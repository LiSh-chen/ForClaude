"""目標：能不能讓目前最佳策略（結構轉折 + 量能確認）的 Sharpe 更高？

前幾輪已經把「訊號本身」的參數（觀察窗、濾網門檻）搜過一輪了，Sharpe 卡在
0.43 附近。Sharpe = 平均報酬 / 報酬波動度，單純再繼續調訊號參數邊際效益已經
很低，這裡改攻兩個真正能動到「波動度」而不是「訊號準不準」的槓桿：

  1. 投資組合風控參數（產業曝險上限、大盤時機門檻）
     前幾輪為了把「大盤時機」「全域鎖」跟「訊號品質」分開測試，一直故意用
     寬鬆值（產業上限 30%、regime 門檻放寬）。現在訊號本身已經選定，回頭
     測試收緊這些風控參數會不會透過「降低同時期集中曝險」「避開雜訊盤整期」
     來平滑權益曲線、拉高 Sharpe（即使犧牲一些總報酬）。

  2. 訊號混合（結構轉折 + 超跌反彈）
     結構轉折（趨勢延續邏輯）跟超跌反彈（均值回歸邏輯）是兩種經濟意義不同、
     理論上低相關的訊號來源。把兩者放進同一個資金池、共用全域風控一起跑，
     用分散化本身來降低權益曲線波動——這是機構常見的「多訊號組合」做法，
     不需要訊號本身變準，光靠低相關性疊加就可能拉高 Sharpe。

用法：
    python scripts/optimize_sharpe_from_db.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant.backtest import run_backtest
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.signals import build_pool_mask
from tw_quant.storage import get_data_store

# 上一輪找到的最佳結構轉折參數
BEST_DT_WIN = 20
BEST_SW_WIN = 15
BEST_VOL_X = 1.3
BEST_ATR_MULT = 2.5
BEST_LOOKBACK = 10

MIN_TRADES_FOR_RANKING = 15


def build_relaxed_base_config() -> StrategyConfig:
    """前幾輪用的寬鬆基準（拿來當對照組）。"""
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


# ---------------------------------------------------------------------------
# 訊號：結構轉折 + 量能確認（沿用上一輪驗證過的邏輯）
# ---------------------------------------------------------------------------


def market_structure_shift_v2_mask(
    master: pd.DataFrame, downtrend_window: int, swing_window: int, volume_confirm_mult: float = 1.0
) -> pd.Series:
    low_min_now = ind.rolling_min(master, "low", downtrend_window)
    low_min_earlier = ind.shift_by_group(low_min_now, master, periods=downtrend_window)
    low_min_now_prior = ind.shift_by_group(low_min_now, master, periods=1)
    low_min_earlier_prior = ind.shift_by_group(low_min_earlier, master, periods=1)
    downtrend_context = (low_min_now_prior < low_min_earlier_prior).fillna(False)

    swing_high = ind.rolling_max(master, "high", swing_window)
    swing_high_prior = ind.shift_by_group(swing_high, master, periods=1)
    breaks_structure = (master["close"] > swing_high_prior).fillna(False)

    if volume_confirm_mult > 1.0:
        avg_vol = ind.sma(master, "volume", 20)
        avg_vol_prior = ind.shift_by_group(avg_vol, master, periods=1)
        volume_ok = (master["volume"] > avg_vol_prior * volume_confirm_mult).fillna(False)
    else:
        volume_ok = pd.Series(True, index=master.index)

    return downtrend_context & breaks_structure & volume_ok


def mean_reversion_mask(master: pd.DataFrame, lookback: int = 5) -> pd.Series:
    ll_n = ind.rolling_min(master, "low", lookback)
    ll_n_prior = ind.shift_by_group(ll_n, master, periods=1)
    close_prior = ind.shift_by_group(master["close"], master, periods=1)
    near_low = (close_prior <= ll_n_prior * 1.02).fillna(False)
    confirm = (master["close"] > close_prior).fillna(False)
    return near_low & confirm


def mss_signal_fn(master, prices, margin_short, cfg):
    pool = build_pool_mask(master, cfg.pool)
    entry = pool & market_structure_shift_v2_mask(master, BEST_DT_WIN, BEST_SW_WIN, BEST_VOL_X)
    return entry.values, np.zeros(len(master), dtype=bool)


def blended_signal_fn(master, prices, margin_short, cfg):
    """A 軌=結構轉折，B 軌=超跌反彈，兩者共用同一個資金池跟全域風控。"""
    pool = build_pool_mask(master, cfg.pool)
    entry_a = pool & market_structure_shift_v2_mask(master, BEST_DT_WIN, BEST_SW_WIN, BEST_VOL_X)
    entry_b = pool & mean_reversion_mask(master, lookback=5)
    return entry_a.values, entry_b.values


HEADER = (
    f"{'label':<40}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(label: str, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{label:<40}  {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
        f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f}  {m['n_trades']:>5.0f} {m['win_rate']:>6.1%} "
        f"{rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def run_one(prices, margin_short, cfg, entry_fn) -> dict:
    result = run_backtest(prices, margin_short, cfg, historical_mdd=None, entry_signal_fn=entry_fn)
    return metrics_from_result(result, cfg.initial_capital, prices=prices)


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin_short = store.load_margin_short()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔股票的資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）\n"
    )

    # 基準：上一輪找到的最佳結構轉折+量能確認組合，用寬鬆風控（對照組）
    baseline_cfg = build_relaxed_base_config()
    baseline_cfg.sizing.atr_multiplier = BEST_ATR_MULT
    baseline_cfg.sizing.chandelier_lookback = BEST_LOOKBACK
    baseline_m = run_one(prices, margin_short, baseline_cfg, mss_signal_fn)

    print("=== 基準（寬鬆風控，上一輪最佳結構轉折+量能確認）===")
    print(HEADER)
    print(_fmt_row("baseline: industry=30%, regime=(0.25,0.9)", baseline_m))

    # ------------------------------------------------------------------
    # 實驗一：收緊投資組合風控，看能不能平滑權益曲線、拉高 Sharpe
    # ------------------------------------------------------------------
    print("\n=== 實驗一：收緊產業曝險上限 / 大盤時機門檻 ===")
    print(HEADER)
    exp1_rows = []
    for industry_cap in (0.10, 0.15, 0.20, 0.30):
        for breadth, vol_ratio in ((0.25, 0.9), (0.35, 1.0), (0.50, 1.20)):
            cfg = copy.deepcopy(baseline_cfg)
            cfg.global_risk.max_industry_exposure_pct = industry_cap
            cfg.regime.breadth_threshold = breadth
            cfg.regime.volume_ratio_threshold = vol_ratio
            m = run_one(prices, margin_short, cfg, mss_signal_fn)
            label = f"ind_cap={industry_cap:.0%}, regime=({breadth},{vol_ratio})"
            row = {"label": label, "industry_cap": industry_cap, "breadth": breadth, "vol_ratio": vol_ratio}
            row.update(m)
            exp1_rows.append(row)

    df1 = pd.DataFrame(exp1_rows)
    ranked1 = df1[df1["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked1.head(15).iterrows():
        print(_fmt_row(r["label"], r.to_dict()))
    if ranked1.empty:
        print(f"  （沒有組合達到最低 {MIN_TRADES_FOR_RANKING} 筆交易門檻）")

    # ------------------------------------------------------------------
    # 實驗二：混合結構轉折 + 超跌反彈，用分散化拉高 Sharpe
    # ------------------------------------------------------------------
    print("\n=== 實驗二：結構轉折 + 超跌反彈 混合資金池 ===")
    print(HEADER)

    mean_rev_only_fn = lambda master, prices_, margin_short_, cfg: (
        np.zeros(len(master), dtype=bool),
        (build_pool_mask(master, cfg.pool) & mean_reversion_mask(master, 5)).values,
    )
    mean_rev_m = run_one(prices, margin_short, baseline_cfg, mean_rev_only_fn)
    print(_fmt_row("D 超跌反彈 單獨（對照）", mean_rev_m))
    print(_fmt_row("C 結構轉折 單獨（=基準）", baseline_m))

    blended_m = run_one(prices, margin_short, baseline_cfg, blended_signal_fn)
    print(_fmt_row("C+D 混合資金池", blended_m))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "n_trades/win%/RR/EV%/PF 已把回測結束時還未平倉的部位用最後收盤價算進去）"
    )


if __name__ == "__main__":
    main()
