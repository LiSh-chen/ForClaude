"""延續上一輪的發現——結構轉折(C) + 超跌反彈(D) 混合資金池把 Sharpe 從 0.43
拉到 0.60，是靠分散化而不是訊號變準。這裡朝 Sharpe 1.0 繼續推進，做兩件事：

  1. 混合組合的穩健度檢查：把 D 的 near_low_win 換幾組、把「單日新增曝險
     上限」（目前固定 6%）放寬，看 0.60 是不是禁得起參數變動的穩健結果，
     還是也只是矇到的。放寬單日曝險上限的理由：兩個訊號疊在一起，同一天
     觸發的候選數變多，6% 的上限可能變成新的瓶頸，把分散化的好處又卡掉一部分。

  2. 三訊號混合：再疊加規格書策略 B（券資比軋空，上一輪單獨測試時風險
     極低但報酬太小），看能不能在幾乎不增加風險的前提下再貢獻一點分散化。

用法：
    python scripts/push_sharpe_from_db.py
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
from tw_quant.signals import build_pool_mask, generate_strategy_b_signals
from tw_quant.storage import get_data_store

BEST_DT_WIN = 20
BEST_SW_WIN = 15
BEST_VOL_X = 1.3
BEST_ATR_MULT = 2.5
BEST_LOOKBACK = 10

MIN_TRADES_FOR_RANKING = 15


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    cfg.sizing.atr_multiplier = BEST_ATR_MULT
    cfg.sizing.chandelier_lookback = BEST_LOOKBACK
    return cfg


def mss_mask(master: pd.DataFrame) -> pd.Series:
    low_min_now = ind.rolling_min(master, "low", BEST_DT_WIN)
    low_min_earlier = ind.shift_by_group(low_min_now, master, periods=BEST_DT_WIN)
    low_min_now_prior = ind.shift_by_group(low_min_now, master, periods=1)
    low_min_earlier_prior = ind.shift_by_group(low_min_earlier, master, periods=1)
    downtrend_context = (low_min_now_prior < low_min_earlier_prior).fillna(False)

    swing_high = ind.rolling_max(master, "high", BEST_SW_WIN)
    swing_high_prior = ind.shift_by_group(swing_high, master, periods=1)
    breaks_structure = (master["close"] > swing_high_prior).fillna(False)

    avg_vol = ind.sma(master, "volume", 20)
    avg_vol_prior = ind.shift_by_group(avg_vol, master, periods=1)
    volume_ok = (master["volume"] > avg_vol_prior * BEST_VOL_X).fillna(False)

    return downtrend_context & breaks_structure & volume_ok


def mean_reversion_mask(master: pd.DataFrame, lookback: int) -> pd.Series:
    ll_n = ind.rolling_min(master, "low", lookback)
    ll_n_prior = ind.shift_by_group(ll_n, master, periods=1)
    close_prior = ind.shift_by_group(master["close"], master, periods=1)
    near_low = (close_prior <= ll_n_prior * 1.02).fillna(False)
    confirm = (master["close"] > close_prior).fillna(False)
    return near_low & confirm


def two_way_signal_fn(near_low_win: int):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        entry_a = pool & mss_mask(master)
        entry_b = pool & mean_reversion_mask(master, near_low_win)
        return entry_a.values, entry_b.values

    return _fn


def three_way_signal_fn(near_low_win: int, margin_short_ratio_pr_threshold: float):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        c_entry = pool & mss_mask(master)

        local_cfg = copy.deepcopy(cfg)
        local_cfg.strategy_b.margin_short_ratio_pr_threshold = margin_short_ratio_pr_threshold
        sig_f = generate_strategy_b_signals(master, margin_short, local_cfg.pool, local_cfg.ignition, local_cfg.strategy_b)
        f_entry = sig_f["entry_signal"]

        entry_a = (c_entry | f_entry).values  # C 跟 F 都是低頻訊號，合併進 A 軌
        entry_b = (pool & mean_reversion_mask(master, near_low_win)).values
        return entry_a, entry_b

    return _fn


HEADER = (
    f"{'label':<48}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(label: str, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{label:<48}  {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
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

    base_cfg = build_base_config()

    print("=== 實驗一：混合組合穩健度檢查（D 的 near_low_win x 單日新增曝險上限）===")
    print(HEADER)
    rows = []
    for near_low_win in (5, 10, 15, 20):
        entry_fn = two_way_signal_fn(near_low_win)
        for daily_risk_cap in (0.06, 0.10, 0.15):
            cfg = copy.deepcopy(base_cfg)
            cfg.global_risk.max_daily_new_risk_pct = daily_risk_cap
            m = run_one(prices, margin_short, cfg, entry_fn)
            label = f"D_win={near_low_win}, daily_risk_cap={daily_risk_cap:.0%}"
            row = {"label": label, "near_low_win": near_low_win, "daily_risk_cap": daily_risk_cap}
            row.update(m)
            rows.append(row)

    df1 = pd.DataFrame(rows)
    ranked1 = df1[df1["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked1.iterrows():
        print(_fmt_row(r["label"], r.to_dict()))

    best1 = ranked1.iloc[0]
    print(f"\n最佳：{best1['label']}（Sharpe={best1['sharpe']:.2f}）")

    print("\n=== 實驗二：三訊號混合（結構轉折 + 超跌反彈 + 券資比軋空）===")
    print(HEADER)
    best_near_low_win = int(best1["near_low_win"])
    best_daily_risk_cap = float(best1["daily_risk_cap"])

    cfg2 = copy.deepcopy(base_cfg)
    cfg2.global_risk.max_daily_new_risk_pct = best_daily_risk_cap

    two_way_m = run_one(prices, margin_short, cfg2, two_way_signal_fn(best_near_low_win))
    print(_fmt_row("C+D（用實驗一最佳設定，對照組）", two_way_m))

    for pr_threshold in (85, 90):
        entry_fn = three_way_signal_fn(best_near_low_win, pr_threshold)
        m = run_one(prices, margin_short, cfg2, entry_fn)
        print(_fmt_row(f"C+D+F（券資比pr={pr_threshold}）", m))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "n_trades/win%/RR/EV%/PF 已把回測結束時還未平倉的部位用最後收盤價算進去）"
    )


if __name__ == "__main__":
    main()
