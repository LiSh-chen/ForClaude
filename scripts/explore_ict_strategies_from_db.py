"""測試 ICT（Inner Circle Trader）風格的價格行為訊號在真實資料上有沒有 edge。

ICT 方法論的核心概念大多是為「有日內/分鐘級資料」設計的（liquidity sweep、
fair value gap、market structure shift 這些詞彙的原始定義都在講盤中的價格
行為），我們手上只有日線 OHLC，所以這裡做的是「把概念在精神上忠實地搬到
日線級別」的改編版本，不是原教旨的日內交易法：

  A. Liquidity Sweep（流動性掃蕩後軋回）
     T 日最低價跌破過去 N 日的支撐低點（掃過空方停損/多方防守單），但 T 日
     收盤價又拉回支撐之上（假跌破、軋回）——ICT 術語裡典型的「停損獵殺後
     反轉」。跟前一輪測過的「超跌反彈」概念上接近，但要求更嚴格：要先跌破
     再收回，不是單純跌到低點附近。

  B. Fair Value Gap 回測進場（多方缺口回補反彈）
     gap_age 天前，K 棒序列出現「兩天前最高價 < 當天最低價」的價格跳空缺口
     （中間那根是噴出量能造成的失衡），且從形成到現在都還沒被完全回補過；
     今天價格首度跌進缺口區間，但收盤又拉回缺口之上——ICT 術語裡的「缺口
     回測不破、繼續原方向」。

  C. Market Structure Shift（結構轉折，簡化版）
     用「近期波段低點比更早的波段低點更低」近似「處於空頭結構（做較低的
     低點）」，當價格收盤突破近期波段高點時，視為結構由空翻多的轉折訊號。
     這是簡化版本，真正的 ICT MSS 需要明確的樞紐點（swing point）辨識，
     這裡用滾動高低點近似，不是嚴格版本。

三種訊號都套用系統既有的魚池篩選（流動性 + 站上 60 日均線）、吊燈停利出場、
完整交易成本模型，跟前幾輪用同一套引擎做公平比較。

用法：
    python scripts/explore_ict_strategies_from_db.py
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

MIN_TRADES_FOR_RANKING = 15
ATR_MULT_GRID = (2.0, 2.5, 3.0)
LOOKBACK_GRID = (10, 15)


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


# ---------------------------------------------------------------------------
# A. Liquidity Sweep（流動性掃蕩後軋回）——原始型態判斷跟魚池篩選分開，
# 方便單元測試直接測型態邏輯，不用湊滿 252 日暖身期。
# ---------------------------------------------------------------------------


def liquidity_sweep_mask(master: pd.DataFrame, sweep_lookback: int) -> pd.Series:
    prior_low = ind.rolling_min(master, "low", sweep_lookback)
    prior_low_prior = ind.shift_by_group(prior_low, master, periods=1)  # 不含今天的過去 N 日低點

    swept = master["low"] < prior_low_prior  # 今天跌破過去支撐
    reclaimed = master["close"] > prior_low_prior  # 但收盤拉回支撐之上
    return (swept & reclaimed).fillna(False)


def liquidity_sweep_signal_fn(sweep_lookback: int):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        entry = pool & liquidity_sweep_mask(master, sweep_lookback)
        return entry.values, np.zeros(len(master), dtype=bool)

    return _fn


# ---------------------------------------------------------------------------
# B. Fair Value Gap 回測進場
# ---------------------------------------------------------------------------


def fvg_fill_mask(master: pd.DataFrame, gap_age: int) -> pd.Series:
    high_2_before_gap_day = ind.shift_by_group(master["high"], master, periods=gap_age + 2)
    low_at_gap_day = ind.shift_by_group(master["low"], master, periods=gap_age)
    gap_exists = low_at_gap_day > high_2_before_gap_day
    gap_bottom = high_2_before_gap_day

    if gap_age >= 2:
        min_low_since_gap = master.groupby("stock_id", sort=False)["low"].transform(
            lambda s: s.shift(1).rolling(gap_age - 1, min_periods=1).min()
        )
    else:
        min_low_since_gap = pd.Series(np.inf, index=master.index)
    not_yet_filled = min_low_since_gap > gap_bottom

    dips_into_gap = master["low"] <= gap_bottom
    reclaims = master["close"] > gap_bottom

    return (
        gap_exists.fillna(False)
        & not_yet_filled.fillna(True)
        & dips_into_gap.fillna(False)
        & reclaims.fillna(False)
    )


def fvg_fill_signal_fn(gap_age: int):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        entry = pool & fvg_fill_mask(master, gap_age)
        return entry.values, np.zeros(len(master), dtype=bool)

    return _fn


# ---------------------------------------------------------------------------
# C. Market Structure Shift（簡化版）
# ---------------------------------------------------------------------------


def market_structure_shift_mask(master: pd.DataFrame, downtrend_window: int, swing_window: int) -> pd.Series:
    low_min_now = ind.rolling_min(master, "low", downtrend_window)
    low_min_earlier = ind.shift_by_group(low_min_now, master, periods=downtrend_window)
    low_min_now_prior = ind.shift_by_group(low_min_now, master, periods=1)
    low_min_earlier_prior = ind.shift_by_group(low_min_earlier, master, periods=1)
    downtrend_context = (low_min_now_prior < low_min_earlier_prior).fillna(False)

    swing_high = ind.rolling_max(master, "high", swing_window)
    swing_high_prior = ind.shift_by_group(swing_high, master, periods=1)
    breaks_structure = (master["close"] > swing_high_prior).fillna(False)

    return downtrend_context & breaks_structure


def market_structure_shift_signal_fn(downtrend_window: int, swing_window: int):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        entry = pool & market_structure_shift_mask(master, downtrend_window, swing_window)
        return entry.values, np.zeros(len(master), dtype=bool)

    return _fn


# ---------------------------------------------------------------------------
# 共用搜尋/報表邏輯
# ---------------------------------------------------------------------------

HEADER = (
    f"{'strategy':<10} {'param':<16} {'exit':<14}  "
    f"{'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(name: str, param: str, exit_label: str, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{name:<10} {param:<16} {exit_label:<14}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>5.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def run_grid(name: str, prices, margin_short, base_cfg, param_grid, fn_builder) -> pd.DataFrame:
    print(f"\n=== {name}：搜尋 {len(param_grid) * len(ATR_MULT_GRID) * len(LOOKBACK_GRID)} 組合 ===")
    print(HEADER)
    rows = []
    for param_label, entry_fn in [(str(p), fn_builder(*(p if isinstance(p, tuple) else (p,)))) for p in param_grid]:
        for atr_mult in ATR_MULT_GRID:
            for lookback in LOOKBACK_GRID:
                cfg = copy.deepcopy(base_cfg)
                cfg.sizing.atr_multiplier = atr_mult
                cfg.sizing.chandelier_lookback = lookback
                result = run_backtest(prices, margin_short, cfg, historical_mdd=None, entry_signal_fn=entry_fn)
                m = metrics_from_result(result, cfg.initial_capital, prices=prices)
                exit_label = f"atr={atr_mult}/lb={lookback}"
                row = {"strategy": name, "param": param_label, "exit": exit_label}
                row.update(m)
                rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(5).iterrows():
        print(_fmt_row(name, r["param"], r["exit"], r.to_dict()))
    if ranked.empty:
        print(f"  （沒有組合達到最低 {MIN_TRADES_FOR_RANKING} 筆交易門檻）")
    return df


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
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）"
    )

    base_cfg = build_base_config()

    df_a = run_grid("A_liq_sweep", prices, margin_short, base_cfg, [5, 10, 20, 40], liquidity_sweep_signal_fn)
    df_b = run_grid("B_fvg_fill", prices, margin_short, base_cfg, [3, 5, 8, 13], fvg_fill_signal_fn)
    df_c = run_grid(
        "C_mss",
        prices,
        margin_short,
        base_cfg,
        [(10, 5), (10, 10), (20, 10), (20, 20)],
        market_structure_shift_signal_fn,
    )

    all_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
    n_total = len(all_df)
    n_profitable = (all_df["total_return"] > 0).sum()
    print(f"\n三種 ICT 風格訊號合計 {n_total} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / n_total:.1%}）")

    print("\n=== 總表：三種訊號各自最佳組合 ===")
    print(HEADER)
    for name, df in [("A_liq_sweep", df_a), ("B_fvg_fill", df_b), ("C_mss", df_c)]:
        ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
        if ranked.empty:
            print(f"{name:<10} (無達門檻組合)")
        else:
            r = ranked.iloc[0]
            print(_fmt_row(name, r["param"], r["exit"], r.to_dict()))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "出場統一用吊燈停利 + ATR 停損；n_trades/win%/RR/EV%/PF 已把回測結束時還未平倉的部位"
        "用最後收盤價算進去；這幾種訊號都是日線改編版，不是 ICT 原教旨的日內交易法）"
    )


if __name__ == "__main__":
    main()
