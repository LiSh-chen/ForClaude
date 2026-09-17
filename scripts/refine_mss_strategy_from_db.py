"""深入優化「市場結構轉折」(C_mss)——目前這輪探索裡風險調整後體質最健康的
候選（Sharpe 0.36、Calmar 0.27、最大回撤只有 12.9%，樣本數 49~51 筆合理，
參數不是孤島）。目標：在同樣的低回撤前提下提升 EV。

加兩個 ICT 概念裡真正常用、但前一版簡化掉的濾網：

  1. breakout_strength（突破幅度門檻）：不是「剛好收在波段高點之上」就算數，
     要求收盤價要真正拉開一段距離（ICT 術語裡的「displacement」精神），
     濾掉勉強擦過高點又拉回的假突破。
  2. volume_confirm（量能確認）：突破當天成交量要顯著放大（同樣是
     displacement 的一部分——真正的結構轉折通常伴隨資金力道），濾掉量縮
     的無效突破。

同時把 downtrend_window / swing_window 抓更細的網格，確認上一輪看到的好結果
是一片穩健的高原、還是單一參數孤島。

用法：
    python scripts/refine_mss_strategy_from_db.py
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

DOWNTREND_WINDOW_GRID = (15, 20, 25)
SWING_WINDOW_GRID = (10, 15, 20)
BREAKOUT_STRENGTH_GRID = (0.0, 0.01, 0.02)  # 收盤價至少要高於波段高點的百分比
VOLUME_CONFIRM_GRID = (1.0, 1.3, 1.6)  # 突破當天量能至少要是 N 日均量的幾倍（1.0=不濾）
VOLUME_AVG_WINDOW = 20

ATR_MULT_GRID = (2.5, 3.0)
LOOKBACK_GRID = (10, 15)


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


def market_structure_shift_v2_mask(
    master: pd.DataFrame,
    downtrend_window: int,
    swing_window: int,
    breakout_strength: float = 0.0,
    volume_confirm_mult: float = 1.0,
) -> pd.Series:
    """跟原版 market_structure_shift_mask 邏輯相同（近期波段低點比更早的波段
    低點更低 -> 空頭結構前提；收盤突破近期波段高點 -> 轉折訊號），
    多加兩個可選濾網：breakout_strength（突破幅度）、volume_confirm_mult（量能確認）。
    """
    low_min_now = ind.rolling_min(master, "low", downtrend_window)
    low_min_earlier = ind.shift_by_group(low_min_now, master, periods=downtrend_window)
    low_min_now_prior = ind.shift_by_group(low_min_now, master, periods=1)
    low_min_earlier_prior = ind.shift_by_group(low_min_earlier, master, periods=1)
    downtrend_context = (low_min_now_prior < low_min_earlier_prior).fillna(False)

    swing_high = ind.rolling_max(master, "high", swing_window)
    swing_high_prior = ind.shift_by_group(swing_high, master, periods=1)
    breaks_structure = (master["close"] > swing_high_prior * (1.0 + breakout_strength)).fillna(False)

    if volume_confirm_mult > 1.0:
        avg_vol = ind.sma(master, "volume", VOLUME_AVG_WINDOW)
        avg_vol_prior = ind.shift_by_group(avg_vol, master, periods=1)
        volume_ok = (master["volume"] > avg_vol_prior * volume_confirm_mult).fillna(False)
    else:
        volume_ok = pd.Series(True, index=master.index)

    return downtrend_context & breaks_structure & volume_ok


def mss_v2_signal_fn(downtrend_window: int, swing_window: int, breakout_strength: float, volume_confirm_mult: float):
    def _fn(master, prices, margin_short, cfg):
        pool = build_pool_mask(master, cfg.pool)
        entry = pool & market_structure_shift_v2_mask(
            master, downtrend_window, swing_window, breakout_strength, volume_confirm_mult
        )
        return entry.values, np.zeros(len(master), dtype=bool)

    return _fn


HEADER = (
    f"{'dt_win':>6} {'sw_win':>6} {'brk%':>6} {'vol_x':>6}  {'exit':<14}  "
    f"{'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(r: dict) -> str:
    pf = r["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = r["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{r['downtrend_window']:>6.0f} {r['swing_window']:>6.0f} {r['breakout_strength']:>6.1%} "
        f"{r['volume_confirm_mult']:>6.1f}  {r['exit']:<14}  "
        f"{r['total_return']:>9.2%} {r['cagr']:>7.2%} {r['max_dd']:>7.2%} {r['sharpe']:>7.2f} {r['calmar']:>7.2f}  "
        f"{r['n_trades']:>5.0f} {r['win_rate']:>6.1%} {rr_str} {r['ev_pct']:>7.2%} {pf_str}"
    )


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

    total_combos = (
        len(DOWNTREND_WINDOW_GRID) * len(SWING_WINDOW_GRID) * len(BREAKOUT_STRENGTH_GRID)
        * len(VOLUME_CONFIRM_GRID) * len(ATR_MULT_GRID) * len(LOOKBACK_GRID)
    )
    print(f"總共搜尋 {total_combos} 組合...\n")

    rows = []
    for dt_win in DOWNTREND_WINDOW_GRID:
        for sw_win in SWING_WINDOW_GRID:
            for brk in BREAKOUT_STRENGTH_GRID:
                for vol_x in VOLUME_CONFIRM_GRID:
                    entry_fn = mss_v2_signal_fn(dt_win, sw_win, brk, vol_x)
                    for atr_mult in ATR_MULT_GRID:
                        for lookback in LOOKBACK_GRID:
                            cfg = copy.deepcopy(base_cfg)
                            cfg.sizing.atr_multiplier = atr_mult
                            cfg.sizing.chandelier_lookback = lookback
                            result = run_backtest(
                                prices, margin_short, cfg, historical_mdd=None, entry_signal_fn=entry_fn
                            )
                            m = metrics_from_result(result, cfg.initial_capital, prices=prices)
                            row = {
                                "downtrend_window": dt_win, "swing_window": sw_win,
                                "breakout_strength": brk, "volume_confirm_mult": vol_x,
                                "exit": f"atr={atr_mult}/lb={lookback}",
                            }
                            row.update(m)
                            rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)

    print(f"=== 依 Sharpe 排序前 25 名（要求 n_trades >= {MIN_TRADES_FOR_RANKING}）===")
    print(HEADER)
    for _, r in ranked.head(25).iterrows():
        print(_fmt_row(r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    print(f"\n=== 依 EV% 排序前 15 名（要求 n_trades >= {MIN_TRADES_FOR_RANKING}）===")
    print(HEADER)
    ranked_ev = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("ev_pct", ascending=False)
    for _, r in ranked_ev.head(15).iterrows():
        print(_fmt_row(r.to_dict()))

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")
        return

    # 穩健度檢查：固定住排名第一組合的 breakout_strength/volume_confirm/exit，
    # 印出 downtrend_window x swing_window 的 Sharpe 矩陣，看好結果是一片高原
    # 還是單一孤島。
    best = ranked.iloc[0]
    print(
        f"\n=== 穩健度檢查：固定 brk={best['breakout_strength']:.1%}, vol_x={best['volume_confirm_mult']}, "
        f"{best['exit']}，看 downtrend_window x swing_window 的 Sharpe 分布 ==="
    )
    fixed = df[
        (df["breakout_strength"] == best["breakout_strength"])
        & (df["volume_confirm_mult"] == best["volume_confirm_mult"])
        & (df["exit"] == best["exit"])
    ]
    pivot = fixed.pivot_table(index="downtrend_window", columns="swing_window", values="sharpe")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "出場統一用吊燈停利 + ATR 停損；n_trades/win%/RR/EV%/PF 已把回測結束時還未平倉的部位"
        "用最後收盤價算進去；brk = 收盤價需高於波段高點的最低幅度；vol_x = 突破當天量能"
        f"需達 {VOLUME_AVG_WINDOW} 日均量的幾倍，1.0 代表不濾）"
    )


if __name__ == "__main__":
    main()
