"""測試長假風控疊加層：能不能靠「長假前主動清空曝險」改善目前最佳策略
（結構轉折 + 量能確認）的風險調整後報酬。

背景：scripts/analyze_calendar_effects_from_db.py 發現節前最後一天的隔日
（跨假期）報酬率平均 -0.51%，t 值將近 -10，是這整個探索過程效應量最大的
日曆規律。這裡把它做成一個風控疊加規則：在「長假前最後一天的前一個交易日」
收盤，強制平倉所有現有部位、且當天不產生新候選，讓帳上在長假期間保持空手，
用 tw_quant.backtest.run_backtest 的 pre_holiday_exit_dates 參數實作。

用法：
    python scripts/test_holiday_overlay_from_db.py
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

BEST_DT_WIN = 20
BEST_SW_WIN = 15
BEST_VOL_X = 1.3
BEST_ATR_MULT = 2.5
BEST_LOOKBACK = 10


def build_base_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    cfg.sizing.atr_multiplier = BEST_ATR_MULT
    cfg.sizing.chandelier_lookback = BEST_LOOKBACK
    return cfg


def mss_signal_fn(master, prices, margin_short, cfg):
    pool = build_pool_mask(master, cfg.pool)
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

    entry = pool & downtrend_context & breaks_structure & volume_ok
    return entry.values, np.zeros(len(master), dtype=bool)


def compute_pre_holiday_trigger_dates(prices: pd.DataFrame) -> set[pd.Timestamp]:
    """從交易日曆抓「長假前最後一天的前一個交易日」——這天收盤要強制平倉，
    隔天（也就是長假前最後一天）開盤才會執行出場，讓我們在長假前最後一天
    收盤前就已經空手。長假定義：跟下一個交易日相隔超過 3 個日曆天。
    """
    dates = sorted(prices["date"].unique())
    triggers = set()
    for i in range(len(dates) - 1):
        gap_days = (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
        if gap_days > 3 and i - 1 >= 0:
            triggers.add(pd.Timestamp(dates[i - 1]))
    return triggers


HEADER = (
    f"{'label':<28}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
    f"{'n_trd':>5} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(label: str, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{label:<28}  {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
        f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f}  {m['n_trades']:>5.0f} {m['win_rate']:>6.1%} "
        f"{rr_str} {m['ev_pct']:>7.2%} {pf_str}"
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
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）\n"
    )

    trigger_dates = compute_pre_holiday_trigger_dates(prices)
    print(f"偵測到 {len(trigger_dates)} 個長假風控觸發日（長假前最後一天的前一個交易日）\n")

    base_cfg = build_base_config()

    result_baseline = run_backtest(prices, margin_short, base_cfg, historical_mdd=None, entry_signal_fn=mss_signal_fn)
    m_baseline = metrics_from_result(result_baseline, base_cfg.initial_capital, prices=prices)

    result_overlay = run_backtest(
        prices, margin_short, base_cfg, historical_mdd=None, entry_signal_fn=mss_signal_fn,
        pre_holiday_exit_dates=trigger_dates,
    )
    m_overlay = metrics_from_result(result_overlay, base_cfg.initial_capital, prices=prices)

    print(HEADER)
    print(_fmt_row("C（不做長假風控）", m_baseline))
    print(_fmt_row("C + 長假風控疊加層", m_overlay))

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "n_trades/win%/RR/EV%/PF 已把回測結束時還未平倉的部位用最後收盤價算進去；"
        "長假風控 = 長假前最後一天的前一個交易日收盤強制平倉現有部位、當天不產生新候選）"
    )


if __name__ == "__main__":
    main()
