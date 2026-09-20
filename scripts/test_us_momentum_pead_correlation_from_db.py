"""動量策略（top_n=3）跟 PEAD+族群篩選策略的逐日報酬率相關係數，回答
「這兩個策略組合起來有沒有真正的分散效果」，再看一個最單純的 50/50
混合（每天用固定權重平均兩者的逐日報酬率，不做更複雜的風險平價）表現
如何，跟 QQQ 買進持有、跟兩個策略各自單獨表現放在一起比較。

背景：使用者觀察到動量策略在樣本內大贏 QQQ、樣本外小贏；PEAD+族群篩選
策略則是樣本內輸 QQQ、樣本外（風險調整後）小贏，問組合起來有沒有機會
全面打敗 QQQ。在檢驗這個之前，先確認兩個策略的報酬率相關係數夠不夠低
——如果兩者高度相關（畢竟都偏好最近漲勢強的大型股，重疊機率不低），
組合起來就沒有真正的分散效果，只是把資金拆成兩份、稀釋各自的優勢，
不會有「1+1>2」的效果。

2026-09-20 發現並修好 tw_quant/event_drift_backtest.py 的一個 bug：
equity_curve 原本橫跨完整歷史，不管 start_date/end_date 設定，導致
CAGR 分母用整段歷史年數而非真正窗格年數，嚴重低估——本腳本用的是修好
之後的版本，PEAD+族群篩選這裡印出來的數字會跟修 bug 之前報告過的不同
（CAGR 應該會提高很多），這是預期中的更正，不是新的發現。

固定用先前找到的最佳組合：動量 momentum_window=126/rebalance=21/top_n=3；
PEAD entry_lag=1/holding_days=60/驚喜幅度門檻=10%/族群篩選回看90天/
前2強產業。

反未來函數：兩個策略各自的訊號計算完全沿用各自腳本已經驗證過的邏輯
（factor_backtest.py 的動量排名、_earnings_surprise_frame 的驚喜幅度、
filter_events_by_sector_momentum 的族群動能篩選），這裡只是把兩條已經
算好的逐日報酬率序列拿來對齊、算相關係數跟簡單混合，不涉及新的訊號
計算，沒有新的反未來函數風險。

用法：
    python scripts/test_us_momentum_pead_correlation_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.data_snapshot import (
    load_us_earnings_snapshot,
    load_us_index_membership_snapshot,
    load_us_prices_snapshot,
)
from tw_quant.event_drift_backtest import EventDriftConfig, run_event_drift_backtest
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.leverage import metrics_from_equity_curve
from tw_quant.us_config import build_us_config
from tw_quant.us_universe import filter_prices_by_index_membership

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_us_pead_earnings_drift_from_db import _earnings_surprise_frame  # noqa: E402
from test_us_pead_sector_momentum_from_db import filter_events_by_sector_momentum  # noqa: E402

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N = 3

PEAD_ENTRY_LAG_DAYS = 1
PEAD_HOLDING_DAYS = 60
PEAD_THRESHOLD = 0.10
PEAD_SECTOR_LOOKBACK = 90
PEAD_TOP_K_SECTORS = 2
PEAD_MAX_CONCURRENT = 20

IN_SAMPLE_START = "2023-09-19"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}

HEADER = f"{'標的':>18} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}"


def _fmt_row(label: str, m: dict) -> str:
    return (
        f"{label:>18} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}"
    )


def _momentum_equity(us_prices: pd.DataFrame, base_cfg, start_date, end_date) -> pd.Series:
    factor_cfg = FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=False)
    result = run_factor_backtest(us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs)
    return result.equity_curve["equity"]


def _pead_sector_equity(us_prices: pd.DataFrame, events_all: pd.DataFrame, base_cfg, start_date, end_date) -> pd.Series:
    events_filtered = filter_events_by_sector_momentum(events_all, us_prices, PEAD_SECTOR_LOOKBACK, PEAD_TOP_K_SECTORS)
    drift_cfg = EventDriftConfig(
        entry_lag_days=PEAD_ENTRY_LAG_DAYS, holding_days=PEAD_HOLDING_DAYS,
        signal_threshold=PEAD_THRESHOLD, max_concurrent_positions=PEAD_MAX_CONCURRENT,
    )
    result = run_event_drift_backtest(
        us_prices, events_filtered, base_cfg, drift_cfg,
        start_date=start_date, end_date=end_date, cost_module=us_costs,
    )
    return result.equity_curve["equity"]


def _print_period(label: str, mom_eq: pd.Series, pead_eq: pd.Series, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")

    mom_ret = mom_eq.pct_change().dropna()
    pead_ret = pead_eq.pct_change().dropna()
    combined = pd.DataFrame({"mom": mom_ret, "pead": pead_ret}).dropna()
    corr = combined["mom"].corr(combined["pead"])
    print(f"共同交易日數：{len(combined)}，逐日報酬率相關係數：{corr:.3f}")

    blend_ret = (combined["mom"] + combined["pead"]) / 2
    blend_eq = (1 + blend_ret).cumprod()

    print(HEADER)
    print(_fmt_row("動量 top_n=3（單獨）", metrics_from_equity_curve(mom_eq)))
    print(_fmt_row("PEAD+族群篩選（單獨）", metrics_from_equity_curve(pead_eq)))
    print(_fmt_row("50/50 固定權重混合", metrics_from_equity_curve(blend_eq)))
    print(
        f"{'QQQ買進持有':>18} "
        f"{qqq_bench['total_return']:>9.2%} {qqq_bench['cagr']:>7.2%} {qqq_bench['max_dd']:>7.2%} "
        f"{qqq_bench['sharpe']:>7.2f} {qqq_bench['calmar']:>7.2f}"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()
    earnings = load_us_earnings_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)
    if earnings.empty:
        print(
            "快照裡沒有任何美股財報公布資料（data/us_earnings_snapshot.parquet 是空的）"
            "——需要先跑過 ingest_us_earnings_data.yml。",
            file=sys.stderr,
        )
        sys.exit(1)

    n_rows_before = len(us_prices)
    us_prices = filter_prices_by_index_membership(us_prices, membership)
    n_rows_dropped = n_rows_before - len(us_prices)
    print(
        f"存活者偏差部分修正：依指數加入日期過濾後，丟掉 {n_rows_dropped} / {n_rows_before} 列"
        f"（{n_rows_dropped / n_rows_before:.1%}）\n"
    )

    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的價量資料\n")
    print(
        f"固定：動量 momentum_window={MOM_WINDOW}、rebalance={REBALANCE_FREQ_DAYS}、top_n={TOP_N}；"
        f"PEAD entry_lag={PEAD_ENTRY_LAG_DAYS}、holding_days={PEAD_HOLDING_DAYS}、"
        f"驚喜幅度門檻={PEAD_THRESHOLD:.0%}、族群篩選回看{PEAD_SECTOR_LOOKBACK}天/前{PEAD_TOP_K_SECTORS}強產業\n"
    )

    events_all = _earnings_surprise_frame(earnings).rename(columns={"surprise": "signal"})
    base_cfg = build_us_config()

    mom_in = _momentum_equity(us_prices, base_cfg, IN_SAMPLE_START, None)
    pead_in = _pead_sector_equity(us_prices, events_all, base_cfg, IN_SAMPLE_START, None)
    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", mom_in, pead_in, QQQ_IN_SAMPLE)

    mom_oos = _momentum_equity(us_prices, base_cfg, None, OOS_END)
    pead_oos = _pead_sector_equity(us_prices, events_all, base_cfg, None, OOS_END)
    _print_period("樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", mom_oos, pead_oos, QQQ_OOS)

    print(
        "\n（誠實揭露：PEAD+族群篩選這裡印出來的數字用的是修好 equity_curve\n"
        "窗格 bug 之後的正確版本，跟之前報告過的數字（bug 修復前）不一樣，\n"
        "CAGR 應該會明顯提高——這是更正，不是新發現，見腳本開頭說明；\n"
        "50/50 混合只是每天固定用兩者逐日報酬率的簡單平均，不是真的每天\n"
        "重新平衡兩筆各自獨立資金部位的精確模擬（沒有考慮混合部位之間的\n"
        "現金調度、也沒有額外的交易成本），是最單純的近似，用來快速判斷\n"
        "組合有沒有方向上的價值，不是精確的可執行策略；相關係數只反映\n"
        "報酬率序列的線性相關，沒有測過兩者實際持股名單的重疊程度，兩者\n"
        "可以同時具備低報酬率相關但持股高度重疊的情況（不常見但不是不可能）。）"
    )


if __name__ == "__main__":
    main()
