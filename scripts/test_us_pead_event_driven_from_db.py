"""美股財報驚喜漂移（PEAD）事件驅動版：財報公布後幾個交易日內就進場，固定
持有一段天數後出場，不等待任何全域調倉日曆。

背景：scripts/test_us_pead_earnings_drift_from_db.py 用因子式定期調倉引擎
（tw_quant/factor_backtest.py），訊號在兩次調倉之間發生的變化要等到下次
調倉才會被採用，天生稀釋掉 PEAD 假設「公布後最初幾週到幾個月漂移最明顯」
這個時間尺度。這裡改用專門的事件驅動引擎
（tw_quant/event_drift_backtest.py），每次財報公布各自獨立管理「延遲
進場 -> 固定天數後出場」的生命週期，測試「進場時機更貼近事件」本身有沒有
差——這不是要取代原本的定期調倃版本，兩支腳本並存，互相對照。

沿用跟定期調倉版本完全相同的驚喜幅度定義與資料源（見
scripts/test_us_pead_earnings_drift_from_db.py 的 _earnings_surprise_frame，
這裡直接 import 重用，不重複定義），只換掉「怎麼決定進出場時機」這一層，
確保兩支腳本的差異只來自進場時機、不是訊號本身算法不同。

固定只做多正向意外（signal_threshold=0.0），對稱地測試進場延遲
（entry_lag_days：財報公布後第幾個交易日進場）跟持有天數
（holding_days：固定持有幾個交易日後出場）的敏感度網格。

反未來函數：跟定期調倉版本一樣，entry_lag_days 本身就是「事件當天不能
交易」的保守處理；驚喜幅度計算方式、事件時間點完全共用同一套，見
tw_quant/event_drift_backtest.py 開頭的完整背景說明。

2026-09-21 架構修正：改用完整未過濾的 us_prices + membership 參數，
取代先前先用 filter_prices_by_index_membership 預過濾再傳進引擎的舊
寫法（誤傷 MRVL 等 14 檔股票，詳見 tw_quant/us_universe.py 檔頭）；
「樣本外」窗口也改成明確傳 OOS_START，不再依賴 start_date=None 的隱含
語意（資料庫擴充到 2006 年後，None 會混入 2008 危機期間）。

用法：
    python scripts/test_us_pead_event_driven_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.data_snapshot import (
    load_us_earnings_snapshot,
    load_us_index_membership_snapshot,
    load_us_prices_snapshot,
)
from tw_quant.event_drift_backtest import EventDriftConfig, run_event_drift_backtest
from tw_quant.us_config import build_us_config

# 重用定期調倉版本的驚喜幅度計算，確保兩支腳本的差異只來自進出場時機
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_us_pead_earnings_drift_from_db import _earnings_surprise_frame  # noqa: E402

ENTRY_LAG_GRID = (1, 3, 5)  # 財報公布後第幾個交易日進場
HOLDING_DAYS_GRID = (20, 40, 60)  # 固定持有幾個交易日後出場
MAX_CONCURRENT_POSITIONS = 20
MIN_TRADES_FOR_RANKING = 10

IN_SAMPLE_START = "2023-09-19"
OOS_START = "2018-09-20"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}

HEADER = (
    f"{'entry_lag':>9} {'hold_days':>9}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(entry_lag: int, holding_days: int, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{entry_lag:>9} {holding_days:>9}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def _run_grid(us_prices: pd.DataFrame, events: pd.DataFrame, base_cfg, start_date, end_date, membership) -> pd.DataFrame:
    rows = []
    for entry_lag in ENTRY_LAG_GRID:
        for holding_days in HOLDING_DAYS_GRID:
            drift_cfg = EventDriftConfig(
                entry_lag_days=entry_lag, holding_days=holding_days,
                signal_threshold=0.0, max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
            )
            result = run_event_drift_backtest(
                us_prices, events, base_cfg, drift_cfg,
                start_date=start_date, end_date=end_date, cost_module=us_costs, membership=membership,
            )
            m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
            row = {"entry_lag": entry_lag, "holding_days": holding_days}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def _print_period(label: str, df: pd.DataFrame, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)

    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(int(r["entry_lag"]), int(r["holding_days"]), r.to_dict()))
    if ranked.empty:
        print("沒有任何組合達到最低成交筆數門檻，無法排名，以下是全部組合：")
        print(df.to_string(index=False))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")
    print(
        f"（對照：QQQ 買進持有同期間總報酬 {qqq_bench['total_return']:.2%}、"
        f"CAGR {qqq_bench['cagr']:.2%}、MDD {qqq_bench['max_dd']:.2%}、"
        f"Sharpe {qqq_bench['sharpe']:.2f}、Calmar {qqq_bench['calmar']:.2f}）"
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

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的價量資料（{earliest.date()} ~ {latest.date()}）")

    events = _earnings_surprise_frame(earnings).rename(columns={"surprise": "signal"})
    n_events = len(events)
    n_positive = (events["signal"] >= 0.0).sum()
    print(f"財報驚喜事件共 {n_events} 筆，其中 {n_positive} 筆是正向意外（signal_threshold=0.0 只做多這些）")

    base_cfg = build_us_config()
    print(
        f"固定網格：entry_lag_days={ENTRY_LAG_GRID}、holding_days={HOLDING_DAYS_GRID}、"
        f"max_concurrent_positions={MAX_CONCURRENT_POSITIONS}\n"
        "（跟定期調倉版本用同一套驚喜幅度定義，差異只在進出場時機——事件公布後\n"
        "第幾個交易日進場、固定持有幾天，不是等下一個排定的調倉日）\n"
    )

    df_in = _run_grid(us_prices, events, base_cfg, IN_SAMPLE_START, None, membership)
    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", df_in, QQQ_IN_SAMPLE)

    df_oos = _run_grid(us_prices, events, base_cfg, OOS_START, OOS_END, membership)
    _print_period("樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", df_oos, QQQ_OOS)

    print(
        "\n（誠實揭露：這裡只有簡單的「同時最多持有 N 個部位」上限，沒有更精細的\n"
        "全域風控排隊（例如依訊號強度動態調整部位大小、產業曝險上限）；同一天\n"
        "多個事件搶額度時，一律優先給驚喜幅度較高的，可能不是最優的資金配置\n"
        "方式；固定持有天數到就出場，不管當下賺賠、不做停損停利，是刻意的簡化\n"
        "以便乾淨地測試「進場時機」這個單一變數；驚喜幅度定義、事件時間點跟\n"
        "定期調倉版本完全共用同一套（見 test_us_pead_earnings_drift_from_db.py），\n"
        "那支腳本列出的其他限制（驚喜幅度公式簡化、財報公布當天盤前/盤後未知、\n"
        "存活者偏差只部分修正）這裡同樣適用。這是第一輪探索結果，不代表已經\n"
        "驗證出可以實際使用的策略，也不代表這個版本一定比定期調倉版本更好。）"
    )


if __name__ == "__main__":
    main()
