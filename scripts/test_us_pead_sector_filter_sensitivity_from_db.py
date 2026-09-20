"""美股 PEAD + 族群動能篩選的參數敏感度檢驗。

背景：scripts/test_us_pead_sector_momentum_from_db.py 只測過一組固定的
族群篩選參數（回看60個交易日、取前3強產業），樣本外最佳組合（持有60天、
驚喜幅度門檻10%）Sharpe 從無篩選版本的 0.39 大幅改善到 0.68，逼近 QQQ
同期的 0.69——但只測一組參數沒辦法排除「剛好選對這組參數」的可能。這裡
固定住那組表現最好的進出場參數（entry_lag=1、holding_days=60、驚喜幅度
門檻=10%），只對族群篩選本身的兩個參數（回看天數、前K強產業）做網格，
確認這個樣本外改善在合理的參數範圍內是不是普遍存在，而不是單一組合的
運氣。

刻意不同時搜索進出場參數（holding_days/門檻）——如果族群參數也一起
調，等於在 4×3×9×9=972 種組合裡挑出表現最好的那一個，那才是真正的
「調參調出來的好結果」，這裡只固定驗證已經找到的那一個進出場設定，換
不同的族群參數看它撐不撐得住，是比較保守、比較不容易自欺欺人的驗證
方式。

重用 test_us_pead_sector_momentum_from_db.py 的族群動能計算/事件篩選
函式跟 test_us_pead_earnings_drift_from_db.py 的驚喜幅度計算（都是
import，不重複定義）。

用法：
    python scripts/test_us_pead_sector_filter_sensitivity_from_db.py
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
from tw_quant.us_universe import filter_prices_by_index_membership

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_us_pead_earnings_drift_from_db import _earnings_surprise_frame  # noqa: E402
from test_us_pead_sector_momentum_from_db import filter_events_by_sector_momentum  # noqa: E402

ENTRY_LAG_DAYS = 1
FIXED_HOLDING_DAYS = 60  # 前次找到的最佳組合
FIXED_THRESHOLD = 0.10  # 前次找到的最佳組合
MAX_CONCURRENT_POSITIONS = 20

SECTOR_LOOKBACK_GRID = (20, 40, 60, 90)  # 60 是前次測過的值
TOP_K_SECTORS_GRID = (2, 3, 5)  # 3 是前次測過的值（共 11 個有效 GICS 產業）

IN_SAMPLE_START = "2023-09-19"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}

# 前次「沒有族群篩選（對照組）」在同一組 holding_days=60/門檻=10% 的樣本外表現，
# 當作這裡評估「篩選有沒有普遍幫助」的基準線（非本腳本重算，直接沿用前次結果）
BASELINE_NO_FILTER_OOS_SHARPE = 0.21  # hold=60/threshold=10%，見前次結果

HEADER = (
    f"{'lookback':>8} {'top_k':>5} {'n_evt':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7}"
)


def _fmt_row(lookback: int, top_k: int, n_evt: int, m: dict) -> str:
    return (
        f"{lookback:>8} {top_k:>5} {n_evt:>6}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%}"
    )


def run_sector_param_grid(us_prices: pd.DataFrame, events_all: pd.DataFrame, base_cfg) -> pd.DataFrame:
    """對族群篩選的兩個參數（回看天數、前K強產業）做網格，固定住
    entry_lag/holding_days/驚喜幅度門檻，樣本內外都跑，回傳長格式結果。
    獨立成函式方便測試（用小的合成資料驗證回傳的列數/欄位對不對，
    不用真的驗證回測數字本身——數字正確性已經由 event_drift_backtest
    自己的測試涵蓋）。
    """
    rows = []
    for lookback in SECTOR_LOOKBACK_GRID:
        for top_k in TOP_K_SECTORS_GRID:
            events_filtered = filter_events_by_sector_momentum(events_all, us_prices, lookback, top_k)
            drift_cfg = EventDriftConfig(
                entry_lag_days=ENTRY_LAG_DAYS, holding_days=FIXED_HOLDING_DAYS,
                signal_threshold=FIXED_THRESHOLD, max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
            )
            for period, start_date, end_date in (("in_sample", IN_SAMPLE_START, None), ("oos", None, OOS_END)):
                result = run_event_drift_backtest(
                    us_prices, events_filtered, base_cfg, drift_cfg,
                    start_date=start_date, end_date=end_date, cost_module=us_costs,
                )
                m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
                row = {"lookback": lookback, "top_k": top_k, "n_events": len(events_filtered), "period": period}
                row.update(m)
                rows.append(row)
    return pd.DataFrame(rows)


def _print_period(label: str, df: pd.DataFrame, qqq_bench: dict, baseline_sharpe: float) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)
    ranked = df.sort_values("sharpe", ascending=False)
    for _, r in ranked.iterrows():
        print(_fmt_row(int(r["lookback"]), int(r["top_k"]), int(r["n_events"]), r.to_dict()))

    n_beats_baseline = (df["sharpe"] > baseline_sharpe).sum()
    print(
        f"\n{len(df)} 組族群參數裡，有 {n_beats_baseline} 組 Sharpe 超過無篩選對照組的 {baseline_sharpe:.2f}"
        f"（{n_beats_baseline / len(df):.1%}）"
    )
    print(
        f"（對照：QQQ 買進持有同期間 Sharpe {qqq_bench['sharpe']:.2f}、"
        f"總報酬 {qqq_bench['total_return']:.2%}、CAGR {qqq_bench['cagr']:.2%}）"
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

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的價量資料（{earliest.date()} ~ {latest.date()}）")

    events_all = _earnings_surprise_frame(earnings).rename(columns={"surprise": "signal"})
    print(f"財報驚喜事件共 {len(events_all)} 筆")

    base_cfg = build_us_config()
    print(
        f"固定：entry_lag_days={ENTRY_LAG_DAYS}、holding_days={FIXED_HOLDING_DAYS}、"
        f"驚喜幅度門檻={FIXED_THRESHOLD:.0%}（前次找到的最佳組合）\n"
        f"族群篩選參數網格：回看天數={SECTOR_LOOKBACK_GRID}、前K強產業={TOP_K_SECTORS_GRID}\n"
    )

    grid = run_sector_param_grid(us_prices, events_all, base_cfg)

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", grid[grid["period"] == "in_sample"], QQQ_IN_SAMPLE, float("nan"))
    _print_period("樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", grid[grid["period"] == "oos"], QQQ_OOS, BASELINE_NO_FILTER_OOS_SHARPE)

    print(
        "\n（誠實揭露：這裡只固定驗證前次找到的單一進出場設定\n"
        "（holding_days=60、驚喜幅度門檻=10%），對族群篩選參數本身做網格，\n"
        "刻意不同時搜索進出場參數，避免把「挑出最好的一個組合」誤當成\n"
        "「這個策略普遍有效」；族群動能/資金流向定義、產業分類、財報驚喜幅度\n"
        "公式等限制，跟前面幾支 PEAD 腳本相同，這裡不重複列。這是第一輪\n"
        "探索結果，不代表已經驗證出可以實際使用的策略。）"
    )


if __name__ == "__main__":
    main()
