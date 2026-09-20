"""動量策略（top_n=3, momentum_window=126, rebalance_freq_days=21——跟報告
主要引用的那組參數完全相同、不重新調參）套在涵蓋 2008 金融風暴的更早期間，
看這組已經在 2018-2023 樣本外驗證過的參數，遇到真正的系統性金融危機時
表現如何。

背景：2026-09-20 使用者要求「找樣本外的週期測試一次，這次要涵蓋到 2008
年前後的金融風暴」。原始 data/us_prices_snapshot.parquet 只回溯到
2018-09-21，這支腳本假設執行前已經先用加大 BACKFILL_YEARS 的
scripts/ingest_us_daily_data.py（透過 .github/workflows/us_daily_data_ingest.yml
的 workflow_dispatch，backfill_years=20）把現有 577 檔成分股的股價回填到
~2006 年，讓 2008 危機發生時已經過了 min_history_days=252 +
momentum_window=126 的暖身期。

★★★ 存活者偏差警語（務必先讀，這是本次測試最大的方法論限制）★★★
現有的「剔除股回補」機制（tw_quant/sp500_history.py + fja05680/sp500 歷史
成分股快照）目前只回補了「2019 年之後」被剔除指數的 74 檔股票——對
2008-2009 危機期間被剔除指數的股票（雷曼兄弟、Bear Stearns、Washington
Mutual、Wachovia、National City、Fannie Mae、Freddie Mac、Countrywide、
通用汽車舊實體、CIT Group 等，正是這場危機的主角）完全沒有回補。
在價格回填後重跑一次 scripts/backfill_removed_sp500_stocks.py 可以讓
find_missing_intervals() 自動去抓這批 2008 年代的缺漏區間（因為它是依
「資料庫目前實際涵蓋的日期範圍」動態決定要找哪段缺漏，不是寫死 2019
之後），但很多當年直接破產下市、股票變壁紙的公司 yfinance 本來就查無
歷史資料，就算重跑也補不回來。

這代表：這支腳本測出來的 2008 危機期間報酬/回撤，是用「今天還存在的
（或至少沒有完全下市查無資料的）577+N 檔股票」去模擬當年，等於是用
「倖存者的世界」去經歷一次危機——實際危機的殺傷力（尤其是最大回撤）
很可能被低估，因為策略永遠不可能選到那些已經從資料庫消失的股票，也就
不會經歷它們暴跌到 0 的過程。這裡的結果只能當作「這組參數在這個修正過
（但仍不完整）的資料集上表現出的方向性訊號」，不能當作「這組策略在真實
2008 年會有多少最大回撤」的可靠估計——真正的數字幾乎肯定更差。

用法：
    python scripts/test_us_momentum_2008_crisis_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider
from tw_quant.us_universe import filter_prices_by_index_membership

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N = 3

# 危機核心期：2007-10（次貸危機已升溫、S&P500 十月見頂）~ 2009-12（V轉
# 反彈告一段落），涵蓋 Bear Stearns（2008-03）、雷曼兄弟倒閉（2008-09-15）、
# 到 2009-03 谷底反彈的完整過程。
CRISIS_START = "2007-10-01"
CRISIS_END = "2009-12-31"

# 既有的樣本外驗證是從 2018-09-20 開始，這裡延伸到危機期後、銜接既有
# 樣本外起點之前，避免跟既有報告的區間重疊計算兩次。
BRIDGE_START = None  # 用完整可用歷史起點
BRIDGE_END = "2018-09-19"

EXISTING_OOS_START = pd.Timestamp("2018-09-20")

HEADER = f"{'期間':<32} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7}"


def _fmt_row(label: str, m: dict) -> str:
    return (
        f"{label:<32} {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} "
        f"{m['calmar']:>7.2f}  {m['n_trades']:>6.0f} {m['win_rate']:>6.1%}"
    )


def _pool_size_snapshot(us_prices: pd.DataFrame, cfg, as_of: str) -> int:
    """粗略估計某一天有多少檔股票同時滿足流動性/趨勢/歷史長度篩選——
    用來具體量化「危機當下真正可交易的股票池有多小」，讓存活者偏差警語
    不是空話，而是有數字佐證。
    """
    from tw_quant.signals import build_pool_mask

    master = us_prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    pool = build_pool_mask(master, cfg.pool)
    master["pool"] = pool.values
    day = master[master["date"] == pd.Timestamp(as_of)]
    return int(day["pool"].sum())


def main() -> None:
    us_prices_raw = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if us_prices_raw.empty:
        print("快照裡沒有任何美股價量資料。先跑過 backfill_years 加大的 US ingest workflow。", file=sys.stderr)
        sys.exit(1)

    earliest_raw = us_prices_raw["date"].min()
    if earliest_raw > pd.Timestamp(CRISIS_START):
        print(
            f"[警告] 目前資料最早只到 {earliest_raw.date()}，晚於危機測試起點 {CRISIS_START}，"
            "還沒跑過加大 BACKFILL_YEARS 的回填，這支腳本的結果會不完整或直接空手。",
            file=sys.stderr,
        )

    n_rows_before = len(us_prices_raw)
    us_prices = filter_prices_by_index_membership(us_prices_raw, membership)
    n_rows_dropped = n_rows_before - len(us_prices)
    print(
        f"存活者偏差部分修正：依指數加入/剔除區間過濾後，丟掉 {n_rows_dropped} / {n_rows_before} 列"
        f"（{n_rows_dropped / n_rows_before:.1%}）——見檔頭★存活者偏差警語★，"
        "這裡仍然不含 2008 年代被剔除指數、目前資料庫完全沒有資料的股票\n"
    )

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {latest.date()}）\n")

    base_cfg = build_us_config()

    for probe_date in ["2007-10-01", "2008-09-15", "2009-03-09", "2018-09-20"]:
        try:
            n_pool = _pool_size_snapshot(us_prices, base_cfg, probe_date)
            print(f"  股票池快照 {probe_date}：{n_pool} 檔符合流動性/趨勢/歷史長度篩選")
        except Exception as exc:  # noqa: BLE001
            print(f"  股票池快照 {probe_date}：計算失敗（{exc}）")
    print()

    provider = YFinanceUSDataProvider()
    print("抓取 QQQ 價格歷史（危機期間對照基準）...")
    qqq_df = provider.fetch_price(
        "QQQ", start_date=str(earliest.date()), end_date=str((latest + pd.Timedelta(days=1)).date()), industry="ETF"
    )
    qqq_close = qqq_df.sort_values("date").set_index("date")["close"] if not qqq_df.empty else None

    factor_cfg = FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=False)

    print(f"固定參數（跟報告主要引用的組合相同，非重新調參）：momentum_window={MOM_WINDOW}、rebalance_freq_days={REBALANCE_FREQ_DAYS}、top_n={TOP_N}\n")
    print("=== 動量策略 2008 金融風暴延伸測試（美股 S&P 500，完整交易成本）===")
    print(HEADER)

    windows = [
        ("危機核心期 2007-10~2009-12", CRISIS_START, CRISIS_END),
        (f"延伸樣本外 {earliest.date()}~2018-09-19", BRIDGE_START, BRIDGE_END),
    ]

    results = {}
    for label, start, end in windows:
        result = run_factor_backtest(
            us_prices, base_cfg, factor_cfg,
            start_date=start, end_date=end,
            cost_module=us_costs,
        )
        m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
        results[label] = (m, result)
        print(_fmt_row(label, m))

    print(f"\n（對照：既有樣本外驗證區間 2018-09-20~2023-09-18 的結果已在 test_us_momentum_out_of_sample_from_db.py，這裡完全不重複）\n")

    if qqq_close is not None:
        print("=== QQQ 買進持有同期間對照 ===")
        for label, start, end in windows:
            s = pd.Timestamp(start) if start else qqq_close.index.min()
            e = pd.Timestamp(end) if end else qqq_close.index.max()
            window_close = qqq_close[(qqq_close.index >= s) & (qqq_close.index <= e)]
            if len(window_close) < 2:
                print(f"{label:<32} （資料不足，跳過）")
                continue
            total_ret = float(window_close.iloc[-1] / window_close.iloc[0] - 1)
            running_max = window_close.cummax()
            max_dd = float(((window_close - running_max) / running_max).min())
            print(f"{label:<32} total_ret={total_ret:>9.2%}  max_dd={max_dd:>8.2%}")

    print(
        "\n"
        "★ 誠實提醒（見檔頭完整說明，這裡再摘要一次）★\n"
        "上面股票池快照如果顯示 2008-09-15（雷曼倒閉日）附近的可交易檔數明顯少於\n"
        "2018-09-20，代表暖身期或資料覆蓋率本身就有缺口，數字要打更大折扣。更根本的\n"
        "問題是：現有的存活者偏差修正機制只回補了 2019 年後被剔除指數的股票，對\n"
        "2008 年代被剔除、且很多已經下市查無 yfinance 資料的公司完全無法回補——這裡\n"
        "測出來的最大回撤幾乎肯定低估了真實 2008 年會發生的損失，因為策略永遠不會\n"
        "選到那些已經從資料庫消失、股價歸零的股票。這組結果只適合當作方向性參考，\n"
        "不是危機韌性的可靠估計。"
    )


if __name__ == "__main__":
    main()
