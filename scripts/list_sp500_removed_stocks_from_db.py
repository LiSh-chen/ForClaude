"""列出「存活者偏差」還沒解決的那一半：曾經是 S&P 500 成分股、在我們資料庫
涵蓋的期間內被剔除指數、而且資料庫裡完全沒有這檔股票資料的名單，連同它們
在指數裡的區間（何時加入、何時被剔除）。

背景：tw_quant/us_universe.py 的 filter_prices_by_index_membership 只解決
「新進戶」那一半的存活者偏差；「被剔除」這一半沒辦法解決，因為
us_data_provider.fetch_sp500_constituents 每次都只抓「今天」的成分股名單
回填歷史，任何在資料庫涵蓋期間內被剔除的股票從一開始就沒被抓進資料庫。
維基百科的「List of S&P 500 companies」頁面經過完整掃描確認沒有可用的歷史
異動紀錄表（見 debug_sp500_full_table_scan.py 的一次性診斷結果，已刪除）。

fja05680/sp500 這個社群維護的 GitHub repo 提供了時間點成分股快照
（見 tw_quant/sp500_history.py 的完整背景說明），可以拿來重建「誰在哪段
期間是成分股」，跟資料庫現有的股票代號比對，抓出「曾經在指數裡、資料庫卻
完全沒有」的股票名單。

這只是列名單的診斷腳本，不寫入資料庫——下一步（測 yfinance 能不能抓到
這些股票的歷史股價、真的把資料補進資料庫）要等這份名單先讓人看過、確認
方向後才做。

用法：
    python scripts/list_sp500_removed_stocks_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.sp500_history import (
    build_membership_intervals,
    fetch_snapshot_csv_text,
    find_missing_intervals,
    parse_snapshot_table,
)
from tw_quant.storage import get_data_store


def _fmt_date(d) -> str:
    return "至今" if pd.isna(d) else pd.Timestamp(d).date().isoformat()


def main() -> None:
    store = get_data_store()
    us_prices = store.load_us_prices()
    if us_prices.empty:
        print("資料庫裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    db_stock_ids = set(us_prices["stock_id"].unique())
    window_start = us_prices["date"].min()
    window_end = us_prices["date"].max()
    print(f"資料庫目前有 {len(db_stock_ids)} 檔美股，涵蓋 {window_start.date()} ~ {window_end.date()}\n")

    print("抓取 fja05680/sp500 的歷史成分股快照 CSV ...")
    csv_text = fetch_snapshot_csv_text()
    snapshot = parse_snapshot_table(csv_text)
    print(
        f"快照涵蓋 {snapshot['date'].min().date()} ~ {snapshot['date'].max().date()}，"
        f"共 {len(snapshot)} 筆異動紀錄\n"
    )

    intervals = build_membership_intervals(snapshot)
    missing = find_missing_intervals(db_stock_ids, intervals, window_start, window_end)

    if missing.empty:
        print("在資料庫涵蓋期間內，快照裡出現過的股票代號都已經在資料庫裡——沒有找到缺漏。")
        print("（這可能代表快照資料源跟我們的代號規則有落差，例如公司改名/換代號的情況，建議人工抽查幾檔。）")
        return

    n_stocks = missing["stock_id"].nunique()
    n_intervals = len(missing)
    print(f"找到 {n_stocks} 檔股票、共 {n_intervals} 段區間，曾在資料庫涵蓋期間內是成分股，但資料庫裡完全沒有資料：\n")

    for stock_id, group in missing.groupby("stock_id"):
        periods = "; ".join(
            f"{_fmt_date(row.clipped_start)} ~ {_fmt_date(row.end_date)}" for row in group.itertuples()
        )
        print(f"  {stock_id:<8} {periods}")

    print(
        "\n（誠實揭露：這份名單來自社群維護、非官方權威的資料源"
        "（fja05680/sp500，資料源自書籍附帶資料 + 維護者人工核對維基百科，"
        "約每兩個月更新一次），可能有遺漏或誤植，正式補資料前建議抽查幾檔\n"
        "確認剔除/加入日期是否合理；「end_date=至今」代表這份快照資料裡沒觀察到\n"
        "剔除紀錄，不代表保證還在指數裡——也可能是快照資料本身還沒更新到最新異動。\n"
        "下一步：測試 yfinance 能不能抓到這些股票的歷史股價——很多已下市/被收購的\n"
        "股票 yfinance 抓不到，這點要老實測試,不能假設一定抓得到。)"
    )


if __name__ == "__main__":
    main()
