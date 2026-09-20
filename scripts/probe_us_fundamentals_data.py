"""美股財報基本面資料可行性探路：不建立任何正式資料管線，純粹用一小批
股票實際呼叫 yfinance 的季報損益表／財報公布日期 API，確認：

  1. 涵蓋度夠不夠——多少檔股票查得到資料。
  2. 回溯歷史夠不夠長——季報/財報公布日最多能拿到幾筆、涵蓋幾年。
  3. 財報公布日期精不精確——是不是真的公布日，還是季末代理日期。PEAD
     （財報驚喜漂移）跟 valuation 因子的歷史重建都需要「真正公布的那天」
     才知道市場當時已經知道什麼、還不知道什麼，如果 yfinance 給的其實是
     季末日期，直接拿來用就是偷看未來資訊。
  4. EPS 預期值/實際值/驚喜幅度資料完不完整。

背景：使用者想探討美股財報基本面策略（PEAD、營收成長動能、valuation
因子），這三個方向都仰賴 yfinance 的季報財務資料，但這些資料的品質/
深度從沒在這個專案裡驗證過——跟先前已經驗證堪用的 FinMind（台股）/
yfinance 價量資料（美股）不一樣，沒有已知先例。在投入建立正式資料管線
之前，先用小批次驗貨，再決定 PEAD/營收動能/valuation 三個方向裡，哪些
可行、哪些要另找資料源（例如 SEC EDGAR）。

抽樣用現有已經抓過的 577 檔美股快照（data/us_prices_snapshot.parquet，
不需要連資料庫），均勻抽 40 檔，涵蓋大中小型股都有機會抽到。

只印報告，不寫入資料庫、不建立任何新的 storage 表格。

用法：
    python scripts/probe_us_fundamentals_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_snapshot import load_us_prices_snapshot
from tw_quant.us_data_provider import YFinanceUSDataProvider

SLEEP_SECONDS = 1.0  # 對 Yahoo Finance 客氣一點，避免連續呼叫被暫時限速
SAMPLE_SIZE = 40
MIN_QUARTERS_USABLE = 8  # 至少 2 年季報才算堪用起點（門檻寬鬆，先看能不能到）
QUARTER_END_DAYS = {(3, 31), (6, 30), (9, 30), (12, 31)}


def pick_sample_tickers(all_ids: list[str], n: int) -> list[str]:
    """從完整股票代號清單均勻抽 n 檔（等間隔索引，不是隨機抽樣），確保
    可重現、且不會因為清單本身的排序（通常是字母序）而集中抽到某個區段。
    """
    ids = sorted(set(all_ids))
    if len(ids) <= n:
        return ids
    idx = np.linspace(0, len(ids) - 1, n).round().astype(int)
    return sorted({ids[i] for i in idx})


def _find_row_label(index, keywords: tuple[str, ...]) -> str | None:
    """在損益表的科目列名裡，用不分大小寫的關鍵字找第一個符合的列名——
    不同 yfinance 版本/不同公司回傳的確切科目名稱可能不一致（例如
    "Total Revenue" vs "TotalRevenue" vs 根本沒有這個科目)，這裡故意用
    模糊比對而不是要求完全相符，順便讓探路報告能誠實反映「這個科目到底
    存不存在」。
    """
    for label in index:
        text = str(label).lower().replace(" ", "")
        if all(k in text for k in keywords):
            return label
    return None


def _find_col_label(columns, keywords: tuple[str, ...]) -> str | None:
    """跟 _find_row_label 一樣的模糊比對，用在 earnings_dates 的欄位名稱上。"""
    for label in columns:
        text = str(label).lower().replace(" ", "")
        if all(k in text for k in keywords):
            return label
    return None


def _is_quarter_end_date(date: pd.Timestamp) -> bool:
    return (date.month, date.day) in QUARTER_END_DAYS


def probe_one(provider: YFinanceUSDataProvider, stock_id: str) -> dict:
    """探測單一股票的季報基本面資料，回傳判讀用的摘要。provider 用注入的
    方式傳入，方便測試（用假 provider 替換掉真的 yfinance 呼叫，不用連網）。
    """
    try:
        data = provider.fetch_quarterly_fundamentals(stock_id)
    except Exception as exc:  # noqa: BLE001 -- 40 檔裡任何一檔失敗都不該中斷其他檔
        return {"stock_id": stock_id, "error": f"{type(exc).__name__}: {exc}"}

    income = data.get("quarterly_income_stmt")
    earnings = data.get("earnings_dates")
    result: dict = {"stock_id": stock_id, "error": None}

    if income is None or income.empty:
        result.update(n_quarters_revenue=0, revenue_range=None, has_revenue_row=False, income_row_labels=[])
    else:
        result["income_row_labels"] = list(income.index)
        revenue_label = _find_row_label(income.index, ("revenue",))
        result["has_revenue_row"] = revenue_label is not None
        if revenue_label is not None:
            non_null_cols = income.columns[income.loc[revenue_label].notna()]
            result["n_quarters_revenue"] = len(non_null_cols)
            result["revenue_range"] = (
                (min(non_null_cols).date(), max(non_null_cols).date()) if len(non_null_cols) else None
            )
        else:
            result["n_quarters_revenue"] = 0
            result["revenue_range"] = None

    if earnings is None or earnings.empty:
        result.update(n_earnings_rows=0, earnings_range=None, pct_with_actual_eps=0.0, pct_dates_on_quarter_end=None)
    else:
        idx_dates = earnings.index
        if getattr(idx_dates, "tz", None) is not None:
            idx_dates = idx_dates.tz_localize(None)
        result["n_earnings_rows"] = len(earnings)
        result["earnings_range"] = (idx_dates.min().date(), idx_dates.max().date())

        actual_col = _find_col_label(earnings.columns, ("reported", "eps"))
        result["has_actual_eps_col"] = actual_col is not None
        if actual_col is not None:
            result["pct_with_actual_eps"] = float(earnings[actual_col].notna().mean())
        else:
            result["pct_with_actual_eps"] = 0.0

        result["pct_dates_on_quarter_end"] = float(pd.Series([_is_quarter_end_date(d) for d in idx_dates]).mean())

    return result


def main() -> None:
    us_prices = load_us_prices_snapshot()
    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    tickers = pick_sample_tickers(list(us_prices["stock_id"].unique()), SAMPLE_SIZE)
    print(f"抽樣 {len(tickers)} 檔股票測試 yfinance 季報基本面資料（每檔間隔 {SLEEP_SECONDS} 秒）\n")

    provider = YFinanceUSDataProvider()
    results = []
    schema_printed = False
    for i, stock_id in enumerate(tickers, start=1):
        r = probe_one(provider, stock_id)
        results.append(r)
        if r.get("error"):
            print(f"[{i}/{len(tickers)}] ✗ {stock_id}：{r['error']}")
        else:
            print(
                f"[{i}/{len(tickers)}] {stock_id}："
                f"營收季數={r['n_quarters_revenue']}（{r['revenue_range']}）、"
                f"財報公布日筆數={r['n_earnings_rows']}（{r['earnings_range']}）、"
                f"有實際EPS比例={r['pct_with_actual_eps']:.0%}、"
                f"日期卡在季末比例={r['pct_dates_on_quarter_end']}"
            )
            if not schema_printed and r["income_row_labels"]:
                schema_printed = True
                print(f"    （schema 一次性紀錄，供後續正式管線參考——income 科目列名: {r['income_row_labels']}）")
        time.sleep(SLEEP_SECONDS)

    ok = [r for r in results if not r.get("error")]
    n_failed = len(tickers) - len(ok)
    n_usable_revenue = sum(1 for r in ok if r["n_quarters_revenue"] >= MIN_QUARTERS_USABLE)
    n_usable_earnings = sum(
        1 for r in ok if r["n_earnings_rows"] >= MIN_QUARTERS_USABLE and r["pct_with_actual_eps"] > 0.5
    )
    avg_quarters_revenue = float(np.mean([r["n_quarters_revenue"] for r in ok])) if ok else 0.0
    avg_earnings_rows = float(np.mean([r["n_earnings_rows"] for r in ok])) if ok else 0.0
    quarter_end_pcts = [r["pct_dates_on_quarter_end"] for r in ok if r.get("pct_dates_on_quarter_end") is not None]
    avg_pct_quarter_end = float(np.mean(quarter_end_pcts)) if quarter_end_pcts else float("nan")

    print(f"\n=== 總覽（{len(tickers)} 檔抽樣）===")
    print(f"抓取失敗：{n_failed} 檔")
    print(f"季報營收堪用（>= {MIN_QUARTERS_USABLE} 季）：{n_usable_revenue}/{len(ok)} 檔，平均 {avg_quarters_revenue:.1f} 季")
    print(
        f"財報公布日+實際EPS堪用（>= {MIN_QUARTERS_USABLE} 筆且過半有實際值）："
        f"{n_usable_earnings}/{len(ok)} 檔，平均 {avg_earnings_rows:.1f} 筆"
    )
    print(f"財報公布日卡在季末日期的比例（越高代表這個日期越不能當「真正公布日」用）：{avg_pct_quarter_end:.0%}")

    print(
        "\n（誠實揭露：這是探路用的一次性抽樣，不是正式資料管線；quarterly_income_stmt\n"
        "只回傳 yfinance 目前保留的季度，通常不會有很長的歷史——上面季數/筆數如果\n"
        "偏少，代表這條路能做到的回測長度會遠短於價量資料的 8 年；「日期卡在季末\n"
        "比例」如果很高，代表 earnings_dates 給的可能是季末代理日期而不是真正的\n"
        "公布日，PEAD／valuation 都需要真正的公布日才能不偷看未來，這種情況下這個\n"
        "資料源這條路可能整個走不通，需要另找資料源（例如 SEC EDGAR）才能繼續做\n"
        "PEAD/valuation；營收成長動能對日期精確度的要求比較低，即使公布日不精確，\n"
        "用季末日期＋合理的公布延遲估計也還有機會繼續做。）"
    )


if __name__ == "__main__":
    main()
