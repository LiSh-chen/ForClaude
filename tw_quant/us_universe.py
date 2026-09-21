"""美股股票池的時間點成分股過濾：修正「用今天的 S&P 500 名單回填過去歷史」
天生帶有的存活者偏差。

背景（見 docs/research_findings.md）：`tw_quant/us_data_provider.py` 的
`fetch_sp500_constituents` 每次執行都抓「今天」的 503 檔成分股，往回
回填 8 年歷史——任何這段期間被剔除指數的股票完全不在資料庫裡，而現在
503 檔裡有 24.5%（123 檔）是 2018-09-20 之後才加入指數的、10.5%（53 檔）
是 2023-09-19 之後才加入的。這對動量策略尤其危險：動量策略買最近漲最多
的股票，而「今天還在指數裡」本身就已經是一種事後贏家篩選，兩者疊加會
高估報酬。

`tw_quant/storage.py` 的 `us_index_membership` 表存的是每檔股票在指數裡的
「區間」——`(stock_id, start_date, end_date)`，`end_date` 是 NULL 代表還沒
觀察到被剔除（可能是目前仍是成分股、也可能是還沒補資料）。這裡把每一列
(stock_id, date) 對照該股票所有已知區間，date 不落在任何一段區間內的列
整個丟掉。這樣後面所有依賴 prices 逐股累積的滾動指標（均線/波動率/暖身期
天數）自然算不到「還沒加入」或「已經被剔除」的日子，不需要在每個策略
訊號函式裡另外加判斷式。

現況（2026-09 更新）：
- 目前 503 檔成分股的「加入日期」半邊，靠 `fetch_sp500_constituents` 抓到
  的維基百科 "Date added" 欄位補齊，這半邊原本就有解。
- 「被剔除、已經不在今天成分股名單」的半邊，原本完全無解——直到找到
  fja05680/sp500 這個社群維護的歷史成分股快照（見
  tw_quant/sp500_history.py），重建出 177 檔缺漏股票的時間點區間，其中
  74 檔 yfinance 抓得到歷史股價、已經用
  scripts/backfill_removed_sp500_stocks.py 補進資料庫、寫入這裡用到的
  `(stock_id, start_date, end_date)` 區間。
- ★ 仍然不解決的部分：177 檔裡另外 101 檔（因併購/破產/私有化/改名下市，
  yfinance 完全查無資料，見對話紀錄 2026-09-19 的
  test_yfinance_full_backfill_scan.py 全樣本測試結果）依然完全抓不到
  資料——這裡沒辦法無中生有，需要另一個能提供這些下市股票歷史股價的
  資料源才能补上。AVB、EQR 這 2 檔則是 yfinance 抓到的資料量異常（區間
  對得上但只抓回一個月資料），原因待查，暫不計入已回填名單。

★★ 2026-09-21 架構修正（filter_prices_by_index_membership 的已知副作用）★★
使用者發現：`scripts/ingest_us_daily_data.py` 每次執行都用 Wikipedia 當下
「Date added」欄位整批覆寫 `us_index_membership.start_date`——如果一家公司
曾經一度被剔除指數、之後又重新納入，Wikipedia 的「Date added」只會記錄
「最近一次」加入日期，不會保留更早那段成分股身份。全樣本清查（630 檔）
發現至少 14 檔長期上市公司（MRVL、FLEX、CASY、COHR、CIEN、FIX、CRH、EME
等，原始股價資料完整涵蓋 2006~2026 年）因此被誤判成分股身份只有近期
幾十~兩百多天，被 `filter_prices_by_index_membership` 砍掉絕大部分真實
歷史，導致這些股票的 60 日均線、5 日均量、252 天最短歷史門檻全部算不準，
系統性地被 `tw_quant.signals.build_pool_mask` 排除在股票池外——即使股價
資料本身完整無缺。

根本問題是「先過濾價格序列、再算指標」這個順序本身：指標（均線/均量/
歷史長度/動量排名）需要完整、連續的真實股價序列才能算對，「是否為當前
指數成分股」則是另一件事——不該混在一起處理。正確作法是：**指標永遠用
完整未過濾的 us_prices 算，「是否為成分股」只在最後決定「這天能不能被
選中」時，額外用 `membership_eligibility_mask` 當一個獨立的資格遮罩疊加
上去**（T-1 為止已知資訊，用法見 tw_quant/factor_backtest.py
run_factor_backtest 的 membership 參數）——不再讓它去汙染價格序列本身。

`filter_prices_by_index_membership`（整段過濾掉不合格的列）保留下來給
還沒改用新架構的舊腳本相容用，但新的動量策略腳本一律改用下面的
`membership_eligibility_mask`（只回傳遮罩，不動價格序列）。
"""

from __future__ import annotations

import pandas as pd


def membership_eligibility_mask(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """跟 filter_prices_by_index_membership 判斷邏輯完全一致（membership 須
    有 stock_id、start_date、end_date 三欄；一檔股票可能有不只一段區間；
    end_date 是 NaT 代表還沒觀察到終點），差別是這裡只回傳跟 prices 等長、
    同順序的布林遮罩（True=這天這檔股票是已知的指數成分股），不砍任何列。

    這樣呼叫端可以把「是否為成分股」當成跟均線/均量/歷史長度同一層級的
    資格條件疊加進 build_pool_mask 的判定，而不用在指標計算之前就先把
    價格序列砍出一堆缺口——後者正是 2026-09-21 發現的架構性 bug 根源
    （見本檔案開頭的完整說明）。

    沒有任何區間紀錄的股票（stock_id 根本不在 membership 裡）視為 True
    （不知道，不代表排除），跟 filter_prices_by_index_membership 一致。
    """
    if membership.empty:
        return pd.Series(True, index=prices.index)

    m = membership.dropna(subset=["start_date"])
    if m.empty:
        return pd.Series(True, index=prices.index)

    eligible = pd.Series(True, index=prices.index)
    for stock_id, intervals in m.groupby("stock_id"):
        stock_rows = prices["stock_id"] == stock_id
        if not stock_rows.any():
            continue
        dates = prices.loc[stock_rows, "date"]
        in_any_interval = pd.Series(False, index=dates.index)
        for interval in intervals.itertuples():
            after_start = dates >= interval.start_date
            before_end = dates < interval.end_date if pd.notna(interval.end_date) else True
            in_any_interval |= after_start & before_end
        eligible.loc[stock_rows] = in_any_interval

    return eligible


def filter_prices_by_index_membership(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """membership 須有 stock_id、start_date、end_date 三欄（見
    tw_quant.storage.DataStore.load_us_index_membership）。一檔股票可能有
    不只一段區間（中途被剔除又重新加入）；end_date 是 NaT 代表這段區間
    還沒觀察到終點（開放式）。

    沒有任何區間紀錄的股票（stock_id 根本不在 membership 裡）視為「不知道，
    不過濾」——缺資料不代表排除，保守起見寧可不誤殺。

    ★ 注意（2026-09-21）：這個函式會把不合格的列整段砍掉，砍完之後任何
    依賴 prices 逐股累積的滾動指標（均線/歷史天數）都會被連帶弄錯——見
    本檔案開頭的架構修正說明。新的動量策略腳本不要再用這個函式配合
    run_factor_backtest，改用 membership_eligibility_mask() 當資格遮罩、
    傳 membership 參數給 run_factor_backtest，讓它在不破壞價格序列的
    前提下處理。這個函式保留給還沒改用新架構、或本來就不需要逐股累積
    指標的用途（例如純粹統計「過濾後還剩幾列」）。
    """
    eligible = membership_eligibility_mask(prices, membership)
    return prices[eligible].reset_index(drop=True)
