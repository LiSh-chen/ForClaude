"""美股股票池的時間點成分股過濾：部分修正「用今天的 S&P 500 名單回填
過去歷史」天生帶有的存活者偏差。

背景（見 docs/research_findings.md）：`tw_quant/us_data_provider.py` 的
`fetch_sp500_constituents` 每次執行都抓「今天」的 503 檔成分股，往回
回填 8 年歷史——任何這段期間被剔除指數的股票完全不在資料庫裡，而現在
503 檔裡有 24.5%（123 檔）是 2018-09-20 之後才加入指數的、10.5%（53 檔）
是 2023-09-19 之後才加入的。這對動量策略尤其危險：動量策略買最近漲最多
的股票，而「今天還在指數裡」本身就已經是一種事後贏家篩選，兩者疊加會
高估報酬。

這裡只解決「新進戶」這一半：把每一列 (stock_id, date) 對照這檔股票的
指數加入日期（見 tw_quant/storage.py 的 us_index_membership 表，欄位來自
`fetch_sp500_constituents` 抓到的維基百科 "Date added" 欄位），date 早於
加入日期的列整個丟掉。這樣後面所有依賴 prices 逐股累積的滾動指標（均線/
波動率/暖身期天數）自然算不到那些還沒加入指數的日子，不需要在每個策略
訊號函式裡另外加判斷式——一檔股票剛加入指數時，也會像真實情況一樣需要
累積一段暖身期才夠格被選進魚池，不是加入當天就能立刻被排名。

★ 不解決的部分：被踢出指數、已經不在「今天的成分股名單」裡的股票依然
完全抓不到資料，這裡沒辦法無中生有——需要另一個時間點成分股名單的資料
來源（維基百科目前的頁面沒有可用的歷史異動紀錄表，已確認過，見對話紀錄）
才能补上這一半。
"""

from __future__ import annotations

import pandas as pd


def filter_prices_by_index_membership(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """membership 須有 stock_id、date_added 兩欄（見
    tw_quant.storage.DataStore.load_us_index_membership）。

    沒有加入日期紀錄的股票（date_added 缺值、或 stock_id 根本不在
    membership 裡）視為「一直都在指數裡」，不過濾——缺資料不代表排除，
    保守起見寧可不誤殺。
    """
    if membership.empty:
        return prices

    date_added = membership.dropna(subset=["date_added"]).set_index("stock_id")["date_added"]
    if date_added.empty:
        return prices

    added_on = prices["stock_id"].map(date_added)
    eligible = added_on.isna() | (prices["date"] >= added_on)
    return prices[eligible].reset_index(drop=True)
