"""回測網頁化入口。把 tw_quant 既有的 6 種策略引擎包成互動網頁：選策略、
調參數、按下「執行回測」看結果，不用碰程式碼或觸發 GitHub Actions。

跑法：
  本機： streamlit run streamlit_app.py
        （不設定 DATABASE_URL 時會自動退回本機 SQLite data/tw_market.db，
        方便先驗證網頁本身能不能動；要接雲端 Postgres 就設環境變數
        DATABASE_URL=postgresql://...）
  Streamlit Community Cloud：把這個 repo 接到 streamlit.io，Main file path
        填 streamlit_app.py，在 App settings -> Secrets 填：
            DATABASE_URL = "postgresql://user:pass@host:5432/dbname"
        （跟 GitHub Actions 用的是同一個 DATABASE_URL，同一個資料庫，不用
        另外複製一份資料。）

這裡只負責頁面骨架（標題、側欄選單、共用參數）跟把選到的策略分派給
webapp/strategies.py 對應的 render 函式——實際的回測邏輯全部重用
tw_quant/ 底下既有、已經在 GitHub Actions 上驗證過的引擎，網頁本身
不重新實作任何策略邏輯。
"""

from __future__ import annotations

import streamlit as st

from webapp import strategies

st.set_page_config(page_title="台股/美股回測研究室", page_icon="📈", layout="wide")

st.title("📈 台股/美股量化策略回測研究室")
st.caption(
    "重用 docs/research_findings.md 記錄的既有回測引擎，讓你在網頁上直接調參數、重新驗證——"
    "所有數字都是即時對雲端資料庫現算，不是預先算好的靜態報告。"
)

STRATEGIES: dict[str, dict] = {
    "結構轉折 C（Market Structure Shift）": {"render": strategies.render_structure_shift_c, "markets": ["台股", "美股"]},
    "配對交易（Pairs Trading）": {"render": strategies.render_pairs_trading, "markets": ["台股", "美股"]},
    "RSI / 布林通道均值回歸": {"render": strategies.render_rsi_bollinger, "markets": ["台股", "美股"]},
    "PEAD 月營收意外漂移": {"render": strategies.render_pead_revenue_drift, "markets": None},
    "營收動能 + 價量突破組合": {"render": strategies.render_pead_breakout_combo, "markets": None},
    "股本效應比較（大股本 vs 小股本）": {"render": strategies.render_market_cap_effect, "markets": None},
}

with st.sidebar:
    st.header("策略設定")
    choice = st.selectbox("選擇策略", list(STRATEGIES.keys()))
    spec = STRATEGIES[choice]

    market = None
    if spec["markets"]:
        market = st.radio("市場", spec["markets"], horizontal=True)
    else:
        st.caption("此策略僅支援台股（依賴 FinMind 月營收/股本資料，美股目前沒有對應的資料管線）。")

    with st.expander("進階設定"):
        initial_capital = st.number_input(
            "起始資金", min_value=100_000.0, value=10_000_000.0, step=100_000.0,
            help="台股/美股預設都是 10,000,000，刻意保持一致方便跨市場比較（見 docs/research_findings.md 第1節）。",
        )

    st.divider()
    st.caption(
        "資料由 GitHub Actions 排程每日更新，這裡讀的是同一個雲端 Postgres——"
        "不會、也無法從這個網頁直接觸發資料回填（那一段仍然只能透過 GitHub Actions 執行）。"
    )

st.subheader(choice)

try:
    if market is not None:
        spec["render"](market, initial_capital)
    else:
        spec["render"](initial_capital)
except Exception as exc:  # noqa: BLE001 - 網頁層級的最後防線，把資料庫連線等錯誤轉成看得懂的訊息
    st.error(
        "執行回測時發生錯誤。如果是資料庫連線問題，請確認 Streamlit Cloud 的 "
        "App settings -> Secrets 有正確設定 DATABASE_URL；如果是本機執行，"
        "確認 data/tw_market.db 存在或已設定 DATABASE_URL。"
    )
    st.exception(exc)
