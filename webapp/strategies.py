"""6 種策略各自的參數 UI + 執行 + 結果展示。每個 render_* 函式對應側欄選單
一個項目，直接呼叫 tw_quant 既有的回測引擎，不重寫任何回測邏輯——這裡只
負責「把研究腳本裡原本寫死的常數換成使用者可調的輸入元件」。

所有策略都用同一組「已知最佳參數」當預設值（直接取自
docs/research_findings.md 記錄的結果），方便使用者先看到跟報告一致的
基準，再自己調整比較。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tw_quant import costs as tw_costs
from tw_quant import us_costs
from tw_quant.backtest import run_backtest
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.config import StrategyConfig
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.pairs_trading import PairsTradingConfig, run_pairs_trading_backtest
from tw_quant.us_config import build_us_config

from webapp import data as data_mod
from webapp.engines import (
    make_bollinger_signal_fn,
    make_breakout_combo_signal_fn,
    make_mss_signal_fn,
    make_rsi_signal_fn,
)

EMPTY_MARGIN_SHORT = pd.DataFrame(columns=["date", "stock_id", "margin_purchase_balance", "short_balance"])


def _base_config(market: str, initial_capital: float) -> StrategyConfig:
    cfg = build_us_config() if market == "美股" else StrategyConfig()
    cfg.initial_capital = initial_capital
    return cfg


def _cost_module(market: str):
    return us_costs if market == "美股" else tw_costs


def display_result(result, metrics: dict, key_prefix: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("總報酬", f"{metrics['total_return']:.2%}")
    c2.metric("CAGR", f"{metrics['cagr']:.2%}")
    c3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    c4.metric("MDD", f"{metrics['max_dd']:.2%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Calmar", f"{metrics['calmar']:.2f}")
    c6.metric("交易筆數", f"{metrics['n_trades']:.0f}")
    c7.metric("勝率", f"{metrics['win_rate']:.1%}")
    pf = metrics["profit_factor"]
    c8.metric("獲利因子 PF", "inf" if pf == float("inf") else f"{pf:.2f}")

    if not result.equity_curve.empty:
        st.line_chart(result.equity_curve["equity"])
    else:
        st.warning("沒有產生任何權益曲線資料點——請確認參數是否過於嚴格（例如魚池篩選條件全部不成立）。")

    with st.expander(f"完整指標與交易明細（共 {len(result.trades)} 筆）"):
        st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()})
        if not result.trades.empty:
            st.dataframe(result.trades, use_container_width=True)
            csv = result.trades.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下載交易明細 CSV", csv, file_name=f"{key_prefix}_trades.csv", mime="text/csv")
        else:
            st.write("沒有任何交易。")


# ---------------------------------------------------------------------------
# 策略一：結構轉折 C（Market Structure Shift + 量能確認）
# ---------------------------------------------------------------------------

def render_structure_shift_c(market: str, initial_capital: float) -> None:
    st.caption("跌破前波低點後、又收復前一段區間高點，且帶量確認——台股/美股共用同一套訊號邏輯。")
    prices = data_mod.load_tw_prices() if market == "台股" else data_mod.load_us_prices()
    if prices.empty:
        st.error(f"資料庫裡沒有任何{market}價量資料。")
        return
    margin_short = data_mod.load_tw_margin_short() if market == "台股" else EMPTY_MARGIN_SHORT
    st.caption(f"資料涵蓋：{data_mod.describe_coverage(prices)}")

    col1, col2, col3 = st.columns(3)
    downtrend_window = col1.slider("下跌結構窗格天數", 5, 60, 20)
    swing_window = col2.slider("突破前段高點窗格天數", 5, 40, 15)
    volume_mult = col3.slider("量能確認倍數", 1.0, 3.0, 1.3, 0.1)

    col4, col5, col6 = st.columns(3)
    atr_mult = col4.slider("吊燈停利 ATR 倍數", 1.5, 4.0, 2.5, 0.1)
    chandelier_lookback = col5.slider("吊燈停利窗格天數", 5, 30, 10)
    max_industry_pct = col6.slider("單一產業曝險上限", 0.05, 0.50, 0.30, 0.05)

    col7, col8 = st.columns(2)
    breadth_threshold = col7.slider("大盤廣度門檻（regime）", 0.10, 0.90, 0.25, 0.05)
    volume_ratio_threshold = col8.slider("大盤量能比門檻（regime）", 0.50, 2.00, 0.90, 0.05)

    if st.button("執行回測", key="run_mss", type="primary"):
        cfg = _base_config(market, initial_capital)
        cfg.regime.breadth_threshold = breadth_threshold
        cfg.regime.volume_ratio_threshold = volume_ratio_threshold
        cfg.global_risk.max_industry_exposure_pct = max_industry_pct
        cfg.sizing.atr_multiplier = atr_mult
        cfg.sizing.chandelier_lookback = chandelier_lookback

        signal_fn = make_mss_signal_fn(downtrend_window, swing_window, volume_mult)
        with st.spinner("回測執行中..."):
            result = run_backtest(
                prices, margin_short, cfg, entry_signal_fn=signal_fn, cost_module=_cost_module(market)
            )
            metrics = metrics_from_result(result, cfg.initial_capital, prices=prices)
        display_result(result, metrics, key_prefix=f"mss_{market}")


# ---------------------------------------------------------------------------
# 策略二：配對交易（Pairs Trading）
# ---------------------------------------------------------------------------

def render_pairs_trading(market: str, initial_capital: float) -> None:
    st.caption("同產業股票兩兩做共整合檢定，挑最顯著的配對做價差均值回歸交易。")
    prices = data_mod.load_tw_prices() if market == "台股" else data_mod.load_us_prices()
    if prices.empty:
        st.error(f"資料庫裡沒有任何{market}價量資料。")
        return
    st.caption(f"資料涵蓋：{data_mod.describe_coverage(prices)}")

    if market == "美股":
        st.warning(
            "美股 503 檔股票的共整合檢定運算量遠大於台股，即使用相關係數預篩+每產業候選數上限，"
            "仍可能需要數分鐘才能跑完，請耐心等候，不要重複點擊。"
        )

    col1, col2, col3 = st.columns(3)
    formation_window = col1.slider("形成期窗格天數", 60, 504, 252, 6)
    reformation_freq = col2.slider("換股週期（交易日）", 21, 126, 63, 7)
    top_n_pairs = col3.slider("每期最多交易配對數", 1, 30, 10)

    col4, col5, col6 = st.columns(3)
    entry_z = col4.slider("進場 |z| 門檻", 1.0, 4.0, 2.0, 0.1)
    exit_z = col5.slider("出場 |z| 門檻", 0.1, 2.0, 0.5, 0.1)
    stop_z = col6.slider("停損 |z| 門檻", 2.0, 6.0, 4.0, 0.1)

    col7, col8, col9 = st.columns(3)
    zscore_window = col7.slider("z-score 滾動窗格天數", 5, 60, 20)
    max_holding_days = col8.slider("最長持有天數", 5, 60, 20)
    coint_pvalue = col9.slider("共整合 p-value 門檻", 0.01, 0.10, 0.05, 0.01)

    default_corr = 0.6 if market == "美股" else 0.0
    default_cap = 25 if market == "美股" else 0
    col10, col11 = st.columns(2)
    pre_filter_corr = col10.slider(
        "相關係數預篩門檻（0 = 不篩選，完全窮舉）", 0.0, 0.9, default_corr, 0.05,
        help="美股股票池大，建議保留預設值避免共整合檢定跑太久；台股 200 多檔規模下關掉也還算得動。",
    )
    limit_per_industry = col11.checkbox("限制每產業候選數（依成交金額取前 N 大）", value=(market == "美股"))
    max_per_industry = None
    if limit_per_industry:
        max_per_industry = st.number_input("每產業最多候選檔數", min_value=5, max_value=100, value=default_cap or 25, step=5)

    if st.button("執行回測", key="run_pairs", type="primary"):
        cfg = _base_config(market, initial_capital)
        rt_cfg = PairsTradingConfig(
            formation_window=formation_window,
            reformation_freq_days=reformation_freq,
            top_n_pairs=top_n_pairs,
            coint_pvalue_threshold=coint_pvalue,
            zscore_window=zscore_window,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_holding_days=max_holding_days,
            pre_filter_min_abs_corr=pre_filter_corr,
            max_stocks_per_industry=max_per_industry,
        )
        with st.spinner("回測執行中，共整合檢定可能需要一段時間..."):
            result = run_pairs_trading_backtest(prices, cfg, rt_cfg, cost_module=_cost_module(market))
            metrics = metrics_from_result(result, cfg.initial_capital, prices=prices)
        display_result(result, metrics, key_prefix=f"pairs_{market}")


# ---------------------------------------------------------------------------
# 策略三：RSI 超賣 / 布林通道乖離反彈
# ---------------------------------------------------------------------------

def render_rsi_bollinger(market: str, initial_capital: float) -> None:
    st.caption("短期均值回歸：RSI 超賣或跌破布林通道下軌乖離越深，排名越前，等權重持有固定天數後換股。")
    prices = data_mod.load_tw_prices() if market == "台股" else data_mod.load_us_prices()
    if prices.empty:
        st.error(f"資料庫裡沒有任何{market}價量資料。")
        return
    st.caption(f"資料涵蓋：{data_mod.describe_coverage(prices)}")

    signal_type = st.radio("排名依據", ["RSI", "布林通道 z-score", "布林通道 z-score + 量能確認"], horizontal=True)

    col1, col2 = st.columns(2)
    if signal_type == "RSI":
        rsi_window = col1.slider("RSI 天數", 5, 30, 14)
        bb_window, volume_mult = 20, 1.5
    else:
        bb_window = col1.slider("布林通道天數", 10, 40, 20)
        volume_mult = col2.slider("量能確認倍數", 1.0, 3.0, 1.5, 0.1) if signal_type.endswith("量能確認") else 1.5
        rsi_window = 14

    col3, col4 = st.columns(2)
    hold_days = col3.slider("持有天數（調倉週期）", 1, 42, 5)
    top_n = col4.slider("持有檔數", 5, 50, 10)

    ascending = st.toggle("買排名最前段（超賣／均值回歸，預設）", value=True)
    st.caption("關閉則改買排名最後段（動量／追強勢）——跟原始均值回歸假設方向相反，純粹拿來對照。")

    if st.button("執行回測", key="run_rsi_bb", type="primary"):
        cfg = _base_config(market, initial_capital)
        if signal_type == "RSI":
            signal_fn = make_rsi_signal_fn(rsi_window)
        elif signal_type == "布林通道 z-score":
            signal_fn = make_bollinger_signal_fn(bb_window, False, volume_mult)
        else:
            signal_fn = make_bollinger_signal_fn(bb_window, True, volume_mult)

        factor_cfg = FactorConfig(rebalance_freq_days=hold_days, top_n=top_n, ascending=ascending)
        with st.spinner("回測執行中..."):
            result = run_factor_backtest(prices, cfg, factor_cfg, signal_fn=signal_fn, cost_module=_cost_module(market))
            metrics = metrics_from_result(result, cfg.initial_capital, prices=prices)
        display_result(result, metrics, key_prefix=f"rsi_bb_{market}")


# ---------------------------------------------------------------------------
# 策略四：PEAD 月營收意外漂移（僅台股，需要月營收資料）
# ---------------------------------------------------------------------------

def render_pead_revenue_drift(initial_capital: float) -> None:
    st.caption("月營收年增率明顯高於近半年平均水準（正向意外），買進意外程度排名最高的股票。僅台股——美股沒有對應的營收資料管線。")
    from scripts.test_pead_revenue_drift_from_db import attach_revenue_signal, revenue_signal_fn

    prices = data_mod.load_tw_prices()
    revenue = data_mod.load_tw_month_revenue()
    if prices.empty:
        st.error("資料庫裡沒有任何台股價量資料。")
        return
    if revenue.empty:
        st.error("資料庫裡沒有任何月營收資料。")
        return
    st.caption(f"價量資料涵蓋：{data_mod.describe_coverage(prices)}；有月營收資料的股票數：{revenue['stock_id'].nunique()}")
    st.caption("「營收意外」的定義（次月 15 號才算公開已知、用近 6 個月平均年增率當基準）為反未來函數安全假設，不開放調整。")

    col1, col2 = st.columns(2)
    rebalance = col1.slider("調倉週期（交易日）", 10, 90, 63, 1)
    top_n = col2.slider("持有檔數", 5, 40, 20)

    if st.button("執行回測", key="run_pead", type="primary"):
        cfg = _base_config("台股", initial_capital)
        with st.spinner("計算營收意外訊號並回測中..."):
            master_with_signal = attach_revenue_signal(prices, revenue)
            factor_cfg = FactorConfig(rebalance_freq_days=rebalance, top_n=top_n, ascending=False)
            result = run_factor_backtest(master_with_signal, cfg, factor_cfg, signal_fn=revenue_signal_fn)
            metrics = metrics_from_result(result, cfg.initial_capital, prices=prices)
        display_result(result, metrics, key_prefix="pead")


# ---------------------------------------------------------------------------
# 策略五：營收動能 + 價量突破組合（僅台股）
# ---------------------------------------------------------------------------

def render_pead_breakout_combo(initial_capital: float) -> None:
    st.caption("月營收 YoY>15% 或創 12 個月新高，加上當日價量突破確認。僅台股。")
    from scripts.test_pead_breakout_combo_from_db import attach_revenue_momentum_flag

    prices = data_mod.load_tw_prices()
    margin_short = data_mod.load_tw_margin_short()
    revenue = data_mod.load_tw_month_revenue()
    if prices.empty:
        st.error("資料庫裡沒有任何台股價量資料。")
        return
    if revenue.empty:
        st.error("資料庫裡沒有任何月營收資料。")
        return
    st.caption(f"資料涵蓋：{data_mod.describe_coverage(prices)}")
    st.info(
        "先前用系統既有參數組合測過：符合條件的候選很稀少（3 年裡約 56 筆），"
        "且容易在同一產業群聚，常常被既有的單一產業曝險上限（10~30%）全域風控擋掉，"
        "曾經測出全部 0 筆成交的結果——這裡讓你自己調參數看看能不能找到有成交的組合。",
        icon="ℹ️",
    )

    col1, col2, col3 = st.columns(3)
    breakout_window = col1.slider("價格突破窗格天數", 5, 60, 20)
    volume_mult = col2.slider("成交量突破倍數", 1.0, 4.0, 2.0, 0.1)
    atr_mult = col3.slider("吊燈停利 ATR 倍數", 1.5, 4.0, 2.0, 0.1)
    max_industry_pct = st.slider("單一產業曝險上限", 0.05, 0.50, 0.10, 0.05)

    if st.button("執行回測", key="run_breakout_combo", type="primary"):
        cfg = _base_config("台股", initial_capital)
        cfg.regime.breadth_threshold = 0.25
        cfg.regime.volume_ratio_threshold = 0.9
        cfg.sizing.atr_multiplier = atr_mult
        cfg.global_risk.max_industry_exposure_pct = max_industry_pct

        with st.spinner("計算營收動能旗標並回測中..."):
            master_with_revenue = attach_revenue_momentum_flag(prices, revenue)
            revenue_flag_lookup = master_with_revenue.set_index(["stock_id", "date"])["revenue_momentum_ok_shifted"]
            signal_fn = make_breakout_combo_signal_fn(breakout_window, volume_mult, revenue_flag_lookup)
            result = run_backtest(prices, margin_short, cfg, entry_signal_fn=signal_fn)
            metrics = metrics_from_result(result, cfg.initial_capital, prices=prices)
        display_result(result, metrics, key_prefix="breakout_combo")


# ---------------------------------------------------------------------------
# 策略六：股本效應比較（僅台股，依股本切成大/小兩組後對照）
# ---------------------------------------------------------------------------

def render_market_cap_effect(initial_capital: float) -> None:
    st.caption("把現有股票池依股本切成大/小兩組，同一個策略分別在兩組上跑，比較股本大小對績效的影響。僅台股。")
    from scripts.test_pead_revenue_drift_from_db import attach_revenue_signal, revenue_signal_fn
    from scripts.test_market_cap_effect_from_db import PAR_VALUE, compute_share_capital_yi

    prices = data_mod.load_tw_prices()
    margin_short = data_mod.load_tw_margin_short()
    revenue = data_mod.load_tw_month_revenue()
    shares_issued = data_mod.load_tw_shares_issued()
    if prices.empty:
        st.error("資料庫裡沒有任何台股價量資料。")
        return
    if shares_issued.empty:
        st.error("資料庫裡沒有已發行股數資料。")
        return

    share_capital_yi = compute_share_capital_yi(shares_issued)
    st.caption(
        f"有股本資料的股票數：{len(share_capital_yi)} / {prices['stock_id'].nunique()}"
        f"（股本 = 已發行股數 × 面額 {PAR_VALUE:.0f} 元 / 1 億，用資料庫最早一筆已發行股數）"
    )

    split_method = st.radio("切分方式", ["依中位數切成大/小兩半（預設）", "依絕對股本門檻（億元）"], horizontal=True)
    if split_method.startswith("依絕對"):
        threshold = st.number_input("股本門檻（億元，小於等於為小股本組）", min_value=1.0, value=50.0, step=5.0)
        small_ids = share_capital_yi[share_capital_yi <= threshold].index.tolist()
        large_ids = share_capital_yi[share_capital_yi > threshold].index.tolist()
    else:
        median = share_capital_yi.median()
        small_ids = share_capital_yi[share_capital_yi <= median].index.tolist()
        large_ids = share_capital_yi[share_capital_yi > median].index.tolist()

    st.caption(
        f"小股本組 {len(small_ids)} 檔（{share_capital_yi[small_ids].min():.1f} ~ {share_capital_yi[small_ids].max():.1f} 億）、"
        f"大股本組 {len(large_ids)} 檔（{share_capital_yi[large_ids].min():.1f} ~ {share_capital_yi[large_ids].max():.1f} 億）"
        if small_ids and large_ids
        else "切分後其中一組沒有股票，請調整門檻。"
    )

    strategy_choice = st.selectbox("要比較哪個策略", ["結構轉折 C", "PEAD 月營收意外漂移"])

    if strategy_choice == "結構轉折 C":
        col1, col2, col3 = st.columns(3)
        downtrend_window = col1.slider("下跌結構窗格天數", 5, 60, 20)
        swing_window = col2.slider("突破前段高點窗格天數", 5, 40, 15)
        volume_mult = col3.slider("量能確認倍數", 1.0, 3.0, 1.3, 0.1)
    else:
        col1, col2 = st.columns(2)
        rebalance = col1.slider("調倉週期（交易日）", 10, 90, 63, 1)
        top_n = col2.slider("持有檔數", 5, 40, 20)

    if st.button("執行回測", key="run_cap_effect", type="primary") and small_ids and large_ids:
        cfg = _base_config("台股", initial_capital)
        cfg.regime.breadth_threshold = 0.25
        cfg.regime.volume_ratio_threshold = 0.9
        cfg.global_risk.max_industry_exposure_pct = 0.30
        cfg.sizing.atr_multiplier = 2.5
        cfg.sizing.chandelier_lookback = 10

        universes = {"全部": None, "小股本組": small_ids, "大股本組": large_ids}
        rows = []
        with st.spinner("依三組股票池分別回測中..."):
            for label, ids in universes.items():
                sub_prices = prices if ids is None else prices[prices["stock_id"].isin(ids)]
                if strategy_choice == "結構轉折 C":
                    sub_margin = margin_short if ids is None else margin_short[margin_short["stock_id"].isin(ids)]
                    signal_fn = make_mss_signal_fn(downtrend_window, swing_window, volume_mult)
                    result = run_backtest(sub_prices, sub_margin, cfg, entry_signal_fn=signal_fn)
                else:
                    sub_revenue = revenue if ids is None else revenue[revenue["stock_id"].isin(ids)]
                    master_with_signal = attach_revenue_signal(sub_prices, sub_revenue)
                    factor_cfg = FactorConfig(rebalance_freq_days=rebalance, top_n=top_n, ascending=False)
                    result = run_factor_backtest(master_with_signal, cfg, factor_cfg, signal_fn=revenue_signal_fn)
                m = metrics_from_result(result, cfg.initial_capital, prices=sub_prices)
                rows.append({"股票池": label, "檔數": sub_prices["stock_id"].nunique(), **m})

        df = pd.DataFrame(rows).set_index("股票池")
        display_cols = ["檔數", "total_return", "cagr", "max_dd", "sharpe", "calmar", "n_trades", "win_rate", "profit_factor"]
        st.dataframe(
            df[display_cols].style.format(
                {"total_return": "{:.2%}", "cagr": "{:.2%}", "max_dd": "{:.2%}", "sharpe": "{:.2f}", "calmar": "{:.2f}", "win_rate": "{:.1%}", "profit_factor": "{:.2f}"}
            ),
            use_container_width=True,
        )
        st.bar_chart(df["sharpe"])
