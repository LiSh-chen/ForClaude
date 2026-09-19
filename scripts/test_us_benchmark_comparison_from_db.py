"""擴大對照組：把 VOO（S&P 500）、VTI（美股全市場）、VT（全球股市）
跟 QQQ 一起比較，同時對照 10.11 節的動量持股檔數策略，找出目前為止
所有選項裡表現最好的「投資組合」。

背景：整個 10.1~10.13 節的探討都是拿 QQQ（那斯達克 100）當唯一的
買進持有對照組，但 QQQ 是集中在科技/成長股的指數，不是唯一合理的
被動投資選項——如果單純買 VOO（更廣泛的 S&P 500）、VTI（美國幾乎
所有上市公司）、甚至 VT（含國際股市，分散度最高）就已經能打敗或
接近 QQQ，那整個「主動選股能不能打敗大盤」的問題，答案可能取決於
選哪個大盤當對照組，不是只有 QQQ 這一個答案。

這裡做兩件事：
  1. 把 VOO/VTI/VT 個別的買進持有績效，用跟 QQQ 完全一樣的期間、
     完全一樣的指標公式（跟 tw_quant.backtest.summarize_performance
     一致）算出來，跟 QQQ、跟 10.11 節的動量策略（top_n=1/2/3/10）
     放在同一張表比較。
  2. 額外算一個「四檔 ETF 期初等金額分配、之後不再調整權重」的簡單
     被動組合（不是每日/每月重新平衡到等權重，是最初買進後就不動，
     權重會隨後續漲跌自然漂移，比較貼近「一次性建倉買進持有」的實際
     操作情境），看單純分散持有這四檔指數 ETF 本身，會不會就是比
     單押 QQQ 更好的答案。

反未來函數：跟前面所有腳本一樣，動量排名計算永遠用完整歷史，只用
start_date/end_date 限制交易日期；ETF 買進持有指標直接用收盤價序列
計算，不涉及排名或訊號，沒有未來函數的疑慮。

用法：
    python scripts/test_us_benchmark_comparison_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider
from tw_quant.us_universe import filter_prices_by_index_membership

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N_GRID = (1, 2, 3, 10)

ETF_TICKERS = ["QQQ", "VOO", "VTI", "VT"]
ETF_LABELS = {
    "QQQ": "QQQ（那斯達克100）", "VOO": "VOO（S&P 500）",
    "VTI": "VTI（美股全市場）", "VT": "VT（全球股市）",
}

IN_SAMPLE_START = "2023-09-19"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

HEADER = f"{'標的':>28} {'總報酬':>10} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Calmar':>7}"


def _buyhold_metrics(close: pd.Series) -> dict:
    """跟 tw_quant.backtest.summarize_performance 完全一樣的公式，直接吃
    收盤價序列（已排序、已限定日期範圍）。
    """
    eq = close.reset_index(drop=True)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    years = max(len(eq) / 252, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    running_peak = eq.cummax()
    max_dd = ((running_peak - eq) / running_peak).max()
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0
    return {"total_return": total_return, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar}


def _fmt_row(label: str, m: dict) -> str:
    return (
        f"{label:>28} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}"
    )


def _run_momentum(us_prices: pd.DataFrame, base_cfg, top_n: int, start_date, end_date) -> dict:
    factor_cfg = FactorConfig(
        momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=top_n, ascending=False
    )
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs
    )
    return metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)


def _print_period(label: str, us_prices, base_cfg, etf_closes: dict[str, pd.DataFrame], start_date, end_date) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)

    blend_components = []
    for ticker in ETF_TICKERS:
        df = etf_closes[ticker]
        # start_date/end_date 為 None 時代表「不設下限/上限」，用該檔 ETF 自己
        # 的資料範圍當邊界，不能直接丟給 pd.Timestamp(None)（會變成 NaT，
        # 任何比較都是 False，篩出空序列）
        lo = pd.Timestamp(start_date) if start_date is not None else df["date"].min()
        hi = pd.Timestamp(end_date) if end_date is not None else df["date"].max()
        s = df[(df["date"] >= lo) & (df["date"] <= hi)].sort_values("date")
        close = s["close"].reset_index(drop=True)
        m = _buyhold_metrics(close)
        print(_fmt_row(ETF_LABELS[ticker], m))
        blend_components.append(close / close.iloc[0])

    min_len = min(len(c) for c in blend_components)
    blend_index = sum(c.iloc[:min_len].reset_index(drop=True) for c in blend_components) / len(blend_components)
    blend_m = _buyhold_metrics(blend_index)
    print(_fmt_row("四檔ETF期初等額不再平衡", blend_m))

    print()
    for top_n in TOP_N_GRID:
        m = _run_momentum(us_prices, base_cfg, top_n, start_date, end_date)
        print(_fmt_row(f"動量策略 top_n={top_n}", m))


def main() -> None:
    store = get_data_store()
    us_prices = store.load_us_prices()
    membership = store.load_us_index_membership()

    if us_prices.empty:
        print("資料庫裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    n_rows_before = len(us_prices)
    us_prices = filter_prices_by_index_membership(us_prices, membership)
    n_rows_dropped = n_rows_before - len(us_prices)
    print(
        f"存活者偏差部分修正：依指數加入日期過濾後，丟掉 {n_rows_dropped} / {n_rows_before} 列"
        f"（{n_rows_dropped / n_rows_before:.1%}，不解決被剔除股票完全消失那一半，"
        "見 tw_quant/us_universe.py）\n"
    )

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {latest.date()}）")

    provider = YFinanceUSDataProvider()
    etf_closes = {}
    for ticker in ETF_TICKERS:
        print(f"抓取 {ticker} 價格歷史...")
        df = provider.fetch_price(
            ticker, start_date=str(earliest.date()), end_date=str((latest + pd.Timedelta(days=1)).date()), industry="ETF"
        )
        if df.empty:
            print(f"{ticker} 資料抓取失敗，中止。", file=sys.stderr)
            sys.exit(1)
        etf_closes[ticker] = df.sort_values("date").reset_index(drop=True)
    print()

    base_cfg = build_us_config()
    print(
        f"固定參數：momentum_window={MOM_WINDOW}、rebalance_freq_days={REBALANCE_FREQ_DAYS}"
        f"（跟 10.5/10.11 節一致，非調參挑選）；持股檔數：{TOP_N_GRID}"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", us_prices, base_cfg, etf_closes, IN_SAMPLE_START, None)
    _print_period("樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", us_prices, base_cfg, etf_closes, None, OOS_END)

    print(
        "\n（誠實揭露：「四檔ETF期初等額不再平衡」是期初各投入 1/4 資金、之後完全\n"
        "不調整權重的被動組合，不是每日/每月重新平衡到等權重，權重會隨後續漲跌\n"
        "自然漂移，比較貼近真實「一次性建倉買進持有」的操作；VOO/VTI/VT 的持股\n"
        "高度重疊（VTI 幾乎包含 VOO 的全部成分股，VT 又包含 VTI），三者報酬差異\n"
        "主要來自中小型股跟國際股的占比不同，不是三個獨立不相關的資產類別，這個\n"
        "混合組合的分散效果比看起來的還要有限）"
    )


if __name__ == "__main__":
    main()
