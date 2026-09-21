"""把 test_us_dual_momentum_from_db.py 已經驗證過的「QQQ 自身趨勢濾網
（絕對動能）」機制，接到報告主要引用的 top_n=3 參數上，測試涵蓋 2008
金融風暴的延伸期間——直接回答「這個策略如果實際上線，遇到系統性市場
反轉時，有沒有簡單的停損/防禦機制可以用」這個問題。

背景：2026-09-20/21 對話紀錄——把 top_n=3/momentum_window=126/
rebalance_freq_days=21（報告主要引用的那組參數）延伸到 2006-2018 年
（涵蓋 2008 危機），無濾網版本結果：
  危機核心期 2007-10~2009-12：total_ret -45.71%、max_dd 77.21%（QQQ 同期
    -11.12%/53.40%，策略大幅跑輸且回撤更深）
  延伸樣本外 2006-09~2018-09：total_ret 51.56%、max_dd 79.04%（QQQ 同期
    400.24%/53.40%）
使用者判斷這段資料存活者偏差太重（2009-03-09 股票池只剩 13 檔）、參考性
偏低，但問題本身成立：動能策略買「上一輪最強勢」的股票，市場系統性反轉
時要等到下次調倉（最多 21 個交易日）才有機會換股，這段期間毫無防禦。這裡
測試 test_us_dual_momentum_from_db.py 已經驗證過的「QQQ 收盤價 vs N 日
均線」濾網（多頭日才做原本的相對動能選股，空頭日對全部股票回傳 NaN，
讓 run_factor_backtest 在該次調倉日全部出清換現金）能不能改善這個問題。

跟 test_us_dual_momentum_from_db.py 的差異：那支腳本刻意用中庸參數
（top_n=20，避免疊加選擇偏誤去驗證「濾網本身有沒有用」這個單一假說），
這裡刻意反過來——直接套用報告實際引用、面向使用者的 top_n=3，因為我們
要回答的是「這個會拿去用的策略，加上濾網後表現如何」，不是方法論驗證，
兩支腳本目的不同、不算重複造輪子。

反未來函數：延續 factor_backtest 的既有設計——動量排名跟趨勢濾網都用
shift(1)、永遠傳完整 us_prices（不切片），只用 start_date/end_date
限制「哪些日期允許實際調倉」。

★ 存活者偏差警語沿用（見 test_us_momentum_2008_crisis_from_db.py 檔頭）：
2008 年代被剔除指數、yfinance 查無資料的公司完全不在資料庫裡，這裡的
危機核心期數字仍然只能當方向性參考。

用法：
    python scripts/test_us_momentum_top3_trend_filter_2008_crisis_from_db.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_us_dual_momentum_from_db import make_trend_filtered_momentum_signal_fn  # noqa: E402

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N = 3
TREND_MA_GRID = (50, 100, 150, 200)

HEADER = (
    f"{'trend_ma':>9} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'n_bear_days':>12}"
)


def _fmt_row(trend_ma, m: dict, n_bear_days) -> str:
    return (
        f"{trend_ma!s:>9} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {n_bear_days!s:>12}"
    )


def _run(us_prices: pd.DataFrame, base_cfg, signal_fn, start_date, end_date) -> dict:
    factor_cfg = FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=False)
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, signal_fn=signal_fn, cost_module=us_costs,
    )
    return metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)


def _print_period(label: str, us_prices, base_cfg, qqq, start_date, end_date) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)

    baseline = _run(us_prices, base_cfg, None, start_date, end_date)
    print(_fmt_row("無濾網", baseline, "-"))

    for ma_window in TREND_MA_GRID:
        qqq_ma = qqq["close"].rolling(ma_window).mean()
        is_bull = qqq["close"] > qqq_ma
        s = pd.Timestamp(start_date) if start_date else qqq["date"].min()
        e = pd.Timestamp(end_date) if end_date else qqq["date"].max()
        window_mask = (qqq["date"] >= s) & (qqq["date"] <= e)
        n_bear_days = int((~is_bull[window_mask]).sum())
        signal_fn = make_trend_filtered_momentum_signal_fn(qqq, ma_window, momentum_window=MOM_WINDOW)
        m = _run(us_prices, base_cfg, signal_fn, start_date, end_date)
        print(_fmt_row(ma_window, m, n_bear_days))

    window_close = qqq[(qqq["date"] >= (pd.Timestamp(start_date) if start_date else qqq["date"].min())) & (qqq["date"] <= (pd.Timestamp(end_date) if end_date else qqq["date"].max()))].sort_values("date")["close"]
    if len(window_close) >= 2:
        qqq_total_ret = float(window_close.iloc[-1] / window_close.iloc[0] - 1)
        running_max = window_close.cummax()
        qqq_max_dd = float(((window_close - running_max) / running_max).min())
        print(f"\n（對照：QQQ 買進持有同期間 total_ret={qqq_total_ret:.2%}、max_dd={qqq_max_dd:.2%}）")


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    n_rows_before = len(us_prices)
    us_prices = filter_prices_by_index_membership(us_prices, membership)
    n_rows_dropped = n_rows_before - len(us_prices)
    print(
        f"存活者偏差部分修正：依指數加入/剔除區間過濾後，丟掉 {n_rows_dropped} / {n_rows_before} 列"
        f"（{n_rows_dropped / n_rows_before:.1%}）——2008 年代被剔除指數、查無資料的公司仍然不在資料庫裡\n"
    )

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {latest.date()}）")

    print("抓取 QQQ 價格歷史作為趨勢濾網（外部序列，不寫入 us_prices，不影響選股魚池）...")
    provider = YFinanceUSDataProvider()
    qqq = provider.fetch_price(
        "QQQ", start_date=str(earliest.date()), end_date=str((latest + pd.Timedelta(days=1)).date()), industry="ETF"
    )
    if qqq.empty:
        print("QQQ 資料抓取失敗，中止。", file=sys.stderr)
        sys.exit(1)
    qqq = qqq.sort_values("date").reset_index(drop=True)
    print(f"QQQ 資料：{qqq['date'].min().date()} ~ {qqq['date'].max().date()}，{len(qqq)} 筆\n")

    base_cfg = build_us_config()
    print(
        f"固定參數（報告主要引用的組合，非重新調參）：momentum_window={MOM_WINDOW}、"
        f"rebalance_freq_days={REBALANCE_FREQ_DAYS}、top_n={TOP_N}；趨勢濾網均線天數網格：{TREND_MA_GRID}"
    )

    windows = [
        ("危機核心期 2007-10~2009-12", "2007-10-01", "2009-12-31"),
        (f"延伸樣本外 {earliest.date()}~2018-09-19", None, "2018-09-19"),
        ("既有樣本外 2018-09-20~2023-09-18", "2018-09-20", "2023-09-18"),
        ("樣本內 2023-09-19~資料庫最新日期", "2023-09-19", None),
    ]
    for label, start, end in windows:
        _print_period(label, us_prices, base_cfg, qqq, start, end)

    print(
        "\n（誠實揭露：趨勢濾網只在每次調倉日檢查一次，regime 中途翻轉要等到下次調倉才會\n"
        "反映；現金部位報酬率算 0%，沒有計入無風險利率，對這個策略是保守假設；\n"
        "危機核心期跟延伸樣本外兩段仍然背負 2008 年代存活者偏差未解決的限制（見\n"
        "test_us_momentum_2008_crisis_from_db.py 檔頭），trend_ma 能不能把回撤壓下來，\n"
        "這裡看到的效果可能被同一個限制放大或縮小，不是乾淨的因果結論）"
    )


if __name__ == "__main__":
    main()
