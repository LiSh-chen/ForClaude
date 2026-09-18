"""雙動能（動量選股 + QQQ 自身趨勢濾網）：能不能真的打敗 QQQ 買進持有。

背景：修正存活者偏差後（見 docs/research_findings.md 第10.5節「2026-09-18
更新」），純動量選股策略的優勢大幅縮水——樣本外驗證的最佳組合甚至已經
輸給 QQQ 買進持有（總報酬 52.00% vs 107.60%）。根本問題是下檔風險沒被
控制：2022 熊市這類系統性下跌期間，動量策略照樣全倉持有一籃子「上一輪
最強勢」的股票一起跳水，樣本外 MDD（40.20%）反而比 QQQ 買進持有本身
（35.12%）更差。

這裡加一層 Gary Antonacci 式的「絕對動能」濾網：用 QQQ 自身的價格趨勢
（收盤價 vs N 日均線）判斷大盤處於多頭還是空頭，只有多頭時才做原本的
相對動能選股，空頭時全部出清換成現金（保守假設現金報酬率 0%，沒有計入
無風險利率，對這個策略不利但更誠實）。這個濾網只依賴 QQQ 自己的歷史
股價，不受 S&P 500 成分股清單的存活者偏差影響。

動量選股參數刻意沿用一組合理但沒有被挑選成「歷史最佳」的中庸參數
（momentum_window=126 約半年、rebalance_freq_days=21 約月調倉、
top_n=20 適度分散，不是網格搜尋出來的最佳解），避免疊加另一層選擇偏誤；
只針對「趨勢濾網的均線天數」這一個新變數做網格測試，這樣才是乾淨地
驗證「加趨勢濾網有沒有用」這個單一假說，而不是又在重新調一次參數。

反未來函數：趨勢濾網用 shift(1) 後的 QQQ 收盤價 vs 均線（T-1 為止已知
的資訊才能決定 T 日該不該進場），動量排名沿用 factor_backtest 既有的
shift(1) 邏輯，一樣永遠傳完整 us_prices（不切片），只用 start_date/
end_date 限制「哪些日期允許實際調倉」。

用法：
    python scripts/test_us_dual_momentum_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider
from tw_quant.us_universe import filter_prices_by_index_membership

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N = 20
TREND_MA_GRID = (50, 100, 150, 200)

IN_SAMPLE_START = "2023-09-19"
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

HEADER = (
    f"{'trend_ma':>9} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'n_bear_days':>12}"
)


def make_trend_filtered_momentum_signal_fn(qqq: pd.DataFrame, trend_ma_window: int, momentum_window: int = MOM_WINDOW):
    """回傳一個 signal_fn：多頭日沿用預設的落後報酬率動量排名，空頭日
    對全部股票回傳 NaN（讓 run_factor_backtest 在該次調倉日把持股全部
    出清、不買進新部位，等同持有現金）。momentum_window 開放覆寫（預設
    MOM_WINDOW）方便用短天期合成資料測試，不用真的準備 126 天以上的假資料。
    """
    qqq_sorted = qqq.sort_values("date").reset_index(drop=True)
    qqq_ma = qqq_sorted["close"].rolling(trend_ma_window).mean()
    is_bull = qqq_sorted["close"] > qqq_ma
    bull_by_date = pd.Series(is_bull.values, index=qqq_sorted["date"]).shift(1)  # T-1 資訊才知道 T 日該不該進場

    def signal_fn(master: pd.DataFrame) -> pd.Series:
        ret = master.groupby("stock_id", sort=False)["close"].pct_change(momentum_window)
        momentum = ind.shift_by_group(ret, master, periods=1)
        bull_today = master["date"].map(bull_by_date).fillna(False)
        return momentum.where(bull_today.values, other=pd.NA)

    return signal_fn


def _fmt_row(trend_ma: int | str, m: dict, n_bear_days: int | str = "") -> str:
    return (
        f"{trend_ma!s:>9} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {n_bear_days!s:>12}"
    )


def _run(us_prices: pd.DataFrame, base_cfg, signal_fn, start_date, end_date) -> dict:
    factor_cfg = FactorConfig(
        momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=False
    )
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date,
        signal_fn=signal_fn, cost_module=us_costs,
    )
    return metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)


def _print_period(label: str, us_prices, base_cfg, qqq, start_date, end_date, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)

    baseline = _run(us_prices, base_cfg, None, start_date, end_date)
    print(_fmt_row("無濾網", baseline, "-"))

    for ma_window in TREND_MA_GRID:
        qqq_ma = qqq["close"].rolling(ma_window).mean()
        is_bull = qqq["close"] > qqq_ma
        n_bear_days = int((~is_bull).sum())
        signal_fn = make_trend_filtered_momentum_signal_fn(qqq, ma_window)
        m = _run(us_prices, base_cfg, signal_fn, start_date, end_date)
        print(_fmt_row(ma_window, m, n_bear_days))

    print(
        f"\n（對照：QQQ 買進持有同期間總報酬 {qqq_bench['total_return']:.2%}、"
        f"CAGR {qqq_bench['cagr']:.2%}、MDD {qqq_bench['max_dd']:.2%}、"
        f"Sharpe {qqq_bench['sharpe']:.2f}、Calmar {qqq_bench['calmar']:.2f}）"
    )


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
        f"固定動量參數（非調參最佳解，避免疊加選擇偏誤）：momentum_window={MOM_WINDOW}、"
        f"rebalance_freq_days={REBALANCE_FREQ_DAYS}、top_n={TOP_N}；"
        f"趨勢濾網均線天數網格：{TREND_MA_GRID}"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期，跟原始 QQQ 對照同期間）", us_prices, base_cfg, qqq, IN_SAMPLE_START, None, QQQ_IN_SAMPLE)
    _print_period("樣本外（挑動量參數/濾網網格時都沒看過的更早期間，2018-09-20 ~ 2023-09-18）", us_prices, base_cfg, qqq, None, OOS_END, QQQ_OOS)

    print(
        "\n（誠實揭露：趨勢濾網只在每次調倉日檢查一次，regime 中途翻轉要等到下次調倉\n"
        "才會反映；現金部位報酬率算 0%，沒有計入短期公債等無風險利率，對這個策略\n"
        "是保守而非有利的假設；trend_ma_window 網格雖然只有 4 個值、不算大規模調參，\n"
        "但終究是事後看著回測結果選出來的，樣本外那欄的數字才是比較公允的判斷依據）"
    )


if __name__ == "__main__":
    main()
