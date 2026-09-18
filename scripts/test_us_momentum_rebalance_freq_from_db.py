"""測試調倉頻率對「動量選股 + QQQ 趨勢濾網」策略的影響。

背景：10.7 節否證了加大盤趨勢濾網這個方向——樣本外驗證顯示濾網不但沒有
把 MDD 壓低多少（39.84% → 38.38%，只少 1.46 個百分點），總報酬、Sharpe
反而都比不加濾網更差。分析認為可能原因之一：濾網只在每次調倉日
（原本 rebalance_freq_days=21，約月調倉）才重新檢查一次 regime，中途從
多頭翻空頭要等到下個月調倉才反映，來不及躲開月中的急跌。

這裡直接測試這個假設：把調倉頻率拉高（5/10 個交易日，約週/雙週），讓
「動量排名」跟「趨勢濾網」都更頻繁地重新評估，看能不能真的改善下檔風險
控制；同時也對照調倉頻率本身（不管有沒有濾網）對純動量策略績效的影響——
調倉越頻繁換手成本通常越高，但理論上也可能更快跟上動量輪動、更快停損。

動量參數維持 momentum_window=126、top_n=20 不變，跟 10.5/10.7 節一致；
趨勢濾網固定用 150 日均線（10.7 節網格裡偏中段、不是特別挑出來表現最好
的那個，避免再疊加一層選擇偏誤）。調倉頻率網格：5/10/21/42/63/126/252
個交易日，分別約當週/雙週/月/雙月/季/半年/年調倉，5 跟 10 是第一輪新加的
（原本的動量網格只測過 21/63），126/252 是第二輪加的，用來檢驗「更頻繁
調倉」跟「更低頻調倉」兩個方向。

**低頻調倉的樣本數警語**：momentum_window 本身要 126 天暖身，樣本內窗格只有
約 3 年（~750 個交易日），rebalance_freq_days=252 意味著整個樣本內窗格只會
真的調倉 2~3 次；樣本外窗格約 5 年，252 天調倉也只有 4~5 次。調倉次數這麼少
時，總報酬/Sharpe 很大程度上取決於少數幾次進出場剛好卡在哪個時間點，統計
意義遠不如高頻調倉那幾組（動輒兩三百筆交易）可靠，結果要打更大的折扣看待。

反未來函數：跟前面幾個腳本一樣，永遠傳完整 us_prices（不切片），只用
start_date/end_date 限制「哪些日期允許實際調倉」。

用法：
    python scripts/test_us_momentum_rebalance_freq_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider
from tw_quant.us_universe import filter_prices_by_index_membership

from test_us_dual_momentum_from_db import MOM_WINDOW, TOP_N, make_trend_filtered_momentum_signal_fn  # noqa: E402

REBALANCE_FREQ_GRID = (5, 10, 21, 42, 63, 126, 252)
TREND_MA_FIXED = 150

IN_SAMPLE_START = "2023-09-19"
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

HEADER = (
    f"{'rebal_d':>8} {'filter':>8} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7}"
)


def _run(us_prices: pd.DataFrame, base_cfg, signal_fn, rebalance_freq_days: int, start_date, end_date) -> dict:
    factor_cfg = FactorConfig(
        momentum_window=MOM_WINDOW, rebalance_freq_days=rebalance_freq_days, top_n=TOP_N, ascending=False
    )
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date,
        signal_fn=signal_fn, cost_module=us_costs,
    )
    return metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)


def _fmt_row(rebal_d: int, label: str, m: dict) -> str:
    return (
        f"{rebal_d:>8} {label:>8} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%}"
    )


def _print_period(label: str, us_prices, base_cfg, qqq, start_date, end_date, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)

    trend_signal_fn = make_trend_filtered_momentum_signal_fn(qqq, TREND_MA_FIXED)

    for rebal_d in REBALANCE_FREQ_GRID:
        baseline = _run(us_prices, base_cfg, None, rebal_d, start_date, end_date)
        print(_fmt_row(rebal_d, "無濾網", baseline))
        filtered = _run(us_prices, base_cfg, trend_signal_fn, rebal_d, start_date, end_date)
        print(_fmt_row(rebal_d, "有濾網", filtered))

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
        f"固定參數：momentum_window={MOM_WINDOW}、top_n={TOP_N}、trend_ma_window={TREND_MA_FIXED}"
        f"（10.7 節網格裡的中段值，非挑選出來表現最好的那個）；調倉頻率網格：{REBALANCE_FREQ_GRID} 個交易日"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期，跟原始 QQQ 對照同期間）", us_prices, base_cfg, qqq, IN_SAMPLE_START, None, QQQ_IN_SAMPLE)
    _print_period("樣本外（挑參數/濾網網格時都沒看過的更早期間，2018-09-20 ~ 2023-09-18）", us_prices, base_cfg, qqq, None, OOS_END, QQQ_OOS)

    print(
        "\n（誠實揭露：調倉頻率越高，複委託每股固定手續費的累積影響越大——已經反映在\n"
        "報酬數字裡，不是額外要扣的成本；trend_ma_window 固定用單一值，沒有跟調倉頻率\n"
        "一起網格化，是為了控制變數只測「調倉頻率」這一個維度，但也代表沒有測試\n"
        "「調倉頻率 × 濾網均線天數」交叉組合裡可能更好的搭配；126/252 天這兩組低頻\n"
        "調倉的樣本數很小（樣本內窗格只夠真的調倉 2~3 次、樣本外窗格 4~5 次），結果\n"
        "統計意義遠不如高頻那幾組可靠，n_trd 欄位本身就是最直接的提醒）"
    )


if __name__ == "__main__":
    main()
