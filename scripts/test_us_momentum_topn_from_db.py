"""測試持股檔數（top_n）對動量策略的影響：更集中的動量組合能不能貼近 QQQ。

背景：QQQ 這幾年的漲幅集中在少數幾檔巨型科技/AI股（見 10.5 節開頭動機）。
10.5 節的原始網格只測過 top_n∈{10,20,30}，三者之間差距不算大，都是相對
分散的組合；10.10 節（等權重被動參照組）進一步確認：不主動選股、單純
分散持有魚池內所有合格股票，樣本外一樣遠遠落後 QQQ——這暗示「分散持有」
這個大方向本身可能就難以貼近 QQQ 這種極端集中的報酬曲線。

這裡直接測試持股檔數這一個維度，把範圍拉得更極端：1/2/3/5 檔（非常集中，
賭少數幾檔真正的動量龍頭，top_n=1 是單押一檔的極端情況）一路到 50 檔
（比原本測過的 30 檔更分散），看報酬曲線的形狀有沒有隨著集中度提高而
更貼近 QQQ——集中度越高，理論上越可能捕捉到「少數贏家帶動指數」的效果，
但代價通常是波動度、換手時的市場衝擊成本、單一持股權重過高的風險都會
上升。

第二輪加了 1、2 檔：第一輪（3~50 檔）的結果顯示 top_n=3 樣本外 Sharpe
（0.55）明顯優於其他檔數，是整個 10.5~10.11 節裡第二高的樣本外 Sharpe，
只是 MDD 沒有跟著變好、單一持股權重已經達 1/3（見 10.11 節）——這裡
繼續往更極端的方向測，看這個「集中度越高越好」的趨勢會不會延續到單押
1、2 檔，還是在某個檔數之後開始反轉（過度集中導致單一選錯股的風險
壓過額外報酬）。

動量參數維持 momentum_window=126、rebalance_freq_days=21（10.5 節原始
網格裡的中段值，跟 10.7~10.9 節維持一致，不是重新調參挑出來的最佳解），
只針對 top_n 這一個新變數做網格測試，避免再疊加一層選擇偏誤。

反未來函數：跟前面幾個腳本一樣，永遠傳完整 us_prices（不切片），只用
start_date/end_date 限制「哪些日期允許實際調倉」，動量排名計算永遠用
完整歷史。

用法：
    python scripts/test_us_momentum_topn_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store
from tw_quant.us_config import build_us_config
from tw_quant.us_universe import filter_prices_by_index_membership

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N_GRID = (1, 2, 3, 5, 10, 20, 30, 50)

IN_SAMPLE_START = "2023-09-19"
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

HEADER = (
    f"{'top_n':>6} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7}"
)


def _fmt_row(top_n: int, m: dict) -> str:
    return (
        f"{top_n:>6} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%}"
    )


def _run(us_prices: pd.DataFrame, base_cfg, top_n: int, start_date, end_date) -> dict:
    factor_cfg = FactorConfig(
        momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=top_n, ascending=False
    )
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs
    )
    return metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)


def _print_period(label: str, us_prices, base_cfg, start_date, end_date, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)
    for top_n in TOP_N_GRID:
        m = _run(us_prices, base_cfg, top_n, start_date, end_date)
        print(_fmt_row(top_n, m))
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

    base_cfg = build_us_config()

    print(
        f"固定參數：momentum_window={MOM_WINDOW}、rebalance_freq_days={REBALANCE_FREQ_DAYS}"
        f"（10.5 節原始網格中段值，非調參挑選）；持股檔數網格：{TOP_N_GRID}"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期，跟原始 QQQ 對照同期間）", us_prices, base_cfg, IN_SAMPLE_START, None, QQQ_IN_SAMPLE)
    _print_period("樣本外（挑持股檔數網格時完全沒看過的更早期間，2018-09-20 ~ 2023-09-18）", us_prices, base_cfg, None, OOS_END, QQQ_OOS)

    print(
        "\n（誠實揭露：top_n 越小，單一持股權重越高——top_n=1 是單押一檔全倉、\n"
        "top_n=2 是對半分——真實下單時的市場衝擊成本、個股風險集中度都會遠高於\n"
        "這裡的回測假設（複委託固定手續費模型沒有模擬大額單一持股的滑價，更沒有\n"
        "模擬單一個股基本面利空對整個投資組合的衝擊），數字要打更大的折扣看待；\n"
        "1~5 檔這種極端集中的組合，樣本內外的表現對「每次調倉剛好選中/沒選中對\n"
        "的那 1~2 檔龍頭股」高度敏感，運氣成分比 20/30/50 檔的組合大得多，n_trd\n"
        "欄位也會因為只有 1~2 個持股名額而偏少，統計意義隨之下降）"
    )


if __name__ == "__main__":
    main()
