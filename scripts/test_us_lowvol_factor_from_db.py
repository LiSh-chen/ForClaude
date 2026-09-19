"""低波動因子策略：買近期日報酬波動度最低的股票，能不能打敗 QQQ 買進持有。

背景：10.5~10.8 節測過的動量、雙動能、各種調倉頻率，樣本外驗證全部打不過
QQQ 買進持有。這裡換一個完全不同方向的因子——低波動異常（low-volatility
anomaly）：學術文獻裡記載得很久的一個現象，波動度最低的一籃子股票長期
風險調整後報酬經常優於大盤，尤其在下跌/盤整期間相對抗跌；代價通常是在
像 2023~2026 這種由少數幾檔巨型科技/AI股帶動、高波動股漲更多的牛市裡，
容易明顯跑輸大盤。這裡直接拿美股 S&P 500 真實資料檢驗這個抵換在這個
資料集上成不成立。

「品質」因子沒有一起做：第11節已經確認 yfinance 免費 tier 的季度財報
（季營收、獲利能力等基本面資料）只回傳最近 5~7 季，撐不起一個 3~8 年的
回測窗格，這個限制對 ROE/負債比/毛利率這些品質因子指標同樣適用——目前
沒有可行的資料來源，這裡誠實跳過，不做假的、資料撐不住的「品質」代理
指標。

不需要另外寫複雜訊號：跟動量一樣重用 tw_quant.factor_backtest.
run_factor_backtest 的 signal_fn 掛鉤機制，只是排名依據換成「T-1 為止
vol_window 日的日報酬率標準差」，ascending=True（買排名最低，也就是
波動度最低）而不是動量的 ascending=False。

動量選股的教訓（10.5節存活者偏差、10.7節樣本內好看/樣本外否證）這裡
從一開始就套用：直接用存活者偏差已修正的 universe，且動量網格的
rebalance_freq_days 固定用最初始未經調參挑選的月調倉（21天），只針對
vol_window 這一個新變數做網格測試（63/126/252 天），避免又疊加一層
事後調參的選擇偏誤。

反未來函數：跟前面幾個腳本一樣，永遠傳完整 us_prices（不切片），只用
start_date/end_date 限制「哪些日期允許實際調倉」，波動度排名計算永遠用
完整歷史，不會因切片重新累積暖身期；排名依據本身用 shift(1) 位移，
T 日開盤前已知的資訊才會被用到。

用法：
    python scripts/test_us_lowvol_factor_from_db.py
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
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.us_config import build_us_config
from tw_quant.us_universe import filter_prices_by_index_membership

VOL_WINDOW_GRID = (63, 126, 252)  # 約季/半年/年，跟動量網格常見的天數對應
REBALANCE_FREQ_DAYS = 21  # 月調倉，非調參挑選的預設值
TOP_N = 20
MIN_TRADES_FOR_RANKING = 5

IN_SAMPLE_START = "2023-09-19"
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

HEADER = (
    f"{'vol_win':>8} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7}"
)


def make_low_vol_signal_fn(vol_window: int):
    """回傳排名依據為「T-1 為止 vol_window 日日報酬率標準差」的 signal_fn，
    數值越小代表波動度越低；搭配 FactorConfig(ascending=True) 買排名最低
    （波動度最低）的股票。
    """

    def signal_fn(master: pd.DataFrame) -> pd.Series:
        daily_ret = master.groupby("stock_id", sort=False)["close"].pct_change()
        vol = daily_ret.groupby(master["stock_id"], sort=False).transform(lambda s: s.rolling(vol_window).std())
        return ind.shift_by_group(vol, master, periods=1)

    return signal_fn


def _fmt_row(vol_win: int, m: dict) -> str:
    return (
        f"{vol_win:>8} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%}"
    )


def _run(us_prices: pd.DataFrame, base_cfg, vol_window: int, start_date, end_date) -> dict:
    # momentum_window 這裡沒有作用——signal_fn 有給值時 run_factor_backtest 完全
    # 不會用到 factor_cfg.momentum_window，波動度窗格由 make_low_vol_signal_fn 的
    # 參數決定，維持預設值只是避免混淆成「這裡也在用動量窗格」
    factor_cfg = FactorConfig(rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=True)
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date,
        signal_fn=make_low_vol_signal_fn(vol_window), cost_module=us_costs,
    )
    return metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)


def _print_period(label: str, us_prices, base_cfg, start_date, end_date, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")
    print(HEADER)
    for vol_win in VOL_WINDOW_GRID:
        m = _run(us_prices, base_cfg, vol_win, start_date, end_date)
        print(_fmt_row(vol_win, m))
    print(
        f"\n（對照：QQQ 買進持有同期間總報酬 {qqq_bench['total_return']:.2%}、"
        f"CAGR {qqq_bench['cagr']:.2%}、MDD {qqq_bench['max_dd']:.2%}、"
        f"Sharpe {qqq_bench['sharpe']:.2f}、Calmar {qqq_bench['calmar']:.2f}）"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
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
        f"固定參數：rebalance_freq_days={REBALANCE_FREQ_DAYS}（月調倉，非調參挑選）、top_n={TOP_N}、"
        f"排名依據=日報酬率標準差（ascending=True，買波動度最低）；波動度計算窗格網格：{VOL_WINDOW_GRID} 個交易日"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期，跟原始 QQQ 對照同期間）", us_prices, base_cfg, IN_SAMPLE_START, None, QQQ_IN_SAMPLE)
    _print_period("樣本外（挑波動度窗格網格時完全沒看過的更早期間，2018-09-20 ~ 2023-09-18）", us_prices, base_cfg, None, OOS_END, QQQ_OOS)

    print(
        "\n（誠實揭露：低波動因子的經典抵換是「風險調整後報酬較優、但在集中度極高的\n"
        "牛市裡容易跑輸大盤」——2023~2026 這段期間 QQQ 的漲幅集中在少數幾檔高波動的\n"
        "巨型科技/AI股，正是低波動策略理論上最吃虧的環境，樣本內結果要用這個背景\n"
        "去理解；「品質」因子因為 yfinance 免費 tier 財報資料窗格太短（見第11節），\n"
        "沒有一起測，不是被否證，是目前的資料撐不住這個測試）"
    )


if __name__ == "__main__":
    main()
