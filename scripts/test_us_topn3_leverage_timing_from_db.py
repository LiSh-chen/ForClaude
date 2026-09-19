"""top_n=3 動量策略加開槓桿時機的分析：固定槓桿 vs 依波動度動態調整
槓桿（vol-targeting），會不會比不開槓桿更好、又會不會被保證金追繳
強制斷頭。

背景：使用者問「機構的槓桿時機」能不能學，討論後決定先只在單一策略
（top_n=3，10.11 節樣本內外表現最一致的候選人）上驗證，不要一次套用
到好幾個策略——選項越多，越容易純粹因為運氣挑到好看的組合，這是這
整個研究系列反覆強調的教訓，開槓桿這種風險更高的變體更要謹慎。

三種情境比較：
  1. 不開槓桿（對照組，就是 10.11 節本來的 top_n=3 結果）
  2. 固定 1.5 倍槓桿（天真作法，槓桿倍數不隨市況調整）
  3. 依波動度動態調整槓桿（vol-targeting：年化波動度目標 20%，
     槓桿夾在 0.5~2.0 倍之間，波動度越高槓桿越低，最極端時甚至降到
     0.5 倍等於保留一半現金）

槓桿模擬用 tw_quant/leverage.py：投資組合層級的簡化模擬（把 top_n=3
本身已經算出來的逐日報酬率序列複利 + 疊加槓桿 + 借款利息 + 每日檢查
維持保證金率，不足時強制減碼回 1 倍），不是逐股層級的借股模擬，細節
與假設（借款年利率 8%、維持保證金率 30%）見 tw_quant/leverage.py
開頭的完整說明。這兩個假設都是文獻上常見的量級估計，不是這個資料集
量測出來的真實數字，真實情況依券商/帳戶規模可能有不小差異。

反未來函數：top_n=3 本身的動量排名跟前面所有腳本一樣，永遠傳完整
us_prices（不切片），只用 start_date/end_date 限制交易日期；槓桿
時機的波動度估計用 shift(1)，T 日的槓桿只會用到 T-1 為止已知的
報酬率（見 tw_quant/leverage.py 的 vol_target_leverage_series）。

用法：
    python scripts/test_us_topn3_leverage_timing_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.leverage import LeverageConfig, metrics_from_equity_curve, simulate_leveraged_equity, vol_target_leverage_series
from tw_quant.us_config import build_us_config
from tw_quant.us_universe import filter_prices_by_index_membership

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N = 3

CONSTANT_LEVERAGE = 1.5
LEV_CFG = LeverageConfig(
    target_vol=0.20, vol_lookback_days=21, min_leverage=0.5, max_leverage=2.0,
    maintenance_margin_pct=0.30, annual_borrow_rate=0.08, releverage_freq_days=REBALANCE_FREQ_DAYS,
)

IN_SAMPLE_START = "2023-09-19"
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

HEADER = (
    f"{'情境':>18} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'追繳次數':>8} {'最低維持率':>10} {'期末槓桿':>8}"
)


def _fmt_row(label: str, m: dict, n_calls: int | str, min_ratio: float | str, end_lev: float | str) -> str:
    n_calls_s = f"{n_calls}" if isinstance(n_calls, int) else n_calls
    min_ratio_s = f"{min_ratio:.1%}" if isinstance(min_ratio, float) else min_ratio
    end_lev_s = f"{end_lev:.2f}" if isinstance(end_lev, float) else end_lev
    return (
        f"{label:>18} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{n_calls_s:>8} {min_ratio_s:>10} {end_lev_s:>8}"
    )


def _print_period(label: str, us_prices, base_cfg, start_date, end_date, qqq_bench: dict) -> None:
    print(f"\n=== {label} ===")

    factor_cfg = FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=False)
    result = run_factor_backtest(us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs)
    eq = result.equity_curve["equity"].reset_index(drop=True)
    dates = pd.Series(result.equity_curve.index)
    daily_returns = eq.pct_change().fillna(0.0)

    print(HEADER)

    m_base = metrics_from_equity_curve(eq)
    print(_fmt_row("不開槓桿（對照組）", m_base, "-", "-", "-"))

    const_result = simulate_leveraged_equity(dates, daily_returns, CONSTANT_LEVERAGE, LEV_CFG, base_cfg.initial_capital)
    m_const = metrics_from_equity_curve(const_result["equity"])
    print(_fmt_row(
        f"固定{CONSTANT_LEVERAGE}倍槓桿", m_const,
        int(const_result["margin_call"].sum()), float(const_result["margin_ratio"].min()),
        float(const_result["leverage"].iloc[-1]),
    ))

    vol_target = vol_target_leverage_series(daily_returns, LEV_CFG)
    dyn_result = simulate_leveraged_equity(dates, daily_returns, vol_target, LEV_CFG, base_cfg.initial_capital)
    m_dyn = metrics_from_equity_curve(dyn_result["equity"])
    print(_fmt_row(
        "波動度動態槓桿", m_dyn,
        int(dyn_result["margin_call"].sum()), float(dyn_result["margin_ratio"].min()),
        float(dyn_result["leverage"].iloc[-1]),
    ))

    print(
        f"\n（對照：QQQ 買進持有同期間總報酬 {qqq_bench['total_return']:.2%}、"
        f"CAGR {qqq_bench['cagr']:.2%}、MDD {qqq_bench['max_dd']:.2%}、"
        f"Sharpe {qqq_bench['sharpe']:.2f}、Calmar {qqq_bench['calmar']:.2f}）"
    )

    if int(const_result["margin_call"].sum()) > 0:
        call_dates = dates[const_result["margin_call"]].dt.date.tolist()
        print(f"固定槓桿觸發保證金追繳的日期：{call_dates}")
    if int(dyn_result["margin_call"].sum()) > 0:
        call_dates = dates[dyn_result["margin_call"]].dt.date.tolist()
        print(f"動態槓桿觸發保證金追繳的日期：{call_dates}")

    print("\n動態槓桿倍數統計：", dyn_result["leverage"].describe()[["min", "25%", "50%", "75%", "max"]].to_dict())


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
        f"固定動量參數（10.11 節同一組，非重新調參）：momentum_window={MOM_WINDOW}、"
        f"rebalance_freq_days={REBALANCE_FREQ_DAYS}、top_n={TOP_N}\n"
        f"槓桿假設：固定槓桿={CONSTANT_LEVERAGE}倍；動態槓桿目標年化波動度={LEV_CFG.target_vol:.0%}、"
        f"範圍[{LEV_CFG.min_leverage},{LEV_CFG.max_leverage}]倍、每{LEV_CFG.releverage_freq_days}個交易日重新評估一次；"
        f"維持保證金率={LEV_CFG.maintenance_margin_pct:.0%}、借款年利率={LEV_CFG.annual_borrow_rate:.0%}"
        "（投資組合層級簡化模擬，不是逐股保證金重現，見 tw_quant/leverage.py 開頭說明）"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", us_prices, base_cfg, IN_SAMPLE_START, None, QQQ_IN_SAMPLE)
    _print_period("樣本外（2018-09-20 ~ 2023-09-18，公允的比較基準）", us_prices, base_cfg, None, OOS_END, QQQ_OOS)

    print(
        "\n（誠實揭露：這裡的槓桿模擬是投資組合層級的簡化（複利日報酬率+扣利息+\n"
        "每日檢查維持保證金率），不是逐股借股的精確重現；維持保證金率30%、借款\n"
        "年利率8%都是文獻常見量級的假設值，不是從這個資料集或真實券商合約量測\n"
        "出來的數字，真實條件可能讓這裡的結論整個翻過來（例如複委託帳戶的融資\n"
        "利率、對集中度過高部位的維持保證金要求，都可能比這裡假設的更嚴苛）；\n"
        "這裡只在 top_n=3 單一策略上驗證，不代表其他策略套用同樣的槓桿時機邏輯\n"
        "會有一樣的結論，範圍刻意不擴大）"
    )


if __name__ == "__main__":
    main()
