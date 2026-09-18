"""動量持股檔數集中度網格的第三段獨立驗證：重新切三段，不沿用原本的樣本內/外邊界。

背景：10.11 節發現 top_n=2 在原本的樣本外窗格（2018-09-20~2023-09-18）
五項指標全面贏過 QQQ 買進持有，是本專案至今唯一一次。但要小心：我們
手上的美股資料範圍就是 2018-09-20~資料庫最新日期（8 年回填上限，見
docs/research_findings.md 第10節開頭），而這整段時間已經被原本的「樣本內」
（2023-09-19~最新，用來跟 QQQ 第一次對照）跟「樣本外」（2018-09-20~
2023-09-18，用來驗證動量/雙動能/調倉頻率/持股檔數）兩段瓜分完畢、沒有
任何一天是完全沒被用過的——嚴格意義上的「第三段從沒看過的資料」並不
存在，因為根本沒有更多歷史資料可以切。

這裡做能力範圍內最誠實的替代方案：把整段可用資料（2018-09-20~最新）
重新切成三段大致等長、彼此互不重疊的期間（邊界跟原本的樣本內/樣本外
邊界不同），各自獨立回測（資金重置、不跨期間延續部位），檢查 top_n=2
（以及 1、3 這兩個鄰居）在這三段裡的表現是不是一致地贏過/接近 QQQ，
還是只集中在其中一段（例如剛好卡到 2020 COVID 反彈或某一段特定的
科技股行情），換一種切法就不成立。這不是真正全新的資料，是同一份
8 年歷史的不同切法，用來測試 top_n=2 的優勢對「切在哪裡」穩不穩健，
跟真正拿到全新、從未用過的歷史資料是兩回事，這裡誠實區分清楚。

動量參數維持跟 10.5/10.11 節一致：momentum_window=126、
rebalance_freq_days=21（原始網格中段值，非調參挑選）；只測 top_n∈
{1,2,3}（10.11 節裡表現最突出的三個，特別是 top_n=2）加上 10 當作
一個分散度較高的對照組，不重跑到 50 是為了控制篇幅，反正 10.11 節已
確認 top_n 越大表現越平庸、越不是這裡要驗證的重點。

反未來函數：跟前面幾個腳本一樣，永遠傳完整 us_prices（不切片），只用
start_date/end_date 限制「哪些日期允許實際調倉」；QQQ 買進持有的對照
數字用跟 tw_quant.backtest.summarize_performance 完全相同的公式（交易
日數/252 年化、simple cummax drawdown、日報酬年化 Sharpe）在本腳本內
重新計算，不是套用之前跑過的固定常數（因為這裡的三段期間邊界是全新的，
沒有現成的 QQQ 對照數字可以重用）。

用法：
    python scripts/test_us_momentum_topn_three_way_split_from_db.py
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

HEADER = (
    f"{'top_n':>6} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'stocks_held':>40}"
)


def _qqq_buyhold_metrics(qqq: pd.DataFrame, start_date, end_date) -> dict:
    """跟 tw_quant.backtest.summarize_performance 用完全相同的公式，
    但直接吃收盤價序列（QQQ 只有一檔、沒有交易紀錄），方便跟策略端的
    回測結果做同一套定義下的公平比較。
    """
    s = qqq[(qqq["date"] >= pd.Timestamp(start_date)) & (qqq["date"] <= pd.Timestamp(end_date))]
    eq = s.sort_values("date")["close"].reset_index(drop=True)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    years = max(len(eq) / 252, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    running_peak = eq.cummax()
    max_dd = ((running_peak - eq) / running_peak).max()
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0
    return {"total_return": total_return, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar}


def _fmt_row(top_n: int, m: dict, stocks_held: set[str]) -> str:
    held_str = ",".join(sorted(stocks_held)[:6]) + ("..." if len(stocks_held) > 6 else "")
    return (
        f"{top_n:>6} "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {held_str:>40}"
    )


def _run(us_prices: pd.DataFrame, base_cfg, top_n: int, start_date, end_date) -> tuple[dict, set[str]]:
    factor_cfg = FactorConfig(
        momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=top_n, ascending=False
    )
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs
    )
    m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
    stocks_held = set(result.trades["stock_id"]) if not result.trades.empty else set()
    stocks_held |= set(result.open_positions.keys())
    return m, stocks_held


def _print_period(label: str, us_prices, base_cfg, qqq, start_date, end_date) -> None:
    print(f"\n=== {label}（{pd.Timestamp(start_date).date()} ~ {pd.Timestamp(end_date).date()}） ===")
    print(HEADER)
    for top_n in TOP_N_GRID:
        m, stocks_held = _run(us_prices, base_cfg, top_n, start_date, end_date)
        print(_fmt_row(top_n, m, stocks_held))
    qqq_m = _qqq_buyhold_metrics(qqq, start_date, end_date)
    print(
        f"\n（對照：QQQ 買進持有同期間總報酬 {qqq_m['total_return']:.2%}、"
        f"CAGR {qqq_m['cagr']:.2%}、MDD {qqq_m['max_dd']:.2%}、"
        f"Sharpe {qqq_m['sharpe']:.2f}、Calmar {qqq_m['calmar']:.2f}）"
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

    print("抓取 QQQ 價格歷史（外部序列，不寫入 us_prices，不影響選股魚池）...")
    provider = YFinanceUSDataProvider()
    qqq = provider.fetch_price(
        "QQQ", start_date=str(earliest.date()), end_date=str((latest + pd.Timedelta(days=1)).date()), industry="ETF"
    )
    if qqq.empty:
        print("QQQ 資料抓取失敗，中止。", file=sys.stderr)
        sys.exit(1)
    qqq = qqq.sort_values("date").reset_index(drop=True)

    all_dates = sorted(us_prices["date"].unique())
    n = len(all_dates)
    third = n // 3
    p1_start, p1_end = all_dates[0], all_dates[third - 1]
    p2_start, p2_end = all_dates[third], all_dates[2 * third - 1]
    p3_start, p3_end = all_dates[2 * third], all_dates[-1]

    print(
        f"把完整可用資料（{n} 個交易日）重新切成三段大致等長、互不重疊的期間\n"
        f"（邊界跟原本樣本內/樣本外的切法不同，見腳本開頭說明）：\n"
        f"  P1: {pd.Timestamp(p1_start).date()} ~ {pd.Timestamp(p1_end).date()}（{third} 個交易日）\n"
        f"  P2: {pd.Timestamp(p2_start).date()} ~ {pd.Timestamp(p2_end).date()}（{third} 個交易日）\n"
        f"  P3: {pd.Timestamp(p3_start).date()} ~ {pd.Timestamp(p3_end).date()}（{n - 2 * third} 個交易日）"
    )

    base_cfg = build_us_config()
    print(
        f"\n固定參數：momentum_window={MOM_WINDOW}、rebalance_freq_days={REBALANCE_FREQ_DAYS}"
        f"（跟 10.5/10.11 節一致，非調參挑選）；持股檔數網格：{TOP_N_GRID}"
    )

    _print_period("P1（最早段）", us_prices, base_cfg, qqq, p1_start, p1_end)
    _print_period("P2（中段）", us_prices, base_cfg, qqq, p2_start, p2_end)
    _print_period("P3（最新段）", us_prices, base_cfg, qqq, p3_start, p3_end)

    print(
        "\n（誠實揭露：這三段彼此不重疊，但個別跟原本的樣本內/樣本外窗格有重疊——\n"
        "這裡驗證的是「top_n=2 的優勢對切法穩不穩健」，不是「拿到全新資料再測一次」，\n"
        "嚴格意義上的全新樣本外資料在目前 8 年回填範圍內並不存在；stocks_held 欄位\n"
        "只列前 6 檔（依代號排序，不代表進場順序），用來肉眼檢查優勢是不是靠同一檔/\n"
        "同一小撮股票撐起來的，不是嚴謹的統計檢定）"
    )


if __name__ == "__main__":
    main()
