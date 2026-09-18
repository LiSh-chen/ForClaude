"""等權重持有魚池內所有合格股票、只做被動的定期補位（不主動選股、不排名）：
拿來當「非戰之罪」的參照組。

背景：10.5~10.9 節每一種主動選股策略（動量、雙動能濾網、7 種調倉頻率、
低波動因子）樣本外驗證後全部打不過 QQQ 買進持有。在繼續嘗試新策略之前，
先回答一個更基本的問題：這是因為我們的選股邏輯本身有問題，還是因為
QQQ 這段期間的漲幅本來就集中在少數幾檔巨型科技/AI股，極度難用任何
「分散持有一籃子股票」的方法追上（不管選股邏輯是什麼）？

這裡測一個完全不主動選股的參照組：等權重持有策略引擎的合格魚池
（`build_pool_mask` 的流動性 + 站上 60 日均線篩選，跟前面每個策略共用
同一套魚池邏輯）裡所有股票，重用既有的 `run_factor_backtest`：把
`top_n` 設得比魚池股數還大（魚池最多 503 檔），這樣每次調倉「排名」
完全不影響誰被選中——魚池內全部股票都會被買進。調倉頻率維持月調倉
（21天，未經調參的預設值）而不是「只買一次、之後完全不碰」：原因是
若把調倉頻率拉到「整段回測期間只執行一次」，那唯一的進場日剛好卡在
資料庫最早的一天（2018-09-20），此時任何股票都還沒有滿足魚池 252 日
最低歷史長度門檻，會導致樣本外那段測試從頭到尾一檔都買不到、報酬率
變成 0%（一次合成資料的煙霧測試就抓到這個 bug）。改成月調倉後，前
252 個交易日魚池雖然是空的，但等到門檻滿足，隨後的每次調倉都會自然
把當時合格的股票全部納入，已經持有且依然合格的股票不會被平倉重買
（只有真的跌破均線/流動性不足才會被換掉），換手率遠低於任何一個主動
排名策略，但不是嚴格意義的「買進後完全不動」。

如果連這個「不主動選股、只做被動定期補位」的參照組都大幅落後 QQQ，
代表問題主要出在 QQQ 本身這段期間的集中度太極端，不是我們的選股邏輯
特別差；如果這個參照組反而接近甚至打敗 QQQ，代表前面測過的主動選股
邏輯（動量排名、低波動排名）本身在拖累績效，比「不特別選股」還差。

反未來函數：跟前面幾個腳本一樣，永遠傳完整 us_prices（不切片），只用
start_date/end_date 限制「哪些日期允許實際調倉」。

用法：
    python scripts/test_us_equal_weight_buyhold_from_db.py
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

TOP_N = 999  # 比魚池股數（最多 503 檔）還大，等同「全部買進」
REBALANCE_FREQ_DAYS = 21  # 月調倉，非調參挑選；只是定期把魚池最新狀態同步進持股

IN_SAMPLE_START = "2023-09-19"
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

HEADER = f"{'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  {'n_held':>6}"


def _fmt_row(m: dict, n_held: int) -> str:
    return (
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{n_held:>6}"
    )


def _run_and_print(label: str, us_prices: pd.DataFrame, base_cfg, start_date, end_date, qqq_bench: dict) -> None:
    factor_cfg = FactorConfig(
        momentum_window=21, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=False
    )
    result = run_factor_backtest(us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs)
    m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
    n_held = len(result.open_positions)

    print(f"\n=== {label} ===")
    print(HEADER)
    print(_fmt_row(m, n_held))
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
        "策略：每月等權重買進魚池內全部合格股票（流動性 + 站上60日均線），"
        "依然合格的舊持股不換手，只補位新符合資格/剔除新失格的股票"
    )

    _run_and_print("樣本內（2023-09-19 ~ 資料庫最新日期，跟原始 QQQ 對照同期間）", us_prices, base_cfg, IN_SAMPLE_START, None, QQQ_IN_SAMPLE)
    _run_and_print("樣本外（2018-09-20 ~ 2023-09-18）", us_prices, base_cfg, None, OOS_END, QQQ_OOS)

    print(
        "\n（這不是嚴格意義上的「買進整個 S&P 500」——魚池篩選（流動性+站上60日均線）\n"
        "會排除一部分股票，且是「等權重、定期補位合格股票」而非真正持續追蹤指數\n"
        "成分的等權重基準，比真正的等權重 S&P 500 指數基準粗糙，但足以回答「分散\n"
        "持有本身能不能打敗 QQQ」這個問題；n_held 是回測結束時實際持有的檔數，\n"
        "不是整段期間曾經買過的檔數）"
    )


if __name__ == "__main__":
    main()
