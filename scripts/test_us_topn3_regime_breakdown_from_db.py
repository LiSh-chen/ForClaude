"""top_n=3 動量策略依市場狀態（動能強勢期／盤整期／修正下跌期）拆解表現，
驗證「強勢期有超額報酬、平穩期不輸大盤」這個假設是不是真的成立，還是樣本外
整段打平只是不同市況互相抵銷的結果。

背景：使用者觀察到 top_n=3 在樣本內（2023下半年起的科技動能噴出期）大幅
超越 QQQ，樣本外（2018~2023，混合 COVID 崩盤、2022 熊市、多次多頭段的 5
年期間）只小幅打平，因而提出前述假設。原本的樣本內/樣本外切法沒辦法驗證
這個假設，因為樣本外那段本身就是好幾種市況混在一起，光看整段加總的總報酬
分不出「全程小贏」跟「強勢期贏、下跌期輸，加總打平」的差別。

做法：不再用人為切的樣本內/樣本外日期分界，改用 tw_quant/market_regime_breakdown.py
對 QQQ 本身的價格序列算一個客觀、反未來函數的市場狀態標籤（T-1為止 126 個
交易日的已實現報酬率，三分位分類——跟策略自己選股用的動量落後窗格
momentum_window=126 天一致，方便解讀），套用在整段可用歷史（不再切成
兩段，用完整連續回測換取更多天數/更多獨立區段），再把策略跟 QQQ 買進
持有各自的逐日報酬率依標籤分組比較。

固定參數（10.11 節同一組，非重新調參）：momentum_window=126、
rebalance_freq_days=21、top_n=3。

反未來函數：動量排名一樣永遠用完整 us_prices（不切片）；市場狀態標籤用
shift(1)，T 日的標籤只用到 T-1 為止已知的 QQQ 報酬率；三分位門檻用全樣本
分位數決定——這是事後歸因分組用的統計手法，不是即時交易訊號，不影響、也
沒有用到策略本身逐日報酬率的計算，細節見 tw_quant/market_regime_breakdown.py
開頭說明。

2026-09-21 架構修正：改用完整未過濾的 us_prices + membership 參數，
取代先前先用 filter_prices_by_index_membership 預過濾再傳進引擎的舊
寫法（誤傷 MRVL 等 14 檔股票，詳見 tw_quant/us_universe.py 檔頭）。

用法：
    python scripts/test_us_topn3_regime_breakdown_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.market_regime_breakdown import (
    REGIME_HIGH,
    REGIME_LOW,
    REGIME_MID,
    regime_episode_stats,
    trailing_return_regime_labels,
)
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N = 3
REGIME_LOOKBACK_DAYS = MOM_WINDOW  # 跟策略自己的動量窗格一致，方便解讀

REGIME_ORDER = [REGIME_HIGH, REGIME_MID, REGIME_LOW]

HEADER = (
    f"{'市場狀態':>14} {'天數':>6} {'佔比':>7} {'獨立區段':>7} {'最長區段':>7}  "
    f"{'策略累積報酬':>12} {'QQQ累積報酬':>11} {'贏率':>7} {'策略年化Sharpe':>13} {'QQQ年化Sharpe':>12}"
)


def _annualized_sharpe(daily_ret: pd.Series) -> float:
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0


def _fmt_row(label: str, n: int, total_n: int, n_episodes: int, max_len: int, strat_ret: pd.Series, qqq_ret: pd.Series) -> str:
    strat_compound = float((1 + strat_ret).prod() - 1)
    qqq_compound = float((1 + qqq_ret).prod() - 1)
    win_rate = float((strat_ret.values > qqq_ret.values).mean())
    return (
        f"{label:>14} {n:>6} {n / total_n:>6.1%} {n_episodes:>7} {max_len:>6}天  "
        f"{strat_compound:>11.2%} {qqq_compound:>10.2%} {win_rate:>6.1%} "
        f"{_annualized_sharpe(strat_ret):>13.2f} {_annualized_sharpe(qqq_ret):>12.2f}"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {latest.date()}）")

    provider = YFinanceUSDataProvider()
    print("抓取 QQQ 價格歷史...")
    qqq_df = provider.fetch_price(
        "QQQ", start_date=str(earliest.date()), end_date=str((latest + pd.Timedelta(days=1)).date()), industry="ETF"
    )
    if qqq_df.empty:
        print("QQQ 資料抓取失敗，中止。", file=sys.stderr)
        sys.exit(1)
    qqq_close = qqq_df.sort_values("date").set_index("date")["close"]

    base_cfg = build_us_config()
    print(
        f"固定參數（10.11 節同一組，非重新調參）：momentum_window={MOM_WINDOW}、"
        f"rebalance_freq_days={REBALANCE_FREQ_DAYS}、top_n={TOP_N}；"
        f"市場狀態分類窗格={REGIME_LOOKBACK_DAYS}天（跟策略動量窗格一致）\n"
        "不再切樣本內/樣本外，用完整連續歷史跑一次回測，換取更多天數/更多獨立區段"
    )

    factor_cfg = FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=TOP_N, ascending=False)
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=None, end_date=None, cost_module=us_costs,
        membership=membership,
    )
    strat_equity = result.equity_curve["equity"]
    strat_ret_all = strat_equity.pct_change()

    qqq_ret_all = qqq_close.pct_change()
    regime_labels = trailing_return_regime_labels(qqq_close, lookback_days=REGIME_LOOKBACK_DAYS)

    combined = pd.DataFrame({"strat_ret": strat_ret_all, "qqq_ret": qqq_ret_all, "regime": regime_labels}).dropna()
    total_n = len(combined)
    print(f"\n可比對的共同交易日數：{total_n}（{combined.index.min().date()} ~ {combined.index.max().date()}）\n")

    print(HEADER)
    for regime in REGIME_ORDER:
        mask = combined["regime"] == regime
        group = combined[mask]
        if group.empty:
            print(f"{regime:>14}   （這段期間沒有出現這個市場狀態）")
            continue
        n_episodes, max_len = regime_episode_stats(combined["regime"], regime)
        print(_fmt_row(regime, len(group), total_n, n_episodes, max_len, group["strat_ret"], group["qqq_ret"]))

    print()
    n_episodes_all = {r: regime_episode_stats(combined["regime"], r)[0] for r in REGIME_ORDER}
    print(
        "全樣本加總（跟前面各別分組結果應該互相一致，當作交叉檢查）：\n"
        + _fmt_row("全部日期", total_n, total_n, sum(n_episodes_all.values()), int(combined["regime"].value_counts().max()), combined["strat_ret"], combined["qqq_ret"])
    )

    print(
        "\n（誠實揭露：市場狀態標籤用 QQQ 自身 T-1 為止 126 個交易日已實現報酬率的\n"
        "全樣本三分位分類，門檻是事後歸因用的統計手法、不是即時交易訊號，不影響\n"
        "策略本身逐日報酬率的計算（策略的動能排名跟進出場邏輯完全獨立算好，這裡\n"
        "只是把已經算出來的逐日報酬率依日期分組）；「獨立區段」欄位提醒：如果某個\n"
        "市場狀態的天數集中在很少幾段連續期間（例如整個 COVID 崩盤就是修正期裡的\n"
        "一段），那組的『規律』有多少代表性要打折扣，不能直接當成大數法則下的統計\n"
        "顯著結果——這整段可用歷史也才 8 年，重大熊市/崩盤事件不過兩三次，樣本數\n"
        "本質上就很小，這個拆解能說明的是『這次觀察到的資料呈現什麼樣貌』，不是\n"
        "能外推到未來任何一次動能期/修正期都會重演同樣結果的統計證明）"
    )


if __name__ == "__main__":
    main()
