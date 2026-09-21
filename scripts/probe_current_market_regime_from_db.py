"""診斷「目前市況」：QQQ 相對各條均線的位置（趨勢濾網現在的訊號）、
最近從高點的回落幅度、目前市場狀態分類（動能強勢期／盤整期／修正期），
以及策略「最近一次調倉」實際會持有哪些股票——回答「依目前市況來看，這個
策略現在處於什麼位置」這個問題，不是重新回測，是對資料庫最新狀態的
現況快照。

用法：
    python scripts/probe_current_market_regime_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.market_regime_breakdown import trailing_return_regime_labels
from tw_quant.signals import build_pool_mask
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider
from tw_quant.us_universe import filter_prices_by_index_membership

MOM_WINDOW = 126
TOP_N = 3
TREND_MA_GRID = (50, 100, 150, 200)


def main() -> None:
    us_prices_raw = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()
    us_prices = filter_prices_by_index_membership(us_prices_raw, membership)
    latest_date = us_prices["date"].max()

    print(f"資料庫最新日期：{latest_date.date()}\n")

    provider = YFinanceUSDataProvider()
    qqq_df = provider.fetch_price("QQQ", start_date="2022-01-01", end_date=str((latest_date + pd.Timedelta(days=1)).date()), industry="ETF")
    qqq = qqq_df.sort_values("date").reset_index(drop=True)
    qqq_latest_close = qqq["close"].iloc[-1]
    qqq_latest_date = qqq["date"].iloc[-1]

    print(f"=== QQQ 現況（{qqq_latest_date.date()} 收盤 ${qqq_latest_close:.2f}）===")
    for ma_window in TREND_MA_GRID:
        ma = qqq["close"].rolling(ma_window).mean().iloc[-1]
        status = "多頭（站上均線）" if qqq_latest_close > ma else "空頭（跌破均線，趨勢濾網現在會判定出清換現金）"
        pct_vs_ma = qqq_latest_close / ma - 1
        print(f"  {ma_window:>3} 日均線 = ${ma:.2f}，現價 {pct_vs_ma:+.2%} -> {status}")

    running_max = qqq["close"].cummax().iloc[-1]
    dd_from_peak = qqq_latest_close / running_max - 1
    peak_date = qqq.loc[qqq["close"].idxmax(), "date"]
    print(f"\n  目前距歷史新高（{peak_date.date()}，${running_max:.2f}）回落：{dd_from_peak:+.2%}")

    regime_labels = trailing_return_regime_labels(qqq.set_index("date")["close"], lookback_days=MOM_WINDOW)
    current_regime = regime_labels.iloc[-1]
    trailing_126d_ret = qqq["close"].iloc[-1] / qqq["close"].iloc[-1 - MOM_WINDOW] - 1 if len(qqq) > MOM_WINDOW else None
    print(f"\n  目前市場狀態分類（T-1 為止 126 日已實現報酬率三分位，跟策略動量窗格一致）：{current_regime}")
    if trailing_126d_ret is not None:
        print(f"  QQQ 最近 126 個交易日報酬率：{trailing_126d_ret:+.2%}")

    print("\n=== 策略現況：最近一次調倉會持有哪些股票 ===")
    base_cfg = build_us_config()
    master = us_prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    pool = build_pool_mask(master, base_cfg.pool)
    ret = master.groupby("stock_id", sort=False)["close"].pct_change(MOM_WINDOW)
    momentum = ind.shift_by_group(ret, master, periods=1)
    master["pool"] = pool.values
    master["momentum"] = momentum.values if hasattr(momentum, "values") else momentum

    last_day = master[master["date"] == latest_date]
    eligible = last_day[last_day["pool"] & last_day["momentum"].notna()]
    top_picks = eligible.sort_values("momentum", ascending=False).head(TOP_N)
    n_pool = int(last_day["pool"].sum())

    print(f"  {latest_date.date()} 股票池符合篩選條件：{n_pool} 檔")
    if top_picks.empty:
        print("  （最新交易日沒有符合資格的股票，可能還沒到下次調倉日或資料不足）")
    else:
        for _, row in top_picks.iterrows():
            print(f"  {row['stock_id']:<6} 126日動量 {row['momentum']:+.2%}，收盤價 ${row['close']:.2f}")

    print(
        "\n（這是「如果今天是調倉日」的即時排名快照，不代表策略實際上次調倉真的\n"
        "買了這些股票——調倉日是固定頻率排程決定的，不是每天都調倉；這裡純粹是\n"
        "拿最新一天的資料跑一次排名邏輯，看現在的市況會選出哪些股票）"
    )


if __name__ == "__main__":
    main()
