"""在三腿日內策略（開盤上衝 + 午盤放空 + 盤中翻多）上疊加技術指標過濾器。

樣本內（2001-2020）掃了 RSI(14)／MACD(12,26,9)／布林通道 %B(20,2)／成交量比率
（對20日均量），全部用昨收算、避免未來函數，把每個時段的交易依指標三分位
分桶比較。結果：
- 開盤上衝、午盤放空兩腿在四個指標的所有分桶都穩定顯著（t 都 > 2.7），
  代表這兩腿的邊際不特別依賴進場當下的技術面狀態，加濾網意義不大。
- 盤中翻多腿明顯不同：昨日成交量比率落在最高三分位時 t=4.23（單日子區間
  1.78~2.80，全部通過穩健性檢查），最低三分位時 t=0.51、幾乎沒有邊際
  ——這是唯一一個濾網後有實質改善、又通過子區間穩健性的組合。

門檻值（成交量比率前 1/3 分位切點）用樣本內資料算一次、寫死，樣本外驗證時
直接套用同一個數字，不是在樣本外重新計算三分位——不然就是變相用到未來
資訊。

用法：
    python scripts/backtest_lunch_reversal_volume_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
IS_CUTOFF = "2021-01-01"


def report(trades: pd.DataFrame, label: str) -> None:
    n = len(trades)
    print(f"\n=== {label}：n={n} 毛損益={trades['pnl_twd'].sum():,.0f} ===")
    for cost in COST_SCENARIOS:
        net = apply_costs(trades, cost)["net_twd"].sum()
        print(f"  {cost.label}: net={net:,.0f}")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    # 門檻值只用樣本內資料算一次
    is_indicators = daily_indicators(is_df)
    vol_threshold = is_indicators["vol_ratio_lag1"].quantile(2 / 3)
    print(f"成交量比率門檻（樣本內算出，樣本外套用同一個數字）: {vol_threshold:.4f}\n")

    for label, sub_df in [("樣本內 2001-2020", is_df), ("樣本外 2021-2023（只驗證一次）", oos_df)]:
        open_trades = backtest_opening(sub_df)
        lunch_trades = backtest_lunch(sub_df)
        short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
        long_leg = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]

        long_leg_filtered = filter_trades_by_volume(long_leg, sub_df, vol_threshold)

        print(f"\n{'#'*10} {label} {'#'*10}")
        report(open_trades, "開盤上衝（不變）")
        report(short_leg, "午盤放空（不變）")
        report(long_leg, "盤中翻多（原版，全部交易日）")
        report(long_leg_filtered, "盤中翻多（成交量濾網後）")

        combined_unfiltered = pd.concat([open_trades, short_leg, long_leg], ignore_index=True)
        combined_filtered = pd.concat([open_trades, short_leg, long_leg_filtered], ignore_index=True)
        report(combined_unfiltered, "三腿合併（原版）")
        report(combined_filtered, "三腿合併（盤中翻多加濾網）")


if __name__ == "__main__":
    main()
