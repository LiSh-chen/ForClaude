"""三腳策略（開盤上衝＋午盤放空＋盤中翻多+成交量濾網）對滑價的敏感度。

之前所有回測的成本模型只有期交稅（精確公式）+ 手續費（三個固定情境），
沒有另外計入滑價——TradeCost 現在多了 `slippage_points_round_trip`
（見 tw_quant/hammer_signal_backtest.py），這裡拿它掃一輪滑價點數，看
淨損益在哪個滑價水準轉負。

小台跳動點 = 1 點 = 50 元，這個策略平均每筆邊際只有 1~2 點，滑價門檻
理論上會跟邊際本身同一個量級，不是有安全邊際的關係——用這個腳本的數字
確認實際會在哪個滑價點數轉負。

用法：
    python scripts/backtest_three_leg_slippage_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import TradeCost, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "three_leg_slippage_sensitivity.parquet"

IS_CUTOFF = "2021-01-01"
VOL_THRESHOLD = 1.1  # 樣本內算出、後續一路沿用的固定門檻
SLIPPAGE_POINTS = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def combined_trades(data_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    open_trades = backtest_opening(data_df)
    lunch_trades = backtest_lunch(data_df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg_filtered = filter_trades_by_volume(long_leg, full_df, VOL_THRESHOLD)
    return pd.concat([open_trades, short_leg, long_leg_filtered], ignore_index=True)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    is_trades = combined_trades(is_df, df)
    oos_trades = combined_trades(oos_df, df)
    print(f"IS n={len(is_trades)}, OOS n={len(oos_trades)}\n")

    rows = []
    for slip in SLIPPAGE_POINTS:
        cost_mid = TradeCost("中", commission_round_trip=60.0, slippage_points_round_trip=slip)
        cost_low = TradeCost("低", commission_round_trip=30.0, slippage_points_round_trip=slip)
        rows.append(dict(
            slippage_points=slip,
            is_mid_net_twd=apply_costs(is_trades, cost_mid)["net_twd"].sum(),
            oos_mid_net_twd=apply_costs(oos_trades, cost_mid)["net_twd"].sum(),
            oos_low_net_twd=apply_costs(oos_trades, cost_low)["net_twd"].sum(),
        ))

    result = pd.DataFrame(rows)
    result.to_parquet(OUT_PATH, index=False)

    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print(result.to_string(index=False))
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
