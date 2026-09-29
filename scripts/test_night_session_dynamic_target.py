"""使用者修正：影片講的「400點振幅空間」應該不是固定數字，而是根據當天
日盤/下午盤振幅「推算」出來的——也就是當天日盤走得越開，晚上夜盤（尤其
美股開盤後）預期也會走得越開。這是「當日波動群聚」(volatility clustering
across sessions within the same day)假設，跟先前測的「方向性動能/持續
偏向」是不同的假設，值得獨立測。

第一步先只驗這個波動群聚關係本身存不存在（不牽涉方向/停利）：

    corr(日盤振幅, 夜盤21:30後振幅)
    IS(2017-2021): n=1135 corr=0.563 R2=0.317（有關係，但只解釋31.7%的變異）
    OOS(2022-2023): n=484 corr=0.240 R2=0.058（明顯衰退，但方向沒變、仍>0）
    逐年相關係數在0.01~0.64之間跳動很大（2017年幾乎零關係，2020年關係最強）

    IS擬合：post_range ≈ 35.0 + 0.313 * day_range
    用使用者提的例子代入：日盤538點 -> 預測約202點；下午盤511點 -> 預測
    約195點。都遠低於影片講的400點，代表即使承認「日盤大->夜盤也會大」
    這個方向是對的，400點這個具體數字仍然偏高估——那天大概是特別的
    單一案例，不是這個關係式的典型推算結果。

第二步：把這個關係套進策略——用IS擬合的固定公式(35.0+0.313*day_range，
不隨OOS重新擬合，避免用未來資訊)，把停利目標從固定150/175/200點改成
「當天推算post_range的某個比例」，测試k=0.5/0.75/1.0（"去頭去尾"對應
不追求吃滿整段、只吃預期範圍的一部分），進場/停損/時間停損邏輯跟前一版
(test_night_session_target_take.py)完全相同，只有停利目標從固定值
換成動態值，才能看出「讓停利目標貼近當天真實可用空間」這個修正本身
有沒有用。

用法：
    python scripts/test_night_session_dynamic_target.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import TradeCost, apply_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"

NIGHT_COST = TradeCost("夜盤（流動性較差，估3點滑價）", commission_round_trip=60.0, slippage_points_round_trip=3.0)

DAY_START, DAY_END = time(8, 45), time(13, 45)
RANGE_START, RANGE_END = time(15, 0), time(21, 30)
TIME_STOP = time(1, 0)
MIN_MOVE_POINTS = 10.0
STOP_LOSS_POINTS = 100.0
TARGET_FRACTIONS = [0.5, 0.75, 1.0]

# IS(2017-05~2021-12)擬合，套用到全樣本(含OOS)不重新估計
FIT_INTERCEPT = 35.0
FIT_SLOPE = 0.313
MIN_TARGET_POINTS = 50.0  # 避免推算值太小時停利目標近乎雜訊

IS_END = "2022-01-01"
OOS_END = "2024-01-01"


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def build_day_range(df: pd.DataFrame) -> pd.Series:
    t = df["datetime"].dt.time
    day = df[(t >= DAY_START) & (t <= DAY_END)].copy()
    day["trading_date"] = day["datetime"].dt.date
    g = day.groupby("trading_date").agg(high=("high", "max"), low=("low", "min"), n=("close", "size"))
    g = g[g["n"] > 200]
    result = (g["high"] - g["low"]).rename("day_range")
    result.index = pd.to_datetime(result.index)
    return result


def night_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    night = df[(t >= RANGE_START) | (t <= time(5, 0))].copy()
    is_evening = night["datetime"].dt.time >= RANGE_START
    night["own_date"] = night["datetime"].dt.date
    # trading_date = 這段夜盤所屬的「當天日盤」日期（凌晨K棒屬於前一個日曆日的夜盤）
    night["trading_date"] = np.where(is_evening, night["own_date"],
                                      night["own_date"] - pd.Timedelta(days=1))
    night["trading_date"] = pd.to_datetime(night["trading_date"])
    night["is_evening_row"] = is_evening
    return night


def simulate(df: pd.DataFrame, day_range: pd.Series, target_fraction: float) -> pd.DataFrame:
    night = night_session_frame(df)
    trades = []
    for td, g in night.groupby("trading_date"):
        if td not in day_range.index:
            continue
        predicted_post_range = FIT_INTERCEPT + FIT_SLOPE * day_range.loc[td]
        target = max(predicted_post_range * target_fraction, MIN_TARGET_POINTS)

        g = g.sort_values("datetime").reset_index(drop=True)
        is_evening_row = g["is_evening_row"]
        gt = g["datetime"].dt.time
        pre_mask = is_evening_row & (gt >= RANGE_START) & (gt < RANGE_END)
        post_mask = (is_evening_row & (gt >= RANGE_END)) | (~is_evening_row)
        if pre_mask.sum() < 30 or not post_mask.any():
            continue
        pre = g.loc[pre_mask]
        range_high, range_low = pre["high"].max(), pre["low"].min()

        post = g.loc[post_mask].reset_index(drop=True)
        entry_idx = direction = None
        for i in range(len(post)):
            c = post["close"].iloc[i]
            if c >= range_high + MIN_MOVE_POINTS:
                entry_idx, direction = i, "long"
                break
            if c <= range_low - MIN_MOVE_POINTS:
                entry_idx, direction = i, "short"
                break
        if entry_idx is None or entry_idx + 1 >= len(post):
            continue

        entry_bar = post.iloc[entry_idx + 1]
        sign = 1 if direction == "long" else -1
        entry_price = entry_bar["open"] + sign * (NIGHT_COST.slippage_points_round_trip / 2)
        entry_dt = entry_bar["datetime"]

        exit_price = exit_dt = exit_reason = None
        for j in range(entry_idx + 1, len(post)):
            r = post.iloc[j]
            rt = r["datetime"].time()
            favorable = (r["high"] - entry_price) if direction == "long" else (entry_price - r["low"])
            adverse = (entry_price - r["low"]) if direction == "long" else (r["high"] - entry_price)
            if favorable >= target:
                exit_price = entry_price + sign * target - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "target"
                break
            if adverse >= STOP_LOSS_POINTS:
                exit_price = entry_price - sign * STOP_LOSS_POINTS - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "stop"
                break
            is_evening_bar = rt >= RANGE_START
            if (not is_evening_bar) and (rt >= TIME_STOP):
                exit_price = r["close"] - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "time_stop"
                break
        if exit_price is None:
            last = post.iloc[-1]
            exit_price = last["close"] - sign * (NIGHT_COST.slippage_points_round_trip / 2)
            exit_dt, exit_reason = last["datetime"], "session_end"

        pnl_points = (exit_price - entry_price) * sign
        trades.append(dict(
            trading_date=td, direction=direction, target=target, entry_dt=entry_dt,
            entry_price=entry_price, exit_dt=exit_dt, exit_price=exit_price,
            exit_reason=exit_reason, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    return trades_df.sort_values("entry_dt").reset_index(drop=True)


def report(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"  {label}: n=0")
        return
    t, n, m = tstat(trades["pnl_points"])
    net = apply_costs(trades, TradeCost("已含夜盤滑價的淨額", commission_round_trip=60.0))["net_twd"].sum()
    win_rate = (trades["pnl_points"] > 0).mean() * 100
    reasons = trades["exit_reason"].value_counts(normalize=True) * 100
    print(f"  {label}: n={n} t={t:.2f} mean={m:.1f}點 勝率={win_rate:.1f}% 淨損益={net:,.0f}元 "
          f"平均動態目標={trades['target'].mean():.0f}點")
    print(f"    出場原因分布: {dict(reasons.round(1))}")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    day_range = build_day_range(df)

    print("=" * 78)
    print("動態停利目標 = IS擬合公式(35.0+0.313*當天日盤振幅) * 比例係數")
    print("=" * 78)

    for frac in TARGET_FRACTIONS:
        trades = simulate(df, day_range, frac)
        if trades.empty:
            print(f"\n比例{frac}: 無交易")
            continue
        is_trades = trades[trades["trading_date"] < IS_END]
        oos_trades = trades[(trades["trading_date"] >= IS_END) & (trades["trading_date"] < OOS_END)]
        print(f"\n--- 停利目標比例 k={frac} ---")
        report(is_trades, "IS(2017-05~2021-12)")
        report(oos_trades, "OOS(2022-2023)")


if __name__ == "__main__":
    main()
