"""使用者提出新假設：不做「突破預期區間後跟著動能走」（已經測過四種操作化
都是零信號），改做「在已知的預期波動區間內做震盪價差」——也就是碰到區間
上緣就放空賭拉回、碰到區間下緣就做多賭反彈，賭的是均值回歸而不是突破
延續。這是跟先前完全相反的經濟邏輯（fade vs follow），值得獨立驗證，
不是重測已否決的假設。

沿用上一步(test_night_session_dynamic_target.py)驗證過的當日波動群聚
公式（IS(2017-2021)擬合，固定套用、不隨OOS重新估計）：
    predicted_post_range ≈ 35.0 + 0.313 * 當天日盤振幅(08:45-13:45)

規則：
1. 中心價 = 21:30(美股開盤/預期區間起點)當下的收盤價
2. 預期區間 = 中心價 ± predicted_post_range/2（對稱切兩邊，最簡單的版本；
   知道這是簡化，實際上進場時的多空傾向不見得對稱，但避免額外調整拿
   OOS或加開自由度）
3. 21:30後第一次觸及區間上緣 -> 放空；觸及下緣 -> 做多（賭拉回中心價）
4. 停利 = 回拉到「中心價方向」的一段距離（測試回拉50%/75%/100%的band_half
   三個版本，"去頭去尾"精神：不強求拉回中心，拉回一部分就獲利了結）
5. 停損 = 在停利方向的反方向繼續走了band_half的50%（代表這次不是均值
   回歸、而是真的突破延續，及早停損避免繼續攤平虧損）
6. 01:00時間停損、04:45保底出場，跟前幾版夜盤測試一致

不開網格搜尋，只測三個預先指定的停利回拉比例。

用法：
    python scripts/test_night_session_range_fade.py
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

FIT_INTERCEPT = 35.0
FIT_SLOPE = 0.313
MIN_BAND_HALF = 25.0  # 避免預測區間太窄時進場條件形同雜訊

PULLBACK_FRACTIONS = [0.5, 0.75, 1.0]  # 停利=回拉band_half的這個比例
STOP_FRACTION = 0.5  # 停損=繼續走band_half的這個比例（突破延續失敗出場）

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
    night["trading_date"] = np.where(is_evening, night["own_date"],
                                      night["own_date"] - pd.Timedelta(days=1))
    night["trading_date"] = pd.to_datetime(night["trading_date"])
    night["is_evening_row"] = is_evening
    return night


def simulate(df: pd.DataFrame, day_range: pd.Series, pullback_fraction: float) -> pd.DataFrame:
    night = night_session_frame(df)
    trades = []
    for td, g in night.groupby("trading_date"):
        if td not in day_range.index:
            continue
        predicted_post_range = FIT_INTERCEPT + FIT_SLOPE * day_range.loc[td]
        band_half = max(predicted_post_range / 2, MIN_BAND_HALF)

        g = g.sort_values("datetime").reset_index(drop=True)
        is_evening_row = g["is_evening_row"]
        gt = g["datetime"].dt.time
        # 找21:30當下或之後第一根K棒的收盤價當中心價
        anchor_mask = (is_evening_row & (gt >= RANGE_END)) | (~is_evening_row)
        if not anchor_mask.any():
            continue
        anchor_idx0 = anchor_mask[anchor_mask].index[0]
        center_price = g["close"].iloc[anchor_idx0]
        upper_band = center_price + band_half
        lower_band = center_price - band_half

        post = g.loc[anchor_idx0:].reset_index(drop=True)
        entry_idx = direction = None
        for i in range(1, len(post)):
            c = post["close"].iloc[i]
            if c >= upper_band:
                entry_idx, direction = i, "short"  # 碰上緣，賭拉回，放空
                break
            if c <= lower_band:
                entry_idx, direction = i, "long"   # 碰下緣，賭反彈，做多
                break
        if entry_idx is None or entry_idx + 1 >= len(post):
            continue

        entry_bar = post.iloc[entry_idx + 1]
        sign = 1 if direction == "long" else -1
        entry_price = entry_bar["open"] + sign * (NIGHT_COST.slippage_points_round_trip / 2)
        entry_dt = entry_bar["datetime"]

        target_dist = band_half * pullback_fraction
        stop_dist = band_half * STOP_FRACTION

        exit_price = exit_dt = exit_reason = None
        for j in range(entry_idx + 1, len(post)):
            r = post.iloc[j]
            rt = r["datetime"].time()
            favorable = (r["high"] - entry_price) if direction == "long" else (entry_price - r["low"])
            adverse = (entry_price - r["low"]) if direction == "long" else (r["high"] - entry_price)
            if favorable >= target_dist:
                exit_price = entry_price + sign * target_dist - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "target"
                break
            if adverse >= stop_dist:
                exit_price = entry_price - sign * stop_dist - sign * (NIGHT_COST.slippage_points_round_trip / 2)
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
            trading_date=td, direction=direction, band_half=band_half, entry_dt=entry_dt,
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
    print(f"  {label}: n={n} t={t:.2f} mean={m:.1f}點 勝率={win_rate:.1f}% 淨損益={net:,.0f}元")
    print(f"    出場原因分布: {dict(reasons.round(1))}")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    day_range = build_day_range(df)

    print("=" * 78)
    print("震盪價差（碰區間上緣放空/下緣做多，賭拉回中心價）")
    print(f"預期區間公式：{FIT_INTERCEPT}+{FIT_SLOPE}*日盤振幅，band_half=預期區間/2")
    print("=" * 78)

    for frac in PULLBACK_FRACTIONS:
        trades = simulate(df, day_range, frac)
        if trades.empty:
            print(f"\n回拉比例{frac}: 無交易")
            continue
        is_trades = trades[trades["trading_date"] < IS_END]
        oos_trades = trades[(trades["trading_date"] >= IS_END) & (trades["trading_date"] < OOS_END)]
        print(f"\n--- 停利=回拉band_half的{frac*100:.0f}% (停損=突破band_half的{STOP_FRACTION*100:.0f}%) ---")
        report(is_trades, "IS(2017-05~2021-12)")
        report(oos_trades, "OOS(2022-2023)")

    print("\n" + "=" * 78)
    print("方向分布檢查（做多 vs 放空，全樣本）")
    print("=" * 78)
    trades_ref = simulate(df, day_range, 0.75)
    if not trades_ref.empty:
        print(trades_ref["direction"].value_counts())
        for d in ["long", "short"]:
            sub = trades_ref[trades_ref["direction"] == d]
            t, n, m = tstat(sub["pnl_points"])
            print(f"  {d}: n={n} t={t:.2f} mean={m:.1f}點")


if __name__ == "__main__":
    main()
