"""使用者指正上一版的設計缺陷：「中心價=21:30收盤價」是隨便選一個時間點
當基準，不合理。正確的邏輯應該是——先追蹤夜盤從開盤(15:00)以來累積的
實際振幅，等它真的達到「預期振幅」（用日盤振幅推算，見
test_night_session_dynamic_target.py的擬合公式）之後，才進場賭「這波
已經達標、該歇一下了」的回檔/反彈，而不是在固定時間點武斷畫一個對稱
區間。

跟上一版(test_night_session_range_fade.py)的差異：
- 上一版：中心=21:30當下收盤價（固定時間點快照），區間=中心±預期振幅/2
  （對稱、跟實際走勢無關的假想區間）
- 這一版：從夜盤開盤(15:00)開始逐分鐘追蹤累積的running_high/running_low，
  running_range = running_high - running_low 第一次達到「預期振幅」的
  當下才觸發——這個「預期振幅」用IS擬合的全夜盤版本(38.69+0.4169*日盤
  振幅，R2=0.374，比只算21:30後那段的擬合R2=0.317略高，兩個IS窗格
  一致：2017-2021)，方向由「這一根K棒是創了新高還是新低」決定
  （創新高->賭回檔放空，創新低->賭反彈做多），兩者同時發生(同一根K棒
  創新高又創新低)的罕見情況直接跳過避免歧義。

停利=達標時累積振幅(achieved_range)的某個比例（測30/50/75%，"去頭去尾"
精神：不奢求完全回到起點,吃到回檔/反彈的一部分就收工），停損=繼續朝
原方向延伸achieved_range的30%（代表這不是達標後歇息、是真的還在噴，
及早停損）。01:00時間停損、05:00保底出場。

不開網格搜尋，只測3個預先指定的停利比例。

用法：
    python scripts/test_night_session_range_exhaustion.py
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
NIGHT_START = time(15, 0)
TIME_STOP = time(1, 0)

# IS(2017-2021)擬合，套用到全樣本(含OOS)不重新估計，跟前一版用同一份IS窗格
FIT_INTERCEPT = 38.69
FIT_SLOPE = 0.4169
MIN_PREDICTED_RANGE = 50.0

TARGET_FRACTIONS = [0.3, 0.5, 0.75]  # 停利=achieved_range的這個比例
STOP_FRACTION = 0.3  # 停損=繼續延伸achieved_range的這個比例

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
    night = df[(t >= NIGHT_START) | (t <= time(5, 0))].copy()
    is_evening = night["datetime"].dt.time >= NIGHT_START
    night["own_date"] = night["datetime"].dt.date
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
        predicted_range = max(FIT_INTERCEPT + FIT_SLOPE * day_range.loc[td], MIN_PREDICTED_RANGE)

        g = g.sort_values("datetime").reset_index(drop=True)
        if len(g) < 30:
            continue

        running_high = g["high"].iloc[0]
        running_low = g["low"].iloc[0]
        trigger_idx = direction = None
        for i in range(1, len(g)):
            bar = g.iloc[i]
            set_new_high = bar["high"] > running_high
            set_new_low = bar["low"] < running_low
            if set_new_high:
                running_high = bar["high"]
            if set_new_low:
                running_low = bar["low"]
            running_range = running_high - running_low
            if running_range >= predicted_range:
                if set_new_high and set_new_low:
                    break  # 同一根K棒同時創新高新低，方向歧義，跳過這一晚
                if set_new_high:
                    trigger_idx, direction = i, "short"
                elif set_new_low:
                    trigger_idx, direction = i, "long"
                else:
                    continue  # 累積振幅達標但這根K棒本身沒創極值，繼續等下一根
                break

        if trigger_idx is None or trigger_idx + 1 >= len(g):
            continue
        achieved_range = running_range
        extreme_price = running_high if direction == "short" else running_low

        entry_bar = g.iloc[trigger_idx + 1]
        sign = 1 if direction == "long" else -1
        entry_price = entry_bar["open"] + sign * (NIGHT_COST.slippage_points_round_trip / 2)
        entry_dt = entry_bar["datetime"]

        target_dist = achieved_range * target_fraction
        stop_dist = achieved_range * STOP_FRACTION

        exit_price = exit_dt = exit_reason = None
        for j in range(trigger_idx + 1, len(g)):
            r = g.iloc[j]
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
            is_evening_bar = rt >= NIGHT_START
            if (not is_evening_bar) and (rt >= TIME_STOP):
                exit_price = r["close"] - sign * (NIGHT_COST.slippage_points_round_trip / 2)
                exit_dt, exit_reason = r["datetime"], "time_stop"
                break
        if exit_price is None:
            last = g.iloc[-1]
            exit_price = last["close"] - sign * (NIGHT_COST.slippage_points_round_trip / 2)
            exit_dt, exit_reason = last["datetime"], "session_end"

        pnl_points = (exit_price - entry_price) * sign
        trades.append(dict(
            trading_date=td, direction=direction, predicted_range=predicted_range,
            achieved_range=achieved_range, trigger_time=entry_bar["datetime"].time(),
            entry_dt=entry_dt, entry_price=entry_price, exit_dt=exit_dt, exit_price=exit_price,
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
    print("震幅達標後反轉（追蹤累積振幅達到IS擬合預期值才進場，賭達標後歇息）")
    print(f"預期振幅公式：{FIT_INTERCEPT}+{FIT_SLOPE}*日盤振幅")
    print("=" * 78)

    ref_trades = None
    for frac in TARGET_FRACTIONS:
        trades = simulate(df, day_range, frac)
        if trades.empty:
            print(f"\n停利比例{frac}: 無交易")
            continue
        if ref_trades is None:
            ref_trades = trades
        is_trades = trades[trades["trading_date"] < IS_END]
        oos_trades = trades[(trades["trading_date"] >= IS_END) & (trades["trading_date"] < OOS_END)]
        print(f"\n--- 停利=achieved_range的{frac*100:.0f}% (停損=延伸achieved_range的{STOP_FRACTION*100:.0f}%) ---")
        report(is_trades, "IS(2017-05~2021-12)")
        report(oos_trades, "OOS(2022-2023)")

    if ref_trades is not None:
        print("\n" + "=" * 78)
        print("觸發時間分布 + 方向分布檢查（全樣本，用50%停利版本為代表）")
        print("=" * 78)
        print(f"n={len(ref_trades)}")
        hours = ref_trades["trigger_time"].apply(lambda x: x.hour)
        print("觸發時刻(小時)分布:")
        print(hours.value_counts().sort_index())
        print("\n方向分布:")
        print(ref_trades["direction"].value_counts())
        for d in ["long", "short"]:
            sub = ref_trades[ref_trades["direction"] == d]
            t, n, m = tstat(sub["pnl_points"])
            print(f"  {d}: n={n} t={t:.2f} mean={m:.1f}點")


if __name__ == "__main__":
    main()
