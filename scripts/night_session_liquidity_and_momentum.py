"""夜盤單獨探討：先用實際成交量確認流動性顧慮的嚴重程度，再針對流動性
相對較好的時段（美股開盤前後）測試一個新假設——不是重跑已經否決過的
VWAP持續偏向假設（scan_night_session_trend.py，結果是乾淨的零信號），
而是測試「夜盤前段（15:00開盤到美股開盤前）的走勢，跟夜盤後段（美股
開盤後到05:00收盤）走勢之間是否有動能或反轉關係」——經濟邏輯是台股
夜盤常跟著美股情緒走，美股開盤前後才是夜盤資訊含量/流動性相對最高的
時段，這之前沒被獨立測試過。

用法：
    python scripts/night_session_liquidity_and_momentum.py
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

# 夜盤流動性明顯比日盤差，用比日盤更保守（更高）的滑價假設，反映實際
# 下單時更容易吃到價差/深度不足的問題（日盤策略一般用1點，這裡用3點）
NIGHT_COST = TradeCost("夜盤（流動性較差，估3點滑價）", commission_round_trip=60.0, slippage_points_round_trip=3.0)


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def _night_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    night = df[(t >= time(15, 0)) | (t <= time(5, 0))].copy()
    # 15:00後的資料算入「隔一個交易日」的夜盤，00:00~05:00算入同一段夜盤（沿用前一天的標籤）
    is_evening = night["datetime"].dt.time >= time(15, 0)
    night["night_date"] = night["datetime"].dt.date
    night.loc[is_evening, "night_date"] = night.loc[is_evening, "datetime"].dt.date + pd.Timedelta(days=1)
    return night


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    t = df["datetime"].dt.time
    night = df[(t >= time(15, 0)) | (t <= time(5, 0))].copy()
    day_same_period = df[(t >= time(8, 45)) & (t <= time(13, 45)) & (df["datetime"] >= "2017-05-15")].copy()

    print("=" * 70)
    print("1) 夜盤 vs 日盤 流動性比較（2017-05-15起同期）")
    print("=" * 70)
    print(f"夜盤資料涵蓋: {night['datetime'].min()} ~ {night['datetime'].max()}")
    print(f"夜盤整體平均每分鐘量: {night['volume'].mean():.1f}")
    print(f"日盤整體平均每分鐘量(同期): {day_same_period['volume'].mean():.1f}")
    print(f"比例: 夜盤約為日盤的 {night['volume'].mean()/day_same_period['volume'].mean():.1%}")
    night_hourly = night.copy()
    night_hourly["hour"] = night_hourly["datetime"].dt.hour
    print("\n夜盤每小時平均每分鐘量：")
    print(night_hourly.groupby("hour")["volume"].mean().round(1))

    print("\n" + "=" * 70)
    print("2) 夜盤前段(15:00開盤->美股開盤前) vs 後段(美股開盤->05:00收盤) 動能/反轉測試")
    print("=" * 70)
    ns = _night_session_frame(df)

    rows = []
    for nd, g in ns.groupby("night_date"):
        g = g.sort_values("datetime").reset_index(drop=True)
        if len(g) < 30:
            continue
        night_open = g["open"].iloc[0]
        night_close = g["close"].iloc[-1]
        # 美股開盤時間隨日光節約變動(21:30或22:30台北時間)，用22:00當一致的錨點
        gt = g["datetime"].dt.time
        # 錨點在傍晚時段(22:00)，用裸時間比較必須排除跨午夜後的凌晨K棒
        # （凌晨00:00~05:00的time物件字面上「小於」22:00，但在夜盤時序上其實
        # 是排在22:00之後，這是這次會話已經踩過一次的同款bug，這裡直接照
        # 正確寫法：明確限定在「傍晚且<=22:00」這個區間內才算「錨點之前」）
        anchor_mask = (gt >= time(15, 0)) & (gt <= time(22, 0))
        if not anchor_mask.any() or anchor_mask.all():
            continue
        anchor_idx = anchor_mask[anchor_mask].index[-1]
        anchor_price = g["close"].iloc[anchor_idx]
        entry_bar = g.iloc[anchor_idx + 1] if anchor_idx + 1 < len(g) else None
        if entry_bar is None:
            continue
        pre_move = anchor_price - night_open
        post_move = night_close - anchor_price
        rows.append(dict(night_date=nd, night_open=night_open, anchor_price=anchor_price,
                          entry_price_raw=entry_bar["open"], night_close=night_close,
                          pre_move=pre_move, post_move=post_move))

    res = pd.DataFrame(rows)
    print(f"有效夜盤樣本數: {len(res)}")
    corr = res["pre_move"].corr(res["post_move"])
    print(f"前段走勢 vs 後段走勢 相關係數: {corr:.3f}")

    print("\n--- 動能規則：前段漲(跌)超過門檻，22:00後續做多(空)，抱到05:00收盤 ---")
    for threshold in [0, 10, 20, 30, 50]:
        sub = res[res["pre_move"].abs() >= threshold].copy()
        if len(sub) < 10:
            print(f"門檻{threshold}點: 樣本數不足({len(sub)})，略過")
            continue
        direction = np.sign(sub["pre_move"])
        # 用進場K棒開盤價(entry_price_raw)成交，扣夜盤滑價；出場用夜盤收盤價，同樣扣滑價
        entry_price = sub["entry_price_raw"] + direction * (NIGHT_COST.slippage_points_round_trip / 2)
        exit_price = sub["night_close"] - direction * (NIGHT_COST.slippage_points_round_trip / 2)
        pnl_points = (exit_price - entry_price) * direction
        trades = pd.DataFrame(dict(pnl_points=pnl_points, entry_price=entry_price, exit_price=exit_price))
        zero_slip_cost = TradeCost("僅稅+手續費", commission_round_trip=60.0)
        t = apply_costs(trades, zero_slip_cost)
        print(f"門檻{threshold}點: n={len(sub)}, t={tstat(pnl_points):.3f}, "
              f"mean_net(已含夜盤滑價)={t['net_twd'].mean():.1f}, sum_net={t['net_twd'].sum():,.0f}")


if __name__ == "__main__":
    main()
