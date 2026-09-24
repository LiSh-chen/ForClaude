"""在「其他時段」（不是已驗證的 08:45-09:00 / 12:00-12:30 / 12:30-13:00）上，
用更多技術指標（KD、Williams %R、ATR、CCI，加上原本的 RSI/MACD/布林/成交量）
找看看有沒有救得起來的候選。

檢查對象：
1. 09:00-09:15 立即回吐（空）——之前被否決（子區間衰減到不顯著）
2. 13:15-13:45 收盤前上衝（多）——之前被否決，這裡改用 RSI 高分桶篩選看
   能不能救回來
3. 夜盤 23:45(進場)->01:30+1(出場) 做多——樣本內掃描時最強的夜盤候選，但
   t 值本來就沒到 |t|>=3 的門檻

結論：
- 09:00-09:15：RSI 高分桶全樣本 t=4.67，但 2016-2020 子區間 t=0.10，完全
  消失，維持否決。
- 13:15-13:45：RSI 高分桶（門檻用樣本內算，寫死給樣本外套用）樣本內看起來
  不錯（filtered mid-cost net=+45,155），但**樣本外驗證直接失效**：原版
  OOS t 只有 0.17（邊際幾乎消失），RSI 濾網後 OOS t=0.58、扣成本三種情境
  全部虧錢。否決。
- 夜盤 23:45->01:30+1：8 個指標分桶後沒有一個 |t|>=3，而且不同指標指向
  的最強分桶彼此不一致（不像日盤那樣動能指標一致指向同一個方向），判定
  是雜訊，樣本量也不夠（每個分桶只有 n≈296）。否決。

這一輪沒有找到新的可用策略；目前唯一驗證過、通過樣本外檢驗的仍然是原本
三個時段（開盤上衝、午盤放空、盤中翻多+成交量濾網）。

用法：
    python scripts/explore_other_windows_with_indicators.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.technical_indicators import LAG1_COLUMNS, add_indicators, build_daily_bars, daily_indicators  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
IS_CUTOFF = "2021-01-01"


def day_leg_trades(data_df: pd.DataFrame, start_min: int, end_min: int, direction: str) -> pd.DataFrame:
    t = data_df["datetime"].dt.time
    day_df = data_df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_df["minute_of_day"] = day_df["datetime"].dt.hour * 60 + day_df["datetime"].dt.minute
    day_df["date"] = day_df["datetime"].dt.date

    rows = []
    for d, g in day_df.groupby("date"):
        g = g.sort_values("datetime")
        mo, op = g["minute_of_day"].to_numpy(), g["open"].to_numpy()
        pos_s, pos_e = np.searchsorted(mo, start_min), np.searchsorted(mo, end_min)
        if pos_s >= len(mo) or pos_e >= len(mo):
            continue
        ret = op[pos_e] - op[pos_s]
        if direction == "short":
            ret = -ret
        rows.append(dict(trading_date=d, entry_price=op[pos_s], exit_price=op[pos_e], pnl_points=ret))
    trades = pd.DataFrame(rows)
    trades["pnl_twd"] = trades["pnl_points"] * 50
    return trades


def bucket_scan(trades: pd.DataFrame, indicators: pd.DataFrame, date_col: str = "trading_date") -> None:
    merged = trades.merge(indicators, left_on=date_col, right_on="date", how="left")
    for col in [c for c in indicators.columns if c not in ("date",)]:
        s = merged.dropna(subset=[col]).copy()
        s["bucket"] = pd.qcut(s[col], 3, labels=["低", "中", "高"], duplicates="drop")
        g = s.groupby("bucket", observed=True)["pnl_points"].agg(["size", "mean"])
        g["tstat"] = s.groupby("bucket", observed=True)["pnl_points"].apply(
            lambda x: x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 else np.nan
        )
        best = g["tstat"].abs().idxmax()
        print(f"  {col}: " + "  ".join(f"{b}t={g.loc[b,'tstat']:.2f}" for b in g.index) +
              f"  (最強: {best}, n={g.loc[best,'size']})")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    print("=== 09:00->09:15 fade（空），全部指標分桶（樣本內）===")
    fade_trades = day_leg_trades(is_df, 9 * 60, 9 * 60 + 15, "short")
    bucket_scan(fade_trades, daily_indicators(is_df))

    print("\n=== 13:15->13:45 close rally（多），全部指標分桶（樣本內）===")
    rally_trades_is = day_leg_trades(is_df, 13 * 60 + 15, 13 * 60 + 45, "long")
    bucket_scan(rally_trades_is, daily_indicators(is_df))

    rsi_threshold = daily_indicators(is_df)["rsi14_lag1"].quantile(2 / 3)
    print(f"\n=== 13:15->13:45 RSI 濾網 OOS 驗證（門檻={rsi_threshold:.2f}，樣本內算出後寫死）===")
    for label, data_df in [("樣本內", is_df), ("樣本外（只驗證一次）", oos_df)]:
        trades = day_leg_trades(data_df, 13 * 60 + 15, 13 * 60 + 45, "long")
        ind = daily_indicators(data_df)
        merged = trades.merge(ind[["date", "rsi14_lag1"]], left_on="trading_date", right_on="date", how="left")
        filtered = merged[merged["rsi14_lag1"] >= rsi_threshold]
        for name, t in [("原版", trades), ("RSI濾網後", filtered)]:
            se = t["pnl_points"].std() / np.sqrt(len(t))
            tax = 0.00002 * 50 * (t["entry_price"].abs() + t["exit_price"].abs())
            nets = {c: (t["pnl_twd"] - tax - c).sum() for c in (30, 60, 100)}
            print(f"  [{label}] {name}: n={len(t)} tstat={t['pnl_points'].mean()/se:.2f} " +
                  "  ".join(f"net{c}={v:,.0f}" for c, v in nets.items()))

    print("\n=== 夜盤 23:45->01:30(+1) 做多，全部指標分桶（樣本內，用當天而非落後指標）===")
    night_daily = add_indicators(build_daily_bars(is_df))
    night_daily["date"] = night_daily["date"].dt.date
    ind_cols = ["rsi14", "macd_hist", "bb_pctb", "vol_ratio", "stoch_k", "willr14", "atr_ratio", "cci20"]
    night_indicators = night_daily[["date"] + ind_cols]

    t = is_df["datetime"].dt.time
    night_df = is_df[(t >= time(15, 0)) | (t < time(5, 0))].copy()
    night_df["minute_of_day"] = night_df["datetime"].dt.hour * 60 + night_df["datetime"].dt.minute
    early = night_df["datetime"].dt.time < time(5, 0)
    night_df.loc[early, "minute_of_day"] += 1440
    night_df["session_date"] = night_df["datetime"].dt.date
    night_df.loc[early, "session_date"] = night_df.loc[early, "datetime"].dt.date - pd.Timedelta(days=1)

    rows = []
    for sd, g in night_df.groupby("session_date"):
        g = g.sort_values("datetime")
        mo, op = g["minute_of_day"].to_numpy(), g["open"].to_numpy()
        s_min, e_min = 23 * 60 + 45, 1 * 60 + 30 + 1440
        pos_s, pos_e = np.searchsorted(mo, s_min), np.searchsorted(mo, e_min)
        if pos_s >= len(mo) or pos_e >= len(mo):
            continue
        rows.append(dict(session_date=sd, pnl_points=op[pos_e] - op[pos_s]))
    night_trades = pd.DataFrame(rows)
    bucket_scan(night_trades, night_indicators, date_col="session_date")

    print("\n結論：這一輪沒有找到新的可用策略，三個已驗證時段維持不變。")


if __name__ == "__main__":
    main()
