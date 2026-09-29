"""測試方案2：低波動期間賣選擇權勒式(short strangle)收權利金。

做法：
1. 用 taifex_tx_multi_contract.parquet 的 near_expiry 欄位找出每個近月合約
   「變成近月的第一天」(=換月日，也是新一輪策略的進場日)。
2. 只在換月日當天的體制(前一日收盤已知)是低波動時，才進場這個月份的勒式。
3. 進場日：抓當天TXO該到期月份的完整履約價鏈，選離現貨約3%的價外
   買權+賣權各賣1口，收權利金(用結算價)。
4. 到期日：用該合約在near-month序列裡最後一天的near_price當最終結算價
   （近似，不是TAIFEX官方到期特別結算價，因為抓不到那個特殊數字，用
   最後一個交易日的收盤/結算價當代理）。
5. P&L = 收到的權利金 - 到期時價內的賠付金額，換算TWD(點值50)。

只用IS(2001-2020)先看整體+子區間穩健性，OOS/2024-2026視情況再花。

用法：
    python scripts/test_low_vol_strangle.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
REGIME_PATH = REPO_ROOT / "data" / "regime_classification_daily.parquet"
OUT_PATH = REPO_ROOT / "data" / "low_vol_strangle_trades.parquet"
ENDPOINT = "https://www.taifex.com.tw/cht/3/optDataDown"
POINT_VALUE = 50.0
OTM_PCT = 0.03
IS_CUTOFF = "2021-01-01"


def fetch_day_options(date_str: str, expiry: str) -> pd.DataFrame | None:
    """date_str: YYYY/MM/DD，抓那一天全部TXO資料，篩出指定到期月份。"""
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["curl", "-sSL", "-m", "40", "-X", "POST", ENDPOINT,
                 "-d", "down_type=1", "-d", "commodity_id=TXO", "-d", "commodity_id2=",
                 "-d", f"queryStartDate={date_str}", "-d", f"queryEndDate={date_str}"],
                capture_output=True, timeout=45,
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                text = result.stdout.decode("big5", errors="replace")
                if "交易日期" not in text:
                    return None
                df = pd.read_csv(io.StringIO(text), dtype=str, index_col=False)
                df.columns = [c.strip() for c in df.columns]
                df["到期月份(週別)"] = df["到期月份(週別)"].str.strip()
                df["交易時段"] = df["交易時段"].str.strip()
                df["買賣權"] = df["買賣權"].str.strip()
                df = df[(df["交易時段"] == "一般") & (df["到期月份(週別)"] == expiry)]
                if df.empty:
                    return None
                df["strike"] = pd.to_numeric(df["履約價"], errors="coerce")
                df["settlement"] = pd.to_numeric(df["結算價"].str.replace(",", "", regex=False), errors="coerce")
                df["close"] = pd.to_numeric(df["收盤價"].str.replace(",", "", regex=False), errors="coerce")
                df["price"] = df["settlement"].fillna(df["close"])
                df.loc[df["price"] == 0, "price"] = np.nan
                return df[["strike", "買賣權", "price"]].dropna()
        except Exception as e:
            print(f"    fetch失敗 attempt{attempt+1}: {e}")
        time.sleep(3)
    return None


def pick_strike(options: pd.DataFrame, side: str, target: float) -> float | None:
    sub = options[options["買賣權"] == side]
    if sub.empty:
        return None
    idx = (sub["strike"] - target).abs().idxmin()
    return sub.loc[idx, "strike"]


def main() -> None:
    tx = pd.read_parquet(TX_PATH)[["date", "near_expiry", "near_price"]].sort_values("date").reset_index(drop=True)
    regime = pd.read_parquet(REGIME_PATH)[["date", "high_vol_regime"]]
    df = tx.merge(regime, on="date", how="left")

    # 每個near_expiry第一天=換月日(進場日)，最後一天=到期日(結算日)
    grouped = df.groupby("near_expiry")
    rollover_days = grouped["date"].min().reset_index().rename(columns={"date": "entry_date"})
    expiry_settle = grouped.agg(expiry_date=("date", "max"), final_price=("near_price", "last")).reset_index()
    contracts = rollover_days.merge(expiry_settle, on="near_expiry")
    contracts = contracts.merge(df[["date", "high_vol_regime", "near_price"]],
                                 left_on="entry_date", right_on="date", how="left")
    contracts = contracts.rename(columns={"near_price": "entry_spot"}).drop(columns=["date"])

    low_vol_contracts = contracts[(contracts["high_vol_regime"] == 0.0) &
                                   (contracts["entry_date"] < IS_CUTOFF)].sort_values("entry_date")
    print(f"IS(2001-2020)期間，換月日剛好是低波動體制的合約數: {len(low_vol_contracts)}")

    trades = []
    for i, row in enumerate(low_vol_contracts.itertuples(), 1):
        entry_str = row.entry_date.strftime("%Y/%m/%d")
        opts = fetch_day_options(entry_str, row.near_expiry)
        if opts is None:
            print(f"[{i}/{len(low_vol_contracts)}] {row.near_expiry} ({entry_str}): 抓不到資料，跳過")
            continue
        call_target = row.entry_spot * (1 + OTM_PCT)
        put_target = row.entry_spot * (1 - OTM_PCT)
        call_strike = pick_strike(opts, "買權", call_target)
        put_strike = pick_strike(opts, "賣權", put_target)
        if call_strike is None or put_strike is None:
            print(f"[{i}/{len(low_vol_contracts)}] {row.near_expiry}: 找不到履約價，跳過")
            continue
        call_premium = opts[(opts["買賣權"] == "買權") & (opts["strike"] == call_strike)]["price"].iloc[0]
        put_premium = opts[(opts["買賣權"] == "賣權") & (opts["strike"] == put_strike)]["price"].iloc[0]

        final = row.final_price
        call_payout = max(final - call_strike, 0)
        put_payout = max(put_strike - final, 0)
        pnl_points = (call_premium + put_premium) - (call_payout + put_payout)

        trades.append(dict(
            entry_date=row.entry_date, expiry=row.near_expiry, expiry_date=row.expiry_date,
            entry_spot=row.entry_spot, final_price=final,
            call_strike=call_strike, put_strike=put_strike,
            call_premium=call_premium, put_premium=put_premium,
            pnl_points=pnl_points, pnl_twd=pnl_points * POINT_VALUE,
        ))
        if i % 20 == 0 or i == len(low_vol_contracts):
            print(f"[{i}/{len(low_vol_contracts)}] 進度：已完成{len(trades)}筆")
        time.sleep(2)

    trades_df = pd.DataFrame(trades)
    trades_df.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")

    if trades_df.empty:
        print("沒有成功的交易，無法統計")
        return

    n = len(trades_df)
    t = trades_df["pnl_points"].mean() / (trades_df["pnl_points"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    print(f"\n總筆數: {n}")
    print(f"平均每筆損益: {trades_df['pnl_points'].mean():.1f}點 ({trades_df['pnl_twd'].mean():,.0f}元)")
    print(f"t值: {t:.2f}")
    print(f"總損益: {trades_df['pnl_twd'].sum():,.0f}元")
    print(f"勝率: {(trades_df['pnl_points'] > 0).mean()*100:.1f}%")
    print(f"最大單筆虧損: {trades_df['pnl_twd'].min():,.0f}元")


if __name__ == "__main__":
    main()
