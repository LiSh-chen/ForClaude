"""策略實驗室統一匯出：跑 tw_quant/strategy_lab.py 註冊的策略（任意組合、
任意濾網），輸出格式跟三腿策略歷史交易紀錄最終版相容（日K總覽data.json +
逐年1分K+交易紀錄 years/*.json），供 strategy_lab_chart/index.html 顯示。

**改參數的方法**：改下面 ACTIVE 這個列表——每個項目是
(strategy_id, cfg覆寫或None, 濾網鏈)，改完重新執行這支腳本，
前端頁面重新整理就會看到新結果。不是網頁端即時運算。

用法：
    python scripts/export_strategy_lab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.technical_indicators import build_daily_bars, daily_indicators  # noqa: E402
from tw_quant.strategy_lab import (  # noqa: E402
    build_registry, run_strategy, combine, apply_volume_filter, VolumeFilterSpec,
    apply_trend_regime_filter, TrendRegimeFilterSpec,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "scripts" / "_strategy_lab_export"
YEARS_DIR = OUT_DIR / "years"
LOW_COST = COST_SCENARIOS[0]  # 來回30元，這次會話統一使用的計算基礎
IS_CUTOFF = "2021-01-01"

# ============================================================
# 這裡改參數：每個 tuple = (strategy_id, cfg覆寫(None=用註冊時的預設),
# 濾網函式list(每個是 (df, trades) -> trades 的callable，None=不加濾網))
# strategy_id 對照 tw_quant/strategy_lab.py 的 build_registry()：
#   og=開盤上衝, lu=午盤放空, re=盤中翻多, vwap_trend=VWAP趨勢日,
#   sr_breakout=支撐壓力順勢突破, sr_fade=支撐壓力逆勢fade, donchian=唐奇安,
#   rsi2=RSI2均值回歸, orb=開盤區間突破, liqsweep=ICT流動性掃單, night=夜盤熱區
# ============================================================
ACTIVE = [
    ("og", None, None),
    ("lu", None, None),
    ("re", None, "volume_filter"),  # 盤中翻多沿用原本的成交量濾網
    ("vwap_trend", None, None),
    ("sr_breakout", None, None),
    ("sr_fade", None, None),
    ("donchian", None, None),
    ("rsi2", None, None),
    ("orb", None, None),
    ("liqsweep", None, None),
    ("night", None, None),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    YEARS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    reg = build_registry()

    vol_threshold = daily_indicators(is_df)["vol_ratio_lag1"].quantile(2 / 3)

    print("=" * 70)
    print("1) 逐一跑註冊策略")
    print("=" * 70)
    all_trades = {}
    meta = {}
    for sid, cfg_override, filt in ACTIVE:
        trades = run_strategy(reg, sid, df, cfg_override)
        if filt == "volume_filter":
            trades = apply_volume_filter(trades, df, VolumeFilterSpec(threshold=vol_threshold))
        elif filt == "trend_regime_filter":
            trades = apply_trend_regime_filter(trades, df, TrendRegimeFilterSpec())
        trades = apply_costs(trades, LOW_COST)
        all_trades[sid] = trades
        meta[sid] = dict(label=reg[sid].label, verdict=reg[sid].verdict, n=len(trades),
                          sum_net=round(float(trades["net_twd"].sum()), 1) if len(trades) else 0.0)
        print(f"  {sid:12s} {reg[sid].label:12s} verdict={reg[sid].verdict:10s} "
              f"n={len(trades):5d}  net_sum={meta[sid]['sum_net']:>12,.0f}")

    print("\n" + "=" * 70)
    print("2) 建立日K總覽 data.json")
    print("=" * 70)
    daily = build_daily_bars(df).sort_values("date").reset_index(drop=True)

    # 每個策略的「每日已實現淨損益」用「出場日」歸屬（多日持有的策略只在平倉
    # 那天記一筆已實現損益，不是逐日mark-to-market）
    daily_by_sid = {}
    for sid, trades in all_trades.items():
        if trades.empty:
            daily_by_sid[sid] = pd.Series(dtype=float)
            continue
        g = trades.groupby("exit_date")["net_twd"].sum()
        g.index = pd.to_datetime(g.index)
        daily_by_sid[sid] = g

    records = []
    running_eq = {sid: 0.0 for sid in all_trades}
    total_eq = 0.0
    for _, row in daily.iterrows():
        d = row["date"]
        rec = {"t": d.strftime("%Y-%m-%d"), "o": int(row["open"]), "h": int(row["high"]),
               "l": int(row["low"]), "c": int(row["close"])}
        day_total = 0.0
        for sid in all_trades:
            val = daily_by_sid[sid].get(d, 0.0)
            rec[sid + "_n"] = round(float(val), 1)
            running_eq[sid] += val
            day_total += val
        rec["net"] = round(day_total, 1)
        total_eq += day_total
        rec["eq"] = round(total_eq, 1)
        records.append(rec)

    with open(OUT_DIR / "data.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    print(f"data.json: {len(records)} 筆日資料")

    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"strategies": meta, "cost_label": LOW_COST.label,
                    "order": [sid for sid, _, _ in ACTIVE]}, f, ensure_ascii=False, indent=2)
    print("meta.json 已寫入")

    print("\n" + "=" * 70)
    print("3) 建立逐年 1分K + 交易紀錄")
    print("=" * 70)
    from datetime import time
    t = df["datetime"].dt.time
    day_session = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_session["date_str"] = day_session["datetime"].dt.strftime("%Y-%m-%d")
    day_session["moy"] = day_session["datetime"].dt.hour * 60 + day_session["datetime"].dt.minute
    day_session["year"] = day_session["datetime"].dt.year

    all_trades_combined = combine(list(all_trades.values()))
    # 用「進場年份 或 出場年份」把交易複製進相關年度檔案，讓跨年度持有的交易
    # 在對應的月份都能正確顯示進/出場標記
    all_trades_combined["entry_year"] = pd.to_datetime(all_trades_combined["entry_date"]).dt.year
    all_trades_combined["exit_year"] = pd.to_datetime(all_trades_combined["exit_date"]).dt.year
    net_map = {}
    for sid, trades in all_trades.items():
        for _, r in trades.iterrows():
            net_map[(sid, r["entry_date"], r["exit_date"])] = round(float(r["net_twd"]), 1)

    for year, g in day_session.groupby("year"):
        days_out = []
        for d, dg in g.groupby("date_str"):
            dg = dg.sort_values("datetime")
            bars = dg[["moy", "open", "high", "low", "close"]].values.astype(int).tolist()
            days_out.append({"d": d, "bars": bars})

        yr_trades = all_trades_combined[(all_trades_combined["entry_year"] == year) |
                                          (all_trades_combined["exit_year"] == year)]
        trades_out = []
        for _, r in yr_trades.iterrows():
            net = net_map.get((r["strategy_id"], r["entry_date"], r["exit_date"]), 0.0)
            trades_out.append({
                "leg": r["strategy_id"], "dir": r["direction"],
                "ed": r["entry_date"], "et": r["entry_time"],
                "xd": r["exit_date"], "xt": r["exit_time"],
                "en": round(float(r["entry_price"]), 1), "ex": round(float(r["exit_price"]), 1),
                "n": net, "approx": bool(r["approx_time"]),
            })

        payload = {"days": days_out, "trades": trades_out}
        with open(YEARS_DIR / f"{year}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"  {year}: {len(days_out)} 交易日, {len(trades_out)} 筆交易(含跨年複製)")

    print("\n完成，輸出於", OUT_DIR)


if __name__ == "__main__":
    main()
