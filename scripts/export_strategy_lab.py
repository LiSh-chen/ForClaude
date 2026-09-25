"""策略實驗室統一匯出：跑 tw_quant/strategy_lab.py 註冊的策略（任意組合、
任意濾網），輸出格式跟三腿策略歷史交易紀錄最終版相容（日K總覽data.json +
逐年1分K+交易紀錄 years/*），供 strategy_lab_chart/index.html 顯示。

**這是「初始資料」匯出，不是參數的唯一調整入口**：前端頁面本身內建了用
JavaScript重新實作的回測引擎（strategy_lab_chart/engine.js，跟這支腳本
呼叫的Python策略模組做過逐筆對拍驗證），使用者可以直接在網頁上調整每個
策略的參數即時重跑，不需要改這支腳本。這支腳本主要負責把「目前註冊的
預設參數」批次跑一次，產生頁面首次載入時的初始畫面，以及提供engine.js
需要的原始1分K/日K資料（含volume、含夜盤區段）。如果要新增/移除註冊的
策略或改變預設組合，才需要改下面 ACTIVE 列表重新執行。

輸出格式：
- data.json / meta.json：日K總覽 + 策略中繼資料（跟之前一樣）。
- years/{year}.json：該年度交易紀錄（進出場時間價位損益），只給「1分K單月
  交易明細」的標記用，不再含K棒本身。
- years/{year}_day.bin / years/{year}_night.bin：該年度日盤(08:45-13:45)/
  夜盤熱區(21:00-23:45)1分K（含volume），緊湊二進位格式取代JSON陣列（見
  write_minute_bin() 說明），前端K線圖跟JS回測引擎共用同一份資料。
- trades_index.json：{策略id: [[進場日,進場時間,出場日,出場時間], ...]}，
  只給前端算「併發口數/保證金需求」用（多個策略同時掛倉需要多少保證金），
  含時間才能正確判斷同一天觸發的不同交易時段有沒有真的重疊，不含價位損益。

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
    print("1b) 建立 trades_index.json（每個策略的進/出場日期，供前端算併發口數/保證金用）")
    print("=" * 70)
    # 存進出場時間(不只日期)，讓前端可以用精確的時間區間做掃描線演算法算併發
    # 口數——三腿策略同一天三腿都會觸發，但彼此時段依序不重疊(08:45-09:00/
    # 12:00-12:30/12:30-13:00)，只比較日期會誤判成同時併發3口。
    trades_index = {}
    for sid, trades in all_trades.items():
        trades_index[sid] = [[r["entry_date"], r["entry_time"], r["exit_date"], r["exit_time"]]
                              for _, r in trades.iterrows()]
    with open(OUT_DIR / "trades_index.json", "w", encoding="utf-8") as f:
        json.dump(trades_index, f, separators=(",", ":"))
    print(f"trades_index.json 已寫入，共 {sum(len(v) for v in trades_index.values())} 筆交易的進出場日期")

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
               "l": int(row["low"]), "c": int(row["close"]), "v": int(row["volume"])}
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
        json.dump({
            "strategies": meta, "cost_label": LOW_COST.label,
            "order": [sid for sid, _, _ in ACTIVE],
            # 給前端JS回測引擎用的固定常數：re腿的量能濾網門檻是用IS(2001-2020)
            # 資料算出來的三分位數，前端不能重新計算（會用到OOS資料，變相看未來），
            # 一定要用這個匯出時鎖定的固定值。
            "re_volume_filter_threshold": round(float(vol_threshold), 6),
            "cost_scenarios": [
                {"label": c.label, "commission_round_trip": c.commission_round_trip,
                 "tax_rate_per_side": c.tax_rate_per_side, "point_value": c.point_value}
                for c in COST_SCENARIOS
            ],
        }, f, ensure_ascii=False, indent=2)
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

    # 夜盤流動性熱區（21:00-23:45，night策略需要）；這個時間窗不跨零點，
    # 交易日直接用K棒自己的日曆日期分組，跟 night_liquid_window_trend_strategy.py 一致
    night_session = df[(t >= time(21, 0)) & (t <= time(23, 45))].copy()
    night_session["date_str"] = night_session["datetime"].dt.strftime("%Y-%m-%d")
    night_session["moy"] = night_session["datetime"].dt.hour * 60 + night_session["datetime"].dt.minute
    night_session["year"] = night_session["datetime"].dt.year

    all_trades_combined = combine(list(all_trades.values()))
    # 用「進場年份 或 出場年份」把交易複製進相關年度檔案，讓跨年度持有的交易
    # 在對應的月份都能正確顯示進/出場標記
    all_trades_combined["entry_year"] = pd.to_datetime(all_trades_combined["entry_date"]).dt.year
    all_trades_combined["exit_year"] = pd.to_datetime(all_trades_combined["exit_date"]).dt.year
    net_map = {}
    for sid, trades in all_trades.items():
        for _, r in trades.iterrows():
            net_map[(sid, r["entry_date"], r["exit_date"])] = round(float(r["net_twd"]), 1)

    night_by_year = {y: g for y, g in night_session.groupby("year")}

    def write_minute_bin(path, session_df):
        """把一年份的1分K（含volume）打包成緊湊二進位格式，取代JSON陣列——
        JSON把每個整數編碼成文字（每個數字約5~9 bytes含逗號/括號），二進位
        固定4 bytes/數字，23年份加總下來省將近一半體積，也讓前端在「全歷史
        即時重算」時抓取全部年份的1分K快很多（不用JSON.parse幾百萬個數字）。
        格式（little-endian int32，瀏覽器原生Int32Array預設讀法就是這個）：
        [numDays, (dateYYYYMMDD, barCount, [moy,o,h,l,c,v]*barCount)*numDays]
        """
        import numpy as np
        header = [len(session_df.groupby("date_str")) if not session_df.empty else 0]
        blocks = [np.array(header, dtype="<i4")]
        if not session_df.empty:
            for d, dg in session_df.groupby("date_str"):
                dg = dg.sort_values("datetime")
                date_int = int(d.replace("-", ""))
                bars = dg[["moy", "open", "high", "low", "close", "volume"]].values.astype("<i4")
                blocks.append(np.array([date_int, len(dg)], dtype="<i4"))
                blocks.append(bars.reshape(-1))
        flat = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
        flat.astype("<i4").tofile(path)

    for year, g in day_session.groupby("year"):
        write_minute_bin(YEARS_DIR / f"{year}_day.bin", g)
        ng = night_by_year.get(year)
        write_minute_bin(YEARS_DIR / f"{year}_night.bin", ng if ng is not None else g.iloc[0:0])

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

        payload = {"trades": trades_out}
        with open(YEARS_DIR / f"{year}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        n_day = len(g.groupby("date_str")) if not g.empty else 0
        n_night = len(ng.groupby("date_str")) if ng is not None else 0
        print(f"  {year}: {n_day} 交易日(bin,含量能), {n_night} 夜盤交易日(bin), "
              f"{len(trades_out)} 筆交易(含跨年複製)")

    print("\n完成，輸出於", OUT_DIR)


if __name__ == "__main__":
    main()
