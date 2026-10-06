"""績效追蹤：對每一期推薦（及整個候選池）計算進場後 5/20/60/120 個交易日與迄今的報酬、超額報酬、停損/目標觸及。"""
from __future__ import annotations
import datetime as dt

import pandas as pd

from . import config as C
from .utils import isnum, market_of

HORIZONS = [5, 20, 60, 120]


def _fwd(df: pd.DataFrame, bench: pd.DataFrame, entry_date: str, entry_px: float | None = None) -> dict | None:
    """以 entry_date（含）之前最後一個收盤為進場價，計算各期報酬。"""
    c = df["Close"].dropna()
    idx = c.index.searchsorted(pd.Timestamp(entry_date), side="right") - 1
    if idx < 0:
        return None
    e = c.iloc[idx] if entry_px is None else entry_px
    out = {"entry": float(e), "n_days": int(len(c) - 1 - idx)}
    bc = bench["Close"].dropna()
    for h in HORIZONS + ["last"]:
        j = len(c) - 1 if h == "last" else idx + h
        if j > len(c) - 1:
            out[f"r{h}"], out[f"a{h}"] = None, None
            continue
        d = c.index[j]
        r = float(c.iloc[j] / e - 1)
        bi = bc.index.searchsorted(pd.Timestamp(entry_date), side="right") - 1
        bj = bc.index.searchsorted(d, side="right") - 1
        br = float(bc.iloc[bj] / bc.iloc[bi] - 1) if bi >= 0 and bj >= 0 else None
        out[f"r{h}"] = r
        out[f"a{h}"] = r - br if br is not None else None
    path = df.iloc[idx + 1:]
    if len(path):
        out["max_dd"] = float((path["Close"] / e - 1).min())
        out["max_high"] = float(path["High"].max() / e - 1)
        out["min_low"] = float(path["Low"].min() / e - 1)
    else:
        out["max_dd"] = out["max_high"] = out["min_low"] = 0.0
    out["last_price"] = float(c.iloc[-1])
    out["last_date"] = str(c.index[-1].date())
    return out


def summarize(positions: list[dict]) -> dict:
    s = {}
    for h in HORIZONS:
        xs = [p for p in positions if p.get(f"r{h}") is not None]
        al = [p[f"a{h}"] for p in xs if p.get(f"a{h}") is not None]
        s[str(h)] = {"n": len(xs), "avg_ret": sum(p[f"r{h}"] for p in xs) / len(xs) if xs else None,
                     "avg_alpha": sum(al) / len(al) if al else None,
                     "win": sum(1 for p in xs if p[f"r{h}"] > 0) / len(xs) if xs else None,
                     "win_alpha": sum(1 for a in al if a > 0) / len(al) if al else None}
    return s


def run_tracker(provider) -> dict:
    uni = C.load_universe()
    data = C.path("data")
    pick_files = sorted((data / "picks").glob("*.json"))
    if not pick_files:
        print("[track] 還沒有任何推薦紀錄。")
        C.save_json(data / "performance.json", {"generated": str(dt.date.today()), "positions": [], "summary": {}, "cohorts": []})
        return {}
    picks_by_date = {f.stem: C.load_json(f) for f in pick_files}
    tickers = {p["ticker"] for pj in picks_by_date.values() for p in pj["picks"]}
    for f in sorted((data / "candidates").glob("*.json")) if (data / "candidates").exists() else []:
        tickers |= {r["ticker"] for r in C.load_json(f)["rows"]}
    benches = set(uni["benchmarks"].values())
    prices = provider.prices(sorted(tickers | benches))
    positions, cohorts = [], []
    for date, pj in picks_by_date.items():
        coh = []
        for p in pj["picks"]:
            df, b = prices.get(p["ticker"]), prices.get(uni["benchmarks"][p["market"]])
            if df is None or b is None:
                continue
            f = _fwd(df, b, date, p["entry_price"])
            if not f:
                continue
            tg = p["target"]
            n = max(f["n_days"], 0)
            exp = (1 + tg["upside"]) ** (n / tg["horizon_days"]) - 1 if n else 0.0
            rec = {"date": date, "ticker": p["ticker"], "name": p["zh_name"], "market": p["market"], "sector": p["sector"],
                   "entry": p["entry_price"], "target": tg["base"], "target_upside": tg["upside"], "stop": p["stop"],
                   "expected_ret_now": exp, "ret_vs_expected": f["r" + "last"] - exp, **{k: v for k, v in f.items() if k not in ("entry",)},
                   "stop_hit": f["min_low"] <= p["stop"] / p["entry_price"] - 1, "target_hit": f["max_high"] >= tg["upside"],
                   "params_version": pj.get("params_version")}
            positions.append(rec)
            coh.append(rec)
        if coh:
            cohorts.append({"date": date, "n": len(coh), "avg_ret": sum(x["rlast"] for x in coh) / len(coh),
                            "avg_alpha": (sum(x["alast"] for x in coh if x["alast"] is not None) / max(1, sum(1 for x in coh if x["alast"] is not None))),
                            "n_days": coh[0]["n_days"]})

    # 候選池前瞻報酬（用於檢驗各因子的預測力，而非只看被選中的股票）
    pool = {}
    for f in sorted((data / "candidates").glob("*.json")) if (data / "candidates").exists() else []:
        cj = C.load_json(f)
        day = {}
        for r in cj["rows"]:
            df, b = prices.get(r["ticker"]), prices.get(uni["benchmarks"][r["market"]])
            if df is None or b is None:
                continue
            fw = _fwd(df, b, cj["date"], r["price"])
            if fw:
                day[r["ticker"]] = {k: (round(fw[k], 5) if isnum(fw.get(k)) else None) for k in ("r5", "a5", "r20", "a20", "r60", "a60", "r120", "a120", "rlast", "alast")}
        pool[cj["date"]] = day
    C.save_json(data / "pool_returns.json", pool)
    perf = {"generated": str(dt.date.today()), "positions": positions, "cohorts": cohorts, "summary": summarize(positions),
            "by_market": {m: summarize([p for p in positions if p["market"] == m]) for m in ("TW", "US")}}
    C.save_json(data / "performance.json", perf)
    print(f"[track] 追蹤 {len(positions)} 筆推薦；候選池 {sum(len(v) for v in pool.values())} 筆觀察。")
    return perf
