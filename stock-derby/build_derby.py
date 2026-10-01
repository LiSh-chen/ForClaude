"""Build data/derby.json: the race card (賽馬報) + race replay for every pool x distance.

pool     : spx (S&P 500) | ndx (Nasdaq-100) | all (union)
distance : 1m / 3m / 6m  = 21 / 63 / 126 trading days
field    : the 10 most-traded names (avg dollar volume, 60 days before the start)
Everything shown on the card (近績 / 跑法 / 能力) uses data up to the race start only;
the race itself is the most recent completed window ending on the latest trading day.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader as dl
import factors as F

DISTANCES = {"1m": 21, "3m": 63, "6m": 126}
FIELD = 10
N_FORM = 4
KEY_FACTORS = Path(__file__).parent / "data" / "key_factors.json"
OUT = Path(__file__).parent / "data" / "derby.json"
SECTOR_ZH = {"Information Technology": "科技", "Communication Services": "通訊", "Consumer Discretionary": "非必需消費",
             "Consumer Staples": "必需消費", "Financials": "金融", "Health Care": "醫療", "Industrials": "工業",
             "Energy": "能源", "Utilities": "公用事業", "Real Estate": "房地產", "Materials": "原物料"}


def style_label(early_rank: float) -> tuple[str, str]:
    if early_rank <= 2.5:
        return "領放", "起跑即搶頭香，靠前段優勢領先"
    if early_rank <= 4.5:
        return "跟前", "緊跟領先集團，伺機而動"
    if early_rank <= 7:
        return "居中", "中段待機，不急不徐"
    return "後上", "前段墊後，靠末腳追趕"


def ranks_of(returns: pd.Series) -> pd.Series:
    """1 = best; horses without data (NaN) get NaN and are ignored by the others."""
    return returns.rank(ascending=False, method="min")


def window_ranks(close: pd.DataFrame, field: list[str], lo: int, hi: int) -> tuple[pd.Series, pd.Series] | None:
    """(rank at 1/3 point, rank at the finish) among the field for window [lo, hi]; None if no history."""
    if lo < 0:
        return None
    w = close.iloc[lo:hi + 1][field]
    cum = w / w.iloc[0] - 1
    third = max(1, (hi - lo) // 3)
    return ranks_of(cum.iloc[third]), ranks_of(cum.iloc[-1])


def stars(score: float) -> int:
    return int(np.clip(1 + score // 20, 1, 5))


def comment(h: dict, factors: list[dict]) -> str:
    ab = sorted(h["ability"].items(), key=lambda kv: -kv[1])
    labels = {f["id"]: f["label"] for f in factors}
    parts = []
    if h["form"] and h["form"][0] is not None:
        wins = sum(1 for x in h["form"] if x is not None and x <= 3)
        parts.append(f"近{sum(x is not None for x in h['form'])}戰{wins}次進前三" if wins else "近績欠佳")
    parts.append(f"跑法{h['style']}")
    parts.append(f"強項「{labels[ab[0][0]]}」")
    if ab[-1][1] < 35:
        parts.append(f"弱點「{labels[ab[-1][0]]}」")
    return "，".join(parts)


def build_race(pool_tickers, panels_, fac, kf, sector_of, dk):
    N = DISTANCES[dk]
    close = panels_["close"]
    end = len(close) - 1
    start = end - N
    ok = [t for t in pool_tickers if t in close.columns and pd.notna(close[t].iloc[start]) and pd.notna(close[t].iloc[end])
          and close[t].iloc[start - 252:start].notna().sum() >= 200]
    pop = panels_["turnover"].iloc[start - 60:start][ok].mean().sort_values(ascending=False)
    field = list(pop.index[:FIELD])

    # ability: percentile of each key factor within the pool at the race start, sign-adjusted
    pct = {}
    for f in kf:
        s = fac[f["id"]].iloc[start][ok].dropna()
        r = s.rank(pct=True) * 100
        pct[f["id"]] = r if f["sign"] > 0 else 100 - r
    w = np.array([max(f['sign'] * f[dk]['ic'], 0.002) for f in kf])
    w = w / w.sum()

    cum = close.iloc[start:end + 1][field]
    cum = cum / cum.iloc[0] - 1
    final_rank = ranks_of(cum.iloc[-1])

    past = [window_ranks(close, field, start - (k + 1) * N, start - k * N) for k in range(N_FORM)]
    horses = []
    for no, t in enumerate(field, 1):
        form = [int(p[1][t]) if p and pd.notna(p[1][t]) else None for p in past]
        early = [p[0][t] for p in past[:3] if p and pd.notna(p[0][t])]
        early_avg = float(np.mean(early)) if early else 5.5
        style, style_desc = style_label(early_avg)
        ability = {f["id"]: round(float(pct[f["id"]].get(t, 50)), 1) for f in kf}
        total = round(float(sum(wi * ability[f["id"]] for wi, f in zip(w, kf))), 1)
        raw = {f["id"]: (None if pd.isna(fac[f["id"]].iloc[start][t]) else round(float(fac[f["id"]].iloc[start][t]), 4)) for f in kf}
        h = {"no": no, "ticker": t, "sector": SECTOR_ZH.get(sector_of.get(t, ""), "—"),
             "form": form, "style": style, "style_desc": style_desc, "early_rank": round(early_avg, 1),
             "ability": ability, "raw": raw, "total": total, "stars": stars(total),
             "price_start": round(float(close[t].iloc[start]), 2), "price_end": round(float(close[t].iloc[end]), 2),
             "ret": round(float(cum[t].iloc[-1]), 4), "result": int(final_rank[t]),
             "path": [round(float(x), 4) for x in cum[t].values]}
        h["comment"] = comment(h, kf)
        horses.append(h)
    order = sorted(horses, key=lambda h: -h["total"])
    marks = ["◎", "○", "▲", "△", "△"]
    for i, h in enumerate(order):
        h["pred"] = i + 1
        h["mark"] = marks[i] if i < len(marks) else ""
    return {"start": str(close.index[start].date()), "end": str(close.index[end].date()), "days": N,
            "dates": [str(d.date()) for d in close.index[start:end + 1]], "horses": horses}


def main() -> None:
    kf_doc = json.loads(KEY_FACTORS.read_text())
    kf = kf_doc["key_factors"]
    df = dl.load_long("2009-01-01")
    p = dl.panels(df)
    fac = F.compute_all(p, dl.earnings())
    sector_of = dl.sectors(df)
    spx, ndx = set(dl.sp500_members()), set(dl.ndx_members())
    ndx.discard("GOOG"); spx.discard("GOOG")           # same company as GOOGL
    pools = {"spx": sorted(spx), "ndx": sorted(ndx), "all": sorted(spx | ndx)}
    out = {"asof": str(p["close"].index[-1].date()),
           "key_factors": kf, "study": {k: kf_doc[k] for k in ("universe_size", "period", "months")},
           "pools": {pk: {dk: build_race(pt, p, fac, kf, sector_of, dk) for dk in DISTANCES}
                     for pk, pt in pools.items()}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB), asof {out['asof']}")
    for pk in pools:
        h = out["pools"][pk]["3m"]["horses"]
        print(pk, "3m:", [(x["ticker"], x["form"], x["style"], x["total"], x["result"]) for x in h[:3]])


if __name__ == "__main__":
    main()
