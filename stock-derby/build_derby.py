"""Build data/derby.json: race card (賽馬報) + race replay for every pool x distance.

pool     : spx (S&P 500) | ndx (Nasdaq-100) | all (union)
distance : 1m / 3m / 6m  = calendar months (start = last trading day on/before end - N months)
race     : the most recent completed window, ending on the latest trading day with price coverage
field    : the 10 stocks of the pool with the highest return over that window (起跑價 -> 終點價, adjusted closes)
card     : 近績 / 跑法 / 能力 only use data up to the start date (so the ability ranking is a genuine
           pre-race "prediction" that can be compared with the result).
validation: the same prediction replayed over past non-overlapping races (in-sample, see caveats in the UI).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader as dl
import factors as F

DIST_MONTHS = {"1m": 1, "3m": 3, "6m": 6}
STRIDE = {"1m": 21, "3m": 63, "6m": 126}       # trading days between validation races (no overlap)
FIELD = 10
N_FORM = 4
N_VALID = 40
KEY_FACTORS = Path(__file__).parent / "data" / "key_factors.json"
OUT = Path(__file__).parent / "data" / "derby.json"
MANIFEST = Path(__file__).parent / "assets" / "horses" / "manifest.json"
ADJ = [("疾風", "Swift"), ("烈焰", "Blaze"), ("飛雲", "Skycloud"), ("銀箭", "Silver Arrow"), ("金蹄", "Gold Hoof"),
       ("雷霆", "Thunder"), ("追月", "Moonchaser"), ("踏雪", "Snowstep"), ("流星", "Comet"), ("驚鴻", "Swan Dive"),
       ("破浪", "Wavebreaker"), ("鐵衛", "Ironclad"), ("赤兔", "Red Hare"), ("青驄", "Bluemane"), ("白虹", "White Rainbow"),
       ("紫電", "Violet Bolt"), ("玄影", "Shadowrun"), ("烈陽", "Sunfire"), ("凌霄", "Skyward"), ("奔雷", "Rumble")]
SECTOR_ZH = {"Information Technology": "科技", "Communication Services": "通訊", "Consumer Discretionary": "非必需消費",
             "Consumer Staples": "必需消費", "Financials": "金融", "Health Care": "醫療", "Industrials": "工業",
             "Energy": "能源", "Utilities": "公用事業", "Real Estate": "房地產", "Materials": "原物料"}
MARKS = ["◎", "○", "▲", "△", "△"]


def months_back(dates: pd.DatetimeIndex, i: int, m: int) -> int:
    """Index of the last trading day on/before dates[i] - m calendar months (-1 if before the data starts)."""
    return int(dates.searchsorted(dates[i] - pd.DateOffset(months=m), side="right")) - 1


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
    """(rank at the 1/3 point, rank at the finish) among the field for window [lo, hi]; None if no history."""
    if lo < 0 or hi <= lo:
        return None
    w = close.iloc[lo:hi + 1][field]
    cum = w / w.iloc[0] - 1
    third = max(1, (hi - lo) // 3)
    return ranks_of(cum.iloc[third]), ranks_of(cum.iloc[-1])


def stars(score: float) -> int:
    return int(np.clip(1 + score // 20, 1, 5))


def weights_for(kf: list[dict], dk: str) -> np.ndarray:
    w = np.array([max(f[dk]["ic"], 0.002) for f in kf])      # oriented ICs; floor keeps every factor in play
    return w / w.sum()


def ability_pct(ofac: dict, kf: list[dict], start: int, eligible: list[str]) -> pd.DataFrame:
    """Percentile (0-100, higher = better) of each key factor among all eligible stocks at the start date."""
    return pd.DataFrame({f["id"]: ofac[f["id"]].iloc[start][eligible].rank(pct=True) * 100 for f in kf})


def total_score(pct: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """IC-weighted mean of the percentiles; missing factors are dropped and the weights renormalised."""
    vals = pct.to_numpy()
    ww = np.where(np.isnan(vals), 0.0, w)
    return pd.Series((np.nan_to_num(vals) * ww).sum(1) / np.where(ww.sum(1) == 0, np.nan, ww.sum(1)), index=pct.index)


def select_field(close: pd.DataFrame, tickers: list[str], s: int, e: int) -> tuple[list[str], pd.Series, list[str]]:
    ok = [t for t in tickers if t in close.columns and pd.notna(close[t].iloc[s]) and pd.notna(close[t].iloc[e])]
    ret = (close[ok].iloc[e] / close[ok].iloc[s] - 1).sort_values(ascending=False)
    return list(ret.index[:FIELD]), ret, ok


def assign_looks(field: list[str], manifest: dict) -> dict[str, dict]:
    """Horse name + sprite sheet per ticker. Listed companies get a logo-inspired design; the rest a stable
    generic coat (by ticker hash, never two identical coats in one field) and an adjective+ticker name."""
    generic = list(manifest["generic"])
    used: set[str] = set()
    looks = {}
    for t in field:
        b = manifest["brands"].get(t)
        if b:
            looks[t] = {"company": b["company"], "horse_zh": b["zh"], "horse_en": b["en"], "sprite": t,
                        "coat": b["features"], "inspired_by": b["inspired_by"], "silks": b["silks"]}
            continue
        h = sum(ord(ch) * (i + 1) for i, ch in enumerate(t))
        k = h % len(generic)
        while generic[k] in used:
            k = (k + 1) % len(generic)
        used.add(generic[k])
        zh, en = ADJ[h % len(ADJ)]
        g = manifest["generic"][generic[k]]
        looks[t] = {"company": "", "horse_zh": f"{zh}{t}", "horse_en": f"{en} {t}", "sprite": f"generic_{generic[k]}",
                    "coat": f"{g['zh']}・{g['features']}", "inspired_by": "", "silks": None}
    return looks


def comment(h: dict, kf: list[dict]) -> str:
    labels = {f["id"]: f["label"] for f in kf}
    have = {k: v for k, v in h["ability"].items() if v is not None}
    parts = [f"{h['horse_zh']}（{h['company']}）" if h["company"] else h["horse_zh"]]
    done = [x for x in h["form"] if x is not None]
    if done:
        top3 = sum(1 for x in done if x <= 3)
        parts.append(f"近{len(done)}戰{top3}次進前三" if top3 else f"近{len(done)}戰未進前三")
    parts.append(f"跑法{h['style']}")
    if have:
        ab = sorted(have.items(), key=lambda kv: -kv[1])
        parts.append(f"強項「{labels[ab[0][0]]}」")
        if ab[-1][1] < 35:
            parts.append(f"弱點「{labels[ab[-1][0]]}」")
    if len(have) < len(kf):
        parts.append("上市未滿一年，部分能力值無資料")
    return "，".join(parts)


def build_race(tickers, p, ofac, kf, sector_of, dk, manifest) -> dict:
    close = p["close"]
    dates = close.index
    m, N = DIST_MONTHS[dk], STRIDE[dk]
    cov = close.reindex(columns=[t for t in tickers if t in close.columns]).notna().mean(axis=1)
    e = int(dates.get_loc(cov[cov >= 0.9].index[-1]))
    s = months_back(dates, e, m)
    field_unsorted, ret, ok = select_field(close, tickers, s, e)
    pct = ability_pct(ofac, kf, s, ok)
    score = total_score(pct, weights_for(kf, dk))
    pred_order = list(score.loc[field_unsorted].sort_values(ascending=False).index)
    final_rank = ranks_of(ret.loc[field_unsorted])

    # chain of earlier same-length windows for 近績 / 跑法
    bounds = [s]
    for _ in range(N_FORM):
        bounds.append(months_back(dates, bounds[-1], m) if bounds[-1] > 0 else -1)
    past = [window_ranks(close, pred_order, bounds[k + 1], bounds[k]) if bounds[k + 1] >= 0 else None for k in range(N_FORM)]
    cum = close.iloc[s:e + 1][pred_order]
    cum = cum / cum.iloc[0] - 1

    looks = assign_looks(pred_order, manifest)
    horses = []
    for no, t in enumerate(pred_order, 1):
        form = [int(q[1][t]) if q and pd.notna(q[1][t]) else None for q in past]
        early = [q[0][t] for q in past[:3] if q and pd.notna(q[0][t])]
        early_avg = float(np.mean(early)) if early else 5.5
        style, style_desc = style_label(early_avg)
        ability = {f["id"]: (None if pd.isna(pct.loc[t, f["id"]]) else round(float(pct.loc[t, f["id"]]), 1)) for f in kf}
        raw = {f["id"]: (None if pd.isna(ofac[f["id"]].iloc[s][t]) else round(float(ofac[f["id"]].iloc[s][t]), 4)) for f in kf}
        total = round(float(score[t]), 1)
        h = {"no": no, "ticker": t, "sector": SECTOR_ZH.get(sector_of.get(t, ""), "—"), **looks[t],
             "form": form, "style": style, "style_desc": style_desc, "early_rank": round(early_avg, 1) if early else None,
             "ability": ability, "raw": raw, "total": total, "stars": stars(total), "pred": no, "mark": MARKS[no - 1] if no <= 5 else "",
             "price_start": round(float(close[t].iloc[s]), 2), "price_end": round(float(close[t].iloc[e]), 2),
             "ret": round(float(ret[t]), 4), "result": int(final_rank[t]),
             "path": [round(float(x), 4) for x in cum[t].values]}
        h["comment"] = comment(h, kf)
        horses.append(h)
    return {"start": str(dates[s].date()), "end": str(dates[e].date()), "months": m, "days": e - s,
            "dates": [str(d.date()) for d in dates[s:e + 1]], "horses": horses,
            "weights": [{"id": f["id"], "label": f["label"], "w": round(float(w), 3)} for f, w in zip(kf, weights_for(kf, dk))],
            "coverage": {"pool_size": len(tickers), "with_prices": len(ok),
                         "missing": sorted(t for t in tickers if t not in ok)[:40]},
            "validation": validate(tickers, p, ofac, kf, dk, e)}


def validate(tickers, p, ofac, kf, dk, live_end: int) -> dict:
    """Replay the same method over earlier non-overlapping races ending before the live one."""
    close = p["close"]
    dates = close.index
    m, N = DIST_MONTHS[dk], STRIDE[dk]
    w = weights_for(kf, dk)
    rhos, top1, top3 = [], [], []
    for k in range(1, N_VALID + 1):
        e = live_end - k * N
        s = months_back(dates, e, m) if e > 0 else -1
        if s < 300:                                   # need >1y of history for the factors
            break
        field, ret, ok = select_field(close, tickers, s, e)
        if len(field) < FIELD:
            continue
        score = total_score(ability_pct(ofac, kf, s, ok), w).loc[field].dropna()
        if len(score) < FIELD:
            continue
        actual = ranks_of(ret.loc[score.index])
        pred = ranks_of(score)
        rhos.append(float(np.corrcoef(pred, actual)[0, 1]))
        order = pred.sort_values()
        top1.append(float(actual[order.index[0]]))
        top3.append(float(actual[order.index[:3]].mean()))
    n = len(rhos)
    if n < 3:
        return {"n": n}
    r = np.array(rhos)
    return {"n": n, "mean_rho": round(float(r.mean()), 3), "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(n))), 2),
            "top1_avg_rank": round(float(np.mean(top1)), 2), "top3_avg_rank": round(float(np.mean(top3)), 2),
            "first": str(dates[months_back(dates, live_end - n * N, m)].date()), "last": str(dates[live_end - N].date())}


def main() -> None:
    kf_doc = json.loads(KEY_FACTORS.read_text())
    kf = kf_doc["key_factors"]
    df = dl.load_long("2009-01-01")
    p = dl.panels(df)
    fac = F.compute_all(p, dl.earnings())
    ofac = {f["id"]: (-fac[f["id"]] if f["reversed"] else fac[f["id"]]) for f in kf}     # higher = better, always
    sector_of = dl.sectors(df)
    manifest = json.loads(MANIFEST.read_text())
    spx, ndx = set(dl.sp500_members()), set(dl.ndx_members())
    spx.discard("GOOG"); ndx.discard("GOOG")           # same company as GOOGL
    pools = {"spx": sorted(spx), "ndx": sorted(ndx), "all": sorted(spx | ndx)}
    pools_out = {pk: {dk: build_race(pt, p, ofac, kf, sector_of, dk, manifest) for dk in DIST_MONTHS} for pk, pt in pools.items()}
    out = {"asof": pools_out["all"]["1m"]["end"], "key_factors": kf,
           "study": {k: kf_doc[k] for k in ("universe_size", "period", "months")},
           "ndx_status": dl.ndx_status(),
           "pools": pools_out}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB), asof {out['asof']}, NDX list: {out['ndx_status']['source']}")
    for pk in pools:
        for dk in DIST_MONTHS:
            r = out["pools"][pk][dk]
            print(pk, dk, r["start"], "->", r["end"], [(h["ticker"], f"{h['ret']:+.1%}") for h in sorted(r["horses"], key=lambda h: h["result"])[:10]],
                  r["validation"])


if __name__ == "__main__":
    main()
