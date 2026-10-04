"""Build data/panel.json: the compact inputs the browser needs to run the derby by itself.

Nothing about a race is pre-computed.  The page picks the pool / momentum cycle / lookback the user chose,
ranks the latest top 10 from the price panel, prices the odds, replays the preliminary and settles races that the
user started - all in the browser.  The daily build therefore only refreshes:
  dates, px    daily adjusted closes of every pool member (last N_DATES sessions)
  pools        which tickers are in S&P 500 / Nasdaq-100 / both
  ab           the five key-factor abilities (raw value + percentile inside each pool) as of the latest close
  key_factors  factor names, meanings, IC study results, weights per horizon
  looks        horse name / sprite / coat for every ticker
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader as dl
import factors as F

IC_KEY = {"d": "1m", "w": "1m", "w2": "1m", "m": "1m", "q": "3m", "h": "6m"}     # which study horizon weights the abilities
N_DATES = 1300                                   # ~5 years of sessions: 4 history periods + 6x lookback of a half-year cycle
MIN_COVER = 0.9                                  # a session counts only when >=90% of the pool has a close
HERE = Path(__file__).parent
KEY_FACTORS = HERE / "data" / "key_factors.json"
OUT = HERE / "data" / "panel.json"
MANIFEST = HERE / "assets" / "horses" / "manifest.json"
ADJ = [("疾風", "Swift"), ("烈焰", "Blaze"), ("飛雲", "Skycloud"), ("銀箭", "Silver Arrow"), ("金蹄", "Gold Hoof"),
       ("雷霆", "Thunder"), ("追月", "Moonchaser"), ("踏雪", "Snowstep"), ("流星", "Comet"), ("驚鴻", "Swan Dive"),
       ("破浪", "Wavebreaker"), ("鐵衛", "Ironclad"), ("赤兔", "Red Hare"), ("青驄", "Bluemane"), ("白虹", "White Rainbow"),
       ("紫電", "Violet Bolt"), ("玄影", "Shadowrun"), ("烈陽", "Sunfire"), ("凌霄", "Skyward"), ("奔雷", "Rumble")]
SECTOR_ZH = {"Information Technology": "科技", "Communication Services": "通訊", "Consumer Discretionary": "非必需消費",
             "Consumer Staples": "必需消費", "Financials": "金融", "Health Care": "醫療", "Industrials": "工業",
             "Energy": "能源", "Utilities": "公用事業", "Real Estate": "房地產", "Materials": "原物料"}


def weights(kf: list[dict]) -> dict[str, list[float]]:
    """Factor weights per study horizon (oriented ICs, floored so every factor stays in play)."""
    out = {}
    for key in ("1m", "3m", "6m"):
        w = np.array([max(f[key]["ic"], 0.002) for f in kf])
        out[key] = [round(float(x), 4) for x in w / w.sum()]
    return out


def pool_pct(ofac: dict, kf: list[dict], idx: int, members: list[str]) -> pd.DataFrame:
    """Percentile (0-100, higher = better) of each key factor among the pool members at row `idx`."""
    return pd.DataFrame({f["id"]: ofac[f["id"]].iloc[idx].reindex(members).rank(pct=True) * 100 for f in kf})


def looks_for(tickers: list[str], manifest: dict, sector_of: dict) -> dict[str, dict]:
    """Horse identity per ticker.  Listed brands get their own design; everyone else a stable generic coat
    (`g` = preferred index into the generic list, the page steps on if two horses of a field collide)."""
    generic = list(manifest["generic"])
    out = {}
    for t in tickers:
        sec = SECTOR_ZH.get(sector_of.get(t, ""), "—")
        b = manifest["brands"].get(t)
        if b:
            out[t] = {"sec": sec, "co": b["company"], "zh": b["zh"], "en": b["en"], "sp": t, "coat": b["features"],
                      "ins": b["inspired_by"], "silks": b["silks"]}
            continue
        h = sum(ord(ch) * (i + 1) for i, ch in enumerate(t))
        zh, en = ADJ[h % len(ADJ)]
        out[t] = {"sec": sec, "zh": f"{zh}{t}", "en": f"{en} {t}", "g": h % len(generic)}
    return out


def main() -> None:
    kf_doc = json.loads(KEY_FACTORS.read_text())
    kf = [{**f, **F.horse_term(f["id"], f["reversed"])} for f in kf_doc["key_factors"]]
    df = dl.load_long("2009-01-01")
    p = dl.panels(df)
    fac = F.compute_all(p, dl.earnings())
    ofac = {f["id"]: (-fac[f["id"]] if f["reversed"] else fac[f["id"]]) for f in kf}      # higher = better, always
    sector_of = dl.sectors(df)
    manifest = json.loads(MANIFEST.read_text())
    spx, ndx = set(dl.sp500_members()), set(dl.ndx_members())
    spx.discard("GOOG"); ndx.discard("GOOG")                    # same company as GOOGL
    close = p["close"]
    allm = sorted((spx | ndx) & set(close.columns))
    cov = close[allm].notna().mean(axis=1)
    good = cov[cov >= MIN_COVER].index
    last = close.index.get_loc(good[-1])
    close = close.iloc[:last + 1]
    rows = close.index[cov.reindex(close.index) >= MIN_COVER][-N_DATES:]
    px = close.loc[rows, allm]
    idx = close.index.get_loc(rows[-1])

    pools = {"spx": sorted(spx & set(allm)), "ndx": sorted(ndx & set(allm)), "all": allm}
    pct = {k: pool_pct(ofac, kf, idx, m) for k, m in pools.items()}
    ab = {}
    for t in allm:
        raw = [None if pd.isna(ofac[f["id"]].iloc[idx].get(t)) else round(float(ofac[f["id"]].iloc[idx][t]), 4) for f in kf]
        ab[t] = {"raw": raw, "pct": {k: [None if pd.isna(pct[k].loc[t, f["id"]]) else round(float(pct[k].loc[t, f["id"]]), 1) for f in kf]
                                       for k in pools if t in pct[k].index}}
    out = {"asof": str(rows[-1].date()), "dates": [str(d.date()) for d in rows], "tk": allm,
           "px": [[None if pd.isna(v) else round(float(v), 3) for v in px[t].to_numpy()] for t in allm],
           "pools": pools, "ab": ab, "weights": weights(kf), "ic_key": IC_KEY,
           "key_factors": kf, "study": {k: kf_doc[k] for k in ("universe_size", "period", "months")},
           "looks": looks_for(allm, manifest, sector_of),
           "generic": [{"key": k, "zh": g["zh"], "features": g["features"]} for k, g in manifest["generic"].items()],
           "ndx_status": dl.ndx_status()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {len(allm)} tickers x {len(rows)} sessions, asof {out['asof']}, "
          f"NDX list: {out['ndx_status']['source']}")


if __name__ == "__main__":
    main()
