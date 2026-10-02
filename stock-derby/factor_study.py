"""Empirically pick the five factors that matter most for forward returns.

For every month-end since 2012, rank-IC (Spearman) of each candidate factor vs the
forward 1/3/6-month return across the current S&P 500 + Nasdaq-100 universe.
Selection: |mean IC across the three horizons|, greedily skipping a factor whose
average cross-sectional rank correlation with an already-chosen one exceeds 0.7.
Output: data/key_factors.json (+ full IC table).  Run: python stock-derby/factor_study.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader as dl
import factors as F

HORIZONS = {"1m": 21, "3m": 63, "6m": 126}
OUT = Path(__file__).parent / "data" / "key_factors.json"
CORR_CAP = 0.7


def month_ends(index: pd.DatetimeIndex, start="2012-01-01") -> list[pd.Timestamp]:
    s = pd.Series(index, index=index)
    return list(s[s.index >= start].groupby([s.index.year[s.index >= start], s.index.month[s.index >= start]]).last())


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, dates) -> pd.Series:
    vals = {}
    for d in dates:
        a, b = factor.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= 100:
            vals[d] = a[m].rank().corr(b[m].rank())
    return pd.Series(vals)


def main() -> None:
    df = dl.load_long("2010-06-01")
    p = dl.panels(df)
    tickers = sorted(set(dl.sp500_members()) | set(dl.ndx_members()))
    p = {k: v.reindex(columns=[t for t in tickers if t in v.columns]) for k, v in p.items()}
    fac = F.compute_all(p, dl.earnings())
    close = p["close"]
    dates = [d for d in month_ends(close.index) if close.index.get_loc(d) + 126 < len(close)]
    fwd = {k: close.shift(-n) / close - 1 for k, n in HORIZONS.items()}

    table = {}
    for name, fr in fac.items():
        row = {}
        for k, n in HORIZONS.items():
            ic = rank_ic(fr, fwd[k], dates)
            step = max(1, n // 21)                       # de-overlap forward windows for the t-stat
            sub = ic.iloc[::step]
            row[k] = {"ic": round(float(ic.mean()), 4),
                      "t": round(float(sub.mean() / (sub.std() / np.sqrt(len(sub)))), 2),
                      "hit": round(float((np.sign(ic) == np.sign(ic.mean())).mean()), 2)}
        row["avg_ic"] = round(float(np.mean([row[k]["ic"] for k in HORIZONS])), 4)
        table[name] = row

    ranked = sorted(table, key=lambda k: -abs(table[k]["avg_ic"]))
    sample = dates[::3]
    chosen: list[str] = []
    skipped: dict[str, str] = {}
    for name in ranked:
        dup = None
        for other in chosen:
            cs = [fac[name].loc[d].rank().corr(fac[other].loc[d].rank()) for d in sample]
            if abs(np.nanmean(cs)) > CORR_CAP:
                dup = other
                break
        if dup:
            skipped[name] = f"與 {dup} 高度相關，略過"
        else:
            chosen.append(name)
        if len(chosen) == 5:
            break

    result = {
        "universe_size": len(tickers),
        "months": len(dates),
        "period": [str(dates[0].date()), str(dates[-1].date())],
        "key_factors": [{"id": k, "label": F.META[k][0], "group": F.META[k][1], "fmt": F.META[k][2],
                         "sign": 1 if table[k]["avg_ic"] >= 0 else -1, **table[k]} for k in chosen],
        "skipped": skipped,
        "all": {k: {"label": F.META[k][0], "group": F.META[k][1], **table[k]} for k in ranked},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for k in ranked:
        t = table[k]
        print(f"{'*' if k in chosen else ' '} {k:13s} {F.META[k][1]} avgIC={t['avg_ic']:+.4f}  "
              + "  ".join(f"{h}:{t[h]['ic']:+.3f}(t={t[h]['t']:+.1f})" for h in HORIZONS))


if __name__ == "__main__":
    main()
