"""Load wide price/volume panels (dates x tickers) for the derby.

S&P 500 history comes from the repo's committed parquet snapshots (kept fresh by
the existing daily US ingest). Nasdaq-100 names that are not in that snapshot are
topped up by fetch_extras.py into data/extra_prices.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent
SNAPSHOTS = ["us_prices_snapshot.parquet", "us_prices_snapshot.part2.parquet"]
EXTRA_CSV = HERE / "data" / "extra_prices.csv"
RECENT_CSV = HERE / "data" / "recent_prices.csv"        # last ~2 weeks for every ticker, refreshed by fetch_extras.py
COLS = ["date", "stock_id", "open", "high", "low", "close", "volume"]


def load_long(since: str = "2009-01-01") -> pd.DataFrame:
    parts = [pd.read_parquet(REPO / "data" / f, columns=COLS + ["industry"]) for f in SNAPSHOTS
             if (REPO / "data" / f).exists()]
    if EXTRA_CSV.exists():
        ex = pd.read_csv(EXTRA_CSV, parse_dates=["date"])
        if "open" not in ex.columns:
            ex["open"] = float("nan")             # saved before opens were kept; the next fetch_extras run fills them
        ex["industry"] = ""
        parts.append(ex[COLS + ["industry"]])
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= since].dropna(subset=["close"])           # a half-finished session (NaN close) is not a row
    df = df.drop_duplicates(["date", "stock_id"], keep="last")
    if RECENT_CSV.exists():
        rec = pd.read_csv(RECENT_CSV, parse_dates=["date"]).dropna(subset=["close"])
        if "open" not in rec.columns:
            rec["open"] = float("nan")
        df = splice_recent(df, rec)
    return df


def splice_recent(base: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """Append sessions newer than `base` from `recent`, rescaled so the two series agree on their last common day.
    (The stored history was dividend/split-adjusted at an earlier download, the recent one at today's.)"""
    last = base.groupby("stock_id")["date"].max().rename("last_base")
    m = base[["stock_id", "date", "close"]].merge(recent[["stock_id", "date", "close"]], on=["stock_id", "date"], suffixes=("_b", "_r"))
    if m.empty:
        return base
    m = m.sort_values("date").groupby("stock_id").tail(1).set_index("stock_id")
    ratio = (m["close_b"] / m["close_r"]).rename("ratio")
    new = recent.join(last, on="stock_id").join(ratio, on="stock_id")
    new = new[new["last_base"].notna() & new["ratio"].notna() & (new["date"] > new["last_base"])].copy()
    for c in ("open", "high", "low", "close"):
        new[c] = new[c] * new["ratio"]
    new["industry"] = new["stock_id"].map(base.sort_values("date").drop_duplicates("stock_id", keep="last").set_index("stock_id")["industry"]).fillna("")
    return pd.concat([base, new[base.columns]], ignore_index=True)


def panels(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {c: df.pivot(index="date", columns="stock_id", values=c).sort_index()
           for c in ["open", "high", "low", "close", "volume"]}
    out["turnover"] = out["close"] * out["volume"]
    return out


def sectors(df: pd.DataFrame) -> dict[str, str]:
    s = df[df["industry"] != ""].sort_values("date").drop_duplicates("stock_id", keep="last")
    return dict(zip(s["stock_id"], s["industry"]))


def sp500_members() -> list[str]:
    m = pd.read_parquet(REPO / "data" / "us_index_membership_snapshot.parquet")
    return sorted(m.loc[m["end_date"].isna(), "stock_id"])


NDX_CACHE = HERE / "data" / "ndx_members.json"
NDX_STATUS = HERE / "data" / "ndx_status.json"


def ndx_members() -> list[str]:
    """Latest known Nasdaq-100 list: fresh/cached download if there is one, else the built-in fallback."""
    if NDX_CACHE.exists():
        return json.loads(NDX_CACHE.read_text())
    return json.loads((HERE / "universe.json").read_text())["ndx"]


def ndx_status() -> dict:
    """Where the Nasdaq-100 list came from (shown on the page so a stale list is never passed off as current)."""
    if NDX_STATUS.exists():
        return json.loads(NDX_STATUS.read_text())
    return {"source": "fallback", "fetched_at": None, "errors": ["never downloaded"]}


EXTRA_EARNINGS = HERE / "data" / "extra_earnings.csv"      # fetched by fetch_extras.py for tickers the snapshot does not cover


def merge_earnings(snap: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([snap, extra], ignore_index=True) if len(extra) else snap.copy()
    both["date"] = pd.to_datetime(both["date"])
    return both.drop_duplicates(["stock_id", "date"], keep="last")


def earnings() -> pd.DataFrame:
    p = REPO / "data" / "us_earnings_snapshot.parquet"
    e = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "stock_id", "eps_estimate", "eps_actual", "surprise_pct"])
    extra = pd.read_csv(EXTRA_EARNINGS) if EXTRA_EARNINGS.exists() else pd.DataFrame()
    return merge_earnings(e, extra)
