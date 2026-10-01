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
COLS = ["date", "stock_id", "high", "low", "close", "volume"]


def load_long(since: str = "2009-01-01") -> pd.DataFrame:
    parts = [pd.read_parquet(REPO / "data" / f, columns=COLS + ["industry"]) for f in SNAPSHOTS
             if (REPO / "data" / f).exists()]
    if EXTRA_CSV.exists():
        ex = pd.read_csv(EXTRA_CSV, parse_dates=["date"])
        ex["industry"] = ""
        parts.append(ex[COLS + ["industry"]])
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= since]
    return df.drop_duplicates(["date", "stock_id"], keep="last")


def panels(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {c: df.pivot(index="date", columns="stock_id", values=c).sort_index()
           for c in ["high", "low", "close", "volume"]}
    out["turnover"] = out["close"] * out["volume"]
    return out


def sectors(df: pd.DataFrame) -> dict[str, str]:
    s = df[df["industry"] != ""].sort_values("date").drop_duplicates("stock_id", keep="last")
    return dict(zip(s["stock_id"], s["industry"]))


def sp500_members() -> list[str]:
    m = pd.read_parquet(REPO / "data" / "us_index_membership_snapshot.parquet")
    return sorted(m.loc[m["end_date"].isna(), "stock_id"])


def ndx_members() -> list[str]:
    cache = HERE / "data" / "ndx_members.json"
    if cache.exists():
        return json.loads(cache.read_text())
    return json.loads((HERE / "universe.json").read_text())["ndx"]


def earnings() -> pd.DataFrame:
    p = REPO / "data" / "us_earnings_snapshot.parquet"
    e = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date", "stock_id", "eps_actual", "surprise_pct"])
    e["date"] = pd.to_datetime(e["date"])
    return e
