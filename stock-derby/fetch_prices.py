"""Fetch daily US closing prices for the racers and merge them into data/prices.csv.

The CSV is wide (date,AAPL,MSFT,...) and is committed to the repo, so GitHub is
the only datastore.  Runs incrementally: only days after the last stored date
are requested (plus a few days of overlap so late corrections are picked up).

    python stock-derby/fetch_prices.py            # incremental
    python stock-derby/fetch_prices.py --days 120 # initial backfill
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CSV_PATH = HERE / "data" / "prices.csv"
TICKERS_PATH = HERE / "tickers.json"
OVERLAP_DAYS = 5


def load_tickers() -> list[str]:
    return [t["ticker"] for t in json.loads(TICKERS_PATH.read_text())]


def merge_prices(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Combine two wide frames indexed by date; new values win, rows sorted, empty rows dropped."""
    combined = new.combine_first(old) if not old.empty else new.copy()
    combined = combined.sort_index().dropna(how="all")
    combined.index.name = "date"
    return combined


def read_csv(path: Path = CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col="date")


def download(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="column")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index).strftime("%Y-%m-%d")
    close.index.name = "date"
    return close[[t for t in tickers if t in close.columns]].round(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="force backfill of this many calendar days")
    args = ap.parse_args()

    tickers = load_tickers()
    old = read_csv()
    if args.days or old.empty:
        start = (pd.Timestamp.today() - pd.Timedelta(days=args.days or 120)).strftime("%Y-%m-%d")
    else:
        start = (pd.Timestamp(old.index.max()) - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")

    new = download(tickers, start)
    if new.empty:
        raise SystemExit("no data returned from the price source; leaving CSV untouched")
    merged = merge_prices(old, new)
    merged = merged.reindex(columns=tickers)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(CSV_PATH)
    print(f"{len(merged)} trading days, {merged.index.min()} .. {merged.index.max()}")


if __name__ == "__main__":
    main()
