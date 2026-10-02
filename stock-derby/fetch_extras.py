"""Daily top-up for what the repo's S&P 500 snapshot does not cover.

1. Refresh the Nasdaq-100 member list (Wikipedia; falls back to universe.json).
2. For Nasdaq-100 tickers missing from the snapshot, download OHLCV with yfinance
   into data/extra_prices.csv (long format, incremental, committed to the repo).

Never fails the pipeline: with no network the derby is simply built from the snapshot.
"""
from __future__ import annotations

import io
import json
import sys

import pandas as pd

import data_loader as dl

OVERLAP_DAYS = 7
HISTORY_DAYS = 365 * 4 + 30


def merge_long(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union of two long frames; on (date, stock_id) the new row wins."""
    both = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
    both["date"] = pd.to_datetime(both["date"])
    return both.drop_duplicates(["date", "stock_id"], keep="last").sort_values(["stock_id", "date"])


UA = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


def _from_wikipedia() -> list[str]:
    import requests

    html = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=UA, timeout=30)
    html.raise_for_status()
    for t in pd.read_html(io.StringIO(html.text)):
        col = next((c for c in t.columns if str(c).lower() in ("ticker", "symbol")), None)
        if col is not None and len(t) > 80:
            return [x.strip().replace(".", "-") for x in t[col].astype(str)]
    raise ValueError("Nasdaq-100 table not found on Wikipedia")


def _from_nasdaq_api() -> list[str]:
    import requests

    r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                     headers={**UA, "Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    return [row["symbol"].strip().replace(".", "-") for row in r.json()["data"]["data"]["rows"]]


def fetch_ndx_list() -> tuple[list[str], dict]:
    """Try each source; accept only a plausible size (95-110). Returns (tickers, status)."""
    import datetime as dt

    errors = []
    for name, fn in (("wikipedia", _from_wikipedia), ("nasdaq.com", _from_nasdaq_api)):
        try:
            lst = sorted(set(fn()))
            if not 95 <= len(lst) <= 110:
                raise ValueError(f"implausible size {len(lst)}")
            return lst, {"source": name, "fetched_at": dt.datetime.utcnow().strftime("%Y-%m-%d"), "count": len(lst), "errors": errors}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    raise RuntimeError("; ".join(errors))


def download(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    frames = []
    for t in tickers:
        try:
            d = raw[t] if len(tickers) > 1 else raw
            d = d.dropna(subset=["Close"])
        except KeyError:
            continue
        if d.empty:
            continue
        frames.append(pd.DataFrame({"date": pd.to_datetime(d.index).strftime("%Y-%m-%d"), "stock_id": t,
                                    "high": d["High"].values, "low": d["Low"].values,
                                    "close": d["Close"].values, "volume": d["Volume"].values}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    try:
        ndx, status = fetch_ndx_list()
        dl.NDX_CACHE.write_text(json.dumps(ndx))
        print(f"Nasdaq-100 list refreshed from {status['source']}: {len(ndx)} tickers")
    except Exception as e:  # noqa: BLE001
        ndx = dl.ndx_members()
        old = dl.ndx_status()
        status = {"source": "cache" if dl.NDX_CACHE.exists() else "fallback", "fetched_at": old.get("fetched_at") if dl.NDX_CACHE.exists() else None,
                  "count": len(ndx), "errors": [str(e)]}
        print(f"Nasdaq-100 list fetch FAILED ({e}); using {status['source']} list of {len(ndx)}", file=sys.stderr)
    dl.NDX_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=1))

    snap = dl.load_long(str(pd.Timestamp.today().date() - pd.Timedelta(days=15)))
    have = set(snap.loc[snap["stock_id"].isin(ndx) & snap["industry"].ne(""), "stock_id"])
    missing = sorted(set(ndx) - have)
    print(f"{len(missing)} Nasdaq-100 tickers not in the S&P snapshot: {missing}")
    if not missing:
        return

    old = pd.read_csv(dl.EXTRA_CSV) if dl.EXTRA_CSV.exists() else pd.DataFrame()
    if len(old):
        start = (pd.to_datetime(old["date"]).max() - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
        new_tickers = [t for t in missing if t not in set(old["stock_id"])]
    else:
        start, new_tickers = None, missing
    start = start or (pd.Timestamp.today() - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    try:
        new = download(missing, start)
        if new_tickers and len(old):          # brand-new tickers need full history
            new = merge_long(new, download(new_tickers, (pd.Timestamp.today() - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")))
    except Exception as e:  # noqa: BLE001
        print(f"extras download failed: {e}", file=sys.stderr)
        return
    if new.empty:
        print("no extra data returned; leaving CSV untouched", file=sys.stderr)
        return
    out = merge_long(old, new)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.round(4).to_csv(dl.EXTRA_CSV, index=False)
    print(f"extras: {out['stock_id'].nunique()} tickers, {len(out)} rows")


if __name__ == "__main__":
    main()
