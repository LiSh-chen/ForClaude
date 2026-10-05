"""Daily top-up for what the repo's S&P 500 snapshot does not cover.

1. Refresh the Nasdaq-100 member list from Wikipedia's "List of NASDAQ-100 companies" (then slickcharts / nasdaq.com,
   then the last cached list, then universe.json). Runs on every daily build, so the list follows index changes.
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


def parse_ndx_tables(html: str, header_hints=("ticker", "symbol")) -> list[str]:
    """Find the constituents table in any page: a table with >= 90 rows and a Ticker/Symbol-like column.
    Raises with a summary of every table seen, so a failure in CI is diagnosable from ndx_status.json."""
    seen = []
    for t in pd.read_html(io.StringIO(html)):
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        seen.append(f"{t.shape[0]}x{t.shape[1]}{cols[:3]}")
        idx = next((i for i, c in enumerate(cols) if any(h in c.lower() for h in header_hints)), None)
        if idx is not None and len(t) >= 90:
            vals = t.iloc[:, idx].astype(str).str.strip().str.replace(".", "-", regex=False)
            return [v for v in vals if v.replace("-", "").isalpha() and v.isupper()]
    raise ValueError(f"no constituents table; saw {len(seen)} tables: " + "; ".join(seen[:8]))


def _get(url: str):
    import requests

    r = requests.get(url, headers={**UA, "Accept": "text/html,application/json"}, timeout=20)
    r.raise_for_status()
    return r


def _from_wikipedia() -> list[str]:
    return parse_ndx_tables(_get("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies").text)


def _from_slickcharts() -> list[str]:
    return parse_ndx_tables(_get("https://www.slickcharts.com/nasdaq100").text)


def _from_nasdaq_api() -> list[str]:
    r = _get("https://api.nasdaq.com/api/quote/list-type/nasdaq100")
    return [row["symbol"].strip().replace(".", "-") for row in r.json()["data"]["data"]["rows"]]


def fetch_ndx_list() -> tuple[list[str], dict]:
    """Try each source; accept only a plausible size (95-110). Returns (tickers, status)."""
    import datetime as dt

    errors = []
    for name, fn in (("wikipedia", _from_wikipedia), ("slickcharts", _from_slickcharts), ("nasdaq.com", _from_nasdaq_api)):
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
                                    "open": d["Open"].values, "high": d["High"].values, "low": d["Low"].values,
                                    "close": d["Close"].values, "volume": d["Volume"].values}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_recent(tickers: list[str], days: int = 14) -> None:
    """Refresh data/recent_prices.csv: the last `days` calendar days of every ticker in one batch download.
    load_long() splices these onto the stored history, so a stale or half-finished upstream snapshot cannot hold the app back."""
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        new = download(sorted(tickers), start)
    except Exception as e:  # noqa: BLE001
        print(f"recent top-up failed: {e}", file=sys.stderr)
        return
    if new.empty:
        print("recent top-up returned nothing; keeping the previous file", file=sys.stderr)
        return
    new = new.dropna(subset=["close"])
    dl.RECENT_CSV.parent.mkdir(exist_ok=True)
    new.round(4).to_csv(dl.RECENT_CSV, index=False)
    print(f"recent: {new['stock_id'].nunique()} tickers, {new['date'].min()} .. {new['date'].max()}")


def download_earnings(tickers: list[str]) -> pd.DataFrame:
    """Reported quarters (EPS estimate / actual / surprise %) from Yahoo for tickers the stored snapshot has no earnings for."""
    import yfinance as yf

    rows = []
    for t in tickers:
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=16)
        except Exception as e:  # noqa: BLE001
            print(f"earnings {t}: {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["Reported EPS"])
        for idx, r in df.iterrows():
            rows.append({"date": pd.Timestamp(idx).tz_localize(None).strftime("%Y-%m-%d") if pd.Timestamp(idx).tzinfo else pd.Timestamp(idx).strftime("%Y-%m-%d"),
                         "stock_id": t, "eps_estimate": r.get("EPS Estimate"), "eps_actual": r.get("Reported EPS"),
                         "surprise_pct": r.get("Surprise(%)")})
    return pd.DataFrame(rows, columns=["date", "stock_id", "eps_estimate", "eps_actual", "surprise_pct"])


def fetch_missing_earnings(tickers: list[str]) -> None:
    """Fill the earnings the S&P snapshot does not have (Nasdaq-only names, new index members) into data/extra_earnings.csv."""
    snap = dl.REPO / "data" / "us_earnings_snapshot.parquet"
    have = set(pd.read_parquet(snap, columns=["stock_id"])["stock_id"]) if snap.exists() else set()
    missing = sorted(set(tickers) - have)
    if not missing:
        return
    new = download_earnings(missing)
    if new.empty:
        print(f"earnings: nothing returned for {len(missing)} tickers", file=sys.stderr)
        return
    old = pd.read_csv(dl.EXTRA_EARNINGS) if dl.EXTRA_EARNINGS.exists() else pd.DataFrame()
    out = dl.merge_earnings(old, new) if len(old) else new
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(dl.EXTRA_EARNINGS, index=False)
    print(f"earnings: {out['stock_id'].nunique()} extra tickers, {len(out)} rows (asked for {len(missing)})")


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

    try:
        spx = dl.sp500_members()
    except Exception:  # noqa: BLE001
        spx = []
    fetch_recent(sorted(set(spx) | set(ndx)))
    try:
        fetch_missing_earnings(sorted(set(spx) | set(ndx)))
    except Exception as e:  # noqa: BLE001
        print(f"earnings top-up failed: {e}", file=sys.stderr)
    snap = dl.load_long(str(pd.Timestamp.today().date() - pd.Timedelta(days=15)))
    have = set(snap.loc[snap["stock_id"].isin(ndx) & snap["industry"].ne(""), "stock_id"])
    missing = sorted(set(ndx) - have)
    print(f"{len(missing)} Nasdaq-100 tickers not in the S&P snapshot: {missing}")
    if not missing:
        return

    old = pd.read_csv(dl.EXTRA_CSV) if dl.EXTRA_CSV.exists() else pd.DataFrame()
    if len(old) and ("open" not in old.columns or old["open"].isna().mean() > 0.5):
        old = pd.DataFrame()                  # history saved before opens were kept: download it again in full
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
