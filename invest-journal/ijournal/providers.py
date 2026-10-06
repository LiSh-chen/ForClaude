"""資料提供者：LiveProvider（yfinance + RSS，在 GitHub Actions 上使用）與 DemoProvider（離線合成資料，僅供驗證流程）。"""
from __future__ import annotations
import datetime as dt
import email.utils
import hashlib
import random
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .utils import isnum, market_of

UA = {"User-Agent": "Mozilla/5.0 (compatible; invest-journal/1.0; +https://github.com/LiSh-chen/ForClaude)"}


# ---------------------------------------------------------------- RSS
def _parse_date(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        d = email.utils.parsedate_to_datetime(s)
    except Exception:
        try:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def parse_feed(xml_bytes: bytes) -> list[dict]:
    """同時支援 RSS 2.0 與 Atom。"""
    root = ET.fromstring(xml_bytes)
    out = []
    strip = lambda tag: tag.split("}")[-1]
    for el in root.iter():
        if strip(el.tag) not in ("item", "entry"):
            continue
        rec = {"title": "", "link": "", "summary": "", "published": None}
        for ch in el:
            tg = strip(ch.tag)
            if tg == "title":
                rec["title"] = (ch.text or "").strip()
            elif tg == "link":
                rec["link"] = (ch.text or ch.attrib.get("href") or "").strip()
            elif tg in ("description", "summary", "content"):
                txt = re.sub(r"<[^>]+>", " ", ch.text or "")
                rec["summary"] = re.sub(r"\s+", " ", txt).strip()[:400]
            elif tg in ("pubDate", "published", "updated", "date"):
                rec["published"] = rec["published"] or _parse_date(ch.text)
        if rec["title"]:
            out.append(rec)
    return out


class LiveProvider:
    name = "live"

    def __init__(self, sleep: float = 0.15):
        self.sleep = sleep
        self.fetch_log: list[dict] = []

    # ---- 價格
    def prices(self, tickers: list[str], days: int = 800) -> dict[str, pd.DataFrame]:
        import yfinance as yf
        out: dict[str, pd.DataFrame] = {}
        period = "3y" if days > 600 else "2y"
        chunks = [tickers[i:i + 40] for i in range(0, len(tickers), 40)]
        for chunk in chunks:
            for attempt in range(3):
                try:
                    df = yf.download(chunk, period=period, interval="1d", auto_adjust=True, group_by="ticker",
                                     progress=False, threads=True)
                    break
                except Exception:
                    time.sleep(2 * (attempt + 1))
                    df = None
            if df is None or df.empty:
                continue
            for t in chunk:
                try:
                    sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                    sub = sub.dropna(subset=["Close"]).copy()
                    if len(sub):
                        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
                        out[t] = sub[["Open", "High", "Low", "Close", "Volume"]]
                except Exception:
                    continue
        return out

    # ---- 基本面
    def fundamentals(self, ticker: str) -> dict:
        import yfinance as yf
        for attempt in range(3):
            try:
                info = yf.Ticker(ticker).info or {}
                if info:
                    time.sleep(self.sleep)
                    return info
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return {}

    # ---- 新聞
    def news(self, sources: dict) -> list[dict]:
        items, seen = [], set()
        for f in sources["feeds"]:
            t0 = time.time()
            try:
                r = requests.get(f["url"], headers=UA, timeout=20)
                r.raise_for_status()
                recs = parse_feed(r.content)
                ok, err = True, ""
            except Exception as e:  # 單一來源失敗不影響整體
                recs, ok, err = [], False, str(e)[:120]
            n_new = 0
            for rec in recs:
                key = re.sub(r"\W+", "", rec["title"].lower())[:80]
                if key in seen:
                    continue
                seen.add(key)
                rec.update(source=f["name"], market=f["market"], weight=f.get("weight", 1.0))
                items.append(rec)
                n_new += 1
            self.fetch_log.append({"source": f["name"], "ok": ok, "n": n_new, "error": err, "sec": round(time.time() - t0, 1)})
        return items


# ---------------------------------------------------------------- Demo（合成資料；絕不可當成真實行情）
class DemoProvider:
    """以代號雜湊做種子的確定性合成資料。目的：在沒有網路的環境驗證整條流程。"""
    name = "demo"

    def __init__(self, end: Optional[dt.date] = None):
        self.end = pd.Timestamp(end or dt.date.today()).normalize()
        self.fetch_log: list[dict] = [{"source": "DEMO 合成新聞", "ok": True, "n": 0, "error": "", "sec": 0}]

    @staticmethod
    def _rng(key: str) -> random.Random:
        return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))

    def prices(self, tickers, days: int = 800):
        idx = pd.bdate_range(end=self.end, periods=days)
        out = {}
        for t in tickers:
            r = self._rng("p" + t)
            drift = r.uniform(-0.0002, 0.0012)
            vol = r.uniform(0.008, 0.03)
            ret = np.array([r.gauss(drift, vol) for _ in range(len(idx))])
            base = r.uniform(20, 800) if market_of(t) == "US" else r.uniform(30, 1500)
            close = base * np.exp(np.cumsum(ret))
            hi = close * (1 + np.abs(np.array([r.gauss(0, vol / 2) for _ in idx])))
            lo = close * (1 - np.abs(np.array([r.gauss(0, vol / 2) for _ in idx])))
            vols = np.array([r.uniform(0.5, 1.5) for _ in idx]) * (2e6 if market_of(t) == "US" else 8e6)
            out[t] = pd.DataFrame({"Open": close, "High": hi, "Low": lo, "Close": close, "Volume": vols}, index=idx)
        return out

    def fundamentals(self, ticker: str) -> dict:
        r = self._rng("f" + ticker)
        us = market_of(ticker) == "US"
        eps = r.uniform(1, 12)
        price_guess = eps * r.uniform(10, 40)
        return {
            "shortName": f"DEMO {ticker}", "currency": "USD" if us else "TWD", "financialCurrency": "USD" if us else "TWD",
            "marketCap": r.uniform(2e10, 3e12) if us else r.uniform(1e11, 2e13),
            "sharesOutstanding": r.uniform(2e8, 5e9), "trailingEps": eps, "forwardEps": eps * r.uniform(1.0, 1.4),
            "forwardPE": r.uniform(10, 40), "trailingPE": r.uniform(12, 50), "pegRatio": r.uniform(0.6, 3.0),
            "returnOnEquity": r.uniform(0.03, 0.40), "returnOnAssets": r.uniform(0.01, 0.2),
            "grossMargins": r.uniform(0.15, 0.7), "operatingMargins": r.uniform(0.03, 0.4), "profitMargins": r.uniform(0.02, 0.3),
            "revenueGrowth": r.uniform(-0.05, 0.5), "earningsGrowth": r.uniform(-0.1, 0.6),
            "debtToEquity": r.uniform(5, 220), "currentRatio": r.uniform(0.8, 3.0),
            "freeCashflow": r.uniform(-1e8, 5e10), "operatingCashflow": r.uniform(1e8, 8e10), "totalRevenue": r.uniform(1e9, 2e11),
            "beta": r.uniform(0.6, 1.9), "targetMeanPrice": price_guess * r.uniform(0.9, 1.4), "numberOfAnalystOpinions": r.randint(2, 40),
            "recommendationMean": r.uniform(1.5, 3.0), "enterpriseToEbitda": r.uniform(6, 30), "priceToSalesTrailing12Months": r.uniform(1, 15),
            "dividendYield": r.uniform(0, 0.04),
        }

    def news(self, sources: dict) -> list[dict]:
        from .config import load_universe
        uni = load_universe()
        r = self._rng("news" + str(self.end.date()))
        now = dt.datetime.now(dt.timezone.utc)
        items = []
        templates_zh = ["{k}需求強勁 法人看好後市", "{k}報價上漲 訂單能見度延伸", "{k}概念股大漲 外資上調目標價", "{k}供應鏈示警 庫存調整疲弱"]
        templates_en = ["{k} demand surges as analysts upgrade outlook", "{k} stocks rally on record orders", "Why {k} could face a selloff on weak demand"]
        for s in uni["sectors"]:
            w = r.uniform(0.2, 1.0)
            for _ in range(int(w * 12)):
                zh = r.random() < 0.5
                k = r.choice(s["keywords_zh"] if zh else s["keywords_en"])
                tpl = r.choice(templates_zh if zh else templates_en)
                items.append({"title": tpl.format(k=k), "link": "https://example.invalid/demo", "summary": "", "source": "DEMO",
                              "market": "TW" if zh else "US", "weight": 1.0, "published": now - dt.timedelta(hours=r.uniform(0, 40))})
        return items
