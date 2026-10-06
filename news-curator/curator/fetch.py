"""抓取並解析 RSS 2.0 / Atom,只用標準庫。"""
from __future__ import annotations

import html
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

UA = "news-curator/0.1 (+https://github.com/LiSh-chen/ForClaude)"
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@dataclass
class Article:
    title: str
    url: str
    summary: str
    source: str
    published: datetime | None
    source_weight: float = 0.5
    # 評分後填入
    score: float = 0.0
    why: str = ""
    one_liner: str = ""
    dims: dict = field(default_factory=dict)


def clean_text(s: str | None, limit: int = 600) -> str:
    s = html.unescape(_TAG.sub(" ", s or ""))
    s = _WS.sub(" ", s).strip()
    return s[:limit]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(el: ET.Element, *names: str) -> str:
    for c in el:
        if _local(c.tag) in names and (c.text or "").strip():
            return c.text.strip()
    return ""


def parse_date(s: str) -> datetime | None:
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def parse_feed(xml_bytes: bytes, source: str, weight: float = 0.5) -> list[Article]:
    root = ET.fromstring(xml_bytes)
    out: list[Article] = []
    for it in root.iter():
        if _local(it.tag) not in ("item", "entry"):
            continue
        title = clean_text(_child_text(it, "title"), 300)
        url = _child_text(it, "link")
        if not url:  # Atom: <link href=...>
            for c in it:
                if _local(c.tag) == "link" and c.get("href"):
                    if c.get("rel", "alternate") == "alternate":
                        url = c.get("href")
                        break
        summary = clean_text(_child_text(it, "description", "summary", "encoded", "content"))
        date = parse_date(_child_text(it, "pubDate", "published", "updated", "date"))
        if title and url:
            out.append(Article(title, url.strip(), summary, source, date, weight))
    return out


def fetch_feed(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_all(feeds: list[dict], log=print) -> list[Article]:
    arts: list[Article] = []
    for f in feeds:
        try:
            got = parse_feed(fetch_feed(f["url"]), f["name"], f.get("weight", 0.5))
            log(f"  ✓ {f['name']}: {len(got)} 篇")
            arts += got
        except Exception as e:  # 單一來源失敗不影響整體
            log(f"  ✗ {f['name']}: {type(e).__name__}: {e}")
    return arts


def recent(arts: list[Article], days: int, now: datetime | None = None) -> list[Article]:
    now = now or datetime.now(timezone.utc)
    return [a for a in arts if a.published is None or (now - a.published).total_seconds() <= days * 86400]


def dedupe(arts: list[Article], seen_urls: set[str] = frozenset()) -> list[Article]:
    """依網址與標題正規化去重,並排除寄過的。"""
    keys: set[str] = set()
    out = []
    for a in arts:
        u = a.url.split("#")[0].rstrip("/")
        t = re.sub(r"\W+", "", a.title.lower())
        if u in seen_urls or u in keys or t in keys:
            continue
        keys.update((u, t))
        out.append(a)
    return out
