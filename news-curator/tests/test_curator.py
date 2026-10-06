import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from curator import digest, fetch, score, subscribers

RSS = b"""<rss version="2.0"><channel>
<item><title>Study finds new mechanism</title><link>https://x.com/a</link>
<description>&lt;p&gt;Researchers discover data&lt;/p&gt;</description>
<pubDate>Mon, 05 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>No link item</title></item></channel></rss>"""
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>T</title>
<link rel="alternate" href="https://y.com/b"/><summary>S</summary>
<updated>2026-10-05T10:00:00Z</updated></entry></feed>"""


def test_parse_rss_and_atom():
    r = fetch.parse_feed(RSS, "S")
    assert len(r) == 1 and r[0].summary == "Researchers discover data" and r[0].published
    a = fetch.parse_feed(ATOM, "S")
    assert a[0].url == "https://y.com/b"


def test_recent_and_dedupe():
    now = datetime(2026, 10, 6, tzinfo=timezone.utc)
    old = fetch.Article("Old", "https://o", "", "S", now - timedelta(days=30))
    new = fetch.Article("New", "https://n/", "", "S", now - timedelta(days=1))
    dup = fetch.Article("new!", "https://other", "", "S", now)
    assert fetch.recent([old, new], 7, now) == [new]
    assert fetch.dedupe([new, dup, new]) == [new]
    assert fetch.dedupe([new], {"https://n"}) == []


def test_compose_penalizes_trivia():
    d = dict(novelty=8, depth=8, impact=8, evidence=8, trivia=0)
    assert score.compose(d) == 80.0
    assert score.compose({**d, "trivia": 10}) == pytest.approx(48.0)


def test_extract_json_with_noise():
    assert score.extract_json('好的:\n[{"id":0}]') == [{"id": 0}]


def test_heuristic_prefers_substance():
    good = fetch.Article("Researchers discover mechanism", "u1", "A study with data and evidence " * 15, "Nature", None, 0.9)
    bad = fetch.Article("10 best ways to save: coupon deal", "u2", "short", "Blog", None, 0.5)
    score.score_heuristic([good, bad])
    assert good.score > bad.score


def test_select_threshold_and_diversity():
    arts = [fetch.Article(f"t{i}", f"u{i}", "", "A" if i < 4 else "B", None) for i in range(6)]
    for i, a in enumerate(arts):
        a.score = 90 - i * 10
    picked = score.select(arts, top_n=5, min_score=50, per_source=2)
    assert [a.title for a in picked] == ["t0", "t1", "t4"]  # A 最多 2 篇;t5=40 低於門檻


def test_digest_escapes_html():
    a = fetch.Article("<script>x</script>", "https://a?x=1&y=2", "", "S", None)
    a.one_liner, a.why, a.score = "o", "w", 70
    out = digest.render_html([a])
    assert "<script>" not in out and "&amp;y=2" in out


def test_subscribers_roundtrip(tmp_path, monkeypatch):
    f = tmp_path / "s.txt"
    monkeypatch.delenv("NEWS_SUBSCRIBERS", raising=False)
    assert subscribers.add("A@b.com", f) and not subscribers.add("a@b.com", f)
    with pytest.raises(ValueError):
        subscribers.add("nope", f)
    monkeypatch.setenv("NEWS_SUBSCRIBERS", "c@d.com, bad,\ne@f.org")
    assert subscribers.load(f) == ["c@d.com", "e@f.org", "a@b.com"]
    assert subscribers.remove("a@b.com", f) and not subscribers.remove("a@b.com", f)
