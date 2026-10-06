import datetime as dt
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["IJ_OUT"] = tempfile.mkdtemp()
(Path(os.environ["IJ_OUT"]) / "config").mkdir()

from ijournal import config as C  # noqa: E402
from ijournal.news import scan  # noqa: E402
from ijournal.providers import DemoProvider, parse_feed  # noqa: E402
from ijournal.tracker import _fwd  # noqa: E402
from ijournal.valuation import dcf_per_share, value_stock  # noqa: E402

P = C.load_params()


def test_parse_rss_and_atom():
    rss = b"<rss><channel><item><title>A</title><link>http://x</link><pubDate>Tue, 06 Oct 2026 01:00:00 GMT</pubDate></item></channel></rss>"
    assert parse_feed(rss)[0]["title"] == "A"
    atom = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>B</title><link href="http://y"/><updated>2026-10-06T01:00:00Z</updated></entry></feed>'
    r = parse_feed(atom)[0]
    assert r["title"] == "B" and r["link"] == "http://y" and r["published"] is not None


def test_news_sector_and_ticker_matching():
    uni, src = C.load_universe(), C.load_sources()
    now = dt.datetime.now(dt.timezone.utc)
    items = [{"title": "台積電 CoWoS 擴產 訂單強勁", "summary": "", "link": "", "source": "t", "market": "TW", "weight": 1.0, "published": now},
             {"title": "Nvidia stock surges on record AI chip demand", "summary": "", "link": "", "source": "t", "market": "US", "weight": 1.0, "published": now},
             {"title": "old news about GPU", "summary": "", "link": "", "source": "t", "market": "US", "weight": 1.0, "published": now - dt.timedelta(days=5)}]
    r = scan(items, uni, src, now)
    assert r["sector"]["ai_chips"]["TW"]["n"] == 1 and r["sector"]["ai_chips"]["US"]["n"] == 1  # 過期新聞被排除
    assert r["sector"]["ai_chips"]["US"]["sent"] > 0
    assert "2330.TW" in r["tickers"] and "NVDA" in r["tickers"]


def _f(**kw):
    base = {"eps_fwd_px": 10.0, "eps_ttm_px": 9.0, "same_ccy": True, "fcf_ps": 6.0, "beta": 1.0, "target_mean": 150.0, "n_analysts": 10,
            "target_low": 100, "target_high": 200, "eps_growth": 0.2, "rev_growth": 0.15}
    base.update(kw)
    return base


def test_valuation_blend_and_scenarios():
    v = value_stock(100.0, _f(), {"vol_ann": 0.3}, "US", {"pe": 18.0, "g": 0.1, "n": 5}, P)
    assert v and v["n_methods"] == 4
    assert abs(sum(m["weight"] for m in v["methods"].values()) - 1) < 1e-9
    assert v["bear"] < v["base"] < v["bull"]
    assert v["confidence"] in "高中低"


def test_outlier_method_dropped_and_insufficient_data_rejected():
    v = value_stock(100.0, _f(target_mean=1000.0), {"vol_ann": 0.3}, "US", {"pe": 18.0, "g": 0.1, "n": 5}, P)
    assert "consensus" in v["dropped"] and "consensus" not in v["methods"]
    assert value_stock(100.0, _f(eps_fwd_px=None, eps_ttm_px=None, fcf_ps=None), {"vol_ann": 0.3}, "US", {"pe": None, "g": None, "n": 0}, P) is None


def test_target_shrink_reduces_upside():
    p2 = dict(P, target_shrink=0.5)
    a = value_stock(100.0, _f(), {"vol_ann": 0.3}, "US", {"pe": 18.0, "g": 0.1, "n": 5}, P)
    b = value_stock(100.0, _f(), {"vol_ann": 0.3}, "US", {"pe": 18.0, "g": 0.1, "n": 5}, p2)
    assert abs(b["upside"] - a["upside"] * 0.5) < 1e-9


def test_dcf_negative_fcf_skipped():
    assert dcf_per_share({"fcf_ps": -1.0}, 0.1, "US", 1.0, P)[0] is None


def test_forward_returns_and_alpha():
    idx = pd.bdate_range("2026-01-01", periods=30)
    stock = pd.DataFrame({"Close": [100 + i for i in range(30)], "High": [101 + i for i in range(30)], "Low": [99 + i for i in range(30)]}, index=idx)
    bench = pd.DataFrame({"Close": [100.0] * 30, "High": [100.0] * 30, "Low": [100.0] * 30}, index=idx)
    f = _fwd(stock, bench, str(idx[0].date()))
    assert abs(f["r5"] - 0.05) < 1e-9 and abs(f["a5"] - 0.05) < 1e-9
    assert f["r60"] is None  # 尚未到期
    assert f["n_days"] == 29


def test_end_to_end_with_feedback_loop():
    from ijournal.daily import run_daily
    from ijournal.review import run_review
    from ijournal.tracker import run_tracker
    prov = DemoProvider(end=dt.date(2026, 9, 30))
    dates = [run_daily(prov, asof=d) for d in ("2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22")]
    assert all(dates)
    assert run_daily(prov, asof="2026-06-01") is None  # 同日不重複產生
    data = C.path("data")
    pj = C.load_json(data / "picks" / "2026-06-01.json")
    assert len(pj["picks"]) <= 10 and all(p["target"]["bear"] < p["target"]["base"] for p in pj["picks"])
    perf = run_tracker(prov)
    assert perf["positions"] and perf["summary"]["20"]["n"] > 0
    out = Path(run_review("monthly", tune=True, date="2026-09-30"))
    assert out.exists()
    # 樣本不足（僅 4 個推薦日）→ 不應調整任何參數
    assert C.load_params()["version"] == 1
    # 放寬門檻 → 會調整並留下稽核紀錄
    run_review("monthly", tune=True, date="2026-09-30", min_dates=2)
    assert C.load_params()["version"] == 2 and C.load_json(C.history_path())[-1]["changes"]
    w = C.load_params()["factor_weights"]
    assert abs(sum(w.values()) - 1) < 1e-9 and all(0.03 <= x <= 0.35 for x in w.values())
    from ijournal.site import build_site
    build_site()
    assert (C.path("site") / "index.html").exists() and (C.path("site") / "performance.html").exists()
