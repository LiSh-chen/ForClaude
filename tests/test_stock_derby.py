import sys
from pathlib import Path

import numpy as np
import json
import math
import pandas as pd
import pytest

D = Path(__file__).parent.parent / "stock-derby"
sys.path.insert(0, str(D))
import build_panel as bp  # noqa: E402
import factors as F  # noqa: E402
import fetch_extras as fe  # noqa: E402


def test_factors_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.exp(rng.normal(0, .01, (400, 3)).cumsum(0)), index=idx, columns=list("abc"))
    p = {"close": close, "high": close * 1.01, "low": close * .99, "volume": close * 0 + 1e6}
    e = pd.DataFrame(columns=["date", "stock_id", "eps_actual", "surprise_pct"])
    base = F.compute_all(p, e)
    p2 = {k: v.copy() for k, v in p.items()}
    p2["close"].iloc[-1] *= 5                                  # tamper with the last day only
    alt = F.compute_all(p2, e)
    for name in ("mom_12_1", "mom_3m", "high_52w", "vol_60", "rsi_14"):
        pd.testing.assert_frame_equal(base[name].iloc[:-1], alt[name].iloc[:-1])


def test_merge_long_new_wins():
    old = pd.DataFrame({"date": ["2026-01-02", "2026-01-03"], "stock_id": "A", "close": [1.0, 2.0]})
    new = pd.DataFrame({"date": ["2026-01-03", "2026-01-06"], "stock_id": "A", "close": [2.5, 3.0]})
    out = fe.merge_long(old, new)
    assert list(out["close"]) == [1.0, 2.5, 3.0]


def test_recent_prices_are_spliced_onto_the_stored_history():
    import data_loader as dl
    base = pd.DataFrame({"stock_id": "A", "date": pd.to_datetime(["2026-09-29", "2026-09-30"]), "high": [10.1, 10.4],
                         "low": [9.8, 9.9], "close": [10.0, 10.3], "volume": 1, "industry": "IT"})
    rec = pd.DataFrame({"stock_id": "A", "date": pd.to_datetime(["2026-09-30", "2026-10-01"]), "high": [20.8, 21.2],
                        "low": [19.8, 20.0], "close": [20.6, 21.0], "volume": 1})        # re-adjusted: everything x2
    out = dl.splice_recent(base, rec)
    assert list(out["date"].dt.strftime("%m-%d")) == ["09-29", "09-30", "10-01"]
    assert abs(out["close"].iloc[-1] - 10.5) < 1e-9                                     # rescaled to the stored basis
def test_horse_wording_covers_every_candidate_factor():
    for fid in F.META:
        t0, t1 = F.horse_term(fid, False), F.horse_term(fid, True)
        assert t0["horse"] and t1["horse"] and t0["param"] != t1["param"]


def test_sprite_frames_differ_so_gallop_animates():
    pytest.importorskip("PIL")
    from PIL import Image, ImageChops
    im = Image.open(D / "assets" / "horses" / "NVDA.png")
    fw = im.height * 4 // 3
    fr = [im.crop((i * fw, 0, i * fw + fw, im.height)) for i in range(6)]
    assert all(ImageChops.difference(fr[i], fr[(i + 1) % 6]).getbbox() for i in range(6))


def test_parse_ndx_tables_finds_the_constituents_table_and_reports_what_it_saw():
    rows = "".join(f"<tr><td>Co{i}</td><td>{t}</td></tr>" for i, t in enumerate(f"AB{chr(65 + i % 26)}{chr(65 + i // 26)}" for i in range(100)))
    html = ("<table><tr><th>Year</th><th>Index</th></tr><tr><td>1</td><td>2</td></tr></table>"
            f"<table><tr><th>Company</th><th>Ticker</th></tr>{rows}</table>")
    out = fe.parse_ndx_tables(html)
    assert len(out) == 100 and all(t.isupper() for t in out)
    import pytest as _pt
    with _pt.raises(ValueError, match="saw 1 tables"):
        fe.parse_ndx_tables("<table><tr><th>A</th></tr><tr><td>1</td></tr></table>")


def test_missing_earnings_are_filled_from_the_extra_file(monkeypatch, tmp_path):
    import types
    import data_loader as dl
    snap = pd.DataFrame({"date": ["2026-07-30"], "stock_id": ["AAA"], "eps_estimate": [1.0], "eps_actual": [1.1], "surprise_pct": [10.0]})
    extra = pd.DataFrame({"date": ["2026-07-31"], "stock_id": ["MSTR"], "eps_estimate": [-1.0], "eps_actual": [-0.5], "surprise_pct": [50.0]})
    merged = dl.merge_earnings(snap, extra)
    assert set(merged["stock_id"]) == {"AAA", "MSTR"}

    idx = pd.DatetimeIndex(["2026-07-31", "2026-10-30"], tz="America/New_York")                 # one reported, one still to come
    df = pd.DataFrame({"EPS Estimate": [-1.0, -0.8], "Reported EPS": [-0.5, float("nan")], "Surprise(%)": [50.0, float("nan")]}, index=idx)
    fake = types.SimpleNamespace(Ticker=lambda t: types.SimpleNamespace(get_earnings_dates=lambda limit: df))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = fe.download_earnings(["MSTR"])
    assert len(out) == 1 and out.iloc[0]["stock_id"] == "MSTR" and out.iloc[0]["date"] == "2026-07-31" and out.iloc[0]["surprise_pct"] == 50.0


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    out = tmp_path_factory.mktemp("panel") / "panel.json"
    old = bp.OUT
    bp.OUT = out
    try:
        bp.main()
    finally:
        bp.OUT = old
    return json.loads(out.read_text())


def test_panel_structure_and_alignment(panel):
    p = panel
    n = len(p["dates"])
    assert 250 < n <= bp.N_DATES and p["dates"] == sorted(p["dates"]) and p["asof"] == p["dates"][-1]
    assert len(p["tk"]) == len(p["px"]) and all(len(r) == n for r in p["px"])
    assert set(p["pools"]) == {"spx", "ndx", "all"} and set(p["pools"]["spx"]) | set(p["pools"]["ndx"]) == set(p["pools"]["all"])
    assert set(p["pools"]["all"]) <= set(p["tk"]) and set(p["looks"]) == set(p["tk"]) == set(p["ab"])
    last = [r[-1] for r in p["px"]]
    assert sum(v is not None for v in last) / len(last) >= bp.MIN_COVER, "the latest session must have (almost) full coverage"
    assert all(v > 0 for r in p["px"] for v in r if v is not None)
    assert len(p["key_factors"]) == 5 and {"1m", "3m", "6m"} == set(p["weights"])
    for w in p["weights"].values():
        assert abs(sum(w) - 1) < 1e-3
    assert set(p["ic_key"]) == {"d", "w", "w2", "m", "q", "h"}


def test_panel_abilities_are_percentiles_per_pool(panel):
    p = panel
    for pk, ts in p["pools"].items():
        vals = [p["ab"][t]["pct"][pk] for t in ts]
        assert all(len(v) == 5 for v in vals)
        for k in range(5):
            col = [v[k] for v in vals if v[k] is not None]
            assert col and 0 <= min(col) and max(col) <= 100 and max(col) > 95


def test_every_ticker_has_a_name_and_a_sprite_that_exists(panel):
    gen = {g["key"] for g in panel["generic"]}
    for t, l in panel["looks"].items():
        assert l["zh"] and l["en"]
        if "sp" in l:
            assert (D / "assets" / "horses" / f"{l['sp']}.png").exists()
        else:
            assert 0 <= l["g"] < len(gen)
    for g in gen:
        assert (D / "assets" / "horses" / f"generic_{g}.png").exists()


def test_key_factors_have_horse_wording_and_params(panel):
    for f in panel["key_factors"]:
        assert f["horse"] and f["meaning"] and f["param"] and f["id"] in F.META
        assert f["3m"]["ic"] is not None
