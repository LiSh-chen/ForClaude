import importlib.util
import sys
from pathlib import Path

import numpy as np
import json
import pandas as pd
import pytest

D = Path(__file__).parent.parent / "stock-derby"
sys.path.insert(0, str(D))
import build_derby as bd  # noqa: E402
import factors as F  # noqa: E402
import fetch_extras as fe  # noqa: E402


def test_style_label_boundaries():
    assert bd.style_label(1.0)[0] == "領放"
    assert bd.style_label(4.0)[0] == "跟前"
    assert bd.style_label(6.0)[0] == "居中"
    assert bd.style_label(9.0)[0] == "後上"


def test_stars_range():
    assert bd.stars(0) == 1 and bd.stars(100) == 5 and bd.stars(55) == 3


def test_ranks_best_is_one_and_nan_ignored():
    r = bd.ranks_of(pd.Series({"a": 0.1, "b": 0.3, "c": np.nan}))
    assert r["b"] == 1 and r["a"] == 2 and np.isnan(r["c"])


def test_window_ranks_uses_only_window_and_handles_missing_history():
    close = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6.0], "b": [1, 1, 1, 1, 1, 1.0], "c": [np.nan] * 3 + [1, 1, 3.0]})
    early, fin = bd.window_ranks(close, ["a", "b", "c"], 0, 5)
    assert fin["a"] == 1 and fin["b"] == 2 and np.isnan(fin["c"])
    assert bd.window_ranks(close, ["a"], -1, 3) is None


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


def _doc():
    import json
    return json.loads((D / "data" / "derby.json").read_text())


def _all_races(d):
    for pool in d["pools"].values():
        for dist, blk in pool.items():
            for race in blk["races"]:
                yield dist, race


def test_derby_json_structure():
    d = _doc()
    assert set(d["pools"]) == {"spx", "ndx", "all"} and len(d["key_factors"]) == 5
    for pool in d["pools"].values():
        assert set(pool) == {"1m", "3m", "6m", "q"}
        for dist, blk in pool.items():
            assert 1 <= len(blk["races"]) <= bd.N_RACES
            assert [r["k"] for r in blk["races"]] == sorted(r["k"] for r in blk["races"])
    for dist, race in _all_races(d):
        assert len(race["horses"]) == 10


def test_every_horse_has_sprite_sheet_and_name():
    pytest.importorskip("PIL")        # Pillow isn't in requirements.txt (only the sprite tools need it)
    from PIL import Image
    d = _doc()
    man = json.loads((D / "assets" / "horses" / "manifest.json").read_text())
    for dist, race in _all_races(d):
        sprites = [h["sprite"] for h in race["horses"]]
        assert len(set(sprites)) == len(sprites), "no two identical coats in one field"
        for h in race["horses"]:
            im = Image.open(D / "assets" / "horses" / f"{h['sprite']}.png")
            assert im.size == (man["frame_w"] * (man["run_frames"] + 1), man["frame_h"])
            assert h["horse_zh"] and h["horse_en"]


def test_finished_races_match_real_prices_and_prediction_rules():
    d = _doc()
    for dist, race in _all_races(d):
        hs = race["horses"]
        assert [h["pred"] for h in hs] == list(range(1, 11))              # numbered by prediction
        assert [h["total"] for h in hs] == sorted((h["total"] for h in hs), reverse=True)
        if race["status"] != "done":
            assert all("ret" not in h and "result" not in h for h in hs), "unknown results must not leak"
            continue
        for h in hs:
            assert abs(h["price_end"] / h["price_start"] - 1 - h["ret"]) < 2e-3
        by_ret = sorted(hs, key=lambda h: -h["ret"])
        assert [h["result"] for h in by_ret] == list(range(1, 11))        # result = return rank
        if "dates" in race:
            assert race["dates"][0] == race["start"] and race["dates"][-1] == race["end"]
            assert all(h["path"][0] == 0 and abs(h["path"][-1] - h["ret"]) < 1e-3 for h in hs)


def test_quarter_race_is_upcoming_with_sane_odds_and_settlement_data():
    d = _doc()
    for pk, pool in d["pools"].items():
        races = pool["q"]["races"]
        up = races[0]
        assert up["status"] == "upcoming" and up["end"] is None and up["planned_end"] > up["start"]
        assert abs(sum(h["p_win"] for h in up["horses"]) - 1) < 0.01
        for h in up["horses"]:
            assert h["odds_win"] >= 1.2 and h["odds_place"] >= 1.1 and h["odds_place"] < h["odds_win"]
        # the horse the card likes best must not pay more than the one it likes least
        assert up["horses"][0]["odds_win"] <= up["horses"][-1]["odds_win"]
        for r in races[1:]:
            assert r["status"] == "done" and r["id"] in d["settled"]
            assert sorted(d["settled"][r["id"]]["result"].values()) == list(range(1, 11))


def test_key_factors_have_horse_wording_and_params():
    d = _doc()
    for f in d["key_factors"]:
        assert f["horse"] and f["meaning"] and f["param"] and f["avg_ic"] >= 0 and "sign" not in f


def test_race_bounds_quarter_and_months():
    dates = pd.bdate_range("2025-01-01", "2026-10-01")
    live = dates.get_loc(pd.Timestamp("2026-09-30"))             # data through the quarter-end close -> Q4 is the race in progress
    q0 = bd.race_bounds(dates, "q", 0, live)
    assert q0["e"] is None and q0["planned_end"] == "2026-12-31"
    assert dates[q0["s"]] == pd.Timestamp("2026-09-30")          # starts at the last quarter-end close
    q1 = bd.race_bounds(dates, "q", 1, live)
    assert dates[q1["e"]] == dates[q0["s"]] and dates[q1["s"]] <= pd.Timestamp("2026-06-30")
    m1 = bd.race_bounds(dates, "3m", 1, live)
    assert m1["e"] == bd.race_bounds(dates, "3m", 0, live)["s"]   # consecutive, non-overlapping races


def test_odds_table_is_a_proper_distribution_with_house_edge():
    sc = pd.Series({"a": 90.0, "b": 60.0, "c": 50.0, "d": 40.0, "e": 30.0, "f": 20.0})
    od = bd.odds_table(sc)
    assert abs(od.p_win.sum() - 1) < 1e-6 and abs(od.p_place.sum() - 3) < 1e-6
    assert od.odds_win.is_monotonic_increasing                    # better score -> lower payout
    assert (od.p_win * od.odds_win).max() <= 1 - bd.TAKEOUT + 0.1 # house keeps its cut (rounding/floor aside)


def test_horse_wording_covers_every_candidate_factor():
    for fid in F.META:
        t0, t1 = F.horse_term(fid, False), F.horse_term(fid, True)
        assert t0["horse"] and t1["horse"] and t0["param"] != t1["param"]


def test_sprite_frames_differ_so_gallop_animates():
    pytest.importorskip("PIL")
    from PIL import Image, ImageChops
    im = Image.open(D / "assets" / "horses" / "NVDA.png")
    fr = [im.crop((i * 48, 0, i * 48 + 48, 36)) for i in range(6)]
    assert all(ImageChops.difference(fr[i], fr[(i + 1) % 6]).getbbox() for i in range(6))


def test_months_back_is_calendar_based():
    dates = pd.bdate_range("2026-01-01", "2026-09-30")
    i = len(dates) - 1                                   # 2026-09-30
    assert dates[bd.months_back(dates, i, 3)] == pd.Timestamp("2026-06-30")
    assert dates[bd.months_back(dates, i, 1)] == pd.Timestamp("2026-08-28")   # 08-30 is a Sunday
    assert bd.months_back(dates, 5, 6) == -1             # before the data starts


def test_select_field_is_top_ten_by_return():
    cols = {f"S{i}": [100.0, 100.0 + i] for i in range(15)}
    cols["NEW"] = [np.nan, 150.0]                         # no start price -> not eligible
    close = pd.DataFrame(cols)
    field, ret, ok = bd.select_field(close, list(cols), 0, 1)
    assert field == [f"S{i}" for i in range(14, 4, -1)] and "NEW" not in ok


def test_total_score_renormalises_missing_factors():
    pct = pd.DataFrame({"a": [100.0, 50.0], "b": [100.0, np.nan]}, index=["x", "y"])
    sc = bd.total_score(pct, np.array([0.5, 0.5]))
    assert sc["x"] == 100.0 and sc["y"] == 50.0           # y's weight moves entirely to factor a


def test_parse_ndx_tables_finds_the_constituents_table_and_reports_what_it_saw():
    rows = "".join(f"<tr><td>Co{i}</td><td>{t}</td></tr>" for i, t in enumerate(f"AB{chr(65 + i % 26)}{chr(65 + i // 26)}" for i in range(100)))
    html = ("<table><tr><th>Year</th><th>Index</th></tr><tr><td>1</td><td>2</td></tr></table>"
            f"<table><tr><th>Company</th><th>Ticker</th></tr>{rows}</table>")
    out = fe.parse_ndx_tables(html)
    assert len(out) == 100 and all(t.isupper() for t in out)
    import pytest as _pt
    with _pt.raises(ValueError, match="saw 1 tables"):
        fe.parse_ndx_tables("<table><tr><th>A</th></tr><tr><td>1</td></tr></table>")
