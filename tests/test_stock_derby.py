import importlib.util
import sys
from pathlib import Path

import numpy as np
import json
import math
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
        assert set(pool) == set(bd.DISTS) == {"d", "w", "w2", "m", "q", "h"}
        for blk in pool.values():
            rs = blk["races"]
            assert 2 <= len(rs) <= bd.N_HIST + 1
            assert [r["k"] for r in rs] == sorted(r["k"] for r in rs) and rs[0]["k"] == 0
            assert rs[0]["status"] == "upcoming" and all(r["status"] == "done" for r in rs[1:])
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
            fh = man["frame_h"] if h["art"] == "pixel" else round(man["frame_h"] * 2.5)       # smooth styles are drawn at 2.5x
            assert im.height == fh and im.width == fh * 4 // 3 * (man["run_frames"] + 1)
            assert h["art"] in ("pixel", "flat", "sketch", "neon")
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


def test_race_bounds_calendar_periods():
    dates = pd.bdate_range("2025-01-01", "2026-10-01")
    live = dates.get_loc(pd.Timestamp("2026-09-30"))            # data through the quarter-end close
    q0 = bd.race_bounds(dates, "q", 0, live)
    assert q0["e"] is None and q0["label"] == "2026Q4" and dates[q0["s"]] == pd.Timestamp("2026-09-30")
    q1 = bd.race_bounds(dates, "q", 1, live)
    assert dates[q1["e"]] == dates[q0["s"]] and q1["label"] == "2026Q3"
    w0 = bd.race_bounds(dates, "w", 0, live)                    # Wed 09-30 -> the week of 09-28 is in progress
    assert dates[w0["s"]] == pd.Timestamp("2026-09-25") and w0["label"] == "09/28~10/02"
    w1 = bd.race_bounds(dates, "w", 1, live)
    assert dates[w1["e"]] == pd.Timestamp("2026-09-25") and w1["e"] == w0["s"]            # consecutive, non-overlapping
    d0, d1 = bd.race_bounds(dates, "d", 0, live), bd.race_bounds(dates, "d", 1, live)
    assert d0["s"] == live and d1["e"] == live and d1["s"] == live - 1
    h0 = bd.race_bounds(dates, "h", 0, live)
    assert h0["label"] == "2026H2" and dates[h0["s"]] == pd.Timestamp("2026-06-30")
    m2 = bd.race_bounds(dates, "m", 0, live)
    assert m2["label"] == "2026-10" and dates[m2["s"]] == pd.Timestamp("2026-09-30")


def test_next_session_skips_weekends_and_holidays():
    assert bd.next_session(pd.Timestamp("2026-10-02")) == pd.Timestamp("2026-10-05")      # Fri -> Mon
    assert bd.next_session(pd.Timestamp("2026-11-25")) == pd.Timestamp("2026-11-27")      # Thanksgiving 11-26
    assert bd.last_session_on_or_before(pd.Timestamp("2026-10-03")) == pd.Timestamp("2026-10-02")


def test_recent_prices_are_spliced_onto_the_stored_history():
    import data_loader as dl
    base = pd.DataFrame({"stock_id": "A", "date": pd.to_datetime(["2026-09-29", "2026-09-30"]), "high": [10.1, 10.4],
                         "low": [9.8, 9.9], "close": [10.0, 10.3], "volume": 1, "industry": "IT"})
    rec = pd.DataFrame({"stock_id": "A", "date": pd.to_datetime(["2026-09-30", "2026-10-01"]), "high": [20.8, 21.2],
                        "low": [19.8, 20.0], "close": [20.6, 21.0], "volume": 1})        # re-adjusted: everything x2
    out = dl.splice_recent(base, rec)
    assert list(out["date"].dt.strftime("%m-%d")) == ["09-29", "09-30", "10-01"]
    assert abs(out["close"].iloc[-1] - 10.5) < 1e-9                                     # rescaled to the stored basis
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
    fw = im.height * 4 // 3
    fr = [im.crop((i * fw, 0, i * fw + fw, im.height)) for i in range(6)]
    assert all(ImageChops.difference(fr[i], fr[(i + 1) % 6]).getbbox() for i in range(6))


def test_months_back_is_calendar_based():
    dates = pd.bdate_range("2026-01-01", "2026-09-30")
    i = len(dates) - 1                                   # 2026-09-30
    assert dates[bd.months_back(dates, i, 3)] == pd.Timestamp("2026-06-30")
    assert dates[bd.months_back(dates, i, 1)] == pd.Timestamp("2026-08-28")   # 08-30 is a Sunday
    assert bd.months_back(dates, 5, 6) == -1             # before the data starts


def test_field_is_previous_periods_top_ten():
    cols = {f"S{i}": [100.0, 100.0 + i, 110.0] for i in range(15)}
    cols["NEW"] = [np.nan, 150.0, 160.0]                  # no price at the start of the previous period -> not eligible
    close = pd.DataFrame(cols)
    field, mom, ok = bd.select_field_prior(close, list(cols), 0, 1)
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


def test_nyse_calendar_and_quarter_length():
    hol = {h.date().isoformat() for h in bd.nyse_holidays(2026)}
    assert {"2026-11-26", "2026-12-25", "2026-07-03", "2026-04-03"} <= hol      # Thanksgiving, Christmas, July 4 observed, Good Friday
    assert bd.trading_days_between(pd.Timestamp("2026-09-30"), pd.Timestamp("2026-12-31")) == 64


def test_in_play_odds_get_more_expensive_as_the_season_runs():
    cum = np.array([0.30, 0.10, 0.05, 0.0, -0.05])
    sig = np.full(5, 0.02)
    p_open = np.full(5, 0.2)
    early = bd.live_odds(p_open, np.full(5, 0.6), cum, sig, 60, 0.05)
    late = bd.live_odds(p_open, np.full(5, 0.6), cum, sig, 10, 0.85)
    assert abs(early.p_win.sum() - 1) < 1e-6 and abs(late.p_win.sum() - 1) < 1e-6
    assert late.odds_win.iloc[0] < early.odds_win.iloc[0]          # the leader gets cheaper to back... i.e. pays less
    assert late.odds_win.iloc[0] < late.odds_win.iloc[-1]          # and pays less than a straggler
    assert bd.tau(0.0) == bd.TAKEOUT and bd.tau(1.0) == bd.TAKE_END and bd.tau(0.5) > bd.tau(0.2)
    assert (late.odds_win >= bd.MIN_ODDS).all()


def test_live_block_of_the_season_in_progress():
    d = _doc()
    for pool in d["pools"].values():
        up = pool["q"]["races"][0]
        L = up["live"]
        assert 0 <= L["u"] <= 1 and L["total_days"] >= L["day"] + 1 and len(L["dates"]) == L["day"] + 1
        assert L["betting_open"] == (L["day"] <= max(0, math.ceil(L["close_u"] * L["total_days"]) - 1)) and abs(L["tau"] - bd.tau(L["u"])) < 1e-3
        assert L["next_first_session"] > up["planned_end"] and pd.Timestamp(L["next_first_session"]).weekday() < 5     # the next period starts on a session
        assert L["close_at"].endswith("Z") and L["close_at"][11:] in ("13:30:00Z", "14:30:00Z")             # 09:30 New York
        for h in up["horses"]:
            assert len(h["live_path"]) == L["day"] + 1 and h["live_path"][0] == 0
            assert h["odds_win"] >= bd.MIN_ODDS and "odds_win_open" in h


def test_every_cycle_has_a_live_race_with_capped_odds_and_replayable_history():
    d = _doc()
    for pk, pool in d["pools"].items():
        for dist, blk in pool.items():
            up = blk["races"][0]
            assert abs(sum(h["p_win"] for h in up["horses"]) - 1) < 0.02, (pk, dist)
            for h in up["horses"]:
                assert bd.MIN_ODDS <= h["odds_win"] <= bd.MAX_ODDS and bd.MIN_ODDS <= h["odds_place"] <= bd.MAX_ODDS
                assert h["odds_place"] <= h["odds_win"] + 1e-9
            for r in blk["races"][1:]:                          # every finished period can be replayed and settled
                assert "dates" in r and all("path" in h for h in r["horses"]) and r["id"] in d["settled"]
