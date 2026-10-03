"""Build data/derby.json: race cards (馬報), replays and quarterly betting odds for every pool x distance.

pool     : spx (S&P 500) | ndx (Nasdaq-100) | all (union)
distance : 1m / 3m / 6m = calendar months, ending on the latest trading day with price coverage;
           q = calendar quarters (race 0 is the quarter in progress: result unknown, odds only)
races    : for each distance the latest race plus the 4 before it (N_RACES entries), each with its own card
field    : 1m/3m/6m -> the 10 stocks with the highest return over the race window (review races);
           q        -> the 10 stocks with the highest return over the *previous* quarter (entries known before the start)
card     : 近績 / 跑法 / 能力 only use data up to the start date, so the ability ranking is a genuine pre-race
           prediction that can be compared with the result.
odds     : simulated from the card (0.7 ability + 0.3 recent form) with a Plackett-Luce model; virtual points only.
validation: the same prediction replayed over past races (in-sample, see caveats in the UI).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader as dl
import factors as F

# cycle -> (pandas freq, number of base units per period).  A period is a calendar block: day = one session,
# w = Mon-Sun week, w2 = two such weeks, m = month, q = quarter, h = half-year.
CYCLES = {"d": None, "w": ("W-SUN", 1), "w2": ("W-SUN", 2), "m": ("M", 1), "q": ("Q", 1), "h": ("Q", 2)}
DISTS = list(CYCLES)
IC_KEY = {"d": "1m", "w": "1m", "w2": "1m", "m": "1m", "q": "3m", "h": "6m"}   # which study horizon weights the abilities
N_HIST = 5                                     # finished periods kept per cycle (plus the one in progress)
TAKEOUT = 0.15                                 # house cut in the simulated odds
ODDS_TEMP = 30.0                               # high on purpose: the ability signal is weak, so odds stay flat
TAKE_END = 0.40                                # house cut at the end of the season: tau(u) = 15% + 25% * u^2
CLOSE_U = 0.80                                 # betting closes once 80% of the season has been run
MIN_ODDS = 1.05
MAX_ODDS = 99.0                                # boards cap long shots, like a real odds board
VOL_SCALE = 0.85                               # backtest (40 quarters): leaders won more often than a plain random walk says
LIVE_SIMS = 4000
RHO = 0.4                                      # one-factor correlation between horses in the remaining-days simulation
FIELD = 10
N_FORM = 4
N_VALID = {"d": 120, "w": 100, "w2": 60, "m": 60, "q": 40, "h": 24}   # past periods replayed for the backtest
KEY_FACTORS = Path(__file__).parent / "data" / "key_factors.json"
OUT = Path(__file__).parent / "data" / "derby.json"
MANIFEST = Path(__file__).parent / "assets" / "horses" / "manifest.json"
ADJ = [("疾風", "Swift"), ("烈焰", "Blaze"), ("飛雲", "Skycloud"), ("銀箭", "Silver Arrow"), ("金蹄", "Gold Hoof"),
       ("雷霆", "Thunder"), ("追月", "Moonchaser"), ("踏雪", "Snowstep"), ("流星", "Comet"), ("驚鴻", "Swan Dive"),
       ("破浪", "Wavebreaker"), ("鐵衛", "Ironclad"), ("赤兔", "Red Hare"), ("青驄", "Bluemane"), ("白虹", "White Rainbow"),
       ("紫電", "Violet Bolt"), ("玄影", "Shadowrun"), ("烈陽", "Sunfire"), ("凌霄", "Skyward"), ("奔雷", "Rumble")]
SECTOR_ZH = {"Information Technology": "科技", "Communication Services": "通訊", "Consumer Discretionary": "非必需消費",
             "Consumer Staples": "必需消費", "Financials": "金融", "Health Care": "醫療", "Industrials": "工業",
             "Energy": "能源", "Utilities": "公用事業", "Real Estate": "房地產", "Materials": "原物料"}
ARTS = ["pixel"]                              # only the pixel library is wired in; smooth styles live in assets/horses_styles (experimental)
MARKS = ["◎", "○", "▲", "△", "△"]




def months_back(dates: pd.DatetimeIndex, i: int, m: int) -> int:
    """Index of the last trading day on/before dates[i] - m calendar months (-1 if before the data starts)."""
    if i <= 0:
        return -1
    return int(dates.searchsorted(dates[i] - pd.DateOffset(months=m), side="right")) - 1


def idx_on(dates: pd.DatetimeIndex, d: pd.Timestamp) -> int:
    """Index of the last trading day on/before date d (-1 if before the data starts)."""
    return int(dates.searchsorted(d, side="right")) - 1


def style_label(early_rank: float) -> tuple[str, str]:
    if early_rank <= 2.5:
        return "領放", "起跑即搶頭香，靠前段優勢領先"
    if early_rank <= 4.5:
        return "跟前", "緊跟領先集團，伺機而動"
    if early_rank <= 7:
        return "居中", "中段待機，不急不徐"
    return "後上", "前段墊後，靠末腳追趕"


def ranks_of(returns: pd.Series) -> pd.Series:
    """1 = best; horses without data (NaN) get NaN and are ignored by the others."""
    return returns.rank(ascending=False, method="min")


def window_ranks(close: pd.DataFrame, field: list[str], lo: int, hi: int) -> tuple[pd.Series, pd.Series] | None:
    """(rank at the 1/3 point, rank at the finish) among the field for window [lo, hi]; None if no history."""
    if lo < 0 or hi <= lo:
        return None
    w = close.iloc[lo:hi + 1][field]
    cum = w / w.iloc[0] - 1
    third = max(1, (hi - lo) // 3)
    return ranks_of(cum.iloc[third]), ranks_of(cum.iloc[-1])


def stars(score: float) -> int:
    return int(np.clip(1 + score // 20, 1, 5))


def weights_for(kf: list[dict], dk: str) -> np.ndarray:
    key = IC_KEY[dk]
    w = np.array([max(f[key]["ic"], 0.002) for f in kf])      # oriented ICs; floor keeps every factor in play
    return w / w.sum()


def ability_pct(ofac: dict, kf: list[dict], start: int, eligible: list[str]) -> pd.DataFrame:
    """Percentile (0-100, higher = better) of each key factor among all eligible stocks at the start date."""
    return pd.DataFrame({f["id"]: ofac[f["id"]].iloc[start][eligible].rank(pct=True) * 100 for f in kf})


def total_score(pct: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """IC-weighted mean of the percentiles; missing factors are dropped and the weights renormalised."""
    vals = pct.to_numpy()
    ww = np.where(np.isnan(vals), 0.0, w)
    return pd.Series((np.nan_to_num(vals) * ww).sum(1) / np.where(ww.sum(1) == 0, np.nan, ww.sum(1)), index=pct.index)


def select_field_prior(close: pd.DataFrame, tickers: list[str], p0: int, s: int) -> tuple[list[str], pd.Series, list[str]]:
    """Quarterly races: entries = best performers of the previous window [p0, s] (known before the start)."""
    ok = [t for t in tickers if t in close.columns and pd.notna(close[t].iloc[p0]) and pd.notna(close[t].iloc[s])]
    mom = (close[ok].iloc[s] / close[ok].iloc[p0] - 1).sort_values(ascending=False)
    return list(mom.index[:FIELD]), mom, ok


_HOL: dict[int, set] = {}


def is_session(ts: pd.Timestamp) -> bool:
    ts = ts.normalize()
    if ts.weekday() >= 5:
        return False
    if ts.year not in _HOL:
        _HOL[ts.year] = {h.normalize() for h in nyse_holidays(ts.year)}
    return ts not in _HOL[ts.year]


def next_session(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.normalize() + pd.Timedelta(days=1)
    while not is_session(d):
        d += pd.Timedelta(days=1)
    return d


def last_session_on_or_before(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.normalize()
    while not is_session(d):
        d -= pd.Timedelta(days=1)
    return d


def sessions_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """NYSE sessions in (start, end]."""
    out, d = [], start.normalize()
    while True:
        d = next_session(d)
        if d > end.normalize():
            return out
        out.append(d)


def period_label(dist: str, block: int) -> str:
    if dist == "d":
        return ""
    freq, span = CYCLES[dist]
    first = pd.Period(ordinal=block * span, freq=freq)
    last = pd.Period(ordinal=(block + 1) * span - 1, freq=freq)
    if dist == "q":
        return f"{first.year}Q{first.quarter}"
    if dist == "h":
        return f"{first.year}H{1 if first.quarter <= 2 else 2}"
    if dist == "m":
        return str(first)
    return f"{first.start_time:%m/%d}~{last.end_time - pd.Timedelta(days=2):%m/%d}"      # weeks: Monday .. Friday


def race_bounds(dates: pd.DatetimeIndex, dist: str, k: int, live_end: int) -> dict:
    """Window of period k (0 = the period in progress, whose next session has not been played yet).
    s / e = start / end index (e is None while running), chain = [s, start of the period before, ...] for 近績 / 跑法."""
    ns = next_session(dates[live_end])
    if dist == "d":
        e = None if k == 0 else live_end - (k - 1)
        s = live_end if k == 0 else e - 1
        chain = [s - j if s - j >= 0 else -1 for j in range(N_FORM + 1)]
        end_ts = ns if k == 0 else dates[e]
        return {"s": s, "e": e, "chain": chain, "end_ts": end_ts, "label": str((ns if k == 0 else dates[e]).date())}
    freq, span = CYCLES[dist]
    b = pd.Period(ns, freq).ordinal // span - k

    def end_of(block):
        return pd.Period(ordinal=(block + 1) * span - 1, freq=freq).end_time.normalize()

    ends = [end_of(b - 1 - j) for j in range(N_FORM + 1)]
    chain = [idx_on(dates, x) for x in ends]
    end_ts = end_of(b)
    e = None if k == 0 else min(idx_on(dates, end_ts), live_end)
    return {"s": chain[0], "e": e, "chain": chain, "end_ts": end_ts, "label": period_label(dist, b)}


def odds_table(score: pd.Series) -> pd.DataFrame:
    """Plackett-Luce win / top-3 probabilities from evaluation scores, then simulated payout odds."""
    strength = np.exp((score - score.mean()) / ODDS_TEMP)
    rng = np.random.default_rng(20261001)
    keys = np.log(strength.to_numpy()) + rng.gumbel(size=(20000, len(strength)))
    place = (-keys).argsort(1).argsort(1)                                     # 0 = winner in each simulation
    p_win = (place == 0).mean(0)
    p_top3 = (place < 3).mean(0)
    return pd.DataFrame({"p_win": p_win, "p_place": p_top3,
                         "odds_win": np.minimum(MAX_ODDS, np.maximum(1.2, np.round((1 - TAKEOUT) / np.maximum(p_win, 1e-3), 1))),
                         "odds_place": np.minimum(MAX_ODDS, np.maximum(1.1, np.round((1 - TAKEOUT) / np.maximum(p_top3, 1e-3), 1)))}, index=score.index)


def nyse_holidays(year: int) -> list[pd.Timestamp]:
    """NYSE full-day closures (weekend holidays are observed on the Friday / Monday)."""
    from dateutil.easter import easter

    def nth(month, weekday, n):                       # n-th weekday of a month (n=-1: last)
        days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
        days = days[(days.month == month) & (days.weekday == weekday)]
        return days[n]

    def observed(d):
        return d - pd.Timedelta(days=1) if d.weekday() == 5 else d + pd.Timedelta(days=1) if d.weekday() == 6 else d

    hol = [observed(pd.Timestamp(year, 1, 1)), nth(1, 0, 2), nth(2, 0, 2), pd.Timestamp(easter(year)) - pd.Timedelta(days=2),
           nth(5, 0, -1), observed(pd.Timestamp(year, 6, 19)), observed(pd.Timestamp(year, 7, 4)), nth(9, 0, 0),
           nth(11, 3, 3), observed(pd.Timestamp(year, 12, 25))]
    return hol


def trading_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Number of NYSE sessions in (start, end]."""
    return len(sessions_between(start, end))


def tau(u: float) -> float:
    return TAKEOUT + (TAKE_END - TAKEOUT) * u ** 2


def live_probs(cum: np.ndarray, sig: np.ndarray, n_rem: int, seed: int = 20261002) -> tuple[np.ndarray, np.ndarray]:
    """P(win), P(top 3) per horse by Monte Carlo of the remaining days (one-factor random walk, zero drift)."""
    k = len(cum)
    if n_rem <= 0:
        order = np.argsort(-cum)
        pw, pp = np.zeros(k), np.zeros(k)
        pw[order[0]], pp[order[:3]] = 1, 1
        return pw, pp
    rng = np.random.default_rng(seed)
    sig = sig * VOL_SCALE
    z = np.sqrt(RHO) * rng.standard_normal((LIVE_SIMS, 1)) + np.sqrt(1 - RHO) * rng.standard_normal((LIVE_SIMS, k))
    final = np.log1p(cum)[None, :] + sig[None, :] * np.sqrt(n_rem) * z - 0.5 * (sig ** 2)[None, :] * n_rem
    rank = (-final).argsort(1).argsort(1)
    return (rank == 0).mean(0), (rank < 3).mean(0)


def live_odds(p_open_win, p_open_place, cum, sig, n_rem: int, u: float) -> pd.DataFrame:
    """In-play prices: geometric blend of the pre-season card odds and the live model, weighted by the elapsed
    fraction u, with a house cut that rises from 15% to 40% over the season. Later bets cost more."""
    pw_live, pp_live = live_probs(np.asarray(cum), np.asarray(sig), n_rem)
    eps = 1e-4
    w = np.exp((1 - u) * np.log(np.maximum(np.asarray(p_open_win), eps)) + u * np.log(np.maximum(pw_live, eps)))
    p_win = w / w.sum()
    q = np.exp((1 - u) * np.log(np.maximum(np.asarray(p_open_place), eps)) + u * np.log(np.maximum(pp_live, eps)))
    p_place = q / q.sum() * 3
    for _ in range(5):                                   # keep probabilities <= 1 while summing to 3
        p_place = np.minimum(p_place, 0.97)
        p_place = p_place / p_place.sum() * 3
    t = tau(u)
    return pd.DataFrame({"p_win": p_win, "p_place": np.minimum(p_place, 0.97),
                         "odds_win": np.minimum(MAX_ODDS, np.maximum(MIN_ODDS, np.round((1 - t) / np.maximum(p_win, 1e-3), 1))),
                         "odds_place": np.minimum(MAX_ODDS, np.maximum(MIN_ODDS, np.round((1 - t) / np.maximum(np.minimum(p_place, 0.97), 1e-3), 1)))})


def form_score(form: list) -> float:
    done = [x for x in form if x is not None]
    return float(np.mean([(11 - r) * 10 for r in done])) if done else 55.0


def assign_looks(field: list[str], manifest: dict) -> dict[str, dict]:
    """Horse name + sprite sheet per ticker. Listed companies get a logo-inspired design; the rest a stable
    generic coat (by ticker hash, never two identical coats in one field) and an adjective+ticker name."""
    generic = list(manifest["generic"])
    used: set[str] = set()
    looks = {}
    for t in field:
        b = manifest["brands"].get(t)
        if b:
            looks[t] = {"company": b["company"], "horse_zh": b["zh"], "horse_en": b["en"], "sprite": t, "art": b["art"],
                        "coat": b["features"], "inspired_by": b["inspired_by"], "silks": b["silks"]}
            continue
        h = sum(ord(ch) * (i + 1) for i, ch in enumerate(t))
        k = h % len(generic)
        while generic[k] in used:
            k = (k + 1) % len(generic)
        used.add(generic[k])
        zh, en = ADJ[h % len(ADJ)]
        g = manifest["generic"][generic[k]]
        art = ARTS[(h // 7) % len(ARTS)]                          # art style is stable per ticker as well
        looks[t] = {"company": "", "horse_zh": f"{zh}{t}", "horse_en": f"{en} {t}",
                    "sprite": f"generic_{generic[k]}" + ("" if art == "pixel" else f"_{art}"), "art": art,
                    "coat": f"{g['zh']}・{g['features']}", "inspired_by": "", "silks": None}
    return looks


def comment(h: dict, kf: list[dict]) -> str:
    labels = {f["id"]: f["horse"] for f in kf}
    have = {k: v for k, v in h["ability"].items() if v is not None}
    parts = []
    ran = [x for x in h["form"] if x is not None]
    if ran:
        top3 = sum(1 for x in ran if x <= 3)
        parts.append(f"近4期出場{len(ran)}次、{top3}次前三" if top3 else f"近4期出場{len(ran)}次、未進前三")
    else:
        parts.append("首次出賽")
    parts.append(f"跑法{h['style']}" if h["early_rank"] is not None else "跑法未知")
    if have:
        ab = sorted(have.items(), key=lambda kv: -kv[1])
        parts.append(f"強項「{labels[ab[0][0]]}」")
        if ab[-1][1] < 35:
            parts.append(f"弱項「{labels[ab[-1][0]]}」")
    miss = [labels[k] for k, v in h["ability"].items() if v is None]
    if miss:
        parts.append(f"缺資料：{'、'.join(miss)}（其餘項目重新配分計算）")
    return "，".join(parts)


_REC: dict = {}


def period_record(close, tickers, pool_key, dist, k, live_end) -> dict:
    """Real results of the finished period k: {ticker: (final rank, rank at the 1/3 point)} for the horses that
    were in that period's field.  Used for 近績 / 跑法 (a horse that was not in the field simply has no entry)."""
    key = (pool_key, dist, k)
    if key in _REC:
        return _REC[key]
    dates = close.index
    b = race_bounds(dates, dist, k, live_end)
    s_, e_, chain = b["s"], b["e"], b["chain"]
    rec: dict = {}
    if s_ >= 0 and e_ is not None and e_ > s_ and chain[1] >= 0 and chain[1] < s_:
        field, _, _ = select_field_prior(close, tickers, chain[1], s_)
        field = [t for t in field if pd.notna(close[t].iloc[e_])]
        if len(field) >= FIELD:
            final = ranks_of(close[field].iloc[e_] / close[field].iloc[s_] - 1)
            third = s_ + max(1, (e_ - s_) // 3)
            early = ranks_of(close[field].iloc[third] / close[field].iloc[s_] - 1)
            rec = {t: (int(final[t]), int(early[t])) for t in field}
    _REC[key] = rec
    return rec


def evaluate(close, ofac, kf, tickers, dist, b) -> dict | None:
    """Field + pre-race scores for one period b = race_bounds(...). The field is always the previous period's
    FIELD best performers (known before the start). None if the data does not allow it."""
    s, e, chain = b["s"], b["e"], b["chain"]
    if s < 0 or chain[1] < 0 or chain[1] >= s:
        return None
    field, _, _ = select_field_prior(close, tickers, chain[1], s)
    if e is not None:
        field = [t for t in field if pd.notna(close[t].iloc[e])]
    if len(field) < FIELD:
        return None
    ret = (close[field].iloc[e] / close[field].iloc[s] - 1) if e is not None else None
    eligible = [t for t in tickers if t in close.columns and pd.notna(close[t].iloc[s])]
    pct = ability_pct(ofac, kf, s, eligible)
    score = total_score(pct, weights_for(kf, dist)).fillna(50.0)      # newly listed horses: neutral score
    return {"field": field, "ret": ret, "pct": pct, "score": score, "ok": eligible}


def build_race(tickers, p, ofac, kf, sector_of, dist, k, live_end, manifest) -> dict | None:
    close = p["close"]
    dates = close.index
    b = race_bounds(dates, dist, k, live_end)
    ev = evaluate(close, ofac, kf, tickers, dist, b)
    if ev is None:
        return None
    s, e, chain = b["s"], b["e"], b["chain"]
    pred_order = list(ev["score"].loc[ev["field"]].sort_values(ascending=False).index)
    pk = ev_pool_id(tickers)
    recs = [period_record(close, tickers, pk, dist, k + j, live_end) for j in range(1, N_FORM + 1)]      # previous periods, newest first
    done = e is not None
    final_rank = ranks_of(ev["ret"].loc[pred_order]) if done else None
    looks = assign_looks(pred_order, manifest)

    horses = []
    for no, t in enumerate(pred_order, 1):
        form = [rec[t][0] if t in rec else None for rec in recs]               # real finishing ranks; None = did not run
        early = [rec[t][1] for rec in recs if t in rec][:3]
        early_avg = float(np.mean(early)) if early else None
        style, style_desc = style_label(early_avg) if early else ("未知", "首次出賽，還沒有跑法紀錄")
        pct = ev["pct"]
        ability = {f["id"]: (None if pd.isna(pct.loc[t, f["id"]]) else round(float(pct.loc[t, f["id"]]), 1)) for f in kf}
        raw = {f["id"]: (None if pd.isna(ofac[f["id"]].iloc[s][t]) else round(float(ofac[f["id"]].iloc[s][t]), 4)) for f in kf}
        total = round(float(ev["score"][t]), 1)
        h = {"no": no, "ticker": t, "sector": SECTOR_ZH.get(sector_of.get(t, ""), "—"), **looks[t],
             "form": form, "style": style, "style_desc": style_desc, "early_rank": round(early_avg, 1) if early else None, "n_ran": sum(x is not None for x in form),
             "ability": ability, "raw": raw, "total": total, "stars": stars(total), "pred": no,
             "mark": MARKS[no - 1] if no <= 5 else "", "price_start": round(float(close[t].iloc[s]), 2),
             "prior_ret": round(float(close[t].iloc[s] / close[t].iloc[chain[1]] - 1), 4),
             "eval": round(0.7 * total + 0.3 * form_score(form), 1)}
        if done:
            h.update({"price_end": round(float(close[t].iloc[e]), 2), "ret": round(float(ev["ret"][t]), 4),
                      "result": int(final_rank[t])})
        h["comment"] = comment(h, kf)
        horses.append(h)

    race = {"id": f"{ev_pool_id(tickers)}|{dist}|{dates[s].date()}", "k": k, "dist": dist, "label": b["label"],
            "status": "done" if done else "upcoming",
            "start": str(dates[s].date()), "end": str(dates[e].date()) if done else None,
            "days": (e - s) if done else None, "horses": horses,
            "momentum_window": [str(dates[chain[1]].date()), str(dates[s].date())],
            "weights": [{"id": f["id"], "w": round(float(w), 3)} for f, w in zip(kf, weights_for(kf, dist))],
            "coverage": {"pool_size": len(tickers), "with_prices": len(ev["ok"]),
                         "missing": sorted(t for t in tickers if t not in ev["ok"])[:40]}}
    od = odds_table(pd.Series({h["ticker"]: h["eval"] for h in horses}))                 # opening odds, from the card only
    for h in horses:
        r = od.loc[h["ticker"]]
        h.update({"p_win": round(float(r.p_win), 3), "p_place": round(float(r.p_place), 3),
                  "odds_win": float(r.odds_win), "odds_place": float(r.odds_place)})
    if done:
        cum = close.iloc[s:e + 1][pred_order]
        cum = cum / cum.iloc[0] - 1
        race["dates"] = [str(d.date()) for d in dates[s:e + 1]]
        for h in horses:
            h["path"] = [round(float(x), 4) for x in cum[h["ticker"]].values]
            del h["odds_win"], h["odds_place"], h["p_win"], h["p_place"]               # odds only matter while a race can still be bet on
    else:
        race["planned_end"] = str(last_session_on_or_before(b["end_ts"]).date()) if dist != "d" else str(b["end_ts"].date())
        add_live(race, close, dates, s, live_end, od, b["end_ts"])
    return race


def session_open_utc(d: pd.Timestamp) -> str:
    from zoneinfo import ZoneInfo
    return pd.Timestamp(d.year, d.month, d.day, 9, 30, tz=ZoneInfo("America/New_York")).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def add_live(race: dict, close: pd.DataFrame, dates: pd.DatetimeIndex, s: int, live_end: int, od: pd.DataFrame,
             end_ts: pd.Timestamp) -> None:
    """Standings so far and in-play prices for the period in progress."""
    names = [h["ticker"] for h in race["horses"]]
    day = int(live_end - s)
    sessions = sessions_between(dates[s], end_ts)
    total = max(len(sessions), day + 1) if day else max(len(sessions), 1)
    u = min(1.0, day / total)
    seg = close.iloc[s:live_end + 1][names]
    cum = seg / seg.iloc[0] - 1
    sig = np.log(close[names]).diff().iloc[max(1, live_end - 59):live_end + 1].std().to_numpy()
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 0.025)
    lo = live_odds(od.loc[names, "p_win"].to_numpy(), od.loc[names, "p_place"].to_numpy(), cum.iloc[-1].to_numpy(), sig, total - day, u)
    ranks = ranks_of(cum.iloc[-1])
    for h, (_, r) in zip(race["horses"], lo.iterrows()):
        t = h["ticker"]
        h.update({"live_path": [round(float(x), 4) for x in cum[t].values], "live_ret": round(float(cum[t].iloc[-1]), 4),
                  "live_rank": int(ranks[t]), "live_price": round(float(close[t].iloc[live_end]), 2),
                  "odds_win_open": h["odds_win"], "odds_place_open": h["odds_place"],
                  "odds_win": float(r.odds_win), "odds_place": float(r.odds_place),
                  "p_win": round(float(r.p_win), 3), "p_place": round(float(r.p_place), 3)})
    n_last = max(0, math.ceil(CLOSE_U * total) - 1)               # last number of finished sessions at which a bet is still taken
    close_session = sessions[min(n_last, len(sessions) - 1)] if sessions else end_ts
    race["live"] = {"as_of": str(dates[live_end].date()), "day": day, "total_days": total, "u": round(u, 3),
                    "tau": round(tau(u), 3), "betting_open": bool(day <= n_last), "close_u": CLOSE_U,
                    "close_at": session_open_utc(close_session), "first_session": str(sessions[0].date()) if sessions else None,
                    "next_first_session": str(next_session(sessions[-1] if sessions else end_ts).date()),      # when the next period starts playing
                    "dates": [str(d.date()) for d in dates[s:live_end + 1]]}


_POOL_ID: dict[int, str] = {}


def ev_pool_id(tickers) -> str:
    return _POOL_ID[id(tickers)]


def validate(tickers, p, ofac, kf, dist, live_end: int) -> dict:
    """Replay the same method over earlier finished races (consecutive, non-overlapping windows)."""
    close = p["close"]
    dates = close.index
    w = weights_for(kf, dist)
    rhos, top1, top3 = [], [], []
    for k in range(1, N_VALID[dist] + 1):
        b = race_bounds(dates, dist, k, live_end)
        if b["s"] < 300 or b["e"] is None or b["e"] <= b["s"]:      # need >1y of history for the factors
            break
        ev = evaluate(close, ofac, kf, tickers, dist, b)
        if ev is None:
            continue
        score = ev["score"].loc[ev["field"]].dropna()
        if len(score) < FIELD:
            continue
        actual = ranks_of(ev["ret"].loc[score.index])
        pred = ranks_of(score)
        rhos.append(float(np.corrcoef(pred, actual)[0, 1]))
        order = pred.sort_values()
        top1.append(float(actual[order.index[0]]))
        top3.append(float(actual[order.index[:3]].mean()))
    n = len(rhos)
    if n < 3:
        return {"n": n}
    r = np.array(rhos)
    return {"n": n, "mean_rho": round(float(r.mean()), 3), "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(n))), 2),
            "top1_avg_rank": round(float(np.mean(top1)), 2), "top3_avg_rank": round(float(np.mean(top3)), 2),
            "first": str(dates[race_bounds(dates, dist, n, live_end)["s"]].date()),
            "last": str(dates[race_bounds(dates, dist, 1, live_end)["e"]].date())}


def main() -> None:
    kf_doc = json.loads(KEY_FACTORS.read_text())
    kf = [{**f, **F.horse_term(f["id"], f["reversed"])} for f in kf_doc["key_factors"]]
    df = dl.load_long("2009-01-01")
    p = dl.panels(df)
    fac = F.compute_all(p, dl.earnings())
    ofac = {f["id"]: (-fac[f["id"]] if f["reversed"] else fac[f["id"]]) for f in kf}     # higher = better, always
    sector_of = dl.sectors(df)
    manifest = json.loads(MANIFEST.read_text())
    spx, ndx = set(dl.sp500_members()), set(dl.ndx_members())
    spx.discard("GOOG"); ndx.discard("GOOG")           # same company as GOOGL
    pools = {"spx": sorted(spx), "ndx": sorted(ndx), "all": sorted(spx | ndx)}
    close = p["close"]
    pools_out, settled = {}, {}
    for pk, pt in pools.items():
        _POOL_ID[id(pt)] = pk
        cov = close.reindex(columns=[t for t in pt if t in close.columns]).notna().mean(axis=1)
        live_end = int(close.index.get_loc(cov[cov >= 0.9].index[-1]))
        pools_out[pk] = {}
        for dk in DISTS:
            races = [r for k in range(N_HIST + 1) if (r := build_race(pt, p, ofac, kf, sector_of, dk, k, live_end, manifest)) is not None]
            pools_out[pk][dk] = {"races": races, "validation": validate(pt, p, ofac, kf, dk, live_end)}
            for r in races:                                  # everything a stored bet needs to be settled later
                if r["status"] == "done":
                    settled[r["id"]] = {"end": r["end"], "result": {h["ticker"]: h["result"] for h in r["horses"]}}
    asof = pools_out["all"]["d"]["races"][1]["end"]
    out = {"asof": asof, "key_factors": kf, "study": {k: kf_doc[k] for k in ("universe_size", "period", "months")},
           "ndx_status": dl.ndx_status(), "pools": pools_out, "settled": settled}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB), asof {asof}, NDX list: {out['ndx_status']['source']}")
    for dk in DISTS:
        r0 = pools_out["all"][dk]["races"][0]
        L = r0["live"]
        print("all", dk, r0["label"], r0["start"], "->", r0["planned_end"], f"day {L['day']}/{L['total_days']}", "close_at", L["close_at"],
              "|", ", ".join(f"{h['ticker']}@{h['odds_win']}" for h in r0["horses"][:3]), "| rho", pools_out["all"][dk]["validation"].get("mean_rho"))


if __name__ == "__main__":
    main()
