"""產業評分 → 重點產業 → 個股多因子評分 → 選出每市場前 N 檔。"""
from __future__ import annotations
from collections import defaultdict

from .news import sector_news_score
from .utils import avg, isnum, lin, median, wavg
from .valuation import growth_estimate, peer_stats, value_stock

FACTORS = ["sector", "leader", "quality", "growth", "valuation", "momentum", "news"]


def rank_scale(vals: dict) -> dict:
    """把 {key: value} 轉成 0–100 的百分位排名（缺值為 None）。"""
    items = sorted((k for k, v in vals.items() if isnum(v)), key=lambda k: vals[k])
    n = len(items)
    out = {k: None for k in vals}
    for i, k in enumerate(items):
        out[k] = 100.0 * i / (n - 1) if n > 1 else 50.0
    return out


def score_sectors(uni, scan_res, feats, funds, history, p) -> dict:
    """回傳 {market: {sector_id: {...score...}}}"""
    sw = p["sector_weights"]
    res = {"US": {}, "TW": {}}
    for m in res:
        raw = {}
        for s in uni["sectors"]:
            members = [t for t, st in uni["stocks"].items() if st["sector"] == s["id"] and st["market"] == m and t in feats]
            if len(members) < 2:
                continue
            nsc = sector_news_score(scan_res, s["id"], m, history)
            mom = avg([avg([feats[t].get("rel_3m"), feats[t].get("rel_6m")]) for t in members])
            r1m = avg([feats[t].get("ret_1m") for t in members])
            breadth = avg([1.0 if feats[t]["last"] > feats[t]["ma50"] else 0.0 for t in members])
            fund = avg([avg([funds[t].get("rev_growth"), funds[t].get("eps_growth")]) for t in members if t in funds])
            raw[s["id"]] = {"name": s["name"], "members": members, "news": nsc, "mom": mom, "ret_1m": r1m, "breadth": breadth, "fund_growth": fund}
        heat_rank = rank_scale({k: v["news"]["heat"] + 3 * max(0, v["news"]["sent"]) for k, v in raw.items()})
        mom_rank = rank_scale({k: wavg([(v["mom"], 0.7), (v["breadth"], 0.3)]) for k, v in raw.items()})
        fund_rank = rank_scale({k: v["fund_growth"] for k, v in raw.items()})
        for k, v in raw.items():
            comps = [(heat_rank[k], sw["news"]), (mom_rank[k], sw["momentum"]), (fund_rank[k], sw["fundamental"])]
            v["score"] = wavg(comps)
            v["heat_rank_score"], v["mom_rank_score"], v["fund_rank_score"] = heat_rank[k], mom_rank[k], fund_rank[k]
            tags = []
            if heat_rank[k] is not None and heat_rank[k] >= 66:
                tags.append("市場焦點")
            if (fund_rank[k] or 0) >= 66 and (mom_rank[k] or 100) < 66:
                tags.append("潛力（基本面強、股價尚未過熱）")
            elif (fund_rank[k] or 0) >= 66 and (heat_rank[k] or 0) >= 50:
                tags.append("成長動能")
            if v["news"]["accel"] is not None and v["news"]["accel"] > 0.5:
                tags.append("新聞升溫")
            v["tags"] = tags
        ordered = sorted(raw, key=lambda k: -(raw[k]["score"] or 0))
        for i, k in enumerate(ordered, 1):
            raw[k]["rank"] = i
        res[m] = raw
    return res


def score_stocks(uni, feats, funds, sec_scores, scan_res, p) -> dict:
    """為宇宙內每檔股票計分並估值。回傳 {ticker: row}。"""
    stocks = uni["stocks"]
    # 先算每檔的成長估計，供同業統計
    for t, f in funds.items():
        pf = feats.get(t)
        if pf:
            f["_g"] = growth_estimate(f, p)
    by_group = defaultdict(list)
    for t, st in stocks.items():
        if t in funds and t in feats:
            by_group[(st["sector"], st["market"])].append(t)
    by_market = defaultdict(list)
    for t in funds:
        if t in stocks and t in feats:
            by_market[stocks[t]["market"]].append(funds[t])

    rows = {}
    for t, st in stocks.items():
        if t not in feats or t not in funds:
            continue
        pf, f, m = feats[t], funds[t], st["market"]
        grp = by_group[(st["sector"], m)]
        peers = peer_stats([funds[x] for x in grp if x != t], p)
        if peers["n"] < 3:  # 同業樣本不足 → 用整個市場中位數
            peers = peer_stats([x for x in by_market[m] if x is not f], p)
        val = value_stock(pf["last"], f, pf, m, peers, p)
        # 龍頭分數：同產業同市場市值排名
        caps = sorted(((funds[x].get("market_cap") or 0, x) for x in grp), reverse=True)
        pos = [x for _, x in caps].index(t) if t in [x for _, x in caps] else len(caps) - 1
        leader = 100.0 if len(caps) <= 1 else 100.0 * (1 - pos / (len(caps) - 1))
        quality = avg([lin(f.get("roe"), 0, 0.25), lin(f.get("op_margin"), 0, 0.30), lin(f.get("gross_margin"), 0.10, 0.60),
                       lin(f.get("fcf_margin"), 0, 0.20), lin(f.get("de"), 2.0, 0.0), lin(f.get("current_ratio"), 0.8, 2.0)], min_n=3)
        growth = avg([lin(f.get("rev_growth"), -0.05, 0.30), lin(f.get("eps_growth"), -0.10, 0.40)])
        valscore = None
        if val:
            valscore = wavg([(lin(val["upside"], -0.10, 0.40), 0.6), (lin(f.get("peg"), 3.0, 0.8), 0.4)])
            if f.get("pe_fwd") and f["pe_fwd"] > 60 and valscore is not None:
                valscore *= 0.8
        mom = wavg([(lin(pf.get("rel_3m"), -0.20, 0.30), 0.4), (lin(pf.get("rel_6m"), -0.20, 0.40), 0.3),
                    (100.0 if pf["last"] > pf["ma50"] and pf.get("ma200") and pf["ma50"] > pf["ma200"] else 50.0 if pf["last"] > pf["ma50"] else 20.0, 0.3)])
        if mom is not None and pf.get("dist200") is not None and pf["dist200"] > 0.5:
            mom = max(0.0, mom - 15)  # 過熱懲罰
        nd = scan_res["tickers"].get(t)
        news = 40.0 if not nd else min(100.0, 50.0 + 12 * min(nd["heat"], 4) ** 0.7 + 25 * nd["sent"])
        ss = sec_scores.get(m, {}).get(st["sector"])
        scores = {"sector": ss["score"] if ss else None, "leader": leader, "quality": quality, "growth": growth,
                  "valuation": valscore, "momentum": mom, "news": news}
        comp = wavg([(scores[k], p["factor_weights"][k]) for k in FACTORS])
        rows[t] = {"ticker": t, "name": f.get("name") or st["name"], "zh_name": st["name"], "market": m, "sector": st["sector"],
                   "price": pf["last"], "as_of": pf["date"], "scores": scores, "composite": comp, "val": val, "f": f, "pf": pf,
                   "rank_in_sector": (pos + 1), "n_in_sector": len(caps), "sector_rank": ss["rank"] if ss else None,
                   "news": nd, "dual_of": st.get("dual_of")}
    return rows


def eligible(r: dict, p: dict) -> tuple[bool, str]:
    s, m = p["selection"], r["market"]
    f, pf, sc = r["f"], r["pf"], r["scores"]
    if r["val"] is None:
        return False, "資料不足，無法估值"
    if r["val"]["spread"] > s["max_method_spread"]:
        return False, "各估值方法分歧過大"
    if (f.get("market_cap") or 0) < s["min_market_cap"][m]:
        return False, "市值過小"
    if pf["turnover"] < s["min_avg_turnover"][m]:
        return False, "流動性不足"
    if r["val"]["upside"] < s["min_upside"]:
        return False, "目標價上檔空間不足"
    if sc["quality"] is None or sc["quality"] < s["min_quality_score"]:
        return False, "財務品質不足"
    lq = s["leader_or_quality"]
    if not (sc["leader"] >= lq["leader_score"] or sc["quality"] >= lq["quality_score"]):
        return False, "非龍頭且財務績優度不足"
    if (f.get("eps_ttm_px") or 0) <= 0 and (f.get("eps_fwd_px") or 0) <= 0:
        return False, "獲利為負"
    return True, ""


def pick(rows: dict, sec_scores: dict, p: dict) -> dict:
    """依序挑選：先台股再美股（美股略過已選入台股的雙重上市標的）。回傳 {market: [rows]}，另附 notes。"""
    s = p["selection"]
    picked: dict[str, list] = {"TW": [], "US": []}
    notes: dict[str, list] = {"TW": [], "US": []}
    chosen = set()
    for m in ("TW", "US"):
        ranked_sectors = sorted(sec_scores[m], key=lambda k: -(sec_scores[m][k]["score"] or 0))
        focus = ranked_sectors[: s["focus_sectors_per_market"]]
        for pool_name, sectors in (("focus", focus), ("relaxed", ranked_sectors[s["focus_sectors_per_market"]:])):
            if len(picked[m]) >= s["picks_per_market"]:
                break
            cands = sorted((r for r in rows.values() if r["market"] == m and r["sector"] in sectors and r["composite"] is not None),
                           key=lambda r: -r["composite"])
            cnt = defaultdict(int)
            for r in picked[m]:
                cnt[r["sector"]] += 1
            for r in cands:
                if len(picked[m]) >= s["picks_per_market"]:
                    break
                ok, why = eligible(r, p)
                if not ok:
                    continue
                if cnt[r["sector"]] >= s["max_per_sector"]:
                    continue
                if r["dual_of"] in chosen or r["ticker"] in chosen:
                    continue
                r["pick_pool"] = pool_name
                picked[m].append(r)
                cnt[r["sector"]] += 1
                chosen.add(r["ticker"])
            if pool_name == "relaxed" and picked[m]:
                if any(r.get("pick_pool") == "relaxed" for r in picked[m]):
                    notes[m].append("重點產業內合格標的不足，已放寬至排名較後的產業補足名額。")
        for i, r in enumerate(picked[m], 1):
            r["pick_rank"] = i
    return {"picks": picked, "notes": notes}
