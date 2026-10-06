"""新聞掃描：把標題/摘要對應到產業與個股，計算熱度與情緒。"""
from __future__ import annotations
import datetime as dt
import math
import re
from collections import defaultdict


def _compile(words):
    ascii_w = [re.escape(w) for w in words if w.isascii()]
    cjk_w = [re.escape(w) for w in words if not w.isascii()]
    pats = []
    if ascii_w:
        pats.append(r"(?<![A-Za-z0-9])(?:" + "|".join(ascii_w) + r")(?![A-Za-z0-9])")
    if cjk_w:
        pats.append("(?:" + "|".join(cjk_w) + ")")
    return re.compile("|".join(pats), re.I) if pats else None


def recency_weight(pub, now) -> float:
    if pub is None:
        return 0.6
    h = (now - pub).total_seconds() / 3600
    if h < 0:
        return 1.0
    return 1.0 if h <= 12 else 0.7 if h <= 24 else 0.4 if h <= 48 else 0.0


def sentiment(text: str, pos, neg) -> float:
    p = len(pos.findall(text)) if pos else 0
    n = len(neg.findall(text)) if neg else 0
    return (p - n) / (p + n + 1.0)


def scan(items: list[dict], uni: dict, sources: dict, now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    pos, neg = _compile(sources["positive_words"]), _compile(sources["negative_words"])
    sec_re = {s["id"]: _compile(s["keywords_zh"] + s["keywords_en"]) for s in uni["sectors"]}
    tk_re = {}
    for t, st in uni["stocks"].items():
        names = [st["name"]] + st["aliases"]
        if len(t) >= 4 and t.isascii() and not t[0].isdigit():
            names.append(t)
        if t[0].isdigit():
            names.append(t.split(".")[0])
        names = [n for n in names if n and len(n) >= 2]
        pats = [_compile(names)] if names else []
        if not t[0].isdigit():
            pats.append(re.compile(r"\$" + re.escape(t) + r"\b"))
        tk_re[t] = pats

    sector = {s["id"]: {m: {"heat": 0.0, "n": 0, "sent": 0.0, "heads": []} for m in ("US", "TW")} for s in uni["sectors"]}
    tickers = defaultdict(lambda: {"heat": 0.0, "n": 0, "sent": 0.0, "heads": []})
    used = 0
    for it in items:
        rw = recency_weight(it.get("published"), now)
        if rw <= 0:
            continue
        used += 1
        w = rw * it.get("weight", 1.0)
        text = it["title"] + " " + it.get("summary", "")
        s = sentiment(it["title"], pos, neg)
        rec = {"title": it["title"], "link": it.get("link", ""), "source": it["source"], "sent": round(s, 2), "w": w}
        for sid, rx in sec_re.items():
            if rx is None:
                continue
            # 標題命中算 1，僅摘要命中算 0.5
            hit = 1.0 if rx.search(it["title"]) else 0.5 if rx.search(text) else 0.0
            if hit:
                d = sector[sid][it["market"]]
                d["heat"] += w * hit
                d["n"] += 1
                d["sent"] += s * w * hit
                d["heads"].append(dict(rec, score=w * hit))
        for t, pats in tk_re.items():
            if any(p.search(it["title"]) for p in pats if p):
                d = tickers[t]
                d["heat"] += w
                d["n"] += 1
                d["sent"] += s * w
                d["heads"].append(rec)
    for sid in sector:
        for m in sector[sid]:
            d = sector[sid][m]
            d["sent"] = d["sent"] / d["heat"] if d["heat"] else 0.0
            d["heads"] = sorted(d["heads"], key=lambda x: -x["score"])[:8]
    for t, d in tickers.items():
        d["sent"] = d["sent"] / d["heat"] if d["heat"] else 0.0
        d["heads"] = sorted(d["heads"], key=lambda x: -x["w"])[:5]
    return {"sector": sector, "tickers": dict(tickers), "n_items": len(items), "n_used": used}


def sector_news_score(scan_res: dict, sid: str, market: str, history: dict | None = None) -> dict:
    """該市場的產業新聞分數（0–100）：本市場熱度 + 0.5×另一市場熱度（全球題材外溢）。
    若有歷史資料，再加入『相對近 20 日均值的升溫幅度』。"""
    other = "TW" if market == "US" else "US"
    d, o = scan_res["sector"][sid][market], scan_res["sector"][sid][other]
    heat = d["heat"] + 0.5 * o["heat"]
    sent = (d["sent"] * d["heat"] + 0.5 * o["sent"] * o["heat"]) / heat if heat else 0.0
    accel = None
    if history:
        past = [h.get(market, {}).get(sid) for h in history.values()]
        past = [p for p in past if p is not None][-20:]
        if len(past) >= 5:
            base = sum(past) / len(past)
            accel = heat / base - 1 if base > 0 else None
    return {"heat": heat, "sent": sent, "accel": accel, "n": d["n"] + o["n"]}
