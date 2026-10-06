"""每日流程編排。"""
from __future__ import annotations
import datetime as dt
import json

import pandas as pd

from . import config as C
from . import llm
from .features import norm_fundamentals, price_features
from .journal import candidate_record, pick_record, write_journal
from .news import scan
from .selection import eligible, pick, score_sectors, score_stocks
from .utils import market_of


def _index_stats(prices, uni):
    out = []
    for ix in uni["indices"]:
        df = prices.get(ix["t"])
        if df is None or len(df) < 70:
            continue
        c = df["Close"]
        out.append({"name": ix["name"], "last": float(c.iloc[-1]), "ret_1d": float(c.iloc[-1] / c.iloc[-2] - 1),
                    "ret_1m": float(c.iloc[-1] / c.iloc[-22] - 1), "ret_3m": float(c.iloc[-1] / c.iloc[-64] - 1)})
    return out


def run_daily(provider, force: bool = False, asof: str | None = None) -> str | None:
    uni, params, src = C.load_universe(), C.load_params(), C.load_sources()
    tickers = list(uni["stocks"]) + [i["t"] for i in uni["indices"]] + list(uni["benchmarks"].values())
    tickers = list(dict.fromkeys(tickers))
    print(f"[daily] 抓取 {len(tickers)} 檔價格…")
    prices = provider.prices(tickers)
    if asof:
        cut = pd.Timestamp(asof)
        prices = {t: d[d.index <= cut] for t, d in prices.items()}
    bdates = [prices[b].index[-1] for b in uni["benchmarks"].values() if b in prices and len(prices[b])]
    if not bdates:
        raise SystemExit("取不到大盤價格，無法決定資料基準日（請檢查網路/資料來源）。")
    date = str(max(bdates).date())
    jpath = C.path("journal") / f"{date}.md"
    if jpath.exists() and not force:
        print(f"[daily] {date} 的日誌已存在，略過（用 --force 重跑）。")
        return None

    feats = {}
    for t, st in uni["stocks"].items():
        b = prices.get(uni["benchmarks"][st["market"]])
        f = price_features(prices.get(t), b)
        if f and f["date"] >= str((max(bdates) - pd.Timedelta(days=7)).date()):  # 排除停牌/下市（資料落後超過 7 天）
            feats[t] = f
    print(f"[daily] 有效價格 {len(feats)}/{len(uni['stocks'])} 檔；抓取基本面…")
    funds = {}
    for t in feats:
        funds[t] = norm_fundamentals(provider.fundamentals(t), feats[t]["last"], "TWD" if market_of(t) == "TW" else "USD")
    print("[daily] 掃描新聞…")
    items = provider.news(src)
    scan_res = scan(items, uni, src)

    data = C.path("data")
    nh_path = data / "news_history.json"
    nh = C.load_json(nh_path, {})
    sec_scores = score_sectors(uni, scan_res, feats, funds, {d: h for d, h in nh.items() if d < date}, params)
    rows = score_stocks(uni, feats, funds, sec_scores, scan_res, params)
    sel = pick(rows, sec_scores, params)
    picks, notes = sel["picks"], sel["notes"]
    if not any(picks.values()):
        print("[daily] 今日沒有任何標的通過篩選；仍會寫入日誌說明。")

    # 前期名單、績效摘要、參數歷史
    prev = None
    for f in sorted((data / "picks").glob("*.json") if (data / "picks").exists() else [], reverse=True):
        if f.stem < date:
            pj = C.load_json(f)
            prev = {"date": f.stem, "picks": pj["picks"]}
            break
    perf = C.load_json(data / "performance.json")
    ctx = {
        "params": params, "uni": uni, "sec_scores": sec_scores, "picks": picks, "notes": notes, "rows": rows, "scan": scan_res,
        "fetch_log": provider.fetch_log, "provider": provider.name, "indices": _index_stats(prices, uni), "prev": prev,
        "perf_summary": perf["summary"] if perf and perf.get("summary") else None,
        "param_history": C.load_json(C.history_path(), []), "elig": lambda r: eligible(r, params),
    }
    # 選用 LLM 評論
    if llm.available():
        ctx["llm"] = llm.commentary({
            "date": date, "indices": ctx["indices"],
            "top_sectors": {m: [{"name": v["name"], "score": v["score"], "tags": v["tags"], "headlines": [h["title"] for h in scan_res["sector"][k][m]["heads"][:3]]}
                                for k, v in sorted(sec_scores[m].items(), key=lambda kv: kv[1]["rank"])[:5]] for m in sec_scores},
            "picks": [{"ticker": r["ticker"], "name": r["zh_name"], "sector": r["sector"], "upside": r["val"]["upside"], "scores": r["scores"],
                       "roe": r["f"].get("roe"), "rev_growth": r["f"].get("rev_growth"), "pe_fwd": r["f"].get("pe_fwd")} for m in picks for r in picks[m]],
        })

    md = write_journal(date, ctx)
    jpath.write_text(md, encoding="utf-8")
    pdir = data / "picks"
    pdir.mkdir(exist_ok=True)
    C.save_json(pdir / f"{date}.json", {
        "date": date, "params_version": params["version"], "provider": provider.name, "llm": bool(ctx.get("llm")),
        "picks": [pick_record(r) for m in ("TW", "US") for r in picks[m]],
        "focus_sectors": {m: [k for k, v in sorted(sec_scores[m].items(), key=lambda kv: kv[1]["rank"])[: params["selection"]["focus_sectors_per_market"]]] for m in sec_scores},
    })
    cdir = data / "candidates"
    cdir.mkdir(exist_ok=True)
    picked = {r["ticker"] for m in picks for r in picks[m]}
    focus = {m: set(k for k, v in sec_scores[m].items() if v["rank"] <= params["selection"]["focus_sectors_per_market"]) for m in sec_scores}
    C.save_json(cdir / f"{date}.json", {"date": date, "rows": [candidate_record(r, picked, focus) for r in rows.values() if r["composite"] is not None]})
    nh[date] = {m: {k: scan_res["sector"][k][m]["heat"] + 0.5 * scan_res["sector"][k]["TW" if m == "US" else "US"]["heat"] for k in scan_res["sector"]} for m in ("US", "TW")}
    C.save_json(nh_path, dict(sorted(nh.items())[-120:]))
    print(f"[daily] 完成：{jpath}（台股 {len(picks['TW'])} 檔、美股 {len(picks['US'])} 檔）")
    return date
