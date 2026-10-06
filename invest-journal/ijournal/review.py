"""定期檢討報告：驗證過往推薦與目標價，歸因，提出改進並（可選）回饋參數。"""
from __future__ import annotations
import datetime as dt

import pandas as pd

from . import config as C
from .selection import FACTORS
from .tracker import HORIZONS
from .tuning import apply_tuning
from .utils import avg, isnum, num, pct

METHODS = ["pe", "peg", "dcf", "consensus"]


def _add(lst, v):
    if isnum(v):
        lst.append(v)


def spearman(x: pd.Series, y: pd.Series):
    m = x.notna() & y.notna()
    return x[m].rank().corr(y[m].rank()) if m.sum() >= 3 else None


def diagnostics(min_obs: int, horizon: int, params: dict) -> dict:
    data = C.path("data")
    pool = C.load_json(data / "pool_returns.json", {})
    perf = C.load_json(data / "performance.json", {"positions": []})
    factor_ics = {k: [] for k in FACTORS}
    method_ics = {k: [] for k in METHODS}
    comp_ic, ic_dates, sel_alpha, pool_alpha, focus_alpha = [], 0, [], [], []
    mdates = 0
    for f in sorted((data / "candidates").glob("*.json")) if (data / "candidates").exists() else []:
        cj = C.load_json(f)
        fw = pool.get(cj["date"], {})
        rows = [dict(r, a=fw[r["ticker"]].get(f"a{horizon}")) for r in cj["rows"] if r["ticker"] in fw and isnum(fw[r["ticker"]].get(f"a{horizon}"))]
        if len(rows) < min_obs:
            continue
        ic_dates += 1
        a = pd.Series([r["a"] for r in rows])
        for k in FACTORS:
            s = pd.Series([r["scores"].get(k) for r in rows], dtype="float64")
            if s.notna().sum() >= min_obs and s.nunique() > 1:
                _add(factor_ics[k], spearman(s, a))
        c = pd.Series([r["composite"] for r in rows], dtype="float64")
        if c.nunique() > 1:
            _add(comp_ic, spearman(c, a))
        got_m = False
        for k in METHODS:
            s = pd.Series([r["method_upside"].get(k) for r in rows], dtype="float64")
            if s.notna().sum() >= min_obs and s.nunique() > 1:
                _add(method_ics[k], spearman(s, a))
                got_m = True
        mdates += got_m
        pool_alpha += [r["a"] for r in rows]
        sel_alpha += [r["a"] for r in rows if r["picked"]]
        focus_alpha += [r["a"] for r in rows if r["in_focus"]]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    pos = [p for p in perf["positions"] if p.get(f"r{horizon}") is not None]
    exp = lambda p: (1 + p["target_upside"]) ** (horizon / 250) - 1
    bias = mean([p[f"r{horizon}"] - exp(p) for p in pos])
    return {"ic_dates": ic_dates, "method_ic_dates": mdates, "factor_ic": {k: mean(v) for k, v in factor_ics.items()},
            "method_ic": {k: mean(v) for k, v in method_ics.items()}, "comp_ic": mean(comp_ic), "n_matured": len(pos), "bias": bias,
            "sel_alpha": mean(sel_alpha), "pool_alpha": mean(pool_alpha), "focus_alpha": mean(focus_alpha), "n_sel": len(sel_alpha), "n_pool": len(pool_alpha),
            "perf": perf, "horizon": horizon}


def run_review(period: str, tune: bool = False, date: str | None = None, min_dates: int | None = None) -> str:
    params = C.load_params()
    sname = {x["id"]: x["name"] for x in C.load_universe()["sectors"]}
    t = params["tuning"]
    date = date or str(dt.date.today())
    H = t["horizon_days"]
    d = diagnostics(t["min_obs_per_date"], H, params)
    perf = d["perf"]
    pos = perf.get("positions", [])
    L, A = [], None
    A = L.append
    label = "週" if period == "weekly" else "月"
    A(f"# {label}度檢討報告 {date}\n")
    A(f"> 追蹤中推薦 {len(pos)} 筆，涵蓋 {len({p['date'] for p in pos})} 個推薦日。參數版本 v{params['version']}。僅供研究，不構成投資建議。\n")
    if not pos:
        A("目前尚無推薦紀錄可供檢討。\n")
    # 1 總覽
    A("## 一、績效總覽（相對大盤）\n")
    A("基準：台股對加權指數、美股對 S&P 500。進場價為推薦日收盤價（未計交易成本與稅費，實際績效會更低）。\n")
    A("| 範圍 | 持有期 | 到期檔數 | 平均報酬 | 平均超額報酬 | 勝率（絕對） | 勝率（贏大盤） |")
    A("|---|---|---|---|---|---|---|")
    for name, summ in (("全部", perf.get("summary", {})), ("台股", perf.get("by_market", {}).get("TW", {})), ("美股", perf.get("by_market", {}).get("US", {}))):
        for h in HORIZONS:
            s = summ.get(str(h))
            if s and s["n"]:
                A(f"| {name} | {h} 日 | {s['n']} | {pct(s['avg_ret'], 1, True)} | {pct(s['avg_alpha'], 1, True)} | {pct(s['win'], 0)} | {pct(s['win_alpha'], 0)} |")
    A("")
    if pos:
        A("**各期推薦批次迄今表現**\n")
        A("| 推薦日 | 檔數 | 已持有交易日 | 平均報酬 | 平均超額報酬 |")
        A("|---|---|---|---|---|")
        for c in sorted(perf["cohorts"], key=lambda x: x["date"], reverse=True)[:15]:
            A(f"| {c['date']} | {c['n']} | {c['n_days']} | {pct(c['avg_ret'], 1, True)} | {pct(c['avg_alpha'], 1, True)} |")
        A("")
        A("## 二、最佳與最差個股（迄今）\n")
        srt = sorted([p for p in pos if p.get("alast") is not None], key=lambda p: p["alast"], reverse=True)
        A("| | 推薦日 | 代號 | 名稱 | 產業 | 迄今報酬 | 超額 | 目標價進度 |")
        A("|---|---|---|---|---|---|---|---|")
        for tag, grp in (("最佳", srt[:5]), ("最差", srt[-5:])):
            for p in grp:
                prog = (p["last_price"] - p["entry"]) / (p["target"] - p["entry"]) if p["target"] != p["entry"] else None
                A(f"| {tag} | {p['date']} | {p['ticker']} | {p['name']} | {sname.get(p['sector'], p['sector'])} | {pct(p['rlast'], 1, True)} | {pct(p['alast'], 1, True)} | {pct(prog, 0)} |")
        A("")
    # 3 選股效果
    A(f"## 三、選股有效性（{H} 個交易日超額報酬）\n")
    if d["n_sel"]:
        A(f"- 入選個股平均超額報酬：{pct(d['sel_alpha'], 2, True)}（{d['n_sel']} 筆）")
        A(f"- 重點產業候選池平均：{pct(d['focus_alpha'], 2, True)}；全宇宙平均：{pct(d['pool_alpha'], 2, True)}（{d['n_pool']} 筆）")
        A(f"- 選股附加價值（入選 − 全宇宙）：{pct((d['sel_alpha'] or 0) - (d['pool_alpha'] or 0), 2, True)}；綜合分與後續超額報酬的 Rank IC：{num(d['comp_ic'], 3)}\n")
    else:
        A(f"尚無已滿 {H} 個交易日的樣本。\n")
    # 4 因子
    A("## 四、因子與估值方法的預測力（日均 Rank IC）\n")
    A(f"IC＝每日橫斷面中，因子分數與後續 {H} 日超額報酬的 Spearman 相關，再對日期平均；有效日期數：因子 {d['ic_dates']}、估值方法 {d['method_ic_dates']}。IC>0 代表該因子有預測力。\n")
    A("| 因子 | 目前權重 | IC |")
    A("|---|---|---|")
    for k in FACTORS:
        A(f"| {k} | {params['factor_weights'][k]:.3f} | {num(d['factor_ic'][k], 3)} |")
    A("\n| 估值方法 | 目前權重 | IC（隱含上檔 vs 實際超額報酬） |")
    A("|---|---|---|")
    for k in METHODS:
        A(f"| {k} | {params['method_weights'][k]:.3f} | {num(d['method_ic'][k], 3)} |")
    A("")
    # 5 目標價
    A("## 五、目標價檢驗\n")
    mp = [p for p in pos if p.get("n_days", 0) >= 5]
    if mp:
        sh = sum(1 for p in mp if p["stop_hit"])
        th = sum(1 for p in mp if p["target_hit"])
        A(f"- 觸及 base 目標價：{th}/{len(mp)}；觸及參考停損：{sh}/{len(mp)}")
        A(f"- 目標價『隱含路徑』檢驗：把 12 個月上檔空間按複利攤到 {H} 日作為預期報酬，實際 − 預期的平均偏差 = **{pct(d['bias'], 2, True)}**（{d['n_matured']} 筆）。")
        A("  - 偏差為負 → 目標價系統性偏樂觀；為正 → 偏保守。需注意此期間大盤整體漲跌也會影響此數字。")
        lr = [p for p in mp if p.get("ret_vs_expected") is not None]
        A(f"- 迄今（不分持有期）平均 實際−預期：{pct(avg([p['ret_vs_expected'] for p in lr]), 2, True)}\n")
    else:
        A("樣本尚不足。12 個月目標價真正的驗證要等到一年後；現階段以『隱含路徑』與短期超額報酬作為先行指標。\n")
    # 6 產業
    if pos:
        A("## 六、產業別表現（迄今超額報酬）\n")
        A("| 產業 | 筆數 | 平均超額 | 平均報酬 |")
        A("|---|---|---|---|")
        by = {}
        for p in pos:
            by.setdefault(p["sector"], []).append(p)
        for sct, ps in sorted(by.items(), key=lambda kv: -(avg([x["alast"] for x in kv[1]]) or -9)):
            A(f"| {sname.get(sct, sct)} | {len(ps)} | {pct(avg([x['alast'] for x in ps]), 1, True)} | {pct(avg([x['rlast'] for x in ps]), 1, True)} |")
        A("")
    # 7 診斷
    A("## 七、診斷與改進行動\n")
    acts = []
    if not d["n_sel"] and not pos:
        acts.append("資料累積中：持續每日推薦，待有到期樣本後才進行實質調整。")
    else:
        if d["n_matured"] < t["min_matured_picks"]:
            acts.append(f"到期樣本僅 {d['n_matured']} 筆（< {t['min_matured_picks']}），任何結論統計上都不可靠；**不調整目標價折減係數**，繼續累積。")
        if d["ic_dates"] < (min_dates or t["min_dates"]):
            acts.append(f"可計算 IC 的日期僅 {d['ic_dates']} 天（< {min_dates or t['min_dates']}），因子權重暫不調整。")
        if isnum(d["comp_ic"]) and d["ic_dates"] >= (min_dates or t["min_dates"]):
            acts.append("綜合分 IC 為正，整體選股邏輯有預測力，維持框架。" if d["comp_ic"] > 0.02 else
                        "綜合分 IC 接近零或為負，目前打分框架尚未展現預測力：檢查各因子 IC，考慮移除長期 IC<0 的因子或擴大樣本期後再判斷。")
        if isnum(d["bias"]) and d["n_matured"] >= t["min_matured_picks"]:
            if d["bias"] < -t["bias_threshold"]:
                acts.append(f"目標價偏樂觀（偏差 {pct(d['bias'], 1, True)}）→ 下調 target_shrink，並檢討成長率假設（growth_clip）與目標本益比來源。")
            elif d["bias"] > t["bias_threshold"]:
                acts.append(f"目標價偏保守（偏差 {pct(d['bias'], 1, True)}）→ 可小幅回調 target_shrink（上限 1.0）。")
        if pos:
            worst = sorted([p for p in pos if p.get("alast") is not None], key=lambda p: p["alast"])[:3]
            if worst:
                acts.append("逐一檢討最差的個股（見第二節）：回頭閱讀當期日誌，確認是『論點被推翻』（需修正選股規則）還是『短期雜訊』（不應過度反應）。最差：" + "、".join(f"{p['ticker']}({pct(p['alast'], 0, True)})" for p in worst))
            if len({p['sector'] for p in pos}) and pos:
                top_sec = max(((s, len([x for x in pos if x['sector'] == s])) for s in {p['sector'] for p in pos}), key=lambda kv: kv[1])
                if top_sec[1] / len(pos) > 0.3:
                    acts.append(f"產業集中度偏高（{sname.get(top_sec[0], top_sec[0])} 占 {top_sec[1] / len(pos):.0%} 的推薦）→ 檢查 max_per_sector 與產業分數是否過度追逐熱門題材。")
    for a in acts:
        A(f"- {a}")
    A("")
    # 8 參數回饋
    A("## 八、回饋到系統的參數調整\n")
    if tune:
        res = apply_tuning(d, date, min_dates)
        if res["changed"]:
            A(f"已更新參數至 **v{res['version']}**，下一期日誌起生效：\n")
            for c in res["changes"]:
                A(f"- {c}")
        else:
            A("本次**未調整**任何參數。")
        for s in res["skipped"]:
            A(f"- 未調整：{s}")
    else:
        A("本報告為純檢討（週報），不調整參數；月報會在樣本足夠時回饋參數。")
    A("")
    out = C.path("reviews") / f"{date}-{period}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[review] 已寫入 {out}")
    return str(out)
