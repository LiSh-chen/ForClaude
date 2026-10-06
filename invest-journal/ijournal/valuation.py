"""目標價估算（仿大型機構研究報告的做法）。

四種方法加權後得到 12 個月 base 目標價：
  1. 同業相對本益比（Comps）：預估 EPS × 目標本益比（同業中位數，依成長率溢/折價）
  2. PEG：預估 EPS × (成長率% × 目標 PEG)
  3. 兩階段 DCF：每股自由現金流成長 10 年並線性收斂到永續成長，以 CAPM 求折現率
  4. 券商共識目標價（僅作交叉檢驗，權重較低）
權重與 target_shrink 來自 config/params.json，會被績效檢討回饋調整。
再以方法間分歧度與波動度估算 bull/bear 情境，並以情境機率計算期望值。
"""
from __future__ import annotations
import math

from .utils import clip, isnum, median


def growth_estimate(f: dict, p: dict) -> float | None:
    lo, hi = p["valuation"]["growth_clip"]
    g = None
    if f.get("eps_fwd_px") and f.get("eps_ttm_px") and f["eps_ttm_px"] > 0:
        g = f["eps_fwd_px"] / f["eps_ttm_px"] - 1
    if g is None or g <= -0.5:
        cands = [x for x in (f.get("eps_growth"), f.get("rev_growth")) if isnum(x)]
        g = sum(cands) / len(cands) if cands else None
    return clip(g, lo, hi) if g is not None else None


def peer_stats(peers: list[dict], p: dict) -> dict:
    pes = [x["pe_fwd"] for x in peers if isnum(x.get("pe_fwd")) and 3 < x["pe_fwd"] < 100]
    gs = [x["_g"] for x in peers if isnum(x.get("_g"))]
    return {"pe": median(pes), "g": median(gs), "n": len(pes)}


def dcf_per_share(f: dict, g0: float, market: str, beta: float | None, p: dict) -> tuple[float | None, dict]:
    v = p["valuation"]
    if not f.get("fcf_ps") or f["fcf_ps"] <= 0:
        return None, {}
    b = beta if isnum(beta) else 1.0
    wacc = clip(v["rf"][market] + clip(b, 0.5, 2.0) * v["erp"], *v["wacc_bounds"])
    tg = v["terminal_growth"]
    g1 = clip(g0, 0.0, 0.25)
    cf, pv = f["fcf_ps"], 0.0
    for yr in range(1, 11):
        g = g1 + (tg - g1) * (yr - 1) / 9.0
        cf *= 1 + g
        pv += cf / (1 + wacc) ** yr
    tv = cf * (1 + tg) / (wacc - tg)
    pv += tv / (1 + wacc) ** 10
    return pv, {"wacc": wacc, "g1": g1, "tg": tg}


def value_stock(price: float, f: dict, pf: dict, market: str, peers: dict, p: dict) -> dict | None:
    """回傳目標價結構；資料不足回傳 None。"""
    v = p["valuation"]
    g = growth_estimate(f, p)
    methods: dict[str, dict] = {}
    eps = f.get("eps_fwd_px")
    if eps and eps > 0 and peers.get("pe"):
        adj = 1.0
        if g is not None and peers.get("g") is not None:
            adj = clip(1 + 0.5 * (g - peers["g"]), 0.8, 1.3)
        tpe = clip(peers["pe"] * adj, *v["pe_bounds"])
        methods["pe"] = {"label": "同業相對本益比", "value": eps * tpe,
                         "detail": f"預估EPS {eps:.2f} × 目標本益比 {tpe:.1f}（同業中位數 {peers['pe']:.1f} × 成長調整 {adj:.2f}）"}
    if eps and eps > 0 and g is not None and g >= 0.05:
        fpe = clip(g * 100 * v["peg_target"], *v["pe_bounds"])
        methods["peg"] = {"label": "PEG 成長調整", "value": eps * fpe,
                          "detail": f"預估EPS {eps:.2f} × 合理本益比 {fpe:.1f}（成長率 {g * 100:.1f}% × PEG {v['peg_target']}）"}
    if f.get("same_ccy") and g is not None:
        val, d = dcf_per_share(f, g, market, f.get("beta"), p)
        if val:
            methods["dcf"] = {"label": "兩階段 DCF", "value": val,
                              "detail": f"每股FCF {f['fcf_ps']:.2f}，前期成長 {d['g1'] * 100:.1f}%→永續 {d['tg'] * 100:.1f}%，WACC {d['wacc'] * 100:.1f}%"}
    if f.get("target_mean") and (f.get("n_analysts") or 0) >= 2:
        methods["consensus"] = {"label": "券商共識目標價", "value": f["target_mean"],
                                "detail": f"{int(f['n_analysts'])} 位分析師平均（區間 {f.get('target_low') or '—'}–{f.get('target_high') or '—'}）"}
    # 離群剔除：估值低於 0.35 倍或高於 3 倍現價者，多半是資料/假設失真，不納入加權
    dropped = {}
    for k in list(methods):
        r = methods[k]["value"] / price
        if r < 0.35 or r > 3.0:
            dropped[k] = methods.pop(k)
    own = [k for k in methods if k != "consensus"]
    if not own or len(methods) < 2:
        return None
    w = p["method_weights"]
    tot = sum(w[k] for k in methods)
    for k, m in methods.items():
        m["weight"] = w[k] / tot
        m["upside"] = m["value"] / price - 1
    blend = sum(m["value"] * m["weight"] for m in methods.values())
    blend = min(blend, price * (1 + v["max_upside"]))
    base = price * (1 + (blend / price - 1) * p["target_shrink"])
    vals = [m["value"] for m in methods.values()]
    spread = (max(vals) - min(vals)) / (sum(vals) / len(vals))
    confidence = "高" if len(methods) >= 3 and spread <= 0.35 else "中" if spread <= 0.6 else "低"
    d = clip(0.5 * spread + 0.25 * (pf.get("vol_ann") or 0.3), 0.12, 0.40)
    bull, bear = base * (1 + d), max(base * (1 - 1.5 * d), price * 0.4)
    pr = p["scenario_prob"]
    ev = pr["bull"] * bull + pr["base"] * base + pr["bear"] * bear
    return {
        "methods": methods, "growth": g, "peer_pe": peers.get("pe"), "blend": blend, "base": base, "bull": bull, "bear": bear,
        "dispersion": d, "ev": ev, "upside": base / price - 1, "ev_upside": ev / price - 1,
        "spread": spread, "confidence": confidence, "dropped": dropped,
        "rr": (base - price) / (price - bear) if bear < price else None, "n_methods": len(methods),
    }
