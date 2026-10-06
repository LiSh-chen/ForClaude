"""把檢討結果回饋到評估流程：在『證據足夠』時，小步幅調整因子權重、估值方法權重與目標價折減係數。
設計原則：①樣本門檻 ②單次調整幅度上限 ③權重上下限 ④每次變更留下可稽核紀錄 ⑤樣本不足就不動。"""
from __future__ import annotations
import copy
import datetime as dt

from . import config as C
from .utils import clip, isnum


def _reweight(w: dict, ic: dict, t: dict) -> tuple[dict, list[str]]:
    new, notes = dict(w), []
    for k, v in w.items():
        if not isnum(ic.get(k)):
            continue
        step = clip(t["ic_gain"] * ic[k], -t["max_step"], t["max_step"])
        new[k] = clip(v * (1 + step), t["min_weight"], t["max_weight"])
    s = sum(new.values())
    new = {k: v / s for k, v in new.items()}
    for k in w:
        if abs(new[k] - w[k]) >= 0.002:
            notes.append(f"{k}: {w[k]:.3f}→{new[k]:.3f}（IC {ic[k]:+.3f}）")
    return new, notes


def apply_tuning(diag: dict, date: str, min_dates: int | None = None) -> dict:
    p = C.load_params()
    t = p["tuning"]
    md = min_dates if min_dates is not None else t["min_dates"]
    new = copy.deepcopy(p)
    changes, skipped = [], []
    if diag["ic_dates"] >= md:
        new["factor_weights"], c1 = _reweight(p["factor_weights"], diag["factor_ic"], t)
        changes += [f"因子權重 {x}" for x in c1]
    else:
        skipped.append(f"因子權重：僅 {diag['ic_dates']} 個有效日期（門檻 {md}），維持不變")
    if diag["method_ic_dates"] >= md:
        new["method_weights"], c2 = _reweight(p["method_weights"], diag["method_ic"], t)
        changes += [f"估值方法權重 {x}" for x in c2]
    else:
        skipped.append(f"估值方法權重：僅 {diag['method_ic_dates']} 個有效日期（門檻 {md}），維持不變")
    if diag["n_matured"] >= t["min_matured_picks"] and isnum(diag["bias"]):
        lo, hi = t["shrink_bounds"]
        s0 = p["target_shrink"]
        if diag["bias"] < -t["bias_threshold"]:
            new["target_shrink"] = clip(s0 - t["shrink_step"], lo, hi)
        elif diag["bias"] > t["bias_threshold"]:
            new["target_shrink"] = clip(s0 + t["shrink_step"], lo, hi)
        if new["target_shrink"] != s0:
            changes.append(f"target_shrink: {s0:.2f}→{new['target_shrink']:.2f}（20 日實現報酬相對目標價隱含路徑的平均偏差 {diag['bias'] * 100:+.1f}pp）")
    else:
        skipped.append(f"目標價折減：僅 {diag['n_matured']} 筆到期推薦（門檻 {t['min_matured_picks']}），維持不變")
    if changes:
        new["version"] = p["version"] + 1
        new["updated"] = date
        C.save_json(C.params_path(), new)
        hist = C.load_json(C.history_path(), [])
        hist.append({"version": new["version"], "date": date, "changes": changes, "skipped": skipped})
        C.save_json(C.history_path(), hist)
    return {"changed": bool(changes), "changes": changes, "skipped": skipped, "version": new["version"]}
