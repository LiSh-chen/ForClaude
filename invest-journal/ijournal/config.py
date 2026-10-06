"""設定與路徑。環境變數 IJ_OUT 可把所有輸出（data/journal/reviews/site）導到別處（demo/測試用）。"""
from __future__ import annotations
import json
import os
from pathlib import Path

from .utils import market_of

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def out_root() -> Path:
    return Path(os.environ.get("IJ_OUT", ROOT))


def path(name: str) -> Path:
    """name ∈ data, journal, reviews, site"""
    p = out_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save_json(p: Path, obj) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def params_path() -> Path:
    # 參數永遠存在『輸出根目錄』的 config/（demo 時不會污染真實參數）
    return out_root() / "config" / "params.json"


def load_params() -> dict:
    p = load_json(params_path())
    return p if p else load_json(CONFIG_DIR / "params.json")


def history_path() -> Path:
    return out_root() / "config" / "params_history.json"


def load_universe() -> dict:
    """回傳 {'sectors': [...], 'stocks': {ticker: {...}}, 'benchmarks':..., 'indices':...}"""
    raw = load_json(CONFIG_DIR / "universe.json")
    stocks = {}
    for s in raw["sectors"]:
        for spec in s["tickers"]:
            parts = (spec.split("|") + ["", ""])[:3]
            t, name, aliases = parts[0].strip(), parts[1].strip(), [a.strip() for a in parts[2].split(",") if a.strip()]
            stocks[t] = {"ticker": t, "name": name or t, "aliases": aliases, "sector": s["id"], "market": market_of(t)}
    # 同一檔雙重上市（ADR 與本股）：避免同日重複推薦同一家公司
    stocks.get("TSM", {})["dual_of"] = "2330.TW"
    return {"sectors": raw["sectors"], "stocks": stocks, "benchmarks": raw["benchmarks"], "indices": raw["indices"]}


def load_sources() -> dict:
    return load_json(CONFIG_DIR / "sources.json")
