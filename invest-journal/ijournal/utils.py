from __future__ import annotations
import math
from typing import Iterable, Optional


def isnum(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def lin(x, lo: float, hi: float) -> Optional[float]:
    """線性映射到 0–100（lo→0, hi→100，超出截斷）。lo>hi 代表『越小越好』。"""
    if not isnum(x) or hi == lo:
        return None
    v = (x - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, v))


def avg(vals: Iterable, min_n: int = 1) -> Optional[float]:
    v = [x for x in vals if isnum(x)]
    return sum(v) / len(v) if len(v) >= min_n else None


def wavg(pairs: Iterable[tuple]) -> Optional[float]:
    """加權平均，缺值者略過並重新正規化權重。"""
    num = den = 0.0
    for v, w in pairs:
        if isnum(v) and isnum(w) and w > 0:
            num += v * w
            den += w
    return num / den if den > 0 else None


def median(vals: Iterable) -> Optional[float]:
    v = sorted(x for x in vals if isnum(x))
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def pct(x, d: int = 1, sign: bool = False) -> str:
    if not isnum(x):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def num(x, d: int = 2) -> str:
    return f"{x:,.{d}f}" if isnum(x) else "—"


def big(x) -> str:
    """市值等大數字縮寫。"""
    if not isnum(x):
        return "—"
    a = abs(x)
    if a >= 1e12:
        return f"{x / 1e12:.2f}兆"
    if a >= 1e8:
        return f"{x / 1e8:.0f}億"
    return f"{x:,.0f}"


def market_of(ticker: str) -> str:
    return "TW" if ticker.endswith((".TW", ".TWO")) else "US"


def md_escape(s: str) -> str:
    return (s or "").replace("|", "／").replace("[", "［").replace("]", "］").replace("\n", " ").strip()
