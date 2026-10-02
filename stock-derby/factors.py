"""Candidate stock "ability" factors (technical / chip / financial).

Every factor is a (dates x tickers) frame using only information available at
that date's close.  factor_study.py measures which of them actually relate to
forward returns; build_derby.py turns the winners into horse abilities.

NOTE on "chip" (籌碼): free US data has no institutional/margin positioning, so
chip factors are volume-flow proxies (up-volume share, Chaikin money flow, volume surge).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# id -> (label, group, format)  format: "pct" | "num"
META = {
    "mom_12_1":    ("12個月動能(扣近1月)", "技術", "pct"),
    "mom_3m":      ("3個月動能", "技術", "pct"),
    "rev_1m":      ("近1月漲幅(短線反轉)", "技術", "pct"),
    "high_52w":    ("距52週高點·回檔幅度", "技術", "pct"),
    "vol_60":      ("波動率·爆發力(60日)", "技術", "pct"),
    "trend_200":   ("相對200日均線", "技術", "pct"),
    "rsi_14":      ("RSI(14)", "技術", "num"),
    "vol_surge":   ("量能放大(20日/120日均量)", "籌碼", "num"),
    "updown_vol":  ("上漲日成交量占比(60日)", "籌碼", "pct"),
    "cmf_20":      ("資金流向 CMF(20日)", "籌碼", "num"),
    "eps_surprise": ("最近財報EPS驚喜", "財務", "pct"),
    "eps_yoy":     ("EPS年增率", "財務", "pct"),
}


def _earnings_panels(e: pd.DataFrame, index: pd.DatetimeIndex, columns) -> dict[str, pd.DataFrame]:
    e = e.sort_values("date").drop_duplicates(["stock_id", "date"], keep="last").copy()
    e["yoy"] = (e["eps_actual"] - e.groupby("stock_id")["eps_actual"].shift(4)) / \
        e.groupby("stock_id")["eps_actual"].shift(4).abs()
    e["surprise"] = e["surprise_pct"].clip(-100, 100) / 100
    e["yoy"] = e["yoy"].clip(-2, 2)
    out = {}
    for src, name in [("surprise", "eps_surprise"), ("yoy", "eps_yoy")]:
        w = e.pivot(index="date", columns="stock_id", values=src)
        w.index = w.index + pd.Timedelta(days=1)          # known only the day after the report
        w = w.reindex(w.index.union(index)).sort_index().ffill(limit=100).reindex(index)
        out[name] = w.reindex(columns=columns)
    return out


def compute_all(p: dict[str, pd.DataFrame], earn: pd.DataFrame) -> dict[str, pd.DataFrame]:
    c, h, l, v = p["close"], p["high"], p["low"], p["volume"]
    ret = c.pct_change()
    delta = c.diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean()
    mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    f = {
        "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "mom_3m": c / c.shift(63) - 1,
        "rev_1m": c / c.shift(21) - 1,
        "high_52w": c / c.rolling(252, min_periods=200).max() - 1,
        "vol_60": ret.rolling(60).std() * np.sqrt(252),
        "trend_200": c / c.rolling(200, min_periods=150).mean() - 1,
        "rsi_14": 100 - 100 / (1 + rs),
        "vol_surge": v.rolling(20).mean() / v.rolling(120, min_periods=80).mean(),
        "updown_vol": (v * (ret > 0)).rolling(60).sum() / v.rolling(60).sum(),
        "cmf_20": (mfm * v).rolling(20).sum() / v.rolling(20).sum(),
    }
    f.update(_earnings_panels(earn, c.index, c.columns))
    return f
