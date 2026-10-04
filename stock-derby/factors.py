"""Candidate stock "ability" factors (technical / chip / financial).

Every factor is a (dates x tickers) frame using only information available at
that date's close.  factor_study.py measures which of them actually relate to
forward returns; build_panel.py turns the winners into horse abilities.

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


# Label used when a factor is entered *reversed* (so that "higher value = better" holds for every ability).
REV_LABEL = {
    "mom_12_1": "12個月跌幅(扣近1月)", "mom_3m": "3個月跌幅", "rev_1m": "近1月跌幅(短線反轉)",
    "high_52w": "距52週高點回檔幅度", "vol_60": "低波動度(60日)", "trend_200": "低於200日均線幅度",
    "rsi_14": "RSI超賣程度(100-RSI)", "vol_surge": "量能萎縮程度", "updown_vol": "下跌日成交量占比(60日)",
    "cmf_20": "資金流出程度(-CMF)", "eps_surprise": "EPS不如預期幅度", "eps_yoy": "EPS年增率衰退幅度",
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


# Horse-racing wording for each ability.  name/meaning describe it the way a racing paper would; `param` is the
# actual statistic behind it (shown only when the user taps the ability).  (normal, reversed) pairs: a reversed
# ability is the same statistic with the sign flipped so that "higher = better" still holds.
HORSE = {
    "vol_60": (("爆發力", "短線震盪越劇烈，越有一鼓作氣暴衝的本錢"), ("穩健度", "走勢越平穩，越不容易失蹄"),
               ("60日年化波動率，越大分數越高", "60日年化波動率取反，越小分數越高")),
    "eps_yoy": (("成長力", "獲利比去年同期成長越多，體能進步越明顯"), ("反彈本錢", "獲利衰退後有回升空間"),
                ("最近一季 EPS 相對去年同期的成長率，越大越高", "EPS 年增率取反（衰退越多分數越高）")),
    "high_52w": (("高位氣勢", "股價貼近 52 週高點，氣勢正旺"), ("後勁", "離高點越遠，後段追趕的空間越大"),
                 ("現價 ÷ 52 週高點 − 1，越接近高點越高", "回檔幅度＝1 − 現價 ÷ 52 週高點，回檔越深分數越高")),
    "eps_surprise": (("超水準", "最近一次財報大幅優於分析師預期，等於跑出超水準的成績"), ("谷底蓄勢", "財報不如預期後的反彈潛力"),
                     ("最近一次財報 EPS 優於分析師預期的幅度，越大越高", "EPS 低於預期的幅度，越大分數越高")),
    "mom_12_1": (("耐力", "過去一年（扣掉最近一個月）一路領先，長途賽跑的底子"), ("補漲力", "長期落後的補漲空間"),
                 ("近 12 個月累積漲幅（扣除最近 1 個月），越大越高", "近 12 個月跌幅，跌越多分數越高")),
    "mom_3m": (("衝刺力", "近一季持續向前衝的力道"), ("回氣力", "近一季走弱後的喘息回升"),
               ("近 3 個月累積漲幅，越大越高", "近 3 個月跌幅，跌越多分數越高")),
    "rev_1m": (("近期氣勢", "最近一個月的表現"), ("反彈力", "最近一個月跌深後的反彈力"),
               ("近 1 個月漲幅，越大越高", "近 1 個月跌幅，跌越多分數越高")),
    "trend_200": (("步伐穩健", "股價站在長期均線之上，步伐穩定"), ("低檔蓄力", "股價低於長期均線，蓄勢待發"),
                  ("現價相對 200 日均線的幅度，越高越好", "低於 200 日均線的幅度，越低分數越高")),
    "rsi_14": (("氣勢", "短線買盤強勁"), ("體力回復", "短線超賣後體力回復"),
               ("RSI(14)，越高越好", "100 − RSI(14)，超賣程度越高分數越高")),
    "vol_surge": (("場內人氣", "近期成交量放大，關注度升溫"), ("冷靜期", "成交量萎縮後的沉澱"),
                  ("近 20 日均量 ÷ 近 120 日均量，越高越好", "上式取反，量縮越多分數越高")),
    "updown_vol": (("助跑力", "上漲日成交量占比高，買盤主導"), ("逆風力", "下跌日成交量占比高，逆風中的反彈力"),
                   ("近 60 日上漲日成交量占比，越高越好", "近 60 日下跌日成交量占比，越高分數越高")),
    "cmf_20": (("資金護航", "資金持續流入"), ("資金退潮", "資金流出後的反彈潛力"),
               ("Chaikin 資金流向 CMF(20日)，越高越好", "−CMF(20日)，資金流出越多分數越高")),
}


def horse_term(fid: str, reversed_: bool) -> dict:
    n, r, p = HORSE[fid]
    name, meaning = r if reversed_ else n
    return {"horse": name, "meaning": meaning, "param": p[1] if reversed_ else p[0]}
