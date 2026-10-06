"""由價格序列與基本面 info 計算特徵。"""
from __future__ import annotations
import math
import pandas as pd

from .utils import isnum


def price_features(df: pd.DataFrame, bench: pd.DataFrame | None) -> dict | None:
    if df is None or len(df) < 60:
        return None
    c = df["Close"].dropna()
    last = float(c.iloc[-1])

    def ret(n):
        return float(last / c.iloc[-1 - n] - 1) if len(c) > n else None

    f = {"last": last, "date": str(c.index[-1].date()), "ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_1y": ret(252)}
    f["ma50"] = float(c.tail(50).mean())
    f["ma200"] = float(c.tail(200).mean()) if len(c) >= 200 else None
    f["dist200"] = last / f["ma200"] - 1 if f["ma200"] else None
    f["high52"] = float(c.tail(252).max())
    f["from_high"] = last / f["high52"] - 1
    hl = df[["High", "Low", "Close"]].dropna()
    pc = hl["Close"].shift(1)
    tr = pd.concat([hl["High"] - hl["Low"], (hl["High"] - pc).abs(), (hl["Low"] - pc).abs()], axis=1).max(axis=1)
    f["atr14"] = float(tr.tail(14).mean())
    f["vol_ann"] = float(c.pct_change().tail(60).std() * math.sqrt(252))
    f["turnover"] = float((df["Close"] * df["Volume"]).tail(20).mean())
    if bench is not None and len(bench) > 130:
        b = bench["Close"].dropna()
        b3 = b.iloc[-1] / b.iloc[-64] - 1
        b6 = b.iloc[-1] / b.iloc[-127] - 1
        f["rel_3m"] = f["ret_3m"] - float(b3) if f["ret_3m"] is not None else None
        f["rel_6m"] = f["ret_6m"] - float(b6) if f["ret_6m"] is not None else None
    else:
        f["rel_3m"], f["rel_6m"] = f["ret_3m"], f["ret_6m"]
    return f


def norm_fundamentals(info: dict, price: float, price_ccy_hint: str) -> dict:
    g = lambda k: info.get(k) if isnum(info.get(k)) else None
    ccy, fccy = info.get("currency") or price_ccy_hint, info.get("financialCurrency") or info.get("currency") or price_ccy_hint
    f = {
        "name": info.get("shortName") or info.get("longName"), "currency": ccy, "financial_currency": fccy,
        "same_ccy": (str(ccy).upper() == str(fccy).upper()),
        "market_cap": g("marketCap"), "shares": g("sharesOutstanding"),
        "pe_ttm": g("trailingPE"), "pe_fwd": g("forwardPE"), "eps_ttm": g("trailingEps"), "eps_fwd": g("forwardEps"),
        "peg": g("pegRatio") or g("trailingPegRatio"),
        "roe": g("returnOnEquity"), "roa": g("returnOnAssets"), "gross_margin": g("grossMargins"),
        "op_margin": g("operatingMargins"), "net_margin": g("profitMargins"),
        "rev_growth": g("revenueGrowth"), "eps_growth": g("earningsGrowth") if g("earningsGrowth") is not None else g("earningsQuarterlyGrowth"),
        "de": (g("debtToEquity") / 100.0) if g("debtToEquity") is not None else None,
        "current_ratio": g("currentRatio"), "fcf": g("freeCashflow"), "ocf": g("operatingCashflow"), "revenue": g("totalRevenue"),
        "beta": g("beta"), "target_mean": g("targetMeanPrice"), "target_high": g("targetHighPrice"), "target_low": g("targetLowPrice"),
        "n_analysts": g("numberOfAnalystOpinions"), "rec_mean": g("recommendationMean"),
        "ev_ebitda": g("enterpriseToEbitda"), "ps": g("priceToSalesTrailing12Months"), "div_yield": g("dividendYield"),
    }
    # 以「價格 / 本益比」反推 EPS，避免 ADR 幣別（財報幣別≠交易幣別）造成錯誤
    if f["pe_fwd"] and f["pe_fwd"] > 0:
        f["eps_fwd_px"] = price / f["pe_fwd"]
    elif f["eps_fwd"] and f["same_ccy"]:
        f["eps_fwd_px"] = f["eps_fwd"]
    else:
        f["eps_fwd_px"] = None
    if f["pe_ttm"] and f["pe_ttm"] > 0:
        f["eps_ttm_px"] = price / f["pe_ttm"]
    elif f["eps_ttm"] and f["same_ccy"]:
        f["eps_ttm_px"] = f["eps_ttm"]
    else:
        f["eps_ttm_px"] = None
    f["fcf_ps"] = f["fcf"] / f["shares"] if f["fcf"] and f["shares"] and f["same_ccy"] else None
    f["fcf_margin"] = f["fcf"] / f["revenue"] if f["fcf"] is not None and f["revenue"] else None
    return f
