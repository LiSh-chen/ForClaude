"""在已經算好的（不開槓桿）策略逐日報酬率之上，疊加一層動態槓桿 + 保證金
維持率模擬。

背景：使用者問「機構的槓桿時機」能不能學，討論後決定先在單一策略
（top_n=3，樣本內外表現最一致的候選人）上驗證，範圍先不擴大到其他
策略。這裡刻意不做逐股層級的借股/保證金模擬（那需要對每一檔持股個別
追蹤融資成數，是更大的工程），改用投資組合層級的簡化模擬：整個策略
的日報酬率序列複利，槓桿倍數要嘛固定、要嘛依波動度動態調整
（vol-targeting：波動度越高，槓桿越低），並且逐日檢查維持保證金率，
不足時強制減碼回到不槓桿（1倍）——這個簡化在文獻上很常見（槓桿型
ETF 公開說明書就是用這種「對標的指數的日報酬率複利 + 每日重設槓桿」
的邏輯描述建構方式的），但不是真實複委託帳戶保證金規則的逐股精確
重現（Reg-T 對不同股票的融資成數不同、對集中度過高的帳戶通常會
加收更高的維持保證金，這裡用單一維持保證金率概括，沒有模擬到那個
顆粒度）。

反未來函數：vol_target_leverage_series 用 shift(1)，T 日開盤前只能
用 T-1 為止已經發生的報酬率去估波動度、決定 T 日的目標槓桿。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LeverageConfig:
    target_vol: float = 0.20  # 年化波動度目標（vol-targeting 用）
    vol_lookback_days: int = 21  # 估已實現波動度的回看天數
    min_leverage: float = 0.5  # 波動度極高時最低可以降到多少槓桿（<1 代表持有部分現金）
    max_leverage: float = 2.0  # 槓桿上限
    maintenance_margin_pct: float = 0.30  # 維持保證金率 = 權益 / 部位市值，低於此值觸發強制減碼
    annual_borrow_rate: float = 0.08  # 借款年利率（假設值，真實利率依券商/帳戶規模而異）
    releverage_freq_days: int = 21  # 每隔幾個交易日重新評估一次目標槓桿


def vol_target_leverage_series(daily_returns: pd.Series, cfg: LeverageConfig) -> pd.Series:
    """每天算一個「如果今天要重新評估，目標槓桿該是多少」：用 T-1 為止
    vol_lookback_days 天的已實現波動度（年化）反推 leverage = target_vol /
    realized_vol，夾在 [min_leverage, max_leverage] 之間。回傳的序列已經
    shift(1)，T 日的值只用到 T-1 為止已知的報酬率。
    """
    realized_vol = daily_returns.rolling(cfg.vol_lookback_days).std() * math.sqrt(252)
    target_lev = (cfg.target_vol / realized_vol).clip(cfg.min_leverage, cfg.max_leverage)
    return target_lev.shift(1)


def simulate_leveraged_equity(
    dates: pd.Series,
    unlevered_returns: pd.Series,
    leverage_target: pd.Series | float,
    cfg: LeverageConfig,
    initial_capital: float,
) -> pd.DataFrame:
    """把槓桿疊加在 unlevered_returns（不開槓桿策略的逐日報酬率序列）上，
    回傳逐日 DataFrame：date, equity, leverage（當下實際生效的槓桿倍數）、
    margin_ratio（權益/部位市值）、margin_call（今天是否觸發強制減碼）。

    leverage_target：可以是固定數字（常數槓桿）或跟 dates 對齊的 Series
    （動態槓桿，例如 vol_target_leverage_series 的輸出，NaN 代表當天
    資訊不足還不能決定目標槓桿，沿用前一次的槓桿）。

    releverage_freq_days（cfg 裡）決定多久重新評估一次目標槓桿；觸發
    保證金追繳的當下會立刻強制減碼回 1 倍，不用等到下次排定的重新
    評估日，之後維持 1 倍直到下一個排定的重新評估日才會依當時的
    target_leverage 重新加碼。
    """
    n = len(dates)
    is_series = isinstance(leverage_target, pd.Series)
    daily_borrow_rate = cfg.annual_borrow_rate / 252

    equity = initial_capital
    current_leverage = 1.0
    V = initial_capital  # 部位市值
    B = 0.0  # 借款金額

    rows = []
    for i in range(n):
        r = unlevered_returns.iloc[i]
        r = 0.0 if pd.isna(r) else float(r)

        should_relever = (i == 0) or (cfg.releverage_freq_days > 0 and i % cfg.releverage_freq_days == 0)
        if should_relever:
            target = leverage_target.iloc[i] if is_series else leverage_target
            if not (is_series and pd.isna(target)):
                current_leverage = float(np.clip(target, cfg.min_leverage, cfg.max_leverage))
                V = equity * current_leverage
                B = V - equity

        V = V * (1 + r)
        B = B * (1 + daily_borrow_rate)
        equity = V - B
        equity = max(equity, 0.0)  # 權益理論上可以歸零（本金全部虧光），不允許變負數

        margin_ratio = (equity / V) if V > 0 else 0.0
        margin_call = margin_ratio < cfg.maintenance_margin_pct

        if margin_call:
            V = equity
            B = 0.0
            current_leverage = 1.0 if equity > 0 else 0.0

        rows.append(
            {
                "date": dates.iloc[i] if hasattr(dates, "iloc") else dates[i],
                "equity": equity,
                "leverage": (V / equity) if equity > 0 else 0.0,
                "margin_ratio": margin_ratio,
                "margin_call": bool(margin_call),
            }
        )

    return pd.DataFrame(rows)


def metrics_from_equity_curve(values: pd.Series) -> dict:
    """跟 tw_quant.backtest.summarize_performance 同一套公式。"""
    eq = values.reset_index(drop=True) if hasattr(values, "reset_index") else pd.Series(values)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    years = max(len(eq) / 252, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[0] > 0 else -1.0
    running_peak = eq.cummax()
    max_dd = ((running_peak - eq) / running_peak.replace(0, np.nan)).max()
    max_dd = 0.0 if pd.isna(max_dd) else max_dd
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0
    return {"total_return": total_return, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar}
