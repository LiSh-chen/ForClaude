"""多檔 ETF 加權組合模擬——支援「期初買進後不再調整」跟「定期再平衡回
目標權重」兩種模式，用來回答「QQQ 搭配其他 ETF 的組合，會不會比單押
QQQ 更好」這個問題（見 docs/research_findings.md 10.14 節後續延伸，
使用者明確要求驗證不同組合/配比、以及再平衡與否的差異）。

跟 tw_quant/us_costs.py 共用同一套 CostConfig：期初買進、以及每一次
再平衡的買進/賣出，都真的計手續費（複委託每股固定費 + 賣出方向 SEC
規費），不是零成本假設。再平衡越頻繁，這裡的成本侵蝕就越真實地反映
在報酬數字裡，不是事後才另外扣。

反未來函數：逐日往前跑，第 i 天要不要再平衡、平衡到多少股，只用第 i
天（及之前）的收盤價計算，不會用到未來資料。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tw_quant.config import CostConfig
from tw_quant.us_costs import entry_cost, exit_proceeds


def simulate_weighted_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    cost_cfg: CostConfig,
    initial_capital: float,
    rebalance_freq_days: int | None,
) -> tuple[pd.Series, int, list[dict]]:
    """prices：index 是日期（已對齊、已去除任一檔缺值的共同交易日），
    欄位是各檔 ticker 的收盤價，必須涵蓋 weights 的所有 key。

    weights：{ticker: 權重}，總和必須是 1.0。

    rebalance_freq_days：None 代表期初買進後永遠不再調整（權重隨漲跌
    自然漂移，最貼近「買了就不動」的操作）；否則每滿 N 個交易日，把
    持股依當時總市值重新調回原始目標權重（賣多買少都真的計交易成本）。

    回傳 (逐日投資組合市值序列, 再平衡次數, 成交紀錄)。成交紀錄是
    list[dict]，每筆一個 {date, ticker, action ("buy"/"sell"), price,
    shares}，包含期初建倉跟每次再平衡的買賣，給視覺化/持股歷史還原用。
    """
    tickers = list(weights.keys())
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"weights 總和必須是 1.0，收到 {sum(weights.values())}")

    dates = prices.index
    if len(dates) == 0:
        raise ValueError("prices 是空的，無法模擬")

    shares = {t: 0 for t in tickers}
    cash = initial_capital
    fills: list[dict] = []

    first_px = prices.iloc[0]
    for t in tickers:
        price = float(first_px[t])
        target_dollar = weights[t] * initial_capital
        n = int(target_dollar // price)
        cash -= n * price + entry_cost(price, n, cost_cfg)
        shares[t] = n
        if n > 0:
            fills.append({"date": dates[0], "ticker": t, "action": "buy", "price": price, "shares": n})

    values = np.empty(len(dates))
    n_rebalances = 0
    for i in range(len(dates)):
        px = prices.iloc[i]
        port_value = cash + sum(shares[t] * float(px[t]) for t in tickers)

        if rebalance_freq_days is not None and i > 0 and i % rebalance_freq_days == 0:
            n_rebalances += 1
            for t in tickers:
                price = float(px[t])
                target_dollar = weights[t] * port_value
                target_shares = int(target_dollar // price)
                diff = target_shares - shares[t]
                if diff > 0:
                    cash -= diff * price + entry_cost(price, diff, cost_cfg)
                    fills.append({"date": dates[i], "ticker": t, "action": "buy", "price": price, "shares": diff})
                elif diff < 0:
                    _, net = exit_proceeds(price, -diff, cost_cfg)
                    cash += net
                    fills.append({"date": dates[i], "ticker": t, "action": "sell", "price": price, "shares": -diff})
                shares[t] = target_shares
            port_value = cash + sum(shares[t] * float(px[t]) for t in tickers)

        values[i] = port_value

    return pd.Series(values, index=dates), n_rebalances, fills


def metrics_from_equity_curve(values: pd.Series) -> dict:
    """跟 tw_quant.backtest.summarize_performance 同一套公式，直接吃一段
    已排序、已限定日期範圍的市值序列（收盤價或投資組合市值皆可，公式對
    純量縮放不敏感）。"""
    eq = values.reset_index(drop=True)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    years = max(len(eq) / 252, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    running_peak = eq.cummax()
    max_dd = ((running_peak - eq) / running_peak).max()
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0
    return {"total_return": total_return, "cagr": cagr, "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar}
