"""測試 tw_quant/backtest_stats.py 的交易統計工具。

背景：trade_stats() 曾經漏掉在回傳值裡包含 win_rate（內部算出來但沒放進
回傳的 dict），因為 metrics_from_result() 是先用 summarize_performance()
（也有算 win_rate，但只算已實現交易）打底、再用 trade_stats() 的結果
update() 上去，這個沒回傳的 bug 不會讓程式報錯，只會讓 win_rate 悄悄retain
用了「已實現交易」的版本而不是「含未平倉部位」的版本——直到有腳本繞過
summarize_performance()、只單獨呼叫 trade_stats() 時，才會發現 win_rate
根本不在回傳值裡（變成 0.0）。這裡把這個欄位補回來，並用測試鎖住。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest_stats import trade_stats


def test_trade_stats_returns_win_rate():
    trades = pd.DataFrame(
        {
            "pnl": [100.0, -50.0, 200.0, -30.0],
            "pnl_pct": [0.10, -0.05, 0.20, -0.03],
            "entry_date": pd.to_datetime(["2024-01-01"] * 4),
            "exit_date": pd.to_datetime(["2024-01-05"] * 4),
        }
    )
    stats = trade_stats(trades)
    assert "win_rate" in stats
    assert stats["win_rate"] == 0.5


def test_trade_stats_empty_trades_returns_all_expected_keys():
    stats = trade_stats(pd.DataFrame())
    expected_keys = {
        "win_rate", "avg_win_pct", "avg_loss_pct", "risk_reward_ratio",
        "ev_pct", "profit_factor", "avg_holding_days",
    }
    assert expected_keys.issubset(stats.keys())
    assert stats["win_rate"] == 0.0


def test_trade_stats_ev_pct_matches_win_rate_weighted_formula():
    trades = pd.DataFrame(
        {
            "pnl": [100.0, -50.0],
            "pnl_pct": [0.10, -0.05],
            "entry_date": pd.to_datetime(["2024-01-01"] * 2),
            "exit_date": pd.to_datetime(["2024-01-05"] * 2),
        }
    )
    stats = trade_stats(trades)
    expected_ev = stats["win_rate"] * stats["avg_win_pct"] + (1 - stats["win_rate"]) * stats["avg_loss_pct"]
    assert abs(stats["ev_pct"] - expected_ev) < 1e-9
