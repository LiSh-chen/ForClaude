"""測試同日跳空回補回測引擎（tw_quant/intraday_roundtrip.py）。

用 monkeypatch 把魚池篩選（build_pool_mask）換成「全部通過」，這樣可以用
小範例直接測跳空幅度排名/門檻邏輯本身，不用湊滿 252 日暖身期跟成交量/
趨勢條件（這些已經在 tw_quant/signals.py 自己的測試裡驗證過）。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import intraday_roundtrip as irt
from tw_quant.config import StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe


def _always_true_pool(master, pool_cfg):
    return pd.Series(True, index=master.index)


def _two_day_gap_frame() -> pd.DataFrame:
    rows = [
        # day1：基準收盤價，隔天才有 prior_close 可以算跳空
        {"date": pd.Timestamp("2024-01-01"), "stock_id": "S1", "industry": "IND", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000, "turnover_value": 1e7},
        {"date": pd.Timestamp("2024-01-01"), "stock_id": "S2", "industry": "IND", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000, "turnover_value": 1e7},
        {"date": pd.Timestamp("2024-01-01"), "stock_id": "S3", "industry": "IND", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000, "turnover_value": 1e7},
        # day2：S1 跳空 -5%、S2 跳空 -10%、S3 跳空 -1%（低於門檻）
        {"date": pd.Timestamp("2024-01-02"), "stock_id": "S1", "industry": "IND", "open": 95.0, "high": 98.0, "low": 94.0, "close": 97.0, "volume": 100000, "turnover_value": 1e7},
        {"date": pd.Timestamp("2024-01-02"), "stock_id": "S2", "industry": "IND", "open": 90.0, "high": 93.0, "low": 89.0, "close": 92.0, "volume": 100000, "turnover_value": 1e7},
        {"date": pd.Timestamp("2024-01-02"), "stock_id": "S3", "industry": "IND", "open": 99.0, "high": 100.0, "low": 98.0, "close": 99.5, "volume": 100000, "turnover_value": 1e7},
    ]
    return pd.DataFrame(rows)


def test_selects_biggest_gap_down_within_top_n(monkeypatch):
    monkeypatch.setattr(irt, "build_pool_mask", _always_true_pool)
    prices = _two_day_gap_frame()
    cfg = StrategyConfig()
    # top_n=1 讓 per_stock_budget 等於 100% 資金，用整股(lot)無條件捨去後
    # 幾乎不留現金餘裕付手續費；小幅加碼資金只是留手續費的緩衝空間，不影響
    # 股數（still floors to the same lot count），純粹避免這個測試專屬的
    # 邊界情況觸發「現金不足」拒單（真實回測 top_n 通常是 15~30，每檔只佔
    # 一小部分資金，floor 捨去自然就留了餘裕，不會有這個問題）。
    cfg.initial_capital *= 1.01
    rt_cfg = irt.IntradayRoundtripConfig(gap_threshold=0.02, top_n=1)

    result = irt.run_intraday_roundtrip_backtest(prices, cfg, rt_cfg)

    day2 = pd.Timestamp("2024-01-02")
    day2_trades = result.trades[result.trades["entry_date"] == day2]
    # top_n=1，S2 跳空最大（-10% < -5%），S3 沒過門檻 -> 只該買 S2
    assert len(day2_trades) == 1
    assert day2_trades.iloc[0]["stock_id"] == "S2"


def test_excludes_gaps_below_threshold(monkeypatch):
    monkeypatch.setattr(irt, "build_pool_mask", _always_true_pool)
    prices = _two_day_gap_frame()
    cfg = StrategyConfig()
    # 門檻拉到 20%，三檔都不該過
    rt_cfg = irt.IntradayRoundtripConfig(gap_threshold=0.20, top_n=10)

    result = irt.run_intraday_roundtrip_backtest(prices, cfg, rt_cfg)
    assert result.trades.empty


def test_entry_and_exit_are_always_same_day(monkeypatch):
    monkeypatch.setattr(irt, "build_pool_mask", _always_true_pool)
    prices = _two_day_gap_frame()
    cfg = StrategyConfig()
    rt_cfg = irt.IntradayRoundtripConfig(gap_threshold=0.02, top_n=10)

    result = irt.run_intraday_roundtrip_backtest(prices, cfg, rt_cfg)
    assert not result.trades.empty
    assert (result.trades["entry_date"] == result.trades["exit_date"]).all()


def test_pnl_matches_hand_computed_value_for_single_trade(monkeypatch):
    monkeypatch.setattr(irt, "build_pool_mask", _always_true_pool)
    prices = _two_day_gap_frame()
    cfg = StrategyConfig()
    cfg.initial_capital *= 1.01  # 同上：留手續費緩衝，見另一個 top_n=1 測試的註解
    rt_cfg = irt.IntradayRoundtripConfig(gap_threshold=0.02, top_n=1)

    result = irt.run_intraday_roundtrip_backtest(prices, cfg, rt_cfg)
    trade = result.trades.iloc[0]

    # 用系統的成本模型手動算一次同一筆交易的損益，交叉驗證回測引擎算的一致
    from tw_quant import costs as cost_mod

    entry_price = 90.0  # S2 開盤價
    exit_price = 92.0  # S2 收盤價
    shares = trade["shares"]
    fee_in = cost_mod.entry_cost(entry_price, shares, cfg.costs)
    cost_basis = shares * entry_price + fee_in
    _, net_proceeds = cost_mod.exit_proceeds(exit_price, shares, cfg.costs)
    expected_pnl = net_proceeds - cost_basis

    assert abs(trade["pnl"] - expected_pnl) < 1e-6


def test_no_open_positions_and_equity_stays_positive_on_synthetic_data():
    """在合成資料上跑一次完整流程（含真正的魚池篩選），確認不會崩潰、
    權益曲線不會變成負值或 NaN，且這個引擎的結構是「不留倉」（open_positions
    永遠是空的，因為每筆交易當天就結清）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=20, n_days=400, seed=3))
    cfg = StrategyConfig()
    rt_cfg = irt.IntradayRoundtripConfig(gap_threshold=0.02, top_n=5)

    result = irt.run_intraday_roundtrip_backtest(data["prices"], cfg, rt_cfg)

    assert result.open_positions == {}
    assert not result.equity_curve["equity"].isna().any()
    assert (result.equity_curve["equity"] > 0).all()
