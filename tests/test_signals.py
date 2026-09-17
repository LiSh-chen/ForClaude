import pandas as pd

from tw_quant.config import PoolConfig, SqueezeFilterConfig, IgnitionConfig, StrategyBConfig
from tw_quant.data_provider import generate_synthetic_universe, SyntheticUniverseConfig
from tw_quant import signals


def _small_universe():
    return generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=6, n_days=400, seed=11))


def test_strategy_a_signals_unaffected_by_future_data():
    data = _small_universe()
    prices = data["prices"]
    pool_cfg, squeeze_cfg, ignition_cfg = PoolConfig(), SqueezeFilterConfig(), IgnitionConfig()

    sig_before = signals.generate_strategy_a_signals(prices, pool_cfg, squeeze_cfg, ignition_cfg)

    mutated = prices.copy()
    cutoff = mutated["date"].max() - pd.Timedelta(days=60)
    future_mask = mutated["date"] > cutoff
    mutated.loc[future_mask, "close"] *= 5.0
    mutated.loc[future_mask, "high"] *= 5.0
    mutated.loc[future_mask, "low"] *= 5.0
    mutated.loc[future_mask, "volume"] *= 10

    sig_after = signals.generate_strategy_a_signals(mutated, pool_cfg, squeeze_cfg, ignition_cfg)

    past_mask = ~future_mask
    pd.testing.assert_series_equal(
        sig_before.loc[past_mask, "entry_signal"].reset_index(drop=True),
        sig_after.loc[past_mask, "entry_signal"].reset_index(drop=True),
    )


def test_calendar_defense_excludes_shareholder_and_ex_dividend_months():
    dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    mask = signals.build_calendar_defense_mask(df, StrategyBConfig())
    excluded_months = {3, 4, 6, 7, 8}
    for d, ok in zip(dates, mask):
        assert ok == (d.month not in excluded_months)


def test_ignition_requires_both_breakout_and_volume_spike():
    n = 80
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = [10.0] * (n - 1) + [20.0]  # 最後一天大幅突破
    df = pd.DataFrame(
        {
            "stock_id": ["A"] * n,
            "date": dates,
            "close": close,
            "high": close,
            "low": [c - 0.1 for c in close],
            "volume": [1000] * (n - 1) + [1000],  # 量沒有放大
        }
    )
    ignition_cfg = IgnitionConfig(breakout_window=60, volume_avg_window=5, volume_multiplier=2.0)
    mask = signals.build_ignition_mask(df, ignition_cfg)
    assert not mask.iloc[-1]  # 有價格突破但沒有爆量，不應觸發

    df.loc[df.index[-1], "volume"] = 5000  # 補上爆量
    mask2 = signals.build_ignition_mask(df, ignition_cfg)
    assert mask2.iloc[-1]
