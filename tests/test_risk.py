from tw_quant.config import PositionSizingConfig
from tw_quant.risk import compute_initial_stop, compute_position_size, compute_chandelier_stop


def test_initial_stop_uses_pct_floor_when_tighter():
    cfg = PositionSizingConfig()
    # entry*0.95 = 95; high-2.5*atr = 98 - 2.5*0.5 = 96.75 -> floor 95 is looser, so max picks 96.75
    stop = compute_initial_stop(entry_price=100, prior_10d_high=98, atr_at_signal=0.5, cfg=cfg)
    assert abs(stop - 96.75) < 1e-9


def test_initial_stop_uses_atr_stop_when_it_is_higher():
    cfg = PositionSizingConfig()
    # atr stop = 98 - 2.5*5 = 85.5 which is below the 95 floor -> max picks 95 (the floor)
    stop = compute_initial_stop(entry_price=100, prior_10d_high=98, atr_at_signal=5, cfg=cfg)
    assert abs(stop - 95.0) < 1e-9


def test_position_size_floors_to_lot():
    cfg = PositionSizingConfig()
    # risk_per_share=6（entry 的 12%），避開市值上限防禦，單純測試無條件捨去至張
    result = compute_position_size(capital_base=1_000_000, entry_price=50, stop_price=44, cfg=cfg)
    # risk_budget=20000, risk_per_share=6 -> est=3333.33 shares -> floor to 3000
    assert result.shares == 3000
    assert not result.rejected


def test_position_size_rejects_sub_lot():
    cfg = PositionSizingConfig()
    # 極小風險預算 + 大風險距離 -> 股數不足 1 張
    result = compute_position_size(capital_base=1000, entry_price=50, stop_price=10, cfg=cfg)
    assert result.rejected
    assert result.shares == 0


def test_position_size_caps_at_max_position_value_pct():
    cfg = PositionSizingConfig(max_position_value_pct_of_capital=0.10)
    # 若無市值上限，股數會遠超過 10% 資金
    result = compute_position_size(capital_base=1_000_000, entry_price=100, stop_price=99, cfg=cfg)
    assert not result.rejected
    assert result.position_value <= 100_000 + 1e-6


def test_chandelier_stop_formula():
    cfg = PositionSizingConfig(atr_multiplier=2.5)
    stop = compute_chandelier_stop(rolling_10d_high=110, atr_today=4, cfg=cfg)
    assert abs(stop - (110 - 10)) < 1e-9
