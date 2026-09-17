from tw_quant.config import GlobalRiskConfig
from tw_quant.portfolio_risk import TradeCandidate, apply_global_lock


def _candidate(stock_id, industry, potential_loss, position_value, liquidity):
    return TradeCandidate(
        stock_id=stock_id,
        industry=industry,
        strategy="A",
        entry_price=100,
        stop_price=95,
        shares=1000,
        position_value=position_value,
        potential_loss=potential_loss,
        liquidity_rank_value=liquidity,
    )


def test_daily_risk_cap_rejects_beyond_6_percent():
    cfg = GlobalRiskConfig(max_daily_new_risk_pct=0.06, max_industry_exposure_pct=1.0)
    capital = 1_000_000
    candidates = [
        _candidate("A", "IND1", potential_loss=20_000, position_value=10_000, liquidity=500),
        _candidate("B", "IND2", potential_loss=20_000, position_value=10_000, liquidity=400),
        _candidate("C", "IND3", potential_loss=20_000, position_value=10_000, liquidity=300),
        _candidate("D", "IND4", potential_loss=20_000, position_value=10_000, liquidity=200),
    ]
    result = apply_global_lock(candidates, capital, {}, cfg)
    # 6% of 1,000,000 = 60,000 -> 恰好只能容納前 3 檔（每檔 20,000）
    assert [c.stock_id for c in result.accepted] == ["A", "B", "C"]
    assert result.rejected[0][0].stock_id == "D"


def test_liquidity_tie_breaker_prioritizes_higher_turnover():
    cfg = GlobalRiskConfig(max_daily_new_risk_pct=0.02, max_industry_exposure_pct=1.0)
    capital = 1_000_000
    # 只夠容納一檔 (2% = 20,000)，兩檔互斥，流動性高者應獲勝
    candidates = [
        _candidate("LOW_LIQ", "IND1", potential_loss=20_000, position_value=10_000, liquidity=100),
        _candidate("HIGH_LIQ", "IND2", potential_loss=20_000, position_value=10_000, liquidity=9999),
    ]
    result = apply_global_lock(candidates, capital, {}, cfg)
    assert [c.stock_id for c in result.accepted] == ["HIGH_LIQ"]


def test_industry_cap_blocks_over_concentration():
    cfg = GlobalRiskConfig(max_daily_new_risk_pct=1.0, max_industry_exposure_pct=0.10)
    capital = 1_000_000
    existing = {"IND1": 90_000}
    candidates = [_candidate("A", "IND1", potential_loss=1_000, position_value=20_000, liquidity=100)]
    result = apply_global_lock(candidates, capital, existing, cfg)
    assert result.accepted == []
    assert "產業" in result.rejected[0][1]


def test_industry_cap_allows_when_within_budget():
    cfg = GlobalRiskConfig(max_daily_new_risk_pct=1.0, max_industry_exposure_pct=0.10)
    capital = 1_000_000
    existing = {"IND1": 50_000}
    candidates = [_candidate("A", "IND1", potential_loss=1_000, position_value=40_000, liquidity=100)]
    result = apply_global_lock(candidates, capital, existing, cfg)
    assert [c.stock_id for c in result.accepted] == ["A"]
