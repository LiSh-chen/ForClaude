from tw_quant.config import MDDConfig
from tw_quant.mdd import MDDManager, DegradeState


def test_no_breach_when_within_threshold():
    cfg = MDDConfig(circuit_multiplier=1.5)
    m = MDDManager(historical_mdd=0.20, cfg=cfg)
    m.on_equity_update(1000)
    m.on_equity_update(900)  # dd = 10% < 20%*1.5=30%
    assert m.state == DegradeState.NORMAL
    assert m.capital_scale == 1.0


def test_breach_triggers_stage1_50pct():
    cfg = MDDConfig(circuit_multiplier=1.5)
    m = MDDManager(historical_mdd=0.10, cfg=cfg)
    m.on_equity_update(1000)
    m.on_equity_update(850)  # dd = 15% == 10%*1.5
    assert m.state == DegradeState.STAGE1_50PCT
    assert m.capital_scale == 0.5


def test_stays_stage1_if_ev_negative_after_20_trades():
    cfg = MDDConfig(circuit_multiplier=1.5, stage1_trade_count=20, stage2_trade_count=30)
    m = MDDManager(historical_mdd=0.10, cfg=cfg)
    m.on_equity_update(1000)
    m.on_equity_update(850)
    for _ in range(25):
        m.on_trade_closed(pnl=-1, current_equity=800)
    assert m.state == DegradeState.STAGE1_50PCT


def test_promotes_to_stage2_when_stabilized_and_ev_positive():
    cfg = MDDConfig(circuit_multiplier=1.5, stage1_trade_count=20, stage2_trade_count=30)
    m = MDDManager(historical_mdd=0.10, cfg=cfg)
    m.on_equity_update(1000)
    m.on_equity_update(850)
    for _ in range(20):
        m.on_trade_closed(pnl=-1, current_equity=830)
    # 熔斷以來新低是 830；接下來獲利交易讓權益回升到高於這個新低
    m.on_trade_closed(pnl=50, current_equity=900)
    assert m.state == DegradeState.STAGE2_75PCT
    assert m.capital_scale == 0.75


def test_recalibrates_to_normal_after_50_trades():
    cfg = MDDConfig(circuit_multiplier=1.5, stage1_trade_count=20, stage2_trade_count=30, recompute_trade_count=50)
    m = MDDManager(historical_mdd=0.10, cfg=cfg)
    m.on_equity_update(1000)
    m.on_equity_update(850)
    for i in range(50):
        m.on_trade_closed(pnl=10, current_equity=900 + i)
    assert m.state == DegradeState.NORMAL
    assert m.capital_scale == 1.0
    assert m.trades_since_breach == 0
