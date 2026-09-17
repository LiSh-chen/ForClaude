"""全策略參數集中管理。所有魔術數字對應規格書章節，改參數只改這裡。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegimeConfig:
    """壹、大盤環境判定 (Regime Switch)"""

    breadth_ma_window: int = 20
    breadth_threshold: float = 0.50
    # 敏感度測試網格，規格要求同步測試 45%/55%
    breadth_sensitivity_grid: tuple[float, ...] = (0.45, 0.50, 0.55)

    volume_avg_window: int = 5
    volume_ratio_threshold: float = 1.20
    # 敏感度測試網格，規格要求同步測試 110%/130%
    volume_sensitivity_grid: tuple[float, ...] = (1.10, 1.20, 1.30)


@dataclass
class PoolConfig:
    """貳-A/B 基礎魚池"""

    min_history_days: int = 252  # 排除資料不足新股
    min_avg_volume_lots: int = 2000  # 5 日均量門檻（張）
    trend_ma_window: int = 60  # 站上 60 日均線


@dataclass
class SqueezeFilterConfig:
    """策略 A 壓縮濾網：近 40 日震幅 < 過去 252 日 PR10"""

    range_window: int = 40
    pr_lookback: int = 252
    pr_threshold: float = 10.0  # percentile, 0-100


@dataclass
class IgnitionConfig:
    """點火扣板機（策略 A/B 共用）"""

    breakout_window: int = 60
    volume_avg_window: int = 5
    volume_multiplier: float = 2.0


@dataclass
class StrategyBConfig:
    """策略 B：軋空異常突破"""

    excluded_months: tuple[int, ...] = (3, 4, 6, 7, 8)  # 股東會 + 除權息月份
    margin_short_ratio_pr_lookback: int = 252
    margin_short_ratio_pr_threshold: float = 90.0


@dataclass
class PositionSizingConfig:
    """肆、部位計算與流動性限制"""

    risk_pct_per_trade: float = 0.02
    initial_stop_pct_of_entry: float = 0.95  # 進場價 * 0.95 防呆下限
    atr_window: int = 14  # 規格未指定 ATR 天數，採業界慣例 14 日（可調）
    atr_multiplier: float = 2.5
    chandelier_lookback: int = 10  # 過去 10 日最高價
    lot_size: int = 1000  # 1 張 = 1000 股
    max_position_value_pct_of_capital: float = 0.20  # 市值上限防禦


@dataclass
class GlobalRiskConfig:
    """貳、投資組合全域風控"""

    max_daily_new_risk_pct: float = 0.06  # 單日新增曝險上限（潛在虧損總和）
    max_industry_exposure_pct: float = 0.10  # 單一產業曝險上限


@dataclass
class CostConfig:
    """伍、交易摩擦成本"""

    tax_rate: float = 0.003  # 證交稅（賣出時課徵）
    fee_rate: float = 0.001425  # 手續費（買賣雙邊）
    exit_slippage_ticks: int = 2  # 出場預留 2 檔不利滑價


@dataclass
class WFAConfig:
    """伍、WFA 滾動驗證"""

    train_years: int = 3
    test_years: int = 1
    param_variation_alert_pct: float = 0.30  # 相鄰訓練窗最佳參數變異 > 30% 觸發警報
    min_annual_trades: int = 100  # 大數檢驗門檻
    # 大數檢驗放寬順序：先放寬 PR10->PR15，再降 60 日->40 日
    relaxed_squeeze_pr_threshold: float = 15.0
    relaxed_breakout_window: int = 40
    breakout_window_grid: tuple[int, ...] = (40, 50, 60, 70, 80)


@dataclass
class MDDConfig:
    """陸-2、MDD 熔斷與降級恢復矩陣"""

    circuit_multiplier: float = 1.5  # 觸及歷史 MDD 1.5 倍觸發熔斷
    stage1_trade_count: int = 20  # 第 1~20 筆
    stage1_capital_pct: float = 0.50
    stage2_trade_count: int = 30  # 第 21~50 筆（累計到第 50 筆）
    stage2_capital_pct: float = 0.75
    recompute_trade_count: int = 50  # 第 50 筆後重新校準


@dataclass
class StrategyConfig:
    initial_capital: float = 10_000_000.0
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    squeeze: SqueezeFilterConfig = field(default_factory=SqueezeFilterConfig)
    ignition: IgnitionConfig = field(default_factory=IgnitionConfig)
    strategy_b: StrategyBConfig = field(default_factory=StrategyBConfig)
    sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    global_risk: GlobalRiskConfig = field(default_factory=GlobalRiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    wfa: WFAConfig = field(default_factory=WFAConfig)
    mdd: MDDConfig = field(default_factory=MDDConfig)
