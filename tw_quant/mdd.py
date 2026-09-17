"""陸-2、MDD 熔斷與降級過渡期 (Graceful Degradation)

規格書條文本身有一處寬嚴不一致，這裡明講並保留原文字面行為，方便使用者依
自己的風險偏好調整：
  - 第 21~50 筆訊號要「恢復至 75%」明確寫了條件（資金曲線回穩且 EV 轉正）；
  - 但「第 50 筆以後」恢復 100% 的敘述沒有附帶條件。
本實作對兩段都採字面解讀：50% -> 75% 需要條件通過；75%（或尚未通過條件而仍在
50%）-> 100% 則在累積滿 50 筆新訊號後無條件觸發並重新校準 MDD 基準。
如果你更保守，希望第 50 筆後也要 EV 為正才能恢復滿水位，把
`MDDManager._maybe_recalibrate` 裡的無條件呼叫改成加上 `self.ev_since_breach() > 0`
即可。

「資金曲線回穩」的量化定義：自熔斷後最新一筆權益沒有再創熔斷以來的新低
（即最近權益 >= 熔斷以來的最低權益）。這是規格書沒有明講、由本實作自行定義
的判準，實盤上線前建議與風控團隊再確認一次。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tw_quant.config import MDDConfig


class DegradeState(Enum):
    NORMAL = "normal"
    STAGE1_50PCT = "stage1_50pct"
    STAGE2_75PCT = "stage2_75pct"


@dataclass
class MDDManager:
    historical_mdd: float  # 歷史（回測）最大回撤，以正數比例表示，如 0.15 代表 -15%
    cfg: MDDConfig

    state: DegradeState = DegradeState.NORMAL
    peak_equity: float = 0.0
    trough_equity_since_breach: float = float("inf")
    trades_since_breach: int = 0
    pnls_since_breach: list[float] = field(default_factory=list)
    breach_count: int = 0

    def __post_init__(self):
        self._initialized_peak = False

    @property
    def capital_scale(self) -> float:
        if self.state == DegradeState.NORMAL:
            return 1.0
        if self.state == DegradeState.STAGE1_50PCT:
            return self.cfg.stage1_capital_pct
        return self.cfg.stage2_capital_pct

    def current_drawdown(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)

    def on_equity_update(self, equity: float) -> None:
        if not self._initialized_peak:
            self.peak_equity = equity
            self._initialized_peak = True
        else:
            self.peak_equity = max(self.peak_equity, equity)

        if self.state != DegradeState.NORMAL:
            self.trough_equity_since_breach = min(self.trough_equity_since_breach, equity)

        if self.state == DegradeState.NORMAL:
            dd = self.current_drawdown(equity)
            if dd >= self.historical_mdd * self.cfg.circuit_multiplier - 1e-9:
                self._trigger_breach(equity)

    def _trigger_breach(self, equity: float) -> None:
        self.state = DegradeState.STAGE1_50PCT
        self.trades_since_breach = 0
        self.pnls_since_breach = []
        self.trough_equity_since_breach = equity
        self.breach_count += 1

    def ev_since_breach(self) -> float:
        if not self.pnls_since_breach:
            return 0.0
        return sum(self.pnls_since_breach) / len(self.pnls_since_breach)

    def equity_curve_stabilized(self, current_equity: float) -> bool:
        """最近權益未創熔斷以來新低，視為回穩（見檔頭說明）。"""
        return current_equity >= self.trough_equity_since_breach

    def on_trade_closed(self, pnl: float, current_equity: float) -> None:
        if self.state == DegradeState.NORMAL:
            return

        self.pnls_since_breach.append(pnl)
        self.trades_since_breach += 1

        if self.state == DegradeState.STAGE1_50PCT and self.trades_since_breach > self.cfg.stage1_trade_count:
            window = self.pnls_since_breach[-self.cfg.stage2_trade_count :]
            ev_positive = (sum(window) / len(window)) > 0
            if ev_positive and self.equity_curve_stabilized(current_equity):
                self.state = DegradeState.STAGE2_75PCT

        self._maybe_recalibrate()

    def _maybe_recalibrate(self) -> None:
        """第 50 筆新訊號後，依這 50 筆重新計算實盤 MDD 基準，恢復 100% 正常部位。"""
        if self.state == DegradeState.NORMAL:
            return
        if self.trades_since_breach >= self.cfg.recompute_trade_count:
            realized_mdd = self.current_drawdown(self.trough_equity_since_breach)
            self.historical_mdd = max(realized_mdd, self.historical_mdd)
            self.state = DegradeState.NORMAL
            self.trades_since_breach = 0
            self.pnls_since_breach = []
            self.trough_equity_since_breach = float("inf")
