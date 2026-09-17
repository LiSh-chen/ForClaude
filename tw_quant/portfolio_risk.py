"""貳、投資組合全域風控 (Global Portfolio Risk Management)

在委託送出「前」對當日所有觸發進場的候選標的做全局鎖檢查：
1. 單日新增曝險上限：當日所有新進場候選的「潛在虧損」總和 <= 總資金 6%
2. 產業集中度上限：單一產業（既有部位 + 當日新進場）曝險 <= 總資金 10%
3. 擁擠篩選器：候選數超過額度時，依 T 日成交金額（流動性）由大至小排序，
   排名在後、無法被額度容納者強制捨棄

實作採「貪婪依序驗證」：候選依流動性排序後逐一嘗試放入，只要still在兩個
額度限制內就接受，額度不足則跳過該檔（繼續檢查排序較後、風險較小的候選是否
仍放得下），而不是整批攔腰砍掉——這樣能在滿足規格「流動性優先」精神的同時，
避免因為一檔大部位卡住額度就浪費掉其餘還放得下的候選。
"""

from __future__ import annotations

from dataclasses import dataclass

from tw_quant.config import GlobalRiskConfig


@dataclass
class TradeCandidate:
    stock_id: str
    industry: str
    strategy: str
    entry_price: float
    stop_price: float
    shares: int
    position_value: float
    potential_loss: float
    liquidity_rank_value: float  # T 日成交金額


@dataclass
class GlobalLockResult:
    accepted: list[TradeCandidate]
    rejected: list[tuple[TradeCandidate, str]]
    daily_new_risk_used: float
    industry_exposure_after: dict[str, float]


def apply_global_lock(
    candidates: list[TradeCandidate],
    total_capital: float,
    existing_industry_exposure: dict[str, float],
    cfg: GlobalRiskConfig,
) -> GlobalLockResult:
    max_daily_risk = total_capital * cfg.max_daily_new_risk_pct
    max_industry = total_capital * cfg.max_industry_exposure_pct

    sorted_candidates = sorted(candidates, key=lambda c: c.liquidity_rank_value, reverse=True)

    accepted: list[TradeCandidate] = []
    rejected: list[tuple[TradeCandidate, str]] = []
    industry_exposure = dict(existing_industry_exposure)
    daily_risk_used = 0.0

    for c in sorted_candidates:
        if daily_risk_used + c.potential_loss > max_daily_risk + 1e-9:
            rejected.append((c, "單日新增曝險上限已滿（6%），依流動性排序捨棄"))
            continue

        current_industry_exposure = industry_exposure.get(c.industry, 0.0)
        if current_industry_exposure + c.position_value > max_industry + 1e-9:
            rejected.append((c, f"產業 {c.industry} 曝險上限已滿（10%），依流動性排序捨棄"))
            continue

        accepted.append(c)
        daily_risk_used += c.potential_loss
        industry_exposure[c.industry] = current_industry_exposure + c.position_value

    return GlobalLockResult(accepted, rejected, daily_risk_used, industry_exposure)
