"""top_n=1~3 集中持股的市場衝擊成本敏感度分析：真實滑價會吃掉多少優勢。

背景：10.11/10.12 節反覆強調同一個保留意見——top_n=1~3 這種高度集中的
動量組合，複委託固定每股手續費模型完全沒有模擬「單一持股占投資組合
20~50%」這種規模的真實下單市場衝擊成本，回測數字可能高估實際可執行的
報酬。這裡不再只是口頭提醒，直接做一次量化的敏感度分析：用每筆交易
當時的成交金額對照該檔股票當時的 20 日均成交金額（ADV），估算「如果
這筆交易的市場衝擊成本跟參與率的平方根成正比」，這筆交易實際上可能
多付出多少滑價成本，然後看扣掉這筆估算成本之後，top_n=1~3 的優勢還
剩多少。

**這是敏感度分析，不是精確校準的模型**：真實的市場衝擊成本取決於很多
這裡沒有資料的因素（訂單分批執行的方式、當下的委託簿深度、是不是用
演算法單慢慢吃、當天新聞流等等），業界文獻對「參與率平方根法則」的
係數估計差異也很大。這裡用三組不同嚴格程度的係數（樂觀/中等/悲觀）
呈現一個範圍，不是宣稱知道真實成本是多少。

模型：對每一筆交易（進場、出場分開算），
    參與率 = 這筆交易的成交金額 / 該股票當時 20 日均成交金額（T-1 為止）
    這筆交易的額外滑價成本 = 成交金額 × 係數 × sqrt(參與率)
係數用「參與率 100% 時的滑價成本（以 bps 表示）」來校準三組情境：
樂觀 50bps、中等 150bps、悲觀 400bps——現有複委託固定每股手續費模型
已經有的成本不受影響，這裡估算的是疊加在既有成本之上的「額外」市場
衝擊成本。

反未來函數：ADV 用 T-1 為止的 20 日均成交金額，不會用到交易當天還沒
發生的成交量。

**已知簡化**：這裡只調整總報酬（原始總報酬 - 估算的總滑價成本佔期初
資金的比例），是一次性的線性調整，不是重建整條每日權益曲線，所以
Sharpe/MDD/CAGR 這裡不重新計算；真正嚴謹的做法需要把每筆滑價成本
併回逐日權益曲線再重算全部指標，這裡先用比較快、比較直觀的版本看
量級對不對，如果初步結果顯示滑價影響很大，才值得投入更多工夫做完整版。

用法：
    python scripts/test_us_momentum_topn_slippage_sensitivity_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import indicators as ind
from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store
from tw_quant.us_config import build_us_config
from tw_quant.us_universe import filter_prices_by_index_membership

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N_GRID = (1, 2, 3, 10)
ADV_WINDOW = 20

# 參與率 100% 時的滑價成本（bps），三組情境
IMPACT_SCENARIOS = {"樂觀": 0.0050, "中等": 0.0150, "悲觀": 0.0400}

IN_SAMPLE_START = "2023-09-19"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)


def _build_adv_lookup(us_prices: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    master = us_prices.sort_values(["stock_id", "date"]).reset_index(drop=True)
    adv = master.groupby("stock_id", sort=False)["turnover_value"].transform(lambda s: s.rolling(ADV_WINDOW).mean())
    adv_lagged = ind.shift_by_group(adv, master, periods=1)
    return {(sid, d): v for sid, d, v in zip(master["stock_id"], master["date"], adv_lagged) if pd.notna(v)}


def _impact_cost(notional: float, adv: float | None, coef: float) -> float:
    if adv is None or adv <= 0 or notional <= 0:
        return 0.0
    participation = notional / adv
    return notional * coef * np.sqrt(participation)


def _analyze(us_prices: pd.DataFrame, base_cfg, top_n: int, start_date, end_date, adv_lookup: dict) -> dict:
    factor_cfg = FactorConfig(
        momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=top_n, ascending=False
    )
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs
    )
    baseline_m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)

    participation_rates = []
    impact_by_scenario = {name: 0.0 for name in IMPACT_SCENARIOS}

    def _add_leg(stock_id, date, price, shares):
        notional = price * shares
        adv = adv_lookup.get((stock_id, pd.Timestamp(date)))
        if adv and adv > 0:
            participation_rates.append(notional / adv)
        for name, coef in IMPACT_SCENARIOS.items():
            impact_by_scenario[name] += _impact_cost(notional, adv, coef)

    for _, t in result.trades.iterrows():
        _add_leg(t["stock_id"], t["entry_date"], t["entry_price"], t["shares"])
        _add_leg(t["stock_id"], t["exit_date"], t["exit_price"], t["shares"])
    for stock_id, pos in result.open_positions.items():
        _add_leg(stock_id, pos.entry_date, pos.entry_price, pos.shares)

    adjusted_returns = {
        name: baseline_m["total_return"] - impact_by_scenario[name] / base_cfg.initial_capital
        for name in IMPACT_SCENARIOS
    }

    return {
        "baseline_total_return": baseline_m["total_return"],
        "n_trades": baseline_m["n_trades"],
        "median_participation": float(np.median(participation_rates)) if participation_rates else 0.0,
        "max_participation": float(np.max(participation_rates)) if participation_rates else 0.0,
        "impact_pct_of_capital": {n: impact_by_scenario[n] / base_cfg.initial_capital for n in IMPACT_SCENARIOS},
        "adjusted_total_return": adjusted_returns,
    }


def _print_period(label: str, us_prices, base_cfg, start_date, end_date, adv_lookup) -> None:
    print(f"\n=== {label} ===")
    header = (
        f"{'top_n':>6} {'原始總報酬':>10} {'中位參與率':>11} {'最大參與率':>11}  "
        + "  ".join(f"滑價後總報酬({name})" for name in IMPACT_SCENARIOS)
    )
    print(header)
    for top_n in TOP_N_GRID:
        r = _analyze(us_prices, base_cfg, top_n, start_date, end_date, adv_lookup)
        adj_str = "  ".join(f"{r['adjusted_total_return'][name]:>14.2%}" for name in IMPACT_SCENARIOS)
        print(
            f"{top_n:>6} {r['baseline_total_return']:>9.2%} {r['median_participation']:>10.2%} "
            f"{r['max_participation']:>10.2%}  {adj_str}"
        )


def main() -> None:
    store = get_data_store()
    us_prices = store.load_us_prices()
    membership = store.load_us_index_membership()

    if us_prices.empty:
        print("資料庫裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    n_rows_before = len(us_prices)
    us_prices = filter_prices_by_index_membership(us_prices, membership)
    n_rows_dropped = n_rows_before - len(us_prices)
    print(
        f"存活者偏差部分修正：依指數加入日期過濾後，丟掉 {n_rows_dropped} / {n_rows_before} 列"
        f"（{n_rows_dropped / n_rows_before:.1%}，不解決被剔除股票完全消失那一半，"
        "見 tw_quant/us_universe.py）\n"
    )

    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    n_stocks = us_prices["stock_id"].nunique()
    print(f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {latest.date()}）")

    print(f"建立 {ADV_WINDOW} 日均成交金額（ADV）查找表（T-1 為止，避免未來函數）...")
    adv_lookup = _build_adv_lookup(us_prices)

    base_cfg = build_us_config()

    print(
        f"固定參數：momentum_window={MOM_WINDOW}、rebalance_freq_days={REBALANCE_FREQ_DAYS}"
        f"（跟 10.5/10.11/10.12 節一致）；持股檔數：{TOP_N_GRID}；"
        f"市場衝擊成本情境（參與率100%時的滑價 bps）：{ {k: f'{v:.2%}' for k, v in IMPACT_SCENARIOS.items()} }"
    )

    _print_period("樣本內（2023-09-19 ~ 資料庫最新日期）", us_prices, base_cfg, IN_SAMPLE_START, None, adv_lookup)
    _print_period("樣本外（2018-09-20 ~ 2023-09-18）", us_prices, base_cfg, None, OOS_END, adv_lookup)

    print(
        "\n（誠實揭露：這裡的「總報酬」調整是線性近似——原始總報酬減去估算的總滑價\n"
        "成本占期初資金的比例，不是重建逐日權益曲線後重新計算，Sharpe/MDD/CAGR\n"
        "這裡沒有一併調整；平方根市場衝擊模型的係數是三組情境性假設，不是針對\n"
        "這些股票實際校準過的數字，真實成本可能落在這個範圍之外；中位數/最大\n"
        "參與率欄位本身就是重要資訊——如果參與率普遍很低，代表這些持股（多半是\n"
        "S&P 500 裡流動性最好的成分股）市場衝擊成本疑慮可能被高估了，不是每一筆\n"
        "集中持股都一樣危險）"
    )


if __name__ == "__main__":
    main()
