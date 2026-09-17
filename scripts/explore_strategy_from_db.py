"""在真實資料上搜尋正期望值（正 EV）的參數組合。

背景：`run_backtest_from_db.py`（規格書預設參數）完全不成交，
`run_sensitivity_test_from_db.py`（demo 寬鬆參數）在 3x3 regime 門檻網格
上全部虧損（-6.97% ~ -29.08%），且對門檻參數極度敏感（懸崖警報）。
regime 門檻網格只影響「何時可以進場」，進出場邏輯本身（壓縮濾網 + 點火
突破 + 吊燈停利）在那次測試中固定不變、卻全部虧損，代表問題很可能出在
進出場邏輯本身，不只是大盤時機濾網。

這支腳本把 regime/產業曝險上限固定在寬鬆值（隔離掉「大盤時機」與「全域鎖
單純擋單」這兩個變數），改成網格搜尋進出場邏輯本身的參數：
  - sizing.atr_multiplier      停損/吊燈停利的 ATR 倍數（原始 2.5，太緊可能提早洗出）
  - sizing.chandelier_lookback 吊燈停利回看天數（原始 10，太短會被正常拉回洗掉）
  - squeeze.pr_threshold       壓縮濾網嚴格度（原始 10，太嚴會篩掉太多真訊號）
  - ignition.volume_multiplier 點火量能倍數（原始 2.0，太高會錯過訊號）

每組合都完整計算報酬相關指標：total_return / cagr / max_dd / Sharpe /
Calmar（cagr / max_dd）/ 勝率 / 風報比（平均獲利% / 平均虧損%）/
單筆期望值 EV%（= 勝率*平均獲利% + (1-勝率)*平均虧損%）/ 獲利因子。

搜尋完後：
  1. 印出依 Sharpe 排序的前 N 名（要求最低成交筆數，避免小樣本雜訊）。
  2. 對最佳組合印出完整交易層級統計。
  3. 對最佳組合做「前 67% vs 後 33%」的簡易切分驗證——正式 WFA
     需要 4 年以上歷史，目前資料庫只有 ~3 年，這是資料不夠長時的替代檢查，
     不能取代之後資料累積夠長後的正式 WFA。

用法：
    python scripts/explore_strategy_from_db.py
    python scripts/explore_strategy_from_db.py --top 30
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.config import StrategyConfig
from tw_quant.storage import get_data_store

ATR_MULTIPLIER_GRID = (1.5, 2.0, 2.5, 3.0, 4.0)
CHANDELIER_LOOKBACK_GRID = (10, 15, 20)
SQUEEZE_PR_GRID = (10, 20, 30, 50)
IGNITION_VOL_MULT_GRID = (1.2, 1.5, 2.0)

MIN_TRADES_FOR_RANKING = 20

METRIC_KEYS = (
    "total_return", "cagr", "max_dd", "sharpe", "calmar", "n_trades",
    "win_rate", "annual_trades", "risk_reward_ratio", "ev_pct", "profit_factor",
    "avg_win_pct", "avg_loss_pct", "avg_holding_days",
)


def build_base_config() -> StrategyConfig:
    """固定 regime 與產業曝險上限在寬鬆值，隔離掉「大盤時機」與「全域鎖單純
    擋單」這兩個變數，讓網格搜尋專注在進出場邏輯本身。
    """
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


def trade_stats(trades: pd.DataFrame) -> dict:
    """勝率以外的交易層級統計：平均獲利/虧損報酬率、風報比、單筆期望值 EV%、
    獲利因子、平均持有天數。
    """
    if trades.empty:
        return {
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "risk_reward_ratio": 0.0,
            "ev_pct": 0.0, "profit_factor": 0.0, "avg_holding_days": 0.0,
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades)
    avg_win_pct = wins["pnl_pct"].mean() if not wins.empty else 0.0
    avg_loss_pct = losses["pnl_pct"].mean() if not losses.empty else 0.0  # <= 0

    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    risk_reward_ratio = (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct < 0 else float("inf")
    ev_pct = win_rate * avg_win_pct + (1 - win_rate) * avg_loss_pct

    holding_days = (pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])).dt.days

    return {
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "risk_reward_ratio": risk_reward_ratio,
        "ev_pct": ev_pct,
        "profit_factor": profit_factor,
        "avg_holding_days": holding_days.mean(),
    }


def run_one(prices: pd.DataFrame, margin_short: pd.DataFrame, cfg: StrategyConfig) -> dict:
    """跑一次回測，回傳績效指標 + 交易層級統計（風報比/EV/獲利因子等）合併後的 dict，
    另外保留 result 供呼叫端需要時取用（例如印出實際交易明細）。
    """
    result = run_backtest(prices, margin_short, cfg, historical_mdd=None)
    metrics = summarize_performance(result, cfg.initial_capital)
    metrics["calmar"] = (metrics["cagr"] / metrics["max_dd"]) if metrics["max_dd"] > 0 else 0.0
    metrics.update(trade_stats(result.trades))
    metrics["result"] = result
    return metrics


def _print_metrics_block(metrics: dict, indent: str = "  ") -> None:
    labels = {
        "total_return": "總報酬率",
        "cagr": "年化報酬率 (CAGR)",
        "max_dd": "最大回撤 (MDD)",
        "sharpe": "夏普率 (Sharpe)",
        "calmar": "卡瑪比率 (Calmar = CAGR / MDD)",
        "n_trades": "成交筆數",
        "win_rate": "勝率",
        "annual_trades": "年化交易次數",
        "risk_reward_ratio": "風報比（平均獲利% / 平均虧損%）",
        "ev_pct": "單筆期望值 EV（勝率加權後平均報酬率）",
        "profit_factor": "獲利因子（總獲利 / 總虧損）",
        "avg_win_pct": "平均獲利交易報酬率",
        "avg_loss_pct": "平均虧損交易報酬率",
        "avg_holding_days": "平均持有天數",
    }
    pct_keys = {"total_return", "cagr", "max_dd", "win_rate", "ev_pct", "avg_win_pct", "avg_loss_pct"}
    for k in METRIC_KEYS:
        if k not in metrics:
            continue
        v = metrics[k]
        label = labels.get(k, k)
        if k in pct_keys:
            print(f"{indent}{label}: {v:.2%}")
        elif isinstance(v, float):
            print(f"{indent}{label}: {v:.2f}")
        else:
            print(f"{indent}{label}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    store = get_data_store()
    prices = store.load_prices()
    margin_short = store.load_margin_short()

    if prices.empty:
        print("資料庫裡沒有任何價量資料。", file=sys.stderr)
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔股票的資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）\n"
    )

    base_cfg = build_base_config()

    combos = []
    for atr_mult in ATR_MULTIPLIER_GRID:
        for lookback in CHANDELIER_LOOKBACK_GRID:
            for squeeze_pr in SQUEEZE_PR_GRID:
                for vol_mult in IGNITION_VOL_MULT_GRID:
                    combos.append((atr_mult, lookback, squeeze_pr, vol_mult))

    print(f"總共搜尋 {len(combos)} 組進出場參數組合（regime 與產業曝險上限固定在寬鬆值）...\n")

    rows = []
    for atr_mult, lookback, squeeze_pr, vol_mult in combos:
        cfg = copy.deepcopy(base_cfg)
        cfg.sizing.atr_multiplier = atr_mult
        cfg.sizing.chandelier_lookback = lookback
        cfg.squeeze.pr_threshold = squeeze_pr
        cfg.ignition.volume_multiplier = vol_mult

        metrics = run_one(prices, margin_short, cfg)
        metrics.pop("result")
        row = {
            "atr_mult": atr_mult,
            "lookback": lookback,
            "squeeze_pr": squeeze_pr,
            "vol_mult": vol_mult,
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)

    print(f"=== 搜尋結果：依 Sharpe 排序前 {args.top} 名（要求 n_trades >= {MIN_TRADES_FOR_RANKING}）===")
    header = (
        f"{'atr':>5} {'lb':>4} {'sq_pr':>6} {'vol':>5}  "
        f"{'total_ret':>10} {'cagr':>8} {'max_dd':>8} {'sharpe':>7} {'calmar':>7}  "
        f"{'n_trd':>6} {'win%':>7} {'RR':>6} {'EV%':>8} {'PF':>6}"
    )
    print(header)
    for _, r in ranked.head(args.top).iterrows():
        pf = r["profit_factor"]
        pf_str = "  inf " if pf == float("inf") else f"{pf:6.2f}"
        rr = r["risk_reward_ratio"]
        rr_str = "  inf" if rr == float("inf") else f"{rr:6.2f}"
        print(
            f"{r['atr_mult']:>5.1f} {r['lookback']:>4.0f} {r['squeeze_pr']:>6.0f} {r['vol_mult']:>5.1f}  "
            f"{r['total_return']:>9.2%} {r['cagr']:>7.2%} {r['max_dd']:>7.2%} {r['sharpe']:>7.2f} {r['calmar']:>7.2f}  "
            f"{r['n_trades']:>6.0f} {r['win_rate']:>6.1%} {rr_str} {r['ev_pct']:>7.2%} {pf_str}"
        )
    print(
        "\n（RR = 風報比 = 平均獲利% / 平均虧損%；EV% = 勝率加權後單筆期望報酬率；"
        "PF = 獲利因子 = 總獲利 / 總虧損；calmar = CAGR / MDD）"
    )

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")
        return

    best = ranked.iloc[0]
    print(
        f"\n=== 最佳組合詳細檢視：atr_mult={best['atr_mult']}, lookback={best['lookback']:.0f}, "
        f"squeeze_pr={best['squeeze_pr']:.0f}, vol_mult={best['vol_mult']} ==="
    )
    best_cfg = copy.deepcopy(base_cfg)
    best_cfg.sizing.atr_multiplier = best["atr_mult"]
    best_cfg.sizing.chandelier_lookback = int(best["lookback"])
    best_cfg.squeeze.pr_threshold = best["squeeze_pr"]
    best_cfg.ignition.volume_multiplier = best["vol_mult"]

    full_metrics = run_one(prices, margin_short, best_cfg)
    result = full_metrics.pop("result")
    _print_metrics_block(full_metrics)
    print(f"  尚未平倉部位數: {len(result.open_positions)}")

    # 簡易切分驗證：前 67% vs 後 33%（正式 WFA 需要 4 年以上歷史，
    # 目前資料庫還不夠長，這是資料不夠長時的替代檢查）
    all_dates = sorted(prices["date"].unique())
    split_idx = int(len(all_dates) * 0.67)
    split_date = pd.Timestamp(all_dates[split_idx])
    print(f"\n=== 簡易切分驗證（非正式 WFA，僅供參考）：切分點 {split_date.date()} ===")

    train_prices = prices[prices["date"] < split_date]
    train_margin = margin_short[margin_short["date"] < split_date]
    test_prices = prices[prices["date"] >= split_date]
    test_margin = margin_short[margin_short["date"] >= split_date]

    train_metrics = run_one(train_prices, train_margin, best_cfg)
    train_metrics.pop("result")
    test_metrics = run_one(test_prices, test_margin, best_cfg)
    test_metrics.pop("result")

    print(f"  前段（訓練期，< {split_date.date()}）:")
    _print_metrics_block(train_metrics, indent="    ")
    print(f"  後段（模擬盲測期，>= {split_date.date()}）:")
    _print_metrics_block(test_metrics, indent="    ")


if __name__ == "__main__":
    main()
