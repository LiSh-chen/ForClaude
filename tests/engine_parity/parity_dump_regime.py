"""快速版：只算高波動體制濾網套用在og/lu/re上的parity dump，不重跑
parity_dump.py裡其他20多組（那些跑一次要4-5分鐘，這裡只需要3組新增的）。

用法：
    python tests/engine_parity/parity_dump_regime.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tw_quant.opening_rally_strategy import OpeningRallyConfig, backtest as bt_og
from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest as bt_lu
from tw_quant.strategy_lab import apply_volatility_regime_filter, VolatilityRegimeFilterSpec, _std

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "parity_out"
OUT_DIR.mkdir(exist_ok=True)


def dump(name, records):
    with open(OUT_DIR / f"{name}.json", "w") as f:
        json.dump(records, f)
    print(f"  {name}: {len(records)} 筆")


def std_to_norm(std_df):
    out = []
    for _, r in std_df.iterrows():
        out.append({
            "direction": r["direction"],
            "entryDate": r["entry_date"], "entryTime": r["entry_time"],
            "entryPrice": round(float(r["entry_price"]), 4),
            "exitDate": r["exit_date"], "exitTime": r["exit_time"],
            "exitPrice": round(float(r["exit_price"]), 4),
            "pnlPoints": round(float(r["pnl_points"]), 4), "approxTime": bool(r["approx_time"]),
        })
    return out


def main():
    df = pd.read_parquet(REPO_ROOT / "data" / "txf_1min.parquet")
    print(f"loaded {len(df)} rows")

    og_std = _std(bt_og(df, OpeningRallyConfig()), "og", None, "trading_date", "entry_price",
                   "trading_date", "exit_price", "pnl_points", entry_time_col="entry_dt", exit_time_col="exit_dt")
    og_filtered = apply_volatility_regime_filter(og_std, df, VolatilityRegimeFilterSpec(high_vol_only=True))
    dump("og_highvol_filtered", std_to_norm(og_filtered))

    lu_raw = bt_lu(df, LunchReversalConfig())
    lu_sub = lu_raw[lu_raw["leg"] == "short_lunch_dip"].assign(direction="short")
    lu_std = _std(lu_sub, "lu", "direction", "trading_date", "entry_price",
                   "trading_date", "exit_price", "pnl_points", entry_time_col="entry_dt", exit_time_col="exit_dt")
    lu_filtered = apply_volatility_regime_filter(lu_std, df, VolatilityRegimeFilterSpec(high_vol_only=True))
    dump("lu_highvol_filtered", std_to_norm(lu_filtered))

    re_std = _std(lu_raw[lu_raw["leg"] == "long_afternoon_rebound"], "re", None, "trading_date", "entry_price",
                   "trading_date", "exit_price", "pnl_points", entry_time_col="entry_dt", exit_time_col="exit_dt")
    re_filtered = apply_volatility_regime_filter(re_std, df, VolatilityRegimeFilterSpec(high_vol_only=True))
    dump("re_highvol_filtered", std_to_norm(re_filtered))

    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
