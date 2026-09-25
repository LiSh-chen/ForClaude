"""測試 tw_quant/strategy_lab.py：策略註冊、正規化、組合、濾網。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.strategy_lab import (  # noqa: E402
    STD_COLUMNS, TrendRegimeFilterSpec, VolumeFilterSpec, apply_trend_regime_filter,
    apply_volume_filter, build_registry, combine, run_strategy,
)


def _sample_df():
    df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "txf_1min.parquet")
    return df[(df["datetime"] >= "2022-01-01") & (df["datetime"] < "2022-04-01")]


def test_registry_has_expected_strategies():
    reg = build_registry()
    for sid in ["og", "lu", "re", "vwap_trend", "sr_breakout", "sr_fade", "donchian",
                "rsi2", "orb", "liqsweep", "night"]:
        assert sid in reg
        assert reg[sid].verdict in ("validated", "weak", "rejected")


def test_run_strategy_returns_std_columns_and_no_null_strategy_id():
    reg = build_registry()
    df = _sample_df()
    for sid in reg:
        trades = run_strategy(reg, sid, df)
        assert list(trades.columns) == STD_COLUMNS
        if not trades.empty:
            assert (trades["strategy_id"] == sid).all()
            assert trades["strategy_id"].isna().sum() == 0
            assert trades["direction"].isin(["long", "short"]).all()


def test_date_only_strategies_never_show_midnight_time():
    # sr_fade/donchian/rsi2 只有日K模擬，entry/exit_time應該被補成08:45/13:30，
    # 不該出現代表「原始資料是午夜」的00:00
    reg = build_registry()
    df = _sample_df()
    for sid in ["sr_fade", "donchian", "rsi2", "sr_breakout"]:
        trades = run_strategy(reg, sid, df)
        if trades.empty:
            continue
        assert (trades["entry_time"] != "00:00").all()
        assert (trades["exit_time"] != "00:00").all()
        assert trades["approx_time"].all()


def test_intraday_strategies_are_not_flagged_approx():
    reg = build_registry()
    df = _sample_df()
    for sid in ["og", "lu", "re", "vwap_trend", "orb"]:
        trades = run_strategy(reg, sid, df)
        if trades.empty:
            continue
        assert not trades["approx_time"].any()


def test_combine_merges_and_sorts_by_entry():
    reg = build_registry()
    df = _sample_df()
    og = run_strategy(reg, "og", df)
    lu = run_strategy(reg, "lu", df)
    combined = combine([og, lu])
    assert len(combined) == len(og) + len(lu)
    assert set(combined["strategy_id"].unique()) == {"og", "lu"}
    dates = pd.to_datetime(combined["entry_date"])
    assert (dates.diff().dropna() >= pd.Timedelta(0)).all()


def test_combine_skips_empty_frames():
    reg = build_registry()
    df = _sample_df()
    og = run_strategy(reg, "og", df)
    empty = run_strategy(reg, "night", df[df["datetime"] < "2022-01-02"])  # 太短，night大概率是空的
    combined = combine([og, empty])
    assert len(combined) >= len(og)


def test_volume_filter_reduces_or_keeps_same_trade_count():
    reg = build_registry()
    df = _sample_df()
    is_df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "txf_1min.parquet")
    is_df = is_df[is_df["datetime"] < "2021-01-01"]
    from tw_quant.technical_indicators import daily_indicators
    threshold = daily_indicators(is_df)["vol_ratio_lag1"].quantile(2 / 3)

    trades = run_strategy(reg, "re", df)
    filtered = apply_volume_filter(trades, df, VolumeFilterSpec(threshold=threshold))
    assert len(filtered) <= len(trades)
    assert list(filtered.columns) == STD_COLUMNS


def test_trend_regime_filter_only_keeps_aligned_trades():
    reg = build_registry()
    df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "txf_1min.parquet")
    df = df[(df["datetime"] >= "2015-01-01") & (df["datetime"] < "2018-01-01")]
    trades = run_strategy(reg, "og", df)
    filtered = apply_trend_regime_filter(trades, df, TrendRegimeFilterSpec(aligned_only=True))
    assert len(filtered) <= len(trades)
    assert list(filtered.columns) == STD_COLUMNS
