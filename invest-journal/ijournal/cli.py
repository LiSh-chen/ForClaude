from __future__ import annotations
import argparse
import datetime as dt
import os


def main():
    ap = argparse.ArgumentParser(prog="ijournal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("daily", help="產生當日研究日誌")
    d.add_argument("--demo", action="store_true", help="使用離線合成資料（只用於驗證流程）")
    d.add_argument("--force", action="store_true")
    d.add_argument("--asof", help="YYYY-MM-DD，截斷資料到該日（回補/測試用）")
    t = sub.add_parser("track", help="追蹤過往推薦的實際績效")
    t.add_argument("--demo", action="store_true")
    r = sub.add_parser("review", help="產生檢討報告")
    r.add_argument("--period", choices=["weekly", "monthly"], default="weekly")
    r.add_argument("--tune", action="store_true", help="依檢討結果調整 params.json（樣本足夠才會生效）")
    r.add_argument("--min-dates", type=int, help="覆寫調參所需最少日期數（測試用）")
    r.add_argument("--date")
    sub.add_parser("site", help="重建靜態網站")
    a = ap.parse_args()

    if a.cmd == "daily":
        from . import config as C
        from .daily import run_daily
        from .providers import DemoProvider, LiveProvider
        prov = DemoProvider(end=dt.date.today()) if a.demo else LiveProvider()
        run_daily(prov, force=a.force, asof=a.asof)
        from .site import build_site
        build_site()
    elif a.cmd == "track":
        from .providers import DemoProvider, LiveProvider
        from .tracker import run_tracker
        run_tracker(DemoProvider(end=dt.date.today()) if a.demo else LiveProvider())
        from .site import build_site
        build_site()
    elif a.cmd == "review":
        from .review import run_review
        run_review(a.period, tune=a.tune, date=a.date, min_dates=a.min_dates)
        from .site import build_site
        build_site()
    elif a.cmd == "site":
        from .site import build_site
        build_site()
