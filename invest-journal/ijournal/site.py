"""把日誌/檢討/績效渲染成靜態網站（相對路徑，可放在任意子路徑，如 GitHub Pages 的 /journal/）。"""
from __future__ import annotations
import html
import re
from pathlib import Path

import markdown

from . import config as C
from .utils import isnum, pct

CSS = """
:root{--bg:#f7f7f4;--fg:#1c2421;--card:#fff;--muted:#68726d;--line:#dcdfda;--accent:#1f6f4a;--up:#c0392b;--down:#1a8f4c}
@media (prefers-color-scheme:dark){:root{--bg:#121614;--fg:#e6eae7;--card:#1a1f1c;--muted:#9aa59f;--line:#2c342f;--accent:#5bbd8a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.7 system-ui,-apple-system,"Noto Sans TC",sans-serif}
header{background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
nav{max-width:1000px;margin:0 auto;padding:10px 16px;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
nav b{margin-right:auto}nav a{color:var(--accent);text-decoration:none;font-weight:600}
main{max-width:1000px;margin:0 auto;padding:12px 16px 60px}
h1{font-size:1.5rem}h2{margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:4px}h3{margin-top:1.6rem}
a{color:var(--accent)}blockquote{margin:10px 0;padding:6px 14px;border-left:4px solid var(--accent);background:var(--card);color:var(--muted)}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:13.5px;margin:10px 0;min-width:100%}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--card)}
code{background:var(--card);padding:1px 4px;border-radius:4px}.muted{color:var(--muted)}
ul.arch{list-style:none;padding:0}ul.arch li{padding:6px 0;border-bottom:1px solid var(--line)}
svg text{fill:var(--fg);font-size:11px}
footer{max-width:1000px;margin:0 auto;padding:0 16px 40px;color:var(--muted);font-size:12.5px}
"""
NAV = [("index.html", "最新日誌"), ("archive.html", "日誌列表"), ("performance.html", "績效追蹤"), ("reviews.html", "檢討報告"), ("methodology.html", "方法論")]


def md2html(text: str) -> str:
    h = markdown.markdown(text, extensions=["tables", "sane_lists"])
    return re.sub(r"<table>", '<div class="tw"><table>', h).replace("</table>", "</table></div>")


def page(title: str, body: str, root: str = "") -> str:
    nav = "".join(f'<a href="{root}{u}">{t}</a>' for u, t in NAV)
    return (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}｜投資研究日誌</title><style>{CSS}</style></head><body>"
            f'<header><nav><b>📒 投資研究日誌</b>{nav}</nav></header><main>{body}</main>'
            f'<footer>本站內容由系統自動產生，僅供研究與教育用途，不構成投資建議；目標價為模型估算，投資有風險。</footer></body></html>')


def _bar_svg(items: list[tuple[str, float]], w=900, h=220) -> str:
    if not items:
        return ""
    m = max(abs(v) for _, v in items) or 0.01
    bw = max(8, min(40, (w - 60) // len(items) - 4))
    mid = h / 2 - 10
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="各期推薦批次平均超額報酬"><line x1="30" x2="{w}" y1="{mid}" y2="{mid}" stroke="#999"/>']
    for i, (lab, v) in enumerate(items):
        x = 70 + i * (bw + 4)
        bh = abs(v) / m * (mid - 20)
        y = mid - bh if v >= 0 else mid
        out.append(f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{max(bh, 1):.1f}" fill="{"#1f6f4a" if v >= 0 else "#c0392b"}"><title>{lab}: {v * 100:+.1f}%</title></rect>')
        if len(items) <= 24:
            out.append(f'<text x="{x + bw / 2}" y="{h - 6}" text-anchor="middle" transform="rotate(-60 {x + bw / 2} {h - 6})">{lab[5:]}</text>')
    out.append(f'<text x="2" y="12">+{m * 100:.0f}%</text></svg>')
    return "".join(out)


def build_site() -> None:
    out = C.path("site")
    jdir, rdir, data = C.path("journal"), C.path("reviews"), C.path("data")
    (out / "journal").mkdir(exist_ok=True)
    (out / "reviews").mkdir(exist_ok=True)
    journals = sorted(jdir.glob("*.md"), reverse=True)
    reviews = sorted(rdir.glob("*.md"), reverse=True)
    # 日誌頁
    for f in journals:
        body = md2html(f.read_text(encoding="utf-8"))
        (out / "journal" / f"{f.stem}.html").write_text(page(f.stem, body, "../"), encoding="utf-8")
    for f in reviews:
        (out / "reviews" / f"{f.stem}.html").write_text(page(f.stem, md2html(f.read_text(encoding="utf-8")), "../"), encoding="utf-8")
    # 首頁
    if journals:
        latest = journals[0]
        (out / "index.html").write_text(page("最新日誌", md2html(latest.read_text(encoding="utf-8"))), encoding="utf-8")
    else:
        (out / "index.html").write_text(page("最新日誌", "<h1>尚無日誌</h1><p>第一份日誌將在排程首次執行後出現。</p>"), encoding="utf-8")
    # 列表
    rows = []
    for f in journals:
        pj = C.load_json(data / "picks" / f"{f.stem}.json") or {"picks": []}
        tk = "、".join(p["ticker"] for p in pj["picks"])
        rows.append(f'<li><a href="journal/{f.stem}.html"><b>{f.stem}</b></a> <span class="muted">{html.escape(tk)}</span></li>')
    (out / "archive.html").write_text(page("日誌列表", f'<h1>日誌列表</h1><ul class="arch">{"".join(rows) or "<li>尚無</li>"}</ul>'), encoding="utf-8")
    rrows = [f'<li><a href="reviews/{f.stem}.html"><b>{f.stem}</b></a></li>' for f in reviews]
    (out / "reviews.html").write_text(page("檢討報告", f'<h1>檢討報告</h1><p class="muted">每週追蹤績效、每月檢討並在樣本足夠時回饋參數。</p><ul class="arch">{"".join(rrows) or "<li>尚無</li>"}</ul>'), encoding="utf-8")
    # 績效
    perf = C.load_json(data / "performance.json") or {}
    body = ["<h1>績效追蹤</h1>"]
    if perf.get("positions"):
        summ = perf["summary"]
        t = ['<div class="tw"><table><tr><th>持有期</th><th>到期檔數</th><th>平均報酬</th><th>平均超額報酬</th><th>勝率</th><th>贏大盤比例</th></tr>']
        for h, s in summ.items():
            if s["n"]:
                t.append(f"<tr><td>{h} 日</td><td>{s['n']}</td><td>{pct(s['avg_ret'], 1, True)}</td><td>{pct(s['avg_alpha'], 1, True)}</td><td>{pct(s['win'], 0)}</td><td>{pct(s['win_alpha'], 0)}</td></tr>")
        t.append("</table></div>")
        body += ["<h2>到期績效</h2>", "".join(t)]
        coh = sorted(perf["cohorts"], key=lambda c: c["date"])
        body += ["<h2>各期推薦批次迄今平均超額報酬</h2>", _bar_svg([(c["date"], c["avg_alpha"]) for c in coh[-40:] if isnum(c["avg_alpha"])])]
        t = ['<h2>全部推薦明細</h2><div class="tw"><table><tr><th>推薦日</th><th>代號</th><th>名稱</th><th>進場價</th><th>目標價</th><th>現價</th><th>報酬</th><th>超額</th><th>天數</th><th>狀態</th></tr>']
        for p in sorted(perf["positions"], key=lambda p: (p["date"], p["ticker"]), reverse=True):
            st = "達標" if p["target_hit"] else "觸停損" if p["stop_hit"] else "持有"
            t.append(f"<tr><td>{p['date']}</td><td>{p['ticker']}</td><td>{html.escape(p['name'])}</td><td>{p['entry']:.2f}</td><td>{p['target']:.2f}</td><td>{p['last_price']:.2f}</td><td>{pct(p['rlast'], 1, True)}</td><td>{pct(p['alast'], 1, True)}</td><td>{p['n_days']}</td><td>{st}</td></tr>")
        t.append("</table></div>")
        body.append("".join(t))
        body.append(f'<p class="muted">更新於 {perf["generated"]}。報酬以推薦日收盤價為進場價，未計成本。</p>')
    else:
        body.append("<p>尚無追蹤資料。</p>")
    (out / "performance.html").write_text(page("績效追蹤", "".join(body)), encoding="utf-8")
    # 方法論
    mp = C.ROOT / "docs" / "methodology.md"
    (out / "methodology.html").write_text(page("方法論", md2html(mp.read_text(encoding="utf-8")) if mp.exists() else "<p>—</p>"), encoding="utf-8")
    print(f"[site] 已建置 {len(journals)} 篇日誌、{len(reviews)} 篇檢討 → {out}")
