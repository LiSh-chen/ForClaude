"""把精選文章排成電子報 (HTML + 純文字)。"""
from __future__ import annotations

import html
from datetime import date

from .fetch import Article


def subject(arts: list[Article], today: date | None = None) -> str:
    today = today or date.today()
    return f"【精選】{today:%m/%d} 值得讀的 {len(arts)} 則 — {arts[0].one_liner[:24]}…" if arts else f"【精選】{today:%m/%d}"


def render_text(arts: list[Article], unsub_hint: str = "") -> str:
    lines = ["本期只挑真正有料的內容,沒達標寧可不湊數。\n"]
    for i, a in enumerate(arts, 1):
        lines += [f"{i}. {a.title}", f"   {a.one_liner}", f"   為什麼值得讀:{a.why}",
                  f"   {a.source}  ·  價值分 {a.score:g}", f"   {a.url}\n"]
    if unsub_hint:
        lines.append(unsub_hint)
    return "\n".join(lines)


def render_html(arts: list[Article], unsub_html: str = "") -> str:
    e = html.escape
    cards = "".join(f"""
<div style="margin:0 0 22px;padding:0 0 18px;border-bottom:1px solid #e5e5e5">
  <div style="font-size:12px;color:#888">{i:02d} · {e(a.source)} · 價值分 {a.score:g}</div>
  <a href="{e(a.url, quote=True)}" style="font-size:19px;font-weight:600;color:#111;text-decoration:none;line-height:1.4">{e(a.title)}</a>
  <p style="margin:8px 0 6px;color:#333;font-size:15px;line-height:1.6">{e(a.one_liner)}</p>
  <p style="margin:0;color:#0b6b3a;font-size:14px;line-height:1.6"><b>為什麼值得讀</b>:{e(a.why)}</p>
</div>""" for i, a in enumerate(arts, 1))
    return f"""<!doctype html><html><body style="margin:0;background:#f6f6f4">
<div style="max-width:620px;margin:0 auto;padding:28px 20px;background:#fff;font-family:-apple-system,'Noto Sans TC',sans-serif">
  <h1 style="font-size:22px;margin:0 0 4px">精選新聞</h1>
  <p style="color:#666;font-size:13px;margin:0 0 24px">本期只挑真正有料的內容,沒達標寧可不湊數。</p>
  {cards}
  <p style="color:#999;font-size:12px">{unsub_html}</p>
</div></body></html>"""
