"""用 SMTP 寄信。每位訂閱者各寄一封,避免互相看到 email。"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


def build(subject: str, text: str, html_body: str, sender: str, to: str) -> EmailMessage:
    m = EmailMessage()
    m["Subject"], m["From"], m["To"] = subject, sender, to
    # 取消訂閱:回信即可 (一鍵退訂 header,主流信箱會顯示「取消訂閱」按鈕)
    m["List-Unsubscribe"] = f"<mailto:{sender}?subject=unsubscribe>"
    m.set_content(text)
    m.add_alternative(html_body, subtype="html")
    return m


def send_all(subject: str, text: str, html_body: str, recipients: list[str], log=print) -> int:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user, pw = os.environ.get("SMTP_USER", ""), os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("MAIL_FROM") or user
    ok = 0
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls(context=ssl.create_default_context())
        if user:
            s.login(user, pw)
        for to in recipients:
            try:
                s.send_message(build(subject, text, html_body, sender, to))
                ok += 1
            except Exception as e:
                log(f"  ✗ 寄給 {to} 失敗: {e}")
    return ok
