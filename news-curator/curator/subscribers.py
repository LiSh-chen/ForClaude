"""訂閱者名單。

Email 屬於個資,repo 若為公開就不能把名單 commit 進去。來源有兩個,會合併:
  1. 環境變數 NEWS_SUBSCRIBERS (逗號/換行分隔) — GitHub Actions 請放 Secret
  2. 本機檔案 subscribers.txt (每行一個,已列入 .gitignore) — 本機測試用
"""
from __future__ import annotations

import os
import re
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "subscribers.txt"
_EMAIL = re.compile(r"^[^@\s,]+@[^@\s,]+\.[^@\s,]+$")


def valid(email: str) -> bool:
    return bool(_EMAIL.match(email))


def _parse(text: str) -> list[str]:
    seen, out = set(), []
    for tok in re.split(r"[,\n;]", text):
        e = tok.strip().lower()
        if e and not e.startswith("#") and valid(e) and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def load(file: Path = FILE) -> list[str]:
    text = os.environ.get("NEWS_SUBSCRIBERS", "")
    if file.exists():
        text += "\n" + file.read_text(encoding="utf-8")
    return _parse(text)


def add(email: str, file: Path = FILE) -> bool:
    email = email.strip().lower()
    if not valid(email):
        raise ValueError(f"不是有效的 email: {email}")
    current = _parse(file.read_text(encoding="utf-8")) if file.exists() else []
    if email in current:
        return False
    file.write_text("\n".join(current + [email]) + "\n", encoding="utf-8")
    return True


def remove(email: str, file: Path = FILE) -> bool:
    email = email.strip().lower()
    current = _parse(file.read_text(encoding="utf-8")) if file.exists() else []
    if email not in current:
        return False
    file.write_text("".join(f"{e}\n" for e in current if e != email), encoding="utf-8")
    return True
