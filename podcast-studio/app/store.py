"""以資料夾 + project.json 保存專案狀態（MVP 不用資料庫）。"""
from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from . import config

_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_lock = threading.Lock()


class NotFound(Exception):
    pass


def valid_id(pid: str) -> bool:
    return bool(_ID_RE.match(pid))


def project_dir(pid: str) -> Path:
    if not valid_id(pid):
        raise NotFound(pid)
    return config.data_dir() / pid


def create(title: str, consent_name: str) -> dict[str, Any]:
    pid = uuid.uuid4().hex[:12]
    d = project_dir(pid)
    (d / "samples").mkdir(parents=True)
    (d / "segments").mkdir()
    proj: dict[str, Any] = {
        "id": pid,
        "title": title or "未命名節目",
        "consent": {"name": consent_name},
        "samples": [],
        "voice_id": None,
        "brief": None,
        "script": None,
        "script_confirmed": False,
        "render": {"state": "idle", "done": 0, "total": 0, "error": None},
        "final": None,
    }
    save(proj)
    return proj


def load(pid: str) -> dict[str, Any]:
    f = project_dir(pid) / "project.json"
    if not f.exists():
        raise NotFound(pid)
    with _lock:
        return json.loads(f.read_text("utf-8"))


def save(proj: dict[str, Any]) -> None:
    f = project_dir(proj["id"]) / "project.json"
    tmp = f.with_suffix(".tmp")
    with _lock:
        tmp.write_text(json.dumps(proj, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(f)
