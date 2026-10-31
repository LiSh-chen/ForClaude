"""環境變數設定。沒有填 API key 時自動進入「試跑模式」（假劇本 / 假聲音）。"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    p = Path(os.environ.get("PODCAST_DATA_DIR", BASE_DIR / "storage"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def anthropic_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def anthropic_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")


def elevenlabs_key() -> str:
    return os.environ.get("ELEVENLABS_API_KEY", "").strip()


def elevenlabs_model() -> str:
    return os.environ.get("ELEVENLABS_MODEL", "eleven_v3")


def elevenlabs_guest_voice() -> str:
    """雙人對談時 AI 搭檔使用的現成聲音 voice_id。"""
    return os.environ.get("ELEVENLABS_GUEST_VOICE_ID", "").strip()
