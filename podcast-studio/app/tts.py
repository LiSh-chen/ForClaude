"""語音合成與聲音克隆。

Provider 介面：
  clone_voice(sample_paths, name) -> voice_id
  synthesize(text, emotion, voice_id) -> mp3 bytes
目前實作：MockTTS（試跑，用 ffmpeg 產生音調）與 ElevenLabsTTS。
要換成 MiniMax / Fish Audio / 自架模型，只要新增一個同介面的 class。
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Protocol

import httpx

from . import config

# 情緒 → ElevenLabs v3 的 audio tag
EMOTION_TAGS = {
    "neutral": "",
    "excited": "[excited]",
    "curious": "[curious]",
    "thoughtful": "[thoughtful]",
    "warm": "[warmly]",
    "serious": "[serious]",
    "surprised": "[surprised]",
    "laughing": "[laughs]",
}


class TTSError(Exception):
    pass


class Provider(Protocol):
    name: str
    is_mock: bool

    def clone_voice(self, sample_paths: list[Path], name: str) -> str: ...
    def synthesize(self, text: str, emotion: str, voice_id: str) -> bytes: ...


def normalize_for_tts(text: str) -> str:
    """去除 markdown 符號、壓縮空白。數字/英文正規化是之後要加強的地方。"""
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_text(text: str, limit: int = 220) -> list[str]:
    """長句依標點切開，避免單次合成過長造成音色/語氣漂移。"""
    if len(text) <= limit:
        return [text]
    parts = [p for p in re.split(r"(?<=[。！？!?；;])", text) if p.strip()]
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        out.append(cur)
    return out


class MockTTS:
    name = "mock"
    is_mock = True

    def clone_voice(self, sample_paths: list[Path], name: str) -> str:
        return "mock-host"

    def synthesize(self, text: str, emotion: str, voice_id: str) -> bytes:
        freq = 330 if voice_id == "mock-guest" else 220
        freq += {"excited": 40, "surprised": 60, "serious": -30}.get(emotion, 0)
        dur = max(0.8, len(text) * 0.17)
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={dur:.2f}",
             "-af", "volume=0.15,afade=t=in:d=0.05,afade=t=out:st=%.2f:d=0.1" % max(0, dur - 0.1),
             "-ac", "1", "-ar", "44100", "-b:a", "64k", "-f", "mp3", "-"],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise TTSError(proc.stderr.decode(errors="ignore"))
        return proc.stdout


class ElevenLabsTTS:
    name = "elevenlabs"
    is_mock = False
    BASE = "https://api.elevenlabs.io/v1"

    def __init__(self) -> None:
        self.headers = {"xi-api-key": config.elevenlabs_key()}

    def clone_voice(self, sample_paths: list[Path], name: str) -> str:
        files = [("files", (p.name, p.read_bytes(), "audio/mpeg")) for p in sample_paths]
        try:
            r = httpx.post(f"{self.BASE}/voices/add", headers=self.headers,
                           data={"name": name}, files=files, timeout=300)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise TTSError(f"建立克隆聲音失敗（需付費方案的 Instant Voice Clone）：{_err(e)}") from e
        return r.json()["voice_id"]

    def synthesize(self, text: str, emotion: str, voice_id: str) -> bytes:
        tag = EMOTION_TAGS.get(emotion, "")
        payload = {"text": f"{tag} {text}".strip(), "model_id": config.elevenlabs_model()}
        last: Exception | None = None
        for attempt in range(3):
            try:
                r = httpx.post(f"{self.BASE}/text-to-speech/{voice_id}", params={"output_format": "mp3_44100_128"},
                               headers=self.headers, json=payload, timeout=180)
                if r.status_code in (429, 500, 502, 503):
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                return r.content
            except httpx.HTTPError as e:
                last = e
                time.sleep(2 ** attempt)
        raise TTSError(f"語音合成失敗：{_err(last)}")


def _err(e: Exception | None) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        return f"{e.response.status_code} {e.response.text[:200]}"
    return str(e)


def get_provider() -> Provider:
    return ElevenLabsTTS() if config.elevenlabs_key() else MockTTS()


def guest_voice_id(provider: Provider) -> str:
    if provider.is_mock:
        return "mock-guest"
    vid = config.elevenlabs_guest_voice()
    if not vid:
        raise TTSError("雙人對談需要設定 ELEVENLABS_GUEST_VOICE_ID（搭檔使用的現成聲音）")
    return vid
