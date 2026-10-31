"""渲染流程：克隆聲音 → 逐段合成（有快取）→ 串接成品。"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from . import audio, store, tts

_active: set[str] = set()
_active_lock = threading.Lock()


def is_active(pid: str) -> bool:
    with _active_lock:
        return pid in _active


def start(pid: str) -> bool:
    """啟動背景渲染；同一專案已在跑則回傳 False。"""
    with _active_lock:
        if pid in _active:
            return False
        _active.add(pid)
    threading.Thread(target=_run, args=(pid,), daemon=True).start()
    return True


def _seg_key(text: str, emotion: str, voice_id: str, provider: str) -> str:
    return hashlib.sha1(f"{provider}|{voice_id}|{emotion}|{text}".encode()).hexdigest()[:16]


def pause_before(prev: dict | None, seg: dict) -> int:
    if prev is None:
        return 300
    ms = 550 if prev["speaker"] != seg["speaker"] else 380
    if seg["emotion"] == "thoughtful":
        ms += 250
    return ms


def _set(proj: dict, **render) -> None:
    proj["render"].update(render)
    store.save(proj)


def _run(pid: str) -> None:
    try:
        proj = store.load(pid)
        provider = tts.get_provider()
        script = proj["script"]
        proj["final"] = None
        _set(proj, state="running", done=0, error=None, total=0)

        samples = [store.project_dir(pid) / "samples" / s["file"] for s in proj["samples"]]
        if not samples:
            raise tts.TTSError("請先上傳或錄製至少一段聲音樣本")
        if not proj.get("voice_id") or proj.get("voice_provider") != provider.name:
            proj["voice_id"] = provider.clone_voice(samples, f"{proj['title']}-{pid}")
            proj["voice_provider"] = provider.name
            store.save(proj)

        voices = {"host": proj["voice_id"]}
        if any(s["speaker"] == "guest" for s in script["segments"]):
            voices["guest"] = tts.guest_voice_id(provider)

        # 展開成「合成單位」（長句會被切開），先算總數供進度條使用
        units = []
        for seg in script["segments"]:
            for piece in tts.split_text(tts.normalize_for_tts(seg["text"])):
                units.append({**seg, "text": piece})
        _set(proj, total=len(units))

        seg_dir: Path = store.project_dir(pid) / "segments"
        items: list[tuple[Path, int]] = []
        prev = None
        for i, u in enumerate(units):
            vid = voices[u["speaker"]]
            f = seg_dir / f"{_seg_key(u['text'], u['emotion'], vid, provider.name)}.mp3"
            if not f.exists():  # 只重新合成被改過的片段
                f.write_bytes(provider.synthesize(u["text"], u["emotion"], vid))
            items.append((f, pause_before(prev, u)))
            prev = u
            _set(proj, done=i + 1)

        out = store.project_dir(pid) / "final.mp3"
        dur = audio.assemble(items, out, script.get("title") or proj["title"])
        proj["final"] = {"file": "final.mp3", "seconds": round(dur, 1), "mock": provider.is_mock}
        _set(proj, state="done")
    except Exception as e:  # noqa: BLE001 - 任何錯誤都要回報給前端而不是讓執行緒默默死掉
        try:
            proj = store.load(pid)
            _set(proj, state="error", error=str(e))
        except Exception:  # noqa: BLE001
            pass
    finally:
        with _active_lock:
            _active.discard(pid)
