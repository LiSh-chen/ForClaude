"""ffmpeg 音訊處理：樣本標準化、片段串接、響度標準化、輸出 mp3。"""
from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

SAMPLE_RATE = 44100


class AudioError(Exception):
    pass


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise AudioError(p.stderr.decode(errors="ignore")[-500:] or "ffmpeg 失敗")


def duration(path: Path) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(p.stdout.strip())
    except ValueError as e:
        raise AudioError("無法讀取音訊長度，檔案可能損毀或格式不支援") from e


def normalize_sample(src: Path, dst: Path) -> float:
    """任何格式 → 單聲道 44.1kHz mp3，去除直流偏移並稍微壓低底噪。回傳秒數。"""
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
          "-af", "highpass=f=70", "-b:a", "128k", str(dst)])
    return duration(dst)


def _silence_wav(path: Path, ms: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(SAMPLE_RATE * ms / 1000))


def assemble(items: list[tuple[Path, int]], out_mp3: Path, title: str) -> float:
    """items = [(片段音檔, 該片段前的停頓毫秒)]。輸出響度標準化(-16 LUFS)的 mp3，並寫入 AI 生成聲明。"""
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        lines: list[str] = []
        for i, (seg, pause_ms) in enumerate(items):
            if pause_ms > 0:
                sil = tmp / f"s{i}.wav"
                _silence_wav(sil, pause_ms)
                lines.append(f"file '{sil}'")
            wav = tmp / f"a{i}.wav"
            _run(["ffmpeg", "-y", "-v", "error", "-i", str(seg), "-ac", "1", "-ar", str(SAMPLE_RATE),
                  "-c:a", "pcm_s16le", str(wav)])
            lines.append(f"file '{wav}'")
        listfile = tmp / "list.txt"
        listfile.write_text("\n".join(lines), "utf-8")
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listfile),
              "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(SAMPLE_RATE), "-ac", "1", "-b:a", "128k",
              "-metadata", f"title={title}",
              "-metadata", "comment=Voice synthesized by AI with the speaker's consent",
              str(out_mp3)])
    return duration(out_mp3)
