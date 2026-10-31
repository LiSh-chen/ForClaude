import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PODCAST_DATA_DIR", str(tmp_path / "storage"))
    for k in ("ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_GUEST_VOICE_ID"):
        monkeypatch.delenv(k, raising=False)
