"""FileVoteCache — resumable on-disk persistence of raw votes (the 'raw label' store).

One JSON file per (frame, expert, anchor signature). Re-running the pipeline reads existing
files and only calls the model for gaps. Empty-vote files are treated as MISSING so a quota
failure never permanently poisons a frame.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


class FileVoteCache:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(frame_path: str, expert: str, anchor_sig: str) -> str:
        h = f"{Path(frame_path).name}|{expert}|{anchor_sig}"
        return hashlib.sha1(h.encode()).hexdigest()[:16]

    def _file(self, key: str) -> Path:
        return self._root / f"{key}.json"

    def get(self, key: str) -> list[dict] | None:
        f = self._file(key)
        if not f.exists():
            return None
        votes = json.loads(f.read_text()).get("votes", [])
        return votes or None      # empty == treat as missing (retry on next run)

    def put(self, key: str, payload: dict) -> None:
        self._file(key).write_text(json.dumps(payload))
