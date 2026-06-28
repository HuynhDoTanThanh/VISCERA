"""Writers.

RawStore      → raw_store/ : per-run manifest (raw votes already cached under raw_labels/).
TrainingStore → training_store/ : a paired image+label dataset —
                  images/<name>.jpg   the frame, converted to JPEG
                  labels/<name>.json  that image's label (concepts + decision + trust)
                one label file per image, matched by file stem.
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from ..config import Settings
from ..domain.entities import Anchor, CuratedFrame


class RawStore:
    def __init__(self, settings: Settings):
        self._s = settings

    def write_run(self, run_id: str, split: str, anchors: list[Anchor], stats: dict) -> Path:
        d = self._s.run_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "split": split,
            "experts": [e.key for e in self._s.experts],
            "votes_per_expert": self._s.votes_per_expert,
            "anchors": [asdict(a) for a in anchors],
            "raw_labels_dir": str(self._s.raw_label_dir),
            "stats": stats,
        }, indent=2))
        return d / "manifest.json"


class TrainingStore:
    """Two folders: images/ (jpg) and labels/ (one json per image)."""

    def __init__(self, settings: Settings):
        self._s = settings
        root = settings.training_store / settings.dataset_name if settings.dataset_name \
            else settings.training_store
        self.root = root
        self.images = root / "images"
        self.labels = root / "labels"
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)

    def write_pair(self, cf: CuratedFrame, jpg_quality: int = 92) -> None:
        """Write images/<stem>.jpg and labels/<stem>.json for one curated frame."""
        stem = Path(cf.frame.name).stem
        img_path = self.images / f"{stem}.jpg"
        if not img_path.exists():
            Image.open(cf.frame.path).convert("RGB").save(img_path, "JPEG", quality=jpg_quality)
        label = {
            "image": f"images/{stem}.jpg",
            "name": stem,
            "split": cf.frame.split,
            "label": cf.frame.label,            # 1=neo, 0=ndbe, -1=unlabeled
            "center": cf.frame.center,
            "decision": cf.decision.value,
            "frame_trust": cf.frame_trust,
            "suspicion": cf.suspicion,
            "verified": cf.verified,
            # ALL concepts emitted (fixed 35-class label). `trust` is the soft weight to train with
            # (down-weight low-trust); `supervise` flags whether it cleared the trust threshold.
            "concepts": {k: {"value": v.value, "trust": v.trust, "supervise": v.supervise}
                         for k, v in cf.cells.items()},
        }
        (self.labels / f"{stem}.json").write_text(json.dumps(label, indent=2))
