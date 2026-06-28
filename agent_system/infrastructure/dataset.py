"""DatasetLoader — builds Frame entities from the labelled CSVs and the unlabeled image dir."""
from __future__ import annotations
import csv
from pathlib import Path

from ..config import Settings
from ..domain.entities import UNLABELED, Frame


class DatasetLoader:
    def __init__(self, settings: Settings):
        self._s = settings

    def _labeled(self, csv_path: Path, split: str) -> list[Frame]:
        frames: list[Frame] = []
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if r.get("aug") not in (None, "", "orig"):   # originals only — no aug leakage
                    continue
                frames.append(Frame(path=self._s.abspath(r["path"]), label=int(r["label"]),
                                    center=r.get("center", ""), split=split))
        return frames

    def train(self) -> list[Frame]:
        return self._labeled(self._s.train_csv, "train")

    def val(self) -> list[Frame]:
        return self._labeled(self._s.val_csv, "val")

    def unlabeled(self, limit: int | None = None) -> list[Frame]:
        paths = sorted(str(p) for p in self._s.unlabeled_dir.rglob("*.png"))
        if limit:
            paths = paths[:limit]
        return [Frame(path=p, label=UNLABELED, center="", split="unlabeled") for p in paths]

    def load(self, split: str, limit: int | None = None) -> list[Frame]:
        return {"train": self.train, "val": self.val,
                "unlabeled": lambda: self.unlabeled(limit)}[split]()
