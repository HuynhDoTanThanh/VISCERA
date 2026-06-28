"""AnchorSelector — choose the in-context few-shot reference frames.

Anchors must be UNAMBIGUOUS (a wrong anchor poisons every label). We score each candidate by
sureness = agreement × assessable × label_consistency × quality and take the surest neo + ndbe.
Crucially we also include 1–2 HARD-NEGATIVE anchors: NDBE frames that *look* suspicious yet are
confidently benign — they calibrate the model against the dominant failure mode (over-calling
concepts on true negatives, the #1 false-positive driver).
"""
from __future__ import annotations

import numpy as np

from ..domain.concepts import ROBUST_CORE
from ..domain.entities import Anchor, FrameAggregate

_QUALITY = ("blur", "glare", "exposure", "debris", "mucus_bubbles")


def _sureness(agg: FrameAggregate, label: int) -> tuple[float, float]:
    agree = np.mean([agg.cells[c].reliability for c in ROBUST_CORE])
    assess = np.mean([agg.cells[c].mask for c in ROBUST_CORE])
    susp = float(np.mean([agg.cells[c].value for c in ROBUST_CORE]))
    consistency = susp if label == 1 else (1.0 - susp)
    quality = 1.0 - np.mean([agg.cells[q].value for q in _QUALITY if q in agg.cells])
    return float(agree * assess * consistency * quality), susp


class AnchorSelector:
    def __init__(self, n_neo: int = 2, n_ndbe: int = 3, n_hard: int = 1):
        self._n_neo, self._n_ndbe, self._n_hard = n_neo, n_ndbe, n_hard

    def select(self, aggregates: list[FrameAggregate]) -> list[Anchor]:
        neo, ndbe = [], []
        for agg in aggregates:
            if not agg.frame.is_labeled:
                continue
            s, susp = _sureness(agg, agg.frame.label)
            rec = {"agg": agg, "sureness": s, "suspicion": susp}
            (neo if agg.frame.label == 1 else ndbe).append(rec)

        neo.sort(key=lambda r: -r["sureness"])
        # easy (clearly benign) ndbe: low suspicion, high sureness
        easy = sorted(ndbe, key=lambda r: -r["sureness"])
        # hard negatives: confidently-read NDBE with the HIGHEST suspicion (look-alikes)
        hard = sorted([r for r in ndbe if r["sureness"] > np.median([x["sureness"] for x in ndbe])],
                      key=lambda r: -r["suspicion"])

        chosen: list[Anchor] = []
        chosen += [self._anchor(r, "neo", False) for r in neo[: self._n_neo]]
        hard_pick = hard[: self._n_hard]
        chosen += [self._anchor(r, "ndbe", True) for r in hard_pick]
        taken = {r["agg"].frame.path for r in hard_pick}
        for r in easy:
            if len(chosen) >= self._n_neo + self._n_ndbe:
                break
            if r["agg"].frame.path not in taken:
                chosen.append(self._anchor(r, "ndbe", False))
        return chosen

    @staticmethod
    def _anchor(rec: dict, kind: str, hard: bool) -> Anchor:
        fr = rec["agg"].frame
        return Anchor(path=fr.path, kind=kind, center=fr.center, hard=hard)
