"""Aggregator — fold raw votes (across experts × votes) into per-concept (value, reliability, mask).

reliability r is the agreement across ALL pooled votes; with a multi-MODEL panel it measures
cross-family consensus (genuine independence), not one model agreeing with itself.
"""
from __future__ import annotations

import numpy as np

from ..domain.concepts import BY_NAME, CONCEPTS, normalize_vote
from ..domain.entities import ConceptCell, Frame, FrameAggregate, Vote


class Aggregator:
    def aggregate(self, frame: Frame, votes: list[Vote]) -> FrameAggregate:
        norm = [normalize_vote(v.values) for v in votes]            # [(vals, mask), ...]
        cells: dict[str, ConceptCell] = {}
        for c in CONCEPTS:
            xs = [vals[c.name] for vals, mask in norm if mask[c.name] == 1]
            m = float(np.mean([mask[c.name] for _, mask in norm])) if norm else 0.0
            if xs:
                value = float(np.mean(xs))
                if BY_NAME[c.name].kind == "ordinal":
                    rel = float(max(0.0, 1.0 - 2.0 * np.std(xs)))   # tight spread => reliable
                else:
                    p = float(np.mean(xs))                           # fraction in abnormal direction
                    rel = float(abs(2 * p - 1))                      # 1 = unanimous, 0 = 50/50
            else:
                value, rel = 0.0, 0.0
            cells[c.name] = ConceptCell(value=value, reliability=rel, mask=m)
        return FrameAggregate(frame=frame, cells=cells, n_votes=len(votes))
