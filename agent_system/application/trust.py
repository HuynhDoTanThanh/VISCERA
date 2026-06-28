"""TrustScorer + Curator — turn an aggregated frame into calibrated, supervised training labels.

trust(frame, concept) = gate × mask × consensus(reliability, cross_anchor)
  consensus = r × x         when a cross-anchor re-extraction exists (x = 1 − |c_A − c_B|)
            = r × penalty   otherwise
Validated on val: kept labels (trust ≥ 0.6) flip ~1% under a new anchor set; masked ones ~18%.
"""
from __future__ import annotations

import numpy as np

from ..config import Settings
from ..domain.concepts import ALL_CONCEPTS, BY_NAME, ROBUST_CORE, is_discriminative
from ..domain.entities import (ConceptCell, CuratedFrame, Decision, FrameAggregate,
                               TrustedCell, UNLABELED)

# soft role prior on the supervision target (all > 0 when every class is enabled)
ROLE_WEIGHT = {"discriminative": 1.0, "context": 0.8, "quality": 0.8,
               "gestalt": 0.7, "center_cue": 0.6}


def _default_gate(enable_all: bool) -> dict[str, float]:
    if enable_all:                       # supervise every class, role-weighted; core boosted
        return {n: (1.0 if n in ROBUST_CORE else ROLE_WEIGHT[BY_NAME[n].role])
                for n in ALL_CONCEPTS}
    # discriminative-only (legacy): core = 1.0, other discriminative = 0.5
    return {n: (1.0 if n in ROBUST_CORE else 0.5)
            for n in ALL_CONCEPTS if is_discriminative(n)}


class TrustScorer:
    def __init__(self, settings: Settings, gate_weights: dict[str, float] | None = None):
        self._s = settings
        self._gate = gate_weights or _default_gate(settings.enable_all_concepts)

    def cell_trust(self, cell: ConceptCell, gate_w: float, x_agree: float | None) -> float:
        # trust = gate × mask × reliability, optionally × cross-anchor agreement (a stricter ≤1
        # factor that only lowers fragile labels). With the 2-model panel, reliability already is
        # cross-model agreement, so the no-cross-anchor path is NOT extra-penalised
        # (unverified_penalty defaults to 1.0). Role gate is the per-role trust ceiling.
        factor = x_agree if x_agree is not None else self._s.unverified_penalty
        return float(gate_w * cell.mask * cell.reliability * factor)

    def score(self, agg: FrameAggregate, x: dict[str, float] | None) -> dict[str, TrustedCell]:
        out: dict[str, TrustedCell] = {}
        for name, cell in agg.cells.items():
            gate_w = self._gate.get(name, 0.0)
            if gate_w <= 0.0:
                continue
            xa = (1.0 - abs(cell.value - x[name])) if (x and name in x) else None
            t = self.cell_trust(cell, gate_w, xa)
            out[name] = TrustedCell(value=round(cell.value, 4), trust=round(t, 4),
                                    supervise=t >= self._s.trust_supervise)
        return out


class Curator:
    def __init__(self, settings: Settings):
        self._s = settings

    def curate(self, agg: FrameAggregate, cells: dict[str, TrustedCell],
               verified: bool) -> CuratedFrame:
        # All concepts are kept (emit-all); trust is the soft weight. Frame-level decision uses the
        # core, TRUST-WEIGHTED, so low-trust concepts contribute little instead of being dropped.
        core = [cells[k] for k in ROBUST_CORE if k in cells]
        trusts = [c.trust for c in core]
        frame_trust = float(np.mean(trusts)) if trusts else 0.0
        wsum = float(np.sum(trusts))
        suspicion = (float(np.average([c.value for c in core], weights=trusts))
                     if wsum > 0 else 0.0)
        decision = self._decide(agg.frame.label, frame_trust, suspicion)
        return CuratedFrame(frame=agg.frame, decision=decision,
                            frame_trust=round(frame_trust, 4), suspicion=round(suspicion, 4),
                            verified=verified, cells=cells)

    def _decide(self, label: int, frame_trust: float, suspicion: float) -> Decision:
        if frame_trust < self._s.frame_trust_min:
            return Decision.ABSTAIN
        if label == 1:
            return Decision.POSITIVE
        if label == 0:
            return Decision.TRUE_NEGATIVE
        return (Decision.HARD_NEG_CANDIDATE if suspicion >= self._s.suspicion_hi
                else Decision.CONFIDENT_NEGATIVE)
