"""Core domain entities — immutable value objects passed between use-cases."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

UNLABELED = -1


@dataclass(frozen=True)
class Frame:
    """One endoscopy image. label: 1=neo, 0=ndbe, -1=unlabeled."""
    path: str
    label: int = UNLABELED
    center: str = ""
    split: str = ""

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def is_labeled(self) -> bool:
        return self.label in (0, 1)


@dataclass(frozen=True)
class Anchor:
    """A labelled reference frame shown in-context. kind: 'neo' | 'ndbe'. hard: a look-alike."""
    path: str
    kind: str
    center: str = ""
    hard: bool = False


@dataclass(frozen=True)
class Vote:
    """One raw model reading: concept -> raw value (str/int) or 'not_assessable'."""
    expert: str
    lens: int
    values: dict


@dataclass(frozen=True)
class ConceptCell:
    """Aggregated reading of ONE concept on ONE frame."""
    value: float        # c: soft value in [0,1]
    reliability: float  # r: vote/expert agreement in [0,1]
    mask: float         # m: assessable fraction in [0,1]


@dataclass(frozen=True)
class FrameAggregate:
    frame: Frame
    cells: dict[str, ConceptCell]
    n_votes: int


class Decision(str, Enum):
    POSITIVE = "POSITIVE"                      # labelled neoplasia
    TRUE_NEGATIVE = "TRUE_NEGATIVE"            # labelled NDBE
    HARD_NEG_CANDIDATE = "HARD_NEG_CANDIDATE"  # unlabeled, looks neoplastic @1% prevalence (FPR lever)
    CONFIDENT_NEGATIVE = "CONFIDENT_NEGATIVE"  # unlabeled, clearly benign
    ABSTAIN = "ABSTAIN"                        # too little trustworthy signal — PU-safe, not mined


@dataclass(frozen=True)
class TrustedCell:
    value: float
    trust: float
    supervise: bool


@dataclass(frozen=True)
class CuratedFrame:
    """The unit written to the training store: a frame + its supervised concept targets."""
    frame: Frame
    decision: Decision
    frame_trust: float
    suspicion: float
    verified: bool                       # cross-anchor confirmation available?
    cells: dict[str, TrustedCell]

    @property
    def supervised(self) -> dict[str, TrustedCell]:
        return {k: v for k, v in self.cells.items() if v.supervise}
