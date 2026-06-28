"""Ports — the abstract interfaces the application depends on. Infrastructure provides the
concrete implementations, so use-cases stay testable and free of vendor/IO details.
"""
from __future__ import annotations
from typing import Protocol, Sequence

from .domain.entities import Anchor


class VLMClient(Protocol):
    """A vision-language model. One `read` = one multimodal turn returning parsed JSON.

    Generic on purpose: the multi-stage extraction agent composes many different prompts
    (context, region, per-chunk, refine, confirm) on top of this single primitive.
    """

    def read(self, system: str, anchors: Sequence[Anchor], image_path: str, user_text: str,
             temperature: float, seed: int = 0, extra_images: Sequence[str] | None = None) -> dict | None:
        """system + (optional) anchors + query image + (optional) zoom crops + user_text -> JSON."""
        ...

    async def aread(self, system: str, anchors: Sequence[Anchor], image_path: str, user_text: str,
                    temperature: float, seed: int = 0,
                    extra_images: Sequence[str] | None = None) -> dict | None:
        """Async variant of `read` — the production pipeline path. Bounded by a shared semaphore."""
        ...

    def infer_text(self, system: str, user_text: str, temperature: float) -> dict | None:
        """Text-only JSON inference (no image) — used by the audit meta-reviewer."""
        ...


class VoteCache(Protocol):
    """Resumable persistence for raw votes, keyed by (frame, expert, anchor signature)."""

    def get(self, key: str) -> list[dict] | None: ...

    def put(self, key: str, payload: dict) -> None: ...
