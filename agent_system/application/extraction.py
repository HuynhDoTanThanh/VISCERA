"""Extractor — produce raw votes for one frame from the expert panel, cached & resumable.

Delegates the actual reading of one (frame, expert) to an ExtractionStrategy (single-shot or the
multi-stage predict→refine→confirm agent), so the extraction protocol is swappable without
touching caching, the panel loop, or downstream aggregation.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Sequence

from ..config import Settings
from ..domain.entities import Anchor, Frame, Vote
from ..ports import VoteCache
from .strategies import ExtractionStrategy


def anchor_signature(anchors: Sequence[Anchor]) -> str:
    return ",".join(Path(a.path).name for a in anchors)


class Extractor:
    def __init__(self, clients: dict, cache: VoteCache, settings: Settings,
                 strategy: ExtractionStrategy):
        self._clients = clients            # {expert_key: VLMClient}
        self._cache = cache
        self._s = settings
        self._strategy = strategy
        # per-expert vote overrides (single-shot): {key: votes} where votes > 0
        self._evotes = {sp.key: sp.votes for sp in settings.experts if sp.votes}

    @property
    def clients(self) -> dict:
        return self._clients

    def _votes_for(self, key: str | None = None) -> int:
        if self._s.extraction_strategy == "multistage":
            return self._s.ms_votes_per_expert
        if key is not None and self._evotes.get(key, 0) > 0:
            return self._evotes[key]
        return self._s.votes_per_expert

    def _cache_ns(self, key: str) -> str:
        ns = f"{key}.{self._s.extraction_strategy}"
        if self._s.extraction_strategy == "multistage":
            ns += f".{self._s.extraction_version}"
        elif self._s.single_quality:
            ns += ".q"
        return ns

    def extract(self, frame: Frame, anchors: Sequence[Anchor]) -> tuple[list[Vote], dict]:
        sig = anchor_signature(anchors)
        all_votes: list[Vote] = []
        raw: dict[str, list[dict]] = {}
        for key, client in self._clients.items():
            # namespace by strategy (and version, for multistage) so readings never collide and a
            # logic change invalidates stale multistage votes
            ck = self._cache.key(frame.path, self._cache_ns(key), sig)
            cached = self._cache.get(ck)
            if cached is not None:
                cached = cached[:self._votes_for(key)]   # honor per-expert vote count
                raw[key] = cached
                all_votes += [Vote(expert=key, lens=i, values=v) for i, v in enumerate(cached)]
                continue
            votes_raw, metas = [], []
            for i in range(self._votes_for(key)):
                values, meta = self._strategy.read(frame, anchors, client, seed=i + 1)
                if not values:
                    continue
                votes_raw.append(values)
                metas.append(meta)
                all_votes.append(Vote(expert=key, lens=i, values=values))
            self._cache.put(ck, {"path": frame.path, "expert": key, "anchor_sig": sig,
                                 "strategy": self._s.extraction_strategy,
                                 "votes": votes_raw, "meta": metas})
            raw[key] = votes_raw
        return all_votes, raw

    async def aextract(self, frame: Frame, anchors: Sequence[Anchor]) -> tuple[list[Vote], dict]:
        """Async extract: experts run concurrently, and each expert's votes are gathered too.
        Cache I/O stays synchronous (fast local JSON). Mirrors `extract`'s lens/cache semantics."""
        sig = anchor_signature(anchors)

        async def for_expert(key, client):
            ck = self._cache.key(frame.path, self._cache_ns(key), sig)
            cached = self._cache.get(ck)
            if cached is not None:
                cached = cached[:self._votes_for(key)]   # honor per-expert vote count
                votes = [Vote(expert=key, lens=i, values=v) for i, v in enumerate(cached)]
                return key, cached, votes
            results = await asyncio.gather(
                *[self._strategy.aread(frame, anchors, client, seed=i + 1)
                  for i in range(self._votes_for(key))])
            votes_raw, metas, votes = [], [], []
            for i, (values, meta) in enumerate(results):
                if not values:
                    continue
                votes_raw.append(values)
                metas.append(meta)
                votes.append(Vote(expert=key, lens=i, values=values))
            self._cache.put(ck, {"path": frame.path, "expert": key, "anchor_sig": sig,
                                 "strategy": self._s.extraction_strategy,
                                 "votes": votes_raw, "meta": metas})
            return key, votes_raw, votes

        per = await asyncio.gather(*[for_expert(k, c) for k, c in self._clients.items()])
        all_votes: list[Vote] = []
        raw: dict[str, list[dict]] = {}
        for key, votes_raw, votes in per:
            raw[key] = votes_raw
            all_votes += votes
        return all_votes, raw
