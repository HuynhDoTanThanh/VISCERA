"""LabelPipeline — the top-level use-case: frames → curated training labels.

For each frame (concurrently): extract votes (cached) → aggregate → trust-score → curate.
Raw votes land in the raw store via the cache; curated labels are returned for the training store.
An optional cross-anchor provider supplies the verification signal that makes trust 'best-sure'.
"""
from __future__ import annotations
import asyncio
from typing import Callable, Sequence

from ..config import Settings
from ..domain.entities import Anchor, CuratedFrame, Frame
from .aggregation import Aggregator
from .extraction import Extractor
from .trust import Curator, TrustScorer
from ..infrastructure.limiter import AdaptiveLimiter, FixedLimiter

CrossAnchorProvider = Callable[[Frame], dict | None]


class LabelPipeline:
    def __init__(self, extractor: Extractor, aggregator: Aggregator, scorer: TrustScorer,
                 curator: Curator, settings: Settings, logger=None,
                 cross_anchor: CrossAnchorProvider | None = None):
        self._extract = extractor
        self._agg = aggregator
        self._scorer = scorer
        self._curator = curator
        self._s = settings
        self._log = logger
        self._x = cross_anchor

    async def _aone(self, frame: Frame, anchors: Sequence[Anchor]) -> CuratedFrame | None:
        try:
            votes, _ = await self._extract.aextract(frame, anchors)
            if not votes:
                return None
            agg = self._agg.aggregate(frame, votes)
            x = self._x(frame) if self._x else None
            cells = self._scorer.score(agg, x)
            return self._curator.curate(agg, cells, verified=x is not None)
        except Exception as e:  # noqa: BLE001 — isolate per-frame failures
            if self._log:
                self._log.warning(f"frame failed {frame.name}: {str(e)[:120]}")
            return None

    async def arun(self, frames: list[Frame], anchors: Sequence[Anchor],
                   on_result: Callable[[CuratedFrame], None] | None = None) -> list[CuratedFrame]:
        """Async run. A call-level limiter (set on every client) bounds in-flight VLM requests —
        either a fixed cap (--workers N) or an AIMD-adaptive controller (--workers auto) that
        auto-tunes to the gateways' sweet spot. A frame-level semaphore bounds active frames so
        huge runs don't spawn a coroutine per (frame × expert × vote) all at once."""
        n = max(1, self._s.max_workers)
        if self._s.adaptive_concurrency:
            call_lim = AdaptiveLimiter(start=min(64, n), max_limit=n, logger=self._log)
            if self._log:
                self._log.info(f"[adaptive] auto-concurrency on (start≤64, max {n})")
        else:
            call_lim = FixedLimiter(n)
        frame_sem = asyncio.Semaphore(n)                 # cap active frames → bounds memory
        for client in self._extract.clients.values():
            client.set_limiter(call_lim)

        async def guarded(fr: Frame) -> CuratedFrame | None:
            async with frame_sem:
                return await self._aone(fr, anchors)

        results: list[CuratedFrame] = []
        total = len(frames)
        done = 0
        tasks = [asyncio.create_task(guarded(fr)) for fr in frames]
        try:
            for fut in asyncio.as_completed(tasks):
                cf = await fut
                done += 1
                if cf is not None:
                    results.append(cf)
                    if on_result:
                        on_result(cf)
                if self._log and (done % 25 == 0 or done == total):
                    extra = f" · limit~{call_lim.limit}" if self._s.adaptive_concurrency else ""
                    self._log.info(f"{done}/{total} curated ({len(results)} ok){extra}")
        finally:
            for client in self._extract.clients.values():
                client.set_limiter(None)
        return results
