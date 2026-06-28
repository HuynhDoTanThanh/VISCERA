"""Concurrency limiters for the async pipeline.

FixedLimiter    — a plain cap (= the old asyncio.Semaphore behaviour).
AdaptiveLimiter — AIMD auto-tuning: grows the in-flight limit while calls succeed with stable
                  latency; multiplicatively backs off on errors or latency inflation. Finds the
                  gateway's sweet spot (e.g. ~256/endpoint for flash3) without manual --workers,
                  and recovers automatically if a gateway degrades mid-run.

Both expose: `async acquire()` and `async release(ok: bool, latency: float)`. Call sites time the
request and report whether it produced a usable result (ok=False for an error/None).
"""
from __future__ import annotations
import asyncio
from collections import deque


class FixedLimiter:
    def __init__(self, limit: int):
        self._sem = asyncio.Semaphore(limit)
        self.limit = limit

    async def acquire(self):
        await self._sem.acquire()

    async def release(self, ok: bool = True, latency: float = 0.0):
        self._sem.release()


class SemaphoreLimiter:
    """Adapts a pre-existing asyncio.Semaphore to the limiter interface (used by offline tools
    that already build their own semaphore and call client.set_semaphore())."""
    def __init__(self, sem: asyncio.Semaphore):
        self._sem = sem
        self.limit = getattr(sem, "_value", 0)

    async def acquire(self):
        await self._sem.acquire()

    async def release(self, ok: bool = True, latency: float = 0.0):
        self._sem.release()


class AdaptiveLimiter:
    """AIMD controller on the concurrency limit.

    - additive increase: +`step` after a clean window (low errors, stable latency)
    - multiplicative decrease: ×`backoff` on errors; ×0.9 on latency inflation
    The limit is re-evaluated every `window` completed calls.
    """

    def __init__(self, start: int = 32, min_limit: int = 8, max_limit: int = 2048,
                 step: int = 16, backoff: float = 0.7, window: int = 30,
                 err_thresh: float = 0.03, lat_inflation: float = 1.6, logger=None):
        self._limit = float(start)
        self._min, self._max = min_limit, max_limit
        self._step, self._backoff = step, backoff
        self._window, self._err_thresh, self._lat_inflation = window, err_thresh, lat_inflation
        self._inflight = 0
        self._cond = asyncio.Condition()
        self._lat = deque(maxlen=window)
        self._errs = deque(maxlen=window)
        self._since = 0
        self._lat_baseline: float | None = None       # best (lowest) median latency seen
        self._log = logger

    @property
    def limit(self) -> int:
        return int(self._limit)

    async def acquire(self):
        async with self._cond:
            while self._inflight >= int(self._limit):
                await self._cond.wait()
            self._inflight += 1

    async def release(self, ok: bool, latency: float):
        async with self._cond:
            self._inflight -= 1
            self._lat.append(latency)
            self._errs.append(0 if ok else 1)
            self._since += 1
            woke = 1
            if self._since >= self._window and len(self._errs) >= self._window:
                woke = self._adjust()
                self._since = 0
            self._cond.notify(max(1, woke))

    def _adjust(self) -> int:
        """Recompute the limit. Returns how many extra waiters to wake (on increase)."""
        err_rate = sum(self._errs) / len(self._errs)
        med = sorted(self._lat)[len(self._lat) // 2]
        prev = int(self._limit)

        if err_rate > self._err_thresh:
            self._limit = max(self._min, self._limit * self._backoff)
        elif self._lat_baseline and med > self._lat_baseline * self._lat_inflation:
            self._limit = max(self._min, self._limit * 0.9)      # queueing → ease off
        else:
            self._limit = min(self._max, self._limit + self._step)
            if self._lat_baseline is None or med < self._lat_baseline:
                self._lat_baseline = med

        new = int(self._limit)
        if self._log and new != prev:
            self._log.info(f"[adaptive] limit {prev}→{new} (err={err_rate:.0%} "
                           f"med_lat={med:.1f}s base={self._lat_baseline or 0:.1f}s)")
        return max(1, new - prev)
