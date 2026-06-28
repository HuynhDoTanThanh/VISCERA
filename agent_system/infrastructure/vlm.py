"""ProxyVLMClient — VLMClient backed by an OpenAI-compatible gateway (the local 9router proxy at
/v1/chat/completions, serving the gemini families; the cloud gateway for Claude). Builds the
multimodal prompt, invokes, and parses robust JSON. The only place that talks to a model.
"""
from __future__ import annotations
import asyncio
import base64
import itertools
import json
import mimetypes
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import ModelSpec, Settings
from ..domain.entities import Anchor
from .limiter import SemaphoreLimiter


@lru_cache(maxsize=4096)
def _encode(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def _image_block(path: str) -> dict:
    # OpenAI chat-completions multimodal format: a base64 data URL under image_url.
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{_encode(path)}"}}


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def parse_json(content) -> dict | None:
    """Tolerant JSON extraction from a model response (handles fences, prose, trailing commas)."""
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content
                           if isinstance(b, dict) and b.get("type") == "text")
    text = re.sub(r"```(?:json)?", "", str(content).strip()).strip().strip("`").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    for candidate in (m.group(0), re.sub(r",\s*([}\]])", r"\1", m.group(0))):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


class ProxyVLMClient:
    """Implements the VLMClient port for one expert (model)."""

    def __init__(self, spec: ModelSpec, settings: Settings):
        self._spec = spec
        self._s = settings
        base_url = settings.cloud_url if spec.is_cloud else settings.proxy_url
        api_key = settings.cloud_key if spec.is_cloud else settings.proxy_key
        # base_url may be a comma-separated list of gateway endpoints — calls round-robin across
        # them so one run spreads load over N gateways (each saturates ~256 concurrency / ~500 cpm).
        self._endpoints = [u.strip() for u in base_url.split(",") if u.strip()] or [base_url]
        self._api_key = api_key
        self._rr = itertools.count()                     # next() is atomic under the GIL
        # One ChatOpenAI per (endpoint, temperature): reuses httpx keep-alive connections.
        self._clients: dict[tuple[int, float], ChatOpenAI] = {}
        self._clients_lock = threading.Lock()
        # Concurrency limiter (bounds in-flight calls; fixed or AIMD-adaptive). Set by the pipeline
        # before an async run; None on the sync path (offline tools).
        self._limiter = None

    def set_semaphore(self, sem) -> None:
        """Back-compat: wrap a raw asyncio.Semaphore (offline tools call this)."""
        self._limiter = SemaphoreLimiter(sem) if sem is not None else None

    def set_limiter(self, limiter) -> None:
        self._limiter = limiter

    @property
    def key(self) -> str:
        return self._spec.key

    def _client(self, temperature: float, endpoint_idx: int = 0,
                model: str | None = None) -> ChatOpenAI:
        model = model or self._spec.model
        ckey = (endpoint_idx, temperature, model)
        client = self._clients.get(ckey)
        if client is None:
            with self._clients_lock:
                client = self._clients.get(ckey)
                if client is None:
                    client = ChatOpenAI(
                        model=model, base_url=self._endpoints[endpoint_idx],
                        api_key=self._api_key, max_tokens=self._s.max_tokens,
                        timeout=self._s.timeout_s, temperature=temperature)
                    self._clients[ckey] = client
        return client

    def _models(self) -> tuple[str, ...]:
        """Primary model, then the fallback (if configured) — tried in order on failure."""
        if self._spec.fallback_model:
            return (self._spec.model, self._spec.fallback_model)
        return (self._spec.model,)

    def _next_endpoint(self) -> int:
        return next(self._rr) % len(self._endpoints)

    def _anchor_blocks(self, anchors: Sequence[Anchor], seed: int) -> list[dict]:
        import numpy as np
        order = list(anchors)
        if seed:
            rng = np.random.default_rng(seed)
            order = [order[i] for i in rng.permutation(len(order))]
        blocks: list[dict] = []
        for a in order:
            tag = "NEOPLASIA (positive)" if a.kind == "neo" else "NDBE / non-dysplastic (negative)"
            if a.hard:
                tag += " — looks suspicious but is BENIGN (calibrate against over-calling)"
            blocks += [_image_block(a.path), _text_block(f"^ REFERENCE — label: {tag}")]
        return blocks

    def _read_blocks(self, system: str, anchors: Sequence[Anchor], image_path: str,
                     user_text: str, seed: int, extra_images: Sequence[str] | None) -> list[dict]:
        """Build the multimodal prompt. Encodes images (base64) — call inside the permit."""
        blocks = [_text_block(system)] + self._anchor_blocks(anchors, seed)
        blocks += [_image_block(image_path), _text_block("^ QUERY FRAME (full)")]
        for xp in (extra_images or []):
            blocks += [_image_block(xp), _text_block("^ ZOOMED CROP of the focal area (same frame)")]
        blocks.append(_text_block(user_text))
        return blocks

    def _invoke(self, blocks: list[dict], temperature: float, endpoint_idx: int = 0) -> dict | None:
        # Try the primary model with retries; if it never yields parseable JSON, fall back to the
        # spec's fallback_model (if any) with its own retries.
        for model in self._models():
            for _ in range(1 + self._s.vote_retries):
                try:
                    resp = self._client(temperature, endpoint_idx, model).invoke(
                        [HumanMessage(content=blocks)])
                except Exception:  # noqa: BLE001 — network/model error; retry then give up
                    continue
                parsed = parse_json(resp.content)
                if parsed is not None:
                    return parsed
        return None

    async def _ainvoke(self, blocks: list[dict], temperature: float, endpoint_idx: int = 0) -> dict | None:
        for model in self._models():
            for _ in range(1 + self._s.vote_retries):
                try:
                    resp = await self._client(temperature, endpoint_idx, model).ainvoke(
                        [HumanMessage(content=blocks)])
                except Exception:  # noqa: BLE001 — network/model error; retry then give up
                    continue
                parsed = parse_json(resp.content)
                if parsed is not None:
                    return parsed
        return None

    # --- VLMClient port (sync — used by offline tools) ----------------------------------
    def read(self, system: str, anchors: Sequence[Anchor], image_path: str, user_text: str,
             temperature: float, seed: int = 0, extra_images: Sequence[str] | None = None) -> dict | None:
        blocks = self._read_blocks(system, anchors, image_path, user_text, seed, extra_images)
        return self._invoke(blocks, temperature, self._next_endpoint())

    def infer_text(self, system: str, user_text: str, temperature: float) -> dict | None:
        return self._invoke([_text_block(system), _text_block(user_text)], temperature,
                            self._next_endpoint())

    # --- async port (the production pipeline) -------------------------------------------
    async def aread(self, system: str, anchors: Sequence[Anchor], image_path: str, user_text: str,
                    temperature: float, seed: int = 0,
                    extra_images: Sequence[str] | None = None) -> dict | None:
        # Acquire the concurrency permit FIRST, then build the heavy (base64) blocks, so coroutines
        # parked on the semaphore stay cheap and peak memory tracks the permit count, not frame count.
        idx = self._next_endpoint()
        lim = self._limiter
        if lim is None:
            blocks = self._read_blocks(system, anchors, image_path, user_text, seed, extra_images)
            return await self._ainvoke(blocks, temperature, idx)
        # Acquire a permit FIRST, then build the heavy (base64) blocks, so coroutines parked on the
        # limiter stay cheap. Report (ok, latency) so the adaptive controller can tune concurrency.
        await lim.acquire()
        t0 = time.monotonic()
        ok = False
        try:
            blocks = self._read_blocks(system, anchors, image_path, user_text, seed, extra_images)
            res = await self._ainvoke(blocks, temperature, idx)
            ok = res is not None
            return res
        finally:
            await lim.release(ok, time.monotonic() - t0)
