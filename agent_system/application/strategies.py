"""Extraction strategies — how ONE (frame, expert) reading is produced.

SingleShotStrategy : the original 1-prompt-for-all-concepts reading (kept for ablation).
MultiStageStrategy : predict → refine → confirm, chunked into focused prompts. Confident,
                     cross-checked values survive; low-confidence or unconfirmed ones are emitted
                     as `not_assessable` (a mask) so the existing reliability/trust math down-weights
                     them instead of being fed noise.

Both return (vote_values, meta) where vote_values is a schema-conformant {concept: raw} dict that
`normalize_vote`/`Aggregator` consume unchanged.
"""
from __future__ import annotations
import asyncio
import json
from typing import Protocol, Sequence

from ..domain.concept_schema import NA

from ..config import Settings
from ..domain.concepts import (BY_NAME, CONTEXT_QUALITY, DISCRIMINATIVE_ALL,
                               ROBUST_CORE, SURFACE_COLOR, VASCULAR)
from ..domain.entities import Anchor, Frame
from ..ports import VLMClient
from ..prompts import SYSTEM as SINGLE_SYSTEM, SYSTEM_QUALITY, query_prompt, quality_query_prompt
from ..prompts_stages import (CHUNK_FOCUS, CONFIRM_SYSTEM, CONTEXT_SYSTEM, EXPERT, LESION_SYSTEM,
                              REFINE_SYSTEM, REGION_SYSTEM, chunk_prompt, confirm_prompt,
                              context_prompt, lesion_morphology_prompt, refine_prompt, region_prompt)
from ..prompts import LENSES
from ..infrastructure.imaging import crop_region

# focal-lesion sub-group split: presence/demarcation/border read on the FULL frame (whole-field
# context); only Paris/size — which truly need a lesion — are deferred to the conditional crop stage.
FOCAL_CORE = ("demarcation", "colocalization", "lesion_present", "border_sharpness", "overall_suspicion")
FOCAL_MORPH = ("paris_type", "lesion_size")

# the multi-scale crop helps resolution-limited FOCAL features but hurts whole-field concepts
# (surface/demarcation lose context), so it is scoped to these chunks only.
CROP_CHUNKS = {"vascular"}


class ExtractionStrategy(Protocol):
    def read(self, frame: Frame, anchors: Sequence[Anchor], client: VLMClient,
             seed: int) -> tuple[dict | None, dict]:
        ...

    async def aread(self, frame: Frame, anchors: Sequence[Anchor], client: VLMClient,
                    seed: int) -> tuple[dict | None, dict]:
        ...


# ---------------------------------------------------------------------------------------------
class SingleShotStrategy:
    """All concepts in one prompt (the baseline). `seed` cycles the lens + perturbs anchors."""

    def __init__(self, settings: Settings):
        self._s = settings

    def _setup(self, seed):
        lens = LENSES[(seed - 1) % len(LENSES)]
        temp = self._s.base_temp + 0.1 * ((seed - 1) % 3)
        if self._s.single_quality:
            system, prompt = SYSTEM_QUALITY, quality_query_prompt(lens)
        else:
            system, prompt = SINGLE_SYSTEM, query_prompt(lens)
        return lens, temp, system, prompt

    def read(self, frame, anchors, client, seed):
        lens, temp, system, prompt = self._setup(seed)
        values = client.read(system, anchors, frame.path, prompt, temp, seed)
        return values, {"strategy": "single", "quality": self._s.single_quality, "lens": lens}

    async def aread(self, frame, anchors, client, seed):
        lens, temp, system, prompt = self._setup(seed)
        values = await client.aread(system, anchors, frame.path, prompt, temp, seed)
        return values, {"strategy": "single", "quality": self._s.single_quality, "lens": lens}


# ---------------------------------------------------------------------------------------------
def _cell(raw) -> dict:
    """Normalize a stage output entry to {'v':..., 'conf':float}."""
    if isinstance(raw, dict) and "v" in raw:
        try:
            conf = float(raw.get("conf", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        return {"v": raw["v"], "conf": max(0.0, min(1.0, conf)), "why": raw.get("why", "")}
    if raw is None:
        return {"v": NA, "conf": 0.0, "why": ""}
    return {"v": raw, "conf": 1.0, "why": ""}


def _abnormal_direction(name: str, raw) -> bool | None:
    """True/False if the raw value is in the abnormal/present direction, None if unassessable."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in (NA, "na", "")):
        return None
    c = BY_NAME[name]
    try:
        if c.kind == "ordinal":
            return float(raw) >= 2.0
        if c.kind == "binary":
            return str(raw).strip().lower() in ("yes", "true", "1")
        return str(raw) in c.abnormal
    except (TypeError, ValueError):
        return None


class MultiStageStrategy:
    """Context → (region) → chunked reading → (refine) → (confirm)."""

    def __init__(self, settings: Settings):
        self._s = settings

    def read(self, frame, anchors, client, seed):
        s, img, t = self._s, frame.path, self._s.base_temp
        meta: dict = {"strategy": "multistage", "stages": []}

        # Stage A — context & quality
        ctx = client.read(CONTEXT_SYSTEM, [], img, context_prompt(CONTEXT_QUALITY), t, seed) or {}
        meta["stages"].append("context")

        # Stage B — region grounding (+ multi-scale crop of the focal area)
        region, crop = None, None
        if s.ms_region:
            region = client.read(REGION_SYSTEM, anchors, img, region_prompt(), t, seed)
            meta["region"] = region
            meta["stages"].append("region")
            if s.ms_crop and region and region.get("focal_area") in ("yes", True):
                crop = crop_region(img, region.get("bbox"), s.raw_store / "crops", s.ms_crop_size)
                meta["crop"] = bool(crop)
        extra = [crop] if crop else None

        # Stage C — chunked concept reading (vascular chunk always runs; per-concept NA per prompt).
        # The zoomed crop (when present) is shown alongside the full frame for resolution-limited
        # features. Focal MORPHOLOGY (Paris/size/border) is deferred to a conditional sub-stage.
        draft: dict[str, dict] = {n: _cell(ctx.get(n)) for n in CONTEXT_QUALITY}
        groups = [("surface_color", SURFACE_COLOR), ("vascular", VASCULAR),
                  ("focal_lesion", FOCAL_CORE)]
        for key, names in groups:
            use_extra = extra if key in CROP_CHUNKS else None   # crop only for resolution-limited chunks
            res = client.read(EXPERT, anchors, img, chunk_prompt(names, CHUNK_FOCUS[key], ctx, region),
                              t, seed, extra_images=use_extra) or {}
            for n in names:
                draft[n] = _cell(res.get(n))
        meta["stages"].append("chunks")

        # Stage C2 — conditional lesion morphology (only when a lesion / real border exists)
        lesion = str(draft.get("lesion_present", {}).get("v", "")).lower() in ("yes", "true", "1")
        try:
            demarc = float(draft.get("demarcation", {}).get("v")) >= 2
        except (TypeError, ValueError):
            demarc = False
        if lesion or demarc:
            res = client.read(LESION_SYSTEM, anchors, img, lesion_morphology_prompt(FOCAL_MORPH),
                              t, seed, extra_images=extra) or {}
            for n in FOCAL_MORPH:
                draft[n] = _cell(res.get(n))
            meta["stages"].append("morphology")
        else:
            for n in FOCAL_MORPH:
                draft[n] = {"v": NA, "conf": 1.0, "why": "no lesion/border present"}

        # Stage D — refine via cross-concept consistency (discriminative subset)
        if s.ms_refine:
            names = [n for n in DISCRIMINATIVE_ALL if n in draft]
            plain = {n: draft[n]["v"] for n in names}
            ref = client.read(REFINE_SYSTEM, [], img,
                              refine_prompt(json.dumps(plain), names), t, seed)
            if ref:
                for n in names:
                    if n in ref:
                        draft[n] = _cell(ref.get(n))
                meta["stages"].append("refine")

        # Stage E — skeptical confirmation (core discriminative); disagreement => mask
        confirm = {}
        if s.ms_confirm:
            names = [n for n in ROBUST_CORE if n in draft]
            confirm = client.read(CONFIRM_SYSTEM, anchors, img, confirm_prompt(names),
                                  max(0.1, t - 0.2), seed) or {}
            meta["stages"].append("confirm")

        return self._finalize(draft, confirm, meta)

    def _finalize(self, draft, confirm, meta):
        # Apply confidence floor + confirmation agreement → mask the shaky cells.
        s = self._s
        vote: dict = {}
        conf_map: dict = {}
        masked = []
        for n, cell in draft.items():
            v, conf = cell["v"], cell["conf"]
            conf_map[n] = conf
            if conf < s.floor_for(n):           # per-concept floor (vascular lower than default)
                vote[n] = NA
                masked.append(n)
                continue
            if n in confirm:
                cv = _cell(confirm.get(n))["v"]
                d_draft, d_conf = _abnormal_direction(n, v), _abnormal_direction(n, cv)
                if d_draft is not None and d_conf is not None and d_draft != d_conf:
                    vote[n] = NA          # two readings disagree on presence → not assessable
                    masked.append(n)
                    continue
            vote[n] = v
        meta["confidence"] = conf_map
        meta["masked"] = masked
        return vote, meta

    async def aread(self, frame, anchors, client, seed):
        s, img, t = self._s, frame.path, self._s.base_temp
        meta: dict = {"strategy": "multistage", "stages": []}

        # Stage A — context & quality
        ctx = await client.aread(CONTEXT_SYSTEM, [], img, context_prompt(CONTEXT_QUALITY), t, seed) or {}
        meta["stages"].append("context")

        # Stage B — region grounding (+ multi-scale crop of the focal area)
        region, crop = None, None
        if s.ms_region:
            region = await client.aread(REGION_SYSTEM, anchors, img, region_prompt(), t, seed)
            meta["region"] = region
            meta["stages"].append("region")
            if s.ms_crop and region and region.get("focal_area") in ("yes", True):
                crop = crop_region(img, region.get("bbox"), s.raw_store / "crops", s.ms_crop_size)
                meta["crop"] = bool(crop)
        extra = [crop] if crop else None

        # Stage C — chunked concept reading. The 3 groups are independent given (ctx, region),
        # so they run concurrently (within-vote parallelism on top of the cross-vote fan-out).
        draft: dict[str, dict] = {n: _cell(ctx.get(n)) for n in CONTEXT_QUALITY}
        groups = [("surface_color", SURFACE_COLOR), ("vascular", VASCULAR),
                  ("focal_lesion", FOCAL_CORE)]

        async def chunk(key, names):
            use_extra = extra if key in CROP_CHUNKS else None
            res = await client.aread(EXPERT, anchors, img,
                                     chunk_prompt(names, CHUNK_FOCUS[key], ctx, region),
                                     t, seed, extra_images=use_extra) or {}
            return names, res

        for names, res in await asyncio.gather(*[chunk(k, n) for k, n in groups]):
            for n in names:
                draft[n] = _cell(res.get(n))
        meta["stages"].append("chunks")

        # Stage C2 — conditional lesion morphology (only when a lesion / real border exists)
        lesion = str(draft.get("lesion_present", {}).get("v", "")).lower() in ("yes", "true", "1")
        try:
            demarc = float(draft.get("demarcation", {}).get("v")) >= 2
        except (TypeError, ValueError):
            demarc = False
        if lesion or demarc:
            res = await client.aread(LESION_SYSTEM, anchors, img, lesion_morphology_prompt(FOCAL_MORPH),
                                     t, seed, extra_images=extra) or {}
            for n in FOCAL_MORPH:
                draft[n] = _cell(res.get(n))
            meta["stages"].append("morphology")
        else:
            for n in FOCAL_MORPH:
                draft[n] = {"v": NA, "conf": 1.0, "why": "no lesion/border present"}

        # Stage D — refine via cross-concept consistency (discriminative subset)
        if s.ms_refine:
            names = [n for n in DISCRIMINATIVE_ALL if n in draft]
            plain = {n: draft[n]["v"] for n in names}
            ref = await client.aread(REFINE_SYSTEM, [], img,
                                     refine_prompt(json.dumps(plain), names), t, seed)
            if ref:
                for n in names:
                    if n in ref:
                        draft[n] = _cell(ref.get(n))
                meta["stages"].append("refine")

        # Stage E — skeptical confirmation (core discriminative); disagreement => mask
        confirm = {}
        if s.ms_confirm:
            names = [n for n in ROBUST_CORE if n in draft]
            confirm = await client.aread(CONFIRM_SYSTEM, anchors, img, confirm_prompt(names),
                                         max(0.1, t - 0.2), seed) or {}
            meta["stages"].append("confirm")

        return self._finalize(draft, confirm, meta)
