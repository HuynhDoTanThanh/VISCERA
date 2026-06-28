"""Clinical-concept vocabulary, exposed as a clean domain service.

The canonical definitions live in `cf.concept_schema` (the project's single source of truth, also
used by the existing reliability study). This module re-exports them behind a small, stable
domain API so the rest of agent_system never imports `cf` directly.
"""
from __future__ import annotations

from .concept_schema import (  # noqa: F401  (re-exported as the domain vocabulary)
    BY_NAME,
    CONCEPTS,
    DISCRIMINATIVE,
    normalize_vote,
)

# The robust core confirmed to generalise (held-out val + within-center + cross-anchor robustness).
ROBUST_CORE: tuple[str, ...] = (
    "mucosal_irregularity", "nodularity", "demarcation", "lesion_present",
    "focal_erythema", "surface_effacement", "colocalization",
)

ALL_CONCEPTS: tuple[str, ...] = tuple(c.name for c in CONCEPTS)

# --- chunk groups for multi-stage extraction (focused, related features per prompt) ----
CONTEXT_QUALITY: tuple[str, ...] = (
    "modality", "magnification", "distance", "view", "landmark", "interpretable_fraction",
    "blur", "glare", "exposure", "mucus_bubbles", "debris", "blood",
    "black_border", "overlay_graphics", "dominant_color", "vessels_assessable",
)
SURFACE_COLOR: tuple[str, ...] = (
    "mucosal_pattern_type", "mucosal_irregularity", "nodularity", "depression_ulceration",
    "surface_effacement", "focal_erythema", "color_heterogeneity", "whitish_focal_area",
    "color_change_locality",
)
VASCULAR: tuple[str, ...] = ("vascular_irregularity", "dilated_vessels", "focal_abnormal_vessels")
FOCAL_LESION: tuple[str, ...] = (
    "demarcation", "border_sharpness", "colocalization", "lesion_present",
    "paris_type", "lesion_size", "overall_suspicion",
)

# concepts whose consistency the refine stage enforces
DISCRIMINATIVE_ALL: tuple[str, ...] = tuple(DISCRIMINATIVE)


def is_discriminative(name: str) -> bool:
    return BY_NAME[name].role == "discriminative"


def kind(name: str) -> str:
    return BY_NAME[name].kind


def value_range(name: str) -> str:
    """The allowed raw-value spec for one concept (matches normalize_vote's expectations)."""
    c = BY_NAME[name]
    if c.kind == "ordinal":
        return '<int 0-4 or "not_assessable">'
    if c.kind == "binary":
        return '<"yes" | "no" | "not_assessable">'
    return '"' + "|".join(c.values + ("not_assessable",)) + '"'


def definitions(names) -> str:
    """Human-readable feature definitions for a subset (grounding for a focused prompt)."""
    out = []
    for n in names:
        d = BY_NAME[n].desc
        out.append(f"- {n}: {d}" if d else f"- {n}")
    return "\n".join(out)


# Explicit 0-4 level descriptors for the ordinal concepts (calibrates the scales; shared by the
# single-shot quality prompt and the multi-stage chunks).
RUBRICS: dict[str, str] = {
    "focal_erythema": "0 uniform colour · 1 faint tinge · 2 mild distinct red area · "
                      "3 clearly redder focal area · 4 intense focal erythema",
    "mucosal_irregularity": "0 regular uniform pit/ridge · 1 slightly uneven · 2 mildly irregular · "
                            "3 clearly irregular/distorted · 4 markedly distorted/effaced",
    "vascular_irregularity": "0 regular branching network · 1 mild caliber variation · "
                             "2 focal disruption · 3 clearly irregular/dilated · 4 markedly abnormal",
    "demarcation": "0 none, blends in · 1 vague edge · 2 partial border · 3 mostly sharp bounded · "
                   "4 sharply demarcated, obvious",
    "overall_suspicion": "0 clearly NDBE · 1 probably benign · 2 indeterminate · "
                         "3 probably neoplastic · 4 clearly neoplastic",
}

_TIER_NAMES = {"S": "acquisition context", "A": "image quality / artifacts", "B": "colour",
               "C": "surface / mucosal pattern", "D": "vascular", "E": "demarcation / focality",
               "F": "lesion morphology", "G": "overall"}


def annotated_spec() -> str:
    """Full JSON schema grouped by tier with per-ordinal rubric comments — gives the single-shot
    model structure and calibrated scales without extra API calls."""
    lines, last = [], None
    for c in CONCEPTS:
        if c.tier != last:
            lines.append(f"  // Tier {c.tier} — {_TIER_NAMES.get(c.tier, '')}")
            last = c.tier
        rub = f"   // {RUBRICS[c.name]}" if c.name in RUBRICS else ""
        lines.append(f'  "{c.name}": {value_range(c.name)},{rub}')
    return "{\n" + "\n".join(lines) + "\n}"
