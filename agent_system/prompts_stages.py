"""Prompt assets for the MULTI-STAGE extraction agent (v3 — hardened for the hard classes).

Skills applied:
- ordinal RUBRIC anchoring (each 0-4 level defined) → calibrated scales, less snapping
- CONTRASTIVE rating vs the NDBE reference (more/same/less → score)
- MODALITY-conditioned vascular assessment + explicit abstention / anti-inference rules
- MULTI-SCALE: a zoomed crop of the focal area is shown alongside the full frame
- per concept we collect {"v": <schema value>, "conf": 0.0-1.0, "why": "<visual evidence>"}.
"""
from __future__ import annotations

from .domain.concepts import RUBRICS, definitions, value_range

EXPERT = (
    "You are an expert endoscopist (Barrett's surveillance) and a careful image-quality reviewer. "
    "Report ONLY directly observable, atomic visual features — never decide 'is this neoplasia'. "
    "Default to ABSENT/low unless a feature is clearly present (false positives are far costlier "
    "than misses). For ordinal features, FIRST compare to the NDBE reference frame (more / same / "
    "less), THEN map to the 0-4 scale using the rubric. Do NOT infer vessels from redness, or a "
    "lesion from colour alone. If a feature cannot be judged (obscured, wrong modality/resolution, "
    "out of view) output \"not_assessable\" — never guess. When a ZOOMED CROP of the focal area is "
    "provided, judge focal features on the crop, cross-checked against the full frame."
)

def _schema(names) -> str:
    lines = []
    for n in names:
        rng = value_range(n)
        rub = f"   // scale: {RUBRICS[n]}" if n in RUBRICS else ""
        lines.append(f'  "{n}": {{"v": {rng}, "conf": <0.0-1.0>, "why": "<≤12 words>"}}{rub}')
    return "{\n" + ",\n".join(lines) + "\n}"


# --- Stage A — context & quality (modality first; gates everything downstream) -------------
CONTEXT_SYSTEM = (
    "You are a meticulous endoscopy image-quality and acquisition reviewer. Identify the imaging "
    "MODALITY first (it flips colour/vessel meaning), then describe acquisition and quality only. "
    "Do NOT let centre-identifying artifacts (black borders, overlay text) influence anything."
)


def context_prompt(names) -> str:
    return ("Think briefly, then describe ONLY this frame's acquisition context and quality.\n"
            "Output ONLY this JSON (no prose, no fence):\n" + _schema(names))


# --- Stage B — region grounding (returns a normalized bbox for the multi-scale crop) -------
REGION_SYSTEM = EXPERT


def region_prompt() -> str:
    return (
        "Step 1: scan the mucosa. Step 2: decide if ONE area stands out from the surrounding "
        "Barrett's background (different colour, surface, vessels, or a raised/depressed spot).\n"
        "If yes, give its bounding box as normalized [x0,y0,x1,y1] in 0-1 (x→right, y→down).\n"
        "Output ONLY JSON: {\"focal_area\": <\"yes\"|\"no\">, \"bbox\": [x0,y0,x1,y1] or null, "
        "\"extent\": \"<small|medium|large|none>\", \"why\": \"<≤15 words of the visual cue, or "
        "'uniform field'>\"}"
    )


# --- Stage C — focused concept chunks (deep, evidence-grounded, contrastive) ----------------
def chunk_prompt(group_names, focus: str, context: dict | None, region: dict | None) -> str:
    ctx = ""
    if context:
        mod = context.get("modality", {})
        mod = mod.get("v") if isinstance(mod, dict) else mod
        ctx = f"Known context — modality: {mod}. "
    if region:
        ctx += (f"A focal area was reported ({region.get('extent')}); a zoomed crop is provided — "
                "assess focal features THERE. " if region.get("focal_area") in ("yes", True)
                else "No focal area was reported; be sceptical of any focal-feature call. ")
    return (
        f"{focus}\n{ctx}\nFeature definitions:\n{definitions(group_names)}\n\n"
        "For EACH feature: compare to the reference frames, reason from the pixels, give a "
        "calibrated confidence (low if ambiguous), and cite the visual cue in 'why'. Be strict — "
        "default to absent/low; output not_assessable rather than guessing. Output ONLY this JSON "
        "(no prose, no fence):\n" + _schema(group_names)
    )


CHUNK_FOCUS = {
    "surface_color": "Focus on the MUCOSAL surface/pit pattern and COLOUR (regular vs "
                     "irregular/effaced; uniform vs patchy/erythematous/whitish).",
    "vascular": "Focus ONLY on the VASCULAR pattern. MODALITY RULE: under white-light, only rate "
                "vessels if you can trace INDIVIDUAL vessels — otherwise not_assessable (do not "
                "infer from redness). Under NBI/BLI/virtual-chromo, assess the brown intrapapillary "
                "capillary loops and their regularity.",
    "focal_lesion": "Focus on DEMARCATION and focality — a sharply bounded area unlike the "
                    "surround, and whether a discrete lesion is present.",
}


# --- Stage C2 — conditional lesion morphology (only when a lesion/border exists) ------------
LESION_SYSTEM = EXPERT


def lesion_morphology_prompt(names) -> str:
    return (
        "A discrete lesion or a demarcated focal area is present (see the zoomed crop). Describe "
        "ONLY its morphology. Output not_assessable for any field you cannot judge. Output ONLY "
        "this JSON:\n" + _schema(names)
    )


# --- Stage D — refine via cross-concept consistency ----------------------------------------
REFINE_SYSTEM = (
    "You are auditing a draft reading of this frame for INTERNAL CONSISTENCY against what the "
    "pixels show. Correct only values that are wrong or contradictory; keep the rest. Default to "
    "absent/lower when uncertain."
)

CONSISTENCY_RULES = (
    "Consistency rules to enforce (correct only clear violations; do NOT blanket-downgrade):\n"
    "1. colocalization='yes' requires a surface abnormality AND, in the SAME spot, either a "
    "vascular abnormality OR — when vessels are not assessable — a clear colour/demarcation "
    "abnormality. Do not force 'no' merely because vessels are unassessable.\n"
    "2. paris_type / lesion_size are meaningful ONLY if lesion_present='yes' — else not_assessable.\n"
    "3. high demarcation (>=3) should coincide with at least one focal surface/vascular/colour "
    "abnormality; if none, lower it.\n"
    "4. if overall_suspicion is high but every atomic feature is low/absent, reconcile (usually "
    "lower overall_suspicion).\n"
    "5. keep each vascular feature's own assessability judgement; do not null vascular features "
    "you can actually see just because the view is imperfect.\n"
)


def refine_prompt(draft_json: str, names) -> str:
    return (
        "Draft reading to verify and correct:\n" + draft_json + "\n\n" + CONSISTENCY_RULES +
        "\nReturn the corrected reading as ONLY this JSON (same keys):\n" + _schema(names)
    )


# --- Stage E — skeptical confirmation (independent second opinion) -------------------------
CONFIRM_SYSTEM = (
    "You are a SECOND, sceptical endoscopist double-checking only the discriminative features. "
    "Assume the frame is a benign NDBE look-alike unless the evidence is clear. Rate independently."
)


def confirm_prompt(names) -> str:
    return (
        "Independently rate ONLY these features, defaulting to absent/low unless clearly present. "
        "Compare against the reference frames.\nOutput ONLY this JSON:\n" + _schema(names)
    )
