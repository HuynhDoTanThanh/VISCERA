"""Clinical-concept vocabulary — the single source of truth for what the VLM experts extract.

Each concept is:
  - ATOMIC & PERCEPTUAL (colour / geometry / texture / presence) — never "is this neoplasia"
  - CLINICALLY GROUNDED in BING (mucosal & vascular regularity), Paris (lesion morphology),
    ARGOS/Amsterdam dysplasia criteria
  - given an explicit `not_assessable` escape -> uncertainty becomes a MASK channel, not noise
  - ORDINAL (0-4 Likert) for the discriminative items -> averaged across experts/votes into a
    soft value in [0,1]

This module builds the prompt JSON spec AND normalises a raw vote dict into a numeric vector, so
extraction and aggregation agree on the same definitions. Pure stdlib (no I/O, no deps).
"""
from __future__ import annotations
from dataclasses import dataclass, field

# Roles drive how a concept is used downstream:
#   discriminative -> enters the candidate "core" set
#   context        -> conditions interpretation (e.g. modality), not a suspicion signal
#   quality        -> OOD / FPR protection + masking
#   center_cue     -> must be DE-BIASED, never drive suspicion (black boxes / overlays)
#   gestalt        -> the old holistic p_neo; ONE weak feature, never the decision
ROLES = ("discriminative", "context", "quality", "center_cue", "gestalt")

NA = "not_assessable"  # universal escape value; recorded as the mask channel


@dataclass(frozen=True)
class Concept:
    name: str
    tier: str
    role: str
    kind: str                       # "ordinal" | "categorical" | "binary"
    values: tuple = ()              # categorical labels (ordinal uses 0..4, binary uses yes/no)
    desc: str = ""
    abnormal: tuple = field(default_factory=tuple)  # categorical value(s) in the abnormal direction

    def spec(self) -> str:
        """One line of the JSON schema shown to the model."""
        if self.kind == "ordinal":
            rng = '<0-4 or "not_assessable">'
        elif self.kind == "binary":
            rng = '<"yes"|"no"|"not_assessable">'
        else:
            opts = "|".join(self.values + (NA,))
            rng = f'"{opts}"'
        return f'"{self.name}": {rng}'


# --------------------------------------------------------------------------------------
# The vocabulary. Order = display order in the prompt.
# --------------------------------------------------------------------------------------
CONCEPTS: list[Concept] = [
    # Tier S — acquisition context (near-perfect reliability; conditions interpretation)
    Concept("modality", "S", "context", "categorical",
            ("white_light", "virtual_chromo", "dye_chromo"), "imaging mode"),
    Concept("magnification", "S", "context", "binary", desc="near-focus / magnified view"),
    Concept("distance", "S", "context", "categorical", ("close", "medium", "far")),
    Concept("view", "S", "context", "categorical", ("en_face", "tangential", "luminal_overview")),
    Concept("landmark", "S", "context", "categorical",
            ("GEJ_Zline", "gastric_folds", "squamous_island", "tubular_only", "none")),
    Concept("interpretable_fraction", "S", "context", "categorical", ("low", "medium", "high")),

    # Tier A — quality / artifact (reliability ★★★★★; OOD + FPR protection; center cues)
    Concept("blur", "A", "quality", "categorical", ("none", "mild", "severe"), abnormal=("severe",)),
    Concept("glare", "A", "quality", "categorical", ("none", "some", "heavy"), abnormal=("heavy",)),
    Concept("exposure", "A", "quality", "categorical", ("ok", "over", "under"), abnormal=("over", "under")),
    Concept("mucus_bubbles", "A", "quality", "binary"),
    Concept("debris", "A", "quality", "binary"),
    Concept("blood", "A", "quality", "binary"),
    Concept("black_border", "A", "center_cue", "binary", desc="black mask / redaction box (center cue)"),
    Concept("overlay_graphics", "A", "center_cue", "binary", desc="text / markers / overlay (center cue)"),

    # Tier B — color / chromatic (reliability ★★★★; discrim ★★★)
    Concept("dominant_color", "B", "context", "categorical",
            ("salmon_pink", "red", "pale", "nbi_brown_cyan")),
    Concept("focal_erythema", "B", "discriminative", "ordinal",
            desc="reddish area distinct from surrounding mucosa"),
    Concept("color_heterogeneity", "B", "discriminative", "categorical",
            ("uniform", "patchy"), abnormal=("patchy",)),
    Concept("whitish_focal_area", "B", "discriminative", "binary"),
    Concept("color_change_locality", "B", "discriminative", "categorical",
            ("none", "focal", "diffuse"), abnormal=("focal",)),

    # Tier C — surface / mucosal pattern (BING-mucosal; discrim ★★★★)
    Concept("mucosal_pattern_type", "C", "context", "categorical",
            ("flat", "villous_ridged", "circular_pit", "irregular_distorted", "featureless_absent"),
            abnormal=("irregular_distorted", "featureless_absent")),
    Concept("mucosal_irregularity", "C", "discriminative", "ordinal",
            desc="0 regular-uniform pattern -> 4 markedly irregular/distorted (key BING axis)"),
    Concept("nodularity", "C", "discriminative", "binary", desc="raised area / nodule"),
    Concept("depression_ulceration", "C", "discriminative", "binary"),
    Concept("surface_effacement", "C", "discriminative", "binary", desc="loss of surface pattern"),

    # Tier D — vascular (BING-vascular; discrim ★★★★ when assessable)
    Concept("vessels_assessable", "D", "context", "binary", desc="are vessels visible at all"),
    Concept("vascular_irregularity", "D", "discriminative", "ordinal",
            desc="0 regular network -> 4 markedly irregular (key BING axis); NA if not assessable"),
    Concept("dilated_vessels", "D", "discriminative", "binary", desc="dilated / caliber-variable"),
    Concept("focal_abnormal_vessels", "D", "discriminative", "binary",
            desc="vessels in a spot differ from surround"),

    # Tier E — demarcation / focality (the single strongest concept)
    Concept("demarcation", "E", "discriminative", "ordinal",
            desc="sharply demarcated focal area distinct from background BE (0 none -> 4 sharp/obvious)"),
    Concept("border_sharpness", "E", "discriminative", "categorical",
            ("sharp", "gradual"), abnormal=("sharp",)),
    Concept("colocalization", "E", "discriminative", "binary",
            desc="surface AND vascular abnormality in the SAME demarcated spot (very specific)"),

    # Tier F — lesion morphology / Paris (conditional on a lesion; reliable shape call)
    Concept("lesion_present", "F", "discriminative", "binary", desc="discrete focal lesion"),
    Concept("paris_type", "F", "context", "categorical",
            ("0-Is_Ip", "0-IIa", "0-IIb", "0-IIc", "0-III"),
            abnormal=("0-Is_Ip", "0-IIa", "0-IIc", "0-III")),
    Concept("lesion_size", "F", "context", "categorical", ("small", "medium", "large")),

    # Tier G — weak gestalt (ONE feature; never the decision)
    Concept("overall_suspicion", "G", "gestalt", "ordinal",
            desc="holistic suspicion 0-4; intentionally treated as one weak input among many"),
]

BY_NAME = {c.name: c for c in CONCEPTS}
DISCRIMINATIVE = [c.name for c in CONCEPTS if c.role == "discriminative"]
CONTEXT = [c.name for c in CONCEPTS if c.role == "context"]


def json_spec() -> str:
    """The compact JSON object the model must emit (one key per concept)."""
    return "{" + ", ".join(c.spec() for c in CONCEPTS) + "}"


def concept_guide() -> str:
    """Human-readable definitions appended to the system prompt for grounding."""
    return "\n".join(f"- {c.name}: {c.desc}" for c in CONCEPTS if c.desc)


# --------------------------------------------------------------------------------------
# Normalize a raw vote dict -> numeric concept vector in [0,1] + mask (1 = assessable).
# --------------------------------------------------------------------------------------
def normalize_vote(vote: dict) -> tuple[dict, dict]:
    """Return ({name: value in [0,1]}, {name: mask 0/1}) for one expert/vote.

    - ordinal:     v/4              (NA -> mask 0, value 0.0)
    - binary:      yes->1, no->0    (NA -> mask 0)
    - categorical: 1.0 if the value is in the `abnormal` set else 0.0 (the abnormality scalar)
    """
    vals, mask = {}, {}
    for c in CONCEPTS:
        raw = vote.get(c.name, NA)
        if raw == NA or raw is None or (isinstance(raw, str) and raw.strip().lower() in (NA, "na", "")):
            vals[c.name], mask[c.name] = 0.0, 0
            continue
        mask[c.name] = 1
        if c.kind == "ordinal":
            try:
                vals[c.name] = max(0.0, min(1.0, float(raw) / 4.0))
            except (TypeError, ValueError):
                vals[c.name], mask[c.name] = 0.0, 0
        elif c.kind == "binary":
            vals[c.name] = 1.0 if str(raw).strip().lower() in ("yes", "true", "1") else 0.0
        else:  # categorical -> abnormality scalar
            vals[c.name] = 1.0 if str(raw) in c.abnormal else 0.0
    return vals, mask
