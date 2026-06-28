"""Prompt assets for concept extraction — system instruction, the 5 perception lenses, and the
query template. Kept separate from logic so prompts can be iterated without touching the pipeline.
"""
from __future__ import annotations

from .domain.concept_schema import concept_guide, json_spec  # canonical clinical vocabulary
from .domain.concepts import annotated_spec

SYSTEM = (
    "You are an expert endoscopist specialising in Barrett's esophagus (BE) surveillance, and a "
    "careful image-quality reviewer. For the QUERY frame, report ONLY directly observable, atomic "
    "visual features as ONE JSON object — do NOT decide whether it is neoplasia, only describe what "
    "you SEE.\n"
    "Use the ordinal 0-4 scale where asked (0 = absent/normal, 4 = marked) and use the FULL range "
    "with fine gradations — never snap to only 0/4. Default to ABSENT unless a feature is clearly "
    "present (false positives are far costlier than misses here). If a feature genuinely cannot be "
    "assessed (obscured, wrong modality, out of view), output \"not_assessable\" rather than "
    "guessing — uncertainty becomes a mask, not noise.\n"
    "Identify the modality FIRST: it flips the colour/vessel distribution (an irregular vascular "
    "pattern means different things under NBI vs white-light). Do NOT let centre-identifying "
    "artifacts (black borders, overlay text/graphics) influence mucosal/vascular judgements.\n"
    "The labelled REFERENCE frames are anchors for what regular vs irregular looks like in this "
    "dataset — calibrate your ratings against them.\n\nFeature definitions:\n" + concept_guide()
)

# Five complementary perception lenses; one per vote so all are exercised (votes_per_expert>=5).
LENSES: tuple[str, ...] = (
    "Focus especially on the MUCOSAL surface/pit pattern (regular vs irregular/effaced).",
    "Focus especially on the VASCULAR pattern (regular network vs irregular/dilated/abnormal).",
    "Focus especially on DEMARCATION and focality (a sharply bounded area unlike the surround).",
    "Consider whether any suspicious appearance is a benign mimic (inflammation, glare, mucus).",
    "Take a holistic gestalt view weighing all features together.",
)

QUERY_TEMPLATE = (
    "^ QUERY FRAME. Describe ONLY this frame's visual features vs the references.\n{lens}\n"
    "Output ONLY JSON (no prose, no fence):\n"
)


def query_prompt(lens: str) -> str:
    return QUERY_TEMPLATE.format(lens=lens) + json_spec()


# --- enhanced single-shot ("quality") prompt — same ONE call, better calibration ----------
# Adds: contrastive-to-reference rating + anti-inference rules (system) and a tier-grouped,
# rubric-annotated schema (query). No extra API cost vs the baseline single prompt.
SYSTEM_QUALITY = SYSTEM + (
    "\n\nRating method: for ORDINAL features, FIRST compare to the NDBE reference frames "
    "(more / same / less irregular), THEN map to the 0-4 scale using the rubric given beside each "
    "feature in the schema. Do NOT infer vessels from redness, or a lesion from colour alone. "
    "Assess each feature on its own evidence; output not_assessable when you genuinely cannot judge."
)

QUALITY_TEMPLATE = (
    "^ QUERY FRAME. Describe ONLY this frame's visual features, calibrated against the reference "
    "frames.\n{lens}\nThe schema is grouped by tier; each ordinal feature lists its 0-4 rubric. "
    "Output ONLY JSON (values only, drop the // comments; no prose, no fence):\n"
)


def quality_query_prompt(lens: str) -> str:
    return QUALITY_TEMPLATE.format(lens=lens) + annotated_spec()


# --- optional self-verification pass -------------------------------------------------------
VERIFY_SYSTEM = (
    "You are auditing a draft reading of this endoscopy frame. You are given the frame and a draft "
    "JSON of atomic visual features. Correct ONLY values that disagree with what the pixels show; "
    "be strict and default to ABSENT/lower when uncertain. Return the corrected JSON object only."
)


def verify_prompt(draft_json: str) -> str:
    return ("^ QUERY FRAME. Draft features to verify and correct:\n" + draft_json +
            "\nReturn the corrected JSON object only:\n" + json_spec())
