"""Agentic faithfulness audit — independent confirmation that cached labels match the pixels.

The statistical gates (confirm.py) prove labels are reliable, generalizing and not a center
shortcut, but not that they are FAITHFUL to the image (a panel can agree on a wrong reading). This
re-grades a stratified sample with a fresh, strict prompt and measures agreement with the cached
labels, then an LLM meta-reviewer turns gates + faithfulness into a GO / GO_WITH_CAVEATS / NO_GO.
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..domain.concept_schema import BY_NAME, NA
from ..domain.entities import Frame, FrameAggregate
from ..ports import VLMClient

BOUNDARY_LO, BOUNDARY_HI = 0.35, 0.65

ADJ_SYSTEM = (
    "You are a senior endoscopist auditing an AI panel's reading of a Barrett's surveillance frame. "
    "Look ONLY at the image. For each listed feature output a calibrated 0.0-1.0 score for how "
    "strongly it is actually visible (0 = clearly absent, 1 = clearly marked). Be strict and "
    "literal — do not infer neoplasia, only score what the pixels show. Output ONLY the JSON object."
)

META_SYSTEM = (
    "You are a meticulous ML reviewer deciding whether VLM-extracted clinical concepts are "
    "trustworthy enough to SUPERVISE a neoplasia detector at ~1% prevalence (confident false "
    "positives are catastrophic). You get (a) a statistical confirmation table and (b) an "
    "independent re-grade's faithfulness, split into a representative 'baseline' stratum (judge "
    "typical faithfulness here) and an adversarial 'risk' stratum (hardest frames, expected worse — "
    "weak risk faithfulness should DOWNGRADE to GO_WITH_CAVEATS, not NO_GO). Output ONLY JSON: "
    '{"verdict":"GO"|"GO_WITH_CAVEATS"|"NO_GO","confirmed":[..],"use_with_weighting":[..],'
    '"drop":[..],"rationale":"..."}'
)


def _rubric(core) -> str:
    lines = [f'  "{c}": <0.0-1.0: {BY_NAME[c].desc or c.replace("_", " ")}>' for c in core]
    return "{\n" + ",\n".join(lines) + "\n}"


def stratify(aggs: dict, frames: list[Frame], core: list[str], n: int,
             baseline_frac: float = 0.4, seed: int = 0) -> list[dict]:
    """Pick the n most informative frames, tagged baseline (random) vs risk (adversarial)."""
    rng = np.random.default_rng(seed)
    scored = []
    for fr in frames:
        agg = aggs.get(fr.path)
        if not agg:
            continue
        unrel = np.mean([1.0 - agg.cells[c].reliability for c in core]) if core else 0.0
        boundary = np.mean([1.0 if BOUNDARY_LO <= agg.cells[c].value <= BOUNDARY_HI else 0.0
                            for c in core]) if core else 0.0
        susp = np.mean([agg.cells[c].value for c in core]) if core else 0.0
        lookalike = susp if fr.label == 0 else 0.0
        scored.append((0.4 * unrel + 0.3 * boundary + 0.3 * lookalike, fr, agg))
    scored.sort(key=lambda t: -t[0])
    k_top = int(round(n * (1.0 - baseline_frac)))
    top = [(*t, "risk") for t in scored[:k_top]]
    rest = scored[k_top:]
    extra = [(*rest[i], "baseline") for i in rng.permutation(len(rest))[: n - len(top)]] if rest else []
    return [{"path": fr.path, "label": fr.label, "stratum": st, "agg": agg}
            for _, fr, agg, st in top + extra]


def audit_faithfulness(samples, core, client: VLMClient, workers: int = 8) -> dict:
    def work(s):
        adj = client.read(ADJ_SYSTEM, [], s["path"],
                          "^ QUERY FRAME. Score these features as JSON:\n" + _rubric(core), 0.2)
        return s, adj

    pairs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(work, s) for s in samples]):
            s, adj = f.result()
            if adj:
                pairs.append((s, adj))

    def stats(subset):
        per = {}
        for c in core:
            diffs, agrees = [], []
            for s, adj in subset:
                if c not in adj:
                    continue
                try:
                    a = max(0.0, min(1.0, float(adj[c])))
                except (TypeError, ValueError):
                    continue
                cached = s["agg"].cells[c].value
                diffs.append(abs(cached - a))
                agrees.append(1.0 if (cached >= 0.5) == (a >= 0.5) else 0.0)
            if diffs:
                per[c] = {"n": len(diffs), "mae": float(np.mean(diffs)),
                          "agree@0.5": float(np.mean(agrees))}
        return per

    base = [(s, a) for s, a in pairs if s["stratum"] == "baseline"]
    risk = [(s, a) for s, a in pairs if s["stratum"] == "risk"]
    return {"n_sampled": len(samples), "n_graded": len(pairs),
            "n_baseline": len(base), "n_risk": len(risk),
            "per_concept_baseline": stats(base), "per_concept_risk": stats(risk)}


def meta_review(confirm_text: str, audit: dict, client: VLMClient) -> dict | None:
    payload = (f"STATISTICAL CONFIRMATION:\n{confirm_text}\n\n"
               f"FAITHFULNESS — baseline (n={audit['n_baseline']}):\n"
               + json.dumps(audit["per_concept_baseline"], indent=1)
               + f"\n\nFAITHFULNESS — risk (n={audit['n_risk']}):\n"
               + json.dumps(audit["per_concept_risk"], indent=1))
    return client.infer_text(META_SYSTEM, payload, 0.3)
