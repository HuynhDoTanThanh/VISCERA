"""A/B the single-shot prompt: baseline vs the rubric-anchored + contrastive 'quality' prompt.
Same 1 call/vote cost; measures whether the prompt enhancement actually helps on val."""
from __future__ import annotations
import dataclasses
import random

import numpy as np

from ..domain.concept_schema import CONCEPTS
from ..config import Settings
from ..domain.concepts import ROBUST_CORE
from ..infrastructure.cache import FileVoteCache
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from ..cli import load_anchors
from .ab_strategy import run_strategy, metrics


def main():
    base = Settings(); base.ensure_dirs()
    spec = next(e for e in base.experts if e.key == "proagent")
    settings = dataclasses.replace(base, experts=(spec,))
    anchors = load_anchors(settings.raw_store.parent / "anchors.json")
    frames = DatasetLoader(settings).val()
    neo = [f for f in frames if f.label == 1]; ndbe = [f for f in frames if f.label == 0]
    sample = neo + random.Random(0).sample(ndbe, 60)
    clients = {spec.key: ProxyVLMClient(spec, settings)}
    cache = FileVoteCache(settings.raw_label_dir)
    names = [c.name for c in CONCEPTS]
    disc = [c.name for c in CONCEPTS if c.role == "discriminative"]

    s_base = dataclasses.replace(settings, extraction_strategy="single", single_quality=False,
                                 votes_per_expert=3)
    s_qual = dataclasses.replace(settings, extraction_strategy="single", single_quality=True,
                                 votes_per_expert=3)
    print("[tune] baseline single…")
    a_b = run_strategy("single", s_base, clients, cache, sample, anchors, 16)
    print("[tune] quality single…")
    a_q = run_strategy("single", s_qual, clients, cache, sample, anchors, 16)
    m_b, _, pos = metrics(a_b, sample, names)
    m_q, _, _ = metrics(a_q, sample, names)

    def g(m, grp): return float(np.nanmean([m[n]["auroc_ass"] for n in grp]))
    L = [f"# Single prompt A/B — baseline vs quality (val {len(sample)} frames, {pos} neo, proagent)\n",
         f"**core (7)          {g(m_b,ROBUST_CORE):.3f} → {g(m_q,ROBUST_CORE):.3f}  "
         f"(Δ {g(m_q,ROBUST_CORE)-g(m_b,ROBUST_CORE):+.3f})**",
         f"**discriminative(15) {g(m_b,disc):.3f} → {g(m_q,disc):.3f}  (Δ {g(m_q,disc)-g(m_b,disc):+.3f})**",
         f"**all (35)          {g(m_b,names):.3f} → {g(m_q,names):.3f}  (Δ {g(m_q,names)-g(m_b,names):+.3f})**\n",
         f"{'concept':22}{'base':>8}{'quality':>9}{'Δ':>8}{'ass_b':>7}{'ass_q':>7}"]
    L.append("-" * 61)
    for n in list(ROBUST_CORE) + [c for c in disc if c not in ROBUST_CORE]:
        b, q = m_b[n], m_q[n]
        L.append(f"{n:22}{b['auroc_ass']:>8.3f}{q['auroc_ass']:>9.3f}"
                 f"{q['auroc_ass']-b['auroc_ass']:>+8.3f}{b['assess']:>7.2f}{q['assess']:>7.2f}")
    report = "\n".join(L)
    (base.raw_store.parent.parent.parent / "outputs" / "reports" / "single_prompt_ab.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
