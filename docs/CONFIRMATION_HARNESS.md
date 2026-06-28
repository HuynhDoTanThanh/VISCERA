# Label Confirmation Harness

**Question it answers:** *Can the Phase-1 VLM concept labels be trusted as supervision for the
Phase-2/3 foundation model?* The reliability study (`run_reliability.py`) only shows which
concepts are reliable-and-discriminative on **train**. That is necessary but not sufficient —
it does not prove the signal **generalizes**, is **not a center shortcut**, is **measurable**, or
is **faithful to the pixels**. This harness turns those into hard gates and a go/no-go verdict.

## Layers

| Layer | File | API? | What it proves / produces |
|---|---|---|---|
| 1 — Statistical | `cf/confirm.py` | No | held-out generalization, bootstrap AUROC CIs, center-leakage + within-center, reliability, assessability → per-concept PASS/REVIEW/FAIL |
| 2 — Agentic | `cf/audit.py` | Yes (proxy) | independent re-grade of a stratified sample (faithfulness) + LLM meta-review verdict |
| 3 — Fair recheck | `scripts/recheck_failcases.py --protocol anchored` | Yes (proxy) | re-extract fail-cases with a DIFFERENT anchor set (real protocol) → STABLE_WRONG / AMBIGUOUS / NOISE; the **cross-anchor robustness** signal |
| 4 — Trust + curation | `cf/trust.py`, `scripts/run_curate.py` | optional | per-(frame,concept) calibrated TRUST → SUPERVISE/MASK + per-frame foundation decision; the **best-sure** label manifest |

## Why anchored, not strict

The Layer-2 standalone rubric changes three variables at once (no anchors, strict 0–1 scale,
single pass) and systematically low-balls — it overstates the fail rate (it even under-called real
lesions). The **anchored** recheck (Layer 3) changes exactly ONE variable — the in-context anchor
set — so disagreement measures genuine label robustness. Always prefer `--protocol anchored`.

## Trust = the "best-sure" signal (validated)

`trust(frame,concept) = gate × assessability × consensus(r, cross_anchor)`, where
`cross_anchor = 1 − |c(anchorA) − c(anchorB)|`. When a cross-anchor re-extraction exists the two
signals are multiplied, so a "confident-but-fragile" label (votes agree, value swings across
anchors) is driven down — the failure within-vote `r` alone misses.

Validated on the 300-frame val cross-anchor pool at `TRUST_SUPERVISE = 0.6`:

| | count | cross-anchor flip rate |
|---|---|---|
| SUPERVISED (kept) | 1,893 (45%) | **1.1%** |
| MASKED (quarantined) | 2,307 | 17.6% |

i.e. the labels the curation keeps flip ~1% under a different anchor set; the ones it drops flip
~18%. Within-vote `r` alone is only moderately calibrated (`r≥0.8` ⇒ 3.2% flip, ρ(r, robustness)
≈ 0.47), so the cross-anchor signal is what makes the kept labels truly sure.

## Gates (a discriminative concept is CONFIRMED iff it passes all)

- **Generalizes:** held-out (val) AUROC 95% bootstrap CI lower bound > 0.50
- **Reliable:** inter-vote agreement ≥ 0.60
- **Measurable:** assessable in ≥ 50% of frames (else it is mostly a mask channel)
- **Not a center shortcut:** within-center label-AUROC ≥ 0.55 **and** concept-predicts-center
  AUROC < 0.70. (center_2 has ~4× the neoplasia prevalence of center_1, so a center-coupled
  concept looks discriminative on the pooled set while teaching the head a site shortcut.)

Thresholds live at the top of `cf/confirm.py`.

## Run it

```bash
# Layer 1 only — fast, offline, no quota:
.venv/bin/python -m scripts.run_confirm

# + Layer 2 agentic faithfulness audit and meta-review verdict (local proxy):
.venv/bin/python -m scripts.run_confirm --audit --audit-n 60 --workers 16

# Layer 3 — fair anchored fail-case recheck (the honest fail rate + cross-anchor signal):
.venv/bin/python -m scripts.recheck_failcases --split val --protocol anchored --pool 300

# Layer 4 — produce best-sure foundation labels (re-extracts a different anchor set, then curates):
.venv/bin/python -m scripts.run_curate --split train     --extract-cross-anchor --workers 24
.venv/bin/python -m scripts.run_curate --split val       --extract-cross-anchor --workers 24
.venv/bin/python -m scripts.run_curate --split unlabeled --extract-cross-anchor --workers 24
```

Layer-4 outputs `outputs/cache/labels_<split>.jsonl` — one row per frame with its foundation
decision (POSITIVE / TRUE_NEGATIVE / HARD_NEG_CANDIDATE / CONFIDENT_NEGATIVE / ABSTAIN) and the
SUPERVISED concept values + trust. This is what Phase-2/3 consumes.

Outputs: `outputs/reports/label_confirmation.md` (human) and `label_confirmation.json`
(machine — `confirmed_core` / `review` / `failed`, per-concept stats, audit, verdict).

## How to read the audit

The audit samples two strata:
- **baseline** — random, representative frames. Judge **typical** label faithfulness here.
- **risk** — adversarially chosen hardest frames (experts split, boundary values, hard NDBE
  look-alikes). A stress test; expected to look worse. A drop here means *down-weight by
  reliability / abstain on hard frames*, not necessarily *drop the concept*.

## Current result (gemini-only, single family)

- **9 CONFIRMED** (Layer 1): mucosal_irregularity, nodularity, demarcation, lesion_present,
  focal_erythema, surface_effacement, color_change_locality, colocalization, color_heterogeneity
- **5 REVIEW**: dilated_vessels / vascular_irregularity / focal_abnormal_vessels (assessable
  <50% — keep as masked features), border_sharpness (low agreement), whitish_focal_area
  (within-center AUROC 0.52 → center confound)
- **1 FAILED**: depression_ulceration (no held-out signal)
- **Meta-verdict:** GO_WITH_CAVEATS. Baseline faithfulness is high (0.92–1.00 agreement) for 8/9;
  `color_heterogeneity` is weak even at baseline → drop. Use the rest with reliability-weighting,
  abstaining on the risk stratum.

## Independence caveat & next step

The adjudicator is currently a fresh **gemini** pass — a weaker independence check than a second
VLM family. When a valid cloud key is available, add the `claude` expert (extraction is resumable;
it won't touch the gemini caches) and set the audit model to it for a true cross-family
faithfulness confirmation. The `r` (reliability) channel will then also reflect cross-family
agreement as the design intends, not just within-gemini vote spread.
