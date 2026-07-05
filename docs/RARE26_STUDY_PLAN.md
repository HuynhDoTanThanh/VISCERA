# RARE26 — Study Plan & Experiment Tracker

> Living document. Competition metric = **PPV@90R at 1% prevalence (bootstrap median)** on a **hidden NEW-center** test; AUROC/AUPRC (95% CI) also reported. Deploy: offline `--network=none`, image-only, per-image / ~16-frame stack, **no test-batch stats, no train-mode BatchNorm in the shipped graph**. Trust the 5 metrics only. Track **AUROC/AUPRC** as the stable signal; PPV@90R is high-variance even on the official leaderboard.

---

## 1. Baselines (leaderboard = ground truth)

| exp | config | **PPV@90R** | PPV CI | AUROC | AUPRC | note |
|---|---|---|---|---|---|---|
| **exp1** (hidden test) | DINOv2 ViT-B + [cls⊕patch_mean], 3-seed ship + 5-view TTA | **0.0181** | [0.0106, 0.0901] | 0.845 | 0.356 | our submission |
| exp1 (RARE25 val) | same | 0.0109 | [0.0098, 0.0990] | 0.866 | 0.562 | |
| **TOP-1 target** (hidden test) | (competitor) | **0.0271** | [0.0140, 0.1203] | 0.895 | 0.382 | **beat this** |
| top-1 (RARE25 val) | (competitor) | 0.0385 | [0.0146, 0.3917] | 0.961 | 0.663 | big val ranking gap |

Field: RARE25 winner ≈ **0.035** (~40-model decorrelated ensemble), field < 0.06.

**Key strategic fact:** exp1 vs top-1 PPV@90R CIs overlap **~85%** → the score gap is **within noise**. The only *detectable* gap is **ranking (AUROC/AUPRC)**, where top-1 genuinely leads (val AUROC 0.961 vs 0.866). **So we win by maximizing AUROC/AUPRC + robustness, not by chasing a PPV@90R point.**

## 2. Diagnosis (drives everything)

Pure **operating-point / TAIL** problem. Implied FPR at the 90%-recall threshold ≈ **49%** (test) / 83% (val): the hardest ~10% of positives score so low the threshold admits half the negatives. The lever is **ranking of the tail** — which negatives sit above the 90R threshold — **not** mean AUROC and **not** calibration. **PPV@90R is invariant to any monotone score transform → temperature/Platt/isotonic are strict NO-OPs.**

## 3. Tracking protocol (discipline first — the metric is noisy)

1. **Local proxy = LOCO** (train center_1→score center_2, and reverse) with the 5-metric report (`evaluate.report_full`). Val (same-center) does NOT test the new-center shift.
2. **Select by AUROC/AUPRC (stable at ~50–130 pos); estimate by PPV@90R-worst + CI.** Never rank configs by the PPV@90R point.
3. **Acceptance gate = paired bootstrap:** keep/submit a change only when the paired Δ CI (new − best) on LOCO AUROC/AUPRC is **clear of 0**. Otherwise it's noise.
4. **Submission budget:** only submit Δ-CI-clear-of-0 gains.
5. **Log** every run in §8 (seed, git SHA, weights, CLI).

---

## 4. Research Agenda (to win) — verified 2026-07-05

Multi-agent research (22 agents) across loss / OOD-attention / imbalance / margin / ensemble; every item below passed 4 hard filters: offline+per-image safe · reorders the tail (not a monotone no-op) · viable at ~127 positives · targets cross-center. Ranked by **tail-leverage × cheapness × proven-ness** (not the source priority integers).

### Tier-1 — do now

| # | Method | Mechanism | Why it lifts AUROC/AUPRC → PPV@90R tail | Integration point | Risk | LOCO gate |
|---|--------|-----------|------------------------------------------|-------------------|------|-----------|
| 1 | **Model-in-the-loop hard-negative mining** (160k pool + in-batch OHEM) | Score the unlabeled corpus with the CURRENT model, add its top false-positives as label-0, retrain, repeat 2–3 rounds | **Most direct lever on the exact quantity PPV@90R measures.** Moves confusable NDBE look-alikes DOWN vs tail positives → fewer negatives admitted at 90R. Lifts AUPRC most. | `mine_hardneg.py` add `--score-with model.pt` (reuse `_score_finetuned`, `infer.py`) over `unl_hardneg.txt` → `unl_modelFP.txt`; feed `finetune.py --neg-list unl_modelFP.txt --neg-cap N`. OHEM: `neg.topk(k)` in `pairwise_rank_loss`. | **PU contamination** — at ~1% pool prevalence the top-N contain real positives; cap N, never hard-label the very top, center-balance the mine. | Paired-Δ AUPRC CI clear of 0 (both LOCO dirs); reject if AUROC drops (= PU contamination). |
| 2 | **Decorrelation-gated ensemble** (FP-tail Jaccard admission) | Fund a new member only if the negatives it puts above its 90R threshold DIFFER from those the ensemble already fails on | **The proven winning path** (RARE25 0.035 = ~40-model decorrelated ensemble). Averaging rankers with disjoint FP-tails demotes each other's confusables. Ensemble averaging is genuinely non-monotone → real reordering. | New `build_ensemble.py`: greedily add the `.pt` maximizing LOCO PPV@90R of the running average s.t. `fp_tail_jaccard(cand,current) < tau` (~0.5 cap). `_score_finetuned` already prob-averages a comma list — gate only picks which commas. | **Select-on-noise** over a ~13-pos LOCO tail. Fix tau as a CAP (don't maximize), seed-average selection, union both dirs, fall back to average-all. | Must beat average-all on paired AUROC/AUPRC CI clear of 0; else ship average-all. |
| 3 | **SWAD** (dense stochastic weight averaging) | Average trainable weights over the flat part of the trajectory; ship ONE averaged model | Flatter minima → lower-variance, better cross-center ranking; **directly attacks the run-to-run tail volatility** that makes PPV@90R noisy. Different weights = different function (reorders). | `finetune.py` loop: running mean of `state_dict` over last-N epochs (only last-K unfrozen blocks+norm+head move); save alongside best-epoch; `cfg` unchanged so `infer.py` unmodified. Composes with WiSE-FT. | Averaging outside a flat basin hurts → average last-N unconditionally / gate on loss plateau, NOT noisy LOCO PPV@90R. Zero added params → low-data safe. | Ship-both-and-compare; averaged ≥ best-epoch on LOCO AUROC/AUPRC; default to averaged on ties (variance). |

### Tier-2 — next

| # | Method | Mechanism | Why | Integration | Risk |
|---|--------|-----------|-----|-------------|------|
| 4 | **Seed × aug-strength diversity pool** | Retrain same arch with different seeds AND aug RNG/strength → cheap decorrelated members feeding gate #2 | Aug-STRENGTH variation makes members fail on different center-2-like negatives = genuine tail decorrelation (pure-seed = only variance reduction) | `finetune.py` already has `--seed`; add `--aug-strength` scaling ColorJitter/Blur/RRC; produce 6–10 members | ~80% already shipped (3-seed); only new code = the knob + gate. Don't count seeds as real diversity |
| 5 | **SAM** (sharpness-aware minimization) | ε-ball worst-case loss → flat minima; ships a normal graph | Flat minima generalize the tail ordering across the shift; pairs with SWAD (SAM finds flat, SWAD averages within), stabilizes the pool | Wrap AdamW (`layerwise_param_groups`) in SAM/ASAM, rho~0.05; 2 fwd/2 bwd | 2× backward; **AMP fragility** (see pAUC float32 guard); non-targeted → use as SWAD/ensemble feeder, not standalone |
| 6 | **Fourier/amplitude aug — prefer RandConv over reference-FDA** | Randomize low-freq amplitude (acquisition color) keep phase (structure); identity at inference | Forces ranking by structure not center color → unseen-center positives less likely to fall in the low-score tail. Cheap decorrelation axis | Custom transform before `ToTensor` in `FrameDS.train`; eval transform identity. RandConv > FDA (only 2 source centers) | **Signal-erasure** — diagnostic mucosal color is partly low-freq; tune small (beta 0.005–0.02); drop if AUROC regresses |

### Research bets — higher risk / weak cross-center

| # | Method | Why it *might* help | Risk |
|---|--------|---------------------|------|
| 7 | **Group-DRO margin (worst-CENTER)** — per-center pAUC/rank margin, backprop the MAX center loss | Targets the cross-center wall by upweighting the harder seen center; needs center in-batch (`FrameDS.__getitem__` + sampler) | **2-center DRO can't target an unseen 3rd**; can sacrifice the easier center; CVaR-only path is near-redundant with active `soft_pauc90`. Only after Tier-1/2. |

---

## 5. Disqualified / do-not-revisit

- **Any pure calibration/temperature/Platt/isotonic** — rank-invariant NO-OP (PPV@90R invariant to monotone transforms). Platt in `ship.py` stays a reporting nicety, never a lever.
- **SOPA-s / one-way partial-AUC DRO (LibAUC)** — no cross-center mechanism; a smoothed DRO variant of the **already-active** `soft_pauc90` (same 90R objective). Covered.
- **CVaR-over-positives (center-stratified)** — fails data scale (α→1–2 samples/batch on 127 pos); cross-center piece = 2-center GroupDRO vs unseen 3rd; center-adversarial already null (GRL/DANN).
- **SOAP / AP-loss / Smooth-AP (AUPRC surrogate)** — no cross-center mechanism; misaligned with 90R; redundant with `pairwise_rank` + `soft_pauc90`.
- **All-positive embedding/logit memory bank** — fails cross-center; head-overfit-to-memorized-positives at 127 pos.
- **PU learning (nnPU) for the negative pool** — fails cross-center (label noise is center-agnostic); only modifies BCE (mean term), leaves the rank+pAUC tail lever untouched.
- **SupCon / Balanced-Contrastive aux loss** — no genuine cross-center mechanism (learns a center shortcut); high-risk at 127 pos.
- **Sub-center ArcFace / prototype-cosine head** — unsound unseen-center DG; untunable s/m/K + unstable sub-center routing at 127 pos.
- **"Measurement discipline as a method"** — it is a RULER, not a lever; home is §7 (prerequisite).

---

## 6. First 3 experiments (concrete sequence)

Cheapest/lowest-risk → most impactful. **Do not spend a submission until a paired-Δ CI on AUROC or AUPRC (LOCO, both directions) is clear of 0.**

**Exp 1 — SWAD (variance floor first).** `finetune.py` running mean of `state_dict` over last-N epochs → `ckpt_swad.pt`. `--swad --swad-last-n 5`. Measure LOCO (both dirs) AUROC/AUPRC/PPV@90R via `paired_bootstrap` of swad vs best-epoch. Ship SWAD if paired-Δ AUROC/AUPRC CI clear of 0 (or Δ≈0 with tighter PPV variance).

**Exp 2 — Model-in-the-loop hard-neg mining, round 1.** `mine_hardneg.py --score-with ckpt_swad.pt --topn 3000` → `unl_modelFP.txt`; retrain `--swad --neg-list unl_modelFP.txt --neg-cap 3000`. **Hold the LOCO val center's unlabeled pool OUT of the mine**; center-balance. Adopt if paired-Δ AUPRC CI clear of 0 AND AUROC doesn't regress (regression = PU contamination → lower N / drop top frames). Iterate rounds 2–3 while Δ>0.

**Exp 3 — Decorrelation-gated ensemble.** Train 6–10 (seed × `--aug-strength`) members on the Exp-2 recipe; greedy FP-tail-Jaccard gate (`build_ensemble.py --tau 0.5 --loco both`) → comma `--model` list. Ship gated ensemble only if it beats average-all on paired AUROC/AUPRC CI clear of 0; else ship average-all. Freeze tau as a fixed cap.

---

## 7. Measurement upgrades (make the LOCO gate trustworthy) — PREREQUISITE

The metric is noise-dominated (PPV@90R CIs overlap ~85%). Tighten the ruler BEFORE trusting any gate.

- **Group/lesion-level bootstrap, not per-frame** — frames from one video/lesion are correlated; per-frame under-estimates SD and manufactures false significance. Resample by lesion/video ID in `paired_bootstrap`.
- **2-fold LOCO, both directions**, report each fold — a lever that helps only one direction is a center shortcut.
- **Seed-averaged metric** over ≥3 seeds with seed SD — a "win" must not be one lucky init.
- **Compute + post the MDE** given lesion-bootstrap SD and 127 positives; treat sub-MDE PPV@90R moves as "not measurable," fall back to AUROC/AUPRC.
- **Gate on AUROC/AUPRC paired-Δ, not the PPV@90R point** (PPV@90R = ship target, noisy judge; tie-breaker only).
- **Anti-leakage** — hold the LOCO val center's unlabeled pool out of the mine; compute all gates on LOCO-val only.
- **Fixed caps, not maximized hyperparameters** — tau, mining-N, SWAD-window are a-priori CAPS; maximizing them against a ~13-pos tail is a select-on-noise engine.

---

## 8. Experiment log (append one row per run)

| date | exp | change | seed(s) | git SHA | LOCO PPV@90R-worst [CI] | LOCO AUROC | LOCO AUPRC | Δ-CI clear of 0? | submitted? leaderboard PPV@90R [CI] | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-05 | exp1 | baseline (ship 3-seed + 5-view TTA) | 0,1,2 | f89c89a | — | — | — | — | ✅ 0.0181 [0.0106, 0.0901] | AUROC 0.845, AUPRC 0.356; top-1 = 0.0271 |

## 9b. Submission-budget strategy (1 submission / WEEK)

**Experiments are LOCAL and free; only the hidden-test eval is rate-limited. NEVER submit to test an idea — submit to DELIVER a locally-validated bundle.** Attribution comes from local LOCO ablation, not from submissions.

1. **Strengthen the local proxy so decisions rarely need the hidden test:**
   - **Acquire the RARE25 validation set locally** if it is public (prior challenge / Zenodo / GC archive). It IS the "Validation RARE25" the platform scores on every submission → an unlimited, on-distribution proxy (NOT our optimistic 2-center `out/val`). **Highest-value action — check first.**
   - Upgrade LOCO: lesion-bootstrap, 2-fold both directions, seed-averaged (§7).
2. **All Tier-1/2 levers COMPOSE** into one recipe: `mined-neg data → SAM → SWAD → seed×aug pool → gated ensemble → 5-view TTA`. Develop as 2 bundles: **A (safe: SWAD + aug-diversity + gated ensemble; SWAD is free to ablate — post-hoc average of the same run)** and **B (risky: hard-neg mining; ablate alone for PU contamination).**
3. **Merge calendar:** Week 0 = no submit (build proxy + local ablation). **Submit #1 = the maximal bundle** of everything that passed the local gate. Submit #2+ = Tier-2 increments (ViT-L member) as each passes locally.
4. **Best-vs-last decides aggressiveness:** if the leaderboard keeps your **BEST** submission → bundle everything into Submit #1 (a failed bundle costs nothing — exp1 stays); if it keeps your **LAST** → submit only the safe Bundle A first, keep mining revertible. **CONFIRM THIS POLICY.**

## 9. Decision rules (how to not fool yourself)

- **Better ⇔ paired-Δ CI vs current best is clear of 0 on LOCO AUROC/AUPRC.** Point gains without that = noise.
- **Expect leaderboard PPV@90R to stay noisy** (CI ~[0.01, 0.12]). Judge progress on AUROC/AUPRC trend.
- **One lever at a time**, logged, so proxy-vs-leaderboard correlation stays interpretable.
- **Realistic target:** 0.018 → field 0.035–0.06. A jump to 0.6 is not physically on the table at 1% prevalence — never chase in-distribution val numbers.
