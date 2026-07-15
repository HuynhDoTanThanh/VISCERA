# RARE26 — Study Plan & Experiment Tracker

> Living document. Competition metric = **PPV@90R at 1% prevalence (bootstrap median)** on a **hidden NEW-center** test; AUROC/AUPRC (95% CI) also reported. Deploy: offline `--network=none`, image-only, per-image / ~16-frame stack, **no test-batch stats, no train-mode BatchNorm in the shipped graph**. Trust the 5 metrics only. Track **AUROC/AUPRC** as the stable signal; PPV@90R is high-variance even on the official leaderboard.
>
> **📍 CURRENT STATE (2026-07-12): read §19 (the winning plan) + §18 (its diagnosis) FIRST.** Three real submissions scored: **exps/2 (AUROC 0.854) is our BEST**, exp1 0.845, exps/3 0.756. Diagnosed failure = **per-center score-shift** (not tail-ranking); added capacity overfits 2 centers. **§19 = the operative plan** (defend exps/2 floor → per-distribution robust-z score normalization + decorrelated ensemble + WiSE-FT freeze + color aug), derived from a 32-agent adversarial workflow + local grounding. §4–§17 are historical, **superseded by §18–§19 where they conflict.**

---

## 1. Baselines (leaderboard = ground truth)

| exp | config | **PPV@90R** | PPV CI | AUROC | AUPRC | note |
|---|---|---|---|---|---|---|
| **exp1** (hidden test) | DINOv2 ViT-B + [cls⊕patch_mean], mild-aug, no-semi, 30ep, 3-seed + 5-TTA | **0.0181** | [0.0106, 0.0901] | 0.845 | 0.356 | prior best |
| exp1 (RARE25 val) | same | 0.0109 | [0.0098, 0.0990] | 0.866 | 0.562 | |
| **exps/2** (hidden test) | **DINOv2 mean-pool + SEMI + concept-init, 336, mild-aug, 12ep**, 3-seed + 5-TTA | **0.0177** | [0.0099, 0.1259] | **0.854** | **0.401** | **🏆 NEW BEST (AUROC +0.009 vs exp1, PPV tied)** |
| exps/2 (RARE25 val) | same | 0.0129 | [0.0102, 0.1306] | **0.873** | 0.613 | **best on both eval sets** |
| **exps/3** (hidden test) | DINOv3 + **CG-AMIL attention** + concept-init + semi + **448 + strong-aug + 30ep** | **0.0117** | [0.0086, 0.0515] | **0.756** | 0.300 | **⚠ REGRESSED −0.098 AUROC vs exps/2** |
| exps/3 (RARE25 val) | same | 0.0111 | [0.0087, 0.0479] | 0.835 | 0.543 | regressed −0.038 vs exps/2 on SAME set |
| **TOP-1 target** (hidden test) | (competitor) | **0.0271** | [0.0140, 0.1203] | 0.895 | 0.382 | **beat this** |
| top-1 (RARE25 val) | (competitor) | 0.0385 | [0.0146, 0.3917] | 0.961 | 0.663 | big val ranking gap |

Field: RARE25 winner ≈ **0.035** (~40-model decorrelated ensemble), field < 0.06.

**⚠ 2026-07-12 REALITY CHECK — 3 real submissions now scored (see §18/§19):** ranked by the STABLE AUROC signal: **exps/2 (0.854) > exp1 (0.845) ≫ exps/3 (0.756)** — consistent on BOTH eval sets. The **decisive lesson is ARCHITECTURE WEIGHT, not the semi/concept levers**:
> - **exps/2 = exp1's SIMPLE arch (DINOv2 mean-pool @336) + semi + concept-init + SHORT (12ep)** → **best model** (AUROC +0.009 vs exp1, PPV tied). So **semi-consistency + concept-init + short training HELP on a light backbone.**
> - **exps/3 = the SAME semi/concept + HEAVY arch (DINOv3 + attention-MIL + 448 + strong-aug + 30ep)** → **worst** (−0.098 AUROC). The heavy capacity overfit the 2 centers (val→test gap ≈4× exp1's).
>
> So §18's blanket "demote semi/concept" was too coarse: **the culprit is added CAPACITY/training-length that overfits 2 centers, not the semi/concept mechanisms** (which help when the arch stays light). **exps/2 is the new floor to defend and the base to build on.** Full winning plan in §19.

**Key strategic fact (unchanged, now reinforced):** PPV@90R CIs overlap ~85% → score gaps are within noise; the only *detectable* signal is **AUROC/AUPRC**. exps/3 losing 0.09 AUROC is the real, above-noise regression. **We win by maximizing new-center AUROC + robustness — and, per §18, by REVERSING complexity, not adding it.**

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

- **Any pure calibration/temperature/Platt/isotonic applied GLOBALLY to one model's final scores** — rank-invariant NO-OP (PPV@90R invariant to a *global* monotone transform). Platt in `ship.py` stays a reporting nicety, never a lever. **⚠ CORRECTION (2026-07-12, §18):** this does NOT extend to **per-center / per-stack** normalization or **pre-ensemble per-member** recalibration — those are monotone *within a group* but **reorder across the pooled test** → NOT global monotone → **NOT no-ops.** Per-center score normalization is the diagnosed fix for score-shift and was the RARE25 winner's key trick. Promoted to a Tier-1 lever in §18.
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
| 2026-07-05 | exp1 | baseline (ship 3-seed + 5-view TTA) | 0,1,2 | f89c89a | — | — | — | — | ✅ 0.0181 [0.0106, 0.0901] | AUROC 0.845, AUPRC 0.356; top-1 = 0.0271; **STILL BEST** |
| 2026-07-12 | exps/3 | DINOv3 + CG-AMIL attn + concept-init + semi + 448 + strong-aug (30/15ep) | 0,1,2 | 5d2b2f7 | — | 0.756 | 0.300 | ❌ AUROC −0.089 vs exp1 | ✅ 0.0117 [0.0086, 0.0515] | **REGRESSED** on stable AUROC (both eval sets); val→test gap 4× exp1 = overfit. See §18. exps/2 (dinov2+semi@336) built+validated, not yet submitted. |

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

---

# DEEP SURVEY & RESEARCH NOTES — 2026-07-11

> Written to answer: *survey the challenge + metric precisely; verdict on OOD / attention / margin "layers"; and design a semi-supervised way to use the large "suspicious" pool with confidence filtering in Phase-2 WITHOUT overfitting, toward a foundation model that generalises to an unseen center.* Everything below lives **under** the existing discipline (§2–§3, §7, §9): the metric is measurement-dominated, so a mechanism is worth nothing until its paired-Δ AUROC/AUPRC CI on LOCO (both directions) is clear of 0.

## 10. The metric, made exact — what "win PPV@90R @ 1% prevalence" literally requires

**Definition.** Rank every image by score. Set the threshold τ at **90% recall** (sensitivity) — i.e. τ = the score of the positive at the 10th percentile from the bottom. At **1% prevalence** the grader re-weights the negative mass so positives are 1% of the population, then reports **PPV = TP/(TP+FP)** at that τ, as the **median of a bootstrap**. AUROC/AUPRC reported alongside.

**The identity that governs everything.** With prevalence π=0.01 and recall fixed at 0.9:

```
PPV = 0.9·π / (0.9·π + FPR·(1−π)) = 0.009 / (0.009 + 0.99·FPR)
⇒  FPR@90R = 0.00909 · (1/PPV − 1)
```

| target PPV@90R | required **FPR at the 90%-recall threshold** | vs exp1 |
|---|---|---|
| 0.0181 (exp1) | **49.6%** | — |
| 0.0271 (top-1) | **32.6%** | −17 pts |
| 0.035 (RARE25 winner) | **25.1%** | −25 pts (≈ halve) |
| 0.05 | 17.3% | |
| 0.06 (field ceiling) | 14.2% | |

**So the entire competition is one number: the fraction of negatives that outrank the hardest 10% of positives on an unseen center.** exp1 lets ~half the negatives through at recall-90; the winner lets a quarter through. To win we must roughly **halve the negative mass sitting above τ_90R**.

**Why the tail is the whole game.** τ_90R is set by ~the **12th–13th lowest-scoring positive** (10% of ~127). The metric is blind to how well we rank the *easy* 90% of positives — they're all above τ by construction. It is decided by (a) where the **subtle/tail positives** land and (b) how much **negative mass** sits above them. This is a **high-sensitivity partial-AUC / tail-ranking** problem, *not* mean AUROC.

**Why it's noisy (→ gate on AUROC/AUPRC).** τ is fixed by ~13 boundary positives; one genuinely-ambiguous or mislabeled hard positive crossing the boundary swings which negatives are counted → the 8×-wide CI. Curve-point PPV@90R is a high-variance judge; AUROC/AUPRC (all positives, threshold-free) are the stable signal. **Rank-invariance** ⇒ any monotone transform (temperature/Platt/isotonic) is a strict NO-OP (already in §5).

**The single objective, three attack surfaces.** Minimise `Σ 1[neg_score > τ_90R]` on an **unseen** center:
1. **Push tail positives UP** — attention pooling, tail-weighted margin, more/diverse positives.
2. **Push false-positive negatives DOWN** — hard-neg mining / one-sided PU (the most direct lever; §12A).
3. **Make both robust to center shift** — SWAD/SAM/WiSE-FT, strong-aug consistency, domain-adaptive SSL (§13).
Every method below is scored by *which surface it moves* and *whether it survives the 4 filters* (offline+per-image · reorders the tail · viable at 127 pos · cross-center).

## 11. OOD / attention / margin "layers" — honest verdicts against the metric

### 11.1 "OOD layer"
There is **no inference-time OOD module** we can ship: test-time adaptation, BatchNorm-in-train, and any test-batch statistic are **disqualified by the per-image / `--network=none` contract** (would NaN at batch-of-1 = lost submission). The OOD "layer" that actually ships is **training-time robustness**, already partly present:
- **WiSE-FT anchor** (active, α=0.7) — weight-space interpolation back toward the SSL init recovers linear-probe OOD robustness (Wortsman 2022) while keeping FT capacity (Kumar 2022).
- **SWAD / SAM** (§4 Tier-1/2) — flat minima generalise the tail *ordering* across the shift.
- **Strong-aug consistency** (RandConv/color/blur as a center-nuisance proxy) — see §12B; this is the closest thing to an "OOD layer" that has a real mechanism and ships clean.
- The genuinely new OOD/foundation lever is **domain-adaptive continued SSL** → §13.
> DANN / center-adversarial GRL / feature-projection center-invariance were already tested → **null** (2-center adversary learns a shortcut, doesn't target an unseen 3rd). Do not revisit.

### 11.2 Attention layer — **PROMOTE to a real candidate** (attention-MIL patch pooling)
The head currently pools patches by a **plain mean**: `feat = [cls ⊕ mean(patch_tokens)]` (`finetune.py:84`). A frame is a **bag of 576 patches**; neoplasia occupies **a few patches**; **mean-pool dilutes a subtle lesion signal with 500+ normal-mucosa patches** → the *tail* positive (few positive patches) gets averaged *below* τ_90R. That is exactly the failure the metric punishes.

**Fix = gated attention-MIL pooling** (Ilse et al. 2018) over the patch tokens: `patch_pool = Σ_i a_i·v_i`, `a_i = softmax(w·(tanh(V h_i) ⊙ sigm(U h_i)))`. It up-weights lesion patches → lifts precisely the tail positives → a **direct FPR@90R mechanism** (surface #1). Fully offline, per-image, permutation-invariant (no batch stats).
- **Ship-safe:** keep `cls` in the concat as a residual (`[cls ⊕ attn_pool(patches)]`), small attention hidden dim (128), so it degrades gracefully to ~mean-pool if attention doesn't help.
- **Risk (real):** +params on 127 positives → overfit; attention can learn a **center shortcut** (attend to scope logo/black border). **Mitigation:** small dim + dropout on attention logits + WiSE-FT anchor + strong-aug consistency (§12B) so attention can't lock onto center-specific background; **gate on LOCO AUROC/AUPRC both directions**, and reject if same-center val lifts while LOCO doesn't (shortcut signature).
- **Integration:** replace the `.mean(1)` in `Net.forward` with an `AttnPool` module; ~30 lines; `cfg` unchanged so `infer.py`/container graph stays LayerNorm-only + per-image. **Kill criterion:** paired-Δ AUROC CI not clear of 0 after 3 seeds → revert to mean-pool (it's a free ablation of the same run if you log both pooled features).

### 11.3 Margin layer — split the verdict
- **Metric-learning HEADS (ArcFace/CosFace/LDAM/sub-center) — stay DISQUALIFIED (§5).** They optimise angular separation around **class prototypes** that need many stable positives to estimate; at 127 positives across 2 centers the positive prototype is unstable and **learns a center-specific direction** → fails the unseen 3rd center. They also optimise a *classification* geometry, not the *high-sensitivity tail ordering* the metric scores. Untunable s/m/K. Do not revisit.
- **Margin in the RANKING loss — already correct and active, but UNIFORM.** `pairwise_rank_loss` (softplus margin=1.0) and `soft_pauc90` (q=0.2) both fire every batch (the **`PosBalancedBatchSampler` is confirmed ACTIVE**, `pos_per_batch=8`, `finetune.py:234` — the old "inert sampler" concern is resolved). The upgrade that survives all 4 filters:
  - **Tail-weighted ranking margin (OHEM)** — the pairs that decide τ_90R are (lowest positives × highest negatives). Uniform mean over all P×N pairs spends most gradient on easy pairs. Weight toward the hard tail: per positive, take `neg.topk(k)` hardest negatives in `pairwise_rank_loss`; optionally focal-style down-weight of already-separated pairs. This concentrates margin exactly at the operating point → surface #1+#2. **Cheap** (a few lines in the existing loss), **on-mechanism**, low overfit risk (no new params). Add as **Tier-1.5**. Gate as usual; freeze k as a fixed cap (don't tune it against the 13-pos tail).

## 12. Semi-supervised use of the "suspicious" pool with confidence filtering — full design (the main new ask)

**The trap (state it first).** Vanilla confidence-filtered self-training (FixMatch/pseudo-label) keeps **high-confidence = EASY** samples. But PPV@90R is decided by the **HARD tail**. So "add the confident positives" adds the ones **we already rank high** → *zero tail leverage*, and at ~1% pool prevalence a positive-pseudo-label filter suffers **PU contamination + confirmation bias** → it inflates same-center AUROC (and the 0.65 val mirage) while doing nothing for the unseen-center tail. **Naive self-training is an AUROC-mover and a PPV@90R non-mover, and it is exactly the overfit the user is worried about.** The design below is built to avoid that.

**Reframe: three uses of the pool, ranked by *metric leverage*, not by confidence.**

| | use | what to keep | surface | leverage | risk |
|---|---|---|---|---|---|
| **A** | **One-sided PU: confident-NEGATIVE mining** | frames teacher is confident are negative **but the model currently scores HIGH** (hard FPs) → add as label-0 | #2 (push FP down) | **highest** | low — pool is ~99% neg, so "confident-negative" is reliable |
| **B** | **Consistency regularisation on ALL unlabeled frames** | no labels at all — enforce f(weak-aug) ≈ f(strong-aug) | #3 (center-robust) | modest, insurance | low |
| **C** | **Confident-POSITIVE pseudo-labels** | only to fight 127-positive **starvation** (diversity), NOT as tail signal | #1 (via less overfit) | small | **highest** (confirmation bias) — do last |

### 12A — One-sided PU (confident-negative) = the real mover
This is **hard-neg mining (§4 Tier-1) re-derived as one-sided self-training**, and the plumbing already exists (`--neg-list/--neg-cap` → injected as label-0, `finetune.py:223-225`). The "confidence filter" is: **keep frames the EMA-teacher scores confidently negative AND that the current model ranks near/above τ_90R** (hard, boundary negatives). Adding those pushes down exactly the FP mass PPV@90R counts. The PU prior — *at 1% prevalence an unlabeled frame is 99% likely negative* (Kiryo 2017's insight, used as a PRIOR, not the nnPU loss which was disqualified) — is what makes the negative direction safe where the positive direction isn't.
- **Guards:** cap N; **never hard-label the very top** of the mine (that's where real positives hide); **center-balance** the mine; **hold the LOCO-val center's pool OUT**. Reject the round if AUROC **drops** (= PU contamination → lower N / drop the top frames).

### 12B — Consistency regularisation = label-free OOD robustness (the "foundation" mechanism at fine-tune time)
On every unlabeled frame, draw a **weak** view (teacher) and a **strong** view (student: RandConv/heavy color/blur = a proxy for the scope+lighting nuisance a new center introduces) and penalise disagreement: `λ(t)·‖σ(student_strong) − σ(teacher_weak).detach()‖²` (FixMatch/UDA + Mean-Teacher, Sohn 2020 / Xie 2020 / Tarvainen 2017). This needs **no correct label** — it flattens the decision surface and buys **center-invariance**, so whatever ranking the supervised head learns **transfers** to an unseen center (the tail positives are less likely to collapse under the shift). This is the safest way to *use the pool* and the training-time answer to "a model other centers can understand."

### 12C — Confident-positive pseudo-labels = anti-starvation diversity ONLY
With 127 positives the sampler oversamples them ~heavily → overfit to those 127. A **few** high-precision pseudo-positives increase positive **diversity** → less overfit, better tail coverage. But this is the dangerous direction, so: **EMA-teacher targets** (not the live model), **augmentation-stability filter** (accept only if the prediction is stable across K views — raw softmax is miscalibrated under shift, so use *stability/agreement*, not magnitude), **cap N small**, **center-balance**, **ramp-up λ(t)**, **never the very-top-confidence** (those are easy, info-free — prefer the *suspicious mid-confidence-but-stable* ones). Do this **only after A/B pass** and only if it lifts LOCO (not just val).

### The overfit-proofing (why this recipe generalises where naive self-training doesn't)
- **EMA teacher** (τ≈0.99–0.999) decorrelates pseudo-targets from the student's current errors → kills confirmation bias (Tarvainen 2017).
- **Weak-target / strong-input** asymmetry (FixMatch) → the student learns invariance, not the teacher's mistakes.
- **Stability filter** (K-view agreement) instead of raw confidence → robust to shift-induced miscalibration.
- **Ramp-up λ(t)** + **caps** → early noisy pseudo-labels can't dominate.
- **One-sided PU** → exploits only the *reliable* (negative) direction.
- **WiSE-FT anchor + SWAD + mostly-frozen/low-LR backbone (LP>FT)** → bounds weight drift OOD.
- **The gate is the real guard:** accept only if **LOCO AUROC/AUPRC paired-Δ CI clear of 0 (both dirs)**; **reject if same-center val jumps but LOCO doesn't** — that divergence is the memorisation/overfit signature (it's how we caught the 0.65→0.018 mirage). This is precisely the "without overfit" the ask demands, made falsifiable.

### Integration (concrete, `phase3/finetune.py`)
- New flags: `--unlabeled-list`, `--consistency-weight`, `--consistency-rampup`, `--ema-decay`, `--pseudo-pos-conf`, `--pseudo-pos-cap`. (A's `--neg-list/--neg-cap` already exist.)
- `PairAugDS`: unlabeled dataset returning (weak, strong) tensor pairs; a second `DataLoader` iterated alongside `dl`.
- `ema = deepcopy(net)`; update after each step; `@torch.no_grad()` teacher forward on the weak view.
- In the loop: `loss += rampup(ep)·consistency(net(strong), ema(weak).detach())`; periodically re-score the pool with `ema` to refresh (A) hard-negatives and (C) stable pseudo-positives (cap N, center-balanced, LOCO-val pool held out).
- Composes with existing `bce+rank+pauc`, the attention pool (§11.2), tail-weighted margin (§11.3), WiSE-FT and SWAD. `cfg`/`infer.py`/container graph unchanged (all extra machinery is **train-time only**).

## 13. Foundation model for cross-center generalisation ("other centers can understand")

**Domain-adaptive continued SSL (DAPT).** Continue the **DINOv2 self-distillation** objective on the **144k in-domain endoscopy frames** (label-free) → adapt the general LVD backbone to endoscopy texture/color/lesion statistics → more center-agnostic features → tail positives separate better on unseen centers. This is the single biggest *foundation* lever and the label-free way to exploit the whole pool.
- **Evidence caveat (do not skip the gate):** in RARE25 the in-domain **GastroNet** pretrain **underperformed** ImageNet — in-domain SSL is **not guaranteed** to help and can *forget* general features or **overfit to 2 centers' style** → worse cross-center. So: **short** continued-SSL, **low LR**, and **WiSE-FT the SSL weights back toward the LVD init** too. **Gate:** DAPT encoder vs current SSL encoder on LOCO AUROC/AUPRC — adopt only if paired-Δ CI clear of 0. Position as a **research bet**, not Tier-1.
- **The DG toolkit that actually ships** (vs disqualified DANN/projection): **strong-aug consistency (§12B) + RandConv (§4 #6) + SWAD + WiSE-FT.** These are the center-invariance levers with a real, offline, per-image-safe mechanism.

**The full "foundation that generalises" stack:** `DAPT encoder (gated) → semi-supervised fine-tune (§12: one-sided PU + consistency + capped pseudo-pos) → attention-MIL pool (§11.2) + tail-weighted margin (§11.3) → SWAD → decorrelation-gated seed×aug ensemble → 5-view TTA`, every stage gated on LOCO AUROC/AUPRC.

## 14. How this merges into the plan (respect the 1-submission/week budget)

All of §10–§13 are **local and free**; only the hidden-test eval is rate-limited. Ordered by leverage×cheapness×safety, extending §6, gated per §7/§9 (**paired-Δ AUROC/AUPRC CI clear of 0, both LOCO directions; reject on val-lift-without-LOCO-lift**):

0. **Build the compass first** — RARE25-val local proxy + lesion-bootstrap 2-fold seed-averaged LOCO (§7). *No lever is trustworthy before this.*
1. **SWAD** (variance floor — §4 Exp1). Free post-hoc average of the same run.
2. **One-sided PU / hard-neg mining, round 1** (§12A / §4 Exp2). The most direct FPR@90R lever; plumbing exists.
3. **Tail-weighted rank margin (OHEM)** (§11.3). A few lines, no new params, on-mechanism.
4. **Attention-MIL pooling** (§11.2). Real tail mechanism; log both pooled features so mean-pool stays a free fallback.
5. **Consistency regularisation** (§12B). Label-free robustness/insurance for the unseen center.
6. **Research bets** (only if 1–5 leave a measurable AUROC gap to top-1): **DAPT encoder** (§13), **capped pseudo-positives** (§12C), **SAM/RandConv** (§4 Tier-2).
7. **Decorrelation-gated seed×aug ensemble + 5-view TTA** (§4 Exp3) — the proven winning wrapper, applied on top of whatever passed.

**Submit #1 = the maximal bundle of every step that passed the local gate** (if the leaderboard keeps BEST; else ship the safe subset first — **still need to CONFIRM best-vs-last policy**, §9b).

**Honest expected value.** #2 (one-sided PU) is the real mover; #3–#4 are on-mechanism tail lifts; #1/#5 are variance/robustness insurance; #6 are uncertain bets with hard gates. **None of this puts 0.6 on the table** — the ceiling at 1% prevalence is ~0.06, and placement has a large luck component for the whole field (§2). We win by **halving FPR@90R on the unseen center** (0.50→~0.25) via surfaces #1–#3, and by **not fooling ourselves** with same-center numbers — measured on AUROC/AUPRC, the only detectable signal.

## 15. WEEK-1 EXECUTION — code shipped 2026-07-11

**Decision (for the single weekly submission):** ship the **graph-safe bundle first** — `SWAD + tail-weighted OHEM margin` on top of the exp1 recipe, wrapped in the existing 3-seed + 5-view TTA. **Defer** attention-MIL pooling / consistency-reg / DAPT (change the graph or need new infra → deployment risk or unvalidated) and **hard-neg mining to week-2** (its PU-contamination failure mode needs a clean, leakage-controlled ablation — don't risk the first submission on the one lever that can silently corrupt labels). Rationale: under 1-submission/week, EV = leverage × cheapness × local-gate-ability × **zero deployment risk**; SWAD+OHEM are the only top-leverage levers that satisfy all four (no `Net`/`viscera_model.py` change, cleanly LOCO-gateable without the mining leakage problem).

**Code landed (all train-time/post-hoc only — shipped graph & offline container unchanged, unit-tested):**
- `finetune.py` — `--ohem-k K` (tail-weighted margin: per-positive top-k hardest negatives in `pairwise_rank_loss`; 0=off=old behaviour). `--swad --swad-last-n N` (running mean of `state_dict` over the last-N epochs; averages float params, passes non-float buffers through; ships the averaged model under `--holdout none`, saves `*_swad.pt` + prints SWAD-vs-best-epoch AUROC/AUPRC under LOCO; WiSE-FT applied to both).
- `mine_hardneg.py` — `--score-with a.pt[,b.pt]` model-in-the-loop mining: scores the VLM-negative pool (`--pool HARD_NEG_CANDIDATE,CONFIDENT_NEGATIVE`) with the current model, emits the top-`--topn` false-positives (minus `--skip-top` PU guard, minus `--exclude-dir` for LOCO anti-leakage) → `unl_modelFP.txt` → feeds the existing `finetune.py --neg-list`.
- `colab_full_pipeline.ipynb` — new **LOCO GATE** cell (baseline vs bundle, both directions, gate on AUROC/AUPRC) before the ship cell; ship cell now `STAGE2_FLAGS='--unfreeze 6 --wise-ft 0.7 --epochs 30 --swad --swad-last-n 5 --ohem-k 8'` with the week-2 mining lines documented inline.

**Run order (Colab GPU; local & free until the submission):**
1. **Gate (no submit):** run the LOCO GATE cell → 4 fine-tunes (baseline/bundle × center_2/center_1). Adopt the bundle only if AUROC/AUPRC **hold or improve on BOTH** centers (PPV@90R point is a noisy tie-breaker only).
2. **Ship:** run the ship cell (3 seeds, `--holdout none`, bundle flags) → `ship_seed{0,1,2}.pt` → `viscera_model.py` (unchanged) → offline container → **the one weekly submission.**
3. **Week-2 (prepare in parallel):** `mine_hardneg --score-with ship_seed0.pt` → LOCO-ablate `--neg-list unl_modelFP.txt` with `--exclude-dir` on the held-out center's pool; adopt only if AUPRC↑ **and AUROC does not drop** (drop = PU contamination → lower `--topn` / raise `--skip-top`).

**Still open (blocks full trust):** (a) leaderboard **best-vs-last** policy (decides whether to bundle mining into submit #1 or stage it); (b) **RARE25-val** downloadable? (unlimited on-distribution compass to supplement 2-center LOCO). No `video`/`lesion` column in `train.csv` → §7 lesion-bootstrap not yet available (frame-level LOCO only).

**Log:**
| date | change | status |
|---|---|---|
| 2026-07-11 | `--swad`, `--ohem-k`, `mine_hardneg --score-with` shipped + unit-tested; notebook LOCO-gate + bundle ship cells | code in `main`; awaiting Colab LOCO-gate run before submit |

## 16. Novelty & paper strategy — completing the solution as a publishable contribution

**The hard truth first (or the paper dies in review).** Winning and novelty conflict on this metric. The *safe* win is the RARE25 recipe — a big decorrelated ensemble — which is **not a methodological contribution** (unpublishable as "novel method"). And because the metric is **measurement-dominated** (SD ≫ margins; §2, §7), **final placement is partly luck → a paper cannot claim "our novel method won" from the leaderboard row alone.** Non-negotiable integrity rule for every claim in the paper: *validated on LOCO paired-Δ AUROC/AUPRC, both directions, seed-averaged, with CIs, reported against the measurement noise floor — independent of final rank.* The leaderboard is corroboration, never proof. A reviewer will (correctly) reject any lift smaller than the MDE.

**Where novelty must come from.** Anyone can bolt DINOv2 + an ensemble onto ImageNet. Our **only** defensible, non-reproducible asset is the **144k VLM-concept-scored frames + the concept-role taxonomy already coded** (`pretrain_concept.py`: discriminative core `full15`; acquisition/scope-style `ACQ_QUALITY`→detached AUX; `route_concepts(detach/main/grl/drop)`). *That is the novelty engine.* (Note the earlier retired verdict — "concept-supervised **pretraining of the backbone** is null vs SSL" — is **not** contradicted here: we retire concepts as a *representation spine* but use them as a *signal* for mining + invariance, which that verdict explicitly left alive. Different mechanism, honest continuity.)

### The recommended spine — "Concept-guided operating-point robustness"
Two novel, concept-driven, metric-aligned mechanisms, both grounded in existing code, both falsifiable on LOCO:

**(A) Concept-Confounded Tail Mining (CTM).** *Claim:* the negatives sitting above τ_90R are not random — they are **concept-confounded** (they share *diagnostic* concepts — `vascular_irregularity`, `whitish_focal_area`, `border_sharpness` — with true neoplasia). *Novelty:* mine hard negatives in the **diagnostic-concept space**, not just model-score space — NDBE frames that maximally activate the FP-driving concepts — and add them as label-0. Interpretable ("we show *which concept* the model confuses and fix it"), metric-aligned (surface #2), and it *extends the score-only miner already shipped* (`mine_hardneg --score-with`) with a concept-ranking term. Ablation: CTM vs score-only mining vs random-neg — does concept-guided cut FPR@90R more?

> **CTM empirical validation (2026-07-11, local on the 170,200×35 concept matrix; `mine_hardneg --concept-rank`).** (1) A trust-weighted sum of the **full15 diagnostic concepts is itself an AUROC=0.878 neoplasia classifier** (≈ the full DINOv2 model's 0.845–0.87) — the VLM concepts carry the signal. (2) **38.6% of labeled negatives are concept-confounded** (sit in the positive concept-range) — the NDBE look-alike population the metric punishes. (3) **Honest negative result:** concept-space *alone* cannot cleanly separate confounded-neg from unlabeled-pos — positives score high on *everything*, so the `surface−decisive` axis barely separates (pos +0.139 vs neg +0.132) and standalone mining cuts PU contamination only 5.1%→3.7%. (4) **The real lever is the DECISIVE-hallmark PU guard:** decisive concepts (architecture/vascular) are the stronger signal (AUROC 0.903 > surface 0.847), so *dropping unlabeled frames whose decisive score ≥ the labeled-positive median* removes the likely unlabeled positives before mining — the deployable CTM = model-FP mine ∩ high-surface-confounder ∩ PU-guarded. This turns the plan's #1 mining risk (PU contamination) into a controlled, interpretable, concept-guided filter. Shipped + tested; the paper's CTM ablation is CTM-PU-guard vs score-only vs random-neg on LOCO.

**(B) Concept-Mediated Center Invariance (CMI).** *Claim:* center shift is **mediated by acquisition concepts** (color, brightness, scope, blur/glare/exposure). *Novelty:* feature-adversarial DANN/GRL already **failed** here (2-center adversary learns a shortcut) — so instead enforce that the **diagnosis logit is invariant to interventions on the acquisition-concept subspace**: style-augmentation consistency (RandConv/color = a do-operation on acquisition concepts) **plus** a penalty that the diagnosis is uninformative about the *predicted* acquisition concepts. This is a **causal/concept-mediated** invariance, not a feature-statistical one — and it upgrades the existing `context_route detach` from "don't shape the trunk" to "actively make the decision counterfactually-invariant to scope style." Ablation: CMI vs detach-only vs GRL vs none — does it cut the held-out-center FPR@90R?

Together: *"The false positives that cost PPV@90R on an unseen center are concept-confounded and acquisition-mediated; we use VLM-derived clinical concepts to mine them (CTM) and to make the decision invariant to them (CMI)."* That is a clean, novel, interpretable paper thesis that targets the exact metric and uses the exact asset no competitor has.

### Secondary contributions (true, and they harden the paper)
- **Operating-point-aware SSL (a mechanistic insight / mini negative-result):** confidence-filtered self-training optimizes the *easy* region while a high-recall metric is decided by the *hard tail* → naive FixMatch is an AUROC-mover and a PPV@90R non-mover; **one-sided PU + consistency** is the principled fix (§12). Publishable as a "why standard SSL misfires for operating-point metrics" result.
- **Robust model selection under measurement-dominated challenge metrics (§7):** MDE for PPV@90R@1% at ~127 positives, lesion/group bootstrap, AUROC-selection > PPV-selection, LOCO-both-directions. A methods contribution reviewers respect; it *is* why we didn't chase the 0.65 mirage.

### Paper skeleton (A = spine, B/C = support)
Metric-as-objective (§10 FPR@90R identity) → CTM → CMI → operating-point SSL → decorrelation-gated ensemble → measurement protocol → LOCO ablations with CIs → leaderboard corroboration. Title direction: *"Concept-Guided Operating-Point Robustness for Cross-Center Barrett's Neoplasia Detection."*

### What "completing the solution" means, concretely
Week-1 ship stays the **graph-safe bundle** (SWAD+OHEM; §15) — that protects the leaderboard while the novel spine is built and LOCO-validated offline. Then fold in whichever concept mechanism clears its LOCO gate. **The novel spine is developed and gated locally; it only reaches a submission once its paired-Δ AUROC/AUPRC CI is clear of 0.** This keeps the win safe AND the paper honest.

## 17. One-shot review — two adversarial workflows (2026-07-11)

Two multi-agent reviews (14-agent container + 27-agent training) with per-finding verification. **No CRITICAL/HIGH; no training-breaking, checkpoint-corrupting, or container-crashing bugs.** The container was run end-to-end offline on the real TIFF -> valid output; architecture parity empirically confirmed (179==179 keys, 0 missing). Fixes applied:

**Container review (5 confirmed; 3 fixed):**
- [MED] Paired gate built but never called — `evaluate_center` returned scores that finetune.py discarded (`r,_`). -> finetune.py persists `{out}_loco.npz`; notebook calls `ev.gate()`. Decision is the paired test, not eyeballing.
- [LOW] BICUBIC/BILINEAR train-serve skew -> `viscera_model.py` forced `Image.BILINEAR` to match FrameDS.
- [LOW] uint16 truncation (`astype(uint8)` mod-256) -> rescale-by-max guard (no-op for the 8-bit example).
- [LOW notes] Docker base/deps unpinned -> validate the *saved tar* (`docker load`), not a rebuild; PPV apples-to-oranges -> moot (decision is the AUROC/AUPRC paired gate).

**Training review (16 confirmed: 3 MED, 13 LOW/PARTIAL; MED all fixed):**
- [MED] SWAD was a silent no-op — cosine->0 left the SWAD window at ~1e-7 LR (average == final epoch). Confirmed the pre-registered hypothesis. -> `eta_min=lr*0.1` when `--swad`; the paired gate now decides if functional-SWAD helps.
- [MED] LOCO selection on epoch-max PPV@90R = selection-on-noise + biased the SWAD-vs-best comparison -> select on **AUPRC** (stable); PPV kept as diagnostic.
- [MED] Ship committed to concept-init (measured ~null) untested -> LOCO cell is now a **3-leg gate** (cbase/cbundle/sslbundle) testing BOTH *[levers] bundle-vs-baseline* AND *[encoder] concept-init-vs-raw-SSL*, with explicit FAIL->drop-lever / FAIL->drop-`--init` decisions.
- [LOW, deferred to LOCO A/B] soft_pauc90 q=0.2 targets ~80R (defensible variance tradeoff); no LR warmup; no drop_path; mild aug / no FOV-mask randomization; wd on norms/bias; positive-oversampling coupled to `--neg-cap`; confneg 1:1 weight. All optional tuning levers gated on LOCO, not defects. Defensive: `--init` asserts loud; pure-FT ship warns.

**Net:** the shipped graph is sound; the fixes make the *decision process* rigorous (real gate, stable selection, encoder tested) and repair the one inert lever (SWAD). Nothing blocks the submission; the LOCO gate now tells you which of {baseline, bundle} x {concept, SSL} to ship.

---

# §18. REALITY CHECK & REVISED STRATEGY — 2026-07-12 (post first real result)

> We shipped the fancy stack (exps/3) and got the **first honest new-center number**. It **regressed vs the simple exp1**. This section records the deep diagnosis, states plainly **what it falsifies in §4–§16**, and rewrites the plan around the **actual** failure mechanism (per-center score-shift), grounded in a fresh cross-center-DG SOTA sweep + the RARE25 winner's recipe. **This section supersedes the tail-lever agenda where they conflict.**

## 18.1 The result — complexity REGRESSED the stable signal

| model | fine-tuning intensity | RARE25-val AUROC | hidden AUROC | **val→test gap** |
|---|---|---|---|---|
| **exp1** (DINOv2, mean-pool, mild-aug, no semi) | light | 0.866 | **0.845** | **0.02** (generalizes) |
| **exps/3** (DINOv3, CG-AMIL attn, 448, strong-aug, semi, concept, 30/15ep) | heavy | 0.835 | **0.756** | **0.08** (overfit ≈4×) |

Every lever we added — attention-MIL pooling, DINOv3, 448, strong-aug, semi, concept-init, more epochs — **lowered new-center AUROC on both eval sets** and **quadrupled the generalization gap**. Not catastrophic (PPV CIs overlap), but the direction is consistent on the **stable** signal. **exp1 is still our best model.**

**Context (not a disaster):** RARE25 2nd-place (MaxViT@384) collapsed **0.945→0.771** cross-center — the *same wall*. Our 0.756 sits right at that band. The winner's **0.92 AUROC / 0.035 PPV came from a 40-model ensemble + per-prevalence recalibration + color aug**, NOT a single clever model. We are not far off the achievable single-model ceiling; we are far off the **ensemble+recalibration** wrapper.

## 18.2 Deep diagnosis — the mechanism (measured, not guessed)

Ran feature probes on 619 val frames (seed0 fine-tuned features vs frozen DINOv3, sklearn):

1. **Frozen DINOv3 linear-probe transfer c1→c2 = 0.776 ≈ hidden AUROC 0.756.** Fine-tuning on 2 centers bought **~zero** new-center gain over the raw backbone. The tuned in-sample transfer (0.978) and same-center val (0.99) are **mirage** — 2-center alignment that evaporates on center 3.
2. **Representation is ~0.996 center-separable — identical for frozen (0.996) and fine-tuned.** The largest axis of feature variation is *which center*, not neoplasia. Intrinsic to DINOv3 on endoscopy (per-center light/scope/color), **not** caused by our training.
3. **🔫 SCORE-SHIFT is the killer.** center_2 negatives score ~3.5× higher than center_1 (neg p95 0.050 vs 0.014; one c2 negative = 0.977). The 90R operating threshold (0.42) is tuned to the 2 training centers. **A new center brings its own higher negative baseline → its normal mucosa floods above threshold → FPR explodes → PPV → ~random 0.01.** This is exactly 0.0117.
4. **LP-FT confirmed (Kumar 2022):** freezing beats fine-tuning here (frozen 0.776 > FT 0.756), and *lighter* FT (exp1) beats *heavier* FT (exps/3). Fine-tuning distorts the foundation features that carry cross-center generality.

## 18.3 What this FALSIFIES / REVISES in the prior plan

- **The §4–§16 "tail-lever" thesis is not wrong in theory but wrong in priority.** Attention-MIL (§11.2), semi (§12), concept-mining (§16), strong-aug, more epochs — as implemented — **added capacity that overfit 2 centers** and **do not attack the diagnosed gap** (per-center score-shift / acquisition-color). They lifted the mirage val, not the unseen center. **Demote all of them below the DG/recalibration levers.**
- **2-center LOCO was too weak to catch this** (it rewarded the same 2-center alignment that fails on center 3). The honest compass is **frozen-transfer + val→test gap**, not LOCO PPV.
- **§5 correction (done):** per-center / per-stack **score normalization** and **pre-ensemble recalibration** are NON-monotone across the pooled test → real levers, not no-ops. This is the single biggest missed lever and the winner's key trick.
- **The premise "we win by halving FPR@90R via tail ranking" (§10) is incomplete:** at AUROC≈0.78 the FPR@90R is dominated by **cross-center calibration/score-shift**, not tail ranking. Fix the shift first; then rank.

## 18.4 REVISED winning strategy — ranked by (new-center leverage × feasibility with 2 centers)

Grounded in the diagnosis + a 6-bucket cross-center-DG SOTA sweep + the RARE25 winner's actual recipe. **This is the new Tier-1.**

| # | Lever | Why it targets THE mechanism | Cost | Evidence |
|---|---|---|---|---|
| **1** | **Per-center / per-stack SCORE normalization + prevalence-affine recalibration** (robust-z or rank on the stack's scores before pooling; recalibrate each ensemble member pre-average) | **Directly cancels the score-shift** that floods FPs at 90R. Non-monotone across centers → real PPV@90R lever. **The winner's key trick.** | trivial, inference-only, offline-safe | diagnosis §18.2(3); IMSY winner |
| **2** | **Acquisition-domain augmentation: FDA amplitude-swap (β≈0.005–0.01) + RandStainNA/LAB-HSV color randomization + white-balance/CLAHE jitter** — replace the geometric rand-m6 | Manufactures **"unseen centers"** from our 2 → attacks the 0.996 separability / the color axis that IS the domain gap. **Biggest OOD-AUROC lever with 2 sources.** Winner used targeted color aug. | retrain (aug only) | FDA (Yang 2020), RandStainNA (Shen 2022) |
| **3** | **Freeze the backbone (linear-probe or LoRA) + WiSE-FT high-α** — stop distorting foundation features | Frozen 0.776 > FT 0.756; lighter-FT exp1 > heavier-FT exps/3. Preserve the OOD-general representation; adapt cheaply. **Winner used DINOv3-ViT-L LoRA (near-frozen).** | cheaper than full FT | LP-FT (Kumar 2022), WiSE-FT (Wortsman 2022) |
| **4** | **Big decorrelated ensemble + SWAD + model soup** — dinov2(exp1) ⊕ dinov3 ⊕ GastroNet-style ⊕ seeds ⊕ aug-strength | The **proven winning wrapper** (40-model soup). Averaging decorrelated rankers + flat minima = variance reduction for the **noisy median** metric. Domain-count-agnostic (works at 2). | many cheap members | SWAD (Cha 2021); every RARE25 top team |
| **5** | **Two-way pAUC / AUC-margin loss on the (linear) head; explore one-class/anomaly scoring on the 99.6%-separable normal manifold** | Optimizes the exact high-recall corner, not mean AUROC. One-class is **unexplored by the whole field** (report flags it) and fits our normal-manifold finding → novelty upside. | head-only retrain | pAUC-DRO (Zhu/Yang 2022); field gap |

**Explicitly SKIP** (need >2 domains or are unstable/BN-dependent here): DANN/GRL (already null), Fishr, SagNet, RSC, CORAL (2-covariance = weak), vanilla Tent (LayerNorm ViT, entropy-collapse on 1-class stacks).

## 18.5 IMMEDIATE ACTIONS

1. **🚨 Resolve best-vs-last submission policy NOW.** If the leaderboard keeps your **LAST** submission, then submitting exps/3 (0.0117) **replaced** and **worsened** your standing vs exp1 (0.0181). **Re-submit exp1 (or exp1⊕exps/3 prob-ensemble) to restore the better score.** If it keeps BEST, exp1 stands and this is non-urgent. *This is a free, high-priority correction.*
2. **Do NOT ship exps/2 or exps/3 as the final** unless it beats exp1 on frozen-transfer / val→test gap. exp1 is the floor to defend.
3. **Cheapest first experiment = lever #1 (score normalization), inference-only, no retrain** — can be prototyped on the existing scores today; then #2 (color aug) as the next retrain.

## 18.6 Revised paper thesis (the honest negative result is STRONGER)

The original CTM/CMI "concept-guided tail mining" spine (§16) was **falsified by the real data** — concept/attention/semi machinery regressed OOD. The honest, defensible, and more novel thesis:

> **"On cross-center Barrett's neoplasia detection, added model capacity overfits the few training centers and *lowers* unseen-center performance; a frozen-backbone linear-probe transfer predicts the leaderboard almost exactly (0.776≈0.756). The operative failure is not tail-ranking but **per-center score-shift**, which we quantify (negatives drift 3.5× across centers) and fix at inference with per-stack score normalization — recovering PPV@90R without retraining."**

That is a clean measurement + diagnosis + targeted-fix paper, corroborated by the winner's recipe, and it uses the honest data instead of chasing a mirage. Secondary contributions from §16 that SURVIVE: the measurement-discipline protocol (§7) and the "why standard SSL misfires for operating-point metrics" negative result (§16 secondary).

**Bottom line to win:** stop adding model complexity; **fix the score-shift (lever #1), simulate centers with color aug (lever #2), freeze the backbone (lever #3), wrap in a decorrelated ensemble (lever #4)** — and defend exp1 as the floor. The ceiling is ~0.035 (winner) at this prevalence; getting there is an **ensemble + recalibration** problem, not a novel-single-model problem.

---

# §19. THE WINNING PLAN — 2026-07-12 (post-3-submissions; 32-agent adversarial workflow + local grounding)

> Derived by a multi-agent workflow: **4 independent winning strategies** (defend-&-recalibrate / domain-invariance / ensemble-maximalist / novelty) → **27 unique levers** → **adversarially verified (24 survived)** → synthesized, then **grounded in local experiments** on the 619-frame val. This is the operative plan; §18 is its diagnosis, §4–§16 are historical. **Base model = exps/2** (the new best; §1).

## 19.1 Situation (honest)
Three real submissions rank **exps/2 (AUROC 0.854) > exp1 (0.845) ≫ exps/3 (0.756)** on both eval sets. The cross-center wall is an **operating-point / score-shift** failure (§18.2), NOT a representation gap capacity can close (frozen-LP transfer 0.776 ≈ fine-tuned 0.756; added capacity *regressed* us). **The win is the RARE25 winner's recipe — decorrelated ensemble + per-member/per-distribution recalibration + acquisition-invariance — layered on a defended exps/2 floor, not a novel single model.** PPV@90R's CI spans ~6× the margins → select on **LOCO AUROC** (stable), treat PPV as a directional tie-breaker. Final rank has a real luck component; we maximize the bootstrap-**median** by cutting variance and de-flooding the tail.

## 19.2 Empirical grounding (local, 619-frame val — SAME-CENTER, so deltas are mechanism-directional, absolutes are mirage)

| transform (on exps/2 unless noted) | pool PPV@90R | pool AUROC | pool FPR@90R | read |
|---|---|---|---|---|
| exps/2 raw | 0.471 | 0.978 | 0.0102 | base |
| **+ per-center robust-z norm** | **0.517** | 0.976 | **0.0085** | ✅ score-norm helps (−FPR) |
| synthetic +0.15 center offset, RAW | 0.433 | 0.978 | 0.0119 | ⬇ score-shift degrades |
| synthetic +0.15/0.30/0.45, **robust-z** | **0.517** | 0.976 | 0.0085 | ✅ **norm CANCELS the offset — invariant to shift** |
| per-center **RANK** norm | 0.10 | 0.951 | 0.082 | ❌ rank-norm HARMFUL (confirmed 2×) |
| **exps/2 ⊕ exps/3 prob-mean** | **0.842** | **0.992** | **0.0017** | ✅ decorrelated ensemble stacks strongly |
| exps/2 ⊕ exps/3 rank-mean | 0.572 | 0.990 | 0.0068 | combo detail is noise-sensitive → LOCO decides |

**Two levers empirically de-risked:** (1) **per-distribution robust-z score normalization** provably cancels an injected center-offset (the diagnosed killer); use robust-z, NEVER rank. (2) **Decorrelated ensemble** is a candidate gain (but see the LOCO correction below). *Caveat: same-center val — must re-confirm on LOCO before shipping.*

### 19.2b — LOCO frozen-LP transfer: the HONEST compass (`phase3/loco_probe.py`, 2026-07-13)
Fit a linear probe on ONE center's FROZEN features, test the OTHER (predicts the leaderboard: dinov3 frozen 0.776 ≈ exps/3 hidden 0.756):

| frozen-LP config | c1→c2 AUROC | c2→c1 AUROC | mean | PPV@90R (boot-median, both dirs) |
|---|---|---|---|---|
| **DINOv2** | **0.910** | **0.949** | **0.93** | 0.032 / 0.043 |
| DINOv3 | 0.776 | 0.895 | 0.835 | 0.015 / 0.034 |
| dv2 ⊕ dv3 ensemble | 0.881 | 0.932 | 0.91 | 0.022 / 0.054 |

**🎯 DECISIVE — DINOv2 features are far more center-invariant than DINOv3** (mean cross-center AUROC **0.93 vs 0.835**). This is the **root cause** of exps/2 > exps/3 and retro-predicts both leaderboard rows. **Corrections to the plan:**
- **Drop DINOv3.** DINOv2 is the backbone for every future member. (Our entire exps/3 direction was the wrong backbone.)
- **The dv2⊕dv3 ensemble HURTS** (0.93→0.91) — dinov3 drags. The earlier same-center val "ensemble helps" (§19.2) was a mirage. **The decorrelated ensemble must be DINOv2 × DINOv2** (frozen-LP ⊕ fine-tuned exps/2 ⊕ seeds/augs), **never dinov3.**
- **Frozen-DINOv2 linear-probe is a top single config** (LOCO 0.91–0.95; max foundation preservation, Kumar LP-FT) — a prime standalone AND ensemble member. Exported to `phase3/cache/frozen_lp_dinov2.npz`.
- Note the frozen-LP LOCO (c1↔c2, ~0.93) overestimates the true 3rd-center by ~0.08 (exps/2 hidden 0.854) — c1↔c2 is easier than an unseen 3rd center. It is a valid **relative** compass, not an absolute.

## 19.3 Ranked levers — 24/27 survived adversarial verification

| Tier | Lever | Mechanism | Fixes | Compute | Risk | Gate |
|---|---|---|---|---|---|---|
| **1** | **Per-distribution robust score alignment** (offline, in-container) | Subtract a robust **low quantile (p5–p20)** of the *pooled* new-center scores (its negative-floor proxy) before the 90R threshold — **not** per-16-frame median/rank (a stack can be all-negative) | score-shift | CPU/local | med | LOCO c1↔c2: FPR@90R↓, PPV median↑, **never-worse guardrail**; stress all-negative stacks |
| **1** | **Decorrelated ensemble + per-member affine recalibration** | Blind prob-average of sane fixed members (DINOv2 exps/2 × frozen DINOv3-LP × seeds); per-member affine baked as constants before averaging | variance (metric is median) | CPU/local | low | LOCO ensemble AUROC ≥ best member; **fixed** decorrelation rule (Spearman<0.85), NO greedy argmin on 2 folds |
| **1** | **Frozen/LoRA base + WiSE-FT high-α** | Head-only/LoRA on frozen backbone; interpolate FT weights toward SSL init (α~0.7, `finetune.py --wise-ft`) | foundation-feature distortion | Colab-cheap | low | α by paired LOCO ΔAUROC both dirs; never below frozen-LP floor |
| **2** | **Acquisition-domain color aug** (FDA β~0.005–0.01 + RandStainNA/LAB-HSV + WB/CLAHE), **replacing** geometric rand-m6 | Synthesize unseen-center appearance from 2 centers | color-gap→score-shift | Colab-retrain | med | center-probe AUROC <0.90 **AND** LOCO AUROC↑; light-FT only |
| **2** | **Third decorrelated family** (GastroNet/ResNet50 frozen features) | CNN inductive bias → rank-decorrelated errors (winner's pairing) | variance | Colab-cheap | med | admit only if ensemble LOCO AUROC↑ AND Spearman<0.9 |
| **2** | **SWAD** dense weight averaging | flat-minima average over FT trajectory | val→test gap | Colab | low | SWAD vs last-ckpt LOCO AUROC |
| **2** | **Semi-consistency on 144k pool, LIGHT arch only** | Mean-Teacher/one-sided-PU (as in exps/2, which it HELPED) — keep it on DINOv2 mean-pool, never on heavy arch | center-robustness | Colab | med | it already cleared: exps/2>exp1. Keep light; LOCO-gate any change |
| **3** | **pAUC/AUC-margin head** (`soft_pauc90 q=0.2`, float32, built) | shapes the 90R corner on frozen features | tail-ranking (in-domain) | CPU | low | paired LOCO A/B beats BCE at matched AUROC |
| **3** | **Mahalanobis/kNN-to-normal as an ENSEMBLE MEMBER only** | one-class score vs per-center negative manifold | decorrelation | CPU | high | LOCO ranks pos above benign outliers; kill on FP flooding |

## 19.4 Explicitly REJECTED (falsified or unsafe)
- **More model capacity / heavy arch** (DINOv3+attention+448+strong-aug+long-train) — overfits 2 centers, regressed us (exps/3). **Proven.**
- **Raw one-class anomaly as the PRIMARY score** — on a new center *everything normal reads anomalous* → amplifies score-shift. Only viable as a decorrelated member.
- **Per-center RANK normalization** — destroys the tail (grounding: 0.10). Use robust-z.
- **Center-debias k-sweep as ensemble members** — all share the single 2-center axis (near-identical, no decorrelation).
- **Concept-bottleneck representation** — PPV collapses to the wall (retired, §16 note).
- **DANN / CORAL / Fishr / SagNet / vanilla-Tent** — null/unstable with 2 domains or BN-dependent.
- **Global calibration as a PPV lever, same-center-val selection** — no-op / mirage (§5, §18).

## 19.5 IMMEDIATE FREE ACTIONS (this week, no retrain)
1. **Floor is secure** — exps/2 is both our **latest AND best** submission (0.0177 PPV / 0.854 AUROC). Whether the board keeps best or last, we sit at our best. **Rule: never submit anything that doesn't beat exps/2 on LOCO.**
2. **Build the LOCO harness** (extend `phase3/evaluate.py`): c1→c2 & c2→c1 reporting AUROC, PPV@90R (bootstrap median+CI), FPR@90R, + a per-member **Spearman matrix**. The decision instrument for every lever.
3. **Cache per-frame scores** for exps/2 (TTA) and a **frozen DINOv3 ViT-B linear-probe** on all labeled val, tagged by center/stack → all recalibration/ensemble experiments become instant CPU.
4. **Prototype low-quantile per-distribution alignment + robust-z** on cached scores; LOCO-gate with the never-worse guardrail (grounding already shows +mechanism).
5. **Build exps/2 ⊕ frozen-DINOv3-LP recalibrated prob-average ensemble.** (Grounding: ensemble is the biggest confirmed gain. Decide by LOCO whether the second member is the frozen-LP or exps/3 — the frozen-LP avoids exps/3's overfit; exps/3 helped on val but is weak on the real center.)

## 19.6 Week-by-week calendar
- **W1 (CPU, no submit):** actions 1–5. Ship candidate = exps/2 ⊕ frozen-DINOv3-LP + affine recal + low-quantile alignment **only if it beats exps/2 on BOTH LOCO directions**; else keep exps/2. Expected AUROC 0.854→~0.86, PPV median ~0.018→~0.022. Kill if either LOCO direction regresses.
- **W2 (Colab):** WiSE-FT α-sweep + expand the frozen-probe bank (DINOv3, GastroNet features). Blind-average, fixed decorrelation rule, no grid argmin. Submit only if LOCO AUROC gain > fold-noise. Expected +0.005–0.015 AUROC, tighter CI.
- **W3 (Colab):** color-aug light-FT member (FDA+RandStainNA, no rand-m6). Gate: center-probe <0.90 AND LOCO AUROC↑; kill if AUROC drops. Add as decorrelated member. Expected +0–0.02 (uncertain).
- **W4:** add GastroNet/CNN member + SWAD if they clear the admission gate; finalize container (parity + `--network none`); pAUC head as a free A/B.

## 19.7 Submission strategy
~1/week, measurement-dominated. **Submit only when the LOCO AUROC gain exceeds the fold-noise band; PPV is a noisy tie-breaker, never a selector.** Every candidate beats the exps/2 LOCO floor on both directions or we ship exps/2 unchanged. Never spend a submission on a same-center-val improvement (mirage).

## 19.8 Honest ceiling & paper
Single-model cross-center AUROC is capacity-saturated ~0.77–0.85. Realistic pooled target ~0.85–0.88; PPV@90R median ~0.022–0.030, optimistic tail brushing the winner's 0.035. **0.02–0.035 is a genuinely good result; 0.06 is the field ceiling; 0.6 is impossible.** CI will still span ~[0.011, 0.07]. **Paper survives any leaderboard draw — "Recalibration beats representation under center shift":** foundation features are 99.6% center-separable and capacity-saturated (frozen-LP == fine-tuned; added capacity regresses); the PPV@90R failure is a per-center **negative score-shift**, not a ranking deficit (quantified by the neg-p95 gap + the synthetic offset→robust-z-recovery experiment); a cheap **offline per-distribution score alignment + per-member affine recalibration + decorrelated ensemble** closes most of the gap **without retraining**, under a LOCO+bootstrap selection discipline. Rigorous negative results (DANN, concept-bottleneck, one-class-as-primary, geometric aug, capacity, same-center mirage) are the second contribution.

## 19.10 IMPLEMENTED (2026-07-13) — code shipped + the next winning train

All Tier-1/2 levers are now coded, unit-tested, and gated. Backbone decision from §19.2b is baked into the recipe: **DINOv2, drop DINOv3.**

**Code landed (`main`):**
- `phase3/loco_probe.py` — the honest cross-center compass (frozen-LP LOCO c1↔c2, AUROC/PPV@90R/FPR@90R + bootstrap CI). Produced the decisive **dinov2 0.93 ≫ dinov3 0.835** result; exports `frozen_lp_dinov2.npz`.
- `phase3/frozen_lp_member.py` — the **frozen-DINOv2 linear-probe member** (top LOCO config; max foundation preservation). Standalone scorer `score_frames()` + `refit()` on all labeled data → shippable `(mean,scale,coef,intercept)` npz. Usable as its own submission OR a decorrelated member for exps/2 (dinov2-FT ⊕ dinov2-frozen-LP; **never dinov3**).
- `phase3/finetune.py` — **`--aug domain`** (§19 lever #2: `AcquisitionAug` = white-balance + HSV stain jitter + reference-free FDA amplitude perturbation + gamma; attacks the per-center color axis, NOT geometric rand-m6; also drives the semi strong-view for center-nuisance consistency) + **`--img`** flag (dinov2 recipe = 336).
- `RARE25-Submission/model/viscera_model.py` — **§19 lever #1 `SCORE_ALIGN_Q`** (per-stack low-quantile de-flooring; monotone within a stack, cancels per-center offset across the pooled test; default OFF, A/B-ready; proven shift-invariant in grounding).

**The next winning train (Colab GPU) — LOCO-GATE the domain-aug lever, then ship:**
```bash
# 1) LOCO GATE (honest): does color/FDA domain-aug beat the exps/2 recipe on the UNSEEN center? Both directions.
for HO in center_2 center_1; do
 for AUG in mild domain; do
  python -m phase3.finetune --backbone dinov2 --img 336 --holdout $HO --epochs 12 --unfreeze 6 \
    --wise-ft 0.7 --init concept_encoder.pt --aug $AUG \
    --semi-manifest phase3/cache/unl_manifest.npz --semi-n 300000 --semi-bs 192 --semi-steps 10 \
    --out loco_${AUG}_${HO}.pt   # prints LOCO AUROC/AUPRC on the held-out center
 done
done
# ADOPT --aug domain ONLY if its held-out AUROC >= mild on BOTH holdouts (else keep mild = exps/2 recipe).

# 2) SHIP (winning recipe): DINOv2 mean-pool @336 + [domain|mild per the gate] + light semi + WiSE-FT, 3 seeds
for S in 0 1 2; do
  python -m phase3.finetune --backbone dinov2 --img 336 --holdout none --epochs 12 --seed $S --unfreeze 6 \
    --wise-ft 0.7 --init concept_encoder.pt --aug domain \
    --semi-manifest phase3/cache/unl_manifest.npz --semi-n 300000 --semi-bs 192 --semi-steps 10 \
    --out ship_dv2dom_seed$S.pt
done
# 3) FROZEN-LP member (top LOCO config; ensemble with the ships): refit on all labeled + drop into the container
python -c "from phase3.frozen_lp_member import refit; import csv; \
  L=lambda f:( [r['path'] for r in csv.DictReader(open(f)) if r.get('aug','orig')=='orig'], \
               [int(r['label']) for r in csv.DictReader(open(f)) if r.get('aug','orig')=='orig']); \
  tp,tl=L('dataset/train.csv'); vp,vl=L('dataset/val.csv'); \
  refit('dinov2.pth', tp+vp, tl+vl, 'phase3/cache/frozen_lp_dinov2_full.npz')"
```
**Container A/B options** (each ONE submission, LOCO-gate first): (a) exps/2 as-is (current best); (b) exps/2 + `SCORE_ALIGN_Q=0.10` (de-flooding); (c) new dinov2+domain ships; (d) ships ⊕ frozen-DINOv2-LP. Ship whichever wins LOCO; never submit a same-center-val-only gain.

## 19.9 What would actually beat the winner
Their 0.035 = ~40 members + affine recalibration + color aug. We **cannot out-ensemble 40 models** with 2 centers + 127 positives. Our only lever with headroom past 0.035 is **per-distribution score alignment that generalizes to the true hidden center** — if the new center's negative floor is estimable from its own pooled test scores and subtracting it de-floods FPR@90R more cleanly than their per-prevalence affine did. That single offline, network-free, non-monotone transform — proven in grounding to be shift-invariant, and LOCO-gated to never regress below exps/2 — is the bet. Everything else defends the floor and shrinks variance so the noisy median lands high.

---

# §20. THE NOVEL SPINE — CRISP (Concept-Residualized Invariant-operating-point Scoring)

> From a 17-agent research→ideate→adversarial-verify→synthesize workflow (2026-07-15). 5 novel methods invented, 4 survived; the adversarial pass **demoted the negative-tail-alignment idea (NTSA)** — its "minimal sufficient invariance" claim is false (recall=90% is set by the *positive* low-quantile too) — and selected **CRISP** as the spine because it is the only idea that is simultaneously **2-center-safe, offline/per-stack-safe, AND locally verifiable with a RANK-INDEPENDENT mechanism metric.** That last property is decisive under a measurement-dominated metric (CI ~6×, MDE > any plausible margin): **we can prove the mechanism fires even when PPV cannot resolve it.**

## 20.1 The spine — CRISP
**Mechanism.** Model the lesion logit as `s(x) = lesion(x) + f(nuisance(x)) + noise`, where `nuisance(x)` = the few acquisition/scope/border/overlay concepts that (a) shift across centers by construction and (b) are label-orthogonal on source. We already predict these **offline, per-image, from the frozen GastroNet-DINOv2 aux concept heads** (no VLM at deploy). On abundant **source NEGATIVES** (labeled negs + ~6k pool ≈ pure neg at 1% prevalence) fit `μ₀(n)=E[s|n,y=0]` and `σ₀(n)=SD[s|n,y=0]` (isotonic / 2-layer). Deploy the **Frisch–Waugh–Lovell residual** `r(x) = (s − μ₀(n̂(x))) / σ₀(n̂(x))`. When a new center pushes brightness/scope/border, `μ₀(n)` moves *with* the concept and subtracts the nuisance-explained lift **per frame**, so negatives stay centered while the nuisance-orthogonal lesion residual survives.

**Why it wins (tie to FPR@90R).** The flood is negatives risen for a *nuisance* reason; CRISP subtracts a per-frame nuisance-predicted baseline → those negatives fall back below the recall-preserving threshold → **lower new-center FPR@90R → higher PPV@90R** at fixed recall. `r` is a per-frame, feature-dependent, **non-monotone** transform across the pooled test → a legitimate rank-metric lever (a global affine would be a no-op). Also variance-reduces the bootstrap-median (threshold no longer set by a stochastic nuisance tail).

**Novelty.** Closest prior art: ComBat site-harmonization (discrete site, feature-level), double-ML/FWL residualization (never applied to a *detector score* for OOD), multicalibration. CRISP's new assembly: **source-only, score-level FWL residualization against *learned continuous interpretable nuisance concepts* for cross-center operating-point transfer, requiring ZERO target-center statistics** → provably cannot degenerate into the ruled-out per-center rank-norm, needs no 3rd-center samples. Not a concept *bottleneck* (representation untouched; concepts only estimate a baseline to subtract).

**Honesty.** Expected PPV lift is **small-to-null** — FWL removes only the *concept-spanned fraction* of a 0.996-separable drift. **The paper's load-bearing claim is the MECHANISM, not the rank.**

## 20.2 Support levers (compose with the spine)
- **NTSA (training-time, default OFF `--neg-align-weight 0`)** — sliced-W2 alignment of the *upper* negative-logit quantiles across source centers. Shapes negatives to co-locate *before* scoring; CRISP residualizes the *remaining* drift at deploy. Ship as an ensemble member + secondary paper angle (reframed as "empirically dominant factor", NOT a sufficiency theorem).
- **PANC-shift only** — per-stack low-quantile null-floor with EB-shrinkage toward the offline 144k anchor (cheap safe last-mile). **Drop the scale/affine mode** (= the ruled-out per-group rank-norm; ~16-frame σ adds variance).
- **DeFLoRA-as-engineering** — frozen-GastroNet-DINOv2 LoRA-bank ensemble (r=8, M=8–16) + WiSE-FT + frozen-LP cross-family member → toward the winner's ~40-model decorrelated recipe. Keep the ensemble + **plain** de-floor; do NOT sell style-gradient decorrelation as validated novelty.

## 20.3 Explicitly rejected (adversarial pass)
- **COCTA** (concept-CAV feature transport) — un-gateable by 2-center LOCO (like MixStyle); orthogonality to the single c1–c2 axis gives no guarantee vs a 3rd center's new drift. *Non-extrapolation wall.*
- **NTSA sufficiency theorem** — false (recall set by positive low-quantile too). Keep NTSA as an empirical support lever, drop the theorem.
- **PANC affine/scale** — two-moment version of the ruled-out per-group rank-norm.
- **MixStyle as a headline** — un-gateable (§19; validated slight-hurt on frozen-LOCO because single-center training can't mix cross-center).

## 20.4 Validation — falsifiable, RANK-INDEPENDENT (the decisive property)
Extend `phase3/loco_probe.py` (the compass that predicted the leaderboard):
1. **Mechanism metric (headline, noise-free):** cross-center **negative-score drift reduction** — KS / mean-gap between held-out-center negative-`s` and source negative-`s`, **raw vs residualized `r`**, on BOTH LOCO legs. Claim = drift shrinks with recall preserved.
2. **Synthetic-shift battery:** `--aug domain` (FDA/RandStainNA) moves nuisance concepts; raw floods (FPR@90R↑) while `r` holds. Report **threshold-transfer recall** (apply source 90R threshold to shifted set → should stay ≈0.90).
3. **Recall tripwire + fallback-to-raw** if residualizing drops held-out recall (concept became label-correlated).
4. **Paired bootstrap** on AUPRC/FPR@90R with `mde()`. **Kill criterion:** if drift-reduction doesn't clear MDE on both legs → CRISP ships as raw; NTSA/PANC-shift/DeFLoRA carry the submission.

## 20.5 Implementation (our stack)
- **k≈5–8 aux nuisance heads** on frozen features (reuse `pretrain_concept.py` + `build_concept_targets.py` VLM labels); select by low `|corr(concept,label)|` on source + high cross-center variance (`concept_audit.py`). Per-image, `--network none` safe.
- **`phase3/nuisance_residual.py`** — fit `μ₀(n),σ₀(n)` on source negatives, save beside the `.pt`.
- **`phase3/infer.py` / `viscera_model.py`** — after `s, n̂`, emit `r=(s−μ₀(n̂))/σ₀(n̂)`; per-stack pooling unchanged.
- **`finetune.py`** — `--neg-align-weight/-qlo/-qhi/-Q` (NTSA, default 0) + `StratifiedCenterBatchSampler` + thread `center_id`; LoRA bank behind `--lora-rank/--lora-members`.
- **Ship** — `--holdout none` both-center model, WiSE-FT 0.7, 3-seed × LoRA ensemble, residualize each member before pooling, ONE global threshold.

## 20.6 The paper
**Title:** *Partialling Out the Scanner: Concept-Residualized Score Calibration for Cross-Center Operating-Point Transfer in Rare-Lesion Detection.*
**Thesis:** cross-center failure at a high-recall operating point is an *operating-point* failure from observed nuisance (acquisition/scope) covariates confounding the score — not a discrimination failure; partialling them out per-frame via an offline concept predictor yields a nuisance-orthogonal score whose 90R threshold transfers with **no target statistics**.
**Contributions:** (1) CRISP — source-only score-level FWL residualization against learned nuisance concepts; (2) a **rank-independent mechanism protocol** (negative-drift KS + threshold-transfer recall under LOCO/synthetic-shift) for validating operating-point methods under a measurement-dominated metric; (3) **honest negative results:** feature-invariance (DANN/CORAL/COCTA) + per-group rank-norm are unsound/harmful at 2 centers; NTSA sufficiency is empirical not a theorem.
**Key figure:** overlaid source vs new-center *negative*-score histograms, raw (flooding the 90R line) vs residualized (re-aligned), with threshold-transfer recall annotated.
**Strongest objection → rebuttal.** *6 concepts can't span a 0.996-separable drift.* → we don't claim full cancellation; we quantify the *fraction* removed, show it clears MDE on drift-reduction with recall preserved, fit on abundant negatives (not 127 positives) → never the DANN 2-center overfit, never ships worse than raw.

## 20.7 Honest expected outcome + fallback
Hidden-test estimate **PPV@90R ≈ 0.018–0.025** (exps/2 0.0177 + ensemble/de-floor variance reduction; CRISP adds a small median lift + CI tightening); ceiling ~0.035. **The defensible deliverable is the mechanism ablation, not the rank.** **Fallback if CRISP fails its gate:** ship exps/2 recipe + WiSE-FT + LoRA-bank ensemble + plain per-stack de-floor + PANC-shift — all 2-center/offline/per-stack-safe — and publish CRISP as the negative-result contribution.

## 20.8 CRISP TESTED → FAILED its gate; REFRAME to the 3 pillars (2026-07-15)
Ran `phase3/crisp.py` (rank-independent mechanism test, both LOCO legs). **CRISP does NOT reduce cross-center negative-score drift** (KS 0.943→0.945, 0.893→0.909; transfer FPR@90R unchanged; target AUROC −0.02). **Why (concept audit):** only **2 of 14** nuisance concepts actually shift across centers (`overlay_graphics` center-shift 0.53, `black_border` 0.11); the rest are dead/near-constant. The detector score doesn't depend on overlay/border, so residualizing removes ~nothing of a 0.996-separable drift. **CRISP is falsified as a positive method** (kept as a negative-result contribution). This also confirms: **"cross-center transfer" is an indefensible framing at 2 centers.**

**REFRAMED operative method (3 pillars — the honest, defensible frame):**
1. **Semi-supervised** — Mean-Teacher + one-sided-PU on the 144k GastroNet-domain pool (measured win: exps/2 > exp1). Label-efficiency, not a center claim.
2. **VLM-Concept Teaching** — Stage-1 (`pretrain_concept.py`) distills the 35 clinical concepts as dense supervision; **role-aware routing** = diagnostic→trunk, nuisance→**GRL** (adversarial suppression of the audit-proven center-cue concepts `overlay_graphics`/`black_border`). This IS the "OOD layer."
3. **OOD generalization levers** — `--aug domain` (color/FDA), MixStyle (param-free), WiSE-FT 0.7. All train-time, ship graph unchanged.

**Winning recipe (Colab-ready, `phase3/colab_full_pipeline.ipynb`, dinov2@448):** Stage-1 concept-teaching (GRL nuisance-suppression) → **gate cell (BUNDLE vs BASE on held-out center)** → Stage-2 ship `--backbone dinov2 --img 448 --init concept_encoder.pt --cg-head --aug domain --mixstyle --wise-ft 0.7 --semi-* --epochs 15`, 3 seeds → container. **Honest:** the measurable wins are dinov2 backbone + semi + (gated) color-aug + concept-teaching; MixStyle/attention are optional un-gateable/below-noise riders. Run the gate before shipping.
