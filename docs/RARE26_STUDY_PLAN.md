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
