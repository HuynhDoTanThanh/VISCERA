# RARE2026 — Experiment Table (done vs not-done)

**Goal:** cross-center Barrett neoplasia, metric = PPV@90R @1% prevalence (hidden NEW-center test).

## Legend
- **Status:** ✅ done · 🔄 coded, not run on leaderboard · ⬜ NOT done yet
- **Finetune (FT) strategy:** `full-FT` = unfreeze last K blocks + head · `frozen-LP` = frozen backbone + linear probe · `head-only` · `LoRA` · `pretrain` = Stage-1 · `post-hoc` = inference-time · `train-time` = loss/aug during training
- **Measured on:** 🟢 **LB** = real hidden new-center leaderboard (TRUTH) · 🔵 **LOCO** = frozen-LP c1↔c2 (honest compass, predicted the LB) · 🟡 **val** = same-center (MIRAGE) · ⚪ **synth** = directional only
- ⚠ 31-positive val → any AUROC delta < ~0.03 is **below the noise floor** (unreliable)

---

## 1. Main results — the 5 real submissions
| exp | encoder | FT | header | concept | semi | **img** | aug | 🟢 PPV@90R | 🟢 AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|
| exp1 | GastroNet-DINOv2 ViT-B | full-FT | cls⊕mean | ✗ | ✗ | **336** | mild | 0.0181 | 0.845 | 0.356 |
| **exps/2** 🏆 | GastroNet-DINOv2 ViT-B | full-FT | cls⊕mean | ✓ | ✓ | **336** | mild | **0.0177** | **0.854** | 0.401 |
| exps/3 | DINOv3 ViT-B | full-FT | CG-AMIL attn | ✓ | ✓ | 448 | strong-geom | 0.0117 | 0.756 | 0.300 |
| exps/4 | GastroNet-DINOv2 ViT-B | full-FT | CG-AMIL attn | ✓ | ✓ | 448 | domain+mixstyle | 0.0155 | 0.829 | 0.355 |
| exps/5 | GastroNet-DINOv2 ViT-B | full-FT | cls⊕mean | ✓ | ✓ | 448 | mild | 0.0128 | **0.797** | 0.351 |

**exp-4 validation (RARE25 val):** PPV@90R 0.0114 · AUROC 0.8626 · AUPRC 0.6109. **exp-5 validation (RARE25 val):** PPV@90R 0.0121 · AUROC 0.8481 · AUPRC 0.5771.

### ⚑ exp-5 analysis — RESOLUTION @448 is the regression, NOT the bundle (corrects the §7 conclusion)
exp-5 = exp-4 **minus** the bundle (mean-pool, no CG-AMIL / MixStyle / aug-domain), held at @448. The plan was "drop the bundle → recover exps/2." Instead exp-5 came out the **WORST dinov2 run**: PPV 0.0128, AUROC **0.797** — below even exp-4. The clean read is on **AUROC (stable; PPV@90R is noise at 31 pos, all CIs overlap ~0.010–0.11)**:

| | @336 (exp1, exps/2) | @448 (exp4, exp5) |
|---|---|---|
| LB AUROC | 0.845, **0.854** | 0.829, **0.797** |

**Both @336 runs beat both @448 runs — a consistent 4-point signal, not noise.** The regression vs exps/2 was never the bundle; it was **resolution**. I isolated the wrong variable in §7 (exp5 vs exp4 held @448 to test "the bundle", but the real killer vs exps/2 was the @336→@448 change that exp5 kept). Mechanism: @448 exposes more high-frequency detail → more scope/acquisition-specific texture the model latches onto → worse transfer to the true 3rd center. The honest 2-center harness (§7) **could not catch this** — the @448 penalty only manifests on the unseen 3rd center, which no 2-center bench sees (it read the @448 anchor at AUROC 0.992 — the mirage).

Nuance: **at @448 the bundle actually helps** (exp4 0.829 > exp5 0.797) — aug-domain + MixStyle partially regularize the high-res center shortcuts — but neither recovers the @336 baseline. **Fix = stay at @336, not "@448 + bundle" nor "@448 simple."** The winning recipe remains exps/2 @336; it is still the best submission after 3 attempts (exp3/4/5) to beat it all regressed.

### exp-4 analysis — the bundle does NOT beat the simple anchor (2nd confirmation)
exp-4 = the **winning dinov2 backbone** + everything we could bolt on (@448, CG-AMIL attention head, MixStyle, `--aug domain`). It **did not improve** on exps/2: PPV −0.0022, AUROC −0.025, AUPRC −0.046 — every delta **within the noise floor** (AUROC Δ<0.03; PPV 95% CIs overlap almost entirely, 0.010–0.11). So exp-4 ≈ exps/2 statistically, at **~2× the compute** and much more complexity.

This is the **2nd time** the `CG-AMIL + @448 + strong-aug` bundle fails to beat `cls⊕mean @336` (1st = exps/3 on dinov3, which actually regressed hard). Controlling the backbone (both dinov2) isolates the cause: **the header/resolution/aug bundle is the dead weight, not the encoder.** Occam → the anchor stays **exps/2** (simple cls⊕mean @336 + concept + semi). Do **not** spend the next submission adding more single-model levers.

The real new asset is the **decorrelated CNN member** (§H): ConvNeXt head-only LP passes LOCO at AUROC **0.932 / 0.976** — the diversity that single-model tweaks can't buy.

---

## 2. Master experiment table

### A · ENCODER / backbone
| method | FT strategy | status | measured | result | verdict |
|---|---|---|---|---|---|
| **GastroNet-DINOv2 ViT-B** | full-FT | ✅ | 🟢 LB | AUROC **0.854** | ✅ **the backbone** |
| GastroNet-DINOv2 ViT-B | frozen-LP | ✅ | 🔵 LOCO | **0.929** | best compass config |
| DINOv3 ViT-B (generic) | full-FT | ✅ | 🟢 LB | 0.756 | ✗ −0.10 |
| DINOv3 ViT-B (generic) | frozen-LP | ✅ | 🔵 LOCO | 0.835 | ✗ |
| DINOv2 ⊕ DINOv3 | frozen-LP | ✅ | 🔵 LOCO | 0.906 | dinov3 **drags** |
| GastroNet ResNet50 (CNN) | full-FT | ⬜ | — | — | not public → self-train |
| Public DINOv2/DINOv3 **ViT-L** | full-FT / frozen-LP | ⬜ | — | — | free, testable — **next** |
| Self-trained concept-CNN (on 144k pool) | full-FT | ⬜ | — | — | GI-matched diversity — **proposed** |
| SurgMotion-L (V-JEPA2, surgical) | full-FT | ⬜ | — | — | gated + video/surgical mismatch |

### B · FINETUNE STRATEGY (the added factor)
| strategy | status | measured | result | verdict |
|---|---|---|---|---|
| **full-FT** (unfreeze last 6 blocks + head) | ✅ | 🟢 LB | exps/2 0.854 | ✅ used for all ships (LoRA "maybe weak") |
| **frozen-LP** (backbone frozen) | ✅ | 🔵 LOCO | 0.929 | best OOD compass; max foundation-preservation (Kumar LP-FT) |
| head-only | 🔄 | — | — | `--head-only` coded, not run |
| **LoRA** | ⬜ | — | — | NOT done (user: prefer full-FT) |
| WiSE-FT anchor (α=0.7) on full-FT | ✅ | 🟢 used | — | ✅ recovers OOD, prevents drift |

### C · HEADER / pooling
| method | FT | status | measured | result | verdict |
|---|---|---|---|---|---|
| **cls ⊕ mean** | frozen-LP | ✅ | 🔵 LOCO | **0.929** | ✅ current |
| mean-only | frozen-LP | ✅ | 🔵 LOCO | 0.921 | ok |
| cls⊕max / max / cls | frozen-LP | ✅ | 🔵 LOCO | 0.917 / 0.901 / 0.881 | − |
| CG-AMIL attention-MIL | frozen-LP | ✅ | 🔵 LOCO | 0.943* | *noisy 0.89–0.94 = **below noise floor** |
| CG-AMIL attention-MIL | full-FT | ✅ | 🟢 LB | regressed (exps/3 **AND** exps/4) | ✗✗ 2× no-gain under full-FT — **retire** |

### D · CONCEPT-SUPERVISED PRETRAINING (VLM-Concept Teaching)
| method | FT | status | measured | result | verdict |
|---|---|---|---|---|---|
| concept-init + semi | pretrain→full-FT | ✅ | 🟢 LB | 0.854 (+0.009) | ✅ but confounded w/ semi |
| concept as representation spine | pretrain | ✅ | prior | ~null vs SSL | retired |
| GRL center-adversarial routing | pretrain | ✅ | 🟢 LB | null on 3rd center | 2-center shortcut |
| **CRISP** (concept-residual OOD score) | post-hoc | ✅ | 🔵 LOCO | drift 0.943→0.945 | ❌ **FAILED** (nuisance concepts don't span drift) |

### E · OOD LAYER / generalization
| method | FT | status | measured | result | verdict |
|---|---|---|---|---|---|
| WiSE-FT (weight-space) | post-hoc | ✅ | 🟢 used | anchor | ✅ keep |
| color/FDA aug (`--aug domain`) | train-time | ✅ | 🟢 LB | in exp-4 bundle, no gain | ✗ no measurable LB win (confounded w/ @448+attn) |
| MixStyle (feature-stat mixing) | train-time | ✅ | 🟢 LB | in exp-4 bundle, no gain | ✗ un-gateable rider; drop |
| per-stack/center score de-floor (`SCORE_ALIGN_Q`) | post-hoc | ✅ | 🔵 **LOCO §7** | no-op on ViT (0.543→0.543); harmful on CNN (AUROC 0.950→0.837) | ❌ **DEAD** — no per-center floor gap even on the honest bench |
| per-center robust-z norm | post-hoc | ✅ | 🟡 val | exp4 0.748→**0.664** (hurts); prior 0.471→0.517 | ✗ **HARMFUL** on same-center (IQR-divide adds noise when centers align) |
| DANN / CORAL / Fishr / Tent | train/post | ✅ | prior | null / need >2 domains | ✗ rejected |

### F · LOSS
| method | FT | status | measured | result | verdict |
|---|---|---|---|---|---|
| **BCE + pairwise-rank + soft-pAUC@90 (q=0.2)** | train-time | ✅ | 🟢 active | — | ✅ the tail objective |
| OHEM tail-margin (`--ohem-k`) | train-time | 🔄 | — | — | coded, not run |
| logit-adjusted BCE | train-time | ⬜ | — | — | NOT done |
| feature-space positive synthesis | train-time | ✅ | 🔵 LOCO | +0.020 AUPRC / +0.003 PPV | weak-positive support lever |
| generative (diffusion) pixel positives | train-time | ⬜ | — | — | NOT done (Track B heavy) |

### G · SEMI-SUPERVISED LOSS (144k pool)
| method | FT | status | measured | result | verdict |
|---|---|---|---|---|---|
| **Mean-Teacher + one-sided-PU** (light arch) | train-time | ✅ | 🟢 LB | 0.854 (**+0.009**) | ✅ **measured win** |
| same semi on HEAVY arch | train-time | ✅ | 🟢 LB | regressed (exps/3) | ✗ (confounded) |
| consistency w/ color-aug strong view | train-time | 🔄 | — | — | coded (`--aug domain` semi), not gated |

### H · ENSEMBLE / FUSION (inference)
| method | status | measured | result | verdict |
|---|---|---|---|---|
| 5-view TTA + 3-seed prob-ensemble | ✅ | 🟢 shipped | — | ✅ baseline wrapper |
| multi-scale TTA (448+384+512) | ✅ | 🟡 val | no help; hurts c1 | ✗ |
| CNN member — ConvNeXt-**Tiny** head-only LP | ✅ member | 🔵 LOCO | AUROC **0.932 / 0.976** (c2/c1) | ✅ strong + decorrelated |
| CNN member — ConvNeXt-**Large** head-only LP | ✅ member | 🔵 LOCO | AUROC **0.909 / 0.965** (c2/c1) | ✗ **no gain over Tiny** (worse, within noise) — revert to Tiny |
| **D2F+ ViT ⊕ CNN (any weight)** | ✅ | 🔵 **LOCO §7** | ViT alone PPV **0.543**; every w<1 drops it (0.33→0.08); CNN tail-poisons | ❌ **FAILED honest harness — drop the CNN member** |
| decorrelated multi-backbone (dinov2⊕dinov3) | ✅ | 🔵 LOCO | dinov3 drags 0.929→0.906 | ✗ (use CNN member instead) |
| per-member affine recalibration (→1%) + per-stack de-floor | ⬜ | — | — | **the real operating-point lever — do this FIRST** |

---

## 3. NOT-DONE — the queue (ranked by EV to win, re-ranked after exp-5 → resolution finding §8)
| # | item | track | why | blocker |
|---|---|---|---|---|
| 0 | **exps/2 @336 IS the ship** — it is still the best board score (0.0177) after exp3/4/5 all regressed | — | 4-point AUROC signal: @336 > @448. Stop chasing @448 variants | none — weights in exps/2/ |
| 1 | **@336 replicas / small wins on the @336 recipe** — more seeds, or exps/2 + `--aug domain` (color-only, no @448) | A | @336 is the resolution; explore levers WITHOUT changing it | Colab ship @336 |
| 2 | **affine→1% recalibration** on exps/2 (winner's trick) | post-hoc | the ONE lever aimed at the 3rd-center score-shift the 2-center bench can't see; post-hoc, no retrain | faith-based (unverifiable locally) |
| 3 | **logit-adjusted BCE** at @336 | loss | agaldran's robust workhorse for the operating point | small code |
| 4 | Generative (diffusion) hard positives | B novelty | break 127-pos wall (winner didn't) | heavy |
| ~~x~~ | ~~ANY @448 recipe~~ | — | **RETIRED §8** — both @448 runs (exp4, exp5) < both @336 runs on stable AUROC | — |
| ~~x~~ | ~~D2F+ ViT⊕CNN ensemble~~ | — | **FAILED honest harness (§7)** — CNN drags PPV@90R at every weight (tail-poisoning) | — |
| ~~x~~ | ~~per-stack/center de-floor (`SCORE_ALIGN_Q`)~~ | — | **FAILED honest harness (§7)** — no-op on ViT, harmful on CNN; no per-center floor gap | — |
| ~~x~~ | ~~CG-AMIL / MixStyle / aug-domain bundle~~ | — | **retired** — no LB gain (exps/3, exps/4); at @336 the bundle is untested but @448 is the confound | — |
| ~~x~~ | ~~ConvNeXt Large/Tiny member~~ | — | **retired** — member itself doesn't help the ensemble (§7) | — |

---

## 4. THE BETTER SOLUTION (post-exp-4 + D2F+ val run)

**Diagnosis — the bottleneck is the OPERATING POINT, not ranking.** Across every submission AUROC is healthy (0.83–0.85) while PPV@90R sits at 0.012–0.018. A model that *ranks* neoplasia well (AUROC 0.83) but scores PPV 0.015 is not a ranking failure — it is a **score-shift** failure: the 90%-recall threshold learned on the source centers lands in a high-FPR region on the new center (memory: score-shift is the killer). Levers that only improve ranking (bigger backbone, attention head, a 2nd ensemble member) **cannot** move an operating-point metric much — and the D2F+ val run shows this directly.

**What the two new runs proved:**
- **ConvNeXt-Large is not worth it.** CNN member LOCO dropped 0.932/0.976 (Tiny) → **0.909/0.965** (Large). A ~4× bigger frozen encoder made the head-only member *slightly worse* on the honest legs. **Revert to convnext_tiny.**
- **Equal-weight D2F+ HURTS the tail.** On val, rank-averaging the strong ViT anchor with the weaker CNN dragged center_2 PPV@90R **0.396 → 0.048** (AUROC barely moved 0.976 → 0.959). PPV@90R lives entirely in the tail; splicing in a member with a noisier tail poisons the 90%-recall threshold even when average ranking is preserved. (val is a same-center mirage — but the *mechanism* is metric-real.)

**Plan — the decisive experiment is the honest LOCO harness; the operating-point lever must be re-designed:**
1. **de-floor is NOT the silver bullet (local exp, §5).** On exp4 the per-center floor gap is ~1e-4 and `SCORE_ALIGN_Q` de-floor is a literal **no-op** (0.748→0.748); robust-z **hurts** (→0.664). The per-center asymmetry that exists is in the **UPPER negative tail** (center_2 q99=0.101 vs center_1 0.038, 2.6×) — exactly what floods FPR@90R, and exactly what a low-quantile de-floor **cannot** touch. So the operating-point lever must target the **upper tail / threshold region** (per-center high-quantile normalization, or the winner's **affine recalibration to 1% prevalence**), not the floor.
2. **Run the honest LOCO harness (notebook cells D2F-4a/4b)** — the only bench that sees the TRUE new-center shift (each frame scored center-blind). It A/B's de-floor AND sweeps the weighted ensemble on one pooled proxy. Same-center val (cell 18) and the local exp are same-center mirages; this is the decision-maker.
3. **Anchor stays simple** = exps/2 recipe (dinov2 ViT-B @336, cls⊕mean, concept + semi). Not the exp-4 @448/attention bundle (retired, 2× no gain).
4. **Ensemble only if it survives the LOCO gate**, fused **weighted** (anchor ≫ CNN, convnext_tiny), never equal rank-average, and only if it beats the anchor on **both** legs.
5. **Paper novelty stays orthogonal:** honest-negatives (CRISP fail, MixStyle/DANN null, bundle's 2× no-gain, Large≈Tiny, equal-weight-hurts-tail, **de-floor mechanism-mismatch**) + generative positives (#5).

**Bottom line:** ranking is solved (AUROC ~0.83), the operating point is not (PPV ~0.015). But the obvious fix — low-quantile de-floor — is mechanistically mismatched (§5: the flood is in the upper neg tail, not the floor). The decisive next step is the **honest LOCO harness (4a/4b)** to measure the true new-center shift and gate BOTH the (re-designed, upper-tail/affine) operating-point lever AND the weighted ensemble; ship the simple exps/2 anchor + whichever survives.

---

## 5. Local score-shift experiment (2026-07-18, exp4 weights, dataset/val orig, CPU)
Scored the 619 orig val frames with the exp4 ship (1 seed, orig view) and probed the score-shift the operating-point levers assume. **Caveat: exp4 saw both centers → same-center regime; this diagnoses the MECHANISM, not the LB gap.**

| finding | number | implication |
|---|---|---|
| per-center **negative floor** gap (median) | **0.0001** (q10 0.0002 vs 0.0003) | no low-end shift on same-center → de-floor has nothing to cancel |
| per-center **upper neg tail** (q99) | center_1 **0.038** vs center_2 **0.101** (2.6×) | THIS is what floods FPR@90R; a low-quantile de-floor can't reach it |
| pooled PPV@90R: raw → de-floor | 0.7481 → **0.7481** (no-op) | `SCORE_ALIGN_Q` mechanically inert here |
| pooled PPV@90R: raw → robust-z | 0.7481 → **0.6644** (−0.084) | spread-normalization HARMS when centers align |
| single-center de-floor | raw == de-floor (exactly) | confirms the monotonicity subtlety: de-floor is a no-op within one center; only acts across pooled centers |

**Takeaway:** the low-quantile de-floor lever is aimed at the wrong part of the distribution. Redesign the operating-point lever toward the **upper negative tail** (per-center high-q normalization) or **affine→1% recalibration**, and validate it on the honest LOCO harness — not same-center val.

---

## 6. PPV@90R is NOISE-DOMINATED across epochs; AUPRC-selection is right (LOCO center_2, D2F-4a, 2026-07-18)
Full 12-epoch curve of the ViT-anchor LOCO leg (dinov2 @448 simple recipe, holdout center_2). **Correction:** an earlier read of only ep1–5 looked like a monotone decay — the full curve shows it is not.

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10–12 |
|---|---|---|---|---|---|---|---|---|---|---|
| **PPV@90R** | **0.597** | 0.248 | 0.184 | 0.161 | 0.067 | 0.223 | 0.326 | 0.438 | 0.421 | 0.421 |
| AUROC | 0.989 | 0.989 | 0.970 | 0.972 | 0.954 | 0.964 | 0.956 | 0.960 | 0.960 | 0.960 |
| AUPRC | 0.970 | 0.944 | 0.933 | 0.941 | 0.910 | 0.932 | 0.927 | 0.932 | 0.932 | 0.931 |

**PPV@90R swings 0.067–0.597 across epochs with all CIs overlapping ~[0.02, 0.84] (49 held-out pos) — it is noise-dominated, not a clean epoch trend.** AUROC/AUPRC are stable (0.95–0.99 / 0.91–0.97). So: **do NOT epoch-select on PPV@90R** — the existing **selection-on-AUPRC** is correct and it saved ep1 (AUPRC 0.970, which also happens to have the best PPV). The "fewer-epochs" hypothesis from the truncated read is **not supported**; the ship's AUPRC-based selection is fine. (Real caveat still stands: this is one leg, center_2 is optimistic vs the true 3rd center, and 49 pos is very few — read AUROC/AUPRC, treat any single PPV@90R point as noise.)

---

## 7. DECISIVE — honest LOCO harness kills BOTH de-floor and the ensemble (D2F-4a/4b, 2026-07-21)
Completed harness, both legs @448: pooled center-blind proxy (n=619, 31 pos; center_1 scored by the model that held out center_1, center_2 by the model that held out center_2).

**(A) DE-FLOOR (`SCORE_ALIGN_Q=0.10`, on raw scores):**
| member | PPV@90R raw | PPV@90R +defloor | AUROC raw→defloor |
|---|---|---|---|
| ViT anchor | **0.5429** | **0.5429** (exact no-op) | 0.992 → 0.992 |
| CNN member | 0.0752 | 0.0360 (worse) | 0.950 → 0.837 |

**(B) WEIGHTED ENSEMBLE** (rank-fuse `w·ViT+(1−w)·CNN`, raw):
| w_ViT | 1.0 | 0.8 | 0.7 | 0.6 | 0.5 | 0.3 | 0.0 |
|---|---|---|---|---|---|---|---|
| PPV@90R | **0.543** | 0.331 | 0.229 | 0.221 | 0.229 | 0.124 | 0.075 |
| AUROC | 0.992 | 0.989 | 0.987 | 0.984 | 0.980 | 0.971 | 0.950 |

**Verdicts (on the honest bench, not the val mirage):**
1. **de-floor is DEAD** — exact no-op on the ViT anchor (no per-center floor gap to cancel, confirming §5), and actively harmful on the CNN (AUROC 0.950→0.837). Keep `SCORE_ALIGN_Q=None`.
2. **the D2F+ ensemble is DEAD** — the ViT anchor (PPV 0.543 / AUROC 0.992) dwarfs the CNN at the operating point (PPV 0.075) despite the CNN's decent AUROC (0.950). Adding *any* CNN weight drops PPV@90R monotonically — the winner's ResNet⊕ViT lever does not transfer here because our CNN's tail is far noisier than the ViT's. **Drop the CNN member; ship the anchor alone.**
3. **the 2-center wall is the real ceiling** — the ViT anchor already hits AUROC 0.992 / PPV 0.543 on this bench, i.e. there is almost no center_1↔center_2 shift left to exploit, which is exactly why both levers find nothing. But the leaderboard's 3rd unseen center reads 0.015 — a shift **no 2-center bench can measure**. Both levers we built target the (small) 2-center shift, not the (large) 3rd-center one.

*(Caveat: cell 4b's "+defloor" column under (B) applies de-floor to rank-transformed scores — a mis-application that collapses PPV to ~0.02; ignore it. The meaningful de-floor numbers are (A), on raw scores.)*

**Action:** ship the **simple @448 anchor** (queue #1) — no ensemble, no de-floor, no CG-AMIL bundle. The only remaining lever that could touch the 3rd-center shift is **affine→1% recalibration** (post-hoc, faith-based — the bench structurally can't validate it). Everything else on the ranking side is at ceiling on the only honest bench we have. **[SUPERSEDED by §8 — this "ship @448 anchor" action produced exp-5, which regressed; the @448 resolution was the culprit the 2-center harness couldn't see.]**

---

## 8. DECISIVE — RESOLUTION @448 is the regression; @336 is the ship (exp-5 LB, 2026-07-23)
exp-5 (the "simple @448 anchor" §7 recommended) scored **PPV 0.0128 / AUROC 0.797** — the worst dinov2 run. Reading the **stable** metric (AUROC; PPV@90R is noise at 31 pos):

| recipe | img | LB AUROC | LB PPV@90R |
|---|---|---|---|
| exps/2 (concept+semi) | **336** | **0.854** | 0.0177 |
| exp1 (simple) | **336** | 0.845 | 0.0181 |
| exp4 (bundle) | 448 | 0.829 | 0.0155 |
| exp5 (simple) | 448 | **0.797** | 0.0128 |

**Both @336 runs beat both @448 runs — a consistent 4-experiment AUROC signal.** The regression across exp3/4/5 was **resolution**, not the header/aug bundle. Why §7 missed it: the honest LOCO harness ran @448 and read the anchor at AUROC 0.992 — but that is the **2-center bench**, and the @448 penalty only appears on the **true 3rd center** (higher resolution → more scope/acquisition-specific high-frequency texture → worse transfer to an unseen center). No 2-center bench can measure a 3rd-center penalty; only the LB can, and it did.

Nuance already visible in the table: **at @448, the bundle helps** (exp4 0.829 > exp5 0.797 — aug-domain/MixStyle partially regularize the high-res shortcuts) but does not recover @336. So the fix is **not** "add the bundle back"; it is **stay at @336**.

**Correction of the §7 error:** exp-5 was designed to isolate "the bundle" by holding @448 — but the real difference vs the winning exps/2 was the @336→@448 change, which exp-5 kept. Isolated the wrong variable. **Lesson: when a same-2-center bench says a lever is neutral/good, it can still be a 3rd-center regression — resolution/aug that add center-specific capacity are exactly the levers a 2-center bench is blind to.**

**Bottom line:** **exps/2 @336 remains the best submission (0.0177) and is the ship.** After exp3 (dinov3), exp4 (@448 bundle), exp5 (@448 simple) all regressed, the evidence says: keep GastroNet-DINOv2 **@336**, concept+semi, cls⊕mean, WiSE-FT. The only untried lever aimed at the real 3rd-center shift is post-hoc **affine→1% recalibration** on exps/2; everything on the ranking/resolution/ensemble side has been tried and lost.

**Net:** the epoch curve is a warning about *reading PPV@90R at all* at this sample size, not a new lever. The decisive question stays the completed harness (both legs) → de-floor A/B + weighted-ensemble gate (§4).
