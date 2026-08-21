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

---

## 9. DEEP CODE AUDIT (2026-07-23) — one real logic bug, one prevalence bug, and the 300k-generalization lever
Two parallel bug-hunt agents (metric + training loop) + manual audit of preprocessing, semi pipeline, and the 300k pool.

### BUG 1 (logic, cross-center) — finetune-LOCO LEAKS the held-out center → optimistic compass
`finetune.py` adds the `--neg-list` and `--semi-manifest` (288k) pools to training with **no center filter**, and `unl_manifest.npz` has an **empty `center` field** for every unlabeled frame (names are numeric IDs, `source_path=None`) — so they **cannot** be center-filtered. Under `--holdout`, the held-out center's unlabeled frames remain in the semi pool and are trained on as consistency targets = **unsupervised domain adaptation TO the test center**. Consequence: **finetune-LOCO is optimistic.** This is exactly why the §7 harness (semi ON) read AUROC 0.992 / "2-center wall, nothing to exploit" — leak-inflated. The leak-free `loco_probe.py` (frozen-LP, no semi) is the compass that **predicted the leaderboard** (dinov3 0.776 ≈ hidden 0.756). **The ship (`--holdout none`) is unaffected — all LB scores are real; only the local LOCO compass was lying.** Fix: added `--loco-no-semi` (drops both unlabeled pools under `--holdout` for a leak-free labeled-only compass) + a loud LEAK WARNING. → §7's "nothing to exploit" is retracted; there may be more cross-center headroom than the leaked harness showed.

### BUG 2 (prevalence) — `.009` instead of `.01`
`loco_probe.py:70` and a notebook cell used `.009/(.009+.99f)` (0.9% prevalence, and internally inconsistent) — understates PPV up to ~10% and breaks cross-cell comparability. Fixed to `.01/(.01+.99f)`. The trusted harness `evaluate.py` was already correct (curve-point, 1% via resampling, median bootstrap, AUPRC selection).

### Audited CLEAN (no bug)
WiSE-FT mixing (correct two states, direction, in-place save), pos/neg sampler, `bce+rank+pauc` (all three terms, signs, reductions), concept-init loading (loud assert vs silent SSL revert), EMA/Mean-Teacher consistency + one-sided PU, AUPRC-based selection, `AcquisitionAug` (FFT phase preserved, amplitude jittered), and — verified on the real test TIFF — SimpleITK reads RGB identical to PIL (mean abs diff 0.0, no BGR/channel bug). Minor: LOCO gate/selection scores are computed on the pre-WiSE-FT model (documented; ship is the WiSE-FT'd model).

### The 300k-pool finding → exp6
The 288k semi-consistency **strong view** is `a.aug`-dependent: at `mild` it is **geometric** RandAugment, which does NOT perturb the per-center **color** axis (the 0.996-separability root disease). Only `--aug domain` makes it AcquisitionAug (white-balance+HSV+FDA+gamma). So at the winning resolution (@336, `aug=mild`), the 300k pool has only ever taught geometric invariance — its cross-center generalization power was never engaged. Supervised loss →0.0006 by ep8 (memorizes the 127 positives); semi + WiSE-FT are the only regularizers, so pointing them at the color axis is the lever. **exp6 = exps/2 @336 + `--aug domain`** (one change vs the best) engages both the labeled color-randomization AND the 288k color-consistency = the winner's color-aug lever, never tried at @336. Gate it leak-free with `--loco-no-semi` (notebook cell) before submitting.

**Bottom line:** the pipeline is mechanically sound (no score-capping bug in the ship path); the real defects were (1) a leaked LOCO compass that mis-guided our lever selection and (2) the 300k pool teaching the wrong invariance at @336. exp6 fixes (2); `--loco-no-semi` + `loco_probe.py` fix (1) so future gating is honest.

---

## 10. DEEP REVIEW (2026-08-18) — the metric is RANK-ONLY, CG-AMIL was never trained, and exp6 was confounded

10 parallel review agents over every subsystem + web verification of the challenge rules. Four findings change what
we should do; three earlier conclusions in this document are **retracted**.

### 10.0 CALENDAR — the ranking event is 8 days after open-dev closes *(verified on grand-challenge.org)*
| phase | dates | submissions |
|---|---|---|
| Open Development | → **Aug 31 2026 EoD** | leaderboard practice only — **not the final ranking** |
| **Closed Testing** | **Sept 1–7 2026** | **ONE per team**, "only the most recent will be evaluated" |
| Report | → Sept 14 2026 | 2–3 pages |

**The open-dev board score (exps/2 = 0.0177, top-1 0.0271) does not decide anything.** Every remaining open-dev slot
should be spent *validating the container we will upload once in September*. Also verified: `phases-rules/` requires
**code published under the MIT license to be eligible for the final leaderboard** — `RARE25-Submission/LICENSE` is
currently **CC-BY-NC-4.0**, an eligibility blocker. External data + public pretrained models are explicitly allowed
with disclosure. No rule restricts test-time/transductive methods.

### 10.1 MECHANISM — PPV@90R depends ONLY on the rank order of our scores
From `rare26.grand-challenge.org/task-evaluation/`, verbatim: *all* non-dysplastic images are retained in every
iteration; neoplasia images are *"sampled with replacement… targeting a ratio of one neoplasia case per 100
non-neoplasia cases"*; PPV is read *"at the threshold where Recall = 0.90"*; 1000 iterations; **median**.

The organizers **re-derive the threshold from our own scores**. Therefore **any strictly monotone transform of the
final per-frame score leaves PPV@90R, AUROC and AUPRC exactly unchanged.** With all negatives kept and positives
resampled to `n_neg/100`:

> **PPV@90R = 0.9 / (0.9 + 100 · FPR@90R)**  ⇒  **FPR@90R = 0.009·(1/PPV − 1)**

| run | PPV@90R | ⇒ FPR@90R |
|---|---|---|
| exps/2 (ours) | 0.0177 | **0.50** — half of all new-center negatives outrank our bottom-decile positive |
| RARE25 winner (IMSY) | 0.035 | **0.25** |

**The whole task is: halve FPR@90R.** This retires a queue item and reframes §4:

- ❌ **RETRACTED — queue #2 "affine→1% recalibration" is an exact no-op.** Platt, temperature, sigmoid recal,
  whole-set quantile-matching, "overshoot recall at inference" — all monotone, all provably zero. This also explains
  §5's puzzle ("single-center de-floor: raw == de-floor exactly") — de-floor is monotone *within* a center.
- ❌ **RETRACTED — §4's "score-shift, not ranking" diagnosis is a category error.** A shift of our score
  distribution cannot move a threshold that adapts to our scores. The problem *is* ranking — specifically the deep
  tail: too many new-center negatives beat our weakest positives.
- ✅ Calibration is only real **pre-fusion, per-member** (it re-weights members inside an average — which is
  non-monotone overall). That is what IMSY's "post-hoc calibration" must mean, and it is the version worth copying.

### 10.2 BUG (critical) — `--cg-head` attention has **never been trained**, in any run
`layerwise_param_groups()` built optimizer groups from `backbone.blocks`, `backbone.norm` and `head` only. `net.attn`
(the gated attention-MIL pooling, **196,736 params**) is a *sibling* of `head` and was in **no** param group, so AdamW
never updated it. Verified by executing the pre-fix code: gradients are produced, `optimizer=42.5426M` vs
`trainable=42.7393M`, and the attention weights are **bit-identical after an optimizer step**.

**Consequence: exps/3 and exps/4 shipped a randomly-initialised, frozen attention pooling.** Every "CG-AMIL
regressed → retire attention-MIL" verdict in §C, §1 and the queue measured *random projections*, not attention-MIL.
The frozen-backbone ablation that liked attention (LOCO 0.943 vs mean 0.929) used a *different* code path and stands.
**Fixed** (+ an assert that every trainable tensor is in the optimizer). The CG-AMIL verdict is now **un-tested**, not
negative — but it is a ranking-side lever, so it is not top priority given §10.1.

### 10.3 CONFOUND (critical) — exp6 as written was **not** "exps/2 + one variable"
`pretrain_concept.py` had **no `--img` flag** and inherited `IMG` from `featurize.py`, which was flipped **336→448**
in commit `f4271e2`. exps/2 predates that commit (its cfg has no `img`/`backbone` key at all), so **exps/2's concept
encoder is @336**. Any Stage-1 built today runs **@448**, and the Drive cache key was `concept_encoder_{BACKBONE}.pt`
— **no resolution in the name** — so exp6 would have loaded a 448-trained encoder under a @336 ship: a second,
uncontrolled variable, and it is precisely the variable §8 identified as the killer.

**Fixed:** `--img` added to `pretrain_concept.py` (stamped into cfg), notebook pins one `FT_IMG = 336` used by both
stages, the Drive cache key is now `concept_encoder_{BACKBONE}_{FT_IMG}.pt`, legacy unversioned encoders are refused,
and `finetune.py` warns loudly on any Stage-1/Stage-2 resolution mismatch.

### 10.4 BUG — the §9 "prevalence fix" was itself a regression
`loco_probe.py` `ppv1()` was changed to `.01/(.01+.99f)`, which **drops the recall factor** from the numerator
(overstates PPV ~10%). The original `.009/(.009+.99f)` was right. Now corrected to the organizers' exact
definition, `R/(R + 100·FPR)`. §9 BUG 2 is **retracted**.

### 10.5 THE UNLABELED POOL — the real reason experiments feel weak
Measured from `phase3/cache/unl_manifest.npz`:

| | |
|---|---|
| actual pool | **144,887 frames** — the "288k manifest"/"300k pool" in §9 and the notebook are **double-counted**; `--semi-n 300000` is a non-binding cap |
| buckets | CONFIDENT_NEGATIVE 107,476 · **HARD_NEG_CANDIDATE 31,012** · ABSTAIN 6,399 |
| fields available | `name, dir, img_path, suspicion, decision, frame_trust` |
| fields Stage-2 reads | **`img_path` + `suspicion` only** (2 of 6) |
| coverage/epoch (ship flags) | 26 labeled batches × 10 semi-steps × 256 = **66,560 views** ≈ 4.6 passes over the run |

Coverage is **not** the bottleneck — signal *quality* is:

1. **The consistency loss self-extinguishes.** `((ps-pt)**2).mean()` in *probability* space: once the PU term drives
   65% of the pool to saturated-negative logits, this term is ~1e-4 and the 145k frames stop contributing gradient
   while still consuming ~90% of the compute. Only ~312 optimizer steps exist in the whole run (26 × 12).
2. **The 31,012 HARD_NEG_CANDIDATE frames — the exact FP-tail population that sets FPR@90R — receive no
   discriminative signal at all.** The only target is `suspicion < 0.15`, which excludes them by construction.
3. **The model-in-the-loop hard-FP miner is coded and has never been used** — no ship command passes `--neg-list`.
4. **The EMA teacher is built, updated every step, then discarded**; the student is shipped.
5. `suspicion` defaults to 0.0 when the VLM refused, so ABSTAIN frames were hard-taught as confident negatives —
   exactly where unlabeled positives hide.
6. Center is *not* in the manifest, but **`dir` is** — so the pool *can* be pseudo-center-labeled, which would let
   LOCO keep semi ON honestly instead of dropping it (`--loco-no-semi` is blind to exp6's own mechanism).

**Shipped fixes (all default-off except the guards):** `--semi-mode fixmatch` (confidence-masked pseudo-label CE),
`--semi-use-decision` (gate the PU target on the VLM `decision` field), `--ship-ema`, `--pauc-q`, a per-epoch `semi=`
loss print, and hard failures on a missing `--semi-manifest`/`--neg-list` (these used to disable the pool silently).

### 10.6 TAIL OBJECTIVE — we optimise PPV@80R, not PPV@90R
`soft_pauc90` defends `quantile(pos, q=0.2)` — the **20th**-percentile positive — while the metric thresholds at the
**10th**. There is no trained margin where the score is actually read. `--pauc-q` added; run at `0.0625–0.10` with
`--pos-per-batch 16` (needed to estimate the deeper quantile).

### 10.7 CONTAINER
- ⚠ **`RARE25-Submission/resources/ship_seed*.pt` are byte-identical to exps/5** (md5 `f6b3707…`) — the **worst**
  run (AUROC 0.797). A `do_build.sh` today ships the regressed model.
- ✅ Resolution parity is sound: `viscera_model.py` reads `img` from each checkpoint's cfg (`DEFAULT_IMG=448` is only
  a fallback), so a @336 ship serves at 336. `ENSEMBLE='prob'` and `SCORE_ALIGN_Q=None` are the correct settings —
  `ENSEMBLE='rank'` would compute ranks *within* a ~16-frame stack and destroy the pooled ranking. Keep both.
- The container is invoked **per stack** (the test fixture is a 16-frame stacked TIFF), so per-stack transductive
  tricks cannot cancel a systematic center shift — they only inject cross-stack ranking noise. Consistent with §7.

### 10.8 DOC CORRECTION — GastroNet did *not* underperform in RARE25
`RARE26_STUDY_PLAN.md:240` claims "in RARE25 the in-domain GastroNet pretrain underperformed ImageNet". The official
RARE25 results page says the **winner (IMSY) used a GastroNet-pretrained ResNet-50** plus a LoRA-finetuned DINOv3
ViT-Large, "extensive ensembling, and post-hoc calibration", and that top solutions "relied on pretraining and
ensembling rather than single-model approaches". The decision to demote continued-SSL/DAPT to a "research bet" rests
on a false premise and should be revisited.

### 10.9 REVISED QUEUE (ranked by EV against `FPR@90R = 0.50 → 0.25`, with 14 open-dev days)
| # | item | why it can move a **rank** metric | cost |
|---|---|---|---|
| 0 | **Relicense to MIT; stage exps/2 (not exps/5) in `resources/`** | eligibility + not shipping the worst model | minutes |
| 1 | **exp6 @336 + aug-domain, now un-confounded** (Stage-1 rebuilt @336) | attacks the color axis that inflates new-center FPs; the run as previously configured was invalid | 1 ship |
| 2 | **Hard-FP mining → `--neg-list`** (`mine_hardneg --score-with` exists, never used) | directly deletes FP mass above τ₉₀ₐ; morphology-driven ⇒ transfers | tiny |
| 3 | **`--pauc-q 0.0625 --pos-per-batch 16`** | trains the margin where the metric is actually read | tiny |
| 4 | **`--semi-mode fixmatch --semi-use-decision`** | makes 145k frames contribute non-vanishing gradient | tiny |
| 5 | **Per-member normalisation *before* prob-fusion** (the only legitimate reading of IMSY's calibration) | non-monotone ⇒ real; rank-preserving no-ops are not | small |
| 6 | **GastroNet ResNet-50 member** (public: `huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights`) | the winner's other family; genuine decorrelation, unlike our ConvNeXt | medium |
| 7 | **Train the final closed-phase model on train+val** (+31–48 positives, ≈ +25%) | positives are the binding constraint; val is ordinary challenge training data | small |
| 8 | Pseudo-center-label the pool via `dir` → honest LOCO **with semi ON** | today no semi lever can be gated at all | small |
| ~~x~~ | ~~affine→1% recalibration~~ | **provably zero** (§10.1) | — |
| ~~x~~ | ~~"CG-AMIL is retired"~~ | **verdict void** (§10.2) — never actually trained | — |

**Bottom line:** the pipeline is not weak because the ideas are wrong; it is weak because three of the levers were
never actually running (attention untrained, hard-FP miner unused, semi loss self-extinguishing), one experiment was
silently confounded (Stage-1 @448), and the single lever ranked #2 in the queue is mathematically incapable of
changing the score. Fix those and the remaining budget buys real experiments.

---

## 11. Follow-up audit — Stage-1 contaminates every LOCO gate; the local metric is not the challenge metric

Two claims left unverified in §10 (their verifier tier died on a usage limit). Both confirmed by direct inspection.

### 11.1 LEAK — `--loco-no-semi` is **not** leak-free when `--init concept_encoder.pt` is used
`concept_targets.npz` (170,200 rows) contains, alongside 167,724 unlabeled frames, **all 2,476 LABELED train
frames** with real `center` tags (center_1 1823 / center_2 653) — verified:

```
label values : {-1: 167724, 0: 2349, 1: 127}
center values: {'': 167724, 'center_1': 1823, 'center_2': 653}
```

Stage-1 never reads the binary `label` (it consumes only `paths/value/supervise`), so this is **not** a label leak.
But it distils concepts that are **0.87–0.91 AUROC proxies for the neo label** (`mucosal_irregularity` 0.905,
`demarcation` 0.870, `nodularity` 0.869 — our own §3.5 table) **on the exact images the LOCO leg then evaluates
on**, for 30 epochs. So every concept-initialised LOCO number is optimistic — including the `--loco-no-semi`
compass and the exp6 aug-gate in the notebook, which we were about to trust to decide the submission.

Structurally identical to §9 BUG 1: **the ship is unaffected** (the hidden test was never in Stage-1, so all LB
scores stand) — only the local compass lies. **Fixed:** `pretrain_concept.py --holdout {center_1|center_2|labeled}`
drops labeled frames by their `center` field (the unlabeled ones have `center == ""`, so the filter is exact):

| `--holdout` | frames kept | dropped |
|---|---|---|
| `none` (**ship**) | 170,200 | 0 |
| `center_1` | 168,377 | 1,823 |
| `center_2` | 169,547 | 653 |
| `labeled` | 167,724 | 2,476 (1.5%) |

**Recommended gating protocol:** build ONE encoder with `--holdout labeled` (costs 1.5% of the corpus, honest for
*both* legs, one extra Stage-1 run instead of two) and use it for every gate. Ship with `--holdout none`.

### 11.2 The local harness does not implement the organizers' estimator
`evaluate.py:_resample_idx` fixes `npos = len(pos)` and synthesises `99·npos` negatives **by resampling the negative
pool with replacement**. The organizers do the opposite: **all negatives retained, never resampled**; positives drawn
with replacement to `n_neg/100`. Two consequences: our PPV@90R is not on the same scale as the LB, and the extra
negative-resampling variance **inflates the apparent noise floor** — part of why §6 read PPV@90R as
"noise-dominated". Added `evaluate.bootstrap_challenge()` (exact scheme) as a **separate** function so every
historical `bootstrap()` number stays comparable; `report_full` now also returns `ppv90_gc` and `fpr90`.

A useful side-effect: on our 619-frame val the exact scheme draws only **6 positives per iteration** (588/100), which
is why local PPV@90R is so jumpy. On the real test (RARE25 scale ≈ 23k NDBE) it draws ~230 — **the leaderboard metric
is far better conditioned than our local proxy.** Read AUROC/AUPRC locally; do not over-read local PPV@90R swings.

### 11.3 FULL-POWER final ship (new notebook cell, after the exp6 cell)
For the one closed-phase container, every source of signal is legitimate — the hidden test is in none of it:

| source | train-only ship (exps/2 / exp6) | **full-power ship** |
|---|---|---|
| labeled | 2,476 / **127 pos** | **3,095 / 158 pos** (+ val's 619 source frames = **+24% positives**) |
| optimizer steps / epoch | 26 | ~74 (more negatives ⇒ more batches) |
| semi pool | 144,887, prob-MSE (self-extinguishing) | 144,887, **fixmatch + decision-gated PU** |
| hard negatives | none (`--neg-list` never used) | **3,000 model-mined FPs** — the exact tail the metric reads |
| tail loss | `q=0.2` (⇒ PPV@80R) | **`--pauc-q 0.0625 --pos-per-batch 16 --ohem-k 16`** |
| weights | best-epoch ×3 seeds | best-epoch + **SWAD** + **EMA teacher** ×3 seeds |

`out/val/labels` is already de-augmented (619 source frames, 31 pos), so folding it in does **not** reintroduce the
8× augmentation leak; the cell also hash-dedups by path.

⚠ **Sequencing is mandatory:** this trains on val, so val can no longer measure anything. Freeze the recipe on the
train-only gates (with an `--holdout labeled` concept encoder, §11.1) **first**; run the full-power cell last.

---

## 12. Review-of-the-review (2026-08-18) — one self-correction, one worse bug found underneath

Adversarial re-check of §10–§11. Two findings survived unchanged, one needed correcting, and the correction
uncovered a **silent training corruption on Colab** that is probably the single biggest reason experiments felt weak.

### 12.1 SELF-CORRECTION — `bootstrap_challenge()` is not usable on our val set
My own §11.2 addition quantizes badly at small n. With `n_draw` positives, "first index with recall ≥ 0.9" needs
`tp ≥ ceil(0.9·n_draw)`, so the **effective** recall is:

| negatives | positive draws | effective recall |
|---|---|---|
| 588 (our val) | 6 | **1.000** — this is PPV@100R, not PPV@90R |
| 2,937 (train+val) | 29 | 0.931 |
| 23,000 (real test) | 230 | **0.900** — exact |

So the exact estimator is faithful only on large sets. It now prints a loud warning when effective recall deviates
>0.02 from target. **Do not rank recipes with it locally — use AUROC/AUPRC.** (§11.2's headline still stands: the
LB metric is better conditioned than our local proxy.)

### 12.2 The Stage-1 contamination of §11.1 is REAL — and worse than described
Re-verified at frame level: the 2,476 labeled rows in `concept_targets.npz` are **name-identical to 2,476/2,476 of
the LOCO eval frames**, and carry **25.5 supervised concepts each** (not masked). So §11.1 stands.

But the paths are stored as `dataset/train/<class>/<id>.png` — **not** `out/train/images/*.jpg`.

### 12.3 NEW (critical) — on Colab, Stage-1 trains the labeled frames as BLACK IMAGES
`ConceptDS.__getitem__` wrapped the image read in `try/except Exception` and substituted
`Image.new("RGB", (IMG, IMG))` — a **black frame** — then trained on it with the real concept targets. Meanwhile:

- the notebook extracts **only `out/`**; it never creates `dataset/`;
- cell 6 **preferred the Drive-cached** `concept_targets.npz` over rebuilding.

⇒ If the Drive copy was built on the laptop (paths under `dataset/`), then **every Colab Stage-1 run trained all
2,476 labeled frames — including all 127 positives — as identical black images mapped to real clinical concept
targets, for 30 epochs**, with no error and no log line. That is simultaneously (a) pure label-correlated noise
injected into the encoder and (b) total waste of the highest-value supervision in the corpus.

It also **inverts §11.1 for Colab runs**: frames that were never actually loaded cannot leak. Which bug you had
depends on where the matrix was built — and nothing recorded that. Both are now impossible:

| fix | file |
|---|---|
| blank-image fallback → hard `RuntimeError` | `pretrain_concept.py:ConceptDS` |
| preflight: all labeled rows + ~2k sampled rows must resolve, else `FileNotFoundError` with the rebuild command | `pretrain_concept.py:main` |
| cell 6 validates paths and **rebuilds** instead of reusing an unusable cache | notebook |
| `--holdout labeled` gating encoder wired into **both** gate cells | notebook cells 8/10/13 |

Verified: preflight passes locally (4,450 paths checked, 2,476 labeled in full), and on a matrix with Colab-style
unresolvable paths it fails with the exact diagnosis instead of training on blanks.

### 12.4 Unchanged after re-check
The AttnPool optimizer bug (§10.2, proven by execution), the rank-only metric (§10.1, algebraic), the
Stage-1/Stage-2 resolution split (§10.3), the `--img`-under-spawn bug, and the pool-signal findings (§10.5) all
survive. Gate cell 10 additionally never passed `--loco-no-semi` at all and ran `@448`; both fixed.

---

## 13. exp6 RESULTS + review (2026-08-19)

### 13.1 Stage-1 is HEALTHY — and the new guards proved themselves in production
`preflight OK` on **291,187/291,187** frames: cell 6 detected the Drive-cached matrix as unusable and rebuilt it,
so the blank-image corruption of §12.3 **did not happen**. `concept encoder img=336 | ship img=336` — the
resolution split of §10.3 is closed. Training was well-behaved:

| | ep1 | ep30 |
|---|---|---|
| `main` (concept distillation) | 0.4939 | **0.3815** — monotone, still learning |
| `center_grl` | 0.6516 | **0.6966** |

`center_grl` ending at 0.6966 ≈ **ln 2 = 0.693 = chance**: the center head can no longer predict
`black_border`/`overlay_graphics` above random. The GRL did what it was designed to do.

### 13.2 The notebook's val numbers are INVALID — discard them
Cell 12 (full-power, trained on train+**val**) **overwrote `ship_seed*.pt`**, and cell 16 then scored **val** with
those weights → `PPV=AUROC=AUPRC=1.000` for POOLED/center_1/center_2, and cell 20's "ViT-anchor PPV@90R=1.000".
That is memorization, exactly what the cell-12 banner warned about. Same for the D2F+ ensemble comparison.

### 13.3 exp6, re-scored properly (weights that never saw val)
`exp6/ship_seed{0,1,2}.pt` are the **train-only** models (`train_csv=train_colab.csv`, `semi_mode=mse`,
`swad=False`). Re-scored on the 619 held-out val frames, 3-seed + 5-view TTA (= the container):

| split | n | pos | PPV@90R | 95% CI | AUROC | AUPRC | FPR@90R |
|---|---:|---:|---:|---|---:|---:|---:|
| POOLED | 619 | 31 | 0.700 | [0.418, 1.000] | **0.971** | 0.955 | 0.0034 |
| center_1 | 456 | 12 | 1.000 | [0.688, 1.000] | 1.000 | 0.994 | 0.0000 |
| center_2 | 163 | 19 | 0.333 | [0.016, 1.000] | **0.967** | 0.944 | 0.0208 |

Consistent with the prior same-center val regime (§4 reported AUROC 0.976 on center_2), so **no anomaly — but no
evidence either**: same-center val sits near ceiling for every recipe here (LB AUROC has been 0.80–0.85 while val
reads 0.97), so it cannot rank exp6 against exps/2. The LB is the measurement.

### 13.4 exp6 is a TWO-variable change, not one
Diffing the shipped cfgs against exps/2 (LB 0.0177): `aug: mild→domain` **and** `semi_use_decision: →True`. The
second is a bug fix (ABSTAIN frames with a defaulted `suspicion=0.0` were being hard-taught as negatives) but it
is still a second variable. Everything else matches exps/2's effective behaviour (`cg_head=False`,
`mixstyle=False`, `pauc_q=0.2`, `semi_mode=mse`, `img=336`). **Both gates were skipped** (`RUN_GATE=False`,
`RUN_AUG_GATE=False`), so exp6 ships on rationale — its LB score *is* the experiment.

### 13.5 Corrections to §10 (this run contradicts two of my own claims)
1. **"The semi loss self-extinguishes / is negligible" — WRONG for `mse`.** Printed `semi` = `semi_steps ×
   semi_w × raw`, so at ep12 raw = 0.1200/(10×0.5) = **0.024 vs a supervised loss of 0.0046** — the pool term is
   ~5× the labeled term, not vanishing.
2. **"fixmatch fixes the vanishing gradient" — NOT SUPPORTED.** In the full-power run fixmatch ended at
   `semi=0.0017` (raw 0.00034) vs mse's 0.1200 (raw 0.024) — **~70× smaller**. Once the teacher is confidently
   negative on a 99%-negative pool, the masked CE collapses. Treat `--semi-mode fixmatch` as **unvalidated**;
   the full-power model uses it, which is a risk, not a known improvement.
3. Confirmed instead: the real pathology is **saturation** — supervised loss reaches 0.0126 by ep7 and 0.0046 by
   ep12 on 127 positives in ~26 steps/epoch. More positives and more hard negatives (the full-power levers)
   address this; changing the consistency form does not.

### 13.6 Two local-scoring bugs found while re-scoring (container was never affected)
* `phase3/infer.py` defaulted a checkpoint with no `img` key to **448**, while the container defaults to **336**.
  exps/1 and exps/2 have no `img` key and were trained @336 — so any local re-score of the *winning* run was
  served at the wrong resolution. Aligned to 336.
* Scores were written at `%.6f`; these compress near 0 and produced **81 ties in 619 rows**, which depresses a
  rank metric. The container writes full-precision floats, so local val was *pessimistic* vs what ships. Now
  `repr(float(s))` → 618/619 unique.

### 13.7 CNN member: still a no
LOCO AUROC **0.861 (c2) / 0.952 (c1)** — worse than the ConvNeXt-Tiny of §H (0.932/0.976), and §7 already killed
the ensemble on the honest bench. Do not ship it.

---

## 14. exp6 LEADERBOARD RESULT (2026-08-19) — the first improvement in four submissions

| run | img | aug | PPV@90R | AUROC | AUPRC | implied FPR@90R |
|---|---|---|---:|---:|---:|---:|
| exp1 | 336 | mild | 0.0181 | 0.845 | 0.356 | 0.488 |
| exps/2 | 336 | mild+semi | 0.0177 | 0.854 | 0.401 | 0.499 |
| exps/3 | 448 | dinov3 | 0.0117 | 0.756 | 0.300 | 0.760 |
| exps/4 | 448 | bundle | 0.0155 | 0.829 | 0.355 | 0.572 |
| exps/5 | 448 | simple | 0.0128 | 0.797 | 0.351 | 0.694 |
| **exp6** | **336** | **domain** | **0.0195** | **0.8602** | 0.3900 | **0.453** |
| open-dev top-1 | | | 0.0271 | | | 0.323 |
| RARE25 winner (IMSY) | | | 0.0350 | 0.920 | **0.822** | 0.248 |

exp6 also reports **Validation RARE25: PPV 0.0163 / AUROC 0.9086 / AUPRC 0.6154**.

**Read:** exp6 is the best submission on PPV **and** AUROC **and** it beats exps/2 on the implied FPR. `--aug domain`
at @336 is the first lever to move the board after exps/3-5 all regressed. Honest caveat: ΔAUROC vs exps/2 is
+0.006 — **below the ±0.03 noise floor** — and the PPV CI is [0.0092, 0.1061]. Statistically this is a tie; the
evidence is that it is best on *all three* metrics at once and did not regress. Treat it as "aug-domain is safe and
probably mildly positive", not as a proven +0.002.

### 14.1 THE DIAGNOSTIC — the gap is AUPRC, not AUROC
| | exp6 | winner | ratio |
|---|---:|---:|---|
| AUROC | 0.860 | 0.920 | 1.07× |
| **AUPRC** | **0.390** | **0.822** | **2.11×** |

AUROC is within 7% of the winner while AUPRC is **less than half**. AUPRC is dominated by how well positives are
ranked in the high-precision region — exactly where PPV@90R is read. So the deficit is **not** general ranking and
**not** calibration (which is a proven no-op, §10.1); it is that our hardest positives sit too low relative to the
negative bulk. To hit 0.0271 we must cut FPs above the threshold by **29%**; to hit 0.035, by **45%**.

### 14.2 What the winner did that we have not
IMSY: **40 models** (20 GastroNet-ResNet50 + 20 LoRA DINOv3 ViT-L) from **5-fold CV**, at only 224px, with
per-member affine recalibration before fusion. We ship **3 seeds of one architecture on one split**. Ensemble
scale/diversity is the single largest structural difference, and it is precisely the lever that raises AUPRC.

### 14.3 Ranked plan for the remaining days
| # | lever | why it should move AUPRC/FPR | cost |
|---|---|---|---|
| 1 | **Submit the already-trained full-power model** (`RARE_LG/final/`) — train+val (**158 pos, +24%**) + 3k mined hard-FP negatives + deep tail loss + SWAD + EMA | more positives = better tail ranking; mined FPs delete mass exactly above τ₉₀ᴿ; also ~3× more optimizer steps/epoch (fixes the ep7 saturation) | already trained — build + upload |
| 2 | **Scale the ensemble 3 → 15-20** (5-fold CV × 3 seeds, all @336 aug-domain) | the winner's core method and the most reliable AUPRC lever; variance reduction on the positive tail | ~15-20 GPU-hours on the 96GB Blackwell |
| 3 | **GastroNet ResNet-50 family** (public: `huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights`) | the winner's other half; genuine architectural decorrelation — unlike our ImageNet-ConvNeXt member, this one is in-domain pretrained | medium |
| 4 | **`--pauc-q 0.0625 --pos-per-batch 16 --ohem-k 16`** | we currently defend the 20th-pct positive = we optimise PPV@80R; the metric reads the 10th | tiny (untested) |
| 5 | **Per-member ECDF normalisation before prob-fusion** | the only legitimate reading of IMSY's "affine recalibration" — non-monotone overall, hence real | small |

**Do NOT spend submissions on:** any @448 variant (5-experiment signal), global score recalibration (§10.1 proves
it is an exact no-op), or the ImageNet-ConvNeXt CNN member (§7, and its LOCO got worse in this run).

---

## 15. THE POSITIVE-SIDE LEVER (2026-08-19) — the one thing the pipeline has never done

### 15.1 The argument
exp6: AUROC 0.860 (winner 0.920, **1.07×**) but AUPRC 0.390 (winner 0.822, **2.11×**). AUPRC is dominated by how
the *hard positives* rank. Every use of the 288,711-frame pool to date is on the **negative / regularisation**
side — Mean-Teacher consistency, one-sided-PU negatives, hard-FP mining, colour-consistency. **Nothing has ever
added positive supervision.** We train on 127 positives.

Measured pool composition (sampled from `out/`, n=4,000 → extrapolated):

| bucket | frames |
|---|---:|
| CONFIDENT_NEGATIVE | 214,584 |
| **HARD_NEG_CANDIDATE** | **61,567** |
| **ABSTAIN** | **12,558** |
| VLM suspicion > 0.9 | 16,961 |

At ~1% true prevalence the pool holds on the order of **~2,900 unlabeled true positives ≈ 23× the labeled count**,
concentrated in exactly the two buckets the pipeline quarantines.

### 15.2 Why naive promotion fails, and the triple gate
A pseudo-positive that is really NDBE teaches "NDBE looks neoplastic" and **raises** FPR@90R — the exact opposite
of the goal. Raw VLM suspicion is far too weak alone (its own PPV@90R baseline is ~0.04). `mine_pseudopos.py`
promotes a frame only when three weakly-correlated signals agree:

| gate | signal |
|---|---|
| **A concept** | trust-weighted DECISIVE-hallmark score ≥ the labeled-positive percentile — the exact *inverse* of the PU guard `mine_hardneg.py` already uses to *exclude* likely positives |
| **B model** | detector probability ≥ threshold, scored by models from a **different fold** (else self-training amplifies its own bias) |
| **C bucket** | HARD_NEG_CANDIDATE / ABSTAIN only — never CONFIDENT_NEGATIVE |

Training support (`finetune.py --pos-list --pos-soft 0.85 --pos-cap`): they enter BCE and the pairwise-rank loss
as soft positives, are counted as positives by the balanced sampler, and are **excluded from the soft-pAUC
threshold quantile** (`y >= 0.999`) so an unverified label can never define the operating point. They are dropped
under `--loco-no-semi` like every other unlabeled pool.

### 15.3 CORRECTION — the hard-negative mine contradicted our own design rule
`mine_hardneg.py --pool` defaulted to `HARD_NEG_CANDIDATE,CONFIDENT_NEGATIVE`, and the full-power notebook cell
used that default. But ARCHITECTURE.md §5 established: *"Quarantine HARD_NEG_CANDIDATE (never y=0) — that bucket
is where the ~1% PU true-positives concentrate; pinning to 0 manufactures false negatives → craters recall (fatal
at 90R)."* Mining hard negatives from that bucket **violates the rule** and would have shipped in the full-power
run. Default corrected to **CONFIDENT_NEGATIVE only**; HARD_NEG_CANDIDATE is now the *input to the positive mine*
instead. Clean split: confident-negatives feed the negative mine, hard-negative-candidates feed the positive mine.

### 15.4 Also corrected: `dir` is NOT a domain label
§10.5 suggested center could be recovered from the manifest's `dir` field. **Wrong** — verified: each dir is a
contiguous, non-overlapping 10,000-ID block (density 1.00), i.e. an arbitrary shard of the export, not an
acquisition source. Pseudo-domains must be discovered by clustering embeddings, not read from metadata.

### 15.5 Honest risk register
| risk | severity | mitigation |
|---|---|---|
| PU contamination raises FPR@90R | **high** | soft target 0.85, `--pos-cap`, excluded from the pAUC quantile, triple gate |
| self-training amplifies existing bias | medium | score with a **different fold's** models |
| the three gates are correlated (all appearance-derived) | medium | keep `--topn` small (≤300 ≈ +2× positives), gate on LOCO |
| we cannot validate cheaply (LOCO needs a `--holdout labeled` encoder) | **high** | each LB submission also returns a *Validation RARE25* row — two measurements per submission |

**This is the highest-ceiling and highest-variance lever in the pipeline. The reliable core remains ensembling
(the winner used 40 models; we use 3).** Do not ship pseudo-positives ungated in the single closed-phase slot.

---

## 16. exp7 (MAX) — what changed vs the exp6 that scored 0.0195

Backbone / resolution / aug / unfreeze / WiSE-FT are **unchanged** (dinov2 @336, `--aug domain`, last-6
blocks, α=0.7). Everything that changed is about *how much data reaches the loss* and *how we measure*.

| | exp6 (LB **0.0195**) | **exp7 (MAX)** |
|---|---|---|
| labeled csv | `train_colab.csv` | `trainval_colab.csv` |
| labeled frames | 2,476 (**127 pos**) | 2,476/fold of 3,095 (**158 pos**, +24%) |
| validation | none — `--holdout none` | **5-fold OOF on all 158 positives** |
| unlabeled as y=0 | **NONE** | **214,584** CONFIDENT_NEGATIVE |
| pseudo-positives | **NONE** | **600** @ soft 0.80 (A/B'd) |
| semi pool | all 288k incl. conf-neg (redundant) | 74,125 ambiguous only |
| semi bs × steps | 256 × 10 | 96 × 1 |
| **steps/epoch** | **26** | **2,582** (99×) |
| epochs | 12 | 3 |
| **total optimizer steps** | **312** | **7,746** (25×) |
| pos_per_batch | 8 | 12 |
| pauc_q (tail) | 0.2 → optimises PPV@**80**R | **0.0625** → the real 90R point |
| ohem_k | 0 | 12 |
| ensemble | 3 seeds, 1 split | 5 folds (× seeds later) |
| img / aug / backbone | 336 / domain / dinov2 | **same** |
| unfreeze / wise_ft | 6 / 0.7 | **same** |

**The three that should matter most, in order:**
1. **216,934 supervised negatives vs 2,350.** `FPR@90R` *is* the fraction of negatives above the
   threshold; exp6 learned "normal mucosa" from 2,350 examples and memorised by ep7 (loss 0.0046).
2. **+31 real positives** (val folded in) — positives are the binding constraint on AUPRC, where we
   trail the winner 0.390 vs 0.822.
3. **5-fold OOF** — the first uncontaminated measurement in the project. Same-centre val is at ceiling
   (0.97 vs 0.86 on the board) and LOCO is contaminated by the concept encoder.

Build script renamed `build_exp6.sh` → **`build_submission.sh [SRC] [EXP]`** (defaults to `exp7`), now
handling any ensemble size and excluding `*_ema.pt` from the member glob.

---

## 17. exp8 — 100% of the corpus, no semi, no k-fold

Time pressure removed the 5-fold sweep, so exp8 is a single 3-seed ship. Every one of the 288,711
unlabeled frames is still in the loss, each at the reliability the VLM actually provides for it:

| frames | role | source of the label |
|---:|---|---|
| 214,584 | **hard y = 0** | VLM `decision == CONFIDENT_NEGATIVE` |
| **73,525** | **35-concept distillation** (`--concept-aux`) | VLM's full concept vector |
| 600 | soft y = 0.80 | 4-gate mine (concept ∩ bucket ∩ suspicion≥0.95 ∩ model≥0.90) |
| 3,095 | y ∈ {0,1} | ground truth (158 positives) |
| **288,709** | **100%** | |

### 17.1 Concept distillation replaces the semi loss — measured, not assumed
The ambiguous frames cannot take a binary label (the VLM abstained; §5 forbids pinning them to 0).
But the VLM emitted **all 35 concepts for every one of them**, and that is a dense supervised signal
we were discarding.

| | loss value | cost per step |
|---|---:|---|
| semi consistency (exp7 fold-0, measured) | **0.003** | 2 forwards (student + EMA teacher) |
| concept distillation (measured) | **0.217** | 1 forward |

**~70× the gradient at half the compute.** It also fixes a defect flagged early and never addressed:
`finetune.py` has **no concept term at all**, so 9 epochs of binary fine-tuning drift the encoder off
the Stage-1 clinical axes with only WiSE-FT pulling back. `demarcation`/`nodularity` are
center-invariant by construction, so holding the encoder to them during Stage-2 is a direct brake on
drifting into the center-specific texture that inflates cross-centre FPR@90R.

The auxiliary head is initialised from the Stage-1 `main_head` and distils only the **MAIN**
(diagnostic) concepts — never the center-cue ones, which Stage-1 deliberately pushes out via GRL.

### 17.2 Verified before shipping
* concept head **is in the optimizer** and demonstrably updates (42.5733M covered == 42.5733M
  trainable) — the `AttnPool` bug class does not repeat.
* the container **tolerates the 4 new `concept_head.*` keys**: `viscera_model` loads with
  `strict=False` and asserts only on missing keys, so a 183-tensor exp8 checkpoint loads and scores
  (verified on a synthetic exp8 checkpoint through `VisceraEnsemble`).
* ship mode (`--holdout none`) runs with concept-aux + pseudo-positives + SWAD + EMA together.

### 17.3 Budget and the honest caveat
2,589 steps/epoch × 9 epochs ≈ **3.2 h/seed**, **9.7 GPU-hours** for 3 seeds.

**exp8 has no internal validation.** val is now training data and there are no folds, so the
leaderboard is the only measurement. Every change vs exp6 is therefore mechanism-driven rather than
tuned: 216k negatives because FPR@90R *is* the negative-side rate; +31 real positives because
positives bind AUPRC; concept-aux because Stage-2 was forgetting Stage-1. If time later allows, a
single fold (~3 h) buys an OOF sanity check before the closed-phase submission.

---

## 18. exp7 5-fold RESULT — pseudo-positives FAIL, and two of my own reads were wrong

The 5-fold OOF sweep ran. It is the first uncontaminated measurement in the project and it paid for
itself immediately by killing a lever I had recommended.

### 18.1 The A/B: pseudo-positives lose on every metric
Pooled out-of-fold, n=3,095, 158 positives, last-epoch scores (no selection):

| arm | AUROC | AUPRC | FPR@90R |
|---|---:|---:|---:|
| **nopos** | **0.9718** | **0.8955** | **0.0262** |
| pos (600 @ soft 0.80) | 0.9476 | 0.8864 | 0.0398 — **52% more FPs** |

Paired gate: **AUROC Δ = −0.0237, CI [−0.0509, −0.0012] → FAIL** (significant harm). AUPRC Δ +0.0006,
CI spans 0 → inconclusive. The mechanism is exactly the one the module docstring warned about: a
pseudo-positive that is really NDBE teaches "NDBE looks neoplastic" and raises FPR@90R.

**Why the 4-gate design did not protect us: the gates were not independent.** `suspicion≥0.95` left
2,381 candidates and the model gate removed **19 of them** (2,381 → 2,362, **0.8%**). The detector and
the VLM agree almost perfectly, so there was no cross-checking at all — NDBE look-alikes that both
signals liked went straight in at y=0.80. A gate that agrees with the signal it is meant to check is
not a gate.

### 18.2 ⚠ exp7_pos/ ON DRIVE IS THE LOSING ARM
MAX-C ran with `USE_PSEUDO = True`, so `exp7_pos/` holds `fold*_pos_s0.pt` — the arm the gate just
rejected. **Do not build a container from it.** Re-run MAX-C with `USE_PSEUDO = False` to stage the
`nopos` folds, which are already trained and sitting in `max_folds/`.

### 18.3 CORRECTION — "loss 0.09 means it is not memorising" was wrong
I read the pos-arm loss floor (0.0915) as evidence that 216k negatives had stopped the memorisation
that killed exp6. It was arithmetic, not health: with `pos_weight=7` and a soft target of 0.80 the
irreducible BCE minimum is

> σ\* = pw·y/(pw·y + 1−y) = 0.966 → loss = 0.870 per pseudo-positive
> ≈ 9.9 pseudo per 96-batch → **floor = 0.0897**, vs **observed 0.0915**

The nopos arm shows what was really happening: **0.0281 → 0.0058 → 0.0013 by ep3** — full
memorisation, exactly like exp6's 0.0046. **216k negatives did not prevent it**, because 12 pos/batch
× 2,599 batches replays each of the 127 positives **246× per epoch**.

### 18.4 CORRECTION — the 9-epoch decision followed from that misread
exp8 was set to 9 epochs on the strength of "loss still falling at ep3". That trajectory was the
soft-label floor. With the pos arm removed the loss is at 0.0013 by ep3, so 9 epochs is far past the
cliff. **exp8 is now 3 epochs** — the setting these OOF numbers were actually measured at.

### 18.5 What the OOF numbers do and do not say
`nopos` reads FPR@90R 0.0262 → implied PPV 0.256, against exp6's actual board 0.0195. The OOF folds
are random splits of the **same two centres**, so this is a same-centre number and overstates the
board by ~13×. Its value is the **delta between arms**, which is what killed the pseudo-positive arm.
Absolute OOF is not a leaderboard prediction.

### 18.6 Unchanged
The D2F-4 harness re-ran and again found the CNN member worthless (ViT anchor PPV 0.664 alone; every
ensemble weight below 1.0 is worse). The resolution-mismatch guard fired correctly on those @448 runs
against the @336 concept encoder — working as designed.

---

## 19. exp7 deep-dive — the pseudo-positive harm is concentrated on the BOARD-LIKE centre

All 40 fold artifacts recovered (Drive split the download into `max_folds 2/3/4`). Independently
recomputed from the `.npz` files: the notebook's numbers reproduce exactly
(nopos/final AUROC 0.9718 / AUPRC 0.8955 / FPR@90R 0.0262).

### 19.1 The pooled A/B understated the damage
| centre | n | pos | prevalence | AUPRC nopos → pos | **FPR@90R nopos → pos** |
|---|---:|---:|---:|---|---|
| **center_1** | 2,279 | 61 | **2.7%** | 0.7790 → 0.7304 | **0.0703 → 0.4270  (6.1× worse)** |
| center_2 | 816 | 97 | 11.9% | 0.9569 → 0.9700 | 0.0223 → 0.0042 |

Paired gate per centre:

| | AUROC Δ (pos − nopos) | verdict |
|---|---|---|
| **center_1** | **−0.0702  CI [−0.1338, −0.0180]** | **FAIL** |
| center_2 | +0.0069  CI [−0.0089, +0.0269] | inconclusive |

**All of the harm lands on center_1** — the low-prevalence, harder centre, and the one whose regime
most resembles the hidden test (a new centre at ~1% prevalence). On the easy high-prevalence centre
the pseudo-positives are harmless. The pooled AUROC Δ of −0.0237 averaged those two apart; the
board-relevant number is **−0.0702**.

This also matches §6's asymmetry: training on the harder c1 transfers to c2, but not the reverse.
**center_1 is the honest proxy; center_2 flatters everything.**

### 19.2 The mechanism, measured
| arm | positives q10 | negatives q99 |
|---|---:|---:|
| nopos | **0.0093** | 0.2228 |
| pos | **0.0042** | 0.0449 |

`FPR@90R` is the fraction of negatives above the 10th-percentile positive. Training 600 pseudo-positives
at soft 0.80 **pushed the genuine weak positives down 2.2×** (0.0093 → 0.0042), dragging the threshold
into the negative mass. That is the PU-contamination mechanism the module docstring predicted, now
observed end-to-end: teaching the model that NDBE look-alikes deserve 0.80 compresses the real
positives it must rank above them.

### 19.3 Best available board proxy
`nopos` on **center_1**: AUPRC 0.779, FPR@90R 0.0703 → implied PPV **0.114**, against exp6's actual
board 0.0195. Still 5.8× optimistic (center_1 is a *seen* centre, the board is unseen), but it is the
most board-like number the project has produced and it is the one to track across designs.

### 19.4 Status of the artifacts
All five `fold*_nopos_s0.pt` recovered and verified. They are the **only fully-validated model set in
the project** — measured out-of-fold, per-centre, with a paired gate. exp8 is strictly more data per
model (100% of labels vs 80%) but adds `--concept-aux`, which is **unvalidated**. With submissions
scarce, the nopos ensemble is the evidence-backed artifact and concept-aux should be gated the same
way pseudo-positives were, on one fold, before it rides in a submission.

---

## 20. exp7 LEADERBOARD — null result, and the mechanism is a design error I made

| | PPV@90R | AUROC | AUPRC | Val-RARE25 AUPRC |
|---|---:|---:|---:|---:|
| exp6 | 0.0195 | 0.8602 | 0.3900 | 0.6154 |
| exp7 | **0.0195** | 0.8479 | 0.3790 | **0.5347** |

**Not a submission mix-up** — AUROC/AUPRC and both Val-RARE25 rows differ, so the models are genuinely
different. The identical PPV is a coincidence of the bootstrap median (CI [0.0094, 0.1139]). The real
signal is **Val-RARE25 AUPRC 0.6154 → 0.5347 (−13%)**, on a set the organizers hold out and that is in
neither model's training data. exp7 is mildly **worse**, not tied.

### 20.1 Why: I recommended volume over hardness, and it was backwards
Scored 600 frames from each pool bucket with the exp6 model (which never saw them), against the 90R
threshold set by the 31 held-out val positives (0.09707):

| bucket | median score | **% above the 90R threshold** |
|---|---:|---:|
| val positives | 0.99094 | 87.1% |
| val negatives | 0.00072 | 0.7% |
| **CONFIDENT_NEGATIVE** — the 216k exp7 **added** as y=0 | **0.00153** | **4.7%** |
| **HARD_NEG_CANDIDATE** — the 60k exp7 **excluded** | **0.92238** | **91.5%** |

**19.6× ratio.** exp7 added 216k frames the model already answers correctly — near-zero gradient once
fitted (training loss confirms it: 0.0281 → 0.0058 → 0.0013 by ep3) — while withholding the 60k that
*constitute* the FP tail `FPR@90R` is computed from. The metric is the fraction of negatives above the
threshold; we trained on the ones already below it.

### 20.2 Three treatments of HARD_NEG_CANDIDATE, all now measured
| treatment | result |
|---|---|
| y = 1 (pseudo-positives) | **FAIL** — center_1 FPR@90R 0.0703 → 0.4270 (§19) |
| excluded entirely | **NULL** — exp7, 91.5% of the FP tail never trained |
| y = 0 for all 60k | untested; ~1% contamination × 60k ≈ 600 manufactured false negatives against only 127 real positives (§5's objection, and it is a real one) |

### 20.3 The middle path was already in the repo and never used
`mine_hardneg.py --concept-rank` mines **concept-confounded** frames — high on surface confounders,
**low on decisive architectural hallmarks** — with a PU guard (`decisive < labeled-positive median`)
that drops likely hidden positives. It kept 137,702 of 167,724 unlabeled frames and ranks the
confounded band. Scored with exp6:

| source | % above 90R threshold |
|---|---:|
| CONFIDENT_NEGATIVE | 4.7% |
| **CTM (PU-guarded)** | **57.0%** |
| HARD_NEG_CANDIDATE raw | 91.5% |

Exactly the intended regime: hard enough to produce gradient at the boundary, guarded against the
true positives that made the pseudo-positive arm fail.

### 20.4 exp8 revision
Replace the "volume" negative set with a **mixture**: a moderate CONFIDENT_NEGATIVE sample for broad
normal coverage plus CTM-mined hard negatives at the boundary. Contamination budget matters — at ~4%
CTM contamination, 5,000 frames ≈ 200 manufactured false negatives against 158 real positives, so the
cap stays small and **this must be gated on one fold before it ships**. Every ungated lever in this
project has failed; every gated one has been caught.

---

## 21. exp9 — the design that follows from everything measured

### 21.1 The problem restated correctly
| | FPR@90R |
|---|---:|
| center_2 (OOF, 11.9% prevalence) | 0.0223 |
| center_1 (OOF, 2.7% prevalence) | 0.0703 |
| **hidden test (new centre)** | **0.4525** |

The boundary is *fine* on centres we have seen and collapses 6–20× on one we have not. **The problem
is not the decision boundary, it is that the boundary does not transfer.** Every single-model lever
aimed at sharpening the boundary on our own data has therefore come back null or negative.

### 21.2 exp9 vs exp7, and why each change
| | exp7 (null) | **exp9** | reason |
|---|---|---|---|
| negatives | 215,986 all-easy | **30k easy + 5k CTM** | 4.7% vs 57% above threshold — exp7 trained where the model was already right |
| steps/epoch | 2,599 | **445** | smaller *and* harder |
| positive repetition | 197×/epoch | **34×/epoch** | exp7 memorised by ep3 (loss 0.0013) |
| epochs | 3 | **8** | real LR annealing, SWAD window |
| ambiguous 73.5k frames | semi (loss 0.003) | **concept-aux (0.217)** | 70× the gradient at half the compute |
| ensemble | 5 | **15** (5 folds × 3 seeds) | the winner shipped 40; this is the one proven lever never tried at scale |
| checkpoint | best-epoch | **last-epoch** | best-epoch selection on 31 positives gave *worse* pooled OOF (0.0361 vs 0.0262) |
| pseudo-positives | 600 | **none** | gated and failed (§19) |

Budget: 445 steps/epoch × 8 epochs = 3,560 steps/model, **0.5 h/model, 7.4 GPU-hours for 15** — less
than exp7 cost, because the wasted easy negatives are gone.

### 21.3 What exp9 does NOT claim
CTM is **57% above threshold on our own centres**. That fixes exp7's "no gradient" defect; it does
**not** prove it transfers to a new centre — sharpening a boundary on seen data is exactly what has
failed repeatedly here. Hence `exp9-GATE`: one fold, CTM vs same-size all-easy, read **center_1**
(§19 showed a pooled number can hide a 6× regression that lands entirely on the board-like leg).

Every ungated lever in this project has failed — pseudo-positives, 216k easy negatives, CG-AMIL, the
CNN member. Every gated one was caught before it cost a submission. concept-aux is also still ungated
and rides in both arms, so the gate measures CTM only.

### 21.4 Left on the table, ranked
1. **GastroNet-5M ResNet-50 as a second family** (public weights). The winner's other half; our CNN
   member failed but it was *ImageNet* ConvNeXt, not in-domain. This is the largest untried
   structural lever.
2. **Per-member ECDF normalisation before prob-fusion** — the only non-monotone (hence real) reading
   of IMSY's "affine recalibration"; needs a reference set shipped in `resources/`.
3. **Pseudo-domain discovery by embedding clustering** → GroupDRO/Fishr across discovered domains.
   `dir` is not a domain label (§15.4), so domains must be discovered; this is the only lever that
   attacks transfer *directly* rather than through variance reduction.

### 21.5 Review pass — three defects in exp9, found before it ran
1. **The shipped model was not the measured model.** Under `--fold`, `a.out` is written only at the
   best-AUPRC epoch, so `_loco_final.npz` (last-epoch *scores*) described weights that were never
   saved. Since exp7 showed last-epoch pooled OOF beats best-epoch (FPR 0.0262 vs 0.0361 — selecting
   on ~31 held-out positives is selecting on noise), `finetune.py` now also writes
   `<out>_final.pt` and WiSE-FTs it with the same anchor. exp9-SHIP stages those.
2. **`exp9-SHIP` referenced `E9`, defined only in `exp9-GATE`** — running SHIP without GATE was a
   `NameError`. SHIP now defines it.
3. **Positive repetition was still 270×.** exp6 — still the best board score — replayed each positive
   ~**20×** across all of training; exp7 replayed **592×** and memorised (loss 0.0013 by ep3). exp9
   now uses 5 epochs × 8 pos/batch at 424 steps/epoch = **107×**, five times less than exp7 and as
   close to exp6's regime as a 35k negative sweep allows.

Final budget: 424 steps/epoch × 5 epochs = 2,120 steps/model, negatives swept 5×, **4.4 GPU-hours for
15 members**.

**The largest remaining ungated risk is positive repetition (107× vs exp6's 20×).** It cannot be
reduced further without either starving the negative sweep or dropping `--pos-per-batch` below the
point where the q=0.0625 pAUC quantile is estimable. Watch the training loss: **if it reaches ~0.005
the model is memorising again and epochs should drop to 3.**

### 21.6 Using a 24-hour budget — exp9 alone only needs 4.4h
| spend | hours | why |
|---|---:|---|
| **exp9-GATE, factorial (4 arms)** | **1.2** | two of exp9's three levers ride ungated in all 15 members |
| exp9-SHIP, 15 members | 4.4 | the winner's proven lever, never tried at scale here |
| **GastroNet ResNet-50 family** | ~7 | the winner's *other* half; largest untried structural lever |
| more ViT seeds (15 → 25) | 3 | reliable but diminishing (variance ∝ 1/√n) |
| slack | 8.4 | Colab drops, reruns, container build |

**Spend the first 1.2h on the gate, not on more members.** exp9 carries three levers and only CTM was
scheduled for measurement:

| lever | status |
|---|---|
| CTM hard negatives | 57% at the boundary vs 4.7% — measured on *our* centres, transfer unproven |
| **concept-aux** | caux 0.217 vs semi 0.003 — measured as *gradient*, **never A/B'd for accuracy** |
| **107× positive repetition** | exp6 (best board score) used 20×; exp7 used 592× and memorised |

The gate is one-factor-at-a-time from a common baseline on fold 0 (`full` / `noctm` / `nocaux` /
`lowrep`), reported pooled **and on center_1**, because §19 showed a pooled number can hide a 6×
regression that lands entirely on the board-like leg.

### 21.7 GastroNet-5M ResNet-50 — the enabler is now in place
`cnn_member.py --pretrained-path` loads an arbitrary state_dict instead of timm's ImageNet weights,
with a loud assert if fewer than 50 tensors match. This is what the RARE25 winner's other 20 models
were built on (`huggingface.co/tgwboers/GastroNet-5M_Pretrained_Weights`).

Why this is not a repeat of the failed CNN member: that member was **ImageNet**-pretrained ConvNeXt,
and §7 measured it tail-poisoning the ensemble at every weight. In-domain pretraining is the variable
that was never tested — and our own evidence says it dominates architecture (GastroNet-DINOv2 ViT-B
scored 0.854 while *generic* DINOv3 ViT-B scored 0.756). Gate it exactly like the ViT levers: admit
to the ensemble only if it improves the fused OOF on **center_1**, never on the pooled number alone.

### 21.8 GastroNet ResNet-50 dropped (user call) — MixStyle takes the slot instead

Removing that family leaves ~18h after the gate and ship. The best use is **not** more seeds of the
same recipe: ensembling reduces *variance*, and our problem is *bias* — a boundary fitted to two
centres. FPR@90R is 0.0223/0.0703 on centres we have seen and 0.4525 on one we have not.

**MixStyle is the only implemented lever that attacks that bias directly.** It mixes per-token
feature statistics across the batch, synthesising unseen-centre acquisition styles at the feature
level; param-free, train-only, identity at eval, so the shipped graph is unchanged.

It has **never been measured on its own**. Its single appearance was inside the exps/4 bundle:

| exps/4 | value |
|---|---|
| img | **448** (the resolution 4 experiments proved regressive) |
| cg_head | **True** — and we later found AttnPool was *never in the optimizer*, so that head was random |
| mixstyle | True |
| LB AUROC | 0.829 |

That bundle's failure is attributable to @448 and a random attention head. It says nothing about
MixStyle at @336. Added as a fifth gate arm (`mixsty`) for ~0.3 GPU-hours.

Revised 24h plan: **1.5h** gate (5 arms) · **4.4h** ship 15 members · **7.5h** extend the ensemble to
~40 with *recipe* diversity (varied `--unfreeze`, exp6-style and exp9-style members) · **10.6h** slack.
