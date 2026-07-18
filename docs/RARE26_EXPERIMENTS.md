# RARE2026 — Experiment Table (done vs not-done)

**Goal:** cross-center Barrett neoplasia, metric = PPV@90R @1% prevalence (hidden NEW-center test).

## Legend
- **Status:** ✅ done · 🔄 coded, not run on leaderboard · ⬜ NOT done yet
- **Finetune (FT) strategy:** `full-FT` = unfreeze last K blocks + head · `frozen-LP` = frozen backbone + linear probe · `head-only` · `LoRA` · `pretrain` = Stage-1 · `post-hoc` = inference-time · `train-time` = loss/aug during training
- **Measured on:** 🟢 **LB** = real hidden new-center leaderboard (TRUTH) · 🔵 **LOCO** = frozen-LP c1↔c2 (honest compass, predicted the LB) · 🟡 **val** = same-center (MIRAGE) · ⚪ **synth** = directional only
- ⚠ 31-positive val → any AUROC delta < ~0.03 is **below the noise floor** (unreliable)

---

## 1. Main results — the 4 real submissions
| exp | encoder | FT | header | concept | semi | aug | 🟢 PPV@90R | 🟢 AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| exp1 | GastroNet-DINOv2 ViT-B @336 | full-FT | cls⊕mean | ✗ | ✗ | mild | 0.0181 | 0.845 | 0.356 |
| **exps/2** 🏆 | GastroNet-DINOv2 ViT-B @336 | full-FT | cls⊕mean | ✓ | ✓ | mild | **0.0177** | **0.854** | 0.401 |
| exps/3 | DINOv3 ViT-B @448 | full-FT | CG-AMIL attn | ✓ | ✓ | strong-geom | 0.0117 | 0.756 | 0.300 |
| exps/4 | GastroNet-DINOv2 ViT-B @448 | full-FT | CG-AMIL attn | ✓ | ✓ | domain+mixstyle | 0.0155 | 0.829 | 0.355 |

**exp-4 validation (RARE25 val, separate held-out):** PPV@90R 0.0114 · AUROC 0.8626 · AUPRC 0.6109.

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
| per-stack/center score de-floor (`SCORE_ALIGN_Q`) | post-hoc | ✅ | 🟡 val | **no-op** on exp4 (0.748→0.748); floor gap 1e-4 | ✗ **mechanism-MISMATCHED** — floods are in the UPPER neg tail, de-floor only touches the low quantile |
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
| **D2F+ ViT ⊕ CNN equal rank-avg** | ✅ | 🟡 val | ens **drags** anchor: c2 PPV 0.396→**0.048**, AUROC 0.976→0.959 | ⚠ equal-weight HURTS the PPV@90R tail; val is a mirage → **needs LOCO gate + weighting** |
| decorrelated multi-backbone (dinov2⊕dinov3) | ✅ | 🔵 LOCO | dinov3 drags 0.929→0.906 | ✗ (use CNN member instead) |
| per-member affine recalibration (→1%) + per-stack de-floor | ⬜ | — | — | **the real operating-point lever — do this FIRST** |

---

## 3. NOT-DONE — the queue (ranked by EV to win, re-ranked after the D2F+ val run)
| # | item | track | why | blocker |
|---|---|---|---|---|
| 1 | **Run the honest LOCO harness** (cells D2F-4a/4b) | bench | the ONLY bench that sees the true new-center shift; A/B's de-floor AND the weighted ensemble on one center-blind proxy | Colab (2 ViT-LOCO finetunes) |
| 2 | **Re-designed operating-point lever** — per-center UPPER-tail norm / affine→1% (NOT low-q de-floor, §5) | post-hoc | de-floor is mechanism-mismatched (flood is upper neg tail); affine rescales to target FPR | needs #1 to validate |
| 3 | Diverse **ViT-L** member (public DINOv2/v3-L, frozen-LP) | A ensemble | 2nd decorrelated member if CNN alone isn't enough | Colab train |
| 4 | **logit-adjusted BCE** | loss | agaldran's robust workhorse for the operating point | small code |
| 5 | Generative (diffusion) hard positives | B novelty | break 127-pos wall (winner didn't) | heavy |
| ~~x~~ | ~~CG-AMIL / @448 / MixStyle / aug-domain bundle~~ | — | **retired** — 2× no LB gain (exps/3, exps/4) | — |
| ~~x~~ | ~~ConvNeXt-**Large** member~~ | — | **retired** — no gain over Tiny (0.909/0.965 < 0.932/0.976) | — |

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
