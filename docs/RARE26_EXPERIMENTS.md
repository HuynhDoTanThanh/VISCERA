# RARE2026 — Experiment Table (done vs not-done)

**Goal:** cross-center Barrett neoplasia, metric = PPV@90R @1% prevalence (hidden NEW-center test).

## Legend
- **Status:** ✅ done · 🔄 coded, not run on leaderboard · ⬜ NOT done yet
- **Finetune (FT) strategy:** `full-FT` = unfreeze last K blocks + head · `frozen-LP` = frozen backbone + linear probe · `head-only` · `LoRA` · `pretrain` = Stage-1 · `post-hoc` = inference-time · `train-time` = loss/aug during training
- **Measured on:** 🟢 **LB** = real hidden new-center leaderboard (TRUTH) · 🔵 **LOCO** = frozen-LP c1↔c2 (honest compass, predicted the LB) · 🟡 **val** = same-center (MIRAGE) · ⚪ **synth** = directional only
- ⚠ 31-positive val → any AUROC delta < ~0.03 is **below the noise floor** (unreliable)

---

## 1. Main results — the 3 real submissions
| exp | encoder | FT | header | concept | semi | aug | 🟢 PPV@90R | 🟢 AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| exp1 | GastroNet-DINOv2 ViT-B @336 | full-FT | cls⊕mean | ✗ | ✗ | mild | 0.0181 | 0.845 | 0.356 |
| **exps/2** 🏆 | GastroNet-DINOv2 ViT-B @336 | full-FT | cls⊕mean | ✓ | ✓ | mild | **0.0177** | **0.854** | 0.401 |
| exps/3 | DINOv3 ViT-B @448 | full-FT | CG-AMIL attn | ✓ | ✓ | strong-geom | 0.0117 | 0.756 | 0.300 |

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
| CG-AMIL attention-MIL | full-FT | ✅ | 🟢 LB | regressed (exps/3) | ✗ under full-FT |

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
| color/FDA aug (`--aug domain`) | train-time | 🔄 | — | untested on LB | **gateable — run the gate** |
| MixStyle (feature-stat mixing) | train-time | ✅ | ⚪ synth | 0.893→0.887 | ⚠ un-gateable, param-free rider |
| per-stack score de-floor (`SCORE_ALIGN_Q`) | post-hoc | ✅ | ⚪ synth | small on dinov2 | mechanism-proven, small |
| per-center robust-z norm | post-hoc | ✅ | 🟡 val | 0.471→0.517 | directional; **rank-norm HARMFUL** |
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
| decorrelated multi-backbone ensemble | 🔄 | 🔵 LOCO | dinov3 drags; needs a STRONG diverse member | pending diverse encoder |
| per-member affine recalibration (→1%) | ⬜ | — | — | NOT done (winner's trick) |

---

## 3. NOT-DONE — the queue (ranked by EV to win)
| # | item | track | why | blocker |
|---|---|---|---|---|
| 1 | **Diverse ViT-L member** (public DINOv2/v3-L, frozen-LP + full-FT) | A ensemble | decorrelation (winner used ViT-L) | none — testable now |
| 2 | **Self-trained concept-CNN** on 144k pool (full-FT) | A ensemble | GI-matched CNN diversity + uses VLM concepts | Colab train |
| 3 | **Per-member affine recalibration** + per-stack de-floor | A ensemble | winner's operating-point trick | none |
| 4 | **`--aug domain` LOCO gate** (color/FDA) | OOD | the one gateable OOD lever untested | run gate cell |
| 5 | **logit-adjusted BCE** | loss | agaldran's robust workhorse | small code |
| 6 | Generative (diffusion) hard positives | B novelty | break 127-pos wall (winner didn't) | heavy |
| 7 | SurgMotion-L member | A | downloadable ViT-L | gated + V-JEPA2 |
| 8 | LoRA ensemble | A | — | deprioritized (full-FT preferred) |

**Bottom line:** only 🟢-confirmed wins = **GastroNet-DINOv2 backbone + semi (MT+PU) on a light full-FT model**. The queue's real headroom = **diverse strong ensemble members (#1,#2) + recalibration (#3)**; novelty for the paper = **generative positives (#6)** + the honest negatives (CRISP, MixStyle, DANN).
