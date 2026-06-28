# RARE2026 — Barrett's Neoplasia Detection (PPV@90R)

Detect esophageal **neoplasia** (`neo`) vs non-dysplastic Barrett's (`ndbe`) in endoscopy frames.
Metric: **PPV@90R** at ~1% prevalence (median bootstrap). Hidden test = a **new center** (domain generalization).
Deployment: offline `--network=none` container, **image-only** at test.

## Pipeline
1. **Phase 1 — VLM concept extraction** (`agent_system/`, `scripts/run.sh`): multi-expert VLM extracts 35 atomic
   clinical concepts + suspicion per frame for ~170k frames (dev-time only; no VLM ships).
2. **Phase 2 — SSL backbone** (`dinov2.pth`): DINOv2 ViT-B/14-reg domain-adapted on the frames (frozen).
3. **Phase 3 — detector** (`phase3/`): the deployable scorer + experiments. See **`phase3/ARCHITECTURE.md`**.

## Key results (PPV@90R, new-center / LOCO)
| | PPV@90R |
|---|---|
| VLM `suspicion` baseline | ~0.04 |
| frozen DINOv2 + logistic | LOCO-mean 0.28 / worst 0.04 (in-distribution-flattered) |
| **real RARE25 ceiling (12-center test)** | **winner 0.035; all teams <0.04** |

The leaderboard is decided in [0.03, 0.06]; the dominant levers are a **trustworthy measurement**, a
**decorrelated ensemble**, **clean per-image deployment**, and the **operating-point tail loss** — see
`phase3/ARCHITECTURE.md` §7 and the memory notes. Concept-supervised pretraining (`pretrain_concept.py`)
is under a pre-registered fairness gate (`concept_gate.py`).

## Data (NOT in git — via Google Drive)
`dinov2.pth`, `dataset/train|val/` images, and `out/*.zip` (the 170k concept JSONs + images) are large and
live on Drive. Only code + label CSVs are versioned.

## Run on Colab (A6000/A100)
Open **`phase3/colab_full_pipeline.ipynb`** → git clone + extract Drive zips + run the full pipeline
(concept-pretrain → fair gate → downstream). See that notebook for the Drive upload checklist.

## Phase-3 module map
`featurize` (DINOv2→embeddings) · `evaluate` (curve/paired-bootstrap/LOCO harness) · `dataset` (embed⊕concept join)
· `mine_hardneg` · `build_concept_targets` · `pretrain_concept` (Stage-1 concept distillation + GRL)
· `concept_gate` (fair SSL-vs-concept gate) · `train`/`experiment` (frozen-probe + LOCO ablations)
· `finetune` (end-to-end FT, tail loss) · `ship`/`infer` (deployable .pkl/.pt, per-image).
