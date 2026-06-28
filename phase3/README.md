# RARE26 Phase-3 — Neoplasia detector (PPV@90R)

Frozen domain-adapted **DINOv2 ViT-B/14-reg** → pooled embedding → center-de-biased logistic ensemble →
calibrated score. Offline `--network=none` (image-only; the VLM concepts are training-time signal only).
Target metric: **PPV@90R** at ~1% prevalence, median over bootstrap; hidden test = **new held-out center(s)**
→ optimized and reported on **LOCO** (the new-center proxy), not same-center.

## Results (curve-point PPV@90R, 1% prevalence, bootstrap median)
| Setting | PPV@90R | note |
|---|---|---|
| VLM `suspicion` baseline | ~0.039 | what the VLM alone achieves |
| Frozen probe, **same-center held-out val** | **0.31** | ensemble; optimistic for a new center |
| Frozen probe, **LOCO mean** (new-center proxy) | **0.22–0.28** | honest target |
| Frozen probe, LOCO worst leg (train-c2→c1) | ~0.04 | the hard direction |

The frozen-probe ceiling is ~0.28 LOCO-mean. **Cross-center domain shift — not the center cue — is the wall**
(projecting out 32 center directions drops center-AUROC 1.0→0.54 with no LOCO change). Whitening, frozen
positive-augmentation, and large diverse-negative doses all hurt; small doses are within the ~[0,0.6] noise band
of 49–78 positives/center. **The next ceiling-raiser is end-to-end fine-tuning on a GPU** (`finetune.py`).

## Pipeline (all in `phase3/`)
| module | role |
|---|---|
| `featurize.py` | frozen DINOv2 → cached 4-pool embedding [cls·reg·patch_mean·patch_max] (parallel, resumable shards) |
| `evaluate.py` | rigorous harness: curve-point + operating-point + oracle PPV@R, **paired bootstrap**, LOCO, source-dedup |
| `dataset.py` | join embeddings ⊕ VLM concept supervision (training-time only) |
| `mine_hardneg.py` | manifest of 144k unlabeled; `CONFIDENT_NEGATIVE` (107k) vs `HARD_NEG_CANDIDATE` (31k, quarantined) |
| `train.py` | T1 trainer: pipe → center-debias(k=32) → logistic; pooled-CV select + LOCO check + center gate |
| `experiment.py` | **LOCO-primary** lever-ablation runner (the new-center proxy) |
| `augment_pos.py` | photometric/geometric augmented positives (helps only under fine-tuning) |
| `finetune.py` | **end-to-end GPU fine-tune** — the cross-center ceiling-raiser |
| `ship.py` / `infer.py` | build deployable ensemble artifact / offline image→score container entrypoint |

## Reproduce
```bash
# 1. cache embeddings (done): train + val
.venv/bin/python -m phase3.featurize --csv dataset/train.csv --out phase3/cache/feats_train.npz --workers 6
.venv/bin/python -m phase3.featurize --csv dataset/val.csv   --out phase3/cache/feats_val.npz   --workers 6
# 2. honest LOCO ablations
.venv/bin/python -m phase3.experiment
# 3. build + run the deployable scorer
.venv/bin/python -m phase3.ship  --out phase3/cache/ship_model.pkl
.venv/bin/python -m phase3.infer --model phase3/cache/ship_model.pkl --images-dir <TEST_DIR> --out preds.csv
```

## Cloud GPU fine-tune (the real upside)
```bash
# LOCO legs (estimate new-center gain), then ship a both-center model:
python -m phase3.finetune --holdout center_2 --unfreeze 4 --epochs 40 --bs 64 \
    --neg-list phase3/cache/unl_confneg.txt --neg-cap 6000 --out phase3/cache/ft_c2.pt
python -m phase3.finetune --holdout center_1 ... --out phase3/cache/ft_c1.pt   # other leg
python -m phase3.finetune --holdout none     ... --out phase3/cache/ft_ship.pt # deployable
```
Heavy augmentation (ColorJitter/blur/rotation = cross-center nuisance) + backbone gradients is where
augmentation finally pays off. Select on the LOCO-val PPV@90R printed each epoch.
