# agent_system

Clean-architecture pipeline that turns endoscopy frames into **reliability-weighted clinical-concept
labels** for training the RACE foundation model, with two output stores: raw votes + logs, and a
foundation-ready training set.

## Architecture (dependencies point inward)

```
                 ┌─────────────┐
   cli.py  ─────►│ application │─────►  domain   (entities + concept vocabulary; no I/O)
 (compose)       └─────────────┘            ▲
                   │       │  depends on     │
                   ▼       ▼  ports.py ──────┘
            infrastructure   outputs
            (VLM, cache,     (raw + training
             dataset, log)    store writers)
```

| Layer | Module | Responsibility |
|---|---|---|
| domain | `domain/entities.py`, `domain/concepts.py` | `Frame`, `Vote`, `ConceptCell`, `CuratedFrame`, `Decision`; the clinical vocabulary + chunk groups (canonical defs reused from `cf.concept_schema`) |
| ports | `ports.py` | `VLMClient.read(...)`, `VoteCache` interfaces |
| application | `application/` | `AnchorSelector`, `ExtractionStrategy` (`SingleShot` / `MultiStage`), `Extractor`, `Aggregator`, `TrustScorer`+`Curator`, `LabelPipeline`, `confirm` (gates), `audit` (agentic faithfulness) |
| domain vocab | `domain/concept_schema.py` | the 35-concept clinical vocabulary (self-contained; no external deps) |
| prompts | `prompts.py`, `prompts_stages.py` | single-shot prompt; the 5-stage prompt assets (context/region/chunks/refine/confirm) |
| infrastructure | `infrastructure/` | `ProxyVLMClient`, `FileVoteCache`, `DatasetLoader`, run logging |
| outputs | `outputs/writers.py` | `RawStore`, `TrainingStore` |
| compose | `cli.py` | wires everything, selects strategy, runs a split |

The application depends only on `ports` + `domain`; swapping the proxy for another backend or the
file cache for a DB means writing one adapter — no use-case changes.

## What the pipeline does (per frame, concurrently)

1. **Extract** — panel of experts (`gemini-pro-agent` + `gemini-3-flash-agent`) reads each frame via
   an `ExtractionStrategy`:
   - **single + quality prompt (DEFAULT)** — one call per vote with a rubric-anchored (0-4 levels
     defined), contrastive-to-reference, tier-grouped prompt. Measured to beat the bare single
     prompt on discriminative AUROC (+0.030) and the hard classes (vascular +0.14, colour +0.08)
     at the *same* 1-call cost, and to match/beat multistage at ~1/5 the API. Reliability comes from
     the **2-model panel** (cross-model agreement) + the 5 lenses.
   - **multistage** — a predict→refine→confirm agent (context → region+crop → chunked → morphology →
     refine → confirm). ~5-8× the API; kept for ablation / high-value subsets, not the default.
   Few-shot **anchored** (incl. a hard-negative look-alike), cached & resumable; cache namespaced by
   strategy + prompt variant.
2. **Aggregate** — votes → per-concept `(value, reliability, mask)`. With two model families,
   `reliability` is cross-**model** agreement (genuine independence).
3. **Trust** — `trust = gate × mask × (reliability × cross_anchor)`; calibrated so kept labels flip
   ~1% under a different anchor set vs ~18% for masked ones.
4. **Curate** — per-frame decision: `POSITIVE / TRUE_NEGATIVE / HARD_NEG_CANDIDATE /
   CONFIDENT_NEGATIVE / ABSTAIN`, and a `SUPERVISE/MASK` flag per concept.

## Output stores

```
artifacts/
  anchors.json                         # few-shot anchors (incl. hard negative)
  raw_store/
    logs/<run>.log                     # per-run log
    raw_labels/<sha1>.json             # raw votes per (frame, expert, anchors) — resumable
    crops/<sha1>.png                   # multi-stage region crops (when used)
    runs/<run>/manifest.json           # run config + stats
  training_store/[<name>/]             # paired image+label dataset (--name nests under a subfolder)
    images/<stem>.jpg                  # the frame, converted to JPEG
    labels/<stem>.json                 # that image's label, matched by file stem
```

One **label file per image** (matched by stem). A label:
`{image, name, split, label, center, decision, frame_trust, suspicion, verified,
concepts:{<name>:{value, trust, supervise}}}`.
- **All 35 concepts are always present** (fixed-shape label — easy to train on). `value` ∈ [0,1],
  `trust` ∈ [0,1] is the **soft weight** to train with (down-weight low-trust; don't drop),
  `supervise` flags whether it cleared the trust threshold.
- `label`: 1=neo, 0=ndbe, -1=unlabeled. Train e.g. `loss = Σ trust_c · ℓ(pred_c, value_c)`.

### Pointing at your own data / choosing the output
```bash
# write the dataset to any folder you choose
python -m agent_system.cli --split train --out ./train_dataset
#   -> ./train_dataset/images/*.jpg + ./train_dataset/labels/*.json

# process one unlabeled shard and name the output
python -m agent_system.cli --split unlabeled \
  --unlabeled-dir dataset/unlabeled_data/100 --out ./labels_out --name 100
#   -> ./labels_out/100/images + ./labels_out/100/labels
```
- `--out PATH` — where to write the dataset (default: `agent_system/artifacts/training_store`)
- `--name N` — nest under `<out>/<N>/` (else flat `images/`+`labels/`)
- `--unlabeled-dir PATH` — unlabeled images (recursive PNG search); `--dataset-root PATH` — override root

A training row: `{image, label, center, decision, frame_trust, suspicion, verified,
concepts:{name:{value,trust}}}` — only trust-passing concepts are present (the rest are masked).

## Usage

```bash
# one-time: pick anchors (incl. hard negative) from existing train votes
python -m agent_system.tools.select_anchors

# generate labels for a split (multistage is the default strategy)
python -m agent_system.cli --split val --link-images
python -m agent_system.cli --split unlabeled --limit 9937 --workers 24
python -m agent_system.cli --split train --strategy single   # baseline / faster ablation

# measure the strategy lift on val (per-concept AUROC, reliability, mask-rate)
python -m agent_system.tools.ab_strategy --expert proagent --n-ndbe 60

# confirm the labels are usable as supervision (held-out gates + agentic faithfulness audit)
python -m agent_system.tools.confirm --train-ndbe 200 --val-ndbe 200 --audit --audit-n 60

# optimize generation time (min votes / max workers)
python -m agent_system.tools.optimize_generation --mode votes
python -m agent_system.tools.optimize_generation --mode batch --workers 16 32 48 64
```

Self-contained: the clinical vocabulary lives in `domain/concept_schema.py` — no external `cf`
package. The confirmation gates + agentic audit (formerly in `cf`) are now `application/confirm.py`
and `application/audit.py`.

Config is centralized in `config.py` (endpoints, expert panel, votes, trust gates, paths); override
via env (`AS_*`, `CF_CLOUD_KEY`) or CLI flags. Re-runs reuse `raw_labels/` and only call the model
for gaps.

## Notes
- Cross-anchor verification (`verified=true`, the best-sure trust) requires a second extraction with
  a different anchor set; until then trust is capped by `unverified_penalty` (0.7).
- `cf.concept_schema` remains the single source of truth for the concept vocabulary.
```
