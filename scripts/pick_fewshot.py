"""Pick 5 highly-sure, general few-shot anchor frames from the TRAIN set.

Anchors are the in-context reference frames every extraction sees, so they must be
UNAMBIGUOUS and CANONICAL — a wrong/edge-case anchor poisons every downstream label.
We score each train frame by:

  sureness = agreement(core) × assessable(core) × label_consistency × quality

  - agreement / assessable : mean inter-vote reliability & assessability over the robust core
  - label_consistency      : neo -> high core suspicion, ndbe -> low core suspicion
  - quality                : low blur/glare/exposure/debris/mucus, modality assessable

Then take the surest 2 neo + 3 ndbe, forcing center & modality diversity so the few-shot
set spans the dataset's looks. Writes outputs/reports/fewshot_anchors.json.
"""
from __future__ import annotations
import json

import numpy as np

from cf import config
from cf.aggregate import aggregate_frame
from cf.confirm import load_split

ROBUST_CORE = ["mucosal_irregularity", "nodularity", "demarcation", "lesion_present",
               "focal_erythema", "surface_effacement", "colocalization"]
QUALITY = ["blur", "glare", "exposure", "debris", "mucus_bubbles"]
N_NEO, N_NDBE = 2, 3


def sureness(agg, label):
    agree = np.mean([agg[c]["r"] for c in ROBUST_CORE])
    assess = np.mean([agg[c]["m"] for c in ROBUST_CORE])
    susp = np.mean([agg[c]["c"] for c in ROBUST_CORE])
    consistency = susp if label == 1 else (1.0 - susp)
    quality = 1.0 - np.mean([agg[q]["c"] for q in QUALITY])     # low artifact => high quality
    return float(agree * assess * consistency * quality), float(susp)


def modality(agg_rec):
    # most common raw modality across votes (categorical) for diversity bookkeeping
    vals = []
    for votes in agg_rec.values():
        for v in votes:
            m = v.get("modality")
            if m:
                vals.append(m)
    return max(set(vals), key=vals.count) if vals else "unknown"


def diverse_pick(cands, k):
    """Greedily take top-sureness while preferring new (center, modality) combos."""
    out, seen = [], set()
    for c in sorted(cands, key=lambda x: -x["sureness"]):
        key = (c["center"], c["modality"])
        if key not in seen or len([o for o in out]) < k:
            out.append(c)
            seen.add(key)
        if len(out) == k:
            break
    if len(out) < k:  # backfill by pure sureness
        for c in sorted(cands, key=lambda x: -x["sureness"]):
            if c not in out:
                out.append(c)
            if len(out) == k:
                break
    return out


def main():
    records, frames = load_split(str(config.CACHE_DIR / "index_train.json"))
    neo, ndbe = [], []
    for fr in frames:
        rec = records.get(fr["path"])
        if not rec:
            continue
        agg = aggregate_frame(rec)
        s, susp = sureness(agg, fr["label"])
        item = {"path": fr["path"], "label": fr["label"], "center": fr["center"],
                "modality": modality(rec), "sureness": round(s, 4), "suspicion": round(susp, 3)}
        (neo if fr["label"] == 1 else ndbe).append(item)

    chosen = diverse_pick(neo, N_NEO) + diverse_pick(ndbe, N_NDBE)
    anchors = [{"path": c["path"], "cls": "neo" if c["label"] == 1 else "ndbe",
                "center": c["center"], "modality": c["modality"],
                "sureness": c["sureness"], "suspicion": c["suspicion"]} for c in chosen]
    out = config.REPORT_DIR / "fewshot_anchors.json"
    out.write_text(json.dumps(anchors, indent=2))

    print(f"Picked {len(anchors)} few-shot anchors (from {len(neo)} neo / {len(ndbe)} ndbe train frames):\n")
    print(f"{'cls':5} {'center':9} {'modality':14} {'sure':>6} {'susp':>5}  file")
    for a in anchors:
        print(f"{a['cls']:5} {a['center']:9} {a['modality']:14} {a['sureness']:6.3f} "
              f"{a['suspicion']:5.2f}  {a['path'].split('/')[-1]}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
