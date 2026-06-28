"""Phase-3 dataset assembly: join cached DINOv2 embeddings with the VLM concept supervision.

OFFLINE-LEGAL CONTRACT: at test time the container sees ONLY the image -> embedding z. So concepts
(c=value, r=trust, supervise mask) are TRAINING-TIME supervision / aux targets / mining signals — never
test-time input features. This module returns z (always available) plus the concept arrays (when a
label JSON exists) so a head can use concepts as auxiliary multitask targets and for hard-neg mining.

Concept ordering is the single source of truth in agent_system/domain/concept_schema.py.
"""
from __future__ import annotations
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_system.domain.concept_schema import CONCEPTS  # noqa: E402

CONCEPT_NAMES = [c.name for c in CONCEPTS]
CONCEPT_ROLE = {c.name: c.role for c in CONCEPTS}
DISCRIMINATIVE = [c.name for c in CONCEPTS if c.role == "discriminative"]
CENTER_CUE = [c.name for c in CONCEPTS if c.role == "center_cue"]
GESTALT = [c.name for c in CONCEPTS if c.role == "gestalt"]
NC = len(CONCEPT_NAMES)


def load_feats(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    out = dict(names=d["names"].astype(str), feats=d["feats"].astype(np.float32),
               label=d["label"].astype(np.int64) if "label" in d.files else None,
               center=d["center"].astype(str) if "center" in d.files else None,
               source=d["source"].astype(str) if "source" in d.files else None,
               aug=d["aug"].astype(str) if "aug" in d.files else None,
               pool_names=list(d["pool_names"].astype(str)) if "pool_names" in d.files else None)
    return out


def index_concept_dirs(*label_dirs: str) -> dict:
    """Map frame name -> parsed label JSON, across one or more out/*/labels directories."""
    idx = {}
    for ld in label_dirs:
        for fp in glob.glob(os.path.join(ld, "*.json")):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            idx[d.get("name", os.path.splitext(os.path.basename(fp))[0])] = d
    return idx


def concept_arrays(rec: dict | None):
    """Return (value[NC], trust[NC], supervise[NC], suspicion, decision) for one label record."""
    if rec is None:
        return (np.zeros(NC, np.float32), np.zeros(NC, np.float32), np.zeros(NC, np.float32),
                np.float32(np.nan), "")
    cj = rec.get("concepts", {})
    val = np.zeros(NC, np.float32); tr = np.zeros(NC, np.float32); sup = np.zeros(NC, np.float32)
    for i, name in enumerate(CONCEPT_NAMES):
        c = cj.get(name)
        if c:
            val[i] = float(c.get("value", 0.0)); tr[i] = float(c.get("trust", 0.0))
            sup[i] = 1.0 if c.get("supervise", False) else 0.0
    return val, tr, sup, np.float32(rec.get("suspicion", np.nan)), rec.get("decision", "")


def assemble(feats_npz: str, *concept_label_dirs: str) -> dict:
    """Build the full Phase-3 matrix: z + concept (value/trust/supervise) + suspicion/decision + meta."""
    F = load_feats(feats_npz)
    cidx = index_concept_dirs(*concept_label_dirs) if concept_label_dirs else {}
    n = len(F["names"])
    C = np.zeros((n, NC), np.float32); R = np.zeros((n, NC), np.float32)
    S = np.zeros((n, NC), np.float32); susp = np.full(n, np.nan, np.float32)
    dec = np.empty(n, dtype=object); has_concept = np.zeros(n, bool)
    for i, nm in enumerate(F["names"]):
        rec = cidx.get(nm)
        has_concept[i] = rec is not None
        v, t, s, sp, dc = concept_arrays(rec)
        C[i], R[i], S[i], susp[i], dec[i] = v, t, s, sp, dc
    return dict(
        names=F["names"], z=F["feats"], y=F["label"], center=F["center"],
        source=F["source"] if F["source"] is not None else F["names"], aug=F["aug"],
        c_value=C, c_trust=R, c_supervise=S, suspicion=susp, decision=dec.astype(str),
        has_concept=has_concept, pool_names=F["pool_names"],
        concept_names=np.array(CONCEPT_NAMES),
        discriminative_idx=np.array([CONCEPT_NAMES.index(x) for x in DISCRIMINATIVE]),
        center_cue_idx=np.array([CONCEPT_NAMES.index(x) for x in CENTER_CUE]),
    )


if __name__ == "__main__":
    # quick self-check once feats_train.npz exists
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="phase3/cache/feats_train.npz")
    ap.add_argument("--labels", nargs="*", default=["out/train/labels"])
    a = ap.parse_args()
    if not os.path.exists(a.feats):
        print(f"(featurize first: {a.feats} not found)"); raise SystemExit
    D = assemble(a.feats, *a.labels)
    print(f"assembled n={len(D['names'])}  z={D['z'].shape}  concepts={D['c_value'].shape}")
    print(f"label balance: {np.bincount((D['y']>=0).astype(int))}  pos={int((D['y']==1).sum())}")
    print(f"has_concept: {D['has_concept'].mean()*100:.1f}%  | discriminative concepts: {len(D['discriminative_idx'])}")
    print(f"pool_names: {D['pool_names']}  embed-dim/pool={D['z'].shape[1]//len(D['pool_names'])}")
