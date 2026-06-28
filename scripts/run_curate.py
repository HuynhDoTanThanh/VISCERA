"""Curate "best-sure" foundation labels from the cached concept votes.

Turns raw concept votes into the supervision artifact Phase-2/3 consumes:
  - per (frame, concept): a calibrated TRUST and a SUPERVISE/MASK flag
  - per frame: a decision (POSITIVE / TRUE_NEGATIVE / HARD_NEG_CANDIDATE / CONFIDENT_NEGATIVE / ABSTAIN)

Trust uses the cross-anchor robustness signal when available (the verified "best-sure" mode). On
the val pool this keeps the labels that flip ~1% under a different anchor set and masks those that
flip ~18% (see docs/CONFIRMATION_HARNESS.md). Build that signal with --extract-cross-anchor, which
re-extracts the split with a DIFFERENT anchor set (real protocol, cached separately, resumable).

    # curate labelled splits (run the cross-anchor pass first for best-sure trust):
    .venv/bin/python -m scripts.run_curate --split train --extract-cross-anchor --workers 24
    .venv/bin/python -m scripts.run_curate --split val   --extract-cross-anchor --workers 24
    # the big lever — curate the unlabeled pool into hard-negative candidates / abstain:
    .venv/bin/python -m scripts.run_curate --split unlabeled --extract-cross-anchor --workers 24

Writes outputs/reports/curation_<split>.{json,md} and a compact foundation manifest
outputs/cache/labels_<split>.jsonl (one row per frame: decision + supervised concept values+trust).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cf import config, data
from cf.aggregate import aggregate_frame
from cf.confirm import confirm, load_split
from cf.audit import anchored_regrade_pool, alt_anchors
from cf.extract import _key
from cf.trust import curate


def _cross_anchor_map(frames, core, seed, workers, extract):
    """Return {path: {concept: c_alt}} from the seed-`seed` alt-anchor caches; optionally extract."""
    anchors = alt_anchors(seed=seed)
    sig = ",".join(Path(p).name for p, _, _ in anchors)
    missing = [f for f in frames if not _key(f["path"], "gemini", sig).exists()]
    if missing and extract:
        print(f"[curate] cross-anchor re-extract: {len(missing)} frames missing seed-{seed} votes…")
        anchored_regrade_pool(missing, core, anchors, workers=workers)  # caches to disk
    x = {}
    for f in frames:
        kf = _key(f["path"], "gemini", sig)
        if kf.exists():
            agg = aggregate_frame({"gemini": json.loads(kf.read_text()).get("votes", [])})
            x[f["path"]] = {c: agg[c]["c"] for c in core}
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "unlabeled"], default="val")
    ap.add_argument("--extract-cross-anchor", action="store_true",
                    help="re-extract the split with a different anchor set first (best-sure trust)")
    ap.add_argument("--cross-anchor-seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--train-index", default=str(config.CACHE_DIR / "index_train.json"))
    ap.add_argument("--val-index", default=str(config.CACHE_DIR / "index_val.json"))
    args = ap.parse_args()

    rep = confirm(args.train_index, args.val_index, boot=200)
    core = [r["concept"] for r in rep["rows"]
            if r["role"] == "discriminative" and r["status"] != "FAIL"]
    print(f"[curate] core concepts ({len(core)}): {', '.join(core)}")

    idx = config.CACHE_DIR / f"index_{args.split}.json"
    records, frames = load_split(str(idx))
    print(f"[curate] {args.split}: {len(frames)} frames")

    x_by_path = _cross_anchor_map(frames, core, args.cross_anchor_seed, args.workers,
                                  args.extract_cross_anchor)
    if x_by_path:
        print(f"[curate] cross-anchor verification available for {len(x_by_path)}/{len(frames)} frames")
    else:
        print("[curate] no cross-anchor signal — trust falls back to within-vote r (capped). "
              "Pass --extract-cross-anchor for best-sure labels.")

    cur = curate(records, frames, core, rep, x_by_path=x_by_path or None)

    # compact foundation manifest: one row per frame with supervised concept values + trust
    rdir = config.REPORT_DIR
    man_path = config.CACHE_DIR / f"labels_{args.split}.jsonl"
    n_sup_cells = 0
    with open(man_path, "w") as fh:
        for row in cur["rows"]:
            sup = {c: {"c": v["c"], "trust": v["trust"]}
                   for c, v in row["per_concept"].items() if v["supervise"]}
            n_sup_cells += len(sup)
            fh.write(json.dumps({"path": row["path"], "label": row["label"],
                                 "center": row["center"], "decision": row["decision"],
                                 "frame_trust": row["frame_trust"], "suspicion": row["suspicion"],
                                 "verified": row["verified"], "supervise": sup}) + "\n")

    (rdir / f"curation_{args.split}.json").write_text(json.dumps(cur, indent=2))
    L = [f"# Foundation label curation — {args.split}\n",
         f"- frames: {cur['n_frames']}  ·  core concepts: {len(core)}  ·  "
         f"cross-anchor verified: {cur['cross_anchor_verified']}",
         f"- supervised (frame,concept) cells written: {n_sup_cells}",
         "", "## Frame decisions"]
    for k, v in sorted(cur["counts"].items(), key=lambda kv: -kv[1]):
        L.append(f"   - {k}: {v}")
    L += ["", "Decisions: POSITIVE/TRUE_NEGATIVE = labelled; HARD_NEG_CANDIDATE = unlabeled, looks "
          "neoplastic but ~1% prevalence => prime false-positive trainer; CONFIDENT_NEGATIVE = "
          "unlabeled, clearly benign; ABSTAIN = too few trustworthy concepts (PU-safe, not used "
          "as a hard negative)."]
    (rdir / f"curation_{args.split}.md").write_text("\n".join(L))

    print(f"[curate] decisions: {cur['counts']}")
    print(f"[curate] wrote {man_path} ({n_sup_cells} supervised cells), "
          f"{rdir/f'curation_{args.split}.md'} and .json")


if __name__ == "__main__":
    main()
