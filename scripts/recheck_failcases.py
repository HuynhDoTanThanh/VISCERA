"""Find and RECHECK the worst label fail-cases — frames where an independent VLM re-grade
disagrees with the cached concept labels on the confirmed core set.

Pipeline:
  1. score a fail-rich candidate pool (risk-weighted) over the chosen split
  2. PASS 1  — adjudicate the pool; rank by disagreement (sign-flips vs cached)
  3. take the worst N fail-cases (default 100)
  4. PASS 2  — re-adjudicate those N independently to separate signal from noise
  5. classify each implicated concept:
       STABLE_WRONG  both re-grades land opposite the cached label  -> cached label suspect
       AMBIGUOUS     the two re-grades disagree with each other     -> genuinely hard frame
       NOISE         pass-2 agrees with cached                      -> pass-1 was a fluke

    .venv/bin/python -m scripts.recheck_failcases --split val --n 100 --pool 350 --workers 16

Writes outputs/reports/audit_failcases.{json,md}.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cf import config
from cf.confirm import confirm, load_split
from cf.audit import (stratify, regrade_pool, anchored_regrade_pool, alt_anchors,
                      frame_disagreements)


def classify(cached: float, a1: float, a2: float) -> str:
    cs, s1, s2 = cached >= 0.5, a1 >= 0.5, a2 >= 0.5
    if s1 == cs:                      # pass-1 didn't actually flip (shouldn't happen for implicated)
        return "NOISE"
    if s2 != cs and s1 == s2:         # both re-grades opposite cached, and agree with each other
        return "STABLE_WRONG"
    if s1 != s2:                      # the two re-grades disagree -> hard/ambiguous frame
        return "AMBIGUOUS"
    return "NOISE"                    # pass-2 reverted to cached side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--n", type=int, default=100, help="max fail-cases to recheck")
    ap.add_argument("--pool", type=int, default=350, help="candidate frames adjudicated in pass 1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--protocol", choices=["strict", "anchored"], default="anchored",
                    help="strict = standalone rubric (fast, biased low); "
                         "anchored = real extraction protocol with a different anchor set (FAIR)")
    ap.add_argument("--train-index", default=str(config.CACHE_DIR / "index_train.json"))
    ap.add_argument("--val-index", default=str(config.CACHE_DIR / "index_val.json"))
    args = ap.parse_args()

    # confirmed core set drives what we check (fall back to non-failed discriminative)
    rep = confirm(args.train_index, args.val_index, boot=200)
    core = rep["confirmed_core"] or [r["concept"] for r in rep["rows"]
                                     if r["role"] == "discriminative" and r["status"] != "FAIL"]
    print(f"[recheck] core concepts ({len(core)}): {', '.join(core)}")

    idx = args.val_index if args.split == "val" else args.train_index
    records, frames = load_split(idx)
    pool = stratify(records, frames, core, n=min(args.pool, len(frames)), baseline_frac=0.15)
    by_path = {s["path"]: s for s in pool}

    if args.protocol == "anchored":
        a1_anchors, a2_anchors = alt_anchors(seed=1), alt_anchors(seed=2)
        print(f"[recheck] protocol=anchored (FAIR: real extraction, different anchor set) · "
              f"PASS 1 re-extracting {len(pool)} {args.split} frames (3 votes each)…")
        adj1 = anchored_regrade_pool(pool, core, a1_anchors, workers=args.workers)
    else:
        print(f"[recheck] protocol=strict (standalone rubric) · PASS 1 adjudicating {len(pool)} frames…")
        adj1 = regrade_pool(pool, core, workers=args.workers)

    # rank by disagreement
    scored = []
    for path, a1 in adj1.items():
        d = frame_disagreements(by_path[path], a1, core)
        if d["n_flips"] >= 1:
            scored.append((d["severity"], path, d))
    scored.sort(key=lambda t: -t[0])
    failcases = scored[: args.n]
    print(f"[recheck] {len(scored)} frames flipped >=1 core concept; rechecking worst {len(failcases)}…")

    # PASS 2 — recheck the worst N (third independent reading)
    recheck_samples = [by_path[p] for _, p, _ in failcases]
    if args.protocol == "anchored":
        adj2 = anchored_regrade_pool(recheck_samples, core, a2_anchors, workers=args.workers)
    else:
        adj2 = regrade_pool(recheck_samples, core, workers=args.workers, seed_tag=1)

    cases, counts = [], {"STABLE_WRONG": 0, "AMBIGUOUS": 0, "NOISE": 0}
    concept_blame = {c: 0 for c in core}
    for severity, path, d in failcases:
        s = by_path[path]
        a1, a2 = adj1.get(path, {}), adj2.get(path, {})
        verdicts = {}
        for c in d["flipped"]:
            if c not in a2:
                continue
            try:
                v1 = float(a1[c]); v2 = float(a2[c])
            except (TypeError, ValueError):
                continue
            cached = s["agg"][c]["c"]
            cls = classify(cached, v1, v2)
            verdicts[c] = {"cached": round(cached, 3), "pass1": round(v1, 3),
                           "pass2": round(v2, 3), "rel": round(s["agg"][c]["r"], 3), "class": cls}
            if cls == "STABLE_WRONG":
                concept_blame[c] += 1
        frame_cls = ("STABLE_WRONG" if any(v["class"] == "STABLE_WRONG" for v in verdicts.values())
                     else "AMBIGUOUS" if any(v["class"] == "AMBIGUOUS" for v in verdicts.values())
                     else "NOISE")
        counts[frame_cls] += 1
        cases.append({"path": path, "label": s["label"], "center": s["center"],
                      "stratum": s["stratum"], "n_flips": d["n_flips"], "mae": round(d["mae"], 3),
                      "frame_class": frame_cls, "concepts": verdicts})

    out = {"split": args.split, "protocol": args.protocol, "core": core, "pool": len(pool),
           "n_flipped_total": len(scored), "n_rechecked": len(failcases),
           "frame_class_counts": counts,
           "concept_stable_wrong_counts": {k: v for k, v in
                                           sorted(concept_blame.items(), key=lambda kv: -kv[1])},
           "cases": cases}

    rdir = config.REPORT_DIR
    (rdir / "audit_failcases.json").write_text(json.dumps(out, indent=2))

    # readable summary
    proto_note = ("real extraction protocol, different anchor sets (FAIR)" if args.protocol == "anchored"
                  else "standalone strict rubric (biased conservative)")
    L = [f"# Fail-case recheck — {args.split} split — protocol={args.protocol}\n",
         f"- re-grade = {proto_note}",
         f"- core concepts checked: {', '.join(core)}",
         f"- candidate pool: {len(pool)}  ·  frames flipping >=1 core concept: {len(scored)}  ·  "
         f"rechecked worst: {len(failcases)}",
         f"- frame verdicts: STABLE_WRONG={counts['STABLE_WRONG']}  "
         f"AMBIGUOUS={counts['AMBIGUOUS']}  NOISE={counts['NOISE']}",
         "",
         "Interpretation: STABLE_WRONG = both independent re-grades disagree with the cached label "
         "(label likely wrong); AMBIGUOUS = re-grades disagree with each other (genuinely hard, "
         "down-weight via reliability/abstain); NOISE = second re-grade reverts (pass-1 fluke).",
         "",
         "## Concepts most often STABLE_WRONG"]
    for c, v in out["concept_stable_wrong_counts"].items():
        if v:
            L.append(f"   - {c}: {v}")
    L.append("\n## Worst 25 fail-cases")
    L.append(f"{'frame':40} {'lbl':>3} {'ctr':>8} {'cls':>13}  flipped concepts (cached/p1/p2)")
    L.append("-" * 120)
    for c in cases[:25]:
        fname = Path(c["path"]).name[:38]
        flips = "  ".join(f"{k}:{v['cached']}/{v['pass1']}/{v['pass2']}[{v['class'][:4]}]"
                          for k, v in c["concepts"].items())
        L.append(f"{fname:40} {c['label']:>3} {c['center']:>8} {c['frame_class']:>13}  {flips}")
    (rdir / "audit_failcases.md").write_text("\n".join(L))

    print("\n" + "\n".join(L[:10]))
    print(f"\n[recheck] wrote {rdir/'audit_failcases.md'} and .json")


if __name__ == "__main__":
    main()
