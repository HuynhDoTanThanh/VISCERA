"""Label-confirmation harness driver — is the Phase-1 concept set usable as foundation labels?

Two layers (see cf/confirm.py and cf/audit.py):
  Layer 1  statistical  — held-out generalization, bootstrap CIs, center-leakage + within-center,
                          reliability, assessability  ->  per-concept PASS/REVIEW/FAIL.  No API.
  Layer 2  agentic      — stratified independent re-grade (faithfulness) + LLM meta-review verdict.

    # offline statistical confirmation only (fast, no quota):
    .venv/bin/python -m scripts.run_confirm

    # add the agentic faithfulness audit + meta-review (uses the local gemini proxy):
    .venv/bin/python -m scripts.run_confirm --audit --audit-n 60 --workers 8

Writes outputs/reports/label_confirmation.{md,json}.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cf import config
from cf.confirm import confirm, format_report, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-index", default=str(config.CACHE_DIR / "index_train.json"))
    ap.add_argument("--val-index", default=str(config.CACHE_DIR / "index_val.json"))
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for AUROC CIs")
    ap.add_argument("--audit", action="store_true", help="run Layer-2 agentic faithfulness audit")
    ap.add_argument("--audit-n", type=int, default=60, help="frames to re-grade in the audit")
    ap.add_argument("--audit-split", choices=["val", "train"], default="val")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-meta", action="store_true", help="skip the LLM meta-review verdict")
    args = ap.parse_args()

    for p in (args.train_index, args.val_index):
        if not Path(p).exists():
            print(f"Error: index not found: {p}. Run run_extract first.")
            return

    print("[confirm] Layer 1 — statistical confirmation (held-out generalization + center gates)…")
    rep = confirm(args.train_index, args.val_index, boot=args.boot)
    text = format_report(rep)
    print("\n" + text + "\n")

    if args.audit:
        from cf.audit import stratify, audit_faithfulness, meta_review
        core = rep["confirmed_core"] or [r["concept"] for r in rep["rows"]
                                         if r["role"] == "discriminative" and r["status"] != "FAIL"]
        idx = args.val_index if args.audit_split == "val" else args.train_index
        records, frames = load_split(idx)
        samples = stratify(records, frames, core, n=args.audit_n)
        print(f"[confirm] Layer 2 — agentic re-grade of {len(samples)} {args.audit_split} frames "
              f"on {len(core)} core concepts (model={config.EXPERTS.get('gemini')})…")
        audit = audit_faithfulness(samples, core, workers=args.workers)
        rep["audit"] = audit
        text += "\n\n## Agentic faithfulness audit (independent re-grade)\n"
        text += (f"- sampled {audit['n_sampled']}, graded {audit['n_graded']} "
                 f"(baseline={audit['n_baseline']}, risk={audit['n_risk']}), model {audit['model']}\n")
        text += ("- baseline = representative frames (judge label faithfulness here); "
                 "risk = adversarial hardest frames (stress test, expected worse)\n")
        head = f"\n{'concept':22} {'MAE_base':>9} {'agr_base':>9} {'MAE_risk':>9} {'agr_risk':>9}\n"
        text += head + "-" * (len(head) - 2) + "\n"
        pb, pr = audit["per_concept_baseline"], audit["per_concept_risk"]
        for c in core:
            b, r = pb.get(c), pr.get(c)
            bm = f"{b['mae']:.3f}" if b else "  -  "
            ba = f"{b['agree@0.5']:.2f}" if b else "  -  "
            rm = f"{r['mae']:.3f}" if r else "  -  "
            ra = f"{r['agree@0.5']:.2f}" if r else "  -  "
            text += f"{c:22} {bm:>9} {ba:>9} {rm:>9} {ra:>9}\n"

        if not args.no_meta:
            print("[confirm] meta-review verdict…")
            verdict = meta_review(text, audit)
            if verdict:
                rep["verdict"] = verdict
                text += "\n## Meta-review verdict\n```json\n" + json.dumps(verdict, indent=2) + "\n```\n"
                print("VERDICT:", verdict.get("verdict"))

    rdir = config.REPORT_DIR
    (rdir / "label_confirmation.md").write_text(text)
    (rdir / "label_confirmation.json").write_text(json.dumps(rep, indent=2))
    print(f"[confirm] wrote {rdir/'label_confirmation.md'} and .json")
    print(f"[confirm] CONFIRMED core ({len(rep['confirmed_core'])}): {', '.join(rep['confirmed_core'])}")
    if rep["review"]:
        print(f"[confirm] REVIEW: {', '.join(rep['review'])}")
    if rep["failed"]:
        print(f"[confirm] FAILED: {', '.join(rep['failed'])}")


if __name__ == "__main__":
    main()
