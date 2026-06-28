#!/usr/bin/env python3
"""Classification metrics on the `suspicion` score vs ground-truth `label`.

Positive class = 1. Score = `suspicion` from each label JSON.

Metrics:
    Recall, Precision, F1, Accuracy   -> at a fixed threshold (--thr, default 0.5)
    PPV@90R                           -> max precision among thresholds with recall >= 0.90

Usage:
    python3 eval_metrics.py                              # out/train/labels, thr=0.5
    python3 eval_metrics.py --dir out/val/labels --thr 0.5
    python3 eval_metrics.py --recall-target 0.95         # PPV@95R instead
    python3 eval_metrics.py --exclude-abstain            # drop ABSTAIN frames
"""
import argparse
import glob
import json
import os
import numpy as np


def load(dirpath, exclude_abstain):
    y, s, skipped = [], [], 0
    for f in glob.glob(os.path.join(dirpath, "*.json")):
        d = json.load(open(f))
        if exclude_abstain and d.get("decision") == "ABSTAIN":
            skipped += 1
            continue
        y.append(int(d["label"]))
        s.append(float(d["suspicion"]))
    return np.array(y), np.array(s), skipped


def thr_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, acc=acc)


def ppv_at_recall(y, s, target):
    """Max precision over all thresholds achieving recall >= target."""
    P = int((y == 1).sum())
    if P == 0:
        return None, None
    order = np.argsort(-s, kind="mergesort")        # high score first
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    recall = tp / P
    precision = tp / np.maximum(tp + fp, 1)
    mask = recall >= target
    if not mask.any():
        return None, None
    idx_pool = np.where(mask)[0]
    best = idx_pool[np.argmax(precision[idx_pool])]
    return float(precision[best]), float(ss[best])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/train/labels")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--recall-target", type=float, default=0.90)
    ap.add_argument("--exclude-abstain", action="store_true")
    a = ap.parse_args()

    y, s, skipped = load(a.dir, a.exclude_abstain)
    n, pos, neg = len(y), int((y == 1).sum()), int((y == 0).sum())
    print(f"dir={a.dir}  n={n}  pos={pos}  neg={neg}"
          + (f"  (skipped {skipped} ABSTAIN)" if skipped else ""))

    if n == 0:
        print("no samples."); return

    m = thr_metrics(y, s, a.thr)
    print(f"\n@thr={a.thr:g}   TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Accuracy  : {m['acc']:.4f}")

    rt = a.recall_target
    ppv, thr_at = ppv_at_recall(y, s, rt)
    label = f"PPV@{int(round(rt*100))}R"
    if ppv is None:
        print(f"\n  {label}    : n/a (no positives, or recall target unreachable)")
    else:
        print(f"\n  {label}    : {ppv:.4f}  (at score threshold {thr_at:.4f})")


if __name__ == "__main__":
    main()
