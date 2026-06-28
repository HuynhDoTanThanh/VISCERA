"""Phase-1 driver: multi-expert concept extraction over a labeled subset / unlabeled pool.

    .venv/bin/python -m scripts.run_extract --split train --n-neo 30 --n-ndbe 120 --workers 8
    .venv/bin/python -m scripts.run_extract --split val --workers 8
    .venv/bin/python -m scripts.run_extract --split unlabeled --limit 500 --workers 8

Extraction is cached per (image, expert, anchors) and resumable — re-running fills gaps.
Writes an index of (path, label, caches) so the offline aggregation/reliability sweep
loads records by path without re-calling the API.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cf import config, data
from cf.extract import extract_frame, _key


def build_frames(args):
    train = data.load_labeled(config.TRAIN_CSV)
    if args.split == "train":
        neo = [r for r in train if r["label"] == 1]
        ndbe = [r for r in train if r["label"] == 0]
        rng = random.Random(args.seed)
        frames = (rng.sample(neo, min(args.n_neo, len(neo)))
                  + rng.sample(ndbe, min(args.n_ndbe, len(ndbe))))
    elif args.split == "val":
        frames = data.load_labeled(config.VAL_CSV)
    else:
        frames = data.load_unlabeled(limit=args.limit)
    return frames, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "unlabeled"], default="train")
    ap.add_argument("--n-neo", type=int, default=30)
    ap.add_argument("--n-ndbe", type=int, default=120)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--workers", type=int, default=config.MAX_CONCURRENT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experts", nargs="+", default=None,
                    help="list of experts to extract (defaults to all in config)")
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()

    frames, train = build_frames(args)
    anchors = data.pick_anchors(train, seed=args.seed)
    anchor_sig = ",".join(Path(p).name for p, _, _ in anchors)

    target_experts = args.experts if args.experts is not None else list(config.EXPERTS.keys())
    print(f"[extract] split={args.split} frames={len(frames)} experts={target_experts} "
          f"votes={config.VOTES_PER_EXPERT} anchors={[a[1] for a in anchors]} "
          f"workers={args.workers} force={args.force}", flush=True)

    def work(fr):
        try:
            rec = extract_frame(fr["path"], anchors, experts=target_experts, force=args.force)
            return fr, sum(len(v) for v in rec.values()), True, ""
        except Exception as e:  # noqa: BLE001
            return fr, 0, False, str(e)[:80]

    t0, done, ok, votes = time.time(), 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, fr) for fr in frames]
        for f in as_completed(futs):
            fr, n, success, err = f.result()
            done += 1; ok += success; votes += n
            if not success:
                print(f"  ! {fr['path']}: {err}", flush=True)
            if done % 10 == 0 or done == len(futs):
                el = time.time() - t0
                eta = (len(futs) - done) / (done / el) if done else 0
                print(f"  {done}/{len(futs)}  ok={ok}  votes={votes}  "
                      f"({el:.0f}s, ETA {eta:.0f}s)", flush=True)

    # index for the offline sweep: one cache file per expert
    index = [{"path": fr["path"], "label": fr["label"],
              "caches": {e: str(_key(fr["path"], e, anchor_sig)) for e in target_experts}}
             for fr in frames]
    index = [e for e in index if all(Path(c).exists() for c in e["caches"].values())]
    idx_path = config.CACHE_DIR / f"index_{args.split}.json"
    idx_path.write_text(json.dumps(index))
    print(f"[index] wrote {len(index)} entries -> {idx_path}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
