"""Probe the VLM gateway's concurrency ceiling.

Ramps the number of in-flight requests and measures throughput (calls/min),
latency (p50/p95), and error rate at each level. The ceiling is where calls/min
stops rising and/or errors climb — that's the semaphore size to use for async.

    python -m agent_system.tools.probe_capacity                       # default ramp
    python -m agent_system.tools.probe_capacity --levels 16 32 64 128 192
    python -m agent_system.tools.probe_capacity --max-tokens 512 --frames 8

Note: --max-tokens defaults LOW (probe measures connection/queue concurrency, not
generation length). Real extraction jobs emit up to 8192 tokens, so absolute
latency will be higher in production — but the *ceiling* (where errors appear) holds.
"""
from __future__ import annotations
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import Settings
from ..domain.entities import Anchor
from ..infrastructure.dataset import DatasetLoader
from ..infrastructure.vlm import ProxyVLMClient
from ..prompts import SYSTEM_QUALITY, quality_query_prompt, LENSES
import json
from pathlib import Path


def load_anchors(path: Path) -> list[Anchor]:
    data = json.loads(Path(path).read_text())
    return [Anchor(path=a["path"], kind=a.get("kind") or a["cls"],
                   center=a.get("center", ""), hard=bool(a.get("hard", False)))
            for a in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128, 192])
    ap.add_argument("--frames", type=int, default=8, help="distinct frames cycled across requests")
    ap.add_argument("--max-tokens", type=int, default=8192)  # model needs full budget to emit JSON
    ap.add_argument("--reps", type=float, default=1.0, help="requests per level = reps × level")
    ap.add_argument("--model", default=None, help="gateway model id to probe (default: first expert)")
    args = ap.parse_args()

    import dataclasses
    from ..config import ModelSpec
    s = dataclasses.replace(Settings(), max_tokens=args.max_tokens)  # cheaper/faster probe gens
    s.ensure_dirs()
    expert = ModelSpec("probe", args.model) if args.model else next(sp for sp in s.experts)
    client = ProxyVLMClient(expert, s)
    anchors = load_anchors(s.raw_store.parent / "anchors.json")
    ndbe = [f for f in DatasetLoader(s).val() if f.label == 0]
    pool = random.Random(1).sample(ndbe, min(args.frames, len(ndbe)))
    prompt = quality_query_prompt(LENSES[0])

    def one(frame):
        t0 = time.time()
        try:
            v = client.read(SYSTEM_QUALITY, anchors, frame.path, prompt, s.base_temp, 1)
            return (time.time() - t0, v is not None)
        except Exception:
            return (time.time() - t0, False)

    print(f"[probe] model={expert.model}  frames={len(pool)}  max_tokens={args.max_tokens}")
    print(f"{'conc':>6}{'reqs':>6}{'sec':>7}{'calls/min':>11}{'ok':>5}{'err':>5}"
          f"{'err%':>7}{'p50 s':>8}{'p95 s':>8}")
    print("-" * 63)

    best_tput, prev_tput = 0.0, 0.0
    for c in args.levels:
        n = max(c, int(round(args.reps * c)))
        reqs = [pool[i % len(pool)] for i in range(n)]
        t0 = time.time()
        lat, ok = [], 0
        with ThreadPoolExecutor(max_workers=c) as ex:
            for fut in as_completed([ex.submit(one, f) for f in reqs]):
                dt, good = fut.result()
                lat.append(dt); ok += int(good)
        secs = time.time() - t0
        err = n - ok
        lat.sort()
        p50 = lat[int(0.50 * (len(lat) - 1))]
        p95 = lat[int(0.95 * (len(lat) - 1))]
        tput = n / secs * 60
        best_tput = max(best_tput, tput)
        flag = ""
        if err / n > 0.05:
            flag = "  <- errors climbing"
        elif prev_tput and tput < prev_tput * 1.05:
            flag = "  <- throughput plateau"
        print(f"{c:>6}{n:>6}{secs:>7.0f}{tput:>11.1f}{ok:>5}{err:>5}"
              f"{err/n*100:>6.0f}%{p50:>8.1f}{p95:>8.1f}{flag}")
        prev_tput = tput

    print(f"\n[probe] peak throughput ≈ {best_tput:.0f} calls/min.")
    print("[probe] Set the async semaphore at the first level where throughput stops "
          "rising AND error% stays ~0.")


if __name__ == "__main__":
    main()
