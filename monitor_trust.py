#!/usr/bin/env python3
"""Neo-style live monitor for label trust scores.

Watches out/train/labels for new .json files and streams each one's
frame_trust / decision / suspicion in Matrix-green terminal style.

Usage:
    python3 monitor_trust.py                  # watch out/train/labels, poll 2s
    python3 monitor_trust.py 1                # poll every 1s
    python3 monitor_trust.py 2 out/train/labels
"""
import json
import sys
import time
import glob
import os
from collections import Counter

INTERVAL = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
LABELS = sys.argv[2] if len(sys.argv) > 2 else "out/train/labels"

# ANSI — neo/matrix palette
G = "\033[38;5;46m"   # bright green
DG = "\033[38;5;28m"  # dim green
Y = "\033[38;5;226m"  # yellow
R = "\033[38;5;196m"  # red
DIM = "\033[2m"
B = "\033[1m"
RST = "\033[0m"


def trust_color(t):
    if t >= 0.75:
        return G
    if t >= 0.5:
        return Y
    return R


def bar(t, width=20):
    n = int(round(t * width))
    return G + "█" * n + DG + "░" * (width - n) + RST


def fmt(path, d):
    name = d.get("name", os.path.basename(path))[:40]
    t = float(d.get("frame_trust", 0.0))
    dec = d.get("decision", "?")
    susp = float(d.get("suspicion", 0.0))
    ver = "✓" if d.get("verified") else "·"
    c = trust_color(t)
    return (f"{DG}{time.strftime('%H:%M:%S')}{RST} "
            f"{bar(t)} {c}{B}{t:5.3f}{RST} "
            f"{DG}susp{RST}{susp:4.2f} {ver} "
            f"{c}{dec:<14}{RST} {DIM}{name}{RST}")


def main():
    seen = set()
    trusts = []
    decisions = Counter()

    print(f"\n{G}{B}┌─ NEO TRUST MONITOR ─────────────────────────────────┐{RST}")
    print(f"{DG}  watching {LABELS}  ·  poll {INTERVAL}s  ·  Ctrl-C to stop{RST}\n")

    try:
        while True:
            files = sorted(glob.glob(os.path.join(LABELS, "*.json")))
            new = [f for f in files if f not in seen]
            for f in new:
                seen.add(f)
                try:
                    d = json.load(open(f))
                except Exception as e:
                    print(f"{R}  ! parse fail {os.path.basename(f)}: {e}{RST}")
                    continue
                t = float(d.get("frame_trust", 0.0))
                trusts.append(t)
                decisions[d.get("decision", "?")] += 1
                print(fmt(f, d))

            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print(f"\n{G}stopped — {len(seen)} files seen{RST}")


if __name__ == "__main__":
    main()
