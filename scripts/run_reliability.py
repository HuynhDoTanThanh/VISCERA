"""Phase-1 reliability calibration study script.

Loads the concept votes cache, aggregates the results across experts/votes, and
computes agreement (reliability) and mutual information with the targets.

Usage:
    .venv/bin/python -m scripts.run_reliability --index outputs/cache/index_train.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cf import config
from cf.aggregate import feature_matrix
from cf.reliability import study, format_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(config.CACHE_DIR / "index_train.json"),
                    help="path to run_extract's index JSON file")
    args = ap.parse_args()

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"Error: index file not found at {idx_path}. Did you run run_extract?")
        return

    print(f"[reliability] Loading index from {idx_path}")
    index = json.loads(idx_path.read_text())

    # Build the records mapping and frames list
    records = {}
    frames = []
    for entry in index:
        path = entry["path"]
        label = entry["label"]
        rec = {}
        missing = False
        for exp, cache_path in entry["caches"].items():
            p = Path(cache_path)
            if not p.exists():
                missing = True
                break
            rec[exp] = json.loads(p.read_text()).get("votes", [])
        if not missing:
            records[path] = rec
            frames.append({"path": path, "label": label})

    print(f"[reliability] Loaded {len(frames)} completed frames")
    if not frames:
        print("Error: No completed frames found in index caches.")
        return

    # Compute matrices
    print("[reliability] Aggregating votes and building feature matrices...")
    Xc, Xr, Xm, y, names = feature_matrix(records, frames)

    # Run reliability study
    print("[reliability] Running reliability x discriminativeness study...")
    rows = study(Xc, Xr, Xm, y, names)

    # Format and display table
    table = format_table(rows)
    print("\n" + table + "\n")

    # Save report
    report_path = config.REPORT_DIR / "reliability_report.txt"
    report_path.write_text(table)
    print(f"[reliability] Saved report to {report_path}")


if __name__ == "__main__":
    main()
