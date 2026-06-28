#!/usr/bin/env bash
# Run from the project root. Uses the project .venv explicitly so a shadowing conda/base python on
# PATH can't break imports (the .venv is the project's real environment).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# --- Gateways -------------------------------------------------------------------------------------
# Each 9router endpoint saturates ~256 concurrency / ~500 calls-min. List all your endpoints
# comma-separated; calls round-robin across them. Then set --workers auto:(256 × #gateways):
#   4 gateways -> auto:1024   ·   3 -> auto:768   ·   1 -> auto:256
# export AS_PROXY_URL="http://gw1:20128/v1,http://gw2:20128/v1,http://gw3:20128/v1,http://gw4:20128/v1"
#
# --workers auto:N = AIMD adaptive concurrency: grows while healthy, backs off on errors/latency.
# Safe even with fewer gateways than N implies — it self-limits to each gateway's real ceiling.
WORKERS="auto:1024"

# Config (agent_system/config.py): pro1 + flash3×3 = 4 calls/img (susp-AUROC 0.913 ≈ old 6-call panel).

# Exclude list: dir names (basename) to skip for a quick pass. Space-separated.
# Override at call time, e.g.:  EXCLUDE="dir3 dir7" scripts/run.sh
EXCLUDE="${EXCLUDE:-0 1 10 11 12 13 14 15 16 17 18 19 2 20 21 22}"

# Process every unlabeled_data/<dir>. Resumable: cached votes are skipped, so re-running after more
# data is downloaded only does the new frames.
for d in dataset/unlabeled_data/*/; do
  name=$(basename "$d")
  if [[ " $EXCLUDE " == *" $name "* ]]; then
    echo ">>> skipping excluded dir $name"
    continue
  fi
  echo ">>> unlabeled dir $name"
  $PY -m agent_system.cli --split unlabeled --unlabeled-dir "$d" --name "$name" \
    --out ./out --workers "$WORKERS"
done

# Train split (uncomment if you also need it):
# $PY -m agent_system.cli --split val --name val --out ./out --workers "$WORKERS"
