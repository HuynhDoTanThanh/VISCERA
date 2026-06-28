"""Bootstrap the few-shot anchor set (incl. a hard-negative look-alike) from EXISTING train votes.

Reads the already-cached train concept votes (cf index), aggregates them, runs the AnchorSelector,
and writes agent_system/artifacts/anchors.json. One-time tool — it may read `cf` caches; the core
pipeline never does. Re-run only if you change the anchor policy.

    python -m agent_system.tools.select_anchors
"""
from __future__ import annotations
import json
from pathlib import Path

from ..application.aggregation import Aggregator
from ..application.anchors import AnchorSelector
from ..config import Settings
from ..domain.entities import Frame, Vote
from ..infrastructure.dataset import DatasetLoader


def main():
    settings = Settings()
    settings.ensure_dirs()
    agg = Aggregator()

    # path -> center, from the labelled CSV
    center = {f.path: f.center for f in DatasetLoader(settings).train()}

    # existing train votes (cf index produced by the original extraction)
    idx_path = Settings.dataset_root.parent / "outputs" / "cache" / "index_train.json"
    if not idx_path.exists():
        raise SystemExit(f"need existing train votes at {idx_path}; run the cf extraction first")
    index = json.loads(idx_path.read_text())

    aggregates = []
    for e in index:
        cache_files = list(e["caches"].values())
        votes = []
        for cf_path in cache_files:
            p = Path(cf_path)
            if not p.exists():
                continue
            for raw in json.loads(p.read_text()).get("votes", []):
                votes.append(Vote(expert="gemini", lens=0, values=raw))
        if not votes:
            continue
        fr = Frame(path=e["path"], label=e["label"], center=center.get(e["path"], ""), split="train")
        aggregates.append(agg.aggregate(fr, votes))

    anchors = AnchorSelector(n_neo=settings.n_anchors_neo, n_ndbe=settings.n_anchors_ndbe,
                             n_hard=1).select(aggregates)
    out = settings.raw_store.parent / "anchors.json"
    out.write_text(json.dumps([{"path": a.path, "kind": a.kind, "center": a.center, "hard": a.hard}
                               for a in anchors], indent=2))
    print(f"wrote {len(anchors)} anchors -> {out}")
    for a in anchors:
        print(f"  {a.kind:5} {'HARD' if a.hard else '    '} {a.center:9} {Path(a.path).name}")


if __name__ == "__main__":
    main()
