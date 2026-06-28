"""Validate the label dataset(s) written by the pipeline.

For every dataset folder under a root (a folder that contains a `labels/` subdir — both the flat
`out/labels` and the per-dir `out/<name>/labels` layouts are auto-discovered), check that each label
JSON is well-formed and complete, and that images and labels line up 1:1:

  - labels/ exists (and images/ if present)
  - every label file is valid JSON with all required top-level keys, correct types
  - `decision` is one of the Decision enum values
  - `concepts` has ALL 35 canonical concepts, each with value/trust/supervise of the right type
  - every image has a matching label and vice-versa (missing-label / orphan-label detection)
  - the image referenced by each label actually exists on disk

Exit code 0 = all folders clean, 1 = at least one problem found. Use --strict to also fail when a
folder has images with no label yet (mid-run / incomplete), otherwise that is reported as a warning.

    python -m agent_system.tools.check_labels                 # scan ./out
    python -m agent_system.tools.check_labels out other_dir    # scan specific roots
    python -m agent_system.tools.check_labels --json           # machine-readable summary
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from ..domain.concepts import ALL_CONCEPTS
from ..domain.entities import Decision

REQUIRED_KEYS: dict[str, type | tuple[type, ...]] = {
    "image": str,
    "name": str,
    "split": str,
    "label": int,
    "center": str,
    "decision": str,
    "frame_trust": (int, float),
    "suspicion": (int, float),
    "verified": bool,
    "concepts": dict,
}
CONCEPT_SUBKEYS: dict[str, type | tuple[type, ...]] = {
    "value": (int, float),
    "trust": (int, float),
    "supervise": bool,
}
DECISIONS = {d.value for d in Decision}
CONCEPT_SET = set(ALL_CONCEPTS)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def validate_label(path: Path, dataset_root: Path) -> list[str]:
    """Return a list of problem strings for one label file (empty = valid)."""
    errs: list[str] = []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"]
    except OSError as e:
        return [f"unreadable: {e}"]
    if not isinstance(data, dict):
        return ["top-level JSON is not an object"]

    for key, typ in REQUIRED_KEYS.items():
        if key not in data:
            errs.append(f"missing key '{key}'")
        elif not isinstance(data[key], typ) or (typ is int and isinstance(data[key], bool)):
            errs.append(f"key '{key}' has wrong type "
                        f"({type(data[key]).__name__}, expected {_typename(typ)})")

    if isinstance(data.get("decision"), str) and data["decision"] not in DECISIONS:
        errs.append(f"decision '{data['decision']}' not in {sorted(DECISIONS)}")

    concepts = data.get("concepts")
    if isinstance(concepts, dict):
        names = set(concepts)
        for missing in sorted(CONCEPT_SET - names):
            errs.append(f"concept '{missing}' missing")
        for extra in sorted(names - CONCEPT_SET):
            errs.append(f"unknown concept '{extra}'")
        for name, cell in concepts.items():
            if not isinstance(cell, dict):
                errs.append(f"concept '{name}' is not an object")
                continue
            for sub, styp in CONCEPT_SUBKEYS.items():
                if sub not in cell:
                    errs.append(f"concept '{name}' missing '{sub}'")
                elif not isinstance(cell[sub], styp) or (styp is bool) != isinstance(cell[sub], bool):
                    errs.append(f"concept '{name}.{sub}' wrong type ({type(cell[sub]).__name__})")

    # referenced image must exist
    img_rel = data.get("image")
    if isinstance(img_rel, str) and not (dataset_root / img_rel).exists():
        errs.append(f"referenced image not found: {img_rel}")
    return errs


def _typename(typ) -> str:
    if isinstance(typ, tuple):
        return "/".join(t.__name__ for t in typ)
    return typ.__name__


def check_dataset(root: Path, strict: bool) -> dict:
    """Validate one dataset folder (containing labels/ and optionally images/)."""
    labels_dir = root / "labels"
    images_dir = root / "images"
    report = {
        "folder": str(root),
        "labels": 0, "images": 0,
        "invalid": [],          # [(stem, [errors])]
        "missing_labels": [],   # images with no label
        "orphan_labels": [],    # labels with no image
        "ok": True,
    }

    label_files = sorted(labels_dir.glob("*.json"))
    report["labels"] = len(label_files)
    label_stems = {p.stem for p in label_files}

    for lf in label_files:
        errs = validate_label(lf, root)
        if errs:
            report["invalid"].append((lf.stem, errs))

    if images_dir.is_dir():
        image_files = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        report["images"] = len(image_files)
        image_stems = {p.stem for p in image_files}
        report["missing_labels"] = sorted(image_stems - label_stems)
        report["orphan_labels"] = sorted(label_stems - image_stems)

    hard_problem = bool(report["invalid"] or report["orphan_labels"])
    if strict:
        hard_problem = hard_problem or bool(report["missing_labels"])
    report["ok"] = not hard_problem
    return report


def discover_datasets(root: Path) -> list[Path]:
    """Any folder (root itself or a child) that has a labels/ subdir is a dataset."""
    found = []
    if (root / "labels").is_dir():
        found.append(root)
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "labels").is_dir():
            found.append(child)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="check label dataset folders for format + completeness")
    ap.add_argument("roots", nargs="*", default=["out"],
                    help="dataset root(s) to scan (default: out)")
    ap.add_argument("--strict", action="store_true",
                    help="treat images-without-labels (incomplete) as a failure, not a warning")
    ap.add_argument("--json", action="store_true", help="emit a JSON summary instead of text")
    ap.add_argument("--show", type=int, default=10, help="max example items to print per problem")
    args = ap.parse_args()

    datasets: list[Path] = []
    for r in args.roots:
        root = Path(r)
        if not root.is_dir():
            print(f"!! root not found: {root}", file=sys.stderr)
            continue
        datasets.extend(discover_datasets(root))

    reports = [check_dataset(d, args.strict) for d in datasets]

    if args.json:
        print(json.dumps(reports, indent=2))
        return 0 if all(r["ok"] for r in reports) else 1

    if not reports:
        print("no dataset folders found (looked for a labels/ subdir)")
        return 1

    all_ok = True
    for r in reports:
        status = "OK " if r["ok"] else "FAIL"
        warn = "" if r["ok"] or r["missing_labels"] else ""
        n_missing = len(r["missing_labels"])
        line = (f"[{status}] {r['folder']}  labels={r['labels']} images={r['images']}"
                f"  invalid={len(r['invalid'])} missing_labels={n_missing}"
                f" orphan_labels={len(r['orphan_labels'])}")
        print(line)
        all_ok = all_ok and r["ok"]

        for stem, errs in r["invalid"][:args.show]:
            print(f"    ✗ {stem}.json: {errs[0]}" + (f"  (+{len(errs)-1} more)" if len(errs) > 1 else ""))
        if len(r["invalid"]) > args.show:
            print(f"    … +{len(r['invalid']) - args.show} more invalid label files")
        if r["orphan_labels"]:
            ex = ", ".join(r["orphan_labels"][:args.show])
            print(f"    ✗ {len(r['orphan_labels'])} label(s) with no image: {ex}"
                  + (" …" if len(r["orphan_labels"]) > args.show else ""))
        if r["missing_labels"]:
            tag = "✗" if args.strict else "⚠"
            ex = ", ".join(r["missing_labels"][:args.show])
            print(f"    {tag} {n_missing} image(s) not yet labeled: {ex}"
                  + (" …" if n_missing > args.show else ""))

    bad = [r for r in reports if not r["ok"]]
    print(f"\n{len(reports) - len(bad)}/{len(reports)} folders OK"
          + (f" · {len(bad)} with problems" if bad else ""))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
