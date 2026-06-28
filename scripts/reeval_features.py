"""Re-evaluate the concept/feature list under each cached labeler (flash3 / flash35 / pro-agent).

Uses the votes already cached by scripts/compare_models.py (same 5 few-shot anchors, same train
frames) — no new API calls. For each model it runs the full reliability study (agreement × MI ×
AUROC) and reports the CORE discriminative set, so you can see which features the stronger model
rescues or drops vs the current flash labeler.

    .venv/bin/python -m scripts.reeval_features
"""
from __future__ import annotations
import json
from pathlib import Path

from cf import config
from cf.aggregate import feature_matrix
from cf.concept_schema import BY_NAME
from cf.confirm import load_split
from cf.extract import _key
from cf.reliability import study

MODELS = ["flash3", "flash35", "proagent"]


def fewshot_sig():
    a = json.loads((config.REPORT_DIR / "fewshot_anchors.json").read_text())
    return ",".join(Path(x["path"]).name for x in a)


def model_records(frames, model_key, sig):
    """{path: {model_key: votes}} for frames that have a cache under this (model, anchors)."""
    recs, ok = {}, []
    for fr in frames:
        kf = _key(fr["path"], model_key, sig)
        if kf.exists():
            recs[fr["path"]] = {model_key: json.loads(kf.read_text()).get("votes", [])}
            ok.append(fr)
    return recs, ok


def main():
    sig = fewshot_sig()
    _, frames = load_split(str(config.CACHE_DIR / "index_train.json"))

    results = {}
    for mk in MODELS:
        recs, ok = model_records(frames, mk, sig)
        if not ok:
            print(f"[reeval] {mk}: no cached frames — skipping")
            continue
        Xc, Xr, Xm, y, names = feature_matrix(recs, ok)
        rows = study(Xc, Xr, Xm, y, names)
        results[mk] = {"n": len(ok), "pos": int(sum(f["label"] for f in ok)),
                       "rows": {r["concept"]: r for r in rows},
                       "core": [r["concept"] for r in rows if r["core"]]}
        print(f"[reeval] {mk} ({config.COMPARE_MODELS[mk]}): {len(ok)} frames, "
              f"{results[mk]['pos']} neo, CORE={len(results[mk]['core'])}")

    if not results:
        print("No cached model votes found. Run scripts.compare_models first.")
        return

    # side-by-side discriminative table: agreement & AUROC & core flag per model
    disc = [c.name for c in BY_NAME.values() if c.role == "discriminative"]
    hdr = f"\n{'concept':22} " + "  ".join(f"{m:>16}" for m in results)
    print(hdr)
    print(f"{'':22} " + "  ".join(f"{'agr/AUROC/core':>16}" for _ in results))
    print("-" * len(hdr))
    # order by best model's MI
    order_key = "proagent" if "proagent" in results else list(results)[0]
    disc.sort(key=lambda n: -results[order_key]["rows"][n]["mi"])
    for n in disc:
        cells = []
        for m in results:
            r = results[m]["rows"][n]
            flag = "Y" if r["core"] else " "
            cells.append(f"{r['agreement']:.2f}/{r['auroc']:.2f}/{flag:>1}".rjust(16))
        print(f"{n:22} " + "  ".join(cells))

    # core-set diffs vs flash3
    base = set(results.get("flash3", {}).get("core", []))
    print("\n## CORE set per model")
    for m in results:
        print(f"  {m:10} ({len(results[m]['core'])}): {', '.join(results[m]['core'])}")
    if "flash3" in results:
        for m in results:
            if m == "flash3":
                continue
            cur = set(results[m]["core"])
            print(f"\n  {m} vs flash3:  +rescued {sorted(cur - base)}   -dropped {sorted(base - cur)}")

    (config.REPORT_DIR / "feature_reeval.json").write_text(json.dumps(
        {m: {"n": results[m]["n"], "pos": results[m]["pos"], "core": results[m]["core"],
             "rows": results[m]["rows"]} for m in results}, indent=2))
    print(f"\n[reeval] wrote {config.REPORT_DIR/'feature_reeval.json'}")


if __name__ == "__main__":
    main()
