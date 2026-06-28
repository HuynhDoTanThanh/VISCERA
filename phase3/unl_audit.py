"""Sample unlabeled concept JSONs at scale: labelability (supervise%, trust, value dist) per concept."""
import os, sys, glob, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_system.domain.concept_schema import CONCEPTS, BY_NAME

NAMES = [c.name for c in CONCEPTS]
ROLE = {c.name: c.role for c in CONCEPTS}
random.seed(0)

dirs = sorted(glob.glob("out/[0-9]*/labels"))
files = []
for d in dirs:
    fs = glob.glob(os.path.join(d, "*.json"))
    files.extend(fs)
print(f"# {len(dirs)} unlabeled dirs, {len(files)} total json")
N = 6000
samp = random.sample(files, min(N, len(files)))

val = {n: [] for n in NAMES}
trust = {n: [] for n in NAMES}
sup = {n: [] for n in NAMES}
susp = []
center_count = {}
decisions = {}
for fp in samp:
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    cen = d.get("center", "?"); center_count[cen] = center_count.get(cen, 0)+1
    dec = d.get("decision", "?"); decisions[dec] = decisions.get(dec, 0)+1
    susp.append(d.get("suspicion", np.nan))
    cj = d.get("concepts", {})
    for n in NAMES:
        c = cj.get(n)
        if c is None:
            continue
        val[n].append(float(c.get("value", 0.0)))
        trust[n].append(float(c.get("trust", 0.0)))
        sup[n].append(1.0 if c.get("supervise", False) else 0.0)

print(f"# sampled {len(samp)} | centers={center_count}")
print(f"# decisions={decisions}")
s = np.array(susp, float); s = s[~np.isnan(s)]
print(f"# suspicion: mean={s.mean():.3f} p50={np.percentile(s,50):.3f} p90={np.percentile(s,90):.3f} p99={np.percentile(s,99):.3f} frac>0.5={np.mean(s>0.5):.4f}")

print(f"\n{'concept':26s} {'role':13s} {'%sup':>6s} {'trust':>6s} {'meanV':>6s} {'%V>0':>6s} {'%V>0.5':>7s} {'p90V':>6s}")
for n in NAMES:
    v = np.array(val[n]); t = np.array(trust[n]); sp = np.array(sup[n])
    if len(v)==0:
        print(f"{n:26s} {ROLE[n]:13s} (none)"); continue
    print(f"{n:26s} {ROLE[n]:13s} {sp.mean()*100:6.1f} {t.mean():6.2f} {v.mean():6.3f} {np.mean(v>0)*100:6.1f} {np.mean(v>0.5)*100:7.1f} {np.percentile(v,90):6.2f}")
