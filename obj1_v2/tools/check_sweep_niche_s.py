"""
check_sweep_niche_s.py  (obj1_v2, competition strength scan, 29 Oct 2026)

Checks of patch_sweep_stage2a_niche_s.py after the smoke run of plan niche_s, next to the smoke file of n33
(niche_b, made on 29 Oct), which shares sim seed 2303:
  - n40's command is n33's with 2,000 attempts; n37-n39 add --interaction-strength 0.6, 0.7, 0.8, nothing else;
  - every row of every older plan gets the same file pattern as before the patch;
  - each niche_s row's file pattern finds only its own smoke file among n33 and n37-n40;
  - niche_ok accepts each smoke file for its own row only (n33's file also passes the n40 row: same setting,
    other folder); a file without INTERACTION_STRENGTH counts as 0.4;
  - each smoke file saved the strength its row asked for.
Run from obj1_v2/work, with a copy of sweep_stage2a.py from before the patch:
    python3 ../tools/sweep_stage2a.py --plan niche_s --smoke
    python3 ../tools/check_sweep_niche_s.py ../tools/sweep_stage2a.before_s.py
"""
import fnmatch, glob, sys, tempfile
import numpy as np


def load(path):
    src = open(path).read(); ns = {"__name__": "x"}
    exec(src.split("ap = argparse.ArgumentParser()")[0], ns); return ns


new, old = load("../tools/sweep_stage2a.py"), load(sys.argv[1] if len(sys.argv) > 1 else "../tools/sweep_stage2a.before_s.py")
ok = True


def show(name, good, text):
    global ok
    ok &= bool(good); print(f"{name}: {text} {'PASS' if good else 'FAIL'}")


rows = {r["name"]: r for r in new["plan_rows"]("niche_s", 1)}
n33 = [r for r in new["plan_rows"]("niche_b", 1) if r["name"] == "n33"][0]
c33 = new["command"](n33)
c33[c33.index("--ibm-max-attempts") + 1] = "2000"
show("command", new["command"](rows["n40"]) == c33, "n40 = n33 with 2,000 attempts")
for name, s in (("n37", 0.6), ("n38", 0.7), ("n39", 0.8)):
    c = new["command"](rows[name]); i = c.index("--interaction-strength")
    show("command", c[:i] + c[i + 2:] == c33 and c[i + 1] == str(s), f"{name} = n40 + --interaction-strength {s}")
plans = ["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c", "niche_f", "niche_d", "niche_b"]
same = all(old["row_pat"](r) == new["row_pat"](r) for p in plans for reps in (1, 2) for r in new["plan_rows"](p, reps)
           if "row_pat" in old)
show("older plans", same, "every row's file pattern unchanged")
files = {r: sorted(glob.glob(f"results/stage2a/niche_s_smoke/{new['row_pat'](rows[r])}")) for r in rows}
f33 = sorted(glob.glob(f"results/stage2a/niche_b_smoke/{new['row_pat'](n33)}"))
if any(len(v) != 1 for v in files.values()) or len(f33) != 1:
    sys.exit(f"STOP: run the smoke worlds first (found {[len(v) for v in files.values()]} niche_s and {len(f33)} n33 files)")
names = {r: v[0].split("/")[-1] for r, v in files.items()}
names["n33"] = f33[0].split("/")[-1]
for r in rows:
    hit = sorted(k for k, v in names.items() if fnmatch.fnmatch(v, new["row_pat"](rows[r])))
    want = sorted([r, "n33"]) if r == "n40" else [r]
    show(r, hit == want, f"pattern {new['row_pat'](rows[r])} matches {hit}" + (" (n33 lives in niche_b)" if r == "n40" else ""))
zz = {k: np.load(v[0], allow_pickle=True) for k, v in files.items()}
zz["n33"] = np.load(f33[0], allow_pickle=True)
for r in rows:
    for f in zz:
        want = (r == f) or (r == "n40" and f == "n33")
        got = new["niche_ok"](zz[f], rows[r], True)
        show("niche_ok", got == want, f"row {r}, file of {f}: {got}, expected {want}")
with tempfile.TemporaryDirectory() as t:
    z = zz["n40"]
    np.savez(f"{t}/old.npz", **{k: z[k] for k in z.files if k != "INTERACTION_STRENGTH"})
    zo = np.load(f"{t}/old.npz", allow_pickle=True)
    got = (new["niche_ok"](zo, rows["n40"], True), new["niche_ok"](zo, rows["n38"], True))
    show("niche_ok", got == (True, False), f"file without INTERACTION_STRENGTH: n40 row {got[0]}, n38 row {got[1]}")
for r in rows:
    s = float(zz[r]["INTERACTION_STRENGTH"])
    show(f"{r} file", abs(s - rows[r].get("strength", 0.4)) < 1e-6 and str(zz[r]["FOUND_rule"][0]) == "best",
         f"saved strength {s:.4f}, founding {zz[r]['FOUND_rule'][0]}, {int(zz[r]['attempts_total'])} entered, "
         f"saturated {int(zz[r]['FOUND_saturated'])}, {int(zz[r]['gamma'])} species")
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)