"""
check_sweep_niche_w.py  (obj1_v2, minimum niche width, 30 Oct 2026; version 2 the same day)

Checks of plan niche_w (patch_sweep_stage2a_niche_w.py and patch_sweep_stage2a_niche_w2.py) after its smoke run,
next to the niche_b smoke files that exist (n33 at least; n33-n36 share sim seeds 2303, 2304, 2307, 2308):
  C0 version 2 is applied (marks of both patches in sweep_stage2a.py);
  C1 each niche_w command is its niche_b twin's (same seed) with the --niche-w-min value changed to w, and holds
     --niche-w-min once;
  C2 every row of every older plan gets the same file pattern and, for niche rows, the same command as before
     version 1;
  C3 each niche_w row's pattern finds only its own smoke file among the niche_w and niche_b smoke files
     (n41 and n45 share seed 2303 with n33; only the width part tells them apart);
  C4 niche_ok accepts each smoke file for its own row only (this is the check that rejected the version 1 smoke
     worlds: NICHE_SET fixed w_min at 0.2);
  C5 each smoke file saved the w_min its row asked for, best founding, and for every final species: width inside
     [w_min, 1], niche = environment.niche_traits(id) with the saved settings, the same optimum as at w_min 0.2
     (only width and peak change), and growth field = the niche formula on ENV_habitat.
Version 1 of this check found the first "--niche-w-min" (the niche block's 0.2) in C1 and needed all four niche_b
smoke files; version 2 compares the commands with the value swapped and uses the niche_b smoke files that exist.
Ecological reading: the smoke worlds are tiny (120 attempts); they check the plumbing, not the ecology.

Run from obj1_v2/work, with the copy of sweep_stage2a.py made before version 1:
    python3 ../tools/sweep_stage2a.py --plan niche_w --smoke
    python3 ../tools/check_sweep_niche_w.py ../tools/sweep_stage2a.before_w.py | tee ../logs/check_sweep_niche_w_v2_30oct.txt
"""
import fnmatch, glob, os, sys
import numpy as np

sys.path.insert(0, os.getcwd())


def load(path):
    src = open(path).read(); ns = {"__name__": "x"}
    exec(src.split("ap = argparse.ArgumentParser()")[0], ns); return ns, src


(new, src_new), (old, _) = load("../tools/sweep_stage2a.py"), load(sys.argv[1] if len(sys.argv) > 1 else "../tools/sweep_stage2a.before_w.py")
ok = True


def show(name, good, text):
    global ok
    ok &= bool(good); print(f"{name}: {text} {'PASS' if good else 'FAIL'}")


show("C0", "# obj1_v2 width v2" in src_new and "# obj1_v2 width" in src_new, "versions 1 and 2 of the niche_w patch applied")
rows = {r["name"]: r for r in new["plan_rows"]("niche_w", 1)}
twins = {r["seed"]: r for r in new["plan_rows"]("niche_b", 1)}
show("rows", sorted(rows) == [f"n{k}" for k in range(41, 49)]
     and [rows[f"n{k}"]["w_min"] for k in range(41, 49)] == [0.3] * 4 + [0.4] * 4,
     "n41-n44 at w_min 0.3, n45-n48 at 0.4")
for name, r in rows.items():
    c, t = new["command"](r), list(new["command"](twins[r["seed"]]))
    j = t.index("--niche-w-min"); t[j + 1] = str(r["w_min"])
    show("C1", c == t and c.count("--niche-w-min") == 1, f"{name} = {twins[r['seed']]['name']} with --niche-w-min {r['w_min']} (once)")
plans = ["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c", "niche_f", "niche_d", "niche_b", "niche_s"]
same_pat = all(old["row_pat"](r) == new["row_pat"](r) for p in plans for reps in (1, 2) for r in new["plan_rows"](p, reps))
same_cmd = all(old["command"](r) == new["command"](r) for p in plans for reps in (1, 2) for r in new["plan_rows"](p, reps))
show("C2", same_pat and same_cmd, "every older plan's file patterns and commands unchanged")

files = {r: sorted(glob.glob(f"results/stage2a/niche_w_smoke/{new['row_pat'](rows[r])}")) for r in rows}
fb = {r["name"]: sorted(glob.glob(f"results/stage2a/niche_b_smoke/{new['row_pat'](r)}")) for r in twins.values()}
fb = {k: v for k, v in fb.items() if len(v) == 1}
if any(len(v) != 1 for v in files.values()) or "n33" not in fb:
    sys.exit(f"STOP: each niche_w row needs exactly one smoke file in results/stage2a/niche_w_smoke (found "
             f"{[len(v) for v in files.values()]}) and n33 needs its niche_b smoke file (found {sorted(fb)}). "
             f"Files in rejected/ do not count. Paste: ls results/stage2a/niche_w_smoke results/stage2a/niche_w_smoke/rejected")
names = {r: v[0].split("/")[-1] for r, v in files.items()}
names_b = {r: v[0].split("/")[-1] for r, v in fb.items()}
for r in rows:
    hit = sorted(k for k, v in names.items() if fnmatch.fnmatch(v, new["row_pat"](rows[r])))
    hit_b = sorted(k for k, v in names_b.items() if fnmatch.fnmatch(v, new["row_pat"](rows[r])))
    show("C3", hit == [r] and not hit_b, f"{r}: pattern {new['row_pat'](rows[r])} matches {hit + hit_b}")

zz = {k: np.load(v[0], allow_pickle=True) for k, v in files.items()}
zb = {k: np.load(v[0], allow_pickle=True) for k, v in fb.items()}
allrows = dict(rows, **{t["name"]: t for t in twins.values() if t["name"] in zb})
allz = dict(zz, **zb)
bad = []
for r in allrows:
    for f in allz:
        got = bool(new["niche_ok"](allz[f], allrows[r], True))
        if got != (r == f):
            bad.append(f"row {r} file {f}: {got}")
show("C4", not bad, f"niche_ok for {len(allrows)} rows x {len(allz)} files; own file only" + (f"; wrong: {bad}" if bad else ""))

import environment as env
for r in rows:
    z = zz[r]
    sv = lambda k: float(z[k])
    ids = [int(x) for x in z["species_id"]]
    kw = dict(seed=int(z["TRAIT_seed"]), opt_sd=sv("NICHE_opt_sd"), w_min=sv("NICHE_w_min"), w_max=sv("NICHE_w_max"),
              peak_max=sv("NICHE_peak_max"), tradeoff=sv("NICHE_tradeoff"))
    tr = np.array([env.niche_traits(i, **kw) for i in ids])
    tr02 = np.array([env.niche_traits(i, **dict(kw, w_min=0.2)) for i in ids])
    rf = np.stack([env.niche_r(z["ENV_habitat"], o, w, pk, sv("NICHE_cost")) for o, w, pk in tr])
    good = (abs(sv("NICHE_w_min") - rows[r]["w_min"]) < 1e-6 and str(z["FOUND_rule"][0]) == "best"
            and z["NICHE_w"].min() >= rows[r]["w_min"] - 1e-6 and z["NICHE_w"].max() <= 1.0 + 1e-6
            and np.allclose(z["NICHE_opt"], tr[:, 0], atol=1e-6) and np.allclose(z["NICHE_w"], tr[:, 1], atol=1e-6)
            and np.allclose(z["NICHE_peak"], tr[:, 2], atol=1e-6) and np.allclose(tr[:, 0], tr02[:, 0])
            and np.allclose(z["ENV_r_field"], rf, atol=1e-5))
    show("C5", good, f"{r}: w_min {sv('NICHE_w_min'):.2f}, widths {z['NICHE_w'].min():.3f}-{z['NICHE_w'].max():.3f}, "
                     f"{len(ids)} species, founding {z['FOUND_rule'][0]}, niches and fields as drawn, optima as at w_min 0.2")
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)