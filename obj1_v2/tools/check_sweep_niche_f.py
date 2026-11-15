"""
check_sweep_niche_f.py  (obj1_v2, stage 2d-a, 28 Oct 2026)

Checks of patch_sweep_stage2a_niche_f.py after the niche_f smoke run (n17, n21, n25 share sim seed 2303):
  - each row's file pattern finds only its own world among n5, n9, n13 (scale 3, connectance 0.4,
    contrast 1, 0.5, 0.25), n17 (scale 1), n21 (connectance 1) and n25 (both);
  - niche_c rows find the same files as with the sweep before the patch; niche and niche_k patterns
    are unchanged;
  - niche_ok accepts each smoke file for its own row and rejects it for every other row; a file saved
    before CONNECTANCE existed counts as connectance 0.4.
Run from obj1_v2/work, with a copy of sweep_stage2a.py from before the patch:
    python3 ../tools/sweep_stage2a.py --plan niche_f --smoke --only n17 n21 n25
    python3 ../tools/check_sweep_niche_f.py ../tools/sweep_stage2a.before_f.py
"""
import fnmatch, glob, sys, tempfile, numpy as np
def load(path):
    src = open(path).read(); ns = {"__name__": "x"}
    exec(src.split("ap = argparse.ArgumentParser()")[0], ns); return ns
new, old = load("../tools/sweep_stage2a.py"), load(sys.argv[1] if len(sys.argv) > 1 else "../tools/sweep_stage2a.before_f.py")
rows = {r["name"]: r for p in ("niche_k", "niche_c", "niche_f") for r in new["plan_rows"](p, 1)}
files = {r: sorted(glob.glob(f"results/stage2a/niche_f_smoke/{new['row_pat'](rows[r])}")) for r in ("n17", "n21", "n25")}
f17 = files["n17"][0]; stem = f17.split("/")[-1]
names = {"n5": stem.replace("ls1p0", "ls3p0").replace("hc0p25", ""), "n9": stem.replace("ls1p0", "ls3p0").replace("hc0p25", "hc0p5"),
         "n13": stem.replace("ls1p0", "ls3p0"), "n17": stem, "n21": files["n21"][0].split("/")[-1], "n25": files["n25"][0].split("/")[-1]}
ok = True
for r in ("n9", "n13", "n17", "n21", "n25"):
    hit = [k for k, v in names.items() if fnmatch.fnmatch(v, new["row_pat"](rows[r]))]
    good = hit == [r]; ok &= good; print(f"{r}: pattern {new['row_pat'](rows[r])} matches {hit} {'PASS' if good else 'FAIL'}")
for r in ("n9", "n13"):
    a = sorted(k for k, v in names.items() if k in ("n5", "n9", "n13") and fnmatch.fnmatch(v, old["row_pat"](rows[r])))
    b = sorted(k for k, v in names.items() if k in ("n5", "n9", "n13") and fnmatch.fnmatch(v, new["row_pat"](rows[r])))
    good = a == b == [r]; ok &= good; print(f"{r}: niche_c files found before {a}, now {b} {'PASS' if good else 'FAIL'}")
for r in ("n5", "n1"):
    rr = {**{x["name"]: x for x in new["plan_rows"]("niche", 1)}, **rows}[r]
    good = new["row_pat"](rr) == old["row_pat"](rr); ok &= good; print(f"{r}: pattern unchanged {'PASS' if good else 'FAIL'}")
z = {r: np.load(files[r][0], allow_pickle=True) for r in files}
want = {("n17", "n17"): True, ("n21", "n21"): True, ("n25", "n25"): True, ("n17", "n25"): False, ("n25", "n17"): False,
        ("n21", "n25"): False, ("n25", "n21"): False, ("n21", "n17"): False, ("n13", "n17"): False, ("n13", "n21"): False}
for (r, f), w in want.items():
    got = new["niche_ok"](z[f], rows[r], True); good = got == w; ok &= good
    print(f"niche_ok(row {r}, file of {f}) = {got}, expected {w} {'PASS' if good else 'FAIL'}")
with tempfile.TemporaryDirectory() as t:                 # an old-style n13 file: no CONNECTANCE key, scale 3
    d = {k: z["n17"][k] for k in z["n17"].files if k not in ("CONNECTANCE", "INTERACTION_STRENGTH")}
    d["HABITAT_ls"] = np.float32(3.0); np.savez(f"{t}/old.npz", **d); zo = np.load(f"{t}/old.npz", allow_pickle=True)
    got = (new["niche_ok"](zo, rows["n13"], True), new["niche_ok"](zo, rows["n21"], True), new["niche_ok"](zo, rows["n17"], True))
    good = got == (True, False, False); ok &= good
    print(f"file without CONNECTANCE (counts as 0.4), scale 3: n13 row {got[0]}, n21 row {got[1]}, n17 row {got[2]} {'PASS' if good else 'FAIL'}")
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)