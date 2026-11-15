"""
check_sweep_niche_d.py  (obj1_v2, stage 2d-b, 29 Oct 2026)

Checks of patch_sweep_stage2a_niche_d.py after the smoke runs of n29 (niche_d) and n21 (niche_f),
which share sim seed 2303:
  - n29's command is n21's command plus --disp-per-species 1e-06 0.01, nothing else;
  - each row's file pattern finds only its own world among n13, n17, n21, n25 and n29;
  - rows of the older plans find the same files as with the sweep before the patch;
  - niche_ok accepts each smoke file for its own row and rejects it for the other; a file saved
    before DISP_per_species existed counts as shared d (accepted for n21, refused for n29).
Run from obj1_v2/work, with a copy of sweep_stage2a.py from before the patch:
    python3 ../tools/sweep_stage2a.py --plan niche_d --smoke --only n29
    python3 ../tools/sweep_stage2a.py --plan niche_f --smoke --only n21
    python3 ../tools/check_sweep_niche_d.py ../tools/sweep_stage2a.before_d.py
"""
import fnmatch, glob, sys, tempfile
import numpy as np


def load(path):
    src = open(path).read(); ns = {"__name__": "x"}
    exec(src.split("ap = argparse.ArgumentParser()")[0], ns); return ns


new, old = load("../tools/sweep_stage2a.py"), load(sys.argv[1] if len(sys.argv) > 1 else "../tools/sweep_stage2a.before_d.py")
rows = {r["name"]: r for p in ("niche_c", "niche_f", "niche_d") for r in new["plan_rows"](p, 1)}
ok = True


def show(name, good, text):
    global ok
    ok &= bool(good); print(f"{name}: {text} {'PASS' if good else 'FAIL'}")


c21, c29 = new["command"](rows["n21"]), new["command"](rows["n29"])
i = c29.index("--disp-per-species")
show("command", c29[:i] + c29[i + 3:] == c21 and c29[i:i + 3] == ["--disp-per-species", "1e-06", "0.01"],
     "n29 = n21 + --disp-per-species 1e-06 0.01")
f29 = sorted(glob.glob(f"results/stage2a/niche_d_smoke/{new['row_pat'](rows['n29'])}"))
f21 = sorted(glob.glob(f"results/stage2a/niche_f_smoke/{new['row_pat'](rows['n21'])}"))
if len(f29) != 1 or len(f21) != 1:
    sys.exit(f"STOP: run the two smoke worlds first (found {len(f29)} n29 and {len(f21)} n21 files)")
s21 = f21[0].split("/")[-1]
names = {"n13": s21.replace("_cn1p0", ""), "n17": s21.replace("_cn1p0", "").replace("ls3p0o", "ls1p0o"),
         "n21": s21, "n25": s21.replace("ls3p0o", "ls1p0o"), "n29": f29[0].split("/")[-1]}
for r in ("n13", "n17", "n21", "n25", "n29"):
    hit = [k for k, v in names.items() if fnmatch.fnmatch(v, new["row_pat"](rows[r]))]
    show(r, hit == [r], f"pattern {new['row_pat'](rows[r])} matches {hit}")
for r in ("n9", "n13", "n17", "n21", "n25"):
    a = sorted(k for k, v in names.items() if k != "n29" and fnmatch.fnmatch(v, old["row_pat"](rows[r])))
    b = sorted(k for k, v in names.items() if k != "n29" and fnmatch.fnmatch(v, new["row_pat"](rows[r])))
    show(r, a == b, f"older-plan files found before {a}, now {b}")
z29, z21 = np.load(f29[0], allow_pickle=True), np.load(f21[0], allow_pickle=True)
want = {("n29", "n29"): True, ("n21", "n21"): True, ("n29", "n21"): False, ("n21", "n29"): False, ("n25", "n29"): False}
zz = {"n29": z29, "n21": z21}
for (r, f), w in want.items():
    got = new["niche_ok"](zz[f], rows[r], True)
    show("niche_ok", got == w, f"row {r}, file of {f}: {got}, expected {w}")
with tempfile.TemporaryDirectory() as t:
    d = {k: z21[k] for k in z21.files if not k.startswith("DISP_")}
    np.savez(f"{t}/old.npz", **d); zo = np.load(f"{t}/old.npz", allow_pickle=True)
    got = (new["niche_ok"](zo, rows["n21"], True), new["niche_ok"](zo, rows["n29"], True))
    show("niche_ok", got == (True, False), f"file without DISP_per_species: n21 row {got[0]}, n29 row {got[1]}")
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)