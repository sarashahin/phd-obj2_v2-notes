"""
check_sweep_niche_b.py  (obj1_v2, LVMCM founding, 29 Oct 2026)

Checks of patch_sweep_stage2a_niche_b.py after the smoke run of n33 (niche_b), next to the smoke run of
n29 (niche_d, made on 29 Oct before the founding patch), which share sim seed 2303:
  - n33's command is n29's command plus --founding best, nothing else;
  - each row's file pattern finds only its own world among n21, n29 and n33;
  - rows of the older plans find the same files as with the sweep before the patch;
  - niche_ok accepts each smoke file for its own row and rejects it for the other; a file saved
    before FOUND_rule existed counts as random (accepted for n29, refused for n33);
  - the n33 smoke file is a best-founding world: entered newcomers = attempts, every one with g >= 0.
Run from obj1_v2/work, with a copy of sweep_stage2a.py from before the patch:
    python3 ../tools/sweep_stage2a.py --plan niche_b --smoke --only n33
    python3 ../tools/check_sweep_niche_b.py ../tools/sweep_stage2a.before_b.py
"""
import fnmatch, glob, sys, tempfile
import numpy as np


def load(path):
    src = open(path).read(); ns = {"__name__": "x"}
    exec(src.split("ap = argparse.ArgumentParser()")[0], ns); return ns


new, old = load("../tools/sweep_stage2a.py"), load(sys.argv[1] if len(sys.argv) > 1 else "../tools/sweep_stage2a.before_b.py")
rows = {r["name"]: r for p in ("niche_f", "niche_d", "niche_b") for r in new["plan_rows"](p, 1)}
ok = True


def show(name, good, text):
    global ok
    ok &= bool(good); print(f"{name}: {text} {'PASS' if good else 'FAIL'}")


c29, c33 = new["command"](rows["n29"]), new["command"](rows["n33"])
i = c33.index("--founding")
show("command", c33[:i] + c33[i + 2:] == c29 and c33[i:i + 2] == ["--founding", "best"], "n33 = n29 + --founding best")
f33 = sorted(glob.glob(f"results/stage2a/niche_b_smoke/{new['row_pat'](rows['n33'])}"))
f29 = sorted(glob.glob(f"results/stage2a/niche_d_smoke/{new['row_pat'](rows['n29'])}"))
if len(f33) != 1 or len(f29) != 1:
    sys.exit(f"STOP: run the smoke worlds first (found {len(f33)} n33 and {len(f29)} n29 files)")
s29 = f29[0].split("/")[-1]
names = {"n21": s29.replace("_dps1em06-0p01", ""), "n29": s29, "n33": f33[0].split("/")[-1]}
for r in ("n21", "n29", "n33"):
    hit = [k for k, v in names.items() if fnmatch.fnmatch(v, new["row_pat"](rows[r]))]
    show(r, hit == [r], f"pattern {new['row_pat'](rows[r])} matches {hit}")
for r in ("n17", "n21", "n25", "n29"):
    a = sorted(k for k, v in names.items() if k != "n33" and fnmatch.fnmatch(v, old["row_pat"](rows[r])))
    b = sorted(k for k, v in names.items() if k != "n33" and fnmatch.fnmatch(v, new["row_pat"](rows[r])))
    show(r, a == b, f"older-plan files found before {a}, now {b}")
z33, z29 = np.load(f33[0], allow_pickle=True), np.load(f29[0], allow_pickle=True)
zz = {"n33": z33, "n29": z29}
for (r, f), w in {("n33", "n33"): True, ("n29", "n29"): True, ("n33", "n29"): False, ("n29", "n33"): False}.items():
    got = new["niche_ok"](zz[f], rows[r], True)
    show("niche_ok", got == w, f"row {r}, file of {f}: {got}, expected {w}")
with tempfile.TemporaryDirectory() as t:
    d = {k: z29[k] for k in z29.files if not k.startswith("FOUND_")}
    np.savez(f"{t}/old.npz", **d); zo = np.load(f"{t}/old.npz", allow_pickle=True)
    got = (new["niche_ok"](zo, rows["n29"], True), new["niche_ok"](zo, rows["n33"], True))
    show("niche_ok", got == (True, False), f"file without FOUND_rule: n29 row {got[0]}, n33 row {got[1]}")
ent = np.asarray(z33["LT_entered"]).astype(bool)
good = (str(z33["FOUND_rule"][0]) == "best" and int(z33["attempts_total"]) == int(ent.sum())
        and (np.asarray(z33["LT_g_max"])[ent] >= 0).all() and (np.asarray(z33["LT_g_max"])[~ent] < 0).all()
        and int(z33["FOUND_tested_total"]) == len(z33["LT_id"]))
show("n33 file", good, f"best founding: {int(ent.sum())} entered of {len(z33['LT_id'])} tested = attempts {int(z33['attempts_total'])}, "
     f"entered g >= 0, failed g < 0, saturated {int(z33['FOUND_saturated'])}")
print("all passed" if ok else "FAILED"); sys.exit(0 if ok else 1)