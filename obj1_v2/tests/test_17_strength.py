"""
test_17_strength.py  (obj1_v2, competition strength, 29 Oct 2026)

Question: does --interaction-strength s do what it says in the real runner, and nothing else?
Runs tiny worlds of plan niche_s (n40 at strength 0.4, n38 at 0.7; 40 entered newcomers, 200-step windows,
400-step final run) on the CPU in a temporary copy of obj1_v2/work, so nothing is written to work/results.
  T1  --interaction-strength 0, -0.2 and 1.5 are refused before any world is made
  T2  no option = --interaction-strength 0.4: same file name and every saved array equal (run times aside)
  T3  0.7: "_is0p7_" in the file name after "_fb", INTERACTION_STRENGTH 0.7 saved, every saved competition link
      (C_topk_w without a species' own slot, connectance 1) is 0.7, and alpha_abs_mean = 0.7 (S - 1) / S
  T4  the same random draws: round 0 tests the same candidates with the same best g at 0.4 and 0.7 (an empty
      world has no competition yet), so the two worlds differ only once residents compete
  T5  negative control: a copy of run_all_rps.py without the line that sets the strength must fail T3
  T6  sweep_stage2a.py: each file passes niche_ok for its own row only, and each row's file pattern finds
      only its own file
Description: species at the end of the tiny worlds at 0.4 and 0.7 (no ecological weight at this size).
Needs patch_run_all_rps_strength.py and patch_sweep_stage2a_niche_s.py applied.

Run from obj1_v2/work:
    USE_GPU=0 python3 ../tests/test_17_strength.py | tee ../logs/test_17_strength_29oct.txt
"""
import fnmatch, glob, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

if "--interaction-strength" not in Path("run_all_rps.py").read_text():
    sys.exit("STOP: run_all_rps.py lacks --interaction-strength; apply patch_run_all_rps_strength.py first")
SET_LINE = "assembly_utils.INTERACTION_STRENGTH = float(args.interaction_strength)"


def load(path):
    src = open(path).read(); ns = {"__name__": "x"}
    exec(src.split("ap = argparse.ArgumentParser()")[0], ns); return ns


sw = load("../tools/sweep_stage2a.py")
rows = {r["name"]: r for r in sw["plan_rows"]("niche_s", 1)}
if set(rows) != {"n37", "n38", "n39", "n40"}:
    sys.exit("STOP: plan niche_s not found in sweep_stage2a.py; apply patch_sweep_stage2a_niche_s.py first")
SHRINK = {"--ibm-max-attempts": "40", "--ibm-window-steps": "200", "--tmax": "400"}
ok = True


def show(name, good, text):
    global ok
    ok &= bool(good); print(f"{name} {text}: {'PASS' if good else 'FAIL'}", flush=True)


def tiny(row, extra=()):
    cmd = sw["command"](row, smoke=True)[1:]
    for k, v in SHRINK.items():
        cmd[cmd.index(k) + 1] = v
    return cmd + list(extra)


def equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    try:
        return np.array_equal(a, b, equal_nan=True)
    except TypeError:
        return np.array_equal(a, b)


def copy_work(dst):
    for f in glob.glob("*.py"):
        shutil.copy2(f, dst)


def run(folder, cmd):
    for f in glob.glob(f"{folder}/results/data/*_training.npz"):
        os.remove(f)
    env = dict(os.environ, USE_GPU="0")
    p = subprocess.run([sys.executable] + cmd, cwd=folder, capture_output=True, text=True, env=env)
    made = glob.glob(f"{folder}/results/data/*_training.npz")
    z = dict(np.load(made[0], allow_pickle=True)) if len(made) == 1 else None
    name = os.path.basename(made[0]) if len(made) == 1 else None
    return p.returncode, p.stdout + p.stderr, name, z


def t3(name, z, s):
    if z is None:
        return False, "no file"
    w = np.asarray(z["C_topk_w"], float); S = int(z["gamma"])
    idx = np.asarray(z["C_topk_idx"])
    nz = w[idx != np.arange(idx.shape[0])[:, None]]       # a species listed as its own partner (I1) carries C_ii = 1
    am = np.asarray(z["alpha_abs_mean"], float)
    good = (name is not None and f"_fb_is{str(s).replace('.', 'p')}_k" in name
            and abs(float(z["INTERACTION_STRENGTH"]) - s) < 1e-6 and S >= 2 and nz.size > 0
            and np.allclose(nz, s, atol=1e-6) and np.allclose(am, s * (S - 1) / S, atol=1e-5))
    return good, (f"file {name}; saved {float(z['INTERACTION_STRENGTH']):.4f}; {S} species; links "
                  f"{sorted(set(np.round(nz, 4).tolist()))}; alpha_abs_mean {am.mean():.4f} (expected {s * (S - 1) / max(S, 1):.4f})")


with tempfile.TemporaryDirectory() as T, tempfile.TemporaryDirectory() as M:
    copy_work(T)
    bad = [run(T, tiny(rows["n40"], ["--interaction-strength", v])) for v in ("0", "-0.2", "1.5")]
    show("T1", all(c != 0 and "must be in (0, 1]" in o and n is None for c, o, n, _ in bad),
         "strength 0, -0.2 and 1.5 refused before any world is made")
    c0, o0, n0, z0 = run(T, tiny(rows["n40"]))
    c4, o4, n4, z4 = run(T, tiny(rows["n40"], ["--interaction-strength", "0.4"]))
    c7, o7, n7, z7 = run(T, tiny(rows["n38"]))
    if z0 is None or z4 is None or z7 is None:
        print((o0 + o4 + o7)[-3000:])
        sys.exit("STOP: a tiny world did not finish; see the output above")
    skip = {k for k in z0 if "runtime" in k.lower()}
    same = (n0 == n4 and set(z0) == set(z4)
            and all(equal(z0[k], z4[k]) for k in z0 if k not in skip))
    show("T2", same, f"no option = 0.4 given: file {n0}, {len(z0) - len(skip)} arrays equal (run times aside: {sorted(skip)})")
    good, text = t3(n7, z7, 0.7)
    show("T3", good, f"strength 0.7: {text}")
    r0a = np.asarray(z0["LT_round_in"]) == 0; r0b = np.asarray(z7["LT_round_in"]) == 0
    same0 = (np.array_equal(np.asarray(z0["LT_id"])[r0a], np.asarray(z7["LT_id"])[r0b])
             and np.array_equal(np.asarray(z0["LT_entered"])[r0a], np.asarray(z7["LT_entered"])[r0b])
             and np.allclose(np.asarray(z0["LT_g_max"])[r0a], np.asarray(z7["LT_g_max"])[r0b])
             and int(np.asarray(z0["ASM_tested"])[0]) == int(np.asarray(z7["ASM_tested"])[0]))
    show("T4", same0, f"round 0 at 0.4 and 0.7: {int(r0a.sum())} candidates tested, same ids, entry and best g")
    copy_work(M)
    src = Path(M, "run_all_rps.py").read_text()
    if src.count(SET_LINE) != 1:
        sys.exit("STOP: the set line was not found exactly once in run_all_rps.py")
    Path(M, "run_all_rps.py").write_text("\n".join(l for l in src.split("\n") if SET_LINE not in l))
    cm, om, nm, zm = run(M, tiny(rows["n38"]))
    gm, textm = t3(nm, zm, 0.7)
    show("T5", not gm, f"negative control (strength read but not set) fails T3: {textm}")
    z_by = {"n40": z0, "n38": z7}
    for rname, fname in (("n40", "n40"), ("n38", "n38"), ("n40", "n38"), ("n38", "n40")):
        class F(dict):
            files = list(z_by[fname])
        got = sw["niche_ok"](F(z_by[fname]), rows[rname], True)
        show("T6", got == (rname == fname), f"niche_ok, row {rname}, file of {fname}: {got}")
    names = {"n40": n0, "n38": n7}
    for rname in ("n40", "n38"):
        hit = [k for k, v in names.items() if fnmatch.fnmatch(v, sw["row_pat"](rows[rname]))]
        show("T6", hit == [rname], f"pattern {sw['row_pat'](rows[rname])} finds {hit}")
    print(f"description: species at the end of the tiny worlds: strength 0.4 {int(z0['gamma'])}, 0.7 {int(z7['gamma'])} "
          f"(40 entered newcomers; no ecological weight)")
print("all passed" if ok else "FAILED")
sys.exit(0 if ok else 1)