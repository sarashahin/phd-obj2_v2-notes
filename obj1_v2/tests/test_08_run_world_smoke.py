"""
test_08_run_world_smoke.py  (obj1_v2, written 24 Oct 2026; S9 and the founding run added
the same day with patch_run_all_rps_seed_size.py)

Question: after the Step 1 fixes, does a whole run of run_all_rps.py (assembly, final
run, export) do what the re-simulation needs, and do the results move the way the
ecology says they should?

Runs five tiny communities (120 invasion attempts, 300-step windows, 1,000-step final
run, 20 x 20 grid) into work/results/data, with the world tag "smoke":
  a  sim seed 101, d = 2e-8, four-neighbour kernel      b  the same again
  c  sim seed 102, otherwise as a                       d  sim seed 101, d = 1e-7
  e  as d with the 2Dt kernel (a = 1, b = 1.75)
  f  as d, founded in 1 cell (--seed-size 1); only when run_all_rps.py has --seed-size
Checks (PASS/FAIL = requirement; numbers in brackets are for reading):
  S1 tags carry kernel and seed            S2 same seed gives the same community
  S3 different seeds give unrelated founders and early ranges
  S4 every species' final-run field is the field it got on arrival (rebuilt from its id)
  S5 seeds, kernel and ids are saved       S6 richness bookkeeping incl. pruned per round
  S7 ecology: a higher d gives larger ranges
  S8 ecology: at the same d the fat-tailed kernel gives more outposts (occupied cells
     with no same-species cell within 2 cells). d = 1e-7 is used because at 1e-5 ranges
     fill most of the grid and no outposts can exist (found when this test was built).
  S9 ecology: founding in 1 cell against 5 (runs f and d) gives one patch per species at
     the end of assembly instead of several, and fewer newcomers establish, since a single
     founder dies out more often. Tags carry _f1 and _f5, and seed_size is saved.
     Stand-in runs 24 Oct (seeds 101, 102): median patches 1 against 3, establishment
     0.71-0.75 against 0.97-0.99.
Run from obj1_v2/work after the patch:  USE_GPU=0 python3 ../tests/test_08_run_world_smoke.py
Takes about 3 minutes on a CPU. It writes only into work/results. Each run's file is found by
its time stamp, so the test can be run again in the same folder (the first version failed then).
"""
import os, re, sys, glob, time, shutil, subprocess
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy import ndimage
import assembly_stepwise_ibm as asm

res = lambda c: "PASS" if c else "FAIL"
base = [sys.executable, "run_all_rps.py", "--pool", "100", "--skip-psd", "--skip-ode", "--tmax", "1000", "--record", "200",
        "--no-movie", "--ibm-frac-multi", "0.05", "--ibm-window-steps", "300", "--ibm-max-attempts", "120",
        "--ibm-richness-cap", "None", "--save-env-field", "--ibm-record-mode", "full", "--ibm-F-sat", "None",
        "--ibm-max-rounds", "None", "--save-every-rounds", "500", "--save-every-seconds", "600", "--C-topk", "16",
        "--coocc-sample", "256", "--world-tag-extra", "smoke", "--fp16-time-series", "--obs-budgets", "1,5,10",
        "--env-length-scale", "5", "--env-var-r", "0.002", "--env-seed-field", "789", "--ibm-detection-threshold", "1",
        "--ldd-prob", "0"]
runs = {"a": ["--disp-rate", "2e-8", "--sim-seed", "101"], "b": ["--disp-rate", "2e-8", "--sim-seed", "101"],
        "c": ["--disp-rate", "2e-8", "--sim-seed", "102"], "d": ["--disp-rate", "1e-7", "--sim-seed", "101"],
        "e": ["--disp-rate", "1e-7", "--sim-seed", "101", "--disp-kernel", "2dt", "--disp-a", "1", "--disp-b", "1.75"]}
HAS_F = "--seed-size" in open("run_all_rps.py").read()
if HAS_F:
    runs["f"] = ["--disp-rate", "1e-7", "--sim-seed", "101", "--seed-size", "1"]
os.makedirs("results/smoke", exist_ok=True)
files, tags = {}, {}
for name, extra in runs.items():
    t0 = time.time()
    p = subprocess.run(base + extra, capture_output=True, text=True, env=dict(os.environ, USE_GPU=os.environ.get("USE_GPU", "0")))
    if p.returncode != 0:
        sys.exit(f"run {name} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    new = [f for f in glob.glob("results/data/*smoke*_training.npz") if os.path.getmtime(f) >= t0 - 1]   # written by this run
    if not new: sys.exit(f"run {name}: no training file written")
    made = max(new, key=os.path.getmtime)
    tags[name] = os.path.basename(made)
    files[name] = shutil.copy(made, f"results/smoke/run_{name}.npz")
    print(f"run {name}: {tags[name]}")
Z = {k: np.load(v, allow_pickle=True) for k, v in files.items()}

F = r"(_f\d+)?"           # founding size in the tag once patch_run_all_rps_seed_size.py is in
print(f"S1 tags carry kernel and seed: {res(bool(re.search('_knn4' + F + '_s101', tags['a'])) and '_s102' in tags['c'] and bool(re.search('_k2dta1p0b1p75' + F + '_s101', tags['e'])))}")
same = all(np.array_equal(Z['a'][k], Z['b'][k]) for k in ("P_last_final", "ENV_r_field", "C_topk_idx", "P_init_assembly"))
print(f"S2 same seed, same community: {res(same)}")
fa, fc = Z["a"]["P_init_assembly"][0], Z["c"]["P_init_assembly"][0]          # the founder species at the end of assembly
n = min(len(Z["a"]["P_last_final"]), len(Z["c"]["P_last_final"]), 50)
Pa, Pc = Z["a"]["P_init_assembly"][:n].reshape(n, -1).astype(bool), Z["c"]["P_init_assembly"][:n].reshape(n, -1).astype(bool)
jac = np.mean([(x & y).sum() / max(1, (x | y).sum()) for x, y in zip(Pa, Pc)])
print(f"S3 seeds 101 and 102: founder species maps identical {np.array_equal(fa, fc)}; mean overlap of the first {n} species' ranges "
      f"{jac:.3f} (about 0 if unrelated) {res(not np.array_equal(fa, fc) and jac < 0.1)}")
ok4 = []
for k in ("a", "c", "e"):
    z = Z[k]; ids = z["species_id"]
    rebuilt = np.stack([asm.species_field(int(i), length_scale=float(z["ENV_length_scale"]), var_r=float(z["ENV_var_r"]),
                                          seed_field=int(z["ENV_seed_field"])) for i in ids])
    ok4.append(np.abs(rebuilt - z["ENV_r_field"]).max() < 1e-5)
print(f"S4 final-run field of every species = its own field from arrival (rebuilt from species id): {res(all(ok4))}")
z = Z["e"]
print(f"S5 saved: seed {int(z['seed'])}, SIM_SEED {int(z['SIM_SEED'])}, kernel {z['DISPERSAL_KERNEL'][0]} a {float(z['DISPERSAL_A'])} "
      f"b {float(z['DISPERSAL_B'])}, d {float(z['DISPERSAL_RATE']):g}, ids {len(z['species_id'])} unique {len(set(z['species_id'].tolist()))} "
      f"{res(int(z['seed']) == 101 and str(z['DISPERSAL_KERNEL'][0]) == '2dt' and len(set(z['species_id'].tolist())) == len(z['species_id']))}")
ok6 = []
for k in ("a", "e"):
    z = Z[k]; g = z["ASM_gamma"].astype(int); est = np.diff(np.r_[0, z["ASM_established"]]); pr = z["ASM_pruned"]
    ok6.append(np.array_equal(g, np.r_[1, g[:-1]] + est - pr) and int(z["attempts_total"]) == int(z["ASM_attempts"][-1]))
print(f"S6 richness = previous + established - pruned every round, and no dropped last window: {res(all(ok6))}")
R = {k: Z[k]["P_last_final"].reshape(len(Z[k]["P_last_final"]), -1).sum(1) for k in Z}
ma, md = np.mean(R["a"][R["a"] > 0]), np.mean(R["d"][R["d"] > 0])
print(f"S7 ecology, d = 1e-7 against 2e-8: mean range {md:.1f} against {ma:.1f} cells {res(md > ma)}")
def outposts(P):
    P = P.astype(bool); S = len(P); out = []
    for s in range(S):
        occ = np.argwhere(P[s])
        if len(occ) < 2: continue
        dy = np.abs(occ[:, None, 0] - occ[None, :, 0]); dx = np.abs(occ[:, None, 1] - occ[None, :, 1])
        d = np.maximum(np.minimum(dy, 20 - dy), np.minimum(dx, 20 - dx)); np.fill_diagonal(d, 99)
        out.append(np.mean(d.min(1) > 2))
    return np.mean(out), len(out)
od, nd = outposts(Z["d"]["P_last_final"]); oe, ne = outposts(Z["e"]["P_last_final"])
print(f"S8 ecology, same d: share of occupied cells with no same-species cell within 2 cells, 2Dt {oe:.3f} ({ne} species) "
      f"against four-neighbour {od:.3f} ({nd} species) {res(oe > od)}")
if HAS_F:
    S4c = ndimage.generate_binary_structure(2, 1); row = {}
    for k in ("d", "f"):
        z = Z[k]; P0 = z["P_init_assembly"].astype(bool); R0 = P0.reshape(len(P0), -1).sum(1)
        pt = [ndimage.label(P0[s], S4c)[1] for s in range(len(P0)) if R0[s] >= 2]
        row[k] = (float(np.median(pt)), float(z["ASM_established"][-1] / z["ASM_attempts"][-1]), int(z["seed_size"]))
    good = ("_f1_s101" in tags["f"] and "_f5_s101" in tags["d"] and row["f"][2] == 1 and row["d"][2] == 5
            and row["f"][0] < row["d"][0] and row["f"][1] < row["d"][1])
    print(f"S9 ecology, founded in 1 cell against 5 (d = 1e-7): median patches per species at the end of assembly "
          f"{row['f'][0]:.1f} against {row['d'][0]:.1f}, share of newcomers established {row['f'][1]:.2f} against {row['d'][1]:.2f}, "
          f"saved seed_size {row['f'][2]} and {row['d'][2]} {res(good)}")
else:
    print("S9 skipped: run_all_rps.py has no --seed-size yet (patch_run_all_rps_seed_size.py not applied)")
for k in ("a", "e"):
    z = Z[k]; lost = ((z["IBM_ext_step"] >= 0) & (R[k] == 0)).mean()
    print(f"   run {k}: {len(R[k])} species, lost in the final run {lost:.1%}, mean range {np.mean(R[k][R[k] > 0]):.1f} cells")