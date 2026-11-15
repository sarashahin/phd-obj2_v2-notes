"""
test_12_niche_assembly.py  (obj1_v2, stage 2d-a part B, 27 Oct 2026; C5, C6, U4, U5, U6 added 28 Oct)

Question: does assembly v2 (niche mode, life table, window-end presence) keep field-mode
worlds exactly as they were, keep its books right, and give niche worlds whose species
live where their habitat fits? Runs small assemblies on the CPU with the real IBM.

R1 regression: field mode of v2 against v1 (md5 6df45b19), same settings and seed:
   r, C, N, species ids, fields, every v1 output and every checkpoint state identical.
L  life table and window-end presence (field and niche runs):
   L1 every species id 0..max once; L2 established count = ASM_established + founder,
   failed newcomers leave in their arrival round; L3 removals per round = ASM_pruned;
   L4 species with no removal round = final community; L5 rows per round = ASM_gamma,
   last round's maps = final N >= tau, every row has a cell, peak cells = largest row;
   L6 a species is present at every window end from arrival to removal, never after.
C  niche fields: C1 each final species' field = niche_r(H, niche_traits(id)) exactly, H
   from its seed; C2 saved niches = niche_traits; C3 bad inputs refused; C4 another
   habitat seed changes H only, not the niches (traits depend on the trait seed and id).
C5, C6 trade-off strength (assembly v2.1, after patch_environment_tradeoff.py; skipped before):
   C5 tradeoff 1 given = not given (identical worlds); C6 tradeoff 0.5: every final field =
   niche_r(H, niche_traits(id, tradeoff=0.5)) exactly.
E  ecology, niche world (description plus two checks with a control):
   E1 occupied cells where the species' own growth rate is above 0: 90% or more; control:
   the same maps scored against a shuffled habitat must fall well below (sink cells from
   immigrants allowed). E2 establishment lower than in field mode with the same seed
   (newcomers founded in one random cell fail on poor habitat).
Z  edge case: a habitat where no species can grow (cost above every peak) empties the
   community; assembly keeps going and every newcomer is recorded as failed.
--runner (after patch_run_all_rps_niche.py --apply): tiny runs of run_all_rps.before_niche.py
   and run_all_rps.py in field mode give identical training files apart from the new keys;
   a niche run saves habitat, niches, life table and presence, its tag ends _f1_s<seed>;
   wrong option sets are refused. U4 (after patch_run_all_rps_tradeoff.py): --niche-tradeoff 0.5
   is saved, its peaks are 1.5 (0.2 / width)^0.5, the tag gets k0p5 and still ends _f1_s<seed>;
   k 1 runs keep their tag; the option is refused in field mode and below 0.
   U5 (after patch_run_all_rps_contrast.py): --habitat-contrast 1 gives the same file as no
   option; 0.5 saves contrast 0.5, the habitat the species see is 0.5 x H, every final field is
   niche_r(0.5 H, niches), the tag gets hc0p5; refused in field mode and at 0 or below.
   U6 (after patch_run_all_rps_connectance.py): no option saves connectance 0.4 and strength 0.4,
   some species pairs have no link, the tag has no cn part; --connectance 0.4 gives the same file;
   --connectance 1 links every final species to all others in both directions (deg_in = deg_out
   = S - 1), saves 1.0, the tag gets _cn1p0 before _k; refused at 0, below 0 and above 1.
Ecological reading: field-mode worlds made so far stay valid; in niche worlds rarity can
come from a narrow niche or rare habitat, which the saved habitat layer can show.

Run from obj1_v2/work, after copying v2 in (v1 kept as ../new_code/assembly_stepwise_ibm_v1.py):
    USE_GPU=0 python3 ../tests/test_12_niche_assembly.py [--runner] | tee ../logs/test_12_niche_assembly_27oct.txt
"""
import contextlib, glob, hashlib, importlib.util, io, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd())
import numpy as np
import config, dispersal, environment as env
import assembly_stepwise_ibm as v2

V1 = "../new_code/assembly_stepwise_ibm_v1.py"
for k, a in enumerate(sys.argv):
    if a == "--v1": V1 = sys.argv[k + 1]
if not hasattr(env, "habitat_layer"):
    sys.exit("STOP: environment.py has no niche functions; run ../tools/patch_environment_niche.py --apply first")
if not hasattr(v2, "niche_setup"):
    sys.exit("STOP: assembly_stepwise_ibm.py in this folder is not v2")
if not os.path.exists(V1) or hashlib.md5(open(V1, "rb").read().rstrip(b"\n") + b"\n").hexdigest()[:8] != "6df45b19":
    sys.exit(f"STOP: v1 not found at {V1} with md5 6df45b19 (a missing final newline is allowed; give --v1 <path>)")
spec = importlib.util.spec_from_file_location("asm_v1", V1); v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)

G, THR = config.NUM_PATCHES_Y, 1.0
dispersal.DISPERSAL_RATE, dispersal.DISPERSAL_KERNEL, dispersal.DISPERSAL_A, dispersal.DISPERSAL_B = 1e-6, "2dt", 1.0, 3.0
NICHE = dict(opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=env.NICHE_PEAK_MAX, cost=0.1)
res = lambda c: "PASS" if c else "FAIL"
fails = []
def check(name, cond, text):
    print(f"{name} {text} {res(cond)}"); fails.extend([] if cond else [name])


def assemble(mod, *, seed, attempts, window, field=True, hseed=11, tseed=789, niche=NICHE, states=None, **kw):
    a = dict(seed=seed, max_attempts=attempts, window_steps=window, seed_size=1, max_rounds=None, F_sat=None,
             detection_threshold=THR, record_step=window, dispersal_type="propagule")
    if field:
        a.update(length_scale=5.0, var_r=0.002, seed_field=tseed)
    else:
        a.update(seed_field=tseed, habitat=env.habitat_layer(G, G, 3.0, seed=hseed), niche=niche)
    if states is not None:
        a["checkpoint_fn"] = lambda s: states.append({k: np.array(v, copy=True) for k, v in s.items()})
    a.update(kw)
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.stepwise_assembly_ibm(**a)


# R1 regression, field mode
s1, s2 = [], []
o1 = assemble(v1, seed=4101, attempts=40, window=300, states=s1)
o2 = assemble(v2, seed=4101, attempts=40, window=300, states=s2)
same = all(np.array_equal(a, b) for a, b in zip(o1[:3], o2[:3]))
same_extra = all(np.array_equal(np.asarray(o1[3][k]), np.asarray(o2[3][k])) for k in o1[3])
same_ck = len(s1) == len(s2) and all(set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a) for a, b in zip(s1, s2))
check("R1", same and same_extra and same_ck,
      f"field mode v2 = v1: r, C, N {same}; all {len(o1[3])} v1 outputs {same_extra}; {len(s1)} checkpoint states {same_ck}; "
      f"{len(o2[1])} species after {int(o2[3]['ASM_attempts'][-1])} attempts:")


def books(tag, out):
    r, C, N, x = out
    lt = {k: np.asarray(x[k]) for k in x if k.startswith("LT_")}
    ids, rin, est, rout, peak = lt["LT_id"], lt["LT_round_in"], lt["LT_established"], lt["LT_round_out"], lt["LT_peak_cells"]
    check(f"L1[{tag}]", np.array_equal(ids, np.arange(int(x["attempts_total"]) + 1)), f"ids 0..{ids.max()} once each:")
    failed = (est == 0)
    check(f"L2[{tag}]", est.sum() - 1 == int(x["ASM_established"][-1]) and np.all(rout[failed] == rin[failed]),
          f"{est.sum() - 1} newcomers kept = ASM_established {int(x['ASM_established'][-1])}; {failed.sum()} failed leave in their arrival round:")
    rounds = np.asarray(x["ASM_round"])
    removed = np.array([np.sum((est == 1) & (rout == k)) for k in rounds])
    check(f"L3[{tag}]", np.array_equal(removed, np.asarray(x["ASM_pruned"])), f"removals per round = ASM_pruned (total {removed.sum()}):")
    alive_ids = np.sort(ids[(est == 1) & (rout == -1)])
    check(f"L4[{tag}]", np.array_equal(alive_ids, np.sort(np.asarray(x["species_id"]))), f"{len(alive_ids)} species without a removal round = final community:")
    o = np.asarray(x["ASM_P_offsets"]); nrow = np.diff(o)
    last_ids, last_P = v2.unpack_presence(x, len(rounds) - 1)
    rows_ok = np.array_equal(nrow, np.asarray(x["ASM_gamma"]))
    last_ok = np.array_equal(last_ids, np.asarray(x["species_id"])) and np.array_equal(last_P, N >= THR)
    allP = np.unpackbits(np.asarray(x["ASM_P_bits"]), axis=1, count=G * G).sum(axis=1)
    pk = {}
    for i, c in zip(np.asarray(x["ASM_P_ids"]).tolist(), allP.tolist()):
        pk[i] = max(pk.get(i, 0), c)
    peak_ok = all(peak[k] == max(pk.get(int(i), 0), 1 if rin[k] == -1 else 0) for k, i in enumerate(ids))
    check(f"L5[{tag}]", rows_ok and last_ok and allP.min() >= 1 and peak_ok,
          f"rows per round = ASM_gamma {rows_ok}; last round = final N >= tau {last_ok}; smallest row {allP.min()} cells; peak cells {peak_ok}:")
    seen = {}
    for k in range(len(rounds)):
        for i in v2.unpack_presence(x, k)[0].tolist():
            seen.setdefault(i, []).append(int(rounds[k]))
    cont = True
    for k, i in enumerate(ids.tolist()):
        if est[k] == 0:
            cont &= i not in seen; continue
        first = int(rounds[0]) if rin[k] == -1 else int(rin[k])
        last = int(rounds[-1]) if rout[k] == -1 else int(rout[k]) - 1
        cont &= seen.get(i, []) == list(range(first, last + 1))
    res_rounds = np.where(rout[est == 1] >= 0, rout[est == 1] - np.maximum(rin[est == 1], 0), np.nan)
    check(f"L6[{tag}]", cont, f"present at every window end from arrival to removal, never after "
          f"(median stay of removed species {np.nanmedian(res_rounds) if np.isfinite(res_rounds).any() else float('nan'):.0f} rounds):")
    return lt


books("field", o2)

# niche run
on = assemble(v2, seed=4101, attempts=60, window=1000, field=False)
rN, CN, NN, xN = on
lt_n = books("niche", on)
H = env.habitat_layer(G, G, 3.0, seed=11)
ids = np.asarray(xN["species_id"])
fields_ok = np.array_equal(xN["habitat"], H) and all(
    np.array_equal(xN["r_field"][k], env.niche_r(H, *env.niche_traits(int(i), seed=789), 0.1)) for k, i in enumerate(ids))
check("C1", fields_ok, f"{len(ids)} final fields = niche_r(H, niche_traits(id)) exactly; H from seed 11:")
T = np.array([env.niche_traits(int(i), seed=789) for i in ids]).reshape(-1, 3)
TA = np.array([env.niche_traits(int(i), seed=789) for i in xN["LT_id"]]).reshape(-1, 3)
check("C2", np.array_equal(np.c_[xN["niche_opt"], xN["niche_w"], xN["niche_peak"]], T)
      and np.array_equal(np.c_[xN["LT_opt"], xN["LT_w"], xN["LT_peak"]], TA), f"saved niches = niche_traits for {len(ids)} final and {len(TA)} arrived species:")
bad = []
for kw in (dict(length_scale=5.0), dict(var_r=0.002), dict(seed_field=None), dict(niche={**NICHE, "extra": 1}),
           dict(habitat=np.zeros((G, G + 1))), dict(habitat=None)):
    try:
        assemble(v2, seed=1, attempts=2, window=10, field=False, **kw); bad.append(False)
    except ValueError:
        bad.append(True)
check("C3", all(bad), f"refused: field options in niche mode, no trait seed, wrong niche keys, wrong habitat shape, niche without habitat {bad}:")
H2 = env.habitat_layer(G, G, 3.0, seed=12)
check("C4", not np.allclose(H, H2) and env.niche_traits(7, seed=789) == env.niche_traits(7, seed=789),
      f"habitat seeds 11 and 12 give different layers (corr {np.corrcoef(H.ravel(), H2.ravel())[0, 1]:+.2f}); niches depend on trait seed and id only:")

# C5, C6 trade-off strength
import inspect
if "tradeoff" in inspect.signature(env.niche_traits).parameters and "tradeoff" in getattr(v2, "NICHE_OPTIONAL", ()):
    k_none = assemble(v2, seed=4103, attempts=30, window=500, field=False)
    k_one = assemble(v2, seed=4103, attempts=30, window=500, field=False, niche={**NICHE, "tradeoff": 1.0})
    same5 = all(np.array_equal(a, b) for a, b in zip(k_none[:3], k_one[:3])) and all(
        np.array_equal(np.asarray(k_none[3][k]), np.asarray(k_one[3][k])) for k in k_none[3] if k != "env_mode")
    check("C5", same5, f"tradeoff 1 given = not given: identical worlds ({len(k_none[0])} species):")
    k_half = assemble(v2, seed=4103, attempts=30, window=500, field=False, niche={**NICHE, "tradeoff": 0.5})
    xk = k_half[3]
    ok6 = all(np.array_equal(xk["r_field"][j], env.niche_r(H, *env.niche_traits(int(i), seed=789, tradeoff=0.5), 0.1))
              for j, i in enumerate(np.asarray(xk["species_id"])))
    ok6 &= np.allclose(np.asarray(xk["niche_peak"]) * np.sqrt(np.asarray(xk["niche_w"])), 1.5 * np.sqrt(0.2))
    check("C6", ok6, f"tradeoff 0.5: {len(xk['species_id'])} final fields = niche_r(H, niche_traits(id, tradeoff 0.5)) exactly; peak x width^0.5 constant:")
else:
    print("C5, C6 skipped: environment.py or assembly_stepwise_ibm.py has no trade-off strength yet")

# E ecology
occ = NN >= THR
F = np.asarray(xN["r_field"])
share = (F[occ] > 0).mean()
rng = np.random.default_rng(0)
ctrl = []
for _ in range(20):
    Hs = rng.permutation(H.ravel()).reshape(G, G)
    Fs = np.stack([env.niche_r(Hs, *env.niche_traits(int(i), seed=789), 0.1) for i in ids])
    ctrl.append((Fs[occ] > 0).mean())
fund = (F > 0).sum(axis=(1, 2)); real = occ.sum(axis=(1, 2))
check("E1", share >= 0.90 and np.mean(ctrl) < share - 0.2,
      f"{occ.sum()} occupied cells, {share:.1%} where the species' own r > 0; shuffled habitat {np.mean(ctrl):.1%} (20 shuffles); "
      f"realised / fundamental range median {np.median(real / np.maximum(fund, 1)):.2f}:")
e_field = assemble(v2, seed=4101, attempts=60, window=1000)[3]
est_f = np.asarray(e_field["LT_established"])[1:].mean(); est_n = np.asarray(xN["LT_established"])[1:].mean()
check("E2", est_n < est_f, f"newcomers kept: niche {est_n:.0%}, field {est_f:.0%} (same seed, 60 attempts):")

# Z edge case: nothing can grow
oz = assemble(v2, seed=4102, attempts=12, window=200, field=False, niche={**NICHE, "cost": 2.0})
xz = oz[3]
check("Z1", len(oz[0]) == 0 and int(np.asarray(xz["LT_established"])[1:].sum()) == 0 and int(xz["attempts_total"]) == 12
      and np.all(np.asarray(xz["ASM_gamma"]) == 0),
      f"cost 2: community empty, {int(xz['attempts_total'])} attempts recorded, all failed, gamma 0 every round:")

# runner checks
if "--runner" in sys.argv:
    before = "run_all_rps.before_niche.py"
    if not os.path.exists(before) or "# obj1_v2 niche" not in open("run_all_rps.py").read():
        sys.exit("STOP: --runner needs patch_run_all_rps_niche.py --apply (run_all_rps.before_niche.py kept)")
    base = ["--pool", "40", "--skip-psd", "--skip-ode", "--tmax", "400", "--record", "100", "--no-movie", "--ibm-frac-multi", "0.05",
            "--ibm-window-steps", "200", "--ibm-richness-cap", "None", "--save-env-field", "--ibm-record-mode", "full",
            "--ibm-F-sat", "None", "--ibm-max-rounds", "None", "--C-topk", "16", "--coocc-sample", "64", "--world-tag-extra", "t12",
            "--obs-budgets", "1,5", "--ibm-detection-threshold", "1", "--ldd-prob", "0", "--ibm-max-attempts", "30",
            "--disp-rate", "1e-6", "--disp-kernel", "2dt", "--disp-a", "1", "--disp-b", "3", "--seed-size", "1", "--sim-seed", "4201"]
    field = ["--env-length-scale", "5", "--env-var-r", "0.002", "--env-seed-field", "789"]
    niche = ["--env-mode", "niche", "--habitat-seed", "104201", "--env-seed-field", "204201"]
    env0 = dict(os.environ, USE_GPU="0")

    def run(script, args, folder):
        p = subprocess.run([sys.executable, script] + args, cwd=folder, env=env0, capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr, sorted(glob.glob(f"{folder}/results/data/*_training.npz"))

    with tempfile.TemporaryDirectory() as t:
        for f in glob.glob("*.py"):
            shutil.copy(f, t)
        ra, la, fa = run(before, base + field, t)
        for f in fa: os.rename(f, f + ".old")
        rb, lb, fb = run("run_all_rps.py", base + field, t)
        if ra or rb or len(fa) != 1 or len(fb) != 1:
            check("U1", False, f"runs failed (exit {ra}, {rb}); last lines:\n{la[-600:]}\n{lb[-600:]}")
        else:
            za, zb = np.load(fa[0] + ".old", allow_pickle=True), np.load(fb[0], allow_pickle=True)
            def eq(a, b):                                     # NaN marks "never" in SP_* arrays; NaN = NaN here
                try:
                    return np.array_equal(a, b, equal_nan=True)
                except TypeError:
                    return np.array_equal(a, b)
            same = all(eq(za[k], zb[k]) for k in za.files if k != "runtime_s")
            added = sorted(set(zb.files) - set(za.files))
            check("U1", same and set(za.files) <= set(zb.files),
                  f"field mode: {len(za.files)} keys of the old runner identical in the new file (runtime_s aside); added {added}:")
            os.remove(fb[0])
            rc, lc, fc = run("run_all_rps.py", base + niche, t)
            if rc or len(fc) != 1:
                check("U2", False, f"niche run failed (exit {rc}):\n{lc[-800:]}")
            else:
                z = np.load(fc[0], allow_pickle=True)
                Hn = env.habitat_layer(G, G, 3.0, seed=104201)
                ok = (str(z["ENV_mode"][0]) == "niche" and np.allclose(z["ENV_habitat"], Hn.astype(np.float32))
                      and int(z["HABITAT_seed"]) == 104201 and int(z["TRAIT_seed"]) == 204201
                      and np.allclose(z["NICHE_opt"], [env.niche_traits(int(i), seed=204201)[0] for i in z["species_id"]])
                      and np.allclose(z["ENV_r_field"][:, :, :], np.stack([env.niche_r(Hn, *env.niche_traits(int(i), seed=204201), 0.1) for i in z["species_id"]]), atol=1e-6)
                      and os.path.basename(fc[0]).endswith("_f1_s4201_training.npz") and "_nicheh104201ls3p0" in os.path.basename(fc[0])
                      and all(k in z.files for k in ("LT_id", "ASM_P_bits", "LT_opt")))
                check("U2", ok, f"niche run saves habitat (seed 104201), trait seed 204201, niches, final-run fields, life table and presence; file {os.path.basename(fc[0])}:")
        refused = []
        for extra in (["--env-mode", "niche", "--env-seed-field", "1"], niche + ["--env-var-r", "0.002"], niche + ["--resume", "x.npz"],
                      field + ["--habitat-seed", "5"], ["--env-mode", "niche", "--habitat-seed", "5"]):
            rc, lg, _ = run("run_all_rps.py", base + extra, t)
            refused.append(rc != 0 and ("niche mode" in lg or "need --env-mode niche" in lg))
        check("U3", all(refused), f"refused: no habitat, field option in niche mode, resume in niche mode, habitat without niche, no trait seed {refused}:")
        if "# obj1_v2 tradeoff" in open("run_all_rps.py").read():
            for f in glob.glob(f"{t}/results/data/*_training.npz"): os.remove(f)
            rk, lk, fk = run("run_all_rps.py", base + niche + ["--niche-tradeoff", "0.5"], t)
            if rk or len(fk) != 1:
                check("U4", False, f"tradeoff run failed (exit {rk}):\n{lk[-800:]}")
            else:
                z = np.load(fk[0], allow_pickle=True); name = os.path.basename(fk[0])
                ok4 = (abs(float(z["NICHE_tradeoff"]) - 0.5) < 1e-7 and "p1p5k0p5_k2dt" in name and name.endswith("_f1_s4201_training.npz")
                       and np.allclose(z["NICHE_peak"], 1.5 * (0.2 / z["NICHE_w"].astype(float)) ** 0.5, rtol=1e-5)
                       and "_nicheh104201" in name)
                bad = []
                for extra in (field + ["--niche-tradeoff", "0.5"], niche + ["--niche-tradeoff", "-1"]):
                    rc, lg, _ = run("run_all_rps.py", base + extra, t)
                    bad.append(rc != 0 and "--niche-tradeoff" in lg)
                check("U4", ok4 and all(bad), f"tradeoff 0.5 saved, peaks 1.5 (0.2 / width)^0.5, file {name}; refused in field mode and below 0 {bad}:")
        else:
            print("U4 skipped: run_all_rps.py has no --niche-tradeoff yet")
        if "# obj1_v2 contrast" in open("run_all_rps.py").read():
            for f in glob.glob(f"{t}/results/data/*_training.npz"): os.remove(f)
            kk = niche + ["--niche-tradeoff", "0.5"]
            def eq5(x, y):                                    # NaN = NaN, as in U1
                try:
                    return np.array_equal(x, y, equal_nan=True)
                except TypeError:
                    return np.array_equal(x, y)
            r1, l1, f1 = run("run_all_rps.py", base + kk, t)
            ok5 = r1 == 0 and len(f1) == 1
            if ok5:
                os.rename(f1[0], f1[0] + ".c1")
                r2, l2, f2 = run("run_all_rps.py", base + kk + ["--habitat-contrast", "1.0"], t)
                ok5 = r2 == 0 and len(f2) == 1 and os.path.basename(f2[0]) == os.path.basename(f1[0])
                if ok5:
                    za, zb = np.load(f1[0] + ".c1", allow_pickle=True), np.load(f2[0], allow_pickle=True)
                    ok5 = all(eq5(za[k], zb[k]) for k in za.files if k != "runtime_s") and set(za.files) == set(zb.files)
                for f in glob.glob(f"{t}/results/data/*_training.npz*"): os.remove(f)
            r3, l3, f3 = run("run_all_rps.py", base + kk + ["--habitat-contrast", "0.5"], t)
            if r3 or len(f3) != 1:
                check("U5", False, f"contrast run failed (exit {r3}):\n{l3[-800:]}")
            else:
                z = np.load(f3[0], allow_pickle=True); name = os.path.basename(f3[0])
                Hn = 0.5 * env.habitat_layer(G, G, 3.0, seed=104201)
                Fn = np.stack([env.niche_r(Hn, *env.niche_traits(int(i), seed=204201, tradeoff=0.5), 0.1) for i in z["species_id"]])
                ok3 = (abs(float(z["HABITAT_contrast"]) - 0.5) < 1e-7 and np.allclose(z["ENV_habitat"], Hn.astype(np.float32))
                       and np.allclose(z["ENV_r_field"], Fn, atol=1e-6) and "p1p5k0p5hc0p5_k2dt" in name
                       and name.endswith("_f1_s4201_training.npz"))
                bad = []
                for extra in (field + ["--habitat-contrast", "0.5"], kk + ["--habitat-contrast", "0"], kk + ["--habitat-contrast", "-1"]):
                    rc, lg, _ = run("run_all_rps.py", base + extra, t)
                    bad.append(rc != 0 and "--habitat-contrast" in lg)
                check("U5", ok5 and ok3 and all(bad), f"contrast 1 = no option (same file) {ok5}; contrast 0.5 saved, habitat 0.5 x H, fields exact, "
                      f"file {name}; refused in field mode, at 0 and below 0 {bad}:")
        else:
            print("U5 skipped: run_all_rps.py has no --habitat-contrast yet")
        if "# obj1_v2 connectance" in open("run_all_rps.py").read():
            for f in glob.glob(f"{t}/results/data/*_training.npz*"): os.remove(f)
            kc = niche + ["--niche-tradeoff", "0.5", "--habitat-contrast", "0.25"]
            def eq6(x, y):                                    # NaN = NaN, as in U1
                try:
                    return np.array_equal(x, y, equal_nan=True)
                except TypeError:
                    return np.array_equal(x, y)
            r1, l1, f1 = run("run_all_rps.py", base + kc, t)
            ok6 = r1 == 0 and len(f1) == 1
            if ok6:
                z0 = np.load(f1[0], allow_pickle=True); S0 = len(z0["species_id"])
                ok6 = (abs(float(z0["CONNECTANCE"]) - 0.4) < 1e-7 and abs(float(z0["INTERACTION_STRENGTH"]) - 0.4) < 1e-7
                       and "_cn" not in os.path.basename(f1[0]) and S0 > 2 and int(z0["deg_in"].min()) < S0 - 1)
                os.rename(f1[0], f1[0] + ".c4")
                r2, l2, f2 = run("run_all_rps.py", base + kc + ["--connectance", "0.4"], t)
                ok6 = ok6 and r2 == 0 and len(f2) == 1 and os.path.basename(f2[0]) == os.path.basename(f1[0])
                if ok6:
                    zb = np.load(f2[0], allow_pickle=True)
                    ok6 = all(eq6(z0[k], zb[k]) for k in z0.files if k != "runtime_s") and set(z0.files) == set(zb.files)
                for f in glob.glob(f"{t}/results/data/*_training.npz*"): os.remove(f)
            r3, l3, f3 = run("run_all_rps.py", base + kc + ["--connectance", "1"], t)
            if r3 or len(f3) != 1:
                check("U6", False, f"connectance run failed (exit {r3}):\n{l3[-800:]}")
            else:
                z = np.load(f3[0], allow_pickle=True); name = os.path.basename(f3[0]); S = len(z["species_id"])
                ok1 = (abs(float(z["CONNECTANCE"]) - 1.0) < 1e-7 and abs(float(z["INTERACTION_STRENGTH"]) - 0.4) < 1e-7
                       and S > 2 and np.all(z["deg_in"] == S - 1) and np.all(z["deg_out"] == S - 1)
                       and "p1p5k0p5hc0p25_cn1p0_k2dt" in name and name.endswith("_f1_s4201_training.npz"))
                bad = []
                for extra in (["--connectance", "0"], ["--connectance", "-0.2"], ["--connectance", "1.5"]):
                    rc, lg, _ = run("run_all_rps.py", base + kc + extra, t)
                    bad.append(rc != 0 and "--connectance" in lg)
                check("U6", ok6 and ok1 and all(bad), f"no option: connectance 0.4 saved, some pairs unlinked, no cn tag; 0.4 given = no option "
                      f"(same file) {ok6}; connectance 1: saved {float(z['CONNECTANCE']):g}, fewest links of a species {int(min(z['deg_in'].min(), z['deg_out'].min()))} "
                      f"of {S - 1} others (all needed both ways), file {name}; "
                      f"refused at 0, below 0 and above 1 {bad}:")
        else:
            print("U6 skipped: run_all_rps.py has no --connectance yet")

print(f"\n{'all passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)