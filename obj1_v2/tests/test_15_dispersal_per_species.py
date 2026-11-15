"""
test_15_dispersal_per_species.py  (obj1_v2, stage 2d-b, 29 Oct 2026)

Question: does the per-species dispersal rate (patch_dispersal_per_species.py, assembly v2.2,
patch_run_all_rps_disp.py) do exactly what it says, leave every other world unchanged, and give
the ecological effect it is meant to give?

P  dispersal.py
   P1 rates off: compute_dispersal is identical (bit for bit) to the v2 code path, also after
      rates were set and switched off again (kernels nn4, 2Dt, Gaussian; long-distance share 0, 0.2).
   P2 every species at the shared d: same flux as the shared path (relative difference 1e-12 or less).
   P3 rates of 1e-7 to 1e-1: each species' flux equals d_i x (its own kernel flux), computed
      species by species; total emitted = d_i x B_i at long-distance share 0 and 0.2.
   P4 a rates vector of the wrong length and negative rates are refused.
   P5 species_dispersal_rate: the same value on every call and in any order; inside [d_min, d_max];
      log10 d uniform (Kolmogorov-Smirnov p > 0.01 and mean within 0.05 of the midpoint, 5,000 ids);
      d_min = d_max gives d_min; bad ranges refused; the niche traits of the same ids and the global
      NumPy random state are unchanged by the draws.
A  assembly v2.2 (small niche assemblies on the CPU)
   A1 without disp: every returned array equals v2.1 (new_code/assembly_stepwise_ibm_v2_1.py).
   A2 with disp: in every window the rates set equal species_dispersal_rate of that window's species
      list, in its order (the list rebuilt from the checkpoint ids and the life table); the rates are
      switched off after every window; disp_rate and LT_d equal the rates of species_id and LT_id;
      after an error inside a window the rates are off.
   A3 with disp: niches of every arrived species equal those of the run without disp; the first
      round's newcomers (ids) are the same (the assembly random stream is untouched).
E  ecology, with a control: two species with no competition between them, the same 4 x 4 source
   block (r 0.3) and sinks at g -0.1 elsewhere, 2Dt kernel, 2,000 steps. d 1e-2 against 1e-6: the
   high-d species holds more cells (time-mean of 1 or more individuals) and its sinks next to the
   source hold about what test_14 found for one species (72 at distance 1, within 15%); control:
   both at 1e-6 hold the same cells (difference of 2 or fewer).
--runner (after patch_run_all_rps_disp.py --apply; run_all_rps.before_disp.py kept):
   U1 no option: every key of the old runner's file is identical (runtime_s aside); DISP_per_species 0 added
      (after patch_run_all_rps_founding.py also FOUND_rule "random"; 29 Oct).
   U2 --disp-per-species 1e-6 1e-2: DISP_* saved, SP_disp_rate and LT_d equal species_dispersal_rate of
      species_id and LT_id with the trait seed, tag "_dps1em06-0p01_k2dt"; refused in field mode, with
      DMIN > DMAX, DMIN 0 and DMAX 0.2.
   U3 the rates reach the assembly AND the final run: --disp-per-species 1e-3 1.000000000001e-3 with
      --disp-rate 1e-6 gives the same world as the shared --disp-rate 1e-3 (every key apart from the
      tag, the saved DISPERSAL_RATE and the DISP_* keys). If either part used --disp-rate, the worlds
      would differ.
Ecological reading: species now differ in how far they reach beyond their own cells; nothing else
in a world changes.

Run from obj1_v2/work, after the dispersal patch and copying new_code/assembly_stepwise_ibm.py (v2.2) in:
    USE_GPU=0 python3 ../tests/test_15_dispersal_per_species.py [--runner] | tee ../logs/test_15_dispersal_per_species_29oct.txt
"""
import contextlib, glob, importlib.util, io, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd())
os.environ.setdefault("USE_GPU", "0")
import numpy as np
from scipy import stats
import config, dispersal, environment as env
import assembly_stepwise_ibm as v22
from models_ibm import IBMModel

fails = []
G = config.NUM_PATCHES_Y
host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)


def check(name, cond, text):
    print(f"{name} {text} {'PASS' if cond else 'FAIL'}")
    if not cond:
        fails.append(name)


if not hasattr(dispersal, "set_species_rates"):
    sys.exit("STOP: dispersal.py lacks set_species_rates; apply patch_dispersal_per_species.py")
if "disp=None" not in open(v22.__file__).read():
    sys.exit("STOP: work/assembly_stepwise_ibm.py is not v2.2; copy ../new_code/assembly_stepwise_ibm.py in")

# ---------------------------------------------------------------- P dispersal.py
def v2_path(B):                                          # the v2 code path, written out
    S = B.shape[0]; B2 = B.reshape((S, -1))
    M = dispersal.DISPERSAL_RATE * dispersal.unit_kernel_matrix()
    f = (M.T @ B2.T).T
    ldd = dispersal.LONG_DISTANCE_PROB * f.sum(axis=1) / (B.size // S)
    return ((1 - dispersal.LONG_DISTANCE_PROB) * f + np.broadcast_to(ldd[:, None], f.shape)).reshape(B.shape)


rng = np.random.default_rng(3)
B = rng.integers(0, 3000, size=(25, G, G)) * config.BODY_MASS
d0, k0 = dispersal.DISPERSAL_RATE, dispersal.DISPERSAL_KERNEL
ok1, ok2, ok3, tot = True, True, True, True
for kind, a, b in (("nn4", None, None), ("2dt", 1.0, 3.0), ("gauss", 2.0, None)):
    dispersal.DISPERSAL_KERNEL = kind
    if a is not None: dispersal.DISPERSAL_A = a
    if b is not None: dispersal.DISPERSAL_B = b
    for p in (0.0, 0.2):
        dispersal.LONG_DISTANCE_PROB = p; dispersal.DISPERSAL_RATE = 1e-4
        dispersal.set_species_rates(None)
        base = host(dispersal.compute_dispersal(B))
        ok1 &= np.array_equal(base, host(v2_path(B)))
        dispersal.set_species_rates(np.full(25, 1e-4))
        same = host(dispersal.compute_dispersal(B))
        ok2 &= np.abs(same - base).max() <= 1e-12 * np.abs(base).max()
        rates = 10 ** rng.uniform(-7, -1, 25)
        dispersal.set_species_rates(rates)
        per = host(dispersal.compute_dispersal(B))
        for s in range(25):                                # species by species, shared path at d = rates[s]
            dispersal.set_species_rates(None); dispersal.DISPERSAL_RATE = rates[s]
            one = host(dispersal.compute_dispersal(B[s:s + 1]))[0]
            ok3 &= np.abs(per[s] - one).max() <= 1e-12 * max(np.abs(one).max(), 1e-300)
            tot &= np.isclose(per[s].sum(), rates[s] * B[s].sum(), rtol=1e-10)
        dispersal.set_species_rates(None); dispersal.DISPERSAL_RATE = 1e-4
        ok1 &= np.array_equal(host(dispersal.compute_dispersal(B)), base)
dispersal.DISPERSAL_RATE, dispersal.DISPERSAL_KERNEL, dispersal.LONG_DISTANCE_PROB = d0, k0, 0.0
check("P1", ok1, "rates off: identical to the v2 code path, also after rates were set and switched off (nn4, 2dt, gauss; LDD 0, 0.2):")
check("P2", ok2, "every species at the shared d: same flux as the shared path (rel. diff <= 1e-12):")
check("P3", ok3 and tot, f"rates 1e-7 to 1e-1: each species' flux = its own shared-path flux at its d {ok3}; total emitted = d_i x B_i {tot}:")
bad = []
for arr in (np.full(24, 1e-4), np.r_[-1e-4, np.full(24, 1e-4)]):
    try:
        dispersal.set_species_rates(arr); dispersal.compute_dispersal(B); bad.append(False)
    except ValueError:
        bad.append(True)
    dispersal.set_species_rates(None)
check("P4", all(bad), f"refused: 24 rates for 25 species, a negative rate {bad}:")

st0 = np.random.get_state()[1].copy()
ids = np.arange(5000)
dd = np.array([dispersal.species_dispersal_rate(int(i), seed=202303, d_min=1e-6, d_max=1e-2) for i in ids])
again = np.array([dispersal.species_dispersal_rate(int(i), seed=202303, d_min=1e-6, d_max=1e-2) for i in ids[::-1]])[::-1]
lg = np.log10(dd); ks = stats.kstest((lg + 6) / 4, "uniform").pvalue
niche_same = all(env.niche_traits(int(i), seed=202303, tradeoff=0.5) == env.niche_traits(int(i), seed=202303, tradeoff=0.5) for i in range(50))
nt_before = [env.niche_traits(int(i), seed=202303, tradeoff=0.5) for i in range(50)]
_ = [dispersal.species_dispersal_rate(int(i), seed=202303, d_min=1e-6, d_max=1e-2) for i in range(50)]
nt_after = [env.niche_traits(int(i), seed=202303, tradeoff=0.5) for i in range(50)]
st1 = np.random.get_state()[1]
refused = []
for kw in (dict(d_min=0, d_max=1e-2), dict(d_min=1e-2, d_max=1e-3), dict(d_min=-1e-3, d_max=1e-2)):
    try:
        dispersal.species_dispersal_rate(1, seed=1, **kw); refused.append(False)
    except ValueError:
        refused.append(True)
check("P5", np.array_equal(dd, again) and dd.min() >= 1e-6 and dd.max() <= 1e-2 and ks > 0.01 and abs(lg.mean() + 4) < 0.05
      and dispersal.species_dispersal_rate(7, seed=1, d_min=3e-4, d_max=3e-4) == 3e-4 and all(refused) and niche_same
      and nt_before == nt_after and np.array_equal(st0, st1),
      f"same value in any order; range {dd.min():.2e}-{dd.max():.2e}; log10 mean {lg.mean():.3f} (-4), KS p {ks:.2f}; "
      f"bad ranges refused {refused}; niche traits and global random state unchanged:")

# ---------------------------------------------------------------- A assembly
spec = importlib.util.spec_from_file_location("asm_v2_1", "../new_code/assembly_stepwise_ibm_v2_1.py")
v21 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v21)
H = env.habitat_layer(G, G, 3.0, seed=104201) * 0.25
NICHE = dict(opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=env.NICHE_PEAK_MAX, cost=0.1, tradeoff=0.5)
DISP = dict(d_min=1e-6, d_max=1e-2)
dispersal.DISPERSAL_KERNEL, dispersal.DISPERSAL_A, dispersal.DISPERSAL_B, dispersal.DISPERSAL_RATE = "2dt", 1.0, 3.0, 1e-6


def assemble(mod, states=None, **kw):
    keep = (lambda s: states.append({k: (np.array(v, copy=True) if hasattr(v, "shape") else v) for k, v in s.items()})) \
        if states is not None else None
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.stepwise_assembly_ibm(window_steps=200, record_step=50, seed_size=1, F_sat=None, max_rounds=None,
                                         max_attempts=60, richness_cap=None, frac_multi=0.05, seed=4301,
                                         detection_threshold=1, checkpoint_fn=keep, habitat=H, niche=NICHE,
                                         seed_field=204301, **kw)


def eq(a, b):
    try:
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    except TypeError:
        return np.array_equal(np.asarray(a), np.asarray(b))


o_old, o_new = assemble(v21), assemble(v22)
same = all(eq(x, y) for x, y in zip(o_old[:3], o_new[:3])) and set(o_old[3]) == set(o_new[3]) \
    and all(eq(o_old[3][k], o_new[3][k]) for k in o_old[3])
check("A1", same, f"without disp: r, C, N and all {len(o_old[3])} extra arrays equal v2.1 ({len(o_old[0])} species):")

calls = []
real_set = dispersal.set_species_rates
def spy_set(arr):
    calls.append(None if arr is None else np.array(host(arr), copy=True)); real_set(arr)
dispersal.set_species_rates = spy_set
states = []
r_d, C_d, N_d, x_d = assemble(v22, states=states, disp=DISP)
dispersal.set_species_rates = real_set
rate = lambda i: dispersal.species_dispersal_rate(int(i), seed=204301, **DISP)
lt_id, lt_in = np.asarray(x_d["LT_id"]), np.asarray(x_d["LT_round_in"])
prev, ok_w = np.array([0]), True
sets, offs = calls[0::2], calls[1::2]
for k, s in enumerate(states):
    big = np.r_[prev, lt_id[lt_in == s["round"]]]
    ok_w &= k < len(sets) and sets[k] is not None and np.array_equal(sets[k], [rate(i) for i in big])
    prev = np.asarray(s["ids"])
ok_off = len(calls) == 2 * len(states) and all(c is None for c in offs) and dispersal._SPECIES_RATES is None
ok_ret = np.array_equal(x_d["disp_rate"], [rate(i) for i in x_d["species_id"]]) and np.array_equal(x_d["LT_d"], [rate(i) for i in lt_id])
orig_run = v22.IBMModel.run
def boom(self):
    raise RuntimeError("planted error")
v22.IBMModel.run = boom
try:
    assemble(v22, disp=DISP); after_err = False
except RuntimeError:
    after_err = dispersal._SPECIES_RATES is None
v22.IBMModel.run = orig_run
check("A2", ok_w and ok_off and ok_ret and after_err,
      f"{len(states)} windows: rates = species_dispersal_rate of the window's species list, in order {ok_w}; off after every window "
      f"{ok_off}; disp_rate and LT_d right {ok_ret}; off after an error {after_err}:")
lt_o = {int(i): k for k, i in enumerate(o_new[3]["LT_id"])}
nich = all(np.isclose(x_d["LT_w"][k], o_new[3]["LT_w"][lt_o[int(i)]]) and np.isclose(x_d["LT_opt"][k], o_new[3]["LT_opt"][lt_o[int(i)]])
           for k, i in enumerate(lt_id) if int(i) in lt_o)
first = np.array_equal(lt_id[lt_in == 0], np.asarray(o_new[3]["LT_id"])[np.asarray(o_new[3]["LT_round_in"]) == 0])
check("A3", nich and first, f"niches of the arrived species unchanged by disp {nich}; first round's newcomers the same {first}; "
      f"final species {len(r_d)} with disp, {len(o_new[0])} without:")

# ---------------------------------------------------------------- E ecology
src = np.zeros((G, G), bool); src[8:12, 8:12] = True
yy, xx = np.mgrid[0:G, 0:G]
dist = np.full((G, G), 99)
for y, x in zip(*np.nonzero(src)):
    dist = np.minimum(dist, np.maximum(np.minimum(abs(yy - y), G - abs(yy - y)), np.minimum(abs(xx - x), G - abs(xx - x))))


def two(d1, d2):
    F = np.stack([np.where(src, 0.3, -0.1)] * 2).astype(float)
    N0 = np.stack([np.where(src, 3000, 0)] * 2)
    m = IBMModel(np.array([0.3, 0.3]), np.eye(2), initial_N=N0, r_field=F, nsteps=2000, record_step=10, record_mode="full", seed=5)
    dispersal.set_species_rates(np.array([d1, d2]))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tr = np.asarray(host(m.run()))[-100:] / config.BODY_MASS
    finally:
        dispersal.set_species_rates(None)
    Nm = tr.mean(0)
    return (Nm >= 1).sum(axis=(1, 2)), Nm[:, dist == 1].mean(axis=1)


cells, ring = two(1e-2, 1e-6)
cells_c, _ = two(1e-6, 1e-6)
check("E1", cells[0] - cells[1] >= 8 and abs(ring[0] / 72.3 - 1) <= 0.15 and abs(int(cells_c[0]) - int(cells_c[1])) <= 2,
      f"d 1e-2 against 1e-6: cells held {cells[0]} against {cells[1]}; sinks next to the source {ring[0]:.1f} individuals (test_14: 72.3) "
      f"and {ring[1]:.3f}; control, both 1e-6: {cells_c[0]} and {cells_c[1]} cells:")

# ---------------------------------------------------------------- runner
if "--runner" in sys.argv:
    before = "run_all_rps.before_disp.py"
    if not os.path.exists(before) or "# obj1_v2 per-species d" not in open("run_all_rps.py").read():
        sys.exit("STOP: --runner needs patch_run_all_rps_disp.py --apply (run_all_rps.before_disp.py kept)")
    base = ["--pool", "40", "--skip-psd", "--skip-ode", "--tmax", "400", "--record", "100", "--no-movie", "--ibm-frac-multi", "0.05",
            "--ibm-window-steps", "200", "--ibm-richness-cap", "None", "--save-env-field", "--ibm-record-mode", "full",
            "--ibm-F-sat", "None", "--ibm-max-rounds", "None", "--C-topk", "16", "--coocc-sample", "64", "--world-tag-extra", "t15",
            "--obs-budgets", "1,5", "--ibm-detection-threshold", "1", "--ldd-prob", "0", "--ibm-max-attempts", "30",
            "--disp-kernel", "2dt", "--disp-a", "1", "--disp-b", "3", "--seed-size", "1", "--sim-seed", "4201",
            "--env-mode", "niche", "--habitat-seed", "104201", "--env-seed-field", "204201", "--niche-tradeoff", "0.5",
            "--habitat-contrast", "0.25", "--connectance", "1"]
    env0 = dict(os.environ, USE_GPU="0")

    def run(script, args, folder):
        for f in glob.glob(f"{folder}/results/data/*_training.npz"): os.remove(f)
        p = subprocess.run([sys.executable, script] + args, cwd=folder, env=env0, capture_output=True, text=True)
        fs = sorted(glob.glob(f"{folder}/results/data/*_training.npz"))
        return p.returncode, p.stdout + p.stderr, (np.load(fs[0], allow_pickle=True) if len(fs) == 1 else None), fs

    with tempfile.TemporaryDirectory() as t:
        for f in glob.glob("*.py"):
            shutil.copy(f, t)
        ra, la, za, _ = run(before, base + ["--disp-rate", "1e-6"], t)
        rb, lb, zb, _ = run("run_all_rps.py", base + ["--disp-rate", "1e-6"], t)
        if ra or rb or za is None or zb is None:
            check("U1", False, f"runs failed (exit {ra}, {rb}):\n{la[-500:]}\n{lb[-500:]}")
        else:
            added = set(zb.files) - set(za.files)          # 29 Oct: FOUND_rule ("random") also comes with the founding patch
            ok = all(eq(za[k], zb[k]) for k in za.files if k != "runtime_s") and added - {"FOUND_rule"} == {"DISP_per_species"} \
                and int(zb["DISP_per_species"]) == 0 and ("FOUND_rule" not in added or str(zb["FOUND_rule"][0]) == "random")
            check("U1", ok, f"no option: the {len(za.files)} keys of the old runner identical, only DISP_per_species (0) added"
                  + (" (and FOUND_rule 'random' of the founding patch)" if "FOUND_rule" in added else "") + ":")
        rc, lc, zc, fc = run("run_all_rps.py", base + ["--disp-rate", "1e-6", "--disp-per-species", "1e-6", "1e-2"], t)
        if rc or zc is None:
            check("U2", False, f"per-species run failed (exit {rc}):\n{lc[-800:]}")
        else:
            rt = lambda i: dispersal.species_dispersal_rate(int(i), seed=204201, d_min=1e-6, d_max=1e-2)
            name = os.path.basename(fc[0])
            ok = (int(zc["DISP_per_species"]) == 1 and float(zc["DISP_d_min"]) == 1e-6 and float(zc["DISP_d_max"]) == 1e-2
                  and np.array_equal(zc["SP_disp_rate"], [rt(i) for i in zc["species_id"]])
                  and np.array_equal(zc["LT_d"], [rt(i) for i in zc["LT_id"]]) and "hc0p25_cn1p0_dps1em06-0p01_k2dt" in name)
            badr = []
            for extra in (["--disp-per-species", "1e-2", "1e-3"], ["--disp-per-species", "0", "1e-2"], ["--disp-per-species", "1e-6", "0.2"]):
                r_, l_, _, _ = run("run_all_rps.py", base + ["--disp-rate", "1e-6"] + extra, t)
                badr.append(r_ != 0 and "--disp-per-species" in l_)
            fbase = [x for x in base if x not in ("--niche-tradeoff", "0.5", "--habitat-contrast", "0.25")]
            k = fbase.index("--env-mode"); fbase = fbase[:k] + fbase[k + 6:] + ["--env-length-scale", "5", "--env-var-r", "0.002", "--env-seed-field", "789"]
            r_, l_, _, _ = run("run_all_rps.py", fbase + ["--disp-rate", "1e-6", "--disp-per-species", "1e-6", "1e-2"], t)
            badr.append(r_ != 0 and "--disp-per-species" in l_)
            check("U2", ok and all(badr), f"saved and right, file {name}; refused with DMIN > DMAX, DMIN 0, DMAX 0.2, field mode {badr}:")
        d1 = 1e-3
        rs, ls_, zs, fs = run("run_all_rps.py", base + ["--disp-rate", str(d1)], t)
        rp, lp, zp, fp = run("run_all_rps.py", base + ["--disp-rate", "1e-6", "--disp-per-species", str(d1), repr(d1 * (1 + 1e-12))], t)
        if rs or rp or zs is None or zp is None:
            check("U3", False, f"runs failed (exit {rs}, {rp}):\n{ls_[-500:]}\n{lp[-500:]}")
        else:
            skip = {"runtime_s", "DISPERSAL_RATE"}
            diff = [k for k in zs.files if k not in skip and not k.startswith("DISP_") and not eq(zs[k], zp[k])]
            check("U3", not diff and len(zs["species_id"]) > 2,
                  f"per-species rates of {d1:g} with --disp-rate 1e-6 = shared --disp-rate {d1:g}: keys that differ {diff} "
                  f"({len(zs['species_id'])} species):")

print(f"\n{'all passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)