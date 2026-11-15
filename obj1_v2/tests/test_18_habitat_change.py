"""
test_18_habitat_change.py  (obj1_v2, root cause 23, habitat changing through time, 31 Oct 2026)

Question: does assembly v2.4's habitat_change move the habitat as designed (contrast and spatial scale kept, one step
per round, from its own random stream), make every growth field follow the map of its round, and leave every world
made without the option exactly as v2.3 made it? Does the runner pass the option and save the maps?

R  regression, option off: v2.4 returns every array of v2.3 (new_code/assembly_stepwise_ibm_v2_3.py) and no new key:
   R1 field mode, R2 niche mode with random founding, R3 niche mode with best founding (both with d per species).
H  habitat change, small niche worlds on the CPU (contrast 0.25, habitat scale 3, k 0.5, connectance 1, d per species
   1e-6 to 1e-2; windows of W = 100 steps, tau set so that rho = exp(-W / tau) = 0.9):
   H1 every map of habitat_t has mean 0 and sd = contrast (to 1e-5); its neighbour correlation is within 0.08 of that
      of fresh habitat_layer maps; maps 1 and 5 rounds apart correlate within 0.06 of rho and 0.12 of rho^5.
   H2 a rebuild of the rule written in this test (U <- rho U + sqrt(1 - rho^2) Z, standardised; Z from
      habitat_layer with seed [habitat seed, 104729, round]) gives every map of habitat_t, and every window's growth
      fields, recorded by a spy around IBMModel, equal niche_r(map of that round, niche of each species) - cost (to
      1e-12), for random and for best founding. Control: without the option each species keeps one field in every window.
   H3 own random stream: with and without the option, round 0 is the same window (fields, counts, links, demographic
      seed); round 1 has the same counts, links and seed but other fields; every window's demographic seed is the same;
      the main stream (full generator state, read by a spy at every draw_interactions call) is the same at every draw
      of rounds 0 and 1. The state check is needed: numpy keeps one spare 32-bit draw, so a founding cell can match
      even after the main stream has been used (seen with mutant M3).
   H4 returned: habitat = the last map, habitat_start = the start map, r_field = niche_r(habitat) of the final species,
      habitat_rho = exp(-W / tau), habitat_tau = tau, one map per round.
   H5 refused: field mode, tau 0, contrast -1, a missing key, an extra key, a resumed run.
--runner (after patch_run_all_rps_habitat_tau.py --apply, which keeps run_all_rps.before_habitat_tau.py):
   U1 no option: every key of the old runner's file is identical (runtime_s aside); only HABITAT_tau (-1) is added.
   U2 --habitat-tau 5000: tag "_ht5000"; HABITAT_tau 5000; ENV_habitat_start = U1's ENV_habitat; ENV_habitat =
      ENV_habitat_t[-1]; one map per round; HABITAT_rho = exp(-window / tau).
   U3 refused: --habitat-tau without --env-mode niche, with --habitat-file, and --habitat-tau 0.
--mutants: six planted errors in copies of v2.4 must each fail its check: M1 no re-standardising (H1), M2 residents
   keep their arrival fields (H2), M3 the new maps drawn from the main random stream (H3), M4 the habitat moves in
   round 0 too (H3), M5 rho ignored, a new map every round (H1), M6 the start map returned as the final map (H4).

Run from obj1_v2/work, after copying ../new_code/assembly_stepwise_ibm.py (v2.4) into work:
    USE_GPU=0 python3 ../tests/test_18_habitat_change.py --runner --mutants | tee ../logs/test_18_habitat_change_31oct.txt
"""
import contextlib, glob, importlib.util, os, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd())
os.environ.setdefault("USE_GPU", "0")
import numpy as np
import config, dispersal, environment as env, assembly_utils, models_ibm
import assembly_stepwise_ibm as v24

fails = []


def check(name, cond, text):
    print(f"{name} {text}: {'PASS' if cond else 'FAIL'}", flush=True)
    if not cond:
        fails.append(name)


V24_SRC = open(v24.__file__).read()
if "habitat_change=None," not in V24_SRC:
    sys.exit("STOP: work/assembly_stepwise_ibm.py is not v2.4; copy ../new_code/assembly_stepwise_ibm.py in")
if not os.path.exists("../new_code/assembly_stepwise_ibm_v2_3.py"):
    sys.exit("STOP: ../new_code/assembly_stepwise_ibm_v2_3.py (the v2.3 copy) is missing")
_spec = importlib.util.spec_from_file_location("asm_v2_3", "../new_code/assembly_stepwise_ibm_v2_3.py")
v23 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v23)

dispersal.DISPERSAL_KERNEL, dispersal.DISPERSAL_A, dispersal.DISPERSAL_B, dispersal.DISPERSAL_RATE = "2dt", 1.0, 3.0, 1e-6
dispersal.LONG_DISTANCE_PROB = 0.0
CONN0 = assembly_utils.CONNECTANCE
assembly_utils.CONNECTANCE = 1.0                         # as n29-n33
G = config.NUM_PATCHES_Y
NICHE = dict(opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=1.5, cost=0.1, tradeoff=0.5)
TRAIT_KW = {k: v for k, v in NICHE.items() if k != "cost"}
CONTRAST, LS, HSEED, TSEED = 0.25, 3.0, 102303, 202303
H0 = CONTRAST * env.habitat_layer(G, G, LS, seed=HSEED)
W, RHO = 100, 0.9
TAU = W / np.log(1.0 / RHO)


def make_spy(log):
    real = models_ibm.IBMModel

    class Spy(real):
        def __init__(self, r, C, **kw):
            log.append(dict(r_field=np.array(kw["r_field"], dtype=float, copy=True), N=np.array(kw["initial_N"], copy=True),
                            C=np.array(C, dtype=float, copy=True), seed=int(kw["seed"])))
            super().__init__(r, C, **kw)
    return Spy


@contextlib.contextmanager
def spied(mod, log):
    if log is None:
        yield
        return
    old = mod.IBMModel
    mod.IBMModel = make_spy(log)
    try:
        yield
    finally:
        mod.IBMModel = old


def run(mod, *, founding="random", change=False, field_mode=False, attempts=40, log=None, **extra_kw):
    kw = dict(window_steps=W, record_step=W, max_rounds=None, max_attempts=attempts, seed=2303, frac_multi=0.05,
              detection_threshold=1, dispersal_type="propagule", seed_size=1, F_sat=None)   # F_sat None, as the runner (--ibm-F-sat None)
    if field_mode:
        kw.update(length_scale=5.0, var_r=0.002, seed_field=789)
    else:
        kw.update(habitat=H0, niche=NICHE, seed_field=TSEED, disp=dict(d_min=1e-6, d_max=1e-2))
    if founding == "best":
        kw.update(founding="best")
    if change:
        kw.update(habitat_change=dict(tau=TAU, seed=HSEED, length_scale=LS, contrast=CONTRAST))
    kw.update(extra_kw)
    with spied(mod, log):
        return mod.stepwise_assembly_ibm(**kw)


def same_out(a, b):
    (ra, Ca, Na, ea), (rb, Cb, Nb, eb) = a, b
    if set(ea) != set(eb):
        return False, f"keys differ: {sorted(set(ea) ^ set(eb))}"
    for k in ea:
        x, y = np.asarray(ea[k]), np.asarray(eb[k])
        eq = (x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)))
        if not eq:
            return False, f"{k} differs"
    for x, y, nm in ((ra, rb, "r"), (Ca, Cb, "C"), (Na, Nb, "N")):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            return False, f"{nm} differs"
    return True, f"{len(ea)} keys and r, C, N identical"


def rebuild_maps(n, tau=TAU):
    rho = np.exp(-W / tau); U = H0 / CONTRAST; maps = [H0.copy()]
    for t in range(1, n):
        Z = env.habitat_layer(G, G, LS, seed=[HSEED, 104729, t])
        U = rho * U + np.sqrt(1 - rho ** 2) * Z
        U = (U - U.mean()) / U.std()
        maps.append(CONTRAST * U)
    return maps


def fld(sid, H):
    return env.niche_r(H, *env.niche_traits(int(sid), seed=TSEED, **TRAIT_KW), NICHE["cost"])


def window_ids(extra, best):
    """Species of each window in the order the assembly put them: residents (previous window-end community) then
    the round's newcomers (entered ones under best founding), in id order."""
    P_ids, off = np.asarray(extra["ASM_P_ids"]), np.asarray(extra["ASM_P_offsets"])
    lt_id, lt_in = np.asarray(extra["LT_id"]), np.asarray(extra["LT_round_in"])
    ent = np.asarray(extra["LT_entered"]) == 1 if best else np.ones(len(lt_id), bool)
    out, prev = [], (np.zeros(0, np.int64) if best else np.array([0], np.int64))
    for t in range(len(extra["ASM_round"])):
        new = np.sort(lt_id[(lt_in == t) & ent])
        out.append(np.r_[prev, new])
        prev = P_ids[off[t]:off[t + 1]]
    return out


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def nbr_corr(M):
    return 0.5 * (corr(M, np.roll(M, 1, 0)) + corr(M, np.roll(M, 1, 1)))


FRESH_NC = float(np.mean([nbr_corr(env.habitat_layer(G, G, LS, seed=s)) for s in range(1, 51)]))


# ---------------------------------------------------------------- checks (each takes the module under test)
def h1(m):
    _, _, _, e = run(m, change=True, attempts=60)
    Ht = np.asarray(e["habitat_t"], float); n = len(Ht)
    mean_ok = bool(np.abs(Ht.mean(axis=(1, 2))).max() < 1e-5)
    sd = Ht.std(axis=(1, 2)); sd_ok = bool(np.abs(sd - CONTRAST).max() < 1e-5)
    nc = float(np.mean([nbr_corr(M) for M in Ht]))
    c1 = float(np.mean([corr(Ht[t], Ht[t + 1]) for t in range(n - 1)]))
    c5 = float(np.mean([corr(Ht[t], Ht[t + 5]) for t in range(n - 5)])) if n > 5 else np.nan
    ok = mean_ok and sd_ok and abs(nc - FRESH_NC) <= 0.08 and abs(c1 - RHO) <= 0.06 and abs(c5 - RHO ** 5) <= 0.12
    return ok, (f"{n} maps: largest |mean| {np.abs(Ht.mean(axis=(1, 2))).max():.1e}, sd {sd.min():.6f}-{sd.max():.6f} "
                f"(contrast {CONTRAST}); neighbour correlation {nc:.3f} (fresh maps {FRESH_NC:.3f}); correlation 1 round apart "
                f"{c1:.3f} (rho {RHO}), 5 rounds apart {c5:.3f} (rho^5 {RHO ** 5:.3f})")


def h2(m, control=False):
    worst_f, worst_m, n_win, msg = 0.0, 0.0, 0, []
    for founding in ("random", "best"):
        log = []
        _, _, _, e = run(m, founding=founding, change=True, log=log)
        ids = window_ids(e, founding == "best")
        maps = rebuild_maps(len(e["ASM_round"]))
        if len(log) != len(ids) or len(ids) != len(maps):
            return False, f"{founding}: {len(log)} windows, {len(ids)} rounds, {len(maps)} maps"
        worst_m = max(worst_m, max(float(np.abs(np.asarray(e["habitat_t"][t], float) - maps[t].astype(np.float32)).max())
                                   for t in range(len(maps))))
        for t, (w, sp) in enumerate(zip(log, ids)):
            if w["r_field"].shape[0] != len(sp):
                return False, f"{founding}, round {t}: {w['r_field'].shape[0]} fields for {len(sp)} species"
            if len(sp):
                worst_f = max(worst_f, float(np.abs(w["r_field"] - np.stack([fld(s, maps[t]) for s in sp])).max()))
        n_win += len(log)
        msg.append(f"{founding} founding {len(log)} windows")
    ok = worst_f <= 1e-12 and worst_m <= 1e-6 and n_win > 0
    text = (f"{', '.join(msg)}: largest difference from the rule, fields {worst_f:.1e} (1e-12 allowed), "
            f"saved maps {worst_m:.1e} (float32, 1e-6 allowed)")
    if control:
        log = []
        _, _, _, e = run(m, change=False, log=log)
        ids = window_ids(e, False); first, bad = {}, 0
        for w, sp in zip(log, ids):
            for row, s in zip(w["r_field"], sp):
                s = int(s)
                if s in first and not np.array_equal(first[s], row):
                    bad += 1
                first.setdefault(s, row)
        ok = ok and bad == 0
        text += f"; control without the option: {bad} species with a changed field over {len(log)} windows"
    return ok, text


def stream_run(m, change, log):
    """run() with a spy on draw_interactions: (round index, full state of the main generator) at every call."""
    states, real = [], m.draw_interactions

    def spy(n, rng=None, **kw):
        states.append((len(log), repr(rng.bit_generator.state)))
        return real(n, rng=rng, **kw)
    m.draw_interactions = spy
    try:
        run(m, change=change, log=log)
    finally:
        m.draw_interactions = real
    return [s for s in states if s[0] <= 1]


def h3(m):
    l0, l1 = [], []
    s0 = stream_run(m, False, l0); s1 = stream_run(m, True, l1)
    stream = s0 == s1 and any(t == 1 for t, _ in s0)
    n = min(len(l0), len(l1))
    r0 = all(np.array_equal(l0[0][k], l1[0][k]) for k in ("r_field", "N", "C")) and l0[0]["seed"] == l1[0]["seed"]
    r1 = (n > 1 and np.array_equal(l0[1]["N"], l1[1]["N"]) and np.array_equal(l0[1]["C"], l1[1]["C"])
          and l0[1]["seed"] == l1[1]["seed"] and not np.array_equal(l0[1]["r_field"], l1[1]["r_field"]))
    seeds = all(l0[t]["seed"] == l1[t]["seed"] for t in range(n))
    return (r0 and r1 and seeds and stream), (f"round 0 identical {r0}; round 1 same counts, links and seed, other fields {r1}; "
                                              f"demographic seeds identical over {n} windows {seeds}; main stream identical "
                                              f"at {len(s0)} draws of rounds 0-1 {stream}")


def h4(m):
    _, _, _, e = run(m, change=True)
    Ht = np.asarray(e["habitat_t"])
    final = np.asarray(e["habitat"], float)
    ok_last = np.abs(final - Ht[-1].astype(float)).max() < 1e-6
    ok_start = np.array_equal(np.asarray(e["habitat_start"], float), H0)
    F = np.asarray(e["r_field"]); sp = np.asarray(e["species_id"])
    ok_F = len(sp) > 0 and float(np.abs(F - np.stack([fld(s, final) for s in sp])).max()) < 1e-12
    ok_rho = abs(float(e["habitat_rho"]) - np.exp(-W / TAU)) < 1e-15 and abs(float(e["habitat_tau"]) - TAU) < 1e-9
    ok_n = Ht.shape == (len(e["ASM_round"]), G, G)
    moved = not np.allclose(final, H0)
    return (ok_last and ok_start and ok_F and ok_rho and ok_n and moved), (
        f"final map = last map {ok_last}, start map kept {ok_start}, final fields from the final map {ok_F}, "
        f"rho and tau {ok_rho}, one map per round {ok_n} ({Ht.shape[0]}), the map moved {moved}")


def h5(m):
    cases = {
        "field mode": dict(field_mode=True, habitat_change=dict(tau=TAU, seed=HSEED, length_scale=LS, contrast=CONTRAST)),
        "tau 0": dict(habitat_change=dict(tau=0, seed=HSEED, length_scale=LS, contrast=CONTRAST)),
        "contrast -1": dict(habitat_change=dict(tau=TAU, seed=HSEED, length_scale=LS, contrast=-1.0)),
        "missing key": dict(habitat_change=dict(tau=TAU, seed=HSEED, length_scale=LS)),
        "extra key": dict(habitat_change=dict(tau=TAU, seed=HSEED, length_scale=LS, contrast=CONTRAST, rho=0.5)),
        "resumed run": dict(habitat_change=dict(tau=TAU, seed=HSEED, length_scale=LS, contrast=CONTRAST),
                            init_r=np.array([1.0]), init_C=np.array([[1.0]]), init_N=np.ones((1, G, G), int)),
    }
    refused = []
    for name, kw in cases.items():
        try:
            run(m, attempts=2, **kw)
        except ValueError:
            refused.append(name)
    return len(refused) == len(cases), f"refused {len(refused)} of {len(cases)}: {', '.join(refused)}"


# ---------------------------------------------------------------- R: option off = v2.3
for name, kw in (("R1", dict(field_mode=True)), ("R2", dict()), ("R3", dict(founding="best"))):
    ok, text = same_out(run(v23, **kw), run(v24, **kw))
    check(name, ok, f"{'field mode' if kw.get('field_mode') else kw.get('founding', 'random') + ' founding'}, option off: v2.4 = v2.3: {text}")

for name, fn in (("H1", h1), ("H2", lambda m: h2(m, control=True)), ("H3", h3), ("H4", h4), ("H5", h5)):
    ok, text = fn(v24)
    check(name, ok, text)

# ---------------------------------------------------------------- runner
if "--runner" in sys.argv:
    if not os.path.exists("run_all_rps.before_habitat_tau.py"):
        sys.exit("STOP: run_all_rps.before_habitat_tau.py missing; apply patch_run_all_rps_habitat_tau.py first")
    ARGS = ["--pool", "22510000", "--skip-psd", "--skip-ode", "--tmax", "200", "--record", "100", "--no-movie",
            "--ibm-frac-multi", "0.05", "--ibm-window-steps", "100", "--ibm-richness-cap", "None", "--save-env-field",
            "--ibm-record-mode", "full", "--ibm-F-sat", "None", "--ibm-max-rounds", "None", "--ibm-max-attempts", "30",
            "--world-tag-extra", "t18", "--fp16-time-series", "--obs-budgets", "1,5", "--ibm-detection-threshold", "1",
            "--ldd-prob", "0", "--env-mode", "niche", "--habitat-seed", "102303", "--env-seed-field", "202303",
            "--habitat-ls", "3", "--niche-tradeoff", "0.5", "--habitat-contrast", "0.25", "--connectance", "1.0",
            "--disp-per-species", "1e-06", "0.01", "--disp-rate", "1e-6", "--disp-kernel", "2dt", "--disp-a", "1",
            "--disp-b", "3", "--seed-size", "1", "--sim-seed", "2303"]
    WORK = os.getcwd()

    def runner(script, args, tmp):
        p = subprocess.run([sys.executable, os.path.join(WORK, script)] + args, cwd=tmp, capture_output=True, text=True,
                           env=dict(os.environ, USE_GPU="0"))
        files = sorted(glob.glob(os.path.join(tmp, "results", "data", "*_training.npz")))
        return p, files

    with tempfile.TemporaryDirectory() as t_old, tempfile.TemporaryDirectory() as t_new, tempfile.TemporaryDirectory() as t_ht:
        p_old, f_old = runner("run_all_rps.before_habitat_tau.py", ARGS, t_old)
        p_new, f_new = runner("run_all_rps.py", ARGS, t_new)
        if p_old.returncode or p_new.returncode or len(f_old) != 1 or len(f_new) != 1:
            check("U1", False, f"runs failed (exit {p_old.returncode}, {p_new.returncode}): {(p_old.stderr or p_new.stderr)[-400:]}")
        else:
            a, b = np.load(f_old[0], allow_pickle=True), np.load(f_new[0], allow_pickle=True)
            extra_keys = set(b.files) - set(a.files)
            diff = [k for k in a.files if k != "runtime_s" and not (
                a[k].shape == b[k].shape and (np.array_equal(a[k], b[k], equal_nan=True) if a[k].dtype.kind == "f"
                                              else np.array_equal(a[k], b[k])))]
            ok = (extra_keys == {"HABITAT_tau"} and int(b["HABITAT_tau"]) == -1 and not diff
                  and os.path.basename(f_old[0]) == os.path.basename(f_new[0]))
            check("U1", ok, f"no option: same file name, keys added {sorted(extra_keys)} (HABITAT_tau {int(b['HABITAT_tau']) if 'HABITAT_tau' in b.files else 'missing'}), "
                            f"keys that differ {diff[:5]}")
            p_ht, f_ht = runner("run_all_rps.py", ARGS + ["--habitat-tau", "5000"], t_ht)
            if p_ht.returncode or len(f_ht) != 1:
                check("U2", False, f"run failed (exit {p_ht.returncode}): {p_ht.stderr[-400:]}")
            else:
                c = np.load(f_ht[0], allow_pickle=True)
                Ht = c["ENV_habitat_t"]
                ok = ("_ht5000_k2dt" in os.path.basename(f_ht[0]) and int(c["HABITAT_tau"]) == 5000
                      and np.array_equal(c["ENV_habitat_start"], b["ENV_habitat"]) and np.array_equal(c["ENV_habitat"], Ht[-1])
                      and Ht.shape == (len(c["ASM_round"]), G, G) and abs(float(c["HABITAT_rho"]) - np.exp(-100 / 5000)) < 1e-12
                      and not np.array_equal(c["ENV_habitat"], c["ENV_habitat_start"]))
                check("U2", ok, f"--habitat-tau 5000: tag {'_ht5000_k2dt' in os.path.basename(f_ht[0])}, HABITAT_tau {int(c['HABITAT_tau'])}, "
                                f"start map = static map {np.array_equal(c['ENV_habitat_start'], b['ENV_habitat'])}, final map = last of "
                                f"{Ht.shape[0]} maps {np.array_equal(c['ENV_habitat'], Ht[-1])}, rho {float(c['HABITAT_rho']):.6f}")
        refused = []
        drop = {"--env-mode": 1, "--habitat-seed": 1, "--habitat-ls": 1, "--niche-tradeoff": 1, "--habitat-contrast": 1, "--disp-per-species": 2}
        field_args, skip = [], 0
        for k, x in enumerate(ARGS):
            if skip:
                skip -= 1; continue
            if x in drop:
                skip = drop[x]; continue
            field_args.append(x)
        field_args += ["--env-length-scale", "5", "--env-var-r", "0.002"]
        file_args = [x if x != "--habitat-seed" else "--habitat-file" for x in ARGS]
        file_args[file_args.index("--habitat-file") + 1] = "no_such_map.npy"
        for name, args, msg in (("field mode", field_args + ["--habitat-tau", "5000"], "--habitat-tau needs --env-mode niche"),
                                ("habitat file", file_args + ["--habitat-tau", "5000"], "--habitat-tau needs --env-mode niche"),
                                ("tau 0", ARGS + ["--habitat-tau", "0"], "--habitat-tau must be a whole number")):
            with tempfile.TemporaryDirectory() as t:
                p, f = runner("run_all_rps.py", args, t)
                if p.returncode != 0 and msg in (p.stdout + p.stderr) and not f:
                    refused.append(name)
        check("U3", len(refused) == 3, f"refused {len(refused)} of 3 with the right message: {', '.join(refused)}")

# ---------------------------------------------------------------- mutants
if "--mutants" in sys.argv:
    MUT = {
        "M1": ("H1", 'hc["U"] = (U - U.mean()) / U.std()', 'hc["U"] = U'),
        "M2": ("H2", "            F = np.stack([field(i) for i in ids]) if len(ids) else np.zeros((0, Ny, Nx))\n", ""),
        "M3": ("H3", 'Z = habitat_layer(Ny, Nx, hc["ls"], seed=[hc["seed"], 104729, int(rnd_)])', 'Z = habitat_layer(Ny, Nx, hc["ls"], seed=rng)'),
        "M4": ("H3", "if hc is not None and rnd > start_round:", "if hc is not None and rnd >= start_round:"),
        "M5": ("H1", 'hc["rho"] = float(np.exp(-float(window_steps) / hc["tau"]))', 'hc["rho"] = 0.0'),
        "M6": ("H4", "        H = H_cur[0]                                    # the map of the final fields, for the final run\n", ""),
    }
    FN = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (target, old, new) in MUT.items():
            if V24_SRC.count(old) != 1:
                check(name, False, f"mutation text found {V24_SRC.count(old)} times; mutant not built"); continue
            path = os.path.join(tmp, f"asm_{name}.py")
            open(path, "w").write(V24_SRC.replace(old, new))
            sp = importlib.util.spec_from_file_location(f"asm_{name}", path)
            mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm)
            try:
                ok, text = FN[target](mm)
            except Exception as ex:                      # a mutant that crashes the check is caught too
                ok, text = False, f"raised {type(ex).__name__}: {ex}"
            check(name, not ok, f"caught by {target}: {text[:300]}")

assembly_utils.CONNECTANCE = CONN0
print("\nall passed" if not fails else f"\nFAILED: {', '.join(fails)}")
sys.exit(0 if not fails else 1)