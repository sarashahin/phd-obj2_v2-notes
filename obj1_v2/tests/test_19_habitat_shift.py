"""
test_19_habitat_shift.py  (obj1_v2, root cause 23, fix B part 2: regional shift of the habitat layer, 31 Oct 2026)

Question: does assembly v2.5's shift (habitat_change shift_sd, shift_tau) move the whole layer as designed
(H_t = contrast x (U_t + m_t), m_t a slow series with sd shift_sd, from its own random stream, m_0 = 0), keep every
U_t and round 0 of the twin without the shift, make every field follow the shifted map, and leave every world made
without the shift exactly as v2.4 made it? Does the runner pass the options and save the shift?

S0 regression, shift off: v2.5 returns every array of v2.4 (new_code/assembly_stepwise_ibm_v2_4.py) and no new key, for
   random and best founding, with and without habitat_change.
Shift on, small niche worlds on the CPU (as test_18: contrast 0.25, scale 3, k 0.5, connectance 1, d per species;
windows of W = 100 steps; position rho 0.9; shift sd 1.0 with rho_m = exp(-W / shift_tau) = 0.8, a value other than
the position rho so that one cannot stand in for the other):
   S1 the returned shift series equals a rebuild of the rule written in this test (m_t = rho_m m_t-1 + sqrt(1 - rho_m^2)
      sd eps_t, eps_t from [habitat seed, 104743, round], m_0 = 0) to 1e-12, for random and best founding; the rule
      itself over 20,000 rounds has sd within 3% of shift_sd, lag-1 correlation within 0.02 of rho_m and |mean| below
      0.08; habitat_shift_rho = exp(-W / shift_tau).
   S2 every window's growth fields, recorded by a spy around IBMModel, equal niche_r(contrast x (U_t + m_t)) with U_t
      rebuilt by test_18's rule (1e-12), and every saved map equals contrast x (U_t + m_t) (float32, 1e-6), for random
      and best founding.
   S3 twin: with and without the shift, every saved map minus contrast x m_t is the twin's map (1e-6); round 0 is the
      same window (fields, counts, links, seed); round 1 has the same counts, links and seed but other fields; every
      window's demographic seed is the same; the main stream (full generator state at every draw_interactions call)
      is the same at every draw of rounds 0 and 1.
   S4 returned: one shift per map, the first 0; each map's mean is contrast x its shift (1e-6); habitat = the last map;
      final fields = niche_r(habitat); habitat_shift_sd and habitat_shift_tau as given.
   S5 refused: shift_sd without shift_tau, shift_tau without shift_sd, shift_sd 0, shift_sd -1, shift_tau 0, an unknown
      key, field mode.
--runner (after patch_run_all_rps_habitat_shift.py --apply, which keeps run_all_rps.before_habitat_shift.py):
   U1 no shift option: the old runner's file and the new one are identical (runtime_s aside, no key added), without
      options and with --habitat-tau 5000.
   U2 --habitat-tau 5000 --habitat-shift-sd 1.0 --habitat-shift-tau 2000: tag "_ht5000_hs1p0t2000_"; HABITAT_shift_sd,
      HABITAT_shift_tau, HABITAT_shift_rho = exp(-100 / 2000); one shift per map, the first 0; every map minus
      0.25 x its shift is the --habitat-tau 5000 map of the same round; ENV_habitat = the last map.
   U3 refused with the right message: the shift without --habitat-tau, --habitat-shift-sd without --habitat-shift-tau,
      --habitat-shift-sd 0.
--mutants: six planted errors in copies of v2.5 must each fail their check: M1 eps drawn from the main stream (S3),
   M2 the shift left out of the map (S2), M3 the position rho used for the shift (S1), M4 the sqrt(1 - rho_m^2) factor
   left out (S1), M5 the shift series not recorded (S4), M6 the shift added without the contrast (S4).

Run from obj1_v2/work, after copying ../new_code/assembly_stepwise_ibm.py (v2.5) into work and applying the patch:
    USE_GPU=0 python3 ../tests/test_19_habitat_shift.py --runner --mutants | tee ../logs/test_19_habitat_shift_31oct.txt
"""
import contextlib, glob, importlib.util, os, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd())
os.environ.setdefault("USE_GPU", "0")
import numpy as np
import config, dispersal, environment as env, assembly_utils, models_ibm
import assembly_stepwise_ibm as v25

fails = []


def check(name, cond, text):
    print(f"{name} {text}: {'PASS' if cond else 'FAIL'}", flush=True)
    if not cond:
        fails.append(name)


V25_SRC = open(v25.__file__).read()
if "shift_sd" not in V25_SRC or "habitat_change=None," not in V25_SRC:
    sys.exit("STOP: work/assembly_stepwise_ibm.py is not v2.5; copy ../new_code/assembly_stepwise_ibm.py in")
if not os.path.exists("../new_code/assembly_stepwise_ibm_v2_4.py"):
    sys.exit("STOP: ../new_code/assembly_stepwise_ibm_v2_4.py (the v2.4 copy) is missing")
_spec = importlib.util.spec_from_file_location("asm_v2_4", "../new_code/assembly_stepwise_ibm_v2_4.py")
v24 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v24)

dispersal.DISPERSAL_KERNEL, dispersal.DISPERSAL_A, dispersal.DISPERSAL_B, dispersal.DISPERSAL_RATE = "2dt", 1.0, 3.0, 1e-6
dispersal.LONG_DISTANCE_PROB = 0.0
CONN0 = assembly_utils.CONNECTANCE
assembly_utils.CONNECTANCE = 1.0
G = config.NUM_PATCHES_Y
NICHE = dict(opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=1.5, cost=0.1, tradeoff=0.5)
TRAIT_KW = {k: v for k, v in NICHE.items() if k != "cost"}
CONTRAST, LS, HSEED, TSEED = 0.25, 3.0, 102303, 202303
H0 = CONTRAST * env.habitat_layer(G, G, LS, seed=HSEED)
W, RHO, SD, RHO_M = 100, 0.9, 1.0, 0.8
TAU, TAU_M = W / np.log(1.0 / RHO), W / np.log(1.0 / RHO_M)


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


def hc(shift=True, **over):
    d = dict(tau=TAU, seed=HSEED, length_scale=LS, contrast=CONTRAST)
    if shift:
        d.update(shift_sd=SD, shift_tau=TAU_M)
    d.update(over)
    return d


def run(mod, *, founding="random", change=True, shift=True, field_mode=False, attempts=40, log=None, **extra_kw):
    kw = dict(window_steps=W, record_step=W, max_rounds=None, max_attempts=attempts, seed=2303, frac_multi=0.05,
              detection_threshold=1, dispersal_type="propagule", seed_size=1, F_sat=None)
    if field_mode:
        kw.update(length_scale=5.0, var_r=0.002, seed_field=789)
    else:
        kw.update(habitat=H0, niche=NICHE, seed_field=TSEED, disp=dict(d_min=1e-6, d_max=1e-2))
    if founding == "best":
        kw.update(founding="best")
    if change:
        kw.update(habitat_change=hc(shift))
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


def rebuild_U(n):
    U = H0 / CONTRAST; out = [U.copy()]
    for t in range(1, n):
        Z = env.habitat_layer(G, G, LS, seed=[HSEED, 104729, t])
        U = RHO * U + np.sqrt(1 - RHO ** 2) * Z
        U = (U - U.mean()) / U.std()
        out.append(U)
    return out


def rebuild_m(n):
    rho_m = np.exp(-W / TAU_M)                            # = RHO_M up to rounding, computed as the code does
    m, out = 0.0, [0.0]
    for t in range(1, n):
        eps = float(np.random.default_rng([HSEED, 104743, t]).standard_normal())
        m = rho_m * m + np.sqrt(1 - rho_m ** 2) * SD * eps
        out.append(m)
    return np.array(out)


def fld(sid, H):
    return env.niche_r(H, *env.niche_traits(int(sid), seed=TSEED, **TRAIT_KW), NICHE["cost"])


def window_ids(extra, best):
    P_ids, off = np.asarray(extra["ASM_P_ids"]), np.asarray(extra["ASM_P_offsets"])
    lt_id, lt_in = np.asarray(extra["LT_id"]), np.asarray(extra["LT_round_in"])
    ent = np.asarray(extra["LT_entered"]) == 1 if best else np.ones(len(lt_id), bool)
    out, prev = [], (np.zeros(0, np.int64) if best else np.array([0], np.int64))
    for t in range(len(extra["ASM_round"])):
        new = np.sort(lt_id[(lt_in == t) & ent])
        out.append(np.r_[prev, new])
        prev = P_ids[off[t]:off[t + 1]]
    return out


# ---------------------------------------------------------------- checks (each takes the module under test)
def s1(m):
    worst, n_all = 0.0, []
    for founding in ("random", "best"):
        _, _, _, e = run(m, founding=founding)
        got = np.asarray(e.get("habitat_shift_t", []), float)
        want = rebuild_m(len(e["ASM_round"]))
        if got.shape != want.shape:
            return False, f"{founding}: {got.shape[0]} shifts for {want.shape[0]} rounds"
        worst = max(worst, float(np.abs(got - want).max())); n_all.append(len(got))
    long = rebuild_m(20000)
    sd, mean = float(long.std()), float(long.mean())
    lag1 = float(np.corrcoef(long[:-1], long[1:])[0, 1])
    rho_ok = abs(float(e["habitat_shift_rho"]) - np.exp(-W / TAU_M)) < 1e-15
    ok = worst <= 1e-12 and abs(sd / SD - 1) <= 0.03 and abs(lag1 - RHO_M) <= 0.02 and abs(mean) < 0.08 and rho_ok
    return ok, (f"returned shifts (random {n_all[0]}, best {n_all[1]} rounds) differ from the rule by {worst:.1e}; the rule over "
                f"20,000 rounds: sd {sd:.3f} (shift sd {SD}), lag-1 correlation {lag1:.3f} (rho_m {RHO_M}), mean {mean:+.3f}; "
                f"habitat_shift_rho right {rho_ok}")


def s2(m):
    worst_f, worst_m, n_win, msg = 0.0, 0.0, 0, []
    for founding in ("random", "best"):
        log = []
        _, _, _, e = run(m, founding=founding, log=log)
        n = len(e["ASM_round"]); ids = window_ids(e, founding == "best")
        U, mm = rebuild_U(n), rebuild_m(n)
        maps = [CONTRAST * (U[t] + mm[t]) for t in range(n)]
        if len(log) != len(ids) or len(ids) != n:
            return False, f"{founding}: {len(log)} windows, {len(ids)} rounds"
        worst_m = max(worst_m, max(float(np.abs(np.asarray(e["habitat_t"][t], float) - maps[t].astype(np.float32)).max())
                                   for t in range(n)))
        for t, (w, sp) in enumerate(zip(log, ids)):
            if w["r_field"].shape[0] != len(sp):
                return False, f"{founding}, round {t}: {w['r_field'].shape[0]} fields for {len(sp)} species"
            if len(sp):
                worst_f = max(worst_f, float(np.abs(w["r_field"] - np.stack([fld(s, maps[t]) for s in sp])).max()))
        n_win += len(log); msg.append(f"{founding} founding {len(log)} windows")
    ok = worst_f <= 1e-12 and worst_m <= 1e-6 and n_win > 0
    return ok, (f"{', '.join(msg)}: largest difference from contrast x (U_t + m_t), fields {worst_f:.1e} (1e-12 allowed), "
                f"saved maps {worst_m:.1e} (1e-6 allowed)")


def stream_run(m, shift, log):
    states, real = [], m.draw_interactions

    def spy(n, rng=None, **kw):
        states.append((len(log), repr(rng.bit_generator.state)))
        return real(n, rng=rng, **kw)
    m.draw_interactions = spy
    try:
        out = run(m, shift=shift, log=log)
    finally:
        m.draw_interactions = real
    return [s for s in states if s[0] <= 1], out


def s3(m):
    l0, l1 = [], []
    s0, (_, _, _, e0) = stream_run(m, False, l0)
    s1_, (_, _, _, e1) = stream_run(m, True, l1)
    stream = s0 == s1_ and any(t == 1 for t, _ in s0)
    n = min(len(l0), len(l1))
    r0 = all(np.array_equal(l0[0][k], l1[0][k]) for k in ("r_field", "N", "C")) and l0[0]["seed"] == l1[0]["seed"]
    r1 = (n > 1 and np.array_equal(l0[1]["N"], l1[1]["N"]) and np.array_equal(l0[1]["C"], l1[1]["C"])
          and l0[1]["seed"] == l1[1]["seed"] and not np.array_equal(l0[1]["r_field"], l1[1]["r_field"]))
    seeds = all(l0[t]["seed"] == l1[t]["seed"] for t in range(n))
    H0t, H1t = np.asarray(e0["habitat_t"], float), np.asarray(e1["habitat_t"], float)
    mm = np.asarray(e1.get("habitat_shift_t", np.zeros(len(H1t))), float)
    k = min(len(H0t), len(H1t), len(mm))
    twin = k > 1 and float(np.abs(H1t[:k] - CONTRAST * mm[:k, None, None] - H0t[:k]).max()) <= 1e-6
    ok = r0 and r1 and seeds and stream and twin
    return ok, (f"maps minus the shift = the twin's maps over {k} rounds {twin}; round 0 identical {r0}; round 1 same counts, "
                f"links and seed, other fields {r1}; demographic seeds identical over {n} windows {seeds}; main stream "
                f"identical at {len(s0)} draws of rounds 0-1 {stream}")


def s4(m):
    _, _, _, e = run(m)
    Ht = np.asarray(e["habitat_t"], float); mm = np.asarray(e.get("habitat_shift_t", []), float)
    n_ok = len(mm) == len(Ht) == len(e["ASM_round"]) and len(mm) > 1
    first = n_ok and mm[0] == 0.0
    mean_ok = n_ok and float(np.abs(Ht.mean(axis=(1, 2)) - CONTRAST * mm).max()) <= 1e-6
    final = np.asarray(e["habitat"], float)
    last_ok = float(np.abs(final - Ht[-1]).max()) <= 1e-6
    F, sp = np.asarray(e["r_field"]), np.asarray(e["species_id"])
    f_ok = len(sp) > 0 and float(np.abs(F - np.stack([fld(s, final) for s in sp])).max()) < 1e-12
    par_ok = float(e.get("habitat_shift_sd", -1)) == SD and abs(float(e.get("habitat_shift_tau", -1)) - TAU_M) < 1e-9
    moved = n_ok and float(np.abs(mm).max()) > 0.1
    ok = n_ok and first and mean_ok and last_ok and f_ok and par_ok and moved
    return ok, (f"one shift per map {n_ok} ({len(mm)} for {len(Ht)}), first 0 {first}, map mean = contrast x shift {mean_ok}, "
                f"final map = last map {last_ok}, final fields from the final map {f_ok}, sd and tau {par_ok}, "
                f"the shift moved {moved} (largest |m| {np.abs(mm).max() if len(mm) else 0:.2f})")


def s5(m):
    cases = {
        "shift_sd alone": dict(habitat_change=hc(False, shift_sd=1.0)),
        "shift_tau alone": dict(habitat_change=hc(False, shift_tau=TAU_M)),
        "shift_sd 0": dict(habitat_change=hc(True, shift_sd=0.0)),
        "shift_sd -1": dict(habitat_change=hc(True, shift_sd=-1.0)),
        "shift_tau 0": dict(habitat_change=hc(True, shift_tau=0.0)),
        "unknown key": dict(habitat_change=hc(True, shift_mean=0.5)),
        "field mode": dict(field_mode=True, change=False, habitat_change=hc(True)),
    }
    refused = []
    for name, kw in cases.items():
        try:
            run(m, attempts=2, **kw)
        except ValueError:
            refused.append(name)
    return len(refused) == len(cases), f"refused {len(refused)} of {len(cases)}: {', '.join(refused)}"


# ---------------------------------------------------------------- S0: shift off = v2.4
for founding in ("random", "best"):
    for change in (False, True):
        ok, text = same_out(run(v24, founding=founding, change=change, shift=False), run(v25, founding=founding, change=change, shift=False))
        check("S0", ok, f"{founding} founding, habitat change {'on' if change else 'off'}, no shift: v2.5 = v2.4: {text}")

for name, fn in (("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4), ("S5", s5)):
    ok, text = fn(v25)
    check(name, ok, text)

# ---------------------------------------------------------------- runner
if "--runner" in sys.argv:
    if not os.path.exists("run_all_rps.before_habitat_shift.py"):
        sys.exit("STOP: run_all_rps.before_habitat_shift.py missing; apply patch_run_all_rps_habitat_shift.py first")
    ARGS = ["--pool", "22510000", "--skip-psd", "--skip-ode", "--tmax", "200", "--record", "100", "--no-movie",
            "--ibm-frac-multi", "0.05", "--ibm-window-steps", "100", "--ibm-richness-cap", "None", "--save-env-field",
            "--ibm-record-mode", "full", "--ibm-F-sat", "None", "--ibm-max-rounds", "None", "--ibm-max-attempts", "30",
            "--world-tag-extra", "t19", "--fp16-time-series", "--obs-budgets", "1,5", "--ibm-detection-threshold", "1",
            "--ldd-prob", "0", "--env-mode", "niche", "--habitat-seed", "102303", "--env-seed-field", "202303",
            "--habitat-ls", "3", "--niche-tradeoff", "0.5", "--habitat-contrast", "0.25", "--connectance", "1.0",
            "--disp-per-species", "1e-06", "0.01", "--disp-rate", "1e-6", "--disp-kernel", "2dt", "--disp-a", "1",
            "--disp-b", "3", "--seed-size", "1", "--sim-seed", "2303"]
    WORK = os.getcwd()

    def runner(script, args, tmp):
        p = subprocess.run([sys.executable, os.path.join(WORK, script)] + args, cwd=tmp, capture_output=True, text=True,
                           env=dict(os.environ, USE_GPU="0"))
        return p, sorted(glob.glob(os.path.join(tmp, "results", "data", "*_training.npz")))

    def same_files(fa, fb):
        a, b = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
        added = sorted(set(b.files) ^ set(a.files))
        diff = [k for k in a.files if k in b.files and k != "runtime_s" and not (
            a[k].shape == b[k].shape and (np.array_equal(a[k], b[k], equal_nan=True) if a[k].dtype.kind == "f"
                                          else np.array_equal(a[k], b[k])))]
        return (not added and not diff and os.path.basename(fa) == os.path.basename(fb)), added, diff

    texts, ok_all, ht_file = [], True, None
    for label, extra in (("no options", []), ("--habitat-tau 5000", ["--habitat-tau", "5000"])):
        with tempfile.TemporaryDirectory() as t_old, tempfile.TemporaryDirectory() as t_new:
            p_old, f_old = runner("run_all_rps.before_habitat_shift.py", ARGS + extra, t_old)
            p_new, f_new = runner("run_all_rps.py", ARGS + extra, t_new)
            if p_old.returncode or p_new.returncode or len(f_old) != 1 or len(f_new) != 1:
                ok_all = False; texts.append(f"{label}: runs failed ({(p_old.stderr or p_new.stderr)[-300:]})"); continue
            same, added, diff = same_files(f_old[0], f_new[0])
            ok_all &= same; texts.append(f"{label}: identical {same} (keys added or lost {added}, differing {diff[:4]})")
            if extra:
                z = np.load(f_new[0], allow_pickle=True)
                ht_file = dict(ENV_habitat_t=np.array(z["ENV_habitat_t"]))
    check("U1", ok_all, "; ".join(texts))
    with tempfile.TemporaryDirectory() as t:
        p, f = runner("run_all_rps.py", ARGS + ["--habitat-tau", "5000", "--habitat-shift-sd", "1.0", "--habitat-shift-tau", "2000"], t)
        if p.returncode or len(f) != 1 or ht_file is None:
            check("U2", False, f"run failed (exit {p.returncode}): {p.stderr[-400:]}")
        else:
            c = np.load(f[0], allow_pickle=True)
            Ht, mm = np.asarray(c["ENV_habitat_t"], float), np.asarray(c["ENV_habitat_shift"], float)
            H5 = np.asarray(ht_file["ENV_habitat_t"], float); k = min(len(Ht), len(H5))
            twin = k > 1 and float(np.abs(Ht[:k] - 0.25 * mm[:k, None, None] - H5[:k]).max()) <= 1e-6
            tag = "_ht5000_hs1p0t2000_" in os.path.basename(f[0])
            ok = (tag and float(c["HABITAT_shift_sd"]) == 1.0 and int(c["HABITAT_shift_tau"]) == 2000
                  and abs(float(c["HABITAT_shift_rho"]) - np.exp(-100 / 2000)) < 1e-12 and len(mm) == len(Ht) and mm[0] == 0.0
                  and twin and np.array_equal(c["ENV_habitat"], c["ENV_habitat_t"][-1]))
            check("U2", ok, f"tag {tag}, sd {float(c['HABITAT_shift_sd'])}, tau {int(c['HABITAT_shift_tau'])}, rho "
                            f"{float(c['HABITAT_shift_rho']):.6f}, {len(mm)} shifts for {len(Ht)} maps (first {mm[0]:.1f}), maps minus "
                            f"the shift = the --habitat-tau 5000 maps over {k} rounds {twin}, final map = last map "
                            f"{np.array_equal(c['ENV_habitat'], c['ENV_habitat_t'][-1])}")
    refused = []
    for name, args, msg in (("no --habitat-tau", ARGS + ["--habitat-shift-sd", "1.0", "--habitat-shift-tau", "2000"], "needs --habitat-tau"),
                            ("sd alone", ARGS + ["--habitat-tau", "5000", "--habitat-shift-sd", "1.0"], "go together"),
                            ("sd 0", ARGS + ["--habitat-tau", "5000", "--habitat-shift-sd", "0", "--habitat-shift-tau", "2000"], "must be > 0")):
        with tempfile.TemporaryDirectory() as t:
            p, f = runner("run_all_rps.py", args, t)
            if p.returncode != 0 and msg in (p.stdout + p.stderr) and not f:
                refused.append(name)
    check("U3", len(refused) == 3, f"refused {len(refused)} of 3 with the right message: {', '.join(refused)}")

# ---------------------------------------------------------------- mutants
if "--mutants" in sys.argv:
    MUT = {
        "M1": ("S3", 'eps = float(np.random.default_rng([hc["seed"], 104743, int(rnd_)]).standard_normal())', 'eps = float(rng.standard_normal())'),
        "M2": ("S2", 'H_cur[0] = hc["contrast"] * (hc["U"] + sh["m"])', 'H_cur[0] = hc["contrast"] * hc["U"]'),
        "M3": ("S1", 'sh["m"] = sh["rho"] * sh["m"] + np.sqrt(1.0 - sh["rho"] ** 2) * sh["sd"] * eps',
               'sh["m"] = hc["rho"] * sh["m"] + np.sqrt(1.0 - hc["rho"] ** 2) * sh["sd"] * eps'),
        "M4": ("S1", 'sh["m"] = sh["rho"] * sh["m"] + np.sqrt(1.0 - sh["rho"] ** 2) * sh["sd"] * eps',
               'sh["m"] = sh["rho"] * sh["m"] + sh["sd"] * eps'),
        "M5": ("S4", 'shift_hist.append(sh["m"])', 'pass'),
        "M6": ("S4", 'H_cur[0] = hc["contrast"] * (hc["U"] + sh["m"])', 'H_cur[0] = hc["contrast"] * hc["U"] + sh["m"]'),
    }
    FN = {"S1": s1, "S2": s2, "S3": s3, "S4": s4}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (target, old, new) in MUT.items():
            if V25_SRC.count(old) != 1:
                check(name, False, f"mutation text found {V25_SRC.count(old)} times; mutant not built"); continue
            path = os.path.join(tmp, f"asm_{name}.py")
            open(path, "w").write(V25_SRC.replace(old, new))
            sp = importlib.util.spec_from_file_location(f"asm_{name}", path)
            mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm)
            try:
                ok, text = FN[target](mm)
            except Exception as ex:
                ok, text = False, f"raised {type(ex).__name__}: {ex}"
            check(name, not ok, f"caught by {target}: {text[:300]}")

assembly_utils.CONNECTANCE = CONN0
print("\nall passed" if not fails else f"\nFAILED: {', '.join(fails)}")
sys.exit(0 if not fails else 1)