"""
test_20_habitat_substeps.py  (obj1_v2, root cause 24, gate 7: habitat in small steps inside each window, 1 Nov 2026)

Question: does assembly v2.6's habitat_change substeps M run each window as M pieces on one IBM model and random stream,
move the habitat (and the shift) before every piece except the very first of the run with rho_s = exp(-(W / M) / tau),
write the new growth fields into the model, save a map after every piece, and leave every world made without the option
exactly as v2.5 made it? Does the runner pass the option and save the maps?

U0 regression, no substeps: v2.6 returns every array of v2.5 (new_code/assembly_stepwise_ibm_v2_5.py) and no new key,
   for random and best founding, with the habitat fixed, moving, and moving with the shift.
U1 substeps 1: every v2.5 key and r, C, N are identical; the keys added are exactly habitat_substeps and ASM_F_* (with
   ASM_F_shift); one map per round, equal to ASM_P of the round (ids and cells).
U2 the IBM of work (models_ibm.IBMModel): a run of W steps and the same seed run as M pieces of W / M steps, with the same
   fields written into the model in place between pieces, give the same counts bit for bit; writing other fields
   between pieces changes the counts (the write reaches the dynamics).
Small niche worlds on the CPU (as test_19: contrast 0.25, scale 3, k 0.5, connectance 1, d per species; W = 100 steps;
rho 0.9 per round; shift sd 1.0 with rho_m 0.8 per round; M = 5):
U3 the habitat steps: habitat_layer is called with seeds [seed, 104729, t, j] for j = 1..4 in round 0 and j = 0..4 in
   every later round, in that order; the saved maps (the last piece of each round) equal a rebuild of the rule written
   here (U <- rho_s U + sqrt(1 - rho_s^2) Z, standardised; m <- rho_ms m + sqrt(1 - rho_ms^2) sd eps, eps from
   [seed, 104743, t, j]) to 1e-6, and ASM_F_shift equals the rebuilt shift of every piece to 1e-12; random and best
   founding.
U4 the fields: one IBM model per round, run M times; the fields the model holds at each run equal niche_r of the rebuilt
   map of that piece for every species of the window (1e-12).
U5 the maps: M maps per round, every recorded species has at least one cell, and the last map of every round equals
   ASM_P of that round (ids and cells).
U6 the rule over 20,000 sub-steps: the shift has sd within 3% of shift_sd and lag-M correlation within 0.02 of rho_m per
   round; the layer's correlation across M sub-steps (one round) is within 0.03 of rho per round (mean over 300 pairs).
U7 refused: substeps 0, -1, 3 (does not divide W = 100), 2.5, True, "5".
--runner (after patch_run_all_rps_habitat_substeps.py --apply, which keeps run_all_rps.before_habitat_substeps.py):
   R1 no substeps option: old and new runner give identical files (runtime_s aside, no key added), without options and
      with --habitat-tau 5000 --habitat-shift-sd 1.0 --habitat-shift-tau 2000.
   R2 --habitat-tau 5000 --habitat-substeps 5: tag "_ht5000_hss5_"; HABITAT_substeps 5; 5 maps per round; the last map
      of each round equals the round's ASM_P map.
   R3 refused with the right message: --habitat-substeps without --habitat-tau; 3 with --ibm-window-steps 100; 0.
--mutants: planted errors in copies of v2.6 must each fail their check: N1 the round's rho used at every sub-step (U3),
   N2 the sub-step seed without j (U3), N3 fields not written into the model (U4), N4 a new model for every piece (U4),
   N5 the first piece of the run moved too (U3), N6 species with no cell recorded (U5).

Run from obj1_v2/work, after copying ../new_code/assembly_stepwise_ibm.py (v2.6) into work and applying the patch:
    USE_GPU=0 python3 ../tests/test_20_habitat_substeps.py --runner --mutants | tee ../logs/test_20_habitat_substeps_1nov.txt
"""
import contextlib, glob, importlib.util, io, os, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd())
os.environ.setdefault("USE_GPU", "0")
import numpy as np
import config, dispersal, environment as env, assembly_utils, models_ibm
import assembly_stepwise_ibm as v26

fails = []


def check(name, cond, text):
    print(f"{name} {text}: {'PASS' if cond else 'FAIL'}", flush=True)
    if not cond:
        fails.append(name)


V26_SRC = open(v26.__file__).read()
if "substeps" not in V26_SRC or "def run_window" not in V26_SRC:
    sys.exit("STOP: work/assembly_stepwise_ibm.py is not v2.6; copy ../new_code/assembly_stepwise_ibm.py in")
if not os.path.exists("../new_code/assembly_stepwise_ibm_v2_5.py"):
    sys.exit("STOP: ../new_code/assembly_stepwise_ibm_v2_5.py (the v2.5 copy) is missing")
_spec = importlib.util.spec_from_file_location("asm_v2_5", "../new_code/assembly_stepwise_ibm_v2_5.py")
v25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v25)

dispersal.DISPERSAL_KERNEL, dispersal.DISPERSAL_A, dispersal.DISPERSAL_B, dispersal.DISPERSAL_RATE = "2dt", 1.0, 3.0, 1e-6
dispersal.LONG_DISTANCE_PROB = 0.0
CONN0 = assembly_utils.CONNECTANCE
assembly_utils.CONNECTANCE = 1.0
G = config.NUM_PATCHES_Y
NICHE = dict(opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=1.5, cost=0.1, tradeoff=0.5)
TRAIT_KW = {k: v for k, v in NICHE.items() if k != "cost"}
CONTRAST, LS, HSEED, TSEED = 0.25, 3.0, 102303, 202303
H0 = CONTRAST * env.habitat_layer(G, G, LS, seed=HSEED)
W, RHO, SD, RHO_M, M = 100, 0.9, 1.0, 0.8, 5
TAU, TAU_M = W / np.log(1.0 / RHO), W / np.log(1.0 / RHO_M)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def make_spy(log):
    real = models_ibm.IBMModel

    class Spy(real):
        def __init__(self, r, C, **kw):
            log.append(dict(runs=[], n=len(r)))
            super().__init__(r, C, **kw)
            self._rec = log[-1]

        def run(self):
            self._rec["runs"].append(np.array(_host(self.r_field), dtype=float, copy=True))
            return super().run()
    return Spy


def _host(a):
    return a.get() if hasattr(a, "get") else a


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


def run(mod, *, founding="random", change=True, shift=True, attempts=40, log=None, **extra_kw):
    kw = dict(window_steps=W, record_step=W, max_rounds=None, max_attempts=attempts, seed=2303, frac_multi=0.05,
              detection_threshold=1, dispersal_type="propagule", seed_size=1, F_sat=None,
              habitat=H0, niche=NICHE, seed_field=TSEED, disp=dict(d_min=1e-6, d_max=1e-2))
    if founding == "best":
        kw.update(founding="best")
    if change is True:
        kw.update(habitat_change=hc(shift))
    elif isinstance(change, dict):
        kw.update(habitat_change=change)
    kw.update(extra_kw)
    with spied(mod, log), quiet():
        return mod.stepwise_assembly_ibm(**kw)


def same_out(a, b, extra_ok=()):
    (ra, Ca, Na, ea), (rb, Cb, Nb, eb) = a, b
    added = set(eb) - set(ea)
    if set(ea) - set(eb) or added - set(extra_ok):
        return False, f"keys differ: {sorted(set(ea) ^ set(eb))}"
    for k in ea:
        x, y = np.asarray(ea[k]), np.asarray(eb[k])
        eq = (x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)))
        if not eq:
            return False, f"{k} differs"
    for x, y, nm in ((ra, rb, "r"), (Ca, Cb, "C"), (Na, Nb, "N")):
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            return False, f"{nm} differs"
    return True, f"{len(ea)} keys and r, C, N identical" + (f"; added {sorted(added)}" if added else "")


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


def rebuild_pieces(n_rounds, shift=True):
    """Maps and shifts of every piece by the rule written here: list over rounds of lists over pieces."""
    rho_s = np.exp(-(W / M) / TAU); rho_ms = np.exp(-(W / M) / TAU_M)
    U, m = H0 / CONTRAST, 0.0
    maps, shifts, seeds = [], [], []
    for t in range(n_rounds):
        mp, sp = [], []
        for j in range(M):
            if not (t == 0 and j == 0):
                seeds.append([HSEED, 104729, t, j])
                Z = env.habitat_layer(G, G, LS, seed=[HSEED, 104729, t, j])
                U = rho_s * U + np.sqrt(1 - rho_s ** 2) * Z
                U = (U - U.mean()) / U.std()
                if shift:
                    eps = float(np.random.default_rng([HSEED, 104743, t, j]).standard_normal())
                    m = rho_ms * m + np.sqrt(1 - rho_ms ** 2) * SD * eps
            mp.append(CONTRAST * (U + m)); sp.append(m)
        maps.append(mp); shifts.append(sp)
    return maps, shifts, seeds


def frames(e):
    off, ids, bits = np.asarray(e["ASM_F_offsets"]), np.asarray(e["ASM_F_ids"]), np.asarray(e["ASM_F_bits"])
    out = []
    for k in range(len(off) - 1):
        a, b = int(off[k]), int(off[k + 1])
        out.append((ids[a:b], np.unpackbits(bits[a:b], axis=1, count=G * G).astype(bool)))
    return out


def asm_p(e, t):
    off = np.asarray(e["ASM_P_offsets"]); a, b = int(off[t]), int(off[t + 1])
    return np.asarray(e["ASM_P_ids"])[a:b], np.unpackbits(np.asarray(e["ASM_P_bits"])[a:b], axis=1, count=G * G).astype(bool)


# ---------------------------------------------------------------- checks (each takes the module under test)
def u3(m):
    worst_h, worst_s, msg = 0.0, 0.0, []
    for founding in ("random", "best"):
        calls, real = [], env.habitat_layer

        def spy(ny, nx, ls, *, seed=None, path=None):
            calls.append(list(seed) if isinstance(seed, (list, tuple)) else seed)
            return real(ny, nx, ls, seed=seed, path=path)
        env.habitat_layer = spy
        try:
            _, _, _, e = run(m, founding=founding, change=hc(True, substeps=M))
        finally:
            env.habitat_layer = real
        n = len(e["ASM_round"])
        maps, shifts, seeds = rebuild_pieces(n)
        if calls != seeds:
            return False, f"{founding}: {len(calls)} habitat steps, {len(seeds)} expected; first differing call " \
                          f"{next((c for c, s in zip(calls, seeds) if c != s), calls[len(seeds):len(seeds) + 1])}"
        Ht = np.asarray(e["habitat_t"], float)
        worst_h = max(worst_h, max(float(np.abs(Ht[t] - maps[t][-1].astype(np.float32)).max()) for t in range(n)))
        fs = np.asarray(e.get("ASM_F_shift", []), float)
        want = np.array([s for sp in shifts for s in sp])
        if fs.shape != want.shape:
            return False, f"{founding}: {fs.shape[0]} piece shifts for {want.shape[0]} pieces"
        worst_s = max(worst_s, float(np.abs(fs - want).max())); msg.append(f"{founding} {n} rounds, {len(calls)} steps")
    ok = worst_h <= 1e-6 and worst_s <= 1e-12
    return ok, (f"{'; '.join(msg)}, seeds in order as written; saved maps against the rebuild {worst_h:.1e} (1e-6), "
                f"piece shifts {worst_s:.1e} (1e-12)")


def u4(m):
    worst, n_runs, msg = 0.0, 0, []
    for founding in ("random", "best"):
        log = []
        _, _, _, e = run(m, founding=founding, change=hc(True, substeps=M), log=log)
        n = len(e["ASM_round"]); ids = window_ids(e, founding == "best")
        maps, _, _ = rebuild_pieces(n)
        if len(log) != n or any(len(w["runs"]) != M for w in log):
            return False, f"{founding}: {len(log)} models for {n} rounds; runs per model {sorted({len(w['runs']) for w in log})}"
        for t, (w, sp) in enumerate(zip(log, ids)):
            if w["n"] != len(sp):
                return False, f"{founding}, round {t}: {w['n']} species in the model, {len(sp)} expected"
            for j, F in enumerate(w["runs"]):
                if len(sp):
                    worst = max(worst, float(np.abs(F - np.stack([fld(s, maps[t][j]) for s in sp])).max()))
            n_runs += len(w["runs"])
        msg.append(f"{founding} {n} models")
    ok = worst <= 1e-12
    return ok, f"{', '.join(msg)}, {n_runs} piece runs, {M} per model; fields against niche_r of each piece's map {worst:.1e} (1e-12)"


def u5(m):
    texts = []
    for founding in ("random", "best"):
        _, _, _, e = run(m, founding=founding, change=hc(True, substeps=M))
        n = len(e["ASM_round"]); fr = frames(e)
        rr, pp = np.asarray(e["ASM_F_round"]), np.asarray(e["ASM_F_piece"])
        layout = len(fr) == n * M and list(pp) == list(range(M)) * n and list(rr) == [t for t in range(n) for _ in range(M)]
        nonempty = all(P.any(axis=1).all() for _, P in fr)
        last = all(np.array_equal(fr[t * M + M - 1][0], asm_p(e, t)[0]) and np.array_equal(fr[t * M + M - 1][1], asm_p(e, t)[1])
                   for t in range(n))
        if not (layout and nonempty and last):
            return False, f"{founding}: {len(fr)} maps for {n} rounds x {M}, layout {layout}, no empty species {nonempty}, last = ASM_P {last}"
        texts.append(f"{founding} {len(fr)} maps")
    return True, f"{', '.join(texts)}: {M} per round in order, every species with a cell, last of each round = ASM_P"


def u7(m):
    cases = {"0": 0, "-1": -1, "3": 3, "2.5": 2.5, "True": True, "'5'": "5"}
    refused = []
    for name, v in cases.items():
        try:
            run(m, attempts=2, change=hc(True, substeps=v))
        except (ValueError, TypeError):
            refused.append(name)
    return len(refused) == len(cases), f"refused {len(refused)} of {len(cases)}: {', '.join(refused)}"


# ---------------------------------------------------------------- U0, U1: v2.5 unchanged
for founding in ("random", "best"):
    for label, change, shift in (("fixed habitat", False, False), ("moving", True, False), ("moving with shift", True, True)):
        ok, text = same_out(run(v25, founding=founding, change=change, shift=shift), run(v26, founding=founding, change=change, shift=shift))
        check("U0", ok, f"{founding} founding, {label}, no substeps: v2.6 = v2.5: {text}")
for founding in ("random", "best"):
    a = run(v25, founding=founding); b = run(v26, founding=founding, change=hc(True, substeps=1))
    added = ("habitat_substeps", "ASM_F_round", "ASM_F_piece", "ASM_F_offsets", "ASM_F_ids", "ASM_F_bits", "ASM_F_shift")
    ok, text = same_out(a, b, extra_ok=added)
    e = b[3]; n = len(e["ASM_round"]); fr = frames(e)
    one = len(fr) == n and all(np.array_equal(fr[t][0], asm_p(e, t)[0]) and np.array_equal(fr[t][1], asm_p(e, t)[1]) for t in range(n))
    check("U1", ok and one and set(b[3]) - set(a[3]) == set(added),
          f"{founding} founding, substeps 1 with shift: {text}; one map per round = ASM_P {one}")

# ---------------------------------------------------------------- U2: the IBM in pieces
rng = np.random.default_rng(5)
S = 6
F0 = rng.normal(0.2, 0.3, (S, G, G))
C = np.eye(S) + 0.4 * (1 - np.eye(S))
N0 = rng.integers(0, 50, (S, G, G))
with quiet():
    a = models_ibm.IBMModel(np.ones(S), C, initial_N=N0, r_field=F0, nsteps=W, record_step=W, record_mode="none", seed=11)
    a.run()
    b = models_ibm.IBMModel(np.ones(S), C, initial_N=N0, r_field=F0, nsteps=W // M, record_step=W, record_mode="none", seed=11)
    for j in range(M):
        if j:
            b.r_field[...] = v26._like(F0, b.r_field); b.r_flat = b.r_field.reshape(S, -1)
        b.run()
    c = models_ibm.IBMModel(np.ones(S), C, initial_N=N0, r_field=F0, nsteps=W // M, record_step=W, record_mode="none", seed=11)
    for j in range(M):
        if j:
            F1 = F0.copy(); F1[0] += 0.5
            c.r_field[...] = v26._like(F1, c.r_field); c.r_flat = c.r_field.reshape(S, -1)
        c.run()
Na, Nb, Nc = (np.asarray(_host(x.N)) for x in (a, b, c))
check("U2", np.array_equal(Na, Nb) and not np.array_equal(Na, Nc) and Nc[0].sum() > Na[0].sum(),
      f"one run of {W} steps = {M} pieces of {W // M} with the same fields written in place (total {int(Na.sum())} = "
      f"{int(Nb.sum())}); other fields written between pieces change the counts (species 0: {int(Na[0].sum())} -> {int(Nc[0].sum())})")

for name, fn in (("U3", u3), ("U4", u4), ("U5", u5)):
    ok, text = fn(v26)
    check(name, ok, text)

# U6 the rule alone over many sub-steps
rho_s, rho_ms = np.exp(-(W / M) / TAU), np.exp(-(W / M) / TAU_M)
mm, m_ = [], 0.0
for k in range(20000):
    eps = float(np.random.default_rng([HSEED, 104743, k // M, k % M]).standard_normal())
    m_ = rho_ms * m_ + np.sqrt(1 - rho_ms ** 2) * SD * eps; mm.append(m_)
mm = np.array(mm)
lagM = float(np.corrcoef(mm[:-M], mm[M:])[0, 1])
U, cors = H0 / CONTRAST, []
for t in range(300):
    U_start = U.copy()
    for j in range(M):
        Z = env.habitat_layer(G, G, LS, seed=[HSEED, 104729, 1000 + t, j])
        U = rho_s * U + np.sqrt(1 - rho_s ** 2) * Z; U = (U - U.mean()) / U.std()
    cors.append(float(np.corrcoef(U_start.ravel(), U.ravel())[0, 1]))
ok6 = abs(mm.std() / SD - 1) <= 0.03 and abs(lagM - RHO_M) <= 0.02 and abs(np.mean(cors) - RHO) <= 0.03
check("U6", ok6, f"over 20,000 sub-steps: shift sd {mm.std():.3f} ({SD}), correlation across one round {lagM:.3f} ({RHO_M}); "
                 f"layer correlation across one round, mean of 300: {np.mean(cors):.3f} ({RHO})")
ok, text = u7(v26)
check("U7", ok, text)

# ---------------------------------------------------------------- runner
if "--runner" in sys.argv:
    if not os.path.exists("run_all_rps.before_habitat_substeps.py"):
        sys.exit("STOP: run_all_rps.before_habitat_substeps.py missing; apply patch_run_all_rps_habitat_substeps.py first")
    ARGS = ["--pool", "22510000", "--skip-psd", "--skip-ode", "--tmax", "200", "--record", "100", "--no-movie",
            "--ibm-frac-multi", "0.05", "--ibm-window-steps", "100", "--ibm-richness-cap", "None", "--save-env-field",
            "--ibm-record-mode", "full", "--ibm-F-sat", "None", "--ibm-max-rounds", "None", "--ibm-max-attempts", "30",
            "--world-tag-extra", "t20", "--fp16-time-series", "--obs-budgets", "1,5", "--ibm-detection-threshold", "1",
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

    texts, ok_all = [], True
    for label, extra in (("no options", []),
                         ("--habitat-tau 5000 with shift", ["--habitat-tau", "5000", "--habitat-shift-sd", "1.0", "--habitat-shift-tau", "2000"])):
        with tempfile.TemporaryDirectory() as t_old, tempfile.TemporaryDirectory() as t_new:
            p_old, f_old = runner("run_all_rps.before_habitat_substeps.py", ARGS + extra, t_old)
            p_new, f_new = runner("run_all_rps.py", ARGS + extra, t_new)
            if p_old.returncode or p_new.returncode or len(f_old) != 1 or len(f_new) != 1:
                ok_all = False; texts.append(f"{label}: runs failed ({(p_old.stderr or p_new.stderr)[-300:]})"); continue
            same, added, diff = same_files(f_old[0], f_new[0])
            ok_all &= same; texts.append(f"{label}: identical {same} (keys added or lost {added}, differing {diff[:4]})")
    check("R1", ok_all, "; ".join(texts))
    with tempfile.TemporaryDirectory() as t:
        p, f = runner("run_all_rps.py", ARGS + ["--habitat-tau", "5000", "--habitat-substeps", "5"], t)
        if p.returncode or len(f) != 1:
            check("R2", False, f"run failed (exit {p.returncode}): {p.stderr[-400:]}")
        else:
            c = np.load(f[0], allow_pickle=True)
            tag = "_ht5000_hss5_" in os.path.basename(f[0])
            fr = frames(c); n = len(c["ASM_P_round"])
            last = all(np.array_equal(fr[t * 5 + 4][0], asm_p(c, t)[0]) and np.array_equal(fr[t * 5 + 4][1], asm_p(c, t)[1])
                       for t in range(n))
            ok = tag and int(c["HABITAT_substeps"]) == 5 and len(fr) == 5 * n and last and "ASM_F_shift" not in c.files
            check("R2", ok, f"tag {tag}, HABITAT_substeps {int(c['HABITAT_substeps'])}, {len(fr)} maps for {n} rounds, last map of "
                            f"each round = ASM_P {last}, no shift key without the shift {'ASM_F_shift' not in c.files}")
    refused = []
    for name, args, msg in (("no --habitat-tau", ARGS + ["--habitat-substeps", "5"], "needs --habitat-tau"),
                            ("3 with window 100", ARGS + ["--habitat-tau", "5000", "--habitat-substeps", "3"], "divides --ibm-window-steps"),
                            ("0", ARGS + ["--habitat-tau", "5000", "--habitat-substeps", "0"], "divides --ibm-window-steps")):
        with tempfile.TemporaryDirectory() as t:
            p, f = runner("run_all_rps.py", args, t)
            if p.returncode != 0 and msg in (p.stdout + p.stderr) and not f:
                refused.append(name)
    check("R3", len(refused) == 3, f"refused {len(refused)} of 3 with the right message: {', '.join(refused)}")

# ---------------------------------------------------------------- mutants
if "--mutants" in sys.argv:
    MUT = {
        "N1": ("U3", 'rho = hc["rho_s"]', 'rho = hc["rho"]'),
        "N2": ("U3", 'Z = habitat_layer(Ny, Nx, hc["ls"], seed=[hc["seed"], 104729, int(rnd_), int(j)])',
               'Z = habitat_layer(Ny, Nx, hc["ls"], seed=[hc["seed"], 104729, int(rnd_)])'),
        "N3": ("U4", 'model.r_field[...] = _like(F_big, model.r_field)', 'pass'),
        "N4": ("U4", '                model.run()\n                if hc is not None and hc["m"] is not None:',
               '                if j:\n                    model = IBMModel(r_big, C_big, r_field=F_big, nsteps=window_steps // n_piece, '
               'record_step=record_step, initial_N=_to_host(model.N), seed=new_seed(), **model_kw)\n'
               '                model.run()\n                if hc is not None and hc["m"] is not None:'),
        "N5": ("U3", 'if j > 0:                               # v2.6: the habitat moves; fields written into the model',
               'if j > 0 or rnd == start_round:   # mutant N5'),
        "N6": ("U5", 'on = Pj.any(axis=1)', 'on = np.ones(len(ids_big), bool)'),
    }
    FN = {"U3": u3, "U4": u4, "U5": u5}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (target, old, new) in MUT.items():
            if V26_SRC.count(old) != 1:
                check(name, False, f"mutation text found {V26_SRC.count(old)} times; mutant not built"); continue
            src = V26_SRC.replace(old, new)
            if name == "N4":
                src = src.replace("        def run_window():\n            nonlocal F_big", "        def run_window():\n            nonlocal F_big, model")
            path = os.path.join(tmp, f"asm_{name}.py")
            open(path, "w").write(src)
            sp = importlib.util.spec_from_file_location(f"asm_{name}", path)
            try:
                mm_ = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm_)
            except Exception as ex:
                check(name, False, f"mutant not built ({type(ex).__name__}: {ex})"); continue
            try:
                ok, text = FN[target](mm_)
            except Exception as ex:
                ok, text = False, f"raised {type(ex).__name__}: {ex}"
            check(name, not ok, f"caught by {target}: {text[:300]}")

assembly_utils.CONNECTANCE = CONN0
print("\nall passed" if not fails else f"\nFAILED: {', '.join(fails)}")
sys.exit(0 if not fails else 1)