"""
test_16_founding_best.py  (obj1_v2, LVMCM founding, 29 Oct 2026)

Question: does founding="best" (assembly v2.3, patch_run_all_rps_founding.py) found newcomers as LVMCM
does (O'Sullivan, LVMCM_src commit ba165f8: Metacommunity.cpp invaderSample lines 311-383, main.cpp
line 647), leave every random-founding world unchanged, and give rare-habitat specialists the start that
random founding denies them (founding_filter, 29 Oct)?

R  regression: founding "random" (the default, and given explicitly) returns every array of v2.2
   (new_code/assembly_stepwise_ibm_v2_2.py), and no new key: R1 field mode, R2 niche mode,
   R3 niche mode with a dispersal rate per species.
B  best founding; a spy records every IBM window (start counts, C, growth fields, end counts). Small
   niche assemblies on the CPU (contrast 0.25, k 0.5, d per species 1e-6 to 1e-2). Here g is recomputed
   from the spy's numbers, independently of the assembly code: g = r - C_new,res (N_res x BODY_MASS).
   B1 the world starts empty (as LVMCM): the first window holds only round-0 newcomers.
   B2 every window = the previous community (ASM_P_ids) + the entered newcomers of its round, with their
      own growth fields; residents start from the previous window's end counts; each newcomer starts as
      exactly one individual, in the cell of its highest g; at connectance 1 every C is 0.4 off the
      diagonal and 1 on it.
   B3 every entered newcomer has max g >= 0; every candidate that failed has max g < 0 (connectance 1,
      where its links are known: all 0.4); LT_g_max equals the recomputed max g (difference < 1e-9).
   B4 failed candidates never enter: in no window, not in ASM_P_ids or the final community; their row
      reads established 0, round in = round out, peak 0, attempt -1.
   B5 counting (LVMCM): attempts = entered newcomers = max_attempts; each round's ASM_attempts step =
      its entered newcomers; LT_attempt of the entered = 1..A in id order; ids 0..T-1 with T tested
      candidates; ASM_tested rises to T (less the stop round's tests).
   B6 cap and stop: with cost 1.6 (no species can grow anywhere) the assembly stops in round 0 after
      founding_cap x 1 candidates, runs no window, returns no species, found_saturated 1, round 0;
      with founding_cap 2, every round enters at most what it needs, tries at most 2 x needed, and
      tries exactly 2 x needed when it enters fewer.
   B7 refused: seed_size 5 with "best", founding "middle", founding_cap 0, 1.5 and None.
E  ecology, with a control
   E1 assembly level: 3 seeds, 100 attempts, 500-step windows, founding best against random (same seeds,
      same candidate niches by id). Fundamental range F = cells with r > 0. Pass (set before the run):
      the kept share per candidate with F 1-100 under best is at least 3 times that under random, best
      keeps at least 3 such species, and no candidate with F = 0 enters (best) or is kept (either).
   E2 mechanism: a planted specialist (habitat = the top cells of H, F 20 or fewer) added to a resident
      community, 40 repeats per arm, 300 steps: founded in its best cell (argmax g) it establishes in at
      least 50% of repeats; founded in a random cell (control) in at most 15%.
--runner (after patch_run_all_rps_founding.py --apply; run_all_rps.before_found.py kept):
   U1 no option: every key of the old runner's file is identical (runtime_s aside); only FOUND_rule
      ("random") added.
   U2 --founding best: tag "_fb_k2dt"; FOUND_* and LT_entered, LT_attempt, LT_g_max, ASM_tested saved and
      consistent; the life table and final community equal a direct call of stepwise_assembly_ibm with
      the same settings (so the option reaches the assembly).
   U3 refused: best with --seed-size 5, best with --resume, --founding-cap 0, --founding-cap 50 without
      best, --founding middle.
   U4 --founding-cap 3: tag "_fbc3_k2dt", FOUND_cap 3, and no round tries more than 3 x needed.
--mutants: five planted errors in copies of assembly v2.3 must each fail a B check: M1 random cell
   instead of the best cell, M2 the test ignores competition (g = r), M3 attempts count tested
   candidates, M4 failed candidates enter with no individual, M5 threshold g >= 0.2 instead of 0.

Run from obj1_v2/work, after copying new_code/assembly_stepwise_ibm.py (v2.3) in:
    USE_GPU=0 python3 ../tests/test_16_founding_best.py --runner --mutants | tee ../logs/test_16_founding_best_29oct.txt
"""
import contextlib, glob, importlib.util, io, os, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.getcwd())
os.environ.setdefault("USE_GPU", "0")
import numpy as np
import config, dispersal, environment as env, assembly_utils
import assembly_stepwise_ibm as v23
from models_ibm import IBMModel

fails = []
G, BM = config.NUM_PATCHES_Y, config.BODY_MASS
host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)


def check(name, cond, text):
    print(f"{name} {text} {'PASS' if cond else 'FAIL'}", flush=True)
    if not cond:
        fails.append(name)


if 'founding: str = "random"' not in open(v23.__file__).read():
    sys.exit("STOP: work/assembly_stepwise_ibm.py is not v2.3; copy ../new_code/assembly_stepwise_ibm.py in")
if not os.path.exists("../new_code/assembly_stepwise_ibm_v2_2.py"):
    sys.exit("STOP: ../new_code/assembly_stepwise_ibm_v2_2.py (the v2.2 copy) is missing")
spec = importlib.util.spec_from_file_location("asm_v2_2", "../new_code/assembly_stepwise_ibm_v2_2.py")
v22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v22)

dispersal.DISPERSAL_KERNEL, dispersal.DISPERSAL_A, dispersal.DISPERSAL_B, dispersal.DISPERSAL_RATE = "2dt", 1.0, 3.0, 1e-6
dispersal.LONG_DISTANCE_PROB = 0.0
CONN0, IS = assembly_utils.CONNECTANCE, assembly_utils.INTERACTION_STRENGTH
NICHE = dict(opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=env.NICHE_PEAK_MAX, cost=0.1, tradeoff=0.5)
DISP = dict(d_min=1e-6, d_max=1e-2)


def hab(seed):
    return env.habitat_layer(G, G, 3.0, seed=seed) * 0.25


def fld(i, H, niche, seed_field):
    kw = {k: v for k, v in niche.items() if k != "cost"}
    return env.niche_r(H, *env.niche_traits(int(i), seed=seed_field, **kw), niche["cost"])


def assemble(mod, *, conn=1.0, window=300, attempts=60, seed=4311, **kw):
    assembly_utils.CONNECTANCE = conn
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return mod.stepwise_assembly_ibm(window_steps=window, record_step=50, seed_size=kw.pop("seed_size", 1), F_sat=None,
                                             max_rounds=None, max_attempts=attempts, richness_cap=None, frac_multi=0.05,
                                             seed=seed, detection_threshold=1, **kw)
    finally:
        assembly_utils.CONNECTANCE = CONN0


def eq(a, b):
    try:
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    except TypeError:
        return np.array_equal(np.asarray(a), np.asarray(b))


def same_out(o1, o2):
    return (all(eq(x, y) for x, y in zip(o1[:3], o2[:3])) and set(o1[3]) == set(o2[3])
            and all(eq(o1[3][k], o2[3][k]) for k in o1[3]))


# ---------------------------------------------------------------- R regression
H1 = hab(104311)
for name, kw in (("R1", dict(conn=0.4, length_scale=5.0, var_r=0.002, seed_field=789)),
                 ("R2", dict(habitat=H1, niche=NICHE, seed_field=204311)),
                 ("R3", dict(habitat=H1, niche=NICHE, seed_field=204311, disp=DISP))):
    o_old = assemble(v22, attempts=40, window=200, **kw)
    o_def = assemble(v23, attempts=40, window=200, **kw)
    o_ran = assemble(v23, attempts=40, window=200, founding="random", **kw)
    check(name, same_out(o_old, o_def) and same_out(o_old, o_ran),
          f"{'field mode' if name == 'R1' else 'niche mode' + (' with d per species' if name == 'R3' else '')}: founding random "
          f"(default and explicit) returns r, C, N and all {len(o_old[3])} extra arrays of v2.2, no new key ({len(o_old[0])} species):")


# ---------------------------------------------------------------- B best founding
class Spy(IBMModel):
    log = []

    def run(self):
        rec = dict(N0=host(self.N).copy(), C=host(self.C).copy(), F=host(self.r_field).copy())
        out = super().run()
        rec["N1"] = host(self.N).copy()
        Spy.log.append(rec)
        return out


def spy_run(mod, **kw):
    Spy.log = []
    orig = mod.IBMModel; mod.IBMModel = Spy
    try:
        return assemble(mod, **kw), list(Spy.log)
    finally:
        mod.IBMModel = orig


def bcheck(mod, *, conn=1.0, attempts=60, cap=100, seed=4311, seed_field=204311, H=H1, niche=NICHE, window=300):
    """B1-B5 on one assembly; returns {check: (ok, text)}."""
    (r, C, N, x), W = spy_run(mod, conn=conn, attempts=attempts, seed=seed, window=window, habitat=H, niche=niche,
                              seed_field=seed_field, disp=DISP, founding="best", founding_cap=cap)
    f = lambda i: fld(i, H, niche, seed_field)
    ids, rin, est, rout, peak = (np.asarray(x[k]) for k in ("LT_id", "LT_round_in", "LT_established", "LT_round_out", "LT_peak_cells"))
    ent, att, gmx = np.asarray(x["LT_entered"]).astype(bool), np.asarray(x["LT_attempt"]), np.asarray(x["LT_g_max"])
    pos = {int(i): k for k, i in enumerate(ids)}
    rounds, off, pid = np.asarray(x["ASM_round"]), np.asarray(x["ASM_P_offsets"]), np.asarray(x["ASM_P_ids"])
    comm = lambda k: [] if k < 0 else pid[off[k]:off[k + 1]].tolist()
    new_of = lambda rnd: sorted(ids[(rin == rnd) & ent].tolist())
    sat, stop = int(x["found_saturated"]), int(x["found_stop_round"])
    out = {}
    out["B1"] = (len(W) == len(rounds) > 0 and rounds[0] == 0 and W[0]["N0"].shape[0] == len(new_of(0)) and (rin >= 0).all(),
                 f"{len(W)} windows; first window holds {W[0]['N0'].shape[0] if W else -1} species, all round-0 newcomers:")
    ok_cnt = ok_fld = ok_res = ok_one = ok_cell = ok_C = ok_gent = ok_gfail = True
    gerr, n_new, n_fail = 0.0, 0, 0

    def fail_g(i, Bres):                                 # connectance 1: every link of a candidate is IS
        return f(i) - (np.full(Bres.shape[0], IS) @ Bres).reshape(G, G)
    for k, rnd in enumerate(rounds):
        prev, new = comm(k - 1), new_of(rnd)
        S, nres = W[k]["N0"].shape[0], len(prev)
        if S != nres + len(new):
            ok_cnt = False; continue
        lst = prev + new
        ok_fld &= np.allclose(W[k]["F"], np.stack([f(i) for i in lst]), rtol=0, atol=1e-12)
        N0 = W[k]["N0"]; Bres = N0[:nres].reshape(nres, G * G).astype(float) * BM
        if k > 0:
            plist = comm(k - 2) + new_of(rounds[k - 1])
            ok_res &= np.array_equal(N0[:nres], W[k - 1]["N1"][[plist.index(i) for i in prev]])
        if conn == 1.0:
            ok_C &= np.allclose(W[k]["C"], IS + (1 - IS) * np.eye(S))
        for j, i in enumerate(new):
            g = W[k]["F"][nres + j] - (W[k]["C"][nres + j, :nres] @ Bres).reshape(G, G)
            ok_one &= int(N0[nres + j].sum()) == 1
            ok_cell &= int(N0[nres + j].flat[int(np.argmax(g))]) == 1
            ok_gent &= g.max() >= 0
            gerr = max(gerr, abs(g.max() - gmx[pos[i]])); n_new += 1
        if conn == 1.0:
            for i in ids[(rin == rnd) & ~ent].tolist():
                g = fail_g(i, Bres)
                ok_gfail &= g.max() < 0; gerr = max(gerr, abs(g.max() - gmx[pos[i]])); n_fail += 1
    if conn == 1.0 and sat:                              # the stop round ran no window: residents = final community
        Bres = np.asarray(N).reshape(len(r), G * G).astype(float) * BM
        for i in ids[(rin == stop) & ~ent].tolist():
            g = fail_g(i, Bres)
            ok_gfail &= g.max() < 0; gerr = max(gerr, abs(g.max() - gmx[pos[i]])); n_fail += 1
    out["B2"] = (ok_cnt and ok_fld and ok_res and ok_one and ok_cell and ok_C,
                 f"windows = previous community + entered newcomers {ok_cnt}, own fields {ok_fld}, residents from the previous window's "
                 f"end {ok_res}, one individual each {ok_one}, in the cell of highest g {ok_cell}, C as drawn {ok_C} ({n_new} newcomers):")
    out["B3"] = (ok_gent and ok_gfail and gerr < 1e-9,
                 f"entered max g >= 0 {ok_gent}; failed max g < 0 {ok_gfail} ({n_fail} checked); LT_g_max = recomputed (max diff {gerr:.1e}):")
    fin = set(np.asarray(x["species_id"]).tolist()); entered = set(ids[ent].tolist())
    ok4 = (fin <= entered and set(pid.tolist()) <= entered and (est[~ent] == 0).all() and (rin[~ent] == rout[~ent]).all()
           and (peak[~ent] == 0).all() and (att[~ent] == -1).all())
    out["B4"] = (ok4 and ok_cnt and ok_fld, f"{(~ent).sum()} failed candidates: in no window, not in ASM_P_ids or the final community, "
                 f"rows established 0, round in = out, peak 0, attempt -1 {ok4}:")
    A = int(x["attempts_total"]); steps = np.diff(np.r_[0, np.asarray(x["ASM_attempts"])])
    ok_steps = len(steps) == len(rounds) and all(int(s) == len(new_of(rnd)) for s, rnd in zip(steps, rounds))
    att_ok = np.array_equal(att[ent], np.arange(1, A + 1)) and np.array_equal(np.sort(att[ent]), att[ent])
    T = int(x["tested_total"]); tst = np.asarray(x["ASM_tested"])
    stop_tests = int(((rin == stop) & ~ent).sum()) if sat else 0
    ok5 = (A == int(ent.sum()) and (sat or A == attempts) and ok_steps and att_ok and np.array_equal(ids, np.arange(T))
           and (np.diff(tst) >= 0).all() and int(tst[-1]) == T - stop_tests)
    out["B5"] = (ok5, f"attempts {A} = entered {int(ent.sum())} (asked {attempts}); round steps = entered {ok_steps}; attempts 1..A in id order "
                 f"{att_ok}; ids 0..{T - 1} for {T} tested; ASM_tested ends at {int(tst[-1]) if len(tst) else -1}:")
    out["_x"] = x
    return out


res = bcheck(v23)
for k in ("B1", "B2", "B3", "B4", "B5"):
    check(k, *res[k])
x1 = res["_x"]
res04 = bcheck(v23, conn=0.4, attempts=40, seed=4312)
check("B2b", res04["B2"][0] and res04["B3"][0] and res04["B5"][0],
      f"connectance 0.4 (links drawn at random, so only entered newcomers are recomputed): {res04['B2'][1].rstrip(':')}; "
      f"entered max g >= 0 and LT_g_max = recomputed {res04['B3'][0]}; counting {res04['B5'][0]}:")

# B6 cap and stop
(r0, C0, N0, x0), W0 = spy_run(v23, habitat=H1, niche=dict(NICHE, cost=1.6), seed_field=204311, founding="best", founding_cap=7)
ok_a = (len(r0) == 0 and len(W0) == 0 and int(x0["found_saturated"]) == 1 and int(x0["found_stop_round"]) == 0
        and int(x0["attempts_total"]) == 0 and int(x0["tested_total"]) == 7 and len(x0["LT_id"]) == 7 and int(np.sum(x0["LT_entered"])) == 0)
(r2, C2, N2, x2), W2 = spy_run(v23, habitat=H1, niche=NICHE, seed_field=204311, founding="best", founding_cap=2, attempts=60)
rin2, ent2 = np.asarray(x2["LT_round_in"]), np.asarray(x2["LT_entered"]).astype(bool)
rd2, off2, pid2 = np.asarray(x2["ASM_round"]), np.asarray(x2["ASM_P_offsets"]), np.asarray(x2["ASM_P_ids"])
ok_b, short, A2 = True, 0, 0
all_rounds = list(rd2) + ([int(x2["found_stop_round"])] if int(x2["found_saturated"]) else [])
for k, rnd in enumerate(all_rounds):
    gam = 0 if k == 0 else int(off2[k] - off2[k - 1])
    need = min(1 + int(0.05 * gam), 60 - A2)
    e_k, t_k = int(((rin2 == rnd) & ent2).sum()), int((rin2 == rnd).sum())
    ok_b &= e_k <= need and t_k <= 2 * need and (e_k == need or t_k == 2 * need)
    short += e_k < need; A2 += e_k
check("B6", ok_a and ok_b and len(all_rounds) > 0,
      f"cost 1.6: stopped in round 0 after 7 candidates, no window, no species, saturated {ok_a}; cap 2: {len(all_rounds)} rounds keep "
      f"entered <= needed, tried <= 2 x needed, tried = 2 x needed when short {ok_b} ({short} short rounds, saturated {int(x2['found_saturated'])}):")

bad = []
for kw in (dict(founding="best", seed_size=5), dict(founding="middle"), dict(founding="best", founding_cap=0),
           dict(founding="best", founding_cap=1.5), dict(founding="best", founding_cap=None)):
    try:
        assemble(v23, attempts=3, window=20, habitat=H1, niche=NICHE, seed_field=204311, **kw); bad.append(False)
    except ValueError:
        bad.append(True)
check("B7", all(bad), f"refused: seed_size 5 with best, founding 'middle', cap 0, 1.5, None {bad}:")


# ---------------------------------------------------------------- E ecology
def Fsize(x, H, niche):
    r = (np.asarray(x["LT_peak"])[:, None, None] * np.exp(-(H[None] - np.asarray(x["LT_opt"])[:, None, None]) ** 2
                                                          / (2 * np.asarray(x["LT_w"])[:, None, None] ** 2)) - niche["cost"])
    return (r > 0).reshape(len(r), -1).sum(1)


pool = {m: np.zeros((3, 2), int) for m in ("random", "best")}       # rows F 0, F 1-100, F 101+; cols candidates, kept
f0_ok, rows = True, []
for s in (4321, 4322, 4323):
    Hs = hab(100000 + s)
    for m in ("random", "best"):
        _, _, _, xe = assemble(v23, conn=1.0, window=500, attempts=100, seed=s, habitat=Hs, niche=NICHE, seed_field=200000 + s,
                               disp=DISP, founding=m)
        F = Fsize(xe, Hs, NICHE); rin = np.asarray(xe["LT_round_in"]); kept = np.asarray(xe["LT_established"]).astype(bool)
        cand = rin >= 0
        for b, sel in enumerate((F == 0, (F >= 1) & (F <= 100), F > 100)):
            pool[m][b] += [int((cand & sel).sum()), int((cand & sel & kept).sum())]
        # candidates only: the founder of a random-founding world (round -1) is a resident at the start by definition,
        # also when it can grow nowhere (first run of this test: seeds 4321 and 4322, founder F = 0)
        f0_ok &= not (cand & kept & (F == 0)).any()
        if m == "best":
            f0_ok &= not (cand & np.asarray(xe["LT_entered"]).astype(bool) & (F == 0)).any()
        rows.append(f"seed {s} {m}: {int(cand.sum())} candidates, {int(xe['attempts_total'])} attempts, {int((cand & kept).sum())} kept, "
                    f"{len(xe['species_id'])} species at the end")
share = {m: pool[m][1, 1] / max(pool[m][1, 0], 1) for m in pool}
print("\n".join("  " + t for t in rows))
for m in pool:
    print(f"  {m}: kept per candidate, F 0: {pool[m][0, 1]}/{pool[m][0, 0]}, F 1-100: {pool[m][1, 1]}/{pool[m][1, 0]} ({share[m]:.1%}), "
          f"F 101+: {pool[m][2, 1]}/{pool[m][2, 0]} ({pool[m][2, 1] / max(pool[m][2, 0], 1):.1%})")
check("E1", share["best"] >= 3 * share["random"] and pool["best"][1, 1] >= 3 and f0_ok,
      f"F 1-100 kept per candidate: best {share['best']:.1%} against random {share['random']:.1%} (needs 3x and 3 or more kept: "
      f"{pool['best'][1, 1]}); no F = 0 candidate entered or kept {f0_ok}:")

# E2 planted specialist, founded in its best cell against a random cell
s = 4321; Hs = hab(100000 + s)
r_res, C_res, N_res, x_res = assemble(v23, conn=1.0, window=500, attempts=100, seed=s, habitat=Hs, niche=NICHE, seed_field=200000 + s)
F_res = np.asarray(x_res["r_field"])
opt, w = float(Hs.max()), 0.2
while True:
    r_sp = 1.5 * np.exp(-(Hs - opt) ** 2 / (2 * w ** 2)) - 0.1
    if (r_sp > 0).sum() <= 20:
        break
    w *= 0.8
Fsp = int((r_sp > 0).sum())
g_sp = v23.invasion_growth(r_sp, np.full(len(r_res), IS), N_res)
best_cell = int(np.argmax(g_sp))
Cb = np.full((len(r_res) + 1,) * 2, IS); Cb[:-1, :-1] = C_res; np.fill_diagonal(Cb, 1.0)
rngE = np.random.default_rng(77)


def plant(cell, seed):
    N0 = np.concatenate([N_res, np.zeros((1, G, G), int)]); N0[-1].flat[cell] = 1
    m = IBMModel(np.r_[r_res, 1.0], Cb, initial_N=N0, r_field=np.concatenate([F_res, r_sp[None]]), nsteps=300, record_step=50,
                 record_mode="none", seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        m.run()
    return bool((host(m.N)[-1] >= 1).any())


assembly_utils.CONNECTANCE = 1.0
est_best = sum(plant(best_cell, 1000 + k) for k in range(40))
est_rand = sum(plant(int(rngE.integers(G * G)), 2000 + k) for k in range(40))
assembly_utils.CONNECTANCE = CONN0
check("E2", est_best >= 20 and est_rand <= 6,
      f"planted specialist (width {w:.3f}, F {Fsp} cells, landing chance {Fsp / G / G:.1%}, g in its best cell {g_sp.max():.2f}) in a community "
      f"of {len(r_res)}: established {est_best}/40 from its best cell, {est_rand}/40 from a random cell (control):")

# ---------------------------------------------------------------- runner
if "--runner" in sys.argv:
    before = "run_all_rps.before_found.py"
    if not os.path.exists(before) or "# obj1_v2 founding" not in open("run_all_rps.py").read():
        sys.exit("STOP: --runner needs patch_run_all_rps_founding.py --apply (run_all_rps.before_found.py kept)")
    base = ["--pool", "40", "--skip-psd", "--skip-ode", "--tmax", "400", "--record", "100", "--no-movie", "--ibm-frac-multi", "0.05",
            "--ibm-window-steps", "200", "--ibm-richness-cap", "None", "--save-env-field", "--ibm-record-mode", "full",
            "--ibm-F-sat", "None", "--ibm-max-rounds", "None", "--C-topk", "16", "--coocc-sample", "64", "--world-tag-extra", "t16",
            "--obs-budgets", "1,5", "--ibm-detection-threshold", "1", "--ldd-prob", "0", "--ibm-max-attempts", "30",
            "--disp-kernel", "2dt", "--disp-a", "1", "--disp-b", "3", "--seed-size", "1", "--sim-seed", "4201",
            "--env-mode", "niche", "--habitat-seed", "104201", "--env-seed-field", "204201", "--niche-tradeoff", "0.5",
            "--habitat-contrast", "0.25", "--connectance", "1", "--disp-rate", "1e-6", "--disp-per-species", "1e-6", "1e-2"]
    env0 = dict(os.environ, USE_GPU="0")

    def run(script, args, folder):
        for f in glob.glob(f"{folder}/results/data/*_training.npz"): os.remove(f)
        p = subprocess.run([sys.executable, script] + args, cwd=folder, env=env0, capture_output=True, text=True)
        fs = sorted(glob.glob(f"{folder}/results/data/*_training.npz"))
        return p.returncode, p.stdout + p.stderr, (np.load(fs[0], allow_pickle=True) if len(fs) == 1 else None), fs

    with tempfile.TemporaryDirectory() as t:
        for f in glob.glob("*.py"):
            shutil.copy(f, t)
        ra, la, za, _ = run(before, base, t)
        rb, lb, zb, _ = run("run_all_rps.py", base, t)
        if ra or rb or za is None or zb is None:
            check("U1", False, f"runs failed (exit {ra}, {rb}):\n{la[-500:]}\n{lb[-500:]}")
        else:
            ok = (all(eq(za[k], zb[k]) for k in za.files if k != "runtime_s") and set(zb.files) - set(za.files) == {"FOUND_rule"}
                  and str(zb["FOUND_rule"][0]) == "random")
            check("U1", ok, f"no option: the {len(za.files)} keys of the old runner identical, only FOUND_rule ('random') added:")
        rc, lc, zc, fc = run("run_all_rps.py", base + ["--founding", "best"], t)
        if rc or zc is None:
            check("U2", False, f"best run failed (exit {rc}):\n{lc[-800:]}")
        else:
            name = os.path.basename(fc[0]); ent = np.asarray(zc["LT_entered"]).astype(bool); A = int(zc["attempts_total"])
            ok_keys = (str(zc["FOUND_rule"][0]) == "best" and int(zc["FOUND_cap"]) == 100 and int(zc["FOUND_tested_total"]) == len(zc["LT_id"])
                       and A == int(ent.sum()) == 30 and np.array_equal(zc["LT_attempt"][ent], np.arange(1, A + 1))
                       and (zc["LT_g_max"][ent] >= 0).all() and (zc["LT_g_max"][~ent] < 0).all()
                       and set(np.asarray(zc["species_id"]).tolist()) <= set(np.asarray(zc["LT_id"])[ent].tolist())
                       and "cn1p0_dps1em06-0p01_fb_k2dt" in name)
            Hr = env.habitat_layer(G, G, 3.0, seed=104201) * 0.25
            assembly_utils.CONNECTANCE = 1.0
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, _, _, xd = v23.stepwise_assembly_ibm(window_steps=200, record_step=100, seed_size=1, F_sat=None, max_rounds=None,
                                                            max_attempts=30, richness_cap=None, frac_multi=0.05, seed=4201,
                                                            detection_threshold=1, habitat=Hr, niche=NICHE, seed_field=204201,
                                                            disp=DISP, founding="best", founding_cap=100)
            finally:
                assembly_utils.CONNECTANCE = CONN0
            ok_same = all(eq(zc[k], xd[k]) for k in ("LT_id", "LT_entered", "LT_established", "LT_round_in", "LT_attempt", "species_id"))
            check("U2", ok_keys and ok_same, f"file {name}: FOUND_* and life-table keys saved and consistent {ok_keys}; life table and "
                  f"final community = direct stepwise_assembly_ibm call {ok_same}:")
        badr = []
        for extra, word in ((["--founding", "best", "--seed-size", "5"], "--seed-size 1"), (["--founding", "best", "--resume", "x.npz"], "--resume"),
                            (["--founding", "best", "--founding-cap", "0"], "--founding-cap"), (["--founding-cap", "50"], "--founding-cap"),
                            (["--founding", "middle"], "--founding")):
            args = list(base)
            if "--seed-size" in extra:
                k = args.index("--seed-size"); del args[k:k + 2]
            r_, l_, _, _ = run("run_all_rps.py", args + extra, t)
            badr.append(r_ != 0 and word in l_)
        check("U3", all(badr), f"refused: best with seed size 5, best with --resume, cap 0, cap 50 without best, founding middle {badr}:")
        rd, ld, zd, fd = run("run_all_rps.py", base + ["--founding", "best", "--founding-cap", "3"], t)
        if rd or zd is None:
            check("U4", False, f"cap 3 run failed (exit {rd}):\n{ld[-800:]}")
        else:
            rin4, ent4 = np.asarray(zd["LT_round_in"]), np.asarray(zd["LT_entered"]).astype(bool)
            off4 = np.asarray(zd["ASM_P_offsets"]); rr = list(np.asarray(zd["ASM_round"]))
            rr += [int(zd["FOUND_stop_round"])] if int(zd["FOUND_saturated"]) else []
            ok4, A4 = True, 0
            for k, rnd in enumerate(rr):
                gam = 0 if k == 0 else int(off4[k] - off4[k - 1])
                need = min(1 + int(0.05 * gam), 30 - A4)
                ok4 &= int((rin4 == rnd).sum()) <= 3 * need; A4 += int(((rin4 == rnd) & ent4).sum())
            check("U4", ok4 and int(zd["FOUND_cap"]) == 3 and "_fbc3_k2dt" in os.path.basename(fd[0]),
                  f"cap 3: tag {'_fbc3_k2dt' in os.path.basename(fd[0])}, FOUND_cap {int(zd['FOUND_cap'])}, no round tried more than 3 x needed {ok4}:")

# ---------------------------------------------------------------- mutants
if "--mutants" in sys.argv:
    src = open(v23.__file__).read()
    MUT = {"M1 random cell": [("found_cells.append(int(np.argmax(g)))", "found_cells.append(int(rng.integers(Ny * Nx)))")],
           "M2 test ignores competition": [("g = invasion_growth(field(sid), row[:len(r)], N)", "g = invasion_growth(field(sid), 0 * row[:len(r)], N)")],
           "M3 attempts count tested": [("        attempts += n_cand\n", "        attempts += (tried if best else n_cand)\n")],
           "M4 failed enter with none": [("lt[sid] = [int(rnd), 0, int(rnd), 0]; lt_fail.add(sid)",
                                          "r_big, C_big = expand_RC(r_big, C_big, base_r, row, col); new_list.append(sid); found_cells.append(-1)"),
                                         ("                N_big[len(r) + add].flat[c] = 1\n",
                                          "                if c >= 0: N_big[len(r) + add].flat[c] = 1\n")],
           "M5 threshold 0.2": [("if lt_g[sid] >= 0:", "if lt_g[sid] >= 0.2:")]}
    with tempfile.TemporaryDirectory() as t:
        for k, (name, subs) in enumerate(MUT.items()):
            s2 = src
            for a, b in subs:
                if s2.count(a) != 1:
                    sys.exit(f"STOP: mutant {name}: anchor found {s2.count(a)} times")
                s2 = s2.replace(a, b)
            p = os.path.join(t, f"asm_mut{k}.py"); open(p, "w").write(s2)
            sp = importlib.util.spec_from_file_location(f"asm_mut{k}", p); mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm)
            try:
                rm = bcheck(mm)
                caught = [c for c in ("B1", "B2", "B3", "B4", "B5") if not rm[c][0]]
            except Exception as e:                       # a crash also stops the error from passing unseen
                caught = [f"crash: {type(e).__name__}"]
            check(name.split()[0], bool(caught), f"mutant '{name}' caught by {caught}:")

print(f"\n{'all passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)