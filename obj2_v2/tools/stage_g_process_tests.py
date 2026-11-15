#!/usr/bin/env python3
"""
stage_g_process_tests.py  (obj2_v2, stage G step G4b, 11 Nov 2026; G4 rules signed off by Sara on 11 Nov, the three
open points (torus distance for T2, cyclic control, undefined statistics) on 11 Nov; G4a passed on her data 11 Nov)

Question: do the model's 6-year maps behave like the simulated process, judged against a control that has the same
yearly maps with the time order broken? Run on the 16,347 frozen real_val stacks, both seeds, B3 half-blank reported.

Statistics (one value per 6-year map stack; the same code for the truth, every sample and every control sample)
    T1  year-to-year change: for each of the 5 transitions with an occupied cell in year t or t+1,
        (cells gained + cells lost) / (occupied cells in year t + in year t+1); the mean over those transitions.
    T2  colonisation distance: for each cell gained between year t and t+1 while year t has an occupied cell, the
        distance between cell centres to the nearest cell occupied in year t; the mean over all such gains of the
        stack. Tested on the 20 x 20 torus (the obj1 grid; obj1_v2 dispersal.py v2), plain distance reported
        (Sara, 11 Nov: BNM windows have edges).
    T3  losses where the habitat fell: for each transition, among cells occupied in year t, the share lost where the
        species' growth rate fell (r_t+1 < r_t) minus the share lost where it rose or stayed; defined for a transition
        when both groups hold an occupied cell; the mean over those transitions. r_t = niche_r(H_t, niche of the species),
        H_t the habitat of year t rebuilt by the G0 rule (pieces M-6 .. M-1 of the stack's round), the niche rebuilt
        from the species id by the G4a call (Sara's niche_traits and niche_r from obj1_v2/work/environment.py, every
        setting from the obj1 file).
Control: year t of control sample k is year t of sample (k + 7 t) mod 50 (Sara, 11 Nov): the 6 years of a control sample
come from 6 different samples, and each year's set of 50 maps is unchanged. Built per seed (and for the pooled line).
Undefined values (Sara, 11 Nov): per stack and method, the mean over the samples that define the statistic; a stack
enters a test for a seed only if the truth, the model and the control all define it; stacks dropped are counted by reason.
Per stack: error = |mean over samples - truth|; coverage = truth inside the 5th-95th percentile of the defined samples.

Pass rule (per seed and test, T1, T2 torus, T3): d = model error - control error per entered stack; mean of d per true
class 1 .. 51+ (as in F5), then the mean over classes; PASS when the upper end of its 95% interval from resampling the 20
communities (2,000 draws, seed 31, stage_f_check.class_boot) is below 0, else FAIL. Reported, not tested: true class 0,
every class, B3 half-blank, T2 plain, the mean of the two seeds, the pooled line (first 25 samples of each seed), and the
relative gain 1 - (class-balanced model error) / (class-balanced control error), because with 16,347 stacks a difference
too small to matter can still be resolved (the planted command line passed T1 with a gain of 0.0%). A second verdict
also needs a gain of at least MIN_GAIN (10%); which verdict binds is Sara's choice, made before her run.

Order (leak rule): inputs checked by md5 (truth files by their file md5 only); then only stack_species and stack_round are
read from the truth files (ids, as in G4a); habitat and growth rates rebuilt; every sample and control statistic computed,
written and md5-logged; only then are the true maps opened.

Gates
    M1  inputs: samples_k50.npz and best.pt of each run = g1_report.json; maps_B3_halfblank.npz = g2_report.json and made
        from this frozen set; all maps cover the frozen stacks in order with the G1 layout; truth files = manifest
        truth_md5; obj1 files = manifest source_md5
    M2  control: every year's multiset of 50 maps unchanged in every stack; the 6 years of every control sample from 6
        different samples
    M3  T3 inputs: the rebuilt last piece of every round used = ENV_habitat_t of that round (1e-5, G0 rule); the niche call
        reproduces each file's saved niches and niche_r on the last round's habitat = ENV_r_field (1e-6, G4a rules)
    M4  leak: the statistics file is unchanged after the true maps are opened, and the statistics of 300 stacks recomputed
        with every stack's true maps swapped for another stack's equal the logged ones
    M5  statistic code: T1, T2 (torus and plain) and T3 of the vectorised code equal a plain loop code on 200 real stacks
        (5 samples of seed 1 and the truth)
    real_eval is refused; the output folder must be new.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU)
    python3 ../tools/stage_g_process_tests.py --self-test | tail -1
    python3 -u ../tools/stage_g_process_tests.py --data ../data --runs ../runs/f_seed1 ../runs/f_seed2 \\
        --g2 ../data/stage_g2_baselines_v2 --obj1-work ../../obj1_v2/work \\
        --files "../../obj1_v2/work/results/stage2a/real_val/*_training.npz" --out ../data/stage_g4b_process \\
        > ../logs/stage_g4b_process.txt 2>&1
Needs the same tool folder as G3 (stage_g_baselines.py and the stage tools it loads), stage_g_habitat_check.py and
stage_g_niche_check.py.
"""
import argparse, csv, glob, importlib.util, inspect, json, math, os, subprocess, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g_process_tests.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


GB = _load("stage_g_baselines")
GS, CK = GB.GS, GB.CK
HR, D, X = CK.HR, CK.D, CK.X
HC = _load("stage_g_habitat_check")
NC = _load("stage_g_niche_check")
G, W = X.G, X.WINDOW
TRUE = D.TRUE
SCORE = [1, 2, 3, 4, 5, 6]
K, SHIFT = 50, 7
TESTS = ("T1", "T2", "T2plain", "T3")
TESTED = ("T1", "T2", "T3")
TOL_HAB, TOL_R = 1e-5, 1e-6
N_LEAK, N_LOOP = 300, 200
MIN_GAIN = 0.10                                       # second verdict, if Sara chooses it before the run
SPARSE_MAX = 32                                       # T2: prev maps with at most this many cells use all pairs
BROKEN = os.environ.get("STAGE_G4B_BROKEN", "")       # self-test only: plants one bug so its gate must fail
_LEAK = None                                          # used only by the planted leak bug


def control_index(k=K, shift=SHIFT):
    """IDX[k, t] = sample that gives year t of control sample k."""
    idx = (np.arange(k)[:, None] + shift * np.arange(W)[None, :]) % k
    if BROKEN == "control_drop":
        idx = (np.arange(k)[:, None] + shift * np.arange(W)[None, :]) % (k - 1)
    return idx


def unpack6(pk):
    """(..., 300) packed G1 layout -> (..., 6, G, G) bool."""
    return np.unpackbits(pk, axis=-1, count=W * G * G).astype(bool).reshape(pk.shape[:-1] + (W, G, G))


# ---------------------------------------------------------------- statistics
def _offsets(wrap):
    r = G // 2 if wrap else G - 1
    o = [(dy, dx, math.hypot(dy, dx)) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if (dy, dx) != (0, 0)]
    o.sort(key=lambda t: (t[2], t[0], t[1]))
    return o


OFFS = {True: _offsets(True), False: _offsets(False)}


def _shift(P, dy, dx, wrap):
    if wrap or BROKEN == "t2_plain":
        return np.roll(P, (dy, dx), axis=(1, 2))
    S = np.zeros_like(P)
    S[:, max(dy, 0):G + min(dy, 0), max(dx, 0):G + min(dx, 0)] = P[:, max(-dy, 0):G + min(-dy, 0), max(-dx, 0):G + min(-dx, 0)]
    return S


def nearest(prev, gain, wrap):
    """Sum and count, per map, of the distance from each gained cell to the nearest cell of prev (maps with an empty
    prev give no distance). Maps whose prev holds at most SPARSE_MAX cells: all pairs, minimum taken. Other maps:
    offsets in order of distance, so each gained cell takes its smallest distance (dense prev ends in a few offsets)."""
    B = len(prev)
    npv = prev.sum((1, 2))
    rem = gain & (npv > 0)[:, None, None]
    dsum, dcnt = np.zeros(B), np.zeros(B)
    sp = np.nonzero((npv > 0) & (npv <= SPARSE_MAX) & rem.any((1, 2)))[0]
    if len(sp):
        pi, py, px = np.nonzero(prev[sp])
        cnt = np.bincount(pi, minlength=len(sp))
        pos = np.arange(len(pi)) - np.concatenate([[0], np.cumsum(cnt)[:-1]])[pi]
        Py = np.zeros((len(sp), SPARSE_MAX), np.int64); Px = np.zeros_like(Py)
        Py[pi, pos], Px[pi, pos] = py, px
        gi_all, gy_all, gx_all = np.nonzero(rem[sp])
        for a in range(0, len(gi_all), 200000):                      # blocks of gained cells bound the memory
            gi, gy, gx = gi_all[a:a + 200000], gy_all[a:a + 200000], gx_all[a:a + 200000]
            dy, dx = np.abs(gy[:, None] - Py[gi]), np.abs(gx[:, None] - Px[gi])
            if wrap or BROKEN == "t2_plain":
                dy, dx = np.minimum(dy, G - dy), np.minimum(dx, G - dx)
            d = np.sqrt((dy * dy + dx * dx).astype(np.float64))
            d[np.arange(SPARSE_MAX)[None, :] >= cnt[gi][:, None]] = np.inf
            dsum[sp] += np.bincount(gi, weights=d.min(1), minlength=len(sp))
            dcnt[sp] += np.bincount(gi, minlength=len(sp))
        rem[sp] = False
    act = np.nonzero(rem.any((1, 2)))[0]
    for dy, dx, d in OFFS[wrap]:
        if len(act) == 0:
            break
        R = rem[act]
        hit = R & _shift(prev[act], dy, dx, wrap)
        h = hit.sum((1, 2))
        dsum[act] += d * h; dcnt[act] += h
        R &= ~hit; rem[act] = R
        act = act[R.any((1, 2))]
    return dsum, dcnt


def _nanmean(v, axis):
    ok = ~np.isnan(v)
    n = ok.sum(axis)
    s = np.where(ok, v, 0.0).sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def stats(A, fell, idx=None):
    """A (c, k, 6, G, G) bool, fell (c, 5, G, G) bool -> dict test -> (c, k) float, nan where undefined."""
    if BROKEN == "leak" and _LEAK is not None and idx is not None:
        A = A.copy(); A[:, :, 0] |= _LEAK[idx][:, None, 0]
    c, k = A.shape[:2]
    prev, nxt = A[:, :, :-1], A[:, :, 1:]
    gain, lost = nxt & ~prev, prev & ~nxt
    occ = A.sum((-1, -2)).astype(np.float64)
    g, l = gain.sum((-1, -2)), lost.sum((-1, -2))
    den = occ[..., :-1] + occ[..., 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        t1 = np.where(den > 0, (g + l) / np.maximum(den, 1), np.nan)
    out = dict(T1=_nanmean(t1, -1))
    P, Gn = prev.reshape(-1, G, G), gain.reshape(-1, G, G)
    for name, wrap in (("T2", True), ("T2plain", False)):
        ds, dc = nearest(P, Gn, wrap)
        ds, dc = ds.reshape(c, k, W - 1).sum(-1), dc.reshape(c, k, W - 1).sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(dc > 0, ds / np.maximum(dc, 1), np.nan)
    F = fell[:, None]
    nf, lf = (prev & F).sum((-1, -2)), (lost & F).sum((-1, -2))
    nr, lr = (prev & ~F).sum((-1, -2)), (lost & ~F).sum((-1, -2))
    with np.errstate(invalid="ignore", divide="ignore"):
        d3 = np.where((nf > 0) & (nr > 0), lf / np.maximum(nf, 1) - lr / np.maximum(nr, 1), np.nan)
    out["T3"] = _nanmean(d3, -1)
    return out


def agg(V):
    """(c, k) values with nan -> mean over defined samples, number defined, 5th and 95th percentile."""
    ok = ~np.isnan(V)
    n = ok.sum(1)
    mean = _nanmean(V, 1)
    p5, p95 = np.full(len(V), np.nan), np.full(len(V), np.nan)
    rows = n > 0
    if rows.any():
        p5[rows], p95[rows] = np.nanpercentile(V[rows], [5, 95], axis=1)
    return mean, n.astype(np.int16), p5, p95


def loop_stats(A, fell):
    """Plain code for one 6-year stack (6, G, G) and fell (5, G, G), for gate M5: cell loops for T1 and T3, and for T2
    the distance from every gained cell to every cell of year t (all pairs), minimum taken."""
    def mind(q, cells, wrap):
        dy, dx = np.abs(cells[:, 0] - q[0]), np.abs(cells[:, 1] - q[1])
        if wrap:
            dy, dx = np.minimum(dy, G - dy), np.minimum(dx, G - dx)
        return float(np.sqrt(dy * dy + dx * dx).min())
    t1, t3 = [], []
    t2 = {True: [0.0, 0], False: [0.0, 0]}
    for t in range(W - 1):
        a, b = A[t], A[t + 1]
        g = l = na = nb = 0
        nf = lf = nr = lr = 0
        prev_cells, gains = [], []
        for y in range(G):
            for x in range(G):
                na += int(a[y, x]); nb += int(b[y, x])
                if b[y, x] and not a[y, x]:
                    g += 1; gains.append((y, x))
                if a[y, x]:
                    prev_cells.append((y, x))
                    if not b[y, x]:
                        l += 1
                    if fell[t, y, x]:
                        nf += 1; lf += int(not b[y, x])
                    else:
                        nr += 1; lr += int(not b[y, x])
        if na + nb > 0:
            t1.append((g + l) / (na + nb))
        if prev_cells:
            pc = np.array(prev_cells)
            for q in gains:
                for wrap in (True, False):
                    t2[wrap][0] += mind(q, pc, wrap); t2[wrap][1] += 1
        if nf > 0 and nr > 0:
            t3.append(lf / nf - lr / nr)
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return dict(T1=m(t1), T2=t2[True][0] / t2[True][1] if t2[True][1] else float("nan"),
                T2plain=t2[False][0] / t2[False][1] if t2[False][1] else float("nan"), T3=m(t3))


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-9))


# ---------------------------------------------------------------- environment, habitat and growth rates
def load_env(obj1_work):
    p = Path(obj1_work) / "environment.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; give --obj1-work, the folder that holds obj1's environment.py")
    sys.path.insert(0, str(Path(obj1_work).resolve()))
    spec = importlib.util.spec_from_file_location("environment_obj1_g4b", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for f in ("habitat_layer", "niche_traits", "niche_r"):
        if not hasattr(m, f):
            sys.exit(f"STOP: {p} has no {f}")
    return m, p


def growth_fell(z, env, rounds_sp, out=print):
    """For one community: fell (5, G, G) per (round, species) pair, and the M3 checks of this file."""
    p = HC.params(z)
    if BROKEN == "hab_seed":
        p = dict(p, seed=p["seed"] + 1)
    kw_t, src_t, miss_t = NC.fill(env.niche_traits, z, 1)
    kw_r, src_r, miss_r = NC.fill(env.niche_r, z, 4)
    if miss_t or miss_r:
        return None, dict(ok=False, why=f"no saved value or default for {miss_t + miss_r}")
    Ht = np.asarray(z["ENV_habitat_t"], np.float64)
    Hs = np.asarray(z["ENV_habitat_shift"], np.float64)
    hab, dmax = {}, 0.0
    for r in sorted({int(r) for r, _ in rounds_sp}):
        if r < 1:
            return None, dict(ok=False, why=f"round {r} has no saved previous round")
        maps, _ = HC.rebuild_round(Ht[r - 1], Hs[r - 1], r, p, env.habitat_layer)
        dmax = max(dmax, float(np.abs(maps[-1] - Ht[r]).max()))
        hab[r] = maps[p["M"] - W:]
    saved = np.stack([np.asarray(z[k], np.float64) for k in ("NICHE_opt", "NICHE_w", "NICHE_peak")], 1)
    found = NC.match_saved(env.niche_traits, kw_t, saved, z["ASM_F_ids"])
    a1 = all(len(f) == 1 for f in found)
    dr = float("inf")
    if a1:
        tr = NC.traits(env.niche_traits, [f[0] for f in found], kw_t)
        R = np.asarray(z["ENV_r_field"], np.float64)
        if R.shape[0] == len(tr):
            dr = float(np.abs(NC.r_fields(env.niche_r, Ht[-1], tr, kw_r) - R).max())
    niche = {}
    fell = {}
    for r, s in rounds_sp:
        if s not in niche:
            niche[s] = tuple(float(v) for v in env.niche_traits(int(s), **kw_t)[:3])
        rr = np.stack([np.asarray(env.niche_r(hab[r][t], *niche[s], **kw_r), np.float64) for t in range(W)])
        fell[(r, s)] = rr[1:] < rr[:-1]
    ok = dmax <= TOL_HAB and a1 and dr <= TOL_R
    return fell, dict(ok=ok, rounds=len(hab), hab_diff=dmax, saved=len(saved), matched=int(sum(len(f) == 1 for f in found)),
                      r_diff=dr, species=len(niche), call_t=src_t, call_r=src_r)


# ---------------------------------------------------------------- run
def run(data, runs, g2, obj1_files, obj1_work, out_dir, chunk=256, out=print):
    global _LEAK
    data, g2, out_dir = Path(data), Path(g2), Path(out_dir)
    val_dir = data / "real_val"
    runs = [Path(r) for r in runs]
    for p in [data, g2, out_dir, *runs, *obj1_files, obj1_work]:
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G5")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    if len(runs) != 2:
        sys.exit("STOP: G4b needs the two seeds (--runs seed1 seed2)")
    if not obj1_files:
        sys.exit("STOP: no obj1 file matches --files")
    out_dir.mkdir(parents=True, exist_ok=True)
    res, t0 = {}, time.time()

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    env, envp = load_env(obj1_work)
    fp = data / "stage_d_frozen_val.npz"
    frozen = dict(np.load(fp)); fmd5 = X.file_md5(fp)
    n = len(frozen["row"])
    out(f"stage_g_process_tests.py md5 {X.file_md5(__file__)}; stage_g_habitat_check.py {X.file_md5(HERE / 'stage_g_habitat_check.py')}; "
        f"stage_g_niche_check.py {X.file_md5(HERE / 'stage_g_niche_check.py')}; environment.py {X.file_md5(envp)}; "
        f"frozen set md5 {fmd5} ({n} stacks)")

    # ---- M1 inputs
    ok1, why = True, []
    maps = {}
    for j, r in enumerate(runs):
        rep = json.loads((r / "check_g1" / "g1_report.json").read_text())
        sp = r / "check_g1" / "samples_k50.npz"
        m_s, m_b = X.file_md5(sp), X.file_md5(r / "best.pt")
        z = np.load(sp)
        good = m_s == rep["md5"]["samples"] and m_b == rep["md5"]["best_pt"] and np.array_equal(z["sel"], np.arange(n)) \
            and str(z["layout"]) == GS.LAYOUT and z["maps"].shape[1] == K
        ok1 &= good; why.append(f"{r.name} samples {m_s[:8]} best.pt {m_b[:8]} {good}")
        maps[f"s{j + 1}"] = z["maps"]
    g2rep = json.loads((g2 / "g2_report.json").read_text())
    fb = "maps_B3_halfblank.npz"
    m_b3 = X.file_md5(g2 / fb)
    z = np.load(g2 / fb)
    good = m_b3 == g2rep["md5"][fb] and str(z["frozen_md5"]) == fmd5 and np.array_equal(z["sel"], np.arange(n)) \
        and str(z["layout"]) == GS.LAYOUT and z["maps"].shape[1] == K
    ok1 &= good; why.append(f"{fb} {m_b3[:8]} {good}")
    maps["B3"] = z["maps"]
    man = {row["input_file"]: row for row in D.rows_of(val_dir)}
    by_name = {Path(f).name: f for f in obj1_files}
    files, fidx, rows = frozen["files"], frozen["file_index"], frozen["row"]
    t_ok, o_ok, comm_of = True, True, {}
    for i in np.unique(fidx):
        row = man[str(files[i])]
        t_ok &= X.file_md5(val_dir / row["truth_file"]) == row["truth_md5"]
        o_ok &= row["source"] in by_name and X.file_md5(by_name.get(row["source"], fp)) == row["source_md5"]
        comm_of[int(i)] = row
    why.append(f"truth files = manifest truth_md5 {t_ok}; obj1 files = manifest source_md5 {o_ok} ({len(comm_of)} communities)")
    gate("M1", ok1 and t_ok and o_ok, "inputs as logged, made from this frozen set, stacks in order: " + "; ".join(why))

    # ---- ids only from the truth files
    sp_id = np.zeros(n, np.int64); rd = np.zeros(n, np.int64); comm = np.zeros(n, np.int64)
    early = np.zeros((n, W, G, G), bool) if BROKEN == "leak" else None
    for i, row in comm_of.items():
        k = np.nonzero(fidx == i)[0]
        with np.load(val_dir / row["truth_file"], allow_pickle=False) as zt:
            sp_id[k] = zt["stack_species"][rows[k]]; rd[k] = zt["stack_round"][rows[k]]
            if early is not None:
                early[k] = X.unpack(zt["true_maps"][rows[k]])
        comm[k] = int(row["sim_seed"])
    _LEAK = early
    out(f"   ids read from the truth files (stack_species, stack_round only): {len(np.unique(comm))} communities, "
        f"{len(set(zip(comm.tolist(), sp_id.tolist()))):,} (community, species) pairs, "
        f"{len(set(zip(comm.tolist(), rd.tolist()))):,} (community, round) pairs")

    # ---- M3 habitat and growth rates
    t1 = time.time()
    fell = np.zeros((n, W - 1, G, G), bool)
    ok3, lines3 = True, []
    for i, row in comm_of.items():
        k = np.nonzero(fidx == i)[0]
        z = np.load(by_name[row["source"]], allow_pickle=True)
        pairs = sorted(set(zip(rd[k].tolist(), sp_id[k].tolist())))
        fl, info = growth_fell(z, env, pairs)
        if fl is None:
            ok3 = False; lines3.append(f"{row['source']}: {info['why']}"); continue
        for j in k:
            fell[j] = fl[(int(rd[j]), int(sp_id[j]))]
        ok3 &= info["ok"]
        lines3.append(f"{row['source'][-30:]}: rounds {info['rounds']}, habitat diff {info['hab_diff']:.1e}; saved niches "
                      f"{info['matched']} of {info['saved']}, r field diff {info['r_diff']:.1e}; species {info['species']}")
    for s in lines3:
        out("   " + s)
    gate("M3", ok3, f"T3 inputs: habitat of every used round rebuilt (rule {TOL_HAB:g}), saved niches and ENV_r_field "
         f"reproduced (rule {TOL_R:g}) in every community ({time.time() - t1:.0f} s)")
    share_fell = fell.reshape(n, W - 1, -1).mean(2)
    out(f"   growth rate fell in {share_fell.mean():.3f} of cells per transition (mean over stacks and transitions); "
        f"transitions with cells of both kinds {float(((share_fell > 0) & (share_fell < 1)).mean()):.3f}")

    # ---- sample and control statistics, before any true map is opened
    IDX = control_index()
    yr = np.arange(W)[None, :]
    two = all(len(set(IDX[q].tolist())) == W for q in range(K)) and all(len(set(IDX[:, t].tolist())) == K for t in range(W))
    methods = ["s1", "s2", "pool", "c1", "c2", "cpool", "B3"]
    S = {(m, v, q): np.full(n, np.nan) for m in methods for v in TESTS for q in ("mean", "p5", "p95")}
    S.update({(m, v, "n"): np.zeros(n, np.int16) for m in methods for v in TESTS})
    m2 = True
    t1 = time.time()
    for a in range(0, n, chunk):
        b = min(n, a + chunk); ix = np.arange(a, b)
        A = {m: unpack6(maps[m][a:b]) for m in ("s1", "s2", "B3")}
        A["pool"] = np.concatenate([A["s1"][:, :K // 2], A["s2"][:, :K // 2]], 1)
        for src, cn in (("s1", "c1"), ("s2", "c2"), ("pool", "cpool")):
            A[cn] = A[src][:, IDX, yr]
            pm = np.packbits(A[src].reshape(b - a, K, W, -1), axis=-1)
            pc = np.packbits(A[cn].reshape(b - a, K, W, -1), axis=-1)
            for t in range(W):
                vm = np.sort(np.ascontiguousarray(pm[:, :, t]).view(f"V{pm.shape[-1]}")[..., 0], axis=1)
                vc = np.sort(np.ascontiguousarray(pc[:, :, t]).view(f"V{pc.shape[-1]}")[..., 0], axis=1)
                m2 &= bool(np.array_equal(vm, vc))
        for m in methods:
            st = stats(A[m], fell[a:b], idx=ix)
            for v in TESTS:
                mean, cnt, p5, p95 = agg(st[v])
                S[(m, v, "mean")][a:b], S[(m, v, "n")][a:b], S[(m, v, "p5")][a:b], S[(m, v, "p95")][a:b] = mean, cnt, p5, p95
        if a == 0 or (a // chunk) % 16 == 0:
            out(f"   statistics {b:,} of {n:,} stacks ({(time.time() - t1) / 60:.1f} min)")
    gate("M2", m2 and two, f"control (year t from sample (k + {SHIFT} t) mod {K}): every year's 50 maps unchanged in every "
         f"stack {m2}; 6 different samples in every control sample and each year a permutation {two}")
    sfile = out_dir / "g4b_sample_stats.npz"
    np.savez(sfile, **{f"{m}__{v}__{q}": S[(m, v, q)] for (m, v, q) in S})
    smd5 = X.file_md5(sfile)
    out(f"   sample statistics written before any true map was opened: {sfile.name} md5 {smd5} "
        f"({(time.time() - t1) / 60:.1f} min)")

    # ---- truth
    T = np.zeros((n, W, G, G), bool); tc = np.zeros(n, int)
    for i, row in comm_of.items():
        k = np.nonzero(fidx == i)[0]
        with np.load(val_dir / row["truth_file"], allow_pickle=False) as zt:
            T[k] = X.unpack(zt["true_maps"][rows[k]]); tc[k] = D.tidx(zt["true_cells"][rows[k]])
    tru = {v: np.zeros(n) for v in TESTS}
    for a in range(0, n, 2048):
        b = min(n, a + 2048)
        st = stats(T[a:b, None], fell[a:b])
        for v in TESTS:
            tru[v][a:b] = st[v][:, 0]

    # ---- M4 leak check
    g = X.rng(77, 9)
    pick = np.sort(g.choice(n, min(N_LEAK, n), replace=False))
    if _LEAK is not None:
        _LEAK = T[np.roll(np.arange(n), 1)]
    st = stats(unpack6(maps["s1"][pick]), fell[pick], idx=pick)
    eq = all(same(agg(st[v])[0], S[("s1", v, "mean")][pick]) for v in TESTS)
    gate("M4", X.file_md5(sfile) == smd5 and eq, f"statistics file unchanged after the truth was opened "
         f"({X.file_md5(sfile)[:8]}); {len(pick)} stacks recomputed with swapped true maps equal the logged values {eq}")

    # ---- M5 statistic code against loops
    pick = np.sort(X.rng(77, 10).choice(n, min(N_LOOP, n), replace=False))
    A5 = unpack6(maps["s1"][pick, :5])
    fast = stats(A5, fell[pick]); fast_t = stats(T[pick, None], fell[pick])
    ok5 = True
    for q, j in enumerate(pick):
        for kk in range(5):
            sl = loop_stats(A5[q, kk], fell[j])
            for v in TESTS:
                ok5 &= same(fast[v][q, kk], sl[v])
        sl = loop_stats(T[j], fell[j])
        for v in TESTS:
            ok5 &= same(fast_t[v][q, 0], sl[v])
    gate("M5", ok5, f"T1, T2 torus, T2 plain, T3 of the vectorised code = loop code on {len(pick)} stacks (5 samples of "
         f"seed 1 and the truth)")

    # ---- tests
    defined = {v: ~np.isnan(tru[v]) for v in TESTS}
    rep, verdicts, verdicts_g, csv_cols = {}, {}, {}, {}
    lines = []
    for lab, mm, cc in (("seed 1", "s1", "c1"), ("seed 2", "s2", "c2"), ("pooled", "pool", "cpool")):
        for v in TESTS:
            em = np.abs(S[(mm, v, "mean")] - tru[v]); ec = np.abs(S[(cc, v, "mean")] - tru[v])
            eb = np.abs(S[("B3", v, "mean")] - tru[v])
            dm, dc = S[(mm, v, "n")] > 0, S[(cc, v, "n")] > 0
            enter = defined[v] & dm & dc
            d = np.where(enter, em - ec, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)     # classes with no stack give empty slices
                cb = CK.class_boot(d, tc, comm, classes=SCORE)
                c0 = CK.class_boot(d, tc, comm, classes=[0])
            cov = lambda m: (tru[v] >= S[(m, v, "p5")]) & (tru[v] <= S[(m, v, "p95")])
            per = {}
            for c in [0] + SCORE:
                s = enter & (tc == c)
                sb = s & (S[("B3", v, "n")] > 0)
                row = dict(n=int(s.sum()), n_b3=int(sb.sum()))
                if s.any():
                    row.update(truth=float(tru[v][s].mean()), model=float(S[(mm, v, "mean")][s].mean()),
                               control=float(S[(cc, v, "mean")][s].mean()), err_model=float(em[s].mean()),
                               err_control=float(ec[s].mean()), cov_model=float(cov(mm)[s].mean()),
                               cov_control=float(cov(cc)[s].mean()))
                    src = cb[c] if c in cb else c0[c]
                    row.update(diff=src["mean"], lo=src["lo"], hi=src["hi"])
                if sb.any():
                    row.update(b3=float(S[("B3", v, "mean")][sb].mean()), err_b3=float(eb[sb].mean()), cov_b3=float(cov("B3")[sb].mean()))
                per[TRUE[c]] = row
            bal = cb["balanced"]
            em_c = [per[TRUE[c]]["err_model"] for c in SCORE if per[TRUE[c]]["n"]]
            ec_c = [per[TRUE[c]]["err_control"] for c in SCORE if per[TRUE[c]]["n"]]
            gain = 1 - float(np.mean(em_c)) / float(np.mean(ec_c)) if em_c and np.mean(ec_c) > 0 else float("nan")
            sb_all = enter & (S[("B3", v, "n")] > 0) & np.isin(tc, SCORE)
            dropped = dict(truth=int((~defined[v]).sum()), model=int((~dm).sum()), control=int((~dc).sum()),
                           any=int((~enter).sum()))
            verdict = ("PASS" if (bal["hi"] < 0 and enter.any()) else "FAIL") if (lab != "pooled" and v in TESTED) else "report"
            verdict_g = ("PASS" if (verdict == "PASS" and gain >= MIN_GAIN) else "FAIL") if verdict != "report" else "report"
            rep[f"{lab} {v}"] = dict(entered=int(enter.sum()), dropped=dropped, balanced=bal, relative_gain=gain, classes=per,
                                     verdict=verdict, verdict_min_gain=verdict_g,
                                     b3_balanced_err=float(np.nanmean([per[TRUE[c]].get("err_b3", np.nan) for c in SCORE]))
                                     if sb_all.any() else None)
            if verdict != "report":
                verdicts[f"{lab} {v}"] = verdict
                verdicts_g[f"{lab} {v}"] = verdict_g
            csv_cols[(mm, v)] = em; csv_cols[(cc, v)] = ec
            lines.append((lab, v, enter.sum(), dropped, bal, per, verdict, gain))
    # mean of the two seeds (report)
    for v in TESTS:
        b1, b2 = rep[f"seed 1 {v}"]["balanced"], rep[f"seed 2 {v}"]["balanced"]
        rep[f"mean of seeds {v}"] = dict(diff=(b1["mean"] + b2["mean"]) / 2)

    out("   statistics by true class, entered stacks (truth | model | control | B3 mean; error model | control | B3; "
        "model - control error with 95% interval; coverage model | control):")
    for lab, v, ne, dr, bal, per, verdict, gain in lines:
        out(f"   {lab:7s} {v:7s} entered {ne:,}; dropped: truth undefined {dr['truth']:,}, model {dr['model']:,}, "
            f"control {dr['control']:,}; balanced (classes 1 .. 51+) {bal['mean']:+.4f} [{bal['lo']:+.4f}, {bal['hi']:+.4f}] "
            f"{verdict}; model error {100 * gain:.1f}% below the control's"
            + (f" (with a {100 * MIN_GAIN:.0f}% minimum gain: {rep[f'{lab} {v}']['verdict_min_gain']})" if verdict != "report" else ""))
        for c in [0] + SCORE:
            r = per[TRUE[c]]
            if r["n"] == 0:
                continue
            b3 = f"{r['b3']:.3f}" if "b3" in r else "-"
            eb3 = f"{r['err_b3']:.3f}" if "err_b3" in r else "-"
            out(f"      class {TRUE[c]:>5s} n {r['n']:6,}: {r['truth']:.3f} | {r['model']:.3f} | {r['control']:.3f} | {b3}; "
                f"err {r['err_model']:.3f} | {r['err_control']:.3f} | {eb3}; diff {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]; "
                f"cov {r['cov_model']:.2f} | {r['cov_control']:.2f}")
    out("   mean of the two seeds, balanced model - control error: "
        + "; ".join(f"{v} {rep[f'mean of seeds {v}']['diff']:+.4f}" for v in TESTS))

    # ---- outputs
    with open(out_dir / "per_stack_g4b.csv", "w", newline="") as fh:
        cols = ["stack", "comm", "true_class"] + [f"truth_{v}" for v in TESTS] + \
               [f"{m}_{v}_{q}" for m in methods for v in TESTS for q in ("mean", "n")]
        w = csv.writer(fh); w.writerow(cols)
        for i in range(n):
            w.writerow([i, int(comm[i]), TRUE[tc[i]]] + [f"{tru[v][i]:.6g}" for v in TESTS] +
                       [f"{S[(m, v, q)][i]:.6g}" if q == "mean" else int(S[(m, v, q)][i]) for m in methods for v in TESTS for q in ("mean", "n")])
    report = dict(gates=res, verdicts=verdicts, verdicts_min_gain=verdicts_g, min_gain=MIN_GAIN, tests=rep, md5=dict(sample_stats=smd5, frozen=fmd5, b3=m_b3,
                  environment=X.file_md5(envp), tool=X.file_md5(__file__)), control=f"(k + {SHIFT} t) mod {K}",
                  minutes=round((time.time() - t0) / 60, 1))
    (out_dir / "g4b_report.json").write_text(json.dumps(report, indent=1, default=float))
    out(f"   written g4b_sample_stats.npz, per_stack_g4b.csv, g4b_report.json ({(time.time() - t0) / 60:.1f} min)")
    out(f"stage G4b: gates {sum(res.values())} PASS, {len(res) - sum(res.values())} FAIL; "
        + ", ".join(f"{k} {v}" for k, v in verdicts.items())
        + f"; with a {100 * MIN_GAIN:.0f}% minimum gain: " + ", ".join(f"{k} {v}" for k, v in verdicts_g.items()))
    return res, verdicts, rep


# ---------------------------------------------------------------- self-test (stand-in environment, planted communities)
ENV_EXTRA = '''
def niche_traits(sid, *, seed, opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=1.5, tradeoff=1.0):
    rng = np.random.default_rng([int(seed), 7919, int(sid)])
    opt = float(rng.normal(0.0, opt_sd))
    w = float(np.exp(rng.uniform(np.log(w_min), np.log(w_max))))
    return opt, w, float(peak_max * (w_min / w) ** tradeoff)
def niche_r(H, opt, w, peak, cost):
    return peak * np.exp(-(np.asarray(H) - opt) ** 2 / (2 * w * w)) - cost
'''


def _neigh(A):
    return sum(np.roll(A, s, axis=ax) for s in (1, -1) for ax in (-1, -2)).astype(np.float64)


def _chain(A0, rr, g, p0=0.85, slope=12.0, col=0.18):
    """Toy process on the torus: an occupied cell stays with a chance that falls where the growth rate fell; an empty
    cell is colonised from its 4 neighbours."""
    A = [A0]
    for t in range(W - 1):
        a = A[-1]
        keep = np.clip(p0 + slope * (rr[t + 1] - rr[t]), 0.02, 0.99)
        stay = a & (g.random(a.shape) < keep)
        new = ~a & (g.random(a.shape) < 1 - (1 - col) ** _neigh(a))
        A.append(stay | new)
    return np.stack(A)


def _blob(n0, g):
    A = np.zeros((G, G), bool)
    y, x = g.integers(0, G, 2); A[y, x] = True
    while A.sum() < n0:
        cand = np.argwhere(~A & (_neigh(A) > 0))
        y, x = cand[g.integers(len(cand))]; A[y, x] = True
    return A


def plant(tmp, static_seed2=True, n_comm=6, per=50):
    """A planted obj2 folder: environment.py, obj1-like files (G0 habitat rule, G4a niche keys), real_val manifest and
    truth files, frozen set, two G1 runs and a G2 folder. Seed 1 samples follow the truth's process; seed 2 samples
    (if static_seed2) take each year from a different chain, which keeps the yearly maps and breaks the time order."""
    tmp = Path(tmp)
    (tmp / "obj1").mkdir(parents=True); (tmp / "work").mkdir(); vd = tmp / "data" / "real_val"; vd.mkdir(parents=True)
    src = "import numpy as np\n" + inspect.getsource(HC.toy_layer).replace("def toy_layer", "def habitat_layer") + ENV_EXTRA
    (tmp / "work" / "environment.py").write_text(src)
    spec = importlib.util.spec_from_file_location("env_g4b_plant", tmp / "work" / "environment.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    M, rounds = 8, 10
    rows, files_in, fidx, frow = [], [], [], []
    allm = {"s1": [], "s2": [], "B3": []}
    sizes = [1, 2, 4, 8, 15, 30, 60, 120]
    for ci in range(n_comm):
        seed = 7301 + ci
        g = np.random.default_rng(seed)
        f1 = tmp / "obj1" / f"c{seed}_training.npz"
        pieces = HC.plant_file(f1, rounds=rounds, M=M, W=5000, tau=8000.0, shift_tau=62500.0, seed=105000 + seed)
        zz = dict(np.load(f1))
        ids = np.unique(g.integers(0, 900, 40))
        tseed = 200000 + seed
        prm = dict(seed=tseed, opt_sd=0.4, w_min=0.2, w_max=1.0, peak_max=1.5, tradeoff=1.0)
        keep = np.sort(g.choice(ids, 12, replace=False))
        tr = np.array([env.niche_traits(int(s), **prm) for s in keep])
        cost = 0.1
        Hl = zz["ENV_habitat_t"].astype(np.float64)[-1]
        zz.update(TRAIT_seed=np.int64(tseed), NICHE_opt_sd=np.float64(0.4), NICHE_w_min=np.float64(0.2),
                  NICHE_w_max=np.float64(1.0), NICHE_peak_max=np.float64(1.5), NICHE_tradeoff=np.float64(1.0),
                  NICHE_cost=np.float64(cost), NICHE_opt=tr[:, 0].astype(np.float32), NICHE_w=tr[:, 1].astype(np.float32),
                  NICHE_peak=tr[:, 2].astype(np.float32), ASM_F_ids=np.repeat(ids, 2), SIM_SEED=np.int64(seed),
                  ENV_r_field=np.stack([env.niche_r(Hl, *tr[i], cost) for i in range(len(tr))]).astype(np.float32))
        np.savez(f1, **zz)
        st_s, st_r, tm, tcells = [], [], [], []
        for j in range(per):
            s = int(ids[j % len(ids)]); r = int(g.integers(rounds // 2, rounds))
            nt = env.niche_traits(s, **prm)
            rr = np.stack([env.niche_r(pieces[(r, M - W + t)], *nt, cost) for t in range(W)])
            A0 = _blob(sizes[j % len(sizes)], g)
            truth = _chain(A0, rr, np.random.default_rng([seed, j, 0]))
            ch = [_chain(A0, rr, np.random.default_rng([seed, j, 1, q])) for q in range(K)]
            ch2 = [_chain(A0, rr, np.random.default_rng([seed, j, 2, q])) for q in range(K)]
            if static_seed2:
                ch2 = [np.stack([ch2[(q + 11 * t) % K][t] for t in range(W)]) for q in range(K)]
            b3 = [_chain(A0, rr, np.random.default_rng([seed, j, 3, q]), p0=0.9, slope=0.0, col=0.05) for q in range(K)]
            for key, cs in (("s1", ch), ("s2", ch2), ("B3", b3)):
                allm[key].append(np.packbits(np.stack(cs).reshape(K, -1), axis=-1))
            st_s.append(s); st_r.append(r); tm.append(truth); tcells.append(int(truth[-1].sum()))
            fidx.append(ci); frow.append(j)
        tf, inf = f"s{seed}_truth.npz", f"s{seed}_input.npz"
        np.savez(vd / tf, stack_species=np.array(st_s, np.int64), stack_round=np.array(st_r, np.int32),
                 true_maps=X.pack(np.stack(tm)), true_cells=np.array(tcells, np.int16), sim_seed=np.int64(seed))
        files_in.append(inf)
        rows.append(dict(sim_seed=seed, source=f1.name, source_md5=X.file_md5(f1), truth_file=tf,
                         truth_md5=X.file_md5(vd / tf), input_file=inf))
    with open(vd / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(frow)
    fz = tmp / "data" / "stage_d_frozen_val.npz"
    np.savez(fz, files=np.array(files_in), file_index=np.array(fidx, np.int16), row=np.array(frow, np.int32))
    for j, key in ((1, "s1"), (2, "s2")):
        rd_ = tmp / "runs" / f"f_seed{j}"; (rd_ / "check_g1").mkdir(parents=True)
        (rd_ / "best.pt").write_bytes(f"planted model {j}".encode())
        np.savez(rd_ / "check_g1" / "samples_k50.npz", maps=np.stack(allm[key]), sel=np.arange(n), layout=np.array(GS.LAYOUT))
        (rd_ / "check_g1" / "g1_report.json").write_text(json.dumps(dict(md5=dict(
            samples=X.file_md5(rd_ / "check_g1" / "samples_k50.npz"), best_pt=X.file_md5(rd_ / "best.pt")))))
    g2 = tmp / "data" / "g2"; g2.mkdir()
    np.savez(g2 / "maps_B3_halfblank.npz", maps=np.stack(allm["B3"]), sel=np.arange(n), layout=np.array(GS.LAYOUT),
             frozen_md5=np.array(X.file_md5(fz)))
    (g2 / "g2_report.json").write_text(json.dumps(dict(md5={"maps_B3_halfblank.npz": X.file_md5(g2 / "maps_B3_halfblank.npz")})))
    return dict(data=tmp / "data", runs=[tmp / "runs" / "f_seed1", tmp / "runs" / "f_seed2"], g2=g2,
                files=sorted(str(p) for p in (tmp / "obj1").glob("*_training.npz")), work=tmp / "work")


def unit_checks(t):
    def one(maps6, fell5=None):
        A = np.asarray(maps6, bool)[None, None]
        fl = np.zeros((1, W - 1, G, G), bool) if fell5 is None else np.asarray(fell5, bool)[None]
        return {v: float(x[0, 0]) for v, x in stats(A, fl).items()}
    a = np.zeros((W, G, G), bool); a[:, 3:6, 3:6] = True
    b = np.zeros((W, G, G), bool)
    for y in range(W):
        b[y, 2 * y % G, 0] = True
    r1, r2 = one(a), one(b)
    t("U1", r1["T1"] == 0.0 and r2["T1"] == 1.0, f"T1: a map that never changes gives 0 ({r1['T1']}); one that moves "
      f"every year gives 1 ({r2['T1']})")
    c = np.zeros((W, G, G), bool); c[0, 0, 0] = True; c[1, 0, 0] = True; c[1, 0, 19] = True
    c[1:, 0, 0] = True; c[2, 3, 4] = True
    r = one(c)
    t("U2", abs(r["T2"] - 3.0) < 1e-12 and abs(r["T2plain"] - 12.0) < 1e-12, f"T2: gains at (0, 19) and (3, 4) from (0, 0): "
      f"torus mean (1 + 5) / 2 = {r['T2']}, plain (19 + 5) / 2 = {r['T2plain']}")
    d = np.zeros((W, G, G), bool); d[:, 0, 0:4] = True; d[1:, 0, 0] = False
    f = np.zeros((W - 1, G, G), bool); f[:, 0, 0:2] = True
    r = one(d, f)
    t("U3", abs(r["T3"] - 0.1) < 1e-12,
      f"T3: year 0 has 2 occupied cells where r fell (1 lost) and 2 where it rose (none lost), so 0.5; the 4 later "
      f"transitions lose nothing, so 0; mean (0.5 + 0 + 0 + 0 + 0) / 5 = {r['T3']}")
    e = np.zeros((W, G, G), bool)
    r = one(e)
    t("U4", all(np.isnan(v) for v in r.values()), "an empty stack leaves every statistic undefined")
    V = np.array([[np.nan, 1.0, 3.0], [np.nan, np.nan, np.nan]])
    mean, cnt, p5, p95 = agg(V)
    t("U5", mean[0] == 2.0 and cnt.tolist() == [2, 0] and np.isnan(mean[1]),
      "mean over the samples that define it (2.0 from 2 of 3); none defined gives no mean")
    idx = control_index()
    ok = all(len(set(idx[q].tolist())) == W for q in range(K)) and all(sorted(idx[:, y].tolist()) == list(range(K)) for y in range(W))
    t("U6", ok, "control index: 6 different samples in every control sample; each year a permutation of the 50")
    g = np.random.default_rng(5)
    good = True
    for dens in (0.02, 0.08, 0.3):
        A = g.random((30, 3, W, G, G)) < dens
        fl = g.random((30, W - 1, G, G)) < 0.5
        fast = stats(A, fl)
        good &= all(same(fast[v][i, k], loop_stats(A[i, k], fl[i])[v]) for i in range(30) for k in range(3) for v in TESTS)
    t("U7", good, f"vectorised statistics = loop code on 270 random stacks of 2%, 8% and 30% occupied cells (T2: all-pairs "
      f"path at {SPARSE_MAX} cells or fewer in year t, offset path above)")


def self_test():
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    unit_checks(t)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        P = plant(tmp / "p")
        lines = []
        g, v, rep = run(P["data"], P["runs"], P["g2"], P["files"], P["work"], tmp / "out1", chunk=64, out=lines.append)
        t("S1", all(g.values()), f"planted communities pass M1-M5: {g}")
        s1 = all(v[f"seed 1 {x}"] == "PASS" for x in TESTED)
        s2 = all(v[f"seed 2 {x}"] == "FAIL" for x in TESTED)
        t("S2", s1, "samples from the truth's process beat the year-shuffled control in T1, T2, T3: "
          + ", ".join(f"{x} {rep[f'seed 1 {x}']['balanced']['mean']:+.3f} [{rep[f'seed 1 {x}']['balanced']['hi']:+.3f}]" for x in TESTED))
        t("S3", s2, "static samples (each year from another chain, so like the control) do not; their differences are near 0: "
          + ", ".join(f"{x} {rep[f'seed 2 {x}']['balanced']['mean']:+.3f} [{rep[f'seed 2 {x}']['balanced']['lo']:+.3f}, "
                      f"{rep[f'seed 2 {x}']['balanced']['hi']:+.3f}]" for x in TESTED))
        g2_, _, _ = run(P["data"], P["runs"], P["g2"], P["files"], P["work"], tmp / "out2", chunk=128, out=lambda *a, **k: None)
        a, b = (X.file_md5(tmp / o / "g4b_sample_stats.npz") for o in ("out1", "out2"))
        t("S4", a == b, f"another chunk size writes the same statistics file ({a[:8]})")
        refused = []
        for kw in (dict(out_dir=tmp / "out1"), dict(out_dir=tmp / "real_eval_out")):
            try:
                run(P["data"], P["runs"], P["g2"], P["files"], P["work"], kw["out_dir"], out=lambda *a, **k: None)
                refused.append(False)
            except SystemExit:
                refused.append(True)
        t("S5", all(refused), f"an output folder that holds outputs and a real_eval path are refused {refused}")
        print("   planted run log:", *lines, sep="\n      ")
        z = dict(np.load(P["runs"][0] / "check_g1" / "samples_k50.npz"))
        z["maps"] = z["maps"].copy(); z["maps"][0, 0, 0] ^= 1
        bad = tmp / "bad_md5"
        import shutil
        shutil.copytree(tmp / "p", bad)
        np.savez(bad / "runs" / "f_seed1" / "check_g1" / "samples_k50.npz", **z)
        cases = [("samples_changed", "M1", bad), ("control_drop", "M2", tmp / "p"), ("hab_seed", "M3", tmp / "p"),
                 ("leak", "M4", tmp / "p"), ("t2_plain", "M5", tmp / "p")]
        for j, (bug, gate_, root) in enumerate(cases):
            env = dict(os.environ, STAGE_G4B_BROKEN=bug if bug != "samples_changed" else "")
            code = ("import importlib.util as u; s=u.spec_from_file_location('p', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "g,v,_=m.run(%r, [%r, %r], %r, %r, %r, %r, chunk=128, out=lambda *a, **k: None); print('RES', g)"
                    % (str(Path(__file__).resolve()), str(root / "data"), str(root / "runs" / "f_seed1"), str(root / "runs" / "f_seed2"),
                       str(root / "data" / "g2"), sorted(str(p) for p in (root / "obj1").glob("*_training.npz")), str(root / "work"),
                       str(tmp / f"bug{j}")))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate_}': False" in line, f"{gate_} fails ({line[4:] or p.stderr.strip()[-300:]})")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--obj1-work", default="../../obj1_v2/work")
    ap.add_argument("--files", default="../../obj1_v2/work/results/stage2a/real_val/*_training.npz")
    ap.add_argument("--out", default="../data/stage_g4b_process")
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    g, v, _ = run(a.data, a.runs, a.g2, sorted(glob.glob(a.files)), a.obj1_work, a.out, chunk=a.chunk,
                  out=lambda s: print(s, flush=True))
    sys.exit(0 if all(g.values()) else 1)