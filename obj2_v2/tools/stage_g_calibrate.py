#!/usr/bin/env python3
"""
stage_g_calibrate.py  (obj2_v2, stage G step G3, 10 Nov 2026; rules signed off by Sara on 10 Nov)

Question: after a level correction and a conformal widening fitted on other communities, do the AOO and EOO intervals
of each method hold the truth 90% of the time in every observable group, how wide are they, and is P(AOO <= x)
calibrated? Methods: model seed 1, model seed 2, both seeds pooled (first 25 maps of each), B3 half-blank, B4 (50 maps
each) and B5 (records, one map). All on the 16,347 frozen real_val stacks at the target year.

Rules (written before the run; claude/obj2_v2_results_log_G.md, signed off 10 Nov)
    AOO of a map = occupied window cells at the target year. EOO = area of the minimum convex polygon around the occupied
    cell centres (cell units), raised to AOO when smaller (IUCN Standards and Petitions Committee 2024, Guidelines for
    Using the IUCN Red List Categories and Criteria, version 16, section 4.9, p. 53); 0-2 cells or a line give EOO = AOO.
    True AOO and EOO from the truth map by the same rule.
    Folds: the communities split into two halves by a fixed seed, the same for every method. Each correction is fitted on
    one fold and applied to the other, then the folds swap; a stack's own fold truth never enters its correction.
    Level: scale per fitting group = sum of true values / sum of mean predicted values over the fit fold; it multiplies
    every map's value (AOO and EOO each with their own scale).
    Interval: CQR on the 5th-95th percentile of the scaled maps (Romano, Patterson and Candes 2019, NeurIPS 32): score
    max(lo - y, y - hi) on the fit fold, Q = its ceil(0.9 (n + 1))-th smallest value, interval [lo - Q, hi + Q]. AOO lower
    end at least the cells recorded at the target year (known occupied) and 0 (Sara, Q1); EOO lower end at least 0. The
    upper end is raised to the same floor if it falls below it (Q can be negative), so no interval is empty; the true
    value is never below the floor, so this can only add coverage, and the share of stacks where it happens is reported.
    Fitting groups: option 1 recorded class at the target year (7); option 2 recorded class x records-over-6-years bin;
    a fitting group with fewer than 30 fit stacks uses its recorded class (option 2), and a recorded class with fewer
    than 30 uses all fit-fold stacks.
    Grouping rule: a fitting option passes when every evaluation group (the 7 recorded classes and the 5 records bins)
    with 50 or more stacks (Sara, Q2) has a corrected ratio of 0.8-1.25 and coverage of 0.85-0.95; option 1 is used if
    it passes, else option 2 if it passes, else the method fails the rule (reported with option 1). Chosen per method and
    metric.
    P(AOO <= x), x = 1, 5, 10, 20: share of the corrected maps with AOO <= x, the corrected AOO taken to the nearest whole
    cell (found by the self-test: unrounded, a scale of 1.002 moves a map of exactly x cells above x). Checked on the check fold: Brier score,
    Brier skill score against the fit-fold base rate of each recorded class, and a 10-bin reliability table (largest gap
    between mean P and the observed share over bins with 50 or more stacks, also in standard errors of the observed share,
    because a bin of 50 stacks has a standard error near 0.07 even when P is exact).
    Reports: ratio, coverage and median width per evaluation group; AOO by true class (report only); width of each
    baseline over the model (mean of the two seeds' medians) per group; both seeds, their mean and the pooled line.

Gates
    K1  inputs: each run's samples_k50.npz md5 = its g1_report.json; its best.pt md5 = the one G1 used; each G2 map file
        md5 = g2_report.json; the G2 maps were made from this frozen set; all maps cover the frozen stacks in order
    K2  target-year AOO read from the packed maps = the AOO stored by step 3a (model) and by G2 (baselines), every map
    K3  folds by community: every community in one fold, the two folds of equal size (or differing by one)
    K4  leak check: each fold's corrected outputs (model seed 1, AOO and EOO) are unchanged when that fold's own truth is
        shuffled
    K5  EOO code: the hull area equals a plain monotone-chain hull on 2,000 maps of the real data (model and truth), and
        EOO >= AOO for every map
    real_eval is refused; the output folder must be new.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU)
    python3 ../tools/stage_g_calibrate.py --self-test | tail -1
    python3 -u ../tools/stage_g_calibrate.py --data ../data --runs ../runs/f_seed1 ../runs/f_seed2 \\
        --g2 ../data/stage_g2_baselines_v2 --out ../data/stage_g3_calibration > ../logs/stage_g3_calibration.txt 2>&1
Needs numpy, torch (through the imported stage tools) and stage_g_baselines.py, stage_g_samples.py, stage_f_check.py,
stage_f_train.py, stage_d_headroom.py, obj2_model.py, stage_e_inputs.py, stage_d_sampler.py, export_obj2.py.
"""
import argparse, csv, importlib.util, json, math, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g_calibrate.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


GB = _load("stage_g_baselines")
GS, CK = GB.GS, GB.CK
HR, D, X = CK.HR, CK.D, CK.X
G, W = X.G, X.WINDOW
TCLS, BINS, TRUE = D.TCLS, D.BINS, D.TRUE
LEVEL, QLO, QHI = 0.90, 5, 95
BAND, COV, N_TEST, N_FIT_MIN = (0.8, 1.25), (0.85, 0.95), 50, 30
XS = (1, 5, 10, 20)
TB0 = (W - 1) * G * G // 8                     # first byte of the target year in a packed map (250)
CH = 2048
BROKEN = os.environ.get("STAGE_G3_BROKEN", "")  # self-test only: plants one bug so its gate must fail


# ---------------------------------------------------------------- AOO and EOO of maps
def target_year(packed):
    """(..., 300) packed maps -> (..., G, G) bool of the target year (the last 400 bits)."""
    return np.unpackbits(packed[..., TB0:], axis=-1, count=G * G).astype(bool).reshape(packed.shape[:-1] + (G, G))


_I, _J, _R, _T = [], [], [], []
for i in range(G):
    for j in range(i, G):
        for r in range(i, j + 1):
            _I.append(i); _J.append(j); _R.append(r); _T.append(0.0 if j == i else (r - i) / (j - i))
_I, _J, _R, _T = map(np.array, (_I, _J, _R, _T))
_BY_R = [np.nonzero(_R == r)[0] for r in range(G)]
BIG = 1e6


def hull_area(occ):
    """Area of the convex hull of the occupied cell centres, (B, G, G) bool -> (B,). The hull's left side at row r is
    the lower convex envelope of the leftmost occupied column per row (minimum over chords between occupied rows that
    span r), the right side the upper envelope of the rightmost column; the area is the integral of the width, exact by
    the trapezoid rule because every vertex lies on a row."""
    rows = occ.any(2)
    minc = np.where(rows, occ.argmax(2), BIG).astype(np.float64)
    maxc = np.where(rows, G - 1 - occ[:, :, ::-1].argmax(2), -BIG).astype(np.float64)
    if BROKEN == "hull_rows":
        maxc = np.where(rows, minc, -BIG)
    ok = rows[:, _I] & rows[:, _J]
    lo_i, lo_j = minc[:, _I], minc[:, _J]
    hi_i, hi_j = maxc[:, _I], maxc[:, _J]
    cl = np.where(ok, lo_i + (lo_j - lo_i) * _T, BIG)
    cr = np.where(ok, hi_i + (hi_j - hi_i) * _T, -BIG)
    L = np.stack([cl[:, k].min(1) for k in _BY_R], 1)
    R = np.stack([cr[:, k].max(1) for k in _BY_R], 1)
    inside = L < BIG / 2
    w = np.where(inside, R - L, 0.0)
    both = inside[:, :-1] & inside[:, 1:]
    return ((w[:, :-1] + w[:, 1:]) / 2 * both).sum(1)


def eoo_aoo(occ):
    a = occ.reshape(len(occ), -1).sum(1)
    return np.maximum(hull_area(occ), a), a


def brute_area(occ):
    """Monotone-chain hull and shoelace area of one map's occupied cell centres (self-test and gate K5)."""
    pts = sorted(map(tuple, np.argwhere(occ)))
    if len(pts) < 3:
        return 0.0
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = lower[:-1] + upper[:-1]
    return abs(sum(h[k][0] * h[(k + 1) % len(h)][1] - h[(k + 1) % len(h)][0] * h[k][1] for k in range(len(h)))) / 2.0


def map_metrics(packed):
    """(n, K, 300) packed maps -> AOO (n, K) and EOO (n, K) at the target year."""
    n, K = packed.shape[:2]
    aoo = np.zeros((n, K), np.int32); eoo = np.zeros((n, K))
    flat = packed.reshape(n * K, -1)
    for a in range(0, n * K, CH):
        occ = target_year(flat[a:a + CH])
        e, s = eoo_aoo(occ)
        eoo.reshape(-1)[a:a + CH] = e; aoo.reshape(-1)[a:a + CH] = s
    return aoo, eoo


# ---------------------------------------------------------------- cross-fitted correction
def make_folds(comm, seed=(77, 3)):
    cu = np.unique(comm)
    perm = X.rng(*seed).permutation(len(cu))
    first = set(cu[perm[:len(cu) // 2]].tolist())
    return np.array([0 if c in first else 1 for c in comm])


def fit_params(pred, y, idx):
    """Scale and conformal Q fitted on the stacks idx."""
    den = pred[idx].mean(1).sum()
    s = float(y[idx].sum() / den) if den > 0 else 1.0
    sc = s * pred[idx]
    lo, hi = np.percentile(sc, QLO, axis=1), np.percentile(sc, QHI, axis=1)
    e = np.sort(np.maximum(lo - y[idx], y[idx] - hi))
    k = math.ceil(LEVEL * (len(idx) + 1))
    return s, float(e[min(k, len(idx)) - 1])


def crossfit(pred, y, rec_t, fold, fgrp, parent, kind, only=None):
    """Corrected mean, interval and scaled maps of every stack, each from the other fold's fit. fgrp = fitting group
    per stack; parent = the group a small fitting group falls back to (-1 = all stacks)."""
    n = len(y)
    mean_c = np.zeros(n); L = np.zeros(n); U = np.zeros(n); scaled = np.zeros(pred.shape); par = np.zeros((n, 2))
    Lraw = np.zeros(n); Uraw = np.zeros(n)
    for c in (0, 1):
        if only is not None and c != only:
            continue
        fit = np.nonzero(fold == (1 - c))[0] if BROKEN != "leak" else np.arange(n)
        chk = np.nonzero(fold == c)[0]
        cache = {}
        for g in np.unique(fgrp[chk]):
            fg = fit[fgrp[fit] == g]
            key = ("g", g)
            if len(fg) < N_FIT_MIN:
                pg = parent[chk[fgrp[chk] == g][0]]
                fg, key = (fit, ("all",)) if pg < 0 else (fit[parent[fit] == pg], ("p", pg))
                if len(fg) < N_FIT_MIN:
                    fg, key = fit, ("all",)
            if key not in cache:
                cache[key] = fit_params(pred, y, fg)
            s, q = cache[key]
            ci = chk[fgrp[chk] == g]
            sc = s * pred[ci]
            lo, hi = np.percentile(sc, QLO, axis=1), np.percentile(sc, QHI, axis=1)
            lo_c, hi_c = lo - q, hi + q
            floor = np.maximum(rec_t[ci], 0) if kind == "aoo" else np.zeros(len(ci))
            mean_c[ci] = sc.mean(1); L[ci] = np.maximum(lo_c, floor); U[ci] = np.maximum(hi_c, floor); scaled[ci] = sc
            par[ci] = (s, q); Lraw[ci] = lo_c; Uraw[ci] = hi_c
    return dict(mean=mean_c, L=L, U=U, scaled=scaled, par=par, Lraw=Lraw, Uraw=Uraw)


def evaluate(y, out, groups):
    """Ratio, coverage and median width per evaluation group."""
    cov = (y >= out["L"]) & (y <= out["U"])
    wid = out["U"] - out["L"]
    rows = {}
    for gname, (gi, labels) in groups.items():
        rows[gname] = []
        for j, lab in enumerate(labels):
            m = gi == j
            if m.any():
                rows[gname].append(dict(group=lab, n=int(m.sum()), ratio=float(out["mean"][m].mean() / y[m].mean()) if y[m].mean() > 0
                                        else float("nan"), cov=float(cov[m].mean()), width=float(np.median(wid[m]))))
    return rows


def rule_pass(rows):
    bad = [f"{g} {r['group']}" for g in ("recorded class", "records over 6 years") for r in rows[g] if r["n"] >= N_TEST and
           not (BAND[0] <= r["ratio"] <= BAND[1] and COV[0] <= r["cov"] <= COV[1])]
    return len(bad) == 0, bad


def p_le(scaled, x):
    """Share of a stack's corrected maps with AOO <= x, the corrected AOO taken to the nearest whole cell (true AOO is a
    whole number of cells; without rounding a scale of 1.002 would move a map of exactly x cells above x)."""
    return (scaled < x + 0.5).mean(1)


def p_check(scaled, y, fold, rcls):
    """P(AOO <= x) from the corrected maps; Brier, skill against the other fold's base rate per recorded class, and the
    largest reliability gap over bins with 50 or more stacks."""
    out = {}
    for x in XS:
        P = p_le(scaled, x)
        o = (y <= x).astype(float)
        ref = np.zeros(len(y))
        for c in (0, 1):
            fit, chk = fold == 1 - c, fold == c
            for g in np.unique(rcls[chk]):
                m = fit & (rcls == g)
                ref[chk & (rcls == g)] = o[m].mean() if m.any() else o[fit].mean()
        brier, bref = float(((P - o) ** 2).mean()), float(((ref - o) ** 2).mean())
        b = np.minimum((P * 10).astype(int), 9)
        tab = [dict(bin=k, n=int((b == k).sum()), p=float(P[b == k].mean()) if (b == k).any() else float("nan"),
                    obs=float(o[b == k].mean()) if (b == k).any() else float("nan")) for k in range(10)]
        big = [t for t in tab if t["n"] >= N_TEST]
        gaps = [abs(t["p"] - t["obs"]) for t in big]
        # the same gap in standard errors of the observed share (sampling noise of a bin of n stacks if P were exact)
        zs = [abs(t["p"] - t["obs"]) / math.sqrt(max(t["p"] * (1 - t["p"]), 1e-4) / t["n"]) for t in big]
        out[x] = dict(brier=brier, brier_ref=bref, bss=1 - brier / bref if bref > 0 else float("nan"),
                      gap=max(gaps) if gaps else float("nan"), gap_z=max(zs) if zs else float("nan"),
                      mean_p=float(P.mean()), share=float(o.mean()), table=tab)
    return out


# ---------------------------------------------------------------- run
def run(data, runs, g2, out_dir, val_dir=None, out=print):
    data, g2, out_dir = Path(data), Path(g2), Path(out_dir)
    val_dir = Path(val_dir) if val_dir else data / "real_val"
    runs = [Path(r) for r in runs]
    for p in (data, g2, out_dir, val_dir, *runs):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G5")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    if len(runs) != 2:
        sys.exit("STOP: G3 needs the two seeds (--runs seed1 seed2) for the pooled line")
    res, t0 = {}, time.time()

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    fp = data / "stage_d_frozen_val.npz"
    frozen = dict(np.load(fp)); fmd5 = X.file_md5(fp)
    n = len(frozen["row"])
    out(f"stage_g_calibrate.py md5 {X.file_md5(__file__)}; stage_g_baselines.py {X.file_md5(HERE / 'stage_g_baselines.py')}; "
        f"frozen set md5 {fmd5} ({n} stacks)")
    # ---- K1, K2: inputs
    ok1, why1, ok2, why2 = True, [], True, []
    maps, names = {}, []
    for r in runs:
        rep = json.loads((r / "check_g1" / "g1_report.json").read_text())
        sp = r / "check_g1" / "samples_k50.npz"
        m_s, m_b = X.file_md5(sp), X.file_md5(r / "best.pt")
        z = np.load(sp)
        good = m_s == rep["md5"]["samples"] and m_b == rep["md5"]["best_pt"] and np.array_equal(z["sel"], np.arange(n)) \
            and str(z["layout"]) == GS.LAYOUT
        ok1 &= good; why1.append(f"{r.name} samples {m_s[:8]} best.pt {m_b[:8]} {good}")
        pk = z["maps"]
        if BROKEN == "shift_maps" and r == runs[0]:
            pk = np.roll(pk, 1, axis=0)
        a3 = np.load(r / "check_3a" / "preds_k50.npz")["aoo"][:, W - 1, :]
        same = all(np.array_equal(target_year(pk[a:a + 512]).sum((2, 3)), a3[a:a + 512]) for a in range(0, n, 512))
        ok2 &= same; why2.append(f"{r.name} {same}")
        maps[f"model {r.name}"] = pk; names.append(f"model {r.name}")
    g2rep = json.loads((g2 / "g2_report.json").read_text())
    for lab, f in (("B3", "maps_B3_halfblank.npz"), ("B4", "maps_B4.npz"), ("B5", "maps_B5.npz")):
        m = X.file_md5(g2 / f)
        z = np.load(g2 / f)
        good = m == g2rep["md5"][f] and str(z["frozen_md5"]) == fmd5 and np.array_equal(z["sel"], np.arange(n)) \
            and str(z["layout"]) == GS.LAYOUT
        ok1 &= good; why1.append(f"{f} {m[:8]} {good}")
        same = all(np.array_equal(target_year(z["maps"][a:a + 512]).sum((2, 3)), z["aoo_t"][a:a + 512]) for a in range(0, n, 512))
        ok2 &= same; why2.append(f"{lab} {same}")
        maps[lab] = z["maps"]
    gate("K1", ok1, "inputs as logged by G1 and G2, made from this frozen set, stacks in order: " + "; ".join(why1))
    gate("K2", ok2, "target-year AOO read from the packed maps = AOO stored by step 3a and G2, every map: " + "; ".join(why2))
    s1, s2 = names
    order = [s1, s2, "pooled", "B3", "B4", "B5"]

    # ---- metrics of every map (pooled = the first 25 maps of each seed, not recomputed)
    t1 = time.time()
    met = {k: map_metrics(maps[k]) for k in order if k != "pooled"}
    met["pooled"] = tuple(np.concatenate([met[s1][q][:, :25], met[s2][q][:, :25]], 1) for q in (0, 1))
    out(f"   AOO and EOO of {sum(maps[k].shape[0] * maps[k].shape[1] for k in maps):,} maps ({(time.time() - t1) / 60:.1f} min)")

    # ---- truth and observable groups
    V, _, _, T, tc, comm = HR.load_stacks(val_dir, frozen["files"], frozen["file_index"], frozen["row"], truth=True)
    true_eoo, true_aoo = eoo_aoo(T)
    true_aoo = true_aoo.astype(float)
    rec_t = V[:, -1].reshape(n, -1).sum(1)
    rcls, rbin = D.tidx(rec_t), D.bidx(V.any(1).reshape(n, -1).sum(1))
    fold = make_folds(comm)
    groups = {"recorded class": (rcls, TCLS), "records over 6 years": (rbin, BINS), "true class": (tc, TRUE)}
    cu = np.unique(comm)
    one_fold = all(len(np.unique(fold[comm == c])) == 1 for c in cu)
    sizes = [len(np.unique(comm[fold == f])) for f in (0, 1)]
    gate("K3", one_fold and abs(sizes[0] - sizes[1]) <= 1, f"folds by community: {sizes[0]} and {sizes[1]} communities, "
         f"every community in one fold {one_fold}; {int((fold == 0).sum())} and {int((fold == 1).sum())} stacks")

    # ---- K5: hull code on real maps
    g = X.rng(77, 5)
    m5 = min(1000, n)
    pick = np.sort(g.choice(n, m5, replace=False))
    occ = np.concatenate([target_year(maps[s1][pick, 0]), T[np.sort(g.choice(n, m5, replace=False))]])
    fast = hull_area(occ)
    slow = np.array([brute_area(o) for o in occ])
    allge = all(bool((met[k][1] >= met[k][0] - 1e-9).all()) for k in order) and bool((true_eoo >= true_aoo - 1e-9).all())
    gate("K5", bool(np.abs(fast - slow).max() < 1e-9) and allge,
         f"hull area = monotone-chain hull on {2 * m5:,} real maps (largest difference {np.abs(fast - slow).max():.1e}); EOO >= AOO "
         f"for every map {allge}")

    # ---- cross-fit every method, metric and fitting option
    opt = {1: (rcls, np.full(n, -1)), 2: (rcls * len(BINS) + rbin, rcls)}
    results, chosen = {}, {}
    for k in order:
        aoo_m, eoo_m = met[k]
        for kind, pred, y in (("aoo", aoo_m.astype(float), true_aoo), ("eoo", eoo_m, true_eoo)):
            per = {}
            for o_ in (1, 2):
                fg, pa = opt[o_]
                cf = crossfit(pred, y, rec_t, fold, fg, pa, kind)
                rows = evaluate(y, cf, groups)
                ok, bad = rule_pass(rows)
                per[o_] = dict(cf=cf, rows=rows, ok=ok, bad=bad)
            use = 1 if per[1]["ok"] else (2 if per[2]["ok"] else 1)
            chosen[(k, kind)] = dict(option=use, ok=per[use]["ok"], bad=per[use]["bad"], opt1_ok=per[1]["ok"], opt2_ok=per[2]["ok"])
            results[(k, kind)] = per[use]

    # ---- K4: leak check on model seed 1
    leak_ok = True
    for kind, pred, y in (("aoo", met[s1][0].astype(float), true_aoo), ("eoo", met[s1][1], true_eoo)):
        o_ = chosen[(s1, kind)]["option"]
        fg, pa = opt[o_]
        for c in (0, 1):
            y2 = y.copy()
            idx = np.nonzero(fold == c)[0]
            y2[idx] = y[X.rng(77, 7, c).permutation(idx)]
            a_ = crossfit(pred, y, rec_t, fold, fg, pa, kind, only=c)
            b_ = crossfit(pred, y2, rec_t, fold, fg, pa, kind, only=c)
            leak_ok &= all(np.array_equal(a_[q][idx], b_[q][idx]) for q in ("mean", "L", "U"))
    gate("K4", leak_ok, "each fold's corrected AOO and EOO (model seed 1) unchanged when that fold's own truth is shuffled")

    # ---- reports
    mean_seeds = {kind: {g: [((a["ratio"] + b["ratio"]) / 2, (a["cov"] + b["cov"]) / 2) for a, b in
                             zip(results[(s1, kind)]["rows"][g], results[(s2, kind)]["rows"][g])] for g in ("recorded class", "records over 6 years")}
                  for kind in ("aoo", "eoo")}
    clip = {f"{k} {kind}": [float((cf["L"] > cf["Lraw"]).mean()), float((cf["U"] > cf["Uraw"]).mean())]
            for k in order for kind in ("aoo", "eoo") for cf in [results[(k, kind)]["cf"]]}
    pchk = {k: p_check(results[(k, "aoo")]["cf"]["scaled"], true_aoo, fold, rcls) for k in order}
    for kind in ("aoo", "eoo"):
        out(f"   {kind.upper()} after correction (check folds; ratio / coverage / median width; rule: ratio {BAND[0]}-{BAND[1]}, "
            f"coverage {COV[0]}-{COV[1]} in every group of {N_TEST}+ stacks):")
        for k in order:
            ch = chosen[(k, kind)]
            rows = results[(k, kind)]["rows"]
            out(f"      {k:<14} option {ch['option']} ({'recorded class' if ch['option'] == 1 else 'recorded class x records bin'})"
                f" rule {'PASS' if ch['ok'] else 'FAIL'}" + ("" if ch["ok"] else f" ({', '.join(ch['bad'][:6])})")
                + f"; option 1 {ch['opt1_ok']}, option 2 {ch['opt2_ok']}")
            for gname in ("recorded class", "records over 6 years"):
                out(f"         {gname:<20} " + "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}/{r['width']:.1f}"
                                                        + ("" if r["n"] >= N_TEST else "*") for r in rows[gname]))
        out(f"      {'mean of seeds':<14} " + " | ".join("; ".join(f"{r['group']} {m[0]:.2f}/{m[1]:.2f}" for r, m in
                                                         zip(results[(s1, kind)]["rows"][g], mean_seeds[kind][g]))
                                                for g in ("recorded class", "records over 6 years")))
        out("      stacks per group (* = under " + f"{N_TEST}, reported, not tested): " + " | ".join(
            "; ".join(f"{r['group']} {r['n']}" for r in results[(s1, kind)]["rows"][g]) for g in ("recorded class", "records over 6 years")))
    for kind in ("aoo", "eoo"):
        out(f"   {kind.upper()} by true class (report only; ratio / coverage):")
        for k in order:
            out(f"      {k:<14} " + "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}" for r in results[(k, kind)]["rows"]["true class"]))
    out("   width over the model (baseline median width / mean of the two seeds' median widths), by recorded class:")
    for kind in ("aoo", "eoo"):
        mw = {r["group"]: (r["width"] + q["width"]) / 2 for r, q in zip(results[(s1, kind)]["rows"]["recorded class"],
                                                                         results[(s2, kind)]["rows"]["recorded class"])}
        for k in ("pooled", "B3", "B4", "B5"):
            out(f"      {kind.upper()} {k:<6} " + "; ".join(f"{r['group']} {r['width'] / mw[r['group']]:.2f}" if mw[r["group"]] > 0
                                                       else f"{r['group']} n/a" for r in results[(k, kind)]["rows"]["recorded class"]))
    out("   P(AOO <= x) on the check folds (Brier / skill against the other fold's base rate per recorded class / largest "
        "reliability gap over bins of 50+ stacks, and that gap in standard errors; mean P against the share with true AOO <= x):")
    for k in order:
        out(f"      {k:<14} " + "; ".join(f"x {x}: {pchk[k][x]['brier']:.3f} / {pchk[k][x]['bss']:+.3f} / {pchk[k][x]['gap']:.3f} "
                                         f"({pchk[k][x]['gap_z']:.1f} se); {pchk[k][x]['mean_p']:.3f} vs {pchk[k][x]['share']:.3f}" for x in XS))
    out("   share of stacks with the lower end / upper end raised to the floor (AOO: cells recorded at the target year; EOO: 0): "
        + ", ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in clip.items()))

    # ---- files
    out_dir.mkdir(parents=True, exist_ok=True)
    save = dict(fold=fold, comm=comm, rcls=rcls, rbin=rbin, rec_t=rec_t, true_class=tc, true_aoo=true_aoo, true_eoo=true_eoo)
    for k in order:
        kk = k.replace(" ", "_")
        save[f"aoo_{kk}"] = met[k][0].astype(np.uint16); save[f"eoo_{kk}"] = met[k][1].astype(np.float32)
        for kind in ("aoo", "eoo"):
            cf = results[(k, kind)]["cf"]
            for q in ("mean", "L", "U"):
                save[f"{kind}_{q}_{kk}"] = cf[q]
            save[f"{kind}_scale_q_{kk}"] = cf["par"]
        save[f"p_le_{kk}"] = np.stack([p_le(results[(k, "aoo")]["cf"]["scaled"], x) for x in XS], 1)
    X.save_npz(out_dir / "g3_calibrated.npz", save)
    with open(out_dir / "per_stack_g3.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["stack", "community", "fold", "true_class", "recorded_class", "records_6y_bin", "aoo_true", "eoo_true"]
        for k in order:
            kk = k.replace(" ", "_")
            head += [f"{kind}_{q}_{kk}" for kind in ("aoo", "eoo") for q in ("mean", "lo", "hi")] + [f"p_aoo_le_{x}_{kk}" for x in XS]
        w.writerow(head)
        for i in range(n):
            row = [i, int(comm[i]), int(fold[i]), TRUE[tc[i]], TCLS[rcls[i]], BINS[rbin[i]], int(true_aoo[i]), f"{true_eoo[i]:.2f}"]
            for k in order:
                kk = k.replace(" ", "_")
                for kind in ("aoo", "eoo"):
                    cf = results[(k, kind)]["cf"]
                    row += [f"{cf['mean'][i]:.3f}", f"{cf['L'][i]:.3f}", f"{cf['U'][i]:.3f}"]
                row += [f"{v:.3f}" for v in save[f"p_le_{kk}"][i]]
            w.writerow(row)
    rep = dict(gates=res, md5=dict(tool=X.file_md5(__file__), calibrated=X.file_md5(out_dir / "g3_calibrated.npz")),
               chosen={f"{k} {kind}": v for (k, kind), v in chosen.items()},
               rows={f"{k} {kind}": v["rows"] for (k, kind), v in results.items()},
               p_le={k: {str(x): {kk: vv for kk, vv in v.items()} for x, v in pchk[k].items()} for k in order},
               mean_of_seeds=mean_seeds, clip_share=clip, n=int(n), minutes=round((time.time() - t0) / 60, 1))
    (out_dir / "g3_report.json").write_text(json.dumps(rep, indent=1))
    n_ok = sum(res.values())
    verdict = ", ".join(f"{k} {kind.upper()} {'PASS' if chosen[(k, kind)]['ok'] else 'FAIL'}" for k in order for kind in ("aoo", "eoo"))
    out(f"   written g3_calibrated.npz md5 {rep['md5']['calibrated']}, per_stack_g3.csv, g3_report.json ({(time.time() - t0) / 60:.1f} min)")
    out(f"stage G3: gates {n_ok} PASS, {len(res) - n_ok} FAIL; calibration rule: {verdict}")
    return res, dict(results=results, chosen=chosen, pchk=pchk, met=met, true_aoo=true_aoo, true_eoo=true_eoo, fold=fold)


# ---------------------------------------------------------------- self-test
def unit_checks():
    g = np.random.default_rng(5)
    res = {}
    occ = g.random((600, G, G)) < g.choice([0.002, 0.01, 0.05, 0.3], (600, 1, 1))
    occ[:6] = False; occ[1, 3, 4] = True; occ[2, 5, 2:9] = True; occ[3, 2:12, 7] = True
    for k in range(10):
        occ[4, k, k] = True
    occ[5, 0, 0] = occ[5, 0, 2] = occ[5, 2, 0] = True
    fast = hull_area(occ)
    slow = np.array([brute_area(o) for o in occ])
    e, a = eoo_aoo(occ)
    res["U1"] = (bool(np.abs(fast - slow).max() < 1e-9 and fast[:5].max() == 0 and abs(fast[5] - 2.0) < 1e-12
                      and (e >= a).all() and e[1] == 1 and e[2] == 7), float(np.abs(fast - slow).max()))
    full = np.ones((1, G, G), bool)
    res["U2"] = bool(abs(hull_area(full)[0] - (G - 1) ** 2) < 1e-9 and eoo_aoo(full)[0][0] == G * G)
    mp = g.random((3, 4, W, G, G)) < 0.2
    pk = GS.pack(mp)
    res["U3"] = bool(np.array_equal(target_year(pk), mp[:, :, W - 1]))
    # calibration on synthetic data: true y ~ Poisson(lam), maps ~ Poisson(1.6 lam), 20 communities
    n, K = 6000, 50
    comm = g.integers(0, 20, n)
    lam = g.gamma(2.0, 4.0, n) + 0.5
    y = g.poisson(lam).astype(float)
    pred = g.poisson(1.6 * lam[:, None] * g.gamma(4.0, 0.25, (n, 1)), (n, K)).astype(float)
    grp = np.minimum((lam // 4).astype(int), 6)
    fold = make_folds(comm)
    cf = crossfit(pred, y, np.zeros(n), fold, grp, np.full(n, -1), "aoo")
    cov = float(((y >= cf["L"]) & (y <= cf["U"])).mean())
    ratio = float(cf["mean"].mean() / y.mean())
    res["U4"] = (abs(cov - LEVEL) < 0.02 and abs(ratio - 1) < 0.03, cov, ratio)
    yq = np.arange(1, 101, dtype=float)[::-1].copy()          # scores 1..100 (maps all 0, scale falls back to 1)
    res["U5"] = (fit_params(np.zeros((100, 3)), yq, np.arange(100)) == (1.0, 91.0), fit_params(np.zeros((100, 3)), yq, np.arange(100)))
    # P(AOO <= x): calibrated when the truth is one more draw from the same distribution as the maps (200 maps so the
    # map noise is small); maps 1.6 times too high and not corrected give a clear miss
    lam2 = g.gamma(2.0, 3.0, n) + 0.3
    pred2 = g.poisson(lam2[:, None], (n, 200)).astype(float)
    y2 = g.poisson(lam2).astype(float)
    cf2 = crossfit(pred2, y2, np.zeros(n), fold, np.zeros(n, int), np.full(n, -1), "aoo")
    pc = p_check(cf2["scaled"], y2, fold, np.zeros(n, int))
    pb = p_check(1.6 * pred2, y2, fold, np.zeros(n, int))
    res["U6"] = (all(pc[x]["gap_z"] < 3.5 and abs(pc[x]["mean_p"] - pc[x]["share"]) < 0.01 and pc[x]["bss"] > 0.2 for x in (1, 5, 10))
                 and pb[5]["gap_z"] > 6, {x: (round(pc[x]["gap"], 3), round(pc[x]["gap_z"], 1), round(pc[x]["bss"], 3)) for x in XS},
                 round(pb[5]["gap_z"], 1))
    # floor at the recorded cells: recorded cells are occupied, so the truth is never below them and raising the lower
    # end to them can only keep or raise coverage
    rec = np.minimum(y, 5.0)
    cf3 = crossfit(pred, y, rec, fold, grp, np.full(n, -1), "aoo")
    cov3 = float(((y >= cf3["L"]) & (y <= cf3["U"])).mean())
    res["U7"] = (bool((cf3["L"] >= rec).all() and (cf3["U"] >= cf3["L"]).all() and cov3 >= cov
                      and np.array_equal(cf3["Uraw"], cf["Uraw"])), cov3)
    # an upper end below the floor is raised to it: maps spread far too wide give Q < 0 and a raw interval [3, 3]; the
    # floor is set to 4 (above the truth, impossible in real data) only to reach this branch
    yb = np.full(n, 3.0); pb_ = np.tile(np.linspace(0, 60, 50), (n, 1))
    cf5 = crossfit(pb_, yb, np.full(n, 4.0), fold, np.zeros(n, int), np.full(n, -1), "aoo")
    res["U9"] = (bool(cf5["par"][0, 1] < 0 and np.allclose(cf5["Uraw"], 3) and (cf5["U"] == 4).all() and (cf5["L"] == 4).all()),
                 float(cf5["par"][0, 1]))
    # small fitting groups fall back to their parent, then to all stacks
    tiny = np.where(np.arange(n) < 40, 99, grp)
    cf4 = crossfit(pred, y, np.zeros(n), fold, tiny, np.where(np.arange(n) < 40, 98, -1), "aoo")
    allp = {c: fit_params(pred, y, np.nonzero(fold == 1 - c)[0]) for c in (0, 1)}
    res["U8"] = bool(all(tuple(cf4["par"][i]) == allp[fold[i]] for i in range(40)))
    return res


def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    u = unit_checks()
    t("U1", u["U1"][0], f"hull area = monotone-chain hull on 600 maps incl. 0-2 cells, a row, a column, a diagonal and a "
                        f"triangle (largest difference {u['U1'][1]:.1e}); EOO >= AOO; 1 cell and a 7-cell row give EOO = AOO")
    t("U2", u["U2"], f"a full window: hull area {(G - 1) ** 2} (cell centres), EOO raised to AOO {G * G}")
    t("U3", u["U3"], "the target year read from the last 50 bytes equals the unpacked year 6")
    t("U4", u["U4"][0], f"synthetic maps 1.6 times too high: cross-fitted ratio {u['U4'][2]:.3f}, 90% coverage {u['U4'][1]:.3f}")
    t("U5", u["U5"][0], f"conformal scores 1..100: Q is the ceil(0.9 x 101) = 91st smallest ({u['U5'][1]})")
    t("U6", u["U6"][0], f"P(AOO <= x) from maps drawn like the truth: (gap, gap in standard errors, skill) {u['U6'][1]}; "
                        f"maps 1.6 times too high, uncorrected: gap {u['U6'][2]} standard errors")
    t("U7", u["U7"][0], f"AOO lower ends raised to the recorded cells; coverage kept or raised ({u['U7'][1]:.3f}); raw upper ends unchanged")
    t("U9", u["U9"][0], f"with Q below 0 ({u['U9'][1]:.2f}) a raw upper end below the floor is raised to it, so no interval is empty")
    t("U8", u["U8"], "a fitting group with under 30 fit stacks whose parent is also under 30 uses all fit-fold stacks")
    env0 = dict(os.environ, STAGE_G3_BROKEN="hull_rows")
    code = ("import importlib.util as u; s=u.spec_from_file_location('c', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
            "print('RES', m.unit_checks()['U1'][0])" % str(Path(__file__).resolve()))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env0)
    t("bug hull_rows", "RES False" in p.stdout, f"a hull with the right side taken from the left fails U1 ({p.stdout.strip()[-12:] or p.stderr.strip()[-200:]})")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = X.plant_grids(tmp / "g")
        st = tmp / "sim" / "real_train"; st.mkdir(parents=True); sv = tmp / "sim" / "real_val"; sv.mkdir(parents=True)
        for s_ in (7001, 7002, 7003):
            X.plant_community(st / f"c{s_}_training.npz", s_)
        for s_ in (7101, 7102, 7103, 7104):
            X.plant_community(sv / f"c{s_}_training.npz", s_)
        base = tmp / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.3, py, refpat, tools, out=quiet)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.3, py, refpat, tools, out=quiet)
        D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
        hres, hinfo = HR.run(str(base), str(base / "real_train"), str(base / "real_val"), str(base), out=quiet)
        e1, e2 = round(hinfo["bal"]["B1"], 3), round(hinfo["bal"]["B2"], 3)
        runs = []
        for sd in (1, 2):
            cfg = dict(CK.F.CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=2, K=2, ddim=4,
                       val_set="full", val_chunk=64, lr=1e-3, seed=sd)
            rd = tmp / f"run{sd}"
            CK.F.train(cfg, str(base), str(rd), "cpu", log=quiet)
            CK.run(rd, base, K=50, ddim=4, dev="cpu", expect_b1=e1, expect_b2=e2, out=quiet)
            GS.run_one(rd, base, K=50, ddim=4, dev="cpu", out=quiet)
            runs.append(rd)
        GB.run(base, base / "real_train", base / "real_val", tmp / "g2", runs=runs, K=50, out=quiet)
        lines = []
        r, info = run(base, runs, tmp / "g2", tmp / "g3", out=lines.append)
        t("S1", all(r.get(k) for k in ("K1", "K2", "K3", "K4", "K5")), f"planted data end to end: {r}")
        z = np.load(tmp / "g3" / "g3_calibrated.npz")
        cf = info["results"][("B4", "aoo")]["cf"]
        t("S2", z["aoo_model_run1"].shape[1] == 50 and z["aoo_pooled"].shape[1] == 50 and z["aoo_B5"].shape[1] == 1
          and np.array_equal(z["aoo_pooled"][:, :25], z["aoo_model_run1"][:, :25]) and np.array_equal(z["aoo_pooled"][:, 25:], z["aoo_model_run2"][:, :25])
          and np.allclose(z["aoo_U_B4"], cf["U"]), "outputs: 50 maps per model and baseline, 1 for B5, pooled = first 25 of each seed")
        ch = info["chosen"]
        t("S3", all(k in ch for k in [("pooled", "aoo"), ("B3", "eoo"), ("B5", "aoo")]) and any("rule" in x for x in lines),
          "every method and metric gets a grouping and a rule verdict: " + next((x for x in lines if x.startswith("stage G3")), "")[-160:])
        stop = []
        for args in ((base, runs, tmp / "g2", tmp / "g3"), (base, runs, tmp / "g2", tmp / "real_eval_g3")):
            try:
                run(*args, out=quiet)
                stop.append(False)
            except SystemExit:
                stop.append(True)
        t("S4", all(stop), f"an output folder that holds outputs is refused {stop[0]}; a real_eval path is refused {stop[1]}")
        g2bad = tmp / "g2bad"
        g2bad.mkdir()
        for f in (tmp / "g2").iterdir():
            (g2bad / f.name).write_bytes(f.read_bytes())
        zz = dict(np.load(g2bad / "maps_B4.npz")); zz["maps"] = zz["maps"].copy(); zz["maps"][0, 0, -1] ^= 1
        np.savez(g2bad / "maps_B4.npz", **zz)
        rb, _ = run(base, runs, g2bad, tmp / "g3_bad", out=quiet)
        t("bug changed_map", rb.get("K1") is False and rb.get("K2") is False, f"a B4 map changed after G2 fails K1 and K2 ({rb})")
        for bug, gate in (("shift_maps", "K2"), ("leak", "K4"), ("hull_rows", "K5")):
            env = dict(os.environ, STAGE_G3_BROKEN=bug)
            code = ("import importlib.util as u; s=u.spec_from_file_location('c', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r,_=m.run(%r, [%r, %r], %r, %r, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), str(base), str(runs[0]), str(runs[1]), str(tmp / "g2"), str(tmp / f"b_{bug}")))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--runs", nargs="*", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--out", default="../data/stage_g3_calibration")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    res, _ = run(a.data, a.runs, a.g2, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if res and all(res.values()) else 1)