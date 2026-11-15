#!/usr/bin/env python3
"""
overlap_capacity_check.py  (obj1_v2, 30 Oct 2026; Sara's yes, given on condition that the change follows from the
results and targets the root cause)

WITHDRAWN 30 Oct, before any run on Sara's machine. Its own tests on planted pools shaped like the kept species of
best founding (240 species, optima packed near the habitat range, widths 0.2-1, k 0.5 peaks, 60 cells, 2 seeds) gave
closed-cell capacity at pools of 30 / 60 / 120 / 240 species:
    equal competition (today)            7 / 7-8 / 11 / 13-16   rises with the pool (the cause, confirmed)
    C = 0.4 + k x 0.6 x overlap, k 0.05   6 / 8 / 10 / 12-15     still rises
                                 k 0.2    5-6 / 6-7 / 8-9 / 10-12 still rises
                                 k 0.4    4 / 3-4 / 6-7 / 9       still rises
                                 k 1      1 / 1 / 1 / 1          one species per cell
No strength of the overlap penalty both levels off and keeps several species per cell. Reason: each cell has one
habitat value, so the species that can live in a cell all have similar niches on the habitat axis; a full penalty
excludes all but one, a partial one keeps the growth with the pool. Kept as a record and for the self-test.

Question: does a cell's own capacity (species that persist with no arrivals) grow with the size of the regional
pool because every pair of species competes at the same strength, and would competition set by niche overlap stop
that? Evidence (sink_growth.py v2, 30 Oct): random founding, pools of 27-33 species, 8-9 per closed cell; best
founding, pools of 198-244 species, 26-38; capacity also rose with niche width (24, 30.5, 33.5) although wider
niches overlap more. Reads saved worlds only; nothing is simulated; writes nothing.

Two competition rules, both Lotka-Volterra with C fixed in time and C_ii = 1:
  equal    C_ij = s (the saved INTERACTION_STRENGTH) for every pair: the rule the worlds were made with
  overlap  C_ij = s + (1 - s) a_ij, with a_ij the overlap of the two Gaussian niches on the habitat axis
           (normalised so a species overlaps itself by 1): sqrt(2 w_i w_j / (w_i^2 + w_j^2))
           exp(-(opt_i - opt_j)^2 / (2 (w_i^2 + w_j^2))). Species with the same niche compete as hard as a species
           with itself; species with distant niches compete at s, as now. a = 0 gives the equal rule back, so the
           change adds no free number. (Overlap alone, without the floor s, was tried first while building this
           check: on a planted pool it left one species per cell, because each cell has one habitat value, so the
           habitat axis cannot stand for all the resources species share within a cell.)
Closed cell: the Lotka-Volterra equilibrium of the cell without arrivals, from the species of the pool that can
grow there alone (r > 0), counting species with 1 individual or more. Equal rule: the closed form of
sink_growth.py. Overlap rule: the same equilibrium as a non-negative least-squares problem (exact for a symmetric
positive definite C; the overlap matrix is a Gram matrix, so it is), members below 1 individual dropped one at a
time, lowest first.

Pools: random subsets of 30, 60 and 120 species (3 draws each, fixed seed) and the whole pool, per world;
--cells cells per world (default 100, a fixed random choice).

Calibration first (stop if it fails): the least-squares solver with the equal rule gives the closed form's answer
in 99% of cells or more, at the whole pool of every world.

Reading rule (written before the run, 30 Oct), on best-founding worlds (medians over worlds of the median capacity
per cell):
  A  equal rule, whole pool / pool 30 >= 1.5 -> capacity grows with the pool: cause confirmed. Below 1.5 -> not
     confirmed; the overlap idea is withdrawn.
  B  overlap rule levels off: whole pool / pool 60 <= 1.25; and lands near random founding: whole-pool capacity
     between 6 and 13 (random founding: 8-9 closed, 13 present per cell).
  A and B -> put competition by niche overlap into the IBM (next stage: code, tests, one GPU batch of 4 worlds).
  A, levels off but below 6 -> add a scale c (C_ij = c x overlap) and check c values here first.
  A, does not level off -> overlap is not the fix; discuss.
Also reported (description): species that persist in at least one closed cell at the whole pool, all and those
with true ranges of 1-5 cells, under each rule (the rare species of the aim must keep a place).

Run from obj1_v2/work (needs ../tools/sink_growth.py version 2 next to this file):
    python3 ../tools/overlap_capacity_check.py --self-test | tee ../logs/overlap_capacity_selftest_30oct.txt
    python3 ../tools/overlap_capacity_check.py --world "results/stage2a/niche_d/*_training.npz" \\
        --world "results/stage2a/niche_b/*_training.npz" --world "results/stage2a/niche_w/*_training.npz" \\
        | tee ../logs/overlap_capacity_30oct.txt
"""
import argparse, glob, importlib.util, itertools, os, sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

_spec = importlib.util.spec_from_file_location("sink_growth", Path(__file__).with_name("sink_growth.py"))
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)                     # closed_count (equal rule) and scalar come from there
if not hasattr(sg, "closed_count"):
    sys.exit("STOP: ../tools/sink_growth.py is not version 2 (no closed_count)")

POOLS = (30, 60, 120)
DRAWS = 3


def overlap(opt, w):
    """Normalised overlap of Gaussian niches: 1 on the diagonal, symmetric, a Gram matrix."""
    o, w = np.asarray(opt, float), np.asarray(w, float)
    s2 = w[:, None] ** 2 + w[None, :] ** 2
    return np.sqrt(2 * w[:, None] * w[None, :] / s2) * np.exp(-(o[:, None] - o[None, :]) ** 2 / (2 * s2))


def closed_general(r, C, thr):
    """Persisting species of a closed cell for symmetric positive definite C: B >= 0 minimising 0.5 B'CB - r'B
    (the Lotka-Volterra equilibrium, g = r - C B <= 0 for the excluded, = 0 for members), members below thr
    dropped one at a time, lowest first. r (n,), C (n, n) -> boolean (n,) and B."""
    idx = np.flatnonzero(r > 0)
    keep = np.zeros(len(r), bool); Bfull = np.zeros(len(r))
    while len(idx):
        Cs = C[np.ix_(idx, idx)]
        try:
            L = np.linalg.cholesky(Cs)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Cs + 1e-10 * np.eye(len(idx)))
        B, _ = nnls(L.T, np.linalg.solve(L, r[idx]))
        pos = B > 0
        low = pos & (B < thr)
        if not low.any():
            keep[idx[pos]] = True; Bfull[idx] = B
            return keep, Bfull
        drop = np.flatnonzero(low)[np.argmin(B[low])]
        idx = np.delete(idx, drop)
    return keep, Bfull


def c_overlap(opt, w, s):
    """Overlap rule: C = s + (1 - s) a, diagonal 1."""
    return s + (1 - s) * overlap(opt, w)


def capacity(r_cells, C, s, rule, thr):
    """Persisting species per cell; r_cells (n, M)."""
    if rule == "equal":
        return sg.closed_count(r_cells, s, thr)
    out = np.zeros(r_cells.shape, bool)
    for m in range(r_cells.shape[1]):
        out[:, m] = closed_general(r_cells[:, m], C, thr)[0]
    return out


def load(f, ncell, seed):
    z = np.load(f, allow_pickle=True)
    need = ["ENV_r_field", "NICHE_opt", "NICHE_w", "INTERACTION_STRENGTH", "BODY_MASS", "P_last_final"]
    miss = [k for k in need if k not in z.files]
    if miss:
        return None, f"skipped (missing {miss}): {os.path.basename(f)[:80]}"
    if "CONNECTANCE" not in z.files or abs(float(z["CONNECTANCE"]) - 1) > 1e-6:
        return None, f"skipped (connectance not 1): {os.path.basename(f)[:80]}"
    r = np.asarray(z["ENV_r_field"], float); S, G, _ = r.shape
    cells = np.sort(np.random.default_rng(seed).choice(G * G, min(ncell, G * G), replace=False))
    rng_cells = (np.asarray(z["P_last_final"]) > 0).reshape(S, -1).sum(1)
    w = dict(name=os.path.basename(f), rule_f=sg.scalar(z, "FOUND_rule", "random"), S=S,
             w_min=float(sg.scalar(z, "NICHE_w_min", 0.2)), s=float(z["INTERACTION_STRENGTH"]),
             body=float(z["BODY_MASS"]), r=r.reshape(S, -1)[:, cells], opt=np.asarray(z["NICHE_opt"], float),
             wd=np.asarray(z["NICHE_w"], float), small=(rng_cells >= 1) & (rng_cells <= 5))
    return w, None


def analyse(w, seed):
    S, s, thr = w["S"], w["s"], w["body"]
    Cov = c_overlap(w["opt"], w["wd"], s)
    res = {}
    rng = np.random.default_rng(seed)
    sizes = [n for n in POOLS if n < S] + [S]
    for n in sizes:
        draws = [np.arange(S)] if n == S else [np.sort(rng.choice(S, n, replace=False)) for _ in range(DRAWS)]
        for rule in ("equal", "overlap"):
            meds = []
            for sub in draws:
                k = capacity(w["r"][sub], Cov[np.ix_(sub, sub)], s, rule, thr)
                meds.append(float(np.median(k.sum(0))))
                if n == S:
                    res[("persist", rule)] = (int(k.any(1).sum()), int((k.any(1) & w["small"]).sum()), int(w["small"].sum()))
            res[(n, rule)] = float(np.median(meds))
    Ceq = s * np.ones((S, S)) + (1 - s) * np.eye(S)
    a = sg.closed_count(w["r"], s, thr)
    b = np.stack([closed_general(w["r"][:, m], Ceq, thr)[0] for m in range(w["r"].shape[1])], 1)
    res["calib"] = float(np.mean(np.all(a == b, 0)))
    res["sizes"] = sizes
    return res


def verdict(rise_eq, level_ov, full_ov):
    if not np.isfinite(rise_eq):
        return "not read"
    if rise_eq < 1.5:
        return "A not confirmed: capacity does not grow with the pool under equal competition; the overlap idea is withdrawn"
    if not np.isfinite(level_ov) or level_ov > 1.25:
        return "A confirmed, but the overlap rule does not level off: overlap is not the fix; discuss"
    if full_ov < 6:
        return "A confirmed, overlap levels off but below 6 per cell: check a scale c (C = c x overlap) here first"
    if full_ov <= 13:
        return "A and B: put competition by niche overlap into the IBM (code, tests, one GPU batch of 4 worlds)"
    return "A confirmed, overlap levels off but above 13 per cell: discuss before any change"


def run(patterns, ncell):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        sys.exit("STOP: no files matched")
    ws = []
    for i, f in enumerate(files):
        w, msg = load(f, ncell, 1000 + i)
        print(msg) if msg else ws.append(w)
    if not ws:
        sys.exit("STOP: no world read")
    rows = []
    for i, w in enumerate(ws):
        rows.append((w, analyse(w, 2000 + i)))
    print(f"\nCalibration: least-squares solver with the equal rule = closed form, share of cells ({ncell} per world)")
    ok = True
    for w, res in rows:
        good = res["calib"] >= 0.99; ok &= good
        print(f"   {w['name'][-34:]}: {res['calib']:.1%} {'PASS' if good else 'FAIL'}")
    if not ok:
        print("\nSTOP: the solver does not reproduce the closed form; nothing below is read.")
        return False
    print("\nCapacity per cell with no arrivals (median over cells; pools of 30, 60, 120: median of 3 draws)")
    print("   world                                 founding  w_min  species | equal: 30 / 60 / 120 / all | overlap: 30 / 60 / 120 / all"
          " | persist somewhere, equal / overlap (of them 1-5 cell species)")
    for w, res in rows:
        f = lambda rule: " / ".join(f"{res[(n, rule)]:4.1f}" if (n, rule) in res else "   -" for n in list(POOLS) + [w["S"]])
        pe, po = res[("persist", "equal")], res[("persist", "overlap")]
        print(f"   {w['name'][-36:]:36s}  {w['rule_f']:6s}    {w['w_min']:.1f}    {w['S']:4d}  | {f('equal')} | {f('overlap')} | "
              f"{pe[0]} / {po[0]} ({pe[1]} / {po[1]} of {pe[2]})")
    best = [(w, r) for w, r in rows if w["rule_f"] == "best"]
    if not best:
        print("\nReading rule: not read (no best-founding world)")
        return True
    print("\nReading rule (written before the run), best-founding worlds, medians over worlds:")
    for wm in sorted({w["w_min"] for w, _ in best}):
        g = [(w, r) for w, r in best if abs(w["w_min"] - wm) < 1e-6]
        rise = float(np.median([r[(r["sizes"][-1], "equal")] / r[(30, "equal")] for w, r in g if (30, "equal") in r and r[(30, "equal")] > 0]))
        lev = float(np.median([r[(r["sizes"][-1], "overlap")] / r[(60, "overlap")] for w, r in g if (60, "overlap") in r and r[(60, "overlap")] > 0]))
        full = float(np.median([r[(r["sizes"][-1], "overlap")] for w, r in g]))
        print(f"   w_min {wm:.1f} ({len(g)} worlds): A equal all / 30 = x{rise:.2f}; B overlap all / 60 = x{lev:.2f}, overlap at the whole pool "
              f"{full:.1f} per cell -> {verdict(rise, lev, full)}")
    return True


# ---------------------------------------------------------------- self-test
def self_test():
    rng = np.random.default_rng(0); ok = True

    def check(name, cond, text=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name} {text} {'PASS' if cond else 'FAIL'}")

    o, w = rng.normal(0, 1, 40), rng.uniform(0.2, 1.0, 40); A = overlap(o, w)
    x = np.linspace(-15, 15, 30001); dx = x[1] - x[0]
    u = lambda oo, ww: np.exp(-(x - oo) ** 2 / (2 * ww ** 2))
    num = [np.sum(u(o[i], w[i]) * u(o[j], w[j])) * dx / np.sqrt(np.sum(u(o[i], w[i]) ** 2) * dx * np.sum(u(o[j], w[j]) ** 2) * dx)
           for i, j in ((0, 1), (2, 3), (4, 5))]
    Co = c_overlap(o, w, 0.4); far = c_overlap(np.array([0.0, 50.0]), np.array([0.3, 0.3]), 0.4)
    check("T1", np.allclose(np.diag(A), 1) and np.allclose(A, A.T) and np.linalg.eigvalsh(A).min() > -1e-9
          and np.allclose(num, [A[0, 1], A[2, 3], A[4, 5]], atol=1e-6) and np.allclose(np.diag(Co), 1)
          and np.linalg.eigvalsh(Co).min() > -1e-9 and abs(far[0, 1] - 0.4) < 1e-9 and Co.min() >= 0.4 - 1e-12,
          "overlap: 1 on the diagonal, symmetric, positive semidefinite, equal to the integral of the two niches; "
          "rule C = s + (1 - s) a: diagonal 1, distant niches at s, never below s")

    bad = 0
    for t in range(120):
        n = int(rng.integers(1, 7)); oo, ww = rng.normal(0, 0.5, n), rng.uniform(0.2, 0.8, n)
        C = c_overlap(oo, ww, 0.4) + 1e-6 * np.eye(n); r = rng.normal(0.5, 0.4, n); want = None
        for k in range(n + 1):
            for sub in itertools.combinations(range(n), k):
                B = np.zeros(n)
                if k:
                    ix = list(sub); B[ix] = np.linalg.solve(C[np.ix_(ix, ix)], r[ix])
                    if (B[ix] <= 0).any():
                        continue
                if np.all((r - C @ B)[[i for i in range(n) if i not in sub]] <= 1e-9):
                    want = B > 0
        bad += not np.array_equal(closed_general(r, C, 0.0)[0], want)
    check("T2", bad == 0, f"overlap rule: least-squares equilibrium = brute force in 120 random cells ({bad} differ)")

    bad = 0
    for t in range(200):
        n = int(rng.integers(1, 30)); s = float(rng.choice([0.2, 0.4, 0.8])); r = rng.normal(0.6, 0.3, n)
        thr = float(rng.choice([0.0, 1e-4, 0.05]))
        Ceq = s * np.ones((n, n)) + (1 - s) * np.eye(n)
        bad += not np.array_equal(closed_general(r, Ceq, thr)[0], sg.closed_count(r[:, None], s, thr)[:, 0])
    check("T3", bad == 0, f"equal rule: least-squares solver = sink_growth closed form, with thresholds, 200 cells ({bad} differ)")

    # T4 planted pools: optima spread evenly over the habitat, one habitat value per cell
    def pool_world(S, seed):
        r_ = np.random.default_rng(seed)
        opt = r_.uniform(-0.6, 0.6, S); wd = np.full(S, 0.15); H = r_.uniform(-0.3, 0.3, 50)
        r = 1.2 * np.exp(-(H[None] - opt[:, None]) ** 2 / (2 * wd[:, None] ** 2)) - 0.1
        return dict(name="plant", rule_f="best", S=S, w_min=0.2, s=0.4, body=1e-4, r=r, opt=opt, wd=wd, small=np.zeros(S, bool))
    res = analyse(pool_world(240, 5), 7)
    eq = [res[(n, "equal")] for n in (30, 60, 120, 240)]; ov = [res[(n, "overlap")] for n in (30, 60, 120, 240)]
    check("T4", eq[-1] / eq[0] >= 1.5 and ov[-1] / ov[1] <= 1.25 and res["calib"] == 1.0,
          f"planted pool: equal capacity rises with the pool {eq}; overlap levels off {ov}; solver = closed form")
    check("T5", verdict(1.2, 1.0, 9).startswith("A not") and verdict(2.0, 1.6, 9).startswith("A confirmed, but")
          and verdict(2.0, 1.1, 4).startswith("A confirmed, overlap levels off but below") and verdict(2.0, 1.1, 9).startswith("A and B")
          and verdict(2.0, 1.1, 20).startswith("A confirmed, overlap levels off but above") and verdict(np.nan, 1, 1) == "not read",
          "every branch of the reading rule")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", action="append", default=[])
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.world:
        sys.exit("give --world PATTERN (one or more)")
    sys.exit(0 if run(a.world, a.cells) else 1)