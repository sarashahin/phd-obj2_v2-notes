#!/usr/bin/env python3
"""
bnm_detection_check.py  (obj1_v2, root cause 22: detection in the BNM yardstick, 30 Oct 2026)

Question: in a tetrad that passes the effort rule (list length 8, focal species left out), how often is a present
species recorded in a given year? world_vs_bnm v5.7 can now thin simulated records with --detect p; this measures p
from Sara's BNM records, so p is a number from the data and not a guess.

Why: the re-score of 30 Oct changed verdicts with p (random founding 4, 2, 1 of 8 bands at p 1, 0.8, 0.6; best
founding lost the 1-5 cell share at 0.6), so by the rule written before that run, p must be set from BNM data before
the next simulator change.

Data: the per-year grids of step 10 (--per-year), the same files check D (9 Oct) read: <peryear>/minlist8 (target
year: obs_mask, known_target) and <peryear>/minlist8_y<year> (present_year, known_holdout). Geographic fit half only:
the split rule of world_vs_bnm (fit = easting below the median of the distinct eastings of the reference windows,
and no cell shared with a test window); test-half files are never opened.

Method (records only, no model):
  J(h)  Jaccard of a species' recorded tetrads between two years h apart, on tetrads that pass the effort rule for
        that species in both years, summed over species and windows (check D's definition), h = 1..5.
  If a present species is recorded with chance p each year, independently, and the true range changes slowly, then
        q(h) = J(h) / (1 + J(h)) = p x a(h), with a(h) = Jt(h) / (1 + Jt(h)) and Jt the true Jaccard; a(0) = 1/2.
  So q is extrapolated back to h = 0 by a straight line through h = 1..5 and p = 2 q(0). An exponential fit is
  printed beside it as a check on the shape. 95% interval: 2,000 bootstrap resamples of fit-half windows.
  Classes as check D: tetrads occupied in any of the six years (1-5, 6-20, 21+, all).
  What p holds: misses and populations that are present one year and gone the next both lower it, because records
  cannot tell them apart; p is the share of a species' lasting cells recorded in one year.

Calibration (self-test, planted per-year grids, true ranges 1-160 cells of 400, fit half only):
  T1  no change, p 1: J(h) = 1 at every lag and p = 1; no test-half file opened
  T2  p 1, 0.8, 0.6 with slow change (persist 0.97 and 0.93; q lag 5 / lag 1 down to 0.75, as steep as BNM's 1-5
      class in check D): all classes within 0.05 of the planted p, 1-5 cell class within 0.10
  T5  20% one-year populations at p 0.8 read 0.64: such populations count as misses
  T6  detection 0.4 or 1.0 in the same cells every year (mean 0.7) reads 0.83: cells hard to survey every year read high
  T7  fast change (persist 0.85, q ratio 0.56) reads 0.56 for a planted 0.6: the line reads low, and the rule says so
  T3  a presence outside its year's effort stops the run;  T4  every branch of the reading rule

Reading rule (written before the run, 30 Oct): p_all with its interval.
  interval includes 1     -> keep --detect 1; detection is not a cause
  otherwise               -> re-score all saved worlds at --detect p_all (one run, command printed) and use those
                             verdicts for the next simulator step
  classes differ by more than 0.15 -> report; one p for all is then an approximation, to be said with the results
  q lag 5 / lag 1 below 0.7       -> report; the line then reads low (T7)

Run from obj1_v2/work:
    python3 ../tools/bnm_detection_check.py --self-test | tee ../logs/bnm_detection_selftest_30oct.txt
    python3 ../tools/bnm_detection_check.py | tee ../logs/bnm_detection_30oct.txt
Defaults: --peryear ../../AI_simulation/Empirical_data/bnm_grids_peryear --minlist 8
          --split-ref "../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz"
Reads files only; writes nothing.

Result 30 Oct 2026 (Sara's run; 44 fit-half windows read, 57 test-half unopened, cut at 400 km; self-test 10 PASS):
  p 1-5 cells 0.643 (0.577-0.705), 6-20 0.642 (0.603-0.678), 21+ 0.800 (0.788-0.810), all 0.795 (0.782-0.805);
  line and exponential agree (all 0.795 both); q lag 5 / lag 1: 0.77, 0.87, 0.96, 0.96 (slow change, inside T2's
  range). Rule: p below 1 -> re-score at --detect 0.79; classes differ by 0.16 (above 0.15), so one p is an
  approximation for the range-size bands.
"""
import argparse, glob, itertools, sys, tempfile, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

CLS = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 400, "21+"), (1, 400, "all")]
N_BOOT = 2000
OPENED = []


def load(f):
    OPENED.append(Path(f).name + "@" + Path(f).parent.name)
    return np.load(f, allow_pickle=True)


# ------------------------------------------------ fit half, as world_vs_bnm.py v3 and later
def window_en(f):
    s = Path(f).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])


def split_ref(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"STOP: no reference BNM files matched {pattern}; the fit half cannot be set")
    eas = sorted({window_en(f)[0] for f in files}); cut = eas[len(eas) // 2]
    fit = [f for f in files if window_en(f)[0] < cut]
    if not fit:
        sys.exit("STOP: the reference set has no fit-half window")
    z = load(fit[0])
    side = (int(z["window_cells"]) * float(z["cell_m"]) / 1000) if {"window_cells", "cell_m"} <= set(z.files) \
        else z["obs_mask"].shape[-1] * 2.0
    return dict(cut=cut, side=side, n=len(files), test=[window_en(f) for f in files if window_en(f)[0] >= cut])


def fit_half(names, ref):
    keep, east, overlap = [], 0, 0
    for f in sorted(names):
        e, n = window_en(f)
        if e >= ref["cut"]:
            east += 1
        elif any(abs(e - te) < ref["side"] and abs(n - tn) < ref["side"] for te, tn in ref["test"]):
            overlap += 1
        else:
            keep.append(f)
    return keep, east, overlap


# ------------------------------------------------ Jaccard by lag
def window_counts(pooled, ydirs, name):
    """Per window: intersections and unions per year pair and class, as check D."""
    a = load(pooled / name)
    zs = [load(d / name) for _, d in ydirs]
    Pr = np.stack([z["present_year"] for z in zs] + [a["obs_mask"]]).astype(bool)
    Ef = np.stack([z["known_holdout"] for z in zs] + [a["known_target"]]).astype(bool)
    Y, S = Pr.shape[0], Pr.shape[1]
    Pr = Pr.reshape(Y, S, -1); Ef = Ef.reshape(Y, S, -1)
    if (Pr & ~Ef).any():
        raise ValueError(f"{name}: a presence outside its year's effort")
    size = Pr.any(0).sum(1)
    pairs = list(itertools.combinations(range(Y), 2))
    I = np.zeros((len(pairs), len(CLS))); U = np.zeros_like(I)
    for k, (i, j) in enumerate(pairs):
        both = Ef[i] & Ef[j]
        ii = (Pr[i] & Pr[j] & both).sum(1); uu = ((Pr[i] | Pr[j]) & both).sum(1)
        for c, (lo, hi, _) in enumerate(CLS):
            sel = (size >= lo) & (size <= hi)
            I[k, c] = ii[sel].sum(); U[k, c] = uu[sel].sum()
    return I, U, pairs


def lag_curve(I, U, lags):
    """J by lag per class from summed counts. I, U (pairs, classes)."""
    J = np.where(U > 0, I / np.maximum(U, 1e-12), np.nan)
    H = sorted(set(lags))
    return np.array([[np.nanmean(J[lags == h, c]) for c in range(J.shape[1])] for h in H]), H


def p_from_curve(Jh, H):
    """p from q(h) = J / (1 + J) extrapolated to h 0: straight line (main) and exponential (shape check)."""
    q = Jh / (1 + Jh)
    h = np.asarray(H, float)
    out = []
    for c in range(q.shape[1]):
        y = q[:, c]
        ok = np.isfinite(y) & (y > 0)
        if ok.sum() < 2:
            out.append((np.nan, np.nan)); continue
        b1, b0 = np.polyfit(h[ok], y[ok], 1)
        e1, e0 = np.polyfit(h[ok], np.log(y[ok]), 1)
        out.append((min(2 * b0, 1.0), min(2 * np.exp(e0), 1.0)))
    return np.array(out)


def analyse(peryear, minlist, ref):
    pooled = Path(peryear) / f"minlist{minlist}"
    ydirs = sorted((int(p.name.rsplit("_y", 1)[1]), p) for p in Path(peryear).glob(f"minlist{minlist}_y*") if p.is_dir())
    names = sorted(f.name for f in pooled.glob("bnm_*.npz"))
    if not names or not ydirs:
        sys.exit(f"STOP: no per-year grids under {peryear} (step 10 with --per-year)")
    keep, east, overlap = fit_half(names, ref)
    print(f"per-year BNM grids {pooled}: {len(names)} windows; {len(keep)} fit-half windows read; {east} test-half windows and "
          f"{overlap} sharing cells with a test window left unopened (cut at easting {ref['cut']} km, set on {ref['n']} reference windows)")
    TY = int(load(pooled / keep[0])["target_year"])
    years = [y for y, _ in ydirs] + [TY]
    if years != list(range(years[0], years[-1] + 1)):
        sys.exit(f"STOP: years are not consecutive: {years}")
    per = []
    for nm in keep:
        I, U, pairs = window_counts(pooled, ydirs, nm)
        per.append((I, U))
    lags = np.array([abs(years[i] - years[j]) for i, j in pairs])
    return per, lags, years


def report(per, lags, years, seed=0):
    I = sum(x[0] for x in per); U = sum(x[1] for x in per)
    Jh, H = lag_curve(I, U, lags)
    P = p_from_curve(Jh, H)
    rng = np.random.default_rng(seed); boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(len(per), size=len(per))
        Ib = sum(per[k][0] for k in idx); Ub = sum(per[k][1] for k in idx)
        boot.append(p_from_curve(*lag_curve(Ib, Ub, lags))[:, 0])
    boot = np.array(boot)
    lo, hi = np.nanquantile(boot, 0.025, 0), np.nanquantile(boot, 0.975, 0)
    print(f"\nyears {years}; {len(per)} windows; J(h): Jaccard of recorded tetrads between years h apart, effort passed in both years")
    q = Jh / (1 + Jh); R = q[-1] / q[0]
    print("   class   " + "  ".join(f"J lag {h}" for h in H) + f"   q lag {H[-1]} / lag {H[0]}   p (line, 95% interval)   p (exponential)")
    for c, (_, _, lab) in enumerate(CLS):
        print(f"   {lab:>5s}   " + "  ".join(f"{Jh[i, c]:7.3f}" for i in range(len(H)))
              + f"   {R[c]:15.3f}   {P[c, 0]:.3f} ({lo[c]:.3f}-{hi[c]:.3f})      {P[c, 1]:.3f}")
    return P, lo, hi, R


def verdict(P, lo, hi, R=None):
    p_all, l_all, h_all = P[3, 0], lo[3], hi[3]
    cls = P[:3, 0]
    spread = float(np.nanmax(cls) - np.nanmin(cls)) if np.isfinite(cls).any() else np.nan
    if h_all >= 1.0 - 1e-9:
        txt = "interval includes 1 -> keep --detect 1; detection is not a cause"
    else:
        txt = f"p below 1 -> re-score all saved worlds at --detect {p_all:.2f} and use those verdicts for the next simulator step"
    if np.isfinite(spread) and spread > 0.15:
        txt += f"; classes differ by {spread:.2f} (more than 0.15): one p for all is an approximation, to be said with the results"
    if R is not None and np.isfinite(R[3]) and R[3] < 0.7:
        txt += (f"; fast change (q lag 5 / lag 1 = {R[3]:.2f}, below 0.7): the straight line reads low (self-test T7), "
                "so the true p may be higher")
    return txt, p_all


# ------------------------------------------------ self-test
def self_test():
    ok = True

    def check(name, cond, text=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name} {text} {'PASS' if cond else 'FAIL'}")

    G, S, YRS = 20, 40, list(range(2014, 2020))

    def make(tmp, p, persist, seed, n_win=12, flicker=0.0, p_two=None, bad=False):
        """Planted per-year grids. True cells per species log-uniform 1-160 of 400; slow change keeps the expected
        range size (loss 1 - persist, gains to match); flicker: extra one-year cells, this share of a year's cells;
        p_two: detection fixed per species and cell, half at each value (the same cells hard to survey every year)."""
        rng = np.random.default_rng(seed)
        peryear = Path(tmp) / "py"; ref = Path(tmp) / "ref"
        (peryear / "minlist8").mkdir(parents=True); ref.mkdir()
        for y in YRS[:-1]:
            (peryear / f"minlist8_y{y}").mkdir()
        for w in range(n_win):
            e, n = 40 * (w % 6), 40 * (w // 6)
            name = f"bnm_e{e:04d}_n{n:04d}.npz"
            k = np.exp(rng.uniform(0, np.log(160), S)).astype(int)
            Z = np.zeros((S, G * G), bool)
            for i in range(S):
                Z[i, rng.choice(G * G, k[i], replace=False)] = True
            Pc = np.full((S, G * G), p) if p_two is None else np.where(rng.random((S, G * G)) < 0.5, *p_two)
            for t, y in enumerate(YRS):
                if t:                                                      # slow true change, range size kept
                    n_occ = Z.sum(1, keepdims=True)
                    gain = (1 - persist) * n_occ / np.maximum(G * G - n_occ, 1)
                    u = rng.random(Z.shape)
                    Z = np.where(Z, u >= 1 - persist, u < gain)
                A = Z.copy()
                if flicker:                                                # one-year populations
                    n_occ = Z.sum(1, keepdims=True)
                    A |= ~Z & (rng.random(Z.shape) < flicker / (1 - flicker) * n_occ / np.maximum(G * G - n_occ, 1))
                Ef = np.broadcast_to(rng.random(G * G) < 0.6, (S, G * G)).copy()
                O = A & Ef & (rng.random(Z.shape) < Pc)
                if bad and t == 0 and w == 0:
                    O[0, 0] = True; Ef[0, 0] = False
                O = O.reshape(S, G, G).astype(np.uint8); Ef = Ef.reshape(S, G, G).astype(np.uint8)
                if y == YRS[-1]:
                    np.savez(peryear / "minlist8" / name, obs_mask=O, known_target=Ef, target_year=np.int64(y),
                             window_cells=np.int64(G), cell_m=np.float64(2000))
                    np.savez(ref / name, obs_mask=O, known_target=Ef, window_cells=np.int64(G), cell_m=np.float64(2000))
                else:
                    np.savez(peryear / f"minlist8_y{y}" / name, present_year=O, known_holdout=Ef)
        return str(peryear), str(ref / "bnm_*.npz")

    def run(**kw):
        global OPENED
        with tempfile.TemporaryDirectory() as tmp:
            peryear, refpat = make(tmp, **kw)
            OPENED = []
            ref = split_ref(refpat)
            per, lags, years = analyse(peryear, 8, ref)
            P, lo, hi, R = report(per, lags, years)
            Jh, _ = lag_curve(sum(x[0] for x in per), sum(x[1] for x in per), lags)
            n_test = len([f for f in OPENED if window_en(f.split("@")[0])[0] >= ref["cut"]])
        return P, lo, hi, Jh, n_test, R

    P, lo, hi, Jh, n_test, R = run(p=1.0, persist=1.0, seed=1)
    check("T1", np.allclose(Jh[np.isfinite(Jh)], 1.0) and abs(P[3, 0] - 1) < 1e-9 and n_test == 0,
          f"no change and p 1: J(h) = 1 at every lag, p = 1; test-half files opened: {n_test}")
    for p, persist, seed in ((1.0, 0.97, 2), (0.8, 0.97, 3), (0.6, 0.97, 4), (0.6, 0.93, 5)):
        P, lo, hi, Jh, n_test, R = run(p=p, persist=persist, seed=seed)
        good = (abs(P[3, 0] - p) <= 0.05 and lo[3] - 0.02 <= p <= hi[3] + 0.02 and abs(P[0, 0] - p) <= 0.10
                and n_test == 0)
        check(f"T2 p {p} persist {persist}", good, f"planted p {p}: all {P[3, 0]:.3f} ({lo[3]:.3f}-{hi[3]:.3f}), "
              f"1-5 cells {P[0, 0]:.3f}; test-half files opened: {n_test}")
    P, lo, hi, Jh, n_test, R = run(p=0.8, persist=0.97, seed=6, flicker=0.2)
    check("T5", abs(P[3, 0] - 0.64) <= 0.05, f"what p holds: p 0.8 with 20% one-year populations reads "
          f"{P[3, 0]:.3f}, expected 0.8 x 0.8 = 0.64 (one-year populations count as misses)")
    P, lo, hi, Jh, n_test, R = run(p=0.7, persist=0.97, seed=7, p_two=(0.4, 1.0))
    check("T6", abs(P[3, 0] - 0.829) <= 0.05, f"what p holds: detection 0.4 or 1.0 in fixed cells (mean 0.7) reads "
          f"{P[3, 0]:.3f}, expected (0.16 + 1) / 2 / 0.7 = 0.829 (cells hard to survey every year read high)")
    P, lo, hi, Jh, n_test, R = run(p=0.6, persist=0.85, seed=11)
    txt, _ = verdict(P, lo, hi, R)
    check("T7", 0.5 < P[3, 0] < 0.6 and R[3] < 0.7 and "fast change" in txt,
          f"fast true change (persist 0.85, q ratio {R[3]:.2f}): planted p 0.6 reads {P[3, 0]:.3f} by the line "
          f"({P[3, 1]:.3f} exponential) and the reading rule says it reads low")
    with tempfile.TemporaryDirectory() as tmp:
        peryear, refpat = make(tmp, 0.8, 0.97, seed=8, bad=True)
        ref = split_ref(refpat)
        try:
            analyse(peryear, 8, ref); stopped = False
        except ValueError:
            stopped = True
        check("T3", stopped, "a presence outside its year's effort stops the run")
    t1, _ = verdict(np.array([[0.9, 0.9]] * 4), np.full(4, 0.8), np.full(4, 1.0))
    t2, p2 = verdict(np.array([[0.6, 0.6], [0.65, 0.6], [0.8, 0.8], [0.7, 0.7]]), np.full(4, 0.6), np.full(4, 0.8))
    check("T4", t1.startswith("interval includes 1") and t2.startswith("p below 1") and "differ by 0.20" in t2 and abs(p2 - 0.7) < 1e-9,
          "every branch of the reading rule")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--minlist", type=int, default=8)
    ap.add_argument("--split-ref", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    ref = split_ref(a.split_ref)
    per, lags, years = analyse(a.peryear, a.minlist, ref)
    P, lo, hi, R = report(per, lags, years)
    txt, p_all = verdict(P, lo, hi, R)
    print(f"\nReading rule (written before the run): p (all classes) {P[3, 0]:.3f} ({lo[3]:.3f}-{hi[3]:.3f}) -> {txt}")
    if hi[3] < 1.0 - 1e-9:
        print("\nNext command (from obj1_v2/work):")
        print(f'  BNM="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz"; python3 ../tools/world_vs_bnm.py --tau 1 '
              f'--detect {p_all:.2f} --bnm "$BNM" --survey "$BNM" --each --sim "results/stage2a/niche_d/*_training.npz" '
              f'--sim "results/stage2a/niche_b/*_training.npz" --sim "results/stage2a/niche_w/*_training.npz" '
              f'> ../logs/world_vs_bnm_detect_{p_all:.2f}_30oct.txt')