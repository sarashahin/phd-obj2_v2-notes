#!/usr/bin/env python3
"""
frame_change_check.py  (obj1_v2, gate 7: frame interval and the speed of change, 1 Nov 2026, Sara's yes)

Question: how much does a simulated world change between window-end maps, against how much BNM changes between
years? Two answers come out of it:
  1. the frame interval: how many assembly rounds match one BNM year (Objective 2 trains on maps through time);
  2. whether the shift worlds change too fast: if a world changes more in one round than BNM in one year, window-end
     maps cannot be yearly frames and the habitat change is the first suspect (to fix before the real run).
Reads saved files only (training files and the BNM per-year grids of step 10, geographic fit half); writes nothing.

BNM side (imported from bnm_detection_check.py, unchanged, the J of check D): per fit-half window, J(h) = Jaccard of a
  species' recorded tetrads between two years h apart (h 1-5), on tetrads that pass the effort rule for that species
  in both years, summed over species, per range class (tetrads recorded in any of the six years: 1-5, 6-20, 21+, all).
  95% interval: 2,000 bootstrap resamples of windows.
Simulated side, the same J on window-end maps (ASM_P_*, 1 individual):
  survey   every species gets, in every BNM fit-half window (known_target of the --survey files, the windows
           world_vs_bnm uses), the known cells of a BNM species of that window drawn at random (stream [0, window]),
           fixed for the whole run, so a species is surveyed in the same cells in every map;
  detection each map is thinned on its own stream [11, window, round]: species with --small-cells or fewer present
           surveyed cells recorded with --detect-small, others with --detect (world_vs_bnm v5.8's rule);
  blocks   six maps k rounds apart (k = 1-20) stand for six years; blocks follow each other without overlap over the
           last half of the rounds; J per year pair from counts summed over windows, then averaged over pairs of
           the same lag and over blocks (BNM: over pairs of the same lag);
  class    cells recorded in any of the six maps (as BNM: any of the six years); species absent from some maps count
           there with no cells (as a BNM species not recorded that year).
  k*       per world and pooled: the k whose J_all(k), J_all(2k) ... J_all(5k) is closest to BNM's J_all(1..5) (sum of
           squared differences).

Reading rule (written 1 Nov, before the run):
  R0 checks first: the self-test passes; every world has consecutive recorded rounds and 5 x 20 rounds or more in its
     last half (else k is limited and said so).
  R1 too fast: pooled simulated J_all at k = 1 (one round) below the lower end of BNM's 95% interval for J_all at
     1 year -> a world changes more in one round than BNM in one year; window-end maps are too far apart for yearly
     frames; fix before the real run (habitat change first).
  R2 otherwise the frame interval is k* rounds per BNM year (pooled). At k*, a class whose J(1) or J(5) falls outside
     BNM's interval is named (its change is too fast or too slow); it goes to the cause list (2 fix rounds at most).
  R3 too slow: at k = 20 (the largest tried) pooled J_all at lag 5 (100 rounds) still above the upper end of BNM's
     interval for 5 years -> the worlds change slower than BNM at every k tried; reported with R2's per-class lines.
  Per world k* reported; a pooled k* that 3 of 4 worlds do not share (within a factor 2) is said.
  Limit on R2's class lines for 1-5 and 6-20 cells: BNM's p for small ranges (0.64) already holds one-year
     populations (bnm_detection_check T5), and the simulated maps hold their own flicker at 1 individual, so these
     classes can read too fast from a double count; a class named for this reason is first read again with
     --detect-small 0.80 before any fix. R1 and k* use the all class.

Run from obj1_v2/work:
    python3 ../tools/frame_change_check.py --self-test | tail -1
    python3 ../tools/frame_change_check.py --sim "results/stage2a/niche_h/*_fb_ht62500_hs1p0t62500_*_training.npz" "results/stage2a/niche_h2/*_training.npz" | tee ../logs/frame_change_h2_1nov.txt
"""
import argparse, glob, itertools, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bnm_detection_check as bdc          # noqa: E402  (fit half, window_counts, lag_curve: unchanged)

CLS = bdc.CLS
KS = list(range(1, 21))
H = 5
N_BOOT = 2000


# ------------------------------------------------ simulated worlds
class World:
    def __init__(self, f):
        z = np.load(f, allow_pickle=True)
        need = ["P_last_final", "ASM_P_round", "ASM_P_offsets", "ASM_P_ids", "ASM_P_bits"]
        miss = [k for k in need if k not in z.files]
        if miss:
            raise ValueError(f"missing {miss}")
        self.name = Path(f).name
        self.rounds = np.asarray(z["ASM_P_round"])
        if not np.all(np.diff(self.rounds) == 1):
            raise ValueError("recorded rounds are not consecutive")
        self.shape = tuple(np.asarray(z["P_last_final"]).shape[1:])
        self.off, self.ids, self.bits = np.asarray(z["ASM_P_offsets"]), np.asarray(z["ASM_P_ids"]), np.asarray(z["ASM_P_bits"])
        n = len(self.rounds)
        self.half = list(range(n // 2, n))
        self.all_ids = np.unique(self.ids[self.off[n // 2]:self.off[n]])
        self.pos = {int(s): j for j, s in enumerate(self.all_ids)}
        self.cache = {}

    def frame(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        n = self.shape[0] * self.shape[1]
        return self.ids[a:b], np.unpackbits(self.bits[a:b], axis=1, count=n).astype(bool)


def mask_index(world, masks):
    """Per survey window, the BNM species whose known cells each simulated species gets (fixed for the run)."""
    return [np.random.default_rng([0, w]).integers(len(m), size=len(world.all_ids)) for w, m in enumerate(masks)]


def records(world, masks, midx, w, i, p, p_small, small):
    key = (w, i)
    if key not in world.cache:
        ids, P = world.frame(i)
        g = np.array([world.pos[int(s)] for s in ids], dtype=int)
        K = masks[w][midx[w][g]]
        pres = P & K
        n = pres.sum(1)
        pp = np.where(n <= small, p_small, p)
        u = np.random.default_rng([11, w, int(world.rounds[i])]).random(pres.shape)
        world.cache[key] = (g, np.packbits(pres & (u < pp[:, None]), axis=1))
    g, rb = world.cache[key]
    return g, np.unpackbits(rb, axis=1, count=masks[w].shape[1]).astype(bool)


def block_counts(world, masks, midx, frames, p, p_small, small):
    """I, U (pairs, classes) summed over windows for six maps standing for six years."""
    pairs = list(itertools.combinations(range(len(frames)), 2))
    I = np.zeros((len(pairs), len(CLS))); U = np.zeros_like(I)
    A = len(world.all_ids)
    for w in range(len(masks)):
        R = np.zeros((len(frames), A, masks[w].shape[1]), bool)
        for t, i in enumerate(frames):
            g, rec = records(world, masks, midx, w, i, p, p_small, small)
            R[t, g] = rec
        size = R.any(0).sum(1)
        for k, (a, b) in enumerate(pairs):
            ii = (R[a] & R[b]).sum(1); uu = (R[a] | R[b]).sum(1)
            for c, (lo, hi, _) in enumerate(CLS):
                sel = (size >= lo) & (size <= hi)
                I[k, c] += ii[sel].sum(); U[k, c] += uu[sel].sum()
    return I, U, np.array([b - a for a, b in pairs])


def world_curves(world, masks, p, p_small, small, ks=KS):
    """J(h k) per class for each k, averaged over blocks: dict k -> (J (H, classes), blocks)."""
    midx = mask_index(world, masks)
    out = {}
    for k in ks:
        starts = list(range(world.half[0], world.half[-1] - H * k + 1, (H + 1) * k))
        if not starts:
            continue
        Js = []
        for s in starts:
            I, U, lags = block_counts(world, masks, midx, [s + t * k for t in range(H + 1)], p, p_small, small)
            Jh, _ = bdc.lag_curve(I, U, lags)
            Js.append(Jh)
        out[k] = (np.nanmean(np.array(Js), axis=0), len(starts))
    return out


def best_k(curves, J_bnm):
    d = {k: float(np.nansum((J[:, 3] - J_bnm[:, 3]) ** 2)) for k, (J, _) in curves.items()}
    return min(d, key=d.get), d


def survey_masks(pattern):
    ref = bdc.split_ref(pattern)
    files = sorted(glob.glob(pattern))
    keep, east, overlap = bdc.fit_half(files, ref)
    masks = []
    for f in keep:
        K = np.asarray(bdc.load(f)["known_target"]).astype(bool)
        masks.append(K.reshape(len(K), -1))
    return masks, ref, len(keep), east, overlap


def bnm_curve(peryear, minlist, ref, seed=0):
    per, lags, years = bdc.analyse(peryear, minlist, ref)
    I = sum(x[0] for x in per); U = sum(x[1] for x in per)
    J, Hs = bdc.lag_curve(I, U, lags)
    rng = np.random.default_rng(seed); boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(len(per), size=len(per))
        boot.append(bdc.lag_curve(sum(per[j][0] for j in idx), sum(per[j][1] for j in idx), lags)[0])
    boot = np.array(boot)
    return J, np.nanquantile(boot, 0.025, 0), np.nanquantile(boot, 0.975, 0), Hs, years


def read(worlds_curves, J_b, lo_b, hi_b):
    """Reading rule R1-R3 on the pooled curves; returns (lines, verdict dict)."""
    ks = sorted(set.intersection(*[set(c) for c in worlds_curves]))
    pooled = {k: (np.nanmean([c[k][0] for c in worlds_curves], axis=0), sum(c[k][1] for c in worlds_curves)) for k in ks}
    v = dict(ks=ks)
    if not ks:
        return ["no k common to every world"], dict(verdict="no k")
    J1 = pooled[1][0][0, 3] if 1 in pooled else np.nan
    v["J1_sim"], v["J1_bnm_lo"] = J1, lo_b[0, 3]
    kstar, dist = best_k(pooled, J_b)
    per_world = [best_k(c, J_b)[0] for c in worlds_curves]
    v.update(kstar=kstar, per_world=per_world, dist=dist, pooled=pooled)
    share = sum(1 for kw in per_world if kstar / 2 <= kw <= kstar * 2)
    lines = []
    kmax = max(ks); J5max = pooled[kmax][0][H - 1, 3]
    v["J5max_sim"], v["J5_bnm_hi"] = J5max, hi_b[H - 1, 3]
    if 1 in pooled and J1 < lo_b[0, 3]:
        v["verdict"] = "too fast"
        lines.append(f"R1 too fast: one round J_all {J1:.3f} is below BNM's one-year interval ({lo_b[0, 3]:.3f}-{hi_b[0, 3]:.3f}); "
                     "a world changes more in one round than BNM in one year -> fix before the real run (habitat change first)")
    elif J5max > hi_b[H - 1, 3]:
        v["verdict"] = "too slow"
        lines.append(f"R3 too slow: at {kmax} rounds per year (the largest tried) J_all over 5 years {J5max:.3f} is above BNM's "
                     f"5-year interval (upper end {hi_b[H - 1, 3]:.3f}); the worlds change slower than BNM at every k tried")
    else:
        v["verdict"] = "interval found"
        Jk = pooled[kstar][0]
        out = [lab for c, (_, _, lab) in enumerate(CLS[:3])
               if not (lo_b[0, c] <= Jk[0, c] <= hi_b[0, c]) or not (lo_b[H - 1, c] <= Jk[H - 1, c] <= hi_b[H - 1, c])]
        v["classes_out"] = out
        lines.append(f"R2 frame interval: {kstar} round(s) per BNM year (pooled); classes outside BNM's interval at J(1) or "
                     f"J(5): {', '.join(out) if out else 'none'}")
    lines.append(f"   per world k*: {per_world}; {share} of {len(per_world)} within a factor 2 of the pooled k*"
                 + ("" if share >= min(3, len(per_world)) else " -> worlds disagree"))
    return lines, v


# ------------------------------------------------ self-test
def markov(rng, Z, persist):
    """One round of slow change that keeps the expected range size (as bnm_detection_check's planted grids)."""
    n_occ = Z.sum(1, keepdims=True); cells = Z.shape[1]
    gain = (1 - persist) * n_occ / np.maximum(cells - n_occ, 1)
    u = rng.random(Z.shape)
    return np.where(Z, u >= 1 - persist, u < gain)


def planted_world(path, rng, persist, n_rounds=160, S=40, G=20, turnover=False):
    k = np.exp(rng.uniform(0, np.log(160), S)).astype(int)
    Z = np.zeros((S, G * G), bool)
    for i in range(S):
        Z[i, rng.choice(G * G, k[i], replace=False)] = True
    ids = np.arange(S); rounds, offs, idl, bits = [], [0], [], []
    for t in range(n_rounds):
        if t:
            Z = markov(rng, Z, persist)
            Z[~Z.any(1), rng.integers(G * G)] = True
            if turnover and t % 7 == 0:                       # one species leaves, a new one enters
                j = rng.integers(S); ids = ids.copy(); ids[j] = ids.max() + 1
                Z[j] = False; Z[j, rng.choice(G * G, 5, replace=False)] = True
        rounds.append(t); idl.append(ids.copy()); bits.append(np.packbits(Z, axis=1)); offs.append(offs[-1] + S)
    np.savez(path, P_last_final=Z.reshape(S, G, G).astype(np.uint8), ASM_P_round=np.array(rounds),
             ASM_P_offsets=np.array(offs), ASM_P_ids=np.concatenate(idl), ASM_P_bits=np.concatenate(bits))


def planted_bnm(tmp, rng, persist, k0, p, n_win=12, S=40, G=20, years=range(2014, 2020)):
    """BNM per-year grids from the same slow change, one year every k0 rounds, detection p, fit and test halves."""
    peryear = Path(tmp) / "py"; ref = Path(tmp) / "ref"
    (peryear / "minlist8").mkdir(parents=True); ref.mkdir()
    YRS = list(years)
    for y in YRS[:-1]:
        (peryear / f"minlist8_y{y}").mkdir()
    for w in range(n_win):
        e, n = 40 * (w % 6), 40 * (w // 6)
        name = f"bnm_e{e:04d}_n{n:04d}.npz"
        kk = np.exp(rng.uniform(0, np.log(160), S)).astype(int)
        Z = np.zeros((S, G * G), bool)
        for i in range(S):
            Z[i, rng.choice(G * G, kk[i], replace=False)] = True
        Ef = np.broadcast_to(rng.random(G * G) < 0.6, (S, G * G)).copy()
        for t, y in enumerate(YRS):
            if t:
                for _ in range(k0):
                    Z = markov(rng, Z, persist)
            O = (Z & Ef & (rng.random(Z.shape) < p)).reshape(S, G, G).astype(np.uint8)
            E = Ef.reshape(S, G, G).astype(np.uint8)
            if y == YRS[-1]:
                np.savez(peryear / "minlist8" / name, obs_mask=O, known_target=E, target_year=np.int64(y),
                         window_cells=np.int64(G), cell_m=np.float64(2000))
                np.savez(ref / name, obs_mask=O, known_target=E, window_cells=np.int64(G), cell_m=np.float64(2000))
            else:
                np.savez(peryear / f"minlist8_y{y}" / name, present_year=O, known_holdout=E)
    return str(peryear), str(ref / "bnm_*.npz")


def run_case(tmp, rng, persist_sim, persist_bnm, k0, p=0.8, ks=(1, 2, 3, 4, 5, 6), n_worlds=2, turnover=False):
    peryear, refpat = planted_bnm(tmp, rng, persist_bnm, k0, p)
    bdc.OPENED.clear()
    masks, ref, n_fit, east, overlap = survey_masks(refpat)
    J_b, lo_b, hi_b, Hs, years = bnm_curve(peryear, 8, ref)
    curves = []
    for j in range(n_worlds):
        planted_world(f"{tmp}/w{j}.npz", rng, persist_sim, turnover=turnover)
        curves.append(world_curves(World(f"{tmp}/w{j}.npz"), masks, p, p, 0, ks=ks))
    lines, v = read(curves, J_b, lo_b, hi_b)
    n_test = len([o for o in bdc.OPENED if bdc.window_en(o.split("@")[0])[0] >= ref["cut"]])
    return v, J_b, lo_b, hi_b, n_test, curves


def self_test():
    rng = np.random.default_rng(7); res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        v, J_b, lo, hi, n_test, curves = run_case(tmp, rng, 0.97, 0.97 ** 1, 3, ks=(1, 2, 3, 4, 5, 6))
        pk = v.get("kstar")
        check("T1", v["verdict"] == "interval found" and pk == 3 and n_test == 0,
              f"same change per round, one BNM year = 3 rounds: k* {pk} (per world {v.get('per_world')}); "
              f"test-half files opened {n_test}")
    with tempfile.TemporaryDirectory() as tmp:
        v, *_ = run_case(tmp, rng, 0.80, 0.97, 1, ks=(1, 2, 3))
        check("T2", v["verdict"] == "too fast",
              f"world changes much faster per round (persist 0.80) than BNM per year (0.97): verdict '{v['verdict']}', "
              f"J_all one round {v['J1_sim']:.3f} against BNM lower end {v['J1_bnm_lo']:.3f}")
    with tempfile.TemporaryDirectory() as tmp:
        v, *_ = run_case(tmp, rng, 1.0, 0.97, 1, ks=(1, 2, 3, 4))
        check("T3", v["verdict"] == "too slow",
              f"a world with no change against BNM's slow change: verdict '{v['verdict']}' (J_all at lag 5 x 4 rounds "
              f"{v['J5max_sim']:.3f} above BNM's upper end {v['J5_bnm_hi']:.3f})")
    # T4 lag 0: two detection draws of the same map give J = p / (2 - p) for species recorded widely
    with tempfile.TemporaryDirectory() as tmp:
        planted_world(f"{tmp}/w.npz", rng, 1.0, n_rounds=40)
        Wd = World(f"{tmp}/w.npz")
        masks = [np.ones((5, 400), bool)] * 6
        midx = mask_index(Wd, masks)
        I, U, lags = block_counts(Wd, masks, midx, [20, 21, 22, 23, 24, 25], 0.8, 0.8, 0)
        J = I[:, 2].sum() / U[:, 2].sum()
        check("T4", abs(J - 0.8 / 1.2) < 0.02, f"no change, p 0.8, 21+ class: J {J:.3f} = p / (2 - p) {0.8 / 1.2:.3f}")
        # T5 direct loop: one window, every cell known, p 1 -> J per pair against a plain count over species
        m1 = [np.ones((3, 400), bool)]
        planted_world(f"{tmp}/t.npz", rng, 0.9, n_rounds=40, turnover=True)
        Wt = World(f"{tmp}/t.npz"); mi = mask_index(Wt, m1)
        fr = [20, 22, 24, 26, 28, 30]
        I, U, lags = block_counts(Wt, m1, mi, fr, 1.0, 1.0, 0)
        maps = [dict(zip(Wt.frame(i)[0].tolist(), Wt.frame(i)[1])) for i in fr]
        allid = set().union(*[set(m) for m in maps])
        good = True
        for k, (a, b) in enumerate(itertools.combinations(range(6), 2)):
            ii = uu = 0
            for s in allid:
                A = maps[a].get(s, np.zeros(400, bool)); B = maps[b].get(s, np.zeros(400, bool))
                ii += (A & B).sum(); uu += (A | B).sum()
            good &= (I[k, 3] == ii and U[k, 3] == uu)
        check("T5", good and len(allid) > 40, f"counts against a plain loop over species, with species leaving and "
              f"entering ({len(allid)} ids over 6 maps; a species missing from a map counts with no cells)")
        # T6 mutant: masks drawn again for every map must lower J (the survey would move between years)
        Wd.cache.clear()
        I0, U0, _ = block_counts(Wd, [np.random.default_rng(3).random((30, 400)) < 0.5], [np.arange(len(Wd.all_ids)) % 30],
                                 [20, 21, 22, 23, 24, 25], 1.0, 1.0, 0)
        Wd.cache.clear()
        Mm = [np.random.default_rng(3).random((30, 400)) < 0.5]
        Ib = np.zeros_like(I0); Ub = np.zeros_like(U0)
        for t_, (a, b) in enumerate(itertools.combinations(range(6), 2)):
            ga, ra = Wd.frame(20 + a); gb, rb = Wd.frame(20 + b)
            Ka = Mm[0][np.random.default_rng([5, a]).integers(30, size=len(ga))]
            Kb = Mm[0][np.random.default_rng([5, b]).integers(30, size=len(gb))]
            Ib[t_, 3] = ((ra & Ka) & (rb & Kb)).sum(); Ub[t_, 3] = ((ra & Ka) | (rb & Kb)).sum()
        check("T6", I0[:, 3].sum() / U0[:, 3].sum() > 0.99 and Ib[:, 3].sum() / Ub[:, 3].sum() < 0.6,
              f"fixed survey, no change, p 1: J {I0[:, 3].sum() / U0[:, 3].sum():.3f}; survey drawn again per map "
              f"(the mutant): {Ib[:, 3].sum() / Ub[:, 3].sum():.3f}")
        # T7 the same world read twice gives the same curves (survey and detection come from fixed streams)
        mk = [np.random.default_rng(4).random((30, 400)) < 0.6] * 3
        c1 = world_curves(World(f"{tmp}/t.npz"), mk, 0.8, 0.64, 20, ks=(1, 2))
        c2 = world_curves(World(f"{tmp}/t.npz"), mk, 0.8, 0.64, 20, ks=(1, 2))
        check("T7", all(np.array_equal(c1[k][0], c2[k][0], equal_nan=True) for k in (1, 2)),
              "the same world read twice gives the same J (fixed survey and detection streams)")
    print(f"{sum(res)} of {len(res)} PASS" if all(res) else f"FAIL: {len(res) - sum(res)} of {len(res)}")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", nargs="+", default=None, help="training-file patterns, all one setting")
    ap.add_argument("--survey", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--minlist", type=int, default=8)
    ap.add_argument("--detect", type=float, default=0.80)
    ap.add_argument("--detect-small", type=float, default=0.64)
    ap.add_argument("--small-cells", type=int, default=20)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.sim:
        sys.exit("STOP: give --sim patterns")
    files = sorted({f for p in a.sim for f in glob.glob(p)})
    masks, ref, n_fit, east, overlap = survey_masks(a.survey)
    print(f"survey: {n_fit} fit-half BNM windows (test half left unopened: {east} east of the cut, {overlap} sharing cells)")
    J_b, lo_b, hi_b, Hs, years = bnm_curve(a.peryear, a.minlist, ref)
    print(f"BNM years {years}: J(h) by class, 95% interval over windows")
    for c, (_, _, lab) in enumerate(CLS):
        print(f"   {lab:>5s}  " + "  ".join(f"h{h} {J_b[i, c]:.3f} ({lo_b[i, c]:.3f}-{hi_b[i, c]:.3f})" for i, h in enumerate(Hs)))
    curves = []
    for f in files:
        try:
            W = World(f)
        except ValueError as e:
            print(f"skipped {Path(f).name}: {e}"); continue
        c = world_curves(W, masks, a.detect, a.detect_small, a.small_cells)
        curves.append(c)
        kw, _ = best_k(c, J_b)
        print(f"\n{W.name[-70:]}: {len(W.rounds)} rounds, last half {len(W.half)}; k tried {min(c)}-{max(c)}; k* {kw}")
        for k in (1, 2, 5, 10, 20):
            if k in c:
                J, nb = c[k]
                print(f"   k {k:2d} ({nb} blocks)  J_all lags 1-5: " + " ".join(f"{x:.3f}" for x in J[:, 3])
                      + "   J(1) by class 1-5 / 6-20 / 21+: " + " / ".join(f"{x:.3f}" for x in J[0, :3]))
    if not curves:
        sys.exit("STOP: no world read")
    lines, v = read(curves, J_b, lo_b, hi_b)
    print(f"\npooled over {len(curves)} worlds (k common to all: {v.get('ks', [])[:1]}-{v.get('ks', [])[-1:]})")
    if v.get("kstar"):
        J = v["pooled"][v["kstar"]][0]
        print(f"   k* {v['kstar']}: J by class at lags 1 and 5: " + "; ".join(
            f"{lab} {J[0, c]:.3f} / {J[H - 1, c]:.3f} (BNM {J_b[0, c]:.3f} / {J_b[H - 1, c]:.3f})" for c, (_, _, lab) in enumerate(CLS)))
    print("reading rule (written before the run, 1 Nov):")
    for s in lines:
        print("  " + s)