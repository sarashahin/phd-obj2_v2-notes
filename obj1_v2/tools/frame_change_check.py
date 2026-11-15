#!/usr/bin/env python3
"""
frame_change_check.py  version 2  (obj1_v2, gate 7 and root cause 24: speed of change against BNM, 1 Nov 2026, Sara's yes)

Question: how fast do simulated maps change from frame to frame, against how fast BNM changes from year to year?
It gives the frame interval (how many frames match one BNM year; Objective 2 trains on maps through time) and says
whether the worlds change too fast or too slow for yearly frames.
Reads saved files only (training files and the BNM per-year grids of step 10, geographic fit half); writes nothing.

Why version 2 (1 Nov): version 1 compared J at lag 1. On the habitat chain that mixed two kinds of change: change that
builds up with time (ranges moving) and change that is lost by lag 1 and does not grow with lag (presences that last
less than one frame, and missed records). Records cannot tell BNM's one-year populations from misses
(bnm_detection_check T5), so only the part that builds up can be compared. Version 2 judges speed by
r(h) = q(h) / q(1), q = J / (1 + J), h = 2..5: under the model q(h) = p_eff x a(h) the level p_eff cancels.
Version 1 is kept as frame_change_check_v1.py.

Frames: the maps after every habitat sub-step when the file has them (ASM_F_*, assembly v2.6 --habitat-substeps),
else the window-end maps (ASM_P_*). Both at 1 individual.
BNM side (bnm_detection_check.py, unchanged, the J of check D): per fit-half window, J(h) = Jaccard of a species'
  recorded tetrads between two years h apart (h 1-5), on tetrads that pass the effort rule for that species in both
  years, summed over species, per range class (tetrads recorded in any of the six years: 1-5, 6-20, 21+, all).
  95% intervals of J and r: 2,000 bootstrap resamples of windows.
Simulated side, the same J:
  survey   every species gets, in every BNM fit-half window (known_target of the --survey files, the windows
           world_vs_bnm uses), the known cells of a BNM species of that window drawn at random (stream [0, window]),
           fixed for the whole run;
  detection each frame thinned on its own stream ([11, window, round] for window-end maps, [11, window, round, piece]
           for sub-step maps): species with --small-cells or fewer present surveyed cells with --detect-small, others
           with --detect (world_vs_bnm v5.8's rule);
  blocks   six frames k apart (k = 1-20 frames) stand for six years; up to --max-blocks blocks per k, evenly spaced
           without overlap over the last half of the frames; J per pair from counts summed over windows, then
           averaged over pairs of the same lag and over blocks;
  class    cells recorded in any of the six frames (as BNM: any of the six years).
  k*       per world and pooled: the k whose r_all(2..5) is closest to BNM's (sum of squared differences).

Reading rule (written 1 Nov, before any v2.6 world exists):
  R0 the self-test passes; frames are consecutive.
  R1 too fast: pooled r_all(5) at k = 1 (one frame apart) below the lower end of BNM's interval for r_all(5) -> the
     worlds change more between neighbouring frames than BNM between years; the frames are too far apart or the
     habitat moves too fast; fix before the real run.
  R3 too slow: pooled r_all(5) at k = 20 above the upper end of BNM's interval -> the worlds change slower than BNM
     at every k tried.
  R2 otherwise the frame interval is k* frames per BNM year (pooled); at k* a class (1-5, 6-20, 21+) whose r(5) is
     outside BNM's interval is named; it goes to the cause list (2 fix rounds at most).
  Per world k* reported; a pooled k* that 3 of 4 worlds do not share within a factor 2 is said.
  Description only: the level p_eff (2 q extrapolated to lag 0 by a straight line), simulated against BNM.

Run from obj1_v2/work:
    python3 ../tools/frame_change_check.py --self-test | tail -1
    python3 ../tools/frame_change_check.py --sim "<training files of one setting>" | tee ../logs/frame_change_<setting>.txt
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


def ratio(J):
    """r(h) = q(h) / q(1), q = J / (1 + J); J (H, classes) -> r (H, classes), r(1) = 1."""
    q = J / (1 + J)
    return q / q[0:1]


def p_eff(J):
    """2 q(0) from a straight line through q(1..H), per class."""
    q = J / (1 + J); h = np.arange(1, len(J) + 1, dtype=float); out = []
    for c in range(J.shape[1]):
        ok = np.isfinite(q[:, c])
        out.append(2 * np.polyfit(h[ok], q[ok, c], 1)[1] if ok.sum() >= 2 else np.nan)
    return np.array(out)


# ------------------------------------------------ simulated worlds
class World:
    def __init__(self, f):
        z = np.load(f, allow_pickle=True)
        self.name = Path(f).name
        self.shape = tuple(np.asarray(z["P_last_final"]).shape[1:])
        if "ASM_F_offsets" in z.files:
            self.kind = "sub-step maps"
            rr, pp = np.asarray(z["ASM_F_round"]), np.asarray(z["ASM_F_piece"])
            m = int(pp.max()) + 1
            if not (np.array_equal(pp, np.tile(np.arange(m), len(pp) // m)) and len(pp) % m == 0
                    and np.all(np.diff(rr[::m]) == 1) and np.all(rr == np.repeat(rr[::m], m))):
                raise ValueError("sub-step maps are not complete rounds in order")
            self.off, self.ids, self.bits = np.asarray(z["ASM_F_offsets"]), np.asarray(z["ASM_F_ids"]), np.asarray(z["ASM_F_bits"])
            self.key = [(int(r_), int(p_)) for r_, p_ in zip(rr, pp)]
            self.per_round = m
        else:
            need = ["ASM_P_round", "ASM_P_offsets", "ASM_P_ids", "ASM_P_bits"]
            miss = [k for k in need if k not in z.files]
            if miss:
                raise ValueError(f"missing {miss}")
            self.kind = "window-end maps"
            rr = np.asarray(z["ASM_P_round"])
            if not np.all(np.diff(rr) == 1):
                raise ValueError("recorded rounds are not consecutive")
            self.off, self.ids, self.bits = np.asarray(z["ASM_P_offsets"]), np.asarray(z["ASM_P_ids"]), np.asarray(z["ASM_P_bits"])
            self.key = [(int(r_),) for r_ in rr]
            self.per_round = 1
        n = len(self.key)
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
        u = np.random.default_rng([11, w, *world.key[i]]).random(pres.shape)
        world.cache[key] = (g, np.packbits(pres & (u < pp[:, None]), axis=1))
    g, rb = world.cache[key]
    return g, np.unpackbits(rb, axis=1, count=masks[w].shape[1]).astype(bool)


def block_counts(world, masks, midx, frames, p, p_small, small):
    """I, U (pairs, classes) summed over windows for six frames standing for six years."""
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


def block_starts(world, k, max_blocks):
    starts = list(range(world.half[0], world.half[-1] - H * k + 1, (H + 1) * k))
    if len(starts) > max_blocks:
        starts = [starts[i] for i in np.unique(np.linspace(0, len(starts) - 1, max_blocks).round().astype(int))]
    return starts


def world_curves(world, masks, p, p_small, small, ks=KS, max_blocks=40):
    """J(h k) per class for each k, averaged over blocks: dict k -> (J (H, classes), blocks)."""
    midx = mask_index(world, masks)
    out = {}
    for k in ks:
        starts = block_starts(world, k, max_blocks)
        if not starts:
            continue
        Js = []
        for s in starts:
            I, U, lags = block_counts(world, masks, midx, [s + t * k for t in range(H + 1)], p, p_small, small)
            Js.append(bdc.lag_curve(I, U, lags)[0])
        out[k] = (np.nanmean(np.array(Js), axis=0), len(starts))
    return out


def best_k(curves, r_b):
    d = {k: float(np.nansum((ratio(J)[1:, 3] - r_b[1:, 3]) ** 2)) for k, (J, _) in curves.items()}
    return min(d, key=d.get), d


def survey_masks(pattern):
    ref = bdc.split_ref(pattern)
    keep, east, overlap = bdc.fit_half(sorted(glob.glob(pattern)), ref)
    masks = []
    for f in keep:
        K = np.asarray(bdc.load(f)["known_target"]).astype(bool)
        masks.append(K.reshape(len(K), -1))
    return masks, ref, len(keep), east, overlap


def bnm_curve(peryear, minlist, ref, seed=0):
    per, lags, years = bdc.analyse(peryear, minlist, ref)
    J, Hs = bdc.lag_curve(sum(x[0] for x in per), sum(x[1] for x in per), lags)
    rng = np.random.default_rng(seed); bJ = []
    for _ in range(N_BOOT):
        idx = rng.integers(len(per), size=len(per))
        bJ.append(bdc.lag_curve(sum(per[j][0] for j in idx), sum(per[j][1] for j in idx), lags)[0])
    bJ = np.array(bJ); bR = bJ / (1 + bJ); bR = bR / bR[:, 0:1]
    q = lambda a, x: np.nanquantile(a, x, 0)
    return dict(J=J, J_lo=q(bJ, 0.025), J_hi=q(bJ, 0.975), r=ratio(J), r_lo=q(bR, 0.025), r_hi=q(bR, 0.975),
                Hs=Hs, years=years)


def read(worlds_curves, b):
    """Reading rule R1-R3 on the pooled curves; returns (lines, verdict dict)."""
    ks = sorted(set.intersection(*[set(c) for c in worlds_curves]))
    if not ks:
        return ["no k common to every world"], dict(verdict="no k")
    pooled = {k: (np.nanmean([c[k][0] for c in worlds_curves], axis=0), sum(c[k][1] for c in worlds_curves)) for k in ks}
    kstar, dist = best_k(pooled, b["r"])
    per_world = [best_k(c, b["r"])[0] for c in worlds_curves]
    v = dict(ks=ks, kstar=kstar, per_world=per_world, dist=dist, pooled=pooled)
    r1 = ratio(pooled[ks[0]][0])[H - 1, 3]; rmax = ratio(pooled[ks[-1]][0])[H - 1, 3]
    v.update(r5_first=r1, r5_last=rmax, r5_lo=b["r_lo"][H - 1, 3], r5_hi=b["r_hi"][H - 1, 3])
    lines = []
    if ks[0] == 1 and r1 < b["r_lo"][H - 1, 3]:
        v["verdict"] = "too fast"
        lines.append(f"R1 too fast: at one frame apart r_all(5) {r1:.3f} is below BNM's interval ({b['r_lo'][H - 1, 3]:.3f}-"
                     f"{b['r_hi'][H - 1, 3]:.3f}); the worlds change more between neighbouring frames than BNM between years "
                     "-> fix before the real run")
    elif rmax > b["r_hi"][H - 1, 3]:
        v["verdict"] = "too slow"
        lines.append(f"R3 too slow: at {ks[-1]} frames apart r_all(5) {rmax:.3f} is above BNM's interval (upper end "
                     f"{b['r_hi'][H - 1, 3]:.3f}); the worlds change slower than BNM at every k tried")
    else:
        v["verdict"] = "interval found"
        Rk = ratio(pooled[kstar][0])
        out = [lab for c, (_, _, lab) in enumerate(CLS[:3]) if not (b["r_lo"][H - 1, c] <= Rk[H - 1, c] <= b["r_hi"][H - 1, c])]
        v["classes_out"] = out
        lines.append(f"R2 frame interval: {kstar} frame(s) per BNM year (pooled); classes with r(5) outside BNM's interval "
                     f"at k*: {', '.join(out) if out else 'none'}")
    share = sum(1 for kw in per_world if kstar / 2 <= kw <= kstar * 2)
    lines.append(f"   per world k*: {per_world}; {share} of {len(per_world)} within a factor 2 of the pooled k*"
                 + ("" if share >= min(3, len(per_world)) else " -> worlds disagree"))
    return lines, v


# ------------------------------------------------ self-test
def markov(rng, Z, persist):
    """One step of slow change that keeps the expected range size (as bnm_detection_check's planted grids)."""
    n_occ = Z.sum(1, keepdims=True); cells = Z.shape[1]
    gain = (1 - persist) * n_occ / np.maximum(cells - n_occ, 1)
    u = rng.random(Z.shape)
    return np.where(Z, u >= 1 - persist, u < gain)


def planted_world(path, rng, persist, n_rounds=160, S=40, G=20, turnover=False, flicker=0.0, pieces=None):
    """Planted frames: slow change per frame; flicker: each present cell missing from a frame with this chance, on its
    own (does not build up); pieces: write ASM_F_* with this many frames per round instead of ASM_P_*."""
    k = np.exp(rng.uniform(0, np.log(160), S)).astype(int)
    Z = np.zeros((S, G * G), bool)
    for i in range(S):
        Z[i, rng.choice(G * G, k[i], replace=False)] = True
    ids = np.arange(S); keys, offs, idl, bits = [], [0], [], []
    n_frames = n_rounds * (pieces or 1)
    for t in range(n_frames):
        if t:
            Z = markov(rng, Z, persist)
            Z[~Z.any(1), rng.integers(G * G)] = True
            if turnover and t % 7 == 0:
                j = rng.integers(S); ids = ids.copy(); ids[j] = ids.max() + 1
                Z[j] = False; Z[j, rng.choice(G * G, 5, replace=False)] = True
        F = Z & (rng.random(Z.shape) >= flicker) if flicker else Z
        F[~F.any(1)] = Z[~F.any(1)]
        keys.append(t); idl.append(ids.copy()); bits.append(np.packbits(F, axis=1)); offs.append(offs[-1] + S)
    out = dict(P_last_final=Z.reshape(S, G, G).astype(np.uint8))
    if pieces:
        out.update(ASM_F_round=np.array(keys) // pieces, ASM_F_piece=np.array(keys) % pieces, ASM_F_offsets=np.array(offs),
                   ASM_F_ids=np.concatenate(idl), ASM_F_bits=np.concatenate(bits),
                   ASM_P_round=np.arange(n_rounds), ASM_P_offsets=np.zeros(n_rounds + 1, int), ASM_P_ids=np.zeros(0, int),
                   ASM_P_bits=np.zeros((0, 50), np.uint8))
    else:
        out.update(ASM_P_round=np.array(keys), ASM_P_offsets=np.array(offs), ASM_P_ids=np.concatenate(idl),
                   ASM_P_bits=np.concatenate(bits))
    np.savez(path, **out)


def planted_bnm(tmp, rng, persist, k0, p, n_win=12, S=40, G=20, years=range(2014, 2020)):
    """BNM per-year grids from the same slow change, one year every k0 steps, detection p, fit and test halves."""
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


def run_case(tmp, rng, persist_sim, persist_bnm, k0, p=0.8, ks=(1, 2, 3, 4, 5, 6), n_worlds=2, **wkw):
    peryear, refpat = planted_bnm(tmp, rng, persist_bnm, k0, p)
    bdc.OPENED.clear()
    masks, ref, n_fit, east, overlap = survey_masks(refpat)
    b = bnm_curve(peryear, 8, ref)
    curves = []
    for j in range(n_worlds):
        planted_world(f"{tmp}/w{j}.npz", rng, persist_sim, **wkw)
        curves.append(world_curves(World(f"{tmp}/w{j}.npz"), masks, p, p, 0, ks=ks))
    lines, v = read(curves, b)
    n_test = len([o for o in bdc.OPENED if bdc.window_en(o.split("@")[0])[0] >= ref["cut"]])
    return v, b, n_test, curves


def self_test():
    rng = np.random.default_rng(7); res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        v, b, n_test, curves = run_case(tmp, rng, 0.97, 0.97, 3)
        check("T1", v["verdict"] == "interval found" and v["kstar"] == 3 and n_test == 0,
              f"same change per frame, one BNM year = 3 frames: k* {v['kstar']} (per world {v['per_world']}); test-half "
              f"files opened {n_test}")
    with tempfile.TemporaryDirectory() as tmp:
        v, b, n_test, curves = run_case(tmp, rng, 0.97, 0.97, 3, flicker=0.25)
        lev = min(curves[0], key=lambda k: float(np.nansum((curves[0][k][0][:, 3] - b["J"][:, 3]) ** 2)))
        check("T2", v["kstar"] == 3 and lev != 3,
              f"the same worlds with 25% of presences missing from each frame on their own (flicker): k* by the slope "
              f"{v['kstar']}; version 1's level match would read {lev}")
    with tempfile.TemporaryDirectory() as tmp:
        v, *_ = run_case(tmp, rng, 0.80, 0.97, 1, ks=(1, 2, 3))
        check("T3", v["verdict"] == "too fast", f"world changes much faster per frame (persist 0.80) than BNM per year "
              f"(0.97): '{v['verdict']}' (r_all(5) one frame apart {v['r5_first']:.3f}, BNM lower end {v['r5_lo']:.3f})")
    with tempfile.TemporaryDirectory() as tmp:
        v, *_ = run_case(tmp, rng, 1.0, 0.97, 1, ks=(1, 2, 3, 4))
        check("T4", v["verdict"] == "too slow", f"a world with no change: '{v['verdict']}' (r_all(5) at 4 frames apart "
              f"{v['r5_last']:.3f}, BNM upper end {v['r5_hi']:.3f})")
    with tempfile.TemporaryDirectory() as tmp:
        v, *_ = run_case(tmp, rng, 0.97, 0.97, 3, n_rounds=40, pieces=5)
        check("T5", v["kstar"] == 3 and v["verdict"] == "interval found",
              f"sub-step maps (ASM_F_*, 5 per round, 200 frames): k* {v['kstar']} frames")
        planted_world(f"{tmp}/w.npz", rng, 1.0, n_rounds=40)
        Wd = World(f"{tmp}/w.npz")
        masks = [np.ones((5, 400), bool)] * 6
        I, U, lags = block_counts(Wd, masks, mask_index(Wd, masks), [20, 21, 22, 23, 24, 25], 0.8, 0.8, 0)
        J = I[:, 2].sum() / U[:, 2].sum()
        check("T6", abs(J - 0.8 / 1.2) < 0.02, f"no change, p 0.8, 21+ class: J {J:.3f} = p / (2 - p) {0.8 / 1.2:.3f}")
        m1 = [np.ones((3, 400), bool)]
        planted_world(f"{tmp}/t.npz", rng, 0.9, n_rounds=40, turnover=True)
        Wt = World(f"{tmp}/t.npz"); fr = [20, 22, 24, 26, 28, 30]
        I, U, lags = block_counts(Wt, m1, mask_index(Wt, m1), fr, 1.0, 1.0, 0)
        maps = [dict(zip(Wt.frame(i)[0].tolist(), Wt.frame(i)[1])) for i in fr]
        allid = set().union(*[set(m) for m in maps]); good = True
        for k, (a, b_) in enumerate(itertools.combinations(range(6), 2)):
            ii = uu = 0
            for s in allid:
                A = maps[a].get(s, np.zeros(400, bool)); B = maps[b_].get(s, np.zeros(400, bool))
                ii += (A & B).sum(); uu += (A | B).sum()
            good &= (I[k, 3] == ii and U[k, 3] == uu)
        check("T7", good and len(allid) > 40, f"counts against a plain loop over species, with species leaving and entering "
              f"({len(allid)} ids over 6 frames)")
        mk = [np.random.default_rng(4).random((30, 400)) < 0.6] * 3
        c1 = world_curves(World(f"{tmp}/t.npz"), mk, 0.8, 0.64, 20, ks=(1, 2))
        c2 = world_curves(World(f"{tmp}/t.npz"), mk, 0.8, 0.64, 20, ks=(1, 2))
        check("T8", all(np.array_equal(c1[k][0], c2[k][0], equal_nan=True) for k in (1, 2)),
              "the same world read twice gives the same J (fixed survey and detection streams)")
        Jt = np.array([[0.6, 0.5, 0.65, 0.62], [0.55, 0.46, 0.6, 0.57], [0.5, 0.42, 0.55, 0.52], [0.46, 0.39, 0.5, 0.48],
                       [0.42, 0.36, 0.46, 0.44]])
        r = ratio(Jt); q = Jt / (1 + Jt)
        check("T9", np.allclose(r[0], 1) and np.allclose(r[4], q[4] / q[0]) and np.allclose(ratio(Jt * 1.0), r),
              "r(h) = q(h) / q(1) on a written example; r(1) = 1")
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
    ap.add_argument("--max-blocks", type=int, default=40)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.sim:
        sys.exit("STOP: give --sim patterns")
    files = sorted({f for p in a.sim for f in glob.glob(p)})
    masks, ref, n_fit, east, overlap = survey_masks(a.survey)
    print(f"survey: {n_fit} fit-half BNM windows (test half left unopened: {east} east of the cut, {overlap} sharing cells)")
    b = bnm_curve(a.peryear, a.minlist, ref)
    print(f"BNM years {b['years']}: J(h) and r(h) = q(h) / q(1) by class, 95% interval over windows")
    for c, (_, _, lab) in enumerate(CLS):
        print(f"   {lab:>5s}  J " + "  ".join(f"h{h} {b['J'][i, c]:.3f}" for i, h in enumerate(b["Hs"]))
              + f"   r(5) {b['r'][H - 1, c]:.3f} ({b['r_lo'][H - 1, c]:.3f}-{b['r_hi'][H - 1, c]:.3f})   p_eff {p_eff(b['J'])[c]:.3f}")
    curves = []
    for f in files:
        try:
            W = World(f)
        except ValueError as e:
            print(f"skipped {Path(f).name}: {e}"); continue
        c = world_curves(W, masks, a.detect, a.detect_small, a.small_cells, max_blocks=a.max_blocks)
        curves.append(c)
        kw, _ = best_k(c, b["r"])
        print(f"\n{W.name[-70:]}: {W.kind}, {len(W.key)} frames ({W.per_round} per round), last half {len(W.half)}; k* {kw}")
        for k in (1, 2, 5, 10, 20):
            if k in c:
                J, nb = c[k]; R = ratio(J)
                print(f"   k {k:2d} ({nb} blocks)  r_all lags 2-5: " + " ".join(f"{x:.3f}" for x in R[1:, 3])
                      + "   r(5) 1-5 / 6-20 / 21+: " + " / ".join(f"{x:.3f}" for x in R[H - 1, :3])
                      + f"   p_eff all {p_eff(J)[3]:.3f}")
    if not curves:
        sys.exit("STOP: no world read")
    lines, v = read(curves, b)
    print(f"\npooled over {len(curves)} worlds")
    if v.get("kstar"):
        R = ratio(v["pooled"][v["kstar"]][0])
        print(f"   k* {v['kstar']}: r(5) by class " + "; ".join(
            f"{lab} {R[H - 1, c]:.3f} (BNM {b['r'][H - 1, c]:.3f}, {b['r_lo'][H - 1, c]:.3f}-{b['r_hi'][H - 1, c]:.3f})"
            for c, (_, _, lab) in enumerate(CLS)))
    print("reading rule (written before the run, 1 Nov):")
    for s in lines:
        print("  " + s)