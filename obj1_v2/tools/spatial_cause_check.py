#!/usr/bin/env python3
"""
spatial_cause_check.py  (obj1_v2, gates 3 and 4: similarity decay and rare clumping, 1 Nov 2026, Sara's yes)

Question: why does the similarity of two recorded cells fall with distance much more in our worlds (decay 0.11-0.20 in
n66-n69, 0.22-0.32 in static worlds) than in BNM (0.74 -> 0.70, decay 0.04), and why are rare ranges (20 recorded cells
or fewer) too clumped at 2-3 cells in 3 of 4 worlds? Is it the habitat layout (where each species can grow) or how
species fill their habitat (dispersal, founding, competition)? Which range sizes carry the excess?
Reads saved files only (training files, BNM fit half); writes nothing.

Measures, on BNM windows and on each simulated end map seen through the BNM survey exactly as world_vs_bnm v5.8 sees it
(observe: known cells of a random BNM species of each fit-half window, rng seed 0; detection on stream [0, 7, window],
--detect-small for species with --small-cells or fewer present surveyed cells):
  decay     Sorensen of two recorded cells at Chebyshev distance 1 minus at 5-10, mean over cell pairs pooled over
            windows (world_vs_bnm's alpha_beta, windows, no wrap-around).
  rare      co-occupancy of cases with 20 or fewer recorded cells at distance 1, 2, 3, 4: occupied pairs over the
            exact chance R(R-1)/(K(K-1)) x known pairs (world_vs_bnm's cooc), reported against BNM.
  split     pooled Sorensen S*(d) = sum over pairs 2 shared / sum over pairs (n_i + n_j) is a sum over species, so its
            decay S*(1) - S*(5-10) splits exactly into species parts; summed by recorded range class in each window
            (1-5, 6-20, 21-50, 51+) for the simulated maps and BNM.
Two nulls per simulated world (20 draws each, mean):
  habitat null  each species keeps its number of cells where its own final growth rate is above 0 (ENV_r_field) and
                its number where it is not (sinks). The first are placed at random among its suitable cells; the sinks
                then go to the unsuitable cells nearest those (ties at random), since immigrants from nearby sources
                feed a sink. Range sizes, the habitat layout and spill-over stay; where the species sits inside its
                habitat is random.
  random null   each species' cells placed at random over the whole world (no spatial structure; control).

Reading rule (written 1 Nov, before the run):
  D0 the tool's decay and rare co-occupancy against BNM on the real maps equal world_vs_bnm's printed values of the
     1 Nov run (2 decimals) in every world; random null decay within 0.015 of 0 and its rare co-occupancy within 0.1 of 1
     at every distance. Otherwise nothing below is read.
  Decay, per world (a setting needs 3 of 4 worlds):
     A habitat layout: habitat-null decay above 0.085 (BNM band top) -> the layout of suitable cells alone gives too
       much decay; the fix goes to the habitat layer or the niches.
     B filling: habitat-null decay 0.085 or less, real decay above 0.085 -> where species sit inside their habitat
       makes it; real minus habitat-null decay split by thirds of d: the lowest third holds half or more -> dispersal;
       otherwise founding or competition.
  Rare clumping, per world, at distances 2 and 3 (the failing ones; band 1 +- 0.232 and 0.137 against BNM):
     habitat null outside the band -> habitat layout; habitat null inside and real outside -> filling (d split as above).
  Split: the range class(es) holding half or more of the excess decay (simulated minus BNM, pooled form) are named.
  Expected: none written; the check decides.

Version 2 (1 Nov, after Sara's run of version 1, kept as spatial_cause_check_v1.py):
  - rare clumping rule fixed: "habitat layout" only when the habitat null is outside the band on the same side as the
    real maps; version 1 did not say the side, and on n66-n69 the habitat null sat below the band (0.69-0.74 at 2 cells)
    while the real maps sat above it, so version 1 named the layout for an excess the layout does not make.
  - B filling is split further with a competition null (connectance-1 worlds; C is known: 1 on the diagonal, s off it):
    g_i(x) = r_i(x) - B_i(x) - s sum_{j != i} B_j(x) from ENV_r_field and B_last (sink_growth.py's g). Each species is
    placed at random among the cells it holds plus the empty cells where g > 0 (cells it could hold against the
    residents), keeping its range size. Check first (C0): median g over occupied cells within 0.02 of 0 in every
    world (sink_growth E), else the competition part is not read.
    Rule (written 1 Nov, before the run): competition-null decay above 0.085 -> competition sorting: each species holds
    only the suitable cells where it beats the residents, and those cells are clumped (fix on the competition and
    niche side); 0.085 or less -> not reached: species could hold empty suitable cells but have not arrived (fix on
    the founding and spread side). Rare clumping at 2 and 3 cells the same way, side included.
    Reported: share of empty suitable cells (r > 0) where g > 0.

Run from obj1_v2/work:
    python3 ../tools/spatial_cause_check.py --self-test | tail -1
    python3 ../tools/spatial_cause_check.py --sim "results/stage2a/niche_h/*_fb_ht62500_hs1p0t62500_*_training.npz" "results/stage2a/niche_h2/*_training.npz" | tee ../logs/spatial_cause_1nov.txt
"""
import argparse, glob, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bnm_detection_check as bdc          # noqa: E402  (fit-half rule of world_vs_bnm, unchanged)

SOR = [(1, 1), (2, 2), (3, 4), (5, 10)]
RARE_D = [1, 2, 3, 4]
RARE_BAND = [0.196, 0.232, 0.137, 0.077]
CLASSES = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 50, "21-50"), (51, 10 ** 6, "51+")]
N_NULL = 20
DECAY_TOP = 0.085


# ------------------------------------------------ observation, as world_vs_bnm v5.8
def observe(P, masks, detect, detect_small, small, seed=0):
    """world_vs_bnm v5.8 observe(): (records, known) per survey window."""
    rng = np.random.default_rng(seed); out = []
    for m in masks:
        K = m[rng.integers(len(m), size=len(P))]
        if detect < 1 or detect_small is not None:
            U = np.random.default_rng([seed, 7, len(out)]).random(P.shape)
            pr = np.full(len(P), detect)
            if detect_small is not None:
                pr[(P & K).reshape(len(P), -1).sum(1) <= small] = detect_small
            out.append((P & K & (U < pr.reshape((-1,) + (1,) * (P.ndim - 1))), K))
        else:
            out.append((P & K, K))
    return out


def pair_counts(A):
    """Ordered pairs of True cells at every Chebyshev distance, per array (n, G0, G1), windows (no wrap):
    (n, max distance + 1) integer counts; FFT autocorrelation as world_vs_bnm's pair_counts."""
    n, G0, G1 = A.shape
    P0, P1 = 2 * G0, 2 * G1
    f = np.fft.fft2(A.astype(float), s=(P0, P1))
    out = np.rint(np.real(np.fft.ifft2(np.conj(f) * f))).astype(np.int64)
    dy = np.minimum(np.arange(P0), P0 - np.arange(P0)); dx = np.minimum(np.arange(P1), P1 - np.arange(P1))
    ch = np.maximum(dy[:, None], dx[None, :]).ravel()
    flat = out.reshape(n, -1)
    return np.stack([flat[:, ch == k].sum(1) for k in range(max(RARE_D) + 1)], axis=1)


def cheb(G0, G1, idx):
    yy, xx = np.divmod(idx, G1)
    return np.maximum(np.abs(yy[:, None] - yy[None, :]), np.abs(xx[:, None] - xx[None, :]))


# ------------------------------------------------ statistics
def stats(cases, sp_tags=None):
    """cases: list of (records (S, G, G) bool, known). Returns mean-of-ratios Sorensen by distance (world_vs_bnm),
    pooled Sorensen numerators per (case, species) and denominators, rare co-occupancy obs and exp by distance, and the
    recorded class of every (case, species)."""
    sor_sum, sor_n = np.zeros(len(SOR)), np.zeros(len(SOR))
    num, den, cls_, tag = [], np.zeros(len(SOR)), [], []
    obs, exp = np.zeros(len(RARE_D)), np.zeros(len(RARE_D))
    for w, (o, K) in enumerate(cases):
        S, G0, G1 = o.shape
        X = o.reshape(S, -1).T.astype(float)
        n = X.sum(1); idx = np.flatnonzero(n > 0)
        R = o.reshape(S, -1).sum(1)
        cls_.append(np.array([next(c for c, (lo, hi, _) in enumerate(CLASSES) if lo <= r <= hi) if r > 0 else -1 for r in R]))
        tag.append(np.arange(S) if sp_tags is None else sp_tags)
        nm = np.zeros((S, len(SOR)))
        if len(idx) > 1:
            Xi = X[idx]; d = cheb(G0, G1, idx); up = np.triu(np.ones_like(d, dtype=bool), 1)
            inter = Xi @ Xi.T; sor = 2.0 * inter / (n[idx][:, None] + n[idx][None, :])
            for j, (lo, hi) in enumerate(SOR):
                msk = (d >= lo) & (d <= hi)
                m_up = up & msk
                sor_sum[j] += sor[m_up].sum(); sor_n[j] += m_up.sum()
                Ms = msk.astype(float); np.fill_diagonal(Ms, 0.0)
                nm[:, j] = (Xi * (Ms @ Xi)).sum(0)                   # = 2 x shared pairs per species
                den[j] += float(n[idx] @ Ms.sum(1))
        num.append(nm)
        # rare co-occupancy (cases with 0 < R < K, K >= 4, R <= 20)
        Kc = K.reshape(S, -1).sum(1)
        ok = np.flatnonzero((R > 0) & (Kc >= 4) & (R < Kc) & (R <= 20))
        if len(ok):
            oo, kk = pair_counts(o[ok]), pair_counts(K[ok])
            p2 = R[ok] * (R[ok] - 1) / (Kc[ok] * (Kc[ok] - 1))
            for j, dd in enumerate(RARE_D):
                obs[j] += oo[:, dd].sum() / 2; exp[j] += (p2 * kk[:, dd]).sum() / 2
    sor_m = np.where(sor_n > 0, sor_sum / np.maximum(sor_n, 1), np.nan)
    return dict(sor=sor_m, decay=sor_m[0] - sor_m[3], num=num, den=den, cls=cls_, tag=tag, obs=obs, exp=exp,
                rare=np.where(exp > 0, obs / np.maximum(exp, 1e-12), np.nan))


def pooled(st):
    """Pooled Sorensen S*(d) and its decay split by class: dict with S (distances), decay, by_class (decay part)."""
    tot = sum(nm.sum(0) for nm in st["num"])
    S = tot / np.maximum(st["den"], 1e-12)
    part = np.zeros((len(CLASSES), len(SOR)))
    for nm, cl in zip(st["num"], st["cls"]):
        for c in range(len(CLASSES)):
            part[c] += nm[cl == c].sum(0)
    part /= np.maximum(st["den"], 1e-12)
    return dict(S=S, decay=S[0] - S[3], by_class=part[:, 0] - part[:, 3], share=[float(np.mean(np.concatenate(st["cls"]) == c))
                                                                                for c in range(len(CLASSES))])


def species_decay(st, S_count):
    """Pooled decay part of every simulated species (summed over windows)."""
    out = np.zeros(S_count)
    for nm, tg in zip(st["num"], st["tag"]):
        np.add.at(out, tg, nm[:, 0] / max(st["den"][0], 1e-12) - nm[:, 3] / max(st["den"][3], 1e-12))
    return out


# ------------------------------------------------ nulls
def habitat_null(P, fp, rng):
    """Each species keeps its number of cells inside its own suitable cells (r > 0) and outside them (sinks). Cells
    inside are placed at random among its suitable cells; its sink cells are then put in the unsuitable cells nearest
    to those (torus Chebyshev distance, ties at random), because a sink is fed by immigrants from nearby sources.
    Habitat layout and spill-over stay; where in its habitat a species sits is random."""
    S, G0, G1 = P.shape; out = np.zeros_like(P); Pf, Ff = P.reshape(S, -1), fp.reshape(S, -1); Of = out.reshape(S, -1)
    yy, xx = np.divmod(np.arange(G0 * G1), G1)
    for s in range(S):
        n_in = int((Pf[s] & Ff[s]).sum()); n_out = int((Pf[s] & ~Ff[s]).sum())
        src = rng.choice(np.flatnonzero(Ff[s]), n_in, replace=False) if n_in else np.zeros(0, int)
        Of[s, src] = True
        if n_out:
            pool = np.flatnonzero(~Ff[s])
            if len(src):
                dy = np.abs(yy[pool][:, None] - yy[src][None, :]); dx = np.abs(xx[pool][:, None] - xx[src][None, :])
                dist = np.maximum(np.minimum(dy, G0 - dy), np.minimum(dx, G1 - dx)).min(1)
            else:
                dist = np.zeros(len(pool))
            order = np.lexsort((rng.random(len(pool)), dist))
            Of[s, pool[order[:n_out]]] = True
    return out


def competition_allowed(P, F, B, s):
    """Cells each species holds or could hold against the residents (g > 0); g as sink_growth.py (connectance 1).
    Returns (allowed (S, G, G) bool, g)."""
    tot = B.sum(axis=0, keepdims=True)
    g = F - B - s * (tot - B)
    return P | (g > 0), g


def allowed_null(P, allowed, rng):
    """Each species' range size placed at random among its allowed cells."""
    S = len(P); out = np.zeros_like(P); Pf, Af = P.reshape(S, -1), allowed.reshape(S, -1); Of = out.reshape(S, -1)
    for s_ in range(S):
        Of[s_, rng.choice(np.flatnonzero(Af[s_]), int(Pf[s_].sum()), replace=False)] = True
    return out


def random_null(P, rng):
    S = len(P); out = np.zeros_like(P); Pf = P.reshape(S, -1); Of = out.reshape(S, -1)
    for s in range(S):
        Of[s, rng.choice(Pf.shape[1], int(Pf[s].sum()), replace=False)] = True
    return out


# ------------------------------------------------ BNM
def load_bnm(pattern):
    ref = bdc.split_ref(pattern)
    keep, east, overlap = bdc.fit_half(sorted(glob.glob(pattern)), ref)
    cases, masks = [], []
    for f in keep:
        z = bdc.load(f)
        K = np.asarray(z["known_target"]).astype(bool)
        cases.append((np.asarray(z["obs_mask"]).astype(bool), K)); masks.append(K)
    return cases, masks, ref, east, overlap


# ------------------------------------------------ one world
def analyse_world(P, F, d, masks, bnm, args, n_null=N_NULL, seed=1, B=None, s_comp=None):
    obs = lambda Q: stats(observe(Q, masks, args["detect"], args["detect_small"], args["small"]))
    real = obs(P)
    fp = F > 0
    rng = np.random.default_rng(seed)
    hab = [obs(habitat_null(P, fp, rng)) for _ in range(n_null)]
    rnd = [obs(random_null(P, rng)) for _ in range(n_null)]
    S = len(P)
    sd_real = species_decay(real, S); sd_hab = np.mean([species_decay(h, S) for h in hab], axis=0)
    gap = sd_real - sd_hab
    terc = np.quantile(d, [1 / 3, 2 / 3]) if len(d) >= 3 else [np.inf, np.inf]
    low = d <= terc[0]
    share_low = float(gap[low].sum() / gap.sum()) if gap.sum() > 0 else np.nan
    return dict(real=real, hab=hab, rnd=rnd,
                decay=real["decay"], decay_hab=float(np.mean([h["decay"] for h in hab])),
                decay_rnd=float(np.mean([h["decay"] for h in rnd])),
                rare=real["rare"] / bnm["rare"], rare_hab=np.nanmean([h["rare"] for h in hab], axis=0) / bnm["rare"],
                rare_rnd=np.nanmean([h["rare"] for h in rnd], axis=0),
                pooled=pooled(real), pooled_hab=pooled(hab[0]), share_low=share_low,
                fp_fill=float(np.mean([(P[s] & fp[s]).sum() / max(fp[s].sum(), 1) for s in range(S)])),
                in_fp=float((P & fp).sum() / max(P.sum(), 1)), **comp_part(P, F, B, s_comp, fp, obs, rng, n_null, bnm))


def comp_part(P, F, B, s_comp, fp, obs, rng, n_null, bnm):
    if B is None or s_comp is None:
        return dict(comp=None)
    allowed, g = competition_allowed(P, F, B, s_comp)
    c0 = float(np.median(g[P]))
    empty_suit = fp & ~P
    comp = [obs(allowed_null(P, allowed, rng)) for _ in range(n_null)]
    return dict(comp=dict(c0=c0, c0_ok=abs(c0) <= 0.02, reach=float(np.mean(g[empty_suit] > 0)) if empty_suit.any() else np.nan,
                          decay=float(np.mean([h["decay"] for h in comp])),
                          rare=np.nanmean([h["rare"] for h in comp], axis=0) / bnm["rare"]))


def verdict(r):
    out = {}
    if r["decay_hab"] > DECAY_TOP:
        out["decay"] = "A habitat layout"
    elif r["decay"] > DECAY_TOP:
        out["decay"] = "B filling: " + ("dispersal" if r["share_low"] >= 0.5 else "founding or competition")
    else:
        out["decay"] = "inside the band"
    hi = lambda v, j: v[j] > 1 + RARE_BAND[j]
    lo = lambda v, j: v[j] < 1 - RARE_BAND[j]
    real_side = [("high" if hi(r["rare"], j) else "low" if lo(r["rare"], j) else "in") for j in (1, 2)]
    if all(x == "in" for x in real_side):
        out["rare"] = "inside the band"
    else:
        same = all((x == "high" and hi(r["rare_hab"], j)) or (x == "low" and lo(r["rare_hab"], j)) or x == "in"
                   for x, j in zip(real_side, (1, 2)))
        out["rare"] = "habitat layout" if same else "filling: " + ("dispersal" if r["share_low"] >= 0.5 else "founding or competition")
    c = r.get("comp")
    if c and c["c0_ok"] and out["decay"].startswith("B") and "dispersal" not in out["decay"]:
        out["decay"] += " -> " + ("competition sorting" if c["decay"] > DECAY_TOP else "not reached")
    if c and c["c0_ok"] and out["rare"].startswith("filling") and "dispersal" not in out["rare"]:
        same = all((x == "high" and hi(c["rare"], j)) or (x == "low" and lo(c["rare"], j)) or x == "in"
                   for x, j in zip(real_side, (1, 2)))
        out["rare"] += " -> " + ("competition sorting" if same else "not reached")
    return out


def class_excess(r, bnm_pooled):
    ex = r["pooled"]["by_class"] - bnm_pooled["by_class"]
    tot = ex.sum()
    acc, named = 0.0, []
    for c in np.argsort(-ex):
        if tot <= 0 or ex[c] <= 0:
            break
        named.append(CLASSES[c][2]); acc += ex[c]
        if acc >= 0.5 * tot:
            break
    return ex, named


# ------------------------------------------------ self-test
def smooth_field(rng, G, ls):
    k = np.fft.fftfreq(G)
    spec = np.exp(-0.5 * (2 * np.pi * ls) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
    H = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(G, G))) * spec))
    return (H - H.mean()) / H.std()


def blob(G, size, rng):
    c = rng.integers(G, size=2); m = np.zeros((G, G), bool); rad = 0
    while m.sum() < size:
        yy, xx = np.ogrid[:G, :G]
        dy = np.minimum(np.abs(yy - c[0]), G - np.abs(yy - c[0])); dx = np.minimum(np.abs(xx - c[1]), G - np.abs(xx - c[1]))
        m = np.maximum(dy, dx) <= rad; rad += 1
    cells = np.flatnonzero(m); keep = np.zeros(G * G, bool); keep[rng.choice(cells, size, replace=False)] = True
    return keep.reshape(G, G)


def planted_bnm(tmp, rng, G=20, S=40, n_win=12, frac=0.6):
    ref = Path(tmp) / "bnm"; ref.mkdir()
    for w in range(n_win):
        e, n = 40 * (w % 6), 40 * (w // 6)
        size = np.exp(rng.uniform(0, np.log(200), S)).astype(int)
        O = np.zeros((S, G * G), bool)
        for i in range(S):
            O[i, rng.choice(G * G, size[i], replace=False)] = True
        K = np.broadcast_to(rng.random(G * G) < frac, (S, G * G)).copy()
        np.savez(ref / f"bnm_e{e:04d}_n{n:04d}.npz", obs_mask=(O & K & (rng.random(O.shape) < 0.8)).reshape(S, G, G),
                 known_target=K.reshape(S, G, G), window_cells=np.int64(G), cell_m=np.float64(2000))
    return str(ref / "bnm_*.npz")


def planted_world(kind, rng, G=20, S=50):
    if kind in ("habitat", "spill"):
        H = smooth_field(rng, G, 2.5)
        opt = rng.normal(0, 1, S); w = rng.uniform(0.15, 0.5, S)
        F = np.exp(-(H[None] - opt[:, None, None]) ** 2 / (2 * w[:, None, None] ** 2)) - 0.3
        P = F > 0
        if kind == "spill":                                   # every unsuitable cell next to a suitable one is a sink
            Q = P.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    Q |= np.roll(np.roll(P, dy, 1), dx, 2)
            P = Q
        keep = P.reshape(S, -1).any(1)
        return P[keep], F[keep], rng.uniform(1e-6, 1e-2, keep.sum())
    F = np.ones((S, G, G)); P = np.zeros((S, G, G), bool); d = np.zeros(S)
    size = np.exp(rng.uniform(np.log(3), np.log(300), S)).astype(int)
    for s in range(S):
        clump = (kind == "fill" and s % 2 == 0) or (kind == "class" and 60 <= size[s] <= 90)
        if kind == "class" and s % 4 == 0:
            size[s] = rng.integers(60, 91); clump = True
        if clump:
            P[s] = blob(G, size[s], rng); d[s] = 1e-6
        else:
            P[s].flat[rng.choice(G * G, size[s], replace=False)] = True; d[s] = 1e-2
    return P, F, d


def self_test():
    rng = np.random.default_rng(3); res = []
    args = dict(detect=0.8, detect_small=0.64, small=20)

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    # T1 exactness: pooled split sums to the direct pooled Sorensen; mean of ratios equals a direct loop
    G = 12; o = rng.random((7, G, G)) < 0.3; K = np.ones_like(o)
    st = stats([(o, K)]); pl = pooled(st)
    cells = [(y, x) for y in range(G) for x in range(G) if o[:, y, x].any()]
    num = np.zeros(4); den = np.zeros(4); sm = np.zeros(4); sn = np.zeros(4)
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            (y1, x1), (y2, x2) = cells[a], cells[b]; dd = max(abs(y1 - y2), abs(x1 - x2))
            A, B = o[:, y1, x1], o[:, y2, x2]
            for j, (lo, hi) in enumerate([(1, 1), (2, 2), (3, 4), (5, 10)]):
                if lo <= dd <= hi:
                    num[j] += 2 * (A & B).sum(); den[j] += A.sum() + B.sum(); sm[j] += 2 * (A & B).sum() / (A.sum() + B.sum()); sn[j] += 1
    good = np.allclose(pl["S"], num / den) and np.allclose(st["sor"], sm / sn) and np.allclose(pl["by_class"].sum(), pl["decay"])
    check("T1", good, "pooled Sorensen and its class split equal a direct loop over cell pairs; mean of ratios equals the loop")
    # T2 co-occupancy against a direct loop
    o2 = np.zeros((3, G, G), bool); o2[0, 2:4, 2:5] = True; o2[1].flat[rng.choice(G * G, 9, replace=False)] = True; o2[2, 7, 7] = True
    K2 = rng.random((3, G, G)) < 0.7; K2 |= o2
    st2 = stats([(o2, K2)])
    ob, ex = np.zeros(4), np.zeros(4)
    for s in range(3):
        oc = [(y, x) for y in range(G) for x in range(G) if o2[s, y, x]]; kn = [(y, x) for y in range(G) for x in range(G) if K2[s, y, x]]
        R, Kc = len(oc), len(kn)
        if not (0 < R < Kc and Kc >= 4 and R <= 20):
            continue
        p2 = R * (R - 1) / (Kc * (Kc - 1))
        for j, dd in enumerate(RARE_D):
            ob[j] += sum(1 for a in range(R) for b in range(a + 1, R) if max(abs(oc[a][0] - oc[b][0]), abs(oc[a][1] - oc[b][1])) == dd)
            ex[j] += p2 * sum(1 for a in range(Kc) for b in range(a + 1, Kc) if max(abs(kn[a][0] - kn[b][0]), abs(kn[a][1] - kn[b][1])) == dd)
    check("T2", np.allclose(st2["obs"], ob) and np.allclose(st2["exp"], ex), "rare co-occupancy counts equal a direct loop")
    with tempfile.TemporaryDirectory() as tmp:
        pat = planted_bnm(tmp, rng)
        bdc.OPENED.clear()
        bcases, masks, ref, east, overlap = load_bnm(pat)
        b = stats(bcases); bp = pooled(b)
        n_test = len([x for x in bdc.OPENED if bdc.window_en(x.split("@")[0])[0] >= ref["cut"]])
        check("T3", n_test == 0 and len(masks) == 6, f"fit half only: {len(masks)} windows read, {n_test} test-half files opened")
        out = {}
        for kind in ("habitat", "fill", "class", "spill"):
            P, F, d = planted_world(kind, rng)
            out[kind] = r = analyse_world(P, F, d, masks, b, args, n_null=8)
            r["v"] = verdict(r); r["named"] = class_excess(r, bp)[1]
        r = out["fill"]
        check("T4", abs(r["decay_rnd"]) <= 0.015 and np.all(np.abs(r["rare_rnd"] - 1) <= 0.1),
              f"random null: decay {r['decay_rnd']:+.3f}, rare co-occupancy {np.round(r['rare_rnd'], 2)} (about 0 and 1)")
        r = out["habitat"]
        check("T5", r["v"]["decay"].startswith("A") and abs(r["decay"] - r["decay_hab"]) < 0.02,
              f"species fill their habitat on a smooth layer: decay {r['decay']:.3f}, habitat null {r['decay_hab']:.3f} -> "
              f"'{r['v']['decay']}'")
        r = out["fill"]
        check("T6", r["v"]["decay"] == "B filling: dispersal" and r["share_low"] >= 0.8,
              f"every cell suitable, low-d species in blobs: decay {r['decay']:.3f}, habitat null {r['decay_hab']:.3f}, lowest "
              f"third of d holds {r['share_low']:.0%} of the gap -> '{r['v']['decay']}'; rare -> '{r['v']['rare']}'")
        r = out["spill"]
        check("T9", r["v"]["decay"].startswith("A") and abs(r["decay"] - r["decay_hab"]) < 0.02 and r["in_fp"] < 0.8,
              f"species fill their habitat and every next cell as sinks ({1 - r['in_fp']:.0%} of occupied cells): decay "
              f"{r['decay']:.3f}, habitat null {r['decay_hab']:.3f} -> '{r['v']['decay']}' (sinks stay next to sources)")
        r = out["class"]
        check("T7", r["named"] == ["21-50"] or (r["named"] and r["named"][0] == "21-50"),
              f"only ranges of 60-90 cells (about 21-50 recorded) in blobs: class holding the excess {r['named']}")
    # T10, T11 competition null on planted worlds (all cells suitable)
    with tempfile.TemporaryDirectory() as tmp2:
        pat2 = planted_bnm(tmp2, rng)
        bc2, masks2, *_ = load_bnm(pat2); b2 = stats(bc2)
        Gp, Sp = 20, 30
        cy, cx = rng.integers(Gp, size=Sp), rng.integers(Gp, size=Sp)
        yy, xx = np.mgrid[:Gp, :Gp]
        dist = np.stack([np.maximum(np.minimum(abs(yy - cy[i]), Gp - abs(yy - cy[i])), np.minimum(abs(xx - cx[i]), Gp - abs(xx - cx[i])))
                         for i in range(Sp)])
        Fp = 1.0 - dist / 20.0 + 0.001 * rng.random((Sp, Gp, Gp))
        best = Fp.argmax(0)
        Pc = np.stack([best == i for i in range(Sp)]); Bc = np.where(Pc, Fp, 0.0)
        dd = np.exp(np.linspace(np.log(1e-6), np.log(1e-2), Sp))[rng.permutation(Sp)]
        rc = analyse_world(Pc, Fp, dd, masks2, b2, args, n_null=6, B=Bc, s_comp=0.95); rc["v"] = verdict(rc)
        check("T10", rc["comp"]["c0_ok"] and "competition sorting" in rc["v"]["decay"],
              f"each cell held by its best species, others excluded (s 0.95): C0 {rc['comp']['c0']:+.4f}, empty suitable cells "
              f"with g > 0 {rc['comp']['reach']:.0%}, decay real {rc['decay']:.3f}, competition null {rc['comp']['decay']:.3f} -> "
              f"'{rc['v']['decay']}'")
        Pr = np.zeros((Sp, Gp, Gp), bool)
        for i in range(Sp):
            Pr[i] = dist[i] <= 2
        k = Pr.sum(0)
        Br = np.where(Pr, 1.0 / (1 + 0.4 * (k - 1).clip(0)), 0.0)
        Fr = np.ones((Sp, Gp, Gp))
        rr = analyse_world(Pr, Fr, dd, masks2, b2, args, n_null=6, B=Br, s_comp=0.4); rr["v"] = verdict(rr)
        check("T11", rr["comp"]["c0_ok"] and "not reached" in rr["v"]["decay"],
              f"blobs in an open world, empty cells invadable (g > 0 in {rr['comp']['reach']:.0%}): C0 {rr['comp']['c0']:+.4f}, "
              f"decay real {rr['decay']:.3f}, competition null {rr['comp']['decay']:.3f} -> '{rr['v']['decay']}'")
        fake = dict(rare=np.array([1.0, 1.4, 1.3, 1.0]), rare_hab=np.array([0.5, 0.7, 0.8, 0.9]), decay=0.2, decay_hab=0.02,
                    share_low=0.2, comp=None)
        fake2 = dict(fake, rare_hab=np.array([1.2, 1.5, 1.4, 1.0]))
        check("T12", verdict(fake)["rare"].startswith("filling") and verdict(fake2)["rare"] == "habitat layout",
              "rare rule with sides: habitat null below the band while real is above -> filling; both above -> habitat layout")

    # T8 observe as written in world_vs_bnm v5.8 (patch_world_vs_bnm_detect.py and _detect_size.py), rebuilt here
    Pz = rng.random((30, 20, 20)) < 0.25; Mz = [rng.random((9, 20, 20)) < 0.6 for _ in range(3)]
    got = observe(Pz, Mz, 0.8, 0.64, 20, seed=0)
    r0 = np.random.default_rng(0); want = []
    for k, m in enumerate(Mz):
        Kk = m[r0.integers(len(m), size=len(Pz))]
        Uk = np.random.default_rng([0, 7, k]).random(Pz.shape)
        pr = np.full(len(Pz), 0.8); pr[(Pz & Kk).reshape(len(Pz), -1).sum(1) <= 20] = 0.64
        want.append((Pz & Kk & (Uk < pr[:, None, None]), Kk))
    good = all(np.array_equal(g[0], w_[0]) and np.array_equal(g[1], w_[1]) for g, w_ in zip(got, want))
    check("T8", good, "the survey and detection draws equal world_vs_bnm v5.8's observe, rebuilt from the patch text")
    print(f"{sum(res)} of {len(res)} PASS" if all(res) else f"FAIL: {len(res) - sum(res)} of {len(res)}")
    return all(res)


# ------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", nargs="+", default=None)
    ap.add_argument("--bnm", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--detect", type=float, default=0.80)
    ap.add_argument("--detect-small", type=float, default=0.64)
    ap.add_argument("--small-cells", type=int, default=20)
    ap.add_argument("--nulls", type=int, default=N_NULL)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.sim:
        sys.exit("STOP: give --sim patterns")
    args = dict(detect=a.detect, detect_small=a.detect_small, small=a.small_cells)
    bcases, masks, ref, east, overlap = load_bnm(a.bnm)
    print(f"BNM: {len(bcases)} fit-half windows (test half left unopened: {east} east of the cut, {overlap} sharing cells)")
    b = stats(bcases); bp = pooled(b)
    print(f"BNM Sorensen 1 / 2 / 3-4 / 5-10: " + " / ".join(f"{x:.3f}" for x in b["sor"]) + f"; decay {b['decay']:.3f}; pooled "
          f"decay {bp['decay']:.3f} by class 1-5 / 6-20 / 21-50 / 51+: " + " / ".join(f"{x:+.3f}" for x in bp["by_class"])
          + "; share of cases " + " / ".join(f"{x:.0%}" for x in bp["share"]))
    rows = []
    for f in sorted({f for p in a.sim for f in glob.glob(p)}):
        z = np.load(f, allow_pickle=True)
        miss = [k for k in ("P_last_final", "ENV_r_field", "SP_disp_rate") if k not in z.files]
        if miss:
            print(f"skipped {Path(f).name}: missing {miss}"); continue
        P = np.asarray(z["P_last_final"]).astype(bool); F = np.asarray(z["ENV_r_field"], float); d = np.asarray(z["SP_disp_rate"], float)
        if F.shape != P.shape or len(d) != len(P):
            print(f"skipped {Path(f).name}: fields {F.shape}, maps {P.shape}, d {d.shape} do not match"); continue
        conn = float(z["CONNECTANCE"]) if "CONNECTANCE" in z.files else None
        Bl = np.asarray(z["B_last"], float) if "B_last" in z.files else None
        s_c = float(z["INTERACTION_STRENGTH"]) if "INTERACTION_STRENGTH" in z.files else None
        if conn is None or abs(conn - 1.0) > 1e-9 or Bl is None or s_c is None or Bl.shape != P.shape:
            Bl, s_c = None, None
        r = analyse_world(P, F, d, masks, b, args, n_null=a.nulls, B=Bl, s_comp=s_c); r["v"] = verdict(r)
        ex, named = class_excess(r, bp); r["named"] = named; rows.append(r)
        print(f"\n{Path(f).name[-60:]}: {len(P)} species; {r['in_fp']:.0%} of occupied cells have r > 0; species fill "
              f"{r['fp_fill']:.0%} of their suitable cells on average")
        print(f"   D0 world_vs_bnm numbers: rare co-occupancy / BNM at 1-4 cells {[round(float(x), 2) for x in r['rare']]}; "
              f"similarity decay {r['decay']:.2f}")
        print(f"   controls, random null: decay {r['decay_rnd']:+.3f}; rare co-occupancy {[round(float(x), 2) for x in r['rare_rnd']]}")
        print(f"   decay: real {r['decay']:.3f}, habitat null {r['decay_hab']:.3f}, random null {r['decay_rnd']:+.3f}, BNM {b['decay']:.3f}; "
              f"gap real - habitat null held by the lowest third of d: {r['share_low']:.0%}")
        print(f"   rare / BNM at 1-4 cells: real {[round(float(x), 2) for x in r['rare']]}, habitat null "
              f"{[round(float(x), 2) for x in r['rare_hab']]}")
        print(f"   pooled decay {r['pooled']['decay']:.3f} (BNM {bp['decay']:.3f}); excess by class 1-5 / 6-20 / 21-50 / 51+: "
              + " / ".join(f"{x:+.3f}" for x in ex) + f"; holding half or more: {', '.join(named) if named else 'none'}")
        c = r["comp"]
        if c is None:
            print("   competition null: not read (needs connectance 1, B_last and INTERACTION_STRENGTH)")
        else:
            print(f"   competition null: C0 median g on occupied cells {c['c0']:+.4f} {'PASS' if c['c0_ok'] else 'FAIL'}; "
                  f"empty suitable cells with g > 0 {c['reach']:.1%}; decay {c['decay']:.3f}; rare / BNM "
                  f"{[round(float(x), 2) for x in c['rare']]}")
        print(f"   verdicts: decay '{r['v']['decay']}'; rare clumping '{r['v']['rare']}'")
    if rows:
        print("\nsetting (3 of 4 worlds needed):")
        for key in ("decay", "rare"):
            vs = [r["v"][key] for r in rows]
            top = max(set(vs), key=vs.count)
            print(f"   {key}: {vs} -> {top if vs.count(top) >= min(3, len(vs)) else 'worlds disagree'}")