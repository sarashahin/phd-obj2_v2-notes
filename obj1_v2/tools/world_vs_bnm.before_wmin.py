"""
world_vs_bnm.py  (obj1_v2, written 24 Oct 2026; version 2 the same day: virtual survey,
one report per file, score against BNM)

Puts simulated worlds and the real BNM windows side by side on the same statistics,
so the Stage 2 settings (d, kernel, founding rule, environment) are judged against
Sara's real data, not against guesses.

Statistics, per case (a species in one BNM window, or a species in one simulated
community), pooled over cases:
  range size        occupied cells; share of cases with 1, 2-5, 6-10, 11-20, 21-50, 51+
  richness          species with at least one occupied cell, per window or community
  patches           4-neighbour patches per case (wrap-around in the torus worlds only),
                    for cases of 6-20 cells
  co-occupancy      (all cases, and cases of 20 cells or fewer) for each Chebyshev distance: occupied pairs observed / expected if
                    the same number of cells were placed at random among the cells
                    known for that case (pairs counted over all cells in the worlds,
                    over surveyed cells only in BNM). 1 = no clustering at that distance.
                    Chance is the exact value for R cells placed among K known cells,
                    R(R-1)/(K(K-1)) per pair; step4 V3 used (R/K)^2, which is too high
                    for small ranges (half the chance value at R = 2), so its ratios
                    read low and its 'chance' distance reads short.
BNM presences are lower bounds (only surveyed cells, one target year), so BNM range
sizes are minima; the co-occupancy ratio corrects for effort by using surveyed cells only.
Step 12c recorded for BNM: 2.40 at one cell, chance reached at 5-6 cells; for the old
worlds 7.07 at one cell and chance by about 3.5.

Version 2:
  --survey BNM_PATTERN  sees each simulated world through real BNM effort. known_target
                        is set per species in step10_bnm_gridder.py (the list length leaves
                        out the focal species; with --flight-window the visit must fall in
                        that species' flight months), so for each BNM window every simulated
                        species gets the known cells of a BNM species of that window drawn
                        at random (fixed seed 0). Occupied cells among them are its records.
                        Detection in known cells is taken as perfect (no data to set it).
                        Both sides are then measured the same way, as windows (no wrap).
                        --n-masks N uses N windows evenly spaced (default 0 = all).
  --each                one report per simulated file (one file per setting in a sweep)
  score (with --bnm)    rare-case co-occupancy at 1-4 cells within x1.3 of BNM; range-size
                        shares within 0.15 total variation distance; species per window
                        within x0.5-2 of BNM. Proposed by Claude from the BNM spread across
                        minlist 4, 8 and 12 (24 Oct): rare co-occupancy 2.47-2.62 at one
                        cell, range-size shares within about 5 points, species 32-34.
Version 3 (25 Oct): BNM files (--bnm and --survey) are read from the geographic fit half only,
  the split of steps 13, 14 and 19 set on --split-ref (default minlist8); test-half windows and
  windows sharing cells with them are never opened. Pass bands are to be set again from the fit half.
Version 4 (28 Oct): species per recorded cell (alpha), turnover gamma / mean alpha (beta), Sorensen
  similarity of cells by distance, and the species excess split into a per-cell and a turnover part
  (geometric means over windows, so gamma = alpha x beta exactly). Description only.
Version 5 (28 Oct): pass bands (BNM minlist 8 +- definition spread + seed spread) for species per
  window, species per cell, turnover, similarity decay, range-size TV, 1-5 and 51+ shares and rare
  co-occupancy at 1-4 cells; PASS/FAIL per world and per setting (median inside, 3 of 4 worlds inside).
Version 5.1 (28 Oct): the setting key also names habitat length scale (niche worlds; 3 if not saved) and
  connectance (0.4 if not saved), so the 2 x 2 worlds of plan niche_f are not pooled.
Version 5.2 (29 Oct): worlds with a dispersal rate per species (DISP_per_species 1) get ", d per species
  <DMIN>-<DMAX>" in their setting key, so plan niche_d is not pooled with its twins.
Version 5.3 (29 Oct): worlds with LVMCM founding (FOUND_rule "best") get "founded best cell, cap <K>" in
  their setting key, so plan niche_b is not pooled with its twins; others keep "founded in <seed size>".
Version 5.4 (29 Oct): worlds whose saved competition strength is not 0.4 get ", strength <s>" in their setting
  key, so plan niche_s is not pooled; worlds at 0.4 and files without INTERACTION_STRENGTH keep their label.
Version 5.5 (30 Oct): --tau T counts a simulated species as present only in cells with T or more individuals in
  the last frame (B_last), for observation and scoring only; the assembly is unchanged. tau 1 = P_last_final.
Run from obj1_v2/work (both arguments are optional; quote the patterns):
  python3 ../tools/world_vs_bnm.py --bnm "../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz" \
                                   --sim "../../results/data/data_eval_unseen/*_training.npz"
  python3 ../tools/world_vs_bnm.py --bnm "<BNM>" --survey "<BNM>" --each --sim "results/stage2a/*.npz"
  python3 ../tools/world_vs_bnm.py --self-test
"""
import argparse, glob, hashlib, sys
from pathlib import Path   # obj1_v2 fit-half
import numpy as np
from scipy import ndimage

BINS = [(1, 1, "1"), (2, 5, "2-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, 10**9, "51+")]
DIST = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 6, "5-6"), (7, 10, "7-10")]
STRUCT = ndimage.generate_binary_structure(2, 1)


# obj1_v2 fit-half (25 Oct 2026): Stage 2 reads the geographic fit half of the BNM windows only.
REF_DEFAULT = "../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz"
REF = None                                   # set from --split-ref before any BNM file is read
TAU = 1.0                                    # obj1_v2 tau: presence threshold for observation (--tau)


def window_en(f):
    """Window origin (easting, northing) in km from step 10's file name bnm_e<km>_n<km>."""
    s = Path(f).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])


def split_ref(pattern):
    """The geographic split of steps 13, 14 and 19, set on the reference windows (the
    Objective 3/4 scoring set): fit = easting below the median of the distinct eastings.
    Only one fit-half file is opened, to read the window side."""
    files = sorted(glob.glob(pattern))
    if not files: sys.exit(f"STOP: no reference BNM files matched {pattern}; the fit half cannot be set")
    eas = sorted({window_en(f)[0] for f in files}); cut = eas[len(eas) // 2]
    fit = [f for f in files if window_en(f)[0] < cut]
    if not fit: sys.exit("STOP: the reference set has no fit-half window")
    z = np.load(fit[0], allow_pickle=True)
    side = (int(z["window_cells"]) * float(z["cell_m"]) / 1000) if {"window_cells", "cell_m"} <= set(z.files) \
        else z["obs_mask"].shape[-1] * 2.0
    return dict(cut=cut, side=side, n=len(files), test=[window_en(f) for f in files if window_en(f)[0] >= cut])


def fit_half(files, ref):
    """Windows Stage 2 may read: west of the cut and sharing no cell with a test window.
    Decided from file names alone; nothing is opened."""
    keep, east, overlap = [], 0, 0
    for f in sorted(files):
        e, n = window_en(f)
        if e >= ref["cut"]: east += 1
        elif any(abs(e - te) < ref["side"] and abs(n - tn) < ref["side"] for te, tn in ref["test"]): overlap += 1
        else: keep.append(f)
    return keep, east, overlap


def bnm_glob(pattern):
    """BNM files for Stage 2: the geographic fit half only. Test-half files are never opened."""
    files = sorted(glob.glob(pattern))
    keep, east, overlap = fit_half(files, REF)
    print(f"BNM files {pattern}: {len(files)} matched; {len(keep)} fit-half windows read; {east} test-half windows and "
          f"{overlap} sharing cells with a test window left unopened (cut at easting {REF['cut']} km, set on "
          f"{REF['n']} reference windows)")
    return keep


def patches(m, wrap):
    if m.sum() == 0: return 0
    lab, n = ndimage.label(m, structure=STRUCT)
    if not wrap: return n
    par = list(range(n + 1))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    G0, G1 = m.shape
    for x in range(G1):
        if m[0, x] and m[G0 - 1, x]: par[f(lab[0, x])] = f(lab[G0 - 1, x])
    for y in range(G0):
        if m[y, 0] and m[y, G1 - 1]: par[f(lab[y, 0])] = f(lab[y, G1 - 1])
    return len({f(v) for v in np.unique(lab[m])})


def pair_counts(A, B, wrap):
    """Ordered pairs sum_x A(x) B(x + delta), by displacement delta, for a stack (n, G, G)."""
    n, G0, G1 = A.shape
    if wrap:
        out = np.real(np.fft.ifft2(np.conj(np.fft.fft2(A)) * np.fft.fft2(B)))
        dy = np.minimum(np.arange(G0), G0 - np.arange(G0)); dx = np.minimum(np.arange(G1), G1 - np.arange(G1))
    else:
        P0, P1 = 2 * G0, 2 * G1
        out = np.real(np.fft.ifft2(np.conj(np.fft.fft2(A, s=(P0, P1))) * np.fft.fft2(B, s=(P0, P1))))
        dy = np.minimum(np.arange(P0), P0 - np.arange(P0)); dx = np.minimum(np.arange(P1), P1 - np.arange(P1))
    cheb = np.maximum(dy[:, None], dx[None, :])
    return np.rint(out).astype(np.int64), cheb


def cooc(occ, known, wrap, rmax=None):
    """Pooled observed and expected occupied pairs by distance group, cases with 0 < p < 1
    (and at most rmax occupied cells if given)."""
    R = occ.reshape(len(occ), -1).sum(1); K = known.reshape(len(known), -1).sum(1)
    ok = (R > 0) & (K >= 4) & (R < K) & ((R <= rmax) if rmax else True)
    occ, known, R, K = occ[ok].astype(float), known[ok].astype(float), R[ok], K[ok]
    if not len(occ): return np.zeros(len(DIST)), np.zeros(len(DIST))
    oo, cheb = pair_counts(occ, occ, wrap); kk, _ = pair_counts(known, known, wrap)
    p2 = R * (R - 1) / (K * (K - 1))          # exact chance for R cells placed at random among K known cells
    obs, exp = [], []
    for lo, hi, _ in DIST:
        g = (cheb >= lo) & (cheb <= hi)
        obs.append(oo[:, g].sum()); exp.append((p2[:, None] * kk[:, g]).sum())
    return np.array(obs, float), np.array(exp, float)


SOR_DIST = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 10, "5-10")]   # obj1_v2 alpha


def alpha_beta(cases, wrap):
    """Species per recorded cell (alpha), gamma / mean alpha per window (beta), and the mean Sorensen
    similarity of two recorded cells by Chebyshev distance, pooled over windows."""
    alphas, gam, abar, sor_sum, sor_n = [], [], [], np.zeros(len(SOR_DIST)), np.zeros(len(SOR_DIST))
    for o, _ in cases:
        S, G0, G1 = o.shape
        X = o.reshape(S, -1).T.astype(float)                  # cells x species
        n = X.sum(1); idx = np.flatnonzero(n > 0)
        g = int((o.reshape(S, -1).sum(1) > 0).sum())
        if g == 0 or len(idx) == 0:
            continue
        alphas.append(n[idx]); gam.append(g); abar.append(n[idx].mean())
        Xi = X[idx]; inter = Xi @ Xi.T; sor = 2.0 * inter / (n[idx][:, None] + n[idx][None, :])
        yy, xx = np.divmod(idx, G1)
        dy, dx = np.abs(yy[:, None] - yy[None, :]), np.abs(xx[:, None] - xx[None, :])
        if wrap:
            dy, dx = np.minimum(dy, G0 - dy), np.minimum(dx, G1 - dx)
        d = np.maximum(dy, dx); up = np.triu(np.ones_like(d, dtype=bool), 1)
        for j, (lo, hi, _) in enumerate(SOR_DIST):
            m = up & (d >= lo) & (d <= hi)
            sor_sum[j] += sor[m].sum(); sor_n[j] += m.sum()
    if not gam:
        return dict(alpha_med=np.nan, alpha_mean=np.nan, alpha_gm=np.nan, beta_gm=np.nan, gamma_gm=np.nan,
                    sor=np.full(len(SOR_DIST), np.nan))
    a = np.concatenate(alphas); gam, abar = np.array(gam, float), np.array(abar, float)
    return dict(alpha_med=float(np.median(a)), alpha_mean=float(a.mean()),
                alpha_gm=float(np.exp(np.log(abar).mean())), beta_gm=float(np.exp(np.log(gam / abar).mean())),
                gamma_gm=float(np.exp(np.log(gam).mean())),
                sor=np.where(sor_n > 0, sor_sum / np.maximum(sor_n, 1), np.nan))


def load_bnm(pattern):
    files = bnm_glob(pattern); cases = []   # obj1_v2 fit-half
    for f in files:
        z = np.load(f, allow_pickle=True)
        cases.append((z["obs_mask"].astype(bool), z["known_target"].astype(bool)))
    return files, cases


def survey_masks(pattern, n=0):
    """known_target stacks (species, G, G) of the BNM windows; n > 0 takes n windows evenly spaced."""
    files = bnm_glob(pattern)   # obj1_v2 fit-half
    if n and n < len(files): files = [files[i] for i in np.linspace(0, len(files) - 1, n).round().astype(int)]
    return [np.load(f, allow_pickle=True)["known_target"].astype(bool) for f in files]


def observe(P, masks, seed=0):
    """A simulated world seen through each survey window: every simulated species gets the
    known cells of a BNM species drawn at random from that window. (records, known) per window."""
    if not masks: return [(P, np.ones_like(P))]
    rng = np.random.default_rng(seed); out = []
    for m in masks:
        K = m[rng.integers(len(m), size=len(P))]
        out.append((P & K, K))
    return out


def effort_report(masks):
    """How much the known cells of one window differ between species (read from the BNM files)."""
    differ = [bool((m != m[:1]).any()) for m in masks]
    per_sp = np.concatenate([m.reshape(len(m), -1).sum(1) for m in masks])
    union = [int(m.any(0).sum()) for m in masks]
    print(f"\nvirtual survey: {len(masks)} BNM windows; known cells differ between species in {np.mean(differ):.0%} of windows; "
          f"known cells per species, median {np.median(per_sp):.0f} (quartiles {np.quantile(per_sp, 0.25):.0f}-{np.quantile(per_sp, 0.75):.0f}); "
          f"cells known for any species, median {np.median(union):.0f} of {masks[0].shape[1] * masks[0].shape[2]}")


def presence(z, tau):   # obj1_v2 tau
    """Presence of every species in every cell for observation: P_last_final at tau 1 (as before); above 1, cells
    with tau or more individuals in the last frame (B_last / BODY_MASS, rounded). None when B_last is missing."""
    if tau == 1:
        return z["P_last_final"].astype(bool)
    if "B_last" not in z.files:
        return None
    return np.rint(np.asarray(z["B_last"], dtype=np.float64) / float(z["BODY_MASS"])) >= tau


def load_sim(pattern, masks=None, each=False):
    """Distinct simulated worlds; with each=True one list of cases per file."""
    files, seen, groups = sorted(glob.glob(pattern)), set(), []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "P_last_final" not in z.files: continue
        P = presence(z, TAU)   # obj1_v2 tau
        if P is None:
            print(f"\n{f.split('/')[-1]}: no B_last, so no presence at tau {TAU:g}; skipped"); continue
        h = hashlib.md5(P.tobytes()).hexdigest()
        if h in seen: continue
        seen.add(h); groups.append((f, observe(P, masks or [])))
    if each: return groups
    return files, [c for _, cs in groups for c in cs]


def score(s, b):
    """Distance of a simulated summary s from the BNM summary b, and pass flags."""
    co = s["ratio_sparse"][:4] / b["ratio_sparse"][:4]
    tv = 0.5 * float(np.nansum(np.abs(np.array(s["bins"]) - np.array(b["bins"]))))
    rr = s["rich_med"] / b["rich_med"]
    ok = dict(cooc=bool(np.all((co >= 1 / 1.3) & (co <= 1.3))), ranges=tv <= 0.15, species=0.5 <= rr <= 2)
    return co, tv, rr, ok


def summarise(cases, wrap):
    R = np.concatenate([o.reshape(len(o), -1).sum(1) for o, _ in cases]); R = R[R > 0]
    rich = [int((o.reshape(len(o), -1).sum(1) > 0).sum()) for o, _ in cases]
    pt = [patches(o[s], wrap) for o, _ in cases for s in range(len(o)) if 6 <= o[s].sum() <= 20]
    ob, ex = np.zeros(len(DIST)), np.zeros(len(DIST)); ob2, ex2 = np.zeros(len(DIST)), np.zeros(len(DIST))
    for o, k in cases:
        a, b = cooc(o, k, wrap); ob += a; ex += b
        a, b = cooc(o, k, wrap, rmax=20); ob2 += a; ex2 += b
    ab = alpha_beta(cases, wrap)   # obj1_v2 alpha
    return dict(n_units=len(cases), n_cases=len(R), n_sparse=int(((R > 0) & (R <= 20)).sum()),
                R_med=float(np.median(R)) if len(R) else np.nan,
                bins=[float(np.mean((R >= lo) & (R <= hi))) if len(R) else np.nan for lo, hi, _ in BINS],
                rich_med=float(np.median(rich)) if rich else np.nan,
                patch_med=float(np.median(pt)) if pt else np.nan, patch_multi=float(np.mean(np.array(pt) > 1)) if pt else np.nan,
                ratio=np.where(ex > 0, ob / np.maximum(ex, 1e-12), np.nan),
                ratio_sparse=np.where(ex2 > 0, ob2 / np.maximum(ex2, 1e-12), np.nan), **ab)   # obj1_v2 alpha


BANDS = dict(species=0.207, alpha=0.355, beta=0.308, decay=0.045, tv=0.116, r15=0.081, r51=0.048,   # obj1_v2 bands
             co=(0.196, 0.232, 0.137, 0.077))


def band_values(s, b):
    """The numbers judged against BNM summary b."""
    co, tv, rr, _ = score(s, b)
    return dict(species=s['rich_med'], alpha=s['alpha_gm'], beta=s['beta_gm'], decay=float(s['sor'][0] - s['sor'][3]),
                tv=tv, r15=s['bins'][0] + s['bins'][1], r51=s['bins'][5], co=np.asarray(co, float))


def band_check(v, b):
    """PASS flags of the values v (band_values) against BNM summary b."""
    rel = lambda x, t, w: bool(abs(x / t - 1) <= w)
    return dict(species=rel(v['species'], b['rich_med'], BANDS['species']), alpha=rel(v['alpha'], b['alpha_gm'], BANDS['alpha']),
                beta=rel(v['beta'], b['beta_gm'], BANDS['beta']),
                decay=bool(abs(v['decay'] - (b['sor'][0] - b['sor'][3])) <= BANDS['decay']), tv=bool(v['tv'] <= BANDS['tv']),
                r15=bool(abs(v['r15'] - (b['bins'][0] + b['bins'][1])) <= BANDS['r15']),
                r51=bool(abs(v['r51'] - b['bins'][5]) <= BANDS['r51']),
                co=bool(np.all(np.abs(v['co'] - 1) <= np.array(BANDS['co']))))


BAND_NAMES = dict(species='species per window', alpha='species per cell', beta='turnover', decay='similarity decay',
                  tv='range-size TV', r15='1-5 cell share', r51='51+ share', co='rare clumping 1-4 cells')


def setting_key(f):   # obj1_v2 setting
    """Worlds of one setting share mode, trade-off, contrast, habitat scale, connectance, founding, kernel and window
    (d and seeds may differ). Files saved before these keys existed: habitat scale 3, connectance 0.4."""
    z = np.load(f, allow_pickle=True)
    g = lambda k, d: (z[k].item() if z[k].ndim == 0 else str(z[k][0])) if k in z.files else d
    mode = str(g('ENV_mode', 'field'))
    env = (f"k {float(g('NICHE_tradeoff', 1.0)):g}, contrast {float(g('HABITAT_contrast', 1.0)):g}, "
           f"habitat scale {float(g('HABITAT_ls', 3.0)):g}" if mode == 'niche'
           else f"field var {float(g('ENV_var_r', -1)):g}")
    dps = (f", d per species {float(g('DISP_d_min', 0)):g}-{float(g('DISP_d_max', 0)):g}"   # obj1_v2 per-species d
           if int(g('DISP_per_species', 0)) else "")
    fnd = (f"best cell, cap {int(g('FOUND_cap', 100))}" if str(g('FOUND_rule', 'random')) == 'best'   # obj1_v2 founding
           else f"in {g('seed_size', '?')}")
    st = float(g('INTERACTION_STRENGTH', 0.4))   # obj1_v2 strength
    st = f", strength {st:g}" if abs(st - 0.4) > 1e-6 else ""
    return (f"{mode}, {env}, connectance {float(g('CONNECTANCE', 0.4)):g}{st}{dps}, founded {fnd}, "
            f"kernel {g('DISPERSAL_KERNEL', '?')}, window {g('window_steps', '?')}")


def setting_verdict(rows, b):
    """rows: list of band_values of the worlds of one setting. PASS per number: median inside and 75% of worlds inside."""
    need = int(np.ceil(0.75 * len(rows))); out = {}
    for k in BANDS:
        med = dict(rows[0]); med[k] = np.median(np.array([r[k] for r in rows]), axis=0)
        n_in = sum(band_check(r, b)[k] for r in rows)
        out[k] = (band_check(med, b)[k] and n_in >= need, n_in, med[k])
    return out


def report(name, s):
    print(f"\n{name}: {s['n_units']} windows/communities, {s['n_cases']} cases with an occupied cell")
    print(f"  range size, median {s['R_med']:.0f} cells; share of cases: " +
          ", ".join(f"{lab} {v:.1%}" for (_, _, lab), v in zip(BINS, s["bins"])))
    print(f"  species present per window/community, median {s['rich_med']:.0f}")
    print(f"  patches for cases of 6-20 cells: median {s['patch_med']:.1f}, share with more than one patch {s['patch_multi']:.1%}")
    print("  co-occupancy ratio by distance, all cases (1 = random): " +
          ", ".join(f"{lab} {v:.2f}" for (_, _, lab), v in zip(DIST, s["ratio"])))
    print(f"  co-occupancy ratio by distance, cases of 20 cells or fewer (the rare-species target; {s['n_sparse']} cases): " +
          ", ".join(f"{lab} {v:.2f}" for (_, _, lab), v in zip(DIST, s["ratio_sparse"])))
    print(f"  species per recorded cell (alpha): median {s['alpha_med']:.0f}, mean {s['alpha_mean']:.1f}; "   # obj1_v2 alpha
          f"geometric means over windows: gamma {s['gamma_gm']:.1f} = alpha {s['alpha_gm']:.1f} x turnover (beta) {s['beta_gm']:.2f}")
    print("  similarity of two recorded cells (Sorensen) by distance: " +
          ", ".join(f"{lab} {v:.2f}" for (_, _, lab), v in zip(SOR_DIST, s["sor"])))


def self_test():
    rng = np.random.default_rng(0); G = 20; ok = True
    occ = rng.random((6, G, G)) < 0.05; known = rng.random((6, G, G)) < 0.7; occ &= known
    for wrap in (True, False):
        ob, ex = cooc(occ, known, wrap)
        bo, be = np.zeros(len(DIST)), np.zeros(len(DIST))
        yy, xx = np.divmod(np.arange(G * G), G)
        for s in range(6):
            o, k = occ[s].ravel(), known[s].ravel()
            R, K = o.sum(), k.sum()
            if not (R > 0 and K >= 4 and R < K): continue
            dy, dx = np.abs(yy[:, None] - yy[None, :]), np.abs(xx[:, None] - xx[None, :])
            if wrap: dy, dx = np.minimum(dy, G - dy), np.minimum(dx, G - dx)
            d = np.maximum(dy, dx); np.fill_diagonal(d, 0)
            for j, (lo, hi, _) in enumerate(DIST):
                g = (d >= lo) & (d <= hi)
                bo[j] += (o[:, None] & o[None, :] & g).sum(); be[j] += R * (R - 1) / (K * (K - 1)) * (k[:, None] & k[None, :] & g).sum()
        good = np.allclose(ob, bo) and np.allclose(ex, be); ok &= good
        print(f"self-test, pair counts against a direct loop over all cell pairs ({'torus' if wrap else 'window'}): {'PASS' if good else 'FAIL'}")
    rnd = np.zeros((400, G * G), bool)
    for s in range(400): rnd[s, rng.choice(G * G, int(rng.integers(2, 30)), replace=False)] = True
    ob, ex = cooc(rnd.reshape(400, G, G), np.ones((400, G, G), bool), True)
    r = ob / ex; good = np.all(np.abs(r - 1) < 0.05); ok &= good
    print(f"self-test, random placement gives ratio about 1 at every distance: {np.round(r, 3).tolist()} {'PASS' if good else 'FAIL'}")
    blob = np.zeros((200, G, G), bool)
    for s in range(200):
        y, x = rng.integers(G, size=2); blob[s, y:y + 3, x:x + 3] = True
    ob, ex = cooc(blob, np.ones_like(blob), True); r = ob / ex
    good = r[0] > 5 and r[-1] < 0.5; ok &= good
    print(f"self-test, 3 x 3 blocks are clustered at 1 cell and absent far away: {np.round(r, 2).tolist()} {'PASS' if good else 'FAIL'}")
    t = np.zeros((G, G), bool); t[0, 5] = t[G - 1, 5] = t[10, 10] = True
    good = patches(t, True) == 2 and patches(t, False) == 3; ok &= good
    print(f"self-test, patches across the torus edge join, inside a window they do not: {'PASS' if good else 'FAIL'}")
    P = rng.random((80, G, G)) < 0.04
    full = summarise(observe(P, [np.ones((30, G, G), bool)]), wrap=True); none = summarise(observe(P, []), wrap=True)
    good = np.allclose(full["ratio_sparse"], none["ratio_sparse"], equal_nan=True) and full["bins"] == none["bins"]; ok &= good
    print(f"self-test, a survey with every cell known changes nothing: {'PASS' if good else 'FAIL'}")
    half = np.zeros((30, G, G), bool); half[:, :, :10] = True
    obs = observe(P, [half])[0][0]
    good = not obs[:, :, 10:].any() and np.array_equal(obs[:, :, :10], P[:, :, :10]); ok &= good
    print(f"self-test, a half-grid survey keeps records only in the surveyed half: {'PASS' if good else 'FAIL'}")
    stack = np.zeros((3, G, G), bool); stack[0, :5] = True; stack[1, 5:12] = True; stack[2, 12:] = True   # three species, three effort bands
    (o1, k1), = observe(P, [stack]); (o2, k2), = observe(P, [stack])
    rows = [next(j for j in range(3) if np.array_equal(k1[i], stack[j])) for i in range(len(P))]
    good = (np.array_equal(o1, P & k1) and np.array_equal(k1, k2) and np.array_equal(o1, o2) and len(set(rows)) == 3
            and all(abs(np.mean(np.array(rows) == j) - 1 / 3) < 0.15 for j in range(3))); ok &= good
    print(f"self-test, each species gets one BNM species' known cells, drawn evenly and the same on every call, "
          f"records only inside them: {'PASS' if good else 'FAIL'}")
    uni = np.ones((G, G), bool)
    lo = summarise([(P & uni[None], np.broadcast_to(uni, P.shape))], wrap=False)
    hi = summarise(observe(P, [np.ones((30, G, G), bool)]), wrap=False)
    good = np.allclose(lo["ratio_sparse"], hi["ratio_sparse"], equal_nan=True); ok &= good
    print(f"self-test, per-species masks that are all equal give the same result as one shared mask: {'PASS' if good else 'FAIL'}")
    yy, xx = np.mgrid[0:G, 0:G]; clump = []
    for w in range(8):                                    # clumped effort: known cells in a few discs, as around recorders
        m = np.zeros((40, G, G), bool)
        for sp in range(40):
            for _ in range(3):
                cy, cx = rng.integers(G, size=2); m[sp] |= (yy - cy) ** 2 + (xx - cx) ** 2 <= int(rng.integers(4, 26))
        clump.append(m)
    rnd = np.zeros((300, G * G), bool)
    for sp in range(300): rnd[sp, rng.choice(G * G, int(rng.integers(5, 60)), replace=False)] = True
    r = summarise(observe(rnd.reshape(300, G, G), clump), wrap=False)["ratio_sparse"][:4]
    good = bool(np.all(np.abs(r - 1) < 0.1)); ok &= good
    print(f"self-test, random truth seen through clumped effort gives ratio about 1 at 1-4 cells (the survey adds no "
          f"clustering): {np.round(r, 3).tolist()} {'PASS' if good else 'FAIL'}")
    full = np.ones((5, G, G), bool)   # obj1_v2 alpha: alpha-beta known answers
    ab = alpha_beta([(full, full)], wrap=False)
    good = ab['alpha_med'] == 5 and abs(ab['beta_gm'] - 1) < 1e-12 and np.allclose(ab['sor'], 1); ok &= good
    print(f"self-test, 5 species in every cell: alpha 5, turnover 1, similarity 1 at every distance: {'PASS' if good else 'FAIL'}")
    quad = np.zeros((4, G, G), bool); quad[0, :10, :10] = quad[1, :10, 10:] = quad[2, 10:, :10] = quad[3, 10:, 10:] = True
    ab = alpha_beta([(quad, quad)], wrap=False)
    good = ab['alpha_med'] == 1 and abs(ab['gamma_gm'] - 4) < 1e-12 and abs(ab['beta_gm'] - 4) < 1e-12; ok &= good
    print(f"self-test, 4 species in 4 quadrants: alpha 1, gamma 4, turnover 4: {'PASS' if good else 'FAIL'}")
    rw = [(rng.random((8, G, G)) < 0.1, None) for _ in range(2)]
    for wrap in (False, True):
        ab = alpha_beta(rw, wrap)
        num, den = np.zeros(len(SOR_DIST)), np.zeros(len(SOR_DIST))
        for o, _ in rw:
            cells = [(y, x) for y in range(G) for x in range(G) if o[:, y, x].any()]
            for a_ in range(len(cells)):
                for b_ in range(a_ + 1, len(cells)):
                    (y1, x1), (y2, x2) = cells[a_], cells[b_]
                    dy, dx = abs(y1 - y2), abs(x1 - x2)
                    if wrap: dy, dx = min(dy, G - dy), min(dx, G - dx)
                    dd = max(dy, dx); A, B = o[:, y1, x1], o[:, y2, x2]
                    for j, (lo, hi, _) in enumerate(SOR_DIST):
                        if lo <= dd <= hi: num[j] += 2 * (A & B).sum() / (A.sum() + B.sum()); den[j] += 1
        good = np.allclose(ab['sor'], num / den); ok &= good
        print(f"self-test, Sorensen by distance against a direct loop over cell pairs ({'torus' if wrap else 'window'}): {'PASS' if good else 'FAIL'}")
    good = abs(ab['gamma_gm'] - ab['alpha_gm'] * ab['beta_gm']) < 1e-9 * ab['gamma_gm']; ok &= good
    print(f"self-test, gamma = alpha x turnover for the geometric means: {'PASS' if good else 'FAIL'}")
    Pz = rng.random((40, G, G)) < 0.1
    a1 = alpha_beta(observe(Pz, [np.ones((30, G, G), bool)]), wrap=False); a2 = alpha_beta([(Pz, np.ones_like(Pz))], wrap=False)
    good = a1['alpha_med'] == a2['alpha_med'] and np.allclose(a1['sor'], a2['sor']) and a1['beta_gm'] == a2['beta_gm']; ok &= good
    print(f"self-test, a survey with every cell known leaves alpha, turnover and similarity unchanged: {'PASS' if good else 'FAIL'}")
    b0 = summarise(observe(rng.random((60, G, G)) < rng.uniform(0.01, 0.12, (60, 1, 1)), [np.ones((30, G, G), bool)]), wrap=False)   # obj1_v2 bands: bands
    v0 = band_values(b0, b0); ck = band_check(v0, b0)
    good = all(ck.values()); ok &= good
    print(f"self-test, BNM against itself passes every band: {'PASS' if good else 'FAIL'}")
    v2 = dict(v0); v2['species'] = 2 * v0['species']; v3 = dict(v0); v3['r51'] = v0['r51'] + 0.06; v4 = dict(v0); v4['co'] = v0['co'] * 1.25
    v5 = dict(v0); v5['co'] = v0['co'] * 0.7
    c2, c3, c4, c5 = band_check(v2, b0), band_check(v3, b0), band_check(v4, b0), band_check(v5, b0)
    good = all(not c[k] and sum(c.values()) == 7 for c, k in ((c2, 'species'), (c3, 'r51'), (c4, 'co'), (c5, 'co'))); ok &= good
    print(f"self-test, twice the species, 6 points more 51+ cases, rare clumping x1.25 or x0.7: each fails its own band only: {'PASS' if good else 'FAIL'}")
    sv3 = setting_verdict([v0, v0, v0, v2], b0); sv2 = setting_verdict([v0, v0, v2, v2], b0)
    good = sv3['species'][0] and sv3['species'][1] == 3 and not sv2['species'][0] and all(x[0] for k, x in sv2.items() if k != 'species'); ok &= good
    print(f"self-test, a setting passes with 3 of 4 worlds inside and fails with 2 of 4: {'PASS' if good else 'FAIL'}")
    import tempfile as _tf   # obj1_v2 setting: scale and connectance split settings; missing keys = defaults
    with _tf.TemporaryDirectory() as tmp:
        base = dict(ENV_mode=np.array(['niche']), NICHE_tradeoff=np.float32(0.5), HABITAT_contrast=np.float32(0.25),
                    seed_size=np.int64(1), DISPERSAL_KERNEL=np.array(['2dt']), window_steps=np.int64(5000))
        sets = dict(a={}, b=dict(HABITAT_ls=np.float32(3.0), CONNECTANCE=np.float32(0.4)), c=dict(CONNECTANCE=np.float32(1.0)),
                    d=dict(HABITAT_ls=np.float32(1.0)), e=dict(HABITAT_ls=np.float32(1.0), CONNECTANCE=np.float32(1.0)))
        key = {}
        for n, extra in sets.items():
            np.savez(f"{tmp}/{n}.npz", **base, **extra); key[n] = setting_key(f"{tmp}/{n}.npz")
    good = (key['a'] == key['b'] and len({key[n] for n in 'acde'}) == 4 and 'habitat scale 3, connectance 0.4' in key['a']
            and 'habitat scale 1, connectance 1,' in key['e']); ok &= good
    print(f"self-test, habitat scale and connectance give their own settings; files without them = scale 3, "
          f"connectance 0.4: {'PASS' if good else 'FAIL'}")
    with _tf.TemporaryDirectory() as tmp:   # obj1_v2 per-species d: per-species d splits settings; shared d keeps its label
        for n, extra in dict(a={}, b=dict(DISP_per_species=np.int8(0)),
                             c=dict(DISP_per_species=np.int8(1), DISP_d_min=np.float64(1e-6), DISP_d_max=np.float64(1e-2))).items():
            np.savez(f"{tmp}/{n}.npz", **base, **extra); key[n] = setting_key(f"{tmp}/{n}.npz")
    good = (key['a'] == key['b'] and 'per species' not in key['a'] and key['c'] != key['a']
            and 'connectance 0.4, d per species 1e-06-0.01, founded' in key['c']); ok &= good
    print(f"self-test, per-species d gives its own setting; shared d (0 or not saved) keeps its label: {'PASS' if good else 'FAIL'}")
    with _tf.TemporaryDirectory() as tmp:   # obj1_v2 founding: best founding splits settings; random (or not saved) keeps its label
        for n, extra in dict(a={}, b=dict(FOUND_rule=np.array(['random'])),
                             c=dict(FOUND_rule=np.array(['best']), FOUND_cap=np.int64(100))).items():
            np.savez(f"{tmp}/{n}.npz", **base, **extra); key[n] = setting_key(f"{tmp}/{n}.npz")
    good = (key['a'] == key['b'] and 'founded in 1, kernel' in key['a'] and key['c'] != key['a']
            and 'founded best cell, cap 100, kernel' in key['c']); ok &= good
    print(f"self-test, best founding gives its own setting; random (or not saved) keeps its label: {'PASS' if good else 'FAIL'}")
    with _tf.TemporaryDirectory() as tmp:   # obj1_v2 strength: a strength other than 0.4 gives its own setting
        b0 = {k: v for k, v in base.items() if k != 'INTERACTION_STRENGTH'}
        for n, extra in dict(s0={}, s4=dict(INTERACTION_STRENGTH=np.float32(0.4)), s6=dict(INTERACTION_STRENGTH=np.float32(0.6)),
                             s7=dict(INTERACTION_STRENGTH=np.float32(0.7))).items():
            np.savez(f"{tmp}/{n}.npz", **b0, **extra); key[n] = setting_key(f"{tmp}/{n}.npz")
        good = (key['s0'] == key['s4'] and 'strength' not in key['s0'] and len({key['s0'], key['s6'], key['s7']}) == 3
                and ', strength 0.6,' in key['s6'] and ', strength 0.7,' in key['s7']); ok &= good
    print(f"self-test, competition strength other than 0.4 gives its own setting; 0.4 (or not saved) keeps its label: "
          f"{'PASS' if good else 'FAIL'}")
    import tempfile as _tt   # obj1_v2 tau: presence at tau, known answers
    with _tt.TemporaryDirectory() as tmp:
        Nn = np.array([0, 1, 9, 10, 99, 100, 999, 1000, 5000, 15000, 0, 2], float).reshape(3, 2, 2)
        np.savez(f"{tmp}/w.npz", P_last_final=(Nn >= 1).astype(np.uint8), B_last=(Nn * 1e-4).astype(np.float16),
                 BODY_MASS=np.float32(1e-4))
        np.savez(f"{tmp}/x.npz", P_last_final=(Nn >= 1).astype(np.uint8), BODY_MASS=np.float32(1e-4))
        zw, zx = np.load(f"{tmp}/w.npz"), np.load(f"{tmp}/x.npz")
        good = (np.array_equal(presence(zw, 1), Nn >= 1) and all(np.array_equal(presence(zw, t), Nn >= t) for t in (2, 10, 100, 1000))
                and presence(zx, 10) is None and np.array_equal(presence(zx, 1), Nn >= 1)); ok &= good
    print(f"self-test, presence at tau: tau 1 = P_last_final; tau 2, 10, 100, 1,000 split cells at tau exactly (float16); "
          f"no B_last -> skipped: {'PASS' if good else 'FAIL'}")
    import tempfile                                   # obj1_v2 fit-half: the split and that test windows stay unopened
    global REF
    with tempfile.TemporaryDirectory() as tmp:
        win = [(400, 200), (440, 200), (440, 240), (480, 240), (520, 240)]
        for e, n in win:
            o = np.zeros((3, G, G), bool)
            if e >= 480: o[:] = True                  # test half: every cell occupied (would show if read)
            else: o[:, 0, 0] = True                   # fit half: one cell per species
            np.savez(f"{tmp}/bnm_e{e:04d}_n{n:04d}.npz", obs_mask=o, known_target=np.ones_like(o), window_cells=G, cell_m=2000)
        REF = split_ref(f"{tmp}/bnm_*.npz")
        eas = sorted({e for e, _ in win}); cut19 = eas[len(eas) // 2]       # step 19's split_fit rule, written out
        want = sorted(f"bnm_e{e:04d}_n{n:04d}.npz" for e, n in win if e < cut19)
        keep, east, ov = fit_half(glob.glob(f"{tmp}/bnm_*.npz"), REF)
        good = sorted(Path(f).name for f in keep) == want and east == 2 and ov == 0 and REF["side"] == 40.0; ok &= good
        print(f"self-test, fit half = step 19's rule (easting below the median), test half left out: {'PASS' if good else 'FAIL'}")
        other = [f"x/bnm_e{e:04d}_n{n:04d}.npz" for e, n in ((470, 240), (300, 100), (500, 300))]
        keep, east, ov = fit_half(other, REF)
        good = [Path(f).name for f in keep] == ["bnm_e0300_n0100.npz"] and east == 1 and ov == 1; ok &= good
        print(f"self-test, a window from another folder that shares cells with a test window is refused: {'PASS' if good else 'FAIL'}")
        files, cases = load_bnm(f"{tmp}/bnm_*.npz")
        s = summarise(cases, wrap=False)
        good = len(files) == 3 and s["R_med"] == 1 and len(survey_masks(f"{tmp}/bnm_*.npz")) == 3; ok &= good
        print(f"self-test, --bnm and --survey read fit-half windows only (median range {s['R_med']:.0f}; 400 if a "
              f"test window were read): {'PASS' if good else 'FAIL'}")
    REF = None
    return ok


ap = argparse.ArgumentParser()
ap.add_argument("--bnm"); ap.add_argument("--sim", action="append", default=[])
ap.add_argument("--survey"); ap.add_argument("--n-masks", type=int, default=0)
ap.add_argument("--each", action="store_true")
ap.add_argument("--split-ref", default=REF_DEFAULT, help="BNM windows that set the geographic split (the Objective 3/4 scoring set)")   # obj1_v2 fit-half
ap.add_argument("--tau", type=float, default=1.0,   # obj1_v2 tau
                help="individuals a simulated species needs in a cell to count as present (observation only; default 1)")
ap.add_argument("--self-test", action="store_true")
a = ap.parse_args()
if a.tau < 1:   # obj1_v2 tau
    sys.exit("STOP: --tau is a number of individuals, 1 or more")
TAU = float(a.tau)
if TAU != 1 and not a.self_test:
    print(f"presence for observation: {TAU:g} or more individuals in a cell (the assembly itself used 1)")
if a.self_test:
    sys.exit(0 if self_test() else 1)
REF = split_ref(a.split_ref) if (a.bnm or a.survey) else None   # obj1_v2 fit-half
target = None
if a.bnm:
    files, cases = load_bnm(a.bnm)
    if not cases: sys.exit(f"no BNM files matched {a.bnm}")
    target = summarise(cases, wrap=False)
    report(f"BNM, geographic fit half ({len(files)} windows)", target)
masks = survey_masks(a.survey, a.n_masks) if a.survey else []
if a.survey:
    if not masks: sys.exit(f"no BNM files matched {a.survey}")
    effort_report(masks)
rows = {}   # obj1_v2 bands: band values per setting
for pat in a.sim:
    if a.each:
        groups = load_sim(pat, masks, each=True)
        if not groups: print(f"\nno simulated files with P_last_final matched {pat}"); continue
        for f, cases in groups:
            s = summarise(cases, wrap=not a.survey)
            report(f"simulated: {f.split('/')[-1]}", s)
            if target is not None:
                co, tv, rr, ok = score(s, target)
                v = band_values(s, target); ck = band_check(v, target); rows.setdefault(setting_key(f), []).append(v)   # obj1_v2 bands
                print(f"  against BNM: rare co-occupancy / BNM at 1-4 cells {np.round(co, 2).tolist()}; range-size distance {tv:.3f}; "
                      f"species per window x{rr:.2f}; similarity decay {v['decay']:.2f} (BNM {target['sor'][0] - target['sor'][3]:.2f})")
                print("  bands (28 Oct): " + ", ".join(f"{BAND_NAMES[k]} {'PASS' if ck[k] else 'FAIL'}" for k in BANDS)
                      + f"  ->  {sum(ck.values())} of {len(ck)} pass")
                ga, aa, bb = s['gamma_gm'] / target['gamma_gm'], s['alpha_gm'] / target['alpha_gm'], s['beta_gm'] / target['beta_gm']   # obj1_v2 alpha
                why = ('no species excess' if ga <= 1 else 'mostly per cell: local competition' if aa > bb
                       else 'mostly turnover between cells: habitat or dispersal')
                print(f"  species against BNM (geometric means) x{ga:.2f} = per cell x{aa:.2f} x turnover x{bb:.2f} ({why})")
    else:
        files, cases = load_sim(pat, masks)
        if not cases: print(f"\nno simulated files with P_last_final matched {pat}"); continue
        report(f"simulated: {pat}", summarise(cases, wrap=not a.survey))
if rows and target is not None:   # obj1_v2 bands
    print("\nsettings against the 28 Oct bands (PASS: median inside and at least 75% of worlds inside)")
    for key, rs in rows.items():
        sv = setting_verdict(rs, target)
        print(f"  {key}: {len(rs)} worlds -> {'PASSES' if all(x[0] for x in sv.values()) else 'fails'} "
              f"({sum(x[0] for x in sv.values())} of {len(sv)} numbers)")
        print("    " + "; ".join(f"{BAND_NAMES[k]} {'PASS' if x[0] else 'FAIL'} ({x[1]}/{len(rs)})" for k, x in sv.items()))