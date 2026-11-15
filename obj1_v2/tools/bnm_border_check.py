#!/usr/bin/env python3
"""
bnm_border_check.py  (obj1_v2, 30 Oct 2026)

Question: are the small ranges in BNM windows (1-5 recorded cells) mostly places where the
40 km window cuts through a larger range? If yes, our worlds (the whole 20 x 20 torus, nothing
outside it) cannot make them, and "world = window" (tests README, sketch 37) is the next change.

BNM side: geographic fit half only. Test-half windows and windows sharing cells with them are
never opened (the rule of world_vs_bnm.py v3 and later, copied here unchanged).
  E1  border ratio, per range-size bin (1-5, 6-20, 21-50, 51+ recorded cells): recorded cells in
      the outer ring of the window (1, 2 or 3 cells wide) divided by the number expected if the
      case's cells sat at random among its own known cells (known_target). 1 = no edge effect.
      95% interval from 2,000 bootstrap resamples of windows.
  E2  next door: for each case, the most cells the same species (matched by name) holds in any
      of the 8 neighbouring fit-half windows. Share of cases where that is 21 or more (common
      there) and 1 or more (present). Cases with no fit-half neighbour are counted apart.
  E3  E1 (ring 2) for the 1-5 cell cases, split by E2: common next door, or not.
Control: simulated worlds seen through BNM effort, as world_vs_bnm.py sees them (each species
  gets the known cells of a BNM species of a fit-half window, seed 0); presence = P_last_final
  (1 individual). A torus has no border, so E1 should be about 1 in every bin. A simulated
  species is seen through every window, so its cases are not independent: the simulated
  interval resamples species, the BNM interval resamples windows.

Reading rule (written before the run, 30 Oct, Sara's yes):
  Control first: BNM 51+ cases must give E1 (ring 2) within 0.8-1.25, and the pooled simulated
     E1 (ring 2) must hold 1 inside its 95% interval in every bin with cases. If not, the
     measure is biased and nothing below is read.
  (a) E1 for 1-5 cell cases 1.3 or more, interval above 1, and E2 common 50% or more:
      small ranges are mostly window cuts -> build world = window.
  (b) E1 below 1.3 and E2 common 50% or more: small cases are mostly species common nearby
      that hold few recorded cells here, away from the edge (patchy occupancy or missed
      detections) -> world = window is not the main fix; look at detection and patchiness.
  (c) E2 common below 50%: small cases are mostly species rare in the region too -> true
      rarity; the fix is in how rare species arise in the assembly.

Run from obj1_v2/work:
  python3 ../tools/bnm_border_check.py --self-test | tee ../logs/bnm_border_check_selftest_30oct.txt
  python3 ../tools/bnm_border_check.py \\
      --bnm "../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz" \\
      --sim "results/stage2a/niche_d/*_training.npz" | tee ../logs/bnm_border_check_30oct.txt
Reads files only; writes nothing.
"""
import argparse, glob, hashlib, sys, tempfile
from pathlib import Path
import numpy as np

REF_DEFAULT = "../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz"
BINS = [(1, 5, "1-5"), (6, 20, "6-20"), (21, 50, "21-50"), (51, 10 ** 9, "51+")]
RINGS = (1, 2, 3)
RULE_RING = 2
COMMON = 21
N_BOOT = 2000
OPENED = []                                  # every file opened, for the leak self-test


def load(f):
    OPENED.append(str(Path(f).name))
    return np.load(f, allow_pickle=True)


# ---------------------------------------------------------------- fit half (as world_vs_bnm.py v3)
def window_en(f):
    """Window origin (easting, northing) in km from step 10's file name bnm_e<km>_n<km>."""
    s = Path(f).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])


def split_ref(pattern):
    """fit = easting below the median of the distinct eastings of the reference set."""
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


def fit_half(files, ref):
    """Windows that may be read: west of the cut and sharing no cell with a test window (names only)."""
    keep, east, overlap = [], 0, 0
    for f in sorted(files):
        e, n = window_en(f)
        if e >= ref["cut"]:
            east += 1
        elif any(abs(e - te) < ref["side"] and abs(n - tn) < ref["side"] for te, tn in ref["test"]):
            overlap += 1
        else:
            keep.append(f)
    return keep, east, overlap


# ---------------------------------------------------------------- measures
def ring_mask(G, w):
    m = np.zeros((G, G), bool)
    m[:w] = m[-w:] = True
    m[:, :w] = m[:, -w:] = True
    return m


def bin_index(R):
    out = np.full(len(R), -1)
    for j, (lo, hi, _) in enumerate(BINS):
        out[(R >= lo) & (R <= hi)] = j
    return out


def border_cases(obs, known, ring):
    """Per case with an occupied cell: R, recorded cells in the ring, and the number expected if
    the R cells sat at random among the case's known cells: R x (known in ring) / known."""
    R = obs.reshape(len(obs), -1).sum(1)
    K = known.reshape(len(known), -1).sum(1)
    if np.any((R > 0) & (K == 0)) or np.any(obs & ~known):
        raise ValueError("records outside the known cells")
    ob = (obs & ring).reshape(len(obs), -1).sum(1)
    kb = (known & ring).reshape(len(known), -1).sum(1)
    sel = R > 0
    return np.flatnonzero(sel), R[sel], ob[sel].astype(float), (R[sel] * kb[sel] / K[sel]).astype(float)


def boot_ratio(O, E, seed=1):
    """Pooled ratio sum O / sum E per column and a 95% interval from resampling units (rows)."""
    O, E = np.asarray(O, float), np.asarray(E, float)
    tot_e = E.sum(0)
    r = np.where(tot_e > 0, O.sum(0) / np.where(tot_e > 0, tot_e, 1), np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(O), size=(N_BOOT, len(O)))
    bo, be = O[idx].sum(1), E[idx].sum(1)
    br = np.where(be > 0, bo / np.where(be > 0, be, 1), np.nan)
    with np.errstate(all="ignore"):
        lo = np.array([np.nanquantile(br[:, j], 0.025) if np.isfinite(br[:, j]).any() else np.nan for j in range(O.shape[1])])
        hi = np.array([np.nanquantile(br[:, j], 0.975) if np.isfinite(br[:, j]).any() else np.nan for j in range(O.shape[1])])
    return r, lo, hi


def e1_units(units, G, w):
    """Per unit (window), per bin: observed and expected ring cells; plus case counts."""
    ring = ring_mask(G, w)
    O = np.zeros((len(units), len(BINS))); E = np.zeros_like(O); n = np.zeros(len(BINS), int)
    for u, (obs, known) in enumerate(units):
        _, R, ob, ex = border_cases(obs, known, ring)
        b = bin_index(R)
        for j in range(len(BINS)):
            O[u, j], E[u, j] = ob[b == j].sum(), ex[b == j].sum()
            n[j] += int((b == j).sum())
    return O, E, n


def neighbours(key, side):
    e, n = key
    return [(e + dx * side, n + dy * side) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def e2_and_e3(wins, side, G):
    """E2 counts per bin and the E3 ring-2 split of the 1-5 cell cases, per window."""
    ring = ring_mask(G, RULE_RING)
    keys = sorted(wins)
    cnt = {lab: dict(cases=0, nonb=0, common=0, present=0) for _, _, lab in BINS}
    O3 = np.zeros((len(keys), 2)); E3 = np.zeros((len(keys), 2)); n3 = [0, 0]
    for u, k in enumerate(keys):
        w = wins[k]
        nb = [wins[x] for x in neighbours(k, side) if x in wins]
        idx, R, ob, ex = border_cases(w["obs"], w["known"], ring)
        b = bin_index(R)
        for c, s in enumerate(idx):
            if b[c] < 0:
                continue
            lab = BINS[b[c]][2]; cnt[lab]["cases"] += 1
            if not nb:
                cnt[lab]["nonb"] += 1
                continue
            name = w["names"][s]
            most = max((int(x["obs"][x["index"][name]].sum()) if name in x["index"] else 0) for x in nb)
            com = most >= COMMON
            cnt[lab]["common"] += int(com); cnt[lab]["present"] += int(most >= 1)
            if b[c] == 0:
                g = 0 if com else 1
                O3[u, g] += ob[c]; E3[u, g] += ex[c]; n3[g] += 1
    return cnt, O3, E3, n3


# ---------------------------------------------------------------- data
def read_bnm(pattern, ref):
    files = sorted(glob.glob(pattern))
    keep, east, overlap = fit_half(files, ref)
    print(f"BNM files {pattern}: {len(files)} matched; {len(keep)} fit-half windows read; {east} test-half windows and "
          f"{overlap} sharing cells with a test window left unopened (cut at easting {ref['cut']} km, set on "
          f"{ref['n']} reference windows)")
    wins = {}
    for f in keep:
        z = load(f)
        if "species" not in z.files:
            sys.exit(f"STOP: {f} has no species names; E2 needs them")
        names = [str(x) for x in z["species"]]
        wins[window_en(f)] = dict(obs=z["obs_mask"].astype(bool), known=z["known_target"].astype(bool),
                                  names=names, index={s: i for i, s in enumerate(names)}, file=f)
    return wins


def observe(P, masks, seed=0):
    """As world_vs_bnm.py: every simulated species gets the known cells of a BNM species drawn at
    random from each survey window. (records, known) per window."""
    rng = np.random.default_rng(seed); out = []
    for m in masks:
        K = m[rng.integers(len(m), size=len(P))]
        out.append((P & K, K))
    return out


def read_sims(patterns, masks):
    seen, groups = set(), []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            z = load(f)
            if "P_last_final" not in z.files:
                print(f"  skipped (no P_last_final): {Path(f).name}"); continue
            P = z["P_last_final"].astype(bool)
            h = hashlib.md5(P.tobytes()).hexdigest()
            if h in seen:
                print(f"  skipped (same map as an earlier file): {Path(f).name}"); continue
            if P.shape[1:] != masks[0].shape[1:]:
                sys.exit(f"STOP: {f} grid {P.shape[1:]} differs from the BNM windows {masks[0].shape[1:]}")
            seen.add(h); groups.append((f, species_units(observe(P, masks))))
    return groups


def species_units(windows):
    """Regroup (records, known) per window into one unit per species: the species' cases in every
    window, stacked (n windows, G, G). Resampling these units resamples species."""
    S = len(windows[0][0])
    return [(np.stack([o[s] for o, _ in windows]), np.stack([k[s] for _, k in windows])) for s in range(S)]


# ---------------------------------------------------------------- report
def fmt(r, lo, hi):
    return "n/a" if not np.isfinite(r) else f"{r:.2f} ({lo:.2f}-{hi:.2f})"


def e1_line(name, units, G):
    out = {}
    for w in RINGS:
        O, E, n = e1_units(units, G, w)
        r, lo, hi = boot_ratio(O, E)
        out[w] = (r, lo, hi)
        print(f"  {name}, ring {w} ({ring_mask(G, w).sum()} of {G * G} cells): " +
              ", ".join(f"{lab} {fmt(r[j], lo[j], hi[j])} [{n[j]} cases]" for j, (_, _, lab) in enumerate(BINS)))
    return out


def run(bnm_pattern, sim_patterns, split_pattern):
    ref = split_ref(split_pattern)
    wins = read_bnm(bnm_pattern, ref)
    if not wins:
        sys.exit("STOP: no fit-half BNM window read")
    keys = sorted(wins); G = wins[keys[0]]["obs"].shape[-1]; side = int(round(ref["side"]))
    print(f"\nE1 border ratio (recorded cells in the ring / expected from the case's own known cells; 1 = no edge "
          f"effect; 95% interval over windows)")
    bnm = e1_line("BNM fit half", [(wins[k]["obs"], wins[k]["known"]) for k in keys], G)
    cnt, O3, E3, n3 = e2_and_e3(wins, side, G)
    print(f"\nE2 the same species in the 8 neighbouring fit-half windows (common = {COMMON}+ recorded cells there)")
    for _, _, lab in BINS:
        c = cnt[lab]; m = c["cases"] - c["nonb"]
        if not c["cases"]:
            print(f"  {lab}: no cases"); continue
        print(f"  {lab}: {c['cases']} cases, {c['nonb']} without a fit-half neighbour; of the other {m}: common next door "
              f"{c['common'] / m:.1%}, present next door {c['present'] / m:.1%}" if m else f"  {lab}: {c['cases']} cases, none with a fit-half neighbour")
    r3, lo3, hi3 = boot_ratio(O3, E3)
    print(f"\nE3 ring-{RULE_RING} border ratio of the 1-5 cell cases: common next door {fmt(r3[0], lo3[0], hi3[0])} "
          f"[{n3[0]} cases]; not common next door {fmt(r3[1], lo3[1], hi3[1])} [{n3[1]} cases]")
    sims = []
    if sim_patterns:
        print("\nControl: simulated worlds seen through BNM effort (presence = P_last_final, 1 individual)")
        masks = [wins[k]["known"] for k in keys]
        groups = read_sims(sim_patterns, masks)
        pooled = []
        for f, units in groups:
            sims.append(e1_line(Path(f).name, units, G)[RULE_RING]); pooled += units
        if len(groups) > 1:
            sims.append(e1_line(f"pooled ({len(groups)} worlds)", pooled, G)[RULE_RING])
    verdict(bnm[RULE_RING], cnt, sims)


def verdict(b, cnt, sims):
    r, lo, hi = b
    bad = [] if (np.isfinite(r[3]) and 0.8 <= r[3] <= 1.25) else [f"BNM 51+ ({r[3]:.2f})" if np.isfinite(r[3]) else "BNM 51+ (no cases to check)"]
    if sims:
        sr, slo, shi = sims[-1]                     # pooled simulated worlds (the only one if a single file)
        bad += [f"simulated {BINS[j][2]}" for j in range(len(BINS)) if np.isfinite(sr[j]) and not (slo[j] <= 1 <= shi[j])]
    c = cnt["1-5"]; m = c["cases"] - c["nonb"]; com = c["common"] / m if m else np.nan
    print(f"\nReading rule (ring {RULE_RING}): 1-5 cell border ratio {fmt(r[0], lo[0], hi[0])}; common next door "
          f"{com:.1%}" if np.isfinite(com) else "\nReading rule: no 1-5 cell case has a fit-half neighbour")
    if not sims:
        print("  controls: no simulated file given; the rule needs them")
    if bad:
        print(f"  controls failed: {', '.join(bad)} -> the measure is biased; do not read further")
    elif not sims:
        print("  -> not read (controls missing)")
    elif not np.isfinite(com):
        print("  -> not read (E2 empty)")
    elif com < 0.5:
        print("  -> (c) small cases are mostly species rare in the region too: true rarity")
    elif r[0] >= 1.3 and lo[0] > 1:
        print("  -> (a) small ranges are mostly window cuts: build world = window")
    else:
        print("  -> (b) small cases are common-nearby species with few recorded cells here, away from the edge")


# ---------------------------------------------------------------- self-test
def self_test():
    rng = np.random.default_rng(0); G = 20; ok = True

    def check(name, cond, text=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name} {text} {'PASS' if cond else 'FAIL'}")

    # T1 ring cells and per-case sums against a direct loop
    ring = ring_mask(G, 2)
    check("T1a", ring.sum() == 400 - 16 * 16 and ring_mask(G, 1).sum() == 76 and ring_mask(G, 3).sum() == 400 - 14 * 14,
          "ring sizes 76, 144, 204 cells")
    known = rng.random((30, G, G)) < 0.4
    obs = known & (rng.random((30, G, G)) < 0.1)
    idx, R, ob, ex = border_cases(obs, known, ring)
    good = True
    for c, s in enumerate(idx):
        r_ = sum(obs[s, y, x] for y in range(G) for x in range(G))
        o_ = sum(obs[s, y, x] for y in range(G) for x in range(G) if min(y, x, G - 1 - y, G - 1 - x) < 2)
        k_ = sum(known[s, y, x] for y in range(G) for x in range(G))
        kb_ = sum(known[s, y, x] for y in range(G) for x in range(G) if min(y, x, G - 1 - y, G - 1 - x) < 2)
        good &= (R[c] == r_) and (ob[c] == o_) and np.isclose(ex[c], r_ * kb_ / k_)
    check("T1b", good, "observed and expected ring cells equal a direct loop")
    bad = obs.copy(); bad[0] |= ~known[0]
    try:
        border_cases(bad, known, ring); raised = False
    except ValueError:
        raised = True
    check("T1c", raised, "records outside the known cells are refused")

    # T2 random placement among clumped known cells (effort heavy at the edge) gives about 1
    units = []
    yy, xx = np.mgrid[0:G, 0:G]
    edge_w = np.where(ring, 4.0, 1.0).ravel()
    for _ in range(60):
        kn = np.zeros((40, G * G), bool); oc = np.zeros_like(kn)
        for s in range(40):
            kc = rng.choice(G * G, int(rng.integers(40, 160)), replace=False, p=edge_w / edge_w.sum())
            kn[s, kc] = True
            oc[s, rng.choice(kc, int(rng.integers(1, 6)), replace=False)] = True
        units.append((oc.reshape(40, G, G), kn.reshape(40, G, G)))
    O, E, n = e1_units(units, G, 2); r, lo, hi = boot_ratio(O, E)
    check("T2a", abs(r[0] - 1) < 0.08 and lo[0] < 1 < hi[0], f"random 1-5 cell cases, edge-heavy effort: {r[0]:.3f} ({lo[0]:.3f}-{hi[0]:.3f})")
    area = sum(((o & ring).sum() for o, _ in units)) / sum(o.sum() * ring.sum() / (G * G) for o, _ in units)
    check("T2b", area > 1.3, f"negative control: an area-based expectation would read {area:.2f} on the same data (effort bias)")

    # T3 planted window cuts: discs centred outside or near the window edge
    units = []
    for _ in range(40):
        oc = np.zeros((40, G, G), bool)
        for s in range(40):
            side_ = rng.integers(4)
            c = rng.uniform(-3, 23); off = rng.uniform(-3.5, -1.0)
            cy, cx = [(off, c), (G - 1 - off, c), (c, off), (c, G - 1 - off)][side_]
            rad = rng.uniform(1.6, 3.2)
            oc[s] = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
        units.append((oc, np.ones_like(oc)))
    O, E, n = e1_units(units, G, 2); r, lo, hi = boot_ratio(O, E)
    check("T3", r[0] >= 1.5 and n[0] > 100, f"ranges cut by the window edge: 1-5 cell ratio {r[0]:.2f} [{n[0]} cases]")

    # T4 torus world (wrap) with compact ranges at random places, seen through full effort: about 1
    P = np.zeros((2000, G, G), bool)
    for s in range(2000):
        cy, cx = rng.uniform(0, G, 2); rad = rng.uniform(0.6, 6)
        dy = np.minimum(abs(yy - cy), G - abs(yy - cy)); dx = np.minimum(abs(xx - cx), G - abs(xx - cx))
        P[s] = dy ** 2 + dx ** 2 <= rad ** 2
    units = species_units(observe(P, [np.ones((10, G, G), bool)] * 5))
    O, E, n = e1_units(units, G, 2); r, lo, hi = boot_ratio(O, E)
    check("T4a", np.all(np.abs(r - 1) < 0.1) and np.all((lo <= 1) & (1 <= hi)),
          f"torus world, ranges across the seam: {np.round(r, 3).tolist()} intervals {np.round(lo, 2).tolist()}-{np.round(hi, 2).tolist()} {n.tolist()}")
    check("T4b", len(units) == 2000 and units[0][0].shape == (5, G, G) and np.array_equal(units[7][0][3], P[7]),
          "species units: one per species, its 5 windows stacked")

    # T5 files: fit half only, test half never opened, species matched by name, E2 counts
    global OPENED
    with tempfile.TemporaryDirectory() as tmp:
        names_a = np.array(["Aa", "Bb", "Cc"]); names_b = np.array(["Cc", "Aa", "Bb"])    # different order
        grid = [(e, n) for e in (0, 40, 80) for n in (0, 40)]                             # fit half: e 0 and 40
        for e, n in grid:
            o = np.zeros((3, G, G), bool); k = np.ones_like(o)
            nm = names_b if e == 40 else names_a
            if (e, n) == (0, 0):
                o[0, 0, 5] = True                       # Aa: 1 cell at the edge
                o[2, 10, 10] = True                     # Cc: 1 cell in the middle
            if (e, n) == (40, 0):
                o[1, :5, :5] = True                     # Aa (index 1 here): 25 cells, common next door
            if e >= 80:
                o[:] = True                             # test half: would make every species common if read
            np.savez(f"{tmp}/bnm_e{e:04d}_n{n:04d}.npz", obs_mask=o, known_target=k, species=nm, window_cells=G, cell_m=2000)
        OPENED = []
        ref = split_ref(f"{tmp}/bnm_*.npz")
        wins = read_bnm(f"{tmp}/bnm_*.npz", ref)
        check("T5a", ref["cut"] == 40 and sorted(wins) == [(0, 0), (0, 40)] and not any("e0080" in f for f in OPENED),
              f"cut {ref['cut']} km; windows read {sorted(wins)}; test-half files opened: {[f for f in OPENED if 'e0080' in f]}")
        # move the cut so that e 40 is fit half too: reference set = e 0, 40, 80, 120 -> cut 80
        np.savez(f"{tmp}/bnm_e0120_n0000.npz", obs_mask=np.ones((3, G, G), bool), known_target=np.ones((3, G, G), bool),
                 species=names_a, window_cells=G, cell_m=2000)
        OPENED = []
        ref = split_ref(f"{tmp}/bnm_*.npz"); wins = read_bnm(f"{tmp}/bnm_*.npz", ref)
        cnt, O3, E3, n3 = e2_and_e3(wins, int(ref["side"]), G)
        c = cnt["1-5"]
        check("T5b", ref["cut"] == 80 and not any(("e0080" in f) or ("e0120" in f) for f in OPENED),
              f"cut {ref['cut']} km; test-half files opened: {[f for f in OPENED if ('e0080' in f) or ('e0120' in f)]}")
        check("T5c", c["cases"] == 2 and c["nonb"] == 0 and c["common"] == 1 and c["present"] == 1 and n3 == [1, 1],
              f"E2 by name across a reordered species list: {c}, E3 cases {n3}")
        check("T5d", O3[:, 0].sum() == 1 and O3[:, 1].sum() == 0,
              "the edge case (common next door) is in the ring, the middle case is not")

    # T6 bootstrap interval holds the pooled ratio
    O = rng.poisson(5, (44, 4)).astype(float); E = O + rng.normal(0, 1, O.shape)
    r, lo, hi = boot_ratio(O, np.clip(E, 0.1, None))
    check("T6", np.all((lo <= r) & (r <= hi)), "bootstrap interval holds the pooled ratio")
    OPENED = []
    return ok


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bnm", default=REF_DEFAULT)
ap.add_argument("--sim", action="append", default=[])
ap.add_argument("--split-ref", default=REF_DEFAULT)
ap.add_argument("--self-test", action="store_true")
a = ap.parse_args()
if a.self_test:
    sys.exit(0 if self_test() else 1)
run(a.bnm, a.sim, a.split_ref)