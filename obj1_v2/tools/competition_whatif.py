#!/usr/bin/env python3
"""
competition_whatif.py  (obj1_v2, gates 3 and 4: the competition-strength fix, 1 Nov 2026, Sara's "decide")

Why: spatial_cause_check v2 (Sara's run, 1 Nov) found the extra similarity decay comes from competition sorting in all 4
worlds: empty suitable cells are invadable in only 1.1-2.9% of cases, and a species placed at random among the cells it
can hold against the residents keeps the real decay (0.104-0.190 against real 0.106-0.195), while placed at random in
all its suitable cells it gives 0.015-0.028 (BNM 0.040). Each species holds only the part of its habitat where it beats
the others, and that part is a compact patch. With every pair competing at strength s (connectance 1), a species
persists in a cell only if its growth is close to the best there, so a weaker s should let species hold more of their
habitat and flatten decay. This tool asks, on the CPU, which s does that, before any GPU run.

What it does, per world (n66-n69 end maps), for s in 0.4 (now), 0.35, 0.3, 0.25, 0.2, 0.15, 0.1:
  closed cells  in every cell, the stable Lotka-Volterra equilibrium of the world's own species (ENV_r_field) with
                C = (1 - s) I + s 1 1^T, in closed form (as sink_growth.py's closed_count): the n species with the
                highest r persist, B_i = (r_i - s T) / (1 - s), T = sum r / (1 - s + s n); present where B >= one body
                mass. No dispersal (sinks are 1-2% of occupied cells, invadable empty cells 1-3%).
  observed      through the BNM survey and detection exactly as world_vs_bnm v5.8 (spatial_cause_check.observe):
                similarity decay, rare co-occupancy against BNM at 1-4 cells, species per recorded cell and per
                window (geometric means over windows, as world_vs_bnm).
  Limit: the species pool is the world's present community; with weaker competition an assembly would keep more
  species, so species per cell and per window here are lower bounds.

Reading rule (written 1 Nov, before the run):
  C1 calibration: at s 0.4 the closed-cell maps agree with the real end maps (Jaccard over species x cells 0.85 or more)
     and their decay is within 0.02 of the real decay, in every world. Otherwise nothing below is read.
  R  s for the batch: among the s with decay 0.085 or less in 3 or more of 4 worlds, the one whose median decay is
     closest to BNM's 0.040; reported with its species per cell (band 8.8-18.6), per window (27-41) and rare clumping.
     If species per cell is above the band at that s, it is said (the batch would then fail that band).
     If no s qualifies, competition strength alone cannot fix decay; the next lever is how species compete (links).

Version 1.1 (1 Nov): C1 was skipped in Sara's run because the saved strength is float32 (0.4000000059604645) and the
match to s 0.4 used 1e-9; now 1e-6. By hand from that run, C1 fails in every world (closed cells hold 59-70% of the real
presences; decay differs by +0.069, +0.026, -0.015, +0.065), so the run is not read.

Run from obj1_v2/work:
    python3 ../tools/competition_whatif.py --self-test | tail -1
    python3 ../tools/competition_whatif.py --sim "<training files>" | tee ../logs/competition_whatif_1nov.txt
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_cause_check as scc          # noqa: E402  (observe, stats, load_bnm: world_vs_bnm's measures)

S_LIST = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
DECAY_TOP, BNM_DECAY = 0.085, 0.040


def closed_cells(r, s, thr):
    """Presence (S, G, G) at the stable equilibrium of every closed cell, equal competition s (0 <= s < 1)."""
    S = r.shape[0]
    rf = r.reshape(S, -1)
    order = np.argsort(-rf, axis=0, kind="mergesort")
    rs = np.take_along_axis(rf, order, 0)
    n = np.arange(1, S + 1)[:, None]
    T = np.cumsum(rs, 0) / (1 - s + s * n)
    Bn = (rs - s * T) / (1 - s)                       # biomass of the n-th best if the top n persist
    ok = Bn >= thr
    k = np.where(ok.any(0), S - np.argmax(ok[::-1], 0), 0)
    keep = n <= k[None]
    out = np.zeros_like(rf, dtype=bool)
    np.put_along_axis(out, order, keep, 0)
    return out.reshape(r.shape)


def alpha_gamma(cases):
    """Geometric means over windows of species per recorded cell and species per window (world_vs_bnm's alpha_beta)."""
    a, g = [], []
    for o, _ in cases:
        X = o.reshape(len(o), -1)
        n = X.sum(0); rec = n > 0
        if rec.any():
            a.append(n[rec].mean()); g.append((X.sum(1) > 0).sum())
    return float(np.exp(np.mean(np.log(a)))), float(np.exp(np.mean(np.log(g))))


def measure(P, masks, args, bnm):
    cases = scc.observe(P, masks, args["detect"], args["detect_small"], args["small"])
    st = scc.stats(cases)
    al, ga = alpha_gamma(cases)
    return dict(decay=st["decay"], rare=st["rare"] / bnm["rare"], alpha=al, gamma=ga)


def pick(table):
    """table: s -> list of per-world decays. Returns the s chosen by rule R, or None."""
    good = [s for s, ds in table.items() if sum(d <= DECAY_TOP for d in ds) >= min(3, len(ds))]
    if not good:
        return None
    return min(good, key=lambda s: (abs(np.median(table[s]) - BNM_DECAY), -s))


# ------------------------------------------------ self-test
def lv_numeric(r, s, steps=4000):
    """Deterministic iteration B' = B exp(r - C B) for one cell, from B = 0.01 each (reference for the closed form)."""
    S = len(r); C = (1 - s) * np.eye(S) + s
    B = np.full(S, 0.01)
    for _ in range(steps):
        B = B * np.exp(np.clip(r - C @ B, -5, 5) * 0.5)
    return B


def self_test():
    rng = np.random.default_rng(5); res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    worst, n_cases = 0, 0
    for _ in range(40):
        S = rng.integers(3, 12); s = rng.uniform(0.05, 0.9)
        r = rng.uniform(-0.2, 1.2, S)
        Bn = lv_numeric(r, s)
        pc = closed_cells(r.reshape(S, 1, 1), s, 1e-4).ravel()
        worst += int(not np.array_equal(pc, Bn >= 1e-3)); n_cases += 1
    check("T1", worst == 0, f"closed form equals a numerical solve of the cell's dynamics in {n_cases - worst} of {n_cases} random cells")
    r = rng.uniform(-0.5, 1, (20, 6, 6))
    check("T2", np.array_equal(closed_cells(r, 0.0, 1e-4), r >= 1e-4), "s 0: every species with r above one body mass persists")
    G = 20; k = np.fft.fftfreq(G)
    spec = np.exp(-0.5 * (2 * np.pi * 2.5) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
    H = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(G, G))) * spec)); H = (H - H.mean()) / H.std()
    S = 40; opt = rng.normal(0, 1, S); w = rng.uniform(0.3, 1.0, S)
    F = 0.2 / w[:, None, None] * np.exp(-(H[None] - opt[:, None, None]) ** 2 / (2 * w[:, None, None] ** 2)) - 0.1
    occ = [closed_cells(F, s, 1e-4).sum() for s in S_LIST]
    check("T3", all(a <= b for a, b in zip(occ, occ[1:])), f"weaker competition adds presences on a planted pool: occupied "
          f"cells by s {S_LIST}: {[int(x) for x in occ]}")
    table = {0.4: [0.2, 0.18, 0.15, 0.19], 0.3: [0.1, 0.09, 0.08, 0.12], 0.2: [0.05, 0.06, 0.04, 0.09], 0.1: [0.02, 0.01, 0.02, 0.03]}
    check("T4", pick(table) == 0.2 and pick({0.4: [0.2] * 4, 0.3: [0.1] * 4}) is None,
          "rule R: picks 0.2 (3 of 4 at or below 0.085, median closest to 0.040); none qualifies -> None")
    print(f"{sum(res)} of {len(res)} PASS" if all(res) else f"FAIL: {len(res) - sum(res)} of {len(res)}")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", nargs="+", default=None)
    ap.add_argument("--bnm", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--detect", type=float, default=0.80)
    ap.add_argument("--detect-small", type=float, default=0.64)
    ap.add_argument("--small-cells", type=int, default=20)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.sim:
        sys.exit("STOP: give --sim patterns")
    args = dict(detect=a.detect, detect_small=a.detect_small, small=a.small_cells)
    bcases, masks, *_ = scc.load_bnm(a.bnm)
    bnm = scc.stats(bcases); ba, bg = alpha_gamma(bcases)
    print(f"BNM: {len(bcases)} fit-half windows; decay {bnm['decay']:.3f}; species per recorded cell {ba:.1f}, per window {bg:.1f}")
    table, rows, c1 = {s: [] for s in S_LIST}, {s: [] for s in S_LIST}, []
    for f in sorted({f for p in a.sim for f in glob.glob(p)}):
        z = np.load(f, allow_pickle=True)
        need = ("P_last_final", "ENV_r_field", "BODY_MASS", "INTERACTION_STRENGTH", "CONNECTANCE")
        miss = [k for k in need if k not in z.files]
        if miss:
            print(f"skipped {Path(f).name}: missing {miss}"); continue
        if abs(float(z["CONNECTANCE"]) - 1) > 1e-9:
            print(f"skipped {Path(f).name}: connectance not 1"); continue
        P = np.asarray(z["P_last_final"]).astype(bool); F = np.asarray(z["ENV_r_field"], float); thr = float(z["BODY_MASS"])
        s0 = float(z["INTERACTION_STRENGTH"])
        real = measure(P, masks, args, bnm)
        print(f"\n{Path(f).name[-60:]}: {len(P)} species; strength now {s0:g}")
        print(f"   real end map: decay {real['decay']:.3f}; species per cell {real['alpha']:.1f}, per window {real['gamma']:.1f}; "
              f"rare / BNM {[round(float(x), 2) for x in real['rare']]}")
        for s in S_LIST:
            Q = closed_cells(F, s, thr)
            m = measure(Q, masks, args, bnm)
            line = (f"   s {s:<4g} closed cells: decay {m['decay']:.3f}; species per cell {m['alpha']:.1f}, per window "
                    f"{m['gamma']:.1f}; rare / BNM {[round(float(x), 2) for x in m['rare']]}; occupied cells x{Q.sum() / P.sum():.2f}")
            if abs(s - s0) < 1e-6:                    # INTERACTION_STRENGTH is saved as float32 (0.4000000059604645)
                jac = float((Q & P).sum() / (Q | P).sum())
                ok = jac >= 0.85 and abs(m["decay"] - real["decay"]) <= 0.02
                c1.append(ok)
                line += f"   C1 Jaccard with the real map {jac:.3f}, decay difference {m['decay'] - real['decay']:+.3f} {'PASS' if ok else 'FAIL'}"
            print(line)
            table[s].append(m["decay"]); rows[s].append(m)
    if not c1:
        sys.exit("STOP: no world read")
    print("\nreading rule (written before the run, 1 Nov):")
    if not all(c1):
        print("   C1 FAIL in at least one world: closed cells do not stand for these worlds; nothing below is read")
        sys.exit(0)
    print("   C1 PASS in every world")
    s_pick = pick(table)
    for s in S_LIST:
        print(f"   s {s:<4g}: decay per world {[round(float(x), 3) for x in table[s]]}, median {np.median(table[s]):.3f}; species per "
              f"cell median {np.median([m['alpha'] for m in rows[s]]):.1f}, per window median {np.median([m['gamma'] for m in rows[s]]):.1f}")
    if s_pick is None:
        print("   R: no s brings decay to 0.085 or less in 3 of 4 worlds -> strength alone cannot fix decay; next lever: how "
              "species compete (links)")
    else:
        al = np.median([m["alpha"] for m in rows[s_pick]]); ga = np.median([m["gamma"] for m in rows[s_pick]])
        warn = "" if 8.8 <= al <= 18.6 else " -> species per cell outside the band 8.8-18.6 at this s (lower bound)"
        print(f"   R: s {s_pick:g} for the batch (median decay {np.median(table[s_pick]):.3f}); species per cell {al:.1f}, per "
              f"window {ga:.1f} (lower bounds){warn}")