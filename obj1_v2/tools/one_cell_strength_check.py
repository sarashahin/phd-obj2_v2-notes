"""one_cell_strength_check.py (obj1_v2, 29 Oct 2026; v2 the same evening)

Question: in one cell, how does the number of species that hold the cell by their
own growth change with the competition strength s, at connectance 1 (every
C_ij = s, C_ii = 1)? It is a guide for choosing the strengths of the GPU scan;
it decides nothing on its own (the GPU worlds and the BNM bands decide).

Equation: the IBM's expected update B' = B exp(r - C B) (README sketch 23). With all
C_ij = s < 1 the species that hold the cell at equilibrium are the k with the
highest r for which r_k > s T, T = (sum of their r) / (1 + (k - 1) s). At s >= 1
only the species with the highest r holds the cell.

Growth rates: niche_traits as in environment.py (trait seed 200000 + sim seed, per
species id; optimum normal sd 1.5; width log-uniform 0.2-1.0; peak 1.5 (0.2 / w)^k,
k 0.5), r = peak exp(-(a H - opt)^2 / (2 w^2)) - 0.1 with contrast a 0.25 (plan
niche_b). Assembly as v2.3 --founding best, in one cell: 1 + floor(0.05 S)
newcomers per round, candidates tried in id order, a candidate enters if
r - s (resident biomass) >= 0, at most 100 tries per newcomer needed, 1,000
entered.

Not in it: dispersal (no sinks, no turnover), space, chance (one individual can
die), connectance below 1. Use the ratios between strengths, not the counts.

Checks (all must PASS):
  V1  the top-k rule gives the same count as running the update B' = B exp(r - C B)
  V2  s = 1 keeps exactly one species
  V3  the trait copy here equals environment.niche_traits (run from obj1_v2/work;
      "skipped" if environment.py cannot be imported)

v1 (earlier the same day) drew widths uniform in 0.2-1.0 and traits from one
random stream; v2 copies niche_traits exactly (log-uniform widths, per id).

Run from obj1_v2/work:
    USE_GPU=0 python3 ../tools/one_cell_strength_check.py | tee ../logs/one_cell_strength_check_29oct.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

PEAK_MAX, W_MIN, W_MAX, OPT_SD, COST = 1.5, 0.2, 1.0, 1.5, 0.1
K, CONTRAST, CAP, TARGET = 0.5, 0.25, 100, 1000
SIM_SEEDS = (2303, 2304, 2307, 2308, 2401, 2402, 2403, 2404, 2405, 2406)
STRENGTHS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def traits(sid, seed):
    rng = np.random.default_rng([int(seed), 7919, int(sid)])
    opt = float(rng.normal(0.0, OPT_SD))
    w = float(np.exp(rng.uniform(np.log(W_MIN), np.log(W_MAX))))
    return opt, w, float(PEAK_MAX * (W_MIN / w) ** K)


def pool(seed, n):
    T = np.array([traits(i, seed) for i in range(n)])
    return T[:, 0], T[:, 1], T[:, 2]


def r_at(opt, w, peak, h):
    return peak * np.exp(-(CONTRAST * h - opt) ** 2 / (2 * w ** 2)) - COST


def survivors(r, s):
    r = np.sort(np.asarray(r, float))[::-1]
    r = r[r > 0]
    if len(r) == 0 or s >= 1:
        return r[:1]
    k = len(r)
    while k > 1:
        if r[k - 1] > s * r[:k].sum() / (1 + (k - 1) * s):
            return r[:k]
        k -= 1
    return r[:1]


def biomass_sum(comm, s):
    if len(comm) == 0:
        return 0.0
    return comm[0] if s >= 1 else comm.sum() / (1 + (len(comm) - 1) * s)


def assemble(rpool, s):
    comm, entered, i = np.array([]), 0, 0
    while entered < TARGET:
        need = 1 + int(0.05 * len(comm))
        tried = rpool[i:i + CAP * need]
        if len(tried) < CAP * need:
            raise RuntimeError("trait pool too small; raise POOL")
        idx = np.flatnonzero(tried - s * biomass_sum(comm, s) >= 0)[:need]
        if len(idx) == 0:
            return comm, entered, True
        i += idx[-1] + 1 if len(idx) == need else CAP * need
        entered += len(idx)
        comm = survivors(np.concatenate([comm, tried[idx]]), s)
    return comm, entered, False


def update_count(r, s, steps=60000):
    C = np.full((len(r), len(r)), s)
    np.fill_diagonal(C, 1.0)
    B = np.full(len(r), 1e-3)
    for _ in range(steps):
        B = B * np.exp(r - C @ B)
        B[B < 1e-12] = 0.0
    return int((B > 1e-6).sum())


if __name__ == "__main__":
    POOL = int(os.environ.get("POOL", 150000))
    ok_all = True
    rng = np.random.default_rng(1)
    for s in (0.4, 0.7, 0.9, 0.97):
        r = rng.uniform(0.6, 1.2, 25)
        a, b = len(survivors(r, s)), update_count(r, s)
        ok_all &= a == b
        print(f"V1 s {s}: rule {a}, update {b}  {'PASS' if a == b else 'FAIL'}")
    r = rng.uniform(0.6, 1.2, 25)
    a, b = len(survivors(r, 1.0)), update_count(r, 1.0)
    ok_all &= a == b == 1
    print(f"V2 s 1.0: rule {a}, update {b}  {'PASS' if a == b == 1 else 'FAIL'}")
    try:
        import environment as env
        mine = np.array([traits(i, 202303) for i in range(1000)])
        real = np.array([env.niche_traits(i, seed=202303, tradeoff=K) for i in range(1000)])
        good = np.allclose(mine, real, rtol=0, atol=1e-12)
        ok_all &= good
        print(f"V3 traits equal environment.niche_traits for 1,000 ids (max diff {np.abs(mine - real).max():.1e}): "
              f"{'PASS' if good else 'FAIL'}")
    except ImportError as e:
        print(f"V3 skipped: environment.py not importable here ({e})")
    pools = {seed: pool(200000 + seed, POOL) for seed in SIM_SEEDS}
    for h in (0.0, 1.5):
        mean = {}
        for s in STRENGTHS:
            out = [assemble(r_at(*pools[seed], h), s) for seed in SIM_SEEDS]
            mean[s] = np.mean([len(c) for c, _, _ in out])
            stops = sum(st for _, _, st in out)
            if stops:
                print(f"  habitat {h}, s {s}: {stops} of {len(out)} runs stopped at the cap before 1,000 entered")
        print(f"habitat {h} (seen {CONTRAST * h:g}): " + "  ".join(
            f"s {s} {mean[s]:.1f} (x{mean[s] / mean[0.4]:.2f})" for s in STRENGTHS))
    print("all checks passed" if ok_all else "CHECK FAILED")