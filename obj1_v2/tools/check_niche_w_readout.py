#!/usr/bin/env python3
"""
check_niche_w_readout.py  (obj1_v2, 30 Oct 2026)

What it does
    Re-checks Sara's pasted output of plan niche_w (8 GPU worlds, minimum niche width 0.3 and 0.4)
    scored with world_vs_bnm v5.5 --tau 1, species_compare v3 and test_05, next to n33 (width 0.2).
    Numbers are typed in from the printout; nothing is re-simulated and no BNM file is opened.

Why
    world_vs_bnm's setting summary pooled all 9 worlds as one setting, because its setting key does not
    name NICHE_w_min. This script gives the per-width medians and the per-width band verdict
    (rule of 28 Oct: median inside the band and at least 75% of worlds inside).

Checks
    1. PASS/FAIL flags recomputed from the printed numbers with the 28 Oct bands match the printout.
    2. Per-width medians and per-width setting verdicts.
    3. Expectations written before the run (n33 note, 30 Oct) against the result.

Run from obj1_v2/work (reads no files):
    python3 ../tools/check_niche_w_readout.py | tee ../logs/check_niche_w_readout_30oct.txt
Correct when part 1 ends with "mismatches: 0".
"""
import numpy as np

BNM = dict(rich=34, alpha=13.7, beta=2.34, bins=np.array([8.3, 15.0, 9.3, 11.0, 25.7, 30.7]),
           decay=0.74 - 0.70, rare=np.array([2.49, 1.50, 1.23, 1.12]))
BANDS = dict(species=0.207, alpha=0.355, beta=0.308, decay=0.045, tv=0.116, r15=8.1, r51=4.8,
             co=np.array([0.196, 0.232, 0.137, 0.077]))
NAMES = ["species", "alpha", "beta", "decay", "tv", "r15", "r51", "co"]

# name: w_min, seed, species/window, alpha, beta, bins %, sim (1, 5-10), decay, TV, rare/BNM 1-4, printed PASS set
W = {
 "n33": (0.2, 2303, 206, 42.1, 4.83, [11.4, 21.3, 13.7, 17.9, 22.5, 13.1], (0.62, 0.40), 0.22, 0.208, [1.29, 1.20, 0.93, 0.78], set()),
 "n41": (0.3, 2303, 198, 44.8, 4.35, [9.4, 19.2, 13.4, 18.1, 25.0, 14.9], (0.64, 0.42), 0.22, 0.165, [1.32, 1.25, 0.95, 0.79], {"r15"}),
 "n42": (0.3, 2304, 204, 44.3, 4.55, [10.6, 20.9, 12.0, 16.8, 25.6, 14.1], (0.67, 0.37), 0.29, 0.167, [1.48, 1.52, 1.14, 0.86], set()),
 "n43": (0.3, 2307, 194, 47.1, 4.14, [6.5, 14.7, 12.8, 21.0, 29.7, 15.2], (0.66, 0.36), 0.30, 0.175, [1.26, 1.29, 1.07, 0.93], {"r15"}),
 "n44": (0.3, 2308, 201, 44.3, 4.49, [7.9, 17.0, 13.6, 19.8, 28.4, 13.3], (0.65, 0.33), 0.32, 0.178, [1.42, 1.52, 1.25, 1.01], {"r15"}),
 "n45": (0.4, 2303, 178, 47.4, 3.73, [8.7, 16.7, 11.8, 16.8, 26.8, 19.2], (0.68, 0.46), 0.22, 0.115, [1.39, 1.32, 0.95, 0.77], {"tv", "r15"}),
 "n46": (0.4, 2304, 178, 44.1, 3.99, [9.0, 19.6, 12.6, 14.8, 26.3, 17.7], (0.71, 0.42), 0.29, 0.130, [1.65, 1.74, 1.31, 0.91], {"r15"}),
 "n47": (0.4, 2307, 176, 49.7, 3.56, [6.7, 13.8, 10.7, 19.1, 30.2, 19.5], (0.72, 0.44), 0.28, 0.140, [1.30, 1.38, 1.11, 0.92], {"r15"}),
 "n48": (0.4, 2308, 196, 49.3, 3.90, [6.9, 15.8, 12.9, 18.3, 28.6, 17.5], (0.69, 0.37), 0.32, 0.146, [1.50, 1.59, 1.23, 0.96], {"r15"}),
}
# test_05 lines (richness change over the last 20% of attempts; kept and removed per round, last 20%)
T05 = {"n41": (0.145, 2.47, 0.947), "n45": (0.180, 2.10, 0.429), "n42": (0.217, 3.17, 0.667), "n46": (0.170, 2.10, 0.500),
       "n43": (0.207, 2.45, 0.550), "n47": (0.193, 1.95, 0.429), "n44": (0.178, 2.30, 0.450), "n48": (0.128, 1.95, 0.714)}
# species_compare K1 (true maps): kept species, median suitable cells F, median true range, median individuals per cell
K1 = {"n33": (244, 287, 40, 653), "n41": (221, 370, 55, 521), "n45": (203, 398, 57, 510), "n42": (241, 373, 46, 586),
      "n46": (206, 398, 52, 558), "n43": (222, 381, 74, 518), "n47": (198, 400, 90, 453), "n44": (225, 383, 62, 571),
      "n48": (212, 399, 74, 544)}


def flags(v):
    rich, a, b, bins, dec, tv, co = v[2], v[3], v[4], np.array(v[5]), v[7], v[8], np.array(v[9])
    rel = lambda x, t, w: abs(x / t - 1) <= w + 1e-9
    f = dict(species=rel(rich, BNM["rich"], BANDS["species"]), alpha=rel(a, BNM["alpha"], BANDS["alpha"]),
             beta=rel(b, BNM["beta"], BANDS["beta"]), decay=abs(dec - BNM["decay"]) <= BANDS["decay"] + 1e-9,
             tv=tv <= BANDS["tv"] + 1e-9, r15=abs(bins[:2].sum() - BNM["bins"][:2].sum()) <= BANDS["r15"] + 1e-9,
             r51=abs(bins[5] - BNM["bins"][5]) <= BANDS["r51"] + 1e-9,
             co=bool(np.all(np.abs(co - 1) <= BANDS["co"] + 1e-9)))
    return {k for k, ok in f.items() if ok}


print("1. flags recomputed from the printed numbers (28 Oct bands)")
bad = 0
for n, v in W.items():
    got = flags(v); same = got == v[10]; bad += not same
    print(f"   {n} w_min {v[0]} seed {v[1]}: {len(got)} of 8 {sorted(got)} {'ok' if same else 'BAD, printed ' + str(sorted(v[10]))}")
print(f"   mismatches: {bad}")

print("\n2. medians per width (BNM: species 34, per cell 13.7, turnover 2.34, 1-5 23.3%, 51+ 30.7%, decay 0.04)")
print("   w_min  worlds  species  per cell  turnover  1-5%   51+%   TV     sim 1/5-10   decay  | true: kept  F    range  N/cell | last 20%  kept/rem per round")
for w in (0.2, 0.3, 0.4):
    ns = [n for n, v in W.items() if v[0] == w]
    m = lambda i: float(np.median([W[n][i] for n in ns]))
    r15 = float(np.median([sum(W[n][5][:2]) for n in ns])); r51 = float(np.median([W[n][5][5] for n in ns]))
    s1 = float(np.median([W[n][6][0] for n in ns])); s4 = float(np.median([W[n][6][1] for n in ns]))
    k = lambda i: float(np.median([K1[n][i] for n in ns]))
    t = [T05[n] for n in ns if n in T05]
    tt = (f"{np.median([x[0] for x in t]):+.3f}    {np.median([x[1] for x in t]):.2f} / {np.median([x[2] for x in t]):.2f}"
          if t else "n33: +0.151  3.17 / 1.33 (n33 note)")
    print(f"   {w:.1f}    {len(ns)}       {m(2):6.1f}   {m(3):6.1f}    {m(4):5.2f}     {r15:5.1f}  {r51:5.1f}  {m(8):.3f}  {s1:.2f} / {s4:.2f}  "
          f"{m(7):.3f}  |  {k(0):5.0f}  {k(1):4.0f}  {k(2):4.0f}  {k(3):5.0f}  | {tt}")

print("\n   per-width setting verdict (median inside and at least 75% of worlds inside):")
for w in (0.3, 0.4):
    ns = [n for n, v in W.items() if v[0] == w]
    passed = []
    for b in NAMES:
        inside = sum(b in flags(W[n]) for n in ns)
        if inside / len(ns) >= 0.75:
            passed.append(f"{b} ({inside}/{len(ns)})")
    print(f"   w_min {w}: {len(passed)} of 8 -> {', '.join(passed) or 'none'}")

print("\n3. expectations written before the run (n33 note, 30 Oct)")
a = {w: float(np.median([W[n][3] for n, v in W.items() if v[0] == w])) for w in (0.2, 0.3, 0.4)}
g = {w: float(np.median([W[n][2] for n, v in W.items() if v[0] == w])) for w in (0.2, 0.3, 0.4)}
print(f"   species per window falls:  {g[0.2]:.0f} -> {g[0.3]:.1f} -> {g[0.4]:.0f}  met, weakly (x{g[0.4] / g[0.2]:.2f})")
print(f"   species per cell falls:    {a[0.2]:.1f} -> {a[0.3]:.2f} -> {a[0.4]:.2f}  NOT met (rises)")
print(f"   F and ranges rise:         F 287 -> 377 -> 399, range 40 -> 60 -> 69 (K6, pooled species)  met")
print(f"   optima and d unchanged:    |opt| 0.38 / 0.37 / 0.38, d 8.4e-5 / 7.4e-5 / 1.1e-4  met")
print(f"   1-5 cell share may fall:   stays inside the band at 0.4 (4 of 4)")
print(f"   richness levels off:       +12.8% to +21.7% over the last 20%, kept 1.95-3.17 vs removed 0.43-0.95 per round  NOT met")