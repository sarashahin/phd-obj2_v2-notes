#!/usr/bin/env python3
"""
check_tau_sweep_readout.py  (obj1_v2, 30 Oct 2026)

What it does
    Re-checks the pasted world_vs_bnm output of the observation-threshold (tau) sweep
    on saved worlds (gate 2 in obj1_v2/tests/README.md). Numbers are typed in from the
    printout; nothing is re-simulated and no BNM test-half data are touched.

Checks
    1. Derived numbers (range-size TV, rare clumping / BNM, species ratios) recomputed
       from the printed raw numbers agree with the printed ones.
    2. PASS/FAIL flags recomputed with the 28 Oct BANDS (patch_world_vs_bnm_bands.py, v5)
       agree with the printed flags.
    3. Medians over the niche worlds at each tau, for the readout table.
    4. The one similarity-decay PASS (strength 0.8, tau 1000) and its similarity level.
    5. Founded-best-cell worlds: how much tau removes of the per-cell and per-window excess.

Run from obj1_v2/work (reads no files; the numbers are typed in from the 30 Oct printout):
    python3 ../tools/check_tau_sweep_readout.py | tee ../logs/check_tau_sweep_readout_30oct.txt
Correct when part 1 ends with "mismatches: 0".

Block 0 has no tau header in the paste; it matches the strength-test numbers recorded at
tau 1 (x5.8 species per window, x2.2 per cell at strength 0.8), so it is read as tau 1.
Rounding: printed Sorensen values have 2 decimals, so decay is taken as printed.
"""
import numpy as np

BNM = dict(rich=34, alpha=13.7, beta=2.34, gamma=32.0,
           bins=np.array([8.3, 15.0, 9.3, 11.0, 25.7, 30.7]),
           sor=np.array([0.74, 0.72, 0.71, 0.70]),
           rare=np.array([2.49, 1.50, 1.23, 1.12]))
BANDS = dict(species=0.207, alpha=0.355, beta=0.308, decay=0.045, tv=0.116, r15=0.081, r51=0.048,
             co=np.array([0.196, 0.232, 0.137, 0.077]))
NAMES = ['species', 'alpha', 'beta', 'decay', 'tv', 'r15', 'r51', 'co']

# (tau, world): rich, alpha_gm, beta_gm, gamma_gm, bins %, sor, rare co-occ, printed decay, printed TV, printed PASS set
W = {
 (1, '2304'): (27, 12.2, 2.17, 26.5, [2.2, 8.5, 6.8, 10.2, 33.9, 38.4], [0.81, 0.74, 0.66, 0.60], [3.87, 2.72, 1.74, 0.97], 0.22, 0.159, {'species', 'alpha', 'beta'}),
 (1, '2307'): (28, 10.8, 2.59, 27.9, [1.8, 4.1, 6.4, 17.2, 39.9, 30.6], [0.76, 0.64, 0.55, 0.47], [2.68, 1.97, 1.47, 1.14], 0.29, 0.203, {'species', 'alpha', 'beta', 'r51'}),
 (1, '2308'): (32, 12.7, 2.53, 32.0, [2.1, 4.5, 8.1, 16.0, 39.2, 30.2], [0.77, 0.67, 0.56, 0.48], [3.27, 2.31, 1.63, 1.20], 0.29, 0.184, {'species', 'alpha', 'beta', 'r51'}),
 (10, '2303'): (29, 9.4, 3.00, 28.1, [4.6, 10.6, 10.0, 18.6, 31.0, 25.2], [0.71, 0.58, 0.50, 0.49], [2.63, 1.54, 1.01, 0.82], 0.22, 0.135, {'species', 'alpha', 'beta', 'r15'}),
 (10, '2304'): (26, 9.4, 2.78, 26.1, [3.0, 9.5, 8.4, 14.1, 36.1, 29.0], [0.75, 0.62, 0.50, 0.44], [3.29, 2.15, 1.35, 0.93], 0.31, 0.135, {'alpha', 'beta', 'r51'}),
 (10, '2307'): (28, 8.8, 3.13, 27.6, [2.7, 6.4, 7.7, 21.5, 38.3, 23.3], [0.71, 0.56, 0.46, 0.39], [2.86, 1.96, 1.35, 1.04], 0.33, 0.230, {'species'}),
 (10, '2308'): (32, 9.6, 3.33, 31.8, [2.4, 8.3, 11.7, 19.9, 35.9, 21.8], [0.71, 0.56, 0.44, 0.36], [3.73, 2.53, 1.60, 1.11], 0.35, 0.215, {'species', 'alpha'}),
 (100, '2303'): (29, 7.8, 3.59, 28.1, [4.6, 11.4, 10.8, 21.0, 33.8, 18.4], [0.61, 0.46, 0.38, 0.37], [2.58, 1.52, 1.01, 0.82], 0.24, 0.196, {'species', 'r15'}),
 (100, '2304'): (26, 7.8, 3.34, 26.0, [3.1, 10.4, 10.0, 16.9, 37.9, 21.7], [0.68, 0.51, 0.39, 0.34], [3.24, 2.16, 1.39, 0.92], 0.34, 0.188, set()),
 (100, '2307'): (28, 8.0, 3.45, 27.5, [2.8, 7.3, 8.7, 23.0, 37.9, 20.2], [0.67, 0.51, 0.40, 0.34], [2.88, 1.88, 1.30, 1.00], 0.34, 0.241, {'species'}),
 (100, '2308'): (32, 8.4, 3.75, 31.6, [3.4, 10.3, 13.1, 20.9, 34.1, 18.2], [0.66, 0.50, 0.38, 0.30], [3.40, 2.22, 1.42, 1.05], 0.35, 0.221, {'species'}),
 (1000, '2303'): (22, 5.8, 3.82, 22.0, [5.0, 11.9, 9.5, 20.4, 36.1, 17.1], [0.56, 0.41, 0.33, 0.32], [2.70, 1.55, 1.03, 0.85], 0.24, 0.200, {'r15'}),
 (1000, '2304'): (26, 6.2, 4.11, 25.3, [3.7, 12.4, 11.2, 21.4, 35.2, 16.1], [0.60, 0.42, 0.32, 0.27], [3.23, 1.93, 1.26, 0.96], 0.32, 0.219, {'r15'}),
 (1000, '2307'): (27, 6.4, 4.21, 27.0, [4.3, 11.0, 13.2, 23.3, 33.6, 14.6], [0.61, 0.44, 0.34, 0.29], [2.98, 1.85, 1.29, 1.03], 0.32, 0.240, {'species', 'r15'}),
 (1000, '2308'): (31, 6.9, 4.39, 30.1, [3.2, 12.2, 14.3, 24.2, 33.6, 12.5], [0.58, 0.42, 0.31, 0.25], [3.19, 1.98, 1.25, 0.95], 0.33, 0.261, {'species', 'r15'}),
 (1000, 'is0.8'): (152, 7.4, 20.34, 149.8, [24.1, 42.5, 19.6, 11.0, 2.8, 0.0], [0.15, 0.08, 0.07, 0.07], [2.13, 1.21, 1.05, 0.91], 0.08, 0.536, {'decay'}),
}


def derived(r):
    rich, a, b, g, bins, sor, rare, dec, tv_p, _ = r
    bins, rare = np.array(bins), np.array(rare)
    return dict(tv=0.5 * np.abs(bins - BNM['bins']).sum() / 100, co=rare / BNM['rare'],
                r15=bins[0] + bins[1], r51=bins[5], decay=dec, rich=rich, alpha=a, beta=b, gamma=g)


def flags(d):
    rel = lambda x, t, w: abs(x / t - 1) <= w + 1e-9
    return {n for n, ok in dict(
        species=rel(d['rich'], BNM['rich'], BANDS['species']), alpha=rel(d['alpha'], BNM['alpha'], BANDS['alpha']),
        beta=rel(d['beta'], BNM['beta'], BANDS['beta']),
        decay=abs(d['decay'] - (BNM['sor'][0] - BNM['sor'][3])) <= BANDS['decay'] + 1e-9,
        tv=d['tv'] <= BANDS['tv'], r15=abs(d['r15'] - BNM['bins'][:2].sum()) <= 100 * BANDS['r15'] + 1e-9,
        r51=abs(d['r51'] - BNM['bins'][5]) <= 100 * BANDS['r51'] + 1e-9,
        co=bool(np.all(np.abs(d['co'] - 1) <= BANDS['co'] + 1e-9))).items() if ok}


bad = 0
print('1-2. derived numbers and flags against the printout')
for k, r in W.items():
    d = derived(r)
    tv_ok = abs(d['tv'] - r[8]) <= 0.002          # printed bins are rounded to 0.1%
    got = flags(d)
    same = got == r[9]
    bad += (not tv_ok) + (not same)
    note = '' if same else f'  differs: mine {sorted(got)} printed {sorted(r[9])}'
    print(f'   tau {k[0]:>4} {k[1]:>5}  TV {d["tv"]:.3f} vs {r[8]:.3f} {"ok" if tv_ok else "BAD"}  flags {"ok" if same else "BAD"}{note}')
print(f'   mismatches: {bad}')

print('\n3. medians over niche worlds (tau 1: 3 visible worlds; others: 4)')
print('   tau  sp/win  sp/cell  turnover  sim1  sim5-10  decay  TV     1-5%   51+%  rare/BNM d1  d2')
for tau in (1, 10, 100, 1000):
    rows = [derived(r) | dict(s1=r[5][0], s4=r[5][3]) for (t, w), r in W.items() if t == tau and w[0] == '2']
    m = lambda key: float(np.median([x[key] for x in rows]))
    c1 = float(np.median([x['co'][0] for x in rows])); c2 = float(np.median([x['co'][1] for x in rows]))
    print(f'   {tau:>4}  {m("rich"):5.1f}  {m("alpha"):6.1f}  {m("beta"):7.2f}  {m("s1"):5.2f}  {m("s4"):6.2f}  '
          f'{m("decay"):5.2f}  {m("tv"):.3f}  {m("r15"):5.1f}  {m("r51"):5.1f}  {c1:5.2f}  {c2:5.2f}')
print('   BNM    34.0    13.7     2.34   0.74    0.70   0.04         23.3   30.7   1.00   1.00')

print('\n4. the similarity-decay PASS at strength 0.8, tau 1000')
r = W[(1000, 'is0.8')]
print(f'   decay {r[7]} (band: within 0.045 of BNM 0.04) -> PASS; similarity at 1 cell {r[5][0]} vs BNM 0.74 (x{r[5][0] / 0.74:.2f})')

print('\n5. founded best cell: excess left as tau rises (ratio to BNM, species per window / per cell)')
fb = {'fb A': ([206, 198, 196, 157], [42.1, 26.3, 21.9, 13.5]), 'fb B': ([346, 334, 328, 212], [57.8, 35.3, 30.4, 13.8]),
      's 0.6': ([268, 254, 251, 178], [40.0, 22.9, 19.4, 9.6]), 's 0.7': ([220, 211, 208, 156], [33.1, 18.2, 15.0, 8.4]),
      's 0.8': ([198, 191, 188, 152], [30.5, 15.7, 11.9, 7.4])}
for n, (sp, al) in fb.items():
    print(f'   {n:6s} per window ' + ' '.join(f'{x / 34:5.2f}' for x in sp) +
          '   per cell ' + ' '.join(f'{x / 13.7:5.2f}' for x in al) + '   (tau 1, 10, 100, 1000)')