#!/usr/bin/env python3
"""
candidate_specialist_share.py  (obj1_v2, rare end round 2, pre-check; 2 Nov 2026)

How many newcomer candidates would be habitat specialists at each minimum niche width? Candidates are drawn by the
simulator's own rule (environment.niche_traits: optimum normal(0, 1.5), width log-uniform [w_min, 1], peak
1.5 (w_min / w)^0.5, cost 0.1) and scored on 20 habitat maps (environment.habitat_layer, length scale 3, contrast
0.25, no shift). "Competitive somewhere": growth above 0.7 in at least one cell, about what an invader needs against
s x sum B (0.4 x about 1.7) in a cell holding about 10 species. "Specialist": footprint (r > 0) of 3-40 cells.
Description only: whether specialists are kept and persist is for the IBM.

Run from obj1_v2/work:  python3 ../tools/candidate_specialist_share.py
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from environment import habitat_layer   # noqa: E402

rng = np.random.default_rng(5)
H = 0.25 * np.stack([habitat_layer(20, 20, 3, seed=[9, i]).ravel() for i in range(20)])
for wmin in (0.2, 0.15, 0.1):
    n = 200000
    opt = rng.normal(0, 1.5, n); w = np.exp(rng.uniform(np.log(wmin), 0, n)); pk = 1.5 * (wmin / w) ** 0.5
    Hm = H[rng.integers(0, 20, n)]
    r = pk[:, None] * np.exp(-(Hm - opt[:, None]) ** 2 / (2 * w[:, None] ** 2)) - 0.1
    fp = (r > 0).sum(1); can = r.max(1) > 0.7
    spec = can & (fp >= 3) & (fp <= 40)
    print(f"w_min {wmin}: competitive somewhere {can.mean():.1%} of candidates; of those, specialists (3-40 cells) "
          f"{spec.sum() / can.sum():.1%}")