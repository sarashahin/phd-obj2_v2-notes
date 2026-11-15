#!/usr/bin/env python3
"""
kernel_tail_2dt.py  (obj1_v2, 30 Oct 2026)

What it does
    Shares of 2Dt propagules by distance on the 20 x 20 torus, built the way dispersal.py v2
    describes it: weight (1 + r^2/a^2)^-b, r = Euclidean distance in cells, summed over periodic
    images +-10 grid periods, every image of the source excluded, the row normalised to 1.
    Reads no files and imports nothing from the project. Run from obj1_v2/work:
        python3 ../tools/kernel_tail_2dt.py | tee ../logs/kernel_tail_2dt_30oct.txt

Why
    To check whether the stage 2a kernel (a 1, b 3) already carries rare long-distance dispersal,
    before adding a uniform long-distance share (LONG_DISTANCE_PROB) on top.

Check of the method
    a 1, b 1.75 must reproduce the obj1_v2 audit's shares: 0.366 (4 edge neighbours),
    0.546 (8 neighbours), 0.247 (beyond 2 cells), 0.075 (beyond 5 cells), mean 2.41 cells.
    Distances on the torus; "beyond" uses the Chebyshev distance.

Result 30 Oct
    a 1, b 1.75: 0.3657, 0.5457, 0.2467, 0.0750, mean 2.4141 (audit reproduced)
    a 1, b 3:    0.6737, 0.8733, 0.0262, 0.0015, mean 1.2754
    -> with b 3, 2.6% of propagules land beyond 2 cells and 0.15% beyond 5 cells: rare long jumps.
"""
import numpy as np

G, IMG = 20, 10


def row(a, b):
    sy = sx = 10
    w = np.zeros((G, G))
    ys, xs = np.mgrid[0:G, 0:G]
    for i in range(-IMG, IMG + 1):
        for j in range(-IMG, IMG + 1):
            dy, dx = ys - sy + i * G, xs - sx + j * G
            r2 = dy ** 2 + dx ** 2
            k = (1 + r2 / a ** 2) ** (-b)
            k[r2 == 0] = 0                       # every image of the source excluded
            w += k
    w[sy, sx] = 0
    w /= w.sum()
    dy = np.minimum(abs(ys - sy), G - abs(ys - sy))
    dx = np.minimum(abs(xs - sx), G - abs(xs - sx))
    return w, np.sqrt(dy ** 2 + dx ** 2), np.maximum(dy, dx)


AUDIT = (0.366, 0.546, 0.247, 0.075, 2.41)
for a, b in ((1.0, 1.75), (1.0, 3.0)):
    w, eu, ch = row(a, b)
    s = (w[eu == 1].sum(), w[ch == 1].sum(), w[ch > 2].sum(), w[ch > 5].sum(), (w * eu).sum())
    line = (f"a {a:g} b {b:g}: 4 neighbours {s[0]:.4f}, 8 neighbours {s[1]:.4f}, beyond 2 {s[2]:.4f}, "
            f"beyond 5 {s[3]:.4f}, mean distance {s[4]:.4f}")
    if b == 1.75:
        ok = all(abs(u - v) <= 0.0015 for u, v in zip(s[:4], AUDIT[:4])) and abs(s[4] - AUDIT[4]) <= 0.015
        line += f"   audit shares reproduced: {'PASS' if ok else 'FAIL'}"
    print(line)