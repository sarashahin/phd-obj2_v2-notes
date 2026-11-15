#!/usr/bin/env python3
"""
specialist_habitat_check.py  (obj1_v2, rare end round 2, pre-check before the w_min pilot; 2 Nov 2026)

Question: if the minimum niche width is lowered (0.1 or 0.15), would a species of rare habitat keep a small but
non-zero footprint long enough to persist, under the habitat change of the gate 5 setting (habitat tau 62,500, shift
sd 1.0 tau 62,500, 25 sub-steps of 200 IBM steps per 5,000-step round, contrast 0.25, length scale 3)? This asks only
the habitat side: no competition, no dispersal. It simulates the habitat rule itself (assembly v2.6): before every
piece U <- rho_s U + sqrt(1 - rho_s^2) Z, U standardised again; m <- rho_s m + sqrt(1 - rho_s^2) sd eps;
H = contrast (U + m); rho_s = exp(-200 / 62,500); Z = environment.habitat_layer(20, 20, 3, seed=...). One piece is
one BNM year (gate 7, 4 seeds, 1 Nov).
Footprint of a species (opt, w) at the width bound (peak 1.5, cost 0.1): cells with r = 1.5 exp(-(H - opt)^2 /
(2 w^2)) - 0.1 > 0. A small range needs 3-40 cells (40 cells is about 5 recorded cells at the median survey).

Calibration (before the reading): K1 per-round lag-1 correlation of U at a cell within 0.02 of exp(-5,000 / 62,500)
= 0.923; K2 the shift's sd within 0.85-1.15 and its per-round lag-1 correlation within 0.03 of 0.923 (shift runs);
K3 with the shift at about 0 (|m| < 0.1) the median footprint of (opt 0.6, w 0.1) and (opt 0.5, w 0.2) within 35% of the
Gaussian values for an infinite map (28 and 178 cells).
Reading rule (written 2 Nov, before the first run), on specialists whose median footprint without shift is 3-40 cells:
  P1 the shift empties their habitat for long: mean viable spell (consecutive years with 3 or more cells) under shift
     1.0 shorter than one round (25 years) -> a narrower niche alone cannot give persistent rare species in this
     setting; the shift has to be reconsidered with it.
  P2 the habitat lets them persist: mean viable spell 100 years (4 rounds) or more -> the pilot tests the competition
     side only.
  P3 between: the pilot decides.
Writes nothing.

Run from obj1_v2/work (needs environment.py there):
    python3 ../tools/specialist_habitat_check.py --rounds 2000 | tee ../logs/specialist_habitat_2nov.txt
"""
import argparse, sys
import numpy as np

sys.path.insert(0, ".")
from environment import habitat_layer   # noqa: E402  (the simulator's own habitat layer)

G, LS, CONTRAST, COST, PEAK = 20, 3.0, 0.25, 0.1, 1.5
W, PIECES, TAU = 5000, 25, 62500
OPTS = (0.3, 0.4, 0.5, 0.6, 0.7)
WIDTHS = (0.1, 0.15, 0.2)


def run(rounds, shift_sd, seed):
    rho_s = np.exp(-(W / PIECES) / TAU)
    a = np.sqrt(1 - rho_s ** 2)
    rng = np.random.default_rng([seed, 1])
    U = habitat_layer(G, G, LS, seed=[seed, 0])
    m = 0.0
    O = np.array(OPTS)[:, None, None]; Wd = np.array(WIDTHS)[None, :, None]
    fp = np.zeros((rounds * PIECES, len(OPTS), len(WIDTHS)), np.int16)
    ms = np.zeros(rounds * PIECES); u_round = []
    for t in range(rounds):
        for j in range(PIECES):
            k = t * PIECES + j
            if k > 0:
                U = rho_s * U + a * habitat_layer(G, G, LS, seed=[seed, 104729, t, j])
                U = (U - U.mean()) / U.std()
                m = rho_s * m + a * shift_sd * rng.normal()
            H = (CONTRAST * (U + m)).ravel()[None, None, :]
            r = PEAK * np.exp(-(H - O) ** 2 / (2 * Wd ** 2)) - COST
            fp[k] = (r > 0).sum(-1)
            ms[k] = m
        u_round.append(U.ravel().copy())
    return fp, ms, np.array(u_round)


def spells(alive):
    """Mean length of runs of True and of False (in pieces)."""
    x = np.asarray(alive, int)
    d = np.diff(np.r_[0, x, 0])
    on = np.flatnonzero(d == 1); off = np.flatnonzero(d == -1)
    up = off - on
    x2 = 1 - x; d2 = np.diff(np.r_[0, x2, 0])
    on2 = np.flatnonzero(d2 == 1); off2 = np.flatnonzero(d2 == -1)
    down = off2 - on2
    return (float(up.mean()) if len(up) else 0.0), (float(down.mean()) if len(down) else 0.0), len(up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2303)
    a = ap.parse_args()
    burn = 50 * PIECES
    out = {}
    for sd in (0.0, 1.0):
        fp, ms, ur = run(a.rounds, sd, a.seed)
        out[sd] = (fp[burn:], ms[burn:], ur)
    # calibration
    ur = out[1.0][2]
    k1 = float(np.mean([np.corrcoef(ur[i], ur[i + 1])[0, 1] for i in range(50, len(ur) - 1)]))
    m_round = out[1.0][1][PIECES - 1::PIECES]
    k2_sd = float(np.std(out[1.0][1])); k2_r = float(np.corrcoef(m_round[:-1], m_round[1:])[0, 1])
    near0 = np.abs(out[1.0][1]) < 0.1
    f = out[1.0][0]
    k3a = float(np.median(f[near0, OPTS.index(0.6), WIDTHS.index(0.1)]))
    k3b = float(np.median(f[near0, OPTS.index(0.5), WIDTHS.index(0.2)]))
    rho_r = np.exp(-W / TAU)
    ok1 = abs(k1 - rho_r) <= 0.02
    ok2 = 0.85 <= k2_sd <= 1.15 and abs(k2_r - rho_r) <= 0.03
    ok3 = abs(k3a - 28) <= 0.35 * 28 and abs(k3b - 178) <= 0.35 * 178
    print(f"rounds {a.rounds} (first 50 dropped), {PIECES} pieces per round, one piece = one BNM year")
    print(f"K1 U per-round lag-1 correlation {k1:.3f} (rule {rho_r:.3f} +- 0.02): {'PASS' if ok1 else 'FAIL'}")
    print(f"K2 shift sd {k2_sd:.2f} (0.85-1.15), per-round lag-1 {k2_r:.3f} ({rho_r:.3f} +- 0.03): {'PASS' if ok2 else 'FAIL'}")
    print(f"K3 median footprint at shift about 0: opt 0.6 w 0.1 {k3a:.0f} cells (28), opt 0.5 w 0.2 {k3b:.0f} cells (178): "
          f"{'PASS' if ok3 else 'FAIL'}")
    if not (ok1 and ok2 and ok3):
        sys.exit("calibration failed: nothing below is read")
    print("\nper specialist (width at the bound, peak 1.5): median footprint; share of years with 0 cells / 1-2 / 3-40 / 41+;"
          " mean viable spell (years with 3+ cells) / mean gap (years), shift 0 then shift 1.0")
    cand = []
    for i, o in enumerate(OPTS):
        for j, w in enumerate(WIDTHS):
            row = []
            for sd in (0.0, 1.0):
                x = out[sd][0][:, i, j].astype(int)
                up, down, n = spells(x >= 3)
                row.append((np.median(x), np.mean(x == 0), np.mean((x >= 1) & (x <= 2)), np.mean((x >= 3) & (x <= 40)),
                            np.mean(x > 40), up, down, n))
            r0, r1 = row
            print(f"  opt {o:.1f} w {w:.2f}:  no shift {r0[0]:4.0f} cells; {100 * r0[1]:4.1f}/{100 * r0[2]:4.1f}/{100 * r0[3]:5.1f}/"
                  f"{100 * r0[4]:5.1f}%; spell {r0[5]:7.0f} / gap {r0[6]:5.0f}   |   shift 1.0 {r1[0]:4.0f} cells; "
                  f"{100 * r1[1]:4.1f}/{100 * r1[2]:4.1f}/{100 * r1[3]:5.1f}/{100 * r1[4]:5.1f}%; spell {r1[5]:6.0f} / gap {r1[6]:5.0f} ({r1[7]} spells)")
            if 3 <= r0[0] <= 40:
                cand.append((o, w, r1[5]))
    print("\nreading rule (written 2 Nov, before the first run), specialists with a median footprint of 3-40 cells without shift:")
    if not cand:
        print("  none in the grid"); return
    for o, w, sp in cand:
        v = "P1 shift empties the habitat for long" if sp < 25 else ("P2 habitat lets them persist" if sp >= 100 else "P3 between: the pilot decides")
        print(f"  opt {o:.1f} w {w:.2f}: mean viable spell under shift 1.0 {sp:.0f} years -> {v}")


if __name__ == "__main__":
    main()