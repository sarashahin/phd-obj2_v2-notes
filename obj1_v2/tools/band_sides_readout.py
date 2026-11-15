#!/usr/bin/env python3
"""
band_sides_readout.py  (obj1_v2, gate 5 readout, 1 Nov 2026)

Reads a world_vs_bnm --each log (any number of worlds or frame copies) and prints, per band, how many worlds sit below,
inside and above the 28 Oct band, the median and quartiles, and the same per seed (taken from _s<seed>_ in the name).
Inside or outside is world_vs_bnm's own per-world PASS / FAIL flag; the printed value only gives the side of a FAIL
(values are printed rounded, so a value on a band edge can be either). R0: the flags summed per band must equal
world_vs_bnm's setting line, or nothing is read; R0 also prints how many flags differ from the band applied to the rounded
value (edge cases).
Version 3 (2 Nov): a log with no simulated world stops with that message (it read as a band mismatch before).
Version 2 (1 Nov): version 1 compared names with a leading space, so its R0 compared nothing and passed; fixed, and
inside/outside now comes from the printed flags.
Bands (28 Oct, claude/n33_result_29oct.md): species per window 27-41 (median per window), species per cell 8.8-18.6
(alpha, geometric mean), turnover 1.62-3.06, similarity decay 0.085 or less, range-size TV 0.116 or less, 1-5 cell share
15.2-31.4%, 51+ share 25.9-35.5%, rare clumping: co-occupancy / BNM within 1 +- 0.196, 0.232, 0.137, 0.077 at 1-4 cells.

Run from obj1_v2/work:
    python3 ../tools/band_sides_readout.py ../logs/world_vs_bnm_frames_g5_1nov.txt
"""
import re, sys
import numpy as np

BANDS = {"window": (27, 41), "cell": (8.8, 18.6), "turnover": (1.62, 3.06), "decay": (-np.inf, 0.085),
         "TV": (-np.inf, 0.116), "1-5": (15.2, 31.4), "51+": (25.9, 35.5)}
RARE = [0.196, 0.232, 0.137, 0.077]
NAMES = {"window": "species per window", "cell": "species per cell", "turnover": "turnover", "decay": "similarity decay",
         "TV": "range-size TV", "1-5": "1-5 cell share", "51+": "51+ share", "rare": "rare clumping 1-4 cells"}


def parse(text):
    worlds, cur = [], None
    for line in text.split("\n"):
        if line.startswith("simulated: "):
            m = re.search(r"_s(\d+)_", line)
            cur = dict(name=line.split()[1], seed=m.group(1) if m else "?"); worlds.append(cur)
        elif cur is None:
            continue
        elif "range size, median" in line:
            sh = dict(re.findall(r"(\d+(?:-\d+)?\+?) ([\d.]+)%", line))
            cur["1"] = float(sh["1"]); cur["1-5"] = float(sh["1"]) + float(sh["2-5"]); cur["51+"] = float(sh["51+"])
        elif "species present per window" in line:
            cur["window"] = float(line.split("median")[1])
        elif "geometric means over windows" in line:
            m = re.search(r"alpha ([\d.]+) x turnover \(beta\) ([\d.]+)", line)
            cur["cell"], cur["turnover"] = float(m.group(1)), float(m.group(2))
        elif "against BNM: rare co-occupancy" in line:
            cur["rare_v"] = [float(x) for x in re.search(r"\[([^\]]+)\]", line).group(1).split(",")]
            cur["TV"] = float(re.search(r"range-size distance ([\d.]+)", line).group(1))
            cur["decay"] = float(re.search(r"similarity decay ([\d.]+)", line).group(1))
        elif line.strip().startswith("bands (28 Oct):"):
            body = line.split(":", 1)[1].split("->")[0]
            cur["printed"] = {k.strip(" ,"): v == "PASS" for k, v in re.findall(r"([A-Za-z0-9+\- ,]+?) (PASS|FAIL)", body)}
    return worlds


def side_value(w, k):
    """side from the rounded value alone: -1 below, 0 inside, 1 above"""
    if k == "rare":
        dev = [v - 1 for v in w["rare_v"]]
        out = [abs(d) > b + 1e-12 for d, b in zip(dev, RARE)]
        if not any(out):
            return 0
        worst = max((abs(d) - b, d) for d, b, o in zip(dev, RARE, out) if o)
        return -1 if worst[1] < 0 else 1
    lo, hi = BANDS[k]
    return -1 if w[k] < lo else (1 if w[k] > hi else 0)


def side(w, k):
    """0 when world_vs_bnm printed PASS; otherwise the side of the value (an edge value goes to the nearer edge)"""
    if w["printed"][NAMES[k]]:
        return 0
    v = side_value(w, k)
    if v != 0:
        return v
    if k == "rare":
        return 1
    lo, hi = BANDS[k]
    return -1 if not np.isinf(lo) and abs(w[k] - lo) <= abs(w[k] - hi) else 1


def main(path):
    text = open(path).read()
    worlds = [w for w in parse(text) if "printed" in w]
    if not worlds:   # version 3 (2 Nov): an empty log is named as such, not read as a band mismatch
        sys.exit("stop: no simulated world in this log (world_vs_bnm found no file for its --sim pattern); nothing to read")
    keys = list(BANDS) + ["rare"]
    printed = re.search(r"-> fails .*?\n\s+(.*)", text) or re.search(r"-> passes .*?\n\s+(.*)", text)
    counts = dict((n.strip(" ;,"), int(a)) for n, a, b in re.findall(r"([A-Za-z0-9+\- ;,]+?) (?:PASS|FAIL) \((\d+)/(\d+)\)", printed.group(1))) if printed else {}
    ok = len(counts) == len(keys) and all(NAMES[k] in w["printed"] for w in worlds for k in keys)
    edges = []
    for k in keys:
        mine = sum(w["printed"].get(NAMES[k], False) for w in worlds)
        theirs = counts.get(NAMES[k])
        if theirs is None or mine != theirs:
            ok = False; print(f"R0 MISMATCH {NAMES[k]}: per-world flags {mine}, setting line {theirs}")
        e = sum((side_value(w, k) == 0) != w["printed"][NAMES[k]] for w in worlds)
        if e:
            edges.append(f"{NAMES[k]} {e}")
    print(f"R0 {'PASS' if ok else 'FAIL'}: {len(worlds)} worlds; {len(counts)} bands on the setting line; per-world flags "
          f"{'sum to' if ok else 'differ from'} its counts; flags that differ from the rounded value (edge cases): "
          f"{', '.join(edges) if edges else 'none'}")
    if not ok:
        sys.exit("stop: the band definitions here do not match world_vs_bnm; nothing below is read")
    seeds = sorted({w["seed"] for w in worlds})
    print(f"\n{'band':26s} {'below':>6s} {'inside':>7s} {'above':>6s}   median (quartiles)        per seed: median [below/inside/above]")
    for k in keys:
        s = [side(w, k) for w in worlds]
        if k == "rare":
            vals = np.array([w["rare_v"][3] for w in worlds]); lab = "rare clumping (4 cells shown)"
        else:
            vals = np.array([w[k] for w in worlds]); lab = NAMES[k]
        q = np.percentile(vals, [25, 50, 75])
        per = []
        for sd in seeds:
            idx = [i for i, w in enumerate(worlds) if w["seed"] == sd]
            ss = [s[i] for i in idx]
            per.append(f"{sd}: {np.median(vals[idx]):.3g} [{ss.count(-1)}/{ss.count(0)}/{ss.count(1)}]")
        print(f"{lab:26s} {s.count(-1):6d} {s.count(0):7d} {s.count(1):6d}   {q[1]:.3g} ({q[0]:.3g}-{q[2]:.3g})   " + "; ".join(per))
    if worlds and "rare_v" in worlds[0]:
        print("\nrare clumping by distance (median over worlds; band 1 +- 0.196, 0.232, 0.137, 0.077): " +
              ", ".join(f"{d} {np.median([w['rare_v'][i] for w in worlds]):.2f} "
                        f"[{sum(w['rare_v'][i] - 1 < -RARE[i] for w in worlds)}/{sum(abs(w['rare_v'][i] - 1) <= RARE[i] for w in worlds)}/"
                        f"{sum(w['rare_v'][i] - 1 > RARE[i] for w in worlds)}]" for i, d in enumerate([1, 2, 3, 4])))
    one = np.array([w["1"] for w in worlds])
    print(f"1 cell share: median {np.median(one):.1f}% ({np.percentile(one, 25):.1f}-{np.percentile(one, 75):.1f}); BNM 8.3%")


if __name__ == "__main__":
    main(sys.argv[1])