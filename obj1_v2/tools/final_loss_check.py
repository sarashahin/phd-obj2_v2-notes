#!/usr/bin/env python3
"""
final_loss_check.py  (obj1_v2, real run, cause check for the test_05 deviation; 5 Nov 2026, Sara's yes to the checks)

Why: test_05 on real_train (5 Nov) failed 8 of 100 worlds (seeds 5019, 5030, 5061, 5063, 5069, 5080, 5084, 5096) on the
agreed rule "final-run losses 2% or less". In each world exactly one species was lost in the 10,000-step final run
(1 of 23-49 species = 2.0-4.3%). At 30-50 species the 2% rule allows no loss at all, so a single loss fails a world.
habitat_track_check skipped one species ("all or no cells") in exactly these 8 worlds. The question is whether each lost
species is the normal tail of a moving community frozen at one moment, or a sign of a fault.

What it reads (training files only; writes nothing): the final-run loss (IBM_ext_step >= 0 and no cell in P_last_final,
test_05's definition), the life table (LT_id, LT_round_in, LT_peak_cells), the round-end maps (ASM_P_*), the map at the
start of the final run (P_init_assembly), the growth field on the final habitat (ENV_r_field) and the round history
(ASM_round, ASM_gamma, ASM_pruned).

For each lost species: age in rounds at the end of the assembly, cells at the start of the final run and their rank in
its world, cells at the last 5 round ends, footprint on the final habitat (cells with growth rate above 0), largest
growth rate, and the BNM year of loss (200 steps = 1 year).

Reading rule (written 5 Nov, before the tool was run on any real file):
  A lost species is explained if at least one holds:
    young      arrived in the last 20 rounds of the assembly;
    small      5 cells or fewer, or in the smallest 10% of its world, at the start of the final run;
    shrinking  its cells at the last round end are half or less of its largest count over the last 5 round ends;
    no habitat no cell with growth rate above 0 on the final habitat (it lived on arrivals only).
  All lost species explained -> the normal tail of a moving community; no simulator change; Objective 2 export drops
  species absent on a target frame. Any species explained by none of these -> stop; it goes to Sara with its record.
  Calibration printed beside it: the losses expected in 10,000 steps at the assembly's own removal rate over the last half
  (2 windows of 5,000 steps), against the losses seen.

Run from obj1_v2/work:
    python3 ../tools/final_loss_check.py --self-test | tail -1
    python3 ../tools/final_loss_check.py "results/stage2a/real_train/*_training.npz" | tee ../logs/final_loss_real_train.txt
"""
import argparse, glob, os, sys, tempfile
import numpy as np

YOUNG, SMALL_CELLS, SMALL_Q, LAST = 20, 5, 0.10, 5


def round_maps(z, k):
    """(ids, P) at the k-th recorded round end; P (n, Y, X) bool."""
    o = np.asarray(z["ASM_P_offsets"]); a, b = int(o[k]), int(o[k + 1])
    Y, X = np.asarray(z["P_last_final"]).shape[1:]
    bits = np.asarray(z["ASM_P_bits"])[a:b]
    P = np.unpackbits(bits, axis=1, count=Y * X).astype(bool).reshape(b - a, Y, X)
    return np.asarray(z["ASM_P_ids"])[a:b], P


def check_file(z):
    P = np.asarray(z["P_last_final"]).astype(bool)
    S = len(P)
    R = P.reshape(S, -1).sum(1)
    ext = np.asarray(z["IBM_ext_step"])
    lost = np.flatnonzero((ext >= 0) & (R == 0))
    ids = np.asarray(z["species_id"]).astype(np.int64)
    start = np.asarray(z["P_init_assembly"]).astype(bool).reshape(S, -1).sum(1)
    F = np.asarray(z["ENV_r_field"]).astype(float)
    lt_id, lt_in = np.asarray(z["LT_id"]).astype(np.int64), np.asarray(z["LT_round_in"]).astype(np.int64)
    lt_peak = np.asarray(z["LT_peak_cells"]) if "LT_peak_cells" in z.files else None
    rounds = np.asarray(z["ASM_round"]).astype(np.int64)
    last_round = int(rounds[-1])
    n_rec = len(np.asarray(z["ASM_P_round"]))
    hist = [round_maps(z, k) for k in range(max(0, n_rec - LAST), n_rec)]
    q10 = np.quantile(start, SMALL_Q)
    out = []
    for i in lost:
        sid = int(ids[i])
        j = np.flatnonzero(lt_id == sid)
        r_in = int(lt_in[j[0]]) if len(j) else None
        age = None if r_in is None or r_in < 0 else last_round - r_in
        cells_hist = []
        for hid, hP in hist:
            m = np.flatnonzero(hid == sid)
            cells_hist.append(int(hP[m[0]].sum()) if len(m) else 0)
        fp = int((F[i] > 0).sum())
        why = []
        if age is not None and age < YOUNG: why.append("young")
        if start[i] <= SMALL_CELLS or start[i] <= q10: why.append("small")
        if cells_hist and max(cells_hist) > 0 and cells_hist[-1] <= 0.5 * max(cells_hist): why.append("shrinking")
        if fp == 0: why.append("no habitat")
        out.append(dict(sid=sid, age=age, start=int(start[i]), rank=float((start <= start[i]).mean()), hist=cells_hist,
                        fp=fp, rmax=float(F[i].max()), year=float(ext[i]) / 200.0,
                        peak=int(lt_peak[j[0]]) if lt_peak is not None and len(j) else None, why=why))
    g = np.asarray(z["ASM_gamma"]).astype(float)
    pr = np.asarray(z["ASM_pruned"]).astype(float) if "ASM_pruned" in z.files else None
    h = len(g) // 2
    rate = float(pr[h:].sum() / max(1.0, g[h - 1:-1].sum())) if pr is not None and h >= 1 else float("nan")
    expected = S * (1 - (1 - rate) ** 2) if rate == rate else float("nan")
    return dict(S=S, lost=out, small_alive=int(((start <= SMALL_CELLS) | (start <= q10)).sum()) - sum("small" in o["why"] for o in out),
                rate=rate, expected=expected)


def run(files):
    tot_S = tot_lost = 0; tot_exp = 0.0; unexplained = []
    by_reason = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        seed = int(np.asarray(z["SIM_SEED"]).item()) if "SIM_SEED" in z.files else None
        r = check_file(z)
        tot_S += r["S"]; tot_lost += len(r["lost"]); tot_exp += r["expected"] if r["expected"] == r["expected"] else 0
        if not r["lost"]:
            continue
        print(f"seed {seed}: {r['S']} species, {len(r['lost'])} lost in the final run; assembly removal rate over the last half "
              f"{1000 * r['rate']:.1f} per 1,000 per round -> {r['expected']:.2f} losses expected in 10,000 steps at that rate")
        for o in r["lost"]:
            v = ", ".join(o["why"]) if o["why"] else "NONE"
            for w in o["why"] or ["none"]:
                by_reason[w] = by_reason.get(w, 0) + 1
            if not o["why"]:
                unexplained.append((seed, o["sid"]))
            print(f"  species {o['sid']}: lost in BNM year {o['year']:.0f} of 50; arrived {o['age']} rounds before the end"
                  f"; {o['start']} cells at the start of the final run (smallest {100 * o['rank']:.0f}% of its world); "
                  f"last {len(o['hist'])} round ends {o['hist']}; largest range ever {o['peak']}; footprint on the final habitat "
                  f"{o['fp']} cells, largest growth rate {o['rmax']:+.3f} -> explained by: {v}")
    print(f"\n{len(files)} files; {tot_S} species at the start of the final runs; {tot_lost} lost "
          f"({100 * tot_lost / max(1, tot_S):.2f}%); expected at the assembly's own removal rate: {tot_exp:.0f} "
          f"(seen / expected {tot_lost / max(1e-9, tot_exp):.3f})")
    print("reasons (a species can have more than one): " + (", ".join(f"{k} {v}" for k, v in sorted(by_reason.items())) or "none"))
    if not tot_lost:
        print("no species was lost in any final run")
    elif unexplained:
        print(f"STOP: {len(unexplained)} lost species explained by none of the rules: {unexplained}; to Sara before any use")
    else:
        print("every lost species is explained: the normal tail of a moving community frozen at one moment; no simulator change")
    return not unexplained


# ---------------------------------------------------------------- self-test
def planted(path, case):
    """A small world: 20 species, 30 rounds, 3 recorded round ends. Species index L (id 110, 40 cells, mid-sized: 4 species
    are smaller) is the one lost in the final run; each case changes one thing about it."""
    Y = X = 20; S = 20; L = 10; rng = np.random.default_rng(1)
    ids = np.arange(100, 100 + S)
    sizes = [300, 280, 260, 240, 220, 200, 180, 160, 140, 120, 40, 100, 90, 80, 70, 60, 30, 20, 10, 8]
    start = np.zeros((S, Y, X), bool)
    for s, n in enumerate(sizes):
        start[s].flat[rng.choice(Y * X, n, replace=False)] = True
    ext = np.full(S, -1.0); ext[L] = 3000.0
    F = np.where(start, 0.3, -0.1).astype(np.float32)
    lt_in = {int(i): 2 for i in ids}
    hist_cells = [40, 40, 40]
    if case == "young": lt_in[110] = 26
    if case == "small": start[L] = False; start[L].flat[:3] = True; hist_cells = [3, 3, 3]
    if case == "shrinking": hist_cells = [60, 50, 20]
    if case == "nohab": F[L] = -0.05
    final = start.copy(); final[L] = False
    if case == "comeback": final[L].flat[0] = True          # extinction step set but present at the end: not a loss
    rec, offs, rid, rbits = [27, 28, 29], [0], [], []
    for n in hist_cells:
        P = start.copy(); P[L] = False; P[L].flat[:n] = True
        rid.append(ids); rbits.append(np.packbits(P.reshape(S, -1), axis=1)); offs.append(offs[-1] + S)
    g = np.full(30, S); pr = np.zeros(30); pr[15:] = 1                     # 15 removals over 15 rounds at 20 species
    np.savez(path, P_last_final=final.astype(np.uint8), IBM_ext_step=ext, species_id=ids, P_init_assembly=start.astype(np.uint8),
             ENV_r_field=F, LT_id=np.array(sorted(lt_in)), LT_round_in=np.array([lt_in[k] for k in sorted(lt_in)]),
             LT_peak_cells=np.array(sizes) + 5, ASM_round=np.arange(30), ASM_P_round=np.array(rec),
             ASM_P_offsets=np.array(offs), ASM_P_ids=np.concatenate(rid), ASM_P_bits=np.concatenate(rbits),
             ASM_gamma=g, ASM_pruned=pr, SIM_SEED=np.int64(1))


def self_test():
    res = []
    def check(name, ok, what):
        res.append(ok); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")
    with tempfile.TemporaryDirectory() as tmp:
        get = lambda case: (planted(f"{tmp}/{case}.npz", case), check_file(np.load(f"{tmp}/{case}.npz")))[1]
        r = get("plain")
        o = r["lost"][0] if r["lost"] else {}
        check("T1", len(r["lost"]) == 1 and o.get("sid") == 110 and o.get("why") == [] and o.get("age") == 27
              and o.get("start") == 40 and o.get("hist") == [40, 40, 40] and o.get("fp") == 40,
              "an old, stable, mid-sized species with habitat is found as the loss and explained by none (the stop case)")
        r = get("young"); check("T2", r["lost"] and r["lost"][0]["why"] == ["young"], "a species that arrived 3 rounds before the end is 'young'")
        r = get("small"); check("T3", r["lost"] and "small" in r["lost"][0]["why"], "a species with 3 cells at the start of the final run is 'small'")
        r = get("shrinking"); check("T4", r["lost"] and r["lost"][0]["why"] == ["shrinking"], "60 -> 50 -> 20 cells over the last round ends is 'shrinking'")
        r = get("nohab"); check("T5", r["lost"] and r["lost"][0]["why"] == ["no habitat"], "no cell with growth above 0 on the final habitat is 'no habitat'")
        r = get("comeback"); check("T6", r["lost"] == [], "a species with an extinction step but present at the end is not a loss (test_05's definition)")
        r = get("plain")
        check("T7", abs(r["rate"] - 15 / (20 * 15)) < 1e-9 and abs(r["expected"] - 20 * (1 - 0.95 ** 2)) < 1e-9,
              f"removal rate over the last half 15 / 300 = {r['rate']:.3f}; expected losses in two windows {r['expected']:.2f}")
        planted(f"{tmp}/young.npz", "young"); planted(f"{tmp}/plain.npz", "plain")
        ok = run([f"{tmp}/young.npz"]); ok2 = run([f"{tmp}/plain.npz"])
        check("T8", ok and not ok2, "the verdict is 'explained' for the young case and STOP for the stable case")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    files = sorted({f for p in a.patterns for f in glob.glob(p)})
    if not files:
        sys.exit("STOP: no file matched")
    sys.exit(0 if run(files) else 2)