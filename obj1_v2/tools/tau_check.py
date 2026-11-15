"""
tau_check.py  (obj1_v2, gate 2: presence threshold for observation, 30 Oct 2026; read-only, no new run)

What happens to the true maps (whole 20 x 20 world, no survey) when a species counts as present only in cells with
tau or more individuals? world_vs_bnm --tau scores the same worlds against BNM; this tool shows the world itself.
Per file and per tau (default 1, 10, 100, 1,000):
  visible species     species with tau or more individuals in at least one cell (last frame)
  species per cell    mean over the 400 cells of the species present there at tau
  true range          median cells of the visible species; share of them in 1-5 and in 51+ cells
  kept presences      presences at tau as a share of presences at 1 individual
  flicker             presences that switch state between consecutive final-run frames, per presence
                      (IBM_B frames read at tau; at tau 1 this is the frame-to-frame turnover of the maps)
Reference lines: median individuals per occupied cell (tau 1) and, per species, its largest cell count (median).
Checks (PASS/FAIL; a FAIL means the file is not read as this tool assumes, so nothing else counts):
  T1  presence at the saved detection threshold, rebuilt from B_last, equals P_last_final
  T2  the last IBM_B frame equals B_last, and the last P_t frame equals P_last_final (when saved)
Individuals = B / BODY_MASS, rounded; B is float16, exact up to 1,000 individuals.

Run from obj1_v2/work:
    python3 ../tools/tau_check.py "results/stage2a/niche_d/*_training.npz" "results/stage2a/niche_b/*_training.npz" \\
        "results/stage2a/niche_s/*_training.npz" | tee ../logs/tau_check_30oct.txt
    python3 ../tools/tau_check.py --self-test
"""
import argparse, glob, os, sys, tempfile
import numpy as np


def counts(B, body):
    return np.rint(np.asarray(B, dtype=np.float64) / body)


def label(z):
    g = lambda k, d: np.asarray(z[k]).ravel()[0].item() if k in z.files else d
    rule = str(g("FOUND_rule", "random"))
    parts = [f"seed {int(g('SIM_SEED', -1))}", f"{rule} founding",
             f"strength {float(g('INTERACTION_STRENGTH', 0.4)):g}",
             "d per species" if int(g("DISP_per_species", 0)) else f"d {float(g('DISPERSAL_RATE', -1)):g}",
             f"{int(g('attempts_total', -1))} attempts"]
    return ", ".join(parts)


def flicker(frames_n, tau):
    """Switches between consecutive frames per presence, frames_n (T, S, Y, X) individuals."""
    P = frames_n >= tau
    if len(P) < 2 or not P[:-1].any():
        return np.nan
    return float((P[1:] != P[:-1]).sum() / P[:-1].sum())


def describe(z, taus):
    body = float(z["BODY_MASS"]); thr = float(np.asarray(z["detection_threshold"]).ravel()[0]) if "detection_threshold" in z.files else 1.0
    N = counts(z["B_last"], body)
    P = np.asarray(z["P_last_final"]).astype(bool)
    checks = [("T1", np.array_equal(N >= thr, P), f"presence at the saved threshold ({thr:g}) rebuilt from B_last = P_last_final")]
    frames = None
    if "IBM_B" in z.files:
        IB = np.asarray(z["IBM_B"])
        same = IB.shape[1:] == N.shape and np.array_equal(IB[-1], np.asarray(z["B_last"]))
        if "P_t" in z.files:
            same = same and np.array_equal(np.asarray(z["P_t"])[-1].astype(bool), P)
        checks.append(("T2", same, f"last IBM_B frame = B_last{', last P_t frame = P_last_final' if 'P_t' in z.files else ''}"))
        frames = counts(IB, body)
    occ1 = N >= 1
    ref = dict(n_occ_med=float(np.median(N[occ1])) if occ1.any() else np.nan,
               peak_med=float(np.median(N.reshape(len(N), -1).max(1)[occ1.any(axis=(1, 2))])) if occ1.any() else np.nan,
               S=len(N), frames=0 if frames is None else len(frames))
    rows = []
    for t in taus:
        Pt = N >= t
        R = Pt.reshape(len(Pt), -1).sum(1); vis = R > 0
        rows.append(dict(tau=t, visible=int(vis.sum()), per_cell=float(Pt.sum(0).mean()),
                         R_med=float(np.median(R[vis])) if vis.any() else np.nan,
                         r15=float(np.mean(R[vis] <= 5)) if vis.any() else np.nan,
                         r51=float(np.mean(R[vis] >= 51)) if vis.any() else np.nan,
                         kept=float(Pt.sum() / max(occ1.sum(), 1)),
                         flicker=flicker(frames, t) if frames is not None else np.nan))
    return checks, ref, rows


def report(files, taus):
    ok = True
    for f in files:
        z = np.load(f, allow_pickle=True)
        miss = [k for k in ("B_last", "P_last_final", "BODY_MASS") if k not in z.files]
        if miss:
            print(f"\n{os.path.basename(f)}: lacks {miss}; skipped"); continue
        checks, ref, rows = describe(z, taus)
        print(f"\n{os.path.basename(f)}\n  {label(z)}; {ref['S']} species rows, {ref['frames']} final-run frames")
        for name, good, text in checks:
            ok &= bool(good); print(f"  {name} {text}: {'PASS' if good else 'FAIL'}")
        print(f"  individuals per occupied cell (tau 1), median {ref['n_occ_med']:.0f}; largest cell of each species, median {ref['peak_med']:.0f}")
        print("     tau   visible species   species per cell   true range median   1-5 cells   51+ cells   kept presences   flicker")
        for r in rows:
            print(f"  {r['tau']:>6g}   {r['visible']:>15d}   {r['per_cell']:>16.1f}   {r['R_med']:>17.0f}   {r['r15']:>9.1%}   "
                  f"{r['r51']:>9.1%}   {r['kept']:>14.1%}   {r['flicker']:>7.3f}")
    print("\nall checks passed" if ok else "\nCHECK FAILED")
    return ok


def self_test():
    """Planted world with known individuals: every printed number has a hand value."""
    ok = True
    G = 4
    N = np.zeros((3, G, G))
    N[0, 0, 0], N[0, 0, 1], N[0, 1, 1] = 5000, 50, 3          # species 0: 3 cells at tau 1, 2 at 10, 1 at 100 and 1,000
    N[1, 2, 2] = 9                                            # species 1: 1 cell at tau 1, gone at 10
    N[2][:] = 2000                                            # species 2: all 16 cells at every tau up to 1,000
    frames = np.stack([N, N, N]); frames[1, 0, 1, 1] = 0       # one presence of species 0 vanishes in frame 2 ...
    frames[1, 1, 3, 3] = 20                                   # ... and species 1 appears in a new cell with 20
    frames[0, 2, 0, 0] = 0                                    # species 2 misses one cell in frame 1 only
    with tempfile.TemporaryDirectory() as t:
        B = (frames * 1e-4).astype(np.float16)
        np.savez(f"{t}/w.npz", B_last=B[-1], P_last_final=(N >= 1).astype(np.uint8), BODY_MASS=np.float32(1e-4),
                 detection_threshold=np.float32(1), IBM_B=B, P_t=(frames >= 1).astype(np.uint8))
        z = np.load(f"{t}/w.npz")
        checks, ref, rows = describe(z, (1, 10, 100, 1000))
        r = {x["tau"]: x for x in rows}
        # flicker at tau 1: frame 1 -> 2: 3 switches, 2 -> 3: 2 switches; presences in frames 1, 2: 19 + 20 = 39
        # at tau 10 the 3-individual cell never counts: 2 + 1 switches; presences 17 + 19 = 36
        good = (all(c[1] for c in checks) and len(checks) == 2
                and [r[t]["visible"] for t in (1, 10, 100, 1000)] == [3, 2, 2, 2]
                and np.isclose(r[1]["per_cell"], 20 / 16) and np.isclose(r[10]["per_cell"], 18 / 16) and np.isclose(r[100]["per_cell"], 17 / 16)
                and np.isclose(r[1]["kept"], 1) and np.isclose(r[10]["kept"], 18 / 20) and np.isclose(r[1000]["kept"], 17 / 20)
                and np.isclose(r[1]["flicker"], 5 / 39) and np.isclose(r[10]["flicker"], 3 / 36)
                and np.isclose(r[1]["r15"], 2 / 3) and np.isclose(r[1]["r51"], 0) and r[1]["R_med"] == 3
                and ref["n_occ_med"] == 2000 and ref["peak_med"] == 2000)
        print(f"S1 planted world: visible species 3, 2, 2, 2; species per cell, kept presences, flicker and range shares "
              f"equal the hand values; T1 and T2 PASS: {'PASS' if good else 'FAIL'}")
        ok &= good
        np.savez(f"{t}/m.npz", B_last=B[-1], P_last_final=np.zeros_like(N, dtype=np.uint8), BODY_MASS=np.float32(1e-4))
        checks, _, _ = describe(np.load(f"{t}/m.npz"), (1,))
        good = not checks[0][1]
        print(f"S2 planted error (P_last_final not from B_last) caught by T1: {'PASS' if good else 'FAIL'}")
        ok &= good
        B2 = B.copy(); B2[-1, 2, 0, 0] = np.float16(0.0)
        np.savez(f"{t}/m2.npz", B_last=B[-1], P_last_final=(N >= 1).astype(np.uint8), BODY_MASS=np.float32(1e-4), IBM_B=B2)
        checks, _, _ = describe(np.load(f"{t}/m2.npz"), (1,))
        good = checks[0][1] and not checks[1][1]
        print(f"S3 planted error (last frame differs from B_last) caught by T2: {'PASS' if good else 'FAIL'}")
        ok &= good
        Nb = np.array([1, 10, 100, 1000], float).reshape(1, 2, 2)    # counts exactly at each tau
        np.savez(f"{t}/b.npz", B_last=(Nb * 1e-4).astype(np.float16), P_last_final=(Nb >= 1).astype(np.uint8),
                 BODY_MASS=np.float32(1e-4))
        _, _, rows = describe(np.load(f"{t}/b.npz"), (1, 10, 100, 1000))
        good = [x["kept"] for x in rows] == [1.0, 0.75, 0.5, 0.25]
        print(f"S4 a cell with exactly tau individuals counts as present (kept 4, 3, 2, 1 of 4): {'PASS' if good else 'FAIL'}")
        ok &= good
    print("all passed" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--tau", type=float, nargs="+", default=[1, 10, 100, 1000])
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if any(t < 1 for t in a.tau):
        sys.exit("STOP: tau is a number of individuals, 1 or more")
    files = sorted({f for pat in a.files for f in glob.glob(pat)})
    if not files:
        sys.exit("no files matched")
    sys.exit(0 if report(files, a.tau) else 1)