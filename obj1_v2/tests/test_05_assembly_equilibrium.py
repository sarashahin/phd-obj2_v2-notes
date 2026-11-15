"""
test_05_assembly_equilibrium.py  (obj1_v2, written 23 Oct 2026, kernel-aware 24 Oct, version 3.1 26 Oct)

Question: did assembly reach saturation, did the final run settle, and can ranges
move at all? Reads saved *_training.npz files (one per distinct final map).

Assembly  rounds, attempts, richness at 50%, 80% and 100% of attempts, relative
          richness change over the last 20% of attempts, establishment rate early and
          late, newcomers kept and species removed per round late on. These are
          descriptions only: test_09 shows that no rule read from one assembly history
          tells saturation from fluctuation at about 30 species (26 Oct).
Final run richness at the first and last frame, species lost, cells switching state
          per frame (gains plus losses, over occupied cells).
          Agreed rule (25 Oct): final-run losses 2% or less (PASS/FAIL line).
          Runs longer than --t0 (default 10,000): species present at t0 and absent at
          the end. Agreed rule for the tmax 30,000 controls (26 Oct): none.
Twins     --twin PATTERN: for each file, the file in PATTERN with the same SIM_SEED and a
          final run that ends at t0 or earlier. Checks the assembly is identical
          (richness history, attempts, start-of-run maps, species ids) and whether the
          final run repeats frame by frame up to the twin's end. Identical assembly also
          shows that runs repeat on the machine that made them (GPU).
Spread    expected arrivals into the empty cells next to each range over the whole
          final run, from the saved counts, d and kernel. Below 1 means ranges cannot move.
          species_with_1_or_more is the share of species with an empty neighbouring cell that
          get at least one arrival there; species filling the whole grid have no such cell and
          are counted apart (species_without_edge; v3.1, 26 Oct).
Old files (no DISPERSAL_KERNEL key) ran with d = 2e-8 and four neighbours whatever their
name says (the d bug); new files use the d and kernel they record.
Ecological reading: a settled final run loses almost no species; ranges able to move need
at least one arrival per edge cell per run.
Run from obj1_v2/work (the pattern is optional and positional, as before):
    USE_GPU=0 python3 ../tests/test_05_assembly_equilibrium.py "../../results/data/data_eval_unseen/*_training.npz"
    USE_GPU=0 python3 ../tests/test_05_assembly_equilibrium.py "results/stage2a/long/*_training.npz" \
        --twin "results/stage2a/settle/*_training.npz"
Results 23 Oct (25 evaluation communities, old code): richness +9.3% over the last 20% of
attempts (7.0-11.3%), late establishment 0.71 against 0.84 early, about 110 kept and 88
removed per round late on, 8-11% of species lost in the final run, 0.70% of cells
switching per frame, 0.20 arrivals per edge cell per run (0.19-0.22).
"""
import os, sys, glob, hashlib, argparse
sys.path.insert(0, os.getcwd())
import numpy as np
import dispersal

ap = argparse.ArgumentParser()
ap.add_argument("pattern", nargs="?", default="../../results/data/data_eval_unseen/*_training.npz")
ap.add_argument("--t0", type=float, default=10000.0, help="count losses after this time in longer runs")
ap.add_argument("--twin", default=None, help="files to pair with by SIM_SEED (shorter runs of the same world)")
a = ap.parse_args()
files = sorted(glob.glob(a.pattern))
if not files: sys.exit("no files matched")
BODY = 1e-4
res = lambda c: "PASS" if c else "FAIL"
seen, rows = set(), []
for f in files:
    z = np.load(f, allow_pickle=True); k = set(z.files)
    P = z["P_last_final"].astype(bool)
    h = hashlib.md5(P.tobytes()).hexdigest()
    if h in seen: continue
    seen.add(h)
    G = P.shape[1]
    new = "DISPERSAL_KERNEL" in k
    d = float(z["DISPERSAL_RATE"]) if new else 2e-8
    kind = str(z["DISPERSAL_KERNEL"][0]) if new else "nn4"
    a_ = float(z["DISPERSAL_A"]) if new and kind != "nn4" else None; b_ = float(z["DISPERSAL_B"]) if new and kind != "nn4" else None
    ldd = float(z["LONG_DISTANCE_PROB"]) if "LONG_DISTANCE_PROB" in k else 0.0
    w = dispersal.kernel_weights(kind, a_, b_)                     # share sent to each displacement
    out = dict(file=os.path.basename(f)[:60], S=P.shape[0], kernel=kind, d=d)
    if "SIM_SEED" in k: out["seed"] = int(z["SIM_SEED"])
    if "ASM_gamma" in k and len(z["ASM_gamma"]) > 5:
        g, att, est = z["ASM_gamma"].astype(float), z["ASM_attempts"].astype(float), z["ASM_established"].astype(float)
        A = att[-1]; i80 = np.searchsorted(att, 0.8 * A); i50 = np.searchsorted(att, 0.5 * A)
        cand = np.diff(np.r_[0, att]); newk = np.diff(np.r_[0, est])
        pruned = z["ASM_pruned"].astype(float) if "ASM_pruned" in k else np.r_[1, g[:-1]] + newk - g
        last = slice(i80, None); first = slice(0, max(1, np.searchsorted(att, 0.2 * A)))
        out.update(rounds=len(g), attempts=int(A), g50=int(g[i50]), g80=int(g[i80]), gend=int(g[-1]),
                   g_change_last20=(g[-1] - g[i80]) / g[i80],
                   est_rate_first20=newk[first].sum() / max(1, cand[first].sum()),
                   est_rate_last20=newk[last].sum() / max(1, cand[last].sum()),
                   kept_per_round_last20=newk[last].mean(), removed_per_round_last20=pruned[last].mean())
    if "gamma_t" in k:
        out.update(gamma_first_frame=int(z["gamma_t"][0]), gamma_last_frame=int(z["gamma_t"][-1]))
    R = P.reshape(len(P), -1).sum(1)
    if "IBM_ext_step" in k:
        out.update(lost_in_final_run=float(((z["IBM_ext_step"] >= 0) & (R == 0)).mean()))
    if "P_t" in k:
        Pt = z["P_t"].astype(bool)
        ch = (Pt[1:] != Pt[:-1]).sum(axis=(1, 2, 3)) / np.maximum(1, Pt[:-1].sum(axis=(1, 2, 3)))
        out.update(switching_per_frame=float(ch.mean()))
        if "IBM_t" in k and float(z["IBM_t"][-1]) > a.t0:
            t = z["IBM_t"].astype(float); j = np.flatnonzero(np.isclose(t, a.t0))
            if len(j) == 1:
                at0 = Pt[j[0]].any(axis=(1, 2)); end = P.any(axis=(1, 2))
                out.update(t_end=float(t[-1]), richness_at_t0=int(at0.sum()), richness_at_end=int(end.sum()),
                           lost_after_t0=int((at0 & ~end).sum()), new_after_t0=int((~at0 & end).sum()))
            else:
                out.update(t0_frame="not recorded")
    if "B_last" in k and "IBM_t" in k:
        N = z["B_last"].astype(np.float64) / BODY
        T = float(z["IBM_t"][-1])
        conv = np.real(np.fft.ifft2(np.fft.fft2(N, axes=(1, 2)) * np.fft.fft2(w)[None], axes=(1, 2)))   # sum_y N(y) w(x - y)
        loc = (1 - ldd) * d * T * conv + ldd * d * T * N.sum(axis=(1, 2))[:, None, None] / (G * G)
        nb = lambda m: np.roll(m, 1, -1) + np.roll(m, -1, -1) + np.roll(m, 1, -2) + np.roll(m, -1, -2)
        edge = (~P) & (nb(P.astype(int)) > 0)
        per_sp = np.array([loc[s][edge[s]].mean() if edge[s].any() else np.nan for s in range(len(P))])
        occ = R > 0
        has_edge = occ & ~np.isnan(per_sp)                      # whole-grid species have no empty neighbour
        out.update(N_per_occupied_cell=float(np.median(N[P])), arrivals_per_edge_cell=float(np.nanmedian(per_sp[occ])),
                   species_with_1_or_more=float(np.mean(per_sp[has_edge] >= 1)) if has_edge.any() else float('nan'),
                   species_without_edge=int((occ & np.isnan(per_sp)).sum()))
    out["_path"] = f
    rows.append(out)

print(f"{len(rows)} distinct communities from {len(files)} files")
for r in rows:
    print("\n" + r["file"])
    print("  " + "  ".join(f"{q}={v:.3g}" if isinstance(v, float) else f"{q}={v}" for q, v in r.items() if q not in ("file", "_path")))
    if "lost_in_final_run" in r:
        print(f"  final-run losses {r['lost_in_final_run']:.1%} (agreed rule: 2% or less) {res(r['lost_in_final_run'] <= 0.02)}")
    if "lost_after_t0" in r:
        print(f"  run to t {r['t_end']:g}: richness {r['richness_at_t0']} at t {a.t0:g} and {r['richness_at_end']} at the end; "
              f"lost after t {a.t0:g}: {r['lost_after_t0']}, new after t {a.t0:g}: {r['new_after_t0']} "
              f"(agreed rule: none lost) {res(r['lost_after_t0'] == 0)}")
print("\nmedians over communities (range):")
for q in ("gend", "g_change_last20", "est_rate_first20", "est_rate_last20", "kept_per_round_last20", "removed_per_round_last20",
          "lost_in_final_run", "switching_per_frame", "N_per_occupied_cell", "arrivals_per_edge_cell", "species_with_1_or_more"):
    v = np.array([r[q] for r in rows if q in r and r[q] == r[q]], float)
    print(f"  {q:<26} " + (f"{np.median(v):.3g}  ({v.min():.3g} to {v.max():.3g}, n={v.size})" if v.size else "not in files"))

if a.twin:
    tw = sorted(glob.glob(a.twin))
    print(f"\ntwins: {len(tw)} files matched {a.twin}")
    by_seed = {}
    for f in tw:
        z = np.load(f, allow_pickle=True)
        if "SIM_SEED" in z.files and "IBM_t" in z.files and float(z["IBM_t"][-1]) <= a.t0:
            by_seed.setdefault(int(z["SIM_SEED"]), []).append(f)
    for r in rows:
        s = r.get("seed")
        if s is None: print(f"{r['file']}: no SIM_SEED, not paired"); continue
        cand = by_seed.get(s, [])
        if len(cand) != 1: print(f"{r['file']}: {len(cand)} twins with seed {s} and a run ending by t {a.t0:g}, not paired"); continue
        x, y = np.load(r["_path"], allow_pickle=True), np.load(cand[0], allow_pickle=True)
        same = lambda key: key in x.files and key in y.files and x[key].shape == y[key].shape and np.array_equal(x[key], y[key])
        keys = [q for q in ("ASM_gamma", "ASM_attempts", "ASM_established", "P_init_assembly", "species_id") if q in x.files and q in y.files]
        asm = all(same(q) for q in keys) and len(keys) >= 3
        print(f"seed {s}: {os.path.basename(r['_path'])[:50]} against {os.path.basename(cand[0])[:50]}")
        print(f"  assembly identical ({', '.join(keys)}): {res(asm)}" + ("" if asm else
              "  (differences: " + ", ".join(q for q in keys if not same(q)) + ")"))
        tx, ty = x["IBM_t"].astype(float), y["IBM_t"].astype(float); n = len(ty)
        if "P_t" in x.files and "P_t" in y.files and len(tx) >= n and np.allclose(tx[:n], ty):
            px, py = x["P_t"].astype(bool)[:n], y["P_t"].astype(bool)
            eq = [bool(np.array_equal(px[i], py[i])) for i in range(n)] if px.shape == py.shape else []
            first = eq.index(False) if eq and not all(eq) else None
            print(f"  final run frame by frame up to t {ty[-1]:g}: " + ("identical" if eq and all(eq) else
                  "shapes differ" if not eq else f"first differs at t {ty[first]:g}, {sum(eq)} of {n} frames identical") +
                  " (identical: the long run extends the twin exactly; otherwise it is a replicate of the same assembly)")
        else:
            print("  final-run frames not comparable (no P_t, or the frame times differ)")