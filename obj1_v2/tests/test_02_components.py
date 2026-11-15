"""
test_02_components.py  (obj1_v2, written 24 Oct 2026)

Question: do the building blocks do what they say, and where do they work against the
aim? Known-answer tests on small planted inputs, in seconds.
Interactions  I1 draw_interactions  I2 expand_RC orientation  I3 prune_extinct
              I4 the 16 partners saved for the graph encoder (ties broken by the sort)
Environment   E1 field moments  E2 realised correlation against l  E3 field depends only on
              (seed, position)  E4 fields keyed by list position (the bug fixed in assembly v2)
Records       O1 presence at N = tau under float32 storage  O2 sample_obs_masks
History       H1 extinction times  H2 species_event_times  H3 take_last_snapshots
FLAG lines are design points for Stage 2 and 3: I4 export, E2 length scale on a 20 x 20
torus, E4 (generate_spatial_r itself still works by position; assembly v2 calls it with
one seed per species), O1 threshold type, O2 records not nested and same seed everywhere.
Run from obj1_v2/work:   USE_GPU=0 python3 ../tests/test_02_components.py
Results 24 Oct on Sara's machine: no FAIL; FLAG on I4, E2, E4, O1 (tau 1 and 2), O2.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import config, assembly_utils as au, environment as env
import run_all_rps as rar                      # helper functions only; main() does not run on import
G, BODY, P_C, A_C = config.NUM_PATCHES_Y, config.BODY_MASS, config.CONNECTANCE, config.INTERACTION_STRENGTH
res = lambda c: "PASS" if c else "FAIL"
flag = lambda c: "FLAG" if c else "ok"
print(f"CONNECTANCE {P_C}, INTERACTION_STRENGTH {A_C}, grid {G}x{config.NUM_PATCHES_X}")

# ---------- interactions ----------
rng = np.random.default_rng(0); k = 200_000
row, col = au.draw_interactions(k, rng=rng)
sd = np.sqrt(P_C * (1 - P_C) / k)
fr, fc = (row > 0).mean(), (col > 0).mean()
rc = np.corrcoef(row > 0, col > 0)[0, 1]
print(f"I1 draw_interactions: share non-zero {fr:.4f} / {fc:.4f} (target {P_C}) {res(abs(fr - P_C) < 4 * sd and abs(fc - P_C) < 4 * sd)}; "
      f"values {sorted(map(float, set(np.r_[row, col])))} {res(set(np.r_[row, col]) <= {0.0, A_C})}; "
      f"row-col link r = {rc:+.4f} {res(abs(rc) < 0.01)}; k = 0 gives empty arrays {res(all(a.size == 0 for a in au.draw_interactions(0, rng=rng)))}")
r0, C0 = np.ones(2), np.eye(2)
r1, C1 = au.expand_RC(r0, C0, 1.0, np.array([A_C, 0.0]), np.array([0.0, A_C]))
want = np.array([[1, 0, 0], [0, 1, A_C], [A_C, 0, 1]])
g = r1 - C1 @ np.array([0.0, 0.0, 1.0])                              # only the newcomer present, biomass 1
print(f"I2 expand_RC: C[new, j] = effect of resident j on the newcomer, C[j, new] = effect of the newcomer on j "
      f"{res(np.array_equal(C1, want))}; with only the newcomer present, growth of residents {g[:2]} (expected [1, 0.6]) "
      f"{res(np.allclose(g[:2], [1.0, 1 - A_C]))}; r appended {res(np.array_equal(r1, [1, 1, 1]))}")
S = 6; Cp = np.arange(S * S, dtype=float).reshape(S, S); rp = np.arange(S, dtype=float)
Np = np.arange(S)[:, None, None] * np.ones((S, 2, 2)); other = np.arange(7)
alive = np.array([True, False, True, True, False, True])
r2, C2, N2, o2, keep = au.prune_extinct(alive, rp, Cp, Np, other)
print(f"I3 prune_extinct: keeps order {res(np.array_equal(keep, [0, 2, 3, 5]))}; C is the kept block "
      f"{res(np.array_equal(C2, Cp[np.ix_(keep, keep)]))}; states follow {res(np.array_equal(N2[:, 0, 0], keep) and np.array_equal(r2, keep))}; "
      f"other-length arrays untouched {res(np.array_equal(o2, other))}")
S = 3000; rng = np.random.default_rng(1)
Cs = np.eye(S); m = (rng.random((S, S)) < P_C) & ~np.eye(S, dtype=bool); Cs[m] = A_C
idx, w = rar.topk_interactions(Cs, k=16)
deg = m.sum(1)
print(f"I4 topk_interactions (the 16 partners saved for the graph encoder): weights {sorted(round(float(v), 4) for v in set(w.ravel().tolist()))}; "
      f"median competitors per species {int(np.median(deg))}, so 16 is {16 / np.median(deg):.1%} of them; "
      f"mean partner position / S {(idx / S).mean():.2f} (0.50 if picked at random); choice set by the sort among ties "
      f"{flag(len(set(w.ravel())) == 1)}")

# ---------- environment ----------
ls_list = (2.5, 5.0, 10.0, 20.0); corr = {}
for L in ls_list:
    f = env.generate_spatial_r(500, G, G, L, np.ones(500), 0.002, seed=7)
    z = (f - f.mean((1, 2), keepdims=True)) / f.std((1, 2), keepdims=True)
    corr[L] = [float(np.mean(z * np.roll(z, d, axis=2))) for d in (1, 2, 5)]
f = env.generate_spatial_r(300, G, G, 5.0, np.ones(300), 0.002, seed=7)
print(f"E1 moments: species means {f.mean((1, 2)).min():.6f}-{f.mean((1, 2)).max():.6f}, sd {f.std((1, 2)).min():.6f}-{f.std((1, 2)).max():.6f} "
      f"(target 1, {np.sqrt(0.002):.6f}) {res(np.allclose(f.mean((1, 2)), 1) and np.allclose(f.std((1, 2)), np.sqrt(0.002)))}")
print("E2 correlation at 1, 2, 5 cells: " + "; ".join(f"l={L}: {c[0]:.2f} {c[1]:.2f} {c[2]:.2f}" for L, c in corr.items())
      + f" | l = 5, 10, 20 give the same fields {flag(max(abs(corr[5.0][i] - corr[20.0][i]) for i in range(3)) < 0.1)}")
a = env.generate_spatial_r(60, G, G, 5.0, np.ones(60), 0.002, seed=789)
b = env.generate_spatial_r(260, G, G, 5.0, np.ones(260), 0.002, seed=789)
print(f"E3 field of position i depends only on (seed, i): first 60 of 260 equal the 60 {res(np.array_equal(a, b[:60]))} "
      f"(lets later tests rebuild the fields of positions beyond the final list)")
moved = np.allclose(a[4], env.generate_spatial_r(59, G, G, 5.0, np.ones(59), 0.002, seed=789)[3])
print(f"E4 keyed by list position: remove species 3, the species that moves from position 4 to 3 keeps its field {moved} "
      f"{flag(not moved)}")

# ---------- observations ----------
for thr in (1.0, 2.0, 3.0, 5.0):
    B32 = (np.array([int(thr)]) * BODY).astype(np.float32)
    as_coded, f64 = bool((B32 >= thr * BODY)[0]), bool((B32 >= np.float64(thr) * BODY)[0])
    print(f"O1 tau={thr:g}, N = tau: present with the threshold as coded (Python float) {res(as_coded)}; "
          f"with a NumPy float64 threshold {f64} {flag(not f64)}")
rng = np.random.default_rng(5)
P = (rng.random((200, G, G)) < 0.03).astype(np.uint8); P[0] = 0; P[1, 3, 3] = 1
M = rar.sample_obs_masks(P, budgets=(1, 5, 10))
R = P.reshape(200, -1).sum(1)
sub = all(((M[k] > 0) <= (P > 0)).all() for k in M)
size = all(np.array_equal(M[f"obs_mask_{b}"].reshape(200, -1).sum(1), np.minimum(b, R)) for b in (1, 5, 10))
nest = np.mean([((M["obs_mask_5"][s] > 0) <= (M["obs_mask_10"][s] > 0)).all() for s in range(200) if R[s] > 10])
M2 = rar.sample_obs_masks(P, budgets=(1, 5, 10))
print(f"O2 sample_obs_masks: records inside the range {res(sub)}; records = min(budget, range) {res(size)}; "
      f"5-record sets inside the 10-record sets for {nest:.0%} of species {flag(nest < 1)}; "
      f"same seed 1234 in every call, so the same truth gives the same records {flag(all(np.array_equal(M[k], M2[k]) for k in M))}")

# ---------- history helpers ----------
T, S = 6, 3; t = np.arange(1, T + 1) * 200
Bh = np.zeros((T, S, G, G), np.float32); Bh[:, 0, 0, 0] = 5 * BODY; Bh[:3, 1, 1, 1] = 5 * BODY; Bh[[0, 1, 4, 5], 2, 2, 2] = 5 * BODY
ext = rar._extinction_times_thresholded(Bh, t, 3 * BODY)
print(f"H1 extinction times: never lost -1, lost at frame 4 -> {t[3]}, lost at frame 3 then back -> {t[2]}: got {ext.tolist()} "
      f"{res(ext.tolist() == [-1.0, float(t[3]), float(t[2])])}")
ev = rar.species_event_times((Bh >= 3 * BODY).astype(np.uint8), t)
print(f"H2 species_event_times: first {ev['T_first_any'].tolist()}, last {ev['T_last_any'].tolist()}, recolonisations "
      f"{ev['n_recolonizations'].tolist()} {res(ev['n_recolonizations'].tolist() == [0, 0, 1] and ev['T_last_any'].tolist() == [1200.0, 600.0, 1200.0])}")
Bk, tk = rar.take_last_snapshots(Bh, t, k=4)
print(f"H3 take_last_snapshots: times {tk.tolist()} {res(tk.tolist() == t[-4:].tolist() and np.array_equal(Bk, Bh[-4:]))}")
