"""
test_01_dispersal_kernel.py  (obj1_v2, written 24 Oct 2026)

Question: does dispersal.py v2 send the right number of propagules to the right cells,
with the d that is set at run time, for every kernel?
  A  four-neighbour kernel against the untouched original dispersal.py (../original)
  B  each kernel: rows sum to 1, source 0, non-negative, symmetric, the same for every
     source, equal to an independent loop over periodic images; shares by distance
  C  d set at run time reaches the flux (the d bug of 3 Oct)
  D  total arrivals = d x N for every kernel and long-distance share
  E  switching kernel at run time and back gives the same flux
  F  one real IBM step sends individuals beyond 2 cells only with the fat-tailed kernel
  G  bad settings are refused
Ecological reading: the kernel decides where propagules land, d how many leave. A PASS
here says the plumbing is right; the d and kernel values are chosen in Stage 2 against
range spread and BNM co-occupancy.
Run from obj1_v2/work:   USE_GPU=0 python3 ../tests/test_01_dispersal_kernel.py
Results 24 Oct on Sara's machine: no FAIL; A differs from her 30 Sep file by 3e-16
(rounding order), bit-identical to the 21 Sep code that made the existing worlds.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import importlib.util
import numpy as np
import config, dispersal, models_ibm
from gpu_patch import np as xp
host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
dense = lambda M: host(M.toarray()) if hasattr(M, "toarray") else host(M)
ok = lambda c: "PASS" if c else "FAIL"
Gy, Gx, BODY = config.NUM_PATCHES_Y, config.NUM_PATCHES_X, config.BODY_MASS
d0, N0 = float(dispersal.DISPERSAL_RATE), 10_000
print(f"d = {d0:g}; kernel at import = {dispersal.DISPERSAL_KERNEL}; images = {dispersal.DISPERSAL_IMAGES}")

spec = importlib.util.spec_from_file_location("dispersal_original", "../original/dispersal.py")
orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)
rng = np.random.default_rng(1)
B = xp.asarray(rng.integers(0, 50, size=(60, Gy, Gx)) * BODY)        # 60 species, random counts
for p in (0.0, 0.2):
    dispersal.LONG_DISTANCE_PROB = orig.LONG_DISTANCE_PROB = p
    new, old = host(dispersal.compute_dispersal(B)), host(orig.compute_dispersal(B))
    rel = np.abs(new - old).max() / np.abs(old).max()
    print(f"A  nn4 against original dispersal.py, pLDD {p}: identical {np.array_equal(new, old)}, "
          f"max relative difference {rel:.1e} {ok(rel <= 1e-12)}")
dispersal.LONG_DISTANCE_PROB = 0.0

def brute_row(kind, a, b, n, sy=10, sx=10):                          # independent: loop over every image
    w = np.zeros((Gy, Gx))
    for y in range(Gy):
        for x in range(Gx):
            if (y, x) == (sy, sx): continue
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    dy, dx = y - sy + i * Gy, x - sx + j * Gx
                    r2 = dy * dy + dx * dx
                    w[y, x] += (1 + r2 / a ** 2) ** (-b) if kind == "2dt" else np.exp(-r2 / (2 * a ** 2))
    return (w / w.sum()).ravel()

expected = {("2dt", 1.0, 1.75): (0.546, 0.247, 0.075, 2.41), ("2dt", 1.0, 2.5): (0.790, 0.064, 0.007, 1.47)}
for kind, a, b in (("nn4", None, None), ("2dt", 1.0, 1.75), ("2dt", 1.0, 2.5), ("2dt", 0.5, 1.75), ("gauss", 2.0, None)):
    K = dense(dispersal.unit_kernel_matrix(kind, a, b))
    rows, diag, neg, sym = np.abs(K.sum(1) - 1).max(), np.abs(np.diag(K)).max(), K.min(), np.abs(K - K.T).max()
    s = int(rng.integers(Gy * Gx)); sy, sx = divmod(s, Gx)
    shift = np.abs(K[s].reshape(Gy, Gx) - np.roll(K[0].reshape(Gy, Gx), (sy, sx), axis=(0, 1))).max()
    sm = dispersal.kernel_summary(kind, a, b)
    line = (f"B  {kind:<5} a={a} b={b}: rows sum to 1 {ok(rows < 1e-12)}, source 0 {ok(diag == 0)}, "
            f"non-negative {ok(neg >= 0)}, symmetric {ok(sym < 1e-15)}, same for every source {ok(shift < 1e-15)}; "
            f"4 nearest {sm['nn4']:.3f}, 8 nearest {sm['nn8']:.3f}, beyond 2 {sm['beyond2']:.3f}, "
            f"beyond 5 {sm['beyond5']:.3f}, mean distance {sm['mean_distance']:.2f}")
    if kind == "nn4":
        line += f"; matches the 1/4 rule {ok(np.allclose(K[s].reshape(Gy, Gx)[[(sy-1)%Gy, (sy+1)%Gy, sy, sy], [sx, sx, (sx-1)%Gx, (sx+1)%Gx]], 0.25) and np.isclose(K[s].sum(), 1))}"
    else:
        diff = np.abs(K[10 * Gx + 10] - brute_row(kind, a, b, dispersal.DISPERSAL_IMAGES)).max()
        line += f"; equals an independent loop over images {ok(diff < 1e-12)} (max diff {diff:.1e})"
    if (kind, a, b) in expected:
        e = expected[(kind, a, b)]
        line += f"; matches the 24 Oct table {ok(all(abs(u - v) <= 0.0015 for u, v in zip((sm['nn8'], sm['beyond2'], sm['beyond5']), e[:3])) and abs(sm['mean_distance'] - e[3]) <= 0.015)}"
    print(line)

def one_cell(y=5, x=5):
    Bc = xp.zeros((1, Gy, Gx)); Bc[0, y, x] = N0 * BODY
    return host(dispersal.compute_dispersal(Bc))[0] / BODY

base = one_cell()
dispersal.DISPERSAL_RATE = 50 * d0
print(f"C  d set at run time reaches the flux without rebuilding: x{one_cell()[4, 5] / base[4, 5]:.1f} (expected x50) "
      f"{ok(np.isclose(one_cell()[4, 5], 50 * base[4, 5], rtol=1e-12))}")
dispersal.DISPERSAL_RATE = d0

for kind, a, b in (("nn4", None, None), ("2dt", 1.0, 1.75), ("gauss", 2.0, None)):
    dispersal.DISPERSAL_KERNEL = kind
    if a is not None: dispersal.DISPERSAL_A = a
    if b is not None: dispersal.DISPERSAL_B = b
    for p in (0.0, 0.2):
        dispersal.LONG_DISTANCE_PROB = p
        inc = one_cell()
        print(f"D  {kind:<5} pLDD {p}: total arrivals per step {inc.sum():.6e} = d x N {d0 * N0:.6e} "
              f"{ok(np.isclose(inc.sum(), d0 * N0, rtol=1e-12))}; source cell receives {inc[5, 5]:.2e}")
dispersal.LONG_DISTANCE_PROB = 0.0

dispersal.DISPERSAL_KERNEL = "nn4"; f1 = one_cell()
dispersal.DISPERSAL_KERNEL = "2dt"; dispersal.DISPERSAL_A, dispersal.DISPERSAL_B = 1.0, 1.75; f2 = one_cell()
dispersal.DISPERSAL_KERNEL = "nn4"; f3 = one_cell()
print(f"E  switching kernel at run time: 2dt differs from nn4 {ok(not np.allclose(f1, f2))}; back to nn4 gives the same flux {ok(np.array_equal(f1, f3))}")

oy = np.abs(np.where(np.arange(Gy) <= Gy // 2, np.arange(Gy), np.arange(Gy) - Gy))
ox = np.abs(np.where(np.arange(Gx) <= Gx // 2, np.arange(Gx), np.arange(Gx) - Gx))
far = np.roll(np.maximum(oy[:, None], ox[None, :]) > 2, (10, 10), axis=(0, 1))     # cells > 2 from (10, 10)
d_test = 0.1
dispersal.DISPERSAL_RATE = d_test
for kind in ("nn4", "2dt"):
    dispersal.DISPERSAL_KERNEL = kind
    N_init = np.zeros((1, Gy, Gx), int); N_init[0, 10, 10] = N0
    m = models_ibm.IBMModel(np.array([1.0]), np.array([[1.0]]), initial_N=xp.asarray(N_init),
                            nsteps=1, record_step=1, record_mode="none", seed=3)
    m.run()
    arrived = int(host(m.N)[0][far].sum())
    expect = d_test * N0 * dispersal.kernel_summary()["beyond2"]
    tol = 5 * np.sqrt(max(expect, 1.0))
    print(f"F  IBM one step, {kind}, d = {d_test}: individuals arriving more than 2 cells away {arrived}, "
          f"expected {expect:.1f} {ok(abs(arrived - expect) <= tol if expect > 0 else arrived == 0)}")
dispersal.DISPERSAL_RATE, dispersal.DISPERSAL_KERNEL = d0, "nn4"

for bad in (dict(kind="2dt", a=1.0, b=1.0), dict(kind="2dt", a=0.0, b=2.0), dict(kind="nn8")):
    try:
        dispersal.unit_kernel_matrix(**bad); res = "accepted"
    except ValueError as e:
        res = f"refused ({e})"
    print(f"G  bad setting {bad}: {res} {ok(res.startswith('refused'))}")
