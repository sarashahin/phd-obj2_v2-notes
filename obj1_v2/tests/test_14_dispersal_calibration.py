"""
test_14_dispersal_calibration.py  (obj1_v2, stage 2d-b step 1, 29 Oct 2026)

Question: at which dispersal rate d can the IBM hold a species in cells where it cannot grow
on its own (sink cells next to its source), and how far out? This sets the range of the
per-species d (2d-b, Sara's yes 29 Oct: log-uniform, drawn per species id, no trade-off).

Why: 2 x 2 run (sketch 33): every species sits almost only in its best cells (K3b AUC
0.96-0.997), far cells share 0.31-0.45 of species against BNM's 0.70. In the model only the
dispersal term d x kernel x B can keep a species where its growth rate is below 0.
LVMCM (O'Sullivan, LVMCM_src on GitHub) uses the same form (effort-normalised dispersal, the
emigrants of a cell sum to e B, no emigration loss in the public code) with e = 0.01 in its
example assemblies and 0.1 as default, growth rates of mean 1; our worlds use d = 1e-6 to
3e-6 with growth rates of the same order.

Set-up (real IBMModel, CPU): one species on the 20 x 20 torus, growth rate 0.3 in a 4 x 4
source block (about 3,000 individuals per source cell, as in the niche worlds) and g < 0 in
every other cell (g = -0.1 and -0.3); C = 1; propagule dispersal as in the assembly; d from
1e-6 to 1e-1; kernels 2Dt a 1 b 3 (the stage 2a choice) and nn4 (nearest neighbours, closest
to LVMCM's graph). 2,000 steps; the last 1,000 averaged.

D1 (check of the IBM against its own equation): in sink cells the time-mean number of
   individuals equals the linear steady state N_s = (I (1 - e^g) - d K_ss')^-1 d K_cs' N_c,
   N_c the simulated source mean (per step: survival and births give e^g, then immigrants).
   PASS when the median ratio simulated / expected is within 0.85-1.15 over sink cells with an
   expected 1 or more individuals, for every d and kernel that has such cells. The linear steady
   state exists only while d x (largest eigenvalue of K_ss) < 1 - e^g: propagule dispersal adds
   individuals without taking them from the source, so at higher d the sink cells keep each other
   going and only density dependence limits them. There D1 is printed as "no linear steady state"
   (found on the first run, 29 Oct: d 0.1 with g -0.1 gave ratios 0.15, because the linear
   expectation had no meaning there, not because the IBM was wrong).
D2 (description): mean individuals and share of steps with 1 or more individuals in sink
   cells at Chebyshev distance 1, 2, 3 and 4-6 from the source; the smallest d that keeps 1 or
   more individuals on average at each distance.
Ecological reading: below the d where distance-1 sinks hold an individual, a species lives
only where it can grow (no mass effect); above it, sinks around sources fill in, which raises
species per cell and similarity between cells. The per-species d range must span both.

Run from obj1_v2/work:
    USE_GPU=0 python3 ../tests/test_14_dispersal_calibration.py | tee ../logs/test_14_dispersal_calibration_29oct.txt
"""
import contextlib, io, os, sys
sys.path.insert(0, os.getcwd())
os.environ.setdefault("USE_GPU", "0")
import numpy as np
import config, dispersal
from models_ibm import IBMModel

G = config.NUM_PATCHES_Y
if (config.NUM_PATCHES_X, G, config.BODY_MASS, config.STEP_SIZE) != (20, 20, 1e-4, 1):
    sys.exit("STOP: test written for a 20 x 20 grid, BODY_MASS 1e-4, STEP_SIZE 1")
SRC = (slice(8, 12), slice(8, 12))
R_SRC, STEPS, KEEP, REC = 0.3, 2000, 1000, 10
D_LIST = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
KERNELS = (("2dt", 1.0, 3.0), ("nn4", None, None))
fails = []

src = np.zeros((G, G), bool); src[SRC] = True
yy, xx = np.mgrid[0:G, 0:G]
dist = np.full((G, G), 99)
for y, x in zip(*np.nonzero(src)):
    dy = np.minimum(abs(yy - y), G - abs(yy - y)); dx = np.minimum(abs(xx - x), G - abs(xx - x))
    dist = np.minimum(dist, np.maximum(dy, dx))
BANDS = ((1, 1), (2, 2), (3, 3), (4, 6))


def run(kind, a, b, d, g):
    dispersal.DISPERSAL_KERNEL = kind
    if a is not None:
        dispersal.DISPERSAL_A, dispersal.DISPERSAL_B = a, b
    dispersal.DISPERSAL_RATE = d
    F = np.where(src, R_SRC, g)[None].astype(float)
    N0 = np.where(src, int(round(R_SRC / config.BODY_MASS)), 0)[None]
    m = IBMModel(np.array([R_SRC]), np.eye(1), initial_N=N0, r_field=F, nsteps=STEPS, record_step=REC,
                 record_mode="full", seed=11)
    with contextlib.redirect_stdout(io.StringIO()):
        tr = m.run()
    Nt = np.asarray(tr)[-(KEEP // REC):, 0] / config.BODY_MASS          # records x G x G, individuals
    K = np.asarray(dispersal.unit_kernel_matrix().todense() if hasattr(dispersal.unit_kernel_matrix(), "todense")
                   else dispersal.unit_kernel_matrix())
    return Nt, K


def expected(K, Nc, d, g):
    """linear steady state of the sink cells, or None when it does not exist (sinks self-sustaining)."""
    s, c = ~src.ravel(), src.ravel()
    Kss = K[np.ix_(s, s)]
    if d * np.abs(np.linalg.eigvals(Kss)).max() >= 1 - np.exp(g):
        return None
    A = np.eye(s.sum()) * (1 - np.exp(g)) - d * Kss.T
    return np.linalg.solve(A, d * K[np.ix_(c, s)].T @ Nc.ravel()[c])


print(f"one species, source 4 x 4 cells at r {R_SRC} (C = 1, about {R_SRC / config.BODY_MASS:,.0f} individuals per cell), "
      f"sinks everywhere else; {STEPS} steps, last {KEEP} averaged")
first = {}
for kind, a, b in KERNELS:
    for g in (-0.1, -0.3):
        print(f"\nkernel {kind}{'' if a is None else f' a {a:g} b {b:g}'}, sink growth rate {g}")
        print("  d        source mean   sink mean individuals at distance 1 / 2 / 3 / 4-6     share of steps with 1+ "
              "at 1 / 2 / 3 / 4-6     D1 sim/expected (cells)")
        for d in D_LIST:
            Nt, K = run(kind, a, b, d, g)
            Nm = Nt.mean(0)
            mean_b = [Nm[(dist >= lo) & (dist <= hi)].mean() for lo, hi in BANDS]
            occ_b = [(Nt[:, (dist >= lo) & (dist <= hi)] >= 1).mean() for lo, hi in BANDS]
            ex = expected(K, Nm, d, g); sim = Nm.ravel()[~src.ravel()]
            use = ex >= 1 if ex is not None else None
            if ex is None:
                d1 = f"no linear steady state (d >= 1 - e^g = {1 - np.exp(g):.3f})"
            elif use.sum():
                ratio = np.median(sim[use] / ex[use]); ok = 0.85 <= ratio <= 1.15
                fails.extend([] if ok else [f"D1[{kind}, g {g}, d {d:g}]"])
                d1 = f"{ratio:.3f} ({use.sum()}) {'PASS' if ok else 'FAIL'}"
            else:
                d1 = "no cell expected at 1+"
            for k, (lo, hi) in enumerate(BANDS):
                if mean_b[k] >= 1 and (kind, g, k) not in first:
                    first[(kind, g, k)] = d
            print(f"  {d:<8g} {Nm[src].mean():>10,.0f}    " + " / ".join(f"{v:9.3g}" for v in mean_b)
                  + "      " + " / ".join(f"{v:4.0%}" for v in occ_b) + f"      {d1}")
print("\nsmallest d with 1 or more individuals on average (distance 1 / 2 / 3 / 4-6):")
for kind, a, b in KERNELS:
    for g in (-0.1, -0.3):
        print(f"  {kind}, g {g}: " + " / ".join(f"{first.get((kind, g, k), float('nan')):g}" for k in range(len(BANDS))))
print(f"\n{'all passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)