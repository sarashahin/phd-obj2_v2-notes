"""
test_11_niche.py  (obj1_v2, stage 2d-a first draft, 27 Oct 2026; N8 trade-off strength added 28 Oct)

Question: do the niche functions added to environment.py (patch_environment_niche.py) do
what they say, and does the real IBM (models_ibm.py) give the ranges they predict?
Needs no world files; runs the IBM on CPU for one or two species on the 20 x 20 grid.

N1 habitat layer: mean 0 and sd 1; same seed gives the same layer, other seeds unrelated
   layers; neighbouring cells correlated (patchy habitat); a real map from a .npy file is
   standardised; a wrong shape, missing values or a flat map are refused.
N2 niche_r: peak - cost at the optimum; peak e^-1/2 - cost one width away.
N3 niche_traits: the same (seed, id) gives the same niche; peak x width equal for every
   species and no peak above peak_max; widths log-uniform; optima spread opt_sd; a peak_max
   of 2 or more is refused.
N4 known answer, stability: one species, no dispersal, r constant. The IBM's expected update
   is B' = B exp(r - B): it settles at B* = r for r = 1.0 and 1.5, and flips between two
   values for r = 2.5. This is why peaks stay below 2.
N5 known answer, range: one species alone on a habitat layer occupies the cells where its
   growth rate is above 0, at equilibrium count r / BODY_MASS, and almost none where it is
   clearly below 0 (a thin fringe from immigrants is allowed).
N6 effect: a narrower niche gives a smaller range, and the counts match N5's rule.
N7 known answer, trade-off: specialist (narrow, high peak) and generalist (wide, low peak),
   competition 0.4 both ways. Each cell is compared with the exact Lotka-Volterra
   equilibrium of two species: the specialist wins where the habitat fits it, the
   generalist holds the rest, both where the equilibrium says they coexist.
N8 trade-off strength k (after patch_environment_tradeoff.py; skipped before): k = 1 gives the
   old peaks exactly; k = 0.5 keeps optimum and width, peak x width^0.5 the same for all, peak
   0.67 at width 1; k < 0 refused; known answer: in the exact two-species equilibrium of N7
   the generalist holds more cells under k = 0.5 than under k = 1, and the IBM matches it.
Ecological reading: rarity now has a cause the habitat can show (a narrow niche, or an
optimum on rare habitat), and the generalist is not able to win everywhere.
Run from obj1_v2/work, after patch_environment_niche.py --apply:
    USE_GPU=0 python3 ../tests/test_11_niche.py | tee ../logs/test_11_niche_27oct.txt
"""
import contextlib, io, os, sys, tempfile
sys.path.insert(0, os.getcwd())
import numpy as np
from scipy import stats
import config, dispersal, environment as env
from models_ibm import IBMModel

if not hasattr(env, "habitat_layer"):
    sys.exit("STOP: environment.py has no niche functions; run ../tools/patch_environment_niche.py --apply first")
G, BODY = config.NUM_PATCHES_Y, config.BODY_MASS
res = lambda c: "PASS" if c else "FAIL"
fails = []
def check(name, cond, text):
    print(f"{name} {text} {res(cond)}"); fails.extend([] if cond else [name])


def run_ibm(r_field, C, steps, d, N0=100, seed=1, record=None):
    dispersal.DISPERSAL_RATE = d
    S = r_field.shape[0]
    m = IBMModel(np.ones(S), C, initial_N=np.full((S, G, G), N0, int), r_field=r_field, nsteps=steps,
                 record_step=record or steps, record_mode="full" if record else "none", seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):                  # the IBM prints every 100 steps
        tr = m.run()
    return np.asarray(m.N), tr


# N1 habitat layer
H = env.habitat_layer(G, G, 3.0, seed=5)
check("N1a", abs(H.mean()) < 1e-12 and abs(H.std() - 1) < 1e-12, f"mean {H.mean():.1e}, sd {H.std():.12f}:")
check("N1b", np.array_equal(H, env.habitat_layer(G, G, 3.0, seed=5)), "same seed gives the same layer:")
cors = [np.corrcoef(env.habitat_layer(G, G, 3.0, seed=s).ravel(), env.habitat_layer(G, G, 3.0, seed=s + 1000).ravel())[0, 1] for s in range(50)]
check("N1c", abs(np.mean(cors)) < 0.05, f"other seeds unrelated (mean correlation of 50 pairs {np.mean(cors):+.3f}):")
lag1 = np.mean([np.mean(L * np.roll(L, 1, axis=1)) for L in (env.habitat_layer(G, G, 3.0, seed=s) for s in range(50))])
check("N1d", lag1 > 0.5, f"neighbouring cells correlated at length scale 3 (lag-1 correlation {lag1:.2f}):")
with tempfile.TemporaryDirectory() as t:
    raw = np.add.outer(np.arange(G), np.arange(G)).astype(float) * 3 + 7
    np.save(f"{t}/map.npy", raw); Hm = env.habitat_layer(G, G, None, path=f"{t}/map.npy")
    ok_map = np.allclose(Hm, (raw - raw.mean()) / raw.std())
    refused = []
    for bad in (np.zeros((G, G + 1)), np.where(np.eye(G) > 0, np.nan, 1.0), np.ones((G, G))):
        np.save(f"{t}/bad.npy", bad)
        try: env.habitat_layer(G, G, None, path=f"{t}/bad.npy"); refused.append(False)
        except ValueError: refused.append(True)
check("N1e", ok_map and all(refused), "a real map is standardised; wrong shape, missing values and a flat map refused:")

# N2 niche_r
check("N2", np.isclose(env.niche_r(0.3, 0.3, 0.5, 1.2, 0.1), 1.1) and np.isclose(env.niche_r(0.8, 0.3, 0.5, 1.2, 0.1), 1.2 * np.exp(-0.5) - 0.1),
      "peak - cost at the optimum, peak e^-1/2 - cost one width away:")

# N3 traits
T = np.array([env.niche_traits(i, seed=789) for i in range(3000)])
same = env.niche_traits(1234, seed=789) == env.niche_traits(1234, seed=789) and env.niche_traits(1234, seed=789) != env.niche_traits(1234, seed=790)
area = T[:, 1] * T[:, 2]
check("N3a", same, "the same (seed, id) gives the same niche, another seed another:")
check("N3b", np.allclose(area, 1.5 * 0.2) and T[:, 2].max() <= 1.5 + 1e-12 and T[:, 1].min() >= 0.2 and T[:, 1].max() <= 1.0,
      f"peak x width = {area.mean():.3f} for all (0.300), peaks {T[:, 2].min():.2f}-{T[:, 2].max():.2f}, widths {T[:, 1].min():.3f}-{T[:, 1].max():.3f}:")
u = (np.log(T[:, 1]) - np.log(0.2)) / (np.log(1.0) - np.log(0.2))
check("N3c", stats.kstest(u, "uniform").pvalue > 0.01 and abs(T[:, 0].std() - 1.5) < 0.1,
      f"log-width uniform (KS p {stats.kstest(u, 'uniform').pvalue:.2f}), optimum sd {T[:, 0].std():.2f} (1.5):")
try: env.niche_traits(1, seed=1, peak_max=2.0); refused = False
except ValueError: refused = True
check("N3d", refused, "peak_max of 2 refused:")

# N4 stability of the update (no dispersal)
for r, expect_flip in ((1.0, False), (1.5, False), (2.5, True)):
    _, tr = run_ibm(np.full((1, G, G), r), np.eye(1), 400, 0.0, record=1)
    B = np.asarray(tr)[200:, 0].reshape(200, -1)                   # last 200 steps, every cell
    mean = B.mean(); flip = np.mean(np.abs(np.diff(B, axis=0))) / mean
    ok = (flip > 0.5) if expect_flip else (flip < 0.05 and abs(mean - r) / r < 0.02)
    check("N4", ok, f"r = {r}: mean biomass {mean:.3f} (B* = r = {r}), mean step-to-step change {flip:.1%} of the mean "
          f"({'flips between two values' if expect_flip else 'settles'} expected):")

# N5 one species alone on the habitat
D = 1e-6
H = env.habitat_layer(G, G, 3.0, seed=11)
r5 = env.niche_r(H, 0.0, 0.5, 1.5 * 0.2 / 0.5, 0.1)
N, _ = run_ibm(r5[None], np.eye(1), 3000, D)
occ, good, bad = N[0] >= 1, r5 > 0, r5 < -0.05
ratio = (N[0][good] * BODY / r5[good])
check("N5a", occ[good].mean() >= 0.95 and occ[bad].mean() <= 0.10,
      f"occupied: {occ[good].mean():.1%} of the {good.sum()} cells with r > 0, {occ[bad].mean():.1%} of the {bad.sum()} cells with r < -0.05:")
check("N5b", abs(np.median(ratio) - 1) < 0.05, f"count / (r / BODY_MASS) in cells with r > 0: median {np.median(ratio):.3f} (1):")

# N6 width effect
out6 = []
for w in (0.2, 1.0):
    rw = env.niche_r(H, 0.0, w, 1.5 * 0.2 / w, 0.1)
    Nw, _ = run_ibm(rw[None], np.eye(1), 3000, D)
    out6.append(((Nw[0] >= 1).sum(), (rw > 0).sum()))
check("N6", out6[0][0] < out6[1][0] and all(abs(o - p) <= 0.05 * max(p, 1) + 3 for o, p in out6),
      f"range in cells: narrow {out6[0][0]} (r > 0 in {out6[0][1]}), wide {out6[1][0]} (r > 0 in {out6[1][1]}):")

# N7 specialist against generalist, exact two-species equilibrium per cell
rs = env.niche_r(H, 0.0, 0.2, 1.5, 0.1); rg = env.niche_r(H, 0.0, 1.0, 0.3, 0.1)
C = np.array([[1.0, 0.4], [0.4, 1.0]])
def lv2(r1, r2):                                          # B >= 0, r - C B <= 0, B (r - C B) = 0
    for sup in ((0, 1), (0,), (1,), ()):
        B = np.zeros(2)
        if sup: B[list(sup)] = np.linalg.solve(C[np.ix_(sup, sup)], np.array([r1, r2])[list(sup)])
        g = np.array([r1, r2]) - C @ B
        if (B >= -1e-12).all() and (g[[i for i in (0, 1) if i not in sup]] <= 1e-12).all(): return B
eq = np.array([[lv2(rs[y, x], rg[y, x]) for x in range(G)] for y in range(G)])   # (G, G, 2)
N7, _ = run_ibm(np.stack([rs, rg]), C, 4000, D)
Bsim = np.moveaxis(N7 * BODY, 0, -1)
spec_cells, gen_cells = eq[..., 0] > 0.05, (eq[..., 1] > 0.05) & (eq[..., 0] < 1e-9)
both = (eq[..., 0] > 0.05) & (eq[..., 1] > 0.05)
cor = np.corrcoef(Bsim.ravel(), eq.ravel())[0, 1]
check("N7a", cor > 0.98, f"biomass per cell against the exact equilibrium: correlation {cor:.3f}:")
check("N7b", (N7[0][spec_cells] >= 1).mean() >= 0.95 and (N7[1][gen_cells] >= 1).mean() >= 0.95 and (N7[0][gen_cells & (rs < -0.05)] >= 1).mean() <= 0.10,
      f"specialist in {(N7[0][spec_cells] >= 1).mean():.0%} of its {spec_cells.sum()} cells, generalist in {(N7[1][gen_cells] >= 1).mean():.0%} "
      f"of its {gen_cells.sum()} cells, both in {both.sum()} cells by the equilibrium:")
check("N7c", spec_cells.sum() < gen_cells.sum() + both.sum(), "the specialist holds fewer cells than the generalist:")

# N8 trade-off strength
import inspect
if "tradeoff" in inspect.signature(env.niche_traits).parameters:
    T1 = np.array([env.niche_traits(i, seed=789, tradeoff=1.0) for i in range(3000)])
    T5 = np.array([env.niche_traits(i, seed=789, tradeoff=0.5) for i in range(3000)])
    check("N8a", np.array_equal(T1, T) and np.array_equal(T1[:, 2], 1.5 * 0.2 / T1[:, 1]),
          "k = 1 gives the old niches exactly (peak = 1.5 x 0.2 / width):")
    k5_area = T5[:, 2] * np.sqrt(T5[:, 1])
    check("N8b", np.array_equal(T5[:, :2], T1[:, :2]) and np.allclose(k5_area, 1.5 * np.sqrt(0.2)) and T5[:, 2].max() <= 1.5 + 1e-12,
          f"k = 0.5: same optima and widths, peak x width^0.5 = {k5_area.mean():.4f} for all ({1.5 * np.sqrt(0.2):.4f}), "
          f"peaks {T5[:, 2].min():.2f}-{T5[:, 2].max():.2f} (0.67 at width 1):")
    try: env.niche_traits(1, seed=1, tradeoff=-0.1); refused = False
    except ValueError: refused = True
    check("N8c", refused, "k below 0 refused:")
    gen = {}
    for k in (1.0, 0.5):
        rg_k = env.niche_r(H, 0.0, 1.0, 1.5 * (0.2 / 1.0) ** k, 0.1)
        eq_k = np.array([[lv2(rs[y, x], rg_k[y, x]) for x in range(G)] for y in range(G)])
        gen[k] = (eq_k, rg_k, int((eq_k[..., 1] > 0.05).sum()))
    eq5, rg5, _ = gen[0.5]
    N8, _ = run_ibm(np.stack([rs, rg5]), C, 4000, D)
    cor8 = np.corrcoef((np.moveaxis(N8 * BODY, 0, -1)).ravel(), eq5.ravel())[0, 1]
    check("N8d", gen[0.5][2] > gen[1.0][2] and cor8 > 0.98,
          f"generalist cells in the exact equilibrium: k = 1 {gen[1.0][2]}, k = 0.5 {gen[0.5][2]}; IBM against exact equilibrium at k = 0.5: corr {cor8:.3f}:")
else:
    print("N8 skipped: environment.py has no trade-off strength (patch_environment_tradeoff.py not applied)")

print(f"\n{'all passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)