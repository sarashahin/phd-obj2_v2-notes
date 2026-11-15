"""
founding_medium_check.py  (obj1_v2, LVMCM founding, 29 Oct 2026)

Question: what does a full assembly give under best founding against random founding, at the same
settings and seed? Medium size, CPU: the n29 settings (contrast 0.25, k 0.5, habitat scale 3,
connectance 1, d per species 1e-6 to 1e-2, 2Dt a 1 b 3, one individual per founding, tau 1, sim seed
2303, habitat seed 102303, trait seed 202303, 1,000 attempts), but 1,000-step windows instead of 5,000.

  run <best|random>  assembles one world (about 30 min on one CPU core) into results/medium/medium_<rule>_2303.npz
  report             for each saved world:
                       species, rounds, change over the last 20% of attempts, true range shares,
                       median range, species per cell;
                       10,000 more steps with no newcomers (species lost);
                       sinks: share of present populations (1 or more individuals) whose own
                       g = r - C B in that cell is below 0 (below -0.05), and species per cell with
                       g >= 0 only;
                       true-map turnover and Sorensen by distance (world_vs_bnm's alpha_beta, torus,
                       no survey; for best against random only, not for BNM bands);
                       by fundamental range F: tested, entered, kept, final (best founding).
Run from obj1_v2/work:
    USE_GPU=0 python3 ../tools/founding_medium_check.py run best
    USE_GPU=0 python3 ../tools/founding_medium_check.py run random
    USE_GPU=0 python3 ../tools/founding_medium_check.py report | tee ../logs/founding_medium_check_29oct.txt
Claude's run (29 Oct): README sketch 37.
"""
import contextlib, io, os, sys, time
sys.path.insert(0, os.getcwd()); os.environ.setdefault("USE_GPU", "0")
import numpy as np
import dispersal, environment as env, assembly_utils
import assembly_stepwise_ibm as asm
from models_ibm import IBMModel

G, SEED, OUT = 20, 2303, "results/medium"
H = env.habitat_layer(G, G, 3.0, seed=100000 + SEED) * 0.25
NICHE = dict(opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=env.NICHE_PEAK_MAX, cost=0.1, tradeoff=0.5)
BINS = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
FB = ((0, 0, "0"), (1, 20, "1-20"), (21, 100, "21-100"), (101, 200, "101-200"), (201, 300, "201-300"), (301, 10**9, "301+"))


def setup():
    dispersal.DISPERSAL_KERNEL, dispersal.DISPERSAL_A, dispersal.DISPERSAL_B, dispersal.DISPERSAL_RATE = "2dt", 1.0, 3.0, 1e-6
    dispersal.LONG_DISTANCE_PROB = 0.0
    assembly_utils.CONNECTANCE = 1.0


def run(rule):
    if 'founding: str = "random"' not in open(asm.__file__).read():
        sys.exit("STOP: work/assembly_stepwise_ibm.py is not v2.3")
    setup(); os.makedirs(OUT, exist_ok=True); t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        r, C, N, x = asm.stepwise_assembly_ibm(window_steps=1000, record_step=50, seed_size=1, F_sat=None, max_rounds=None,
                                               max_attempts=1000, richness_cap=None, frac_multi=0.05, seed=SEED,
                                               detection_threshold=1, habitat=H, niche=NICHE, seed_field=200000 + SEED,
                                               disp=dict(d_min=1e-6, d_max=1e-2), founding=rule)
    np.savez_compressed(f"{OUT}/medium_{rule}_{SEED}.npz", r=r, C=C, N=N, minutes=np.float64((time.time() - t) / 60),
                        **{k: v for k, v in x.items() if not isinstance(v, str)})
    print(f"{rule}: {(time.time() - t) / 60:.1f} min, {len(r)} species, {len(x['ASM_round'])} rounds")


def fsize(z):
    opt, w, pk = (np.asarray(z[k], float) for k in ("LT_opt", "LT_w", "LT_peak"))
    F = np.zeros(len(opt), int)
    for s in range(0, len(opt), 2000):
        r = pk[s:s + 2000, None, None] * np.exp(-(H[None] - opt[s:s + 2000, None, None]) ** 2 / (2 * w[s:s + 2000, None, None] ** 2)) - NICHE["cost"]
        F[s:s + 2000] = (r > 0).reshape(len(r), -1).sum(1)
    return F


def report():
    setup()
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "world_vs_bnm.py")).read()
    wvb = {"__name__": "wvb"}; exec(src.split("ap = argparse.ArgumentParser()")[0], wvb)
    for rule in ("best", "random"):
        f = f"{OUT}/medium_{rule}_{SEED}.npz"
        if not os.path.exists(f):
            print(f"{rule}: no file {f}; run it first"); continue
        z = np.load(f, allow_pickle=True)
        N, C = np.asarray(z["N"]), np.asarray(z["C"]); P = N >= 1; S = len(N)
        cells = P.reshape(S, -1).sum(1); gam, att = np.asarray(z["ASM_gamma"]), np.asarray(z["ASM_attempts"])
        k80 = np.searchsorted(att, 0.8 * att[-1])
        print(f"\n{rule}: {float(z['minutes']):.1f} min; {S} species; {len(gam)} rounds; attempts {int(z['attempts_total'])}"
              + (f"; candidates tested {int(z['tested_total'])}" if "tested_total" in z.files else "")
              + f"; change over the last 20% of attempts {(gam[-1] - gam[k80]) / max(gam[k80], 1):+.0%}")
        print("  true range shares 1 / 2-5 / 6-10 / 11-20 / 21-50 / 51+: " + " ".join(f"{np.mean((cells >= a) & (cells <= b)):.0%}" for a, b in BINS)
              + f"; median range {np.median(cells):.0f} cells; species per cell {P.sum(0).mean():.1f}")
        m = IBMModel(z["r"], C, initial_N=N, r_field=z["r_field"], nsteps=10000, record_step=1000, record_mode="none",
                     seed=int(z["final_run_seed"]))
        dispersal.set_species_rates(np.asarray(z["disp_rate"], float))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m.run()
        finally:
            dispersal.set_species_rates(None)
        alive = (np.asarray(m.N) >= 1).any(axis=(1, 2))
        print(f"  10,000 more steps, no newcomers: {alive.sum()} of {S} still there ({1 - alive.mean():.1%} lost)")
        Nf = N.reshape(S, -1); g = np.asarray(z["r_field"]).reshape(S, -1) - C @ (Nf * 1e-4); Pf = Nf >= 1
        print(f"  sinks: g < 0 in {np.mean(g[Pf] < 0):.0%} of present populations ({np.mean(g[Pf] < -0.05):.0%} below -0.05); "
              f"species per cell with g >= 0 {((g >= 0) & Pf).sum(0).mean():.1f} of {Pf.sum(0).mean():.1f}")
        s = wvb["alpha_beta"]([(P, None)], True)
        print(f"  true maps: turnover {s['beta_gm']:.2f}; Sorensen 1 / 2 / 3-4 / 5-10 cells " + " / ".join(f"{v:.2f}" for v in s["sor"]))
        if "LT_entered" in z.files:
            F = fsize(z); ent = np.asarray(z["LT_entered"]).astype(bool); est = np.asarray(z["LT_established"]).astype(bool)
            pos = {int(i): k for k, i in enumerate(z["LT_id"])}; fin = np.zeros(len(F), bool); fin[[pos[int(i)] for i in z["species_id"]]] = True
            for lo, hi, lab in FB:
                mm = (F >= lo) & (F <= hi)
                print(f"  F {lab:>7}: tested {mm.sum():5d}, entered {ent[mm].sum():4d}, kept {est[mm].sum():4d}, final {fin[mm].sum():4d}")


if len(sys.argv) >= 3 and sys.argv[1] == "run" and sys.argv[2] in ("best", "random"):
    run(sys.argv[2])
elif len(sys.argv) == 2 and sys.argv[1] == "report":
    report()
else:
    sys.exit("use: run best | run random | report")