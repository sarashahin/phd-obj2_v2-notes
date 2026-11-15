"""
sweep_stage2a.py  (obj1_v2, written 24 Oct 2026)

Stage 2a: choose the dispersal rate d, the kernel shape and the founding rule by
comparing simulated worlds with Sara's BNM windows (tools/world_vs_bnm.py).

Root causes this step targets (tests 04-06, 24 Oct): ranges frozen (0.2 arrivals per
edge cell per run at d = 2e-8), clustering gone by 3 cells against 5-6 in BNM, and
rarity made of frozen founding cells (87% of the oldest species in 5 cells or fewer).

Plans
  pilot  4 worlds, 1,000 invasion attempts each. Checks the pipeline and run time on
         the GPU, and measures seed-to-seed spread:
           p1  d 1e-6, nn4,         founded in 5 cells, sim seed 2001
           p2  as p1,                                    sim seed 2002 (control: same
               setting, other seed; differences between settings must be larger)
           p3  d 1e-6, 2dt a 1 b 3, founded in 1 cell,   sim seed 2003
           p4  as p1 with d 3e-7 (BODY_MASS x 3e-3),       sim seed 2004 (added 25 Oct;
               against p1 and p2 it changes d only)
  settle 8 worlds (added 25 Oct): 2Dt a 1 b 3, founded in 1 cell, 1,000 attempts; d 1e-6 and 3e-6 x
         assembly window 1,000 and 5,000 steps x 2 seeds. s1 is pilot p3 (seed 2003), copied after
         its metadata is checked; s2-s8 use sim seeds 2302-2308. Asks whether a longer window lets
         the community settle (final-run losses 2% or less, richness change within 2%).
  long   2 worlds (added 26 Oct): settings and sim seeds of settle s3 (2303) and s7 (2307) with a
         30,000-step final run. Pass: no species lost after step 10,000.
  extend 4 worlds (added 26 Oct): settings and sim seeds of settle s3, s4, s7, s8 (2303, 2304, 2307,
         2308) with 2,000 attempts. test_10 compares each with its 1,000-attempt twin.
  niche  4 worlds (added 27 Oct): settings and sim seeds of settle s3, s4, s7, s8 in niche mode (habitat
         scale 3, optimum sd 1.5, width 0.2-1.0, cost 0.1; habitat seed 100000 + sim seed, trait seed
         200000 + sim seed). Each settle twin is the control for the niche effect.
  niche_k 4 worlds (added 28 Oct): n5-n8 = n1-n4 with trade-off strength 0.5 instead of 1 (softer:
         a niche 5x wider has a peak about 2.2x lower). n1-n4 are the control.
  niche_c 8 worlds (added 28 Oct): n5-n8 with habitat contrast 0.5 (n9-n12) and 0.25 (n13-n16)
         instead of 1 (the same habitat map, less different from place to place). n5-n8 are the control.
  niche_f 12 worlds (added 28 Oct): 2 x 2 at contrast 0.25, k 0.5: habitat length scale 1 (n17-n20),
         connectance 1 (n21-n24), both (n25-n28); n13-n16 (scale 3, connectance 0.4) are the twins.
  niche_d 4 worlds (added 29 Oct): n29-n32 = n21-n24 with a dispersal rate per species, log-uniform 1e-6 to
         1e-2 (test_14 range). n21-n24 are the twins.
  niche_b 4 worlds (added 29 Oct): n33-n36 = n29-n32 with LVMCM founding (--founding best: a newcomer enters
         only if it can grow somewhere, as one individual in its best cell; 1,000 entered newcomers).
         n29-n32 are the twins.
  full   4 d x 3 kernels x 2 founding rules = 24 worlds, 2,000 attempts each,
         sim seeds 2101-2124 in the order printed by --dry-run. --reps 2 adds a second
         seed for every setting (2201-2224).
         d in 1e-7, 3e-7, 1e-6, 3e-6; kernel nn4, 2dt a 1 b 2.5, 2dt a 1 b 3; founding 5, 1.
         2dt b 1.75 is left out: a quarter of its propagules land beyond 2 cells, and in
         the 24 Oct test worlds its all-case co-occupancy at 1 cell was 1.31 (too diffuse).
Fixed settings: those of sweep_worlds.py (window 1,000 steps, final run 10,000 steps
recorded every 200, invasion fraction 0.05, record mode full, top-16 partners,
co-occurrence sample 4096, float16 series, record budgets 1,5,10,20,50,100, no
saturation stop), except: tag s2a, long-distance share 0 (the 2dt tail does that
job), environment l 5, variance 0.002, field seed 789, tau 1. The environment, tau and
the stopping rule are Stages 2b-2d; holding them fixed here keeps 2a about dispersal.

Needs run_all_rps.py with Step 1 (patch_run_all_rps_v2.py), the RPS guard and
patch_run_all_rps_seed_size.py; the script stops if --seed-size is missing.

Run from obj1_v2/work:
  python3 ../tools/sweep_stage2a.py --plan pilot --dry-run     # prints the commands only
  python3 ../tools/sweep_stage2a.py --plan pilot --smoke       # 120 attempts, 300-step windows,
                                                               # 1,000-step final run: checks the pipeline
  python3 ../tools/sweep_stage2a.py --plan pilot
Uses the GPU (USE_GPU=1) unless USE_GPU is already set or --cpu is given.
Outputs: each world's training and dataset files are moved to results/stage2a/<plan>/
(<plan>_smoke/ with --smoke);
one log per world in ../logs/stage2a/<plan>/; results/stage2a/<plan>/runs.csv lists
name, setting, seed, minutes, species and status. A world whose training file is
already in results/stage2a/<plan>/ is skipped, so the command can be run again after
a stop. Then score with world_vs_bnm.py (command printed at the end).
"""
import argparse, csv, glob, os, shutil, subprocess, sys, time
from pathlib import Path

BASE = ["--pool", "22510000", "--skip-psd", "--skip-ode", "--tmax", "10000", "--record", "200", "--no-movie",
        "--ibm-frac-multi", "0.05", "--ibm-window-steps", "1000", "--ibm-richness-cap", "None", "--save-env-field",
        "--ibm-record-mode", "full", "--ibm-F-sat", "None", "--ibm-max-rounds", "None", "--save-every-rounds", "500",
        "--save-every-seconds", "600", "--C-topk", "16", "--coocc-sample", "4096", "--world-tag-extra", "s2a",
        "--fp16-time-series", "--obs-budgets", "1,5,10,20,50,100",
        "--env-length-scale", "5", "--env-var-r", "0.002", "--env-seed-field", "789",
        "--ibm-detection-threshold", "1", "--ldd-prob", "0"]
KERNELS = {"nn4": ["--disp-kernel", "nn4"],
           "2dt_a1_b2.5": ["--disp-kernel", "2dt", "--disp-a", "1", "--disp-b", "2.5"],
           "2dt_a1_b3": ["--disp-kernel", "2dt", "--disp-a", "1", "--disp-b", "3"]}


def plan_rows(plan, reps):
    if plan == "niche_b":   # obj1_v2 founding
        return [dict(name=f"n{32 + k}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,
                     habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=0.25, connectance=1.0,
                     disp=(1e-6, 1e-2), founding="best")
                for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]
    if plan == "niche_d":   # obj1_v2 per-species d
        return [dict(name=f"n{28 + k}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,
                     habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=0.25, connectance=1.0,
                     disp=(1e-6, 1e-2))
                for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]
    if plan == "niche_f":   # obj1_v2 connectance
        return [dict(name=f"n{16 + 4 * j + k}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,
                     habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=0.25, **x)
                for j, x in enumerate((dict(habitat_ls=1.0), dict(connectance=1.0), dict(habitat_ls=1.0, connectance=1.0)))
                for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]
    if plan == "niche_c":   # obj1_v2 contrast v2
        return [dict(name=f"n{8 + 4 * j + k}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,
                     habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=c)
                for j, c in enumerate((0.5, 0.25))
                for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]
    if plan == "niche_k":   # obj1_v2 tradeoff
        return [dict(name=f"n{k + 4}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,
                     habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5)
                for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]
    if plan == "niche":   # obj1_v2 niche
        return [dict(name=f"n{k}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,
                     habitat_seed=100000 + s, trait_seed=200000 + s)
                for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]
    if plan == "extend":   # obj1_v2 extend
        return [dict(name=f"e{k}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=2000, window=5000, check_attempts=True)
                for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]
    if plan == "long":   # obj1_v2 long
        return [dict(name="l1", d="1e-6", kernel="2dt_a1_b3", f=1, seed=2303, attempts=1000, window=5000, tmax=30000),
                dict(name="l2", d="3e-6", kernel="2dt_a1_b3", f=1, seed=2307, attempts=1000, window=5000, tmax=30000)]
    if plan == "settle":   # obj1_v2 settle
        cells = (("1e-6", 1000, 2003), ("1e-6", 1000, 2302), ("1e-6", 5000, 2303), ("1e-6", 5000, 2304),
                 ("3e-6", 1000, 2305), ("3e-6", 1000, 2306), ("3e-6", 5000, 2307), ("3e-6", 5000, 2308))
        return [dict(name=f"s{k}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=w,
                     reuse="results/stage2a/pilot" if s == 2003 else None) for k, (d, w, s) in enumerate(cells, 1)]
    if plan == "pilot":
        return [dict(name="p1", d="1e-6", kernel="nn4", f=5, seed=2001, attempts=1000),
                dict(name="p2", d="1e-6", kernel="nn4", f=5, seed=2002, attempts=1000),
                dict(name="p3", d="1e-6", kernel="2dt_a1_b3", f=1, seed=2003, attempts=1000),
                dict(name="p4", d="3e-7", kernel="nn4", f=5, seed=2004, attempts=1000)]   # obj1_v2 p4
    rows = []
    for rep in range(reps):
        i = 0
        for d in ("1e-7", "3e-7", "1e-6", "3e-6"):
            for k in KERNELS:
                for f in (5, 1):
                    i += 1
                    rows.append(dict(name=f"g{i:02d}" + ("" if rep == 0 else f"r{rep + 1}"), d=d, kernel=k, f=f,
                                     seed=2100 + 100 * rep + i, attempts=2000))
    return rows


SMOKE = {"--ibm-window-steps": "300", "--tmax": "1000"}


def meta_ok(z, r, smoke):   # obj1_v2 settle
    """The saved metadata carries the requested setting: seed, founding, window, d and kernel."""
    return (int(z["SIM_SEED"]) == r["seed"] and int(z["seed_size"]) == r["f"]
            and int(z["window_steps"]) == (300 if smoke else int(r.get("window", 1000)))
            and abs(float(z["DISPERSAL_RATE"]) - float(r["d"])) <= 1e-6 * float(r["d"])
            and str(z["DISPERSAL_KERNEL"][0]) == r["kernel"].split("_")[0]
            and (r["kernel"] == "nn4" or (float(z["DISPERSAL_A"]) == float(KERNELS[r["kernel"]][3])
                                          and float(z["DISPERSAL_B"]) == float(KERNELS[r["kernel"]][5]))))


def tmax_ok(z, r, smoke):   # obj1_v2 long
    """Rows that set tmax: the saved final run ends at tmax (1,000 under --smoke)."""
    return "tmax" not in r or abs(float(z["IBM_t"][-1]) - (1000 if smoke else float(r["tmax"]))) < 1e-6


def attempts_ok(z, r, smoke):   # obj1_v2 extend
    """Rows with check_attempts: the saved assembly reached the requested attempts (120 under --smoke)."""
    return not r.get("check_attempts") or int(z["ASM_attempts"][-1]) >= (120 if smoke else int(r["attempts"]))


NICHE_SET = dict(HABITAT_ls=3.0, NICHE_opt_sd=1.5, NICHE_w_min=0.2, NICHE_w_max=1.0, NICHE_cost=0.1)   # obj1_v2 niche


def niche_ok(z, r, smoke):   # obj1_v2 niche
    """Niche rows: the file is a niche world with this row's habitat seed, trait seed and settings.
    Other rows: the file is not a niche world."""
    mode = str(z["ENV_mode"][0]) if "ENV_mode" in z.files else "field"
    if not r.get("niche"):
        return mode == "field"
    return (mode == "niche" and int(z["HABITAT_seed"]) == r["habitat_seed"] and int(z["TRAIT_seed"]) == r["trait_seed"]
            and all(abs(float(z[k]) - v) < 1e-6 for k, v in {**NICHE_SET, "HABITAT_ls": float(r.get("habitat_ls", 3.0))}.items())   # obj1_v2 connectance
            and abs((float(z["NICHE_tradeoff"]) if "NICHE_tradeoff" in z.files else 1.0) - r.get("tradeoff", 1.0)) < 1e-6   # obj1_v2 tradeoff
            and abs((float(z["HABITAT_contrast"]) if "HABITAT_contrast" in z.files else 1.0) - r.get("contrast", 1.0)) < 1e-6   # obj1_v2 contrast
            and abs((float(z["CONNECTANCE"]) if "CONNECTANCE" in z.files else 0.4) - r.get("connectance", 0.4)) < 1e-6   # obj1_v2 connectance
            and disp_ok(z, r)   # obj1_v2 per-species d
            and found_ok(z, r))   # obj1_v2 founding


def disp_ok(z, r):   # obj1_v2 per-species d
    """Per-species d saved as the row asks: rows with disp=(d_min, d_max) need DISP_per_species 1 and the same range;
    other rows need shared d (files without DISP_per_species count as shared)."""
    per = int(z["DISP_per_species"]) if "DISP_per_species" in z.files else 0
    if r.get("disp") is None:
        return per == 0
    lo, hi = r["disp"]
    return per == 1 and abs(float(z["DISP_d_min"]) / lo - 1) < 1e-9 and abs(float(z["DISP_d_max"]) / hi - 1) < 1e-9


def found_ok(z, r):   # obj1_v2 founding
    """Founding rule saved as the row asks: rows with founding="best" need FOUND_rule best; other rows need random
    (files without FOUND_rule count as random)."""
    rule = str(z["FOUND_rule"][0]) if "FOUND_rule" in z.files else "random"
    return rule == r.get("founding", "random")


def row_pat(r):   # obj1_v2 contrast v2   # obj1_v2 connectance
    """File pattern of a row's training file. Niche rows with a contrast, habitat scale or connectance
    name them as the world tag does (ls<scale>o ... hc<contrast>_[cn<connectance>_]k), so rows of one plan
    that share a seed are told apart; other rows keep the plain pattern."""
    c = r.get("contrast")
    if (c in (None, 1, 1.0) and r.get("connectance") is None and r.get("habitat_ls") is None and r.get("disp") is None   # obj1_v2 per-species d
            and r.get("founding") is None):   # obj1_v2 founding
        return f"*_s2a_*_f{r['f']}_s{r['seed']}_training.npz"
    sl = lambda x: str(float(x)).replace(".", "p").replace("-", "m")   # as _sanitize_slug in run_all_rps.py
    ls = f"ls{sl(r.get('habitat_ls', 3.0))}o" if r.get("niche") else ""
    hc = f"hc{sl(c)}_" if c not in (None, 1, 1.0) else ""
    cn = f"cn{sl(r['connectance'])}_" if r.get("connectance") is not None else ""
    dps = f"dps{sl(r['disp'][0])}-{sl(r['disp'][1])}_" if r.get("disp") is not None else ""
    fb = "fb_" if r.get("founding") == "best" else ""   # obj1_v2 founding
    return f"*_s2a_*{ls}*{hc}{cn}{dps}{fb}k*_f{r['f']}_s{r['seed']}_training.npz"


def command(r, smoke=False):
    base = list(BASE)
    if r.get("window"): base[base.index("--ibm-window-steps") + 1] = str(r["window"])   # obj1_v2 settle
    if r.get("tmax"): base[base.index("--tmax") + 1] = str(r["tmax"])   # obj1_v2 long
    if r.get("niche"):   # obj1_v2 niche: field-mode environment options out, niche options in
        for k in ("--env-length-scale", "--env-var-r", "--env-seed-field"):
            j = base.index(k); del base[j:j + 2]
        base += ["--env-mode", "niche", "--habitat-seed", str(r["habitat_seed"]), "--env-seed-field", str(r["trait_seed"]),
                 "--habitat-ls", "3", "--niche-opt-sd", "1.5", "--niche-w-min", "0.2", "--niche-w-max", "1.0",
                 "--niche-cost", "0.1"]
    if r.get("tradeoff") is not None:   # obj1_v2 tradeoff
        base += ["--niche-tradeoff", str(r["tradeoff"])]
    if r.get("contrast") is not None:   # obj1_v2 contrast
        base += ["--habitat-contrast", str(r["contrast"])]
    if r.get("habitat_ls") is not None:   # obj1_v2 connectance
        base[base.index("--habitat-ls") + 1] = str(r["habitat_ls"])
    if r.get("connectance") is not None:
        base += ["--connectance", str(r["connectance"])]
    if r.get("disp") is not None:   # obj1_v2 per-species d
        base += ["--disp-per-species", repr(float(r["disp"][0])), repr(float(r["disp"][1]))]
    if r.get("founding") is not None:   # obj1_v2 founding
        base += ["--founding", r["founding"]]
    if smoke:
        for k, v in SMOKE.items(): base[base.index(k) + 1] = v
    return ([sys.executable, "run_all_rps.py"] + base + ["--ibm-max-attempts", "120" if smoke else str(r["attempts"]),
            "--disp-rate", r["d"]] + KERNELS[r["kernel"]] + ["--seed-size", str(r["f"]), "--sim-seed", str(r["seed"])])


ap = argparse.ArgumentParser()
ap.add_argument("--plan", choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c", "niche_f", "niche_d", "niche_b"], required=True)   # obj1_v2 settle   # obj1_v2 long   # obj1_v2 extend   # obj1_v2 niche   # obj1_v2 tradeoff   # obj1_v2 contrast   # obj1_v2 connectance   # obj1_v2 per-species d   # obj1_v2 founding
ap.add_argument("--reps", type=int, default=1, help="seeds per setting in the full plan")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--cpu", action="store_true", help="force USE_GPU=0")
ap.add_argument("--smoke", action="store_true", help="tiny runs to check the pipeline (outputs to <plan>_smoke)")
ap.add_argument("--only", nargs="*", help="run only these names, e.g. p1 g05")
a = ap.parse_args()

if not Path("run_all_rps.py").is_file():
    sys.exit("STOP: run this from obj1_v2/work (run_all_rps.py not found here)")
src = Path("run_all_rps.py").read_text()
if "--seed-size" not in src or "--sim-seed" not in src:
    sys.exit("STOP: run_all_rps.py lacks --sim-seed or --seed-size; apply patch_run_all_rps_v2.py and "
             "patch_run_all_rps_seed_size.py first")
if a.plan == "niche_b" and ("--founding" not in src   # obj1_v2 founding
                           or 'founding: str = "random"' not in Path("assembly_stepwise_ibm.py").read_text()):
    sys.exit("STOP: niche_b needs --founding in run_all_rps.py (patch_run_all_rps_founding.py) and assembly v2.3 in work")
if a.plan == "niche_d" and "--disp-per-species" not in src:   # obj1_v2 per-species d
    sys.exit("STOP: run_all_rps.py lacks --disp-per-species; apply patch_run_all_rps_disp.py first")
if a.plan == "niche_f" and "--connectance" not in src:   # obj1_v2 connectance
    sys.exit("STOP: run_all_rps.py lacks --connectance; apply patch_run_all_rps_connectance.py first")
if a.plan == "niche_c" and "--habitat-contrast" not in src:   # obj1_v2 contrast
    sys.exit("STOP: run_all_rps.py lacks --habitat-contrast; apply patch_run_all_rps_contrast.py first")
if a.plan == "niche_k" and "--niche-tradeoff" not in src:   # obj1_v2 tradeoff
    sys.exit("STOP: run_all_rps.py lacks --niche-tradeoff; apply patch_run_all_rps_tradeoff.py first")
if a.plan == "niche" and "--env-mode" not in src:   # obj1_v2 niche
    sys.exit("STOP: run_all_rps.py lacks --env-mode; apply patch_run_all_rps_niche.py first")
rows = plan_rows(a.plan, a.reps)
if a.only:
    rows = [r for r in rows if r["name"] in a.only]
    if not rows: sys.exit(f"STOP: no world named {a.only} in plan {a.plan}")
sub = a.plan + ("_smoke" if a.smoke else "")
out_dir, log_dir = Path("results/stage2a") / sub, Path("../logs/stage2a") / sub
env = dict(os.environ)
env["USE_GPU"] = "0" if a.cpu else env.get("USE_GPU", "1")
print(f"plan {sub}: {len(rows)} worlds; USE_GPU={env['USE_GPU']}; outputs to {out_dir}/, logs to {log_dir}/")
for r in rows:
    print(f"  {r['name']}: d {r['d']}, kernel {r['kernel']}, founded in {r['f']} cell(s), sim seed {r['seed']}, "
          f"{120 if a.smoke else r['attempts']} attempts, window {300 if a.smoke else r.get('window', 1000)} steps, final run {1000 if a.smoke else r.get('tmax', 10000)} steps")
if a.dry_run:
    print("\ncommands:")
    for r in rows: print("  " + " ".join(command(r, a.smoke)[1:]))
    sys.exit(0)

out_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
runs_csv = out_dir / "runs.csv"
fields = ["name", "d", "kernel", "f", "seed", "attempts", "window", "tmax", "status", "minutes", "species", "file"]   # obj1_v2 settle   # obj1_v2 long
done_rows = list(csv.DictReader(open(runs_csv))) if runs_csv.is_file() else []
for r in rows:   # obj1_v2 settle: copy a finished identical world, after checking its metadata
    if not r.get("reuse") or a.smoke: continue
    pat = row_pat(r)   # obj1_v2 contrast v2
    if sorted(out_dir.glob(pat)): continue
    found = sorted(Path(r["reuse"]).glob(pat))
    if len(found) != 1: sys.exit(f"STOP: {r['name']} reuses one file {pat} from {r['reuse']}; found {len(found)}")
    import numpy as np
    z = np.load(found[0], allow_pickle=True)
    if not meta_ok(z, r, False): sys.exit(f"STOP: {found[0].name} does not carry the setting of {r['name']}; nothing copied")
    shutil.copy2(found[0], out_dir / found[0].name)
    dsrc = found[0].with_name(found[0].name.replace("_training.npz", "_dataset.npz"))
    if dsrc.is_file(): shutil.copy2(dsrc, out_dir / dsrc.name)
    done_rows = [x for x in done_rows if x["name"] != r["name"]] + [dict(
        name=r["name"], d=r["d"], kernel=r["kernel"], f=r["f"], seed=r["seed"], attempts=r["attempts"], window=r["window"],
        status=f"ok (copied from {r['reuse']})", minutes="", species=str(int(z["gamma"])), file=found[0].name)]
    with open(runs_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(done_rows)
    print(f"{r['name']}: copied from {r['reuse']} after the metadata check ({found[0].name})")
for r in rows:
    pat = row_pat(r)   # obj1_v2 contrast v2
    have = sorted(out_dir.glob(pat))
    if have:
        print(f"{r['name']}: already done ({have[0].name}), skipped"); continue
    log = log_dir / f"{r['name']}_f{r['f']}_s{r['seed']}.log"
    print(f"{r['name']}: running ... log {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(" ".join(command(r, a.smoke)) + "\n\n"); fh.flush()
        p = subprocess.run(command(r, a.smoke), stdout=fh, stderr=subprocess.STDOUT, env=env)
    mins = (time.time() - t0) / 60
    made = [Path(f) for f in glob.glob(f"results/data/{pat}") if os.path.getmtime(f) >= t0 - 1]
    row = dict(r, attempts=120 if a.smoke else r["attempts"], minutes=f"{mins:.1f}", species="", file="",   # obj1_v2 settle
               window=300 if a.smoke else r.get("window", 1000), tmax=1000 if a.smoke else r.get("tmax", 10000))   # obj1_v2 long
    if p.returncode != 0 or len(made) != 1:
        row["status"] = f"FAILED (exit {p.returncode}, {len(made)} training files)"
        print(f"{r['name']}: {row['status']} after {mins:.1f} min; see {log}")
    else:
        train = made[0]; data = train.with_name(train.name.replace("_training.npz", "_dataset.npz"))
        import numpy as np
        z = np.load(train, allow_pickle=True)
        row.update(status="ok", species=str(int(z["gamma"])), file=train.name)
        ok_meta = (meta_ok(z, r, a.smoke) and tmax_ok(z, r, a.smoke) and attempts_ok(z, r, a.smoke) and niche_ok(z, r, a.smoke))   # obj1_v2 niche
        dest = out_dir
        if not ok_meta:                       # kept apart, so a re-run does not count it as done
            row["status"] = "metadata mismatch, file in rejected/"; dest = out_dir / "rejected"; dest.mkdir(exist_ok=True)
        shutil.move(str(train), dest / train.name)
        if data.is_file(): shutil.move(str(data), dest / data.name)
        print(f"{r['name']}: {row['status']}, {mins:.1f} min, {row['species']} species, final run to t {float(z['IBM_t'][-1]):g}, saved seed {int(z['SIM_SEED'])}, "   # obj1_v2 long
              f"seed_size {int(z['seed_size'])}, d {float(z['DISPERSAL_RATE']):g}, kernel {z['DISPERSAL_KERNEL'][0]}")
    done_rows = [x for x in done_rows if x["name"] != r["name"]] + [{k: row.get(k, "") for k in fields}]
    with open(runs_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(done_rows)

print(f"\nruns listed in {runs_csv}. Score them against BNM (from obj1_v2/work):")
print(f'  python3 ../tools/world_vs_bnm.py --bnm "<BNM>" --survey "<BNM>" --each --sim "{out_dir}/*_training.npz"')
print('  with <BNM> = ../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz')
print(f'  and test_05 on the same files: USE_GPU=0 python3 ../tests/test_05_assembly_equilibrium.py "{out_dir}/*_training.npz"')