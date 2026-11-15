"""
patch_sweep_stage2a_real.py  (obj1_v2, the real IBM run for Objective 2; 2 Nov 2026, Sara's yes: option A, no sweep)

Adds three plans to sweep_stage2a.py, one world per seed, all with n70's setting (the gate 5 setting, passed on 4 seeds:
balance, no trend, speed 1 piece = 1 BNM year, guard, test_05): best founding, habitat tau 62,500, regional shift sd
1.0 tau 62,500, 25 habitat sub-steps per 5,000-step window, minimum niche width 0.2, competition strength 0.4,
connectance 1, contrast 0.25, trade-off 0.5, habitat length scale 3, d per species 1e-6 to 1e-2, 2Dt kernel a 1 b 3,
founded in 1 cell, 1,000 entered newcomers (checked in the saved file), final run 10,000 steps.
  real_train  100 worlds, sim seeds 5001-5100 (names rt5001-rt5100)  -> results/stage2a/real_train/  Objective 2 training
  real_val     20 worlds, sim seeds 5101-5120 (names rv5101-rv5120)  -> results/stage2a/real_val/    early stopping, tuning
  real_eval    30 worlds, sim seeds 6001-6030 (names re6001-re6030)  -> results/stage2a/real_eval/   opened once, after training
Habitat seed 100000 + seed and trait seed 200000 + seed, so every world has its own habitat map and its own species
pool; no seed is shared with a test world (2001-2308) or between splits. The leftover --disp-rate is 1e-6 as in n70 (no
effect with --disp-per-species, test_15 U3).
Why one setting and no sweep (Sara, 2 Nov): only this setting passed balance and speed on 4 seeds; the old sweep's
parameters are replaced (field scale and variance by habitat scale and contrast; threshold by detection at export; d by
d per species; uniform jumps by the 2Dt tail) or did nothing (d bug). Variety comes inside each world: habitat map,
niche optimum and width, and d over four orders of magnitude.
Stated limits of this setting (reported, not gates): similarity decay, species per cell, turnover, 51+ share, range-size
TV (competition sorting, row 25); the rare end (1-5 share, 1 cell share, rare clumping); species per window on maps
(median inside, 65% of maps). Option A (Sara, 2 Nov): Objective 2 reweights training species to the BNM fit-half
range-size classes and is checked on held-out BNM fit-half records before the test half is opened.
Quality rule (written 2 Nov, before the run), per world: real_run_check.py settings, seeds and attempts as above; test_05
final-run losses 2% or less; habitat_track_check guard 0.8 or more. A world that fails is reported to Sara before any
use; nothing is dropped silently. Per split (description): richness_balance_check, richness_shift_check,
frame_change_check, world_vs_bnm on end maps.
Leak rule: Objective 2 training reads real_train only, tuning and early stopping read real_val only; real_eval is read
once, after training; real_run_check R3 confirms no seed, habitat map or final map is shared between any two worlds.
Time: about 40 min per world on the GPU (n70 40.1 min; n73-n75 37.5-43.7), about 100 hours for 150 worlds. A stopped
plan is resumed by running the same command again: finished worlds are skipped; a world stopped mid-run starts again.
Needs patch_sweep_stage2a_niche_r2.py (anchors) and assembly v2.6.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_real.py
    python3 ../tools/patch_sweep_stage2a_real.py
    python3 ../tools/patch_sweep_stage2a_real.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 real run"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the real-run plans are already in sweep_stage2a.py")
if "# obj1_v2 rare-end 2" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_r2.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "the rare end kept?).", "niche_r2 docstring line")
edits.append((i + 1, i + 1, [
    "  real_train / real_val / real_eval (added 2 Nov): the real run, n70's setting, one world per seed: 100 training",
    "         (5001-5100), 20 validation (5101-5120), 30 evaluation (6001-6030), each split in its own folder."]))
i = find(lambda s: s.strip() == 'if plan == "niche_r2":   # obj1_v2 rare-end 2', "niche_r2 plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan in ("real_train", "real_val", "real_eval"):   {MARK}',
    f'{p}    b = dict(kernel="2dt_a1_b3", f=1, attempts=1000, check_attempts=True, window=5000, niche=True, tradeoff=0.5,',
    f'{p}             contrast=0.25, connectance=1.0, disp=(1e-6, 1e-2), d="1e-6", founding="best", habitat_tau=62500,',
    f'{p}             habitat_shift_sd=1.0, habitat_shift_tau=62500, habitat_substeps=25)',
    f'{p}    lo, hi, tag = {{"real_train": (5001, 5100, "rt"), "real_val": (5101, 5120, "rv"), "real_eval": (6001, 6030, "re")}}[plan]',
    f'{p}    return [dict(b, name=f"{{tag}}{{s}}", seed=s, habitat_seed=100000 + s, trait_seed=200000 + s) for s in range(lo, hi + 1)]']))
i = find(lambda s: '"niche_r", "niche_r2"]' in s and 'ap.add_argument("--plan"' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_r", "niche_r2"]', '"niche_r", "niche_r2", "real_train", "real_val", "real_eval"]')
                         + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_p", "niche_g5", "niche_r", "niche_r2") and ("--habitat-substeps" not in src'),
         "substeps guard")
edits.append((i, i + 1, [L[i].replace('("niche_p", "niche_g5", "niche_r", "niche_r2")',
                                      '("niche_p", "niche_g5", "niche_r", "niche_r2", "real_train", "real_val", "real_eval")') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_h", "niche_h2", "niche_p", "niche_g5", "niche_r", "niche_r2") and ("--habitat-shift-sd" not in src'),
         "shift guard")
edits.append((i, i + 1, [L[i].replace('"niche_r", "niche_r2")', '"niche_r", "niche_r2", "real_train", "real_val", "real_eval")') + f"   {MARK}"]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "sweep_stage2a.py (now)", "sweep_stage2a.py (patched)", lineterm="", n=1)))
compile(out, "sweep_stage2a.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: ../tools/sweep_stage2a.py")
else:
    print("dry run: nothing written (add --apply to write)")