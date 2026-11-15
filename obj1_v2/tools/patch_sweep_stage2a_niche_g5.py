"""
patch_sweep_stage2a_niche_g5.py  (obj1_v2, gate 5: the 4-seed batch of the setting that will run, 1 Nov 2026, Sara's yes)

Adds plan "niche_g5" to sweep_stage2a.py: n70's setting (niche_p; best founding, habitat tau 62,500, shift sd 1.0 tau
62,500, 25 habitat sub-steps per window, strength 0.4, connectance 1) on the three other seeds, so the setting is judged
on 4 worlds with n70 (seed 2303):
  n73  seed 2304
  n74  seed 2307
  n75  seed 2308
Habitat and trait seeds 100000 + seed and 200000 + seed; the leftover --disp-rate as in niche_h2 (no effect with
--disp-per-species, test_15 U3).
Sara's decision (1 Nov): similarity decay is a stated limit of this model, not a gate: every species is sorted along one
habitat axis, and on a world the size of the BNM window far cells never share habitat (correlation at 5-10 cells about
0 at every scale), so decay stays 0.11-0.20 where BNM has 0.04; the levers tried either did not move it (competition
strength, links, contrast) or broke rare clumping (a finer habitat).
Reading rule (written before the run, 1 Nov), on n70 + n73-n75:
  1. gate 1: richness_balance_check, pooled kept share interval holds 0.5 and balance ratio mu / (0.05 f (1 - mu)) of 1
     or more; richness_shift_check no trend in 3 or more of 4.
  2. gate 7: frame_change_check v2 on the sub-step maps, pooled: R2 (an interval found), no class outside BNM's
     interval; k* is the frame interval for Objective 2.
  3. habitat_track_check guard 0.8 or more in every world; test_05 final-run losses 2% or less.
  4. gate 5, bands: the agreed verdict on 20 window-end maps per world (frame_bands_export + world_vs_bnm: median over
     the 80 maps inside and 75% of maps inside), the 4 end maps reported beside it. Decay reported as the stated limit.
     Every other band that fails is named; each gets at most 2 fix rounds (the finish line), the first a cause check.
  5. If 1-3 pass and every band but decay passes, this setting goes to the real run (gates 6, 8 and 9 are run on its
     exports).
Outputs go to results/stage2a/niche_g5/. Needs patch_sweep_stage2a_niche_p.py.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_g5.py
    python3 ../tools/patch_sweep_stage2a_niche_g5.py
    python3 ../tools/patch_sweep_stage2a_niche_g5.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 gate5"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_g5 plan is already in sweep_stage2a.py")
if "# obj1_v2 substeps" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_p.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "+ connectance 0.4 (n72); one-seed pilot before the 4-seed batch.", "niche_p docstring line")
edits.append((i + 1, i + 1, [
    "  niche_g5 3 worlds (added 1 Nov): n73-n75 = n70's setting on seeds 2304, 2307, 2308; with n70 the setting that",
    "         will run is judged on 4 worlds (gate 5)."]))
i = find(lambda s: s.strip() == 'if plan == "niche_p":   # obj1_v2 substeps', "niche_p plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_g5":   {MARK}',
    f'{p}    b = dict(kernel="2dt_a1_b3", f=1, attempts=1000, window=5000, niche=True, tradeoff=0.5, contrast=0.25,',
    f'{p}             connectance=1.0, disp=(1e-6, 1e-2), founding="best", habitat_tau=62500, habitat_shift_sd=1.0,',
    f'{p}             habitat_shift_tau=62500, habitat_substeps=25)',
    f'{p}    return [dict(b, name=f"n{{72 + k}}", d=d, seed=s, habitat_seed=100000 + s, trait_seed=200000 + s)',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: '"niche_h2", "niche_p"]' in s and 'ap.add_argument("--plan"' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_h2", "niche_p"]', '"niche_h2", "niche_p", "niche_g5"]') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan == "niche_p" and ("--habitat-substeps" not in src'), "niche_p guard")
edits.append((i, i + 1, [L[i].replace('if a.plan == "niche_p" and', 'if a.plan in ("niche_p", "niche_g5") and') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_h", "niche_h2", "niche_p") and ("--habitat-shift-sd" not in src'),
         "shift guard")
edits.append((i, i + 1, [L[i].replace('("niche_h", "niche_h2", "niche_p")', '("niche_h", "niche_h2", "niche_p", "niche_g5")') + f"   {MARK}"]))

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