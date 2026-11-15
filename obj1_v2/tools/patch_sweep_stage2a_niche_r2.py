"""
patch_sweep_stage2a_niche_r2.py  (obj1_v2, rare end, the one extra round Sara chose (option A); 2 Nov 2026)

Adds plan "niche_r2" to sweep_stage2a.py: n78 = n76 (n70's setting with minimum niche width 0.1, shift kept, seed 2303)
run to 2,000 entered newcomers instead of 1,000 (the saved attempts are checked, as in plan extend).
Why: n76 kept balance (0.51, ratio 1.39), speed, guard and test_05 and moved the rare end the right way (1-5 share on maps
16.3% against 13.2%, 1 cell 4.0% against 2.9%, specialists x14), but its richness was still rising when the run stopped
(+85% over the last half; end 35, ceiling 44 (25-84)), so its 20 maps caught a community still filling (species per
window 24 against band 27-41).
Reading rule (written 2 Nov, before the build; claude/n33_result_29oct.md):
  1. richness_shift_check: no trend over the last half.
  2. on the world's 20 window-end maps (frame_bands_export, world_vs_bnm, band_sides_readout): species per window and the
     1-5 cell share each with the median inside the band and 15 or more of 20 maps inside.
  3. balance (kept share interval holds 0.5, ratio 1 or more), frame_change_check R2 with no class outside BNM's
     interval, habitat_track_check guard 0.8 or more, test_05 final-run losses 2% or less.
  All pass -> 4-seed batch at width 0.1 and 2,000 newcomers. Any fails -> the rare end is a stated limit, the real run
  uses n70's setting, and no further round is made (finish line). Specialists, 1 cell share and the other bands are
  reported. One seed: a pass is confirmed only by the batch.
Cost if it passes: about twice n76's 51 min per world, so the real run (150 worlds) would take roughly twice as long.
Outputs go to results/stage2a/niche_r2/. Needs patch_sweep_stage2a_niche_r.py.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_r2.py
    python3 ../tools/patch_sweep_stage2a_niche_r2.py
    python3 ../tools/patch_sweep_stage2a_niche_r2.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 rare-end 2"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_r2 plan is already in sweep_stage2a.py")
if "# obj1_v2 rare-end" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_r.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "the shift (n77): habitat specialists, and whether the shift removes them.", "niche_r docstring line")
edits.append((i + 1, i + 1, [
    "  niche_r2 1 world (added 2 Nov): n78 = n76 run to 2,000 entered newcomers (Sara's option A: does it level off with",
    "         the rare end kept?)."]))
i = find(lambda s: s.strip() == 'if plan == "niche_r":   # obj1_v2 rare-end', "niche_r plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_r2":   {MARK}',
    f'{p}    return [dict(kernel="2dt_a1_b3", f=1, attempts=2000, check_attempts=True, window=5000, niche=True, tradeoff=0.5,',
    f'{p}                 contrast=0.25, connectance=1.0, disp=(1e-6, 1e-2), d="1e-6", seed=2303, habitat_seed=102303,',
    f'{p}                 trait_seed=202303, founding="best", habitat_tau=62500, habitat_substeps=25, w_min=0.1,',
    f'{p}                 habitat_shift_sd=1.0, habitat_shift_tau=62500, name="n78")]']))
i = find(lambda s: '"niche_g5", "niche_r"]' in s and 'ap.add_argument("--plan"' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_g5", "niche_r"]', '"niche_g5", "niche_r", "niche_r2"]') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_p", "niche_g5", "niche_r") and ("--habitat-substeps" not in src'), "substeps guard")
edits.append((i, i + 1, [L[i].replace('("niche_p", "niche_g5", "niche_r")', '("niche_p", "niche_g5", "niche_r", "niche_r2")') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_h", "niche_h2", "niche_p", "niche_g5", "niche_r") and ("--habitat-shift-sd" not in src'),
         "shift guard")
edits.append((i, i + 1, [L[i].replace('"niche_g5", "niche_r")', '"niche_g5", "niche_r", "niche_r2")') + f"   {MARK}"]))

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