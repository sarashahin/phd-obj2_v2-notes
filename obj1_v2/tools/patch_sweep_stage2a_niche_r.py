"""
patch_sweep_stage2a_niche_r.py  (obj1_v2, rare end round 2: narrow niches with and without the shift; 2 Nov 2026, Sara's yes)

Adds plan "niche_r" to sweep_stage2a.py, one seed (2303), twins of n70 (best founding, habitat tau 62,500, 25 habitat
sub-steps, strength 0.4, connectance 1, d per species 1e-6 to 1e-2):
  n76  n70 + minimum niche width 0.1 (shift sd 1.0, tau 62,500 kept)
  n77  n76 without the shift (habitat tau 62,500 and 25 sub-steps kept)
Why: rare_end_check (2 Nov) found no habitat specialists in any world (small footprints 0.0-0.2%): at width 0.2 and
contrast 0.25 a footprint spans +-1.86 habitat sd, so a species of rare habitat cannot win its own cells (row 26). At
width 0.1, 11.1% of competitive candidates are specialists against 2.1% today (candidate_specialist_share.py). But
specialist_habitat_check.py found the shift empties a specialist's habitat in 11-27% of years (viable spells 44-86
BNM years, gaps about 20), while without the shift it keeps 3-40 cells in 90-98% of years (row 27, candidate). n77
separates the two causes; it is not expected to balance (n59, no shift: ratio 0.89).
Reading rule (written 2 Nov, before the build; claude/n33_result_29oct.md):
  1. specialists: rare_end_check small-footprint share 5% or more.
  2. rare end on the world's 20 window-end maps (frame_bands_export, world_vs_bnm, band_sides_readout): 1-5 share median
     inside 15.2-31.4% and 75% of maps inside; 1 cell share and rare clumping reported.
  3. n76 only: balance (kept share interval holds 0.5, ratio 1 or more), richness_shift_check no trend, frame_change_check
     R2 with no class outside BNM's interval, habitat_track_check guard 0.8, test_05 2% or less, species per window median
     inside on the maps.
  n76 passes 1-3 -> 4-seed batch. n76 fails 1 or 2 and n77 passes them -> the shift removes the specialists (row 27);
  the fix needs a new design of the habitat change, beyond the finish line: Sara decides. Neither passes 2 -> the rare
  end goes to Sara. Expected: n77 passes 1, n76 fails 1 or 2. One seed: a pass is confirmed only by the batch.
Outputs go to results/stage2a/niche_r/. Needs patch_sweep_stage2a_niche_g5.py.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_r.py
    python3 ../tools/patch_sweep_stage2a_niche_r.py
    python3 ../tools/patch_sweep_stage2a_niche_r.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 rare-end"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_r plan is already in sweep_stage2a.py")
if "# obj1_v2 gate5" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_g5.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "will run is judged on 4 worlds (gate 5).", "niche_g5 docstring line")
edits.append((i + 1, i + 1, [
    "  niche_r 2 worlds (added 2 Nov): seed 2303, n70's setting with minimum niche width 0.1 (n76) and the same without",
    "         the shift (n77): habitat specialists, and whether the shift removes them."]))
i = find(lambda s: s.strip() == 'if plan == "niche_g5":   # obj1_v2 gate5', "niche_g5 plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_r":   {MARK}',
    f'{p}    b = dict(kernel="2dt_a1_b3", f=1, attempts=1000, window=5000, niche=True, tradeoff=0.5, contrast=0.25,',
    f'{p}             connectance=1.0, disp=(1e-6, 1e-2), d="1e-6", seed=2303, habitat_seed=102303, trait_seed=202303,',
    f'{p}             founding="best", habitat_tau=62500, habitat_substeps=25, w_min=0.1)',
    f'{p}    return [dict(b, name="n76", habitat_shift_sd=1.0, habitat_shift_tau=62500), dict(b, name="n77")]']))
i = find(lambda s: '"niche_p", "niche_g5"]' in s and 'ap.add_argument("--plan"' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_p", "niche_g5"]', '"niche_p", "niche_g5", "niche_r"]') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_p", "niche_g5") and ("--habitat-substeps" not in src'), "substeps guard")
edits.append((i, i + 1, [L[i].replace('if a.plan in ("niche_p", "niche_g5") and', 'if a.plan in ("niche_p", "niche_g5", "niche_r") and') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_h", "niche_h2", "niche_p", "niche_g5") and ("--habitat-shift-sd" not in src'),
         "shift guard")
edits.append((i, i + 1, [L[i].replace('("niche_h", "niche_h2", "niche_p", "niche_g5")', '("niche_h", "niche_h2", "niche_p", "niche_g5", "niche_r")') + f"   {MARK}"]))

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