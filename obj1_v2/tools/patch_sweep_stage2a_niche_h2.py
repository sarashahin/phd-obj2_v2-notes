"""
patch_sweep_stage2a_niche_h2.py  (obj1_v2, root cause 23, gate 5 for the candidate setting, 31 Oct 2026, Sara's yes)

Adds plan "niche_h2" to sweep_stage2a.py: the n66 setting on the three other seeds, so the candidate is judged on 4
worlds (the bands need 3 of 4 worlds inside; balance and richness need more than one world).
  n67  seed 2304   best founding, habitat tau 62,500, shift sd 1.0, shift tau 62,500 (as n66, seed 2303)
  n68  seed 2307
  n69  seed 2308
Habitat and trait seeds 100000 + seed and 200000 + seed, the leftover --disp-rate as the random-founding rows of the same
seed (no effect with --disp-per-species, test_15 U3), everything else n66's.
Reading rule (written before the run, 31 Oct):
  1. richness_balance_check --chain with n66 (niche_h) and n67-n69 (niche_h2) as two links; kept and removed counts of
     the 4 worlds added: pooled kept share interval holds 0.5 and balance ratio mu / (0.05 f (1 - mu)) 1 or more; per
     world reported.
  2. richness_shift_check: no trend in the last half in 3 or more of the 4 worlds (richness swings with the shift but
     does not keep falling or rising).
  3. habitat_track_check guard 0.8 or more in every world; resident_loss_check reported.
  4. world_vs_bnm v5.10 setting verdict on the 4 worlds: the bands n66 passed (window, cell, turnover, TV, 1-5 share,
     51+) must pass as a setting; rare clumping and similarity decay reported (both failed in n66).
Outputs go to results/stage2a/niche_h2/. Needs the niche_h patch (patch_sweep_stage2a_niche_h.py).

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_h2.py
    python3 ../tools/patch_sweep_stage2a_niche_h2.py
    python3 ../tools/patch_sweep_stage2a_niche_h2.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 habitat-shift 2"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_h2 plan is already in sweep_stage2a.py")
if "# obj1_v2 habitat-shift" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_h.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "62,500. n49-n52 and n59 are the twins.", "niche_h docstring line")
edits.append((i + 1, i + 1, [
    "  niche_h2 3 worlds (added 31 Oct): n67-n69 = n66's setting (best founding, habitat tau 62,500, shift sd 1.0,",
    "         shift tau 62,500) on seeds 2304, 2307, 2308; with n66 (seed 2303) the candidate is judged on 4 worlds."]))
i = find(lambda s: s.strip() == 'if plan == "niche_h":   # obj1_v2 habitat-shift', "niche_h plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_h2":   {MARK}',
    f'{p}    b = dict(kernel="2dt_a1_b3", f=1, attempts=1000, window=5000, niche=True, tradeoff=0.5, contrast=0.25,',
    f'{p}             connectance=1.0, disp=(1e-6, 1e-2))',
    f'{p}    return [dict(b, name=f"n{{66 + k}}", d=d, seed=s, habitat_seed=100000 + s, trait_seed=200000 + s, founding="best",',
    f'{p}                 habitat_tau=62500, habitat_shift_sd=1.0, habitat_shift_tau=62500)',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: '"niche_t2", "niche_h"]' in s and 'ap.add_argument("--plan"' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_t2", "niche_h"]', '"niche_t2", "niche_h", "niche_h2"]') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan == "niche_h" and ("--habitat-shift-sd" not in src'), "niche_h guard")
edits.append((i, i + 1, [L[i].replace('if a.plan == "niche_h" and', 'if a.plan in ("niche_h", "niche_h2") and') + f"   {MARK}"]))

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