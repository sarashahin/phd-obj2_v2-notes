"""
patch_sweep_stage2a_niche_t2.py  (obj1_v2, root cause 23, fix B, 31 Oct 2026, Sara's yes)

Adds plan "niche_t2" to sweep_stage2a.py: best founding (twin n33, seed 2303) at two faster habitat speeds, to see
whether fix B can balance best founding while ranges still follow the habitat.
  n59  tau 62,500 steps (12.5 rounds of 5,000; rho 0.923 per round)
  n60  tau 15,625 steps (3.1 rounds; rho 0.726 per round)
Everything else is n57's and n58's (niche_t): contrast 0.25, k 0.5, habitat scale 3, connectance 1, strength 0.4,
d per species 1e-6 to 1e-2, seeds 2303 / 102303 / 202303, 1,000 entered newcomers, window 5,000.
Why these speeds (pilot niche_t, 31 Oct): kept share 0.76 (static), 0.66 (tau 1,000,000), 0.60 (tau 250,000);
losses rose x1.81 and then x1.35 per 4-fold speed, and balance at 250,000 needs about x1.5 more.
Reading rule (written before the run, 31 Oct):
  1. richness_balance_check --chain n33 -> n57 -> n58 -> n59 -> n60: the first speed whose kept share interval holds
     0.5 balances best founding under fix B; none -> fix B alone cannot balance best founding.
  2. guard: habitat_track_check median AUC now 0.8 or more at that speed (ranges still follow the habitat).
  3. world_vs_bnm v5.9 bands at the measured detection, reported.
  Expected: kept share about 0.55 at 62,500 and about 0.5 at 15,625; AUC now falls at 15,625 (AUC -10 at 250,000
  was already 0.85).
The niche_t check of HABITAT_tau and the _ht<tau> file pattern apply unchanged; outputs go to results/stage2a/niche_t2/.
Needs the niche_t patch (patch_sweep_stage2a_niche_t.py) and the runner and assembly it needs.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_t2.py
    python3 ../tools/patch_sweep_stage2a_niche_t2.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_t2.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 habitat-tau 2"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_t2 plan is already in sweep_stage2a.py")
if "# obj1_v2 habitat-tau" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_t.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "the same two speeds. n29-n32 and n33 are the twins.", "niche_t docstring line")
edits.append((i + 1, i + 1, [
    "  niche_t2 2 worlds (added 31 Oct): n59, n60 = n33 (best founding, seed 2303) at habitat tau 62,500 and 15,625",
    "         steps; n57, n58 (tau 1,000,000 and 250,000) and n33 (fixed) are the twins."]))
i = find(lambda s: s.strip() == 'if plan == "niche_t":   # obj1_v2 habitat-tau', "niche_t plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_t2":   {MARK}',
    f'{p}    b = dict(kernel="2dt_a1_b3", f=1, attempts=1000, window=5000, niche=True, tradeoff=0.5, contrast=0.25,',
    f'{p}             connectance=1.0, disp=(1e-6, 1e-2))',
    f'{p}    return [dict(b, name=f"n{{59 + j}}", d="1e-6", seed=2303, habitat_seed=100000 + 2303, trait_seed=200000 + 2303,',
    f'{p}                 founding="best", habitat_tau=t) for j, t in enumerate((62500, 15625))]']))
i = find(lambda s: '"niche_s", "niche_w", "niche_t"]' in s and "ap.add_argument(\"--plan\"" in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_s", "niche_w", "niche_t"]', '"niche_s", "niche_w", "niche_t", "niche_t2"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'if a.plan == "niche_t" and ("--habitat-tau" not in src   # obj1_v2 habitat-tau', "niche_t guard")
edits.append((i, i + 1, [L[i].replace('if a.plan == "niche_t" and', 'if a.plan in ("niche_t", "niche_t2") and') + f"   {MARK}"]))

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