"""
patch_sweep_stage2a_niche_w2.py  (obj1_v2, minimum niche width, version 2, 30 Oct 2026)

Fixes plan niche_w after Sara's smoke run (30 Oct): all 8 smoke worlds were made with the right width (their files
carry w0p3-1p0 and w0p4-1p0) but went to rejected/ with "metadata mismatch". Cause, in my version 1: niche_ok checks
the saved niche settings against NICHE_SET, which fixes NICHE_w_min at 0.2 (patch_sweep_stage2a_niche.py, 27 Oct),
so every world at 0.3 or 0.4 failed. Version 1 was tested on a stand-in without that line. Second issue: the command
kept the niche block's "--niche-w-min 0.2" and appended "--niche-w-min 0.3"; the runner took the last one, but the
command carried two values.
  1. niche_ok: the NICHE_SET check takes the row's w_min (0.2 for rows without one), the way niche_f made it take
     the row's habitat scale. Every older row still checks 0.2.
  2. command: --niche-w-min is set in place in the niche block, so the command holds one value.
Run from obj1_v2/work, after patch_sweep_stage2a_niche_w.py (version 1) was applied:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_w2.py
    python3 ../tools/patch_sweep_stage2a_niche_w2.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_w2.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 width v2"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: version 2 of the niche_w plan is already in sweep_stage2a.py")
if "# obj1_v2 width" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_w.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


OLD_SET = '{**NICHE_SET, "HABITAT_ls": float(r.get("habitat_ls", 3.0))}'
NEW_SET = '{**NICHE_SET, "HABITAT_ls": float(r.get("habitat_ls", 3.0)), "NICHE_w_min": float(r.get("w_min", 0.2))}'
edits = []
i = find(lambda s: s.strip() == "2303, 2304, 2307, 2308; n33 (w_min 0.2, seed 2303) is the twin of n41 and n45.", "niche_w docstring end")
edits.append((i + 1, i + 1, [
    "         v2 (30 Oct): the niche settings check takes the row's w_min; the command sets --niche-w-min in place."]))
i = find(lambda s: OLD_SET in s and "and all(abs(float(z[k]) - v) < 1e-6 for k, v in" in s, "NICHE_SET line in niche_ok")
edits.append((i, i + 1, [L[i].replace(OLD_SET, NEW_SET) + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'base += ["--niche-w-min", str(r["w_min"])]', "appended --niche-w-min in command")
if L[i - 1].strip() != 'if r.get("w_min") is not None:   # obj1_v2 width':
    sys.exit("STOP: the --niche-w-min lines are not those of version 1; nothing written")
edits.append((i, i + 1, [f'{L[i][: len(L[i]) - len(L[i].lstrip())]}base[base.index("--niche-w-min") + 1] = str(r["w_min"])   {MARK}: '
                         'in place, one value']))

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