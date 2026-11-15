"""
patch_sweep_stage2a_p4.py  (obj1_v2, 25 Oct 2026)

Adds pilot world p4 to sweep_stage2a.py: d = BODY_MASS x 3e-3 = 1e-4 x 3e-3 = 3e-7,
four neighbours, founded in 5 cells, sim seed 2004, 1,000 attempts. Everything else
is p1's setting, so p4 against p1 and p2 changes d only (1e-6 to 3e-7), and p1 against
p2 gives the seed-to-seed spread that the d effect must exceed.

d is passed with --disp-rate, as for every other world; config.py is not edited, so
the default d and the earlier tests stay as they were.

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_p4.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_p4.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 p4"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: p4 is already in sweep_stage2a.py")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

edits = []
i = find(lambda s: s.strip() == "pilot  3 worlds, 1,000 invasion attempts each. Checks the pipeline and run time on",
         "docstring pilot count")
edits.append((i, i + 1, [L[i].replace("pilot  3 worlds", "pilot  4 worlds")]))
i = find(lambda s: s.strip() == "p3  d 1e-6, 2dt a 1 b 3, founded in 1 cell,   sim seed 2003", "docstring p3 line")
edits.append((i + 1, i + 1, [
    "           p4  as p1 with d 3e-7 (BODY_MASS x 3e-3),       sim seed 2004 (added 25 Oct;",
    "               against p1 and p2 it changes d only)"]))
old = 'dict(name="p3", d="1e-6", kernel="2dt_a1_b3", f=1, seed=2003, attempts=1000)]'
i = find(lambda s: s.strip() == old, "p3 row in plan_rows")
pad = L[i][: len(L[i]) - len(L[i].lstrip())]
edits.append((i, i + 1, [L[i].replace(old, old[:-1] + ","),
                         f'{pad}dict(name="p4", d="3e-7", kernel="nn4", f=5, seed=2004, attempts=1000)]   {MARK}']))

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
