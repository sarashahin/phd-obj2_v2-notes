"""
patch_run_all_rps_seed_size.py  (obj1_v2, 24 Oct 2026)

Adds --seed-size to run_all_rps.py: the number of random patches each invader is
founded in. It was hard-coded to 5 (seed_size=5 in the assembly call). Step 2a tests
founding in 1 cell against 5, because founding in 5 random cells makes far fragments
that no record can predict (test_06: 87% of even the oldest species sit in 5 cells
or fewer). The founding number goes into the world tag as _f<n>, before _s<seed>.

Run from obj1_v2/work, after Step 1 and the RPS guard:
    python3 ../tools/patch_run_all_rps_seed_size.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_seed_size.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 seed-size"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: --seed-size is already in run_all_rps.py")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: "master seed of this community (assembly, demography, final run); required with --pool" in s, "--sim-seed help")
edits.append((i + 1, i + 1, [
    f"    p.add_argument('--seed-size', type=int, default=5,   {MARK}",
    "                   help='random patches each invader is founded in (one individual each); was fixed at 5')"]))
i = find(lambda s: "3-species RPS benchmark is switched off" in s, "RPS guard")
edits.append((i + 1, i + 1, [
    f"    if args.seed_size < 1:   {MARK}",
    "        raise SystemExit('--seed-size must be at least 1')"]))
i = find(lambda s: '+ f"_s{args.sim_seed}")' in s, "world tag seed suffix")
edits.append((i, i + 1, [L[i].replace('+ f"_s{args.sim_seed}")', '+ f"_f{args.seed_size}_s{args.sim_seed}")') + f"   {MARK}"]))
i = find(lambda s: s.strip() == "seed_size=5,", "assembly call seed_size=5")
edits.append((i, i + 1, [f"{ind(L[i])}seed_size=args.seed_size,   {MARK}"]))

new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "run_all_rps.py (now)", "run_all_rps.py (patched)", lineterm="", n=1)))
compile(out, "run_all_rps.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: run_all_rps.py")
else:
    print("dry run: nothing written (add --apply to write)")