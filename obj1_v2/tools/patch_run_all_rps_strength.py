"""
patch_run_all_rps_strength.py  (obj1_v2, competition strength, 29 Oct 2026)

Adds --interaction-strength s to work/run_all_rps.py: the value of every competition link C_ij (i != j) drawn
by assembly_utils.draw_interactions; C_ii stays 1. Left out, the config value is used (0.4, the model
default), so every world so far is unchanged.
Why (n33, 29 Oct, sketch 38): with LVMCM founding the world holds 3.07 times BNM's species per cell. In the
Lotka-Volterra rule the species that can hold a cell by their own growth fall as s rises (one_cell_strength_check.py:
0.47-0.51 of the s 0.4 count at 0.6, 0.35-0.39 at 0.7, 0.25-0.26 at 0.8).
The random draws do not change: draw_interactions compares the same random numbers with the connectance and
only writes s instead of 0.4 into the links, so a world at s 0.7 tests the same candidates in the same order as
its twin at 0.4 until their communities differ.
 1. --interaction-strength s (0 < s <= 1), set in assembly_utils before assembly, next to --connectance.
    Above 1 another species would hurt a species more than its own kind does; that is left out.
 2. World tag: "_is<s>" before the kernel part (after "_fb") only when s is given and differs from the config value.
 3. Saved: INTERACTION_STRENGTH is already saved in every training file (patch_run_all_rps_connectance.py),
    read from assembly_utils, so it records the value used.
Needs run_all_rps.py with patch_run_all_rps_connectance.py and patch_run_all_rps_founding.py.

Run from obj1_v2/work:
    python3 ../tools/patch_run_all_rps_strength.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_strength.py --apply
    USE_GPU=0 python3 ../tests/test_17_strength.py            # about 2-5 minutes on the CPU
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
run_all_rps.before_strength.py.
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 strength"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --interaction-strength")
if "# obj1_v2 connectance" not in src or "# obj1_v2 founding" not in src:
    sys.exit("STOP: apply patch_run_all_rps_connectance.py and patch_run_all_rps_founding.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("p.add_argument('--connectance'"), "--connectance option")
if not L[i + 1].strip().startswith("help='chance of a competition link per pair and direction"):
    sys.exit("STOP: --connectance help line not where expected; nothing written")
edits.append((i + 2, i + 2, [
    f"    p.add_argument('--interaction-strength', type=float, default=None,   {MARK}",
    "                   help='value of every competition link C_ij, i != j (assembly_utils); default config.INTERACTION_STRENGTH')"]))
j = find(lambda s: s.strip() == "import assembly_utils   # obj1_v2 connectance", "connectance block")
i = find(lambda s: s.strip() == "assembly_utils.CONNECTANCE = float(args.connectance)", "connectance set line")
if not j < i <= j + 5:
    sys.exit("STOP: connectance set line not inside the connectance block; nothing written")
p = ind(L[j])
edits.append((i + 1, i + 1, [
    f"{p}if args.interaction_strength is not None:   {MARK}",
    f"{p}    if not 0 < args.interaction_strength <= 1:",
    f"{p}        raise SystemExit('--interaction-strength must be in (0, 1]')",
    f"{p}    assembly_utils.INTERACTION_STRENGTH = float(args.interaction_strength)"]))
i = find(lambda s: s.lstrip().startswith('world_tag += (f"_k{dispersal.DISPERSAL_KERNEL}"'), "kernel part of the world tag")
if not any("# obj1_v2 founding" in s for s in L[max(0, i - 3):i]):
    sys.exit("STOP: the founding part of the world tag is not just before the kernel part; nothing written")
p = ind(L[i])
edits.append((i, i, [
    f"{p}import config as _cfg   {MARK}",
    f"{p}if args.interaction_strength is not None and float(args.interaction_strength) != float(_cfg.INTERACTION_STRENGTH):",
    f"{p}    world_tag += f\"_is{{_sanitize_slug(float(args.interaction_strength))}}\""]))
find(lambda s: s.strip() == 'train_out["INTERACTION_STRENGTH"] = _np.float32(_au.INTERACTION_STRENGTH)', "INTERACTION_STRENGTH save line")

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "run_all_rps.py (now)", "run_all_rps.py (patched)", lineterm="", n=1)))
compile(out, "run_all_rps.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    Path("run_all_rps.before_strength.py").write_text(src)
    PATH.write_text(out); print("written: run_all_rps.py (the version before is kept as run_all_rps.before_strength.py)")
else:
    print("dry run: nothing written (add --apply to write)")