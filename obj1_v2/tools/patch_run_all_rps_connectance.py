"""
patch_run_all_rps_connectance.py  (obj1_v2, stage 2d-a, 28 Oct 2026)

Adds --connectance c to work/run_all_rps.py: the chance that a newcomer's competition link to
each resident exists, in each direction (assembly_utils.draw_interactions). Left out, the config
value is used (0.4, the model default), so every world so far is unchanged. The strength of a
link stays config.INTERACTION_STRENGTH (0.4).
Why (contrast sweep, 28 Oct, sketch 30): with habitat contrast 0.25 the worlds hold 50 species per
window (BNM 34) and 17 per cell; field worlds held 19-28 per cell. At connectance 0.4, 36% of
species pairs never compete, so species that share cells can stack up.
The random draws do not change: the link mask compares the same random numbers with c, so a
world at c = 1 has the same newcomers, founding cells and demography seeds as its twin at 0.4
until their communities differ.
 1. --connectance c (0 < c <= 1), set in assembly_utils before assembly.
 2. World tag: "_cn<c>" before the kernel part only when c is given and differs from the config value.
 3. Saved in every training file: CONNECTANCE and INTERACTION_STRENGTH.

Run from obj1_v2/work:
    python3 ../tools/patch_run_all_rps_connectance.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_connectance.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 connectance"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --connectance")
if "# obj1_v2 contrast" not in src:
    sys.exit("STOP: apply patch_run_all_rps_contrast.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("p.add_argument('--habitat-contrast'"), "--habitat-contrast option")
if not L[i + 1].strip().startswith("help='niche mode: the species see contrast x H"):
    sys.exit("STOP: --habitat-contrast help line not where expected; nothing written")
edits.append((i + 2, i + 2, [
    f"    p.add_argument('--connectance', type=float, default=None,   {MARK}",
    "                   help='chance of a competition link per pair and direction (assembly_utils); default config.CONNECTANCE')"]))
i = find(lambda s: s.strip() == "dispersal.unit_kernel_matrix()                                   # stops here on a bad kernel setting", "kernel check line")
edits.append((i + 1, i + 1, [
    f"    import assembly_utils   {MARK}",
    "    if args.connectance is not None:",
    "        if not 0 < args.connectance <= 1:",
    "            raise SystemExit('--connectance must be in (0, 1]')",
    "        assembly_utils.CONNECTANCE = float(args.connectance)"]))
i = find(lambda s: s.lstrip().startswith('world_tag += (f"_k{dispersal.DISPERSAL_KERNEL}"'), "kernel part of the world tag")
p = ind(L[i])
edits.append((i, i, [
    f"{p}import config as _cfg   {MARK}",
    f"{p}if args.connectance is not None and float(args.connectance) != float(_cfg.CONNECTANCE):",
    f"{p}    world_tag += f\"_cn{{_sanitize_slug(float(args.connectance))}}\""]))
i = find(lambda s: s.strip().startswith('train_out["ENV_mode"] = _np.array([args.env_mode])'), "ENV_mode save line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}import assembly_utils as _au   {MARK}",
    f"{p}train_out[\"CONNECTANCE\"] = _np.float32(_au.CONNECTANCE)",
    f"{p}train_out[\"INTERACTION_STRENGTH\"] = _np.float32(_au.INTERACTION_STRENGTH)"]))

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
    PATH.write_text(out); print("written: run_all_rps.py")
else:
    print("dry run: nothing written (add --apply to write)")