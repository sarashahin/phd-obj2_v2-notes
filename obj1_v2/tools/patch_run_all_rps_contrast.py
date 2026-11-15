"""
patch_run_all_rps_contrast.py  (obj1_v2, stage 2d-a, 28 Oct 2026)

Adds --habitat-contrast a to work/run_all_rps.py: the habitat layer the species see is
a x H, where H is the standardised layer (mean 0, sd 1) from habitat_layer. a = 1 (default)
is every niche world so far, unchanged; a = 0.5 makes the habitat half as different from
place to place inside the window, compared with how wide the niches are.
Why (world_vs_bnm v4, 28 Oct): niche worlds hold the right number of species per cell at
k = 0.5 but turn over 2.5-2.8 times more between cells than BNM, and their cell similarity
falls with distance (0.41-0.47 to 0.16-0.23) where BNM stays almost flat (0.74 to 0.70).
 1. --habitat-contrast a (a > 0; niche mode only).
 2. H_world = a x habitat_layer(...) when a is not 1 (a = 1 leaves H untouched).
 3. World tag: "hc<a>" after the niche part only when a is not 1, so n1-n8 tags do not change.
 4. Saved: HABITAT_contrast; ENV_habitat is the layer the species see (a x H).
Needs patch_run_all_rps_tradeoff.py.

Run from obj1_v2/work:
    python3 ../tools/patch_run_all_rps_contrast.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_contrast.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 contrast"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --habitat-contrast")
if "# obj1_v2 tradeoff" not in src:
    sys.exit("STOP: apply patch_run_all_rps_tradeoff.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("p.add_argument('--niche-tradeoff'"), "--niche-tradeoff option")
edits.append((i + 2, i + 2, [
    f"    p.add_argument('--habitat-contrast', type=float, default=1.0,   {MARK}",
    "                   help='niche mode: the species see contrast x H (H has sd 1); 1 = as before, 0.5 = habitat half as different')"]))
if not L[i + 1].strip().startswith("help='niche mode: trade-off strength k"):
    sys.exit("STOP: --niche-tradeoff help line not where expected; nothing written")
i = find(lambda s: s.strip() == "raise SystemExit('--niche-tradeoff needs --env-mode niche')", "tradeoff guard")
edits.append((i + 1, i + 1, [
    f"    if not args.habitat_contrast > 0:   {MARK}",
    "        raise SystemExit('--habitat-contrast must be > 0')",
    "    if args.env_mode != 'niche' and args.habitat_contrast != 1.0:",
    "        raise SystemExit('--habitat-contrast needs --env-mode niche')"]))
i = find(lambda s: s.strip().startswith("H_world = habitat_layer(NUM_PATCHES_Y, NUM_PATCHES_X, args.habitat_ls,"), "habitat layer line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}if args.habitat_contrast != 1.0:   {MARK}",
    f"{p}    H_world = args.habitat_contrast * H_world"]))
i = find(lambda s: s.strip() == "+ (f\"k{_sanitize_slug(args.niche_tradeoff)}\" if args.niche_tradeoff != 1.0 else \"\"))   # obj1_v2 tradeoff", "tag k part")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}+ (f\"k{{_sanitize_slug(args.niche_tradeoff)}}\" if args.niche_tradeoff != 1.0 else \"\")   # obj1_v2 tradeoff",
    f"{p}+ (f\"hc{{_sanitize_slug(args.habitat_contrast)}}\" if args.habitat_contrast != 1.0 else \"\"))   {MARK}"]))
i = find(lambda s: s.strip() == "\"NICHE_tradeoff\": _np.float32(args.niche_tradeoff),   # obj1_v2 tradeoff", "saved NICHE_tradeoff")
edits.append((i + 1, i + 1, [f"{ind(L[i])}\"HABITAT_contrast\": _np.float32(args.habitat_contrast),   {MARK}"]))

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