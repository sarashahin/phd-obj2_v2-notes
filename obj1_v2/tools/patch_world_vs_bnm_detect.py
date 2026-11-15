"""
patch_world_vs_bnm_detect.py  (obj1_v2, root cause 22: detection in the BNM yardstick, 30 Oct 2026, Sara's yes)

world_vs_bnm v5.7: --detect p. A simulated world seen through BNM effort (--survey) has each present species recorded
in each surveyed cell with probability p (default 1 = as before). BNM's own records already carry BNM's misses, so
only the simulated side is thinned. Why: world_vs_bnm assumed BNM never misses a present species in a surveyed cell
("no data to set it", 24 Oct), but check D (9 Oct) found that in tetrads well surveyed in both years a species' cells
overlap between consecutive years by Jaccard 0.40 (1-5 cells), 0.46 (6-20) and 0.67 (21+); if all of that change were
missed records, detection would be 0.57, 0.63 and 0.80 (2J / (1 + J)). Every band (species per cell and per window,
range sizes, rare clumping) is judged through this survey, so the target must be right before the simulator changes.
  - p 1: the code path is exactly the old one (no extra random draw), so every earlier result is unchanged.
  - p below 1: the misses come from their own random stream (seed, 7, window number), so the BNM species whose known
    cells each simulated species gets are the same draw as at p 1; only records are removed, never added.
  - without --survey (true maps) nothing changes.
One self-test line is added (28 -> 29 PASS lines). Needs world_vs_bnm v5.6 (patch_world_vs_bnm_wmin.py).

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_detect.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_detect.py --apply
    python3 ../tools/world_vs_bnm.py --self-test | grep -c PASS          # 29
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
world_vs_bnm.before_detect.py.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 detect"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has --detect")
if "# obj1_v2 width" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_wmin.py (v5.6) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "key, so plan niche_w is not pooled; worlds at 0.2, files without NICHE_w_min and field worlds keep their label.",
         "v5.6 docstring line")
edits.append((i + 1, i + 1, [
    "Version 5.7 (30 Oct): --detect p records each present species in each surveyed cell with probability p (simulated",
    "  worlds seen through BNM effort only; BNM records already carry their misses). p 1 = as before, no extra draw."]))
i = find(lambda s: s.startswith("TAU = 1.0") and "# obj1_v2 tau" in s, "TAU line")
edits.append((i + 1, i + 1, [f"DETECT = 1.0                                 {MARK}: chance a present species is recorded in a surveyed cell (--detect)"]))
i = find(lambda s: s.strip() == "out.append((P & K, K))", "observe append line")
p = ind(L[i])
if L[i - 1].strip() != "K = m[rng.integers(len(m), size=len(P))]":
    sys.exit("STOP: observe is not the version 2 text; nothing written")
edits.append((i, i + 1, [
    f"{p}if DETECT < 1:   {MARK}: misses from their own stream, so the known-cell draw above is unchanged",
    f"{p}    D = np.random.default_rng([seed, 7, len(out)]).random(P.shape) < DETECT",
    f"{p}    out.append((P & K & D, K))",
    f"{p}else:",
    f"{p}    out.append((P & K, K))"]))
i = find(lambda s: s.strip() == "ap.add_argument(\"--self-test\", action=\"store_true\")", "--self-test option")
edits.append((i, i, [
    f"ap.add_argument(\"--detect\", type=float, default=1.0,   {MARK}",
    "                help=\"chance a present simulated species is recorded in a surveyed cell (with --survey; default 1)\")"]))
i = find(lambda s: s.strip() == "TAU = float(a.tau)", "TAU = float(a.tau) line")
edits.append((i + 1, i + 1, [
    f"if not 0 < a.detect <= 1:   {MARK}",
    "    sys.exit(\"STOP: --detect is a probability, above 0 and at most 1\")",
    "DETECT = float(a.detect)",
    "if DETECT != 1 and not a.self_test:",
    "    print(f\"detection: a present simulated species is recorded in a surveyed cell with probability {DETECT:g}\")"]))
i = find(lambda s: s.strip().startswith("import tempfile as _tt") and "# obj1_v2 tau" in s, "tau self-test line")
p = ind(L[i])
edits.append((i, i, [
    f"{p}global DETECT   {MARK}: known answers",
    f"{p}rd = np.random.default_rng(11)",
    f"{p}Pd = rd.random((60, G, G)) < 0.3; Md = [rd.random((20, G, G)) < 0.6 for _ in range(3)]",
    f"{p}DETECT = 1.0; d1 = observe(Pd, Md, seed=3)",
    f"{p}DETECT = 0.5; d2 = observe(Pd, Md, seed=3); d3 = observe(Pd, Md, seed=3)",
    f"{p}DETECT = 1.0; d4 = observe(Pd, Md, seed=3)",
    f"{p}share = sum(int(y[0].sum()) for y in d2) / max(1, sum(int(x[0].sum()) for x in d1))",
    f"{p}good = (all(np.array_equal(x[1], y[1]) for x, y in zip(d1, d2)) and all(not (y[0] & ~x[0]).any() for x, y in zip(d1, d2))",
    f"{p}        and abs(share - 0.5) < 0.03 and all(np.array_equal(x[0], y[0]) for x, y in zip(d2, d3))",
    f"{p}        and all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(d1, d4))); ok &= good",
    f"{p}print(f\"self-test, detection 0.5: same known cells as at 1, records only removed, {{share:.3f}} of present surveyed cells \"",
    f"{p}      f\"recorded, same draw on every call, detection 1 unchanged: {{'PASS' if good else 'FAIL'}}\")"]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "world_vs_bnm.py (now)", "world_vs_bnm.py (patched)", lineterm="", n=1)))
compile(out, "world_vs_bnm.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    Path("../tools/world_vs_bnm.before_detect.py").write_text(src)
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py (the version before is kept as ../tools/world_vs_bnm.before_detect.py)")
else:
    print("dry run: nothing written (add --apply to write)")