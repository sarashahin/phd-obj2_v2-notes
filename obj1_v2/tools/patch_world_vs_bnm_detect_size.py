"""
patch_world_vs_bnm_detect_size.py  (obj1_v2, root cause 22: detection by range size, 30 Oct 2026, Sara's yes)

world_vs_bnm v5.8: --detect-small p_s and --small-cells c. A simulated species with c or fewer present-and-surveyed
cells in a survey window is recorded in each of those cells with probability p_s; every other species with --detect p.
Why: bnm_detection_check.py (30 Oct, BNM fit half) measured p 0.643 for species with 1-5 tetrads, 0.642 for 6-20 and
0.800 for 21+ (all 0.795). The classes differ by 0.16, above the 0.15 limit set before the run, and the band this
moves is the 1-5 cell share (best founding passes it at p 0.8, fails at 0.6; at 0.79, width 0.3 fell from 3/4 to 2/4
worlds). BNM classes count tetrads recorded in any of six years within effort; the simulated class counts present
cells inside the BNM species' known cells it is given (one year of effort), the closest match the survey allows.
  - without --detect-small: exactly v5.7 (same random draw, same records).
  - --detect-small equal to --detect: exactly v5.7 too (one uniform draw per cell, compared with each species' p).
  - records are only removed, never added; the known cells are the same draw as at p 1.
One self-test line is added (29 -> 30 PASS lines). Needs world_vs_bnm v5.7 (patch_world_vs_bnm_detect.py).

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_detect_size.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_detect_size.py --apply
    python3 ../tools/world_vs_bnm.py --self-test | grep -c PASS          # 30
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
world_vs_bnm.before_detect_size.py.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 detect-size"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has --detect-small")
if "# obj1_v2 detect" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_detect.py (v5.7) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "worlds seen through BNM effort only; BNM records already carry their misses). p 1 = as before, no extra draw.",
         "v5.7 docstring line")
edits.append((i + 1, i + 1, [
    "Version 5.8 (30 Oct): --detect-small p_s with --small-cells c: a simulated species with c or fewer present surveyed",
    "  cells in a window is recorded with p_s, others with --detect (BNM fit half: 0.64 for 1-20 tetrads, 0.80 for 21+)."]))
i = find(lambda s: s.startswith("DETECT = 1.0") and "# obj1_v2 detect" in s, "DETECT line")
edits.append((i + 1, i + 1, [
    f"DETECT_SMALL = None                          {MARK}: detection for species with SMALL_CELLS or fewer surveyed cells",
    "SMALL_CELLS = 20"]))
i = find(lambda s: s.strip().startswith("if DETECT < 1:") and "# obj1_v2 detect" in s, "observe detection block")
p = ind(L[i])
want = ["D = np.random.default_rng([seed, 7, len(out)]).random(P.shape) < DETECT", "out.append((P & K & D, K))",
        "else:", "out.append((P & K, K))"]
if [s.strip() for s in L[i + 1:i + 5]] != want:
    sys.exit("STOP: observe is not the v5.7 text; nothing written")
edits.append((i, i + 5, [
    f"{p}if DETECT < 1 or DETECT_SMALL is not None:   {MARK}: one uniform draw per cell (v5.7's stream), compared with each species' p",
    f"{p}    U = np.random.default_rng([seed, 7, len(out)]).random(P.shape)",
    f"{p}    pr = np.full(len(P), DETECT)",
    f"{p}    if DETECT_SMALL is not None:",
    f"{p}        pr[(P & K).reshape(len(P), -1).sum(1) <= SMALL_CELLS] = DETECT_SMALL",
    f"{p}    D = U < pr.reshape((-1,) + (1,) * (P.ndim - 1))",
    f"{p}    out.append((P & K & D, K))",
    f"{p}else:",
    f"{p}    out.append((P & K, K))"]))
i = find(lambda s: s.strip() == "ap.add_argument(\"--self-test\", action=\"store_true\")", "--self-test option")
edits.append((i, i, [
    f"ap.add_argument(\"--detect-small\", type=float, default=None,   {MARK}",
    "                help=\"detection for species with --small-cells or fewer present surveyed cells in a window (default: off)\")",
    "ap.add_argument(\"--small-cells\", type=int, default=20,",
    "                help=\"largest number of present surveyed cells that counts as small (with --detect-small; default 20)\")"]))
i = find(lambda s: s.strip() == "DETECT = float(a.detect)", "DETECT = float(a.detect) line")
edits.append((i + 1, i + 1, [
    f"if a.detect_small is not None:   {MARK}",
    "    if not 0 < a.detect_small <= 1 or a.small_cells < 1:",
    "        sys.exit(\"STOP: --detect-small is a probability above 0 and at most 1; --small-cells is 1 or more\")",
    "    DETECT_SMALL, SMALL_CELLS = float(a.detect_small), int(a.small_cells)",
    "    if not a.self_test:",
    "        print(f\"detection by range size: {DETECT_SMALL:g} for species with {SMALL_CELLS} or fewer present surveyed cells \"",
    "              f\"in a window, {DETECT:g} for the rest\")"]))
i = find(lambda s: s.strip().startswith("f\"recorded, same draw on every call, detection 1 unchanged:"), "v5.7 self-test line")
if not L[i - 1].strip().startswith("print(f\"self-test, detection 0.5:"):
    sys.exit("STOP: the v5.7 self-test print is not the expected text; nothing written")
p = ind(L[i - 1])   # indent of the print statement, not of its continuation line
edits.append((i + 1, i + 1, [
    f"{p}global DETECT_SMALL, SMALL_CELLS   {MARK}: known answers",
    f"{p}rz = np.random.default_rng(12); Gz = 20; Pz = np.zeros((90, Gz * Gz), bool); ztop = Gz * Gz // 2",
    f"{p}for sp in range(30): Pz[sp, rz.choice(ztop, 5, replace=False)] = True              # A: 5 cells, all surveyed",
    f"{p}for sp in range(30, 60): Pz[sp, rz.choice(ztop, 100, replace=False)] = True       # B: 100 cells, all surveyed",
    f"{p}for sp in range(60, 90): Pz[sp, rz.choice(ztop, 10, replace=False)] = True; Pz[sp, ztop + rz.choice(ztop, 20, replace=False)] = True",
    f"{p}Pz = Pz.reshape(90, Gz, Gz); mz = np.zeros((20, Gz, Gz), bool); mz[:, : Gz // 2] = True; Mz = [mz] * 4   # C: 30 cells, 10 surveyed",
    f"{p}DETECT, DETECT_SMALL, SMALL_CELLS = 0.8, None, 20; z0 = observe(Pz, Mz, seed=5)",
    f"{p}DETECT_SMALL = 0.8; z1 = observe(Pz, Mz, seed=5)",
    f"{p}DETECT_SMALL = 0.4; z2 = observe(Pz, Mz, seed=5)",
    f"{p}DETECT, DETECT_SMALL = 1.0, None; z3 = observe(Pz, Mz, seed=5)",
    f"{p}zsh = lambda z, lo, hi, n: sum(int(o[lo:hi].sum()) for o, _ in z) / (4 * 30 * n)",
    f"{p}zsa, zsb, zsc = zsh(z2, 0, 30, 5), zsh(z2, 30, 60, 100), zsh(z2, 60, 90, 10)",
    f"{p}good = (all(np.array_equal(x[0], y[0]) for x, y in zip(z0, z1)) and all(np.array_equal(x[1], y[1]) for x, y in zip(z2, z3))",
    f"{p}        and all(not (y[0] & ~x[0]).any() for x, y in zip(z3, z2)) and all(np.array_equal(x[0][30:60], y[0][30:60]) for x, y in zip(z0, z2))",
    f"{p}        and abs(zsa - 0.4) < 0.08 and abs(zsb - 0.8) < 0.03 and abs(zsc - 0.4) < 0.06",
    f"{p}        and all(np.array_equal(o, Pz & k & (np.random.default_rng([5, 7, w]).random(Pz.shape) < 0.8)) for w, (o, k) in enumerate(z0))); ok &= good",
    f"{p}print(f\"self-test, detection by range size (0.4 at 20 or fewer surveyed cells, else 0.8): recorded {{zsa:.3f}} (5 cells), \"",
    f"{p}      f\"{{zsb:.3f}} (100 cells), {{zsc:.3f}} (30 cells, 10 surveyed); v5.7's draw kept; \"",
    f"{p}      f\"large species untouched; records only removed: {{'PASS' if good else 'FAIL'}}\")"]))

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
    Path("../tools/world_vs_bnm.before_detect_size.py").write_text(src)
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py (the version before is kept as ../tools/world_vs_bnm.before_detect_size.py)")
else:
    print("dry run: nothing written (add --apply to write)")