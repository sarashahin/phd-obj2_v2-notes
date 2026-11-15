"""
patch_world_vs_bnm_tau.py  (obj1_v2, gate 2: presence threshold for observation, 30 Oct 2026)

world_vs_bnm v5.5: --tau T counts a simulated species as present (for observation and scoring only) in the cells
where it has T or more individuals in the last frame. Up to now presence was 1 individual (P_last_final), so small
populations kept alive by arrivals counted as presences: in plan niche_s the median occupied cell held 196, 53, 22
and 11 individuals at strength 0.4, 0.6, 0.7 and 0.8 (one species alone holds thousands), and 32-37% of occupied
cells switched state every frame. A BNM record needs a recorder to see the butterfly, which is unlikely where only a
handful live. The assembly is unchanged: establishment and removal still use 1 individual (the LVMCM detection
level, one body mass); only what is called "present" when a world is scored changes (step 2b, "tau for observation
only").
  - tau 1 (default): P_last_final exactly as before, so every earlier result is unchanged.
  - tau above 1: individuals = B_last / BODY_MASS, rounded. B_last is float16; up to 1,000 individuals the rounding is
    exact (the float16 spacing near 0.1 is 6e-5, a third of one body mass), so cells are split at tau without error.
  - a file without B_last is skipped at tau above 1, with a line saying so.
One self-test line is added (26 -> 27 PASS lines). Nothing else changes.
Needs world_vs_bnm v5.4 (patch_world_vs_bnm_strength.py).

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_tau.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_tau.py --apply
    python3 ../tools/world_vs_bnm.py --self-test | grep -c PASS          # 27
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
world_vs_bnm.before_tau.py.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 tau"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has --tau")
if "# obj1_v2 strength" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_strength.py (v5.4) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("key, so plan niche_s is not pooled; worlds at 0.4 and files without INTERACTION_STRENGTH"),
         "v5.4 docstring line")
edits.append((i + 1, i + 1, [
    "Version 5.5 (30 Oct): --tau T counts a simulated species as present only in cells with T or more individuals in",
    "  the last frame (B_last), for observation and scoring only; the assembly is unchanged. tau 1 = P_last_final."]))
i = find(lambda s: s.startswith("REF = None") and "set from --split-ref" in s, "REF = None line")
edits.append((i + 1, i + 1, [f"TAU = 1.0                                    {MARK}: presence threshold for observation (--tau)"]))
i = find(lambda s: s.strip() == "def load_sim(pattern, masks=None, each=False):", "def load_sim")
edits.append((i, i, [
    f"def presence(z, tau):   {MARK}",
    "    \"\"\"Presence of every species in every cell for observation: P_last_final at tau 1 (as before); above 1, cells",
    "    with tau or more individuals in the last frame (B_last / BODY_MASS, rounded). None when B_last is missing.\"\"\"",
    "    if tau == 1:",
    "        return z[\"P_last_final\"].astype(bool)",
    "    if \"B_last\" not in z.files:",
    "        return None",
    "    return np.rint(np.asarray(z[\"B_last\"], dtype=np.float64) / float(z[\"BODY_MASS\"])) >= tau",
    "",
    ""]))
i = find(lambda s: s.strip() == "P = z[\"P_last_final\"].astype(bool); h = hashlib.md5(P.tobytes()).hexdigest()", "load_sim presence line")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}P = presence(z, TAU)   {MARK}",
    f"{p}if P is None:",
    f"{p}    print(f\"\\n{{f.split('/')[-1]}}: no B_last, so no presence at tau {{TAU:g}}; skipped\"); continue",
    f"{p}h = hashlib.md5(P.tobytes()).hexdigest()"]))
i = find(lambda s: s.strip() == "ap.add_argument(\"--self-test\", action=\"store_true\")", "--self-test option")
edits.append((i, i, [
    f"ap.add_argument(\"--tau\", type=float, default=1.0,   {MARK}",
    "                help=\"individuals a simulated species needs in a cell to count as present (observation only; default 1)\")"]))
i = find(lambda s: s.strip() == "a = ap.parse_args()", "parse_args line")
edits.append((i + 1, i + 1, [
    f"if a.tau < 1:   {MARK}",
    "    sys.exit(\"STOP: --tau is a number of individuals, 1 or more\")",
    "TAU = float(a.tau)",
    "if TAU != 1 and not a.self_test:",
    "    print(f\"presence for observation: {TAU:g} or more individuals in a cell (the assembly itself used 1)\")"]))
i = find(lambda s: s.strip().startswith("import tempfile") and "obj1_v2 fit-half" in s, "self-test tempfile line")
edits.append((i, i, [
    f"    import tempfile as _tt   {MARK}: presence at tau, known answers",
    "    with _tt.TemporaryDirectory() as tmp:",
    "        Nn = np.array([0, 1, 9, 10, 99, 100, 999, 1000, 5000, 15000, 0, 2], float).reshape(3, 2, 2)",
    "        np.savez(f\"{tmp}/w.npz\", P_last_final=(Nn >= 1).astype(np.uint8), B_last=(Nn * 1e-4).astype(np.float16),",
    "                 BODY_MASS=np.float32(1e-4))",
    "        np.savez(f\"{tmp}/x.npz\", P_last_final=(Nn >= 1).astype(np.uint8), BODY_MASS=np.float32(1e-4))",
    "        zw, zx = np.load(f\"{tmp}/w.npz\"), np.load(f\"{tmp}/x.npz\")",
    "        good = (np.array_equal(presence(zw, 1), Nn >= 1) and all(np.array_equal(presence(zw, t), Nn >= t) for t in (2, 10, 100, 1000))",
    "                and presence(zx, 10) is None and np.array_equal(presence(zx, 1), Nn >= 1)); ok &= good",
    "    print(f\"self-test, presence at tau: tau 1 = P_last_final; tau 2, 10, 100, 1,000 split cells at tau exactly (float16); \"",
    "          f\"no B_last -> skipped: {'PASS' if good else 'FAIL'}\")"]))

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
    Path("../tools/world_vs_bnm.before_tau.py").write_text(src)
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py (the version before is kept as ../tools/world_vs_bnm.before_tau.py)")
else:
    print("dry run: nothing written (add --apply to write)")