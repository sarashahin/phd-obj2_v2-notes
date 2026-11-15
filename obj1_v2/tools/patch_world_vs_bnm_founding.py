"""
patch_world_vs_bnm_founding.py  (obj1_v2, LVMCM founding, 29 Oct 2026)

world_vs_bnm.py v5.2 groups worlds into settings by mode, trade-off, contrast, habitat scale,
connectance, per-species d, founding cells, kernel and window. Plan niche_b (n33-n36) differs from
its twins n29-n32 only in the founding rule, so without this patch the 8 worlds would be pooled.
Version 5.3 writes "founded best cell, cap <K>" in the setting key of worlds saved with FOUND_rule
"best"; other worlds (FOUND_rule "random" or not saved) keep "founded in <seed size>" as it is.
A self-test line is added.

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_founding.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_founding.py --apply
    python3 ../tools/world_vs_bnm.py --self-test
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 founding"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has version 5.3")
if "# obj1_v2 per-species d" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_disp.py (version 5.2) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


edits = []
i = find(lambda s: s.strip() == "<DMIN>-<DMAX>\" in their setting key, so plan niche_d is not pooled with its twins.", "version 5.2 docstring")
edits.append((i + 1, i + 1, [
    "Version 5.3 (29 Oct): worlds with LVMCM founding (FOUND_rule \"best\") get \"founded best cell, cap <K>\" in",
    "  their setting key, so plan niche_b is not pooled with its twins; others keep \"founded in <seed size>\"."]))
i = find(lambda s: s.strip() == "return (f\"{mode}, {env}, connectance {float(g('CONNECTANCE', 0.4)):g}{dps}, founded in {g('seed_size', '?')}, \"",
         "setting_key return")
if L[i + 1].strip() != "f\"kernel {g('DISPERSAL_KERNEL', '?')}, window {g('window_steps', '?')}\")":
    sys.exit("STOP: setting_key return is not the version 5.2 text; nothing written")
edits.append((i, i + 2, [
    f"    fnd = (f\"best cell, cap {{int(g('FOUND_cap', 100))}}\" if str(g('FOUND_rule', 'random')) == 'best'   {MARK}",
    "           else f\"in {g('seed_size', '?')}\")",
    "    return (f\"{mode}, {env}, connectance {float(g('CONNECTANCE', 0.4)):g}{dps}, founded {fnd}, \"",
    "            f\"kernel {g('DISPERSAL_KERNEL', '?')}, window {g('window_steps', '?')}\")"]))
i = find(lambda s: s.strip() == "print(f\"self-test, per-species d gives its own setting; shared d (0 or not saved) keeps its label: {'PASS' if good else 'FAIL'}\")",
         "version 5.2 self-test print")
edits.append((i + 1, i + 1, [
    f"    with _tf.TemporaryDirectory() as tmp:   {MARK}: best founding splits settings; random (or not saved) keeps its label",
    "        for n, extra in dict(a={}, b=dict(FOUND_rule=np.array(['random'])),",
    "                             c=dict(FOUND_rule=np.array(['best']), FOUND_cap=np.int64(100))).items():",
    "            np.savez(f\"{tmp}/{n}.npz\", **base, **extra); key[n] = setting_key(f\"{tmp}/{n}.npz\")",
    "    good = (key['a'] == key['b'] and 'founded in 1, kernel' in key['a'] and key['c'] != key['a']",
    "            and 'founded best cell, cap 100, kernel' in key['c']); ok &= good",
    "    print(f\"self-test, best founding gives its own setting; random (or not saved) keeps its label: {'PASS' if good else 'FAIL'}\")"]))

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
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py")
else:
    print("dry run: nothing written (add --apply to write)")