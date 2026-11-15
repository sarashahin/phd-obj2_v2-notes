"""
patch_world_vs_bnm_strength.py  (obj1_v2, competition strength scan, 29 Oct 2026)

world_vs_bnm v5.4: worlds whose saved INTERACTION_STRENGTH differs from 0.4 get ", strength <s>" after the
connectance in their setting key, so the worlds of plan niche_s (strength 0.6, 0.7, 0.8) are not pooled with
each other or with their twin at 0.4. Worlds at 0.4, and files without the key, keep their label exactly.
Adds one self-test line (25 -> 26 PASS lines). Nothing else changes; score() is untouched.
Needs world_vs_bnm v5.3 (patch_world_vs_bnm_founding.py).

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_strength.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_strength.py --apply
    python3 ../tools/world_vs_bnm.py --self-test | grep -c PASS          # 26
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
world_vs_bnm.before_strength.py.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 strength"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has the strength label")
if "# obj1_v2 founding" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_founding.py (v5.3) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith('Version 5.3 (29 Oct): worlds with LVMCM founding'), "v5.3 docstring line")
if not L[i + 1].strip().startswith("their setting key, so plan niche_b is not pooled with its twins"):
    sys.exit("STOP: v5.3 docstring not as expected; nothing written")
edits.append((i + 2, i + 2, [
    "Version 5.4 (29 Oct): worlds whose saved competition strength is not 0.4 get \", strength <s>\" in their setting",
    "  key, so plan niche_s is not pooled; worlds at 0.4 and files without INTERACTION_STRENGTH keep their label."]))
j = find(lambda s: s.strip().startswith("fnd = (f\"best cell, cap {int(g('FOUND_cap', 100))}\""), "founding label line")
i = find(lambda s: s.strip() == "return (f\"{mode}, {env}, connectance {float(g('CONNECTANCE', 0.4)):g}{dps}, founded {fnd}, \"",
         "setting key return line")
if not j < i <= j + 2:
    sys.exit("STOP: the setting key return line is not right after the founding label; nothing written")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}st = float(g('INTERACTION_STRENGTH', 0.4))   {MARK}",
    f"{p}st = f\", strength {{st:g}}\" if abs(st - 0.4) > 1e-6 else \"\"",
    f"{p}return (f\"{{mode}}, {{env}}, connectance {{float(g('CONNECTANCE', 0.4)):g}}{{st}}{{dps}}, founded {{fnd}}, \""]))
i = find(lambda s: s.strip() == "print(f\"self-test, best founding gives its own setting; random (or not saved) keeps its label: "
                                "{'PASS' if good else 'FAIL'}\")", "founding self-test line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}with _tf.TemporaryDirectory() as tmp:   {MARK}: a strength other than 0.4 gives its own setting",
    f"{p}    b0 = {{k: v for k, v in base.items() if k != 'INTERACTION_STRENGTH'}}",
    f"{p}    for n, extra in dict(s0={{}}, s4=dict(INTERACTION_STRENGTH=np.float32(0.4)), s6=dict(INTERACTION_STRENGTH=np.float32(0.6)),",
    f"{p}                         s7=dict(INTERACTION_STRENGTH=np.float32(0.7))).items():",
    f"{p}        np.savez(f\"{{tmp}}/{{n}}.npz\", **b0, **extra); key[n] = setting_key(f\"{{tmp}}/{{n}}.npz\")",
    f"{p}    good = (key['s0'] == key['s4'] and 'strength' not in key['s0'] and len({{key['s0'], key['s6'], key['s7']}}) == 3",
    f"{p}            and ', strength 0.6,' in key['s6'] and ', strength 0.7,' in key['s7']); ok &= good",
    f"{p}print(f\"self-test, competition strength other than 0.4 gives its own setting; 0.4 (or not saved) keeps its label: \"",
    f"{p}      f\"{{'PASS' if good else 'FAIL'}}\")"]))

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
    Path("../tools/world_vs_bnm.before_strength.py").write_text(src)
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py (the version before is kept as ../tools/world_vs_bnm.before_strength.py)")
else:
    print("dry run: nothing written (add --apply to write)")