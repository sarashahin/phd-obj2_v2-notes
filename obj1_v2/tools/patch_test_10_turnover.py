"""
patch_test_10_turnover.py  (obj1_v2, 27 Oct 2026)

Adds X4 (turnover) to test_10: species ids count invasion attempts (id = attempt at
arrival), so the saved ids show who replaced whom between 1,000 and 2,000 attempts.
Description only; the verdict X3 does not change.

Run from obj1_v2/work:
    python3 ../tools/patch_test_10_turnover.py            # dry run, prints the diff
    python3 ../tools/patch_test_10_turnover.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tests/test_10_saturation_extension.py"); MARK = "X4 turnover"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: test_10 already has X4")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

edits = []
i = find(lambda s: s.strip() == "test_10_saturation_extension.py  (obj1_v2, 26 Oct 2026)", "title line")
edits.append((i, i + 1, ["test_10_saturation_extension.py  (obj1_v2, 26 Oct 2026; X4 turnover added 27 Oct)"]))
i = find(lambda s: s.startswith("X3  Verdict: the changes of all paired worlds are pooled;"), "X3 docstring")
edits.append((i, i, [
    "X4  Turnover (added 27 Oct): species ids count invasion attempts (id = attempt at arrival), so",
    "    the saved ids show who replaced whom. Share of the 1,000-attempt community still present",
    "    after 2,000 attempts, and share of the 2,000-attempt community that arrived after attempt",
    "    1,000; per world, the share of the community that arrived in the last 20% of its attempts.",
    "    Description only, no pass rule."]))
i = find(lambda s: s.strip() == 'change = L["stats"] - S["stats"]; changes.append(change)', "change line")
edits.append((i + 1, i + 1, [
    '    if "species_id" in S["z"].files and "species_id" in L["z"].files:',
    '        ids_s, ids_l = set(S["z"]["species_id"].tolist()), set(L["z"]["species_id"].tolist())',
    '        A_s, A_l = int(S["z"]["ASM_attempts"][-1]), int(L["z"]["ASM_attempts"][-1])',
    '        young = lambda ids, A: np.mean(np.array(sorted(ids)) > 0.8 * A)',
    '        print(f"  X4 turnover: {len(ids_s & ids_l) / len(ids_s):.0%} of the {len(ids_s)} species at {A_s} attempts are still present at {A_l}; "',
    '              f"{np.mean(np.array(sorted(ids_l)) > A_s):.0%} of the {len(ids_l)} species at {A_l} arrived after attempt {A_s}")',
    '        print(f"     arrived in the last 20% of attempts: {young(ids_s, A_s):.0%} of the community at {A_s}, {young(ids_l, A_l):.0%} at {A_l}")',
    '    else:',
    '        print("  X4 turnover: no species_id in the files")']))

new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "test_10 (now)", "test_10 (patched)", lineterm="", n=1)))
compile(out, "test_10_saturation_extension.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: ../tests/test_10_saturation_extension.py")
else:
    print("dry run: nothing written (add --apply to write)")
