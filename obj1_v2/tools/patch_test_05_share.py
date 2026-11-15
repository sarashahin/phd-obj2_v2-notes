"""
patch_test_05_share.py  (obj1_v2, 26 Oct 2026; signed off by Sara the same day)

test_05 v3 -> v3.1. "species_with_1_or_more" divided by every occupied species, and a
species that fills the whole grid has no empty neighbouring cell, so its value is NaN and
it was counted as below 1 (l1: share 0.483 with a median of 6.77 arrivals). Now the share is
taken over species with at least one empty neighbouring cell, and whole-grid species are
counted on their own (species_without_edge). Nothing else changes.

Run from obj1_v2/work:
    python3 ../tools/patch_test_05_share.py            # dry run, prints the diff
    python3 ../tools/patch_test_05_share.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tests/test_05_assembly_equilibrium.py"); MARK = "species_without_edge"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: test_05 already has the share fix")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: "kernel-aware 24 Oct, version 3 26 Oct)" in s, "version line")
edits.append((i, i + 1, [L[i].replace("version 3 26 Oct)", "version 3.1 26 Oct)")]))
i = find(lambda s: s.strip() == "final run, from the saved counts, d and kernel. Below 1 means ranges cannot move.", "Spread docstring")
edits.append((i + 1, i + 1, [
    "          species_with_1_or_more is the share of species with an empty neighbouring cell that",
    "          get at least one arrival there; species filling the whole grid have no such cell and",
    "          are counted apart (species_without_edge; v3.1, 26 Oct)."]))
i = find(lambda s: s.strip() == "species_with_1_or_more=float(np.nanmean(per_sp[occ] >= 1)))", "share line")
j = i - 1
if L[j].strip() != "out.update(N_per_occupied_cell=float(np.median(N[P])), arrivals_per_edge_cell=float(np.nanmedian(per_sp[occ])),":
    sys.exit("STOP: the line above the share line differs from the tested file; nothing written")
edits.append((j, i + 1, [
    f"{ind(L[j])}has_edge = occ & ~np.isnan(per_sp)                      # whole-grid species have no empty neighbour",
    L[j],
    f"{ind(L[i])}species_with_1_or_more=float(np.mean(per_sp[has_edge] >= 1)) if has_edge.any() else float('nan'),",
    f"{ind(L[i])}species_without_edge=int((occ & np.isnan(per_sp)).sum()))"]))

new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "test_05 (now)", "test_05 (patched)", lineterm="", n=1)))
compile(out, "test_05_assembly_equilibrium.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: ../tests/test_05_assembly_equilibrium.py")
else:
    print("dry run: nothing written (add --apply to write)")
