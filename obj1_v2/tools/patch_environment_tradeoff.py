"""
patch_environment_tradeoff.py  (obj1_v2, stage 2d-a, 28 Oct 2026)

Adds the trade-off strength k to environment.niche_traits:
    peak = peak_max * (w_min / width) ** k
k = 1 (default) is the equal-area trade-off used so far, computed exactly as before, so
worlds n1-n4 and test_11 stay valid. k = 0.5 is the softer trade-off proposed on 28 Oct:
a niche 5 times wider has a peak about 2.2 times lower (0.67 instead of 0.3 at width 1).
Optimum and width are drawn as before, so for each species only its peak changes with k.
Why (niche worlds n1-n4, 28 Oct): newcomers kept had narrower niches (width 0.32-0.38)
than those arriving (0.44-0.46); almost no common species (2-4% of cases in 51+ cells
against BNM 31%) and too many species (x1.7-1.9 per window).

Run from obj1_v2/work, after patch_environment_niche.py:
    python3 ../tools/patch_environment_tradeoff.py            # dry run, prints the diff
    python3 ../tools/patch_environment_tradeoff.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("environment.py"); MARK = "# obj1_v2 tradeoff"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: environment.py already has the trade-off strength")
if "def niche_traits(" not in src:
    sys.exit("STOP: apply patch_environment_niche.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


edits = []
i = find(lambda s: s.strip() == "def niche_traits(sid, *, seed, opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=NICHE_PEAK_MAX):", "niche_traits signature")
edits.append((i, i + 1, [f"def niche_traits(sid, *, seed, opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=NICHE_PEAK_MAX, tradeoff=1.0):   {MARK}"]))
i = find(lambda s: s.strip() == "rare habitat; width log-uniform in [w_min, w_max]; peak = peak_max * w_min / width, so", "docstring peak line")
j = find(lambda s: s.strip() == "peak x width is the same for all (equal area) and no peak exceeds peak_max.", "docstring equal-area line")
if j != i + 1:
    sys.exit("STOP: docstring lines not next to each other; nothing written")
edits.append((i, j + 1, [
    "    rare habitat; width log-uniform in [w_min, w_max]; peak = peak_max * (w_min / width) ** tradeoff,",
    "    so peak x width ** tradeoff is the same for all and no peak exceeds peak_max. tradeoff 1 (default)",
    "    is the equal-area trade-off, computed exactly as before; 0.5 is softer (wide niches keep more",
    "    peak); 0 means no trade-off. Optimum and width do not depend on tradeoff."]))
i = find(lambda s: s.strip() == "if opt_sd < 0:", "opt_sd check")
edits.append((i, i, [
    f"    if tradeoff < 0:   {MARK}",
    "        raise ValueError(f\"tradeoff must be >= 0, got {tradeoff}\")"]))
i = find(lambda s: s.strip() == "return opt, w, float(peak_max * w_min / w)", "return line")
edits.append((i, i + 1, [
    f"    peak = peak_max * w_min / w if tradeoff == 1 else peak_max * (w_min / w) ** tradeoff   {MARK}",
    "    return opt, w, float(peak)"]))

new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "environment.py (now)", "environment.py (patched)", lineterm="", n=1)))
compile(out, "environment.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: environment.py")
else:
    print("dry run: nothing written (add --apply to write)")