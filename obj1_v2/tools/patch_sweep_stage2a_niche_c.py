"""
patch_sweep_stage2a_niche_c.py  (obj1_v2, stage 2d-a, 28 Oct 2026; version 2 the same day)

Adds plan "niche_c" to sweep_stage2a.py: n5-n8 repeated with a lower habitat contrast, at two
values in one run, so the dial is bracketed (review of 28 Oct, sketch 28):
  n9-n12   contrast 0.5    n13-n16  contrast 0.25
same sim seeds, habitat seeds, trait seeds, dispersal, niche settings and trade-off k = 0.5 as
n5-n8; the habitat map is the same, only less different from place to place. n5-n8 are the control.
Also:
  - niche rows have their saved habitat contrast checked (files without it count as 1);
  - rows of one plan that share a seed are told apart by their contrast tag (hc0p5, hc0p25) when
    the sweep looks for finished worlds, so n13 is not taken as done because n9 is.
Version 2 adds contrast 0.25 and the tag-aware file pattern. If version 1 (contrast 0.5 only) was
applied, running this upgrades it in place.
Needs run_all_rps.py with patch_run_all_rps_contrast.py for this plan only.

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_niche_c.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_c.py --apply
Every anchor must be found exactly once (the file pattern line exactly twice), or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 contrast"; MARK2 = "# obj1_v2 contrast v2"
src = PATH.read_text()
if MARK2 in src:
    sys.exit("STOP: niche_c version 2 is already in sweep_stage2a.py")
if "# obj1_v2 tradeoff" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_k.py first")
upgrade = MARK in src
L = src.split("\n")


def find(pred, what, n=1):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != n:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly {n}); nothing written")
    return hits if n > 1 else hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
PLAN = lambda p: [
    f'{p}if plan == "niche_c":   {MARK2}',
    f'{p}    return [dict(name=f"n{{8 + 4 * j + k}}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,',
    f'{p}                 habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=c)',
    f'{p}            for j, c in enumerate((0.5, 0.25))',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']
DOC = ["  niche_c 8 worlds (added 28 Oct): n5-n8 with habitat contrast 0.5 (n9-n12) and 0.25 (n13-n16)",
       "         instead of 1 (the same habitat map, less different from place to place). n5-n8 are the control."]
edits = []
if upgrade:
    i = find(lambda s: s.strip().startswith("niche_c 4 worlds (added 28 Oct): n9-n12 = n5-n8 with habitat contrast 0.5"), "version 1 docstring")
    edits.append((i, i + 2, DOC))
    i = find(lambda s: s.strip() == f'if plan == "niche_c":   {MARK}', "version 1 plan")
    if "contrast=0.5)" not in L[i + 2] or "enumerate(" not in L[i + 3]:
        sys.exit("STOP: version 1 plan lines not as expected; nothing written")
    edits.append((i, i + 4, PLAN(ind(L[i]))))
else:
    i = find(lambda s: s.strip() == "a niche 5x wider has a peak about 2.2x lower). n1-n4 are the control.", "niche_k docstring end")
    edits.append((i + 1, i + 1, DOC))
    i = find(lambda s: s.strip() == 'if plan == "niche_k":   # obj1_v2 tradeoff', "niche_k plan")
    edits.append((i, i, PLAN(ind(L[i]))))
    i = find(lambda s: s.strip().startswith('and abs((float(z["NICHE_tradeoff"]) if "NICHE_tradeoff" in z.files else 1.0)'), "niche_ok tradeoff line")
    p = ind(L[i]); line = L[i].split("   # obj1_v2 tradeoff")[0]
    if not line.endswith(")"):
        sys.exit("STOP: niche_ok tradeoff line does not end the expression; nothing written")
    edits.append((i, i + 1, [
        line[:-1] + "   # obj1_v2 tradeoff",
        f'{p}and abs((float(z["HABITAT_contrast"]) if "HABITAT_contrast" in z.files else 1.0) - r.get("contrast", 1.0)) < 1e-6)   {MARK}']))
    i = find(lambda s: s.strip() == 'base += ["--niche-tradeoff", str(r["tradeoff"])]', "tradeoff option in command")
    edits.append((i + 1, i + 1, [
        f'    if r.get("contrast") is not None:   {MARK}',
        '        base += ["--habitat-contrast", str(r["contrast"])]']))
    i = find(lambda s: s.strip() == 'if a.plan == "niche_k" and "--niche-tradeoff" not in src:   # obj1_v2 tradeoff', "niche_k guard")
    p = ind(L[i])
    edits.append((i, i, [
        f'{p}if a.plan == "niche_c" and "--habitat-contrast" not in src:   {MARK}',
        f'{p}    sys.exit("STOP: run_all_rps.py lacks --habitat-contrast; apply patch_run_all_rps_contrast.py first")']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k"' in s, "--plan choices")
if '"niche_c"' not in L[i]:
    edits.append((i, i + 1, [L[i].replace('"niche", "niche_k"]', '"niche", "niche_k", "niche_c"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == "def command(r, smoke=False):", "def command")
edits.append((i, i, [
    f"def row_pat(r):   {MARK2}",
    "    \"\"\"File pattern of a row's training file; a contrast other than 1 adds its tag part (hc0p5, hc0p25).\"\"\"",
    "    c = r.get(\"contrast\")",
    "    mid = \"\" if c in (None, 1, 1.0) else \"hc\" + str(c).replace(\".\", \"p\") + \"_\"",
    "    return f\"*_s2a_*{mid}*_f{r['f']}_s{r['seed']}_training.npz\" if mid else f\"*_s2a_*_f{r['f']}_s{r['seed']}_training.npz\"",
    "",
    ""]))
for i in find(lambda s: s.strip() == "pat = f\"*_s2a_*_f{r['f']}_s{r['seed']}_training.npz\"", "file pattern line", n=2):
    edits.append((i, i + 1, [f"{ind(L[i])}pat = row_pat(r)   {MARK2}"]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "sweep_stage2a.py (now)", "sweep_stage2a.py (patched)", lineterm="", n=1)))
compile(out, "sweep_stage2a.py", "exec")
print(f"\n{len(edits)} edits ({'upgrade of version 1' if upgrade else 'fresh'}); the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: ../tools/sweep_stage2a.py")
else:
    print("dry run: nothing written (add --apply to write)")