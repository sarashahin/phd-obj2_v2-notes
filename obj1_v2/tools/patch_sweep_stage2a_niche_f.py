"""
patch_sweep_stage2a_niche_f.py  (obj1_v2, stage 2d-a, 28 Oct 2026)

Adds plan "niche_f" to sweep_stage2a.py: a 2 x 2 test at habitat contrast 0.25 (k 0.5), with
n13-n16 (habitat length scale 3, connectance 0.4) as the fourth corner and the twins:
  n17-n20  habitat length scale 1, connectance 0.4
  n21-n24  habitat length scale 3, connectance 1
  n25-n28  habitat length scale 1, connectance 1
same sim seeds, habitat seeds (the same random map, made sharper at scale 1), trait seeds and
dispersal as n13-n16.
Also:
  - niche rows have their saved habitat length scale and connectance checked (files without
    CONNECTANCE count as 0.4);
  - the file pattern used to find finished worlds now names the habitat scale, contrast and
    connectance, so rows of one plan that share a seed are told apart; rows of the older plans
    keep finding their files (checked on the niche_c files).
Needs run_all_rps.py with patch_run_all_rps_connectance.py for this plan only.

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_niche_f.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_f.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 connectance"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_f plan is already in sweep_stage2a.py")
if "# obj1_v2 contrast v2" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_c.py (version 2) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "instead of 1 (the same habitat map, less different from place to place). n5-n8 are the control.", "niche_c docstring end")
edits.append((i + 1, i + 1, [
    "  niche_f 12 worlds (added 28 Oct): 2 x 2 at contrast 0.25, k 0.5: habitat length scale 1 (n17-n20),",
    "         connectance 1 (n21-n24), both (n25-n28); n13-n16 (scale 3, connectance 0.4) are the twins."]))
i = find(lambda s: s.strip() == 'if plan == "niche_c":   # obj1_v2 contrast v2', "niche_c plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_f":   {MARK}',
    f'{p}    return [dict(name=f"n{{16 + 4 * j + k}}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,',
    f'{p}                 habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=0.25, **x)',
    f'{p}            for j, x in enumerate((dict(habitat_ls=1.0), dict(connectance=1.0), dict(habitat_ls=1.0, connectance=1.0)))',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: s.strip() == "and all(abs(float(z[k]) - v) < 1e-6 for k, v in NICHE_SET.items())", "niche_ok settings line")
p = ind(L[i])
edits.append((i, i + 1, [f'{p}and all(abs(float(z[k]) - v) < 1e-6 for k, v in {{**NICHE_SET, "HABITAT_ls": float(r.get("habitat_ls", 3.0))}}.items())   {MARK}']))
i = find(lambda s: s.strip().startswith('and abs((float(z["HABITAT_contrast"]) if "HABITAT_contrast" in z.files else 1.0)'), "niche_ok contrast line")
p = ind(L[i]); line = L[i].split("   # obj1_v2 contrast")[0]
if not line.endswith(")"):
    sys.exit("STOP: niche_ok contrast line does not end the expression; nothing written")
edits.append((i, i + 1, [
    line[:-1] + "   # obj1_v2 contrast",
    f'{p}and abs((float(z["CONNECTANCE"]) if "CONNECTANCE" in z.files else 0.4) - r.get("connectance", 0.4)) < 1e-6)   {MARK}']))
i = find(lambda s: s.strip() == 'def row_pat(r):   # obj1_v2 contrast v2', "row_pat")
j = i + 1
while j < len(L) and L[j].strip():
    j += 1
if j - i != 5:
    sys.exit(f"STOP: row_pat has {j - i} lines, expected 5; nothing written")
edits.append((i, j, [
    f"def row_pat(r):   # obj1_v2 contrast v2   {MARK}",
    "    \"\"\"File pattern of a row's training file. Niche rows with a contrast, habitat scale or connectance",
    "    name them as the world tag does (ls<scale>o ... hc<contrast>_[cn<connectance>_]k), so rows of one plan",
    "    that share a seed are told apart; other rows keep the plain pattern.\"\"\"",
    "    c = r.get(\"contrast\")",
    "    if c in (None, 1, 1.0) and r.get(\"connectance\") is None and r.get(\"habitat_ls\") is None:",
    "        return f\"*_s2a_*_f{r['f']}_s{r['seed']}_training.npz\"",
    "    sl = lambda x: str(float(x)).replace(\".\", \"p\")",
    "    ls = f\"ls{sl(r.get('habitat_ls', 3.0))}o\" if r.get(\"niche\") else \"\"",
    "    hc = f\"hc{sl(c)}_\" if c not in (None, 1, 1.0) else \"\"",
    "    cn = f\"cn{sl(r['connectance'])}_\" if r.get(\"connectance\") is not None else \"\"",
    "    return f\"*_s2a_*{ls}*{hc}{cn}k*_f{r['f']}_s{r['seed']}_training.npz\""]))
i = find(lambda s: s.strip() == 'base += ["--habitat-contrast", str(r["contrast"])]', "contrast option in command")
edits.append((i + 1, i + 1, [
    f'    if r.get("habitat_ls") is not None:   {MARK}',
    '        base[base.index("--habitat-ls") + 1] = str(r["habitat_ls"])',
    '    if r.get("connectance") is not None:',
    '        base += ["--connectance", str(r["connectance"])]']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c"]' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_k", "niche_c"]', '"niche_k", "niche_c", "niche_f"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'if a.plan == "niche_c" and "--habitat-contrast" not in src:   # obj1_v2 contrast', "niche_c guard")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche_f" and "--connectance" not in src:   {MARK}',
    f'{p}    sys.exit("STOP: run_all_rps.py lacks --connectance; apply patch_run_all_rps_connectance.py first")']))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "sweep_stage2a.py (now)", "sweep_stage2a.py (patched)", lineterm="", n=1)))
compile(out, "sweep_stage2a.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: ../tools/sweep_stage2a.py")
else:
    print("dry run: nothing written (add --apply to write)")