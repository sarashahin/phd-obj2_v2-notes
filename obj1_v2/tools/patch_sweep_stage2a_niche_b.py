"""
patch_sweep_stage2a_niche_b.py  (obj1_v2, LVMCM founding, 29 Oct 2026)

Adds plan "niche_b" to sweep_stage2a.py: n33-n36 = n29-n32 (contrast 0.25, k 0.5, habitat scale 3,
connectance 1, d per species 1e-6 to 1e-2, same sim seeds 2303, 2304, 2307, 2308, habitat seeds, trait
seeds and --disp-rate) with LVMCM founding (--founding best): a newcomer enters only if it can grow
somewhere against the residents, as one individual in its best cell, and only entered newcomers count
toward the 1,000 attempts. n29-n32 are the twins: the founding rule is the only change.
Also:
  - niche rows have their saved founding rule checked (files without FOUND_rule count as random);
  - the file pattern names the founding part (_fb_), so n33 is never taken for n29.
Needs run_all_rps.py with patch_run_all_rps_founding.py and assembly v2.3 in work, for this plan only.

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_niche_b.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_b.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 founding"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_b plan is already in sweep_stage2a.py")
if "# obj1_v2 per-species d" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_d.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "1e-2 (test_14 range). n21-n24 are the twins.", "niche_d docstring end")
edits.append((i + 1, i + 1, [
    "  niche_b 4 worlds (added 29 Oct): n33-n36 = n29-n32 with LVMCM founding (--founding best: a newcomer enters",
    "         only if it can grow somewhere, as one individual in its best cell; 1,000 entered newcomers).",
    "         n29-n32 are the twins."]))
i = find(lambda s: s.strip() == 'if plan == "niche_d":   # obj1_v2 per-species d', "niche_d plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_b":   {MARK}',
    f'{p}    return [dict(name=f"n{{32 + k}}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,',
    f'{p}                 habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=0.25, connectance=1.0,',
    f'{p}                 disp=(1e-6, 1e-2), founding="best")',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: s.strip() == 'and disp_ok(z, r))   # obj1_v2 per-species d', "niche_ok disp line")
p = ind(L[i])
edits.append((i, i + 1, [
    f'{p}and disp_ok(z, r)   # obj1_v2 per-species d',
    f'{p}and found_ok(z, r))   {MARK}']))
i = find(lambda s: s.strip() == 'def row_pat(r):   # obj1_v2 contrast v2   # obj1_v2 connectance', "row_pat")
want_if = '    if c in (None, 1, 1.0) and r.get("connectance") is None and r.get("habitat_ls") is None and r.get("disp") is None:   # obj1_v2 per-species d'
want_ret = '    return f"*_s2a_*{ls}*{hc}{cn}{dps}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"'
if L[i + 5] != want_if or L[i + 12] != want_ret:
    sys.exit("STOP: row_pat is not the niche_d version; nothing written")
edits.append((i, i, [
    f"def found_ok(z, r):   {MARK}",
    "    \"\"\"Founding rule saved as the row asks: rows with founding=\"best\" need FOUND_rule best; other rows need random",
    "    (files without FOUND_rule count as random).\"\"\"",
    "    rule = str(z[\"FOUND_rule\"][0]) if \"FOUND_rule\" in z.files else \"random\"",
    "    return rule == r.get(\"founding\", \"random\")",
    "",
    ""]))
edits.append((i + 5, i + 6, [
    '    if (c in (None, 1, 1.0) and r.get("connectance") is None and r.get("habitat_ls") is None and r.get("disp") is None   # obj1_v2 per-species d',
    f'            and r.get("founding") is None):   {MARK}']))
edits.append((i + 12, i + 13, [
    f'    fb = "fb_" if r.get("founding") == "best" else ""   {MARK}',
    '    return f"*_s2a_*{ls}*{hc}{cn}{dps}{fb}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"']))
i = find(lambda s: s.strip() == 'base += ["--disp-per-species", repr(float(r["disp"][0])), repr(float(r["disp"][1]))]', "disp option in command")
edits.append((i + 1, i + 1, [
    f'    if r.get("founding") is not None:   {MARK}',
    '        base += ["--founding", r["founding"]]']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c", "niche_f", "niche_d"]' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_f", "niche_d"]', '"niche_f", "niche_d", "niche_b"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'if a.plan == "niche_d" and "--disp-per-species" not in src:   # obj1_v2 per-species d', "niche_d guard")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche_b" and ("--founding" not in src   {MARK}',
    f'{p}                           or \'founding: str = "random"\' not in Path("assembly_stepwise_ibm.py").read_text()):',
    f'{p}    sys.exit("STOP: niche_b needs --founding in run_all_rps.py (patch_run_all_rps_founding.py) and assembly v2.3 in work")']))

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