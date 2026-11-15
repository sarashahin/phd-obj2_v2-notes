"""
patch_sweep_stage2a_niche_s.py  (obj1_v2, competition strength scan, 29 Oct 2026)

Adds plan "niche_s" to sweep_stage2a.py (Sara's yes, 29 Oct): n33 (seed 2303, contrast 0.25, k 0.5, habitat
scale 3, connectance 1, d per species 1e-6 to 1e-2, LVMCM founding) at a stronger competition link, each run to
2,000 entered newcomers so its richness curve shows whether it levels off:
  n40  strength 0.4 (n33 itself, run to 2,000; the twin at equal effort)
  n37  strength 0.6     n38  strength 0.7     n39  strength 0.8
Same sim seed, habitat seed, trait seed, d range and founding rule as n33; only the strength (and, for all four,
the 2,000 entered newcomers) differ. s 1.0 is left out: with C_ij = C_ii only the species with the highest r can
hold a cell by its own growth (one_cell_strength_check.py), so its answer is known without a run.
Also:
  - niche rows have their saved INTERACTION_STRENGTH checked (files without it count as 0.4, the config value);
  - the file pattern names the strength part (_is0p6_ ...), so n37-n39 are never taken for n40.
A world that stops early because no candidate can enter (FOUND_saturated 1) is kept: that is a result.
Needs run_all_rps.py with patch_run_all_rps_strength.py (and the founding patch) for this plan only.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_s.py
    python3 ../tools/patch_sweep_stage2a_niche_s.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_s.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 strength"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_s plan is already in sweep_stage2a.py")
if "# obj1_v2 founding" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_b.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "n29-n32 are the twins.", "niche_b docstring end")
edits.append((i + 1, i + 1, [
    "  niche_s 4 worlds (added 29 Oct): n33 (seed 2303) run to 2,000 entered newcomers at competition strength 0.4",
    "         (n40, the twin), 0.6 (n37), 0.7 (n38) and 0.8 (n39); n33 (1,000 entered) is the twin of n40."]))
i = find(lambda s: s.strip() == 'if plan == "niche_b":   # obj1_v2 founding', "niche_b plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_s":   {MARK}',
    f'{p}    b = dict(d="1e-6", kernel="2dt_a1_b3", f=1, seed=2303, attempts=2000, window=5000, niche=True,',
    f'{p}             habitat_seed=100000 + 2303, trait_seed=200000 + 2303, tradeoff=0.5, contrast=0.25, connectance=1.0,',
    f'{p}             disp=(1e-6, 1e-2), founding="best")',
    f'{p}    return [dict(b, name="n40")] + [dict(b, name=f"n{{36 + k}}", strength=s) for k, s in enumerate((0.6, 0.7, 0.8), 1)]']))
i = find(lambda s: s.strip() == 'and found_ok(z, r))   # obj1_v2 founding', "niche_ok founding line")
p = ind(L[i])
edits.append((i, i + 1, [
    f'{p}and found_ok(z, r)   # obj1_v2 founding',
    f'{p}and strength_ok(z, r))   {MARK}']))
i = find(lambda s: s.strip() == 'def row_pat(r):   # obj1_v2 contrast v2   # obj1_v2 connectance', "row_pat")
want = '    fb = "fb_" if r.get("founding") == "best" else ""   # obj1_v2 founding'
want_ret = '    return f"*_s2a_*{ls}*{hc}{cn}{dps}{fb}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"'
if L[i + 13] != want or L[i + 14] != want_ret:
    sys.exit("STOP: row_pat is not the niche_b version; nothing written")
edits.append((i, i, [
    f"def strength_ok(z, r):   {MARK}",
    "    \"\"\"Competition strength saved as the row asks: rows with strength=s need INTERACTION_STRENGTH s; other rows need",
    "    0.4 (files without INTERACTION_STRENGTH count as 0.4, the config value).\"\"\"",
    "    s = float(z[\"INTERACTION_STRENGTH\"]) if \"INTERACTION_STRENGTH\" in z.files else 0.4",
    "    return abs(s - float(r.get(\"strength\", 0.4))) < 1e-6",
    "",
    ""]))
edits.append((i + 14, i + 15, [
    f'    st = f"is{{sl(r[\'strength\'])}}_" if r.get("strength") is not None else ""   {MARK}',
    '    return f"*_s2a_*{ls}*{hc}{cn}{dps}{fb}{st}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"']))
i = find(lambda s: s.strip() == 'base += ["--founding", r["founding"]]', "founding option in command")
edits.append((i + 1, i + 1, [
    f'    if r.get("strength") is not None:   {MARK}',
    '        base += ["--interaction-strength", str(r["strength"])]']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c", "niche_f", "niche_d", "niche_b"]' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_d", "niche_b"]', '"niche_d", "niche_b", "niche_s"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'if a.plan == "niche_b" and ("--founding" not in src   # obj1_v2 founding', "niche_b guard")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche_s" and ("--interaction-strength" not in src or "--founding" not in src   {MARK}',
    f'{p}                           or \'founding: str = "random"\' not in Path("assembly_stepwise_ibm.py").read_text()):',
    f'{p}    sys.exit("STOP: niche_s needs --interaction-strength and --founding in run_all_rps.py '
    f'(patch_run_all_rps_strength.py) and assembly v2.3 in work")']))

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