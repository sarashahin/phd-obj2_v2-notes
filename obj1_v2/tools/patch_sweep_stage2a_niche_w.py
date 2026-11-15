"""
patch_sweep_stage2a_niche_w.py  (obj1_v2, minimum niche width, 30 Oct 2026)

Adds plan "niche_w" to sweep_stage2a.py (Sara's yes, 30 Oct). species_compare.py (30 Oct) found that best
founding keeps species whose niche widths pile at the lower bound (median 0.21-0.22 against w_min 0.2; random
founding 0.27-0.29; AUC 0.12), so the lower bound sets how finely species split the habitat. This plan raises it:
  n41-n44  w_min 0.3, sim seeds 2303, 2304, 2307, 2308
  n45-n48  w_min 0.4, the same seeds
Everything else is n33-n36 (niche_b): contrast 0.25, k 0.5, habitat scale 3, connectance 1, strength 0.4,
d per species 1e-6 to 1e-2, best founding, 1,000 entered newcomers, the same habitat and trait seeds. n33 (seed
2303, w_min 0.2) is the twin of n41 and n45. With the same trait seed a species keeps its optimum; its width is
redrawn on [w_min, 1] from the same random number, and its peak follows (peak = 1.5 (w_min / w)^0.5).
Also:
  - niche rows have their saved NICHE_w_min checked (files without it count as 0.2, the runner default);
  - the file pattern names the width part (w0p3-1p0), so the two widths of one seed are told apart.
run_all_rps.py has had --niche-w-min since patch_run_all_rps_niche.py (27 Oct); no runner change is needed.

Run from obj1_v2/work:
    md5sum ../tools/sweep_stage2a.py                      # expected to start 09c065a5 (after niche_s)
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_w.py
    python3 ../tools/patch_sweep_stage2a_niche_w.py       # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_w.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 width"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_w plan is already in sweep_stage2a.py")
if "# obj1_v2 strength" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_s.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "(n40, the twin), 0.6 (n37), 0.7 (n38) and 0.8 (n39); n33 (1,000 entered) is the twin of n40.",
         "niche_s docstring end")
edits.append((i + 1, i + 1, [
    "  niche_w 8 worlds (added 30 Oct): n33-n36 with minimum niche width 0.3 (n41-n44) and 0.4 (n45-n48), sim seeds",
    "         2303, 2304, 2307, 2308; n33 (w_min 0.2, seed 2303) is the twin of n41 and n45."]))
i = find(lambda s: s.strip() == 'if plan == "niche_s":   # obj1_v2 strength', "niche_s plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_w":   {MARK}',
    f'{p}    return [dict(name=f"n{{40 + 4 * j + k}}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000,',
    f'{p}                 niche=True, habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=0.25,',
    f'{p}                 connectance=1.0, disp=(1e-6, 1e-2), founding="best", w_min=w)',
    f'{p}            for j, w in enumerate((0.3, 0.4))',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: s.strip() == 'and strength_ok(z, r))   # obj1_v2 strength', "niche_ok strength line")
p = ind(L[i])
edits.append((i, i + 1, [
    f'{p}and strength_ok(z, r)   # obj1_v2 strength',
    f'{p}and wmin_ok(z, r))   {MARK}']))
i = find(lambda s: s.strip() == 'def row_pat(r):   # obj1_v2 contrast v2   # obj1_v2 connectance', "row_pat")
want_fb = '    fb = "fb_" if r.get("founding") == "best" else ""   # obj1_v2 founding'
want_st = '    st = f"is{sl(r[\'strength\'])}_" if r.get("strength") is not None else ""   # obj1_v2 strength'
want_ret = '    return f"*_s2a_*{ls}*{hc}{cn}{dps}{fb}{st}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"'
if L[i + 13] != want_fb or L[i + 14] != want_st or L[i + 15] != want_ret:
    sys.exit("STOP: row_pat is not the niche_s version; nothing written")
edits.append((i, i, [
    f"def wmin_ok(z, r):   {MARK}",
    "    \"\"\"Minimum niche width saved as the row asks: rows with w_min=w need NICHE_w_min w; other rows need 0.2",
    "    (files without NICHE_w_min count as 0.2, the runner default).\"\"\"",
    "    w = float(z[\"NICHE_w_min\"]) if \"NICHE_w_min\" in z.files else 0.2",
    "    return abs(w - float(r.get(\"w_min\", 0.2))) < 1e-6",
    "",
    ""]))
edits.append((i + 15, i + 16, [
    f'    wm = f"w{{sl(r[\'w_min\'])}}-{{sl(r.get(\'w_max\', 1.0))}}*" if r.get("w_min") is not None else ""   {MARK}',
    '    return f"*_s2a_*{ls}*{wm}{hc}{cn}{dps}{fb}{st}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"']))
i = find(lambda s: s.strip() == 'base += ["--interaction-strength", str(r["strength"])]', "strength option in command")
edits.append((i + 1, i + 1, [
    f'    if r.get("w_min") is not None:   {MARK}',
    '        base += ["--niche-w-min", str(r["w_min"])]']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c", "niche_f", "niche_d", "niche_b", "niche_s"]' in s,
         "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_b", "niche_s"]', '"niche_b", "niche_s", "niche_w"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'if a.plan == "niche_s" and ("--interaction-strength" not in src or "--founding" not in src   # obj1_v2 strength',
         "niche_s guard")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche_w" and ("--niche-w-min" not in src or "--founding" not in src   {MARK}',
    f'{p}                           or \'founding: str = "random"\' not in Path("assembly_stepwise_ibm.py").read_text()):',
    f'{p}    sys.exit("STOP: niche_w needs --niche-w-min and --founding in run_all_rps.py and assembly v2.3 in work")']))

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