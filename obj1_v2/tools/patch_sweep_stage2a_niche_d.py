"""
patch_sweep_stage2a_niche_d.py  (obj1_v2, stage 2d-b, 29 Oct 2026)

Adds plan "niche_d" to sweep_stage2a.py: n29-n32 = n21-n24 (contrast 0.25, k 0.5, habitat scale 3,
connectance 1, same sim seeds 2303, 2304, 2307, 2308, habitat seeds, trait seeds and --disp-rate)
with a dispersal rate per species, log-uniform 1e-6 to 1e-2 (--disp-per-species 1e-06 0.01).
n21-n24 are the twins: the only change is that each species gets its own d.
Also:
  - niche rows have their saved per-species d checked (files without DISP_per_species count as
    shared d);
  - the file pattern names the per-species part (dps1em06-0p01), so n29 is never taken for n21.
Needs run_all_rps.py with patch_run_all_rps_disp.py for this plan only.

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_niche_d.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_d.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 per-species d"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_d plan is already in sweep_stage2a.py")
if "# obj1_v2 connectance" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_f.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "connectance 1 (n21-n24), both (n25-n28); n13-n16 (scale 3, connectance 0.4) are the twins.", "niche_f docstring end")
edits.append((i + 1, i + 1, [
    "  niche_d 4 worlds (added 29 Oct): n29-n32 = n21-n24 with a dispersal rate per species, log-uniform 1e-6 to",
    "         1e-2 (test_14 range). n21-n24 are the twins."]))
i = find(lambda s: s.strip() == 'if plan == "niche_f":   # obj1_v2 connectance', "niche_f plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_d":   {MARK}',
    f'{p}    return [dict(name=f"n{{28 + k}}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,',
    f'{p}                 habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5, contrast=0.25, connectance=1.0,',
    f'{p}                 disp=(1e-6, 1e-2))',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: s.strip().startswith('and abs((float(z["CONNECTANCE"]) if "CONNECTANCE" in z.files else 0.4)'), "niche_ok connectance line")
p = ind(L[i]); line = L[i].split("   # obj1_v2 connectance")[0]
if not line.endswith(")"):
    sys.exit("STOP: niche_ok connectance line does not end the expression; nothing written")
edits.append((i, i + 1, [
    line[:-1] + "   # obj1_v2 connectance",
    f'{p}and disp_ok(z, r))   {MARK}']))
i = find(lambda s: s.strip() == 'def row_pat(r):   # obj1_v2 contrast v2   # obj1_v2 connectance', "row_pat")
want = ['    c = r.get("contrast")',
        '    if c in (None, 1, 1.0) and r.get("connectance") is None and r.get("habitat_ls") is None:',
        '        return f"*_s2a_*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"',
        '    sl = lambda x: str(float(x)).replace(".", "p")']
if L[i + 4:i + 8] != want or not L[i + 11].strip().startswith('return f"*_s2a_*{ls}*{hc}{cn}k*'):
    sys.exit("STOP: row_pat is not the niche_f version; nothing written")
edits.append((i, i, [
    f"def disp_ok(z, r):   {MARK}",
    "    \"\"\"Per-species d saved as the row asks: rows with disp=(d_min, d_max) need DISP_per_species 1 and the same range;",
    "    other rows need shared d (files without DISP_per_species count as shared).\"\"\"",
    "    per = int(z[\"DISP_per_species\"]) if \"DISP_per_species\" in z.files else 0",
    "    if r.get(\"disp\") is None:",
    "        return per == 0",
    "    lo, hi = r[\"disp\"]",
    "    return per == 1 and abs(float(z[\"DISP_d_min\"]) / lo - 1) < 1e-9 and abs(float(z[\"DISP_d_max\"]) / hi - 1) < 1e-9",
    "",
    ""]))
edits.append((i + 5, i + 8, [
    f'    if c in (None, 1, 1.0) and r.get("connectance") is None and r.get("habitat_ls") is None and r.get("disp") is None:   {MARK}',
    '        return f"*_s2a_*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"',
    '    sl = lambda x: str(float(x)).replace(".", "p").replace("-", "m")   # as _sanitize_slug in run_all_rps.py']))
edits.append((i + 11, i + 12, [
    '    dps = f"dps{sl(r[\'disp\'][0])}-{sl(r[\'disp\'][1])}_" if r.get("disp") is not None else ""',
    '    return f"*_s2a_*{ls}*{hc}{cn}{dps}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"']))
i = find(lambda s: s.strip() == 'base += ["--connectance", str(r["connectance"])]', "connectance option in command")
edits.append((i + 1, i + 1, [
    f'    if r.get("disp") is not None:   {MARK}',
    '        base += ["--disp-per-species", repr(float(r["disp"][0])), repr(float(r["disp"][1]))]']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c", "niche_f"]' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_c", "niche_f"]', '"niche_c", "niche_f", "niche_d"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'if a.plan == "niche_f" and "--connectance" not in src:   # obj1_v2 connectance', "niche_f guard")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche_d" and "--disp-per-species" not in src:   {MARK}',
    f'{p}    sys.exit("STOP: run_all_rps.py lacks --disp-per-species; apply patch_run_all_rps_disp.py first")']))

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