"""
patch_sweep_stage2a_niche_k.py  (obj1_v2, stage 2d-a, 28 Oct 2026)

Adds plan "niche_k" to sweep_stage2a.py: worlds n5-n8 repeat n1-n4 exactly (sim seed,
habitat seed, trait seed, dispersal, niche settings) except the trade-off strength,
k = 0.5 instead of 1. Optima and widths are drawn the same, so each species differs from
its n1-n4 twin only in its peak. n1-n4 are the control.
Also: niche rows have their saved trade-off strength checked (files without it count as 1).
Needs run_all_rps.py with patch_run_all_rps_tradeoff.py for this plan only.

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_niche_k.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_k.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 tradeoff"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_k plan is already in sweep_stage2a.py")
if "# obj1_v2 niche" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "200000 + sim seed). Each settle twin is the control for the niche effect.", "niche docstring end")
edits.append((i + 1, i + 1, [
    "  niche_k 4 worlds (added 28 Oct): n5-n8 = n1-n4 with trade-off strength 0.5 instead of 1 (softer:",
    "         a niche 5x wider has a peak about 2.2x lower). n1-n4 are the control."]))
i = find(lambda s: s.strip() == 'if plan == "niche":   # obj1_v2 niche', "niche plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_k":   {MARK}',
    f'{p}    return [dict(name=f"n{{k + 4}}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,',
    f'{p}                 habitat_seed=100000 + s, trait_seed=200000 + s, tradeoff=0.5)',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: s.strip() == "and all(abs(float(z[k]) - v) < 1e-6 for k, v in NICHE_SET.items()))", "niche_ok settings line")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}and all(abs(float(z[k]) - v) < 1e-6 for k, v in NICHE_SET.items())",
    f'{p}and abs((float(z["NICHE_tradeoff"]) if "NICHE_tradeoff" in z.files else 1.0) - r.get("tradeoff", 1.0)) < 1e-6)   {MARK}']))
i = find(lambda s: s.strip() == '"--niche-cost", "0.1"]', "end of niche options in command")
edits.append((i + 1, i + 1, [
    f'    if r.get("tradeoff") is not None:   {MARK}',
    '        base += ["--niche-tradeoff", str(r["tradeoff"])]']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend", "niche"]' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('choices=["pilot", "full", "settle", "long", "extend", "niche"]',
                                      'choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'if a.plan == "niche" and "--env-mode" not in src:   # obj1_v2 niche', "niche guard")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche_k" and "--niche-tradeoff" not in src:   {MARK}',
    f'{p}    sys.exit("STOP: run_all_rps.py lacks --niche-tradeoff; apply patch_run_all_rps_tradeoff.py first")']))

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