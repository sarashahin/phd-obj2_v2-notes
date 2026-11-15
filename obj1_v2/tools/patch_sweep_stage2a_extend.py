"""
patch_sweep_stage2a_extend.py  (obj1_v2, 26 Oct 2026; signed off by Sara the same day)

Adds plan "extend" to sweep_stage2a.py: is assembly saturated after 1,000 attempts?

  e1  setting and sim seed of settle s3 (d 1e-6, 2Dt a 1 b 3, 1 cell, window 5,000, seed 2303)
  e2  setting and sim seed of settle s4 (seed 2304)
  e3  setting and sim seed of settle s7 (d 3e-6, seed 2307)
  e4  setting and sim seed of settle s8 (seed 2308)
  all with 2,000 invasion attempts instead of 1,000; final run 10,000 steps as before.
Same seeds, so each run should repeat its settle twin up to the attempt cap and then go on;
test_10 checks that and compares each world at 1,000 and 2,000 attempts. Saturated (agreed
26 Oct): every change from 1,000 to 2,000 attempts is no larger than the difference between
the two seeds of the same setting at 1,000 attempts. Run after the long control shows
"assembly identical: PASS" (runs repeat on the GPU). About 75 min per world.

Also: for these rows the saved attempt count is checked (at least the requested number).

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_extend.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_extend.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 extend"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the extend plan is already in sweep_stage2a.py")
if "# obj1_v2 long" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_long.py first")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "30,000-step final run. Pass: no species lost after step 10,000.", "long docstring end")
edits.append((i + 1, i + 1, [
    "  extend 4 worlds (added 26 Oct): settings and sim seeds of settle s3, s4, s7, s8 (2303, 2304, 2307,",
    "         2308) with 2,000 attempts. test_10 compares each with its 1,000-attempt twin."]))
i = find(lambda s: s.strip() == 'if plan == "long":   # obj1_v2 long', "long plan")
edits.append((i, i, [
    f'{ind(L[i])}if plan == "extend":   {MARK}',
    f'{ind(L[i])}    return [dict(name=f"e{{k}}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=2000, window=5000, check_attempts=True)',
    f'{ind(L[i])}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: s.strip() == "def command(r, smoke=False):", "def command")
edits.append((i, i, [
    f"def attempts_ok(z, r, smoke):   {MARK}",
    '    """Rows with check_attempts: the saved assembly reached the requested attempts (120 under --smoke)."""',
    '    return not r.get("check_attempts") or int(z["ASM_attempts"][-1]) >= (120 if smoke else int(r["attempts"]))',
    "",
    ""]))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long"]' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('choices=["pilot", "full", "settle", "long"]', 'choices=["pilot", "full", "settle", "long", "extend"]') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith("ok_meta = meta_ok(z, r, a.smoke) and tmax_ok(z, r, a.smoke)"), "ok_meta call")
edits.append((i, i + 1, [f"{ind(L[i])}ok_meta = meta_ok(z, r, a.smoke) and tmax_ok(z, r, a.smoke) and attempts_ok(z, r, a.smoke)   {MARK}"]))

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
