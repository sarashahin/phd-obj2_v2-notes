"""
patch_sweep_stage2a_long.py  (obj1_v2, 26 Oct 2026; signed off by Sara the same day)

Adds plan "long" to sweep_stage2a.py: is a 10,000-step final run long enough?

  l1  setting and sim seed of settle s3 (d 1e-6, 2Dt a 1 b 3, 1 cell, window 5,000, seed 2303)
  l2  setting and sim seed of settle s7 (d 3e-6, 2Dt a 1 b 3, 1 cell, window 5,000, seed 2307)
  both with a final run of 30,000 steps (--tmax 30000, 150 frames) instead of 10,000.
  Pass (agreed 26 Oct): no species lost after step 10,000.
Same seeds, so assembly should repeat s3 and s7 exactly and the first 10,000 steps of the
final run may too; test_05 --twin checks both (it also tests that runs repeat on the GPU).
The world tag holds no tmax, so l1 and l2 write the same checkpoint name as s3 and s7
(checkpoints are read only with --resume, which the sweep never passes).

Also: the final-run length is checked in the saved file (last IBM_t = tmax) for rows that
set it, shown in the listing, and written to runs.csv (column tmax).

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_long.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_long.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 long"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the long plan is already in sweep_stage2a.py")
if "# obj1_v2 settle" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_settle.py first")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "the community settle (final-run losses 2% or less, richness change within 2%).", "settle docstring end")
edits.append((i + 1, i + 1, [
    "  long   2 worlds (added 26 Oct): settings and sim seeds of settle s3 (2303) and s7 (2307) with a",
    "         30,000-step final run. Pass: no species lost after step 10,000."]))
i = find(lambda s: s.strip() == 'if plan == "settle":   # obj1_v2 settle', "settle plan")
edits.append((i, i, [
    f'{ind(L[i])}if plan == "long":   {MARK}',
    f'{ind(L[i])}    return [dict(name="l1", d="1e-6", kernel="2dt_a1_b3", f=1, seed=2303, attempts=1000, window=5000, tmax=30000),',
    f'{ind(L[i])}            dict(name="l2", d="3e-6", kernel="2dt_a1_b3", f=1, seed=2307, attempts=1000, window=5000, tmax=30000)]']))
i = find(lambda s: s.strip() == "def command(r, smoke=False):", "def command")
edits.append((i, i, [
    f"def tmax_ok(z, r, smoke):   {MARK}",
    '    """Rows that set tmax: the saved final run ends at tmax (1,000 under --smoke)."""',
    '    return "tmax" not in r or abs(float(z["IBM_t"][-1]) - (1000 if smoke else float(r["tmax"]))) < 1e-6',
    "",
    ""]))
i = find(lambda s: s.strip().startswith('if r.get("window"): base[base.index("--ibm-window-steps") + 1]'), "window override")
edits.append((i + 1, i + 1, [f'{ind(L[i])}if r.get("tmax"): base[base.index("--tmax") + 1] = str(r["tmax"])   {MARK}']))
i = find(lambda s: 'choices=["pilot", "full", "settle"]' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('choices=["pilot", "full", "settle"]', 'choices=["pilot", "full", "settle", "long"]') + f"   {MARK}"]))
i = find(lambda s: s.strip().endswith("""window {300 if a.smoke else r.get('window', 1000)} steps")"""), "row listing")
edits.append((i, i + 1, [L[i].replace(""" steps")""", """ steps, final run {1000 if a.smoke else r.get('tmax', 10000)} steps")""")]))
i = find(lambda s: s.strip().startswith('fields = ["name", "d", "kernel", "f", "seed", "attempts", "window", "status"'), "runs.csv fields")
edits.append((i, i + 1, [L[i].replace('"window", "status"', '"window", "tmax", "status"') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'window=300 if a.smoke else r.get("window", 1000))', "runs.csv row window")
edits.append((i, i + 1, [f'{ind(L[i])}window=300 if a.smoke else r.get("window", 1000), tmax=1000 if a.smoke else r.get("tmax", 10000))   {MARK}']))
i = find(lambda s: s.strip() == "ok_meta = meta_ok(z, r, a.smoke)   # obj1_v2 settle", "ok_meta call")
edits.append((i, i + 1, [f"{ind(L[i])}ok_meta = meta_ok(z, r, a.smoke) and tmax_ok(z, r, a.smoke)   {MARK}"]))
i = find(lambda s: s.strip().startswith("print(f\"{r['name']}: {row['status']}, {mins:.1f} min,"), "result print")
edits.append((i, i + 1, [L[i].replace("{row['species']} species,", "{row['species']} species, final run to t {float(z['IBM_t'][-1]):g},") + f"   {MARK}"]))

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
