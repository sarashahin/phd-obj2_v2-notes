"""
patch_sweep_stage2a_settle.py  (obj1_v2, 25 Oct 2026; signed off by Sara the same day)

Adds the settling test to sweep_stage2a.py as plan "settle": does a longer assembly
window let the community settle before the final run?

  8 worlds, 2Dt a 1 b 3, founded in 1 cell, 1,000 attempts each:
    s1  d 1e-6, window 1,000, sim seed 2003  = pilot p3, copied from results/stage2a/pilot
        after its metadata (seed, founding, d, kernel, window) is checked
    s2  d 1e-6, window 1,000, sim seed 2302
    s3  d 1e-6, window 5,000, sim seed 2303      s4  as s3, sim seed 2304
    s5  d 3e-6, window 1,000, sim seed 2305      s6  as s5, sim seed 2306
    s7  d 3e-6, window 5,000, sim seed 2307      s8  as s7, sim seed 2308
  Pass (agreed 25 Oct): final-run losses 2% or less, richness change over the last 20% of
  attempts within 2%, and both seeds of a setting give the same verdict on the BNM fit half.

Also: every finished world's saved window_steps is now checked with the other metadata,
and runs.csv gets a window column.

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_settle.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_settle.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 settle"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the settle plan is already in sweep_stage2a.py")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

edits = []
i = find(lambda s: s.strip() == "full   4 d x 3 kernels x 2 founding rules = 24 worlds, 2,000 attempts each,", "docstring full plan")
edits.append((i, i, [
    "  settle 8 worlds (added 25 Oct): 2Dt a 1 b 3, founded in 1 cell, 1,000 attempts; d 1e-6 and 3e-6 x",
    "         assembly window 1,000 and 5,000 steps x 2 seeds. s1 is pilot p3 (seed 2003), copied after",
    "         its metadata is checked; s2-s8 use sim seeds 2302-2308. Asks whether a longer window lets",
    "         the community settle (final-run losses 2% or less, richness change within 2%)."]))
i = find(lambda s: s.strip() == "def plan_rows(plan, reps):", "plan_rows")
edits.append((i + 1, i + 1, [
    f'    if plan == "settle":   {MARK}',
    '        cells = (("1e-6", 1000, 2003), ("1e-6", 1000, 2302), ("1e-6", 5000, 2303), ("1e-6", 5000, 2304),',
    '                 ("3e-6", 1000, 2305), ("3e-6", 1000, 2306), ("3e-6", 5000, 2307), ("3e-6", 5000, 2308))',
    '        return [dict(name=f"s{k}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=w,',
    '                     reuse="results/stage2a/pilot" if s == 2003 else None) for k, (d, w, s) in enumerate(cells, 1)]']))
i = find(lambda s: s.strip() == 'SMOKE = {"--ibm-window-steps": "300", "--tmax": "1000"}', "SMOKE")
edits.append((i + 1, i + 1, [
    "",
    "",
    f"def meta_ok(z, r, smoke):   {MARK}",
    '    """The saved metadata carries the requested setting: seed, founding, window, d and kernel."""',
    '    return (int(z["SIM_SEED"]) == r["seed"] and int(z["seed_size"]) == r["f"]',
    '            and int(z["window_steps"]) == (300 if smoke else int(r.get("window", 1000)))',
    '            and abs(float(z["DISPERSAL_RATE"]) - float(r["d"])) <= 1e-6 * float(r["d"])',
    '            and str(z["DISPERSAL_KERNEL"][0]) == r["kernel"].split("_")[0]',
    '            and (r["kernel"] == "nn4" or (float(z["DISPERSAL_A"]) == float(KERNELS[r["kernel"]][3])',
    '                                          and float(z["DISPERSAL_B"]) == float(KERNELS[r["kernel"]][5]))))']))
i = find(lambda s: s.strip() == "base = list(BASE)", "command base")
edits.append((i + 1, i + 1, [
    f'    if r.get("window"): base[base.index("--ibm-window-steps") + 1] = str(r["window"])   {MARK}']))
i = find(lambda s: s.strip() == 'ap.add_argument("--plan", choices=["pilot", "full"], required=True)', "--plan choices")
edits.append((i, i + 1, [L[i].replace('choices=["pilot", "full"]', 'choices=["pilot", "full", "settle"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == """f"{120 if a.smoke else r['attempts']} attempts")""", "row listing")
edits.append((i, i + 1, [L[i].replace(""" attempts")""", """ attempts, window {300 if a.smoke else r.get('window', 1000)} steps")""")]))
i = find(lambda s: s.strip() == 'fields = ["name", "d", "kernel", "f", "seed", "attempts", "status", "minutes", "species", "file"]', "runs.csv fields")
edits.append((i, i + 1, [L[i].replace('"attempts", "status"', '"attempts", "window", "status"') + f"   {MARK}"]))
i = find(lambda s: s.strip() == "done_rows = list(csv.DictReader(open(runs_csv))) if runs_csv.is_file() else []", "done_rows")
edits.append((i + 1, i + 1, [
    f"for r in rows:   {MARK}: copy a finished identical world, after checking its metadata",
    '    if not r.get("reuse") or a.smoke: continue',
    '    pat = f"*_s2a_*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"',
    "    if sorted(out_dir.glob(pat)): continue",
    '    found = sorted(Path(r["reuse"]).glob(pat))',
    '    if len(found) != 1: sys.exit(f"STOP: {r[\'name\']} reuses one file {pat} from {r[\'reuse\']}; found {len(found)}")',
    "    import numpy as np",
    "    z = np.load(found[0], allow_pickle=True)",
    "    if not meta_ok(z, r, False): sys.exit(f\"STOP: {found[0].name} does not carry the setting of {r['name']}; nothing copied\")",
    "    shutil.copy2(found[0], out_dir / found[0].name)",
    '    dsrc = found[0].with_name(found[0].name.replace("_training.npz", "_dataset.npz"))',
    "    if dsrc.is_file(): shutil.copy2(dsrc, out_dir / dsrc.name)",
    '    done_rows = [x for x in done_rows if x["name"] != r["name"]] + [dict(',
    '        name=r["name"], d=r["d"], kernel=r["kernel"], f=r["f"], seed=r["seed"], attempts=r["attempts"], window=r["window"],',
    '        status=f"ok (copied from {r[\'reuse\']})", minutes="", species=str(int(z["gamma"])), file=found[0].name)]',
    '    with open(runs_csv, "w", newline="") as fh:',
    "        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(done_rows)",
    "    print(f\"{r['name']}: copied from {r['reuse']} after the metadata check ({found[0].name})\")"]))
old = 'row = dict(r, attempts=120 if a.smoke else r["attempts"], minutes=f"{mins:.1f}", species="", file="")'
i = find(lambda s: s.strip() == old, "runs.csv row")
edits.append((i, i + 1, [L[i].replace('species="", file="")', 'species="", file="",') + f"   {MARK}",
                         L[i][: len(L[i]) - len(L[i].lstrip())] + '           window=300 if a.smoke else r.get("window", 1000))']))
first = 'ok_meta = (int(z["SIM_SEED"]) == r["seed"] and int(z["seed_size"]) == r["f"]'
i = find(lambda s: s.strip() == first, "ok_meta block")
rest = ['and abs(float(z["DISPERSAL_RATE"]) - float(r["d"])) <= 1e-6 * float(r["d"])',
        'and str(z["DISPERSAL_KERNEL"][0]) == r["kernel"].split("_")[0]',
        'and (r["kernel"] == "nn4" or (float(z["DISPERSAL_A"]) == float(KERNELS[r["kernel"]][3])',
        'and float(z["DISPERSAL_B"]) == float(KERNELS[r["kernel"]][5]))))']
if [s.strip() for s in L[i + 1:i + 5]] != rest:
    sys.exit("STOP: the ok_meta block differs from the tested file; nothing written")
edits.append((i, i + 5, [L[i][: len(L[i]) - len(L[i].lstrip())] + f"ok_meta = meta_ok(z, r, a.smoke)   {MARK}"]))

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
