"""
patch_sweep_stage2a_niche_p.py  (obj1_v2, root causes 24 and 25, pilot on one seed, 1 Nov 2026, Sara's "decide")

Adds plan "niche_p" to sweep_stage2a.py: a one-seed GPU pilot (seed 2303, n66's setting) before the 4-seed batch.
  n70  n66's setting + --habitat-substeps 25 (speed fix alone; twin of n71, n72)
  n71  n70 + competition strength 0.2 (every pair still competes, half as strongly)
  n72  n70 + connectance 0.4 (36% of pairs do not compete; strength 0.4 where they do)
Why: speed (frame_change_check v2) needs finer steps; similarity decay comes from competition sorting
(spatial_cause_check v2: empty suitable cells invadable in 1.1-2.9%), and the CPU shortcut (competition_whatif) failed
its calibration (closed cells hold only 59-70% of the real presences: about a third of presences are kept by
arrivals), so the competition change has to be tried in the full IBM.
Reading rule (written before the run, 1 Nov):
  1. n70: frame_change_check v2 on the sub-step maps is not "too fast"; k* reported. Balance and test_05 reported.
  2. decay (world_vs_bnm v5.10, end map): n71 and n72 each against 0.085. The candidate for the 4-seed batch is the one
     at 0.085 or less that passes the most other bands; if both, the one with more bands; if neither, competition
     is not the lever for decay and the next is cell-level habitat variation (to build).
  3. every world: test_05 losses 2% or less; habitat_track_check guard 0.8 or more.
One seed decides nothing on its own: the batch of 4 seeds decides.
Also: rows can ask for habitat_substeps (saved HABITAT_substeps checked; files without it count as no sub-steps), and
the file pattern names the sub-step part (_hss<M>_). Needs run_all_rps.py with patch_run_all_rps_habitat_substeps.py
and assembly v2.6 in work.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_p.py
    python3 ../tools/patch_sweep_stage2a_niche_p.py
    python3 ../tools/patch_sweep_stage2a_niche_p.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 substeps"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_p plan is already in sweep_stage2a.py")
if "# obj1_v2 habitat-shift 2" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_h2.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "shift tau 62,500) on seeds 2304, 2307, 2308; with n66 (seed 2303) the candidate is judged on 4 worlds.",
         "niche_h2 docstring line")
edits.append((i + 1, i + 1, [
    "  niche_p 3 worlds (added 1 Nov): seed 2303, n66's setting + --habitat-substeps 25 (n70), + strength 0.2 (n71),",
    "         + connectance 0.4 (n72); one-seed pilot before the 4-seed batch."]))
i = find(lambda s: s.strip() == 'if plan == "niche_h2":   # obj1_v2 habitat-shift 2', "niche_h2 plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_p":   {MARK}',
    f'{p}    b = dict(kernel="2dt_a1_b3", f=1, attempts=1000, window=5000, niche=True, tradeoff=0.5, contrast=0.25,',
    f'{p}             connectance=1.0, disp=(1e-6, 1e-2), d="1e-6", seed=2303, habitat_seed=102303, trait_seed=202303,',
    f'{p}             founding="best", habitat_tau=62500, habitat_shift_sd=1.0, habitat_shift_tau=62500, habitat_substeps=25)',
    f'{p}    c04 = {{k: v for k, v in b.items() if k != "connectance"}}   # connectance 0.4 is the runner default: no option, no tag',
    f'{p}    return [dict(b, name="n70"), dict(b, name="n71", strength=0.2), dict(c04, name="n72")]']))
i = find(lambda s: s.strip() == "and shift_ok(z, r))   # obj1_v2 habitat-shift", "niche_ok shift line")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}and shift_ok(z, r)   # obj1_v2 habitat-shift",
    f"{p}and substeps_ok(z, r))   {MARK}"]))
i = find(lambda s: s.strip() == "def shift_ok(z, r):   # obj1_v2 habitat-shift", "def shift_ok")
edits.append((i, i, [
    f"def substeps_ok(z, r):   {MARK}",
    "    \"\"\"Habitat sub-steps saved as the row asks: rows with habitat_substeps=M need HABITAT_substeps M; other rows need",
    "    none (files without the key count as none).\"\"\"",
    "    m = int(z[\"HABITAT_substeps\"]) if \"HABITAT_substeps\" in z.files else None",
    "    return m == (None if r.get(\"habitat_substeps\") is None else int(r[\"habitat_substeps\"]))",
    "",
    ""]))
i = find(lambda s: s.strip() == "ht += f\"hs{sl(r['habitat_shift_sd'])}t{int(r['habitat_shift_tau'])}_\"", "row_pat hs line")
p = ind(L[i - 1])
edits.append((i + 1, i + 1, [
    f"{p}if r.get(\"habitat_substeps\") is not None:   {MARK}",
    f"{p}    ht += f\"hss{{int(r['habitat_substeps'])}}_\""]))
i = find(lambda s: s.strip() == 'base += ["--habitat-shift-sd", str(float(r["habitat_shift_sd"])), "--habitat-shift-tau", str(int(r["habitat_shift_tau"]))]',
         "--habitat-shift-sd in command")
p = ind(L[i - 1])
edits.append((i + 1, i + 1, [
    f'{p}if r.get("habitat_substeps") is not None:   {MARK}',
    f'{p}    base += ["--habitat-substeps", str(int(r["habitat_substeps"]))]']))
i = find(lambda s: '"niche_h", "niche_h2"]' in s and 'ap.add_argument("--plan"' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_h", "niche_h2"]', '"niche_h", "niche_h2", "niche_p"]') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_h", "niche_h2") and ("--habitat-shift-sd" not in src'), "niche_h guard")
p = ind(L[i])
edits.append((i, i + 1, [
    L[i].replace('if a.plan in ("niche_h", "niche_h2") and', 'if a.plan in ("niche_h", "niche_h2", "niche_p") and') + f"   {MARK}"]))
edits.append((i - 0, i, [
    f'{p}if a.plan == "niche_p" and ("--habitat-substeps" not in src or "substeps" not in Path("assembly_stepwise_ibm.py").read_text()):   {MARK}',
    f'{p}    sys.exit("STOP: niche_p needs --habitat-substeps in run_all_rps.py (patch_run_all_rps_habitat_substeps.py) and assembly v2.6 in work")']))

starts = [e[0] for e in edits]
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "sweep_stage2a.py (now)", "sweep_stage2a.py (patched)", lineterm="", n=1)))
compile(out, "sweep_stage2a.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: ../tools/sweep_stage2a.py")
else:
    print("dry run: nothing written (add --apply to write)")