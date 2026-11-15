"""
patch_sweep_stage2a_niche_h.py  (obj1_v2, root cause 23, fix B part 2, 31 Oct 2026, Sara's yes)

Adds plan "niche_h" to sweep_stage2a.py: the pilot of the regional habitat shift (assembly v2.5, runner
--habitat-shift-sd and --habitat-shift-tau). Twins, one change each (the shift):
  n61-n64  n49-n52 (random founding, habitat tau 1,000,000; 4 of 8 bands) + shift sd 1.0, shift tau 1,000,000
  n65, n66 n59 (best founding, seed 2303, habitat tau 62,500; balance ratio 0.89, rare-species bands pass)
           + shift sd 0.5 and 1.0, shift tau 62,500
Shift tau = the twin's habitat tau (Sara, 31 Oct). Everything else is the twin's.
Also: rows are checked for the saved shift (HABITAT_shift_sd and HABITAT_shift_tau; files without them count as no
shift), and the file pattern names the shift part (ht<tau>_hs<sd>t<tau>_), so twins and shift worlds are told apart.
Reading rule (written before the build, 31 Oct; claude/n33_result_29oct.md):
  1. richness_balance_check --chain twin -> shift, per founding rule: kept share interval holds 0.5 and balance ratio
     mu / (0.05 f (1 - mu)) of 1 or more -> gate 1 fix found for that rule.
  2. habitat_track_check guard 0.8 or more.
  3. resident_loss_check against the twin: R3 and the share of resident-rounds at 51+ cells.
  4. world_vs_bnm v5.10 bands reported; for best founding the 1-5 share and rare clumping must not be lost.
Needs run_all_rps.py with patch_run_all_rps_habitat_shift.py and assembly v2.5 in work, for this plan only.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_h.py
    python3 ../tools/patch_sweep_stage2a_niche_h.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_h.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 habitat-shift"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_h plan is already in sweep_stage2a.py")
if "# obj1_v2 habitat-tau 2" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_t.py and patch_sweep_stage2a_niche_t2.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "steps; n57, n58 (tau 1,000,000 and 250,000) and n33 (fixed) are the twins.", "niche_t2 docstring line")
edits.append((i + 1, i + 1, [
    "  niche_h 6 worlds (added 31 Oct): the whole habitat layer also shifts (--habitat-shift-sd, --habitat-shift-tau).",
    "         n61-n64 = n49-n52 + shift sd 1.0, shift tau 1,000,000; n65, n66 = n59 + shift sd 0.5 and 1.0, shift tau",
    "         62,500. n49-n52 and n59 are the twins."]))
i = find(lambda s: s.strip() == 'if plan == "niche_t2":   # obj1_v2 habitat-tau 2', "niche_t2 plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_h":   {MARK}',
    f'{p}    b = dict(kernel="2dt_a1_b3", f=1, attempts=1000, window=5000, niche=True, tradeoff=0.5, contrast=0.25,',
    f'{p}             connectance=1.0, disp=(1e-6, 1e-2))',
    f'{p}    rand = [dict(b, name=f"n{{60 + k}}", d=d, seed=s, habitat_seed=100000 + s, trait_seed=200000 + s, habitat_tau=1000000,',
    f'{p}                 habitat_shift_sd=1.0, habitat_shift_tau=1000000)',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]',
    f'{p}    best = [dict(b, name=f"n{{65 + j}}", d="1e-6", seed=2303, habitat_seed=100000 + 2303, trait_seed=200000 + 2303,',
    f'{p}                 founding="best", habitat_tau=62500, habitat_shift_sd=sd, habitat_shift_tau=62500)',
    f'{p}            for j, sd in enumerate((0.5, 1.0))]',
    f'{p}    return rand + best']))
i = find(lambda s: s.strip() == "and tau_ok(z, r))   # obj1_v2 habitat-tau", "niche_ok tau line")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}and tau_ok(z, r)   # obj1_v2 habitat-tau",
    f"{p}and shift_ok(z, r))   {MARK}"]))
i = find(lambda s: s.strip() == "def tau_ok(z, r):   # obj1_v2 habitat-tau", "def tau_ok")
edits.append((i, i, [
    f"def shift_ok(z, r):   {MARK}",
    "    \"\"\"Shift saved as the row asks: rows with habitat_shift_sd need HABITAT_shift_sd and HABITAT_shift_tau as given;",
    "    other rows need no shift (files without the keys count as no shift).\"\"\"",
    "    sd = float(z[\"HABITAT_shift_sd\"]) if \"HABITAT_shift_sd\" in z.files else 0.0",
    "    if r.get(\"habitat_shift_sd\") is None:",
    "        return sd == 0.0",
    "    return (abs(sd - float(r[\"habitat_shift_sd\"])) < 1e-9 and \"HABITAT_shift_tau\" in z.files",
    "            and int(z[\"HABITAT_shift_tau\"]) == int(r[\"habitat_shift_tau\"]))",
    "",
    ""]))
i = find(lambda s: s.strip() == 'ht = f"ht{int(r[\'habitat_tau\'])}_" if r.get("habitat_tau") is not None else ""   # obj1_v2 habitat-tau',
         "row_pat ht line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f'{p}if r.get("habitat_shift_sd") is not None:   {MARK}',
    f'{p}    ht += f"hs{{sl(r[\'habitat_shift_sd\'])}}t{{int(r[\'habitat_shift_tau\'])}}_"']))
i = find(lambda s: s.strip() == 'base += ["--habitat-tau", str(int(r["habitat_tau"]))]', "--habitat-tau in command")
p = ind(L[i - 1])
edits.append((i + 1, i + 1, [
    f'{p}if r.get("habitat_shift_sd") is not None:   {MARK}',
    f'{p}    base += ["--habitat-shift-sd", str(float(r["habitat_shift_sd"])), "--habitat-shift-tau", str(int(r["habitat_shift_tau"]))]']))
i = find(lambda s: '"niche_t", "niche_t2"]' in s and 'ap.add_argument("--plan"' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_t", "niche_t2"]', '"niche_t", "niche_t2", "niche_h"]') + f"   {MARK}"]))
i = find(lambda s: s.strip().startswith('if a.plan in ("niche_t", "niche_t2") and ("--habitat-tau" not in src'), "niche_t guard")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche_h" and ("--habitat-shift-sd" not in src   {MARK}',
    f'{p}                            or "shift_sd" not in Path("assembly_stepwise_ibm.py").read_text()):',
    f'{p}    sys.exit("STOP: niche_h needs --habitat-shift-sd in run_all_rps.py (patch_run_all_rps_habitat_shift.py) and assembly v2.5 in work")']))

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