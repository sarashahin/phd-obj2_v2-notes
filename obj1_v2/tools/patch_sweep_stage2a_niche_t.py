"""
patch_sweep_stage2a_niche_t.py  (obj1_v2, root cause 23, habitat changing through time, 31 Oct 2026, Sara's yes)

Adds plan "niche_t" to sweep_stage2a.py: the pilot of fix B (Sara's choice, 30 Oct). Twins of existing worlds, one change
each: the habitat changes slowly through time (--habitat-tau, assembly v2.4).
  n49-n52  n29-n32 (random founding, seeds 2303, 2304, 2307, 2308) with tau 1,000,000 steps (200 rounds of 5,000)
  n53-n56  n29-n32 with tau 250,000 steps (50 rounds)
  n57, n58 n33 (best founding, seed 2303) with tau 1,000,000 and 250,000
Everything else is the twin's: contrast 0.25, k 0.5, habitat scale 3, connectance 1, strength 0.4, d per species 1e-6 to
1e-2, the same sim, habitat and trait seeds, 1,000 attempts (entered newcomers under best founding), window 5,000.
Also:
  - niche rows have their saved HABITAT_tau checked (files without it count as -1, a fixed habitat);
  - the file pattern names the habitat-change part (ht1000000_, ht250000_), so the two speeds of one seed, and the
    random and best founding worlds of seed 2303, are told apart.
Reading rule (written before the run, 30 Oct; claude/n33_result_29oct.md):
  1. richness_balance_check.py --chain, static -> slow -> fast, per founding rule: a kept share whose interval holds 0.5
     at some speed confirms the fix for that rule; none -> fix B is not enough, discuss.
  2. guard: ranges still follow the habitat of the time (median AUC of the final suitability for occupied cells 0.8 or
     more; static worlds about 0.99).
  3. BNM bands at the measured detection (world_vs_bnm --detect 0.80 --detect-small 0.64), reported.
Needs run_all_rps.py with patch_run_all_rps_habitat_tau.py and assembly v2.4 in work, for this plan only.

Run from obj1_v2/work:
    cp ../tools/sweep_stage2a.py ../tools/sweep_stage2a.before_t.py
    python3 ../tools/patch_sweep_stage2a_niche_t.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche_t.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 habitat-tau"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche_t plan is already in sweep_stage2a.py")
if "# obj1_v2 width v2" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_niche_w.py and patch_sweep_stage2a_niche_w2.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "v2 (30 Oct): the niche settings check takes the row's w_min; the command sets --niche-w-min in place.",
         "niche_w v2 docstring line")
edits.append((i + 1, i + 1, [
    "  niche_t 10 worlds (added 31 Oct): the habitat changes slowly through time (--habitat-tau). n49-n52 = n29-n32",
    "         (random founding) at tau 1,000,000 steps, n53-n56 at 250,000; n57, n58 = n33 (best founding, seed 2303) at",
    "         the same two speeds. n29-n32 and n33 are the twins."]))
i = find(lambda s: s.strip() == 'if plan == "niche_w":   # obj1_v2 width', "niche_w plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche_t":   {MARK}',
    f'{p}    b = dict(kernel="2dt_a1_b3", f=1, attempts=1000, window=5000, niche=True, tradeoff=0.5, contrast=0.25,',
    f'{p}             connectance=1.0, disp=(1e-6, 1e-2))',
    f'{p}    rand = [dict(b, name=f"n{{48 + 4 * j + k}}", d=d, seed=s, habitat_seed=100000 + s, trait_seed=200000 + s, habitat_tau=t)',
    f'{p}            for j, t in enumerate((1000000, 250000))',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]',
    f'{p}    best = [dict(b, name=f"n{{57 + j}}", d="1e-6", seed=2303, habitat_seed=100000 + 2303, trait_seed=200000 + 2303,',
    f'{p}                 founding="best", habitat_tau=t) for j, t in enumerate((1000000, 250000))]',
    f'{p}    return rand + best']))
i = find(lambda s: s.strip() == "and wmin_ok(z, r))   # obj1_v2 width", "niche_ok width line")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}and wmin_ok(z, r)   # obj1_v2 width",
    f"{p}and tau_ok(z, r))   {MARK}"]))
i = find(lambda s: s.strip() == 'def row_pat(r):   # obj1_v2 contrast v2   # obj1_v2 connectance', "row_pat")
want_wm = '    wm = f"w{sl(r[\'w_min\'])}-{sl(r.get(\'w_max\', 1.0))}*" if r.get("w_min") is not None else ""   # obj1_v2 width'
want_ret = '    return f"*_s2a_*{ls}*{wm}{hc}{cn}{dps}{fb}{st}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"'
if L[i + 15] != want_wm or L[i + 16] != want_ret:
    sys.exit("STOP: row_pat is not the niche_w version; nothing written")
edits.append((i, i, [
    f"def tau_ok(z, r):   {MARK}",
    "    \"\"\"Habitat change saved as the row asks: rows with habitat_tau=t need HABITAT_tau t; other rows need a fixed",
    "    habitat, -1 (files without HABITAT_tau count as -1).\"\"\"",
    "    t = int(z[\"HABITAT_tau\"]) if \"HABITAT_tau\" in z.files else -1",
    "    return t == int(r.get(\"habitat_tau\", -1))",
    "",
    ""]))
edits.append((i + 16, i + 17, [
    f'    ht = f"ht{{int(r[\'habitat_tau\'])}}_" if r.get("habitat_tau") is not None else ""   {MARK}',
    '    return f"*_s2a_*{ls}*{wm}{hc}{cn}{dps}{fb}{st}{ht}k*_f{r[\'f\']}_s{r[\'seed\']}_training.npz"']))
i = find(lambda s: s.strip() == 'base[base.index("--niche-w-min") + 1] = str(r["w_min"])   # obj1_v2 width v2: in place, one value',
         "--niche-w-min line in command")
edits.append((i + 1, i + 1, [
    f'    if r.get("habitat_tau") is not None:   {MARK}',
    '        base += ["--habitat-tau", str(int(r["habitat_tau"]))]']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend", "niche", "niche_k", "niche_c", "niche_f", "niche_d", "niche_b", "niche_s", "niche_w"]' in s,
         "--plan choices")
edits.append((i, i + 1, [L[i].replace('"niche_s", "niche_w"]', '"niche_s", "niche_w", "niche_t"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == 'if a.plan == "niche_w" and ("--niche-w-min" not in src or "--founding" not in src   # obj1_v2 width',
         "niche_w guard")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche_t" and ("--habitat-tau" not in src   {MARK}',
    f'{p}                           or "habitat_change=None," not in Path("assembly_stepwise_ibm.py").read_text()):',
    f'{p}    sys.exit("STOP: niche_t needs --habitat-tau in run_all_rps.py (patch_run_all_rps_habitat_tau.py) and assembly v2.4 in work")']))

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