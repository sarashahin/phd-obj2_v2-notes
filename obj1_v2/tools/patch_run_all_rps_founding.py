"""
patch_run_all_rps_founding.py  (obj1_v2, LVMCM founding, 29 Oct 2026)

Adds --founding random|best and --founding-cap K to work/run_all_rps.py. Left out, nothing changes
in the world (one key, FOUND_rule "random", is added to the training file).
  1. options: --founding random (default; each newcomer in --seed-size random cells, as before) or
     best (LVMCM: a newcomer enters only if it can grow somewhere against the residents, as one
     individual in its best cell; only entered newcomers count as attempts, so --ibm-max-attempts
     counts entered newcomers). --founding-cap K (default 100): candidates tried per newcomer needed
     in a round; a round with none that can grow ends the assembly (saturated).
  2. refused: best without --seed-size 1; best with --resume; --founding-cap below 1, or set without
     --founding best.
  3. passed to stepwise_assembly_ibm as founding=..., founding_cap=... (assembly v2.3).
  4. world tag: "_fb" before the kernel part with best ("_fbc<K>" when K is not 100).
  5. saved: FOUND_rule in every training file; with best also FOUND_cap, FOUND_tested_total,
     FOUND_saturated, FOUND_stop_round, LT_entered, LT_attempt, LT_g_max (every tested candidate,
     in LT_id order) and ASM_tested (candidates tested up to each round).
Needs new_code/assembly_stepwise_ibm.py v2.3 in work and run_all_rps.py with patch_run_all_rps_disp.py.

Run from obj1_v2/work:
    python3 ../tools/patch_run_all_rps_founding.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_founding.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 founding"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --founding")
if "# obj1_v2 per-species d" not in src:
    sys.exit("STOP: apply patch_run_all_rps_disp.py first")
if 'founding: str = "random"' not in Path("assembly_stepwise_ibm.py").read_text():
    sys.exit("STOP: assembly_stepwise_ibm.py is not v2.3; copy ../new_code/assembly_stepwise_ibm.py into work first")
L = src.split("\n")
code = lambda s: s.split("#")[0].strip()


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("p.add_argument('--disp-per-species'"), "--disp-per-species option")
if not L[i + 2].strip().endswith("'--disp-rate is then not used by the dynamics')"):
    sys.exit("STOP: --disp-per-species help lines not where expected; nothing written")
edits.append((i + 3, i + 3, [
    f"    p.add_argument('--founding', choices=['random', 'best'], default='random',   {MARK}",
    "                   help='random: each newcomer in --seed-size random cells (as before); best: LVMCM, a newcomer enters only if it '",
    "                        'can grow somewhere against the residents, as one individual in its best cell (needs --seed-size 1)')",
    "    p.add_argument('--founding-cap', type=int, default=100,",
    "                   help='--founding best: candidates tried per newcomer needed in a round; a round with none ends the assembly')"]))
i = find(lambda s: s.strip() == "raise SystemExit('--disp-per-species needs 0 < DMIN <= DMAX <= 0.1')", "--disp-per-species range guard")
j = find(lambda s: code(s) == "if args.disp_per_species is not None:" and ind(s) == "    ", "--disp-per-species guard block")
p = ind(L[j])
edits.append((i + 1, i + 1, [
    f"{p}if args.founding == 'best' and args.seed_size != 1:   {MARK}",
    f"{p}    raise SystemExit('--founding best places one individual in one cell (LVMCM); give --seed-size 1')",
    f"{p}if args.founding == 'best' and args.resume:",
    f"{p}    raise SystemExit('--founding best: --resume is not supported; rerun the world from the start')",
    f"{p}if args.founding_cap < 1:",
    f"{p}    raise SystemExit('--founding-cap must be at least 1')",
    f"{p}if args.founding != 'best' and args.founding_cap != 100:",
    f"{p}    raise SystemExit('--founding-cap needs --founding best')"]))
i = find(lambda s: s.strip().startswith("disp=None if args.disp_per_species is None else dict("), "assembly call disp line")
p = ind(L[i])
edits.append((i + 1, i + 1, [f"{p}founding=args.founding, founding_cap=args.founding_cap,   {MARK}"]))
i = find(lambda s: s.lstrip().startswith('world_tag += (f"_k{dispersal.DISPERSAL_KERNEL}"'), "kernel part of the world tag")
p = ind(L[i])
edits.append((i, i, [
    f"{p}if args.founding == 'best':   {MARK}",
    f"{p}    world_tag += \"_fb\" + (f\"c{{args.founding_cap}}\" if args.founding_cap != 100 else \"\")"]))
i = find(lambda s: s.strip() == 'train_out["LT_d"] = _np.asarray(_to_host(extra_ibm["LT_d"]), dtype=_np.float64)', "LT_d save line")
j = find(lambda s: s.strip().startswith('train_out["DISP_per_species"] = _np.int8('), "DISP_per_species save line")
p = ind(L[j])
edits.append((i + 1, i + 1, [
    f"{p}train_out[\"FOUND_rule\"] = _np.array([args.founding])   {MARK}",
    f"{p}if args.founding == 'best':",
    f"{p}    train_out[\"FOUND_cap\"] = _np.int64(args.founding_cap)",
    f"{p}    train_out[\"FOUND_tested_total\"] = _np.int64(extra_ibm[\"tested_total\"])",
    f"{p}    train_out[\"FOUND_saturated\"] = _np.int64(extra_ibm[\"found_saturated\"])",
    f"{p}    train_out[\"FOUND_stop_round\"] = _np.int64(extra_ibm[\"found_stop_round\"])",
    f"{p}    train_out[\"LT_entered\"] = _np.asarray(_to_host(extra_ibm[\"LT_entered\"]), dtype=_np.uint8)",
    f"{p}    train_out[\"LT_attempt\"] = _np.asarray(_to_host(extra_ibm[\"LT_attempt\"]), dtype=_np.int64)",
    f"{p}    train_out[\"LT_g_max\"] = _np.asarray(_to_host(extra_ibm[\"LT_g_max\"]), dtype=_np.float64)",
    f"{p}    train_out[\"ASM_tested\"] = _np.asarray(_to_host(extra_ibm[\"ASM_tested\"]), dtype=_np.int64)"]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "run_all_rps.py (now)", "run_all_rps.py (patched)", lineterm="", n=1)))
compile(out, "run_all_rps.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    Path("run_all_rps.before_found.py").write_text(src)
    PATH.write_text(out); print("written: run_all_rps.py (the version before is kept as run_all_rps.before_found.py)")
else:
    print("dry run: nothing written (add --apply to write)")