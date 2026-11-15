"""
patch_run_all_rps_habitat_substeps.py  (obj1_v2, root cause 24, gate 7, 1 Nov 2026, Sara's yes)

Adds --habitat-substeps M to work/run_all_rps.py: with --habitat-tau, the habitat moves in M equal small steps inside
every window and a presence map is saved after each step (assembly v2.6). The change per round stays as before, so the
balance of the shift setting (gate 1) is kept, and the maps come M times as often (gate 7, the frame interval).
Why (frame_change_check, 1 Nov): the candidate setting changes about 6.6 times faster per round than BNM per year, the
static twin at BNM's rate; slowing the habitat per round broke the balance in n57, n58, n59 and n65.
  1. option --habitat-substeps M (whole number >= 1); needs --habitat-tau, and M must divide --ibm-window-steps.
  2. passed to the assembly as habitat_change substeps=M.
  3. world tag: "_hss<M>" after the _ht/_hs parts, only with the option.
  4. saved: HABITAT_substeps, and the maps after every piece: ASM_F_round, ASM_F_piece, ASM_F_offsets, ASM_F_ids,
     ASM_F_bits (packed as ASM_P_*), ASM_F_shift with a shift.
Without the option nothing changes (test_20 --runner R1). Needs assembly v2.6 in work and run_all_rps.py with
patch_run_all_rps_habitat_shift.py.

Run from obj1_v2/work:
    cp ../new_code/assembly_stepwise_ibm.py assembly_stepwise_ibm.py        # v2.6
    python3 ../tools/patch_run_all_rps_habitat_substeps.py
    python3 ../tools/patch_run_all_rps_habitat_substeps.py --apply
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
run_all_rps.before_habitat_substeps.py (test_20 --runner compares the two).
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 habitat-substeps"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --habitat-substeps")
if "# obj1_v2 habitat-shift" not in src:
    sys.exit("STOP: apply patch_run_all_rps_habitat_shift.py first")
if "substeps" not in Path("assembly_stepwise_ibm.py").read_text():
    sys.exit("STOP: assembly_stepwise_ibm.py is not v2.6; copy ../new_code/assembly_stepwise_ibm.py into work first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "help='time scale of the shift in IBM steps; needs --habitat-shift-sd')", "--habitat-shift-tau help")
p = ind(L[i - 1])
edits.append((i + 1, i + 1, [
    f"{p}p.add_argument('--habitat-substeps', type=int, default=None,   {MARK}",
    f"{p}               help='with --habitat-tau: the habitat moves in this many equal steps inside each window and a '",
    f"{p}                    'map is saved after each (assembly v2.6); must divide --ibm-window-steps')"]))
i = find(lambda s: s.strip() == "raise SystemExit('--habitat-shift-sd must be > 0 and --habitat-shift-tau a whole number of steps, 1 or more')",
         "shift guard end")
p = ind(L[i - 4])
edits.append((i + 1, i + 1, [
    f"{p}if args.habitat_substeps is not None:   {MARK}",
    f"{p}    if args.habitat_tau is None:",
    f"{p}        raise SystemExit('--habitat-substeps needs --habitat-tau')",
    f"{p}    if args.habitat_substeps < 1 or args.ibm_window_steps % args.habitat_substeps:",
    f"{p}        raise SystemExit('--habitat-substeps must be a whole number >= 1 that divides --ibm-window-steps')"]))
i = find(lambda s: s.strip() == "shift_tau=args.habitat_shift_tau))),", "shift part of the habitat_change call")
p = ind(L[i - 2])
edits.append((i, i + 1, [
    L[i].replace("shift_tau=args.habitat_shift_tau))),", "shift_tau=args.habitat_shift_tau)),"),
    f"{p}**({{}} if args.habitat_substeps is None else dict(substeps=args.habitat_substeps))),   {MARK}"]))
i = find(lambda s: s.strip() == 'world_tag += f"_hs{_sanitize_slug(float(args.habitat_shift_sd))}t{int(args.habitat_shift_tau)}"',
         "_hs part of the world tag")
p = ind(L[i - 1])
edits.append((i + 1, i + 1, [
    f"{p}if args.habitat_substeps is not None:   {MARK}",
    f"{p}    world_tag += f\"_hss{{int(args.habitat_substeps)}}\""]))
i = find(lambda s: s.strip() == 'train_out["ENV_habitat_shift"] = _np.asarray(extra_ibm["habitat_shift_t"], dtype=_np.float64)',
         "ENV_habitat_shift save line")
p = ind(L[i - 4])
edits.append((i + 1, i + 1, [
    f"{p}if args.habitat_substeps is not None:   {MARK}",
    f"{p}    train_out[\"HABITAT_substeps\"] = _np.int64(args.habitat_substeps)",
    f"{p}    for _k in (\"ASM_F_round\", \"ASM_F_piece\", \"ASM_F_offsets\", \"ASM_F_ids\", \"ASM_F_bits\", \"ASM_F_shift\"):",
    f"{p}        if _k in extra_ibm:",
    f"{p}            train_out[_k] = _np.asarray(_to_host(extra_ibm[_k]))"]))

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
    Path("run_all_rps.before_habitat_substeps.py").write_text(src)
    PATH.write_text(out); print("written: run_all_rps.py (the version before is kept as run_all_rps.before_habitat_substeps.py)")
else:
    print("dry run: nothing written (add --apply to write)")