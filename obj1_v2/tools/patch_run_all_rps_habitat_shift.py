"""
patch_run_all_rps_habitat_shift.py  (obj1_v2, root cause 23, fix B part 2, 31 Oct 2026, Sara's yes)

Adds --habitat-shift-sd S and --habitat-shift-tau T to work/run_all_rps.py: on top of --habitat-tau, the whole habitat
layer moves up or down together through time (assembly v2.5, habitat_change shift_sd and shift_tau). S is the shift's
sd in units of the layer's sd; T its time scale in IBM steps. Left out, nothing changes and no key is added.
Why (31 Oct): resident_loss_check showed losses fall almost only on small ranges and that no species ever had less
than half its own habitat amount under --habitat-tau alone (each moving map is standardised), so large-range residents
are never pushed down; position-only change reached 0.89 (best founding) and 0.40-0.42 (random founding) of the loss
rate a level needs under the newcomer rule.
  1. options --habitat-shift-sd (number > 0) and --habitat-shift-tau (whole number >= 1), both or neither, and only
     with --habitat-tau.
  2. passed to stepwise_assembly_ibm inside habitat_change as shift_sd and shift_tau.
  3. world tag: "_hs<sd>t<tau>" right after "_ht<tau>", only with the options.
  4. saved, only with the options: HABITAT_shift_sd, HABITAT_shift_tau, HABITAT_shift_rho (per round) and
     ENV_habitat_shift (the shift of every assembly round; ENV_habitat_t and ENV_habitat include it).
Needs run_all_rps.py with patch_run_all_rps_habitat_tau.py, and new_code/assembly_stepwise_ibm.py v2.5 in work.

Run from obj1_v2/work:
    cp ../new_code/assembly_stepwise_ibm.py assembly_stepwise_ibm.py        # v2.5
    python3 ../tools/patch_run_all_rps_habitat_shift.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_habitat_shift.py --apply
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
run_all_rps.before_habitat_shift.py (test_19 --runner compares the two).
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 habitat-shift"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --habitat-shift-sd")
if "# obj1_v2 habitat-tau" not in src:
    sys.exit("STOP: apply patch_run_all_rps_habitat_tau.py first")
if "shift_sd" not in Path("assembly_stepwise_ibm.py").read_text():
    sys.exit("STOP: assembly_stepwise_ibm.py is not v2.5; copy ../new_code/assembly_stepwise_ibm.py into work first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "'(assembly v2.4); default: fixed habitat')", "--habitat-tau help end")
p = ind(L[i - 2])
edits.append((i + 1, i + 1, [
    f"{p}p.add_argument('--habitat-shift-sd', type=float, default=None,   {MARK}",
    f"{p}               help='with --habitat-tau: the whole habitat layer moves up or down together, with this sd in '",
    f"{p}                    'units of the layer sd (assembly v2.5); needs --habitat-shift-tau')",
    f"{p}p.add_argument('--habitat-shift-tau', type=int, default=None,   {MARK}",
    f"{p}               help='time scale of the shift in IBM steps; needs --habitat-shift-sd')"]))
i = find(lambda s: s.strip() == "raise SystemExit('--habitat-tau: --resume is not supported; rerun the world from the start')",
         "--habitat-tau resume refusal")
if L[i - 6].strip() != "if args.habitat_tau is not None:   # obj1_v2 habitat-tau":
    sys.exit("STOP: the --habitat-tau guard block is not the habitat-tau patch's text; nothing written")
p = ind(L[i - 6])
edits.append((i + 1, i + 1, [
    f"{p}if (args.habitat_shift_sd is None) != (args.habitat_shift_tau is None):   {MARK}",
    f"{p}    raise SystemExit('--habitat-shift-sd and --habitat-shift-tau go together')",
    f"{p}if args.habitat_shift_sd is not None:",
    f"{p}    if args.habitat_tau is None:",
    f"{p}        raise SystemExit('--habitat-shift-sd needs --habitat-tau')",
    f"{p}    if not args.habitat_shift_sd > 0 or args.habitat_shift_tau < 1:",
    f"{p}        raise SystemExit('--habitat-shift-sd must be > 0 and --habitat-shift-tau a whole number of steps, 1 or more')"]))
i = find(lambda s: s.strip().startswith("habitat_change=None if args.habitat_tau is None else dict(tau=args.habitat_tau, seed=args.habitat_seed,"),
         "habitat_change line in the assembly call")
if L[i + 1].strip() != "length_scale=args.habitat_ls, contrast=args.habitat_contrast),":
    sys.exit("STOP: the habitat_change call is not the habitat-tau patch's text; nothing written")
p = ind(L[i + 1])
edits.append((i + 1, i + 2, [
    f"{p}length_scale=args.habitat_ls, contrast=args.habitat_contrast,",
    f"{p}**({{}} if args.habitat_shift_sd is None else dict(shift_sd=args.habitat_shift_sd,   {MARK}",
    f"{p}                                                    shift_tau=args.habitat_shift_tau))),"]))
i = find(lambda s: s.strip() == 'world_tag += f"_ht{int(args.habitat_tau)}"', "_ht part of the world tag")
p = ind(L[i - 1])
edits.append((i + 1, i + 1, [
    f"{p}if args.habitat_shift_sd is not None:   {MARK}",
    f"{p}    world_tag += f\"_hs{{_sanitize_slug(float(args.habitat_shift_sd))}}t{{int(args.habitat_shift_tau)}}\""]))
i = find(lambda s: s.strip() == 'train_out["HABITAT_rho"] = _np.float64(extra_ibm["habitat_rho"])', "HABITAT_rho save line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}if args.habitat_shift_sd is not None:   {MARK}",
    f"{p}    train_out[\"HABITAT_shift_sd\"] = _np.float64(args.habitat_shift_sd)",
    f"{p}    train_out[\"HABITAT_shift_tau\"] = _np.int64(args.habitat_shift_tau)",
    f"{p}    train_out[\"HABITAT_shift_rho\"] = _np.float64(extra_ibm[\"habitat_shift_rho\"])",
    f"{p}    train_out[\"ENV_habitat_shift\"] = _np.asarray(extra_ibm[\"habitat_shift_t\"], dtype=_np.float64)"]))

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
    Path("run_all_rps.before_habitat_shift.py").write_text(src)
    PATH.write_text(out); print("written: run_all_rps.py (the version before is kept as run_all_rps.before_habitat_shift.py)")
else:
    print("dry run: nothing written (add --apply to write)")