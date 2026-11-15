"""
patch_run_all_rps_habitat_tau.py  (obj1_v2, root cause 23, 31 Oct 2026, Sara's yes)

Adds --habitat-tau STEPS to work/run_all_rps.py: the habitat layer changes slowly through time, with time scale STEPS
(IBM steps), through assembly v2.4 (habitat_change). Left out, nothing changes in the world; one key, HABITAT_tau = -1,
is added to every training file.
Why (chain of twins, richness_balance_check.py v2, 30 Oct): in every niche world about three newcomers are kept for
each resident lost, while the field-mode twins balance, so richness never levels off (gate 1). In niche worlds each
species has fixed home cells and the habitat never changes, so residents are almost never lost.
  1. option --habitat-tau STEPS (whole number > 0); niche mode with a simulated habitat only (refused with
     --habitat-file: a real map is not moved), and refused with --resume.
  2. passed to stepwise_assembly_ibm as habitat_change=dict(tau, seed=--habitat-seed, length_scale=--habitat-ls,
     contrast=--habitat-contrast). The new random maps come from their own stream, seeded by the habitat seed.
  3. world tag: "_ht<STEPS>" before the kernel part (after "_fb" and "_is"), only with the option.
  4. saved: HABITAT_tau in every training file (-1 without the option); with it also ENV_habitat_start (the map at
     round 0, the static twin's ENV_habitat), ENV_habitat (now the final map, the one the final run's fields use),
     ENV_habitat_t (the map of every assembly round, float32) and HABITAT_rho (per round).
     Tools that rebuild growth fields from ENV_habitat (founding_filter, species_compare) stay right for the final
     fields; the history of the map is in ENV_habitat_t.
Needs new_code/assembly_stepwise_ibm.py v2.4 in work and run_all_rps.py with patch_run_all_rps_strength.py.

Run from obj1_v2/work:
    cp ../new_code/assembly_stepwise_ibm.py assembly_stepwise_ibm.py        # v2.4
    python3 ../tools/patch_run_all_rps_habitat_tau.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_habitat_tau.py --apply
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
run_all_rps.before_habitat_tau.py (test_18 --runner compares the two).
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 habitat-tau"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --habitat-tau")
if "# obj1_v2 strength" not in src or "# obj1_v2 founding" not in src or "# obj1_v2 contrast" not in src:
    sys.exit("STOP: apply patch_run_all_rps_strength.py (and the patches before it) first")
if "habitat_change=None," not in Path("assembly_stepwise_ibm.py").read_text():
    sys.exit("STOP: assembly_stepwise_ibm.py is not v2.4; copy ../new_code/assembly_stepwise_ibm.py into work first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("help='niche mode: the species see contrast x H (H has sd 1)"), "--habitat-contrast help line")
edits.append((i + 1, i + 1, [
    f"    p.add_argument('--habitat-tau', type=int, default=None,   {MARK}",
    "                   help='niche mode: the habitat changes slowly through time, with this time scale in IBM steps '",
    "                        '(assembly v2.4); default: fixed habitat')"]))
i = find(lambda s: s.strip() == "raise SystemExit('--habitat-contrast needs --env-mode niche')", "--habitat-contrast niche guard")
p = ind(L[i - 1])
edits.append((i + 1, i + 1, [
    f"{p}if args.habitat_tau is not None:   {MARK}",
    f"{p}    if args.env_mode != 'niche' or args.habitat_file is not None:",
    f"{p}        raise SystemExit('--habitat-tau needs --env-mode niche with --habitat-seed (a real map is not moved)')",
    f"{p}    if args.habitat_tau < 1:",
    f"{p}        raise SystemExit('--habitat-tau must be a whole number of steps, 1 or more')",
    f"{p}    if args.resume:",
    f"{p}        raise SystemExit('--habitat-tau: --resume is not supported; rerun the world from the start')"]))
i = find(lambda s: s.strip() == "founding=args.founding, founding_cap=args.founding_cap,   # obj1_v2 founding", "assembly call founding line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}habitat_change=None if args.habitat_tau is None else dict(tau=args.habitat_tau, seed=args.habitat_seed,   {MARK}",
    f"{p}                                                         length_scale=args.habitat_ls, contrast=args.habitat_contrast),"]))
i = find(lambda s: s.lstrip().startswith('world_tag += (f"_k{dispersal.DISPERSAL_KERNEL}"'), "kernel part of the world tag")
if not any("# obj1_v2 strength" in s for s in L[max(0, i - 3):i]):
    sys.exit("STOP: the strength part of the world tag is not just before the kernel part; nothing written")
p = ind(L[i])
edits.append((i, i, [
    f"{p}if args.habitat_tau is not None:   {MARK}",
    f"{p}    world_tag += f\"_ht{{int(args.habitat_tau)}}\""]))
i = find(lambda s: s.strip() == "if obs_masks:", "obs_masks block")
if not any(s.strip() == 'train_out["ENV_type"] = _np.array(["niche: shared habitat layer"], dtype=object)' for s in L[max(0, i - 4):i]):
    sys.exit("STOP: the niche save block does not end just before 'if obs_masks:'; nothing written")
p = ind(L[i])
edits.append((i, i, [
    f"{p}train_out[\"HABITAT_tau\"] = _np.int64(-1 if args.habitat_tau is None else args.habitat_tau)   {MARK}",
    f"{p}if args.habitat_tau is not None:",
    f"{p}    train_out[\"ENV_habitat_start\"] = _np.asarray(H_world, dtype=_np.float32)",
    f"{p}    train_out[\"ENV_habitat\"] = _np.asarray(_to_host(extra_ibm[\"habitat\"]), dtype=_np.float32)",
    f"{p}    train_out[\"ENV_habitat_t\"] = _np.asarray(_to_host(extra_ibm[\"habitat_t\"]), dtype=_np.float32)",
    f"{p}    train_out[\"HABITAT_rho\"] = _np.float64(extra_ibm[\"habitat_rho\"])",
    ""]))

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
    Path("run_all_rps.before_habitat_tau.py").write_text(src)
    PATH.write_text(out); print("written: run_all_rps.py (the version before is kept as run_all_rps.before_habitat_tau.py)")
else:
    print("dry run: nothing written (add --apply to write)")