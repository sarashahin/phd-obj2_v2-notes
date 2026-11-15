"""
patch_run_all_rps_disp.py  (obj1_v2, stage 2d-b, 29 Oct 2026)

Adds --disp-per-species DMIN DMAX to work/run_all_rps.py (niche mode only): every species draws
its own dispersal rate, log-uniform in [DMIN, DMAX], from the trait seed (--env-seed-field) and
its id. The assembly windows and the final run both use these rates; --disp-rate is then not
used by the dynamics (it is still saved as DISPERSAL_RATE). Left out, nothing changes.
  1. option --disp-per-species DMIN DMAX; refused outside niche mode and unless
     0 < DMIN <= DMAX <= 0.1 (test_14: from about 0.095 on, sinks with g above -0.1 keep each
     other going and the habitat stops mattering there).
  2. passed to stepwise_assembly_ibm as disp=dict(d_min, d_max).
  3. final run: dispersal.set_species_rates(rates of the final species) around m_ibm.run(),
     switched off afterwards, also on an error.
  4. world tag: "_dps<DMIN>-<DMAX>" before the kernel part, only with the option.
  5. saved in every training file: DISP_per_species (0 or 1); with the option also DISP_d_min,
     DISP_d_max, SP_disp_rate (final species, in species_id order) and LT_d (every species that
     arrived, in LT_id order).
Needs dispersal.py with patch_dispersal_per_species.py and new_code/assembly_stepwise_ibm.py v2.2
in work, and run_all_rps.py with patch_run_all_rps_connectance.py.

Run from obj1_v2/work:
    python3 ../tools/patch_run_all_rps_disp.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_disp.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 per-species d"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --disp-per-species")
if "# obj1_v2 connectance" not in src:
    sys.exit("STOP: apply patch_run_all_rps_connectance.py first")
if "def set_species_rates" not in Path("dispersal.py").read_text():
    sys.exit("STOP: dispersal.py lacks set_species_rates; apply patch_dispersal_per_species.py first")
if "disp=None," not in Path("assembly_stepwise_ibm.py").read_text():
    sys.exit("STOP: assembly_stepwise_ibm.py is not v2.2; copy ../new_code/assembly_stepwise_ibm.py into work first")
L = src.split("\n")
code = lambda s: s.split("#")[0].strip()


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("p.add_argument('--connectance'"), "--connectance option")
if not L[i + 1].strip().startswith("help='chance of a competition link"):
    sys.exit("STOP: --connectance help line not where expected; nothing written")
edits.append((i + 2, i + 2, [
    f"    p.add_argument('--disp-per-species', type=float, nargs=2, default=None, metavar=('DMIN', 'DMAX'),   {MARK}",
    "                   help='niche mode: each species draws its own d, log-uniform in [DMIN, DMAX], from the trait seed and its id; '",
    "                        '--disp-rate is then not used by the dynamics')"]))
i = find(lambda s: s.strip() == "raise SystemExit('--habitat-contrast needs --env-mode niche')", "--habitat-contrast niche guard")
p = ind(L[i - 1])
edits.append((i + 1, i + 1, [
    f"{p}if args.disp_per_species is not None:   {MARK}",
    f"{p}    if args.env_mode != 'niche':",
    f"{p}        raise SystemExit('--disp-per-species needs --env-mode niche')",
    f"{p}    if not 0 < args.disp_per_species[0] <= args.disp_per_species[1] <= 0.1:",
    f"{p}        raise SystemExit('--disp-per-species needs 0 < DMIN <= DMAX <= 0.1')"]))
i = find(lambda s: s.strip().startswith("habitat=H_world, niche=niche_kw,"), "assembly call habitat line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}disp=None if args.disp_per_species is None else dict(d_min=args.disp_per_species[0], d_max=args.disp_per_species[1]),   {MARK}"]))
i = find(lambda s: code(s) == "B_ibm = m_ibm.run()", "final run m_ibm.run()")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}if use_assembly and 'disp_rate' in extra_ibm:   {MARK}: the final run uses each species' own d",
    f"{p}    dispersal.set_species_rates(extra_ibm['disp_rate'])",
    f"{p}try:",
    f"{p}    B_ibm = m_ibm.run()",
    f"{p}finally:",
    f"{p}    dispersal.set_species_rates(None)"]))
i = find(lambda s: s.lstrip().startswith('world_tag += (f"_k{dispersal.DISPERSAL_KERNEL}"'), "kernel part of the world tag")
p = ind(L[i])
edits.append((i, i, [
    f"{p}if args.disp_per_species is not None:   {MARK}",
    f"{p}    world_tag += f\"_dps{{_sanitize_slug(args.disp_per_species[0])}}-{{_sanitize_slug(args.disp_per_species[1])}}\""]))
i = find(lambda s: s.strip().startswith('train_out["INTERACTION_STRENGTH"] = _np.float32(_au.INTERACTION_STRENGTH)'), "INTERACTION_STRENGTH save line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}train_out[\"DISP_per_species\"] = _np.int8(args.disp_per_species is not None)   {MARK}",
    f"{p}if args.disp_per_species is not None:",
    f"{p}    train_out[\"DISP_d_min\"] = _np.float64(args.disp_per_species[0])",
    f"{p}    train_out[\"DISP_d_max\"] = _np.float64(args.disp_per_species[1])",
    f"{p}    train_out[\"SP_disp_rate\"] = _np.asarray(_to_host(extra_ibm[\"disp_rate\"]), dtype=_np.float64)",
    f"{p}    train_out[\"LT_d\"] = _np.asarray(_to_host(extra_ibm[\"LT_d\"]), dtype=_np.float64)"]))

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
    Path("run_all_rps.before_disp.py").write_text(src)
    PATH.write_text(out); print("written: run_all_rps.py (the version before is kept as run_all_rps.before_disp.py)")
else:
    print("dry run: nothing written (add --apply to write)")