"""
patch_run_all_rps_niche.py  (obj1_v2, stage 2d-a part B, 27 Oct 2026)

Wires the niche model into work/run_all_rps.py. Needs assembly_stepwise_ibm.py v2
(new_code, 27 Oct) and environment.py with patch_environment_niche.py applied.

What it changes, and why:
 1. --env-mode field|niche (default field: every world made so far is unchanged).
    Niche mode: --habitat-seed (or --habitat-file, a real map), --habitat-ls,
    --niche-opt-sd, --niche-w-min, --niche-w-max, --niche-cost, --niche-peak-max;
    --env-seed-field becomes the seed of the species niches.
 2. Checks in niche mode: exactly one of --habitat-seed and --habitat-file; a niche seed;
    no field-mode options (--env-length-scale, --env-var-r); no --resume (a resumed world
    is not the world an uninterrupted run gives; stopped worlds are rerun from the start,
    which repeats them exactly on the GPU, long control 26 Oct).
 3. The habitat seed (or map file name) and niche settings go into the world tag, before
    _f<n>_s<seed>, so worlds do not overwrite each other and the sweep still finds them.
 4. Saved in every training file: ENV_mode, the life table (LT_*), the window-end
    presence record (ASM_P_*), ASM_resumed_from_round. Niche mode also saves the habitat
    layer (ENV_habitat), its seed or file, the niche settings, the trait seed, the niche of
    every final species (NICHE_opt, NICHE_w, NICHE_peak) and of every species that ever
    arrived (LT_opt, LT_w, LT_peak).
 5. A run whose assembly ends with no species stops with a message (it could not be saved).

Run from obj1_v2/work:
    python3 ../tools/patch_run_all_rps_niche.py            # dry run: prints the diff, writes nothing
    python3 ../tools/patch_run_all_rps_niche.py --apply    # keeps run_all_rps.before_niche.py, writes the file
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, shutil, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 niche"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already carries the niche changes")
if "# obj1_v2 seed-size" not in src or "# obj1_v2" not in src:
    sys.exit("STOP: apply patch_run_all_rps_v2.py and patch_run_all_rps_seed_size.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []                                     # (start, end, new_lines): replace L[start:end]

i = find(lambda s: "random patches each invader is founded in (one individual each); was fixed at 5')" in s, "--seed-size help")
edits.append((i + 1, i + 1, [
    f"    p.add_argument('--env-mode', choices=['field', 'niche'], default='field',   {MARK}",
    "                   help='field: own random field per species (as before); niche: shared habitat layer and a niche per species')",
    "    p.add_argument('--habitat-seed', type=int, default=None, help='niche mode: seed of the simulated habitat layer, one per world')",
    "    p.add_argument('--habitat-file', type=str, default=None, help='niche mode: real habitat map (.npy, shape Y x X) instead of --habitat-seed')",
    "    p.add_argument('--habitat-ls', type=float, default=3.0, help='niche mode: length scale of the simulated habitat layer, in cells')",
    "    p.add_argument('--niche-opt-sd', type=float, default=1.5, help='niche mode: sd of niche optima (habitat sd is 1)')",
    "    p.add_argument('--niche-w-min', type=float, default=0.2, help='niche mode: narrowest niche width')",
    "    p.add_argument('--niche-w-max', type=float, default=1.0, help='niche mode: widest niche width')",
    "    p.add_argument('--niche-cost', type=float, default=0.1, help='niche mode: cost subtracted from the niche curve')",
    "    p.add_argument('--niche-peak-max', type=float, default=None, help='niche mode: peak of the narrowest niche; default environment.NICHE_PEAK_MAX')"]))

i = find(lambda s: "raise SystemExit('--seed-size must be at least 1')" in s, "seed-size guard")
edits.append((i + 1, i + 1, [
    f"    if args.env_mode == 'niche':   {MARK}",
    "        if (args.habitat_seed is None) == (args.habitat_file is None):",
    "            raise SystemExit('niche mode: give exactly one of --habitat-seed and --habitat-file')",
    "        if args.env_seed_field is None:",
    "            raise SystemExit('niche mode: give --env-seed-field (seed of the species niches, one per world)')",
    "        if args.env_length_scale is not None or args.env_var_r is not None:",
    "            raise SystemExit('niche mode: --env-length-scale and --env-var-r belong to field mode; leave them out')",
    "        if args.resume:",
    "            raise SystemExit('niche mode: --resume is not supported; rerun the world from the start (it repeats exactly)')",
    "    elif args.habitat_seed is not None or args.habitat_file is not None:",
    "        raise SystemExit('--habitat-seed and --habitat-file need --env-mode niche')"]))

i = find(lambda s: s.lstrip().startswith('world_tag += (f"_k{dispersal.DISPERSAL_KERNEL}"'), "kernel part of the world tag")
p = ind(L[i])
edits.append((i, i, [
    f"{p}H_world = niche_kw = None   {MARK}: shared habitat layer and niche settings",
    f"{p}if args.env_mode == 'niche':",
    f"{p}    from environment import habitat_layer, NICHE_PEAK_MAX",
    f"{p}    H_world = habitat_layer(NUM_PATCHES_Y, NUM_PATCHES_X, args.habitat_ls, seed=args.habitat_seed, path=args.habitat_file)",
    f"{p}    niche_kw = dict(opt_sd=args.niche_opt_sd, w_min=args.niche_w_min, w_max=args.niche_w_max, cost=args.niche_cost,",
    f"{p}                    peak_max=NICHE_PEAK_MAX if args.niche_peak_max is None else args.niche_peak_max)",
    p + "    _h = f\"h{args.habitat_seed}\" if args.habitat_file is None else f\"hf{pathlib.Path(args.habitat_file).stem}\"",
    p + "    world_tag += (f\"_niche{_h}ls{_sanitize_slug(args.habitat_ls)}o{_sanitize_slug(args.niche_opt_sd)}\"",
    p + "                  f\"w{_sanitize_slug(args.niche_w_min)}-{_sanitize_slug(args.niche_w_max)}c{_sanitize_slug(args.niche_cost)}\"",
    p + "                  f\"p{_sanitize_slug(niche_kw['peak_max'])}\")"]))

i = find(lambda s: s.strip() == "seed_size=args.seed_size,   # obj1_v2 seed-size", "assembly call seed_size")
edits.append((i + 1, i + 1, [f"{ind(L[i])}habitat=H_world, niche=niche_kw,   {MARK}"]))

i = find(lambda s: "# obj1_v2: species keep their assembly field" in s, "final-run field line")
p = ind(L[i])
edits.append((i, i, [
    f"{p}if use_assembly and len(r_ibm) == 0:   {MARK}",
    f"{p}    raise SystemExit('assembly ended with no species; nothing to simulate or save')"]))

i = find(lambda s: s.strip() == "if obs_masks:", "obs_masks block")
p = ind(L[i])
edits.append((i, i, [
    f"{p}train_out[\"ENV_mode\"] = _np.array([args.env_mode])   {MARK}: mode, life table, window-end presence, niche",
    f"{p}for key in (\"ASM_resumed_from_round\", \"LT_id\", \"LT_round_in\", \"LT_established\", \"LT_round_out\", \"LT_peak_cells\",",
    f"{p}            \"ASM_P_round\", \"ASM_P_offsets\", \"ASM_P_ids\", \"ASM_P_bits\", \"ASM_P_count\"):",
    f"{p}    if key in extra_ibm:",
    f"{p}        train_out[key] = _to_host(extra_ibm[key])",
    f"{p}if args.env_mode == 'niche':",
    f"{p}    train_out.update({{",
    f"{p}        \"ENV_habitat\": _np.asarray(H_world, dtype=_np.float32),",
    f"{p}        \"HABITAT_seed\": _np.int64(-1 if args.habitat_seed is None else args.habitat_seed),",
    f"{p}        \"HABITAT_file\": _np.array([args.habitat_file or \"\"]),",
    f"{p}        \"HABITAT_ls\": _np.float32(args.habitat_ls),",
    f"{p}        \"TRAIT_seed\": _np.int64(args.env_seed_field),",
    f"{p}        \"NICHE_opt_sd\": _np.float32(niche_kw[\"opt_sd\"]), \"NICHE_w_min\": _np.float32(niche_kw[\"w_min\"]),",
    f"{p}        \"NICHE_w_max\": _np.float32(niche_kw[\"w_max\"]), \"NICHE_cost\": _np.float32(niche_kw[\"cost\"]),",
    f"{p}        \"NICHE_peak_max\": _np.float32(niche_kw[\"peak_max\"]),",
    f"{p}        \"NICHE_opt\": extra_ibm[\"niche_opt\"].astype(_np.float32), \"NICHE_w\": extra_ibm[\"niche_w\"].astype(_np.float32),",
    f"{p}        \"NICHE_peak\": extra_ibm[\"niche_peak\"].astype(_np.float32),",
    f"{p}        \"LT_opt\": extra_ibm[\"LT_opt\"].astype(_np.float32), \"LT_w\": extra_ibm[\"LT_w\"].astype(_np.float32),",
    f"{p}        \"LT_peak\": extra_ibm[\"LT_peak\"].astype(_np.float32)}})",
    f"{p}    if \"ENV_type\" in train_out:",
    f"{p}        train_out[\"ENV_type\"] = _np.array([\"niche: shared habitat layer\"], dtype=object)",
    ""]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
diff = list(difflib.unified_diff(L, new, "run_all_rps.py (now)", "run_all_rps.py (patched)", lineterm="", n=1))
print("\n".join(diff))
print(f"\n{len(edits)} edits, {sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))} lines added")
compile(out, "run_all_rps.py", "exec")
print("the edited file compiles")
if "--apply" in sys.argv:
    shutil.copyfile(PATH, "run_all_rps.before_niche.py")
    PATH.write_text(out); print("kept: run_all_rps.before_niche.py; written: run_all_rps.py")
else:
    print("dry run: nothing written (add --apply to write)")