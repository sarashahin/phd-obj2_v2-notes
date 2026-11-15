"""
patch_run_all_rps_v2.py  (obj1_v2, 24 Oct 2026)

Edits work/run_all_rps.py for the re-simulation. Run from obj1_v2/work:
    python3 ../tools/patch_run_all_rps_v2.py            # dry run: prints the diff, writes nothing
    python3 ../tools/patch_run_all_rps_v2.py --apply    # writes the file
Every anchor must be found exactly once, or nothing is written. The untouched file
stays in obj1_v2/original.

What it changes, and why (tests in obj1_v2/tests):
 1. --sim-seed: a master seed per community, passed to the assembly instead of the
    hard-coded seed=1 (test_03 B2: two communities shared their first species'
    interactions and founder cell). Required when --pool is set.
 2. --disp-kernel, --disp-a, --disp-b: choose the dispersal kernel of dispersal.py v2.
    The kernel and the seed go into the world tag, so runs do not overwrite each other.
 3. Final run: species keep the fields they had in assembly (r_field from the
    assembly), and the demographic seed comes from the master seed (test_04 E4:
    species lost in the final run sat on cells poor under their new field).
 4. Saved metadata: kernel settings, master seed, species ids, number pruned per round;
    checkpoints keep species ids, so a resumed run rebuilds the same fields.
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py")
MARK = "# obj1_v2"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already carries the obj1_v2 changes")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []                                     # (start, end, new_lines): replace L[start:end]

i = find(lambda s: "help='override long-distance dispersal probability')" in s, "--ldd-prob help")
edits.append((i + 1, i + 1, [
    f"    p.add_argument('--disp-kernel', choices=['nn4', '2dt', 'gauss'], default=None,   {MARK}",
    "                   help='dispersal kernel of dispersal.py; default config.DISPERSAL_KERNEL')",
    "    p.add_argument('--disp-a', type=float, default=None, help='2Dt scale a or Gaussian sd, in cells')",
    "    p.add_argument('--disp-b', type=float, default=None, help='2Dt shape b, must be > 1')",
    "    p.add_argument('--sim-seed', type=int, default=None,",
    "                   help='master seed of this community (assembly, demography, final run); required with --pool')"]))

i = find(lambda s: "dispersal.LONG_DISTANCE_PROB = float(args.ldd_prob)" in s, "ldd override")
edits.append((i + 1, i + 1, [
    f"    if args.disp_kernel is not None:                                 {MARK}",
    "        dispersal.DISPERSAL_KERNEL = args.disp_kernel",
    "    if args.disp_a is not None:",
    "        dispersal.DISPERSAL_A = float(args.disp_a)",
    "    if args.disp_b is not None:",
    "        dispersal.DISPERSAL_B = float(args.disp_b)",
    "    dispersal.unit_kernel_matrix()                                   # stops here on a bad kernel setting",
    "    if args.pool is not None and args.sim_seed is None:",
    "        raise SystemExit('set --sim-seed: every community needs its own master seed (obj1_v2)')"]))

i = find(lambda s: s.strip() == "init_r = init_C = init_N = None", "init_r = init_C = init_N = None")
edits.append((i, i + 1, [f"{ind(L[i])}init_r = init_C = init_N = init_ids = None   {MARK}"]))

i = find(lambda s: "# ---------- NEW: resume from checkpoint" in s, "resume block comment")
edits.append((i, i, [
    f"        world_tag += (f\"_k{{dispersal.DISPERSAL_KERNEL}}\"                  {MARK}",
    "                      + (f\"a{_sanitize_slug(dispersal.DISPERSAL_A)}b{_sanitize_slug(dispersal.DISPERSAL_B)}\"",
    "                         if dispersal.DISPERSAL_KERNEL != 'nn4' else '') + f\"_s{args.sim_seed}\")",
    ""]))

i = find(lambda s: "init_N = ck['N'] if 'N' in ck.files else None" in s, "resume init_N")
edits.append((i + 1, i + 1, [f"{ind(L[i])}init_ids = ck['species_id'] if 'species_id' in ck.files else None   {MARK}"]))

i = find(lambda s: "occ_counts=state['occ']," in s, "checkpoint payload occ_counts")
edits.append((i + 1, i + 1, [f"{ind(L[i])}species_id=state.get('ids'),   {MARK}"]))

i = find(lambda s: "detection_threshold=args.ibm_det_thr," in s and "#" in s, "assembly call detection_threshold")
if L[i - 1].strip() != "seed=1,":
    sys.exit(f"STOP: expected 'seed=1,' above the detection_threshold line, found '{L[i - 1].strip()}'; nothing written")
edits.append((i - 1, i, [f"{ind(L[i - 1])}seed=args.sim_seed,                       {MARK}"]))

i = find(lambda s: s.strip() == "init_r=init_r, init_C=init_C, init_N=init_N,", "assembly call init_r")
edits.append((i, i + 1, [f"{ind(L[i])}init_r=init_r, init_C=init_C, init_N=init_N, init_ids=init_ids,   {MARK}"]))

i = find(lambda s: "m_ibm = IBMModel(r_ibm, C_ibm, initial_N=N_ibm," in s, "final-run IBMModel")
j = next((k for k in range(i, min(i + 10, len(L))) if "seed_field=args.env_seed_field" in L[k] and L[k].rstrip().endswith(")")), None)
if j is None:
    sys.exit("STOP: end of the final-run IBMModel call not found within 10 lines; nothing written")
edits.append((i, j + 1, [
    f"        _F = extra_ibm.get('r_field') if use_assembly else None      {MARK}: species keep their assembly field",
    "        _final_seed = int(extra_ibm['final_run_seed']) if use_assembly else 1",
    "        m_ibm = IBMModel(r_ibm, C_ibm, initial_N=N_ibm,",
    "                        nsteps=int(args.tmax/STEP_SIZE),",
    "                        record_step=int(args.record/STEP_SIZE),",
    "                        record_mode=args.ibm_record_mode,",
    "                        dispersal_type='propagule', seed=_final_seed,",
    "                        r_field=_F,",
    "                        length_scale=None if _F is not None else args.env_length_scale,",
    "                        var_r=None if _F is not None else args.env_var_r,",
    "                        seed_field=args.env_seed_field)"]))

i = find(lambda s: '"LONG_DISTANCE_PROB": _np.float32(float(dispersal.LONG_DISTANCE_PROB)),' in s, "train_out LONG_DISTANCE_PROB")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}\"DISPERSAL_KERNEL\": _np.array([dispersal.DISPERSAL_KERNEL]),   {MARK}",
    f"{p}\"DISPERSAL_A\": _np.float32(dispersal.DISPERSAL_A),",
    f"{p}\"DISPERSAL_B\": _np.float32(dispersal.DISPERSAL_B),",
    f"{p}\"DISPERSAL_IMAGES\": _np.int64(dispersal.DISPERSAL_IMAGES),",
    f"{p}\"SIM_SEED\": _np.int64(args.sim_seed if args.sim_seed is not None else -1),",
    f"{p}\"species_id\": extra_ibm.get(\"species_id\"),"]))

i = find(lambda s: s.strip() == '"seed": _np.int64(1),', 'train_out "seed"')
edits.append((i, i + 1, [f"{ind(L[i])}\"seed\": _np.int64(args.sim_seed if args.sim_seed is not None else 1),   {MARK}"]))

i = find(lambda s: 'for key in ("ASM_round", "ASM_attempts", "ASM_established",' in s, "ASM key loop")
edits.append((i, i + 1, [L[i].replace('"ASM_established",', '"ASM_established", "ASM_pruned",') + f"   {MARK}"]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
diff = list(difflib.unified_diff(src.split("\n"), new, "original/run_all_rps.py", "work/run_all_rps.py", lineterm="", n=1))
print("\n".join(diff))
print(f"\n{len(edits)} edits, {sum(1 for d in diff if d.startswith('+') and not d.startswith('+++'))} lines added or changed")
compile(out, "run_all_rps.py", "exec")
print("the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: run_all_rps.py")
else:
    print("dry run: nothing written (add --apply to write)")
