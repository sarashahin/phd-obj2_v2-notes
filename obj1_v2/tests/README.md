# Objective 1 v2 tests

Tests for the IBM simulator before the re-simulation. They read code or saved worlds and write only to `obj1_v2/logs` (test_08 also writes tiny worlds into `work/results`). Run them from `obj1_v2/work` (new code) or `obj1_v2/original` (old code); each script puts the current folder first on the import path.

Labels: PASS/FAIL is a requirement, ecological requirements included. FLAG means the code does what it says but works against the aim. A PASS counts only when the numbers also make ecological sense; the "ecological reading" column says what to look for.

| Test | Question | Run (from obj1_v2/work) | Ecological reading |
|---|---|---|---|
| test_01_dispersal_kernel | Do propagules leave at rate d and land by the kernel? | `USE_GPU=0 python3 ../tests/test_01_dispersal_kernel.py` | Plumbing only; d and kernel values are chosen in Stage 2. |
| test_02_components | Do interactions, fields, records and history helpers do what they say? | `USE_GPU=0 python3 ../tests/test_02_components.py` | FLAGs are design points for Stages 2 and 3. |
| test_03_assembly_mini | Does assembly keep its books, seeds and per-species fields right? | `USE_GPU=0 python3 ../tests/test_03_assembly_mini.py` | A species' environment must not change when another species dies. |
| test_04_saved_worlds | What do saved worlds contain (history, fields, interactions, records)? | `USE_GPU=0 python3 ../tests/test_04_saved_worlds.py "<files>"` | Ranges should carry an imprint of their own field; losses in the final run should be few and not tied to a field change. |
| test_05_assembly_equilibrium | Did assembly saturate, did the final run settle, can ranges move? | `USE_GPU=0 python3 ../tests/test_05_assembly_equilibrium.py "<files>"` | Saturation: richness flat, establishment balanced by removal. Ranges can move only with at least one arrival per edge cell per run. |
| test_06_rarity_order | Are rare species rare, or just recent arrivals? | `python3 ../tests/test_06_rarity_order.py "<files>"` | Small ranges piling up at the newest end mean rarity is recency. |
| test_07_shared_seed | Do different communities share species? | `python3 ../tests/test_07_shared_seed.py "<eval files>" "<other files>"` | Observed overlap above the torus-shifted value means leakage across communities. |
| test_08_run_world_smoke | Does a whole run (assembly, final run, export) work after Step 1? | `USE_GPU=0 python3 ../tests/test_08_run_world_smoke.py` | Higher d gives larger ranges; at the same d the fat tail gives more outposts. |

`<files>` for the old evaluation worlds: `"../../results/data/data_eval_unseen/*_training.npz"`. Quote the pattern so Python expands it.

Save every log as `../logs/<test name>_<date>.txt`, for example `| tee ../logs/test_03_assembly_mini_25oct.txt`.

## Results log

| Date | Test | Code | Result |
|---|---|---|---|
| 24 Oct | test_01 | work (dispersal v2) | no FAIL; A differs from the 30 Sep dispersal.py by 3e-16 |
| 24 Oct | test_02 | work | no FAIL; FLAG I4, E2, E4, O1 (tau 1, 2), O2 |
| 24 Oct | test_03 | original assembly | A3 6 dropped, A5 one seed for 49 windows, B2 identical interactions and founder across field seeds, C1 fields lost after a removal (FLAG then; FAIL in the revised test) |
| 24 Oct | test_04 | 25 evaluation worlds | H2 166 dropped, H3 9.6% empty maps, H4 57.0% static, H5 83.7%, H7 0% gaps, E3 own-field imprint +0.049 +- 0.007, E4 lost species on poor cells -0.288 against +0.031, O2 17% nested, O3 0.0% of cells under 2 tau (median 3,911 individuals), O4 median 1,366 |
| 23 Oct | test_05 | 25 evaluation worlds | richness +9.3% over the last 20% of attempts, 8-11% lost in the final run, 0.20 arrivals per edge cell per run |
| 23 Oct | test_06 | 25 evaluation worlds | Spearman -0.178; 87% of the oldest species in 5 cells or fewer |
| - | test_07 | - | not yet run on Sara's files |
| 24 Oct | test_03, test_08 | Step 1 code (Claude's stand-in run) | all PASS; see obj1_v2_audit.md |
| 24 Oct | test_03 | Step 1 code, Sara's machine | 8 PASS lines, 0 FAIL; numbers identical to the stand-in run (49 rounds, 120 species, 49 seeds, C1 100%/100%) |
| 24 Oct | test_08 | Step 1 code, Sara's machine | 8 PASS lines, 0 FAIL; mean range 26.5 against 8.1 cells (d 1e-7 against 2e-8), outposts 0.174 against 0.138 |
| 24 Oct | test_07 | old worlds | stopped: KeyError on a file without P_last_final; fixed the same day (files skipped and listed) |

## Change log

- 23 Oct: test_05, test_06, test_07 written (first run as terminal commands).
- 24 Oct: test_07 skips and lists files without P_last_final (assembly-pool and default-run files); before, the first one stopped the test.
- 24 Oct: test_01 to test_04 and test_08 written; test_03 turned into a requirement test (A3, A5, C1, C2 FAIL on old code); test_04 and test_05 read new files (fields by species id, kernel and d from the file); test_06 counts only true losses.
