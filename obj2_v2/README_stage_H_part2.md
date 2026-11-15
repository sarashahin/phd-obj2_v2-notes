# obj2_v2: stage H run blocks, part 2 (from 14 Nov 2026, habitat checks B; rare-end checks T and S from 15 Nov)

obj2_v2/README_stage_H.md holds the run blocks and status of H0-P4d; it reached a size where a full rewrite per step costs too much, so the blocks from the habitat checks B go here. Rules: claude/obj2_v2_H_rules_draft.md, section "Habitat checks B", and claude/obj2_v2_H_rules_part2.md (T, S, K, sensitivity line). Results: claude/obj2_v2_results_log_H_part4.md.

## Status

| Step | State |
|---|---|
| B-0 | Sara's yes 14 Nov; tool sandbox self-test 12 PASS; block B0-A run by Sara 14 Nov: self-test 12 PASS with her real G3 / G3b tools, gates C0-C5 PASS, b0_rows.csv d6ad71c7, b0_report.json 9edd5d52; R0a: habitat of training strength is not expected to shrink the true 1-cell excess; R0b: blanked half alone PASS for both seeds and pooled (B4 FAIL at records 7-10) |
| B-1 / B-2 | stopped: B-0 showed the habitat cannot shrink the 1-cell excess; Sara asked for options without retraining (step 45) |
| T | Sara's yes 15 Nov; block T-A run by Sara 15 Nov: self-test 13 PASS, gates V0-V4 PASS, t_scores.npz 6f7f67ac, t_report.json 1305724a; rule NO (class 1 and records 4-6 still FAIL against B1 exact for both seeds); ties close 38-65% of the sparse gap; report only |
| K | on hold (Sara 15 Nov) |
| S | on hold (Sara 15 Nov) |
| B-1a | Sara's yes 15 Nov (B-1 before Figure 2); rule in H rules part 2; stage_b1a_habitat_axis.py with test_36, sandbox 11 PASS; block B1a-A sent (tools now come through the repo) |
| B-1b | after B-1a runs; rule to be written in full first |

```
B-0 sketch
real_val frozen set (16,347)  --blank flag-->  blanked 8,173 | shown 8,174
P4-1 cross-fit (land-only, P4 fits, rule E)       same correction for both halves
   per half: G3 evaluate + rule | option B P | true class 1, 2-5
   shown - blanked, 2,000 community resamples (seed 31); minus B4 / B5 on the same draws
R0a: does habitat of training strength shrink the true 1-cell excess?   -> no (pooled -0.205 [-0.832, +0.509])
R0b: does the blanked half alone (the BNM mode) keep G3's rule?           -> yes for seeds and pooled
```

## B-0, the blank-mode split (block B0-A)

Save `obj2_v2/tools/stage_b0_blank_split.py` into tools and `obj2_v2/tests/test_34_b0_blank_split.py` into tests. CPU, under a minute. From obj2_v2/work:

```bash
cd ~/Downloads/"The-Lotka-Volterra-Metacommunity-Model-main (1)"/The-Lotka-Volterra-Metacommunity-Model-main/PSD_Dispersal_pool/obj2_v2/work
md5sum ../tools/stage_b0_blank_split.py ../tests/test_34_b0_blank_split.py ../data/stage_p4_1/p4_1_crossfit.npz ../data/stage_p4_1/p4_1_report.json
python3 -c "import hashlib,sys;[print(hashlib.md5(b'\n'.join(l.rstrip() for l in open(f,'rb').read().split(b'\n') if l[:2]!=bytes([35,33])).rstrip(b'\n')).hexdigest(), f) for f in sys.argv[1:]]" ../tools/stage_b0_blank_split.py ../tests/test_34_b0_blank_split.py
ls -d ../data/stage_b0_blank_split 2>/dev/null || echo "no stage_b0_blank_split"
python3 ../tools/stage_b0_blank_split.py --self-test 2>&1 | tail -n 13
[ ! -e ../data/stage_b0_blank_split ] && python3 -u ../tools/stage_b0_blank_split.py > ../logs/stage_b0_blank_split.txt 2>&1; cat ../logs/stage_b0_blank_split.txt
md5sum ../data/stage_b0_blank_split/*
```

Verification line: tool c55991ae5be96a6a6036e9686d3c75be, test 4ad15593348b3954e3f0cb693b595f6c (download form; fingerprints 7444539963a240834d9f4046b0e2d324, 3d002582413ca0e8195e74a5eaea6e52); inputs 371f424a..., 5e980763...; "no stage_b0_blank_split"; self-test "12 PASS, 0 FAIL"; C0 "expected 5e980763", C1 "16,347 frozen stacks ... blanked 8,173, shown 8,174 ... True", C2 differences 0.0e+00, C3-C5 PASS; last line "stage B-0: gates 6 PASS, 0 FAIL; ...".

B-0 result (Sara's run, 14 Nov): every md5 as sent; self-test 12 PASS; gates C0-C5 PASS (frozen set 7f4844ed; C2 0.0 for every method); b0_rows.csv d6ad71c7e3f8cdf176bc5bd3e3dd7e3f, b0_report.json 9edd5d52015a617496542a6d8a857516. Rechecked from the log: the halves recombine to P4-1's true 1-cell ratios (pooled 3.82, B3 7.09, B5 8.94). Numbers and reading: results log part 4, step 44.

## T, tie-break with B1 exact (block T-A)

```
T sketch (BNM fit half, report only; numbers are an example, not results)
one stack-repeat, holes sorted by the model's share of 50 samples
  model   0.30  0.10  0.02  0.02  0.02  0.00  0.00  0.00    far holes tie at 0-1 of 50
  B1x       .     .   -2.1  -4.0  -6.5  -3.0  -8.2  -9.9    B1 exact orders holes only inside a tie
  T       0.30  0.10  0.02+ 0.02+ 0.02+ 0.00+ 0.00+ 0.00+   added term below 1/100: model order kept
H-R1 again, both seeds, both B1 versions -> rule: no group FAILs, 21-50 / 51+ / 21+ keep a mean >= 0
room T has = share of recorded / unrecorded hole pairs the model ties (printed per group)
limit: on tied pairs T copies B1 exact, so at best about B1 level where the model ties
control (sandbox): B1 exact replaced by noise -> T - model about 0 (+0.0011 [-0.0014, +0.0046])
```

Problem: on the BNM fit and test halves the model ranks below B1 for 20 or fewer records over 6 years (H2b, H6); H6 D1 puts the sparse stacks' far score flat at 0.013-0.015 of 50 samples, so many far holes tie and H-R1 counts each tied pair as half. Idea: keep the model's order, break only its ties with B1 exact. Plan: one tool, fit half only, no parameter, rule written before the run. Expected outputs: t_scores.npz, t_report.json, the log. Open: whether T lifts the sparse groups to B1 (the rule) and how much room the ties give (tie share). Failed or open items so far: none run on real data.

Save `obj2_v2/tools/stage_t_tiebreak.py` into tools and `obj2_v2/tests/test_35_t_tiebreak.py` into tests. stage_f0_spread_check.py (version 2) and test_25 / test_23 must stay where they are; nothing else changes. CPU; the self-test builds test_25's planted pipeline (a few minutes), the run a few minutes. From obj2_v2/work:

```bash
cd ~/Downloads/"The-Lotka-Volterra-Metacommunity-Model-main (1)"/The-Lotka-Volterra-Metacommunity-Model-main/PSD_Dispersal_pool/obj2_v2/work
md5sum ../tools/stage_t_tiebreak.py ../tests/test_35_t_tiebreak.py ../tools/stage_f0_spread_check.py ../data/stage_h3_freeze/h3_freeze.json
python3 -c "import hashlib,sys;[print(hashlib.md5(b'\n'.join(l.rstrip() for l in open(f,'rb').read().split(b'\n') if l[:2]!=bytes([35,33])).rstrip(b'\n')).hexdigest(), f) for f in sys.argv[1:]]" ../tools/stage_t_tiebreak.py ../tests/test_35_t_tiebreak.py ../tools/stage_f0_spread_check.py
ls -d ../data/stage_t_tiebreak 2>/dev/null || echo "no stage_t_tiebreak"
python3 ../tools/stage_t_tiebreak.py --self-test > ../logs/test_35_t_tiebreak_selftest.txt 2>&1; st=$?
grep -E "^(U|T|S|bug)[^ ]* .*: (PASS|FAIL)$|PASS, " ../logs/test_35_t_tiebreak_selftest.txt
[ $st -eq 0 ] && [ ! -e ../data/stage_t_tiebreak ] && python3 -u ../tools/stage_t_tiebreak.py > ../logs/stage_t_tiebreak.txt 2>&1; cat ../logs/stage_t_tiebreak.txt
md5sum ../data/stage_t_tiebreak/*
```

Verification line: tool 276314196474b218c5f5a42faf53923f, test fb884828a212ec449d164b1eac7783c8 (download form; fingerprints 78bedade42d2a846dde8feda2cfac998, 414b3356dc99074e689884b18b0727e9); stage_f0_spread_check.py 75ce6f2414a7719501bc30afbb989c26 (fingerprint 3599c46ba15995431b8eaba743f07660); h3_freeze.json 5516a6fc9e1cfcda3ce1a883aa5467f9; "no stage_t_tiebreak"; self-test 13 lines then "13 PASS, 0 FAIL"; V0 "= signed True ... differ: [] ... version 2 True ... PASS"; V1 AUC differences 0.0e+00, whole-number gap about 1.4e-06 or less; V2 "largest difference 0.0e+00, verdicts same True"; V3 True for both seeds; V4 PASS; last line "stage T: gates 5 PASS, 0 FAIL; H-R1 with T seed 1 ..., seed 2 ...; T worth using by the rule written before the run: YES" (or NO). Send the whole log. If the self-test has a FAIL, the run line does nothing (it needs exit code 0); send ../logs/test_35_t_tiebreak_selftest.txt then.

T result (Sara's run, 15 Nov): every md5 as sent; self-test 13 PASS; gates V0-V4 PASS (V1 whole-number gap 1.4e-6); t_scores.npz 6f7f67acc20e348aee20b62c967a886d, t_report.json 1305724a75dbd36a3507dae1ea7392b0. Rule NO. Sparse set S: T - model +0.0126 / +0.0100; seed 1 vs B1 exact -0.0329 -> -0.0202, seed 2 -0.0156 -> -0.0056. Numbers and reading: results log part 4, step 47.

## B-1a, the habitat axis from the Land Cover Map (block B1a-A)

```
B-1a sketch (BNM fit half, 44 windows; no BNM record or label opened)
LCM 2019 1 km, 21 class %  -->  per 2 km tetrad: sum of its 4 cells, divided by the total  =  composition (21 shares)
fit set: tetrads in L_w (P4-0) with a composition            -->  PCA (covariance)  -->  PC1 = one habitat axis
score = (composition - mean) . PC1  -->  z per window over L_w (as channel 18 in training); sea or no cover -> 0
control for B-1b: window k gets the map of window k + 22 (name order)
next, B-1b: does nearness on this axis to a species' records pick out its hidden records better than the control map?
```

Problem: on BNM the model gets no habitat layer. Idea: give it one axis of land-cover composition, built like the training habitat. Plan: B-1a builds the axis (this block); B-1b scores it on the fit-half holes against a control and the real_val reference; B-2 and the GPU step only if B-1 passes (go rule, 14 Nov). Expected outputs: b1a_habitat.npz, b1a_report.json, the log; the PC1 loadings show what the axis means. Open: whether one axis carries signal for butterflies (B-1b).

Tools now come through the repo (Sara, 15 Nov, G3). CPU, under a minute.

```bash
cd ~/phd-obj2_v2-notes && git pull
cd ~/Downloads/"The-Lotka-Volterra-Metacommunity-Model-main (1)"/The-Lotka-Volterra-Metacommunity-Model-main/PSD_Dispersal_pool/obj2_v2/work
cp ~/phd-obj2_v2-notes/obj2_v2/tools/stage_b1a_habitat_axis.py ../tools/ && cp ~/phd-obj2_v2-notes/obj2_v2/tests/test_36_b1a_habitat_axis.py ../tests/
md5sum ../tools/stage_b1a_habitat_axis.py ../tests/test_36_b1a_habitat_axis.py ../tools/export_obj2.py ../tools/stage_p3c_band_check.py ../data/lcm2019_1km/gb2019lcm1km_percentage_target.tif ../data/stage_p4_0/p4_windows.npz
ls -d ../data/stage_b1a_habitat 2>/dev/null || echo "no stage_b1a_habitat"
python3 ../tools/stage_b1a_habitat_axis.py --self-test > ../logs/test_36_b1a_selftest.txt 2>&1; st=$?
grep -E "^(U|T|bug)[^ ]* .*: (PASS|FAIL)$|PASS, " ../logs/test_36_b1a_selftest.txt
[ $st -eq 0 ] && [ ! -e ../data/stage_b1a_habitat ] && python3 -u ../tools/stage_b1a_habitat_axis.py > ../logs/stage_b1a_habitat_axis.txt 2>&1; cat ../logs/stage_b1a_habitat_axis.txt
md5sum ../data/stage_b1a_habitat/*
```

Verification line: md5 stage_b1a_habitat_axis.py 4282109d0b21f54bc3a6e263365e5e61, test_36 ee75fcb2666a6ad54499b6864220e6e9 (copied from the repo, so the md5 is the file as committed), export_obj2.py 2685f91aac0c2fe89c9344abe112bd7b, stage_p3c_band_check.py 98faa2d95d1a96fa6680972c1100f3ca, map d66aac8836c89b317e9b6b0a09262c66, p4_windows.npz 928cf96c9bcbe9a14bb0f639adb581eb; "no stage_b1a_habitat"; self-test 11 lines PASS and "11 PASS, 0 FAIL"; J0 every part True; J1 "44 fit-half windows ... 0 tetrads differ"; J2 |cos| 1.000000000000 and every difference at 1e-9 or less; J3 1 file; J4 PASS; last line "stage B-1a: gates 5 PASS, 0 FAIL; habitat axis PC1 (...) for 44 fit-half windows". Send the whole log.
