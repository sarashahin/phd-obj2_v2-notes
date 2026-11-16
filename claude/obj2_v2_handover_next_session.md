# Objective 2 v2: handover for the next session (16 Nov 2026, step 59: B-1b rule signed (M3); stage_b1b_k2.py built, sandbox 17 PASS; waiting for Sara's B1b-A run log)

This file was shortened on 15 Nov: the step-by-step history (steps 1-46) lives in the results logs and READMEs below, so it is not repeated here.

Where the files live (Sara, 15 Nov): the private GitHub repo sarashahin/phd-obj2_v2-notes holds the living MD files (paths as in the claude.ai project) and the old versions; MD files are updated there from 15 Nov, one commit per step. The project copies of those MD files stop at 15 Nov; no project file is removed. Open questions for Sara: claude/obj2_v2_open_questions.md in the repo.

Read first, in this order: obj2_v2/README_stage_H_part2.md (status table and run blocks from B-0; earlier blocks H0-P4d in obj2_v2/README_stage_H.md), claude/obj2_v2_H_rules_part2.md (T, S, K and sensitivity-line rules; part 1 is claude/obj2_v2_H_rules_draft.md), claude/obj2_v2_results_log_H_part4.md (from step 40; earlier parts H, H_part2, H_part3), obj2_v2/README_tests_part2.md (tests from test_34; earlier rows in README_tests.md). claude/obj2_v2_root_causes_story.md is the short record from manuscript audit to stage H; claude/obj2_v2_paper_plan.md holds the paper decisions.

## Where the work stands

- Stage G finished (G5d read real_eval once, 11 Nov). Stage H finished: BNM fit half 44 windows, test half 57; test-half labels read once at H6 (12 Nov 22:00:21Z); H-R1 FAIL both seeds (below B1 for 20 or fewer records over 6 years, above at 21+), H-R2 coverage 0.85-0.95 in every group, level high at 0-3 records (ratio 1.25-1.56).
- Products: P3 (all-tetrad AOO) and P4 (land-only AOO, UKCEH LCM 2019; P4-2 second labelled reading of real_eval, land-only rule PASS for all six methods; P4-3 products af01b4ad; P4d paper tables 1991a255 / 0dbe7d5f). Objective 4 claim (Sara 13 Nov): upgrade of recorded AOO with stated limits; EOO out of scope.
- Rare end, stated limits: count level at 0-3 records (test half), map ranking for 20 or fewer records (H-R1), true 1-cell AOO 4.3-4.5 times too large on simulation (P4-2). B-0 (14 Nov): habitat cannot shrink the 1-cell excess; the blank mode used by every BNM product keeps G3's rule.
- Paper (Sara 14 Nov): one paper for all of v2; v1 never submitted; figures before text (Figure 2 posterior samples, figure set, BNM HTML profiles as step18 without EOO, 3D trace movies); questions of steps 38-39 on figures still open.
- 15 Nov: Sara asked for rare-end options without retraining (step 45) and said yes to T and S now, K only if T leaves a gap, yes to a sensitivity line (assumption, not a product). Rules written before any value (H rules part 2). Her "continue" read as yes to Q1-Q3 there (to be confirmed).
- T: stage_t_tiebreak.py (27631419) with test_35 (fb884828); Sara's run 15 Nov: self-test 13 PASS, gates V0-V4 PASS, t_report.json 1305724a; rule NO. Ties close 38-65% of the sparse gap; class 1 and records 4-6 stay below B1 exact. T fixes no root cause and changes no AOO.
- Step 52 (15 Nov, late): the step 51 session was cut off before its reply; B-1a rechecked in a new sandbox (11 PASS); a path bug fixed before any run (map folder ../data/lcm2019_1km/data, as P3c and P4-0); version 2 committed and block B1a-A sent again.
- Sara 15 Nov: K and S on hold; T recorded as a report-only finding. Figure answers: real_eval rows as "display reading 3, no scores", one species per recorded class, seed fixed first, plus one BNM row; Figure 2 and the BNM species pages (as v1) first; movie = denoising; journal Ecological Informatics; one paper; no v3 (v2 is final). Open questions: claude/obj2_v2_open_questions.md (G1-G3 before the Figure 2 draft; P3, R1, R2 asked again).
- B-1a (Sara's run 16 Nov, step 54): self-test 11 PASS, gates J0-J4 PASS, b1a_habitat.npz fac95acf, b1a_report.json 71e2bedb. PC1 (0.375 of the variance) is pasture against arable (improved grassland +0.886, arable -0.424); semi-natural classes carry little weight. Found before B-1b's rule: the control map is 0 where the paired window has sea, which weakens the control on coastal pairs (M1).
- Step 55 (16 Nov, Sara asked after the N33 session was flagged and its sketches were not readable): a hand-drawing guide to the Objective 1 and BNM pipeline, pipeline only, no results: 17 sheets (simulator, BNM data, comparison), a road map of every cause and fix, and the issues still open. Claude Doc "Objective 1 and BNM: a hand-drawing guide to the pipeline" (https://claude.ai/artifact/5z3KzD5fPt6XtTkxPTP8Ve, 18 drawings); text in claude/obj1_bnm_drawing_guide.md. Open question D1: extend it to Objectives 2-4 in the same format.
- Step 56 (16 Nov): Sara asked for the guide as hand sketches with less text. First draft on the Design canvas "Objective 1 and BNM in hand sketches" (https://claude.ai/artifact/Uvjy2DKraqVMPDJVfiRWov): start-here board and sheets 1-5; scripts in claude/obj1_bnm_sketches/. Question D2: style OK, then sheets 6-17, road map and open issues in the same style.
- Step 57 (16 Nov): Sara said the draft left out most of the guide; redrawn in full on the same canvas: 24 boards (start here, sheets 1-17 each with how / problems and fixes / settings / still open, road map 4a-4c, still open 5a-5c). Scripts in claude/obj1_bnm_sketches/. D2 now asks whether the boards read well when drawn by hand.
- Step 58 (16 Nov): Sara said yes to M1 and M2. B-1b's rule written in full in H rules part 2 before any value: own map against window k + 22's map on the same far holes and kept records; PASS when AUC(own) - AUC(control) in S has its 95% interval above 0; real_val reference (8,174 habitat-shown stacks) report only; gates W0-W6. Waits for M3 (sign-off).
- Step 59 (16 Nov): M3 yes. stage_b1b_k2.py (2ba374a6) with test_37 (b68db477); sandbox self-test 17 PASS with her real tools (torch 2.15.0 CPU installed so test_25's planted pipeline runs); block B1b-A in README_stage_H_part2.
- Project space (15 Nov, step 53): after the earlier 17, 182 more project code files removed (all checked to be in the repo first; Sara's yes to S1); project now 734,923 of 2,000,000 (36.7%). The four stray .pyc removed (S2). Tools from B-1a on come through the repo (G3), so the project keeps only MD docs, the 8 kept files, temporal_checks/, the root-level v1 scripts and __init__.py.

## Next steps

0. Sara checks the hand-sketch guide (step 57) and answers D1 and D2.
1. Read Sara's B1b-A log (tool 2ba374a6, test b68db477): self-test 17 PASS, W0-W6, the S row of AUC(own) - AUC(control) (the rule), the support counts per group, and the real_val reference; log it in results log part 4 as the next step. If W1 or W3 fails, stop and find the cause before reading any rule value.
2. B-1b result: if the S interval is above 0, B-2's rule is written in full next (step 41 plan, go rule 14 Nov); otherwise B stops and the paper states the habitat limit with the B-0 and B-1 numbers.
3. Then the Figure 2 draft (G1: blank rows) and the BNM species pages.
4. Tools go to Sara through the repo: she runs git pull and copies the tool and test into obj2_v2/tools and obj2_v2/tests; md5 = the file as committed. Sandbox: torch can be installed from PyPI (download the wheels, install offline) so her planted pipelines run with her real tools.

## Rules that never change after a result

No leakage; real_eval read twice (G5d, then P4-2 labelled); any further reading is a labelled rerun with its fits md5-logged first. Test half opened once at H4, its labels once at H6; a test-half score of T or S is a labelled second look and needs Sara's yes. Floor 50 distinct stacks; holes (3n + 5) // 10; H-R1 stack-first means, 2,000 window resamples, seed 31; H-R2 coverage per stack-repeat; folds = make_folds (seed 77, 3) on window names. No training, fine-tuning or checkpoint choice on BNM data; a fix cycle would be a new model version with its own rules.

## How Sara wants the work run

- Every reply that needs her to run something ends with an exact command block (full cd path) and a verification line; a plan waiting for sign-off says so in one line.
- Reply: before answering, check constraints and edge cases; up to 3 questions if the goal is unclear (why, and Claude's suggested answer), XML task tags, key claim in bold, verification block; short plain English; say "communities".
- Code: sandbox test on planted data first; on her machine plain python3 (not -I). Tools reuse validated tools by import (T imports stage_f0_spread_check.py). Project files come back inline from project_read, so a sandbox needs stubs for deep tool chains: copy the exact functions used, and let the self-test on her machine run with her real tools. Stubs and planted-world scripts stay out of the project.
- After each step update, in the repo, README_stage_H_part2, README_tests_part2, results log H part 4, H rules part 2 (when a rule changes), the open questions file, and the Claude Doc (project HQmk81fegK2SmYRopQBfsS; body node e9c6596e-b7f6 in doc 84e3e63f-aac7-4509-8b92-c0b1ffa867a5).
- Download-form md5 = md5 of the file without its final newline; blocks also give a content fingerprint (no #! line, no trailing spaces).
- Keep each tool under about 700 lines; one step per turn. Project knowledge is near its limit (about 94% on 15 Nov): MD files go to the repo; tools still go to the project for Sara to download; nothing is removed.

cd path: ~/Downloads/"The-Lotka-Volterra-Metacommunity-Model-main (1)"/The-Lotka-Volterra-Metacommunity-Model-main/PSD_Dispersal_pool/obj2_v2/work
