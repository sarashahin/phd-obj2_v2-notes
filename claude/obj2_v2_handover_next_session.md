# Objective 2 v2: handover for the next session (15 Nov 2026: T run, rule NO; K and S on hold; figures next; notes moved to the private GitHub repo)

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
- Sara 15 Nov: K and S on hold; T recorded as a report-only finding; figures next. Open questions listed in claude/obj2_v2_open_questions.md (F1-F3 needed before the figures).

## Next steps

1. Sara's answers to F1-F3 (open questions file); then the Figure 2 tool (Planning, then Draft with a sandbox test), reading v2 files only.
2. Old IBM versions v2_1-v2_4 added to the repo from Sara's machine (block in the 15 Nov reply).
3. K and S stay on hold unless Sara reopens them.

## Rules that never change after a result

No leakage; real_eval read twice (G5d, then P4-2 labelled); any further reading is a labelled rerun with its fits md5-logged first. Test half opened once at H4, its labels once at H6; a test-half score of T or S is a labelled second look and needs Sara's yes. Floor 50 distinct stacks; holes (3n + 5) // 10; H-R1 stack-first means, 2,000 window resamples, seed 31; H-R2 coverage per stack-repeat; folds = make_folds (seed 77, 3) on window names. No training, fine-tuning or checkpoint choice on BNM data; a fix cycle would be a new model version with its own rules.

## How Sara wants the work run

- Every reply that needs her to run something ends with an exact command block (full cd path) and a verification line; a plan waiting for sign-off says so in one line.
- Reply: thinking block with up to 3 questions (why, and Claude's suggested answer), XML task tags, key claim in bold, verification block; short plain English; say "communities".
- Code: sandbox test on planted data first; on her machine plain python3 (not -I). Tools reuse validated tools by import (T imports stage_f0_spread_check.py). Project files come back inline from project_read, so a sandbox needs stubs for deep tool chains: copy the exact functions used, and let the self-test on her machine run with her real tools. Stubs and planted-world scripts stay out of the project.
- After each step update, in the repo, README_stage_H_part2, README_tests_part2, results log H part 4, H rules part 2 (when a rule changes), the open questions file, and the Claude Doc (project HQmk81fegK2SmYRopQBfsS; body node e9c6596e-b7f6 in doc 84e3e63f-aac7-4509-8b92-c0b1ffa867a5).
- Download-form md5 = md5 of the file without its final newline; blocks also give a content fingerprint (no #! line, no trailing spaces).
- Keep each tool under about 700 lines; one step per turn. Project knowledge is near its limit (about 94% on 15 Nov): MD files go to the repo; tools still go to the project for Sara to download; nothing is removed.

cd path: ~/Downloads/"The-Lotka-Volterra-Metacommunity-Model-main (1)"/The-Lotka-Volterra-Metacommunity-Model-main/PSD_Dispersal_pool/obj2_v2/work
