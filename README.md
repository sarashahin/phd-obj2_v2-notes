# phd-obj2_v2-notes (private)

Working notes for Sara Shahin's PhD Objective 2 v2 (MetaDiffusion v2: IBM-simulated training, BNM butterfly atlas hold-out, AOO products). Started 15 Nov 2026 because the claude.ai project knowledge was near its size limit. Nothing here is deleted; every change is a commit, so older versions stay in the history.

Paths mirror the claude.ai project "object2_new_train_inpaint", so a path cited in one file (for example claude/obj2_v2_results_log_H_part4.md) is the same file here.

## Living files (updated here from 15 Nov; the project copies stop at 15 Nov)

- claude/obj2_v2_handover_next_session.md: read first in every session
- claude/obj2_v2_open_questions.md: questions waiting for Sara
- claude/obj2_v2_results_log_H_part4.md: results log from step 40
- claude/obj2_v2_H_rules_part2.md: rules for the rare-end checks T, S, K
- claude/obj2_v2_paper_plan.md: paper and figure plan
- obj2_v2/README_stage_H_part2.md, obj2_v2/README_tests_part2.md: run blocks and tests from B-0
- obj2_v2/tools, obj2_v2/tests: tools and self-tests from 15 Nov (stage_t_tiebreak.py, test_35)

## Archive (old versions, read-only)

- claude/V9_review.md to claude/V16_review.md: reviews of the unsubmitted v1 MetaDiffusion manuscript (4-5 Oct 2026)
- obj1_v2/new_code/assembly_stepwise_ibm_v2_1.py to v2_4.py: older IBM versions (added from Sara's machine)

## Use

On Sara's machine, first time: `git clone https://github.com/sarashahin/phd-obj2_v2-notes.git ~/phd-obj2_v2-notes`; after that `git -C ~/phd-obj2_v2-notes pull`.

## Removed from the claude.ai project on 15 Nov 2026 (copies here only)

claude/V9_review.md to claude/V16_review.md; obj1_v2/new_code/assembly_stepwise_ibm_v2_1.py to v2_4.py; claude/obj2_v2_results_log_H_part4.md, claude/obj2_v2_H_rules_part2.md, claude/obj2_v2_paper_plan.md, obj2_v2/README_stage_H_part2.md, obj2_v2/README_tests_part2.md. Each was checked to be on GitHub before it was removed.

On 15 Nov 2026 (step 53), after Sara's yes to S1, 182 code files were removed from the project: obj2_v2/tools and tests (74) and obj1_v2/tools, tests and new_code (108), the full list in claude/project_cleanup_list.md. Each of the 182 was first checked to be in this repo at the same path (committed, non-empty); no MD or README was touched. Kept in the project: 8 files (6 that Sara's patch tools changed on her machine or that are newer here, 2 sandbox-only tests), plus every MD doc, the temporal_checks/ folder, the root-level v1 scripts and __init__.py (none of those are in the repo yet). The project dropped to 734,923 of 2,000,000 bytes (36.7%). Nothing was removed from GitHub.

## Tools from 15 Nov

New tools and tests are delivered here (Sara, 15 Nov): `git pull`, then copy them into PSD_Dispersal_pool/obj2_v2/tools and tests. The md5 to check is the md5 of the file as committed. Sara's own code folders were added on 15 Nov (commit 462314a).

## .pyc removed (step 53, Sara's yes to S2)

The four __pycache__/*.pyc files that commit 1e717958 committed by mistake (sandbox bytecode, not code) were git-removed on 15 Nov. The .gitignore added in step 52 keeps new ones out.
