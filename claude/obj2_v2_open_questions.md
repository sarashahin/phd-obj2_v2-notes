# Open questions for Sara (15 Nov 2026)

Questions put to Sara in earlier sessions that she has not answered yet. Each has a plain explanation, why it matters, and Claude's suggestion. When she answers, the answer and date go beside the question and the item moves to "Answered".

## Open now (15 Nov, late)

| # | Question | Why it matters | Suggestion |
|---|---|---|---|
| P3 | B-0 in the supplement | Sara wants more testing first (B-1, B-2) | ask after B-1 / B-2 |

## Later

| # | Question | Suggestion |
|---|---|---|
| F4 | BNM species pages: as the v1 step18 pages (Sara, 15 Nov); species scope and phenology read from step18_bnm_profiles.py when the profile tool is planned | as v1, EOO removed |
| P2 | One supplement sentence that EOO was planned, failed and was dropped | Sara will decide later |

## Saved for later consideration (Sara, 15 Nov, in her words: "i will think about it")

Claude's suggestion from the 14 Nov session, kept at her request: "For this paper, change what the model claims at one record; leave the model itself alone." S1, the one-record claim (this paper): for each species, recorded AOO as a hard lower bound, the 90% interval, and P(window AOO under 2.5 tetrads), which is the CR cut-off; a "not decidable" flag when the interval crosses that cut-off; the 0-3 record bias printed beside it. S2, surveyed against never-surveyed (this paper, report only): how much of each rare species' AOO lies on land nobody visited, so "we can't know" becomes "visit these tetrads to settle it"; later a simulation test of whether the model's ranking finds the hidden range with fewer visits than distance alone (not claimable on BNM today). Sara wants more tests of other options that might improve class 1 first; if none helps, she will be asked again about S1 and S2.

## On hold (Sara, 15 Nov)

K (more samples for sparse stacks) and S (level correction by predicted size, with its tercile question) wait. T showed the gap left is in holes the model orders strictly, which K cannot change; S fixes no root cause.

## Answered

- 15 Nov (late, step 53): S1 yes, with "look into the repo first, do it with caution, don't lose any file or MD": 182 project code files removed after all 182 were checked to be in the repo (0 missing, no .md on the list); project 90.3% to 36.7%; 8 files kept. S2 yes: the four stray .pyc removed. B-1b order yes: after the B1a-A log, write B-1b's rule in full, then build.

- 15 Nov (late): G1 yes, Figure 2 simulation rows from habitat-blanked stacks. H1 yes, B-1 before Figure 2 (then B-2 only if B-1 passes; GPU step only if both pass; refit by input mode and a clean test before any BNM claim). S1: list first (shown).

- 15 Nov (late): G2 yes (BNM row drawn by the fixed seed from species with 2-5 records in a coastal window). G3 yes (new tools through the GitHub repo). R1 options (a) and (b): print the test-half bias beside the product at 0-3 records, and an adjusted AOO = AOO / ratio labelled as an assumption. Sara pushed her code folders to the repo (commit 462314a; obj1_v2 tools, tests, new_code; obj2_v2 tools, tests).

- 15 Nov (evening): F1 real_eval, labelled "display reading 3, no scores", species drawn by a seed fixed before any file opens, plus one BNM row (records, samples, posterior, land mask). F2 one species per recorded class. F3 Figure 2 and the BNM species pages first (as v1), then the rest. F5 movie: the model turning noise into a map for one species (the sampler saves its steps). P1 Ecological Informatics. P4 one paper. R3 there is no v3; v2 is the final version (root causes beyond v2 are stated as limits and future work).
- 15 Nov (evening): project space: files verified on GitHub removed from the project (list in README.md).

- 15 Nov: storage. No project file is removed; old files and MD files updated from now on live in the private repo sarashahin/phd-obj2_v2-notes.
- 15 Nov: T is a report-only finding; K and S on hold; figures next.
- 15 Nov: the three detail questions of the T and S rules. B1 exact was T's only tie-breaker and T ran first (done); the tercile question belongs to S (on hold).
