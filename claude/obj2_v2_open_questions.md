# Open questions for Sara (15 Nov 2026)

Questions put to Sara in earlier sessions that she has not answered yet. Each has a plain explanation, why it matters, and Claude's suggestion. When she answers, the answer and date go beside the question and the item moves to "Answered".

## Open now (15 Nov, late)

| # | Question | Why it matters | Suggestion |
|---|---|---|---|
| G1 | (asked again with a plain explanation) The five real_eval rows are simulation only. In half of the simulated evaluation stacks the model was given the habitat layer, in half it was blank. Draw the rows from the blank-habitat stacks? | The BNM row and every BNM product use the blank input, so blank rows compare like with like; shown-habitat rows would show the model with information it never gets on BNM. | blank |
| H1 | Run habitat check B-1 (does the UKCEH Land Cover Map, as one habitat axis, carry the signal the model was trained on?) before Figure 2? | Sara wants the missing BNM habitat fixed first if a reliable, unbiased fix exists; B-1 is CPU only, uses the fit half and files already on her machine, and stops the habitat route early if the signal is weak. Figure 2's simulation rows do not depend on it. | yes, B-1 first, plan as written in results log part 4 step 41 |
| S1 | Make project space: remove project copies of code files now in the repo from her machine (her run versions; 10 of 10 spot-checked md5s equal the md5s logged at her runs) | The project is at about 89%; code is most of it; the repo keeps every version. | yes, after a full listing of what would go |
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

- 15 Nov (late): G2 yes (BNM row drawn by the fixed seed from species with 2-5 records in a coastal window). G3 yes (new tools through the GitHub repo). R1 options (a) and (b): print the test-half bias beside the product at 0-3 records, and an adjusted AOO = AOO / ratio labelled as an assumption. Sara pushed her code folders to the repo (commit 462314a; obj1_v2 tools, tests, new_code; obj2_v2 tools, tests).

- 15 Nov (evening): F1 real_eval, labelled "display reading 3, no scores", species drawn by a seed fixed before any file opens, plus one BNM row (records, samples, posterior, land mask). F2 one species per recorded class. F3 Figure 2 and the BNM species pages first (as v1), then the rest. F5 movie: the model turning noise into a map for one species (the sampler saves its steps). P1 Ecological Informatics. P4 one paper. R3 there is no v3; v2 is the final version (root causes beyond v2 are stated as limits and future work).
- 15 Nov (evening): project space: files verified on GitHub removed from the project (list in README.md).

- 15 Nov: storage. No project file is removed; old files and MD files updated from now on live in the private repo sarashahin/phd-obj2_v2-notes.
- 15 Nov: T is a report-only finding; K and S on hold; figures next.
- 15 Nov: the three detail questions of the T and S rules. B1 exact was T's only tie-breaker and T ran first (done); the tercile question belongs to S (on hold).
