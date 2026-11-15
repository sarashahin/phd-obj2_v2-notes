# Open questions for Sara (15 Nov 2026)

Questions put to Sara in earlier sessions that she has not answered yet. Each has a plain explanation, why it matters, and Claude's suggestion. When she answers, the answer and date go beside the question and the item moves to "Answered".

## Needed before Figure 2 is drafted (asked 15 Nov, evening)

| # | Question | Why it matters | Suggestion |
|---|---|---|---|
| G1 | Draw the five real_eval rows only from habitat-blanked stacks? | Every BNM product and the BNM row use the blank habitat input; the same input on every row keeps the rows comparable. | yes |
| G2 | Which species-window is the BNM row: drawn by the same fixed seed among rows with recorded class 2-5 in coastal windows? | A coastal window shows the land mask; class 2-5 is the sparse end the paper is about and still gives a readable map. | yes |
| G3 | New tools and tests delivered through the GitHub repo (git pull) instead of the project? | The project grows by about 30-40 KB per step and is near its limit. | yes |

## Asked again after the plain explanation (15 Nov, evening)

| # | Question | Suggestion |
|---|---|---|
| P3 | B-0 (habitat shown against blanked, on simulation) in the supplement | yes: it shows the BNM products' blank mode is calibrated and that habitat cannot fix the 1-cell AOO excess |
| R1 | Beside the product at 0-3 records: (a) print the test-half bias (counts ran 1.25-1.56 times too high), (b) also an adjusted AOO = AOO / that ratio, labelled an assumption, (c) neither | (a) and (b) |
| R2 | Report the share of each species' land-only AOO that lies on tetrads nobody surveyed (from P4d, no new run) | yes |

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

- 15 Nov (evening): F1 real_eval, labelled "display reading 3, no scores", species drawn by a seed fixed before any file opens, plus one BNM row (records, samples, posterior, land mask). F2 one species per recorded class. F3 Figure 2 and the BNM species pages first (as v1), then the rest. F5 movie: the model turning noise into a map for one species (the sampler saves its steps). P1 Ecological Informatics. P4 one paper. R3 there is no v3; v2 is the final version (root causes beyond v2 are stated as limits and future work).
- 15 Nov (evening): project space: files verified on GitHub removed from the project (list in README.md).

- 15 Nov: storage. No project file is removed; old files and MD files updated from now on live in the private repo sarashahin/phd-obj2_v2-notes.
- 15 Nov: T is a report-only finding; K and S on hold; figures next.
- 15 Nov: the three detail questions of the T and S rules. B1 exact was T's only tie-breaker and T ran first (done); the tercile question belongs to S (on hold).
