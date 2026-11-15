# Open questions for Sara (15 Nov 2026)

Questions put to Sara in earlier sessions that she has not answered yet. Each has a plain explanation, why it matters, and Claude's suggestion. When she answers, the answer and date go beside the question and the item moves to "Answered".

## Needed before the figures start

| # | Question | Why it matters | Suggestion |
|---|---|---|---|
| F1 | Figure 2 shows true maps beside the model's samples. Which simulated communities: real_val (the development set) or real_eval (the clean test set) as a labelled display reading? | real_val chose the training checkpoint and fitted the level correction, so its maps flatter the model. real_eval is where the scores come from; showing its truth again is a third look, but no number is computed or changed. | real_eval, labelled display reading: truth md5-logged first, no score computed |
| F2 | Which species go in Figure 2: one species per recorded class (1, 2-5, 6-10, 11-20, 21-50), or the old v1 choice (5 random species of 12+ cells from one community)? | One per class shows how maps change with the number of records, which is the paper's grouping. A random seed fixed before any file opens stops picking a nice example. | one per recorded class, seed fixed first |
| F3 | Order of work: Figure 2 for Axel, then the paper figure set, then the BNM species pages, then the movies? | Axel asked for Figure 2 first; the rest reuse its code. | yes |

## Needed later, for the figures

| # | Question | Why it matters | Suggestion |
|---|---|---|---|
| F4 | BNM species pages (v1 step18 layout, EOO removed): which species get a page, and is phenology kept? | Phenology needs the record months, which may not be in the P1 export. | a page per species with a P4-3 product row; phenology only if the months are in the export (checked read-only first) |
| F5 | What do the 3D movies show: the IBM assembly in a few windows, the model turning noise into a map for one species, or both? | The assembly movie needs a trace option and reruns checked against the stored communities; the denoising movie needs one model run. | denoising first; the assembly movie only if Axel asks |

## Paper

| # | Question | Why it matters | Suggestion |
|---|---|---|---|
| P1 | Journal | Sets the word limit (Ecological Informatics: abstract 250, main text guideline 7,000, maximum 10,000, from V9_review; to recheck) | Ecological Informatics; Methods in Ecology and Evolution the other fit; a general-science journal not advised |
| P2 | One sentence in the supplement that EOO was planned, scored at G5d, failed and was dropped? | The reported gates then match the logs. | yes |
| P3 | B-0 (habitat shown against blanked) in the supplement? | It shows the BNM products' blank mode is calibrated and that habitat cannot fix the 1-cell excess. | yes |
| P4 | Axel's view on one paper | Sara asks Axel | - |

## Rare end (from 14 Nov, step 39)

| # | Question | Why it matters | Suggestion |
|---|---|---|---|
| R1 | How the product reads at 1 record: recorded AOO as the lower bound, the 90% interval, P(window AOO under 2.5 tetrads), a "not decidable" flag when the interval spans the cut-off, and the BNM test-half bias beside it (the sensitivity line Sara said yes to on 15 Nov; its exact rule is written before it is made) | One record cannot give a range map; the product should say what it can decide. | yes |
| R2 | Report the never-surveyed share of AOO in this paper (from the P4d columns, report only)? | About 0.58-0.60 of the land-only AOO at recorded classes 0-1 sits on land nobody surveyed, which no check covers. | yes |
| R3 | v3 (Objective 1 rare end, retraining) after this paper, with a clean real test set chosen before any v3 work? | Root causes of the rare-end limits sit in the training mix; reaching them is a new model version. | yes |

## On hold (Sara, 15 Nov)

K (more samples for sparse stacks) and S (level correction by predicted size, with its tercile question) wait. T showed the gap left is in holes the model orders strictly, which K cannot change; S fixes no root cause.

## Answered

- 15 Nov: storage. No project file is removed; old files and MD files updated from now on live in the private repo sarashahin/phd-obj2_v2-notes.
- 15 Nov: T is a report-only finding; K and S on hold; figures next.
- 15 Nov: the three detail questions of the T and S rules. B1 exact was T's only tie-breaker and T ran first (done); the tercile question belongs to S (on hold).
