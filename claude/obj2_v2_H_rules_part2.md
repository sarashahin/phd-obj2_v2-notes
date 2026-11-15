# Objective 2 v2, stage H rules, part 2 (from 15 Nov 2026: rare-end checks T and S)

Part 1 (claude/obj2_v2_H_rules_draft.md) ends with "Habitat checks B". It reached the size where a full rewrite per change is error-prone, so new rules go here. Nothing in part 1 changes.

## Rare-end checks without retraining (Sara, 15 Nov: yes to T and S now, K only if T leaves a gap, yes to the sensitivity line)

Status: rules written 15 Nov before any T or S value was read. Q1-Q3 below: Sara's "continue" (15 Nov) read as yes to the suggested answers; she is asked to confirm. T run by Sara 15 Nov: rule NO (results log part 4, step 47). K: the gap T leaves is in strictly ordered holes, so K is not suggested; Sara to decide. S not built; suggested not needed for the paper.

Why (results log part 4, steps 44-45): B-0 showed the habitat cannot shrink the true 1-cell excess. Sara asked for options without retraining. T aims at the BNM map ranking for 20 or fewer records (H-R1). S aims at the true 1-cell excess of the land-only AOO (P4-2: 4.3-4.5 times). Neither touches the fit-to-test count gap at 0-3 records.

Both are report only. The model, its samples, h3_freeze.json, the P4 fits and every signed product stay as they are.

### T, tie-break the model's hole ranking with B1 exact

- Scope: BNM fit half only (development data). Inputs: H2b's files (h2b_predicted.npz, h2b_scores.npz, h2b_report.json), the fit-half truth files through H2b's read_labels, H2a's baselines. Nothing of the test half, H5, H6 or real_eval is loaded (np.load refused, as F0).
- Score: within each stack-repeat, hole tetrads are ordered by the model's score first; tetrads with the same model score are ordered by the B1 exact score. Tetrads equal on both stay tied. Nothing is fitted and there is no parameter, so no folds are needed.
- How it is computed: score_T = model score + r / (2 x 50), with r the rank of the B1 exact score among the stack-repeat's holes scaled to [0, 1) (0-based, ties share their mean rank, divided by the number of holes). Gate: every model score x 50 is a whole number (tolerance 1e-5, see the correction below), so the smallest gap between two model scores is 1/50 and the added term (below 1/100) never changes the model's own order. If the gate fails the tool stops and the table goes to Sara.
- Correction found in the sandbox before any T value (15 Nov, labelled): the tolerance was written as 1e-6. H2b stores the model score as float32, and float32(k/50) x 50 differs from k by up to 1.43e-6 (k = 36), so a 1e-6 gate would stop the tool on correct data. Tolerance set to 1e-5; the smallest stored gap between two model scores is 0.0199999, still twice the largest added term. The order gate (below) checks the result on every stack-repeat either way.
- Rule, the F0 shape: "T worth using" = YES when, for seed 1 and seed 2, against B1 stored and B1 exact, with H2b's H-R1 code (stack-first means, 2,000 resamples of the 44 windows, seed 31, floor 50 distinct stacks): (i) no group of either grouping FAILs, and (ii) recorded classes 21-50 and 51+ and records 21+ keep a mean of 0 or above. Otherwise NO. Sara decides either way.
- Reported beside it: the unchanged H2b rows; per group the stack-first mean of AUC(T) minus AUC(model) with its interval; the share of recorded-against-unrecorded hole pairs the model ties, per group (the room T has); the sparse set S of H3b pooled (kept recorded class 0-1 in 2019, or 1-10 kept records over 6 years).
- Code gates (tool names V0-V4): V0 inputs = the H3 freeze list, stage_h6_bnm_test_score.py 8b9a38dd, stage_f0_spread_check.py version 2 by content (its loader, H-R1 closure and np.load log are reused), H2b's report gates; V1 alignment as F0's A1, the whole-number gate, B1 exact never NaN at a hole; V2 with the added term set to 0 the H-R1 rows = h2b_report.json (1e-9, same verdicts); V3 the order of any two holes with different model scores is the same before and after and they never tie (every stack-repeat), AUC(T) = AUC(model) on every stack-repeat with no tied recorded / unrecorded pair, and with B1 exact replaced by a constant AUC(T) = AUC(model) exactly; V4 np.load log.
- What a YES means: development evidence only. The fit half was used for every earlier choice and the test-half labels were read once at H6. A test-half score of T is a labelled second look and needs Sara's yes after the fit-half result. T changes the order of cells in a map; it changes no AOO, interval or P in the P3 or P4 products.
- Known limit, stated now: against B1 exact, T can gain only on pairs the model ties, and there it copies B1 exact. So T can lift the model to about B1's level in the sparse groups, and above it only where the model's strict ordering already beats B1.

### S, level correction by recorded class and the model's own raw size

- Scope: real_val only (development data). Inputs: P4-1's p4_1_crossfit.npz (371f424a) and p4_1_report.json (5e980763). No truth file, nothing of real_eval or BNM.
- Size of a stack = the mean land-only AOO over the method's own maps, before any correction (from the maps only; no truth).
- Grouping "option 3": recorded class x size bin, with the recorded class as the parent group. G3's crossfit, evaluate and rule_pass are used unchanged, on G3's community folds, with G3's fallback (fewer than 30 fit stacks: the recorded class, then all).
- Size bins (Q1, read as yes): 3 bins per recorded class, cut at the terciles of the size. For the stacks of a check fold the cut points come from the other fold's stacks, so no number of the check fold enters its own grouping.
- Methods: seed 1, seed 2, pooled (the product line), and B3, B4, B5 beside them.
- Rule: "S worth a labelled reading of real_eval" = YES when, for pooled and for both seeds, out of fold: (i) G3's rule passes (ratio 0.8-1.25 and coverage 0.85-0.95 in every recorded class and records bin with 50 or more stacks); and (ii) the ratio at land-only true class 1 under option 3 minus the same ratio under P4-1's stored cross-fit has its interval below 0 (2,000 resamples of the 20 communities, numpy seed 31, the same draws for both). Otherwise NO. Sara decides either way.
- Reported beside it: rows by size bin; ratio and coverage at true class 1 and 2-5; median interval width by recorded class against P4-1's; mean G3 P(AOO <= 1) and P(AOO <= 5) from the corrected maps at true class 1, labelled "before recalibration"; the rank correlation of size and true land-only AOO within each recorded class; crossed intervals counted (rule E).
- Code gates: inputs = Sara's P4-1 run; with one size bin per class the tool's cross-fit = P4-1's stored mean, L, U for every method at option 1 (1e-9); the check fold's cut points unchanged when that fold's sizes are replaced by random values; each fold's corrected values unchanged when that fold's truth is shuffled; np.load log (only the one npz).
- Correction (my error, step 45): I wrote that S helps only if true AOO grows less than in proportion to the raw size within a class. That is one-sided. S changes nothing when truth is proportional to size within a class; when it is not, the scale differs by bin, and whether the true 1-cell ratio falls depends on which bins the truly 1-cell stacks sit in. The direction is not known in advance; (ii) measures it.
- What a YES means: a candidate only. real_eval has been read twice, so a reading with option 3 would be a labelled third reading with its fits md5-logged first, and any products would be a new labelled version. P4 stays as signed.

### K and the sensitivity line

- K (100 samples for sparsely recorded stacks) runs only if T's result leaves a gap (signed 15 Nov). Its rule is written here before it runs.
- Sensitivity line (signed 15 Nov): shown beside the products at recorded classes 0-1, labelled as an assumption and not a product. Its exact rule (which test-half ratio divides which row) is written here in full before it is made.

### Questions put to Sara 15 Nov (her "continue" read as yes to each suggestion; to be confirmed)

- Q1: S with 3 size bins per recorded class (terciles, cut points from the other fold). Suggested yes: at least 250 fit stacks per bin in every class, and one fixed choice means no picking among bin counts after results.
- Q2: T's tie-breaker is B1 exact only. Suggested yes: one fixed choice; B1 stored is only a reference in the rule.
- Q3: order T first, then S, one tool per step, each tested on planted data before delivery. Suggested yes.
