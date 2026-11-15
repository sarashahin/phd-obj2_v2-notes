# Paper plan for the v2 work (12 Nov 2026; updated 14 Nov: one paper chosen by Sara; 15 Nov: figures pulled ahead)

Status: one paper (option B), chosen by Sara on 14 Nov; the v1 MetaDiffusion manuscript was not submitted and waits for this version (Sara, 14 Nov). Axel's view still to be asked. Journal: open; suggested Ecological Informatics (below). 15 Nov: Sara pulled the figures ahead of T/S because Axel asked to see Figure 2 as in v1.

## Objective 4 claim (Sara, 13 Nov; project instructions updated 14 Nov: "we target AOO upgrading metrics")

Objective 4 is an upgrade of recorded-tetrad AOO, with stated limits. EOO is out of scope since 13 Nov. Evidence and limits: claude/obj2_v2_results_log_H_part3.md, last sections; rule text: claude/obj2_v2_H_rules_draft.md, last section.

## Structure: one paper (option B)

Simulated training data (Objective 1 v2, IBM only; short in the main text, full in the supplement), virtual survey from the BNM fit half, model and training with leak checks, 50 sampled maps to a corrected AOO with a 90% interval and P(AOO <= x), checks on unseen simulated communities (G5d window AOO, P4-2 land-only), the BNM hole test with the test half opened once, the land-only BNM products (P4-3, P4d). Limits stated: sparse map ranking below a distance smoother on BNM (H-R1), count level at 0-3 records on BNM, true 1-cell ranges 4.3-4.5 times too large on simulation, never-surveyed land unchecked (median 0.60 of the land-only AOO at recorded class 0), the simulation's rare end and similarity decay, T3, window AOO only. EOO: planned and scored at G5d under the signed G plan (failed), then dropped; suggested as one sentence in the supplement so the reported gates match the logs (Sara to decide).

Why one paper: the claim is one chain (process simulation, an observation layer read from a real atlas, calibrated AOO, judged on a real-atlas hold-out); a split would leave one paper with no real-data test and the other describing the model again. Objective 1 v2 alone would overlap the ODE/IBM/PSD bioRxiv preprint (10.64898/2026.04.21.719942) and the LVMCM papers. Cost: length; main text on the AOO chain, simulator detail, gates and baselines in the supplement.

## Against the unsubmitted v1

v1: the model ranked hidden cells below a records smoother on simulation (AUC 0.708 against 0.852); the audit found leaks (true range size among the inputs, shared validation communities, a hand-picked checkpoint, a 10-record cap) and no clean real-data hold-out. v2: above the smoother in every true class on 30 unseen communities; rules and leak gates written before each run; AOO calibrated on unseen communities and checked on a real atlas hold-out opened once; land-only products. v2 is the stronger and more defensible paper; v1 should not be submitted.

## Journal (advice given 14 Nov; Sara to decide)

Suggested: Ecological Informatics (v1's target; recorded in claude/V9_review.md: abstract up to 250 words, main text guideline 7,000 words, maximum 10,000; to be checked again in the journal's guide before drafting). Methods in Ecology and Evolution is the other fit (v1 introduction was first drafted for it). A general-science journal is not advised: the result is a calibrated upgrade with stated limits, and the real-data rare-species map test failed.

## Expected review points (my view as a reviewer, 14 Nov)

Major revision is the likely outcome; no paper of this kind should expect acceptance without revision. Points a reviewer will raise: (1) on BNM, maps of sparsely recorded species rank hidden records below a distance smoother, while the aim stresses rare species; the paper must say plainly that the gain on real data is the calibrated count and interval, not the map; (2) true AOO is never checked on real data, the products rest on calibration in simulated communities, and those communities miss BNM's rare end and similarity decay; (3) post-result choices (the land mask after the sea finding, the second labelled reading of real_eval, presentation lines) must be shown as labelled; (4) two seeds, one atlas, one taxon, one country, window AOO only; (5) never-surveyed land has no check; (6) planned analyses that failed (T3, EOO) must be reported. Strengths a reviewer will credit: rules written before results, leak tests with planted bugs, hold-outs opened once, every number recomputed.

## Figures (plan; 15 Nov, waiting on Sara's 3 questions, then one tool per turn with a sandbox test)

The old scripts are on Sara's disk (posterior_per_species.py for Figure 2; step18_bnm_profiles.py for the profiles). They read v1 files and will not run on v2 outputs, so we keep their layout and write v2 tools that read the v2 files. Reads only; no re-scoring; a real_eval truth panel is a labelled display reading (truth md5-logged first, no scores). All outputs are local files (PNG/PDF and HTML), not artifacts.

v2 data confirmed present: real_eval 50 maps per stack in samples_<run>.npz (stage_g5_predict), corrected AOO, interval and P in g5d_predicted.npz, records, class and blank in frozen_eval.npz, truth in the obj1 real_eval files; BNM fit-half samples in stage_h2_predict/samples_<run>.npz and test-half in H5; land-only products in P4-3 (products_land_bnm.npz) and P4d; the H6 group check in per_stack_h6/per_window_h6.

1. Figure 2 (old layout, v2 data), for Axel first. Rows = one species per recorded class (1, 2-5, 6-10, 11-20, 21-50). Columns: truth at 2019, records over 6 years, 4 of the 50 samples, pointwise posterior (mean of 50), corrected AOO with its 90% interval. Species drawn by a seed written before any file opens. Open Q1 (real_val or a labelled real_eval reading), Q2 (species by recorded class, or the old 5 random >= 12-cell species from one community).
2. Paper figure set from logged outputs only (one figure tool with a self-test, every number checked against the logs): pipeline schematic; Figure 2; ranking against the baselines by true class on real_eval; AOO ratio, coverage and width by recorded class (window G5d and land-only P4-2); P(AOO <= x) skill; BNM test half (H-R2 coverage and level, H-R1 by records); land-only products against P3 (P4d) with an example window; change through time (T1, T2, T3) in the supplement; Objective 1 figures from obj1_figures.py.
3. BNM species profile pages (HTML local files, step18 layout), after obj4. Per species: records map, effort, phenology if the record months are in the P1 export, posterior per window with the land mask, land-only AOO with its interval and P bands (CR 2.5, EN 125 tetrads), the H6 group check beside, the never-surveyed share; the EOO section removed; a methods page with the limits. Open Q3 (which species get a page; phenology kept or not).
4. 3D trace movies (content to be agreed): IBM assembly events in a few windows (needs a trace option and a rerun of those windows with the same seeds, checked to reproduce the stored community), and/or the model's denoising from noise to a map for one species.

## Open

- Journal choice; Axel's view on one paper; the EOO sentence; Figure 2 data (real_val, or a labelled real_eval reading); Figure 2 species choice; profile species scope and phenology; what the movies show.
