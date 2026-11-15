# MetaDiffusion v10 (f1464525): review against v9 (4 Oct 2026)

Compared with v9 (206ee59f) by pandoc text, Word XML and figure md5: 20 changed paragraphs, 1 changed figure (rId40.pdf = Figure 4), no tracked changes. Abstract 247 words. Main text 8,742 words without captions and tables (v9 8,711, same count).

## D check
- Dispersal rate d: unchanged from v9 everywhere (Table 1 "Six parameters were varied", d = 2e-8, 5e-8, 1e-7; 2.2, Table 2 caption, 4.6 list six extrapolation communities incl. d = 1e-8 and 2e-7; 2.1 and S6 "rate d").
- KS distances: changed by the Figure 4 edits (abstract, 3.2, Figure 4 + caption, Table 2 row, 4.6). D scale sentence removed from 2.7 and Figure 7 caption. Verified: embedded Figure 4 prints D 0.295/0.076/0.056, means 17.4/17.0, 9.3/9.8, 2.30/2.29; in-distribution run 0.291/0.070/0.043 (largest change 0.013).
- Conflict: "25 (21 + 4)" exists because five files differing only in the d label are identical (C1: 30 files, 25 distinct; 204 training files, 117 distinct). Table 2 has "25 (21 + 4)" next to "6, extrapolation"; no sentence explains 30 to 25. Sara wants d kept as before; evidence (matrix stayed 2e-8 after override; list_world_params.py prints the label) does not support it. Fix is text only (V9_review D1 to D5).

## Points 1 to 5: retraining not needed for any
- 1: prevalence enters only via species_features (build_condition); NO_SPECIES_FEATS zeroes them (run_ablation_spatial.py); 3.5 reports recall 10.5% vs 10.4%, shapes D <= 0.09. Range size untested: mass check on saved ablation maps (FULL vs NO_SPECIES_FEATS hidden-cell mass; rule: median change within 5% and Spearman change < 0.02), heredoc tested on mock data. If it fails, drop "and in mean size" from abstract, contribution (iii), Discussion para 1.
- 2: rerun baseline_smoother_compare.py without the five copies (CPU).
- 3: 2.2 sentence "its parameter combination does not appear in the training pool" is false; replace with C1 result; run C2.
- 4: disclosed in 2.1, 3.5, 4.6, S6; only fixable by re-simulation.
- 5: a) label six as 4 + 2 or new sampling; b) paired ablation from saved maps; c) env-only reference = growth-rate field AUC 0.53 to 0.58 or one-variable GLM; d) length.

## Still open in v10
30 vs 25 unexplained (2.2, Supp. Figure 1, data availability, 3.3, Supp. Table 3 with 3,547); d listed as varied; false 2.2 sentence; 2.5 validation AUC on 20 files (16 copies of training communities), "204 communities" = 117 distinct; smoother on 30 files.
Slips: Discussion para 1 repeats "The reconstructions sit around those records,"; Figure 1 caption ends "are.]" and has "unchanged(Section"; S1 lacks chosen cut-offs (0.70/0.975 x16, 0.65 x7, 0.75 x1, 0.40 x1; 0.40 is the grid edge); T6 and the 4.6 Zurell edit not applied.
Submission: "E-mail: EMAIL" placeholder; no AI declaration; no highlights.
CPU runs before submission: Figure 4 with grid down to 0.30 and --references; clustering column and 3.3 ratios on 25 communities; smoother re-split; C2; ablation mass check; paired ablation.

Verdict: still major revision as written; main reasons are text contradictions needing no run.
