# MetaDiffusion v12 (fb996945): check against v11 (4 Oct 2026)

16 changed paragraphs vs v11 (34611757); no figure changes; no tracked changes. Applied: abstract without "and in mean size" (248 words), contribution (iii), 2.2 "204 simulation files, holding 117 distinct communities", Figure 1 "unchanged (Section", 2.5 validation sentence, 4.6 single dispersal rate clause, S1 chosen cut-offs, S6 duplicate seed sentence removed, Supp. Figure 1 caption, footnote 1, wide-range caption ("four"), smoothing table caption, e-mail filled, IUCN 2012 "2nd ed.", EcoVAE "version 1".

Not applied / new slips: Discussion para 1 still doubled ("records,The reconstructions sit around those records,"); S6 "reached.The"; Figure 1 caption ends "are.]"; Table 2 "6(4" twice; EcoVAE title now italic (book style); no AI declaration; no highlights.

Mass check on 6 distinct communities (714 species): median change +0.1% (IQR -4.9% to +5.4%), Wilcoxon p = 0.854; Spearman with true hidden size 0.119 (full) vs 0.173 (no features). 3.2 caveat LaTeX written with these numbers.

Ablation CSV filtered: 4775 -> 3570 rows (714 x 5 arms), 8 -> 6 communities: figures_map_axel_stage2_new/unseen_eval/ablation/ablation_per_species_distinct.csv.

Figure 7 (rId55, unchanged since V5) still says "No traits" and draws an "agreement threshold" at 0.30 that 2.7 no longer defines; patch for ablation_analyse.py written and tested (label "No species features", line removed, numbers identical).

Heredocs given (all tested on mocks): ablation numbers with paired community bootstrap (V5 item 5); smoother patch (drops R2 files, 13/12 split, prints mean patches, writes _distinct CSV); clustering recompute for 3.3 and Supp. Table 3 with self-check against the paper's 30-file values (3,547 species; 7.07 2.17 1.11 0.88 0.82 0.78) over open/periodic distance and binomial/hypergeometric expectation.

Next: Sara pastes the four outputs; then LaTeX for 3.3, 3.5, 3.6, 4.2, abstract smoother numbers, Table 2 rows, Figure 7 caption, S5.3, Supp. Table 3.
