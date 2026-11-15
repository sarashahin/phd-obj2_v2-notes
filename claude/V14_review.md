# MetaDiffusion v14 (79a94659): check and 3.6 numbers (5 Oct 2026)

13 changed paragraphs vs v13 (all v13 items applied: abstract smoother numbers, Table 2 rows, 2.7, 3.3, 3.5, Figure 7 caption, 3.6 first and K=10 paragraphs, 4.2, 4.3, S5.3, Supp. Table 3 caption). Figure files unchanged (new Figure 7 not inserted yet).

Open: Supp. Table 3 clustering values still 30-file (7.07 2.17 1.11 0.88 0.82 0.78 -> 7.14 2.18 1.11 0.87 0.81 0.78); S5.3 "cells,with"; EcoVAE "ahead of print" (cite version 1); no AI declaration; no highlights; permutation paragraph (9.6/9.5/10.9) pending file check.

Figure 7: Fig_ablation_distinct from ablation_analyse_v2.py is correct ("No species features", no threshold; recall 10.2/2.0/10.3/10.0/10.6).

Smoother self-check ("all 30"): reproduces every old 3.6 number (sigma 1.5, 1,772 species, 31/15%, near 39/18, KS 0.50/0.34, 0.18/0.65, 0.15/0.55, patches 4.55/4.29/1.64, AUC 0.70/0.84, MCC 0.10/0.20, J 0.14/0.30, ENS 31%, diversity 0.60, mean range 9.28/9.20/9.20, Spearman 0.466/0.408) EXCEPT union size (35.5 vs paper 49) and matched-budget recall (smoother 67% vs union 31% against paper 71% vs 36%). Old union numbers not reproducible by the defined method; replaced with the defined method's 25-community values.

25 distinct, K=5: KS size 0.5404 vs 0.3449; mean range truth 9.28, diffusion 8.97, smoother 9.42; Spearman patches 0.493 vs 0.319; union 34.0 cells; matched-budget recall smoother 60.3% vs union 31.7% (pooled 42.9% vs 22.8%). LaTeX for 3.6 paragraphs 2-4 and 6 written.
