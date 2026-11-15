# MetaDiffusion v13 (b28dd153): check against v12 and the 5 Oct reruns

7 changed paragraphs vs v12: Figure 1 "]" removed; Table 2 "6 (4"; 3.2 mass-check sentence; Discussion duplicate removed; 4.6 pointer to 3.2; S6 space; EcoVAE reverted to "bioRxiv, ahead of print" (audit: DOI opens version 2; cite version 1, v1 URL). No AI declaration, no highlights. No figure changes.

Figure 7: Sara's Fig_ablation_distinct came from the ORIGINAL ablation_analyse.py ("No traits", "agreement threshold"); rerun with ablation_analyse_v2.py. Numbers right: full 10.2, no records 2.0 (CI 1.5-2.6), no network 10.3, no env 10.0, no features 10.6; chance 2.4; D no records 0.641/0.489; others max 0.084.

Paired differences (community bootstrap, 6 communities, 714 pairs): no records -8.19 (-9.79 to -6.64); network +0.08 (-1.94 to +1.98); env -0.17 (-1.95 to +1.93); features +0.40 (-1.62 to +2.31). Ablation set: 5 in-distribution + 1 extrapolation (p_LDD 0.3).

Clustering self-check: periodic Chebyshev + hypergeometric expectation reproduces the paper exactly (3,547; 7.07 2.17 1.11 0.88 0.82 0.78). Distinct (2,959 species): 7.14 2.18 1.11 0.87 0.81 0.78.

Smoother on 25 distinct (13 calib / 12 report), sigma 2.0. K=5 (1,447 species): KS size 0.540 vs 0.345, frag 0.187 vs 0.717, spread 0.166 vs 0.668; patches truth 4.41, diff 4.17, smoother 1.29; recall 15 vs 32%, near 18 vs 40%, far 0/0; AUC 0.701 vs 0.844; MCC 0.105 vs 0.202; J 0.142 vs 0.309; ENS 32 vs 32%; diversity 0.593. K=10 (313): KS size 0.403 vs 0.316 (smoother now closer; old text says the reverse), frag 0.157 vs 0.942, spread 0.080 vs 0.786; patches 8.80/8.54/1.61; recall 9 vs 22%; ENS 25 vs 22%.

Still to recompute for 3.6: mean range sizes (old 9.2/9.3), Spearman of patches (old 0.47/0.41), union size (old 49) and budget-matched recall (old 71% vs 36%); extra patch given with an "all" self-check mode. Permutation arms (3.5, second paragraph: 9.6/9.5/10.9) may include the two copies; check which files that run used.
