# MetaDiffusion v11 (34611757): review against v10 (4 Oct 2026)

13 changed paragraphs vs v10 (f1464525); no figure changes; no tracked changes. d corrections applied: abstract ("six held-out communities, four beyond"), 2.1 (d = 2e-8, same in every community), Table 1 (five varied; d one value; four extrapolation), 2.2 (rewritten: 30 files = 25 distinct; false parameter-combination sentence removed), Table 2 caption and rows, 3.1, 3.4, 4.6, data availability, S6 dispersal and assembly.

## Mass check (Sara's run, 8 ablation files)
955 species; hidden-cell mass median change -0.2% (IQR -5.0% to +5.3%), Wilcoxon p = 0.709; Spearman with true hidden size 0.097 (full) vs 0.164 (no features), change +0.067. Rule (|change| < 0.02) not met, direction opposite to a leak. Decision: drop "and in mean size" from abstract (252 -> 248 words), contribution (iii), Discussion para 1; 3.2 caveat to state measured result after dedup rerun.

## New finding (L1 resolved)
Ablation 8 files = 6 distinct communities: env2024 dr2em08_ld0p2 = dr5em08_ld0p2 (117 species each), env789 dr2em08_ld0p2 = dr5em08_ld0p2 (124 each); 955 species = 714 distinct. Rerun mass check and ablation stats dropping the dr2em08_ld0p2 copies (same as R2). Ablation set: 5 in-distribution + 1 extrapolation (p_LDD 0.3).

## L2 resolved
Wide-range supplementary figure (sp #22 N=29, #5 N=37, #8 N=32) matches the sigma_r^2 = 0.008 file (ls2p5_vr0p008_thr3p0_env789): genuine extrapolation; caption "one of the six" -> "one of the four".

## Still open in v11
Text: 2.2 add 117 distinct training communities; footnote 1 and Supp. Figure 1 caption (204 communities; 30 = 24 + 6); smoothing table caption ("six extrapolation"); Table 2 "6(4" spacing; S6 assembly duplicate seed sentence and "Table 1.The"; Figure 1 caption "unchanged(Section" and trailing "]"; 2.5 validation sentence (16 of 20 validation files are training communities); S1 chosen cut-offs; 4.6 add single dispersal rate limitation; E-mail placeholder; AI declaration; highlights.
CPU: smoother comparison on 25 distinct; 3.3 ratios and Supp. Table 3 clustering on 25; mass check and ablation on 6 distinct.

Verdict: still major as is (2.2 states duplicates exist while three analyses still use them); after text fixes and CPU reruns, reviewer recommendation would be minor revision. d correction changes no number; narrows scope (one dispersal rate, 117 distinct training communities).
