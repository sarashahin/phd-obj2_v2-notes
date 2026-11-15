# MetaDiffusion v16 (57f91ba2): review vs v15 (5 Oct 2026)

One change vs v15: Section 4.2 range-size sentence now reads "similar mean predicted range sizes (9.0 and 9.4 cells, against a true mean of 9.3), though per-species counts differ." No figure changes. rId55 = Figure 7 (deduplicated ablation, 714 species, 6 distinct communities).

## Verdict (held consistent, no flip-flop)
Recommendation after the remaining small items: minor revision. The manuscript read as major only while the text contradicted itself (30 vs 25, d listed as varied, false 2.2 parameter-combination sentence, three analyses run on duplicate files); those were fixed across v11 to v15. The residual "major" is editorial risk, not a defect in what she has: a reviewer weighting practical value could still ask for a real-data case, an established SDM baseline, or a second training seed. Keep my call (minor) separate from that risk.

## S5.4 species column (open, fix given)
Sara's check: S5.4 "Species" = 1,186 / 1,334 / 4,805 / 1,869 (total 9,194). Eligible species with true range > K=5 (>=6 cells): all-30 = 493/598/2,328/128 (3,547); distinct-25 = 401/598/1,832/128 (2,959). S5.4 exceeds the eligible set at every p_LDD (0.3: 1,869 vs 128), so the column is not the Section 2.3 set and still counts duplicate files. Cannot be relabelled to a known quantity from the numbers alone.
recon npz stores noisy_input (the exact revealed records), so the far-cell set recomputes with no random seed. Fix: drop the "Species" column, report the eligible deduplicated count (metric-independent, matches 2.3), keep far recall 0.1-0.2% (robust, seed-free on the saved mean). Heredoc verified on mock: dedup 30 to 25, group by _ld, print eligible / far-cell species / recall (mean and per-sample) / median 4-neighbour patches for range>10. Far distance assumed periodic Chebyshev > 2 (clustering convention); confirm against the committed S5.4 script if present.

## Permutation arms (keep as is, scope honestly)
3.5 permutation full-model recall = 10.9% (CI 9.5-12.2) vs deduplicated removal full = 10.2%: a different, earlier run on the eight files. S5.3 sentence "every result uses the six distinct communities" is false for the permutation arms. Edit (LaTeX): replace that clause with "so the removal arms are reported for the six distinct communities; the permutation arms are from an earlier run (Section~\ref{sec:ablation})." Numbers untouched. 3.5 main text can stay. Data availability (line 310) is about code + 30 eval files + weights; it holds for the permutation arms if run_ablation_spatial.py commits the two permutation modes (confirm PERMUTE env/feature modes are in the repo).

## Other
EcoVAE reference may still show "ahead of print" in the PDF (check). AI declaration: Sara adds at the last minute, ignore for now.
