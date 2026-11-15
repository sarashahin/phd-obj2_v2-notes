# MetaDiffusion v15 (b9f5e4db): full read and submission check (5 Oct 2026)

Changes vs v14: 3.6 paragraphs 2-4 and pooling paragraph updated; S5.3 space; Supp. Table 3 values; Figure 7 replaced with the v2 figure (no threshold, "No species features"). Abstract 248 words; main text 8,911 words (Introduction to Data availability, no captions/tables); highlights 5, all <= 85 characters (Sara's check).

Still to fix before submission:
1. 4.2: "gives both methods the same mean predicted range size" is stale (3.6 now 9.0 and 9.4).
2. Permutation result (3.5 second paragraph 9.6/9.5/10.9; 4.3 "removes or permutes"; S5.3 two sentences): Sara has no files for that run, so the numbers cannot be traced; remove, with a caveat sentence in 3.5.
3. S5.4 table species (1,186 / 1,334 / 4,805 / 1,869 = 9,194) exceed the species that can have hidden cells at K = 5 in the evaluation set (3,547 with >= 6 cells in 30 files; 2,959 in 25); provenance check heredoc given.
4. AI declaration: Sara declines ("I don't use it"), but highlights and passages in 2.5, 3.2, 3.5, 3.6, S1, S5.3, 4.6 were drafted with Claude in these sessions; guide asks for the declaration before the references.
5. EcoVAE: export still "ahead of print" with DOI resolving to v2; check PDF.

Reviewer view: valuable as a careful simulation benchmark of what sparse records can recover, with honest limits; biggest weakness: practical value not shown (one stylized simulator; smoother better on pointwise recovery 32 vs 15%, AUC 0.84 vs 0.70, and on range size D 0.34 vs 0.54; shape advantage partly from random fragment placement, as 4.4 says; no real data; one trained model). Own recommendation after the fixes: minor revision; a reviewer weighting practical value could ask for major (real data, established SDM baseline, second training run).
