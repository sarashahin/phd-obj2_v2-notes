# MetaDiffusion v9 (206ee59f): status, LaTeX fixes, new code findings (4 Oct 2026)

Compared with V6 by Word XML and pandoc text; figures compared by md5. Supplementary items use PDF numbering (Methods S1 to S8, Figures S1 to S9). Every search phrase below occurs once in v9.

## Verdict

Still major revision. v9 fixed most paste errors, but no analysis item changed, and the training/validation split adds a new one. Every result still uses 30 evaluation files that hold 25 distinct communities, Table 1 still lists d as varied, and the Figure 4 thresholds still read the evaluated communities' truth.

## What v9 fixed (11 changes against V6)

IUCN comma; doubled Hutchinson/Pulliam citation; "Second"/"Third" limitations restored; doubled Grimm/Cabral sentence; 3.1 and 4.3 fragments; "truth-free" removed in 2.7, 4.6 and S7 (3.2 keeps "One input is not truth-free", correctly); isotonic paragraph and Zadrozny reference restored. Half done: Figure 1 caption full stop (now inside the citation bracket), S7 sentence (still doubled). All 16 figure files are byte-identical to V5.

## 1. Text errors left

T1 Introduction, paragraph 1. Search `could ever be recorded; so there are few places`. Replace the sentence:

```latex
Many species are genuinely range-restricted, by dispersal limitation, habitat availability or local extinction, so there are few places where they could ever be recorded \citep{Gaston2003}; and survey effort is limited and unevenly spread across geography \citep{Anderson2012}, so many occupied places are never visited.
```

T2 Figure 1 caption. `leaves performance unchanged(` to `leaves performance unchanged (`. Replace from `and sampled in` to the end of the caption:

```latex
and sampled in $50$ DDIM steps \citep{Song2021}; the stochasticity parameter $\eta = 0.15$ sets how much randomness is kept between samples, and so how different the ensemble members are.
```

T3 Methods S7, From samples to a range. Search `highest-valued cells by partial sort the`. Replace the sentence:

```latex
To obtain a binary range where a species' size is held fixed, we take the $N$ highest-valued cells by partial sort, with $N$ the true range size, either of the mean or of each sample separately (as in the coverage analysis); this is used only where range size must be controlled, and is stated in each case.
```

T4 Introduction, paragraph 4. Search `In a simulation the ground-truth range`. Replace the last three sentences of the paragraph:

```latex
In a simulation the true range of every species is known. This is the virtual ecologist approach \citep{Zurell2010}: a simulated system and an explicit observer model produce the data, and a method is scored against the simulated truth. Simulation also lets us train on synthetic communities under a controlled observation process and measure how much of a community's spatial structure a handful of records can recover.
```

Section 4.6: `which the virtual ecologist approach supplies and field data cannot (...)` ends at `which the virtual ecologist approach supplies \citep{Zurell2010}`.

T5 Contribution (iii). Search `so the method needs no environmental layer or interaction matrix to be applied`. Replace with:

```latex
so in these simulations, where the environment is weak by design, the method ran as well without an environmental layer or interaction matrix.
```

T6 Section 3.5. After `so there is little for it to carry.` add:

```latex
It was also supplied on a different scale from training (Supplementary Methods~\ref{smeth:model}).
```

## 2. Dispersal corrections that are final now

D1 Table 1. Move the d row from the varied block to the fixed block:

```latex
Local dispersal rate & $d$ & $2\times10^{-8}$ & Spread rate to adjacent cells \\
```

Caption text:

```latex
Simulation parameters. Five parameters were varied across communities; the rest were fixed. Training and evaluation used disjoint sets of random seeds for the environmental fields, so no evaluation community reuses a training field; the seed values are recorded in the released code. Values listed for the varied parameters are those used in training; the four extrapolation communities each take one parameter beyond this set. The attempt cap is tested after each round, so the realised total slightly exceeds it.
```

D2 Section 2.1. Search `four neighbours at rate`. Replace the sentence:

```latex
Biomass spreads to a patch's four neighbours at rate $d = 2\times10^{-8}$ per time step, the same in every community.
```

D3 Methods S6. Search `scaled by the dispersal rate`; make the clause:

```latex
scaled by the dispersal rate $d = 2\times10^{-8}$, the same in every community (the dispersal-rate label in the released file names was not applied by the simulator).
```

After the sentence starting `Rounds continue until the cap on invasion attempts is reached` add:

```latex
Assembly and the forward run used the same random-number seed in every community, so communities differ through their environmental fields and the parameters in Table~\ref{tab:sim}.
```

(run_all_rps.py passes seed=1 to stepwise_assembly_ibm and to IBMModel.)

D4 Relabel the six-community analyses; numbers unchanged. The six are the four extrapolation communities plus two copies of in-distribution ones (d = 2e-7 file = l 10, sigma_r^2 0.002, tau 1, p_LDD 0.2, seed 789; d = 1e-8 file = l 5, sigma_r^2 0.002, tau 1, p_LDD 0, seed 789).

Table 2 caption, search `The six extrapolation communities each carry one parameter outside`:

```latex
Four held-out communities each carry one parameter outside the values in Table~\ref{tab:sim}: correlation length $\ell = 1$ or $20$ (trained: $2.5$--$10$), environmental variance $\sigma_{r}^{2} = 0.008$ (trained: to $0.004$), or long-distance dispersal $p_{LDD} = 0.3$ (trained: to $0.2$). The recall and coverage analyses use these four and two in-distribution communities.
```

Table 2 rows "Recall against records" and "Ensemble coverage": `6, extrapolation` to `6 (4 extrap.\ + 2 in-dist.)`.
Abstract: `In six communities beyond the trained parameters,` to `In six held-out communities, four beyond the trained parameters,` (247 words).
Section 3.1: `Recall pooled over the six extrapolation communities` to `Recall pooled over six held-out communities`.
Section 3.4: `species from the six extrapolation communities` to `species from six held-out communities (four beyond the trained parameters, two within)`.
Section 4.6, search `The recall and coverage results come from the six communities carrying`; replace up to the semicolon:

```latex
The recall and coverage results come from six held-out communities, four with a parameter outside the trained ranges and two within them;
```

Smoothing table caption: `in the six extrapolation communities, fifty samples each` to `in six held-out communities, fifty samples each`.
Table 2 row "wide-ranging species" and Figures S5/S6: keep "extrapolation" only if that community is not the d = 2e-7 or d = 1e-8 file (lookup L2).

D5 Section 2.2, replace the paragraph:

```latex
Training used $204$ simulation files drawn from the parameter values listed as varied in Table~\ref{tab:sim}. All results are on held-out simulations generated at two environmental-field seeds not used in training. These $30$ files hold $25$ distinct communities, because five pairs of files differ only in a dispersal-rate label that the simulator did not apply (Supplementary Methods~\ref{smeth:sim}). Of the $25$, $21$ take only parameter values listed in Table~\ref{tab:sim}, and $4$ carry one value outside them ($\ell = 1$, $\ell = 20$, $\sigma_{r}^{2} = 0.008$ or $p_{LDD} = 0.3$), which separates in-distribution performance from extrapolation. No evaluation community shares its environmental-field seed with a training community.
```

After C1, add the distinct count of the 204 training files; after C2, say how many in-distribution communities share all other parameter values with a training community.

Data availability: `together with the $30$ held-out evaluation communities` to `together with the $30$ evaluation simulation files`, then add:

```latex
File names carry a dispersal-rate label that the simulator did not apply: every community was run at $d = 2\times10^{-8}$, and files that differ only in that label hold identical communities.
```

## 3. New problems in the code

N1 Validation split. data_preprocessing.py (create_dataloaders) permutes the valid `*_training.npz` files with seed 42 and cuts 0.833/0.083/0.084. C1 (run 4 Oct): 245 valid files; train 204 (117 distinct communities), validation 20 (16 identical to a training community), test 21 (17 identical); the 30 evaluation files hold 25 distinct communities, none identical to a training community. Sara: the 20 validation files were never used for inference. So N1 affects only the Section 2.5 sentence reporting validation AUC (0.99 and 0.94); the results are unaffected.

N3 Seeds (code read 4 Oct). sweep_worlds_eval30.py separates the environmental-field seed only (789/2024 vs 123/456). run_all_rps.py fixes seed=1 for assembly (invader interactions, founding patches) and for the forward run; each assembly window builds IBMModel without a seed, so np.random.seed(0) restarts the demographic stream every window. Worlds with the same l, sigma_r^2, tau and p_LDD differ only through their environmental fields; C2 measures how far apart they are (not yet run).

N4 Field follows list position, not species. generate_spatial_r(seed=seed_field) draws field i for position i, and IBMModel rebuilds the fields at every assembly window; after pruning, species move to lower positions and receive another field. The saved ENV_r_field is the final forward run's assignment. Disclose in 2.1/S6; fix for new runs: draw each species' field once at invasion and carry it with r and C.

Figure 4 script v2 (4 Oct): claude/axel_ecological_distribution_figure.py removes the five copies, uses a leak-free nested router (--router legacy reproduces 759/6,072/0.139/0.229/0.102) and draws at exactly 190 x 125 mm.

N2 Section 2.2 claim and fixed seed. Section 2.2 says no evaluation community's parameter combination appears in the training pool. The training folder holds pool22510000_batcha_ls2p5_vr0p004_thr1p0_env123_grid20x20_dr2em08_ld0p0_training.npz; the evaluation set holds the same l, sigma_r^2, tau, d label and p_LDD at seed 789. sweep_worlds_missing5.py generated the hardband, connectivity and wide-range combinations at the training seeds. With seed=1 for every run, a reviewer will ask whether in-distribution evaluation communities are near-copies of their training twins. C2 answers it.

## 4. Reruns, in order

R0 V5 item 1 (inference inputs differ from training). If you rerun inference with matched inputs, do it first; every number changes.
R1 Router: fit the two cut-offs and the wide-range share on calibration communities only; then Figure 4.
R2 Deduplicate the evaluation set; recompute Figure 4 and Section 3.2 (759 species becomes 624), the in-distribution restriction (21 communities), the distance-table clustering column and Section 3.3 (3,547 becomes 2,959), and the abstract's first result sentence. Drop:
- pool22510000_batcha_ls10p0_vr0p001_thr1p0_env789_grid20x20_dr2em08_ld0p2 (= ...dr5em08...)
- pool22510000_batcha_ls10p0_vr0p001_thr1p0_env2024_grid20x20_dr2em08_ld0p2 (= ...dr5em08...)
- pool22510000_batcha_ls5p0_vr0p002_thr1p0_env789_grid20x20_dr2em08_ld0p2 (= ...dr1em07...)
- pool22510000_batcha_ls10p0_vr0p002_thr1p0_env789_grid20x20_dr2em07_ld0p2 (= ...dr1em07...)
- pool22510000_batcha_ls5p0_vr0p002_thr1p0_env789_grid20x20_dr1em08_ld0p0 (= ...dr2em08...)
R3 Smoother comparison: split the 25 distinct communities with none in both halves; retune sigma, refit the isotonic calibrator; recompute 3.6, 4.2 and the abstract's smoother numbers.
R4 After C1/C2: training count in 2.2, footnote 1, Figure S1 and caption, Section 2.5 validation sentence.
R5 Still open from V5: Figure 4 at K = 5 (or "ranges above ten cells" in the abstract); recall and coverage on the 21 in-distribution communities; paired ablation differences with a community bootstrap; an environment-only baseline; length (main text about 8,360 words, guideline 7,000, maximum 10,000); a source or a reference value for the D scale in 2.7.

Lookups: L1 the eight ablation communities (--truth-npz values in run_ablation_spatial.py runs); L2 the community behind Figures S5/S6; L3 checks B (S4 species counts) and C (union recall 31% vs 36%) from the V5 review.

## 5. Submission items

AI declaration (section title and wording from the Ecological Informatics guide, checked 4 Oct; place before the references):

```latex
\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}
During the preparation of this work the authors used Claude (Anthropic) in order to check the text, references and analysis code. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
```

Highlights (3 to 5 bullets, 85 characters including spaces; lengths 85, 82, 80, 71, 83):
- A conditional diffusion model returns ensembles of ranges from a few presence records
- Trained and tested on metacommunity simulations in which every true range is known
- Cells within two steps of a record are found above chance; farther cells are not
- Ensembles bracket the true range shape more often as records accumulate
- Environment and interaction inputs added nothing measurable once records were given

E-mail: title block still reads "E-mail: EMAIL".

Figures: all 16 are the V5 files. Regenerate and insert Figure 4 after R1/R2; redraw Figure 1 (V5 list); redo Figure S1 with the new counts (it still prints "250 two seeds"); fix "1th"/"2th" in Figures 3 and S6.

BibTeX (verified 3 Oct):

```bibtex
% add to existing entries
%   BeerySDM2021    doi = {10.1145/3460112.3471966}
%   Hutchinson1957  doi = {10.1101/SQB.1957.022.01.039}
%   Pulliam2000     doi = {10.1046/j.1461-0248.2000.00143.x}
%   Youden1950      doi = {10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3}
%   Lantuejoul2002  doi = {10.1007/978-3-662-04808-5}
%   CISO            number = {3}
%   Matthews1975    journal = {Biochimica {et} Biophysica Acta (BBA) - Protein Structure}

@misc{EcoVAE,
  author       = {Yan, Yujing and Shao, Bin and Davis, Charles C.},
  title        = {A generative deep learning approach for global species distribution prediction},
  year         = {2024},
  howpublished = {bioRxiv preprint, version 1},
  doi          = {10.1101/2024.12.10.627845},
  url          = {https://www.biorxiv.org/content/10.1101/2024.12.10.627845v1}}

@book{IUCN2012,
  author    = {{IUCN}},
  title     = {{IUCN} Red List Categories and Criteria: Version 3.1},
  edition   = {2nd},
  publisher = {IUCN},
  address   = {Gland, Switzerland and Cambridge, UK},
  year      = {2012},
  isbn      = {978-2-8317-1435-6}}
```

## Checks run for this review

Diff V6 to v9: 11 hunks, listed above; all 16 figure md5s equal V5. Abstract 245 words (247 after D4). Every search phrase occurs once in the Word text. C1 tested on a mock folder: its header-based split equals a full-load reproduction of create_dataloaders, and its counts match an independent recount. C2 tested on mock twins: near-copy gave same degrees 1.000 and Jaccard 0.905 vs 0.027 shuffled; an independent twin gave 0.000 and 0.026 vs 0.026. Not checked: anything that needs Sara's files.

## 6. Figure 4 after the v2 runs (4 Oct)

Runs read from Sara's figures. Regression (legacy router, 30 files): D 0.139/0.229/0.102, means 17.4/16.8, 9.3/10.6, 2.30/2.40, identical to the old figure. v2_all (copies removed, nested router, 25 communities): D size 0.295, patches 0.076, spread 0.056; means 17.4/17.0, 9.3/9.8, 2.30/2.29. v2_indist (21 communities): 0.291/0.070/0.043; means 17.4/16.4, 9.1/9.6, 2.28/2.27.

Decision: v2_all is Figure 4; v2_indist only for the in-distribution sentence; regression run was a check only. Two things visible in v2_all: the two cut-offs split reconstructed sizes into a small and a large group (gap in panel a, plateau in d), and about three quarters of maps sit within a few cells of the 10 records. Before inserting: C3 = rerun v2_all with --references (records only; records + random cells to true size; records + model top cells to true size). n true must print 624 and n reconstructed 4,992 (every map holds the 10 pinned records).

LaTeX for Figure 4 (F1 caption, F2 3.2, F3 abstract, F4 contribution iii, F5 4 opening, F6 4.3, F7 3.6, F8 4.6, F9 Table 2, F10 2.7 scale removal and Fig 7 caption, F11 S1) and the seed/field disclosure (G1 2.1, G2 S6, G3 3.5, G4 4.6) are in the chat reply of 4 Oct; every search phrase occurs once in v9. Abstract after F3 and the short six-community wording: 247 words.
