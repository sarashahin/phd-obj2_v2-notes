# Objective 1 and BNM: a hand-drawing guide to the pipeline

16 Nov 2026 · Sara Shahin. Pipeline only, no results. The drawings live in the Claude Doc "Objective 1 and BNM: a hand-drawing guide to the pipeline" (https://claude.ai/artifact/5z3KzD5fPt6XtTkxPTP8Ve); this file keeps the text for later sessions.

## How to use this guide

Each sheet is one step of the pipeline, drawn by hand on one A4 page, in the order given. Together the sheets tell the story from the old simulator, through the fixes, to the comparison with the BNM butterfly records. Numbers on the sheets are settings and rules; results are left out on purpose.

Every sheet has the same five parts. "What it is" names the step. "How we did it" gives the settings. "Issue and fix" says what was wrong in the old version and what replaced it. "Still open" names what is not solved. "Draw" says what to put on the page, and the picture under the text is the model to copy.

| Word | Meaning on these sheets |
| --- | --- |
| community | one simulated metacommunity on a 20 x 20 grid (never "world") |
| cell | one grid square; it stands for one BNM tetrad (2 km x 2 km) |
| window | 20 x 20 cells, 40 km across; the size of one BNM window |
| step | one IBM time step |
| round | one assembly round: newcomers enter, then the community runs 5,000 steps |
| piece | 200 steps; 25 pieces make one round, and one piece stands for one BNM year |
| newcomer, resident | a species entering this round; a species already in the community |
| footprint | the cells where a species can grow on its own (r > 0) |
| fit half, test half | the western and eastern BNM windows; the fit half is used for tuning, the test half only once in Objective 4 |
| band | the range of a statistic that counts as "like BNM" |

Sources: claude/obj1\_v2\_audit.md, claude/root\_causes\_ledger.md, claude/real\_run\_plan.md, obj1\_v2/tests/README.md (sketches 1-38), claude/teaching\_obj3\_4.md and the tool headers in obj1\_v2/tools.

## Part 1. Objective 1, the simulator

### Sheet 1. Why Objective 1 was redone

**The old simulated communities had nothing an AI could learn from beyond the cells next to a record, so the simulator was rebuilt before the model was retrained.**

What it is. Objective 1 makes the simulated communities that train the Objective 2 model. In the old version the model could not place a species in cells 3 or more cells from a record, and a plain smoother of the records did as well.

How we found the cause. An audit of the simulator code (23 Oct) and tests 01-08 on the old saved communities. The dispersal rate never reached the dynamics, so ranges were frozen where they were founded. Each species was founded in 5 random cells, so its far patches sat where chance dropped them, and no record could point to them.

Issue and fix. One root cause in the simulator spoils every later step, so the fix went into the simulator first (Objective 1 v2), and the model was retrained only after that (Objective 2 v2). The old faults and their fixes are sheets 2-8; the full list is the road map in Part 4.

Still open. Nothing on this sheet; the open items are in Part 5.

Draw. Two 20 x 20 grids side by side. Left, "old": five small patches scattered at random, a record dot in one of them, rings of 1, 2 and 3 cells around the record, and "beyond 2 cells: nothing to learn". Right, "v2": shaded habitat, one clumped range sitting on the good habitat, a record in it, and "habitat and spread explain the range".

Drawing in the Claude Doc: Sheet 1 · old community against v2 community (drawing, not data).

### Sheet 2. The landscape and its habitat

**One shared habitat map now sets where every species can grow; each old species had its own random field.**

What it is. A 20 x 20 torus (the edges wrap round), the size of one BNM window. Every cell has one habitat value H, the same for all species.

How we did it. The map has a length scale of 3 cells and mean 0, sd 1; species see it at contrast 0.25 (0.25 x H). It is drawn from the community's own habitat seed. Through time it drifts slowly (tau 62,500 steps) and the whole layer shifts up and down (regional shift, sd 1.0, tau 62,500); sheet 7 says why.

Issue and fix. Each old species had its own random growth field, keyed by its place in the species list and drawn again every round and at the final run. When one species died, the species after it got a new field (test\_03 C1, test\_04 E4). A length scale above 5 cells changed nothing on a 20 x 20 torus. No ecologist has a species' own growth field, so this input could never be given for BNM. The fix is one shared map with a niche per species (sheet 3). Contrast 1, 0.5 and 0.25 were tried; 0.25 was kept, since contrast 1 sorted species too strongly between neighbouring cells. Length scale 1 was tried and dropped, since it scattered rare species.

Still open. The habitat is one axis only; Part 5.

Draw. A 20 x 20 grid with blobs about 3 cells wide in three shades (good, medium, poor). Beside it, two arrows: "drifts slowly" and "the whole layer shifts".

Drawing in the Claude Doc: Sheet 2 · a shared habitat map (pattern drawn for teaching, not a saved map).

### Sheet 3. Species: a niche on the habitat

**Each species has its own optimum and niche width, so a range can be small because of rare habitat, a narrow niche or slow dispersal.**

What it is. A species' growth rate in a cell follows a bell curve of the habitat there, minus a fixed cost.

```latex
r_i(x) = \mathrm{peak}_i \exp\left(-\frac{(H(x)-\mathrm{opt}_i)^2}{2 w_i^2}\right) - c, \qquad \mathrm{peak}_i = 1.5\,(0.2 / w_i)^{k}
```

How we did it. Optimum sd 1.5, width w 0.2-1.0, cost c 0.1, peak at most 1.5 (one species alone is stable only for r below 2), trade-off k 0.5. Traits come from (trait seed, species id), so a species keeps its niche wherever it sits in the list.

Issue and fix. Every old species had r = 1 on its own small random field, so species were nearly identical. With k = 1 (peak x width the same for all), wide niches were filtered out; k = 0.5 lets generalists in.

Still open. At contrast 0.25, even the narrowest niche (0.2) spans most of the habitat seen, so a specialist of rare habitat cannot win its own cells (row 26). A pilot with width 0.1 helped only in part; Part 5.

Draw. A horizontal axis "habitat H, poor to good". A tall narrow bell (specialist) and a low wide bell (generalist). A horizontal line for the cost; shade where each bell is above it and write "footprint (r > 0)".

Drawing in the Claude Doc: Sheet 3 · two niches on one habitat axis (curves drawn from the formula, example traits).

### Sheet 4. Individuals and competition: one IBM step

**The simulator is a Lotka-Volterra metacommunity counted in individuals, so small populations can die by chance.**

What it is. Each step, in each cell, every species' individuals survive, breed, compete and arrive from other cells. The expected change follows the Lotka-Volterra form:

```latex
B_i(x) \rightarrow B_i(x)\, \exp\!\Big(r_i(x) - \sum_j C_{ij} B_j(x)\Big) + d_i \sum_y K(x - y)\, B_i(y)
```

How we did it. Survival is Binomial, births and arrivals are Poisson; births come from survivors so that mean growth per step is exactly e^g. One individual is a body mass of 1e-4. Every pair competes at C = 0.4 (connectance 1), and C\_ii = 1. A species is lost only at 0 individuals; it counts as present from 1 individual. Recording (survey and detection) is added later, only when a community is compared or exported (Part 3).

Issue and fix. The old code deleted a species with fewer than tau individuals in every cell (tau = the detection threshold), so low-density species vanished and the observation rule changed the community. Now loss is at 0 individuals and tau is for observation only. At the old connectance 0.4, 36% of species pairs never competed and cells held too many species; connectance 1 was kept.

Still open. Every pair competes at 0.4 whatever their niche overlap. A rule tying competition to overlap was tested and was not the fix (row 13d). Competition sorting is in Part 5.

Draw. One cell as a box with a few dots for individuals. Four arrows: "survive? a coin for each individual (Binomial)", "births (Poisson)", "competition from every other species (0.4 each)", "arrivals from other cells (Poisson)".

Drawing in the Claude Doc: Sheet 4 · the four events of one IBM step in one cell.

### Sheet 5. Dispersal

**Dispersal now reaches the dynamics, with a fat-tailed kernel and a rate drawn for each species.**

What it is. Each step a cell sends d\_i x B\_i of species i to other cells, by a kernel K; the source loses nothing (propagule dispersal, as in LVMCM).

How we did it. The 2Dt kernel of [Clark et al. (1999, Ecology 80: 1475-1494)](<https://doi.org/10.1890/0012-9658(1999)080[1475:SDNAFP]2.0.CO;2>), weight (1 + r^2/a^2)^-b with a 1, b 3, on torus distances with periodic images summed and the source cell excluded. d\_i is log-uniform from 1e-6 to 1e-2, drawn per species id from its own random stream, and read at every step. The range comes from test\_14: the low end is the rate of the first settled v2 test communities (1e-6), the high end is the LVMCM example rate (0.01).

Issue and fix. The old d was BODY\_MASS x 0.0002 = 2e-8 per individual per step, built into a matrix once at import, so the d given on the command line never applied. Propagules went only to the 4 neighbours, plus a uniform long-distance share. Ranges were frozen and each cell was an island. dispersal.py v2 reads d at every call; the 2Dt kernel and d per species replaced the rest.

Still open. Nothing on the mechanism; the 2Dt tail carries rare long-distance dispersal, so no separate share is needed (checked 30 Oct).

Draw. Two panels. Left, "old": a centre cell with 4 short arrows to its neighbours, "d 2e-8, never applied". Right, "v2": rings fading outward from the centre and one or two long arrows, "most land near, a few far". Under it, two small ranges: a slow disperser (tight clump) and a fast disperser (spread out).

Drawing in the Claude Doc: Sheet 5 · old and v2 dispersal kernels; two example ranges (drawing, not data).

### Sheet 6. Assembly: the infinite-pool loop

**Communities are built round by round from a pool that never runs out; a candidate enters only if it can grow somewhere, as one individual in its best cell.**

What it is. Stepwise "infinite-pool" assembly, as Objective 1 asks. Candidates are never chosen to fit; the community is what is left after hundreds of rounds of arrivals and losses.

How we did it. One round:

1. Newcomers needed: m = 1 + floor(0.05 S), where S is the number of species now (the LVMCM rule).
2. Draw a candidate: a new niche, a new d, and a competition link of 0.4 to every resident.
3. Invader test (LVMCM): g(x) = r(x) - sum\_j C\_ij B\_j(x) in every cell, with the residents as they are at the start of the round. If its best g is below 0 it never enters (it stays in the life table); otherwise one individual is placed in the cell with the highest g. At most 100 tries per newcomer needed.
4. Run the whole community for 5,000 steps (25 pieces of 200 steps).
5. Keep a newcomer that still has individuals; remove for good a resident with 0 individuals.

Rounds repeat until 1,000 newcomers have entered. A final run of 10,000 steps with no newcomers follows, as the settling check. Each community saves a map per species per piece, the habitat per round and a life table (round in, round out, largest range).

Issue and fix. Old species were founded in 5 random cells, which made far fragments; founding in one random cell then kept out the species of rare habitat, since they rarely landed on a cell where they could grow (founding\_filter). LVMCM's test with best-cell founding replaced both. Every old community used assembly seed 1 and demographic seed 0, so different communities shared founders and draws; now each has its own seed stream. Short rounds left competition unfinished, so the final run still lost species; rounds are now 5,000 steps. The old run stopped at a cap of 10,000 attempts while richness still rose, and the capped round was simulated and thrown away; now the last round is cut to the attempts left and kept. Best founding first let in far too many species; the cause was on the loss side and was fixed on sheet 7.

Still open. A slow creep: slightly more newcomers kept than residents lost, recorded as a measured limit (Part 5).

Draw. A circle of five stations with arrows: "draw candidates", "invader test: can it grow anywhere?" (side arrow "no: never enters"), "1 individual in its best cell", "run 5,000 steps", "keep newcomers alive, remove residents at 0". An exit arrow: "1,000 entered, then final run 10,000 steps".

Drawing in the Claude Doc: Sheet 6 · one assembly round, the stop rule and the final run.

### Sheet 7. Settling: richness must level off

**A community is settled when newcomers kept balance residents lost; a slowly moving habitat is what lets residents be lost.**

What it is. Training maps must not depend on when assembly stopped, so richness must stop rising while species still come and go.

How we did it. The habitat drifts (tau 62,500 steps) and the whole layer shifts (sd 1.0, tau 62,500). Three checks, written before each run: over the last half of the rounds the kept share, kept / (kept + removed), holds 0.5; richness shows no trend (richness\_shift\_check); the final run loses 2% of species or fewer (test\_05).

Issue and fix. In a fixed habitat each resident held its best cells for ever, so residents were almost never lost and richness kept rising under both founding rules (row 23). Lower contrast cost rare species; habitat drift alone did not balance; adding the regional shift did. On the way, three suspects were tested and ruled out: competition strength (0.4-0.8), minimum niche width, and counting presence at 1 individual. A first saturation rule failed its own calibration (test\_09), so saturation was judged by extending runs with the same seeds (test\_10).

Still open. The slow creep of sheet 6.

Draw. Two small plots of richness against rounds. Left, "fixed habitat": a line that keeps rising, "residents never lost". Right, "moving habitat": a flat line with wiggles. Beside it a balance: "newcomers kept = residents lost".

Drawing in the Claude Doc: Sheet 7 · the settling check (curve shapes are a sketch, not data).

### Sheet 8. Time: one saved map per BNM year

**Each round is cut into 25 pieces of 200 steps with a map after each, so one piece changes as much as BNM changes in one year.**

What it is. Change through time is part of the PhD aim, so the step between saved maps must match real years.

How we did it. The habitat moves in 25 small steps per round, with the same total change per round. A map is saved after each piece. The speed is checked by frame\_change\_check v2 against BNM between-year change, by range class, on the fit half:

```latex
q(h) = \frac{J(h)}{1 + J(h)}, \qquad r(h) = \frac{q(h)}{q(1)}
```

Here J(h) is the Jaccard overlap of a species' cells h pieces (or years) apart. Taking the ratio to lag 1 cancels the part that does not build up with time (flicker of tiny populations, missed records), so only real range movement is compared.

Issue and fix. Old final-run maps barely changed, so history frames carried almost nothing. Once the habitat moved, maps changed too fast per round against BNM per year (row 24). Slowing the habitat per round broke the balance of sheet 7, so the round was cut into finer pieces instead.

Still open. Simulated yearly maps hold presences that last less than a piece, and BNM records cannot tell one-year populations from misses; that part is not compared.

Draw. A long bar "one round = 5,000 steps" cut into 25 boxes, each "200 steps = 1 BNM year". Under it a film strip of 6 frames, "2014 ... 2019", with arcs from the first frame to the others labelled "overlap by lag".

Drawing in the Claude Doc: Sheet 8 · pieces, yearly maps and the speed check.

### Sheet 9. The real run: seeds, splits and the quality rule

**150 communities were made with one tested setting, split by community, and each was checked on its own before any use.**

What it is. The communities that train and test the Objective 2 model.

How we did it. The setting is the one that passed the gates on 4 seeds (n70's): sheets 2-8 with best-cell founding, competition 0.4 at connectance 1, 2Dt dispersal with d per species, 1,000 entered newcomers, 25 pieces per round and a final run of 10,000 steps. Variety comes from inside each community: its own habitat map, its own species traits, and d spread over four orders of magnitude.

| Split | Communities | Sim seeds | Used for |
| --- | --- | --- | --- |
| training | 100 | 5001-5100 | fitting the Objective 2 model |
| validation | 20 | 5101-5120 | stopping and tuning only |
| evaluation | 30 | 6001-6030 | read once, after training |

Habitat seed = 100000 + sim seed and trait seed = 200000 + sim seed, so no two communities share a habitat map or a species pool, and none shares a seed with the test communities (2001-2308). Each community must pass three checks, written before the run: real\_run\_check (settings, seeds, 1,000 newcomers, final run 10,000 steps, not resumed, and no seed or map shared with any other community or test file), test\_05 (final-run losses 2% or fewer) and the habitat guard (species still sit on their habitat after it moves). Any species lost in the final run is explained by final\_loss\_check. Balance, trend, speed and the BNM bands are reported per split as description, not as gates.

Issue and fix. The old run was a sweep of 6 settings with shared seeds (field seeds 123 and 456 for training, 789 and 2024 for evaluation), and most of the old validation files shared a training community. Several of the swept settings did nothing, because d never applied. Now one tested setting is used, the split is by community, seed ranges are apart, and R3 of real\_run\_check proves no map is shared.

Still open. Objective 1 names ODE, IBM and PSD; obj1\_v2 is IBM only, since ODE and PSD are a separate project (Part 5).

Draw. Three boxes side by side, "training 100", "validation 20", "evaluation 30", each with its seed range under it and a padlock on evaluation, "read once". Above them a funnel with three filters: "settings and seeds", "final-run losses 2% or fewer", "habitat guard".

Drawing in the Claude Doc: Sheet 9 · the per-community quality rule and the three splits.

### Sheet 10. How each fix was tested

**Each change was made alone, run beside a twin with the same seeds, and judged by a rule written before the run.**

What it is. The method behind every sheet above; it is how a cause was told apart from noise.

How we did it.

1. Find the cause with a check that reads saved files, without a new run where possible.
2. Write the expected result and the pass rule before running.
3. Build the change behind an option, so that "off" gives the old files bit for bit.
4. Test it on planted data, with broken copies (mutants) that must fail.
5. Run the change and its twin: same seeds, one difference. The GPU repeats runs exactly, so the twin differs only by the change.
6. Read the result with the written rule; a PASS counts only if it also makes ecological sense.
7. Keep or drop. A failing band gets at most two fix rounds; after that Sara decides whether it becomes a stated limit.

Issue and fix. Early on, several rules were set after a look at the numbers, and three saturation rules failed their own calibration. From 28 Oct, bands and rules were written before each run.

Still open. Nothing on the method.

Draw. Two identical small grids: "twin A: setting" and "twin B: setting + one change", the same seed written under both. Between them a judge holding a card "rule written first". Three arrows out: "keep", "cause check", "stated limit after 2 rounds".

## Part 2. The BNM butterfly data

### Sheet 11. From raw records to clean records

**Only 2 km records from Great Britain of resident species, 2014-2019, are used; there are no absence records.**

What it is. Butterflies for the New Millennium (BNM) records from the NBN Atlas: "species X seen in tetrad Y in year Z". A tetrad with no record can mean absent or nobody looked.

How we did it. Keep rows with grid size 2,000 m (one tetrad = one IUCN AOO cell, 4 km2), drop Ireland, drop nine non-resident species, and attach the GB Red List 2022 status through a synonym table. Years 2014-2019; 2019 is the target year.

Issue and fix. None new in v2; these steps (steps 9-10 of the earlier pipeline) were reused.

Still open. The records are opportunistic; Part 5.

Draw. A stack of six yearly files, 2014 to 2019, an arrow into a funnel with three filters written on it ("2 km", "GB only", "resident species"), and a small table "species, tetrad, year" coming out.

Drawing in the Claude Doc: Sheet 11 · from the six yearly files to clean records.

### Sheet 12. Effort and windows

**A tetrad counts as searched for a species only after a general visit in its flight months, and windows are cut to the simulator's size.**

What it is. step10\_bnm\_gridder.py turns records into 20 x 20 tetrad windows (40 km) with, for each species and year, its records and the tetrads searched for it.

How we did it. A tetrad passes the effort test for a species in a year if the list there holds 8 or more other species (minlist 8; the focal species is left out of the count) and the visit falls in that species' flight months. A window is kept only if 40 or more tetrads pass in 2019; windows are chosen on effort, never on occupancy. Per-year grids give the same maps for each of 2014-2018. Minlist 4 and 12 are written too; their spread sets part of each band (Part 3).

Why it is built this way. Leaving the focal species out of the list length means effort never selects on the answer. The flight-month rule stops an April visit counting as a search for an August species.

Issue and fix. Before the leak rule (25 Oct), the comparison tool read all windows; from then it reads the fit half only (sheet 13).

Still open. Many tetrads in a window were never searched; Part 5.

Draw. A 20 x 20 grid: black dots for records, grey for searched tetrads, white for never searched. Caption: "white = unknown, not absent". Down the side: "8 or more other species on the list", "in the flight months", "window kept with 40+ searched tetrads in 2019".

Drawing in the Claude Doc: Sheet 12 · one BNM window with records, searched and unsearched tetrads (pattern drawn for teaching, not a real window).

### Sheet 13. Fit half and test half

**The windows are split at 400 km easting; all tuning uses the western half, and the eastern half was kept closed for one test.**

What it is. The leak rule for every use of BNM (Sara, 25 Oct).

How we did it. The cut is the median of the distinct eastings of the minlist 8 windows (400 km): 44 fit windows to the west, 57 test windows to the east. A window from another minlist folder that shares any cell with a test window is never opened, which is decided from file names alone. The fit half gives Objective 1 its yardstick and detection, gives the Objective 2 export its survey footprints and class shares, and holds the fit-half hole test of stage H. The test half was opened once (H4) and its labels read once (H6).

Issue and fix. The earlier alternate split shared windows with the fit half, so it could not be a clean test after tuning; it is not used as a test.

Still open. The fit half was used several times, so only the test half was clean, and it has now been used (Part 5).

Draw. A rough outline of Britain with a vertical line "400 km easting". Left: "fit half, 44 windows: tune here". Right: "test half, 57 windows: opened once", with a padlock.

Drawing in the Claude Doc: Sheet 13 · the leak rule: what each half of the BNM windows is used for (schematic).

### Sheet 14. Detection measured on BNM

**A butterfly that is present is not always recorded; the chance was measured on the fit half, and it depends on range size.**

What it is. The chance that a species present in a searched tetrad gets a record that year.

How we did it. bnm\_detection\_check.py takes the share of a species' lasting tetrads that carry a record in a given year, inside tetrads that pass the effort test that year, on the fit half only. The values used as settings: 0.80 for species with more than 20 tetrads, 0.64 for 20 or fewer. The first estimate came from the overlap J between consecutive years, p = 2J / (1 + J).

Issue and fix. The old yardstick took detection in searched tetrads as perfect ("no data to set it", 24 Oct), so the simulated communities were judged at the wrong level (row 22). world\_vs\_bnm v5.8 applies the measured detection by range size.

Still open. Records cannot tell a missed record from a population that lasted one year, and detection is pooled over species and years (Part 5).

Draw. One searched tetrad with a butterfly in it, and a coin beside it: "recorded? 0.80 (more than 20 tetrads) or 0.64 (20 or fewer)".

Drawing in the Claude Doc: Sheet 14 · detection as a coin toss per present species and tetrad.

## Part 3. Comparing the simulation with BNM

### Sheet 15. The virtual survey

**Each simulated community is seen through real BNM effort and detection before it is compared, so both sides are measured the same way.**

What it is. world\_vs\_bnm.py turns a true simulated map into BNM-like records.

How we did it. For one fit-half BNM window, every simulated species takes the searched tetrads of a BNM species of that window, drawn at random with a fixed seed. Its occupied cells among them are recorded with detection 0.80, or 0.64 if it has 20 or fewer present searched cells. Both sides are then measured as windows, with no wrap-around.

Issue and fix. The old comparison set true simulated maps beside BNM as if every tetrad had been searched, while BNM presences are lower bounds. The survey was added on 24 Oct, the fit-half rule on 25 Oct and measured detection on 30 Oct. The same survey rule later makes the Objective 2 training records (one BNM window per community-round), so training data and BNM are seen alike.

Still open. A simulated community is a torus with no outside, while a BNM window is cut from a larger landscape (Part 5).

Draw. Three small grids in a row with arrows: "true map (simulated)", "searched tetrads of a real BNM species", "records: present and searched, each recorded with chance p". A last arrow: "the same statistics as for BNM".

Drawing in the Claude Doc: Sheet 15 · the virtual survey (pattern drawn for teaching, not data).

### Sheet 16. Eight statistics and their bands

**A setting is judged on eight statistics against bands built from BNM itself, written before the run.**

What it is. The yardstick for every setting tested in Objective 1.

| Statistic | What it measures |
| --- | --- |
| species per window (gamma) | richness of one window |
| species per recorded cell (alpha) | local richness |
| turnover (beta = gamma / alpha) | how much cells differ from each other |
| similarity decay | Sorensen similarity of cells 1 apart minus cells 5-10 apart |
| range-size TV | distance between the range-size shares (1, 2-5, 6-10, 11-20, 21-50, 51+ cells) and BNM's |
| 1-5 cell share | share of small ranges |
| 51+ cell share | share of wide ranges |
| rare clumping | for ranges of 20 cells or fewer: occupied pairs 1-4 cells apart over the chance value, relative to BNM |

How we did it. Band = the BNM minlist 8 value plus or minus (D + S). D is how far minlist 4 or 12 moves the BNM value (definition spread); S is half the range between 4 communities of one setting (seed spread). A setting passes a statistic when the median of its communities is inside the band and 3 of 4 communities are inside; on window-end maps, the median and 75% of maps inside. For rare clumping the chance value for R occupied cells among K searched cells is R(R - 1) / (K(K - 1)) per pair.

Issue and fix. The earlier chance value (R/K)^2 is too high for small ranges, so clumping read too low; the exact form replaced it (24 Oct). Early bands came from all windows; they were set again on the fit half. One end map is a noisy reading while the habitat shifts, so communities are also scored on 20 window-end maps each. Setting labels once pooled different settings; the label now names every option that changed.

Still open. Similarity decay cannot reach its band with one habitat axis on a 20 x 20 community (Part 5).

Draw. A window with a few species in its cells: circle one cell ("alpha: species in a cell"), draw an arrow across the window ("beta: how much cells differ"), and write "gamma = alpha x beta". Under it, a number line with a shaded band and four dots: "median inside and 3 of 4 inside = pass".

Drawing in the Claude Doc: Sheet 16 · alpha, beta, gamma and the pass rule (positions illustrative).

### Sheet 17. Tracing a failing statistic to its cause

**When a band fails, a cause check on saved files decides where the fix goes, before any new run.**

What it is. A set of null models and splits, each asking one question of the saved communities.

How we did it.

| Question | Check |
| --- | --- |
| too many species per cell, or cells too different? | gamma = alpha x beta split (world\_vs\_bnm v4) |
| wrong habitat layout, or wrong filling of the habitat? | spatial\_cause\_check: habitat null and competition null |
| is the fault in the niches or in competition? | footprint\_export: score the map with every cell where r > 0 filled |
| held by arrivals, or by own growth? | sink\_growth: cells cut off from arrivals |
| do maps change too fast because of the habitat? | frame\_change\_check along a chain from fixed to moving habitat |
| are BNM ranges cut by the window edge? | bnm\_border\_check on the fit half |

The answer names the side of the model to change; then one change and its twin (sheet 10), at most two rounds per band.

Issue and fix. Early changes were picked by eye from the band table; from 29 Oct each change followed a cause check.

Still open. Nothing on the method.

Draw. A small tree: "band fails", then "cause check on saved files", then four branches ("habitat layout", "competition sorting", "arrivals", "speed"), then "one change + twin", then "keep" or "stated limit after 2 rounds".

Drawing in the Claude Doc: Sheets 10 and 17 · from a failing band to a decision.

## Part 4. Road map: every cause, how it was found, what was done

**Eight old-code faults were fixed first, then each band failure was traced to a cause and given one change at a time, until a setting passed the gates and the rest became stated limits.**

The order of the work: audit and Step 1 code fixes (23-24 Oct), settling and saturation (26 Oct), niche model and habitat (27-28 Oct), competition links and d per species (28-29 Oct), best-cell founding (29 Oct), detection and moving habitat (30-31 Oct), regional shift and finer time steps (31 Oct-1 Nov), decisions on stated limits (1-2 Nov), real run (2-7 Nov). Row numbers follow claude/root\_causes\_ledger.md.

Drawing in the Claude Doc: Road map · the order of the Objective 1 redo, 23 Oct to 7 Nov 2026.

| # | Cause | Found by | Fix | Status |
| --- | --- | --- | --- | --- |
| 1 | d never reached the dynamics; d was 2e-8 | audit T1, test\_01, test\_05 | d read at every call; d per species 1e-6 to 1e-2 | fixed |
| 2 | four-neighbour steps plus uniform long jumps | audit, test\_01 | 2Dt kernel, a 1, b 3 | fixed |
| 3 | founding in 5 random cells | audit | 1 cell, then best-cell founding (row 13) | fixed |
| 4 | growth field by list position, redrawn each round | test\_02 E4, test\_03 C1, test\_04 E4 | field by species id, kept for life; then the niche model | fixed |
| 5 | one seed for every community | test\_03 A5, B2 | a seed stream per community; habitat and trait seeds from the sim seed | fixed |
| 6 | rounds too short, final run lost species, capped round thrown away | test\_05, test\_03 A3, settling test | 5,000-step rounds; capped round kept | fixed |
| 7 | species nearly identical | audit | niche on a shared habitat; d per species | fixed as a mechanism |
| 8 | species deleted below the detection threshold | audit | lost only at 0 individuals; threshold for observation only | fixed |
| 9 | trade-off k 1 filtered out wide niches | test\_13 K4 | k 0.5 | changed |
| 10 | habitat contrast too strong | contrast sweep | contrast 0.25 | changed |
| 11 | 36% of species pairs never competed | per-cell split (gamma = alpha x beta) | connectance 1 | changed |
| 12 | one d for all species | similarity by distance, test\_13 K3b | d per species | changed |
| 13 | random founding kept out species of rare habitat | founding\_filter | LVMCM best-cell founding | fixed; the species excess it brought was cured by row 23 |
| 14 | competition strength as the dial for that excess | plan niche\_s | none | ruled out |
| 15 | presence at 1 individual (sink populations) | tau sweep on saved files | none | ruled out |
| 16 | long-distance dispersal missing | check of the 2Dt tail | none | not a cause |
| 17 | BNM ranges cut by window edges | bnm\_border\_check | none | not the main cause |
| 22 | yardstick took detection as perfect | overlap between years, bnm\_detection\_check | detection by range size in world\_vs\_bnm | fixed |
| 23 | richness never levelled: residents almost never lost | richness\_balance\_check, resident\_loss\_check | moving habitat with a regional shift | fixed |
| 24 | maps changed too fast per round against BNM years | frame\_change\_check | 25 pieces per round, 1 piece = 1 BNM year | fixed |
| 25 | competition sorting along one habitat axis | spatial\_cause\_check, footprint\_export | tried strength, connectance, contrast, habitat scale | stated limit |
| 26 | no specialists of rare habitat | rare\_end\_check, candidate\_specialist\_share | pilot with width 0.1 (helped in part) | stated limit (rare end) |
| 27 | the regional shift removes specialists | pilot niche\_r | none | not supported |
| - | slow richness creep | balance on the real run | none | measured limit |
| - | chance value (R/K)^2 in the clumping ratio | world\_vs\_bnm self-test | exact R(R - 1) / (K(K - 1)) | fixed |
| - | yardstick read every BNM window | leak rule (25 Oct) | fit half only | fixed |
| - | one end map is a noisy reading | frame\_bands\_export | 20 window-end maps per community | fixed |

Rows 18-21 are the Objective 2 inputs (habitat input, partner list, records, history); they belong to the Objective 2 sheets.

## Part 5. Issues still open

**The open issues are known causes left as stated limits, plus data limits of BNM that the later objectives must report; none is an unknown fault.**

### Objective 1, the simulator

| Issue | Why it matters | What is done now |
| --- | --- | --- |
| Competition sorting along one habitat axis (row 25): each species holds a compact patch of its habitat | similarity decay, species per cell, turnover, 51+ share and range-size spread sit outside BNM; AOO is a range size | stated limit, reported with every result |
| Rare end (row 26): too few 1-cell and 1-5-cell ranges, rare clumping off; no specialists of rare habitat at contrast 0.25 and width 0.2 | rare species are the aim | stated limit; Objective 2 weights recorded classes to BNM fit-half shares and tests on hidden BNM fit-half records |
| Slow richness creep: slightly more newcomers kept than residents lost | communities are close to settled, not exactly | measured limit |
| Species per window on window-end maps: median inside, too few maps inside | one map can sit outside the band | stated limit |
| One synthetic habitat axis; real windows have many land-cover classes, and BNM gives the model no habitat input | habitat learned on simulation may not carry over to BNM | B-1 tests a real land-cover map (UKCEH LCM 2019) on BNM |
| A community is a torus with no outside; a BNM window is cut from a larger landscape | no range margins coming in from outside | edge check on BNM: not the main cause; the design difference stays |
| IBM only; Objective 1 names ODE, IBM and PSD | the thesis text and the code differ in scope | ODE and PSD are a separate project (Sara, 5 Nov); the text must say so |

The rare end shows again downstream: on simulation the model's AOO for truly 1-cell species is too large (P4-2), and the habitat input could not shrink it (B-0).

### BNM data

| Issue | Why it matters | What is done now |
| --- | --- | --- |
| Opportunistic records with no absences; about half of each window was never searched | white tetrads are unknown, not absent | products count AOO on land and searched tetrads (P4 land-only) |
| Sea in coastal windows | the model filled sea cells | land line from UKCEH LCM 2019 (1 km); new: in B-1 the control map is 0 where its paired window has sea, which weakens the control on coastal pairs (M1, waiting for Sara) |
| Detection pooled over species and years; a miss and a one-year population look the same | the recording rule is an average | stated with every product |
| The grids hold no visit dates | two visits per tetrad-year cannot be read from the grids | baseline B3 on BNM uses the detection fitted on simulation, stated as a limit (Sara, 11 Nov) |
| The fit half was reused (survey footprints, detection, class shares, hole test) | fit-half tests see effort patterns the model saw | the test half was the clean test and has been opened once; a new model version would need new windows or another dataset |
| The land-cover axis for BNM (PC1) is pasture against arable, with little weight on semi-natural classes | habitat may not describe butterfly habitat | kept as signed if Sara agrees (M2); named in the habitat limit if B-1 fails |
| Minlist 8 is a definition choice | the target moves with the definition | bands carry the minlist 4 and 12 spread |

### The comparison

| Issue | Why it matters | What is done now |
| --- | --- | --- |
| Similarity decay cannot reach its band: on a 20 x 20 grid far cells never share habitat, while BNM similarity stays high and flat | one of eight statistics can never pass | stated limit (Sara, 1 Nov) |
| "Like BNM" means like 44 fit-half windows of one dataset | the target is one region's butterflies | other datasets can be gridded like step 10 and used as a second profile (sketch 14) |
| The habitat shift makes richness swing within a community | one end map is a noisy reading | scored on 20 window-end maps per community |
| "1 piece = 1 BNM year" is matched on range movement only | flicker shorter than a year is not compared | stated with the speed check |

Draw. A page in two columns: "fixed" (rows 1-8, 13, 22-24) and "still open" (row 25, the rare end, the creep, the habitat axis, the torus, sea and effort in BNM, the used test half). Draw an arrow from "still open" to "stated with every result in Objectives 2-4".

Drawing in the Claude Doc: Part 5 · fixed against still open.
