"""
patch_readme_strength.py  (obj1_v2, 29 Oct 2026; parts B and C added 30 Oct)

Brings obj1_v2/tests/README.md up to date (Sara's standing request: update the README after every step). Three parts,
each added only if the README lacks it, so one run works on any README from the project:
  A (29 Oct, the strength-scan build): the test_17 row, six tool rows, sketch 39, a decisions paragraph, six results-log
    rows and a change-log line
  B (30 Oct, the strength-scan result): sketch 40 (niche_s against BNM, the expectations checked, the 9 gates left),
    a decisions paragraph, one results-log row and a change-log line
  C (30 Oct, gate 2 built): two tool rows, sketch 41 (presence threshold tau, reading rule and expectations written
    before the run, the gate 6 and 8 plan), a decisions paragraph, one results-log row and a change-log line
Nothing else in the README changes. Use the README from the project (the copy on Sara's machine was older).

Run from obj1_v2/work:
    python3 ../tools/patch_readme_strength.py            # dry run, prints the diff
    python3 ../tools/patch_readme_strength.py --apply
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as README.before_strength.md.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tests/README.md")
src = PATH.read_text()
L = src.split("\n")
has = dict(A="39. Effort check and strength scan" in src, B="40. Strength scan result" in src, C="41. Gate 2 built" in src)
todo = [p for p in "ABC" if not has[p]]
if not todo:
    sys.exit("STOP: the README already has sketches 39, 40 and 41")
for p, q in (("A", "B"), ("B", "C")):
    if has[q] and not has[p]:
        sys.exit(f"STOP: the README has part {q} without part {p}; nothing written")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


TEST17 = [

    "| test_17_strength | Competition strength (29 Oct): T1 0, -0.2, 1.5 refused before any world; T2 no option = 0.4 given "
    "(every saved array); T3 0.7: tag `_fb_is0p7_`, INTERACTION_STRENGTH 0.7 saved, every saved link 0.7 (own slot left out, "
    "I1), alpha_abs_mean 0.7 (S - 1) / S; T4 round 0 tests the same candidates with the same best g at 0.4 and 0.7; "
    "T5 negative control (strength read but not set) fails T3; T6 niche_ok and file patterns of plan niche_s. Tiny worlds "
    "(40 entered, 200-step windows) in a temporary copy of work | `USE_GPU=0 python3 ../tests/test_17_strength.py` | "
    "Only the competition links change; the same candidates are tested until the communities differ |"]

TOOLS_A = [

    "| one_cell_strength_check.py | v2 (29 Oct): species that hold one cell by their own growth against competition "
    "strength s (connectance 1, LVMCM counting, niche_b growth rates, no dispersal); V1 rule = Ricker update, V2 s 1 keeps one "
    "species, V3 trait copy = environment.niche_traits. A planning aid: it chooses the strengths to run, nothing else "
    "(v1 drew widths uniformly; fixed) | `USE_GPU=0 python3 ../tools/one_cell_strength_check.py` |",
    "| effort_check.py | Read-only: bookkeeping checks C1-C4, candidates, entered, kept, richness at equal candidates and "
    "at equal rounds, last-20% change, pass rate; pairs by seed (random against best founding; the same best world at "
    "1,000 and 2,000 entered must repeat). Reading rule written before the run in its docstring. `--self-test` on planted "
    "files | `python3 ../tools/effort_check.py \"<files>\"` |",
    "| patch_run_all_rps_strength.py | `--interaction-strength s` (0 < s <= 1): value of every competition link, set in "
    "assembly_utils next to --connectance; tag `_is<s>` after `_fb` only when s is not 0.4; INTERACTION_STRENGTH already "
    "saved; keeps run_all_rps.before_strength.py | `python3 ../tools/patch_run_all_rps_strength.py --apply` |",
    "| patch_sweep_stage2a_niche_s.py | Plan \"niche_s\": n40 = n33 at 2,000 entered (s 0.4), n37-n39 = n40 at s 0.6, 0.7, "
    "0.8; saved strength checked (no key = 0.4); file pattern names `_is<s>_` | "
    "`python3 ../tools/patch_sweep_stage2a_niche_s.py --apply` |",
    "| check_sweep_niche_s.py | Checks of the niche_s patch on the smoke worlds: commands, older patterns unchanged, each "
    "pattern finds its own file, niche_ok right, saved strength | "
    "`python3 ../tools/check_sweep_niche_s.py ../tools/sweep_stage2a.before_s.py` |",
    "| patch_world_vs_bnm_strength.py | world_vs_bnm v5.4: \", strength s\" in the setting key when s is not 0.4; other "
    "labels unchanged. Self-test 26 PASS | `python3 ../tools/patch_world_vs_bnm_strength.py --apply` |"]

S39 = [

    "",
    "39. Effort check and strength scan, built (29 Oct, late; Sara's yes to all three questions)",
    "```",
    "one-cell check v2 (Sara's machine: V1-V3 PASS, numbers as on Claude's run; no dispersal, so a planning aid only)",
    "   s                      0.4         0.5        0.6        0.7        0.8        0.9        1.0",
    "   species holding cell   12.6-13.4   8.7-10.0   6.3-6.4    4.7-4.9    3.3        1.8-2.2    1",
    "   against s 0.4          1           0.69-0.75  0.47-0.51  0.35-0.39  0.25-0.26  0.14-0.16  0.07-0.08",
    "   -> n33 holds x3.07 BNM's species per cell: target near s 0.7 if sink populations fall in step;",
    "      s 1.0 needs no run (one species per cell by its own growth)",
    "step 1  effort check, no run: n33 tried 10,772 candidates, n29 1,000; kept per candidate 3.3% against about 3.5%",
    "        -> compare richness at equal candidates and at equal rounds (model time)",
    "        reading rule, written before: n33 at 1,000 candidates <= 1.5 x n29 -> mostly extra tries;",
    "        >= 2 x -> best founding packs more species into the same tries; between -> both",
    "        result (Sara's run): at 1,000 candidates n33 75 species, n29 29 (x2.59) -> best founding packs more",
    "        species into the same tries; at round 206 (equal model time) 244 against 17; n33 pass rate 56% -> 7%",
    "        (first / last 20% of rounds) -> the local packing is real, so competition strength is the right dial",
    "step 2  plan niche_s, seed 2303, every world to 2,000 entered newcomers (none of our worlds had stopped growing at 1,000)",
    "        n40 s 0.4 (twin; its first 1,000 must repeat n33)    n37 s 0.6    n38 s 0.7    n39 s 0.8",
    "        only the links change: the same candidates in the same order until the communities differ (test_17 T4)",
    "expected (written before the run)",
    "        species per cell and per window fall as s rises; per cell near BNM at about s 0.7",
    "        richness curve flatter at higher s (smaller change over the last 20%)",
    "        turnover and the 51+ share move little: the founding test keeps narrow niches (K4: width 0.45 -> 0.22)",
    "        similarity decay unchanged; flicker (27.4% of cells per frame in n33) may fall with fewer sinks",
    "        a value is kept only if its 1-5 cell share stays in 15.2-31.4%",
    "next if turnover and 51+ still fail: softer trade-off (k 0.25), twin = the chosen s; then world = window (decay)",
    "```"]

D39 = [

    "",
    "Signed off 29 Oct, late (Sara): the effort check first (read-only); the strength scan at 0.6, 0.7 and 0.8 (1.0 left "
    "out: its answer is known); every scan world run to 2,000 entered newcomers, with n33 extended to 2,000 (n40) as the "
    "twin. Found 29 Oct: one_cell_strength_check v1 drew niche widths uniformly, while environment.niche_traits draws "
    "them log-uniform per species id; v2 copies niche_traits and checks the copy (V3); the ratios barely changed. Run 29 Oct "
    "(Sara): one_cell_strength_check v2, V1-V3 PASS. Found 29 Oct: the reason given for running the scan on the GPU (a CPU run "
    "at 1,000 attempts aims at a richness still rising) applies to the GPU run as well (Claude's error); the safeguard is "
    "the 2,000-entered runs and the extension test. Found 29 Oct (review of n33): kept per candidate is the same under both "
    "founding rules (3.3% against about 3.5%), so n33's richness must be compared at equal effort (effort_check). Built and "
    "tested 29 Oct (sketch 39): effort_check, the runner strength patch, plan niche_s, world_vs_bnm v5.4, test_17, "
    "check_sweep_niche_s. sweep_stage2a.py rebuilt by Claude from the project copy and the six plan patches has Sara's md5 "
    "(3ad16038 before niche_b, 2f8736f2 after), so the niche_s patch was tested on her exact file; run_all_rps.py and "
    "world_vs_bnm.py could not be rebuilt exactly (her computer was not connected), so their patches were tested on stand-ins "
    "with the exact anchor lines, and test_17 and the self-test check them on her files."]

R39 = [

    "| 29 Oct | one_cell_strength_check v2 | Sara's machine | V1 rule = update at s 0.4, 0.7, 0.9, 0.97; V2 s 1 keeps 1; "
    "V3 traits = environment.niche_traits (max diff 0.0); every run reached the cap stop; species holding a cell 13.4 / 12.6 "
    "at s 0.4 (habitat 0 / 1.5), ratios 0.47 / 0.51 at 0.6, 0.35 / 0.39 at 0.7, 0.25 / 0.26 at 0.8 |",
    "| 29 Oct | patch_sweep_stage2a_niche_s, check_sweep_niche_s | Sara's sweep_stage2a.py rebuilt exactly (md5 2f8736f2) | "
    "8 edits, compiles; second apply refused; dry runs of the 11 older plans (and smoke, full --reps 2) byte-identical; n40 = "
    "n33 with 2,000 attempts, n37-n39 = n40 + --interaction-strength 0.6, 0.7, 0.8; check on planted smoke files all PASS; "
    "a planted n38 file saved at 0.4: 3 FAIL lines |",
    "| 29 Oct | patch_run_all_rps_strength | stand-in runner with the exact connectance and founding lines, real assembly_utils.py | "
    "3 edits, compiles; second apply refused; no option = 0.4 given (every array); 0.7: tag `_cn1p0_fb_is0p7_k`, saved 0.7, "
    "links 0.7 at the same positions as at 0.4; 0, -0.2, 1.5 refused, 1.0 accepted; mutant (set line removed) caught |",
    "| 29 Oct | test_17_strength | stand-in runner (writes the keys test_17 reads) | T1-T6 all PASS; T5 negative control "
    "caught; own-partner slots (I1) left out of T3 |",
    "| 29 Oct | patch_world_vs_bnm_strength, effort_check | stand-in with the v5.3 lines; planted files | world_vs_bnm: 3 "
    "edits, labels at 0.4 or without the key unchanged, new self-test line PASS, FAIL on a label without the strength; "
    "effort_check --self-test S1-S4 PASS (bookkeeping, curve ends, 4 planted errors caught, non-repeats told apart) |",
    "| 29 Oct | effort_check, the three patches, world_vs_bnm self-test, test_17, niche_s smoke, check_sweep_niche_s | Sara's "
    "machine | effort_check: self-test all PASS, C1-C4 PASS on n33 and n29-n32; at 1,000 candidates n33 75 species against "
    "n29 29 (x2.59): best founding packs more species into the same tries; at round 206 244 against 17; kept per candidate "
    "3.3% (n33) against 3.0-4.3%; n33 pass rate 56.2% / 7.0% in the first / last 20% of rounds; random-founding worlds still "
    "growing (+3.6% to +17.4% over the last 20%). md5 before 5abe8981, 2f8736f2, f2f90ab5 (as expected); sweep after 09c065a5; "
    "world_vs_bnm self-test 26 PASS; test_17 all passed (T3 19 species, links 0.7; T5 caught; T4 had only 1 candidate in round "
    "0, so little power); smoke n40 53, n37 41, n38 40, n39 38 species (0.4 min each, no ecological weight); "
    "check_sweep_niche_s all passed; this README patch first stopped on a local README without the test_16 row (nothing written) |"]

C39 = [
    "- 29 Oct: strength scan built (sketch 39): effort_check, one_cell_strength_check v2, runner --interaction-strength, "
    "plan niche_s, world_vs_bnm v5.4, test_17, check_sweep_niche_s; README, claude/n33_result_29oct.md, "
    "claude/lvmcm_review_29oct.md and the Claude Doc updated."]

S40 = [
    "",
    "40. Strength scan result (plan niche_s, 30 Oct; seed 2303, 2,000 entered newcomers each, best founding; seen through BNM effort)",
    "```",
    "                          n40 s 0.4   n37 s 0.6   n38 s 0.7   n39 s 0.8   BNM         band",
    "species per window        346         268         220         198         34          27-41",
    "species per cell          57.8        40.0        33.1        30.5        13.7        8.8-18.6",
    "turnover                  5.89        6.64        6.57        6.42        2.34        1.62-3.06",
    "similarity 1 / 5-10       0.55/0.32   0.52/0.29   0.53/0.29   0.56/0.32   0.74/0.70",
    "similarity decay          0.23        0.24        0.24        0.24        0.04        <= 0.085",
    "range-size TV             0.262       0.301       0.296       0.302                   <= 0.116",
    "1-5 cell share            35.3%       38.1%       38.0%       37.7%       23.3%       15.2-31.4%",
    "51+ share                 8.6%        6.9%        7.0%        7.7%        30.7%       25.9-35.5%",
    "bands passed              0 of 8      0 of 8      0 of 8      0 of 8",
    "true species              415         331         272         237",
    "change, last 20%          +17.2%      +25.4%      +15.3%      +15.6%      (none levels off)",
    "individuals per occupied  196         53          22          11",
    "  cell (median)",
    "cells switching / frame   0.317       0.358       0.372       0.367",
    "kept per candidate        2.0%        1.2%        0.8%        0.5%",
    "",
    "expectations (sketch 39)                       result",
    "per cell and per window fall as s rises        met, weakly: per cell x0.53 at 0.8 (one-cell check: x0.25)",
    "per cell near BNM at about s 0.7               NOT met: 33.1 against 13.7",
    "flatter richness curve at higher s             NOT met: +15% to +25% at every s",
    "turnover and 51+ move little                   met",
    "similarity decay unchanged                     met",
    "flicker may fall                               NOT met: 0.317 -> 0.367",
    "1-5 cell share inside the band                 NOT met: 35-38%, above the band",
    "",
    "reading",
    "   strength is not the dial: at 0.8 still x5.8 BNM's species per window and x2.2 per cell",
    "   the one-cell check missed by about 2x because it has no dispersal: as s rose, median individuals",
    "   per occupied cell fell 196 -> 11 (one species alone holds thousands), so most presences are small",
    "   populations, most likely sinks fed by arrivals (75% of present populations were sinks in the CPU",
    "   medium check); at tau 1 every one counts as present, and a third of occupied cells switch each frame",
    "   -> root cause candidate: presence threshold for observation at 1 individual (step 2b, never done)",
    "",
    "gates left before the real run (Sara's rule, 27 Oct: every root cause solved, BNM fit, retraining parts done)",
    "simulator, judged on the BNM fit half             status                        test, cost",
    " 1 richness levels off (saturation)               FAIL in every niche world     test_10 extension, GPU overnight",
    " 2 presence counts small sink populations (tau)   new, strong evidence          tau sweep on saved worlds, CPU minutes",
    " 3 similarity decay (never passed) and turnover   FAIL                          world = window (bigger map), code + GPU",
    " 4 range sizes and rare species (TV, 1-5, 51+,    FAIL                          re-scored after 2",
    "   rare clumping)",
    " 5 4 seeds of the chosen setting inside bands     not started                   GPU, 4 worlds",
    "data for Objective 2, build and check (no open science question)",
    " 6 records 10% and 30%, nested, with effort and detection; prevalence from records only",
    " 7 history frames during invasions (needs 2: a third of occupied cells flicker now)",
    " 8 export fixes: I1 own partner, habitat input (noisy copy), leak tests",
    " 9 headroom gate on simulated truth",
    "plus the live 3D view and video of a few windows (Sara's request; a trace option, built in parallel)",
    "```"]
D40 = ("Run 29-30 Oct (Sara): plan niche_s (n37-n40, 34-36 min each), test_05, effort_check, test_13, world_vs_bnm v5.4 "
       "(sketch 40). 0 of 8 bands at every strength; richness still rising in all four. Claude's prediction (species per "
       "cell near BNM at s 0.7) failed by about 2x: it leaned on a check without dispersal, while median individuals per "
       "occupied cell fell from 196 to 11 as s rose and 32-37% of occupied cells switched state every frame. Proposed "
       "(waiting for Sara's yes): stop the strength scan; test a presence threshold for observation (tau, step 2b) on the "
       "saved worlds with no new run, comparing random founding (n29-n32) and best founding (n33, n37-n40); build the "
       "Objective 2 export parts on the CPU in parallel. Gates left: 9 (sketch 40).")
R40 = ["| 30 Oct | plan niche_s n37-n40, test_05, effort_check, test_13, world_vs_bnm v5.4 | Sara's machine, GPU | "
       "n40 (s 0.4) 34.1 min, 415 species; n37 (0.6) 34.9 min, 331; n38 (0.7) 34.8 min, 272; n39 (0.8) 36.2 min, 237; "
       "final-run losses 0-0.9% PASS; richness change over the last 20% +17.2, +25.4, +15.3, +15.6%; individuals per "
       "occupied cell 196, 53, 22, 11; cells switching per frame 0.317, 0.358, 0.372, 0.367; effort_check C1-C4 PASS, "
       "n40's first 205 rounds identical to n33 PASS, kept per candidate 2.0, 1.2, 0.8, 0.5%, richness at 10,000 "
       "candidates 242, 159, 127, 105; test_13 all passed (K2 0.954, 0.944, 0.941, 0.929; K4 final width 0.21; K7 "
       "Spearman(d, range) 0.52, 0.55, 0.57, 0.66); world_vs_bnm 0 of 8 bands in all four: species per window 346, 268, "
       "220, 198; species per cell 57.8, 40.0, 33.1, 30.5; turnover 5.89, 6.64, 6.57, 6.42; decay 0.23-0.24; TV 0.262, "
       "0.301, 0.296, 0.302; 1-5 share 35.3, 38.1, 38.0, 37.7%; 51+ share 8.6, 6.9, 7.0, 7.7%; rare clumping / BNM at 1-4 "
       "cells 1.14-1.23, 1.08-1.13, 0.90-0.92, 0.78-0.79 |"]
C40 = ("- 30 Oct: strength scan result (sketch 40): 0 of 8 bands at s 0.4-0.8; strength is not the dial; presence threshold "
       "(tau) proposed as the next test, on saved worlds; 9 gates left before the real run.")

TOOLS_C = [
    "| patch_world_vs_bnm_tau.py | world_vs_bnm v5.5 (gate 2, 30 Oct): `--tau T` counts a simulated species as present only "
    "in cells with T or more individuals in the last frame (B_last / BODY_MASS, rounded; exact up to 1,000 in float16), for "
    "observation and scoring only; the assembly is unchanged; tau 1 = P_last_final, so earlier output is unchanged; a file "
    "without B_last is skipped above tau 1. Self-test 27 PASS | `python3 ../tools/patch_world_vs_bnm_tau.py --apply` |",
    "| tau_check.py | Read-only, gate 2: the true maps at tau 1, 10, 100, 1,000 (visible species, species per cell, range "
    "median and 1-5 / 51+ shares, kept presences, flicker between final-run frames); T1 presence rebuilt from B_last = "
    "P_last_final, T2 last IBM_B frame = B_last. `--self-test` S1-S4 | `python3 ../tools/tau_check.py \"<files>\"` |"]
S41 = [
    "",
    "41. Gate 2 built: presence threshold for observation (tau), 30 Oct; Sara's yes to the tau sweep and to building gates 6 and 8",
    "```",
    "why     a BNM record needs a recorder to see the butterfly; in the niche_s worlds the median occupied cell holds",
    "        11-196 individuals (one species alone holds thousands) and a third of occupied cells flicker each frame",
    "what    world_vs_bnm --tau T (v5.5): a species counts as present only where it has T or more individuals in the last",
    "        frame; the assembly is unchanged (it still uses 1 individual, the LVMCM detection level of one body mass)",
    "        tau_check: the true maps at each tau (visible species, species per cell, ranges, kept presences, flicker)",
    "run     tau 1, 10, 100, 1,000 on niche_d (random founding, 4 seeds), n33 and niche_s (best founding, s 0.4-0.8);",
    "        no new simulation; tau 1 must repeat the earlier scores exactly",
    "reading rule (written before the run)",
    "        each setting at each tau gets the 28 Oct bands; tau stays at 1,000 or less (a tenth of what one species",
    "        alone holds at r = 1); kept: most bands passed with the 1-5 cell share inside its band; ties -> lower flicker",
    "        the setting kept goes to gate 1 (extension) and gate 5 (4 seeds) in one GPU batch",
    "expected (written before the run)",
    "        species per window and per cell fall as tau rises, more in best-founding worlds (median occupied cell",
    "        11-207 individuals) than in random-founding ones (834-1,550); flicker falls; 1-5 cell share rises;",
    "        turnover and similarity decay: no prediction",
    "strength scan stopped (Sara, 30 Oct): its four worlds stay in the sweep, so the strength value is still chosen",
    "        from data; the BNM bands stay a gate before the real run (a model trained on worlds unlike BNM",
    "        transfers badly: the old recall-beyond-2-cells failure, sketch 1)",
    "found   window-end maps (ASM_P_*) are saved as presence at 1 individual, without counts, so history frames at",
    "        tau above 1 need the assembly to save them at the observation tau (gate 7; the real run uses new code)",
    "gates 6 and 8, plan (stage 1): export_obj2.py reads a finished world and writes two files: inputs (habitat with",
    "        noise, records 10% inside 30% through BNM fit-half effort and detection, prevalence from records,",
    "        partners with the I1 fix) and truth (scoring only); leak tests: no truth key in the input file,",
    "        records inside the truth and nested, training and evaluation seed ranges apart, test-half BNM never opened",
    "```"]
D41 = ("Signed off 30 Oct (Sara): the tau sweep (gate 2) and building gates 6 and 8; the strength scan stopped, on condition "
       "that stopping does no harm to Objectives 2-4 (answered: the four strength worlds stay in the tau sweep, and the BNM "
       "bands stay a gate). Built and tested 30 Oct (sketch 41): world_vs_bnm v5.5 (--tau), tau_check. Found 30 Oct: "
       "window-end maps are saved at 1 individual only (gate 7 needs them at the observation tau). Gates 6 and 8: plan "
       "written, two design questions put to Sara (habitat input for Objective 2; the interaction input at connectance 1).")
R41 = ["| 30 Oct | patch_world_vs_bnm_tau, tau_check | stand-in with the exact v3 lines and the v5.4 docstring; planted worlds and "
       "BNM-like windows | patch: 7 edits, compiles, second apply refused; no option and --tau 1 give output byte-identical to "
       "before; --tau 100 thins the ranges (planted worlds: median 82 -> 32 cells) and skips a file without B_last; --tau 0.5 "
       "refused; self-test line PASS, FAIL for two planted errors (tau ignored; > instead of >=). tau_check --self-test S1-S4 "
       "PASS (hand values for visible species, species per cell, kept presences, flicker, range shares; T1 and T2 catch "
       "their planted errors); three planted errors in the tool each fail S1 or S4 |"]
C41 = ("- 30 Oct: gate 2 built (sketch 41): world_vs_bnm v5.5 --tau, tau_check; strength scan stopped (Sara); gates 6 and 8 "
       "planned.")


edits = []
if "A" in todo:
    i = find(lambda s: s.startswith("| test_16_founding_best"), "test_16 row (get the README from the project if missing)")
    edits.append((i + 1, i + 1, TEST17))
tools = (TOOLS_A if "A" in todo else []) + (TOOLS_C if "C" in todo else [])
if tools:
    last = "| patch_world_vs_bnm_fithalf.py |" if "A" in todo else "| patch_world_vs_bnm_strength.py |"
    i = find(lambda s: s.startswith(last), "last tools row")
    edits.append((i + 1, i + 1, tools))
sk = dict(A=S39, B=S40, C=S41)
dec = dict(A=D39, B=["", D40], C=["", D41])
res = dict(A=R39, B=R40, C=R41)
log = dict(A=C39, B=[C40], C=[C41])
if "A" in todo:
    i = find(lambda s: s.strip() == "verification: s 0.4 given equals no option (as test_12 U6 for connectance); planted errors must be caught",
             "end of sketch 38")
elif "B" in todo:
    i = find(lambda s: s.strip() == "next if turnover and 51+ still fail: softer trade-off (k 0.25), twin = the chosen s; then world = window (decay)",
             "end of sketch 39")
else:
    i = find(lambda s: s.strip() == "plus the live 3D view and video of a few windows (Sara's request; a trace option, built in parallel)",
             "end of sketch 40")
if L[i + 1].strip() != "```":
    sys.exit("STOP: the last sketch does not end where expected; nothing written")
edits.append((i + 2, i + 2, [x for p in todo for x in sk[p]]))
if "A" in todo:
    i = find(lambda s: s.startswith("Run 29 Oct (Sara): the three founding patches, test_16 --runner --mutants"), "last decisions paragraph")
elif "B" in todo:
    i = find(lambda s: s.startswith("Signed off 29 Oct, late (Sara): the effort check first"), "sketch 39 decisions paragraph")
else:
    i = find(lambda s: s.startswith("Run 29-30 Oct (Sara): plan niche_s"), "sketch 40 decisions paragraph")
edits.append((i + 1, i + 1, [x for p in todo for x in dec[p]]))
if "A" in todo:
    i = find(lambda s: s.startswith("| 29 Oct | plan niche_b n33, test_05, founding_filter v2, test_13, world_vs_bnm v5.3 |"), "n33 results row")
elif "B" in todo:
    last = ("| 29 Oct | effort_check, the three patches" if any(x.startswith("| 29 Oct | effort_check, the three patches") for x in L)
            else "| 29 Oct | patch_world_vs_bnm_strength, effort_check")   # a README patched by the first 29 Oct version
    i = find(lambda s: s.startswith(last), "last 29 Oct results row")
else:
    i = find(lambda s: s.startswith("| 30 Oct | plan niche_s n37-n40"), "niche_s results row")
edits.append((i + 1, i + 1, [x for p in todo for x in res[p]]))
i = find(lambda s: s.strip() == "## Change log", "change log heading")
edits.append((i + 2, i + 2, [x for p in reversed(todo) for x in log[p]]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "README.md (now)", "README.md (patched)", lineterm="", n=0)))
print(f"\n{len(edits)} edits (parts {', '.join(todo)})")
if "--apply" in sys.argv:
    Path("../tests/README.before_strength.md").write_text(src)
    PATH.write_text(out); print("written: ../tests/README.md (the version before is kept as ../tests/README.before_strength.md)")
else:
    print("dry run: nothing written (add --apply to write)")