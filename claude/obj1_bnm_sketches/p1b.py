"""Part 1, sheets 6-10, full detail."""
import math
import numpy as np
from sketch import INK, PENCIL, BLUE, ORANGE, RED, GREEN, DRED, esc
from frame import start, heading, cards, bottom, footer, step_cards, tick, xmark


# ------------------------------------------------------------------ sheet 6
def sheet6():
    b = start(206, 6, "Assembly: the infinite-pool loop",
              "Communities are built round by round from a pool that never runs out. A newcomer enters only if it can grow "
              "somewhere, as one individual in its best cell.")
    heading(b, 50, 164, "How we did it: one round")
    b.card(60, 236, 230, 84, "1. draw newcomers needed: m = 1 + floor(0.05 S)", size=18)
    b.card(330, 236, 250, 84, "2. invader test: can it grow anywhere? (best g above 0)", size=18, fill=BLUE, fop=0.10)
    b.card(620, 236, 240, 84, "3. place 1 individual in its best cell", size=18)
    b.card(620, 440, 240, 84, "4. run the whole community 5,000 steps (25 pieces)", size=18)
    b.card(330, 440, 250, 84, "5. keep newcomers still alive; remove residents at 0", size=18)
    b.arrow(294, 278, 326, 278)
    b.arrow(584, 278, 616, 278)
    b.label(586, 250, "yes", 16, BLUE)
    b.arrow(740, 324, 740, 436)
    b.arrow(616, 482, 584, 482)
    b.line(326, 482, 175, 482, INK, 2.2)
    b.arrow(175, 482, 175, 324)
    b.label(70, 380, "repeat until 1,000 newcomers have entered", 18, w=100)
    b.arrow(455, 324, 455, 372, DRED, 2.0, 9)
    b.label(468, 344, "no: never enters (stays in the life table)", 17, DRED, w=140)
    b.label(200, 336, "a candidate = new niche, new d, competition 0.4 with every resident", 16, PENCIL, w=110)
    b.arrow(455, 528, 455, 572)
    b.card(330, 576, 250, 72, "after 1,000 entered: final run 10,000 steps, no newcomers", size=18, fill=ORANGE, fop=0.16)
    b.label(620, 560, "each community saves: a map per species per piece, the habitat per round, and a life table (round in, round out, largest range)", 17, w=250)
    b.label(60, 560, "g(x) = r(x) - sum of C<sub>ij</sub> B<sub>j</sub>(x), residents as at the start of the round", 17, BLUE, w=250)
    cards(b, [
        ("Old species were founded in 5 random cells, which made far fragments; founding in 1 random cell then kept out species of rare habitat (found by founding_filter).",
         "LVMCM invader test with best-cell founding replaced both."),
        ("Every old community used assembly seed 1 and demographic seed 0, so communities shared founders and draws (test_03 A5, B2).",
         "Each community has its own seed stream."),
        ("Short rounds left competition unfinished, so the final run still lost species (test_05, test_03 A3).",
         "Rounds are now 5,000 steps."),
        ("The old run stopped at a cap of 10,000 attempts while richness still rose, and the capped round was simulated and thrown away.",
         "The last round is cut to the attempts left, and kept."),
        ("Best-cell founding first let in far too many species.",
         "The cause was on the loss side; fixed on sheet 7."),
    ], size=18)
    bottom(b, ["m = 1 + floor(0.05 S), the LVMCM rule", "at most 100 tries per newcomer needed",
               "round = 5,000 steps = 25 pieces of 200", "stop at 1,000 entered newcomers", "final run 10,000 steps"],
           "A slow creep: slightly more newcomers kept than residents lost. Recorded as a measured limit (Part 5).", y=690, h=210)
    footer(b, "Sheet 6 · one assembly round, the stop rule and the final run")
    return b.html("Sheet 6 · Assembly: the infinite-pool loop",
                  "A loop of five cards: draw newcomers, invader test, one individual in its best cell, run 5,000 steps, keep or remove; "
                  "a side arrow for newcomers that never enter; an exit to the final run of 10,000 steps.")


# ------------------------------------------------------------------ sheet 7
def sheet7():
    b = start(207, 7, "Settling: richness must level off",
              "A community is settled when newcomers kept balance residents lost. A slowly moving habitat is what lets residents be lost.")
    heading(b, 50, 164, "How we did it")
    for k, (ttl, col) in enumerate((("fixed habitat", PENCIL), ("moving habitat", BLUE))):
        x0, y0 = 80 + k * 300, 250
        b.label(x0, y0 - 34, ttl, 22, col)
        b.arrow(x0, y0 + 170, x0 + 240, y0 + 170, INK, 1.8, 10)
        b.arrow(x0, y0 + 170, x0, y0 - 6, INK, 1.8, 10)
        b.label(x0 + 170, y0 + 176, "rounds", 16, PENCIL)
        b.label(x0 + 6, y0 - 8, "species", 16, PENCIL)
        if k == 0:
            b.curve([(x0 + 8 + i * 28, y0 + 160 - i * 17 + (4 if i % 2 else 0)) for i in range(9)], ORANGE, 2.6)
            b.label(x0 + 20, y0 + 202, "keeps rising: residents never lost", 18, w=240)
        else:
            b.curve([(x0 + 8 + i * 14, y0 + 150 - (90 * (1 - math.exp(-i / 3.0))) + (7 if i % 2 else -5) * (i > 5)) for i in range(17)], BLUE, 2.4)
            b.label(x0 + 20, y0 + 202, "levels off and wiggles", 18, w=240)
    # balance
    cx, cy = 760, 350
    b.line(cx, cy - 80, cx, cy + 50, INK, 2.4)
    b.line(cx - 90, cy - 70, cx + 90, cy - 70, INK, 2.4)
    for sx, t in ((-90, "newcomers kept"), (90, "residents lost")):
        b.line(cx + sx, cy - 70, cx + sx - 26, cy - 20, INK, 1.4, 1)
        b.line(cx + sx, cy - 70, cx + sx + 26, cy - 20, INK, 1.4, 1)
        b.curve([(cx + sx - 30, cy - 20), (cx + sx, cy - 6), (cx + sx + 30, cy - 20)], INK, 2.0)
        b.label(cx + sx - 60, cy + 2, t, 17, w=120, align="center")
    b.line(cx - 40, cy + 50, cx + 40, cy + 50, INK, 2.4)
    b.label(cx - 80, cy - 130, "balance", 22, INK, w=160, align="center")
    # checks
    b.label(80, 494, "three checks, written before each run", 22, INK)
    step_cards(b, ["kept share = kept / (kept + removed) holds 0.5 over the last half of rounds",
                   "richness shows no trend (richness_shift_check)",
                   "the final run loses 2% of species or fewer (test_05)"],
               60, 534, 250, 104, gap=30, size=18, fill=BLUE, fop=0.08, arrows=False)
    cards(b, [
        ("In a fixed habitat each resident held its best cells for ever, so residents were almost never lost and richness kept rising under both founding rules (row 23; richness_balance_check, resident_loss_check).",
         "The habitat drifts slowly and the whole layer shifts (regional shift)."),
        ("Lower contrast cost rare species; habitat drift alone did not balance.",
         "Adding the regional shift did."),
        ("Three suspects for the species excess: competition strength (0.4-0.8), minimum niche width, presence counted at 1 individual.",
         "All three tested and ruled out; none was the cause."),
        ("A first saturation rule failed its own calibration (test_09).",
         "Saturation judged by extending runs with the same seeds (test_10)."),
    ], size=18)
    bottom(b, ["habitat drift: time scale 62,500 steps", "regional shift: sd 1.0, time scale 62,500",
               "kept share target 0.5", "final-run losses 2% or fewer"],
           "The slow creep of sheet 6: slightly more newcomers kept than residents lost.", y=690, h=210)
    footer(b, "Sheet 7 · the settling check (curve shapes are a sketch, not data)")
    return b.html("Sheet 7 · Settling: richness must level off",
                  "Two small plots of species against rounds: rising in a fixed habitat, level in a moving habitat; a balance of newcomers kept "
                  "against residents lost; three checks.")


# ------------------------------------------------------------------ sheet 8
def sheet8():
    b = start(208, 8, "Time: one saved map per BNM year",
              "Each round is cut into 25 pieces of 200 steps, with a map after each, so one piece changes as much as BNM changes in one year.")
    heading(b, 50, 164, "How we did it")
    x0, y0, bw = 70, 252, 30
    b.label(x0, y0 - 34, "1. one round = 5,000 steps", 21)
    for i in range(25):
        b.rect(x0 + i * bw, y0, bw, 40, INK, 1.4, rough=0.35, passes=1, fill=(ORANGE if i == 4 else None), fop=0.6)
    b.arrow(x0 + 4.5 * bw, y0 + 44, x0 + 4.5 * bw + 26, y0 + 70, INK, 1.6, 9)
    b.label(x0 + 4.5 * bw + 32, y0 + 62, "one piece = 200 steps = 1 BNM year; a map is saved after each piece", 18, w=620)
    b.label(x0, y0 + 108, "2. the habitat moves in 25 small steps per round (same total change per round)", 19, w=760)
    # film strip
    fy = 500
    b.label(x0, 392, "3. the speed check: overlap by lag", 21)
    years = ["2014", "2015", "2016", "2017", "2018", "2019"]
    for i, yv in enumerate(years):
        fx = x0 + i * 120
        b.rect(fx, fy, 96, 70, INK, 1.8, rough=0.5)
        b.label(fx, fy + 76, yv, 17, w=96, align="center")
        if i:
            b.arrow(x0 + 48, fy - 2, fx + 48, fy - 2, BLUE, 1.4, 8, bow=-(14 + 12 * i))
    b.label(x0, 604, "J(h) = overlap (Jaccard) of a species' cells h years (or pieces) apart", 18, w=760)
    b.label(x0, 632, "q(h) = J(h) / (1 + J(h)),   r(h) = q(h) / q(1)", 20, BLUE, w=700)
    b.label(x0, 662, "the ratio to lag 1 cancels flicker of tiny populations and missed records, so only real range movement is compared; "
                     "frame_change_check v2 compares it with BNM by range class, on the fit half", 18, w=800)
    cards(b, [
        ("Old final-run maps barely changed, so history frames carried almost nothing.",
         "A moving habitat (sheets 2 and 7)."),
        ("Once the habitat moved, maps changed too fast per round against BNM per year (row 24, frame_change_check).",
         "Not slower habitat: slowing it per round broke the balance of sheet 7."),
        ("So how to match one BNM year?",
         "The round is cut into 25 finer pieces; 1 piece = 1 BNM year."),
    ], size=18)
    bottom(b, ["round = 5,000 steps", "25 pieces of 200 steps", "1 piece = 1 BNM year", "check on the fit half, by range class"],
           "Simulated yearly maps hold presences that last less than a piece, and BNM records cannot tell one-year populations from misses; that part is not compared.",
           y=730, h=170)
    footer(b, "Sheet 8 · pieces, yearly maps and the speed check")
    return b.html("Sheet 8 · Time: one saved map per BNM year",
                  "A long bar of 25 boxes for one round with one box marked as one BNM year; a film strip of six yearly frames with arcs for overlap by lag; the speed formula.")


# ------------------------------------------------------------------ sheet 9
def sheet9():
    b = start(209, 9, "The real run: seeds, splits and the quality rule",
              "150 communities were made with one tested setting, split by community, and each was checked on its own before any use.")
    heading(b, 50, 164, "How we did it")
    b.card(60, 224, 790, 70, "one setting (passed the gates on 4 seeds): sheets 2-8, best-cell founding, competition 0.4 at connectance 1, "
           "2Dt with d per species, 1,000 newcomers, 25 pieces per round, final run 10,000 steps", size=17, fill=ORANGE, fop=0.14)
    # three filters, narrowing
    b.label(80, 316, "150 communities, each checked on its own (rules written before the run)", 18)
    fl = [("1. real_run_check: settings, seeds, 1,000 newcomers, final run 10,000 steps, not resumed, no seed or map shared", 680),
          ("2. test_05: final-run losses 2% or fewer", 560),
          ("3. habitat guard: species still sit on their habitat after it moves", 560)]
    yy = 350
    for t, wdt in fl:
        b.card(455 - wdt // 2, yy, wdt, 52, t, size=17, fill=BLUE, fop=0.08)
        yy += 52 + 14
    fy = 316
    # splits
    sy = 560
    b.arrow(455, 548, 455, sy - 6)
    for i, (nm, n, seeds, use) in enumerate((("training", "100", "5001-5100", "fits the model"),
                                             ("validation", "20", "5101-5120", "stopping and tuning only"),
                                             ("evaluation", "30", "6001-6030", "read once, after training"))):
        x = 70 + i * 270
        b.card(x, sy, 240, 84, f"{nm} {n}", f"seeds {seeds}; {use}", fill=(BLUE if i == 2 else None), fop=0.12)
        if i == 2:
            b.padlock(x + 226, sy - 8, 0.6)
    b.label(70, 656, "habitat seed = 100000 + sim seed; trait seed = 200000 + sim seed: no two communities share a map or a species pool, "
                     "and none shares a seed with the test communities (2001-2308)", 17, w=800)
    cards(b, [
        ("The old run was a sweep of 6 settings with shared seeds (field seeds 123 and 456 for training, 789 and 2024 for evaluation).",
         "One tested setting; variety comes from inside each community (own habitat, own traits, d over 4 orders of magnitude)."),
        ("Most of the old validation files shared a training community.",
         "Split by community, seed ranges apart; R3 of real_run_check proves no map is shared."),
        ("Several of the swept settings did nothing, because d never applied.",
         "d is read at every step (sheet 5)."),
        ("Balance, trend, speed and the BNM bands could hide a bad community if pooled.",
         "Gates are per community; those four are reported per split as description, not as gates."),
    ], size=18)
    bottom(b, ["150 communities", "training 100 / validation 20 / evaluation 30", "habitat seed 100000 + sim seed",
               "trait seed 200000 + sim seed", "final-run losses explained by final_loss_check"],
           "Objective 1 names ODE, IBM and PSD; obj1_v2 is IBM only, since ODE and PSD are a separate project (Part 5).",
           y=726, h=176)
    footer(b, "Sheet 9 · the per-community quality rule and the three splits")
    return b.html("Sheet 9 · The real run",
                  "The tested setting at the top; a funnel with three filters; three boxes for training 100, validation 20 and evaluation 30 with seed ranges, a padlock on evaluation.")


# ------------------------------------------------------------------ sheet 10
def sheet10():
    b = start(210, 10, "How each fix was tested",
              "Each change was made alone, run beside a twin with the same seeds, and judged by a rule written before the run.")
    heading(b, 50, 164, "How we did it")
    rng = np.random.default_rng(10)
    cells = [(int(a), int(c_)) for a, c_ in rng.integers(0, 8, size=(12, 2))]
    for k, (ttl, x) in enumerate((("twin A: setting", 80), ("twin B: setting + one change", 520))):
        b.label(x, 226, ttl, 20, BLUE if k else INK)
        b.grid(x, 260, 8, 16, PENCIL, 0.6, INK, 1.6)
        for cc, rr in cells:
            b.fill_cell(x, 260, 16, cc, rr, ORANGE, 0.8)
        if k:
            for cc, rr in [(1, 6), (2, 6), (6, 1)]:
                b.fill_cell(x, 260, 16, cc, rr, BLUE, 0.6)
        b.label(x, 396, "same seed", 17, PENCIL)
    # judge
    jx, jy = 380, 300
    b.ellipse(jx, jy, 16, 16, INK, 2.0)
    b.line(jx, jy + 16, jx, jy + 66, INK, 2.2)
    b.line(jx, jy + 66, jx - 18, jy + 100, INK, 2.2); b.line(jx, jy + 66, jx + 18, jy + 100, INK, 2.2)
    b.line(jx, jy + 32, jx + 30, jy + 18, INK, 2.2); b.line(jx, jy + 32, jx - 26, jy + 48, INK, 2.2)
    b.rect(jx + 26, jy - 22, 90, 50, INK, 1.6, fill="#fffdf8", fop=1.0, rough=0.4)
    b.label(jx + 30, jy - 18, "rule written first", 15, w=82, align="center", lh=1.1)
    b.label(jx - 60, jy + 108, "the GPU repeats runs exactly, so the twins differ only by the change", 16, PENCIL, w=150, align="center")
    # outcomes
    for i, t in enumerate(["keep", "cause check (sheet 17)", "stated limit after 2 rounds"]):
        b.arrow(700 + 0, 330, 770, 260 + i * 70, INK, 1.6, 9)
        b.label(776, 248 + i * 70, esc(t), 17, w=100)
    b.label(80, 444, "the seven steps", 22)
    steps = ["find the cause with a check that reads saved files", "write the expected result and the pass rule first",
             "build the change behind an option: off gives the old files bit for bit",
             "test on planted data, with broken copies that must fail", "run the change and its twin: same seeds, one difference",
             "read the result with the written rule; a PASS must also make ecological sense",
             "keep or drop; a failing band gets at most two fix rounds, then Sara decides on a stated limit"]
    for i, t in enumerate(steps):
        col, row = i % 4, i // 4
        x, y = 60 + col * 205, 484 + row * 112
        b.card(x, y, 190, 98, f"{i + 1}. {t}", size=16)
    cards(b, [
        ("Early on, several rules were set after a look at the numbers.",
         "From 28 Oct, bands and rules were written before each run."),
        ("Three saturation rules failed their own calibration.",
         "Saturation judged by extending runs with the same seeds (test_10)."),
    ], size=18)
    bottom(b, None, "Nothing on the method.", y=730, h=96)
    footer(b, "Sheet 10 · the twin test and the seven steps")
    return b.html("Sheet 10 · How each fix was tested",
                  "Two small grids, twin A and twin B with the same seed, a judge holding a card saying rule written first, three outcome arrows, and seven numbered step cards.")


SHEETS = [("Sheet06.dc.html", sheet6), ("Sheet07.dc.html", sheet7), ("Sheet08.dc.html", sheet8),
          ("Sheet09.dc.html", sheet9), ("Sheet10.dc.html", sheet10)]
