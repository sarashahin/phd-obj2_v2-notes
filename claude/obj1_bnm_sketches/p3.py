"""Part 3, comparing the simulation with BNM: sheets 15-17."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from sketch import INK, PENCIL, BLUE, ORANGE, RED, GREEN, DRED, esc
from frame import start, heading, cards, bottom, footer, step_cards, tick, xmark


# ------------------------------------------------------------------ sheet 15
def sheet15():
    b = start(215, 15, "The virtual survey",
              "Each simulated community is seen through real BNM effort and detection before it is compared, so both sides are measured the same way.")
    heading(b, 50, 164, "How we did it (world_vs_bnm.py)")
    rng = np.random.default_rng(15)
    n, c = 12, 16
    true = gaussian_filter(rng.normal(size=(n, n)), 1.3, mode="wrap") > 0.35
    f2 = gaussian_filter(rng.normal(size=(n, n)), 1.5, mode="nearest")
    searched = f2 > np.quantile(f2, 0.45)
    kept = rng.random((n, n)) < 0.8
    xs = [70, 345, 620]
    titles = ["1. true map (simulated)", "2. searched (a real BNM species)", "3. records"]
    for k, x in enumerate(xs):
        y = 262
        b.label(x, 220, titles[k], 18, w=230)
        for r_ in range(n):
            for c_ in range(n):
                if k == 0 and true[r_, c_]:
                    b.fill_cell(x, y, c, c_, r_, ORANGE, 0.8)
                if k == 1 and searched[r_, c_]:
                    b.fill_cell(x, y, c, c_, r_, PENCIL, 0.3)
                if k == 2:
                    if searched[r_, c_]:
                        b.fill_cell(x, y, c, c_, r_, PENCIL, 0.18)
                    if true[r_, c_] and searched[r_, c_] and kept[r_, c_]:
                        b.dot(x + (c_ + 0.5) * c, y + (r_ + 0.5) * c, 4, INK)
        b.grid(x, y, n, c, PENCIL, 0.5)
        if k < 2:
            b.arrow(x + n * c + 10, y + 96, x + 265, y + 96, INK, 2.0, 10)
    b.label(70, 470, "from one fit-half BNM window: every simulated species takes the searched tetrads of a BNM species of that window, drawn at random with a fixed seed", 17, w=520)
    b.label(620, 470, "present and searched cells are recorded with chance p (0.80, or 0.64 if 20 or fewer)", 17, w=220)
    b.arrow(700, 540, 700, 572)
    b.card(560, 576, 300, 70, "4. the same statistics as for BNM (sheet 16)", size=18, fill=BLUE, fop=0.10)
    b.label(70, 560, "both sides are measured as windows, with no wrap-around", 18, BLUE, w=460)
    b.label(70, 604, "the same survey rule later makes the Objective 2 training records (one BNM window per community-round), so training data and BNM are seen alike", 17, w=470)
    cards(b, [
        ("The old comparison set true simulated maps beside BNM as if every tetrad had been searched, while BNM presences are lower bounds.",
         "The virtual survey (24 Oct)."),
        ("The comparison read every BNM window, test half included.",
         "Fit-half rule (25 Oct)."),
        ("Detection was taken as perfect.",
         "Measured detection by range size (30 Oct, sheet 14)."),
    ])
    bottom(b, ["world_vs_bnm.py", "one fit-half BNM window per community", "searched tetrads of a random BNM species (fixed seed)",
               "detection 0.80; 0.64 if 20 or fewer present searched cells", "measured as windows, no wrap-around"],
           "A simulated community is a torus with no outside, while a BNM window is cut from a larger landscape (Part 5).",
           y=704, h=196)
    footer(b, "Sheet 15 · the virtual survey (pattern drawn for teaching, not data)")
    return b.html("Sheet 15 · The virtual survey",
                  "Three small grids with arrows: the true simulated map, the searched tetrads of a real BNM species, and the records that result; then the same statistics as for BNM.")


# ------------------------------------------------------------------ sheet 16
def sheet16():
    b = start(216, 16, "Eight statistics and their bands",
              "A setting is judged on eight statistics against bands built from BNM itself, written before the run.")
    heading(b, 50, 164, "How we did it")
    # window with species letters
    gx, gy, c, n = 70, 236, 34, 6
    rng = np.random.default_rng(16)
    letters = "ABCDEFG"
    for r_ in range(n):
        for c_ in range(n):
            k = rng.integers(0, 4)
            sp = "".join(sorted(set(rng.choice(list(letters), size=k)))) if k else ""
            if sp:
                b.label(gx + c_ * c + 2, gy + r_ * c + 6, sp, 13, INK, w=c - 4, align="center", lh=1.0)
    b.grid(gx, gy, n, c, PENCIL, 0.7, INK, 1.8)
    b.ellipse(gx + 1.5 * c, gy + 1.5 * c, 24, 24, BLUE, 2.0, passes=1)
    b.label(gx + n * c + 16, gy - 4, "alpha = species in one cell (circled)", 17, w=200)
    b.arrow(gx + 10, gy + n * c + 18, gx + n * c - 10, gy + n * c + 18, BLUE, 2.0, 10)
    b.label(gx, gy + n * c + 26, "beta = how much cells differ", 17, w=220)
    b.label(gx + n * c + 16, gy + 60, "gamma = species in the window", 17, w=200)
    b.label(gx + n * c + 16, gy + 110, "gamma = alpha x beta", 20, BLUE, w=200)
    # band
    by = 560
    b.label(70, 496, "the pass rule", 21)
    b.arrow(70, by, 470, by, INK, 1.8, 10)
    b.rect(200, by - 18, 150, 36, None, fill=BLUE, fop=0.18, rough=0.3)
    b.line(200, by - 22, 200, by + 22, BLUE, 1.8, 1)
    b.line(350, by - 22, 350, by + 22, BLUE, 1.8, 1)
    b.line(275, by - 26, 275, by + 26, INK, 1.6, 1, dash="4 4")
    for x in (230, 262, 306, 400):
        b.dot(x, by, 6, ORANGE)
    b.label(70, by + 30, "band = BNM value (minlist 8) +/- (D + S); dots = 4 communities of one setting; pass = median inside and 3 of 4 inside", 17, w=420)
    b.label(70, by + 100, "D: how far minlist 4 or 12 moves the BNM value. S: half the range between 4 communities.", 16, PENCIL, w=420)
    # the eight
    lx = 520
    b.label(lx, 236, "the eight statistics", 22)
    stats = ["species per window (gamma)", "species per recorded cell (alpha)", "turnover (beta = gamma / alpha)",
             "similarity decay: Sorensen of cells 1 apart minus cells 5-10 apart",
             "range-size TV distance: shares of 1, 2-5, 6-10, 11-20, 21-50, 51+ cells against BNM",
             "1-5 cell share (small ranges)", "51+ cell share (wide ranges)",
             "rare clumping: ranges of 20 cells or fewer, pairs 1-4 cells apart over chance, relative to BNM"]
    y = 276
    for i, t in enumerate(stats):
        b.label(lx, y, f"{i + 1}. {esc(t)}", 17, w=350, lh=1.2)
        y += 22 * (1 + (len(t) > 40) + (len(t) > 80)) + 8
    b.label(lx, y + 6, "chance for rare clumping: R(R - 1) / (K(K - 1)) per pair, for R occupied cells among K searched", 16, BLUE, w=350)
    cards(b, [
        ("The chance value (R/K)² is too high for small ranges, so clumping read too low (world_vs_bnm self-test).",
         "The exact form R(R - 1) / (K(K - 1)) (24 Oct)."),
        ("Early bands came from all windows.",
         "Set again on the fit half."),
        ("One end map is a noisy reading while the habitat shifts.",
         "Communities also scored on 20 window-end maps each (median and 75% inside)."),
        ("Setting labels once pooled different settings.",
         "The label now names every option that changed."),
    ])
    bottom(b, ["band = BNM minlist 8 value +/- (D + S)", "D = definition spread (minlist 4, 12)", "S = seed spread (half range, 4 communities)",
               "pass: median inside, 3 of 4 inside", "end maps: median and 75% inside"],
           "Similarity decay cannot reach its band with one habitat axis on a 20 x 20 community (Part 5).", y=724, h=176)
    footer(b, "Sheet 16 · alpha, beta, gamma and the pass rule (positions illustrative)")
    return b.html("Sheet 16 · Eight statistics and their bands",
                  "A small window with species letters in cells, one cell circled for alpha and an arrow across for beta; a number line with a shaded band and four dots; the list of eight statistics.")


# ------------------------------------------------------------------ sheet 17
def sheet17():
    b = start(217, 17, "Tracing a failing statistic to its cause",
              "When a band fails, a cause check on saved files decides where the fix goes, before any new run.")
    heading(b, 50, 164, "How we did it")
    b.card(330, 222, 240, 54, "a band fails", size=20, fill=ORANGE, fop=0.18)
    b.arrow(450, 280, 450, 304)
    b.card(250, 308, 400, 54, "cause check on saved files (no new run)", size=19, fill=BLUE, fop=0.10)
    checks = [("too many species per cell, or cells too different?", "gamma = alpha x beta split (world_vs_bnm v4)"),
              ("wrong habitat layout, or wrong filling of the habitat?", "spatial_cause_check: habitat null and competition null"),
              ("is the fault in the niches or in competition?", "footprint_export: score the map with every r > 0 cell filled"),
              ("held by arrivals, or by own growth?", "sink_growth: cells cut off from arrivals"),
              ("do maps change too fast because of the habitat?", "frame_change_check along a chain from fixed to moving habitat"),
              ("are BNM ranges cut by the window edge?", "bnm_border_check on the fit half")]
    b.line(160, 384, 740, 384, INK, 1.8)
    b.line(450, 366, 450, 384, INK, 1.8)
    for i, (q, ck) in enumerate(checks):
        col, row = i % 3, i // 3
        x, y = 60 + col * 280, 400 + row * 128
        if row == 0:
            b.arrow(x + 125, 384, x + 125, y - 4, INK, 1.6, 8)
        b.rect(x, y, 250, 112, INK, 1.8, fill="#fffdf8", fop=1.0, rough=0.5)
        b.label(x + 10, y + 8, esc(q), 17, INK, w=230, lh=1.15)
        b.label(x + 10, y + 56, esc(ck), 15, BLUE, w=230, lh=1.15)
    b.label(130, 668, "the answer names the side of the model to change, then one change + its twin (sheet 10), at most 2 rounds per band: keep, or a stated limit", 17, w=700)
    cards(b, [
        ("Early changes were picked by eye from the band table.",
         "From 29 Oct each change followed a cause check."),
    ])
    bottom(b, None, "Nothing on the method.", y=740, h=96)
    footer(b, "Sheets 10 and 17 · from a failing band to a decision")
    return b.html("Sheet 17 · Tracing a failing statistic to its cause",
                  "A tree: a band fails, a cause check on saved files, six question cards each with the check that answers it, then one change and its twin.")


SHEETS = [("Sheet15.dc.html", sheet15), ("Sheet16.dc.html", sheet16), ("Sheet17.dc.html", sheet17)]
