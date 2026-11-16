"""Start board, Part 4 road map (4a-4c) and Part 5 issues still open (5a-5c)."""
import math
from sketch import Board, INK, PENCIL, BLUE, ORANGE, RED, GREEN, DRED, esc
from frame import start, heading, footer, wrap_lines, tick, xmark, SW, SH

STATUS_FILL = {"fixed": (BLUE, 0.16), "changed": (BLUE, 0.08), "ruled out": (PENCIL, 0.15),
               "not a cause": (PENCIL, 0.15), "not the main cause": (PENCIL, 0.15), "not supported": (PENCIL, 0.15), "stated limit": (ORANGE, 0.25),
               "measured limit": (ORANGE, 0.25)}


def status_key(s):
    for k in STATUS_FILL:
        if s.startswith(k):
            return k
    return "fixed"


def table(b, x0, y0, cols, rows, size=17, status_col=None, max_y=950):
    lh = size * 1.22
    xs = [x0]
    for _, w_ in cols:
        xs.append(xs[-1] + w_)
    hh = 36
    b.rect(x0, y0, xs[-1] - x0, hh, INK, 1.6, fill=ORANGE, fop=0.14, rough=0.4)
    for i, (t, w_) in enumerate(cols):
        b.label(xs[i] + 8, y0 + 6, esc(t), 20, INK, w=w_ - 16)
    y = y0 + hh
    for row in rows:
        n = max(wrap_lines(str(c), cols[i][1] - 18, size) for i, c in enumerate(row))
        h = int(n * lh + 14)
        if status_col is not None:
            col, op = STATUS_FILL[status_key(row[status_col])]
            b.rect(xs[status_col] + 4, y + 4, cols[status_col][1] - 8, h - 8, None, fill=col, fop=op, rough=0.4)
        for i, c in enumerate(row):
            b.label(xs[i] + 8, y + 7, esc(str(c)), size, INK, w=cols[i][1] - 16, lh=1.22)
        b.line(x0, y + h, xs[-1], y + h, PENCIL, 1.0, 1, 0.4)
        y += h
    for x in xs:
        b.line(x, y0, x, y, PENCIL if x not in (xs[0], xs[-1]) else INK, 1.0 if x not in (xs[0], xs[-1]) else 1.8, 1, 0.4)
    b.line(x0, y, xs[-1], y, INK, 1.8, 1, 0.4)
    if y > max_y:
        raise ValueError(f"table overflow: ends at {y}")
    return y


# ------------------------------------------------------------------ start here
def main():
    b = Board(300, SW, SH)
    b.label(56, 32, "Objective 1 and BNM: the pipeline in hand sketches", 54, INK, font="title", lh=1.0)
    b.label(56, 100, "One board per step, in order. Pipeline only, no results. Numbers on the boards are settings and rules.", 23, w=1290)
    heading(b, 56, 160, "The story")
    b.card(60, 220, 300, 130, "Part 1. The simulator", "sheets 1-10: why redone, habitat, niches, one step, dispersal, assembly, settling, time, real run, testing", fill=ORANGE, fop=0.16)
    b.card(400, 220, 250, 130, "Part 2. BNM data", "sheets 11-14: records, effort, fit and test halves, detection", fill=BLUE, fop=0.12)
    b.card(200, 410, 320, 110, "Part 3. Compare them", "sheets 15-17: virtual survey, eight statistics, tracing causes")
    b.card(60, 590, 280, 110, "Part 4. Road map", "boards 4a-4c: the order of the work; every cause, how found, the fix")
    b.card(380, 590, 270, 110, "Part 5. Still open", "boards 5a-5c: simulator, BNM data, comparison", fill=BLUE, fop=0.12)
    b.arrow(210, 354, 300, 406)
    b.arrow(525, 354, 430, 406)
    b.arrow(300, 524, 220, 586)
    b.arrow(420, 524, 500, 586)
    b.label(60, 730, "The 150 simulated communities of Part 1 train the Objective 2 model; Part 3 checks they look like BNM.", 19, w=590)
    # how to read a sheet
    heading(b, 720, 160, "How to read a sheet")
    x, y = 724, 220
    b.rect(x, y, 600, 300, INK, 2.0, fill="#fffdf8", fop=1.0, rough=0.5)
    b.label(x + 14, y + 10, "title + what it is (one line)", 18, INK, w=560)
    b.rect(x + 14, y + 48, 360, 160, PENCIL, 1.4, fill=ORANGE, fop=0.10, rough=0.4)
    b.label(x + 26, y + 58, "How we did it: the sketch, with numbered steps", 18, w=330)
    b.rect(x + 390, y + 48, 196, 236, PENCIL, 1.4, rough=0.4)
    xmark(b, x + 402, y + 64); b.label(x + 422, y + 58, "problem", 17, w=150)
    tick(b, x + 402, y + 98); b.label(x + 422, y + 92, "fix", 17, w=150)
    b.label(x + 402, y + 130, "Problems and fixes: what was wrong, how it was found, what replaced it", 16, w=176)
    b.rect(x + 14, y + 220, 170, 64, PENCIL, 1.4, fill="#f1ece1", fop=1.0, rough=0.4)
    b.label(x + 24, y + 236, "Settings", 18)
    b.rect(x + 196, y + 220, 178, 64, BLUE, 1.4, fill="#e3ebf6", fop=1.0, rough=0.4)
    b.label(x + 206, y + 236, "Still open", 18)
    # word key
    heading(b, 720, 556, "Words on these boards")
    words = [("community", "one simulated metacommunity on a 20 x 20 grid (never \"world\")"),
             ("cell", "one grid square; it stands for one BNM tetrad (2 km x 2 km)"),
             ("window", "20 x 20 cells, 40 km across; the size of one BNM window"),
             ("step / round", "one IBM time step / newcomers enter, then the community runs 5,000 steps"),
             ("piece", "200 steps; 25 pieces make one round; one piece stands for one BNM year"),
             ("newcomer, resident", "a species entering this round; a species already in the community"),
             ("footprint", "the cells where a species can grow on its own (r > 0)"),
             ("fit half, test half", "western BNM windows for tuning; eastern ones for one test"),
             ("band", "the range of a statistic that counts as \"like BNM\"")]
    yy = 608
    for wd, mean in words:
        b.label(724, yy, esc(wd), 19, BLUE, w=180)
        b.label(904, yy, esc(mean), 17, INK, w=446, lh=1.15)
        yy += 22 * max(1, wrap_lines(mean, 446, 17)) + 8
    footer(b, "Start here · from the guide of 16 Nov 2026")
    return b.html("Objective 1 and BNM: the pipeline in hand sketches",
                  "Story map of the five parts, a sketch of how each sheet is laid out, and a key of the words used.")


# ------------------------------------------------------------------ 4a timeline
def road_a():
    b = start(301, "4a", "Road map: the order of the work",
              "Eight old-code faults were fixed first; then each band failure was traced to a cause and given one change at a time, "
              "until a setting passed the gates. The rest became stated limits.")
    stops = [("23-24 Oct", "audit and Step 1 code fixes: the eight old faults", "sheets 1-6"),
             ("26 Oct", "settling and saturation checks", "sheets 6-7"),
             ("27-28 Oct", "niche model and the shared habitat", "sheets 2-3"),
             ("28-29 Oct", "competition links (connectance 1) and d per species", "sheets 4-5"),
             ("29 Oct", "best-cell founding", "sheet 6"),
             ("30-31 Oct", "measured detection and a moving habitat", "sheets 14, 7"),
             ("31 Oct-1 Nov", "regional shift and finer time steps", "sheets 7-8"),
             ("1-2 Nov", "decisions on stated limits", "Part 5"),
             ("2-7 Nov", "the real run: 150 communities", "sheet 9")]
    rows = [(stops[:5], 300), (stops[5:], 640)]
    k = 0
    for row, ly in rows:
        n = len(row)
        x0, x1 = 175, 1225
        b.arrow(x0 - 60, ly, x1 + 90, ly, INK, 2.6, 16)
        step = (x1 - x0) / (5 - 1)
        for i, (d, t, sh) in enumerate(row):
            x = x0 + i * step
            k += 1
            b.ellipse(x, ly, 14, 14, INK, 2.2, fill=(ORANGE if k == 9 else BLUE), fop=0.5)
            b.label(x - 90, ly - 52, esc(d), 24, INK, w=180, align="center", font="title")
            b.card(int(x - 115), ly + 30, 230, 110, f"{k}. {t}", sh, size=19)
    b.label(56, 830, "Each stop used the method of sheet 10: one change, a twin with the same seeds, a rule written first. "
                     "Row numbers on boards 4b and 4c follow claude/root_causes_ledger.md.", 19, w=1290)
    footer(b, "Road map · the order of the Objective 1 redo, 23 Oct to 7 Nov 2026")
    return b.html("Road map 4a · the order of the work",
                  "Two timelines with nine stops from 23 Oct to 7 Nov 2026, each with its date and what was done.")


COLS = [("#", 46), ("cause", 330), ("found by", 300), ("fix", 400), ("status", 224)]
ROWS_A = [
    (1, "d never reached the dynamics; d was 2e-8", "audit T1, test_01, test_05", "d read at every call; d per species 1e-6 to 1e-2", "fixed"),
    (2, "four-neighbour steps plus uniform long jumps", "audit, test_01", "2Dt kernel, a 1, b 3", "fixed"),
    (3, "founding in 5 random cells", "audit", "1 cell, then best-cell founding (row 13)", "fixed"),
    (4, "growth field by list position, redrawn each round", "test_02 E4, test_03 C1, test_04 E4", "field by species id, kept for life; then the niche model", "fixed"),
    (5, "one seed for every community", "test_03 A5, B2", "a seed stream per community; habitat and trait seeds from the sim seed", "fixed"),
    (6, "rounds too short, final run lost species, capped round thrown away", "test_05, test_03 A3, settling test", "5,000-step rounds; capped round kept", "fixed"),
    (7, "species nearly identical", "audit", "niche on a shared habitat; d per species", "fixed as a mechanism"),
    (8, "species deleted below the detection threshold", "audit", "lost only at 0 individuals; threshold for observation only", "fixed"),
    (9, "trade-off k 1 filtered out wide niches", "test_13 K4", "k 0.5", "changed"),
    (10, "habitat contrast too strong", "contrast sweep", "contrast 0.25", "changed"),
    (11, "36% of species pairs never competed", "per-cell split (gamma = alpha x beta)", "connectance 1", "changed"),
    (12, "one d for all species", "similarity by distance, test_13 K3b", "d per species", "changed"),
    (13, "random founding kept out species of rare habitat", "founding_filter", "LVMCM best-cell founding", "fixed; the species excess it brought was cured by row 23"),
]
ROWS_B = [
    (14, "competition strength as the dial for that excess", "plan niche_s", "none", "ruled out"),
    (15, "presence at 1 individual (sink populations)", "tau sweep on saved files", "none", "ruled out"),
    (16, "long-distance dispersal missing", "check of the 2Dt tail", "none", "not a cause"),
    (17, "BNM ranges cut by window edges", "bnm_border_check", "none", "not the main cause"),
    (22, "yardstick took detection as perfect", "overlap between years, bnm_detection_check", "detection by range size in world_vs_bnm", "fixed"),
    (23, "richness never levelled: residents almost never lost", "richness_balance_check, resident_loss_check", "moving habitat with a regional shift", "fixed"),
    (24, "maps changed too fast per round against BNM years", "frame_change_check", "25 pieces per round, 1 piece = 1 BNM year", "fixed"),
    (25, "competition sorting along one habitat axis", "spatial_cause_check, footprint_export", "tried strength, connectance, contrast, habitat scale", "stated limit"),
    (26, "no specialists of rare habitat", "rare_end_check, candidate_specialist_share", "pilot with width 0.1 (helped in part)", "stated limit (rare end)"),
    (27, "the regional shift removes specialists", "pilot niche_r", "none", "not supported"),
    ("-", "slow richness creep", "balance on the real run", "none", "measured limit"),
    ("-", "chance value (R/K)^2 in the clumping ratio", "world_vs_bnm self-test", "exact R(R - 1) / (K(K - 1))", "fixed"),
    ("-", "yardstick read every BNM window", "leak rule (25 Oct)", "fit half only", "fixed"),
    ("-", "one end map is a noisy reading", "frame_bands_export", "20 window-end maps per community", "fixed"),
]


def legend(b, y):
    x = 56
    for k, (col, op) in STATUS_FILL.items():
        if k in ("not a cause", "not the main cause", "not supported", "measured limit"):
            continue
        b.rect(x, y, 26, 20, None, fill=col, fop=op + 0.1, rough=0.3)
        b.label(x + 34, y - 2, k, 17)
        x += 60 + 9 * len(k)


def road_b():
    b = start(302, "4b", "Road map: every cause, how it was found, what was done (1)",
              "Rows 1-13: the eight old-code faults, then the first band failures. Status colour: blue = fixed or changed, grey = ruled out, orange = limit.")
    legend(b, 166)
    table(b, 50, 200, COLS, ROWS_A, size=19, status_col=4)
    footer(b, "Road map 4b · rows 1-13 (claude/root_causes_ledger.md)")
    return b.html("Road map 4b · every cause (rows 1-13)", "A hand-drawn table of causes 1 to 13 with how each was found, the fix and its status.")


def road_c():
    b = start(303, "4c", "Road map: every cause, how it was found, what was done (2)",
              "Rows 14-27 and the late fixes. Rows 18-21 are the Objective 2 inputs (habitat input, partner list, records, history); they belong to the Objective 2 sheets.")
    legend(b, 166)
    table(b, 50, 200, COLS, ROWS_B, size=19, status_col=4)
    footer(b, "Road map 4c · rows 14-27 and late fixes (claude/root_causes_ledger.md)")
    return b.html("Road map 4c · every cause (rows 14-27)", "A hand-drawn table of causes 14 to 27 and four late fixes with how each was found, the fix and its status.")


OCOLS = [("issue", 470), ("why it matters", 380), ("what is done now", 450)]


def open_a():
    b = start(304, "5a", "Still open: the simulator",
              "The open issues are known causes left as stated limits, plus data limits of BNM that later objectives must report; none is an unknown fault.")
    rows = [
        ("Competition sorting along one habitat axis (row 25): each species holds a compact patch of its habitat",
         "similarity decay, species per cell, turnover, 51+ share and range-size spread sit outside BNM; AOO is a range size",
         "stated limit, reported with every result"),
        ("Rare end (row 26): too few 1-cell and 1-5-cell ranges, rare clumping off; no specialists of rare habitat at contrast 0.25 and width 0.2",
         "rare species are the aim",
         "stated limit; Objective 2 weights recorded classes to BNM fit-half shares and tests on hidden BNM fit-half records"),
        ("Slow richness creep: slightly more newcomers kept than residents lost", "communities are close to settled, not exactly", "measured limit"),
        ("Species per window on window-end maps: median inside, too few maps inside", "one map can sit outside the band", "stated limit"),
        ("One synthetic habitat axis; real windows have many land-cover classes, and BNM gives the model no habitat input",
         "habitat learned on simulation may not carry over to BNM", "B-1 tests a real land-cover map (UKCEH LCM 2019) on BNM"),
        ("A community is a torus with no outside; a BNM window is cut from a larger landscape", "no range margins coming in from outside",
         "edge check on BNM: not the main cause; the design difference stays"),
        ("IBM only; Objective 1 names ODE, IBM and PSD", "the thesis text and the code differ in scope",
         "ODE and PSD are a separate project (Sara, 5 Nov); the text must say so"),
    ]
    y = table(b, 50, 170, OCOLS, rows, size=21, max_y=880)
    b.label(56, y + 20, "The rare end shows again downstream: on simulation the model's AOO for truly 1-cell species is too large (P4-2), "
                        "and the habitat input could not shrink it (B-0).", 21, BLUE, w=1290)
    footer(b, "Part 5a · simulator")
    return b.html("Part 5a · Still open: the simulator", "A hand-drawn table of the seven open simulator issues, why each matters and what is done now.")


def open_b():
    b = start(305, "5b", "Still open: the BNM data",
              "Limits of the butterfly records that every later product has to state.")
    rows = [
        ("Opportunistic records with no absences; about half of each window was never searched", "white tetrads are unknown, not absent",
         "products count AOO on land and searched tetrads (P4 land-only)"),
        ("Sea in coastal windows", "the model filled sea cells",
         "land line from UKCEH LCM 2019 (1 km); new: in B-1 the control map is 0 where its paired window has sea, which weakens the control on coastal pairs (M1, waiting for Sara)"),
        ("Detection pooled over species and years; a miss and a one-year population look the same", "the recording rule is an average", "stated with every product"),
        ("The grids hold no visit dates", "two visits per tetrad-year cannot be read from the grids",
         "baseline B3 on BNM uses the detection fitted on simulation, stated as a limit (Sara, 11 Nov)"),
        ("The fit half was reused (survey footprints, detection, class shares, hole test)", "fit-half tests see effort patterns the model saw",
         "the test half was the clean test and has been opened once; a new model version would need new windows or another dataset"),
        ("The land-cover axis for BNM (PC1) is pasture against arable, with little weight on semi-natural classes", "habitat may not describe butterfly habitat",
         "kept as signed if Sara agrees (M2); named in the habitat limit if B-1 fails"),
        ("Minlist 8 is a definition choice", "the target moves with the definition", "bands carry the minlist 4 and 12 spread"),
    ]
    table(b, 50, 170, OCOLS, rows, size=21)
    footer(b, "Part 5b · BNM data")
    return b.html("Part 5b · Still open: the BNM data", "A hand-drawn table of seven BNM data limits, why each matters and what is done now.")


def open_c():
    b = start(306, "5c", "Still open: the comparison, and fixed against open",
              "Four limits of the comparison with BNM, then everything on one page: what was fixed and what is still open.")
    rows = [
        ("Similarity decay cannot reach its band: on a 20 x 20 grid far cells never share habitat, while BNM similarity stays high and flat",
         "one of eight statistics can never pass", "stated limit (Sara, 1 Nov)"),
        ("\"Like BNM\" means like 44 fit-half windows of one dataset", "the target is one region's butterflies",
         "other datasets can be gridded like step 10 and used as a second profile (sketch 14)"),
        ("The habitat shift makes richness swing within a community", "one end map is a noisy reading", "scored on 20 window-end maps per community"),
        ("\"1 piece = 1 BNM year\" is matched on range movement only", "flicker shorter than a year is not compared", "stated with the speed check"),
    ]
    y = table(b, 50, 160, OCOLS, rows, size=19)
    y += 24
    b.rect(60, y, 560, 300, BLUE, 2.0, fill=BLUE, fop=0.08, rough=0.5)
    b.label(80, y + 10, "Fixed", 30, INK, font="title")
    fixed = ["rows 1-8: the old-code faults", "row 13: founding", "row 22: detection", "row 23: richness levels off", "row 24: time pieces"]
    for i, t in enumerate(fixed):
        tick(b, 82, y + 58 + i * 40)
        b.label(110, y + 52 + i * 40, esc(t), 19, w=490)
    b.rect(680, y, 560, 300, ORANGE, 2.0, fill=ORANGE, fop=0.10, rough=0.5)
    b.label(700, y + 10, "Still open", 30, INK, font="title")
    opn = ["row 25: competition sorting", "the rare end (row 26)", "the slow creep", "the one habitat axis", "the torus",
           "sea and effort in BNM", "the used test half"]
    for i, t in enumerate(opn):
        b.label(706, y + 52 + i * 34, "- " + esc(t), 19, w=500)
    b.arrow(1244, y + 150, 1290, y + 150, INK, 2.4, 12)
    b.label(1250, y + 170, "stated with every result in Objectives 2-4", 18, w=110)
    footer(b, "Part 5c · comparison; fixed against still open")
    return b.html("Part 5c · Still open: the comparison", "A table of four comparison limits, then two boxes: fixed and still open, with an arrow to: stated with every result in Objectives 2-4.")


SHEETS = [("Main.dc.html", main), ("RoadMap4a.dc.html", road_a), ("RoadMap4b.dc.html", road_b), ("RoadMap4c.dc.html", road_c),
          ("Open5a.dc.html", open_a), ("Open5b.dc.html", open_b), ("Open5c.dc.html", open_c)]
