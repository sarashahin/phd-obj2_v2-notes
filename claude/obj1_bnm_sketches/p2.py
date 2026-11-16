"""Part 2, BNM data: sheets 11-14."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from sketch import INK, PENCIL, BLUE, ORANGE, RED, GREEN, DRED, esc
from frame import start, heading, cards, bottom, footer, step_cards, tick, xmark

# Rough outline of Great Britain in British National Grid km (hand-picked coastal points, approximate).
GB = [(135, 25), (170, 12), (248, 55), (280, 37), (300, 80), (368, 72), (465, 100), (530, 105), (632, 141),
      (640, 170), (590, 185), (625, 232), (655, 293), (620, 342), (560, 330), (556, 363), (540, 410), (525, 470),
      (490, 510), (450, 535), (437, 568), (400, 653), (370, 678), (325, 673), (365, 710), (340, 730), (394, 806),
      (412, 846), (400, 866), (266, 845), (295, 888), (336, 950), (340, 973), (226, 975), (213, 894), (175, 828),
      (142, 767), (185, 730), (160, 610), (230, 670), (233, 622), (215, 530), (330, 555), (295, 512), (345, 465),
      (335, 390), (275, 383), (230, 380), (215, 325), (258, 282), (175, 225), (265, 193), (318, 175), (355, 175),
      (295, 145), (223, 128), (180, 60), (135, 25)]


def tetrad_doodle(b, x, y, s=44, fill=None, fop=0.25):
    b.rect(x, y, s, s, INK, 1.8, fill=fill, fop=fop, rough=0.4)


# ------------------------------------------------------------------ sheet 11
def sheet11():
    b = start(211, 11, "From raw records to clean records",
              "Butterflies for the New Millennium (BNM) records from the NBN Atlas: \"species X seen in tetrad Y in year Z\". "
              "There are no absence records.")
    heading(b, 50, 164, "How we did it")
    # yearly files
    for i, yv in enumerate(range(2014, 2020)):
        x, y = 70 + i * 10, 236 + i * 14
        b.rect(x, y, 150, 100, INK, 1.5, fill="#fffdf8", fop=1.0, rough=0.4)
        if i == 5:
            b.label(x + 10, y + 10, "BNM records<br>2014 ... 2019", 18, w=130)
    b.label(70, 440, "six yearly files<br>(2019 is the target year)", 18, w=200)
    b.arrow(250, 300, 300, 300)
    fl = ["keep 2 km records (grid size 2,000 m)", "drop Ireland: Great Britain only", "drop nine non-resident species",
          "attach the GB Red List 2022 status (synonym table)"]
    for i, t in enumerate(fl):
        b.card(306 + i * 12, 236 + i * 64, 330 - i * 24, 52, f"{i + 1}. {t}", size=17, fill=BLUE, fop=0.08)
    b.arrow(470, 494, 470, 524)
    # output table
    tx, ty = 330, 530
    cols = [("species", 130), ("tetrad", 110), ("year", 80)]
    xx = tx
    for nm, w_ in cols:
        b.rect(xx, ty, w_, 34, INK, 1.4, fill=ORANGE, fop=0.15, rough=0.3, passes=1)
        b.label(xx + 8, ty + 6, nm, 17)
        for r_ in range(3):
            b.rect(xx, ty + 34 + r_ * 30, w_, 30, INK, 1.2, rough=0.3, passes=1)
        xx += w_
    b.label(tx + 8, ty + 40, "species X", 15, PENCIL)
    b.label(tx + 138, ty + 40, "tetrad Y", 15, PENCIL)
    b.label(tx + 248, ty + 40, "year Z", 15, PENCIL)
    b.label(tx + 330, ty + 56, "clean records", 16, PENCIL, w=200)
    # meaning of a record
    b.label(700, 236, "a record", 22)
    tetrad_doodle(b, 700, 270, 60, ORANGE, 0.2)
    b.butterfly(730, 302, 0.9)
    b.label(770, 280, "seen in this tetrad that year", 16, w=100)
    b.label(700, 360, "no record", 22)
    tetrad_doodle(b, 700, 394, 60)
    b.label(718, 400, "?", 40, INK, w=24, font="title")
    b.label(770, 400, "absent, or nobody looked", 16, w=100)
    b.label(700, 474, "one tetrad = 2 km x 2 km = one IUCN AOO cell (4 km²)", 16, w=170)
    cards(b, [
        ("A tetrad with no record can mean absent or nobody looked.",
         "Effort is built from the records themselves (sheet 12); white tetrads are unknown, not absent."),
        ("Nothing new to fix in these steps for v2.",
         "Steps 9-10 of the earlier pipeline were reused."),
    ])
    bottom(b, ["source: BNM records, NBN Atlas", "grid size 2,000 m (tetrads)", "Great Britain only",
               "resident species only (nine dropped)", "GB Red List 2022 via a synonym table", "years 2014-2019; target 2019"],
           "The records are opportunistic: people record where they choose to go (Part 5).", y=704, h=196)
    footer(b, "Sheet 11 · from the six yearly files to clean records")
    return b.html("Sheet 11 · From raw records to clean records",
                  "A stack of six yearly files, four filter cards, and a small table of species, tetrad and year; beside it, a tetrad with a butterfly "
                  "for a record and a tetrad with a question mark for no record.")


# ------------------------------------------------------------------ sheet 12
def sheet12():
    b = start(212, 12, "Effort and windows",
              "A tetrad counts as searched for a species only after a general visit in its flight months. Windows are cut to the simulator's size.")
    heading(b, 50, 164, "How we did it")
    rng = np.random.default_rng(12)
    f = gaussian_filter(rng.normal(size=(20, 20)), 1.6, mode="nearest")
    searched = f > np.quantile(f, 0.5)
    rec = searched & (rng.random((20, 20)) < 0.3)
    c, gx, gy = 17, 70, 236
    for r_ in range(20):
        for c_ in range(20):
            if searched[r_, c_]:
                b.fill_cell(gx, gy, c, c_, r_, PENCIL, 0.28)
            if rec[r_, c_]:
                b.dot(gx + (c_ + 0.5) * c, gy + (r_ + 0.5) * c, 4.2, INK)
    b.grid(gx, gy, 20, c, PENCIL, 0.5)
    b.label(gx, gy - 30, "one window: 20 x 20 tetrads (40 km)", 19)
    ly = gy + 20 * c + 12
    b.dot(gx + 8, ly + 11, 4.2); b.label(gx + 20, ly, "record", 17)
    b.fill_cell(gx + 100, ly + 3, 16, 0, 0, PENCIL, 0.28); b.label(gx + 122, ly, "searched", 17)
    b.rect(gx + 220, ly + 4, 14, 14, PENCIL, 1.0, rough=0.3, passes=1); b.label(gx + 240, ly, "never searched", 17)
    b.label(gx, ly + 30, "white = unknown, not absent", 20, BLUE)
    # effort test
    px = 470
    b.label(px, 236, "is this tetrad searched for species X in year Z?", 19, w=390)
    b.card(px, 270, 390, 60, "1. the visit list holds 8 or more OTHER species (X itself is left out)", size=17, fill=BLUE, fop=0.08)
    b.card(px, 346, 390, 60, "2. the visit falls in X's flight months", size=17, fill=BLUE, fop=0.08)
    b.label(px, 420, "both yes: searched", 19, BLUE)
    # calendar doodle
    for k, (mo, ok) in enumerate((("April visit", False), ("August visit", True))):
        x = px + k * 200
        b.rect(x, 458, 70, 56, INK, 1.6, rough=0.4)
        b.line(x, 474, x + 70, 474, INK, 1.4, 1)
        b.label(x + 80, 466, mo, 17, w=110)
        (tick if ok else xmark)(b, x + 26, 486)
    b.label(px, 524, "for an August butterfly, only the August visit counts", 17, w=390)
    b.label(px, 572, "a window is kept if 40 or more tetrads pass in 2019; windows are chosen on effort, never on occupancy", 18, w=390)
    b.label(px, 640, "per-year grids give the same maps for 2014-2018", 17, PENCIL, w=390)
    cards(b, [
        ("Counting the focal species in the list length would let effort select on the answer.",
         "The focal species is left out of the count."),
        ("An April visit would count as a search for an August species.",
         "The flight-month rule."),
        ("The minlist is a definition choice (8 is used).",
         "Minlist 4 and 12 are written too; their spread sets part of each band (sheet 16)."),
        ("Before the leak rule (25 Oct), the comparison tool read all windows.",
         "From then it reads the fit half only (sheet 13)."),
    ])
    bottom(b, ["step10_bnm_gridder.py", "window 20 x 20 tetrads, 40 km", "minlist 8 (4 and 12 also written)",
               "searched: 8+ other species, in flight months", "window kept: 40+ searched tetrads in 2019"],
           "Many tetrads in a window were never searched (Part 5).", y=704, h=196)
    footer(b, "Sheet 12 · one BNM window (pattern drawn for teaching, not a real window)")
    return b.html("Sheet 12 · Effort and windows",
                  "A 20 by 20 window with dots for records, grey for searched tetrads and white for never searched; beside it the two-part effort test "
                  "and a calendar doodle for April against August visits.")


# ------------------------------------------------------------------ sheet 13
def sheet13():
    b = start(213, 13, "Fit half and test half",
              "The windows are split at 400 km easting. All tuning uses the western half; the eastern half was kept closed for one test.")
    heading(b, 50, 164, "How we did it")
    sc = 0.47
    ox, oy = 250, 690
    P = lambda e, n: (ox + (e - 100) * sc, oy - n * sc)
    pts = [P(e, n) for e, n in GB]
    b.curve(pts, INK, 2.0, 2, 0.6)
    lx = P(400, 0)[0]
    b.line(lx, oy - 470, lx, oy + 6, BLUE, 2.6, 1, dash="10 7")
    b.label(lx - 60, oy + 8, "400 km easting", 18, BLUE, w=140, align="center")
    rng = np.random.default_rng(13)
    for k in range(9):
        e = rng.uniform(170, 390); n = rng.uniform(60, 520)
        x, y = P(e, n); b.rect(x - 7, y - 7, 14, 14, ORANGE, 1.6, fill=ORANGE, fop=0.3, rough=0.3, passes=1)
    for k in range(9):
        e = rng.uniform(410, 600); n = rng.uniform(120, 480)
        x, y = P(e, n); b.rect(x - 7, y - 7, 14, 14, BLUE, 1.6, fill=BLUE, fop=0.3, rough=0.3, passes=1)
    b.label(60, 240, "fit half (west)<br>44 windows<br>tune here", 22, INK, w=170)
    b.label(60, 340, "used for: the Objective 1 yardstick and detection; the Objective 2 export's survey footprints and class shares; "
                     "the stage H fit-half hole test", 16, w=170)
    b.label(560, 240, "test half (east)<br>57 windows<br>opened once", 22, INK, w=170)
    b.padlock(700, 270, 0.9)
    b.label(560, 340, "opened once at H4; its labels read once at H6", 16, w=170)
    # overlap rule doodle
    b.label(560, 470, "leak rule (Sara, 25 Oct)", 19, w=300)
    b.rect(570, 510, 60, 60, BLUE, 1.8, fill=BLUE, fop=0.15, rough=0.4)
    b.rect(600, 534, 60, 60, PENCIL, 1.8, rough=0.4)
    b.cross(630, 564, 10, DRED, 2.4)
    b.label(680, 510, "a window from another minlist folder that shares any cell with a test window is never opened (decided from file names alone)", 15, w=190)
    b.label(60, 640, "squares show the idea, not the real window positions", 15, PENCIL, w=180)
    cards(b, [
        ("Tuning and testing on the same windows would leak.",
         "Cut at the median of the distinct eastings of the minlist 8 windows (400 km): 44 west, 57 east."),
        ("The earlier alternate split shared windows with the fit half, so it could not be a clean test after tuning.",
         "Not used as a test."),
        ("Windows written at minlist 4 or 12 can overlap a test window.",
         "Never opened if they share any cell with one."),
    ])
    bottom(b, ["cut: median of distinct eastings, minlist 8", "= 400 km easting", "fit half: 44 windows (west)",
               "test half: 57 windows (east)", "test half: opened at H4, labels read at H6"],
           "The fit half was used several times, so only the test half was clean, and it has now been used (Part 5).",
           y=724, h=176)
    footer(b, "Sheet 13 · the leak rule: what each half is used for (schematic)")
    return b.html("Sheet 13 · Fit half and test half",
                  "A rough outline of Great Britain with a dashed line at 400 km easting; small squares for windows west and east; "
                  "labels for what each half is used for, a padlock on the test half, and a doodle of the overlap rule.")


# ------------------------------------------------------------------ sheet 14
def sheet14():
    b = start(214, 14, "Detection measured on BNM",
              "A butterfly that is present is not always recorded. The chance was measured on the fit half, and it depends on range size.")
    heading(b, 50, 164, "How we did it")
    # coin flip
    tetrad_doodle(b, 80, 260, 110, ORANGE, 0.15)
    b.butterfly(135, 315, 1.6)
    b.label(80, 376, "present in a searched tetrad", 17, w=140)
    b.ellipse(300, 315, 34, 34, INK, 2.2, fill=ORANGE, fop=0.25)
    b.label(282, 292, "p", 36, INK, w=36, align="center", font="title", lh=1.0)
    b.arrow(196, 315, 260, 315)
    b.arrow(338, 300, 420, 256, BLUE, 2.0, 10)
    b.arrow(338, 330, 420, 376, DRED, 2.0, 10)
    b.card(426, 226, 190, 60, "recorded", size=19, fill=BLUE, fop=0.12)
    b.card(426, 350, 190, 60, "missed: looks absent", size=19)
    b.label(640, 240, "p = 0.80 for species in more than 20 tetrads", 18, w=220)
    b.label(640, 300, "p = 0.64 for species in 20 or fewer", 18, w=220)
    # how measured
    b.label(70, 450, "how p was measured (bnm_detection_check.py, fit half only)", 20, w=780)
    for i, yv in enumerate(range(2014, 2020)):
        x = 80 + i * 74
        b.rect(x, 492, 60, 60, INK, 1.6, rough=0.4, fill=PENCIL, fop=0.12)
        b.label(x, 556, str(yv), 15, w=60, align="center")
        if i in (0, 1, 3, 5):
            b.dot(x + 30, 522, 6, INK)
    b.label(540, 486, "one tetrad where the species lasts; searched every year; a record in 4 of 6 years", 17, w=320)
    b.label(70, 590, "p = share of a species' lasting tetrads that carry a record in a given year, inside tetrads that pass the effort test that year", 18, w=790)
    b.label(70, 642, "first estimate came from the overlap J between consecutive years: p = 2J / (1 + J)", 18, BLUE, w=790)
    cards(b, [
        ("The old yardstick took detection in searched tetrads as perfect (\"no data to set it\", 24 Oct), so the simulated communities were judged at the wrong level (row 22).",
         "world_vs_bnm v5.8 applies the measured detection by range size."),
    ])
    bottom(b, ["bnm_detection_check.py", "fit half only", "p = 0.80: more than 20 tetrads", "p = 0.64: 20 tetrads or fewer",
               "first estimate p = 2J / (1 + J)"],
           "Records cannot tell a missed record from a population that lasted one year, and detection is pooled over species and years (Part 5).",
           y=704, h=196)
    footer(b, "Sheet 14 · detection as a coin toss per present species and tetrad")
    return b.html("Sheet 14 · Detection measured on BNM",
                  "A searched tetrad with a butterfly, a coin marked p, and two outcomes: recorded or missed. Below, six yearly squares for one tetrad "
                  "with records in four years, and how p was measured.")


SHEETS = [("Sheet11.dc.html", sheet11), ("Sheet12.dc.html", sheet12), ("Sheet13.dc.html", sheet13),
          ("Sheet14.dc.html", sheet14)]
