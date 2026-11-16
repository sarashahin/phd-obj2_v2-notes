"""Part 1, sheets 1-5, full detail."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from sketch import INK, PENCIL, BLUE, ORANGE, RED, GREEN, DRED, esc
from frame import start, heading, cards, bottom, footer, step_cards, tick, xmark


def habitat(seed, n=20, sigma=1.4):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(size=(n, n)), sigma, mode="wrap")
    return (f - f.mean()) / f.std()


def hab_grid(b, x, y, n, c, h, levels=(1 / 3, 2 / 3)):
    q1, q2 = np.quantile(h, levels)
    for r_ in range(n):
        for c_ in range(n):
            v = h[r_, c_]
            if v > q2:
                b.fill_cell(x, y, c, c_, r_, GREEN, 0.55)
            elif v > q1:
                b.fill_cell(x, y, c, c_, r_, GREEN, 0.22)
    b.grid(x, y, n, c, PENCIL, 0.6)


# ------------------------------------------------------------------ sheet 1
def sheet1():
    b = start(201, 1, "Why Objective 1 was redone",
              "Objective 1 makes the simulated communities that train the Objective 2 model. "
              "The old ones gave the model nothing to learn beyond the cells next to a record.")
    heading(b, 50, 164, "How we found it")
    c = 12
    ox, oy = 90, 250
    b.label(ox, oy - 36, "old community", 24, PENCIL)
    b.grid(ox, oy, 20, c, PENCIL, 0.6)
    patches = [[(6, 8), (7, 8), (6, 9)], [(15, 3), (16, 3)], [(3, 16), (3, 17), (4, 17)],
               [(16, 14), (17, 14), (17, 15)], [(11, 18), (12, 18)]]
    for p in patches:
        for cc, rr in p:
            b.fill_cell(ox, oy, c, cc, rr, ORANGE, 0.8)
    rx, ry = ox + 6.5 * c, oy + 8.5 * c
    for rad in (1.5, 2.5, 3.5):
        b.ellipse(rx, ry, rad * c, rad * c, BLUE, 1.4, passes=1, rough=0.5)
    b.dot(rx, ry, 4.5, BLUE)
    b.label(ox, oy + 248, "rings at 1, 2, 3 cells round a record;<br>beyond 2 cells: nothing to learn", 19, INK, w=300)
    b.label(ox + 252, oy + 10, "5 patches dropped<br>at random", 18, INK, w=150)
    b.arrow(ox + 270, oy + 52, ox + 16.6 * c, oy + 3.4 * c, PENCIL, 1.6, 9, bow=-8)
    # v2
    vx, vy = 560, 250
    b.label(vx, vy - 36, "v2 community", 24, BLUE)
    h = habitat(11)
    hab_grid(b, vx, vy, 20, c, h, (0.45, 0.7))
    rr0, cc0 = np.unravel_index(np.argmax(h), h.shape)
    for r_ in range(20):
        for c_ in range(20):
            dr = min(abs(r_ - rr0), 20 - abs(r_ - rr0))
            dc = min(abs(c_ - cc0), 20 - abs(c_ - cc0))
            if h[r_, c_] > np.quantile(h, 0.74) and max(dr, dc) <= 4:
                b.fill_cell(vx, vy, c, c_, r_, ORANGE, 0.85)
    b.dot(vx + (cc0 + 0.5) * c, vy + (rr0 + 0.5) * c, 4.5, BLUE)
    b.label(vx, vy + 248, "one clumped range on good habitat;<br>habitat and spread explain it", 19, INK, w=300)
    b.arrow(368, 370, 540, 370, INK, 2.4, 14)
    b.label(372, 330, "rebuild", 20, INK, w=160, align="center")
    # legend
    b.dot(90 + 8, 566, 4.5, BLUE); b.label(90 + 20, 555, "record", 17)
    b.fill_cell(190, 558, 14, 0, 0, ORANGE, 0.85); b.label(210, 555, "species present", 17)
    b.fill_cell(350, 558, 14, 0, 0, GREEN, 0.55); b.label(370, 555, "good habitat", 17)
    # how found strip
    step_cards(b, ["audit of the simulator code (23 Oct)", "tests 01-08 on the old saved communities",
                   "fix the simulator first: Objective 1 v2", "then retrain the model: Objective 2 v2"],
               50, 600, 182, 86, gap=34, size=18)
    cards(b, [
        ("The model could not place a species 3 or more cells from a record; a plain smoother of the records did as well.",
         "Find the root cause in the simulator, because one fault there spoils every later step."),
        ("The dispersal rate never reached the dynamics, so ranges froze where they were founded.",
         "Dispersal rebuilt so the rate is read at every step (sheet 5)."),
        ("Each species was founded in 5 random cells, so its far patches sat where chance dropped them and no record could point to them.",
         "Founding rebuilt: one individual in the cell where it grows best (sheet 6)."),
        ("Other old faults: growth fields by list position, one seed for all, short rounds, species deleted below the detection line.",
         "Fixed one by one on sheets 2-8; the full list is the road map (Part 4)."),
    ])
    bottom(b, None, "Nothing on this sheet. The open items are in Part 5.", y=730, h=110)
    footer(b, "Sheet 1 · old against v2 community (drawing, not data)")
    return b.html("Sheet 1 · Why Objective 1 was redone",
                  "Old community: five random patches and a record with rings at 1, 2 and 3 cells. v2 community: good habitat with one clumped range. "
                  "Below, four steps: audit, tests 01-08, fix the simulator, retrain the model.")


# ------------------------------------------------------------------ sheet 2
def sheet2():
    b = start(202, 2, "The landscape and its habitat",
              "A 20 x 20 grid whose edges join up (a torus), the size of one BNM window. "
              "Every cell has one habitat value H, the same for all species.")
    heading(b, 50, 164, "How we did it")
    c = 15
    gx, gy = 96, 236
    h = habitat(5)
    hab_grid(b, gx, gy, 20, c, h)
    b.label(gx, gy - 26, "1. one shared map", 21, INK)
    b.arrow(gx + 300 + 4, gy + 75, gx + 300 + 36, gy + 75, BLUE, 2.0, 9)
    b.arrow(gx - 34, gy + 75, gx - 4, gy + 75, BLUE, 2.0, 9)
    b.label(gx, gy + 306, "edges join: walk off the right,<br>come back on the left", 18, BLUE, w=300)
    b.fill_cell(gx, gy + 362, 16, 0, 0, GREEN, 0.55); b.label(gx + 22, gy + 358, "good", 17)
    b.fill_cell(gx + 80, gy + 362, 16, 0, 0, GREEN, 0.22); b.label(gx + 102, gy + 358, "medium", 17)
    b.rect(gx + 176, gy + 363, 14, 14, PENCIL, 1.0, rough=0.3, passes=1); b.label(gx + 196, gy + 358, "poor", 17)
    b.label(gx, gy + 392, "patches about 3 cells wide", 18, PENCIL)
    # 2 species see softened map
    px = 470
    b.label(px, 212, "2. species feel a softened map", 21, INK)
    xs = np.linspace(px, px + 340, 13)
    vals = [math.sin(i * 0.8) * 34 for i in range(13)]
    b.curve([(x, 300 - v) for x, v in zip(xs, vals)], GREEN, 2.2)
    b.curve([(x, 300 - v * 0.25) for x, v in zip(xs, vals)], INK, 2.0, 1, dash="7 5")
    b.label(px + 250, 238, "H", 19, GREEN)
    b.label(px + 250, 312, "0.25 x H", 18, INK)
    # 3 drift
    b.label(px, 360, "3. the map drifts slowly", 21, INK)
    blob = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3)]
    for k, sx in enumerate((px, px + 170)):
        b.grid(sx, 396, 6, 13, PENCIL, 0.6, INK, 1.5)
        for cc, rr in blob:
            b.fill_cell(sx, 396, 13, cc + k, rr, GREEN, 0.55)
    b.arrow(px + 86, 434, px + 162, 434, INK, 1.8, 9, bow=-8)
    b.label(px + 268, 410, "time scale<br>62,500 steps", 17, PENCIL, w=120)
    # 4 layer shift
    b.label(px, 494, "4. the whole layer shifts up and down", 21, INK)
    base = [596 + 14 * math.sin(i * 0.9) for i in range(9)]
    xs2 = np.linspace(px, px + 280, 9)
    b.curve(list(zip(xs2, base)), INK, 2.0)
    b.curve(list(zip(xs2, [v - 36 for v in base])), BLUE, 1.8, 1, dash="7 5")
    b.arrow(px + 310, 600, px + 310, 552, BLUE, 1.8, 9)
    b.arrow(px + 310, 566, px + 310, 614, BLUE, 1.8, 9)
    b.label(px + 330, 556, "sd 1.0, time<br>scale 62,500", 17, PENCIL, w=110)
    b.label(px, 636, "drift + shift let residents be lost (sheet 7)", 18, INK, w=400)
    cards(b, [
        ("Each old species had its own random growth field, keyed by its place in the species list and drawn again every round and at the final run.",
         "One shared habitat map for all species, plus a niche per species (sheet 3)."),
        ("When one species died, the species after it got a new field (found by test_03 C1, test_04 E4).",
         "Traits keyed by species id, so nothing changes when another species dies."),
        ("No ecologist has a species' own growth field, so this input could never be given for BNM. A length scale above 5 cells changed nothing on a 20 x 20 torus.",
         "A habitat map is something a real map can stand in for (land cover, Part 5)."),
        ("Contrast 1 sorted species too strongly between neighbouring cells; length scale 1 scattered rare species.",
         "Contrast 1, 0.5 and 0.25 tried: 0.25 kept. Length scale 1 tried and dropped; 3 kept."),
    ])
    bottom(b, ["length scale 3 cells", "mean 0, sd 1", "contrast 0.25 (species see 0.25 x H)",
               "drawn from the community's habitat seed", "drift time scale 62,500 steps",
               "regional shift sd 1.0, time scale 62,500"],
           "The habitat is one axis only. Real windows have many land-cover classes (Part 5).")
    footer(b, "Sheet 2 · pattern drawn for teaching, not a saved map")
    return b.html("Sheet 2 · The landscape and its habitat",
                  "A 20 by 20 habitat grid in three shades with arrows showing the edges join; a habitat line and its softened copy; "
                  "a patch drifting; the whole layer shifting up and down.")


# ------------------------------------------------------------------ sheet 3
def sheet3():
    b = start(203, 3, "Species: a niche on the habitat",
              "Each species has its own best habitat (optimum) and niche width. A range can be small because "
              "its habitat is rare, its niche is narrow, or it spreads slowly.")
    heading(b, 50, 164, "How we did it")
    x0, x1, yb, yt = 100, 600, 520, 250
    Hmin, Hmax, ymax = -3.0, 3.0, 1.6
    X = lambda v: x0 + (v - Hmin) / (Hmax - Hmin) * (x1 - x0)
    Y = lambda v: yb - v / ymax * (yb - yt)
    b.arrow(x0, yb, x1 + 26, yb, INK, 2.0)
    b.arrow(x0, yb, x0, yt - 24, INK, 2.0)
    b.label(x1 - 160, yb + 6, "habitat: poor to good", 18)
    b.label(x0 + 8, yt - 34, "growth rate r", 18)
    c = 0.1
    sp = dict(opt=1.0, w=0.3); ge = dict(opt=-0.8, w=1.0)
    for s in (sp, ge):
        s["peak"] = 1.5 * (0.2 / s["w"]) ** 0.5
    for s, col in ((ge, BLUE), (sp, ORANGE)):
        hs = np.linspace(Hmin, Hmax, 61)
        b.curve([(X(v), Y(s["peak"] * math.exp(-(v - s["opt"]) ** 2 / (2 * s["w"] ** 2)))) for v in hs], col, 2.6, 2, 0.4)
    b.line(x0, Y(c), x1, Y(c), INK, 1.5, 1, dash="8 6")
    b.label(x0 + 12, Y(c) - 26, "cost c", 17)
    b.label(X(1.0) + 26, Y(sp["peak"]) - 6, "specialist:<br>narrow, tall", 18, INK, w=120)
    b.label(X(-0.8) - 80, Y(ge["peak"]) - 52, "generalist: wide, low", 18, INK, w=200)
    for s, col, yy in ((ge, BLUE, yb + 40), (sp, ORANGE, yb + 56)):
        half = s["w"] * math.sqrt(2 * math.log(s["peak"] / c))
        b.line(X(s["opt"] - half), yy, X(s["opt"] + half), yy, col, 7, 1, 0.4)
    b.label(x0, yb + 68, "bars = footprint: where growth beats the cost (r > 0)", 18)
    # formula
    b.label(660, 236, "the rule", 24, INK, font="title")
    b.label(660, 270, "r = peak x bell(H; optimum, width) - cost", 18, BLUE, w=215)
    b.label(660, 322, "peak = 1.5 x (0.2 / width)<sup>k</sup>", 18, BLUE, w=215)
    b.label(660, 362, "narrow niche gets a taller peak; k = 0.5 sets how much", 18, INK, w=215)
    b.label(660, 440, "traits come from (trait seed, species id): a species keeps its niche wherever it sits in the list", 18, INK, w=215)
    cards(b, [
        ("Every old species had r = 1 on its own small random field, so species were nearly identical.",
         "A niche on the shared habitat: own optimum and width, minus a fixed cost."),
        ("With k = 1 (peak x width the same for all), wide niches were filtered out (test_13 K4).",
         "k = 0.5 lets generalists in."),
        ("One species alone is stable only for r below 2.",
         "Peak capped at 1.5."),
    ])
    bottom(b, ["optimum sd 1.5", "width w 0.2 to 1.0", "cost c 0.1", "peak at most 1.5", "trade-off k 0.5"],
           "At contrast 0.25 even the narrowest niche (0.2) spans most of the habitat seen, so a specialist of rare habitat "
           "cannot win its own cells (row 26). A pilot with width 0.1 helped only in part (Part 5).")
    footer(b, "Sheet 3 · curves from the formula, example traits")
    return b.html("Sheet 3 · Species: a niche on the habitat",
                  "Growth against habitat: a tall narrow bell for a specialist, a low wide bell for a generalist, a dashed cost line, "
                  "and bars under the axis for each footprint. Beside it, the rule in words.")


# ------------------------------------------------------------------ sheet 4
def sheet4():
    b = start(204, 4, "Individuals and competition: one IBM step",
              "The simulator is a Lotka-Volterra metacommunity counted in individuals, so small populations can die by chance.")
    heading(b, 50, 164, "How we did it: each step, in each cell")
    bx, by, bw, bh = 330, 300, 280, 230
    b.rect(bx, by, bw, bh, INK, 2.6)
    b.label(bx + 6, by + bh + 6, "one cell", 18, PENCIL)
    rng = np.random.default_rng(4)
    placed = []
    for kind, n in (("ink", 10), ("open", 7), ("blue", 4)):
        k = 0
        while k < n:
            x, y = bx + 24 + rng.random() * (bw - 48), by + 24 + rng.random() * (bh - 48)
            if all(math.hypot(x - a, y - c_) > 26 for a, c_ in placed):
                placed.append((x, y)); k += 1
                if kind == "ink": b.dot(x, y, 6, INK)
                elif kind == "open": b.ring(x, y, 6, INK)
                else: b.dot(x, y, 6, BLUE)
    # 1 survive
    b.ellipse(104, 262, 26, 26, INK, 2.0, fill=ORANGE, fop=0.25)
    b.label(88, 240, "?", 34, INK, w=32, align="center", font="title", lh=1.0)
    b.label(140, 226, "1. survive? a coin flip for each individual (Binomial)", 18, w=180)
    b.arrow(250, 300, bx - 6, 330, INK, 2.0, bow=-8)
    # 2 births
    b.dot(668, 252, 6); b.arrow(682, 252, 718, 238, INK, 1.6, 9); b.arrow(682, 252, 718, 266, INK, 1.6, 9)
    b.dot(730, 234, 6); b.dot(730, 270, 6)
    b.label(660, 286, "2. births from the survivors (Poisson)", 18, w=200)
    b.arrow(662, 350, bx + bw + 6, 360, INK, 2.0, bow=8)
    # 3 crowding
    b.dot(90, 480, 7); b.dot(140, 480, 7, BLUE)
    b.arrow(104, 480, 126, 480, INK, 1.6, 8); b.arrow(126, 480, 104, 480, BLUE, 1.6, 8)
    b.label(70, 500, "3. competition from every other species, 0.4 each", 18, w=215)
    b.arrow(290, 486, bx - 6, 470, INK, 2.0, bow=8)
    # 4 arrivals
    for k in range(3):
        b.rect(690 + k * 52, 452, 42, 42, PENCIL, 1.4, rough=0.4)
        b.dot(711 + k * 52, 473, 5)
    b.label(680, 504, "4. arrivals from other cells (Poisson)", 18, w=200)
    b.arrow(684, 470, bx + bw + 6, 476, INK, 2.0)
    b.label(80, 588, "expected change:  B<sub>i</sub>  becomes  B<sub>i</sub> x exp(r<sub>i</sub> - sum of C<sub>ij</sub> B<sub>j</sub>)  +  arrivals d<sub>i</sub> x K x B<sub>i</sub>", 19, BLUE, w=780)
    b.label(80, 630, "a species is lost only at 0 individuals; it counts as present from 1", 19, INK, w=780)
    b.label(80, 660, "recording (survey and detection) is added later, only when a community is compared or exported (Part 3)", 18, PENCIL, w=780)
    cards(b, [
        ("The old code deleted a species with fewer than tau individuals in every cell (tau = the detection threshold), so low-density species vanished and the observation rule changed the community.",
         "Loss only at 0 individuals; tau is used for observation only."),
        ("At the old connectance 0.4, 36% of species pairs never competed, and cells held too many species (found by the gamma = alpha x beta split).",
         "Connectance 1: every pair competes at 0.4."),
    ])
    bottom(b, ["survival Binomial; births, arrivals Poisson", "births from survivors: mean growth exactly e^g",
               "one individual = body mass 1e-4", "C = 0.4 for every pair; C_ii = 1",
               "lost at 0; present from 1 individual"],
           "Every pair competes at 0.4 whatever their niche overlap. A rule tying competition to overlap was tested and was not the fix (row 13d). Competition sorting: Part 5.")
    footer(b, "Sheet 4 · the four events of one IBM step in one cell")
    return b.html("Sheet 4 · Individuals and competition: one IBM step",
                  "One cell as a box with dots for individuals of three species; four arrows in: survive by coin flip, births, "
                  "competition from every other species, arrivals from other cells; the update rule written below.")


# ------------------------------------------------------------------ sheet 5
def sheet5():
    b = start(205, 5, "Dispersal",
              "Each step a cell sends d x B of each species to other cells along a kernel. "
              "The home cell loses nothing (propagule dispersal, as in LVMCM).")
    heading(b, 50, 164, "How we did it")
    c = 26
    ox, oy = 70, 250
    b.label(ox, oy - 34, "old", 26, PENCIL, font="title")
    b.grid(ox, oy, 7, c, PENCIL, 0.7, INK, 1.6)
    b.fill_cell(ox, oy, c, 3, 3, INK, 0.75)
    cx, cy = ox + 3.5 * c, oy + 3.5 * c
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        b.arrow(cx + dx * 9, cy + dy * 9, cx + dx * (c + 3), cy + dy * (c + 3), INK, 1.6, 7)
    b.label(ox, oy + 7 * c + 8, "4 next cells only, plus an even share to every far cell", 18, w=190)
    b.label(ox, oy + 7 * c + 76, "d = 2e-8, never used", 19, DRED, w=190)
    b.cross(ox + 7 * c + 26, oy + 30, 12, DRED, 2.6)
    # now
    nx_, ny_ = 470, 360
    b.label(330, 216, "now", 26, BLUE, font="title")
    for rad, op in ((120, 0.07), (90, 0.12), (60, 0.2), (30, 0.34)):
        b.ellipse(nx_, ny_, rad, rad, PENCIL, 1.1, fill=ORANGE, fop=op, passes=1, rough=0.3)
    b.rect(nx_ - 9, ny_ - 9, 18, 18, INK, 1.8, fill=INK, fop=0.8, rough=0.3)
    b.arrow(nx_ + 12, ny_ - 8, 690, 250, BLUE, 1.8, 10, bow=-14)
    b.arrow(nx_ + 12, ny_ + 10, 676, 470, BLUE, 1.8, 10, bow=12)
    b.label(620, 214, "a few go far", 18, BLUE)
    b.label(340, 492, "2Dt kernel: weight (1 + r²/a²)<sup>-b</sup>; most land near, a few far", 18, w=360)
    b.label(700, 300, "torus distances; periodic images summed; home cell left out", 17, PENCIL, w=170)
    # ranges
    for k, (ttl, cells) in enumerate((
            ("slow disperser: tight clump", [(4, 4), (5, 4), (4, 5), (5, 5), (3, 5), (5, 6), (4, 3)]),
            ("fast disperser: spread out", [(1, 1), (2, 2), (4, 1), (6, 3), (5, 5), (8, 4), (2, 7), (7, 8), (4, 6), (8, 7), (1, 5)]))):
        gx = 70 + k * 270
        b.grid(gx, 576, 10, 10, PENCIL, 0.5, INK, 1.3)
        b.label(gx + 110, 590, esc(ttl), 17, w=140)
        for cc, rr in cells:
            b.fill_cell(gx, 576, 10, cc, rr, ORANGE, 0.85)
    b.arrow(620, 650, 870, 650, INK, 1.8)
    for i in range(5):
        x = 630 + i * 58
        b.line(x, 643, x, 657, INK, 1.4, 1)
    b.label(604, 662, "1e-6", 16); b.label(836, 662, "1e-2", 16)
    b.label(620, 590, "d per species, log-uniform, own random stream", 17, w=250)
    cards(b, [
        ("The old d was BODY_MASS x 0.0002 = 2e-8 per individual per step, built into a matrix once at import, so the d given on the command line never applied (audit T1, test_01, test_05).",
         "dispersal.py v2 reads d at every call."),
        ("Propagules went only to the 4 neighbours, plus a uniform long-distance share. Ranges were frozen and each cell was an island.",
         "2Dt kernel (Clark et al. 1999, Ecology 80: 1475-1494), a = 1, b = 3."),
        ("One d for all species (found by similarity against distance, test_13 K3b).",
         "d drawn per species from 1e-6 to 1e-2 (range from test_14)."),
    ])
    bottom(b, ["kernel 2Dt, a = 1, b = 3", "d log-uniform 1e-6 to 1e-2", "low end: first settled v2 test communities",
               "high end: LVMCM example rate 0.01", "d read at every step"],
           "Nothing on the mechanism. The 2Dt tail carries rare long-distance dispersal, so no separate share is needed (checked 30 Oct).")
    footer(b, "Sheet 5 · old and v2 kernels; two example ranges (drawing, not data)")
    return b.html("Sheet 5 · Dispersal",
                  "Old: a centre cell with four short arrows, crossed out. Now: rings fading out from the home cell and two long arrows. "
                  "Below: a tight clump and a spread-out range, and a log scale for d.")


SHEETS = [("Sheet01.dc.html", sheet1), ("Sheet02.dc.html", sheet2), ("Sheet03.dc.html", sheet3),
          ("Sheet04.dc.html", sheet4), ("Sheet05.dc.html", sheet5)]
