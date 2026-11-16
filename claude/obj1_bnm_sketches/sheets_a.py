"""Batch 1: start-here board and sheets 1-5 (Objective 1 simulator)."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from sketch import Board, INK, PENCIL, BLUE, ORANGE, RED, esc


def habitat(seed, n=20, sigma=1.4):
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(size=(n, n)), sigma, mode="wrap")
    return (f - f.mean()) / f.std()


# ---------------------------------------------------------------- start here
def main():
    b = Board(100)
    b.header(None, "Objective 1 and BNM in pictures",
             "One sheet per step. Copy the picture, then say the words under it.")
    # story map
    b.card(64, 184, 240, 80, "old simulator", "ranges stuck where they began", stroke=PENCIL)
    b.card(64, 312, 240, 80, "fix the simulator", "sheets 1 to 10", fill=ORANGE, fop=0.18)
    b.card(64, 440, 240, 80, "150 simulated communities", "they train the Objective 2 model")
    b.card(352, 440, 224, 80, "BNM butterfly records", "sheets 11 to 14", fill=BLUE, fop=0.12)
    b.card(168, 576, 304, 80, "look at both the same way, then compare", "sheets 15 to 17")
    b.card(64, 696, 200, 56, "what we fixed: Part 4")
    b.card(376, 696, 200, 56, "still open: Part 5", fill=BLUE, fop=0.12)
    b.arrow(184, 268, 184, 308)
    b.arrow(184, 396, 184, 436)
    b.arrow(184, 524, 184, 572)
    b.arrow(464, 524, 464, 572)
    b.arrow(240, 660, 180, 692)
    b.arrow(400, 660, 460, 692)
    # word key
    b.label(648, 160, "Words on these sheets", 30, INK, font="title")
    b.line(648, 194, 900, 196, INK, 1.6, 1)
    rows = [
        ("community", "one simulated landscape with its species: 20 x 20 cells, the size of one BNM window (40 km)"),
        ("cell", "one square of the grid = one BNM tetrad (2 km x 2 km)"),
        ("round", "newcomers arrive, then the community runs for 5,000 steps"),
        ("piece", "200 steps = one BNM year; 25 pieces make one round"),
        ("footprint", "the cells where a species can grow on its own"),
        ("fit half, test half", "western BNM windows for tuning; eastern ones opened once"),
    ]
    y0 = 214
    for i, (word, text) in enumerate(rows):
        y = y0 + i * 88
        dx, dy = 652, y + 6
        if i == 0:
            b.grid(dx, dy, 6, 10, PENCIL, 0.8, INK, 1.6)
        elif i == 1:
            b.rect(dx + 12, dy + 6, 40, 40, INK, 2.0, fill=ORANGE, fop=0.25)
        elif i == 2:
            pts = [(dx + 30 + 22 * math.cos(a), dy + 30 + 22 * math.sin(a))
                   for a in np.linspace(-1.2, 4.3, 9)]
            b.curve(pts, INK, 2.0, 2, 0.4)
            b.head(pts[-1][0], pts[-1][1], pts[-2][0], pts[-2][1], INK, 2.0, 11)
        elif i == 3:
            for k in range(5):
                b.rect(dx + k * 13, dy + 18, 13, 22, INK, 1.4, rough=0.4, passes=1)
            b.fill_cell(dx + 26, dy + 18, 13, 0, 0, ORANGE, 0.8)
        elif i == 4:
            b.grid(dx, dy, 6, 10, PENCIL, 0.8, INK, 1.6)
            for c, r in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (4, 4)]:
                b.fill_cell(dx, dy, 10, c, r, ORANGE, 0.8)
        else:
            b.rect(dx, dy + 6, 64, 44, INK, 1.8)
            b.line(dx + 32, dy + 2, dx + 32, dy + 54, BLUE, 2.2, 1)
            b.padlock(dx + 48, dy + 28, 0.55)
        b.label(740, y, esc(word), 23, BLUE)
        b.label(740, y + 28, esc(text), 19, INK, w=330, lh=1.15)
    b.footer("Start here · drawn for teaching, not data")
    return b.html("Objective 1 and BNM in pictures",
                  "Story map: old simulator, fix the simulator, 150 communities and BNM records, compare, then fixed and still open. "
                  "Beside it, small drawings for the words community, cell, round, piece, footprint and the fit and test halves.")


# ---------------------------------------------------------------- sheet 1
def sheet1():
    b = Board(101)
    b.header(1, "Why we rebuilt the simulator",
             "The old maps gave the AI nothing to learn more than 2 cells away from a record.")
    c = 15
    # old grid
    ox, oy = 96, 226
    b.label(ox, oy - 40, "old", 30, PENCIL, font="title")
    b.grid(ox, oy, 20, c)
    patches = [[(6, 8), (7, 8), (6, 9)], [(15, 3), (16, 3)], [(3, 16), (3, 17), (4, 17)],
               [(16, 14), (17, 14), (17, 15)], [(11, 18), (12, 18)]]
    for p in patches:
        for cc, rr in p:
            b.fill_cell(ox, oy, c, cc, rr, ORANGE, 0.75)
    rx, ry = ox + 6.5 * c, oy + 8.5 * c
    for k, rad in enumerate((1.5, 2.5, 3.5)):
        b.ellipse(rx, ry, rad * c, rad * c, BLUE, 1.5, passes=1, rough=0.6)
    b.dot(rx, ry, 5, BLUE)
    b.label(rx + 3.6 * c + 2, ry - 12, "1, 2, 3 cells", 16, BLUE)
    b.arrow(ox + 300 + 50, oy + 40, ox + 16.5 * c + 4, oy + 3.6 * c, PENCIL, 1.8, bow=-14)
    b.label(ox + 300 + 22, oy - 10, "far patches sit where chance dropped them", 18, INK, w=200)
    b.label(ox, oy + 312, "beyond 2 cells: nothing to learn", 21, INK)
    # middle arrow
    b.arrow(452, 376, 610, 376, INK, 2.6, 16)
    b.label(452, 310, "fix the simulator first", 19, INK, w=160, align="center")
    b.label(452, 392, "then retrain the model", 19, INK, w=160, align="center")
    # v2 grid
    vx, vy = 650, 226
    b.label(vx, vy - 40, "v2", 30, BLUE, font="title")
    h = habitat(11)
    good = h > np.quantile(h, 0.62)
    b.grid(vx, vy, 20, c)
    for r_ in range(20):
        for c_ in range(20):
            if good[r_, c_]:
                b.hatch_rect(vx + c_ * c, vy + r_ * c, c, c, 5, 45, PENCIL, 1.0, 0.8)
    rr0, cc0 = np.unravel_index(np.argmax(h), h.shape)
    rng_cells = []
    for r_ in range(20):
        for c_ in range(20):
            dr = min(abs(r_ - rr0), 20 - abs(r_ - rr0))
            dc = min(abs(c_ - cc0), 20 - abs(c_ - cc0))
            if h[r_, c_] > np.quantile(h, 0.72) and max(dr, dc) <= 4:
                rng_cells.append((c_, r_))
    for cc, rr in rng_cells:
        b.fill_cell(vx, vy, c, cc, rr, ORANGE, 0.8)
    b.dot(vx + (cc0 + 0.5) * c, vy + (rr0 + 0.5) * c, 5, BLUE)
    b.label(vx, vy + 312, "habitat and spread explain the range", 21, INK)
    # legend
    lx, ly = 650, 590
    b.dot(lx + 8, ly + 12, 5, BLUE)
    b.label(lx + 24, ly, "a record", 18, INK)
    b.fill_cell(lx + 120, ly + 4, 16, 0, 0, ORANGE, 0.8)
    b.label(lx + 144, ly, "where the species lives", 18, INK)
    b.hatch_rect(lx + 330, ly + 4, 16, 16, 5, 45, PENCIL, 1.0)
    b.rect(lx + 330, ly + 4, 16, 16, PENCIL, 1.2, rough=0.4, passes=1)
    b.label(lx + 354, ly, "good habitat", 18, INK)
    b.note(96, 636, 520, 104, "old",
           "The dispersal rate was never used, so ranges froze where they began. "
           "Each species began in 5 random cells.")
    b.footer("Sheet 1 · drawing, not data")
    return b.html("Sheet 1 · Why we rebuilt the simulator",
                  "Two 20 by 20 grids. Old: five scattered patches, one record with rings at 1, 2 and 3 cells. "
                  "v2: shaded good habitat with one clumped range on it and a record inside.")


# ---------------------------------------------------------------- sheet 2
def sheet2():
    b = Board(102)
    b.header(2, "One habitat map for every species",
             "Every cell has one habitat value. All species read the same map.")
    c = 22
    gx, gy = 96, 202
    h = habitat(5)
    q1, q2 = np.quantile(h, [1 / 3, 2 / 3])
    for r_ in range(20):
        for c_ in range(20):
            x, y = gx + c_ * c, gy + r_ * c
            if h[r_, c_] > q2:
                b.hatch_rect(x, y, c, c, 5, 45, INK, 1.1, 0.85)
                b.hatch_rect(x, y, c, c, 5, -45, INK, 1.1, 0.85)
            elif h[r_, c_] > q1:
                b.hatch_rect(x, y, c, c, 6, 45, PENCIL, 1.0, 0.75)
    b.grid(gx, gy, 20, c, PENCIL, 0.7)
    # wrap arrows
    b.arrow(gx + 440 + 6, gy + 5.5 * c, gx + 440 + 46, gy + 5.5 * c, BLUE, 2.2)
    b.arrow(gx - 44, gy + 5.5 * c, gx - 6, gy + 5.5 * c, BLUE, 2.2)
    b.label(gx - 2, gy - 34, "the edges join up: walk off the right, come back on the left", 18, BLUE)
    # legend
    ly = gy + 440 + 16
    b.hatch_rect(gx, ly + 4, 18, 18, 5, 45, INK, 1.1)
    b.hatch_rect(gx, ly + 4, 18, 18, 5, -45, INK, 1.1)
    b.rect(gx, ly + 4, 18, 18, PENCIL, 1.2, rough=0.4, passes=1)
    b.label(gx + 26, ly, "good", 19)
    b.hatch_rect(gx + 100, ly + 4, 18, 18, 6, 45, PENCIL, 1.0)
    b.rect(gx + 100, ly + 4, 18, 18, PENCIL, 1.2, rough=0.4, passes=1)
    b.label(gx + 126, ly, "medium", 19)
    b.rect(gx + 216, ly + 4, 18, 18, PENCIL, 1.2, rough=0.4, passes=1)
    b.label(gx + 242, ly, "poor", 19)
    b.label(gx, ly + 34, "patches about 3 cells wide; species feel a softened map (0.25 x H)", 18, PENCIL, w=470)
    # right: drift
    px = 620
    b.label(px, 176, "the map drifts slowly", 26, INK, font="title")
    blob1 = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3)]
    for k, sx in enumerate((px, px + 190)):
        b.grid(sx, 216, 6, 14, PENCIL, 0.7, INK, 1.6)
        for cc, rr in blob1:
            b.hatch_rect(sx + (cc + k) * 14, 216 + rr * 14, 14, 14, 4.5, 45, INK, 1.0)
    b.arrow(px + 96, 258, px + 178, 258, INK, 2.0, bow=-10)
    b.label(px + 300, 222, "time scale: 62,500 steps", 18, PENCIL, w=150)
    # right: layer shift
    b.label(px, 330, "the whole layer moves up and down", 26, INK, font="title")
    xs = np.linspace(px, px + 300, 9)
    base = [430 + 18 * math.sin(i * 0.9) for i in range(9)]
    b.curve(list(zip(xs, base)), INK, 2.2)
    b.curve(list(zip(xs, [v - 44 for v in base])), BLUE, 2.0, 1, dash="7 6")
    b.line(px, 480, px + 300, 480, PENCIL, 1.4, 1)
    b.label(px + 250, 484, "cells", 16, PENCIL)
    b.arrow(px + 340, 446, px + 340, 386, BLUE, 2.0)
    b.arrow(px + 340, 404, px + 340, 464, BLUE, 2.0)
    b.label(px + 360, 392, "with the drift, this lets residents be lost (sheet 7)", 18, INK, w=140)
    b.note(px, 536, 450, 112, "old",
           "Each species had its own random map, drawn again every round. "
           "When one species died, the next got a new map.")
    b.note(px, 664, 450, 76, "open", "Habitat is one axis only (Part 5).")
    b.footer("Sheet 2 · pattern drawn for teaching, not a saved map")
    return b.html("Sheet 2 · One habitat map for every species",
                  "A 20 by 20 grid shaded good, medium and poor in patches about 3 cells wide, with arrows showing the edges join. "
                  "Beside it: a patch drifting slowly, and a habitat line moving up and down.")


# ---------------------------------------------------------------- sheet 3
def sheet3():
    b = Board(103)
    b.header(3, "Each species has a niche",
             "A species grows where the habitat suits it: a bell curve minus a fixed cost.")
    x0, x1, yb, yt = 110, 720, 520, 210
    Hmin, Hmax = -3.0, 3.0
    ymax = 1.6

    def X(hv):
        return x0 + (hv - Hmin) / (Hmax - Hmin) * (x1 - x0)

    def Y(v):
        return yb - v / ymax * (yb - yt)

    b.arrow(x0, yb, x1 + 30, yb, INK, 2.2)
    b.arrow(x0, yb, x0, yt - 20, INK, 2.2)
    b.label(x1 - 170, yb + 8, "habitat: poor to good", 19, INK)
    b.label(x0 - 6, yt - 56, "growth", 19, INK)
    c = 0.1
    sp = dict(opt=1.0, w=0.3)
    ge = dict(opt=-0.8, w=1.0)
    for s in (sp, ge):
        s["peak"] = 1.5 * (0.2 / s["w"]) ** 0.5
    for s, col in ((ge, BLUE), (sp, ORANGE)):
        hs = np.linspace(Hmin, Hmax, 61)
        pts = [(X(v), Y(s["peak"] * math.exp(-(v - s["opt"]) ** 2 / (2 * s["w"] ** 2)))) for v in hs]
        b.curve(pts, col, 2.6, 2, 0.5)
    b.line(x0, Y(c), x1, Y(c), INK, 1.6, 1, dash="8 6")
    b.label(x1 + 6, Y(c) - 30, "cost", 18, INK)
    b.label(X(1.0) + 30, Y(sp["peak"]) - 4, "specialist: narrow niche, tall peak", 19, INK, w=170)
    b.label(X(-0.8) - 120, Y(ge["peak"]) - 58, "generalist: wide niche, low peak", 19, INK, w=230)
    # footprints
    for s, col, yy in ((ge, BLUE, yb + 48), (sp, ORANGE, yb + 66)):
        half = s["w"] * math.sqrt(2 * math.log(s["peak"] / c))
        a, z = X(s["opt"] - half), X(s["opt"] + half)
        b.line(a, yy, z, yy, col, 7, 1, 0.5)
    b.label(x0, yb + 80, "footprint (the bars) = where growth beats the cost (r > 0)", 19, INK)
    # right panel
    px = 790
    b.label(px, 200, "the trade-off", 28, INK, font="title")
    b.label(px, 240, "narrow niche, taller peak; wide niche, lower peak", 19, INK, w=280)
    b.label(px, 300, "peak = 1.5 x (0.2 / width)<sup>0.5</sup>", 19, BLUE, w=280)
    b.label(px, 344, "width 0.2 to 1.0<br>cost 0.1<br>peak at most 1.5", 19, PENCIL, w=280)
    b.label(px, 432, "each species keeps its niche for life, wherever it sits in the list", 19, INK, w=280)
    b.note(110, 632, 470, 108, "old",
           "Every species grew at the same rate (r = 1) on its own random patch, so species were near copies.")
    b.note(604, 632, 466, 108, "open",
           "On the softened map even the narrowest niche spans most habitats, so a rare-habitat specialist cannot win its cells.")
    b.footer("Sheet 3 · curves from the formula, example traits")
    return b.html("Sheet 3 · Each species has a niche",
                  "Growth against habitat: a tall narrow bell for a specialist and a low wide bell for a generalist, "
                  "a dashed cost line, and bars under the axis where each bell is above the cost.")


# ---------------------------------------------------------------- sheet 4
def sheet4():
    b = Board(104)
    b.header(4, "One time step in one cell",
             "We count individuals, so a small population can die out by chance.")
    bx, by, bw, bh = 381, 236, 360, 300
    b.rect(bx, by, bw, bh, INK, 2.6)
    b.label(bx + 8, by + bh + 8, "one cell", 20, PENCIL)
    rng = np.random.default_rng(4)
    placed = []

    def free(x, y):
        return all(math.hypot(x - a, y - c_) > 26 for a, c_ in placed)

    for kind, n in (("ink", 12), ("open", 8), ("blue", 5)):
        k = 0
        while k < n:
            x, y = bx + 26 + rng.random() * (bw - 52), by + 26 + rng.random() * (bh - 52)
            if free(x, y):
                placed.append((x, y))
                k += 1
                if kind == "ink":
                    b.dot(x, y, 6, INK)
                elif kind == "open":
                    b.ring(x, y, 6, INK)
                else:
                    b.dot(x, y, 6, BLUE)
    # 1 survive
    b.ellipse(120, 236, 30, 30, INK, 2.2, fill=ORANGE, fop=0.25)
    b.label(102, 212, "?", 38, INK, w=36, align="center", font="title", lh=1.0)
    b.label(164, 196, "1. Survive? A coin flip for each individual", 20, INK, w=200)
    b.arrow(300, 270, bx - 6, 292, INK, 2.2, bow=-10)
    # 2 births
    b.dot(824, 230, 6)
    b.arrow(840, 230, 884, 216, INK, 1.8, 10)
    b.arrow(840, 230, 884, 246, INK, 1.8, 10)
    b.dot(898, 212, 6)
    b.dot(898, 250, 6)
    b.label(800, 272, "2. Births: new young come from the survivors", 20, INK, w=250)
    b.arrow(812, 336, bx + bw + 6, 330, INK, 2.2, bow=10)
    # 3 crowding
    b.dot(110, 470, 7)
    b.dot(160, 470, 7, BLUE)
    b.arrow(124, 470, 146, 470, INK, 1.6, 8)
    b.arrow(146, 470, 124, 470, BLUE, 1.6, 8)
    b.label(80, 498, "3. Crowding: every other species pushes back, 0.4 each", 20, INK, w=270)
    b.arrow(310, 470, bx - 6, 456, INK, 2.2, bow=8)
    # 4 arrivals
    for k in range(3):
        b.rect(880 + k * 54, 440, 44, 44, PENCIL, 1.6, rough=0.5)
        b.dot(902 + k * 54, 462, 5)
    b.label(870, 498, "4. Arrivals from other cells", 20, INK, w=200)
    b.arrow(872, 462, bx + bw + 6, 452, INK, 2.2)
    b.label(bx - 20, by + bh + 44, "gone only at 0 individuals; present from 1", 21, BLUE, w=bw + 40, align="center")
    b.note(96, 632, 474, 108, "old",
           "Species below the detection line were deleted, so the way we look changed what lived.")
    b.note(596, 632, 474, 108, "open",
           "Every pair competes at 0.4, however alike their niches are (Part 5).")
    b.footer("Sheet 4 · the four events of one step")
    return b.html("Sheet 4 · One time step in one cell",
                  "One cell drawn as a box holding dots for individuals of three species, with four arrows in: "
                  "survive by coin flip, births, crowding from other species, arrivals from other cells.")


# ---------------------------------------------------------------- sheet 5
def sheet5():
    b = Board(105)
    b.header(5, "How young spread to other cells",
             "Most land near home, a few land far away. Each species has its own rate.")
    # old
    c = 32
    ox, oy = 96, 214
    b.label(ox, oy - 42, "old", 30, PENCIL, font="title")
    b.grid(ox, oy, 7, c, PENCIL, 0.8, INK, 1.8)
    b.fill_cell(ox, oy, c, 3, 3, INK, 0.75)
    cx, cy = ox + 3.5 * c, oy + 3.5 * c
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        b.arrow(cx + dx * 10, cy + dy * 10, cx + dx * (c + 4), cy + dy * (c + 4), INK, 1.8, 8)
    b.label(ox, oy + 7 * c + 12, "only to the 4 next cells, plus a thin even share to every far cell", 19, INK, w=240)
    b.label(ox + 7 * c + 20, oy + 40, "rate 2e-8, and it was never used", 19, RED, w=130)
    b.cross(ox + 7 * c + 84, oy + 120, 16, RED, 3)
    # now
    nx_, ny_ = 720, 340
    b.label(560, 172, "now", 30, BLUE, font="title")
    for rad, op in ((140, 0.07), (104, 0.12), (68, 0.2), (34, 0.34)):
        b.ellipse(nx_, ny_, rad, rad, PENCIL, 1.2, fill=ORANGE, fop=op, passes=1, rough=0.35)
    b.rect(nx_ - 10, ny_ - 10, 20, 20, INK, 2.0, fill=INK, fop=0.8, rough=0.4)
    b.arrow(nx_ + 14, ny_ - 8, 1030, 222, BLUE, 2.0, bow=-20)
    b.arrow(nx_ + 14, ny_ + 12, 1000, 466, BLUE, 2.0, bow=16)
    b.label(900, 160, "a few go far", 20, BLUE)
    b.label(560, 494, "most land near home; sending young costs the home cell nothing", 19, INK, w=330)
    # ranges
    c2 = 10
    for k, (ttl, cells) in enumerate((
            ("slow mover: tight clump", [(4, 4), (5, 4), (4, 5), (5, 5), (3, 5), (5, 6), (4, 3)]),
            ("fast mover: spread out", [(1, 1), (2, 2), (4, 1), (6, 3), (5, 5), (8, 4), (2, 7), (7, 8), (4, 6), (8, 7), (1, 5)]))):
        gx = 96 + k * 250
        b.grid(gx, 548, 10, c2, PENCIL, 0.6, INK, 1.4)
        for cc, rr in cells:
            b.fill_cell(gx, 548, c2, cc, rr, ORANGE, 0.85)
        b.label(gx + 112, 560, esc(ttl), 18, INK, w=120)
    # rate scale
    b.arrow(620, 600, 1040, 600, INK, 2.0)
    for i in range(5):
        x = 640 + i * 95
        b.line(x, 592, x, 608, INK, 1.6, 1)
    b.label(608, 614, "0.000001", 17, INK)
    b.label(990, 614, "0.01", 17, INK)
    b.label(640, 556, "each species draws its own rate (log scale)", 19, INK)
    b.note(96, 664, 974, 76, "Kernel",
           "2Dt kernel (Clark et al. 1999), a = 1, b = 3: its fat tail gives the rare long jumps, so no extra rule is needed.")
    b.footer("Sheet 5 · old and new spread; two example ranges (drawing, not data)")
    return b.html("Sheet 5 · How young spread to other cells",
                  "Old: a centre cell with four short arrows to its neighbours, crossed out. "
                  "Now: rings fading outward from the home cell and two long arrows. Below: a tight clump and a spread-out range, and a log scale of rates.")


SHEETS = [("Main.dc.html", main), ("Sheet01.dc.html", sheet1), ("Sheet02.dc.html", sheet2),
          ("Sheet03.dc.html", sheet3), ("Sheet04.dc.html", sheet4), ("Sheet05.dc.html", sheet5)]

if __name__ == "__main__":
    import os
    import sys
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, fn in SHEETS:
        with open(os.path.join(out, name), "w") as f:
            f.write(fn())
        print(name)
