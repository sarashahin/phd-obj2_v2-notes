"""Sheet template (1400 x 990): header, How we did it, Problems and fixes, Settings, Still open."""
import math
from sketch import Board, INK, PENCIL, BLUE, ORANGE, RED, GREEN, DRED, esc

SW, SH = 1400, 990
LEFT = (50, 205, 880, 700)      # drawing area x0, y0, x1, y1
CARD_X, CARD_W = 918, 432


def wrap_lines(text, px_width, size=19):
    """Rough line count for Patrick Hand: about 0.39 x size px per character."""
    cpl = max(10, int(px_width / (0.40 * size)))
    words, lines, cur = text.split(), 0, 0
    for w_ in words:
        add = len(w_) + (1 if cur else 0)
        if cur + add > cpl:
            lines += 1
            cur = len(w_)
        else:
            cur += add
    return lines + (1 if cur else 0)


def tick(b, x, y, col=BLUE, s=1.0):
    b.line(x, y + 6 * s, x + 5 * s, y + 12 * s, col, 2.6, 1, 0.3)
    b.line(x + 5 * s, y + 12 * s, x + 15 * s, y - 2 * s, col, 2.6, 1, 0.3)


def xmark(b, x, y, col=DRED, s=1.0):
    b.line(x, y, x + 12 * s, y + 12 * s, col, 2.6, 1, 0.3)
    b.line(x, y + 12 * s, x + 12 * s, y, col, 2.6, 1, 0.3)


def start(seed, num, title, what):
    b = Board(seed, SW, SH)
    big = len(str(num)) > 2 or not str(num).isdigit()
    b.ellipse(78, 70, 34 if big else 30, 28, INK, 2.4)
    b.label(44, 45, str(num), 34 if big else 40, INK, w=68, align="center", font="title", lh=1.0)
    b.label(126, 32, esc(title), 52, INK, font="title", lh=1.0)
    b.label(126, 96, esc(what), 23, INK, w=1220, lh=1.25)
    return b


def heading(b, x, y, text, col=INK):
    b.label(x, y, esc(text), 32, col, font="title", lh=1.0)
    b.line(x, y + 38, x + 9.5 * len(text) + 20, y + 39, col, 1.6, 1, 0.6)


def cards(b, pairs, y0=220, y_max=950, size=19):
    """pairs: list of (problem, fix). Problem text gets a red cross, fix text a blue tick."""
    heading(b, CARD_X, 164, "Problems and fixes")
    tw = CARD_W - 54
    y = y0
    for prob, fix in pairs:
        lp = wrap_lines(prob, tw, size)
        lf = wrap_lines(fix, tw, size)
        lh = size * 1.22
        h = int(16 + lp * lh + 10 + lf * lh + 14)
        b.rect(CARD_X, y, CARD_W, h, PENCIL, 1.5, fill="#fffdf8", fop=1.0, rough=0.6)
        xmark(b, CARD_X + 14, y + 16)
        b.label(CARD_X + 40, y + 12, esc(prob), size, INK, w=tw, lh=1.22)
        yf = y + 16 + lp * lh + 10
        tick(b, CARD_X + 13, yf + 2)
        b.label(CARD_X + 40, yf - 4, esc(fix), size, INK, w=tw, lh=1.22)
        y += h + 14
    if y > y_max + 14:
        raise ValueError(f"cards overflow: ends at {y}")
    return y


def bottom(b, settings, open_text, y=724, h=230):
    """settings: list of short lines (or None); open_text: string."""
    if settings:
        b.rect(50, y, 400, h, PENCIL, 1.6, fill="#f1ece1", fop=1.0, rough=0.6)
        b.label(66, y + 10, "Settings", 28, INK, font="title", lh=1.0)
        b.label(66, y + 46, "<br>".join(esc(s) for s in settings), 18, INK, w=368, lh=1.28)
        ox, ow = 470, 410
    else:
        ox, ow = 50, 830
    b.rect(ox, y, ow, h, BLUE, 1.6, fill="#e3ebf6", fop=1.0, rough=0.6)
    b.label(ox + 16, y + 10, "Still open", 28, INK, font="title", lh=1.0)
    b.label(ox + 16, y + 46, esc(open_text), 19, INK, w=ow - 32, lh=1.25)


def footer(b, text):
    b.label(880, 962, esc(text), 15, PENCIL, w=470, align="right", lh=1.0)


def step_cards(b, items, x0, y, w, h, gap=34, fill=None, fop=0.14, size=19, numbered=True, arrows=True):
    """A row of cards joined by arrows; items: list of strings."""
    xs = []
    for i, t in enumerate(items):
        x = x0 + i * (w + gap)
        label = f"{i + 1}. {t}" if numbered else t
        b.card(x, y, w, h, label, stroke=INK, fill=fill, fop=fop, size=size)
        xs.append(x)
        if i and arrows:
            b.arrow(x - gap + 4, y + h / 2, x - 4, y + h / 2, INK, 2.0, 10)
    return xs
