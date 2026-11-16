"""Hand-drawn style SVG helpers for the Objective 1 sketch sheets.

Every stroke gets a small seeded wobble, so a sheet redraws the same way each time.
"""
import math
import random

INK = "#2a2926"
PENCIL = "#66615a"
BLUE = "#2f5fa7"
ORANGE = "#e0892e"
RED = "#b4432a"
PAPER = "#fbf8f1"
W, H = 1400, 990
GREEN = "#4f8a4b"
DRED = "#a3381f"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Board:
    def __init__(self, seed, w=None, h=None):
        self.W = w or W
        self.H = h or H
        self.r = random.Random(seed)
        self.paths = []
        self.labels = []

    def j(self, a):
        return self.r.uniform(-a, a)

    # ---------- raw path ----------
    def _p(self, d, stroke="none", w=2.0, op=1.0, dash=None, fill="none", fop=None):
        s = f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{w:.2f}" stroke-linecap="round" stroke-linejoin="round"'
        if op != 1:
            s += f' opacity="{op:.2f}"'
        if fop is not None:
            s += f' fill-opacity="{fop:.2f}"'
        if dash:
            s += f' stroke-dasharray="{dash}"'
        self.paths.append(s + "></path>")

    # ---------- strokes ----------
    def line(self, x1, y1, x2, y2, stroke=INK, w=2.2, passes=2, rough=1.0, dash=None, over=0.0):
        L = math.hypot(x2 - x1, y2 - y1) or 1.0
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        x1, y1, x2, y2 = x1 - ux * over, y1 - uy * over, x2 + ux * over, y2 + uy * over
        nx, ny = -uy, ux
        amp = min(2.2, 0.3 + L * 0.006) * rough
        for p in range(passes):
            ax, ay = x1 + self.j(rough * 0.8), y1 + self.j(rough * 0.8)
            bx, by = x2 + self.j(rough), y2 + self.j(rough)
            bow = self.j(amp)
            c1x = ax + (bx - ax) * 0.33 + nx * bow + self.j(rough * 0.5)
            c1y = ay + (by - ay) * 0.33 + ny * bow + self.j(rough * 0.5)
            c2x = ax + (bx - ax) * 0.67 + nx * bow * 0.7 + self.j(rough * 0.5)
            c2y = ay + (by - ay) * 0.67 + ny * bow * 0.7 + self.j(rough * 0.5)
            d = f"M{ax:.1f} {ay:.1f} C{c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} {bx:.1f} {by:.1f}"
            self._p(d, stroke, w if p == 0 else w * 0.45, 1.0 if p == 0 else 0.4, dash)

    @staticmethod
    def _smooth(pts, closed=False):
        if closed:
            pts = pts + pts[:3]
        d = f"M{pts[0][0]:.1f} {pts[0][1]:.1f}"
        n = len(pts)
        rng = range(0, n - 3) if closed else range(0, n - 1)
        if closed:
            d = f"M{pts[1][0]:.1f} {pts[1][1]:.1f}"
        for i in rng:
            if closed:
                p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
            else:
                p0 = pts[i - 1] if i > 0 else pts[i]
                p1, p2 = pts[i], pts[i + 1]
                p3 = pts[i + 2] if i + 2 < n else pts[i + 1]
            c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
            c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
            d += f" C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}"
        return d

    def curve(self, pts, stroke=INK, w=2.2, passes=2, rough=1.0, dash=None):
        for p in range(passes):
            q = [(x + self.j(rough), y + self.j(rough)) for x, y in pts]
            self._p(self._smooth(q), stroke, w if p == 0 else w * 0.5, 1.0 if p == 0 else 0.55, dash)

    def rect(self, x, y, w, h, stroke=INK, sw=2.2, fill=None, fop=1.0, rough=1.0, passes=2):
        if fill:
            q = [(x + self.j(2), y + self.j(2)), (x + w + self.j(2), y + self.j(2)),
                 (x + w + self.j(2), y + h + self.j(2)), (x + self.j(2), y + h + self.j(2))]
            d = "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in q) + " Z"
            self._p(d, fill=fill, fop=fop)
        if stroke:
            o = 3 * rough
            self.line(x, y, x + w, y, stroke, sw, passes, rough, over=o)
            self.line(x + w, y, x + w, y + h, stroke, sw, passes, rough, over=o)
            self.line(x + w, y + h, x, y + h, stroke, sw, passes, rough, over=o)
            self.line(x, y + h, x, y, stroke, sw, passes, rough, over=o)

    def ellipse(self, cx, cy, rx, ry, stroke=INK, w=2.2, fill=None, fop=1.0, passes=2, rough=1.0):
        n = 20
        if fill:
            pts = [(cx + rx * math.cos(2 * math.pi * k / n) + self.j(rough),
                    cy + ry * math.sin(2 * math.pi * k / n) + self.j(rough)) for k in range(n)]
            self._p(self._smooth(pts, closed=True) + " Z", fill=fill, fop=fop)
        if stroke:
            for p in range(passes):
                a0 = self.r.uniform(0, 2 * math.pi)
                sweep = 2 * math.pi * self.r.uniform(1.03, 1.10)
                m = n + 3
                pts = []
                for k in range(m):
                    a = a0 + sweep * k / (m - 1)
                    rr = 1 + self.j(0.035 * rough)
                    pts.append((cx + rx * rr * math.cos(a), cy + ry * rr * math.sin(a)))
                self._p(self._smooth(pts), stroke, w if p == 0 else w * 0.5, 1.0 if p == 0 else 0.55)

    def dot(self, cx, cy, r=4.5, fill=INK):
        self.ellipse(cx, cy, r, r, stroke=fill, w=1.2, fill=fill, passes=1, rough=0.4)

    def ring(self, cx, cy, r=4.5, stroke=INK):
        self.ellipse(cx, cy, r, r, stroke=stroke, w=1.6, passes=1, rough=0.4)

    def arrow(self, x1, y1, x2, y2, stroke=INK, w=2.2, head=13, bow=0.0, dash=None):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        L = math.hypot(x2 - x1, y2 - y1) or 1.0
        nx, ny = -(y2 - y1) / L, (x2 - x1) / L
        cx, cy = mx + nx * bow, my + ny * bow
        if abs(bow) < 0.5:
            self.line(x1, y1, x2, y2, stroke, w, 2 if not dash else 1, dash=dash)
            tx, ty = x2 - x1, y2 - y1
        else:
            pts = []
            for k in range(7):
                t = k / 6
                pts.append(((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t * t * x2,
                            (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t * t * y2))
            self.curve(pts, stroke, w, 2 if not dash else 1, rough=0.6, dash=dash)
            tx, ty = x2 - cx, y2 - cy
        a = math.atan2(ty, tx)
        for s in (-1, 1):
            b = a + math.pi + s * math.radians(27 + self.j(4))
            self.line(x2, y2, x2 + head * math.cos(b), y2 + head * math.sin(b), stroke, w, 1, 0.5)

    def head(self, x2, y2, fx, fy, stroke=INK, w=2.0, size=12):
        a = math.atan2(y2 - fy, x2 - fx)
        for sg in (-1, 1):
            b = a + math.pi + sg * math.radians(27)
            self.line(x2, y2, x2 + size * math.cos(b), y2 + size * math.sin(b), stroke, w, 1, 0.4)

    def cross(self, x, y, s, stroke=RED, w=3):
        self.line(x - s, y - s, x + s, y + s, stroke, w, 1, 1.2)
        self.line(x - s, y + s, x + s, y - s, stroke, w, 1, 1.2)

    # ---------- fills ----------
    def hatch_rect(self, x, y, w, h, gap=6.0, angle=45, stroke=PENCIL, sw=1.2, op=0.9):
        a = math.radians(angle)
        dx, dy = math.cos(a), math.sin(a)
        nx, ny = -dy, dx
        cs = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        proj = [cx * nx + cy * ny for cx, cy in cs]
        t = min(proj) + gap / 2
        while t < max(proj):
            px, py = nx * t, ny * t
            seg = self._clip(px, py, dx, dy, x, y, x + w, y + h)
            if seg:
                (ax, ay), (bx, by) = seg
                sh = 0.8
                ax += dx * self.r.uniform(0, sh); ay += dy * self.r.uniform(0, sh)
                bx -= dx * self.r.uniform(0, sh); by -= dy * self.r.uniform(0, sh)
                d = f"M{ax + self.j(.5):.1f} {ay + self.j(.5):.1f} L{bx + self.j(.5):.1f} {by + self.j(.5):.1f}"
                self._p(d, stroke, sw, op)
            t += gap

    @staticmethod
    def _clip(px, py, dx, dy, x0, y0, x1, y1):
        tmin, tmax = -1e9, 1e9
        for p, d, lo, hi in ((px, dx, x0, x1), (py, dy, y0, y1)):
            if abs(d) < 1e-9:
                if p < lo or p > hi:
                    return None
            else:
                ta, tb = (lo - p) / d, (hi - p) / d
                tmin, tmax = max(tmin, min(ta, tb)), min(tmax, max(ta, tb))
        if tmax <= tmin:
            return None
        return (px + dx * tmin, py + dy * tmin), (px + dx * tmax, py + dy * tmax)

    def hatch_circle(self, cx, cy, r, gap=6.0, angle=45, stroke=PENCIL, sw=1.2, op=0.9):
        a = math.radians(angle)
        dx, dy = math.cos(a), math.sin(a)
        nx, ny = -dy, dx
        t = -r + gap / 2
        while t < r:
            hw = math.sqrt(max(r * r - t * t, 0)) * self.r.uniform(0.9, 1.0)
            ax, ay = cx + nx * t - dx * hw, cy + ny * t - dy * hw
            bx, by = cx + nx * t + dx * hw, cy + ny * t + dy * hw
            self._p(f"M{ax:.1f} {ay:.1f} L{bx:.1f} {by:.1f}", stroke, sw, op)
            t += gap

    def grid(self, x, y, n, cell, stroke=PENCIL, w=0.9, border=INK, bw=2.0, m=None):
        m = m or n
        for i in range(1, n):
            self.line(x + i * cell, y, x + i * cell, y + m * cell, stroke, w, 1, 0.5)
        for i in range(1, m):
            self.line(x, y + i * cell, x + n * cell, y + i * cell, stroke, w, 1, 0.5)
        self.rect(x, y, n * cell, m * cell, border, bw, rough=0.8)

    def fill_cell(self, x, y, cell, c, r, color=ORANGE, op=0.75):
        q = [(x + c * cell + 1.5 + self.j(1), y + r * cell + 1.5 + self.j(1)),
             (x + (c + 1) * cell - 1.5 + self.j(1), y + r * cell + 1.5 + self.j(1)),
             (x + (c + 1) * cell - 1.5 + self.j(1), y + (r + 1) * cell - 1.5 + self.j(1)),
             (x + c * cell + 1.5 + self.j(1), y + (r + 1) * cell - 1.5 + self.j(1))]
        d = "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in q) + " Z"
        self._p(d, fill=color, fop=op)

    def butterfly(self, cx, cy, s=1.0, stroke=INK, fill=ORANGE):
        self.ellipse(cx - 9 * s, cy - 6 * s, 9 * s, 7 * s, stroke, 1.8, fill, 0.55, 1, 0.6)
        self.ellipse(cx + 9 * s, cy - 6 * s, 9 * s, 7 * s, stroke, 1.8, fill, 0.55, 1, 0.6)
        self.ellipse(cx - 7 * s, cy + 6 * s, 6 * s, 5 * s, stroke, 1.8, fill, 0.55, 1, 0.6)
        self.ellipse(cx + 7 * s, cy + 6 * s, 6 * s, 5 * s, stroke, 1.8, fill, 0.55, 1, 0.6)
        self.line(cx, cy - 10 * s, cx, cy + 11 * s, stroke, 2.4, 1, 0.3)
        self.line(cx, cy - 10 * s, cx - 5 * s, cy - 18 * s, stroke, 1.4, 1, 0.3)
        self.line(cx, cy - 10 * s, cx + 5 * s, cy - 18 * s, stroke, 1.4, 1, 0.3)

    def padlock(self, cx, cy, s=1.0, stroke=INK):
        self.rect(cx - 13 * s, cy - 2 * s, 26 * s, 20 * s, stroke, 2.2, fill=BLUE, fop=0.15, rough=0.5)
        pts = [(cx - 8 * s, cy - 2 * s), (cx - 8 * s, cy - 12 * s), (cx, cy - 18 * s),
               (cx + 8 * s, cy - 12 * s), (cx + 8 * s, cy - 2 * s)]
        self.curve(pts, stroke, 2.2, 2, 0.4)
        self.dot(cx, cy + 7 * s, 2.5 * s)

    # ---------- labels ----------
    def label(self, x, y, html, size=20, color=INK, w=None, align="left", font="hand", lh=1.25):
        fam = "'Caveat', 'Segoe Print', cursive" if font == "title" else "'Patrick Hand', 'Segoe Print', cursive"
        st = f"position: absolute; left: {x}px; top: {y}px; margin: 0; font-family: {fam}; font-size: {size}px; line-height: {lh}; color: {color}"
        if font == "title":
            st += "; font-weight: 700"
        if w:
            st += f"; width: {w}px"
        if align != "left":
            st += f"; text-align: {align}"
        self.labels.append(f'<div style="{st}">{html}</div>')

    def card(self, x, y, w, h, main, sub=None, stroke=INK, fill=None, fop=0.12, size=21):
        self.rect(x, y, w, h, stroke, 2.2, fill=fill, fop=fop)
        inner = esc(main) if sub is None else f'{esc(main)}<br><span style="font-size: 17px; color: {PENCIL}">{esc(sub)}</span>'
        st = (f"position: absolute; left: {x}px; top: {y}px; width: {w}px; height: {h}px; box-sizing: border-box; "
              f"padding: 6px 10px; display: flex; align-items: center; justify-content: center; text-align: center; "
              f"font-family: 'Patrick Hand', 'Segoe Print', cursive; font-size: {size}px; line-height: 1.2; color: {INK}")
        self.labels.append(f'<div style="{st}"><span>{inner}</span></div>')

    def note(self, x, y, w, h, kind, text):
        if kind == "old":
            fill, stroke, head = "#e9e3d6", PENCIL, "Old way"
        elif kind == "open":
            fill, stroke, head = "#dfe8f5", BLUE, "Still open"
        else:
            fill, stroke, head = "#f6e2c9", ORANGE, kind
        self.rect(x, y, w, h, stroke, 1.8, fill=fill, fop=1.0, rough=0.8)
        self.label(x + 16, y + 8, esc(head), 26, INK, font="title", lh=1.0)
        self.label(x + 16, y + 38, esc(text), 19, INK, w=w - 32, lh=1.2)

    def header(self, num, title, point):
        if num is not None:
            self.ellipse(80, 66, 28, 26, INK, 2.4)
            self.label(52, 42, str(num), 40, INK, w=56, align="center", font="title", lh=1.0)
            tx = 124
        else:
            tx = 56
        self.label(tx, 34, esc(title), 52, INK, font="title", lh=1.0)
        self.label(tx, 98, esc(point), 25, INK, w=W - tx - 56, lh=1.2)

    def footer(self, text):
        self.label(560, 758, esc(text), 16, PENCIL, w=507, align="right", lh=1.0)

    # ---------- output ----------
    def html(self, title, alt):
        Wd, Hd = self.W, self.H
        svg = (f'<svg role="img" aria-label="{esc(alt)}" width="{Wd}" height="{Hd}" viewBox="0 0 {Wd} {Hd}" '
               f'style="position: absolute; left: 0px; top: 0px; width: {Wd}px; height: {Hd}px">'
               + "".join(self.paths) + "</svg>")
        root = (f'<div style="position: relative; width: {Wd}px; height: {Hd}px; overflow: hidden; background: {PAPER}; '
                f"font-family: 'Patrick Hand', 'Segoe Print', cursive; color: {INK}\">")
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{esc(title)}</title>
<script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@700&amp;family=Patrick+Hand&amp;display=swap" rel="stylesheet">
<style>
body{{margin:0}}
</style>
</helmet>
{root}
{svg}
{chr(10).join(self.labels)}
</div>
</x-dc>
<script type="text/x-dc" data-dc-script data-props='{{"$preview":{{"width":{Wd},"height":{Hd}}}}}'>
class Component extends DCLogic {{
renderVals() {{
return {{}};
}}
}}
</script>
</body>
</html>
"""
