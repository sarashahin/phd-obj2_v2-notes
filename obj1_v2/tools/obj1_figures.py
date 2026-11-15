#!/usr/bin/env python3
"""
obj1_figures.py  (obj1_v2, Objective 1 figures; 7 Nov 2026, Sara's yes to the figure list in claude/obj1_figures_plan.md)
v2 (7 Nov, after Sara's first render): layout only, no number changes. Font fixed to DejaVu Sans (matplotlib's own), so
the layout is the same on every machine; fig4: shorter "Species per cell" title, counts under each group as "77/100",
rare-clumping counts on a second x-label line; fig2a: the median is drawn only over rounds every world reaches (the
worlds that run longest are the poorer ones, so a median over the rest dipped); every figure is checked for text that
overlaps other text or runs outside the figure, and each problem is written to figure_numbers.txt as a [layout] line.

Draws the Objective 1 figures from three inputs that already exist on Sara's machine, and writes each one as a vector PDF
and a 300 dpi PNG, plus figure_numbers.txt with every number a figure shows (for checking against the logs):
  --data        obj1_figure_data.npz (obj1_figure_data.py): per-world histories and three example worlds
  --bnm-logs    world_vs_bnm --each logs of real_train, real_val, real_eval (per-world numbers and the BNM fit half)
  --speed-logs  frame_change_check logs of the same three splits (pooled r(5) by range class and BNM's interval)
No BNM file is opened: every BNM number is read from the logs, which print the fit half only. The split of a log is read
from its file name (real_train, real_val or real_eval).

Figures
  fig1_world      world 5001: habitat at the start and the end, final richness, regional shift by round, growth rate of
                  each final species against the habitat value of its cells (from its saved growth field), range sizes
  fig2_assembly   richness by round (all worlds; median in black), kept against removed per round over the last half,
                  richness through the final run relative to its start
  fig3_ranges     recorded range-size classes: median and 10-90% of worlds per split against the BNM fit half
  fig4_bands      the 8 numbers of the 28 Oct bands per world and split, the band shaded and BNM's value as a line;
                  the count of worlds inside comes from world_vs_bnm's own flags
  fig5_spatial    similarity of two recorded cells and rare-species co-occupancy by distance, against BNM
  fig6_speed      r(5) by range class (pooled per split) against BNM's interval
  figS1_ranges    four species of world 5001, small to widespread, at four times over the last 100 BNM years, and their
                  cells by year
Colours: one hue per split (blue, orange, aqua), each with its own marker; BNM in black; bands in light grey.

Run from obj1_v2/work:
    python3 ../tools/obj1_figures.py --self-test | tail -1
    python3 ../tools/obj1_figures.py --data ../logs/obj1_figure_data.npz --bnm-logs ../logs/world_vs_bnm_real_train.txt ../logs/world_vs_bnm_real_val.txt ../logs/world_vs_bnm_real_eval.txt --speed-logs ../logs/frame_change_real_train.txt ../logs/frame_change_real_val.txt ../logs/frame_change_real_eval.txt --out ../figures/obj1
    xdg-open ../figures/obj1/fig1_world.png
"""
import argparse, os, re, sys, tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS = ["real_train", "real_val", "real_eval"]
LABEL = {"real_train": "training", "real_val": "validation", "real_eval": "evaluation"}
SHORT = {"real_train": "train", "real_val": "val", "real_eval": "eval"}
COL = {"real_train": "#2a78d6", "real_val": "#eb6834", "real_eval": "#1baf7a"}
MARK = {"real_train": "o", "real_val": "s", "real_eval": "^"}
INK, MUTED, BAND = "#0b0b0b", "#6b6a66", "#e4e3df"
BANDS = {"window": (27, 41), "cell": (8.8, 18.6), "turnover": (1.62, 3.06), "decay": (None, 0.085), "TV": (None, 0.116),
         "1-5": (15.2, 31.4), "51+": (25.9, 35.5)}
RARE = [0.196, 0.232, 0.137, 0.077]
FLAG_NAMES = {"species per window": "window", "species per cell": "cell", "turnover": "turnover", "similarity decay": "decay",
              "range-size TV": "TV", "1-5 cell share": "1-5", "51+ share": "51+", "rare clumping 1-4 cells": "rare"}
TITLES = {"window": "Species per window", "cell": "Species per cell", "turnover": "Turnover (beta)",
          "decay": "Similarity decay", "TV": "Range-size distance (TV)", "1-5": "Share of 1-5 cell ranges (%)",
          "51+": "Share of 51+ cell ranges (%)", "rare": "Rare co-occupancy / BNM"}
MM = 1 / 25.4
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "axes.titlelocation": "left", "axes.titleweight": "bold", "font.family": "DejaVu Sans"})


# ---------------------------------------------------------------- reading the logs
def split_of_name(path):
    hits = [s for s in SPLITS if s in os.path.basename(path)]
    if len(hits) != 1:
        sys.exit(f"STOP: cannot tell the split of {path} from its name (needs one of {SPLITS})")
    return hits[0]


def _floats(s):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)]


def parse_block(b):
    """Numbers of one world (or of the BNM fit half) from a world_vs_bnm block."""
    d = {}
    m = re.search(r"share of cases: 1 ([\d.]+)%, 2-5 ([\d.]+)%, 6-10 ([\d.]+)%, 11-20 ([\d.]+)%, 21-50 ([\d.]+)%, 51\+ ([\d.]+)%", b)
    d["shares"] = [float(x) for x in m.groups()]
    d["window"] = float(re.search(r"species present per window/community, median (\d+(?:\.\d+)?)", b).group(1))
    m = re.search(r"alpha ([\d.]+) x turnover \(beta\) ([\d.]+)", b)
    d["cell"], d["turnover"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"cases of 20 cells or fewer[^:]*: 1 ([\d.]+), 2 ([\d.]+), 3 ([\d.]+), 4 ([\d.]+), 5-6 ([\d.]+), 7-10 ([\d.]+)", b)
    d["rare_co"] = [float(x) for x in m.groups()]
    m = re.search(r"Sorensen\) by distance: 1 ([\d.]+), 2 ([\d.]+), 3-4 ([\d.]+), 5-10 ([\d.]+)", b)
    d["sor"] = [float(x) for x in m.groups()]
    m = re.search(r"rare co-occupancy / BNM at 1-4 cells \[([^\]]+)\]; range-size distance ([\d.]+);.*similarity decay ([\d.]+)", b)
    if m:
        d["rare"] = [float(x) for x in m.group(1).split(",")]
        d["TV"], d["decay"] = float(m.group(2)), float(m.group(3))
        d["1-5"], d["51+"] = d["shares"][0] + d["shares"][1], d["shares"][5]
        m2 = re.search(r"bands \(28 Oct\): (.*?)  ->", b)
        d["flags"] = {FLAG_NAMES[k.strip()]: v == "PASS" for k, v in re.findall(r"([^,]+?) (PASS|FAIL)", m2.group(1))}
        m3 = re.search(r"_s(\d+)_training", b)
        d["seed"] = int(m3.group(1)) if m3 else None
    return d


def read_bnm_log(path):
    t = Path(path).read_text()
    head = t.split("\nsimulated: ")[0]
    bnm = parse_block(head.split("BNM, geographic fit half")[1])
    bnm["decay"] = bnm["sor"][0] - bnm["sor"][3]
    worlds = [parse_block(b) for b in t.split("\nsimulated: ")[1:]]
    if not worlds:
        sys.exit(f"STOP: no simulated world in {path}")
    return bnm, worlds


def read_speed_log(path):
    t = Path(path).read_text()
    m = re.search(r"pooled over (\d+) worlds\s*\n\s*k\* (\d+): r\(5\) by class 1-5 ([\d.]+) \(BNM ([\d.]+), ([\d.]+)-([\d.]+)\); "
                  r"6-20 ([\d.]+) \(BNM ([\d.]+), ([\d.]+)-([\d.]+)\); 21\+ ([\d.]+) \(BNM ([\d.]+), ([\d.]+)-([\d.]+)\); "
                  r"all ([\d.]+) \(BNM ([\d.]+), ([\d.]+)-([\d.]+)\)", t)
    if not m:
        sys.exit(f"STOP: no pooled line in {path}")
    v = [float(x) for x in m.groups()]
    out = dict(n=int(v[0]), k=int(v[1]))
    for i, c in enumerate(["1-5", "6-20", "21+", "all"]):
        sim, b, lo, hi = v[2 + 4 * i: 6 + 4 * i]
        out[c] = dict(sim=sim, bnm=b, lo=lo, hi=hi)
    return out


# ---------------------------------------------------------------- helpers
def _drawn_ticklabels(axis, lim):
    """Tick labels inside the axis range; matplotlib keeps labels for ticks outside it, which are never drawn."""
    lo, hi = sorted(lim); tol = 1e-9 * max(1.0, abs(hi - lo))
    return [lab for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()) if lo - tol <= loc <= hi + tol]


def layout_problems(fig):
    """Text that overlaps other text, or runs outside the figure (titles, axis labels, tick labels of every panel)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    items = []
    for ax in fig.axes:
        texts = [getattr(ax, a, None) for a in ("title", "_left_title", "_right_title")]
        texts += [ax.xaxis.label, ax.yaxis.label] + _drawn_ticklabels(ax.xaxis, ax.get_xlim()) + _drawn_ticklabels(ax.yaxis, ax.get_ylim())
        for t in texts:
            if t is not None and t.get_visible() and t.get_text().strip():
                items.append((t.get_text().replace("\n", " "), t.get_window_extent(r)))
    probs = []
    for s, b in items:
        if b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1 or b.y1 > fb.y1 + 1:
            probs.append(f"'{s[:40]}' runs outside the figure")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1].overlaps(items[j][1]):
                probs.append(f"'{items[i][0][:40]}' overlaps '{items[j][0][:40]}'")
    return probs


def rare_xlabel(ins):
    return "distance (cells)\ninside all four bands\n" + ", ".join(ins)


def save(fig, out, name, numbers):
    for p in layout_problems(fig):
        numbers.append(f"[layout] {name}: {p}")
    for ext in ("pdf", "png"):
        fig.savefig(Path(out) / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    numbers.append(f"[{name}] written as {name}.pdf and {name}.png")


def frame(ax):
    """Map panels get a full thin frame (the plot style removes the top and right spines)."""
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color(MUTED)
    ax.set_xticks([]); ax.set_yticks([])


def legend_splits(ax, splits, loc="best", extra=()):
    h = [plt.Line2D([], [], color=COL[s], marker=MARK[s], ls="", ms=4, label=LABEL[s]) for s in splits]
    ax.legend(handles=h + list(extra), loc=loc, frameon=False, handletextpad=0.3)


def unpack_frames(z, ex, Y=20, X=20):
    off = z[f"{ex}_F_offsets"]; ids = z[f"{ex}_F_ids"]; bits = z[f"{ex}_F_bits"]
    frames = []
    for k in range(len(off) - 1):
        a, b = int(off[k]), int(off[k + 1])
        P = np.unpackbits(bits[a:b], axis=1, count=Y * X).astype(bool).reshape(b - a, Y, X)
        frames.append(dict(zip(ids[a:b].tolist(), P)))
    return frames


# ---------------------------------------------------------------- figures
def fig1(z, out, numbers, seed=5001):
    ex = f"ex{seed}"
    if f"{ex}_hab_start" not in z.files:
        numbers.append(f"[fig1_world] skipped: no example world {seed} in the data"); return
    Hs, He, R = z[f"{ex}_hab_start"], z[f"{ex}_hab_end"], z[f"{ex}_r_field"]
    pres = z[f"{ex}_presence"].astype(bool)
    shift = z[f"w{seed}_shift"]
    v = float(max(np.abs(Hs).max(), np.abs(He).max()))
    fig, axs = plt.subplots(2, 3, figsize=(180 * MM, 110 * MM), constrained_layout=True)
    for ax, H, t in ((axs[0, 0], Hs, "a  Habitat at the start"), (axs[0, 1], He, "b  Habitat at the end")):
        im = ax.imshow(H, cmap="PuOr_r", vmin=-v, vmax=v, origin="lower")
        ax.set_title(t); frame(ax)
    fig.colorbar(im, ax=axs[0, :2], shrink=0.8, label="habitat value")
    rich = pres.sum(0)
    im2 = axs[0, 2].imshow(rich, cmap="Blues", origin="lower", vmin=0)
    axs[0, 2].set_title("c  Species per cell at the end"); frame(axs[0, 2])
    fig.colorbar(im2, ax=axs[0, 2], shrink=0.8, label="species")
    ax = axs[1, 0]
    ax.plot(np.arange(len(shift)), shift, color=INK, lw=1)
    ax.axhline(0, color=MUTED, lw=0.5, ls=":")
    ax.set_title("d  Regional shift of the habitat"); ax.set_xlabel("round (1 round = 25 BNM years)"); ax.set_ylabel("shift m (sd units)")
    ax = axs[1, 1]
    for i in range(len(R)):
        o = np.argsort(He.ravel())
        ax.plot(He.ravel()[o], R[i].ravel()[o], color=COL["real_train"], lw=0.6, alpha=0.5)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_title("e  Growth rate against habitat, final species"); ax.set_xlabel("habitat value of the cell"); ax.set_ylabel("growth rate r")
    ax = axs[1, 2]
    cells = pres.reshape(len(pres), -1).sum(1)
    ax.hist(cells, bins=np.arange(0, 401, 20), color=COL["real_train"], edgecolor="white", lw=0.5)
    ax.set_title("f  Range sizes of the final species"); ax.set_xlabel("occupied cells (of 400)"); ax.set_ylabel("species")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    numbers.append(f"[fig1_world] world {seed}: {len(pres)} final species; species per cell median {np.median(rich):.0f} "
                   f"(range {rich.min()}-{rich.max()}); range size median {np.median(cells):.0f} cells; {len(shift)} rounds; "
                   f"shift range {shift.min():+.2f} to {shift.max():+.2f}")
    save(fig, out, "fig1_world", numbers)


def fig2(z, out, numbers):
    seeds = [int(s) for s in z["seeds"]]
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM), constrained_layout=True)
    ax = axs[0]
    G = [z[f"w{s}_gamma"].astype(float) for s in seeds]
    n = max(len(g) for g in G)
    M = np.full((len(G), n), np.nan)
    for i, g in enumerate(G):
        M[i, :len(g)] = g
        ax.plot(np.arange(len(g)), g, color=MUTED, lw=0.3, alpha=0.25)
    keep = np.sum(~np.isnan(M), 0) == len(G)
    med = np.nanmedian(M[:, keep], 0)
    ax.plot(np.arange(n)[keep], med, color=INK, lw=1.2, label="median (rounds all worlds reach)")
    ax.set_title("a  Richness during assembly"); ax.set_xlabel("round"); ax.set_ylabel("species in the world")
    ax.legend(frameon=False, loc="lower right")
    ax = axs[1]
    lim = 0
    for s in seeds:
        k, r = z[f"w{s}_kept"].astype(float), z[f"w{s}_removed"].astype(float)
        h = len(k) // 2
        sp = str(z[f"w{s}_split"])
        x, y = r[h:].mean(), k[h:].mean()
        lim = max(lim, x, y)
        ax.plot(x, y, marker=MARK[sp], color=COL[sp], ms=3.5, ls="", mec="white", mew=0.3, alpha=0.9)
    ax.plot([0, lim * 1.1], [0, lim * 1.1], color=INK, lw=0.6, ls="--")
    ax.set_xlim(0, lim * 1.1); ax.set_ylim(0, lim * 1.1); ax.set_aspect("equal")
    ax.set_title("b  Kept against lost, last half"); ax.set_xlabel("residents lost per round"); ax.set_ylabel("newcomers kept per round")
    legend_splits(ax, [s for s in SPLITS if any(str(z[f"w{x}_split"]) == s for x in seeds)], loc="lower right")
    ax = axs[2]
    rel_end = []
    for s in seeds:
        g, t = z[f"w{s}_final_gamma"].astype(float), z[f"w{s}_final_t"].astype(float)
        ax.plot(t / 200.0, g / g[0], color=MUTED, lw=0.4, alpha=0.3)
        rel_end.append(g[-1] / g[0])
    ax.set_ylim(0.9, 1.05)
    ax.set_title("c  Richness through the final run"); ax.set_xlabel("BNM year"); ax.set_ylabel("richness / richness at the start")
    K = sum(z[f"w{s}_kept"][len(z[f'w{s}_kept']) // 2:].sum() for s in seeds)
    Rm = sum(z[f"w{s}_removed"][len(z[f'w{s}_removed']) // 2:].sum() for s in seeds)
    numbers.append(f"[fig2_assembly] {len(seeds)} worlds; rounds {min(len(g) for g in G)}-{n}; median drawn over rounds 0-{int(keep.sum()) - 1} "
                   f"(reached by all worlds), {med[-1]:.0f} at its last round; last halves kept {K}, removed {Rm} (kept share {K / max(1, K + Rm):.3f}); final-run richness "
                   f"end / start: median {np.median(rel_end):.3f}, lowest {min(rel_end):.3f}")
    save(fig, out, "fig2_assembly", numbers)


def fig3(logs, out, numbers):
    cls = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]
    bnm = logs[SPLITS[0]][0]["shares"] if SPLITS[0] in logs else next(iter(logs.values()))[0]["shares"]
    fig, ax = plt.subplots(figsize=(120 * MM, 60 * MM), constrained_layout=True)
    x = np.arange(len(cls))
    off = {"real_train": -0.18, "real_val": 0.0, "real_eval": 0.18}
    for sp in SPLITS:
        if sp not in logs: continue
        S = np.array([w["shares"] for w in logs[sp][1]])
        med, lo, hi = np.median(S, 0), np.percentile(S, 10, 0), np.percentile(S, 90, 0)
        ax.errorbar(x + off[sp], med, yerr=[med - lo, hi - med], fmt=MARK[sp], color=COL[sp], ms=4, lw=0.8, capsize=0,
                    mec="white", mew=0.4, label=f"{LABEL[sp]} (n = {len(S)})")
        numbers.append(f"[fig3_ranges] {LABEL[sp]}: median shares " + ", ".join(f"{c} {m:.1f}%" for c, m in zip(cls, med)))
    ax.scatter(x, bnm, marker="_", s=260, color=INK, lw=1.6, label="BNM fit half", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(cls); ax.set_xlabel("recorded range size (cells of 400)"); ax.set_ylabel("share of ranges (%)")
    ax.set_title("Range-size classes, simulated (median, 10-90% of worlds) and BNM")
    ax.legend(frameon=False, loc="upper left")
    numbers.append("[fig3_ranges] BNM: " + ", ".join(f"{c} {b:.1f}%" for c, b in zip(cls, bnm)))
    save(fig, out, "fig3_ranges", numbers)


def fig4(logs, out, numbers):
    keys = ["window", "cell", "turnover", "decay", "TV", "1-5", "51+", "rare"]
    bnm = next(iter(logs.values()))[0]
    fig, axs = plt.subplots(2, 4, figsize=(180 * MM, 95 * MM), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, k in zip(axs.ravel(), keys):
        present = [sp for sp in SPLITS if sp in logs]
        if k != "rare":
            lo, hi = BANDS[k]
            vals_all = np.concatenate([[w[k] for w in logs[sp][1]] for sp in present])
            y0 = min(vals_all.min(), lo if lo is not None else vals_all.min()); y1 = max(vals_all.max(), hi)
            ax.axhspan(lo if lo is not None else min(0, y0), hi, color=BAND, zorder=0)
            bv = {"window": bnm["window"], "cell": bnm["cell"], "turnover": bnm["turnover"], "decay": bnm["decay"], "TV": 0.0,
                  "1-5": bnm["shares"][0] + bnm["shares"][1], "51+": bnm["shares"][5]}[k]
            ax.axhline(bv, color=INK, lw=0.9)
            txt = []
            for j, sp in enumerate(present):
                v = np.array([w[k] for w in logs[sp][1]])
                ax.plot(j + rng.uniform(-0.18, 0.18, len(v)), v, MARK[sp], color=COL[sp], ms=2.6, alpha=0.75, mec="none")
                ax.plot([j - 0.28, j + 0.28], [np.median(v)] * 2, color=INK, lw=1.2)
                inside = sum(w["flags"][k] for w in logs[sp][1])
                txt.append(f"{inside}/{len(v)}")
                numbers.append(f"[fig4_bands] {TITLES[k]}, {LABEL[sp]}: median {np.median(v):.3g}, inside {inside}/{len(v)}")
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels([f"{SHORT[s]}\n{t}" for s, t in zip(present, txt)])
            ax.set_xlim(-0.6, len(present) - 0.4)
            ax.set_title(TITLES[k], loc="left")
        else:
            for d in range(4):
                ax.add_patch(plt.Rectangle((d - 0.4, 1 - RARE[d]), 0.8, 2 * RARE[d], color=BAND, zorder=0, lw=0))
            ax.axhline(1, color=INK, lw=0.9)
            for j, sp in enumerate(present):
                V = np.array([w["rare"] for w in logs[sp][1]])
                for d in range(4):
                    ax.plot(d + (j - 1) * 0.22 + rng.uniform(-0.07, 0.07, len(V)), V[:, d], MARK[sp], color=COL[sp], ms=2.2,
                            alpha=0.7, mec="none")
                inside = sum(w["flags"]["rare"] for w in logs[sp][1])
                numbers.append(f"[fig4_bands] rare co-occupancy / BNM, {LABEL[sp]}: medians at 1-4 cells "
                               f"{', '.join(f'{m:.2f}' for m in np.median(V, 0))}; worlds inside all four {inside}/{len(V)}")
            ins = [f"{sum(w['flags']['rare'] for w in logs[sp][1])}/{len(logs[sp][1])}" for sp in present]
            ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"])
            ax.set_xlabel(rare_xlabel(ins))
            ax.set_title(TITLES[k], loc="left")
    h = [plt.Line2D([], [], color=COL[s], marker=MARK[s], ls="", ms=4, label=LABEL[s]) for s in SPLITS if s in logs]
    h += [plt.Line2D([], [], color=INK, lw=1.2, label="median of worlds (short bar); BNM (long line)"),
          plt.Rectangle((0, 0), 1, 1, color=BAND, label="band (28 Oct); n/N = worlds inside")]
    try:
        fig.legend(handles=h, loc="outside lower center", ncol=5, frameon=False)
    except ValueError:  # matplotlib older than 3.7 has no "outside" legends
        fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False)
    save(fig, out, "fig4_bands", numbers)


def fig5(logs, out, numbers):
    bnm = next(iter(logs.values()))[0]
    fig, axs = plt.subplots(1, 2, figsize=(150 * MM, 58 * MM), constrained_layout=True)
    panels = [(axs[0], "sor", ["1", "2", "3-4", "5-10"], "a  Similarity of two recorded cells", "Sorensen similarity",
               "Similarity of two recorded cells"),
              (axs[1], "rare_co", ["1", "2", "3", "4", "5-6", "7-10"], "b  Rare ranges (20 cells or fewer)",
               "co-occupancy ratio (1 = random)", "Co-occupancy of rare ranges (20 cells or fewer)")]
    for ax, key, xl, title, yl, name in panels:
        x = np.arange(len(xl))
        for sp in SPLITS:
            if sp not in logs: continue
            V = np.array([w[key] for w in logs[sp][1]])
            for v in V:
                ax.plot(x, v, color=COL[sp], lw=0.3, alpha=0.15)
            ax.plot(x, np.median(V, 0), color=COL[sp], marker=MARK[sp], lw=1.2, ms=4, mec="white", mew=0.4, label=f"{LABEL[sp]} median")
            numbers.append(f"[fig5_spatial] {name}, {LABEL[sp]}: medians " + ", ".join(f"{a} {m:.2f}" for a, m in zip(xl, np.median(V, 0))))
        ax.plot(x, bnm[key], color=INK, marker="D", lw=1.4, ms=3.5, label="BNM fit half")
        numbers.append(f"[fig5_spatial] {name}, BNM: " + ", ".join(f"{a} {m:.2f}" for a, m in zip(xl, bnm[key])))
        if key == "rare_co":
            ax.axhline(1, color=MUTED, lw=0.5, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_xlabel("distance (cells)"); ax.set_ylabel(yl); ax.set_title(title)
    axs[0].legend(frameon=False, loc="lower left")
    save(fig, out, "fig5_spatial", numbers)


def fig6(speed, out, numbers):
    cls = ["1-5", "6-20", "21+", "all"]
    fig, ax = plt.subplots(figsize=(110 * MM, 60 * MM), constrained_layout=True)
    x = np.arange(len(cls))
    ref = next(iter(speed.values()))
    for i, c in enumerate(cls):
        ax.add_patch(plt.Rectangle((i - 0.32, ref[c]["lo"]), 0.64, ref[c]["hi"] - ref[c]["lo"], color=BAND, lw=0, zorder=0))
        ax.plot([i - 0.32, i + 0.32], [ref[c]["bnm"]] * 2, color=INK, lw=1.2, zorder=1)
    off = {"real_train": -0.15, "real_val": 0.0, "real_eval": 0.15}
    for sp in SPLITS:
        if sp not in speed: continue
        ax.plot(x + off[sp], [speed[sp][c]["sim"] for c in cls], MARK[sp], color=COL[sp], ms=4.5, mec="white", mew=0.4,
                label=f"{LABEL[sp]} (n = {speed[sp]['n']})", ls="")
        numbers.append(f"[fig6_speed] {LABEL[sp]}: frames per BNM year k* {speed[sp]['k']}; r(5) " +
                       ", ".join(f"{c} {speed[sp][c]['sim']:.3f}" for c in cls))
    numbers.append("[fig6_speed] BNM: " + ", ".join(f"{c} {ref[c]['bnm']:.3f} ({ref[c]['lo']:.3f}-{ref[c]['hi']:.3f})" for c in cls))
    ax.set_xticks(x); ax.set_xticklabels([f"{c} cells" if c != "all" else "all" for c in cls])
    ax.set_ylabel("map agreement 5 years apart, r(5)"); ax.set_title("Speed of range change, simulated (pooled) and BNM (interval)")
    legend_splits(ax, [s for s in SPLITS if s in speed], loc="lower right",
                  extra=[plt.Line2D([], [], color=INK, lw=1.2, label="BNM"), plt.Rectangle((0, 0), 1, 1, color=BAND, label="BNM interval")])
    save(fig, out, "fig6_speed", numbers)


def figS1(z, out, numbers, seed=5001):
    ex = f"ex{seed}"
    if f"{ex}_F_bits" not in z.files:
        numbers.append(f"[figS1_ranges] skipped: no example world {seed}"); return
    frames = unpack_frames(z, ex)
    ids_all = set().union(*[set(f) for f in frames])
    full = [i for i in ids_all if all(i in f for f in frames)]
    if len(full) < 4:
        numbers.append("[figS1_ranges] skipped: fewer than 4 species present in every frame"); return
    med = {i: np.median([f[i].sum() for f in frames]) for i in full}
    order = sorted(full, key=lambda i: med[i])
    pick = [order[int(round(q * (len(order) - 1)))] for q in (0.0, 0.33, 0.67, 1.0)]
    T = len(frames); cols = [0, T // 3, 2 * T // 3, T - 1]
    fig, axs = plt.subplots(4, 5, figsize=(170 * MM, 140 * MM), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.8]))
    for r, sid in enumerate(pick):
        for c, k in enumerate(cols):
            ax = axs[r, c]
            ax.imshow(frames[k][sid], cmap="Greys", vmin=0, vmax=1.4, origin="lower")
            frame(ax)
            if r == 0:
                ax.set_title(f"year {k - (T - 1)}" if k != T - 1 else "last year", loc="center", fontweight="normal")
        cells = [f[sid].sum() for f in frames]
        ax = axs[r, 4]
        ax.plot(np.arange(T) - (T - 1), cells, color=COL["real_train"], lw=1)
        ax.set_ylabel("cells"); ax.set_title(f"species {sid}: median {med[sid]:.0f} cells", loc="left", fontweight="normal")
        if r == 3: ax.set_xlabel("BNM year (0 = last)")
        numbers.append(f"[figS1_ranges] species {sid}: cells over the last {T} years median {med[sid]:.0f}, range {min(cells)}-{max(cells)}")
    save(fig, out, "figS1_ranges", numbers)


def make_all(data, bnm_logs, speed_logs, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    z = np.load(data)
    logs = {split_of_name(p): read_bnm_log(p) for p in bnm_logs}
    speed = {split_of_name(p): read_speed_log(p) for p in speed_logs}
    b0 = [v[0] for v in logs.values()]
    if any(b["shares"] != b0[0]["shares"] or b["window"] != b0[0]["window"] for b in b0):
        sys.exit("STOP: the BNM fit-half numbers differ between the logs; they must come from the same BNM files")
    numbers = []
    fig1(z, out, numbers); fig2(z, out, numbers); fig3(logs, out, numbers); fig4(logs, out, numbers)
    fig5(logs, out, numbers); fig6(speed, out, numbers); figS1(z, out, numbers)
    Path(out, "figure_numbers.txt").write_text("\n".join(numbers) + "\n")
    print("\n".join(numbers))
    lay = [x for x in numbers if x.startswith("[layout]")]
    print(f"\nlayout problems: {len(lay)}" + (" (see the [layout] lines above)" if lay else ""))
    print(f"figures and figure_numbers.txt written to {out}")
    return numbers


# ---------------------------------------------------------------- self-test
BNM_HEAD = """BNM, geographic fit half (44 windows): 44 windows/communities, 1434 cases with an occupied cell
  range size, median 29 cells; share of cases: 1 8.3%, 2-5 15.0%, 6-10 9.3%, 11-20 11.0%, 21-50 25.7%, 51+ 30.7%
  species present per window/community, median 34
  co-occupancy ratio by distance, cases of 20 cells or fewer (the rare-species target; 625 cases): 1 2.49, 2 1.50, 3 1.23, 4 1.12, 5-6 0.90, 7-10 0.84
  species per recorded cell (alpha): median 13, mean 14.3; geometric means over windows: gamma 32.0 = alpha 13.7 x turnover (beta) 2.34
  similarity of two recorded cells (Sorensen) by distance: 1 0.74, 2 0.72, 3-4 0.71, 5-10 0.70
"""
WORLD = """simulated: pool_x_f1_s{seed}_training.npz: 44 windows/communities, 1403 cases with an occupied cell
  range size, median 26 cells; share of cases: 1 {s1}%, 2-5 8.3%, 6-10 14.8%, 11-20 16.0%, 21-50 36.8%, 51+ 23.1%
  species present per window/community, median {win}
  co-occupancy ratio by distance, cases of 20 cells or fewer (the rare-species target; 562 cases): 1 2.69, 2 1.80, 3 1.17, 4 0.86, 5-6 0.68, 7-10 0.68
  species per recorded cell (alpha): median 10, mean 10.6; geometric means over windows: gamma 31.9 = alpha 10.5 x turnover (beta) 3.03
  similarity of two recorded cells (Sorensen) by distance: 1 0.62, 2 0.56, 3-4 0.50, 5-10 0.47
  against BNM: rare co-occupancy / BNM at 1-4 cells [1.08, 1.2, 0.95, 0.76]; range-size distance 0.216; species per window x0.94; similarity decay 0.15 (BNM 0.04)
  bands (28 Oct): species per window {wf}, species per cell PASS, turnover PASS, similarity decay FAIL, range-size TV FAIL, 1-5 cell share FAIL, 51+ share FAIL, rare clumping 1-4 cells FAIL  ->  3 of 8 pass
"""
SPEED = """pooled over {n} worlds
   k* 1: r(5) by class 1-5 0.857 (BNM 0.768, 0.680-0.863); 6-20 0.906 (BNM 0.866, 0.816-0.913); 21+ 0.964 (BNM 0.959, 0.951-0.966); all 0.963 (BNM 0.957, 0.949-0.965)
"""


def self_test():
    res = []
    def check(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")
    with tempfile.TemporaryDirectory() as tmp:
        bl = []
        for sp, seeds in (("real_train", [5001, 5002]), ("real_val", [5101]), ("real_eval", [6001])):
            p = f"{tmp}/world_vs_bnm_{sp}.txt"
            Path(p).write_text("header\n" + BNM_HEAD + "\n" + "\n".join(WORLD.format(seed=s, s1=1.0 + i, win=30 + i, wf="PASS" if i == 0 else "FAIL")
                                                                         for i, s in enumerate(seeds)))
            bl.append(p)
            Path(f"{tmp}/frame_change_{sp}.txt").write_text("x\n" + SPEED.format(n=len(seeds)))
        bnm, worlds = read_bnm_log(bl[0])
        check("T1", bnm["shares"] == [8.3, 15.0, 9.3, 11.0, 25.7, 30.7] and bnm["cell"] == 13.7 and abs(bnm["decay"] - 0.04) < 1e-9
              and bnm["rare_co"][0] == 2.49, "BNM fit-half numbers read from the log header")
        w = worlds[1]
        check("T2", w["seed"] == 5002 and w["shares"][0] == 2.0 and w["window"] == 31 and w["1-5"] == 10.3 and w["TV"] == 0.216
              and w["rare"] == [1.08, 1.2, 0.95, 0.76] and w["flags"]["window"] is False and w["flags"]["cell"] is True,
              "per-world numbers and world_vs_bnm's own flags read")
        sp = read_speed_log(f"{tmp}/frame_change_real_val.txt")
        check("T3", sp["n"] == 1 and sp["1-5"] == dict(sim=0.857, bnm=0.768, lo=0.68, hi=0.863) and sp["all"]["hi"] == 0.965,
              "pooled speed line read")
        try:
            split_of_name(f"{tmp}/world_vs_bnm_mixed.txt"); ok4 = False
        except SystemExit:
            ok4 = True
        check("T4", ok4, "a log whose name does not give its split is refused")
        Y = 20; rng = np.random.default_rng(1); data = {}
        for s, spn in ((5001, "real_train"), (5002, "real_train"), (5101, "real_val"), (6001, "real_eval")):
            n = 40 + s % 7
            data.update({f"w{s}_gamma": np.minimum(np.arange(n), 30), f"w{s}_kept": rng.integers(0, 3, n), f"w{s}_removed": rng.integers(0, 3, n),
                         f"w{s}_shift": rng.normal(size=n), f"w{s}_final_gamma": np.full(50, 30), f"w{s}_final_t": np.arange(1, 51) * 200,
                         f"w{s}_split": np.array(spn)})
        S = 6; P = np.zeros((S, Y, Y), bool)
        for i in range(S): P[i].flat[: (i + 1) * 30] = True
        H = rng.normal(size=(Y, Y)) * 0.25
        frames = np.concatenate([np.packbits(P.reshape(S, -1), axis=1)] * 12)
        data.update(ex5001_hab_start=H, ex5001_hab_end=H, ex5001_r_field=np.stack([np.exp(-(H - 0.1 * i) ** 2 / 0.02) - 0.1 for i in range(S)]),
                    ex5001_presence=P.astype(np.uint8), ex5001_F_offsets=np.arange(13) * S, ex5001_F_ids=np.tile(np.arange(S), 12),
                    ex5001_F_bits=frames, seeds=np.array([5001, 5002, 5101, 6001]))
        np.savez(f"{tmp}/d.npz", **data)
        nums = make_all(f"{tmp}/d.npz", bl, [f"{tmp}/frame_change_{s}.txt" for s in SPLITS], f"{tmp}/figs")
        made = sorted(p.name for p in Path(f"{tmp}/figs").iterdir())
        want = [f"{n}.{e}" for n in ("fig1_world", "fig2_assembly", "fig3_ranges", "fig4_bands", "fig5_spatial", "fig6_speed", "figS1_ranges")
                for e in ("pdf", "png")] + ["figure_numbers.txt"]
        check("T5", sorted(want) == made, "all 7 figures written as PDF and PNG, with figure_numbers.txt")
        check("T6", any("Species per window, training: median 30.5, inside 1/2" in x for x in nums) and any("validation: median 30, inside 1/1" in x for x in nums),
              "fig4 counts worlds inside from world_vs_bnm's flags (training 1 of 2, validation 1 of 1)")
        check("T7", any("species 5: cells over the last 12 years median 180" in x for x in nums), "figS1 picks the widest species and counts its cells")
        Path(bl[1]).write_text(Path(bl[1]).read_text().replace("1 8.3%, 2-5 15.0%", "1 9.3%, 2-5 15.0%"))
        try:
            make_all(f"{tmp}/d.npz", bl, [], f"{tmp}/figs2"); ok8 = False
        except SystemExit:
            ok8 = True
        check("T8", ok8, "logs with different BNM fit-half numbers are refused")
        rng2 = np.random.default_rng(5); bl2 = []
        Path(f"{tmp}/big").mkdir()
        for spn, nw in (("real_train", 100), ("real_val", 20), ("real_eval", 30)):
            p = f"{tmp}/big/world_vs_bnm_{spn}.txt"
            Path(p).write_text("header\n" + BNM_HEAD + "\n" + "\n".join(
                WORLD.format(seed=5000 + i, s1=round(float(rng2.uniform(0.5, 5)), 1), win=int(rng2.integers(15, 56)),
                             wf="PASS" if rng2.random() < 0.6 else "FAIL") for i in range(nw)))
            bl2.append(p)
        sl = [f"{tmp}/frame_change_{s}.txt" for s in SPLITS]
        lay = [x for x in make_all(f"{tmp}/d.npz", bl2, sl, f"{tmp}/figs3") if x.startswith("[layout]")]
        check("T9", not lay, f"no text overlaps or runs outside any figure, logs of 100, 20 and 30 worlds ({len(lay)} problems)")
        old_t, old_x = TITLES["cell"], globals()["rare_xlabel"]
        TITLES["cell"] = "Species per cell (geometric mean)"
        n3 = make_all(f"{tmp}/d.npz", bl2, sl, f"{tmp}/figs4")
        TITLES["cell"] = old_t
        check("T10", any(x.startswith("[layout] fig4_bands") and "geometric mean" in x for x in n3),
              "broken version caught: the first long title overlaps its neighbour")
        globals()["rare_xlabel"] = lambda ins: "distance (cells); inside all four: " + ", ".join(ins)
        n4 = make_all(f"{tmp}/d.npz", bl2, sl, f"{tmp}/figs5")
        globals()["rare_xlabel"] = old_x
        check("T11", any(x.startswith("[layout] fig4_bands") and "inside all four" in x for x in n4),
              "broken version caught: the first one-line rare-clumping label")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data"); ap.add_argument("--bnm-logs", nargs="*", default=[]); ap.add_argument("--speed-logs", nargs="*", default=[])
    ap.add_argument("--out", default="../figures/obj1"); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.data or not a.bnm_logs or not a.speed_logs:
        sys.exit("STOP: give --data, --bnm-logs (3 files) and --speed-logs (3 files)")
    make_all(a.data, a.bnm_logs, a.speed_logs, a.out)