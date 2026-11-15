#!/usr/bin/env python3
"""
trace_view.py  (obj1_v2, the real run: 3D trace pages of a few worlds, to show the simulation; 2 Nov 2026, Sara's yes)

For one training file, writes one HTML page that shows the world through time, one frame per habitat piece (200 IBM
steps = 1 BNM year, gate 7):
  1. a space-time cube (x, y, year): the cells held by a few species over the chosen years, one colour per species,
     so a species' range can be followed as it moves, spreads, shrinks or is lost (drag to turn it);
  2. a map that plays year by year (play button and slider): species per cell as a heat map, the chosen species' cells
     on top, and the regional shift of that year in the title;
  3. lines through the same years: species in the world and the range size of each chosen species.
Species are chosen from those present in at least 30% of the shown years, spread over range size (smallest to largest
median range), so rare and widespread species both appear. Presence is at 1 individual, as saved (ASM_F maps).
The page draws with plotly.js 2.35.2: copied into the page when the plotly Python package is installed (works offline),
otherwise loaded from cdnjs (needs internet). Reads the training file only; writes the page.

Run from obj1_v2/work (a few worlds, any split; the evaluation worlds are not used for this):
    python3 ../tools/trace_view.py --self-test | tail -1
    python3 ../tools/trace_view.py "results/stage2a/real_train/*_s5001_training.npz" --years 100 --species 6 --out ../logs/trace_rt5001.html
"""
import argparse, glob, json, os, sys, tempfile
from pathlib import Path
import numpy as np

CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"
COLOURS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#1f78b4"]
NEED = ["ASM_F_round", "ASM_F_piece", "ASM_F_offsets", "ASM_F_ids", "ASM_F_bits"]


def frames(z):
    """List of (round, piece, ids, P (S, n) bool) for every saved piece map."""
    Y, X = int(np.asarray(z["Y"]).item()) if "Y" in z.files else 20, int(np.asarray(z["X"]).item()) if "X" in z.files else 20
    n = Y * X
    off = np.asarray(z["ASM_F_offsets"]).astype(int); ids = np.asarray(z["ASM_F_ids"]).astype(int)
    bits = np.asarray(z["ASM_F_bits"]); rnd = np.asarray(z["ASM_F_round"]).astype(int); pc = np.asarray(z["ASM_F_piece"]).astype(int)
    out = []
    for k in range(len(rnd)):
        a, b = off[k], off[k + 1]
        out.append((int(rnd[k]), int(pc[k]), ids[a:b], np.unpackbits(bits[a:b], axis=1, count=n).astype(bool)))
    return out, Y, X


def build(z, years=100, n_species=6, name=""):
    """The page's data (dict) from one training file."""
    miss = [k for k in NEED if k not in z.files]
    if miss:
        raise ValueError(f"missing {miss}: a v2.6 file with habitat sub-steps is needed")
    fr, Y, X = frames(z)
    fr = fr[-years:]
    T = len(fr)
    shift = None
    if "ASM_F_shift" in z.files:
        shift = [float(v) for v in np.asarray(z["ASM_F_shift"])[-T:]]
    present = {}
    for t, (_, _, ids, P) in enumerate(fr):
        for s, row in zip(ids.tolist(), P):
            present.setdefault(s, {})[t] = np.flatnonzero(row)
    cand = [(s, float(np.median([len(c) for c in d.values()]))) for s, d in present.items() if len(d) >= 0.3 * T]
    cand.sort(key=lambda x: (x[1], x[0]))
    if len(cand) <= n_species:
        chosen = cand
    else:
        idx = np.unique(np.round(np.linspace(0, len(cand) - 1, n_species)).astype(int))
        chosen = [cand[i] for i in idx]
    rich = np.zeros((T, Y * X), int); n_sp = []
    for t, (_, _, ids, P) in enumerate(fr):
        rich[t] = P.sum(0); n_sp.append(int(len(ids)))
    species = []
    for k, (s, med) in enumerate(chosen):
        cube = {"x": [], "y": [], "t": []}
        per_t, size = [], []
        for t in range(T):
            c = present[s].get(t, np.array([], int))
            cube["x"] += (c % X).tolist(); cube["y"] += (c // X).tolist(); cube["t"] += [t] * len(c)
            per_t.append({"x": (c % X).tolist(), "y": (c // X).tolist()}); size.append(int(len(c)))
        species.append(dict(id=int(s), median=med, colour=COLOURS[k % len(COLOURS)], cube=cube, per_t=per_t, size=size))
    first = fr[0]
    return dict(name=name, Y=Y, X=X, T=T, rounds=[f[0] for f in fr], pieces=[f[1] for f in fr], shift=shift,
                richness=rich.tolist(), n_species=n_sp, species=species, start=f"round {first[0]}, piece {first[1]}")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>World trace</title>
<style>
:root{--bg:#ffffff;--fg:#1d2329;--muted:#5b6670;--line:#d9dee3}
@media (prefers-color-scheme: dark){:root{--bg:#14181c;--fg:#e8ecef;--muted:#9aa6b1;--line:#2c343b}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:8px 0 4px} p{color:var(--muted);margin:4px 0 12px}
.box{border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:4px}
.plot{width:100%;height:560px}.short{height:300px}
</style>__PLOTLY__</head><body><main>
<h1>__TITLE__</h1>
<p>__INTRO__</p>
<div class="box"><div id="cube" class="plot"></div></div>
<div class="box"><div id="map" class="plot"></div></div>
<div class="box"><div id="lines" class="plot short"></div></div>
</main><script>
const D = __DATA__;
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
const fg = dark ? '#e8ecef' : '#1d2329', grid = dark ? '#2c343b' : '#d9dee3';
const base = {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:fg}, margin:{l:50,r:20,t:50,b:40}};
const label = s => `species ${s.id} (median ${s.median} cells)`;
Plotly.newPlot('cube', D.species.map(s => ({type:'scatter3d', mode:'markers', name:label(s), x:s.cube.x, y:s.cube.y, z:s.cube.t,
  visible: s.median > 120 ? 'legendonly' : true, marker:{size:3, color:s.colour, opacity:0.75}})), Object.assign({}, base, {title:{text:'Space-time cube: cells held by each species, year by year'},
  scene:{xaxis:{title:'x (cell)', range:[-0.5, D.X - 0.5], gridcolor:grid}, yaxis:{title:'y (cell)', range:[-0.5, D.Y - 0.5], gridcolor:grid},
         zaxis:{title:'year', gridcolor:grid}, aspectmode:'manual', aspectratio:{x:1, y:1, z:1.4}}, legend:{orientation:'h'}}), {responsive:true});
const grid2 = t => { const z = []; for (let r = 0; r < D.Y; r++) z.push(D.richness[t].slice(r * D.X, (r + 1) * D.X)); return z; };
const zmax = Math.max(...D.richness.flat());
const title = t => `Year ${t + 1} of ${D.T} (round ${D.rounds[t]}, piece ${D.pieces[t]})` + (D.shift ? `; regional shift ${D.shift[t].toFixed(2)}` : '');
const OFF = [[-0.25,0.2],[0,0.2],[0.25,0.2],[-0.25,-0.2],[0,-0.2],[0.25,-0.2],[-0.12,0],[0.12,0]];
const mapData = t => [{type:'heatmap', z:grid2(t), zmin:0, zmax:zmax, colorscale:'Greys', reversescale:!dark,
  colorbar:{title:'species<br>per cell', x:-0.22}, hoverinfo:'z'}]
  .concat(D.species.map((s, k) => ({type:'scatter', mode:'markers', name:label(s),
  x:s.per_t[t].x.map(v => v + OFF[k % OFF.length][0]), y:s.per_t[t].y.map(v => v + OFF[k % OFF.length][1]),
  marker:{size:5, symbol:'square', color:s.colour, opacity:0.95}})));
const steps = [...Array(D.T).keys()].map(t => ({method:'animate', label:String(t + 1),
  args:[[String(t)], {mode:'immediate', frame:{duration:0, redraw:true}, transition:{duration:0}}]}));
Plotly.newPlot('map', mapData(0), Object.assign({}, base, {title:{text:title(0)},
  xaxis:{title:'x (cell)', range:[-0.5, D.X - 0.5], constrain:'domain', gridcolor:grid}, yaxis:{title:'y (cell)', range:[-0.5, D.Y - 0.5], scaleanchor:'x', gridcolor:grid},
  legend:{orientation:'v', x:1.02, y:1}, sliders:[{steps:steps, currentvalue:{prefix:'year '}, pad:{t:40}, font:{size:8}}],
  updatemenus:[{type:'buttons', showactive:false, x:-0.05, y:1.12, buttons:[
    {label:'Play', method:'animate', args:[null, {frame:{duration:150, redraw:true}, fromcurrent:true, transition:{duration:0}}]},
    {label:'Pause', method:'animate', args:[[null], {mode:'immediate', frame:{duration:0, redraw:false}}]}]}]}), {responsive:true})
 .then(() => Plotly.addFrames('map', [...Array(D.T).keys()].map(t => ({name:String(t), data:mapData(t), layout:{title:{text:title(t)}}}))));
const yrs = [...Array(D.T).keys()].map(t => t + 1);
Plotly.newPlot('lines', [{type:'scatter', mode:'lines', name:'species in the world', x:yrs, y:D.n_species, line:{color:fg, width:2}}]
  .concat(D.species.map(s => ({type:'scatter', mode:'lines', name:label(s) + ': cells', x:yrs, y:s.size, yaxis:'y2', line:{color:s.colour}}))),
  Object.assign({}, base, {margin:{l:50,r:70,t:50,b:40}, title:{text:'Species in the world and range size of each shown species'}, xaxis:{title:'year', gridcolor:grid},
  yaxis:{title:'species', gridcolor:grid}, yaxis2:{title:'cells held', overlaying:'y', side:'right'}, legend:{orientation:'h', y:-0.3}}), {responsive:true});
</script></body></html>"""


def plotly_tag(use_cdn):
    if not use_cdn:
        try:
            import plotly
            p = Path(os.path.dirname(plotly.__file__)) / "package_data" / "plotly.min.js"
            if p.is_file():
                return "<script>" + p.read_text() + "</script>", "plotly.js copied into the page (works offline)"
        except ImportError:
            pass
    return f'<script src="{CDN}"></script>', "plotly.js from cdnjs (needs internet)"


def write_page(data, out, use_cdn=False):
    tag, how = plotly_tag(use_cdn)
    seed = data["name"]
    intro = (f"{data['T']} years from {data['start']}; one frame per habitat piece (200 IBM steps, about one BNM year). "
             f"{len(data['species'])} species shown, chosen across range sizes; presence at 1 individual. In the cube, species "
             f"above 120 cells start hidden so smaller ranges stay visible: click a name in the legend to show or hide it. "
             f"On the map each species has its own place inside a cell, so species sharing a cell are all seen.")
    html = (PAGE.replace("__PLOTLY__", tag).replace("__TITLE__", f"World trace: {seed}").replace("__INTRO__", intro)
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    Path(out).write_text(html)
    return how, len(html)


# ------------------------------------------------------------------ self-test
def planted(path, T=30, X=20, Y=20):
    """Species 1 holds cell (x=t, y=2) in year t; species 2 holds a fixed 3 x 3 block at x 10-12, y 5-7; species 3 is
    present only in the first 5 years; species 4 joins at year 25."""
    rnd, pc, ids, bits, offs = [], [], [], [], [0]
    for t in range(T):
        rows, sid = [], []
        for s in (1, 2, 3, 4):
            m = np.zeros(Y * X, bool)
            if s == 1:
                m[2 * X + t % X] = True
            elif s == 2:
                for yy in (5, 6, 7):
                    for xx in (10, 11, 12):
                        m[yy * X + xx] = True
            elif s == 3 and t < 5:
                m[0] = True
            elif s == 4 and t >= 25:
                m[399] = True
            if m.any():
                rows.append(np.packbits(m)); sid.append(s)
        rnd.append(t // 5); pc.append(t % 5); ids += sid; bits += rows; offs.append(offs[-1] + len(sid))
    np.savez(path, X=np.int64(X), Y=np.int64(Y), ASM_F_round=np.array(rnd), ASM_F_piece=np.array(pc), ASM_F_offsets=np.array(offs),
             ASM_F_ids=np.array(ids), ASM_F_bits=np.array(bits, np.uint8), ASM_F_shift=np.linspace(-1, 1, T))
    return np.load(path)


def self_test():
    res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        z = planted(f"{tmp}/w_training.npz")
        d = build(z, years=30, n_species=6, name="planted")
        ids = [s["id"] for s in d["species"]]
        check("T1", ids == [1, 2], f"species present in 30% of years or more are shown, smallest range first: {ids} (3 and 4 too brief)")
        s1 = d["species"][0]
        check("T2", s1["cube"]["x"] == [t % 20 for t in range(30)] and set(s1["cube"]["y"]) == {2} and s1["cube"]["t"] == list(range(30)),
              "a species moving one cell east per year (wrapping on the torus) is drawn at x = year mod 20, y = 2 (x and y not swapped)")
        s2 = d["species"][1]
        cells = {(x, y) for x, y in zip(s2["per_t"][7]["x"], s2["per_t"][7]["y"])}
        check("T3", cells == {(x, y) for x in (10, 11, 12) for y in (5, 6, 7)} and s2["size"] == [9] * 30,
              "a fixed 3 x 3 block is drawn at x 10-12, y 5-7 in every year, 9 cells")
        check("T4", d["n_species"][:5] == [3] * 5 and d["n_species"][25:] == [3] * 5 and d["richness"][0][0] == 1
              and d["richness"][29][399] == 1, "species per year and per cell counted from the maps")
        d2 = build(z, years=10, n_species=6, name="planted")
        check("T5", d2["T"] == 10 and d2["rounds"][0] == 4 and abs(d2["shift"][0] - np.linspace(-1, 1, 30)[20]) < 1e-12,
              "the last 10 years are taken, with their rounds and shifts")
        how, size = write_page(d, f"{tmp}/p.html", use_cdn=True)
        html = Path(f"{tmp}/p.html").read_text()
        back = json.loads(html.split("const D = ")[1].split(";\nconst dark")[0])
        check("T6", back == json.loads(json.dumps(d)) and CDN in html and "__" not in html.replace("__proto__", ""),
              f"the page holds the data unchanged and no placeholder is left ({size} bytes, {how})")
        np.savez(f"{tmp}/old_training.npz", P_last_final=np.zeros((2, 20, 20)))
        try:
            build(np.load(f"{tmp}/old_training.npz")); ok7b = False
        except ValueError:
            ok7b = True
        check("T7", ok7b, "a file without piece maps (before assembly v2.6) is refused")
    print(f"\n{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--years", type=int, default=100)
    ap.add_argument("--species", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cdn", action="store_true", help="load plotly.js from cdnjs instead of copying it into the page")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    files = sorted(glob.glob(a.file or ""))
    if len(files) != 1:
        sys.exit(f"STOP: the pattern must match exactly one training file (matched {len(files)})")
    z = np.load(files[0], allow_pickle=True)
    seed = int(np.asarray(z["SIM_SEED"]).item()) if "SIM_SEED" in z.files else -1
    data = build(z, a.years, a.species, name=f"seed {seed}")
    out = a.out or f"trace_s{seed}.html"
    how, size = write_page(data, out, a.cdn)
    print(f"{Path(files[0]).name[-60:]}: {data['T']} years, {len(data['species'])} species shown "
          f"({', '.join(str(s['id']) + ': median ' + str(s['median']) + ' cells' for s in data['species'])}); "
          f"written {out} ({size / 1e6:.1f} MB, {how})")