#!/usr/bin/env python3
"""
stage_f_trace.py  (obj2_v2, stage F step 3b, 3D trace page, 9 Nov 2026; asked by Sara on 8 Nov, agreed 9 Nov)

Question: what do the trained model's maps look like next to the truth and the records, year by year, for real
validation species? One HTML page per run, in the style of obj1_v2/tools/trace_view.py, made only from files the
run and step 3a already wrote (no new sampling, no GPU):
    best.pt's 50 samples per stack (check_3a/preds_k50.npz), the frozen real_val set, its input and truth files,
    B1 from stage_d_headroom_pred.npz, log.csv and run_info.json.

Stacks shown: a fixed random draw of 2 per true class (1, 2-5, 6-10, 11-20, 21-50, 51+) among the stacks step 3a
scored, so the page is not hand-picked; --stacks overrides. For the stack chosen in the list:
    1. space-time cube (x, y, year): true cells, records, and cells the model holds in 10% of samples or more
       (colour = share of samples); drag to turn it
    2. maps that play the 6 years: truth, records on surveyed cells, model share of samples, B1 (pooled 6 years,
       scaled to its largest value; the same in every year)
    3. lines per year: true AOO, expected AOO (mean over the 50 samples) with its 5th-95th percentile band, records
    4. training: loss per epoch, validation score and AUC per true class (8 samples), best epoch marked; another run's
       validation score beside it with --compare
x is the column and y the row of the 20 x 20 window (as in trace_view.py).

Checks (the tool stops on a failure)
    T0  check_3a was made from this best.pt (md5 in report.json); its 50-sample counts and the frozen set match
    T1  each shown stack's AUC (model, 50 samples; B1) recomputed here equals step 3a's per_stack.csv to 1e-5
    T2  expected AOO per year equals the mean of the per-sample AOO stored by step 3a
    real_eval is refused.

Run (from PSD_Dispersal_pool/obj2_v2/work; seconds, CPU; needs the plotly package as for trace_view.py)
    python3 ../tools/stage_f_trace.py --self-test 2>&1 | tail -1
    python3 ../tools/stage_f_trace.py --run ../runs/f_seed1 --data ../data --compare ../runs/f_seed2 --out ../logs/trace_f_seed1.html
"""
import argparse, csv, importlib.util, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"
SHOW_MIN = 0.10
BROKEN = os.environ.get("STAGE_F_TRACE_BROKEN", "")     # self-test only


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_f_trace.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


CK = _load("stage_f_check")
F, HR, E, D, X = CK.F, CK.HR, CK.E, CK.D, CK.X
G, W = CK.G, CK.W


def cells(m):
    """bool (G, G) -> {'x': columns, 'y': rows}"""
    r, c = np.nonzero(m)
    return {"x": c.tolist(), "y": r.tolist()}


def read_log(run):
    rows = list(csv.DictReader(open(Path(run) / "log.csv")))
    ep = [int(r["epoch"]) for r in rows]
    loss = [float(r["train_loss"]) for r in rows]
    val = [(int(r["epoch"]), float(r["val_score"])) for r in rows if r.get("val_score")]
    per = {c: [float(r[f"auc_{c}"]) for r in rows if r.get("val_score")] for c in D.TRUE[1:]}
    info = json.loads((Path(run) / "run_info.json").read_text())
    return dict(name=Path(run).name, epoch=ep, loss=loss, val_epoch=[v[0] for v in val], val=[v[1] for v in val], per=per,
                best_epoch=int(info["end"]["best_epoch"]), best=float(info["end"]["best"]))


def build(run, data, stacks=None, per_class=2, seed=23, compare=None):
    run, data = Path(run), Path(data)
    for p in [run, data] + ([Path(c) for c in compare] if compare else []):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G")
    c3 = run / "check_3a"
    if not (c3 / "preds_k50.npz").is_file() or not (c3 / "report.json").is_file():
        sys.exit(f"STOP: {c3} has no step 3a output; run stage_f_check.py on this run first")
    rep = json.loads((c3 / "report.json").read_text().replace("NaN", "null"))
    t0 = rep["md5"]["best_pt"] == X.file_md5(run / "best.pt") and rep["md5"]["preds"] == X.file_md5(c3 / "preds_k50.npz")
    pz = np.load(c3 / "preds_k50.npz")
    frozen = dict(np.load(data / "stage_d_frozen_val.npz"))
    n = len(frozen["row"])
    K = int(pz["K"])
    t0 = t0 and pz["counts"].shape[0] == n and int(pz["counts"].max()) <= K
    if not t0:
        sys.exit("STOP: T0 check_3a does not match this best.pt, its predictions or the frozen set")
    ps = list(csv.DictReader(open(c3 / "per_stack.csv")))
    auc_m = np.array([float(r["auc_model"]) if r["auc_model"] not in ("nan", "") else np.nan for r in ps])
    auc_b = np.array([float(r["auc_b1"]) if r["auc_b1"] not in ("nan", "") else np.nan for r in ps])
    tcl = np.array([r["true_class"] for r in ps])
    if stacks is None:
        g = X.rng(seed, 5)
        pick = []
        for c in D.TRUE[1:]:
            idx = np.nonzero((tcl == c) & np.isfinite(auc_m))[0]
            pick += sorted(g.choice(idx, min(per_class, len(idx)), replace=False).tolist()) if len(idx) else []
        stacks = pick
    sel = np.array(sorted(set(int(s) for s in stacks)), int)
    voff, vblank = E.frozen_draws(frozen)
    fi, rw = frozen["file_index"][sel], frozen["row"][sel]
    Tc, tc = E.targets(data / "real_val", frozen["files"], fi, rw, voff[sel])
    T = CK.unplace(Tc, voff[sel]) > 0
    VA = E.Inputs(data / "real_val", frozen["files"])
    V, Ef, _ = VA.arrays(fi, rw)
    cnt = CK.unplace(pz["counts"][sel].astype(np.float32), voff[sel]) / K
    if BROKEN == "transpose":
        cnt = cnt.transpose(0, 1, 3, 2)
    hz = dict(np.load(data / "stage_d_headroom_pred.npz"))
    B1 = hz["B1"][sel].astype(np.float64)
    Mk = ~V[:, W - 1]
    am = HR.aucs(cnt[:, W - 1], T[:, W - 1], Mk)
    ab = HR.aucs(B1, T[:, W - 1], Mk)
    t1 = np.allclose(am, auc_m[sel], atol=1e-5, equal_nan=True) and np.allclose(ab, auc_b[sel], atol=1e-5, equal_nan=True)
    if not t1:
        sys.exit(f"STOP: T1 AUC recomputed here differs from step 3a's per_stack.csv (model {np.round(am, 5)} against "
                 f"{np.round(auc_m[sel], 5)})")
    aoo = pz["aoo"][sel].astype(np.float64)                 # (n, 6, K)
    exp = aoo.mean(2)
    occ_sum = cnt.reshape(len(sel), W, -1).sum(2)           # sum of window-cell shares = mean AOO over samples
    t2 = np.allclose(exp, occ_sum, atol=1e-4)
    if not t2:
        sys.exit("STOP: T2 expected AOO differs from the per-sample AOO of step 3a")
    comm = np.array(VA.seed)[fi]
    rec_t = V[:, W - 1].reshape(len(sel), -1).sum(1); rec6 = V.any(1).reshape(len(sel), -1).sum(1)
    out = []
    for j, i in enumerate(sel):
        b1 = B1[j] / B1[j].max() if B1[j].max() > 0 else B1[j]
        out.append(dict(
            stack=int(i), community=int(comm[j]), true_class=D.TRUE[tc[j]], rec_target=int(rec_t[j]), rec6=int(rec6[j]),
            blank=int(frozen["blank"][i]), auc_model=None if np.isnan(am[j]) else round(float(am[j]), 4),
            auc_b1=None if np.isnan(ab[j]) else round(float(ab[j]), 4),
            truth=[cells(T[j, y]) for y in range(W)], records=[cells(V[j, y]) for y in range(W)],
            truth_grid=[T[j, y].astype(int).tolist() for y in range(W)],
            rec_grid=[(V[j, y].astype(int) + Ef[j, y].astype(int)).tolist() for y in range(W)],
            freq=[np.round(cnt[j, y], 3).tolist() for y in range(W)], b1=np.round(b1, 3).tolist(),
            aoo_true=T[j].reshape(W, -1).sum(1).tolist(), aoo_exp=np.round(exp[j], 2).tolist(),
            aoo_p5=np.percentile(aoo[j], 5, axis=1).round(1).tolist(), aoo_p95=np.percentile(aoo[j], 95, axis=1).round(1).tolist(),
            rec_year=V[j].reshape(W, -1).sum(1).tolist()))
    logs = [read_log(run)] + [read_log(c) for c in (compare or [])]
    return dict(run=run.name, K=K, G=G, W=W, show_min=SHOW_MIN, best_epoch=int(rep["best_epoch"]),
                auc_balanced=rep["auc_balanced"]["model"], b1_balanced=rep["auc_balanced"]["B1"], stacks=out, logs=logs)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model trace</title>
<style>
:root{--bg:#ffffff;--fg:#1d2329;--muted:#5b6670;--line:#d9dee3}
@media (prefers-color-scheme: dark){:root{--bg:#14181c;--fg:#e8ecef;--muted:#9aa6b1;--line:#2c343b}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:8px 0 4px} p{color:var(--muted);margin:4px 0 12px}
select{font:inherit;padding:4px;max-width:100%}
.box{border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:4px}
.plot{width:100%;height:560px}.tall{height:760px}.short{height:320px}
</style>__PLOTLY__</head><body><main>
<h1>__TITLE__</h1>
<p>__INTRO__</p>
<p><label for="pick">Stack: </label><select id="pick"></select></p>
<p id="info"></p>
<div class="box"><div id="cube" class="plot"></div></div>
<div class="box"><div id="map" class="plot tall"></div></div>
<div class="box"><div id="lines" class="plot short"></div></div>
<div class="box"><div id="train" class="plot short"></div></div>
</main><script>
if (typeof Plotly === 'undefined') {
  for (const id of ['cube', 'map', 'lines', 'train']) document.getElementById(id).innerHTML = '<p style="padding:12px">plotly.js did not load, so ' +
    'nothing can be drawn. Make the page again with the plotly package installed (python3 -m pip install plotly==5.24.1), ' +
    'or, for a page made with --cdn, open it with internet.</p>';
  throw new Error('plotly.js did not load');
}
const D = __DATA__;
const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
const fg = dark ? '#e8ecef' : '#1d2329', grid = dark ? '#2c343b' : '#d9dee3';
const base = {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{color:fg}, margin:{l:50,r:20,t:50,b:40}};
const yrs = [...Array(D.W).keys()].map(t => t + 1);
const sel = document.getElementById('pick');
D.stacks.forEach((s, i) => { const o = document.createElement('option'); o.value = String(i);
  o.textContent = `stack ${s.stack}, community ${s.community}: true class ${s.true_class}, ${s.rec6} records over 6 years`; sel.appendChild(o); });
const fmt = v => v === null ? 'not scored' : v.toFixed(3);
function cube(s) {
  const tr = {x:[], y:[], z:[]}, rc = {x:[], y:[], z:[]}, md = {x:[], y:[], z:[], c:[]};
  for (let t = 0; t < D.W; t++) {
    s.truth[t].x.forEach((x, k) => { tr.x.push(x); tr.y.push(s.truth[t].y[k]); tr.z.push(t + 1); });
    s.records[t].x.forEach((x, k) => { rc.x.push(x); rc.y.push(s.records[t].y[k]); rc.z.push(t + 1); });
    for (let r = 0; r < D.G; r++) for (let c = 0; c < D.G; c++) { const f = s.freq[t][r][c];
      if (f >= D.show_min) { md.x.push(c); md.y.push(r); md.z.push(t + 1); md.c.push(f); } }
  }
  const ax = n => ({title:n, range:[-0.5, D.G - 0.5], gridcolor:grid});
  Plotly.newPlot('cube', [
    {type:'scatter3d', mode:'markers', name:'model: share of samples >= ' + D.show_min, x:md.x, y:md.y, z:md.z,
     marker:{size:3, color:md.c, colorscale:'Viridis', cmin:0, cmax:1, opacity:0.5, colorbar:{title:'share', x:1.02}}},
    {type:'scatter3d', mode:'markers', name:'true cells', x:tr.x, y:tr.y, z:tr.z, marker:{size:4, symbol:'square', color:'#1b9e77'}},
    {type:'scatter3d', mode:'markers', name:'records', x:rc.x, y:rc.y, z:rc.z, marker:{size:5, symbol:'diamond', color:'#d95f02'}}],
    Object.assign({}, base, {title:{text:'Space-time cube: truth, records and the model, year by year'},
    scene:{xaxis:ax('x (column)'), yaxis:ax('y (row)'), zaxis:{title:'year', range:[0.5, D.W + 0.5], gridcolor:grid},
    aspectmode:'manual', aspectratio:{x:1, y:1, z:1}}, legend:{orientation:'h'}}), {responsive:true});
}
function mapData(s, t) {
  const hm = (z, xa, ya, cs, zmax, title) => ({type:'heatmap', z:z, xaxis:xa, yaxis:ya, colorscale:cs, zmin:0, zmax:zmax,
    showscale:false, hovertemplate:title + ' x %{x} y %{y}: %{z}<extra></extra>'});
  return [hm(s.truth_grid[t], 'x', 'y', [[0,'#f2f2f2'],[1,'#1b9e77']], 1, 'truth'),
          hm(s.rec_grid[t], 'x2', 'y2', [[0,'#ffffff'],[0.5,'#c8d3dc'],[1,'#d95f02']], 2, 'not surveyed 0, surveyed 1, record 2'),
          hm(s.freq[t], 'x3', 'y3', 'Viridis', 1, 'model share'),
          hm(s.b1, 'x4', 'y4', 'Viridis', 1, 'B1 scaled')];
}
function maps(s) {
  const ax = (d, a) => ({domain:d, range:[-0.5, D.G - 0.5], constrain:'domain', gridcolor:grid, anchor:a});
  const lay = Object.assign({}, base, {title:{text:`Year 1 of ${D.W}`}, showlegend:false,
    xaxis:ax([0, 0.46], 'y'), yaxis:ax([0.56, 1], 'x'), xaxis2:ax([0.54, 1], 'y2'), yaxis2:ax([0.56, 1], 'x2'),
    xaxis3:ax([0, 0.46], 'y3'), yaxis3:ax([0, 0.44], 'x3'), xaxis4:ax([0.54, 1], 'y4'), yaxis4:ax([0, 0.44], 'x4'),
    annotations:[['truth', 0.23, 1.0], ['records (orange) on surveyed cells (grey)', 0.77, 1.0],
      [`model: share of ${D.K} samples`, 0.23, 0.46], ['B1 smoother, pooled 6 years (scaled)', 0.77, 0.46]]
      .map(a => ({text:a[0], x:a[1], y:a[2], xref:'paper', yref:'paper', showarrow:false, yanchor:'bottom'})),
    sliders:[{steps:yrs.map(t => ({method:'animate', label:String(t), args:[[String(t - 1)], {mode:'immediate', frame:{duration:0, redraw:true}, transition:{duration:0}}]})),
      currentvalue:{prefix:'year '}, pad:{t:30}}],
    updatemenus:[{type:'buttons', showactive:false, x:-0.05, y:1.1, buttons:[
      {label:'Play', method:'animate', args:[null, {frame:{duration:700, redraw:true}, fromcurrent:true, transition:{duration:0}}]},
      {label:'Pause', method:'animate', args:[[null], {mode:'immediate', frame:{duration:0, redraw:false}}]}]}]});
  Plotly.newPlot('map', mapData(s, 0), lay, {responsive:true})
    .then(() => Plotly.addFrames('map', yrs.map(t => ({name:String(t - 1), data:mapData(s, t - 1), layout:{title:{text:`Year ${t} of ${D.W}`}}}))));
}
function lines(s) {
  Plotly.newPlot('lines', [
    {type:'scatter', mode:'lines', x:yrs, y:s.aoo_p95, line:{width:0}, showlegend:false, hoverinfo:'skip'},
    {type:'scatter', mode:'lines', x:yrs, y:s.aoo_p5, fill:'tonexty', fillcolor:'rgba(68,1,84,0.18)', line:{width:0}, name:'model 5th-95th percentile'},
    {type:'scatter', mode:'lines+markers', x:yrs, y:s.aoo_exp, name:'model expected AOO', line:{color:'#440154'}},
    {type:'scatter', mode:'lines+markers', x:yrs, y:s.aoo_true, name:'true AOO', line:{color:'#1b9e77'}},
    {type:'scatter', mode:'markers', x:yrs, y:s.rec_year, name:'recorded cells', marker:{color:'#d95f02', symbol:'diamond'}}],
    Object.assign({}, base, {title:{text:'Cells occupied per year (AOO in window cells)'}, xaxis:{title:'year', dtick:1, gridcolor:grid},
    yaxis:{title:'cells', rangemode:'tozero', gridcolor:grid}, legend:{orientation:'h', y:-0.3}}), {responsive:true});
}
function train() {
  const L = D.logs[0], tr = [{type:'scatter', mode:'lines', x:L.epoch, y:L.loss, name:L.name + ' training loss', line:{color:fg}}];
  const cols = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02'];
  Object.keys(L.per).forEach((c, k) => tr.push({type:'scatter', mode:'lines', x:L.val_epoch, y:L.per[c], yaxis:'y2',
    name:'AUC ' + c, line:{color:cols[k % cols.length], dash:'dot', width:1}}));
  D.logs.forEach((g, k) => tr.push({type:'scatter', mode:'lines+markers', x:g.val_epoch, y:g.val, yaxis:'y2',
    name:g.name + ' validation score', line:{color:k === 0 ? '#440154' : '#3b75af', width:2}}));
  Plotly.newPlot('train', tr, Object.assign({}, base, {margin:{l:60,r:70,t:50,b:40},
    title:{text:`Training (8 samples per stack); best epoch ${L.best_epoch}, score ${L.best.toFixed(4)}`},
    xaxis:{title:'epoch', gridcolor:grid}, yaxis:{title:'loss', type:'log', gridcolor:grid},
    yaxis2:{title:'AUC', overlaying:'y', side:'right'}, legend:{orientation:'h', y:-0.35},
    shapes:[{type:'line', x0:L.best_epoch, x1:L.best_epoch, yref:'paper', y0:0, y1:1, line:{dash:'dash', color:'#888'}}]}), {responsive:true});
}
function draw(i) {
  const s = D.stacks[i];
  document.getElementById('info').textContent = `Stack ${s.stack} (community ${s.community}); true class ${s.true_class}; ` +
    `recorded cells in the target year ${s.rec_target}, records over 6 years ${s.rec6}; habitat ${s.blank ? 'blanked' : 'given'}; ` +
    `AUC in the target year: model ${fmt(s.auc_model)}, B1 ${fmt(s.auc_b1)}; true AOO ${s.aoo_true[D.W - 1]}, ` +
    `expected ${s.aoo_exp[D.W - 1]} (${s.aoo_p5[D.W - 1]}-${s.aoo_p95[D.W - 1]}).`;
  cube(s); maps(s); lines(s);
}
sel.addEventListener('change', () => draw(Number(sel.value)));
draw(0); train();
</script></body></html>"""


def find_plotly():
    try:
        import plotly
    except ImportError:
        return None
    p = Path(os.path.dirname(plotly.__file__)) / "package_data" / "plotly.min.js"
    return p if p.is_file() else None


def plotly_tag(use_cdn, find=find_plotly):
    if use_cdn:
        return f'<script src="{CDN}"></script>', "plotly.js from cdnjs (needs internet; chosen with --cdn)"
    p = find()
    if p is None:
        raise SystemExit("STOP: the plotly Python package is not installed, so plotly.js cannot be copied into the page. "
                         "Install it: python3 -m pip install plotly==5.24.1, or add --cdn for a page that needs internet.")
    return "<script>" + Path(p).read_text() + "</script>", "plotly.js copied into the page (works offline)"


def write_page(d, out, use_cdn=False, find=find_plotly):
    tag, how = plotly_tag(use_cdn, find)
    intro = (f"Run {d['run']}, best epoch {d['best_epoch']}; frozen real_val set, {len(d['stacks'])} stacks drawn at random per "
             f"true class among those step 3a scored. Class-balanced AUC in the target year with {d['K']} samples: model "
             f"{d['auc_balanced']:.3f}, B1 {d['b1_balanced']:.3f}. Model cells are the share of the {d['K']} samples that "
             f"hold the cell; the cube shows cells held in {int(100 * d['show_min'])}% of samples or more. x is the column, "
             f"y the row of the 20 x 20 window.")
    html = (PAGE.replace("__PLOTLY__", tag).replace("__TITLE__", f"Model trace: {d['run']}").replace("__INTRO__", intro)
            .replace("__DATA__", json.dumps(d, separators=(",", ":"))))
    Path(out).write_text(html)
    return how, len(html)


# ---------------------------------------------------------------- self-test
STUB = r"""
const calls = [], els = {};
function el(id) { if (!els[id]) els[id] = {id, innerHTML:'', textContent:'', children:[], value:'0', listeners:{},
  appendChild(c) { this.children.push(c); }, addEventListener(e, f) { this.listeners[e] = f; }}; return els[id]; }
global.document = {getElementById: el, createElement: () => ({value:'', textContent:''})};
global.window = {matchMedia: () => ({matches:false})};
global.Plotly = {newPlot: (id, data, lay) => { calls.push([id, data.length, data, lay]); return Promise.resolve(); },
                 addFrames: (id, fr) => { calls.push([id + ':frames', fr.length, fr]); return Promise.resolve(); }};
"""


def js_run(html, tmp, pick=1):
    """Run the page's own script in node with a stand-in for the browser and Plotly; returns the calls made."""
    js = html.split("</script></body>")[0].split("<script>")[-1]
    prog = STUB + js + f"""
const s = els['pick']; s.value = '{pick}'; s.listeners['change']();
setTimeout(() => {{ const cube = calls.filter(c => c[0] === 'cube').pop(), fr = calls.filter(c => c[0] === 'map:frames').pop();
  console.log(JSON.stringify({{names: calls.map(c => c[0]), info: els['info'].textContent, cube_x: cube[2][1].x, cube_y: cube[2][1].y,
    cube_z: cube[2][1].z, frames: fr[1], frame6_model: fr[2][5].data[2].z}})); }}, 50);
"""
    p = Path(tmp) / "page.js"; p.write_text(prog)
    r = subprocess.run(["node", str(p)], capture_output=True, text=True)
    return r


def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = X.plant_grids(tmp / "g")
        st = tmp / "sim" / "real_train"; st.mkdir(parents=True); sv = tmp / "sim" / "real_val"; sv.mkdir(parents=True)
        for s_ in (7001, 7002):
            X.plant_community(st / f"c{s_}_training.npz", s_)
        for s_ in (7101, 7102):
            X.plant_community(sv / f"c{s_}_training.npz", s_)
        base = tmp / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.6, py, refpat, tools, out=quiet)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.6, py, refpat, tools, out=quiet)
        D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
        hres, hinfo = HR.run(str(base), str(base / "real_train"), str(base / "real_val"), str(base), out=quiet)
        e1, e2 = round(hinfo["bal"]["B1"], 3), round(hinfo["bal"]["B2"], 3)
        cfg = dict(F.CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=2, K=2, ddim=4,
                   val_set="full", val_chunk=64, lr=1e-3)
        F.train(cfg, str(base), str(tmp / "run"), "cpu", log=quiet)
        CK.run(tmp / "run", base, K=6, ddim=4, dev="cpu", expect_b1=e1, expect_b2=e2, out=quiet)
        d = build(tmp / "run", base)
        frozen = dict(np.load(base / "stage_d_frozen_val.npz"))
        voff, _ = E.frozen_draws(frozen)
        ps = list(csv.DictReader(open(tmp / "run" / "check_3a" / "per_stack.csv")))
        cls = [ps[s["stack"]]["true_class"] for s in d["stacks"]]
        present = sorted({r["true_class"] for r in ps if r["auc_model"] not in ("nan", "") and r["true_class"] != "0"})
        t("T1", sorted(set(cls)) == present and all(cls.count(c) <= 2 for c in present) and build(tmp / "run", base)["stacks"] == d["stacks"],
          f"stacks drawn per true class among scored stacks, at most 2 each, same draw on a rerun ({len(d['stacks'])} stacks)")
        s0 = d["stacks"][0]; i0 = s0["stack"]
        Tc, _ = E.targets(base / "real_val", frozen["files"], frozen["file_index"][[i0]], frozen["row"][[i0]], voff[[i0]])
        Tw = CK.unplace(Tc, voff[[i0]])[0] > 0
        pz = np.load(tmp / "run" / "check_3a" / "preds_k50.npz")
        fq = CK.unplace(pz["counts"][[i0]].astype(np.float32), voff[[i0]])[0] / 6
        ok_t = all(sorted(zip(s0["truth"][y]["x"], s0["truth"][y]["y"])) == sorted((c, r) for r, c in zip(*np.nonzero(Tw[y]))) for y in range(W))
        t("T2", ok_t and np.allclose(np.array(s0["freq"]), np.round(fq, 3)) and s0["aoo_true"] == Tw.reshape(W, -1).sum(1).tolist(),
          "page data equal the truth files and step 3a counts recomputed apart; x = column, y = row")
        how, size = write_page(d, tmp / "p.html", use_cdn=find_plotly() is None)
        html = (tmp / "p.html").read_text()
        back = json.loads(html.split("const D = ")[1].split(";\nconst dark")[0])
        left = [k for k in ("__PLOTLY__", "__TITLE__", "__INTRO__", "__DATA__") if k in html]
        t("T3", back == json.loads(json.dumps(d)) and not left and "typeof Plotly === 'undefined'" in html,
          f"the page holds the data unchanged, no placeholder left, plotly guard present ({size / 1e6:.1f} MB, {how})")
        pick = 1 if len(d["stacks"]) > 1 else 0
        skipped = []
        if shutil.which("node") is None:
            skipped.append("T4")
            print("T4 the page's script in node: SKIPPED (node is not installed here; it ran in the sandbox)")
        r = js_run(html, tmp, pick=pick) if not skipped else None
        try:
            o = json.loads(r.stdout.strip().splitlines()[-1]) if r is not None else None
        except (ValueError, IndexError):
            o = {}
        s1 = d["stacks"][pick]
        if o is not None:
            ex = sum((s1["truth"][y]["x"] for y in range(W)), []); ey = sum((s1["truth"][y]["y"] for y in range(W)), [])
            ez = sum(([y + 1] * len(s1["truth"][y]["x"]) for y in range(W)), [])
            cube_ok = o.get("cube_x") == ex and o.get("cube_y") == ey and o.get("cube_z") == ez
            fr_ok = o.get("frames") == W and np.allclose(np.array(o.get("frame6_model", [[0]])), np.array(s1["freq"][W - 1]))
            t("T4", r.returncode == 0 and {"cube", "map", "lines", "train", "map:frames"} <= set(o.get("names", [])) and cube_ok
              and fr_ok and str(o.get("info", "")).startswith(f"Stack {s1['stack']}"),
              f"the page's script runs in node with a stand-in browser: all panels drawn, the chosen stack's truth in the cube, "
              f"6 year frames with its model map ({r.stderr.strip()[-150:] or 'no error'})")
        env = dict(os.environ); env["STAGE_F_TRACE_BROKEN"] = "transpose"
        code = ("import importlib.util as u; s=u.spec_from_file_location('t', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                "m.build(%r, %r)" % (str(Path(__file__).resolve()), str(tmp / "run"), str(base)))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        t("bug transpose", p.returncode != 0 and "T1" in (p.stdout + p.stderr),
          "model maps with rows and columns swapped stop at T1 (AUC no longer equals step 3a's)")
        rep = json.loads((tmp / "run" / "check_3a" / "report.json").read_text())
        rep["md5"]["best_pt"] = "0" * 32
        (tmp / "run" / "check_3a" / "report.json").write_text(json.dumps(rep))
        stop = False
        try:
            build(tmp / "run", base)
        except SystemExit as e:
            stop = "T0" in str(e)
        t("T5", stop, "a step 3a folder made from another best.pt is refused (T0)")
        stop = False
        try:
            build(tmp / "run", tmp / "real_eval")
        except SystemExit:
            stop = True
        t("T6", stop, "a real_eval path is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL" + (f" ({', '.join(skipped)} skipped: node not installed)" if skipped else ""))
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", default="../runs/f_seed1"); ap.add_argument("--data", default="../data")
    ap.add_argument("--compare", nargs="*", default=None, help="other runs whose validation score is drawn beside")
    ap.add_argument("--stacks", type=int, nargs="*", default=None, help="frozen-set stack numbers instead of the random draw")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cdn", action="store_true")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    d = build(a.run, a.data, stacks=a.stacks, per_class=a.per_class, compare=a.compare)
    out = a.out or f"trace_{Path(a.run).name}.html"
    how, size = write_page(d, out, a.cdn)
    print(f"stage F trace {d['run']}: T0, T1, T2 PASS; {len(d['stacks'])} stacks ("
          + ", ".join(f"{s['stack']} {s['true_class']}" for s in d["stacks"]) + f"); written {out} ({size / 1e6:.1f} MB, {how})")