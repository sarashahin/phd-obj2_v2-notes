#!/usr/bin/env python3
"""
stage_g_baselines.py  (obj2_v2, stage G step G2, 10 Nov 2026; B3, B4, B5 design signed off by Sara on 10 Nov)
v2 (10 Nov, after Sara's first run): B4 refitted on the distance to the nearest record (Sara's Q2: with log B1 the far
corners of the window, log B1 about -5,800, set a slope of +0.0038 and a nearly flat chance); torch warning fixed. B1,
B3 and B5 are unchanged.

Question: is the model better than the simple methods a reviewer would try first, for ranking (AUC) now and, in G3,
for the AOO level and the interval coverage? B1 and B2 (stage D) rank cells but give no maps. G2 adds three baselines
that give maps. All are fitted on real_train only and run on the frozen real_val set (16,347 stacks) from inputs only.

    B3  dynamic occupancy with imperfect detection; mean-field forward filter over the 6 years, 20 x 20 window, no wrap
          start year      logit psi1  = c0 + c1 dh
          colonisation    logit gamma = a0 + a1 nb + a2 dh     nb = mean filtered chance of the in-window neighbours
                                                                    last year (8 inside, 5 on an edge, 3 at a corner)
          extinction      logit eps   = b0 + b1 dh
          detection       p per visit; two visits per surveyed cell-year; no false positives; visits only in surveyed
                          cells. p = 2r / (1 + r), r = share of cell-years with both visits recording among those with
                          at least one (visit1 and visit2 only; it does not depend on the occupancy part)
        dh = |z - mean z at the recorded cells|, z = noisy habitat standardised per stack (stage E channel hab_dist).
        Parameters per records-over-6-years bin (1-3, 4-6, 7-10, 11-20, 21+) by maximum likelihood of the records
        (torch, float64, L-BFGS) on up to 4,000 real_train stacks per bin drawn with the stage D weights (fixed seed);
        the truth is not used. A no-habitat version (a0, a1, b0, c0) is fitted on the same stacks; "B3 half-blank"
        uses it where the frozen set blanks the habitat (what the model saw). Score = filtered chance at the target
        year. Stated limits: no long-jump term beyond the constant a0; mean field treats cells as independent given
        last year's chances; earlier years are filtered (records up to that year), only the target year is scored.
    B4  the records made into a chance by distance: logit = a + b d, d = distance in cells from the cell to the nearest
        cell recorded in any of the 6 years (0 at a recorded cell), 2 parameters by logistic fit on the unrecorded
        target-year cells of the stage D fit sample (6,000 real_train stacks, unweighted rows); a cell recorded in a year
        is on in that year. No true range size is used. It ranks as -d when b < 0.
    B5  records only: score = last year with a record / 6 (0 if never recorded); map of year t = cells recorded in years
        1..t, so its target-year AOO is the cells recorded over the 6 years (records AOO).
    Maps for G3: B3 half-blank and B4, 50 maps per stack drawn cell by cell (independent per cell and year); B5 one
    map (K 1). Same packed layout as G1 (stage_g_samples.LAYOUT).

Gates (rules written before the run)
    P1  same inputs as stage D: input files md5 = the sampler and frozen-set lists; stage_d_headroom_pred.npz made from
        this frozen set; B1 recomputed here from the inputs equals the stored B1 of every stack (float32, exact)
    P2  the fits open real_train files only
    P3  no truth file opened while the real_val predictions and maps are built; all written with md5 before any truth
    P4  no cell recorded at the target year is scored; a random score gives 0.50 (within 0.01 or 4 SE); B4 slope b < 0
        and B4 AUC = AUC of -d in every stack (within 1e-9)
    B1 ties (found while testing G2, 10 Nov): stage D stored B1 as float32, so every cell farther than about 3.6 cells
    from all records (sigma 0.25) is 0 and tied; stage D chose sigma with float64 B1. G2 scores B1 both ways: "B1" as
    stored (the F5 comparator) and "B1 exact" (log-sum-exp, no ties by underflow). It reports the gap by class (flag
    above 0.005) and rechecks F5 (model - B1 exact, same rule and resampling) for each run given with --runs.
    P5  maps: every cell recorded in a year is on in that year in every map; mean sample AOO at the target year matches
        the sum of the chances (pooled |z| <= 4); the packed maps unpack to the drawn maps
    P6  (with --runs) step 3a per_stack.csv of each run is for these stacks (community, file, row), and its B1 AUC (as
        the same 5-decimal text) and true AOO equal the values here
    Flag: fitted p outside 0.37-0.58 (export rule: two visits with 1 - sqrt(1 - d), d 0.64 or 0.80, gives 0.400, 0.553).
    real_eval is refused; the output folder must be new.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU only, so it can run while G1 holds the GPU)
    python3 ../tools/stage_g_baselines.py --self-test | tail -1
    nohup python3 -u ../tools/stage_g_baselines.py --data ../data --out ../data/stage_g2_baselines_v2 \\
        --runs ../runs/f_seed1 ../runs/f_seed2 --compare ../data/stage_g2_baselines --threads 4 \\
        > ../logs/stage_g2_baselines_v2.txt 2>&1 &
Needs torch, numpy, and stage_g_samples.py, stage_f_check.py, stage_f_train.py, stage_d_headroom.py, obj2_model.py,
stage_e_inputs.py, stage_d_sampler.py, export_obj2.py in the same folder.
"""
import argparse, csv, importlib.util, json, os, subprocess, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as TF

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g_baselines.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


GS = _load("stage_g_samples")
CK = GS.CK
HR, E, D, X = CK.HR, CK.E, CK.D, CK.X
G, W, NB = X.G, X.WINDOW, GS.NB
SCORE, TRUE, TCLS, BINS = HR.SCORE, D.TRUE, D.TCLS, D.BINS
N_BIN, N_MIN, K_MAPS, CH = 4000, 100, 50, 256
P_FLAG = (0.37, 0.58)
B4_TOL = 1e-9                                           # B4 is a + b d: same order as -d when b < 0
LOG_DP = 9                                              # log B1 rounded to 1e-9: summation noise (last bits) must not split
                                                        # cells that tie in exact arithmetic (sandbox: -104 against -104 + 1 ulp)
TIE_FLAG = 0.005                                        # exact B1 above stored float32 B1 by more than this in a class
INIT = dict(a0=-3.0, a1=0.0, a2=0.0, b0=-2.0, b1=0.0, c0=-2.0, c1=0.0)
NAMES_H, NAMES_N = ["a0", "a1", "a2", "b0", "b1", "c0", "c1"], ["a0", "a1", "b0", "c0"]
BROKEN = os.environ.get("STAGE_G2_BROKEN", "")         # self-test only: plants one bug so its gate must fail
F64 = torch.float64


# ---------------------------------------------------------------- inputs
def load_obs(d, files, fi, rw):
    """Visit 1, visit 2, effort (n, 6, G, G) bool and noisy habitat (n, G, G) of the listed stacks, in order;
    each file's md5 checked against the manifest."""
    man = {r["input_file"]: r for r in D.rows_of(d)}
    n = len(fi)
    V1 = np.zeros((n, W, G, G), bool); V2 = np.zeros_like(V1); Ef = np.zeros_like(V1); H = np.zeros((n, G, G))
    comm = np.zeros(n, np.int64)
    for i in np.unique(fi):
        k = np.nonzero(fi == i)[0]; r = rw[k]
        row = man[str(files[i])]
        p = Path(d) / row["input_file"]
        if X.file_md5(p) != row["input_md5"]:
            sys.exit(f"STOP: {p} md5 differs from the manifest")
        with np.load(p, allow_pickle=False) as z:
            V1[k] = X.unpack(z["visit1"][r]); V2[k] = X.unpack(z["visit2"][r]); Ef[k] = X.unpack(z["effort"][r])
            H[k] = z["habitat_noisy"][z["stack_hab"][r]].astype(np.float64)
        comm[k] = int(row["sim_seed"])
    return V1, V2, Ef, H, comm


def hab_dist(V, H):
    """|z - mean z at the cells recorded in any year|, z = habitat standardised per stack (stage E channel 20)."""
    n = len(V)
    R6 = V.any(1)
    Hf = H.astype(np.float32).astype(np.float64)
    mu = Hf.reshape(n, -1).mean(1); sd = Hf.reshape(n, -1).std(1); sd[sd == 0] = 1.0
    Z = (Hf - mu[:, None, None]) / sd[:, None, None]
    zm = (Z * R6).reshape(n, -1).sum(1) / np.maximum(R6.reshape(n, -1).sum(1), 1)
    return np.abs(Z - zm[:, None, None])


def p_visit(V1, V2, Ef):
    """Per-visit detection from the two visits: r = both / at least one, p = 2r / (1 + r)."""
    one, both = ((V1 | V2) & Ef).sum(), (V1 & V2 & Ef).sum()
    r = both / max(one, 1)
    return float(2 * r / (1 + r)), int(one), int(both)


# ---------------------------------------------------------------- B3: mean-field dynamic occupancy
_CNT = {}


def nbr_mean(P):
    """Mean of the in-window 8 neighbours, no wrap. P (n, G, G) tensor."""
    k = torch.ones(1, 1, 3, 3, dtype=P.dtype); k[0, 0, 1, 1] = 0
    if P.dtype not in _CNT:
        _CNT[P.dtype] = TF.conv2d(torch.ones(1, 1, G, G, dtype=P.dtype), k, padding=1)[0, 0]
    if BROKEN == "wrap":
        return TF.conv2d(TF.pad(P[:, None], (1, 1, 1, 1), mode="circular"), k)[:, 0] / 8.0
    return TF.conv2d(P[:, None], k, padding=1)[:, 0] / _CNT[P.dtype]


def b3_filter(th, Y, S, DH, p):
    """Forward filter. th = (a0, a1, a2, b0, b1, c0, c1) tensor; Y (n, W, G, G) detections 0-2; S surveyed (bool);
    DH (n, G, G). Returns filtered chances (n, W, G, G) and the log-likelihood of the records."""
    a0, a1, a2, b0, b1, c0, c1 = th
    L1 = torch.where(S, p ** Y * (1 - p) ** (2 - Y), torch.ones_like(Y))
    L0 = torch.where(S & (Y > 0), torch.zeros_like(Y), torch.ones_like(Y))
    e = torch.sigmoid(b0 + b1 * DH)
    pi = torch.sigmoid(c0 + c1 * DH)
    ll, out, psi = 0.0, [], None
    for t in range(W):
        if t:
            g = torch.sigmoid(a0 + a1 * nbr_mean(psi) + a2 * DH)
            pi = psi * (1 - e) + (1 - psi) * g
        pi = pi.clamp(1e-12, 1 - 1e-12)
        num = pi * L1[:, t]; den = num + (1 - pi) * L0[:, t]
        ll = ll + torch.log(den).sum()
        psi = num / den
        out.append(psi)
    return torch.stack(out, 1), ll


def expand(x, hab):
    if hab:
        return x
    z = torch.zeros(1, dtype=x.dtype)
    return torch.cat([x[0:2], z, x[2:3], z, x[3:4], z])          # a0 a1 0 b0 0 c0 0


def fit_b3(Y, S, DH, p, hab, iters=300):
    names = NAMES_H if hab else NAMES_N
    x = torch.tensor([INIT[k] for k in names], dtype=F64, requires_grad=True)
    Yt, St, Dt = torch.from_numpy(Y.astype(np.float64)), torch.from_numpy(S), torch.from_numpy(DH if hab else np.zeros_like(DH))
    ncy = float(Y.size)
    opt = torch.optim.LBFGS([x], lr=1, max_iter=iters, tolerance_grad=1e-10, tolerance_change=1e-14, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        _, ll = b3_filter(expand(x, hab), Yt, St, Dt, p)
        loss = -ll / ncy
        loss.backward()
        return loss
    opt.step(closure)
    loss = closure()
    return dict(theta=[float(v) for v in expand(x.detach(), hab)], names=names, loglik=float(-loss.detach()) * ncy,
                grad=float(x.grad.abs().max()), cell_years=int(ncy), p=p)


@torch.no_grad()
def b3_predict(th, Y, S, DH, p, chunk=2000):
    """Filtered chances of all years (float32, for the maps) and of the target year (float64, for the AUC)."""
    out = np.zeros(Y.shape, np.float32); last = np.zeros((len(Y), G, G))
    th = torch.tensor(th, dtype=F64)
    for a in range(0, len(Y), chunk):
        sl = slice(a, a + chunk)
        ps, _ = b3_filter(th, torch.from_numpy(Y[sl].astype(np.float64)), torch.from_numpy(S[sl]), torch.from_numpy(DH[sl]), p)
        out[sl] = ps.numpy(); last[sl] = ps[:, -1].numpy()
    return out, last


# ---------------------------------------------------------------- B4 and B5
def b1_log(V, sig):
    """Log of the B1 surface (6 pooled years, pinned cells + PIN) by log-sum-exp per stack, so no cell ties by underflow;
    rounded to 1e-9 so that summation noise does not split exact ties. Stage D stored B1 as float32: exp(-d^2 / 2 sigma^2) is 0 in float32 beyond about 3.6 cells at sigma 0.25
    (float64: about 9.4), so every cell that far from all records tied at 0 in the stage D and F5 scores."""
    n = len(V)
    R6 = V.any(1)
    yy, xx = np.mgrid[0:G, 0:G]
    cy, cx = yy.ravel(), xx.ravel()
    out = np.full((n, G * G), -np.inf)
    for i in range(n):
        r = np.flatnonzero(R6[i])
        if len(r) == 0:
            continue
        t = -((cy[:, None] - cy[r][None]) ** 2 + (cx[:, None] - cx[r][None]) ** 2) / (2.0 * sig * sig)
        m = t.max(1)
        out[i] = m + np.log(np.exp(t - m[:, None]).sum(1))
    if BROKEN == "b1_log_float32":
        out = np.log(np.maximum(HR.smooth(R6, sig).astype(np.float32), 1e-45)).reshape(n, -1).astype(np.float64)
    out = out.reshape(n, G, G)
    out = np.where(R6, np.logaddexp(out, np.log(HR.PIN)), out)
    return np.round(out, LOG_DP) if BROKEN != "no_round" else out


def near_dist(V):
    """Distance in cells from each cell to the nearest cell recorded in any of the 6 years (0 at a recorded cell)."""
    n = len(V)
    R6 = V.any(1)
    yy, xx = np.mgrid[0:G, 0:G]
    cy, cx = yy.ravel(), xx.ravel()
    out = np.full((n, G * G), np.inf)
    for i in range(n):
        r = np.flatnonzero(R6[i])
        if len(r):
            out[i] = np.sqrt(((cy[:, None] - cy[r][None]) ** 2 + (cx[:, None] - cx[r][None]) ** 2).min(1))
    return out.reshape(n, G, G)


def b4_x(V, sig=None):
    return near_dist(V)


def b4_chance(fit, V, sig=None, x=None):
    """(n, W, G, G): chance of each year; cells recorded in that year are 1."""
    z = fit["a"] + fit["b"] * (b4_x(V) if x is None else x)
    c = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
    P = np.repeat(c[:, None], W, 1)
    if BROKEN != "maps_ignore_records":
        P[V] = 1.0
    return P.astype(np.float32), z


def b5_maps(V):
    return np.maximum.accumulate(V, axis=1)                   # year t: cells recorded in years 1..t


def b5_score(V):
    last = np.zeros(V.shape[:1] + V.shape[2:])
    for t in range(W):
        last[V[:, t]] = (t + 1) / W
    return last


# ---------------------------------------------------------------- maps
def draw_maps(P, K, key):
    """P (n, W, G, G) chances -> packed maps (n, K, 300), and the target-year AOO of each map (n, K)."""
    n = len(P)
    maps = np.zeros((n, K, NB), np.uint8); aoo = np.zeros((n, K), np.uint16)
    for j, a in enumerate(range(0, n, CH)):
        U = X.rng(88, key, j).random((min(CH, n - a), K, W, G, G), dtype=np.float32)
        M_ = U < P[a:a + CH, None]
        maps[a:a + CH] = GS.pack(M_)
        aoo[a:a + CH] = M_[:, :, W - 1].reshape(len(M_), K, -1).sum(2)
    return maps, aoo


def check_maps(maps, P, V, aoo):
    """P5 parts: recorded cells on in every map; packed maps unpack to their AOO; mean AOO against sum of chances."""
    on, rt, K = True, True, maps.shape[1]
    for a in range(0, len(maps), 1000):
        u = GS.unpack(maps[a:a + 1000])
        on &= bool(u[V[a:a + 1000][:, None].repeat(K, 1)].all())
        rt &= np.array_equal(u[:, :, W - 1].reshape(len(u), K, -1).sum(2), aoo[a:a + 1000])
    pt = P[:, W - 1].reshape(len(P), -1).astype(np.float64)
    exp_ = pt.sum(1); var = (pt * (1 - pt)).sum(1) / K
    z = float((aoo.mean(1) - exp_).sum() / np.sqrt(max(var.sum(), 1e-12)))
    return on, rt, z


# ---------------------------------------------------------------- run
def bin_picks(sampler):
    """Up to N_BIN real_train stacks per records-over-6-years bin, probability proportional to the stage D weight."""
    w, bn = sampler["weight"].astype(np.float64), sampler["bin"].astype(int)
    picks = {}
    for b in range(len(BINS)):
        idx = np.nonzero((bn == b) & (w > 0))[0]
        if len(idx) == 0:
            picks[b] = idx
            continue
        g = X.rng(99, 60, b)
        picks[b] = np.sort(g.choice(idx, min(N_BIN, len(idx)), replace=False, p=w[idx] / w[idx].sum()))
    return picks


def per_stack_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def aoo_rows(exp_, true, lo, hi, groups, labels):
    rows = []
    for j, lab in enumerate(labels):
        m = groups == j
        if m.any():
            rows.append(dict(group=lab, n=int(m.sum()), ratio=float(exp_[m].mean() / true[m].mean()) if true[m].mean() > 0
                             else float("nan"), cov90=float(((true[m] >= lo[m]) & (true[m] <= hi[m])).mean())))
    return rows


def run(data, train_dir, val_dir, out_dir, runs=(), K=K_MAPS, out=print, compare=None):
    data, train_dir, val_dir, out_dir = Path(data), Path(train_dir), Path(val_dir), Path(out_dir)
    for p in (data, train_dir, val_dir, out_dir, *map(str, runs)):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G5")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    res, t0 = {}, time.time()

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    sp, fp, hp = data / "stage_d_train_sampler.npz", data / "stage_d_frozen_val.npz", data / "stage_d_headroom_pred.npz"
    sampler, frozen, hz = dict(np.load(sp)), dict(np.load(fp)), dict(np.load(hp))
    sig = float(hz["sigma"])
    cmd5 = X.file_md5(__file__)
    out(f"stage_g_baselines.py md5 {cmd5}; stage_g_samples.py {X.file_md5(HERE / 'stage_g_samples.py')}; "
        f"stage_d_headroom.py {X.file_md5(HERE / 'stage_d_headroom.py')}; export_obj2.py {X.file_md5(HERE / 'export_obj2.py')}")
    out(f"   sampler md5 {X.file_md5(sp)}; frozen set md5 {X.file_md5(fp)}; headroom predictions md5 {X.file_md5(hp)}; "
        f"B1 sigma {sig:g}; K {K}; torch threads {torch.get_num_threads()}")
    same = all(X.file_md5(train_dir / f) == m for f, m in zip(sampler["files"], sampler["file_md5"])) and \
        all(X.file_md5(val_dir / f) == m for f, m in zip(frozen["files"], frozen["file_md5"]))
    hz_ok = str(hz["frozen_md5"]) == X.file_md5(fp)

    # ---- fits, real_train only
    with HR.OpenLog() as log:
        picks = bin_picks(sampler)
        fdir, ffiles, ffi, frw = train_dir, sampler["files"], sampler["file_index"], sampler["row"]
        if BROKEN == "fit_on_val":
            fdir, ffiles = val_dir, frozen["files"]
            picks = {b: np.nonzero(frozen["rec6"] >= 0)[0][:N_BIN] for b in range(len(BINS))}
            ffi, frw = frozen["file_index"], frozen["row"]
        allp = np.unique(np.concatenate([v for v in picks.values()]))
        V1, V2, Ef, H, _ = load_obs(fdir, ffiles, ffi[allp], frw[allp])
        V = V1 | V2
        bn = D.bidx(V.any(1).reshape(len(V), -1).sum(1))
        DH = hab_dist(V, H)
        Y = V1.astype(np.int8) + V2.astype(np.int8)
        outside = int((V & ~Ef).sum())
        fits = {}
        pool = np.arange(len(allp))
        pp, one, both = p_visit(V1, V2, Ef)
        out(f"   B3 fit: {len(allp)} real_train stacks (up to {N_BIN} per bin, stage D weights); records outside surveyed "
            f"cells {outside} (must be 0); pooled p per visit {pp:.4f} ({both} of {one} recorded cell-years seen on both visits)")
        fits["pooled"] = dict(hab=fit_b3(Y[pool], Ef[pool], DH[pool], pp, True), nohab=fit_b3(Y[pool], Ef[pool], DH[pool], pp, False),
                              n=int(len(pool)), p=pp, fallback=False)
        for b, lab in enumerate(BINS):
            k = np.nonzero(bn == b)[0]
            if len(k) < N_MIN:
                fits[lab] = dict(fits["pooled"], n=int(len(k)), fallback=True)
                out(f"      {lab:>6}: {len(k)} stacks (< {N_MIN}): pooled parameters used")
                continue
            pb, _, _ = p_visit(V1[k], V2[k], Ef[k])
            fits[lab] = dict(hab=fit_b3(Y[k], Ef[k], DH[k], pb, True), nohab=fit_b3(Y[k], Ef[k], DH[k], pb, False),
                             n=int(len(k)), p=pb, fallback=False)
        for lab in ["pooled"] + BINS:
            f = fits[lab]
            out(f"      {lab:>6}: n {f['n']:>5}, p {f['p']:.3f}; habitat " + ", ".join(f"{nm} {v:+.3f}" for nm, v in zip(NAMES_H, f["hab"]["theta"]))
                + f" (max |grad| {f['hab']['grad']:.1e}); no habitat " + ", ".join(f"{nm} {f['nohab']['theta'][NAMES_H.index(nm)]:+.3f}" for nm in NAMES_N)
                + f" (max |grad| {f['nohab']['grad']:.1e})")
        del V1, V2, Ef, H, Y, DH, V
        bfiles, bfi, brw = HR.fit_sample(train_dir, sampler) if BROKEN != "fit_on_val" else (frozen["files"], frozen["file_index"], frozen["row"])
        Vb, _, _, Tb, _, _ = HR.load_stacks(fdir, bfiles, bfi, brw, truth=True)
    fit_paths = log.paths
    m = ~Vb[:, -1]
    xb = b4_x(Vb, sig)[m]
    lf = HR.fit_logit(xb[:, None], Tb[m].astype(float), np.ones(int(m.sum())))
    b4 = dict(a=float(lf["b"][0] - lf["b"][1] * lf["mu"][0] / lf["sd"][0]), b=float(lf["b"][1] / lf["sd"][0]))
    if BROKEN == "b4_flip":
        b4["b"] = -b4["b"]
    out(f"   B4 fit on {len(Vb)} real_train stacks ({int(m.sum())} unrecorded target-year cells): logit = {b4['a']:+.4f} "
        f"{b4['b']:+.4f} d (d = distance to the nearest record, cells); chance at d 1, 2, 5, 10: "
        + ", ".join(f"{1 / (1 + np.exp(-(b4['a'] + b4['b'] * q))):.3f}" for q in (1, 2, 5, 10)) + f"; mean chance {float((1 / (1 + np.exp(-(b4['a'] + b4['b'] * xb)))).mean()):.4f} against "
        f"true share {float(Tb[m].mean()):.4f}")
    del Vb, Tb, xb, m
    inside = lambda p, d: str(p).startswith(str(Path(d).resolve()) + os.sep)
    fit_ok = len(fit_paths) > 0 and all(inside(p, train_dir) for p in fit_paths)
    gate("P2", fit_ok, f"the fits opened {len(fit_paths)} files, all in real_train {fit_ok}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pflag = [f"{lab} {fits[lab]['p']:.3f}" for lab in BINS if not P_FLAG[0] <= fits[lab]["p"] <= P_FLAG[1]]
    (out_dir / "g2_fit.json").write_text(json.dumps(dict(B3=fits, B4=b4, sigma=sig, outside=outside, code_md5=cmd5,
                                                          p_flag=pflag), indent=1))
    out(f"   written g2_fit.json md5 {X.file_md5(out_dir / 'g2_fit.json')}")

    # ---- predictions and maps on the frozen real_val set, inputs only
    vf, vfi, vrw = frozen["files"], frozen["file_index"], frozen["row"]
    blank = frozen["blank"].astype(bool)
    with D.TruthGuard() as guard:
        try:
            V1, V2, Ef, H, comm = load_obs(val_dir, vf, vfi, vrw)
            if BROKEN == "misalign":
                V1, V2, Ef, H = (np.roll(a, 1, 0) for a in (V1, V2, Ef, H))
            if BROKEN == "truth_in_pred":
                HR.load_stacks(val_dir, vf, vfi[:2], vrw[:2], truth=True)
            V = V1 | V2
            n = len(V)
            b1 = HR.b1_score(V, sig).astype(np.float32)
            b1_same = np.array_equal(b1, hz["B1"])
            bn = D.bidx(V.any(1).reshape(n, -1).sum(1))
            DH = hab_dist(V, H)
            Y = V1.astype(np.int8) + V2.astype(np.int8)
            P3h = np.zeros((n, W, G, G), np.float32); P3n = np.zeros_like(P3h)
            S3h = np.zeros((n, G, G)); S3n = np.zeros_like(S3h)
            for b, lab in enumerate(BINS):
                k = np.nonzero(bn == b)[0]
                if len(k):
                    f = fits[lab]
                    P3h[k], S3h[k] = b3_predict(f["hab"]["theta"], Y[k], Ef[k], DH[k], f["p"])
                    P3n[k], S3n[k] = b3_predict(f["nohab"]["theta"], Y[k], Ef[k], np.zeros_like(DH[k]), f["p"])
            b1x = b1_log(V, sig)
            dist = near_dist(V)
            if BROKEN == "maps_ignore_records":
                P3h[V] = 0.5; P3n[V] = 0.5
            P3b = np.where(blank[:, None, None, None], P3n, P3h)
            P4, z4 = b4_chance(b4, V, x=dist)
            m3, a3 = draw_maps(P3b, K, 1)
            m4, a4 = draw_maps(P4, K, 2)
            M5 = b5_maps(V)
            m5 = GS.pack(M5[:, None]); a5 = M5[:, W - 1].reshape(n, -1).sum(1).astype(np.uint16)[:, None]
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    if crashed:
        gate("P3", False, f"truth file opened while the predictions were built ({crashed[:120]})")
        return res, None
    pred = dict(B3=S3h, B3_nohab=S3n, B4_logit=z4, B4_dist=dist.astype(np.float32), B1_exact_log=b1x,
                B5=b5_score(V).astype(np.float32))
    X.save_npz(out_dir / "g2_pred.npz", dict(pred, blank=blank.astype(np.uint8), sigma=np.float64(sig), code_md5=np.array(cmd5),
                                             frozen_md5=np.array(X.file_md5(fp))))
    common = dict(sel=np.arange(n), layout=np.array(GS.LAYOUT), frozen_md5=np.array(X.file_md5(fp)), code_md5=np.array(cmd5))
    X.save_npz(out_dir / "maps_B3_halfblank.npz", dict(common, maps=m3, aoo_t=a3, K=np.int64(K), seed_key=np.int64(1)))
    X.save_npz(out_dir / "maps_B4.npz", dict(common, maps=m4, aoo_t=a4, K=np.int64(K), seed_key=np.int64(2)))
    X.save_npz(out_dir / "maps_B5.npz", dict(common, maps=m5, aoo_t=a5, K=np.int64(1), seed_key=np.int64(0)))
    md5s = {f: X.file_md5(out_dir / f) for f in ("g2_fit.json", "g2_pred.npz", "maps_B3_halfblank.npz", "maps_B4.npz", "maps_B5.npz")}
    out("   written before any real_val truth file was opened: " + ", ".join(f"{f} md5 {v}" for f, v in md5s.items()))
    if compare:
        cp = Path(compare)
        same_maps = {k: np.array_equal(np.load(cp / f)["maps"], mm) for k, f, mm in
                     (("B3 half-blank", "maps_B3_halfblank.npz", m3), ("B5", "maps_B5.npz", m5))}
        old_fit = json.loads((cp / "g2_fit.json").read_text())["B3"]
        same_fit = all(old_fit[lab][h]["theta"] == fits[lab][h]["theta"] for lab in ["pooled"] + BINS for h in ("hab", "nohab"))
        rep_same = dict(same_maps, B3_fit=same_fit)
        out(f"   against the earlier run in {cp.name} (B1, B3, B5 code unchanged; file md5s differ because each file stores "
            f"the tool md5): " + ", ".join(f"{k} maps equal {v}" for k, v in same_maps.items()) + f", B3 parameters equal {same_fit}"
            + ("" if all(rep_same.values()) else " (FLAG)"))
    gate("P3", not guard.opened, f"no truth file opened while the predictions and maps were built (opened {len(guard.opened)})")
    gate("P1", same and hz_ok and b1_same,
         f"input files md5 unchanged since stage D {same}; headroom predictions from this frozen set {hz_ok}; B1 recomputed "
         f"here equals the stored B1 in every stack {b1_same}")
    on3, rt3, z3 = check_maps(m3, P3b, V, a3)
    on4, rt4, z4m = check_maps(m4, P4, V, a4)
    on5 = bool(GS.unpack(m5)[:, 0][V].all()) and np.array_equal(GS.unpack(m5)[:, 0], M5)
    gate("P5", on3 and on4 and on5 and rt3 and rt4 and abs(z3) <= 4 and abs(z4m) <= 4,
         f"recorded cells on in every map (B3 {on3}, B4 {on4}, B5 {on5}); unpacked maps give the stored AOO (B3 {rt3}, "
         f"B4 {rt4}); mean sample AOO against the sum of chances z B3 {z3:+.2f}, B4 {z4m:+.2f} (rule |z| <= 4)")

    # ---- truth, after everything is written
    _, _, _, T, tc, _ = HR.load_stacks(val_dir, vf, vfi, vrw, truth=True)
    Mk = ~V[:, -1] if BROKEN != "score_recorded" else np.ones_like(V[:, -1])
    S = dict(B1=hz["B1"], B1_exact=pred["B1_exact_log"], B2=hz["B2"], B2_halfblank=np.where(blank[:, None, None], hz["B2e"], hz["B2"]),
             B3=pred["B3"], B3_nohab=pred["B3_nohab"], B3_halfblank=np.where(blank[:, None, None], pred["B3_nohab"], pred["B3"]),
             B4=pred["B4_logit"], B5=pred["B5"])
    S["random"] = X.rng(99, 61).random(T.shape).astype(np.float32) if BROKEN != "random_is_b1" else hz["B1"].copy()
    A = {k: HR.aucs(v, T, Mk) for k, v in S.items()}
    bal = {k: HR.balanced(A[k], tc) for k in A}
    rec_scored = int((Mk & V[:, -1]).sum())
    tol = max(0.01, 4 * bal["random"][1])
    A_negd = HR.aucs(-pred["B4_dist"].astype(np.float64), T, Mk)
    d4 = np.abs(A["B4"] - A_negd)
    same4 = bool(b4["b"] < 0 and np.array_equal(np.isnan(A["B4"]), np.isnan(A_negd)) and np.nanmax(d4) <= B4_TOL)
    gate("P4", rec_scored == 0 and abs(bal["random"][0] - 0.5) <= tol and same4,
         f"recorded cells among the scored cells {rec_scored}; random score {bal['random'][0]:.3f} (0.50 within {tol:.3f}); "
         f"B4 slope b {b4['b']:+.4f} (< 0) and B4 AUC = AUC of -d in every stack (largest gap {np.nanmax(d4):.1e}, rule {B4_TOL})")
    true_aoo = T.reshape(n, -1).sum(1)

    # ---- model runs (step 3a per-stack files)
    rec_t = V[:, -1].reshape(n, -1).sum(1); rec6 = V.any(1).reshape(n, -1).sum(1)
    rcls, rbin = D.tidx(rec_t), D.bidx(rec6)
    mod = {}
    if runs:
        ok6, why = True, []
        for r in runs:
            rows = per_stack_rows(Path(r) / "check_3a" / "per_stack.csv")
            if BROKEN == "p6_misorder":
                rows = rows[::-1]
            name = Path(r).name
            if len(rows) != n:
                ok6 = False; why.append(f"{name} {len(rows)} rows"); continue
            key = np.array([[int(q["community"]), int(q["file_index"]), int(q["row"])] for q in rows])
            ok_key = np.array_equal(key, np.stack([comm, vfi, vrw], 1))
            ok_b1 = all(q["auc_b1"] == f"{A['B1'][i]:.5f}" for i, q in enumerate(rows))   # same text as written by step 3a
            ok_t = np.array_equal(np.array([int(q["aoo_true"]) for q in rows]), true_aoo)
            ok6 &= ok_key and ok_b1 and ok_t
            why.append(f"{name}: stacks {ok_key}, B1 AUC {ok_b1}, true AOO {ok_t}")
            mod[name] = dict(auc=np.array([float(q["auc_model"]) for q in rows]),
                             exp=np.array([float(q["aoo_expected"]) for q in rows]),
                             lo=np.array([float(q["aoo_p5"]) for q in rows]), hi=np.array([float(q["aoo_p95"]) for q in rows]))
        gate("P6", ok6, "step 3a per-stack files are for these stacks: " + "; ".join(why))

    # ---- AUC tables
    warnings.simplefilter("ignore", RuntimeWarning)           # classes without stacks print nan
    cols = [*(f"model {k}" for k in mod), "B1", "B1_exact", "B2_halfblank", "B3_halfblank", "B3", "B3_nohab", "B4", "B5"]
    getA = lambda c: mod[c[6:]]["auc"] if c.startswith("model ") else A[c]
    out("   AUC by true class at the target year (stacks scored | " + " | ".join(cols) + "):")
    for c in SCORE:
        k = (tc == c) & ~np.isnan(A["B1"])
        out(f"      {TRUE[c]:>6} {int(k.sum()):>6}  " + "  ".join(f"{np.nanmean(getA(q)[k]):.4f}" for q in cols))
    out("      balanced       " + "  ".join(f"{HR.balanced(getA(q), tc)[0]:.4f}" for q in cols))
    pairs = [("B1_exact", "B1"), ("B3_halfblank", "B1_exact"), ("B3_halfblank", "B2_halfblank"), ("B3", "B3_nohab"),
             ("B2", "B1_exact")] + [(f"model {k}", "B3_halfblank") for k in mod] + [(f"model {k}", "B1_exact") for k in mod]
    boots = {}
    out("   paired differences per true class, 95% interval by resampling communities:")
    for a_, b_ in pairs:
        cb = CK.class_boot(getA(a_) - getA(b_), tc, comm)
        boots[f"{a_} - {b_}"] = cb
        out(f"      {a_} - {b_}: " + "; ".join(f"{TRUE[c]} {cb[c]['mean']:+.4f} [{cb[c]['lo']:+.4f}, {cb[c]['hi']:+.4f}]" for c in SCORE)
            + f"; balanced {cb['balanced']['mean']:+.4f} [{cb['balanced']['lo']:+.4f}, {cb['balanced']['hi']:+.4f}]")
    gap = {TRUE[c]: boots["B1_exact - B1"][c]["mean"] for c in SCORE}
    tie_flag = [k for k, v in gap.items() if v > TIE_FLAG]
    out(f"   B1 ties: exact B1 - stored float32 B1 by class " + ", ".join(f"{k} {v:+.4f}" for k, v in gap.items())
        + (f": FLAG above {TIE_FLAG} in {', '.join(tie_flag)}" if tie_flag else f" (none above {TIE_FLAG})"))
    d9 = HR.balanced(A["B2"], tc)[0] - HR.balanced(A["B1_exact"], tc)[0]
    out(f"   stage D headroom recheck: B2 - B1 exact {d9:+.4f} (stage D rule 0.02 or more; with stored B1 "
        f"{HR.balanced(A['B2'], tc)[0] - HR.balanced(A['B1'], tc)[0]:+.4f})")
    f5r = {}
    for k in mod:
        cb = boots[f"model {k} - B1_exact"]
        v = {TRUE[c]: CK.verdict(cb[c]["mean"], cb[c]["hi"]) for c in SCORE if cb[c]["n"]}
        f5r[k] = "FAIL" if "FAIL" in v.values() else ("PASS" if all(x == "PASS" for x in v.values()) else "NOT RESOLVED")
        out(f"   F5 recheck {k}: model - B1 exact, same rule (PASS mean >= 0, FAIL upper < 0) by class "
            + ", ".join(f"{c_} {x}" for c_, x in v.items()) + f": {f5r[k]}")

    # ---- raw AOO (before the G3 correction)
    q5 = lambda a: np.percentile(a.astype(np.float64), 5, axis=1)
    q95 = lambda a: np.percentile(a.astype(np.float64), 95, axis=1)
    aoo_sets = {"B3_halfblank": (a3.mean(1), q5(a3), q95(a3)), "B4": (a4.mean(1), q5(a4), q95(a4)),
                "B5": (a5[:, 0].astype(float), a5[:, 0].astype(float), a5[:, 0].astype(float))}
    for k, v in mod.items():
        aoo_sets[f"model {k}"] = (v["exp"], v["lo"], v["hi"])
    tabs = {}
    out("   raw target-year AOO (no correction yet): ratio = mean expected / mean true; cov = share of true AOO inside the "
        "5th-95th percentile of the maps (B5 is one map: share exactly right)")
    for name, (ex, lo, hi) in aoo_sets.items():
        tabs[name] = {"recorded class": aoo_rows(ex, true_aoo, lo, hi, rcls, TCLS),
                      "records over 6 years": aoo_rows(ex, true_aoo, lo, hi, rbin, BINS)}
        for g_, rows in tabs[name].items():
            out(f"      {name:<14} by {g_}: " + "; ".join(f"{q['group']} {q['ratio']:.2f} cov {q['cov90']:.2f}" for q in rows))

    # ---- files
    with open(out_dir / "per_stack_g2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stack", "community", "file_index", "row", "true_class", "recorded_class", "records_6y_bin", "blank",
                    "auc_b1", "auc_b1_exact", "auc_b3", "auc_b3_nohab", "auc_b3_halfblank", "auc_b4", "auc_b5", "aoo_true", "aoo_b3hb_mean",
                    "aoo_b3hb_p5", "aoo_b3hb_p95", "aoo_b4_mean", "aoo_b4_p5", "aoo_b4_p95", "aoo_b5"])
        e3, l3, h3 = aoo_sets["B3_halfblank"]; e4, l4, h4 = aoo_sets["B4"]
        for i in range(n):
            w.writerow([i, int(comm[i]), int(vfi[i]), int(vrw[i]), TRUE[tc[i]], TCLS[rcls[i]], BINS[rbin[i]], int(blank[i])]
                       + [f"{A[q][i]:.5f}" for q in ("B1", "B1_exact", "B3", "B3_nohab", "B3_halfblank", "B4", "B5")]
                       + [int(true_aoo[i]), f"{e3[i]:.3f}", f"{l3[i]:.1f}", f"{h3[i]:.1f}", f"{e4[i]:.3f}", f"{l4[i]:.1f}",
                          f"{h4[i]:.1f}", int(a5[i, 0])])
    rep = dict(gates=res, md5=dict(md5s, tool=cmd5), auc_balanced={k: v[0] for k, v in bal.items()},
               auc_balanced_models={k: HR.balanced(v["auc"], tc)[0] for k, v in mod.items()}, paired=boots, aoo_raw=tabs,
               p_flag=pflag, b1_tie_gap=gap, b1_tie_flag=tie_flag, headroom_recheck=d9, F5_recheck=f5r, K=K, n=int(n),
               minutes=round((time.time() - t0) / 60, 1))
    (out_dir / "g2_report.json").write_text(json.dumps(rep, indent=1))
    n_ok = sum(res.values())
    out(f"   written per_stack_g2.csv and g2_report.json ({(time.time() - t0) / 60:.1f} min)")
    flags = ([f"p outside {P_FLAG}: " + ", ".join(pflag)] if pflag else []) + \
        ([f"exact B1 above stored B1 by more than {TIE_FLAG} in {', '.join(tie_flag)}"] if tie_flag else [])
    out(f"stage G2: gates {n_ok} PASS, {len(res) - n_ok} FAIL; F5 recheck " + (", ".join(f"{k} {v}" for k, v in f5r.items()) or "not run")
        + "; flags: " + ("; ".join(flags) or "none"))
    return res, dict(bal=bal, A=A, fits=fits, b4=b4, tc=tc, gap=gap, f5r=f5r)


# ---------------------------------------------------------------- self-test
def simulate(n, th, p, a1_true_nbr, seed, s_rate=0.4):
    """Data from the B3 process: true occupancy with colonisation from the true neighbours' occupancy, two visits."""
    g = np.random.default_rng(seed)
    a0, a1, a2, b0, b1, c0, c1 = th
    DH = np.abs(g.standard_normal((n, G, G)))
    sig = lambda z: 1 / (1 + np.exp(-z))
    Z = np.zeros((n, W, G, G), bool)
    Z[:, 0] = g.random((n, G, G)) < sig(c0 + c1 * DH)
    for t in range(1, W):
        nb = nbr_mean(torch.from_numpy(Z[:, t - 1].astype(np.float64))).numpy() if a1_true_nbr else 0.0
        gam, eps = sig(a0 + a1 * nb + a2 * DH), sig(b0 + b1 * DH)
        u = g.random((n, G, G))
        Z[:, t] = np.where(Z[:, t - 1], u >= eps, u < gam)
    S = g.random((n, W, G, G)) < s_rate
    V1 = Z & S & (g.random(Z.shape) < p); V2 = Z & S & (g.random(Z.shape) < p)
    return Z, S, V1, V2, DH


def slow_filter(th, Y, S, DH, p):
    """Cell-by-cell loop, for U3."""
    a0, a1, a2, b0, b1, c0, c1 = th
    sg = lambda z: 1 / (1 + np.exp(-z))
    n = len(Y)
    out = np.zeros(Y.shape)
    for i in range(n):
        psi = None
        for t in range(W):
            new = np.zeros((G, G))
            for r in range(G):
                for c in range(G):
                    if t == 0:
                        pi = sg(c0 + c1 * DH[i, r, c])
                    else:
                        nb = [psi[rr, cc] for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                              if (rr, cc) != (r, c) and 0 <= rr < G and 0 <= cc < G]
                        gm = sg(a0 + a1 * np.mean(nb) + a2 * DH[i, r, c]); ep = sg(b0 + b1 * DH[i, r, c])
                        pi = psi[r, c] * (1 - ep) + (1 - psi[r, c]) * gm
                    y = Y[i, t, r, c]
                    l1 = p ** y * (1 - p) ** (2 - y) if S[i, t, r, c] else 1.0
                    l0 = (0.0 if y > 0 else 1.0) if S[i, t, r, c] else 1.0
                    new[r, c] = pi * l1 / (pi * l1 + (1 - pi) * l0)
            psi = new
            out[i, t] = psi
    return out


def unit_checks():
    g = np.random.default_rng(7)
    res = {}
    V = g.random((4, W, G, G)) < 0.05; V[:, -1, 3, 3] = True
    Ef = V | (g.random(V.shape) < 0.3); H = g.standard_normal((4, G, G)).astype(np.float16)
    x, _ = E.build(V, Ef, H, np.zeros((4, 2), int), np.zeros(4, bool))
    res["U1"] = bool(np.abs(hab_dist(V, H).astype(np.float32) - x[:, 20, :G, :G]).max() < 1e-5)
    P = g.random((3, G, G))
    nm = nbr_mean(torch.from_numpy(P)).numpy()
    ref = np.zeros_like(P)
    for r in range(G):
        for c in range(G):
            ref[:, r, c] = np.mean([P[:, rr, cc] for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                                    if (rr, cc) != (r, c) and 0 <= rr < G and 0 <= cc < G], 0)
    res["U2"] = bool(np.abs(nm - ref).max() < 1e-12 and abs(nm[0, 0, 0] - (P[0, 0, 1] + P[0, 1, 0] + P[0, 1, 1]) / 3) < 1e-12)
    th = [-2.0, 3.0, -0.5, -1.5, 0.4, -1.0, -0.8]
    Z, S, V1, V2, DH = simulate(2, th, 0.45, True, 3)
    Y = V1.astype(np.int8) + V2.astype(np.int8)
    fast = b3_predict(th, Y, S, DH, 0.45)[0]
    res["U3"] = bool(np.abs(fast - slow_filter(th, Y, S, DH, 0.45)).max() < 1e-6 and (fast[V1 | V2] == 1).all())
    zz = g.random(200000) < 0.5
    v1 = zz & (g.random(200000) < 0.4); v2 = zz & (g.random(200000) < 0.4)
    res["U4"] = abs(p_visit(v1, v2, np.ones(200000, bool))[0] - 0.4) < 0.01
    th0 = [-2.2, 0.0, 0.6, -1.2, 0.5, -1.0, -0.7]
    Z, S, V1, V2, DH = simulate(3000, th0, 0.45, False, 11)
    Y = V1.astype(np.int8) + V2.astype(np.int8)
    p0 = p_visit(V1, V2, S)[0]
    f0 = fit_b3(Y, S, DH, p0, True)
    err = max(abs(a - b) for i, (a, b) in enumerate(zip(f0["theta"], th0)) if i != 1)
    res["U5"] = (err < 0.1 and abs(f0["theta"][1]) < 0.3, err, f0["theta"])
    th1 = [-3.0, 4.0, 0.3, -1.5, 0.3, -1.0, -0.5]
    Z, S, V1, V2, DH = simulate(1500, th1, 0.45, True, 12)
    Y = V1.astype(np.int8) + V2.astype(np.int8)
    f1 = fit_b3(Y, S, DH, p_visit(V1, V2, S)[0], True)
    res["U6"] = (f1["theta"][1] > 1.0, f1["theta"])
    Vb = g.random((60, W, G, G)) < 0.02; Vb[:, 0, 5, 5] = True
    near = HR.smooth(Vb.any(1), 2.0)
    T = Vb[:, -1] | (g.random((60, G, G)) < 0.9 * near / (1 + near))
    Mk = ~Vb[:, -1]
    P4u, z = b4_chance(dict(a=0.5, b=-0.8), Vb)
    _, zf = b4_chance(dict(a=0.5, b=0.8), Vb)
    dd = near_dist(Vb)
    ref = np.full(dd.shape, np.inf)
    for i in range(len(Vb)):
        pts = np.argwhere(Vb[i].any(0))
        for r in range(G):
            for c in range(G):
                ref[i, r, c] = np.sqrt(((pts - [r, c]) ** 2).sum(1)).min()
    a_, b_, c_ = HR.aucs(z, T, Mk), HR.aucs(-dd, T, Mk), HR.aucs(zf, T, Mk)
    res["U7"] = (bool(np.array_equal(dd, ref) and np.array_equal(np.isnan(a_), np.isnan(b_)) and np.nanmax(np.abs(a_ - b_)) <= B4_TOL
                      and np.nanmean(b_) > 0.55 and np.nanmean(c_) < np.nanmean(b_) - 0.1 and (P4u[Vb] == 1).all()
                      and (dd[Vb.any(1)] == 0).all()), float(np.nanmax(np.abs(a_ - b_))))
    bx = b1_log(Vb, 0.25)
    b_ = HR.aucs(bx, T, Mk)
    b64 = HR.b1_score(Vb, 0.25)
    ok = b64 > 1e-250
    nu = lambda A_: np.array([len(np.unique(x)) for x in A_.reshape(len(A_), -1)])
    b32 = b64.astype(np.float32)
    s32 = HR.aucs(b32, T, Mk)
    z32 = b32 == 0
    zs = [len(np.unique(bx[i][z32[i]])) for i in range(len(bx)) if z32[i].sum() >= 5]
    split = len(zs) > 0 and np.mean(zs) >= 2                 # cells tied at 0 in float32 are ordered again (by distance)
    res["U9"] = (bool(np.allclose(np.exp(bx[ok]), b64[ok], rtol=1e-8) and np.isfinite(bx).all() and (nu(bx) >= nu(b32)).all()
                      and split), float(np.nanmean(b_) - np.nanmean(s32)), int(nu(b32).mean()), int(nu(bx).mean()))
    M5 = b5_maps(Vb)
    res["U8"] = bool(np.array_equal(M5[:, -1], Vb.any(1)) and np.array_equal(M5[:, 2], Vb[:, :3].any(1))
                     and np.allclose(b5_score(Vb)[Vb[:, -1]], 1.0))
    return res


def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    u = unit_checks()
    t("U1", u["U1"], "habitat distance equals stage E channel 20 (hab_dist)")
    t("U2", u["U2"], "neighbour mean equals a double loop; a corner uses its 3 in-window neighbours (no wrap)")
    t("U3", u["U3"], "the forward filter equals a cell-by-cell loop; recorded cells get chance 1 (no false positives)")
    t("U4", u["U4"], "p from the two visits recovers a planted 0.40 within 0.01")
    t("U5", u["U5"][0], f"data from the B3 process without neighbours: ML recovers the 6 other parameters within 0.1 "
                        f"(largest error {u['U5'][1]:.3f}) and a1 near 0 ({u['U5'][2][1]:+.3f})")
    t("U6", u["U6"][0], f"data with colonisation from true neighbours (a1 4): fitted a1 {u['U6'][1][1]:+.2f} above 1")
    t("U7", u["U7"][0], f"distance to the nearest record equals a double loop (0 at records); B4 ranks as -d: largest per-stack "
                        f"AUC gap {u['U7'][1]:.1e} (rule {B4_TOL}); a positive slope falls far below; recorded cells have chance 1")
    t("U9", u["U9"][0], f"exact log B1 equals log of B1 where B1 does not underflow, is finite everywhere and splits the float32 zeros; distinct values per "
                        f"stack float32 B1 {u['U9'][2]} against exact {u['U9'][3]}; exact minus float32 AUC here {u['U9'][1]:+.4f}")
    t("U8", u["U8"], "B5 maps are the records so far; B5 score of a target-year record is 1")
    env0 = dict(os.environ, STAGE_G2_BROKEN="wrap")
    code = ("import importlib.util as u, numpy as np, torch; s=u.spec_from_file_location('b', %r); m=u.module_from_spec(s); "
            "s.loader.exec_module(m); print('RES', m.unit_checks()['U2'])" % str(Path(__file__).resolve()))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env0)
    t("bug wrap", "RES False" in p.stdout, f"a torus neighbour mean fails U2 ({p.stdout.strip()[-20:] or p.stderr.strip()[-200:]})")
    for bug, test, what in (("b1_log_float32", "U9", "a log B1 built from float32"),):
        env0 = dict(os.environ, STAGE_G2_BROKEN=bug)
        code = ("import importlib.util as u, numpy as np, torch; s=u.spec_from_file_location('b', %r); m=u.module_from_spec(s); "
                "s.loader.exec_module(m); r=m.unit_checks()[%r]; print('RES', r[0] if isinstance(r, tuple) else r)"
                % (str(Path(__file__).resolve()), test))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env0)
        t(f"bug {bug}", "RES False" in p.stdout, f"{what} fails {test} ({p.stdout.strip()[-20:] or p.stderr.strip()[-200:]})")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = X.plant_grids(tmp / "g")
        st = tmp / "sim" / "real_train"; st.mkdir(parents=True); sv = tmp / "sim" / "real_val"; sv.mkdir(parents=True)
        for s_ in (7001, 7002, 7003):
            X.plant_community(st / f"c{s_}_training.npz", s_)
        for s_ in (7101, 7102):
            X.plant_community(sv / f"c{s_}_training.npz", s_)
        base = tmp / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.3, py, refpat, tools, out=quiet)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.3, py, refpat, tools, out=quiet)
        D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
        hres, hinfo = HR.run(str(base), str(base / "real_train"), str(base / "real_val"), str(base), out=quiet)
        e1, e2 = round(hinfo["bal"]["B1"], 3), round(hinfo["bal"]["B2"], 3)
        cfg = dict(CK.F.CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=2, K=2, ddim=4,
                   val_set="full", val_chunk=64, lr=1e-3)
        CK.F.train(cfg, str(base), str(tmp / "run"), "cpu", log=quiet)
        CK.run(tmp / "run", base, K=4, ddim=4, dev="cpu", expect_b1=e1, expect_b2=e2, out=quiet)
        lines = []
        r, info = run(base, base / "real_train", base / "real_val", tmp / "g2", runs=[tmp / "run"], K=8, out=lines.append)
        t("S1", all(r.get(g) for g in ("P1", "P2", "P3", "P4", "P5", "P6")), f"planted data end to end: {r}")
        t("S2", abs(info["bal"]["B1"][0] - hinfo["bal"]["B1"]) < 1e-12 and abs(info["bal"]["B2"][0] - hinfo["bal"]["B2"]) < 1e-12
          and abs(info["bal"]["random"][0] - 0.5) < 0.05,
          f"B1 and B2 scored here equal stage D exactly ({info['bal']['B1'][0]:.4f}, {info['bal']['B2'][0]:.4f})")
        t("S7", info["bal"]["B1_exact"][0] >= info["bal"]["B1"][0] and "run" in info["f5r"],
          f"exact B1 {info['bal']['B1_exact'][0]:.4f} against stored float32 B1 {info['bal']['B1'][0]:.4f} (ties broken by distance); "
          f"F5 recheck ran: {info['f5r']}")
        b3, b1 = info["bal"]["B3"][0], info["bal"]["B1"][0]
        t("S3", b3 > 0.5 and b3 > info["bal"]["B5"][0], f"B3 {b3:.4f} ranks above random and above records-only B5 "
          f"{info['bal']['B5'][0]:.4f} (B1 {b1:.4f}, B3 without habitat {info['bal']['B3_nohab'][0]:.4f})")
        z = np.load(tmp / "g2" / "maps_B4.npz")
        t("S4", z["maps"].shape[1:] == (8, NB) and str(z["layout"]) == GS.LAYOUT and np.load(tmp / "g2" / "maps_B5.npz")["maps"].shape[1] == 1,
          f"maps in the G1 layout: B4 {z['maps'].shape}, B5 one map")
        lines2 = []
        run(base, base / "real_train", base / "real_val", tmp / "g2b", runs=[tmp / "run"], K=8, out=lines2.append, compare=tmp / "g2")
        same = all(X.file_md5(tmp / "g2" / f) == X.file_md5(tmp / "g2b" / f) for f in
                   ("g2_fit.json", "g2_pred.npz", "maps_B3_halfblank.npz", "maps_B4.npz", "maps_B5.npz"))
        cmpl = next((x for x in lines2 if "against the earlier run" in x), "")
        t("S5", same and "B3 half-blank maps equal True, B5 maps equal True, B3 parameters equal True" in cmpl and "FLAG" not in cmpl,
          "a rerun into a new folder writes byte-equal fit, prediction and map files, and --compare reports them equal")
        stop = []
        for args in ((base, base / "real_train", base / "real_val", tmp / "g2"),
                     (base, base / "real_train", base / "real_val", tmp / "real_eval_g2")):
            try:
                run(*args, K=8, out=quiet)
                stop.append(False)
            except SystemExit:
                stop.append(True)
        t("S6", all(stop), f"an output folder that holds outputs is refused {stop[0]}; a real_eval path is refused {stop[1]}")
        for bug, gate in (("misalign", "P1"), ("fit_on_val", "P2"), ("truth_in_pred", "P3"), ("score_recorded", "P4"),
                          ("random_is_b1", "P4"), ("b4_flip", "P4"), ("maps_ignore_records", "P5"), ("p6_misorder", "P6")):
            env = dict(os.environ, STAGE_G2_BROKEN=bug)
            code = ("import importlib.util as u; s=u.spec_from_file_location('b', %r); m=u.module_from_spec(s); "
                    "s.loader.exec_module(m); r,_=m.run(%r, %r, %r, %r, runs=[%r], K=8, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), str(base), str(base / "real_train"), str(base / "real_val"),
                       str(tmp / f"b_{bug}"), str(tmp / "run")))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        lines3 = []
        run(base, base / "real_train", base / "real_val", tmp / "g2c", runs=[tmp / "run"], K=8, out=lines3.append,
            compare=tmp / "b_maps_ignore_records")
        cmpl = next((x for x in lines3 if "against the earlier run" in x), "")
        t("S8", "B3 half-blank maps equal False" in cmpl and "FLAG" in cmpl,
          "--compare against a run whose B3 maps differ (planted bug maps_ignore_records) reports them unequal and flags it")
        print("   planted run log:", *lines[:40], sep="\n      ")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--train", default=None); ap.add_argument("--val", default=None)
    ap.add_argument("--out", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--K", type=int, default=K_MAPS)
    ap.add_argument("--compare", default=None, help="an earlier G2 output folder: B3 and B5 maps and B3 parameters are compared")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    tr, va = a.train or str(Path(a.data) / "real_train"), a.val or str(Path(a.data) / "real_val")
    res, _ = run(a.data, tr, va, a.out, runs=a.runs, K=a.K, out=lambda s: print(s, flush=True), compare=a.compare)
    sys.exit(0 if res and all(res.values()) else 1)