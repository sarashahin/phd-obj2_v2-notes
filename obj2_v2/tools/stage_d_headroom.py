#!/usr/bin/env python3
"""
stage_d_headroom.py  (obj2_v2, stage D step 2, 8 Nov 2026)

Question: do the new communities hold signal that records alone do not (plan part 5, root cause R22)? If they do not, a
model trained on them cannot beat distance to a record, as the old model could not, and the plan changes before training.

What it does (claude/obj2_v2_plan.md, "Stage D step 2"; rule signed off by Sara on 8 Nov)
    Fit, on a real_train sample only (stacks drawn with the stage D sampler weights, fixed seed):
      B1   records-pinned Gaussian smoother on the 6 pooled years, no wrap; sigma chosen on the sample
      B2r  logistic on the log smoother surface and a flag for cells recorded only in earlier years (records only)
      B2e  B2r + effort: cell surveyed at the target year, years surveyed without a record
      B2   B2e + habitat: |noisy habitat - mean noisy habitat at the species' recorded cells|
    Score, on the frozen real_val set of stage D step 1 (every stack; the fast subset is reported beside it):
      per stack, AUC over the cells with no record at the target year, truth = occupied at the target year;
      mean per true range class (1, 2-5, 6-10, 11-20, 21-50, 51+; true 0 has no occupied cell at the target year),
      then the mean of the classes (class-balanced). Stacks without an occupied or an empty scored cell are left out
      and counted.
    Pass rule (written before the run): B2 beats B1 by 0.02 or more, and B1 stays below 0.95.

Gates (rules written before the run, 8 Nov)
    D6 no truth file opened while the frozen-set predictions are built (guard); predictions written with their md5
       before any real_val truth file is opened
    D7 the fit opens real_train files only; the frozen set and the sampler point to the same input files, md5 unchanged
       since stage D step 1; real_eval refused
    D8 scoring: no cell recorded at the target year is scored; a random score gives 0.50 (within 0.01, or 4 standard
       errors on small sets); stacks left out are counted
    D9 headroom rule as above: PASS, or FAIL and the plan changes before training
    Flags: the habitat part of the gain (B2 - B2e) when each stack is given the habitat map of another community
    above 0.005 (the habitat feature then also acts as a distance proxy, so only the part beyond it counts as habitat
    signal); the gain in the rare true classes (1-20 cells) below 0.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_d_headroom.py --self-test | tail -1
    python3 ../tools/stage_d_headroom.py --data ../data | tee ../logs/stage_d_headroom_8nov.txt
Needs numpy, export_obj2.py and stage_d_sampler.py in the same folder, and the two stage D files in --data.
"""
import argparse, hashlib, importlib.util, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_d_headroom.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


D = _load("stage_d_sampler")
X = D.X
G, WINDOW = X.G, X.WINDOW
TRUE = D.TRUE                                           # "0", "1", "2-5", "6-10", "11-20", "21-50", "51+"
SCORE = [1, 2, 3, 4, 5, 6]                              # true classes scored at the target year
RARE = [1, 2, 3, 4]                                     # true 1-20 cells
BINS = D.BINS
SIGMAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
N_FIT, N_SIG, N_BOOT = 6000, 3000, 1000
GAIN_MIN, B1_MAX, HAB_SHUF_FLAG = 0.02, 0.95, 0.005
PIN = 1e3                                               # pinned cells rank above every smoother value (sum of 400 kernels <= 400)
COLS = {"B2r": ["log_s", "pinned"],                    # no products: a product with the record count can turn the
        "B2e": ["log_s", "pinned", "surv_t", "nondet"],  # smoother slope negative for sparse species (sandbox, 8 Nov)
        "B2": ["log_s", "pinned", "surv_t", "nondet", "dh"]}
BROKEN = os.environ.get("STAGE_D_HEADROOM_BROKEN", "")  # self-test only: plants one bug so its gate must fail


# ---------------------------------------------------------------- scores and features (inputs only)
def kernel(sig):
    i = np.arange(G)
    K = np.exp(-((i[:, None] - i[None]) ** 2) / (2.0 * sig * sig))
    if BROKEN == "wrap":                                # torus distance: must fail the no-wrap test
        dd = np.minimum(abs(i[:, None] - i[None]), G - abs(i[:, None] - i[None]))
        K = np.exp(-(dd ** 2) / (2.0 * sig * sig))
    return K


def smooth(R, sig):
    """Sum over record cells of exp(-d^2 / 2 sigma^2), Euclidean distance in cells, no wrap (zero padding).
    The Gaussian is separable, so the surface is K R K^T per stack."""
    K = kernel(sig)
    return np.einsum("ij,njk,lk->nil", K, R.astype(np.float64), K)


def b1_score(V, sig):
    R6 = V.any(1)
    return smooth(R6, sig) + PIN * R6


def features(V, E, H, sig):
    """Per cell, from records V (n, 6, G, G), effort E (n, 6, G, G) and noisy habitat H (n, G, G)."""
    n = len(V)
    R6, Rt = V.any(1), V[:, -1]
    S = smooth(R6, sig)
    nrec = np.maximum(R6.reshape(n, -1).sum(1), 1)
    hm = (H * R6).reshape(n, -1).sum(1) / nrec
    return dict(log_s=np.log(S + 1e-300),            # 1e-300 keeps the B1 order down to float64 underflow
                pinned=(R6 & ~Rt).astype(np.float64), surv_t=E[:, -1].astype(np.float64),
                nondet=(E & ~V).sum(1).astype(np.float64), dh=np.abs(H - hm[:, None, None]))


def scored_mask(V):
    Rt = V[:, -1]
    return np.ones_like(Rt) if BROKEN == "score_recorded" else ~Rt


def design(f, cols, m):
    return np.column_stack([f[c][m] for c in cols])


# ---------------------------------------------------------------- AUC
def stack_auc(score, lab):
    """Mann-Whitney AUC with ties counted half (equal to export_obj2.auc)."""
    pos = score[lab]; neg = np.sort(score[~lab])
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    lt = np.searchsorted(neg, pos, "left"); le = np.searchsorted(neg, pos, "right")
    return (lt.sum() + 0.5 * (le - lt).sum()) / (len(pos) * len(neg))


def aucs(S, T, M):
    return np.array([stack_auc(S[i][M[i]], T[i][M[i]]) for i in range(len(S))])


def balanced(a, tc, classes=SCORE):
    """Mean per class (NaN left out), then the mean of the classes that have stacks; also its standard error."""
    means, ses = [], []
    for c in classes:
        v = a[(tc == c) & ~np.isnan(a)]
        if len(v):
            means.append(v.mean()); ses.append(v.var() / len(v))
    if not means:
        return np.nan, np.nan
    return float(np.mean(means)), float(np.sqrt(np.sum(ses)) / len(means))


def boot_gain(d, tc, comm, n_boot=N_BOOT):
    """Class-balanced mean of per-stack differences, communities resampled with replacement (fixed seed)."""
    ok = ~np.isnan(d)
    cu = np.unique(comm); ci = np.searchsorted(cu, comm)
    Ssum = np.zeros((len(cu), len(SCORE))); Scnt = np.zeros_like(Ssum)
    for k, c in enumerate(SCORE):
        m = ok & (tc == c)
        np.add.at(Ssum[:, k], ci[m], d[m]); np.add.at(Scnt[:, k], ci[m], 1)
    g = X.rng(99, 20)
    out = []
    for _ in range(n_boot):
        w = np.bincount(g.integers(0, len(cu), len(cu)), minlength=len(cu)).astype(float)
        s, n = w @ Ssum, w @ Scnt
        out.append(np.mean(s[n > 0] / n[n > 0]))
    return np.percentile(out, [2.5, 97.5])


# ---------------------------------------------------------------- logistic (weighted, Newton steps)
def fit_logit(A, y, w, lam=1e-3, iters=100):
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([np.ones(len(A)), (A - mu) / sd])
    b = np.zeros(Z.shape[1]); P = lam * np.eye(Z.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
        g = Z.T @ (w * (y - p)) - P @ b
        Hs = (Z * (w * p * (1 - p))[:, None]).T @ Z + P
        step = np.linalg.solve(Hs, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return dict(b=b, mu=mu, sd=sd)


def logit_score(fit, A):
    return np.column_stack([np.ones(len(A)), (A - fit["mu"]) / fit["sd"]]) @ fit["b"]


def balanced_weights(T, M, tc):
    """Row weights: each stack's occupied and empty scored cells weigh 1/2 each, then each true class weighs the same."""
    n = len(T)
    P = (T & M).reshape(n, -1).sum(1); N = (~T & M).reshape(n, -1).sum(1)
    keep = (P > 0) & (N > 0)
    W = np.where(T, 0.5 / np.maximum(P, 1)[:, None, None], 0.5 / np.maximum(N, 1)[:, None, None]) * keep[:, None, None]
    present = [c for c in SCORE if (keep & (tc == c)).any()]
    for c in present:
        k = keep & (tc == c)
        W[k] *= keep.sum() / len(present) / k.sum()
    return W


# ---------------------------------------------------------------- data
class OpenLog:
    """Logs every np.load path while active (gate D7)."""
    def __enter__(self):
        self.orig, self.paths = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        self.paths.append(str(Path(f).resolve()))
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig
        return False


def load_stacks(d, files, fidx, rows, truth):
    """Records, effort and noisy habitat (and, if truth, the target-year true map, true cells and community seed)
    for the listed stacks, in the listed order."""
    man = {r["input_file"]: r for r in D.rows_of(d)}
    n = len(rows)
    V = np.zeros((n, WINDOW, G, G), bool); E = np.zeros_like(V); H = np.zeros((n, G, G))
    T = np.zeros((n, G, G), bool) if truth else None
    tc = np.zeros(n, int); comm = np.zeros(n, np.int64)
    for i in np.unique(fidx):
        k = np.nonzero(fidx == i)[0]; r = rows[k]
        row = man[str(files[i])]
        with np.load(Path(d) / row["input_file"], allow_pickle=False) as z:
            V[k] = X.unpack(z["visit1"][r]) | X.unpack(z["visit2"][r]); E[k] = X.unpack(z["effort"][r])
            H[k] = z["habitat_noisy"][z["stack_hab"][r]].astype(np.float64)
        comm[k] = int(row["sim_seed"])
        if truth:
            with np.load(Path(d) / row["truth_file"], allow_pickle=False) as zt:
                T[k] = X.unpack(zt["true_maps"][r])[:, -1]
                tc[k] = D.tidx(zt["true_cells"][r])
    return V, E, H, T, tc, comm


def fit_sample(train_dir, sampler):
    """N_FIT real_train stacks drawn with probability proportional to the stage D weight, fixed seed."""
    w = sampler["weight"].astype(np.float64)
    g = X.rng(99, 10)
    pick = np.sort(g.choice(len(w), min(N_FIT, int((w > 0).sum())), replace=False, p=w / w.sum()))
    return sampler["files"], sampler["file_index"][pick], sampler["row"][pick]


def partner(comm):
    """For each stack, a stack of the next community (cyclic order of seeds), at the same position modulo its size."""
    cu = np.unique(comm)
    idx = {c: np.nonzero(comm == c)[0] for c in cu}
    p = np.zeros(len(comm), int)
    for j, c in enumerate(cu):
        nxt = idx[cu[(j + 1) % len(cu)]]
        p[idx[c]] = nxt[np.arange(len(idx[c])) % len(nxt)]
    return p


def predict(V, E, H, sig, fits, comm, chunk=2000):
    """All scores for the frozen set, from inputs only, in chunks of stacks to bound memory."""
    n = len(V)
    keys = ["B1"] + list(COLS) + ["B2_shuffled_habitat"]
    pred = {k: np.zeros((n, G, G), np.float32) for k in keys}
    Hs = H[partner(comm)]                                              # habitat maps of another community
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        allm = np.ones(V[sl][:, 0].shape, bool)
        pred["B1"][sl] = b1_score(V[sl], sig)
        f = features(V[sl], E[sl], H[sl], sig)
        for k, c in COLS.items():
            pred[k][sl] = logit_score(fits[k], design(f, c, allm)).reshape(-1, G, G)
        fs = features(V[sl], E[sl], Hs[sl], sig)
        pred["B2_shuffled_habitat"][sl] = logit_score(fits["B2"], design(fs, COLS["B2"], allm)).reshape(-1, G, G)
    pred["random"] = X.rng(99, 21).random((n, G, G)).astype(np.float32) if BROKEN != "random_is_b1" else pred["B1"].copy()
    return pred


# ---------------------------------------------------------------- run
def run(data, train_dir, val_dir, out_dir, out=print):
    for p in (data, train_dir, val_dir, out_dir):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, after training")
    sp, fp = Path(data) / "stage_d_train_sampler.npz", Path(data) / "stage_d_frozen_val.npz"
    sampler, frozen = dict(np.load(sp)), dict(np.load(fp))
    res = {}

    def check(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    cmd5 = X.file_md5(__file__)
    out(f"stage_d_headroom.py md5 {cmd5}; stage_d_sampler.py md5 {X.file_md5(HERE / 'stage_d_sampler.py')}; "
        f"export_obj2.py md5 {X.file_md5(HERE / 'export_obj2.py')}")
    out(f"   sampler {sp} md5 {X.file_md5(sp)}; frozen set {fp} md5 {X.file_md5(fp)}")
    # ---- D7 inputs unchanged since stage D step 1
    same = all(X.file_md5(Path(train_dir) / f) == m for f, m in zip(sampler["files"], sampler["file_md5"])) and \
        all(X.file_md5(Path(val_dir) / f) == m for f, m in zip(frozen["files"], frozen["file_md5"]))
    # ---- fit on real_train only
    fdir = train_dir if BROKEN != "fit_on_val" else val_dir
    with OpenLog() as log:
        if BROKEN == "fit_on_val":
            files, fi, rw = frozen["files"], frozen["file_index"], frozen["row"]
        else:
            files, fi, rw = fit_sample(train_dir, sampler)
        V, E, H, T, tc, _ = load_stacks(fdir, files, fi, rw, truth=True)
    fit_paths = log.paths
    M = scored_mask(V)
    sig_scores = {}
    ns = min(N_SIG, len(V))
    for s in SIGMAS:
        sig_scores[s] = balanced(aucs(b1_score(V[:ns], s), T[:ns], M[:ns]), tc[:ns])[0]
    sig = max(SIGMAS, key=lambda s: (sig_scores[s], -s))
    out(f"   fit sample: {len(V)} real_train stacks drawn with the stage D weights (seed fixed); true classes "
        + ", ".join(f"{TRUE[c]} {int((tc == c).sum())}" for c in range(len(TRUE))))
    out("   B1 sigma on the first " + str(ns) + " of them (class-balanced AUC): "
        + ", ".join(f"{s:g} {v:.3f}" for s, v in sig_scores.items()) + f" -> sigma {sig:g}")
    f = features(V, E, H, sig)
    Wt = balanced_weights(T, M, tc)
    m = M & (Wt > 0)
    fits = {k: fit_logit(design(f, c, m), T[m].astype(float), Wt[m]) for k, c in COLS.items()}
    for k, c in COLS.items():
        out(f"   {k} coefficients (standardised): " + ", ".join(f"{n} {b:+.3f}" for n, b in zip(["intercept"] + c, fits[k]["b"])))
    del V, E, H, T, f, Wt, m
    inside = lambda p, d: str(p).startswith(str(Path(d).resolve()) + os.sep)
    fit_ok = all(inside(p, train_dir) for p in fit_paths) and not any(inside(p, val_dir) for p in fit_paths)
    check("D7", fit_ok and same and "real_val" in str(val_dir),
          f"the fit opened {len(fit_paths)} files, all in real_train {fit_ok}; input files md5 unchanged since stage D step 1 {same}")
    # ---- predictions on the frozen real_val set, inputs only
    vf, vfi, vrw = frozen["files"], frozen["file_index"], frozen["row"]
    with D.TruthGuard() as guard:
        try:
            Vv, Ev, Hv, _, _, comm = load_stacks(val_dir, vf, vfi, vrw, truth=(BROKEN == "truth_in_pred"))
            Mv = scored_mask(Vv)
            pred = predict(Vv, Ev, Hv, sig, fits, comm)
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        check("D6", False, f"{crashed}")
        return res, None
    os.makedirs(out_dir, exist_ok=True)
    pp = Path(out_dir) / "stage_d_headroom_pred.npz"
    keep = {k: pred[k] for k in ["B1"] + list(COLS)}                   # the two controls are kept as md5 only (size)
    ctrl = {f"md5_{k}": np.array(hashlib.md5(pred[k].tobytes()).hexdigest()) for k in ("B2_shuffled_habitat", "random")}
    X.save_npz(pp, dict(keep, **ctrl, sigma=np.float64(sig), code_md5=np.array(cmd5), frozen_md5=np.array(X.file_md5(fp))))
    pmd5 = X.file_md5(pp)
    out(f"   written {pp} md5 {pmd5} (before any real_val truth file is opened)")
    check("D6", not guard.opened, f"no truth file opened while the frozen-set predictions were built (opened {len(guard.opened)})")
    # ---- truth (after the predictions are written)
    _, _, _, Tv, tcv, _ = load_stacks(val_dir, vf, vfi, vrw, truth=True)
    rec_scored = int((Mv & Vv[:, -1]).sum())
    A = {k: aucs(v, Tv, Mv) for k, v in pred.items()}
    left = int(np.isnan(A["B1"]).sum())
    rnd, rse = balanced(A["random"], tcv)
    tol = max(0.01, 4 * rse)
    check("D8", rec_scored == 0 and abs(rnd - 0.5) <= tol,
          f"cells recorded at the target year among the scored cells {rec_scored}; random score {rnd:.3f} (0.50 within {tol:.3f}); "
          f"stacks left out (no occupied or no empty scored cell) {left} of {len(Mv)}: true 0 {int((tcv == 0).sum())}, others "
          f"{left - int(np.isnan(A['B1'][tcv == 0]).sum())}")
    fast = frozen["fast"].astype(bool)
    out("   class-balanced AUC over unrecorded target-year cells (full frozen set | fast subset):")
    bal = {}
    for k in ("B1", "B2r", "B2e", "B2", "B2_shuffled_habitat", "random"):
        bal[k] = balanced(A[k], tcv)[0]
        out(f"      {k:<20} {bal[k]:.3f} | {balanced(A[k][fast], tcv[fast])[0]:.3f}")
    blank = frozen["blank"].astype(bool)
    Ab = np.where(blank, A["B2e"], A["B2"])
    out(f"      B2 with the frozen half-blank habitat (B2e where blanked) {balanced(Ab, tcv)[0]:.3f}")
    out("   by true class at the target year (stacks scored / B1 / B2r / B2e / B2 / B2 - B1):")
    for c in SCORE:
        k = (tcv == c) & ~np.isnan(A["B1"])
        out(f"      {TRUE[c]:>6}  {int(k.sum()):>6}  " + "  ".join(f"{np.nanmean(A[v][k]) if k.any() else np.nan:.3f}" for v in ("B1", "B2r", "B2e", "B2"))
            + f"  {np.nanmean(A['B2'][k] - A['B1'][k]) if k.any() else np.nan:+.3f}")
    r6 = Vv.any(1).reshape(len(Vv), -1).sum(1); bn = D.bidx(r6)
    out("   by recorded cells over 6 years (stacks scored / B1 / B2 / B2 - B1):")
    for b, lab in enumerate(BINS):
        k = (bn == b) & ~np.isnan(A["B1"])
        out(f"      {lab:>6}  {int(k.sum()):>6}  {np.nanmean(A['B1'][k]) if k.any() else np.nan:.3f}  "
            f"{np.nanmean(A['B2'][k]) if k.any() else np.nan:.3f}  {np.nanmean(A['B2'][k] - A['B1'][k]) if k.any() else np.nan:+.3f}")
    d = A["B2"] - A["B1"]
    gain = balanced(d, tcv)[0]
    lo, hi = boot_gain(d, tcv, comm)
    rare = balanced(d, tcv, RARE)[0]
    hab, hab_s = bal["B2"] - bal["B2e"], bal["B2_shuffled_habitat"] - bal["B2e"]
    out(f"   gain B2 - B1 {gain:+.3f} (communities resampled, 95%: {lo:+.3f} to {hi:+.3f}); rare true classes 1-20 cells {rare:+.3f}")
    out(f"   parts of the gain: logistic on records {bal['B2r'] - bal['B1']:+.3f}, effort {bal['B2e'] - bal['B2r']:+.3f}, habitat {hab:+.3f}; "
        f"the same model given the habitat maps of another community {hab_s:+.3f} (any smooth map near the records acts as a "
        f"distance proxy), so habitat beyond that {hab - hab_s:+.3f}")
    check("D9", gain >= GAIN_MIN and bal["B1"] < B1_MAX,
          f"B2 beats B1 by {gain:+.3f} (rule 0.02 or more) and B1 {bal['B1']:.3f} (rule below 0.95)")
    flags = [t for t, on in (("habitat maps of another community give above 0.005", hab_s > HAB_SHUF_FLAG),
                             ("gain in the rare true classes below 0", rare < 0)) if on]
    n_ok = sum(res.values())
    out(f"stage D step 2: {n_ok} PASS, {len(res) - n_ok} FAIL; flags: " + (", ".join(flags) or "none"))
    return res, dict(gain=gain, b1=bal["B1"], hab=hab, hab_s=hab_s, rare=rare, bal=bal, sigma=sig)


# ---------------------------------------------------------------- self-test
def plant_nohab(path, seed):
    """As export_obj2.plant_community, then the saved habitat maps are replaced by maps unrelated to the species."""
    X.plant_community(path, seed)
    z = dict(np.load(path))
    g = np.random.default_rng(seed + 999)
    z["ENV_habitat_t"] = g.standard_normal(z["ENV_habitat_t"].shape).astype(np.float32)
    np.savez(path, **z)


def full_effort(py, refpat):
    """Planted grids with every cell surveyed in every year (known cells all 1)."""
    for f in list(Path(py).glob("*/bnm_*.npz")) + list(Path(refpat).parent.glob("bnm_*.npz")):
        z = dict(np.load(f))
        for k in ("known_target", "known_holdout"):
            if k in z:
                z[k] = np.ones_like(z[k])
        np.savez(f, **z)


def unit_checks():
    """AUC against export_obj2.auc; smoother against a double loop; no wrap; logistic recovers planted coefficients."""
    g = np.random.default_rng(5)
    ok_auc = True
    for _ in range(300):
        s = g.integers(0, 6, 40).astype(float); lab = g.random(40) < 0.3
        a, b = stack_auc(s, lab), X.auc(s, lab)
        ok_auc &= (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12
    R = g.random((3, G, G)) < 0.05
    sig = 1.7
    S = smooth(R, sig)
    yy, xx = np.mgrid[0:G, 0:G]
    Sb = np.zeros_like(S)
    for n in range(3):
        for y, x in zip(*np.nonzero(R[n])):
            Sb[n] += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sig * sig))
    one = np.zeros((1, G, G), bool); one[0, 0, 0] = True
    corner = smooth(one, 3.0)[0, G - 1, G - 1]
    ok_wrap = abs(corner - np.exp(-2 * (G - 1) ** 2 / 18.0)) < 1e-12
    A = g.standard_normal((60000, 3)); bt = np.array([-0.5, 1.0, -2.0, 0.7])
    y = (g.random(60000) < 1 / (1 + np.exp(-(bt[0] + A @ bt[1:])))).astype(float)
    fit = fit_logit(A, y, np.ones(len(y)))
    braw = fit["b"][1:] / fit["sd"]
    return ok_auc, bool(np.allclose(S, Sb, atol=1e-10)), ok_wrap, bool(np.abs(braw - bt[1:]).max() < 0.1)


def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    a, s, w, l = unit_checks()
    t("U1", a, "per-stack AUC equals export_obj2.auc on 300 cases with ties")
    t("U2", s, "separable smoother equals the sum over records")
    t("U3", w, "no wrap: a record in one corner reaches the far corner at the Euclidean distance")
    t("U4", l, "weighted logistic recovers planted coefficients within 0.1")
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        # signal: species placed by habitat, habitat noise small; nohab: habitat unrelated to the species;
        # null: habitat unrelated and every cell surveyed every year, so effort says nothing beyond the records
        for kind, plant, sigma, full in (("signal", X.plant_community, 0.1, False), ("nohab", plant_nohab, 0.6, False),
                                         ("null", plant_nohab, 0.6, True)):
            py, refpat = X.plant_grids(Path(tmp) / kind)
            if full:
                full_effort(py, refpat)
            st = Path(tmp) / kind / "sim" / "real_train"; st.mkdir(parents=True)
            sv = Path(tmp) / kind / "sim" / "real_val"; sv.mkdir(parents=True)
            for s_ in (7001, 7002, 7003, 7004):
                plant(st / f"c{s_}_training.npz", s_)
            for s_ in (7101, 7102, 7103):
                plant(sv / f"c{s_}_training.npz", s_)
            base = Path(tmp) / kind / "data"
            X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), sigma, py, refpat, tools, out=quiet)
            X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), sigma, py, refpat, tools, out=quiet)
            D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
            out[kind] = (base, run(str(base), str(base / "real_train"), str(base / "real_val"), str(base), out=quiet))
        rs, info = out["signal"][1]
        t("S1", all(rs.values()), f"species placed by habitat: D6-D9 PASS {rs}; gain {info['gain']:+.3f}")
        t("S2", info["hab"] >= 0.01 and info["hab_s"] < info["hab"] / 2,
          f"habitat part {info['hab']:+.3f} (0.01 or more); with another community's habitat maps {info['hab_s']:+.3f} (below half of it)")
        rn, inn = out["nohab"][1]
        t("S3", abs(inn["hab"]) < 0.01 and all(rn[k] for k in ("D6", "D7", "D8")),
          f"habitat unrelated to the species: habitat part {inn['hab']:+.3f} (within 0.01), D6-D8 {rn}")
        r0, i0 = out["null"][1]
        t("S4", not r0["D9"] and all(r0[k] for k in ("D6", "D7", "D8")),
          f"habitat unrelated and every cell surveyed: gain {i0['gain']:+.3f}, D9 fails as it must {r0}")
        base = out["signal"][0]
        p1 = X.file_md5(base / "stage_d_headroom_pred.npz")
        run(str(base), str(base / "real_train"), str(base / "real_val"), str(base), out=quiet)
        t("S5", X.file_md5(base / "stage_d_headroom_pred.npz") == p1, "a rerun writes a byte-equal prediction file")
        want = {"truth_in_pred": "D6", "fit_on_val": "D7", "score_recorded": "D8", "random_is_b1": "D8"}
        for bug, gate in want.items():
            env = dict(os.environ); env["STAGE_D_HEADROOM_BROKEN"] = bug
            code = ("import sys; sys.argv=['x']; import importlib.util as u; s=u.spec_from_file_location('h', %r); "
                    "m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r,_=m.run(%r, %r, %r, %r, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), str(base), str(base / "real_train"), str(base / "real_val"), str(base / f"b_{bug}")))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        env = dict(os.environ); env["STAGE_D_HEADROOM_BROKEN"] = "wrap"
        code = ("import importlib.util as u; s=u.spec_from_file_location('h', %r); m=u.module_from_spec(s); "
                "s.loader.exec_module(m); print('RES', m.unit_checks()[2])" % str(Path(__file__).resolve()))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        t("bug wrap", "RES False" in p.stdout, "a torus smoother fails the no-wrap check")
        stop = False
        try:
            run(str(base), str(base / "real_train"), str(Path(tmp) / "real_eval"), str(base), out=quiet)
        except SystemExit:
            stop = True
        t("S6", stop, "a real_eval folder is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--train", default=None); ap.add_argument("--val", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    tr, va = a.train or str(Path(a.data) / "real_train"), a.val or str(Path(a.data) / "real_val")
    res, _ = run(a.data, tr, va, a.out or a.data)
    sys.exit(0 if res and all(res.values()) else 1)