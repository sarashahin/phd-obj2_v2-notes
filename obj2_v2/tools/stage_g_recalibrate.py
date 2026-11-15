#!/usr/bin/env python3
"""
stage_g_recalibrate.py  (obj2_v2, stage G step G3b, 11 Nov 2026; approach agreed by Sara on 11 Nov)

Question: G3 found the model's P(AOO <= x) skilful but too low at recorded class 0. Does a logistic recalibration,
fitted on the other fold of communities, make it calibrated without losing skill? The maps and the AOO and EOO
intervals are not touched; only P(AOO <= x) changes. All methods get the same treatment.

Rules (written before the run; claude/obj2_v2_results_log_G.md, "Stage G3b rules")
    Input: G3's g3_calibrated.npz (its md5 must equal g3_report.json md5.calibrated) and nothing else. It holds, per
    frozen real_val stack, P(AOO <= x) for x = 1, 5, 10, 20 per method, the true target-year AOO, the cells recorded at
    the target year, the recorded class, the community and the G3 fold.
    Known zeros: true AOO is never below the cells recorded at the target year, so P'(AOO <= x) = 0 where they exceed x.
    Recalibration (Platt 1999, logistic): z = logit(P clipped to [0.01, 0.99]) (0.01 = half of one map in 50);
    P' = 1 / (1 + exp(-(a + b z))). a and b are fitted by maximum likelihood on the fit fold, per recorded class, among
    the stacks where AOO <= x is possible, and applied to the other fold; then the folds swap (the G3 folds).
    A recorded class needs 30 or more fit-fold stacks with 5 or more of each outcome; otherwise the fit over all its
    possible fit-fold stacks is used, under the same condition; otherwise P' = P.
    P' is then made non-decreasing in x (running maximum over x = 1, 5, 10, 20); the number of values raised is reported.
    Scores on the check folds, before (G3's P) and after (P'), as in G3: Brier, Brier skill against the other fold's
    base rate per recorded class, mean P against the observed share overall and per recorded class, and the reliability
    table (10 bins; largest gap over bins of 50+ stacks and in standard errors; also over bins of 200+ stacks).
    Check rule for each model seed and x: R1 Brier after <= Brier before; R2 in recorded classes 0 and 1, |mean P' -
    observed share| <= 0.02; R3 largest reliability gap over bins of 200+ stacks <= 0.05. Pooled and baselines reported.
    Brier after - before with a 95% interval from resampling the 20 communities (2,000 draws, seed 31).

Gates
    L1  input md5 as G3 logged it; every array covers the same stacks; the folds equal G3's rule (on real_val 10 and 10
        communities)
    L2  leak check: each fold's P' (model seed 1, every x) is unchanged when that fold's own truth is shuffled
    L3  P' in [0, 1]; P' = 0 where the recorded cells exceed x; P' non-decreasing in x; the "before" Brier, skill and
        largest gap equal g3_report.json for every method and x (to 1e-9)
    real_eval is refused; the output folder must be new.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, under a minute)
    python3 ../tools/stage_g_recalibrate.py --self-test | tail -1
    python3 -u ../tools/stage_g_recalibrate.py --g3 ../data/stage_g3_calibration --out ../data/stage_g3b_recalibration \\
        > ../logs/stage_g3b_recalibration.txt 2>&1
Needs numpy and stage_g_calibrate.py (with the stage tools it loads) in the same folder.
"""
import argparse, csv, importlib.util, json, math, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g_recalibrate.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


GC = _load("stage_g_calibrate")
X, TCLS, XS, N_TEST = GC.X, GC.TCLS, GC.XS, GC.N_TEST
EPS, N_MIN, E_MIN, RIDGE = 0.01, 30, 5, 1e-6
R2_TOL, R3_TOL, R3_BIN = 0.02, 0.05, 200
BROKEN = os.environ.get("STAGE_G3B_BROKEN", "")  # self-test only: plants one bug so its gate must fail


def logit_clip(P):
    q = np.clip(P, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def fit_logistic(z, o):
    """ML logistic regression o ~ a + b z by Newton steps (tiny ridge keeps a flat z finite)."""
    A = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -35, 35)))
        g = A.T @ (o - p) - RIDGE * w
        H = (A * (p * (1 - p))[:, None]).T @ A + RIDGE * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.abs(step).max() < 1e-12:
            break
    return w


def recalibrate(P, y, rec_t, rcls, fold, only=None):
    """P (n, 4) -> P' (n, 4), each fold from the other fold's fits; parameters per (check fold, x, group)."""
    n = len(y)
    out = np.array(P, float).copy(); pars = []
    for j, x in enumerate(XS):
        possible = rec_t <= x
        o = (y <= x).astype(float)
        z = logit_clip(P[:, j])
        for c in (0, 1):
            if only is not None and c != only:
                continue
            fit = (fold == (1 - c)) if BROKEN != "leak" else np.ones(n, bool)
            chk = fold == c
            allfit = fit & possible
            ok = lambda m: m.sum() >= N_MIN and o[m].sum() >= E_MIN and (1 - o[m]).sum() >= E_MIN
            w_all = fit_logistic(z[allfit], o[allfit]) if ok(allfit) else None
            for g in np.unique(rcls[chk & possible]):
                m = allfit & (rcls == g)
                if ok(m):
                    w, src = fit_logistic(z[m], o[m]), "class"
                elif w_all is not None:
                    w, src = w_all, "all"
                else:
                    w, src = None, "none"
                t = chk & possible & (rcls == g)
                out[t, j] = P[t, j] if w is None else 1 / (1 + np.exp(-(w[0] + w[1] * z[t])))
                pars.append(dict(check_fold=int(c), x=int(x), recorded_class=TCLS[g], source=src, n_fit=int(m.sum()),
                                 events=int(o[m].sum()), a=None if w is None else float(w[0]), b=None if w is None else float(w[1])))
            if BROKEN != "no_zero":
                out[chk & ~possible, j] = 0.0
    before_mono = out.copy()
    if BROKEN != "no_mono":
        out = np.maximum.accumulate(out, axis=1)
    return out, pars, int((out > before_mono + 1e-15).sum())


def evaluate(P, y, fold, rcls, x):
    """G3's P(AOO <= x) check (same bins, same reference), plus bins of 200+ stacks and per recorded class."""
    o = (y <= x).astype(float)
    ref = np.zeros(len(y))
    for c in (0, 1):
        fit, chk = fold == 1 - c, fold == c
        for g in np.unique(rcls[chk]):
            m = fit & (rcls == g)
            ref[chk & (rcls == g)] = o[m].mean() if m.any() else o[fit].mean()
    brier, bref = float(((P - o) ** 2).mean()), float(((ref - o) ** 2).mean())
    b = np.minimum((P * 10).astype(int), 9)
    tab = [dict(bin=k, n=int((b == k).sum()), p=float(P[b == k].mean()) if (b == k).any() else float("nan"),
                obs=float(o[b == k].mean()) if (b == k).any() else float("nan")) for k in range(10)]
    big = [t for t in tab if t["n"] >= N_TEST]
    gaps = [abs(t["p"] - t["obs"]) for t in big]
    zs = [abs(t["p"] - t["obs"]) / math.sqrt(max(t["p"] * (1 - t["p"]), 1e-4) / t["n"]) for t in big]
    gaps200 = [abs(t["p"] - t["obs"]) for t in tab if t["n"] >= R3_BIN]
    per = {TCLS[g]: (float(P[rcls == g].mean()), float(o[rcls == g].mean()), int((rcls == g).sum())) for g in np.unique(rcls)}
    return dict(brier=brier, brier_ref=bref, bss=1 - brier / bref if bref > 0 else float("nan"),
                gap=max(gaps) if gaps else float("nan"), gap_z=max(zs) if zs else float("nan"),
                gap200=max(gaps200) if gaps200 else float("nan"), mean_p=float(P.mean()), share=float(o.mean()),
                per_class=per, table=tab)


def boot_diff(d, comm, n_boot=2000, seed=31):
    cu = np.unique(comm)
    s = np.array([d[comm == c].sum() for c in cu]); k = np.array([(comm == c).sum() for c in cu])
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(cu), (n_boot, len(cu)))
    bs = s[idx].sum(1) / k[idx].sum(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run(g3_dir, out_dir, out=print):
    g3_dir, out_dir = Path(g3_dir), Path(out_dir)
    for p in (g3_dir, out_dir):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G5")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    res, t0 = {}, time.time()

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    rep = json.loads((g3_dir / "g3_report.json").read_text())
    fp = g3_dir / "g3_calibrated.npz"
    m5 = X.file_md5(fp)
    z = np.load(fp)
    out(f"stage_g_recalibrate.py md5 {X.file_md5(__file__)}; stage_g_calibrate.py {X.file_md5(HERE / 'stage_g_calibrate.py')}; "
        f"g3_calibrated.npz md5 {m5}")
    keys = sorted(k[5:] for k in z.files if k.startswith("p_le_"))
    order = [k for k in keys if k.startswith("model_")] + [k for k in ("pooled", "B3", "B4", "B5") if k in keys]
    y, rec_t, rcls, fold, comm = z["true_aoo"], z["rec_t"], z["rcls"], z["fold"], z["comm"]
    if BROKEN == "shift_truth":
        y = np.roll(y, 1)
    n = len(y)
    same_n = all(len(a) == n for a in (rec_t, rcls, fold, comm)) and all(z[f"p_le_{k}"].shape == (n, len(XS)) for k in order)
    nc = [len(np.unique(comm[fold == f])) for f in (0, 1)]
    folds_ok = np.array_equal(fold, GC.make_folds(comm)) and abs(nc[0] - nc[1]) <= 1
    gate("L1", m5 == rep["md5"]["calibrated"] and same_n and folds_ok,
         f"input md5 as G3 logged it ({m5[:8]} against {rep['md5']['calibrated'][:8]}); {n} stacks in every array "
         f"{same_n}; folds by G3's rule ({nc[0]} and {nc[1]} communities) {folds_ok}")
    P0 = {k: z[f"p_le_{k}"].astype(float) for k in order}
    P1, pars, raised = {}, {}, {}
    for k in order:
        P1[k], pars[k], raised[k] = recalibrate(P0[k], y, rec_t, rcls, fold)
    # L2 leak check on the first model seed
    s1 = order[0]
    leak_ok = True
    for c in (0, 1):
        idx = np.nonzero(fold == c)[0]
        y2 = y.copy(); y2[idx] = y[X.rng(77, 9, c).permutation(idx)]
        a_, _, _ = recalibrate(P0[s1], y, rec_t, rcls, fold, only=c)
        b_, _, _ = recalibrate(P0[s1], y2, rec_t, rcls, fold, only=c)
        leak_ok &= np.array_equal(a_[idx], b_[idx])
    gate("L2", leak_ok, f"each fold's recalibrated P ({s1}, every x) unchanged when that fold's own truth is shuffled")
    # L3 structure and agreement with G3
    in01 = all(bool(((P1[k] >= 0) & (P1[k] <= 1)).all()) for k in order)
    zero = all(bool((P1[k][rec_t[:, None] > np.array(XS)[None, :]] == 0).all()) for k in order)
    mono = all(bool((np.diff(P1[k], axis=1) >= -1e-15).all()) for k in order)
    ev0 = {k: {x: evaluate(P0[k][:, j], y, fold, rcls, x) for j, x in enumerate(XS)} for k in order}
    worst = 0.0
    for k in order:
        rk = k.replace("_", " ", 1) if k.startswith("model_") else k
        for x in XS:
            g3 = rep["p_le"][rk][str(x)]
            for q in ("brier", "bss", "gap"):
                a, b = ev0[k][x][q], g3[q]
                if not (isinstance(a, float) and math.isnan(a) and (b is None or (isinstance(b, float) and math.isnan(b)))):
                    worst = max(worst, abs(a - b))
    gate("L3", in01 and zero and mono and worst < 1e-9,
         f"P' in [0, 1] {in01}; P' = 0 where the recorded cells exceed x {zero}; non-decreasing in x {mono}; 'before' Brier, "
         f"skill and gap equal g3_report.json (largest difference {worst:.1e})")
    # scores
    ev1 = {k: {x: evaluate(P1[k][:, j], y, fold, rcls, x) for j, x in enumerate(XS)} for k in order}
    diffs = {k: {x: boot_diff((P1[k][:, j] - (y <= x)) ** 2 - (P0[k][:, j] - (y <= x)) ** 2, comm) for j, x in enumerate(XS)} for k in order}
    out("   P(AOO <= x) on the check folds, before -> after (Brier / skill / largest gap, bins 50+ | 200+ / mean P vs share):")
    for k in order:
        for x in XS:
            b0, b1, d = ev0[k][x], ev1[k][x], diffs[k][x]
            out(f"      {k:<14} x {x:>2}: {b0['brier']:.4f} -> {b1['brier']:.4f} (after - before {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]); "
                f"skill {b0['bss']:+.3f} -> {b1['bss']:+.3f}; gap {b0['gap']:.3f} | {b0['gap200']:.3f} -> {b1['gap']:.3f} | {b1['gap200']:.3f}; "
                f"mean P {b0['mean_p']:.3f} -> {b1['mean_p']:.3f} vs {b1['share']:.3f}")
    out("   mean P against the observed share in recorded classes 0 and 1, before -> after:")
    for k in order:
        out(f"      {k:<14} " + "; ".join(f"x {x}: " + ", ".join(f"class {g} {ev0[k][x]['per_class'][g][0]:.3f} -> {ev1[k][x]['per_class'][g][0]:.3f} vs "
                                                                f"{ev1[k][x]['per_class'][g][1]:.3f}" for g in ("0", "1") if g in ev1[k][x]['per_class'])
                                         for x in XS))
    verdict = {}
    for k in order:
        if not k.startswith("model_"):
            continue
        for x in XS:
            b0, b1 = ev0[k][x], ev1[k][x]
            r1 = b1["brier"] <= b0["brier"] + 1e-15
            r2 = all(abs(b1["per_class"][g][0] - b1["per_class"][g][1]) <= R2_TOL for g in ("0", "1") if g in b1["per_class"])
            r3 = not (b1["gap200"] > R3_TOL)
            verdict[f"{k} x {x}"] = dict(R1=bool(r1), R2=bool(r2), R3=bool(r3), ok=bool(r1 and r2 and r3))
    out("   check rule (R1 Brier not worse; R2 classes 0 and 1 within 0.02 of the observed share; R3 gap over bins of 200+ "
        "stacks <= 0.05): " + "; ".join(f"{k} {'PASS' if v['ok'] else 'FAIL'}" + ("" if v["ok"] else
                                     " (" + ", ".join(r for r in ("R1", "R2", "R3") if not v[r]) + ")") for k, v in verdict.items()))
    out("   fits: " + "; ".join(f"{k} " + ", ".join(f"{s} {sum(p['source'] == s for p in pars[k])}" for s in ("class", "all", "none"))
                                for k in order) + "; values raised by the running maximum over x: "
        + ", ".join(f"{k} {raised[k]}" for k in order))
    for k in order[:2]:
        out(f"      {k} slopes b (check fold, x, class): " + "; ".join(f"{p['check_fold']}/{p['x']}/{p['recorded_class']} "
                                                                       f"{p['b']:+.2f}" for p in pars[k] if p["b"] is not None))
    out_dir.mkdir(parents=True, exist_ok=True)
    save = dict(fold=fold, comm=comm, rcls=rcls, rec_t=rec_t, true_aoo=y, xs=np.array(XS))
    for k in order:
        save[f"p_le_{k}"] = P0[k]; save[f"p_recal_{k}"] = P1[k]
    X.save_npz(out_dir / "g3b_recal.npz", save)
    with open(out_dir / "per_stack_g3b.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stack", "community", "fold", "recorded_class", "aoo_true"] +
                   [f"p_recal_aoo_le_{x}_{k}" for k in order for x in XS])
        for i in range(n):
            w.writerow([i, int(comm[i]), int(fold[i]), TCLS[rcls[i]], int(y[i])] + [f"{P1[k][i, j]:.4f}" for k in order for j in range(len(XS))])
    m_out = X.file_md5(out_dir / "g3b_recal.npz")
    rep_out = dict(gates=res, md5=dict(tool=X.file_md5(__file__), input=m5, recal=m_out), verdict=verdict,
                   before={k: {str(x): {q: v for q, v in d.items() if q != "table"} for x, d in ev0[k].items()} for k in order},
                   after={k: {str(x): d for x, d in ev1[k].items()} for k in order},
                   brier_diff={k: {str(x): v for x, v in diffs[k].items()} for k in order}, fits=pars, raised=raised,
                   n=int(n), minutes=round((time.time() - t0) / 60, 2))
    (out_dir / "g3b_report.json").write_text(json.dumps(rep_out, indent=1))
    n_ok = sum(res.values())
    out(f"   written g3b_recal.npz md5 {m_out}, per_stack_g3b.csv, g3b_report.json ({(time.time() - t0) / 60:.1f} min)")
    out(f"stage G3b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; check rule: "
        + ", ".join(f"{k} {'PASS' if v['ok'] else 'FAIL'}" for k, v in verdict.items()))
    return res, dict(P0=P0, P1=P1, ev0=ev0, ev1=ev1, verdict=verdict, pars=pars, diffs=diffs)


# ---------------------------------------------------------------- self-test
def synth_g3(path, seed=3, n=8000, miscal=True):
    """A g3_calibrated.npz and g3_report.json with G3's keys: 20 communities, recorded classes, true AOO and P from
    50 maps; with miscal, the maps' P is too low where nothing is recorded (as G3 found on real_val)."""
    g = np.random.default_rng(seed)
    comm = g.integers(5101, 5121, n)
    rcls = g.choice(7, n, p=[0.25, 0.3, 0.1, 0.1, 0.1, 0.1, 0.05])
    lo = np.array([0, 1, 2, 6, 11, 21, 51]); hi = np.array([0, 1, 5, 10, 20, 50, 120])
    rec_t = np.array([g.integers(lo[c], hi[c] + 1) for c in rcls])
    lam = rec_t * g.gamma(3, 1, n) + g.gamma(1.5, 2.0, n)
    y = np.maximum(g.poisson(lam), rec_t).astype(float)
    lam_m = lam * (1.6 if miscal else 1.0)
    maps = g.poisson(lam_m[:, None], (n, 50))                      # model maps need not hold every record
    fold = GC.make_folds(comm)
    path.mkdir(parents=True)
    arrs = dict(fold=fold, comm=comm, rcls=rcls, rec_t=rec_t, true_aoo=y)
    rep = dict(md5={}, p_le={})
    for k, mm in (("model_run1", maps), ("model_run2", g.poisson(lam_m[:, None], (n, 50))),
                  ("pooled", maps), ("B3", np.maximum(g.poisson(0.5 * lam[:, None], (n, 50)), rec_t[:, None])),
                  ("B5", np.minimum(rec_t, rec_t)[:, None].repeat(1, 1))):
        arrs[f"p_le_{k}"] = np.stack([GC.p_le(mm.astype(float), x) for x in XS], 1)
        pc = GC.p_check(mm.astype(float), y, fold, rcls)
        rep["p_le"][k.replace("_", " ", 1) if k.startswith("model_") else k] = {str(x): {q: pc[x][q] for q in ("brier", "bss", "gap")} for x in XS}
    X.save_npz(path / "g3_calibrated.npz", arrs)
    rep["md5"]["calibrated"] = X.file_md5(path / "g3_calibrated.npz")
    (path / "g3_report.json").write_text(json.dumps(rep))
    return arrs


def self_test():
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    g = np.random.default_rng(1)
    zz = g.normal(0, 1.5, 40000); oo = (g.random(40000) < 1 / (1 + np.exp(-(-0.5 + 1.4 * zz)))).astype(float)
    w = fit_logistic(zz, oo)
    t("U1", abs(w[0] + 0.5) < 0.05 and abs(w[1] - 1.4) < 0.05, f"logistic fit recovers a -0.5, b 1.4 ({w[0]:+.3f}, {w[1]:+.3f})")
    flat = fit_logistic(np.full(500, 0.3), (g.random(500) < 0.2).astype(float))
    t("U2", np.isfinite(flat).all(), f"a flat predictor (one P value) gives a finite fit ({flat[0]:+.2f}, {flat[1]:+.2f})")
    n = 20000
    comm = g.integers(0, 20, n); fold = GC.make_folds(comm); rcls = g.choice(2, n); rec_t = rcls.copy()
    ptrue = g.beta(0.6, 3, n)
    P_mis = np.clip(1 / (1 + np.exp(-(np.log(ptrue / (1 - ptrue)) - 0.6))), 0, 1)        # too low
    Pm = np.repeat(np.round(P_mis * 50)[:, None] / 50, 4, 1)
    y = np.where(g.random(n) < ptrue, 0, 30).astype(float)                             # AOO <= x with chance ptrue, every x
    P1, pars, _ = recalibrate(Pm, y, rec_t, rcls, fold)
    e0, e1 = evaluate(Pm[:, 0], y, fold, rcls, 1), evaluate(P1[:, 0], y, fold, rcls, 1)
    t("U3", e1["brier"] < e0["brier"] and abs(e1["mean_p"] - e1["share"]) < 0.01 and abs(e0["mean_p"] - e0["share"]) > 0.03,
      f"P too low: mean P {e0['mean_p']:.3f} -> {e1['mean_p']:.3f} against {e1['share']:.3f}; Brier {e0['brier']:.4f} -> {e1['brier']:.4f}")
    Pc = np.repeat((g.random((n, 50)) < ptrue[:, None]).mean(1)[:, None], 4, 1)
    P2, pars2, _ = recalibrate(Pc, y, rec_t, rcls, fold)
    e0c, e1c = evaluate(Pc[:, 0], y, fold, rcls, 1), evaluate(P2[:, 0], y, fold, rcls, 1)
    bs = [p["b"] for p in pars2 if p["b"] is not None]
    t("U4", e1c["brier"] < e0c["brier"] + 0.001 and abs(e1c["mean_p"] - e1c["share"]) < 0.01,
      f"P already calibrated: Brier {e0c['brier']:.4f} -> {e1c['brier']:.4f}, slopes {min(bs):.2f}-{max(bs):.2f}")
    rec_b = np.where(g.random(n) < 0.3, 7, 0)
    P3, _, raised = recalibrate(Pc, y, rec_b, rcls, fold)
    t("U5", bool((P3[rec_b > 1, 0] == 0).all() and (P3[rec_b > 5, 1] == 0).all() and (np.diff(P3, axis=1) >= 0).all()),
      f"recorded cells above x give P' = 0 (7 recorded: x 1 and 5); P' non-decreasing in x ({raised} values raised)")
    few = np.where(rcls == 1, 1, 0)
    yf = np.where(((few == 0) & (g.random(n) < 0.3)) | ((few == 1) & (g.random(n) < 0.0002)), 0, 30).astype(float)
    P4, pars4, _ = recalibrate(Pc, yf, rec_t * 0, few, fold)
    srcs = {p["recorded_class"]: p["source"] for p in pars4 if p["x"] == 1}
    t("U6", srcs.get("1") == "all", f"a recorded class with under 5 events uses the fit over all classes ({srcs})")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synth_g3(tmp / "g3")
        lines = []
        r, info = run(tmp / "g3", tmp / "out", out=lines.append)
        t("S1", all(r.values()) and len(r) == 3, f"planted G3 output end to end: {r}")
        v = info["verdict"]; ev0, ev1 = info["ev0"], info["ev1"]
        gain = ev1["model_run1"][5]["brier"] < ev0["model_run1"][5]["brier"]
        t("S2", gain and "model_run1 x 5" in v, f"maps 1.6 times too high: Brier at x 5 {ev0['model_run1'][5]['brier']:.4f} -> "
          f"{ev1['model_run1'][5]['brier']:.4f}; verdicts {[(k, w['ok']) for k, w in v.items()][:4]}")
        stop = []
        for args in ((tmp / "g3", tmp / "out"), (tmp / "g3", tmp / "real_eval_out")):
            try:
                run(*args, out=lambda *a, **k: None); stop.append(False)
            except SystemExit:
                stop.append(True)
        t("S3", all(stop), f"an output folder that holds outputs is refused {stop[0]}; a real_eval path is refused {stop[1]}")
        bad = tmp / "g3bad"; bad.mkdir()
        for f in (tmp / "g3").iterdir():
            (bad / f.name).write_bytes(f.read_bytes())
        zz2 = dict(np.load(bad / "g3_calibrated.npz")); zz2["p_le_B3"] = zz2["p_le_B3"].copy(); zz2["p_le_B3"][0, 3] = 0.5
        np.savez(bad / "g3_calibrated.npz", **zz2)
        rb, _ = run(bad, tmp / "out_bad", out=lambda *a, **k: None)
        t("bug changed_input", rb.get("L1") is False and rb.get("L3") is False, f"a P changed after G3 fails L1 and L3 ({rb})")
        for bug, gate in (("leak", "L2"), ("no_zero", "L3"), ("no_mono", "L3"), ("shift_truth", "L3")):
            env = dict(os.environ, STAGE_G3B_BROKEN=bug)
            code = ("import importlib.util as u; s=u.spec_from_file_location('r', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r,_=m.run(%r, %r, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), str(tmp / "g3"), str(tmp / f"b_{bug}")))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    ap.add_argument("--out", default="../data/stage_g3b_recalibration")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    res, _ = run(a.g3, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if res and all(res.values()) else 1)