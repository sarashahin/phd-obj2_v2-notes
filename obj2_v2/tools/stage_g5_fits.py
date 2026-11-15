#!/usr/bin/env python3
"""
stage_g5_fits.py  (obj2_v2, stage G5 part G5a, 11 Nov 2026; G5 rules signed off by Sara on 11 Nov)

Question: G5-4 of the signed G5 rules (claude/obj2_v2_G5_rules_draft.md). Refit, on all 20 real_val communities, the
AOO and EOO level correction with its conformal widening (G3) and the logistic recalibration of P(AOO <= x) (G3b), and
write them, md5-logged, while real_eval has not been exported. G5d applies these fits to real_eval unchanged, with the
apply code of this tool.

Rules (G5-4; the G3 and G3b rules unchanged)
    Input: G3's g3_calibrated.npz and G3b's g3b_recal.npz (md5 = their reports) and nothing else. They hold, per frozen
    real_val stack, the target-year AOO and EOO of every map of every method, the true AOO and EOO, the cells recorded
    at the target year, the recorded class, the records-over-6-years bin, the community, the G3 fold, G3's P(AOO <= x)
    and G3b's recalibrated P.
    Correction: per method and metric, the grouping option G3 chose (g3_report.json "chosen"); scale and Q by G3's
    fit_params on all stacks of a fitting group (all communities); a fitting group with fewer than 30 stacks uses its
    recorded class (option 2), and a recorded class with fewer than 30 uses all stacks. A value is stored for every
    possible fitting group (7, or 7 x 5), so a group first seen in real_eval still has a fit.
    Applied as in G3: interval [p5 - Q, p95 + Q] of the scaled maps; AOO lower end at least the cells recorded at the
    target year, EOO lower end at least 0; the upper end raised to the floor if below it.
    Recalibration: G3b's logistic fit per x and recorded class on all real_val stacks where AOO <= x is possible. Its
    input is G3b's input, G3's cross-fitted P, whose scale came from other communities, as real_eval's will. A class
    needs 30 stacks and 5 of each outcome, else the all-class fit, else P unchanged; P = 0 where the recorded cells
    exceed x; running maximum over x.
    Option B (Sara, 11 Nov): the recalibrated P at x 1 and 5, G3's P (from the corrected maps) at x 10 and 20, then the
    running maximum over x.
    Brier reference for G5-5: the real_val share with true AOO <= x per recorded class (all communities; the share over
    all stacks for a class with no stacks).

Gates
    A0  real_eval not exported: <data>/real_eval does not exist, before and after the fits are written; if it exists,
        nothing is written
    A1  inputs: g3_calibrated.npz and g3b_recal.npz md5 = their reports; G3b was made from this G3 file; the arrays
        both files hold are equal; every predicted EOO is a whole number of half cells (so its float32 copy is exact);
        a grouping option is read from g3_report.json for every method and metric
    A2  replica: this tool's fit and apply code, run on G3's folds (fit on one fold, apply to the other), gives G3's
        stored corrected mean, interval ends, scale and Q and P(AOO <= x), and G3b's recalibrated P, for every method
        and metric (largest difference below 1e-6)
    A3  fits on all stacks: every fitting group's stack count equals its count over all real_val stacks (or its
        recorded class's, or all stacks); applied to real_val, no interval is empty, AOO lower ends are at least the
        recorded cells, P is in [0, 1], 0 where the recorded cells exceed x, never falls with x, and option B is built
        as stated
    A4  the fits file reloads equal to the fits in memory; its md5 and the time written are logged
    real_eval paths are refused; the output folder must be new.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, about a minute)
    python3 ../tools/stage_g5_fits.py --self-test | tail -1
    python3 -u ../tools/stage_g5_fits.py --g3 ../data/stage_g3_calibration --g3b ../data/stage_g3b_recalibration \\
        --data ../data --out ../data/stage_g5_fits > ../logs/stage_g5_fits.txt 2>&1
Needs numpy and stage_g_calibrate.py, stage_g_recalibrate.py (with the stage tools they load) in the same folder.
"""
import argparse, importlib.util, json, os, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g5_fits.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


GC = _load("stage_g_calibrate")
GR = _load("stage_g_recalibrate")
X, D = GC.X, GC.D
TCLS, BINS, XS = GC.TCLS, GC.BINS, GC.XS
NC, NB, NX = len(TCLS), len(BINS), len(XS)
TOL = 1e-6
KINDS = ("aoo", "eoo")
BROKEN = os.environ.get("STAGE_G5A_BROKEN", "")  # self-test only: plants one bug so its gate must fail


# ---------------------------------------------------------------- correction (G3)
def group_of(rcls, rbin, option):
    return np.asarray(rcls) if option == 1 else np.asarray(rcls) * NB + np.asarray(rbin)


def fit_correction(pred, y, rcls, rbin, option, mask):
    """Scale and Q for every possible fitting group, from the stacks in mask; G3's fallback (under 30 fit stacks: the
    recorded class for option 2, then all stacks). src 0 own group, 1 recorded class, 2 all stacks."""
    fit = np.nonzero(mask)[0]
    fg = group_of(rcls, rbin, option)
    ng = NC if option == 1 else NC * NB
    scale, q = np.zeros(ng), np.zeros(ng)
    src, nfit = np.zeros(ng, np.int8), np.zeros(ng, np.int64)
    cache = {}
    for g in range(ng):
        idx, key, s_ = fit[fg[fit] == g], ("g", g), 0
        if len(idx) < GC.N_FIT_MIN:
            pg = -1 if (option == 1 or BROKEN == "parent_skip") else g // NB
            idx, key, s_ = (fit, ("all",), 2) if pg < 0 else (fit[rcls[fit] == pg], ("p", pg), 1)
            if len(idx) < GC.N_FIT_MIN:
                idx, key, s_ = fit, ("all",), 2
        if key not in cache:
            cache[key] = GC.fit_params(pred, y, idx)
        scale[g], q[g] = cache[key]
        src[g], nfit[g] = s_, len(idx)
    return dict(option=int(option), scale=scale, q=q, src=src, nfit=nfit)


def apply_correction(pred, rec_t, rcls, rbin, tab, kind):
    """G3's corrected mean, interval and scaled maps, with a fitted table."""
    g = group_of(rcls, rbin, tab["option"])
    s, q = tab["scale"][g], tab["q"][g]
    sc = s[:, None] * pred
    lo, hi = np.percentile(sc, GC.QLO, axis=1), np.percentile(sc, GC.QHI, axis=1)
    lo_c, hi_c = lo - q, hi + q
    floor = np.maximum(rec_t, 0) if kind == "aoo" else np.zeros(len(pred))
    if BROKEN == "no_floor":
        floor = np.full(len(pred), -np.inf)
    return dict(mean=sc.mean(1), L=np.maximum(lo_c, floor), U=np.maximum(hi_c, floor), scaled=sc, Lraw=lo_c, Uraw=hi_c,
                scale=s, q=q)


# ---------------------------------------------------------------- recalibration (G3b) and option B
def fit_recal(P, y, rec_t, rcls, mask):
    """Logistic fit per x and recorded class from the stacks in mask (G3b's rule). src 0 class fit, 1 all-class fit,
    2 none (P unchanged)."""
    W = np.full((NX, NC, 2), np.nan); W_all = np.full((NX, 2), np.nan)
    src = np.full((NX, NC), 2, np.int8)
    ncls, ev = np.zeros((NX, NC), np.int64), np.zeros((NX, NC), np.int64)
    nall = np.zeros(NX, np.int64)
    for j, x in enumerate(XS):
        possible = rec_t <= x
        o = (y <= x).astype(float)
        z = GR.logit_clip(P[:, j])
        allfit = mask & possible
        ok = lambda m: m.sum() >= GR.N_MIN and o[m].sum() >= GR.E_MIN and (1 - o[m]).sum() >= GR.E_MIN
        w_all = GR.fit_logistic(z[allfit], o[allfit]) if ok(allfit) else None
        nall[j] = allfit.sum()
        if w_all is not None:
            W_all[j] = w_all
        for g in range(NC):
            m = allfit & (rcls == g)
            ncls[j, g], ev[j, g] = m.sum(), o[m].sum()
            if ok(m):
                W[j, g], src[j, g] = GR.fit_logistic(z[m], o[m]), 0
            elif w_all is not None:
                W[j, g], src[j, g] = w_all, 1
    return dict(w=W, w_all=W_all, src=src, nclass=ncls, nall=nall, events=ev)


def apply_recal(P, rec_t, rcls, tab):
    out = np.array(P, float).copy()
    for j, x in enumerate(XS):
        possible = rec_t <= x
        z = GR.logit_clip(P[:, j])
        for g in range(NC):
            t = possible & (rcls == g)
            if not t.any():
                continue
            w, s_ = tab["w"][j, g], tab["src"][j, g]
            if BROKEN == "recal_all" and np.isfinite(tab["w_all"][j]).all():
                w, s_ = tab["w_all"][j], 1
            if s_ == 2:
                continue
            out[t, j] = 1 / (1 + np.exp(-(w[0] + w[1] * z[t])))
        out[~possible, j] = 0.0
    return np.maximum.accumulate(out, axis=1)


def option_b(p_g3, p_rec):
    comb = np.concatenate([p_rec[:, :2], p_g3[:, 2:]], 1)
    if BROKEN == "optb_order":
        comb = np.concatenate([p_g3[:, :2], p_rec[:, 2:]], 1)
    return np.maximum.accumulate(comb, axis=1)


def base_rates(y, rcls):
    br, ball = np.zeros((NX, NC)), np.zeros(NX)
    for j, x in enumerate(XS):
        o = y <= x
        ball[j] = o.mean()
        for g in range(NC):
            m = rcls == g
            br[j, g] = o[m].mean() if m.any() else ball[j]
    return br, ball, np.bincount(rcls, minlength=NC)


def apply_all(fits, kk, aoo, eoo, rec_t, rcls, rbin):
    """Everything G5d needs for one method: corrected AOO and EOO, G3's P, the recalibrated P and option B."""
    ca = apply_correction(np.asarray(aoo, float), rec_t, rcls, rbin, fits["corr"][(kk, "aoo")], "aoo")
    ce = apply_correction(np.asarray(eoo, float), rec_t, rcls, rbin, fits["corr"][(kk, "eoo")], "eoo")
    p_g3 = np.stack([GC.p_le(ca["scaled"], x) for x in XS], 1)
    p_rec = apply_recal(p_g3, rec_t, rcls, fits["recal"][kk])
    return dict(aoo=ca, eoo=ce, p_g3=p_g3, p_recal=p_rec, p_b=option_b(p_g3, p_rec))


# ---------------------------------------------------------------- fits file
def save_fits(path, fits):
    a = dict(methods=np.array(fits["methods"]), xs=np.array(XS), base_rate=fits["base_rate"],
             base_rate_all=fits["base_rate_all"], base_n=fits["base_n"], communities=fits["communities"],
             g3_md5=np.array(fits["g3_md5"]), g3b_md5=np.array(fits["g3b_md5"]))
    for (kk, kind), t in fits["corr"].items():
        for q in ("option", "scale", "q", "src", "nfit"):
            a[f"corr__{kk}__{kind}__{q}"] = np.asarray(t[q])
    for kk, t in fits["recal"].items():
        for q in ("w", "w_all", "src", "nclass", "nall", "events"):
            a[f"recal__{kk}__{q}"] = np.asarray(t[q])
    X.save_npz(path, a)


def load_fits(path):
    z = np.load(path)
    methods = [str(m) for m in z["methods"]]
    corr = {(kk, kind): dict(option=int(z[f"corr__{kk}__{kind}__option"]), **{q: z[f"corr__{kk}__{kind}__{q}"] for q in
                                                                               ("scale", "q", "src", "nfit")})
            for kk in methods for kind in KINDS}
    recal = {kk: {q: z[f"recal__{kk}__{q}"] for q in ("w", "w_all", "src", "nclass", "nall", "events")} for kk in methods}
    return dict(methods=methods, corr=corr, recal=recal, base_rate=z["base_rate"], base_rate_all=z["base_rate_all"],
                base_n=z["base_n"], communities=z["communities"], g3_md5=str(z["g3_md5"]), g3b_md5=str(z["g3b_md5"]))


def same_fits(a, b):
    eq = lambda u, v: np.array_equal(np.asarray(u), np.asarray(v), equal_nan=np.asarray(u).dtype.kind == "f")
    if a["methods"] != b["methods"] or a["g3_md5"] != b["g3_md5"] or a["g3b_md5"] != b["g3b_md5"]:
        return False
    if not all(eq(a[k], b[k]) for k in ("base_rate", "base_rate_all", "base_n", "communities")):
        return False
    return all(all(eq(a["corr"][key][q], b["corr"][key][q]) for q in ("option", "scale", "q", "src", "nfit")) for key in a["corr"]) \
        and all(all(eq(a["recal"][kk][q], b["recal"][kk][q]) for q in a["recal"][kk]) for kk in a["recal"])


def report_name(kk):
    return kk.replace("_", " ", 1) if kk.startswith("model_") else kk


# ---------------------------------------------------------------- run
def run(g3_dir, g3b_dir, data_dir, out_dir, out=print):
    g3_dir, g3b_dir, data_dir, out_dir = Path(g3_dir), Path(g3b_dir), Path(data_dir), Path(out_dir)
    for p in (g3_dir, g3b_dir, out_dir):
        if "real_eval" in str(p):
            sys.exit("STOP: the G5 fits come from real_val only; a real_eval path is refused")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    res, t0 = {}, time.time()

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    out(f"stage_g5_fits.py md5 {X.file_md5(__file__)}; stage_g_calibrate.py {X.file_md5(HERE / 'stage_g_calibrate.py')}; "
        f"stage_g_recalibrate.py {X.file_md5(HERE / 'stage_g_recalibrate.py')}")
    ev = data_dir / "real_eval"
    if ev.exists():
        gate("A0", False, f"real_eval not exported: {ev} exists, so the fits are no longer written before the export; nothing written")
        return res, None

    # ---- A1 inputs
    rep3 = json.loads((g3_dir / "g3_report.json").read_text())
    rep3b = json.loads((g3b_dir / "g3b_report.json").read_text())
    f3, f3b = g3_dir / "g3_calibrated.npz", g3b_dir / "g3b_recal.npz"
    m3, m3b = X.file_md5(f3), X.file_md5(f3b)
    z, zb = np.load(f3), np.load(f3b)
    out(f"   g3_calibrated.npz md5 {m3} (report {rep3['md5']['calibrated']}); g3b_recal.npz md5 {m3b} (report {rep3b['md5']['recal']}, "
        f"made from {rep3b['md5']['input']})")
    keys = sorted(k[5:] for k in z.files if k.startswith("p_le_"))
    order = [k for k in keys if k.startswith("model_")] + [k for k in ("pooled", "B3", "B4", "B5") if k in keys]
    fold, comm, rcls, rbin, rec_t = z["fold"], z["comm"], z["rcls"], z["rbin"], z["rec_t"]
    ya, ye = z["true_aoo"].astype(float), z["true_eoo"].astype(float)
    n = len(ya)
    shared = all(np.array_equal(z[k], zb[k]) for k in ("fold", "comm", "rcls", "rec_t", "true_aoo")) and \
        all(f"p_le_{k}" in zb.files and np.array_equal(z[f"p_le_{k}"], zb[f"p_le_{k}"]) for k in order) and \
        all(f"p_recal_{k}" in zb.files for k in order)
    half = all(bool((np.abs(z[f"eoo_{k}"].astype(float) * 2 - np.round(z[f"eoo_{k}"].astype(float) * 2)) == 0).all()) for k in order)
    opts, miss = {}, []
    for k in order:
        for kind in KINDS:
            c = rep3.get("chosen", {}).get(f"{report_name(k)} {kind}")
            if c is None or c.get("option") not in (1, 2):
                miss.append(f"{k} {kind}")
            else:
                opts[(k, kind)] = int(c["option"])
    cu = np.unique(comm)
    a1 = m3 == rep3["md5"]["calibrated"] and m3b == rep3b["md5"]["recal"] and rep3b["md5"]["input"] == m3 and shared and half \
        and not miss and len(order) > 0
    gate("A1", a1, f"inputs as G3 and G3b logged them, G3b made from this G3 file {rep3b['md5']['input'] == m3}; shared arrays equal "
         f"{shared}; predicted EOO in whole half cells {half}; options read for {len(opts)} of {2 * len(order)} method-metrics"
         + (f" (missing: {', '.join(miss)})" if miss else "") + f"; {n} stacks, {len(cu)} communities, methods {order}")
    if not a1:
        return res, None
    out("   grouping options (g3_report.json chosen; 1 = recorded class, 2 = recorded class x records bin): "
        + ", ".join(f"{report_name(k)} {kind.upper()} {opts[(k, kind)]}" for k in order for kind in KINDS))

    # ---- A2 replica on G3's folds
    worst, wtxt, fold_tabs = 0.0, [], {}
    for k in order:
        for kind in KINDS:
            pred = z[f"{kind}_{k}"].astype(float)
            y = ya if kind == "aoo" else ye
            got = {q: np.zeros(n) for q in ("mean", "L", "U")}
            par, pl = np.zeros((n, 2)), np.zeros((n, NX))
            for c in (0, 1):
                tab = fit_correction(pred, y, rcls, rbin, opts[(k, kind)], fold == 1 - c)
                fold_tabs[(k, kind, c)] = tab
                i = fold == c
                ap = apply_correction(pred[i], rec_t[i], rcls[i], rbin[i], tab, kind)
                for q in got:
                    got[q][i] = ap[q]
                par[i] = np.column_stack([ap["scale"], ap["q"]])
                if kind == "aoo":
                    pl[i] = np.stack([GC.p_le(ap["scaled"], x) for x in XS], 1)
            d = max(max(float(np.abs(got[q] - z[f"{kind}_{q}_{k}"]).max()) for q in got),
                    float(np.abs(par - z[f"{kind}_scale_q_{k}"]).max()))
            if kind == "aoo":
                d = max(d, float(np.abs(pl - z[f"p_le_{k}"]).max()))
            worst = max(worst, d); wtxt.append(f"{report_name(k)} {kind.upper()} {d:.1e}")
        P0 = zb[f"p_le_{k}"].astype(float)
        P1 = np.zeros((n, NX))
        for c in (0, 1):
            tabr = fit_recal(P0, ya, rec_t, rcls, fold == 1 - c)
            i = fold == c
            P1[i] = apply_recal(P0[i], rec_t[i], rcls[i], tabr)
        d = float(np.abs(P1 - zb[f"p_recal_{k}"]).max())
        worst = max(worst, d); wtxt.append(f"{report_name(k)} P' {d:.1e}")
    gate("A2", worst < TOL, f"this tool's fit and apply code on G3's folds gives G3's corrected mean, interval ends, scale, Q and "
         f"P(AOO <= x) and G3b's recalibrated P (largest difference {worst:.1e}; {'; '.join(wtxt)})")

    # ---- fits on all stacks
    use = np.ones(n, bool) if BROKEN != "subset_fit" else fold == 0
    corr, recal = {}, {}
    for k in order:
        for kind in KINDS:
            corr[(k, kind)] = fit_correction(z[f"{kind}_{k}"].astype(float), ya if kind == "aoo" else ye, rcls, rbin,
                                             opts[(k, kind)], use)
        recal[k] = fit_recal(zb[f"p_le_{k}"].astype(float), ya, rec_t, rcls, use)
    br, ball, bn = base_rates(ya, rcls)
    fits = dict(methods=list(order), corr=corr, recal=recal, base_rate=br, base_rate_all=ball, base_n=bn, communities=cu,
                g3_md5=m3, g3b_md5=m3b)

    # ---- A3 counts and structure on real_val
    cnt_ok = True
    for (k, kind), t in corr.items():
        fg = group_of(rcls, rbin, t["option"])
        for g in range(len(t["scale"])):
            exp = {0: int((fg == g).sum()), 1: int((rcls == g // NB).sum()), 2: n}[int(t["src"][g])]
            cnt_ok &= int(t["nfit"][g]) == exp and np.isfinite(t["scale"][g]) and np.isfinite(t["q"][g])
    for k, t in recal.items():
        for j, x in enumerate(XS):
            pos = rec_t <= x
            cnt_ok &= int(t["nall"][j]) == int(pos.sum()) and all(int(t["nclass"][j, g]) == int((pos & (rcls == g)).sum()) for g in range(NC))
    struct, insample = True, {}
    for k in order:
        r = apply_all(fits, k, z[f"aoo_{k}"], z[f"eoo_{k}"], rec_t, rcls, rbin)
        insample[k] = r
        for kind in KINDS:
            struct &= bool((r[kind]["U"] >= r[kind]["L"]).all())
        struct &= bool((r["aoo"]["L"] >= rec_t).all())
        for P in (r["p_recal"], r["p_b"]):
            struct &= bool(((P >= 0) & (P <= 1)).all()) and bool((np.diff(P, axis=1) >= 0).all())
        pb, pr, pg = r["p_b"], r["p_recal"], r["p_g3"]
        struct &= bool((r["p_recal"][rec_t[:, None] > np.array(XS)[None, :]] == 0).all())
        struct &= bool((pb[rec_t > XS[0], 0] == 0).all() and (pb[rec_t > XS[1], 1] == 0).all())
        struct &= np.array_equal(pb[:, 0], pr[:, 0]) and np.array_equal(pb[:, 1], pr[:, 1]) \
            and np.array_equal(pb[:, 2], np.maximum(pr[:, 1], pg[:, 2])) and np.array_equal(pb[:, 3], np.maximum(pb[:, 2], pg[:, 3]))
    gate("A3", cnt_ok and struct, f"every fitting group's stack count is its count over all {n} real_val stacks (or its recorded "
         f"class's, or all) {cnt_ok}; applied to real_val: intervals not empty, AOO lower end >= recorded cells, P in [0, 1], 0 "
         f"where the recorded cells exceed x, non-decreasing in x, option B as stated {struct}")

    # ---- reports (not tests)
    srcname = {0: "own", 1: "class", 2: "all"}
    out("   correction fits on all stacks (scale / Q / fit source / stacks):")
    for k in order:
        for kind in KINDS:
            t = corr[(k, kind)]
            if t["option"] == 1:
                out(f"      {report_name(k):<14} {kind.upper()} option 1: " + "; ".join(
                    f"{TCLS[g]} {t['scale'][g]:.3f}/{t['q'][g]:.2f}/{srcname[int(t['src'][g])]}/{t['nfit'][g]}" for g in range(NC)))
            else:
                cs = {s: int((t["src"] == s).sum()) for s in (0, 1, 2)}
                out(f"      {report_name(k):<14} {kind.upper()} option 2: {len(t['scale'])} groups, own fit {cs[0]}, recorded class {cs[1]}, "
                    f"all {cs[2]}; scale {t['scale'].min():.3f}-{t['scale'].max():.3f}, Q {t['q'].min():.2f}-{t['q'].max():.2f}")
    out("   scale on all stacks against G3's two fold fits, model seeds, by recorded class (all | fold 0 fit, fold 1 fit):")
    for k in order[:2]:
        for kind in KINDS:
            t = corr[(k, kind)]
            if t["option"] != 1:
                continue
            out(f"      {report_name(k):<14} {kind.upper()} " + "; ".join(
                f"{TCLS[g]} {t['scale'][g]:.3f} | {fold_tabs[(k, kind, 1)]['scale'][g]:.3f}, {fold_tabs[(k, kind, 0)]['scale'][g]:.3f}"
                for g in range(NC)))
    out("   recalibration fits on all stacks, model seeds, x 1 and 5 (class: a / b / source / stacks / AOO <= x):")
    for k in order[:2]:
        t = recal[k]
        for j in (0, 1):
            out(f"      {report_name(k):<14} x {XS[j]}: " + "; ".join(
                (f"{TCLS[g]} {t['w'][j, g, 0]:+.2f}/{t['w'][j, g, 1]:+.2f}/{('class', 'all')[int(t['src'][j, g])]}/{t['nclass'][j, g]}/{t['events'][j, g]}"
                 if t["src"][j, g] < 2 else f"{TCLS[g]} none/{t['nclass'][j, g]}") for g in range(NC) if t["nclass"][j, g] > 0)
                + " (other classes cannot have AOO <= x: P = 0)")
    out("   Brier reference for G5-5 (real_val share with true AOO <= x per recorded class): " + "; ".join(
        f"x {x}: " + ", ".join(f"{TCLS[g]} {br[j, g]:.3f}" for g in range(NC)) for j, x in enumerate(XS)))
    out("   in-sample check on real_val (the fitted stacks themselves, not a test; ratio / coverage by recorded class):")
    for k in order:
        r = insample[k]
        for kind in KINDS:
            y = ya if kind == "aoo" else ye
            cov = (y >= r[kind]["L"]) & (y <= r[kind]["U"])
            out(f"      {report_name(k):<14} {kind.upper()} " + "; ".join(
                f"{TCLS[g]} {r[kind]['mean'][rcls == g].mean() / max(y[rcls == g].mean(), 1e-12):.2f}/{cov[rcls == g].mean():.2f}"
                for g in range(NC) if (rcls == g).any()))
    out("   option B mean P against the share with true AOO <= x, model seeds, recorded classes 0 and 1 (in-sample):")
    for k in order[:2]:
        pb = insample[k]["p_b"]
        out(f"      {report_name(k):<14} " + "; ".join(f"x {x}: " + ", ".join(
            f"{TCLS[g]} {pb[rcls == g, j].mean():.3f} vs {(ya[rcls == g] <= x).mean():.3f}" for g in (0, 1) if (rcls == g).any())
            for j, x in enumerate(XS)))

    # ---- write and A4
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "g5_fits.npz"
    save_fits(fp, fits)
    mf = X.file_md5(fp)
    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    back = load_fits(fp)
    gate("A4", same_fits(fits, back) and back["g3_md5"] == m3, f"g5_fits.npz reloads equal to the fits in memory; md5 {mf}; written {when}")
    gate("A0", not ev.exists(), f"real_eval not exported: {ev} does not exist, before and after the fits were written")
    rep = dict(gates=res, md5=dict(tool=X.file_md5(__file__), calibrate_tool=X.file_md5(HERE / "stage_g_calibrate.py"),
                                   recalibrate_tool=X.file_md5(HERE / "stage_g_recalibrate.py"), g3=m3, g3b=m3b, fits=mf),
               written_utc=when, options={f"{report_name(k)} {kind}": v for (k, kind), v in opts.items()},
               n=int(n), communities=[int(c) for c in cu], replica_largest_difference=worst,
               sources={f"{report_name(k)} {kind}": {srcname[s]: int((t["src"] == s).sum()) for s in (0, 1, 2)} for (k, kind), t in corr.items()},
               recal_sources={report_name(k): {("class", "all", "none")[s]: int((t["src"] == s).sum()) for s in (0, 1, 2)} for k, t in recal.items()},
               base_rate={str(x): {TCLS[g]: float(br[j, g]) for g in range(NC)} for j, x in enumerate(XS)},
               minutes=round((time.time() - t0) / 60, 2))
    (out_dir / "g5_fits_report.json").write_text(json.dumps(rep, indent=1))
    n_ok = sum(res.values())
    out(f"   written g5_fits.npz md5 {mf}, g5_fits_report.json ({(time.time() - t0) / 60:.1f} min)")
    out(f"stage G5a: gates {n_ok} PASS, {len(res) - n_ok} FAIL; fits md5 {mf}")
    return res, dict(fits=fits, md5=mf, insample=insample, opts=opts)


# ---------------------------------------------------------------- self-test
FORCE2 = {("model run2", "eoo"), ("B3", "aoo"), ("B5", "aoo"), ("B5", "eoo")}


def synth_g3(path, seed=4, n=6000):
    """A g3_calibrated.npz and g3_report.json with G3's keys, made with G3's own crossfit and p_check: 20 communities, the
    7 recorded classes (51+ rare, so its groups fall back to all stacks), option 2 for four method-metrics (as on
    real_val), predicted EOO in half cells with hull-like 1e-13 noise before the float32 copy."""
    g = np.random.default_rng(seed)
    comm = g.integers(8001, 8021, n)
    cls = g.choice(NC, n, p=[0.3, 0.25, 0.15, 0.12, 0.1, 0.076, 0.004])
    lo, hi = np.array([0, 1, 2, 6, 11, 21, 51]), np.array([0, 1, 5, 10, 20, 50, 120])
    rec_t = np.array([g.integers(lo[c], hi[c] + 1) for c in cls])
    rcls = D.tidx(rec_t)
    rbin = D.bidx(rec_t + g.poisson(1 + 0.5 * rec_t))
    lam = rec_t * g.gamma(3, 1, n) + g.gamma(1.5, 2.0, n)
    ya = np.maximum(g.poisson(lam), rec_t).astype(float)
    ye = ya + 0.5 * g.poisson(2 * ya) + 1e-13 * g.standard_normal(n) * (ya > 0)
    mk = lambda f: g.poisson(lam[:, None] * f * g.gamma(4, 0.25, (n, 1)), (n, 50))
    am = {"model run1": mk(1.4), "model run2": mk(1.4), "B3": mk(0.6), "B4": mk(1.0), "B5": rec_t[:, None]}
    em = {k: v + 0.5 * g.poisson(2 * v) + 1e-13 * g.standard_normal(v.shape) * (v > 0) for k, v in am.items()}  # empty map: EOO 0 exactly
    am["pooled"] = np.concatenate([am["model run1"][:, :25], am["model run2"][:, :25]], 1)
    em["pooled"] = np.concatenate([em["model run1"][:, :25], em["model run2"][:, :25]], 1)
    order = ["model run1", "model run2", "pooled", "B3", "B4", "B5"]
    fold = GC.make_folds(comm)
    opt = {1: (rcls, np.full(n, -1)), 2: (rcls * NB + rbin, rcls)}
    save = dict(fold=fold, comm=comm, rcls=rcls, rbin=rbin, rec_t=rec_t, true_class=D.tidx(ya.astype(int)), true_aoo=ya, true_eoo=ye)
    chosen, pl = {}, {}
    for k in order:
        kk = k.replace(" ", "_")
        save[f"aoo_{kk}"] = am[k].astype(np.uint16); save[f"eoo_{kk}"] = em[k].astype(np.float32)
        for kind, pred, y in (("aoo", am[k].astype(float), ya), ("eoo", em[k], ye)):
            o_ = 2 if (k, kind) in FORCE2 else 1
            cf = GC.crossfit(pred, y, rec_t, fold, *opt[o_], kind)
            chosen[f"{k} {kind}"] = dict(option=o_, ok=True, bad=[], opt1_ok=o_ == 1, opt2_ok=True)
            for q in ("mean", "L", "U"):
                save[f"{kind}_{q}_{kk}"] = cf[q]
            save[f"{kind}_scale_q_{kk}"] = cf["par"]
            if kind == "aoo":
                save[f"p_le_{kk}"] = np.stack([GC.p_le(cf["scaled"], x) for x in XS], 1)
                pc = GC.p_check(cf["scaled"], ya, fold, rcls)
                pl[k] = {str(x): {q: pc[x][q] for q in ("brier", "bss", "gap")} for x in XS}
                floor_share = float((cf["L"] > cf["Lraw"]).mean())
                save.setdefault("_floor", []).append(floor_share)
    floors = save.pop("_floor")
    path.mkdir(parents=True)
    X.save_npz(path / "g3_calibrated.npz", save)
    rep = dict(md5=dict(calibrated=X.file_md5(path / "g3_calibrated.npz")), chosen=chosen, p_le=pl, n=n)
    (path / "g3_report.json").write_text(json.dumps(rep))
    return save, floors


def copy_dir(a, b):
    b.mkdir(parents=True)
    for f in a.iterdir():
        (b / f.name).write_bytes(f.read_bytes())


def self_test():
    quiet = lambda *a, **k: None
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        s, floors = synth_g3(tmp / "g3")
        n = len(s["true_aoo"])
        fold, rcls, rbin, rec_t, ya = s["fold"], s["rcls"], s["rbin"], s["rec_t"], s["true_aoo"]
        # U1 correction: cross-fit with this tool's code = G3's crossfit, both options, small groups
        pred = s["aoo_B3"].astype(float)
        d1, srcs = 0.0, set()
        for o_ in (1, 2):
            fg, pa = (rcls, np.full(n, -1)) if o_ == 1 else (rcls * NB + rbin, rcls)
            cf = GC.crossfit(pred, ya, rec_t, fold, fg, pa, "aoo")
            for c in (0, 1):
                tab = fit_correction(pred, ya, rcls, rbin, o_, fold == 1 - c)
                srcs |= set(int(v) for v in tab["src"])
                i = fold == c
                ap = apply_correction(pred[i], rec_t[i], rcls[i], rbin[i], tab, "aoo")
                d1 = max(d1, *(float(np.abs(ap[q] - cf[q][i]).max()) for q in ("mean", "L", "U")),
                         float(np.abs(np.column_stack([ap["scale"], ap["q"]]) - cf["par"][i]).max()))
        t("U1", d1 == 0.0 and srcs == {0, 1, 2}, f"correction cross-fitted with this tool = G3's crossfit, options 1 and 2 "
          f"(largest difference {d1:.1e}); own-group, recorded-class and all-stack fallbacks all used {sorted(srcs)}")
        # U2 recalibration: cross-fit with this tool's code = G3b's recalibrate, incl. a class with too few events
        P0 = s["p_le_model_run1"].astype(float)
        yr = ya.copy(); yr[rcls == 1] = 200.0                  # recorded class 1 never has AOO <= x: no class fit
        ref, pars, _ = GR.recalibrate(P0, yr, rec_t, rcls, fold)
        mine = np.zeros_like(ref); fb = []
        for c in (0, 1):
            tabr = fit_recal(P0, yr, rec_t, rcls, fold == 1 - c)
            fb.append(int(tabr["src"][0, 0]) == 0 and int(tabr["src"][0, 1]) == 1)
            i = fold == c
            mine[i] = apply_recal(P0[i], rec_t[i], rcls[i], tabr)
        d2 = float(np.abs(mine - ref).max())
        few = fit_recal(P0, yr, rec_t, rcls, np.arange(n) < 10)
        Pz = P0.copy(); Pz[rec_t[:, None] > np.array(XS)[None, :]] = 0
        none_ok = bool((few["src"] == 2).all()) and np.array_equal(apply_recal(P0, rec_t, rcls, few), np.maximum.accumulate(Pz, axis=1))
        t("U2", d2 == 0.0 and all(fb) and none_ok, f"recalibration cross-fitted with this tool = G3b's recalibrate (largest difference "
          f"{d2:.1e}); at x 1 class 0 has its own fit and class 1 (no events) the all-class fit {all(fb)}; with 10 fit stacks "
          f"no fit, P kept apart from the zeros and the running maximum {none_ok}")
        # U3 fit on all stacks = fit_params on the group's own stacks, its recorded class, or all stacks
        tab = fit_correction(pred, ya, rcls, rbin, 2, np.ones(n, bool))
        fg = rcls * NB + rbin
        ok3 = True
        for g in range(NC * NB):
            idx = {0: np.nonzero(fg == g)[0], 1: np.nonzero(rcls == g // NB)[0], 2: np.arange(n)}[int(tab["src"][g])]
            ok3 &= (tab["scale"][g], tab["q"][g]) == GC.fit_params(pred, ya, idx) and len(idx) == tab["nfit"][g]
            ok3 &= (tab["src"][g] == 0) == ((fg == g).sum() >= GC.N_FIT_MIN)
        t("U3", ok3, f"fits on all stacks = G3's fit_params on each group's own stacks, else its recorded class, else all "
          f"({', '.join(f'{v} {int((tab['src'] == v).sum())}' for v in (0, 1, 2))} of {NC * NB} groups)")
        # U4 option B by hand
        pg = np.array([[0.10, 0.20, 0.50, 0.90], [0.0, 0.0, 0.30, 0.40], [0.2, 0.3, 0.1, 0.2]])
        pr = np.array([[0.30, 0.60, 0.99, 0.99], [0.0, 0.0, 0.0, 0.0], [0.25, 0.35, 0.9, 0.9]])
        exp = np.array([[0.30, 0.60, 0.60, 0.90], [0.0, 0.0, 0.30, 0.40], [0.25, 0.35, 0.35, 0.35]])
        t("U4", np.allclose(option_b(pg, pr), exp), "option B by hand: recalibrated P at x 1 and 5, G3's P at x 10 and 20, then "
          "the running maximum (a row whose G3 P(AOO <= 10) is below its P(AOO <= 5) is raised)")
        # U5 Brier reference
        br, ball, bn = base_rates(ya, rcls)
        y0 = ya.copy(); r0 = rcls.copy(); r0[r0 == 6] = 5
        br0, ball0, _ = base_rates(y0, r0)
        t("U5", np.isclose(br[1, 2], (ya[rcls == 2] <= 5).mean()) and np.allclose(br0[:, 6], ball0),
          "Brier reference = share with true AOO <= x per recorded class; a class with no stacks takes the share over all stacks")
        # S1 end to end
        GR.run(tmp / "g3", tmp / "g3b", out=quiet)
        (tmp / "data").mkdir()
        lines = []
        r, info = run(tmp / "g3", tmp / "g3b", tmp / "data", tmp / "out", out=lines.append)
        t("S1", all(r.values()) and set(r) == {"A0", "A1", "A2", "A3", "A4"} and min(floors) >= 0 and max(floors) > 0,
          f"planted G3 and G3b outputs end to end: {r}; AOO lower ends raised to the recorded cells in up to {max(floors):.3f} of stacks")
        opts = info["opts"]
        t("S2", all(opts[(k.replace(' ', '_'), kind)] == 2 for k, kind in FORCE2) and sum(v == 2 for v in opts.values()) == len(FORCE2),
          f"options read from g3_report.json: option 2 for exactly {sorted(FORCE2)}")
        # S3 fits file: apply from the file = apply from memory; a second run gives the same md5
        back = load_fits(tmp / "out" / "g5_fits.npz")
        a_ = apply_all(back, "model_run1", s["aoo_model_run1"], s["eoo_model_run1"], rec_t, rcls, rbin)
        b_ = info["insample"]["model_run1"]
        same = all(np.array_equal(a_[q], b_[q]) for q in ("p_g3", "p_recal", "p_b")) and \
            all(np.array_equal(a_[kd][q], b_[kd][q]) for kd in KINDS for q in ("mean", "L", "U"))
        r2, info2 = run(tmp / "g3", tmp / "g3b", tmp / "data", tmp / "out2", out=quiet)
        t("S3", same and info2["md5"] == info["md5"], f"applied from the written file = applied in memory {same}; a second run "
          f"gives the same md5 ({info['md5'][:8]} and {info2['md5'][:8]})")
        # S4 refusals
        stop = []
        for args in ((tmp / "g3", tmp / "g3b", tmp / "data", tmp / "out"), (tmp / "g3", tmp / "g3b", tmp / "data", tmp / "real_eval_fits")):
            try:
                run(*args, out=quiet); stop.append(False)
            except SystemExit:
                stop.append(True)
        (tmp / "data_ev" / "real_eval").mkdir(parents=True)
        r4, _ = run(tmp / "g3", tmp / "g3b", tmp / "data_ev", tmp / "out_ev", out=quiet)
        none_written = not (tmp / "out_ev").exists()
        t("S4", all(stop) and r4 == {"A0": False} and none_written, f"an output folder that holds outputs is refused {stop[0]}; a "
          f"real_eval path is refused {stop[1]}; with data/real_eval present A0 fails and nothing is written {none_written}")
        # planted input bugs
        copy_dir(tmp / "g3", tmp / "g3_bad")
        zz = dict(np.load(tmp / "g3_bad" / "g3_calibrated.npz")); zz["aoo_B4"] = zz["aoo_B4"].copy(); zz["aoo_B4"][0, 0] += 1
        np.savez(tmp / "g3_bad" / "g3_calibrated.npz", **zz)
        rb, _ = run(tmp / "g3_bad", tmp / "g3b", tmp / "data", tmp / "out_bad", out=quiet)
        t("bug changed_input", rb.get("A1") is False, f"a map AOO changed after G3 fails A1 ({rb})")
        synth_g3(tmp / "g3_other", seed=9)
        GR.run(tmp / "g3_other", tmp / "g3b_other", out=quiet)
        ro, _ = run(tmp / "g3", tmp / "g3b_other", tmp / "data", tmp / "out_other", out=quiet)
        t("bug other_g3b", ro.get("A1") is False, f"a G3b output made from another G3 file fails A1 ({ro})")
        for bug, gate in (("parent_skip", "A2"), ("no_floor", "A2"), ("recal_all", "A2"), ("subset_fit", "A3"), ("optb_order", "A3")):
            env = dict(os.environ, STAGE_G5A_BROKEN=bug)
            code = ("import importlib.util as u; s=u.spec_from_file_location('f', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r,_=m.run(%r, %r, %r, %r, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), str(tmp / "g3"), str(tmp / "g3b"), str(tmp / "data"), str(tmp / f"b_{bug}")))
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
    ap.add_argument("--g3b", default="../data/stage_g3b_recalibration")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../data/stage_g5_fits")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    res, _ = run(a.g3, a.g3b, a.data, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if res and all(res.values()) else 1)