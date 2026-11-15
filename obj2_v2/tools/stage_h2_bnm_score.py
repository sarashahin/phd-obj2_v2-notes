#!/usr/bin/env python3
"""
stage_h2_bnm_score.py  (obj2_v2, stage H step H2b, 12 Nov 2026; H rules signed off by Sara on 11 Nov, H2 points on 12 Nov,
claude/obj2_v2_H_rules_draft.md)

Question: H2b of the signed H rules. Score the fit-half predictions of H2a in the holes: H-R1 ranking (the F5 rule on BNM),
H-R2 recorded count in the holes (thinned by detection, G3's correction and 90% interval cross-fitted on two groups of 22
windows), and the report-only items. Fit half only: test-half scoring is H6, with the correction frozen in H3.

Order
    Phase 1, BNM truth closed (np.load of a *_truth file in the H1 folder is refused): Y0 inputs, Y1 alignment, Y2 scoring
        code on real_val; the count maps of every method, md5 printed.
    Phase 2a, hole masks only (a *_truth.npz array other than hole, pair_md5 or species is refused): Y3 holes and thinning;
        every prediction-side value in the holes written to h2b_predicted.npz and its md5 printed.
    Phase 2b, labels: Y4 labels aligned with the inputs and with H1; rules and reports; Y5 order; Y6 folds and leak.

Rules (as signed; nothing here changes them)
    Items: the scored stack-repeats of bnm_frozen.npz (a hole and a kept record). Groups from inputs only: kept recorded
        class at 2019 (0, 1, 2-5, 6-10, 11-20, 21-50, 51+) and kept records over 6 years (1-3, 4-6, 7-10, 11-20, 21+). A rule
        applies to a group with 50 or more distinct stacks; stack-repeat counts are printed beside.
    H-R1  score per hole tetrad at 2019: the model's share of its 50 samples occupied; B1 stored (float32) and B1 exact log;
        B3 no-habitat chance, B4 logit, B5 score (report). Per stack-repeat AUC over its holes, recorded in 2019 against not
        (rankable: both present). d = AUC(model) - AUC(B1). Group mean: each stack's repeats in the group averaged first,
        then the mean over stacks (Sara, 12 Nov); 95% interval from 2,000 window resamples (seed 31; stage_f_check
        class_boot). PASS mean >= 0, FAIL upper end < 0, else NOT RESOLVED; the rule passes if every tested group of both
        groupings passes; a seed passes only against both B1 versions.
    H-R2  recorded count in the holes. Each map's occupied hole tetrads are thinned by the fit-half detection of the
        stack's kept-record class (1-5, 6-20, 21+ tetrads over 6 years: 0.643, 0.642, 0.800); one uniform per stack-repeat,
        sample and cell from seed (31201, 1, stack-repeat index), the same for every method. Then G3's level correction and
        CQR 90% interval (stage_g_calibrate fit_params, options 1 and 2), cross-fitted on G3's make_folds of the 44 window
        names (seed 77, 3); lower end 0 (no kept record lies in a hole). The fit-size fallback (30) counts distinct stacks
        (with one item per stack it is G3's code exactly, gate Y2). Rule as G3: ratio (mean corrected / mean observed)
        0.8-1.25 and coverage 0.85-0.95 in every tested group of both groupings, coverage counted per stack-repeat; option 1
        if it passes, else option 2, else FAIL (printed with option 1). Methods: seed 1, seed 2, pooled (first 25 maps of
        each), B3, B4, B5 (B5 has no record in a hole, so its ratio is 0 by construction).
    Reported only: AUC by group; H-R1 with pooled truth (hole recorded in any of the 6 years); recall = mean share of a
        method's maps occupied at recorded hole tetrads (and at unrecorded ones) by distance to the nearest kept record of
        any year (Euclidean cells rounded to the nearest whole cell: 1, 2-3, 4-6, 7+); recorded count over every 2019
        surveyed tetrad (the maps thinned over input effort and holes) against all 2019 records; EOO of those thinned
        cells against EOO of all 2019 records (hull of tetrad centres raised to AOO, G3 code), split at the window cap 361;
        GB Red List 2022 status; per window (csv); stacks with 10+ records in 2019 (the 7 Nov version).

Gates
    Y0  inputs: the H1 export as logged (h1_report gates, manifest, bnm_frozen.npz, every input and truth file md5); H1
        and H2a are the fit half; h2a_report.json gates PASS, truth opened 0, every part PASS, every file md5 = the list,
        frozen md5 = H1, best.pt of each run = the one H2a used
    Y1  alignment from inputs only: every map file covers the frozen rows in order (layout, frozen md5, K 50 / 1); B3-B5
        stored AOO = packed AOO; kept records and classes = bnm_frozen; window, repeat and row = bnm_stacks; B4 distance =
        distance recomputed from the inputs; every row blanked
    Y2  scoring code on real_val (no BNM label): the H-R1 group code with one item per stack = stage F step 3a F5 of
        both runs (1e-9, verdicts); the H-R2 correction code = g3_calibrated.npz corrected AOO of every method at its G3
        option (1e-9)
    Y3  holes (phase 2a): read with only hole, pair_md5 and species open; pair md5 and species = inputs and bnm_stacks;
        hole counts = bnm_stacks; no hole tetrad holds a kept record or input effort in any year; thinned share in the
        holes per detection class = the signed value within 4 standard errors; a second pass in other chunks gives the
        same thinned counts
    Y4  labels: recorded holes inside the holes, strict inside pooled; outside the holes the truth records and effort =
        the inputs in all 6 years; every hole inside the original 2019 effort; rankable stack-repeats, distinct stacks and
        hole shares = h1_report.json; a random score gives AUC 0.50 (0.01 or 4 se)
    Y5  order: no BNM truth file opened in phase 1; no label array read before h2b_predicted.npz was written;
        h2b_predicted.npz unchanged after the labels were read
    Y6  folds by window (every window in one fold, sizes equal or one apart); each fold's corrected seed-1 outputs
        unchanged when that fold's own labels are shuffled (G3 K4)

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, about 10 min)
    python3 ../tools/stage_h2_bnm_score.py --self-test | tail -1
    python3 -u ../tools/stage_h2_bnm_score.py > ../logs/stage_h2b_score.txt 2>&1
Defaults: --data ../data, --bnm ../data/bnm_fit, --h2a ../data/stage_h2_predict, --runs ../runs/f_seed1 ../runs/f_seed2,
          --g3 ../data/stage_g3_calibration, --out ../data/stage_h2b_score.
Needs stage_h2_bnm_predict.py, stage_g5_predict.py, stage_g_calibrate.py and the stage tools they load; stage_h1_bnm_export.py
for the self-test.
"""
import argparse, csv, hashlib, importlib.util, json, os, shutil, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_h2_bnm_score.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


H2A = _load("stage_h2_bnm_predict")
P5 = H2A.P5
GC = _load("stage_g_calibrate")
GB, GS, CK, HR, E, D, X = P5.GB, P5.GS, P5.CK, P5.HR, P5.E, P5.D, P5.X
G, W = X.G, X.WINDOW
TCLS, BINS, TRUE, SCORE = D.TCLS, D.BINS, D.TRUE, CK.SCORE
KEY = 31201                                    # seed key of stage H2b (thinning 1)
DET, DLAB = (0.643, 0.642, 0.800), ("1-5", "6-20", "21+")
FLOOR = 50
DIST = ("1", "2-3", "4-6", "7+")
EOO_CAP = (G - 1) ** 2
HOLE_KEYS = ("hole", "pair_md5", "species")
MAP_M = ("s1", "s2", "B3", "B4", "B5")
SCORE_M = ("s1", "s2", "pooled", "B1", "B1x", "B3", "B4", "B5")
R2_M = ("s1", "s2", "pooled", "B3", "B4", "B5")
NAME = dict(s1="seed 1", s2="seed 2", pooled="seeds pooled", B1="B1 stored", B1x="B1 exact", B3="B3", B4="B4", B5="B5")
CH = 128


def broken():
    return os.environ.get("STAGE_H2B_BROKEN", "")    # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def arr_md5(*arrays):
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def dif(a, b):
    na, nb = a is None or (isinstance(a, float) and np.isnan(a)), b is None or (isinstance(b, float) and np.isnan(b))
    return 0.0 if na and nb else (float("inf") if na or nb else abs(float(a) - float(b)))


# ---------------------------------------------------------------- guards
class BnmGuard:
    """Phase 1: np.load of a *_truth file inside the BNM export folder is refused and logged (real_val is allowed)."""
    def __init__(self, bnm):
        self.root = Path(bnm).resolve()

    def __enter__(self):
        self.orig, self.opened = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        if isinstance(f, (str, Path)):
            p = Path(f).resolve()
            if "_truth" in p.name and self.root in p.parents:
                self.opened.append(str(p))
                raise RuntimeError(f"BNM truth file opened in phase 1: {p.name}")
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig
        return False


class KeyLog:
    """Phase 2: every array read from a *_truth.npz file is logged; in mode 'holes' an array other than HOLE_KEYS is refused."""
    def __init__(self):
        self.reads, self.mode = [], "holes"

    def __enter__(self):
        self.cls = np.lib.npyio.NpzFile
        self.orig = self.cls.__getitem__
        log = self

        def getitem(z, key):
            fn = str(getattr(getattr(z, "zip", None), "filename", "") or "")
            if fn.endswith("_truth.npz"):
                log.reads.append((log.mode, key))
                if log.mode == "holes" and key not in HOLE_KEYS:
                    raise RuntimeError(f"label array '{key}' read before h2b_predicted.npz was written ({Path(fn).name})")
            return log.orig(z, key)
        self.cls.__getitem__ = getitem
        return self

    def __exit__(self, *exc):
        self.cls.__getitem__ = self.orig
        return False


# ---------------------------------------------------------------- scoring pieces
def det_class(rec6):
    return np.where(rec6 <= 5, 0, np.where(rec6 <= 20, 1, 2))


def dist_bin(d):
    r = np.floor(d + 0.5)
    return np.where(r <= 1, 0, np.where(r <= 3, 1, np.where(r <= 6, 2, 3)))


def uniforms(items, K):
    """Thinning uniforms (len(items), K, G, G), keyed by stack-repeat index, so they do not depend on the chunking."""
    return np.stack([X.rng(KEY, 1, int(i)).random((K, G * G)) for i in items]).reshape(len(items), K, G, G)


def eoo_of(occ):
    """EOO (G3 code: hull of cell centres raised to AOO) of (B, G, G) maps, in blocks of G3's chunk size."""
    out = np.zeros(len(occ))
    for a in range(0, len(occ), GC.CH):
        out[a:a + GC.CH] = GC.eoo_aoo(occ[a:a + GC.CH])[0]
    return out


def count_maps(maps, k=None):
    """Packed maps (n, K, 300) -> occupied-sample counts at the target year (n, G, G)."""
    c = np.zeros((len(maps), G, G), np.int32)
    for a in range(0, len(maps), 512):
        pk = maps[a:a + 512] if k is None else maps[a:a + 512, :k]
        c[a:a + 512] = GC.target_year(pk).sum(1)
    return c


def item_aucs(ptr, score, lab):
    """Per stack-repeat AUC over its holes (stage D's stack_auc: ties half; NaN when not rankable)."""
    A = np.full(len(ptr) - 1, np.nan)
    for i in range(len(A)):
        a, b = ptr[i], ptr[i + 1]
        A[i] = HR.stack_auc(score[a:b], lab[a:b])
    return A


def hr1_core(d, stack, grp, win, classes):
    """Stack-first group means (Sara, 12 Nov): a stack's items in a group are averaged, then stage F's class_boot runs over
    stacks with window resampling. Every window of the item set stays in the resampling (NaN placeholders), so with one
    item per stack this is class_boot itself (gate Y2). n = distinct stacks."""
    ok = ~np.isnan(d)
    wu = np.unique(win)
    if ok.any():
        key = np.stack([stack[ok], grp[ok]], 1)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        dm = np.bincount(inv, weights=d[ok]) / np.bincount(inv)
        order = np.argsort(inv, kind="stable")
        first = order[np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]]
        ws = win[ok][first]
        gs = uk[:, 1]
    else:
        dm, ws, gs = np.zeros(0), win[:0], grp[:0]
    dd = np.concatenate([dm, np.full(len(wu), np.nan)])
    gg = np.concatenate([gs, np.full(len(wu), classes[0], dtype=gs.dtype)])
    ww = np.concatenate([ws, wu])
    if broken() == "core_shift":
        dd = dd + 1e-6
    return CK.class_boot(dd, gg, ww, classes=classes)


def hr1_rows(cb, labels, floor):
    rows = []
    for j, lab in enumerate(labels):
        r = cb[j]
        tested = r["n"] >= floor
        rows.append(dict(group=lab, n=r["n"], mean=r["mean"], lo=r["lo"], hi=r["hi"], tested=bool(tested),
                         verdict=CK.verdict(r["mean"], r["hi"]) if tested else "reported"))
    return rows


def overall(rows_lists):
    v = [r["verdict"] for rows in rows_lists for r in rows if r["tested"]]
    if not v:
        return "no group tested"
    return "FAIL" if "FAIL" in v else ("PASS" if all(x == "PASS" for x in v) else "NOT RESOLVED")


def both(a, b):
    if a == b == "PASS":
        return "PASS"
    if "FAIL" in (a, b):
        return "FAIL"
    return "no group tested" if a == b == "no group tested" else "NOT RESOLVED"


def crossfit_h(pred, y, rec_t, fold, fgrp, parent, kind, stack, only=None):
    """stage_g_calibrate.crossfit with the fit-size fallback counting distinct stacks; with one item per stack it is the
    G3 code line for line (gate Y2)."""
    n = len(y)
    mean_c = np.zeros(n); L = np.zeros(n); U = np.zeros(n); scaled = np.zeros(pred.shape); par = np.zeros((n, 2))
    Lraw = np.zeros(n); Uraw = np.zeros(n)
    nst = lambda ix: len(np.unique(stack[ix]))
    for c in (0, 1):
        if only is not None and c != only:
            continue
        leak = broken() == "leak" and not np.array_equal(stack, np.arange(n))    # self-test: a leak in the BNM cross-fit only
        fit = np.nonzero(fold == (1 - c))[0] if not leak else np.arange(n)
        chk = np.nonzero(fold == c)[0]
        cache = {}
        for g in np.unique(fgrp[chk]):
            fg = fit[fgrp[fit] == g]
            key = ("g", g)
            if nst(fg) < GC.N_FIT_MIN:
                pg = parent[chk[fgrp[chk] == g][0]]
                fg, key = (fit, ("all",)) if pg < 0 else (fit[parent[fit] == pg], ("p", pg))
                if nst(fg) < GC.N_FIT_MIN:
                    fg, key = fit, ("all",)
            if key not in cache:
                s, q = GC.fit_params(pred, y, fg)
                cache[key] = (s * 1.01 if broken() == "corr_scale" else s, q)
            s, q = cache[key]
            ci = chk[fgrp[chk] == g]
            sc = s * pred[ci]
            lo, hi = np.percentile(sc, GC.QLO, axis=1), np.percentile(sc, GC.QHI, axis=1)
            lo_c, hi_c = lo - q, hi + q
            floor = np.maximum(rec_t[ci], 0) if kind == "aoo" else np.zeros(len(ci))
            mean_c[ci] = sc.mean(1); L[ci] = np.maximum(lo_c, floor); U[ci] = np.maximum(hi_c, floor); scaled[ci] = sc
            par[ci] = (s, q); Lraw[ci] = lo_c; Uraw[ci] = hi_c
    return dict(mean=mean_c, L=L, U=U, scaled=scaled, par=par, Lraw=Lraw, Uraw=Uraw)


def evaluate_h(y, out, groups, stack):
    """G3's evaluate with distinct stacks beside the stack-repeat count."""
    cov = (y >= out["L"]) & (y <= out["U"])
    wid = out["U"] - out["L"]
    rows = {}
    for gname, (gi, labels) in groups.items():
        rows[gname] = []
        for j, lab in enumerate(labels):
            m = gi == j
            if m.any():
                rows[gname].append(dict(group=lab, n=int(m.sum()), stacks=len(np.unique(stack[m])),
                                        ratio=float(out["mean"][m].mean() / y[m].mean()) if y[m].mean() > 0 else float("nan"),
                                        cov=float(cov[m].mean()), width=float(np.median(wid[m]))))
    return rows


def rule_pass_h(rows, floor):
    tested = [(g, r) for g in ("recorded class", "records over 6 years") for r in rows[g] if r["stacks"] >= floor]
    bad = [f"{g} {r['group']}" for g, r in tested if not (GC.BAND[0] <= r["ratio"] <= GC.BAND[1] and GC.COV[0] <= r["cov"] <= GC.COV[1])]
    return bool(tested) and not bad, bad


# ---------------------------------------------------------------- Y2 replicas on real_val
def replica_f5(data, runs):
    fv = dict(np.load(data / "stage_d_frozen_val.npz"))
    hz = dict(np.load(data / "stage_d_headroom_pred.npz"))
    V, _, _, T, tc, comm = HR.load_stacks(data / "real_val", fv["files"], fv["file_index"], fv["row"], truth=True)
    Mk = ~V[:, W - 1]
    voff, _ = E.frozen_draws(fv)
    a_b1 = HR.aucs(hz["B1"], T, Mk)
    sid = np.arange(len(T))
    worst, same, txt = 0.0, True, []
    for r in runs:
        pk = np.load(Path(r) / "check_3a" / "preds_k50.npz")
        s = CK.unplace(pk["counts"][:, W - 1][:, None].astype(np.float32), voff)[:, 0] / int(pk["K"])
        cb = hr1_core(HR.aucs(s, T, Mk) - a_b1, sid, tc, comm, SCORE)
        ref = json.loads((Path(r) / "check_3a" / "report.json").read_text())["F5"]
        d = max(max(dif(cb[c][q], ref["by_class"][TRUE[c]][q]) for c in SCORE for q in ("mean", "lo", "hi")),
                max(dif(cb["balanced"][q], ref["balanced"][q]) for q in ("mean", "lo", "hi")))
        v = {c: CK.verdict(cb[c]["mean"], cb[c]["hi"]) for c in SCORE}
        ov = "FAIL" if "FAIL" in v.values() else ("PASS" if all(x == "PASS" for x in v.values()) else "NOT RESOLVED")
        sv = ov == ref["verdict"] and all(v[c] == ref["by_class"][TRUE[c]]["verdict"] for c in SCORE)
        worst = max(worst, d); same &= sv
        txt.append(f"{Path(r).name} F5 {ov} (report {ref['verdict']}), largest difference {d:.1e}")
    return worst <= 1e-9 and same, "; ".join(txt)


def replica_g3(g3):
    rep = json.loads((g3 / "g3_report.json").read_text())
    fp = g3 / "g3_calibrated.npz"
    z = np.load(fp)
    y, rcls, rbin, rec_t, fold = z["true_aoo"].astype(float), z["rcls"], z["rbin"], z["rec_t"], z["fold"]
    n = len(y)
    opt = {1: (rcls, np.full(n, -1)), 2: (rcls * len(BINS) + rbin, rcls)}
    worst, nm = 0.0, 0
    for key, ch in rep["chosen"].items():
        if not key.endswith(" aoo"):
            continue
        kk = key[:-4].replace(" ", "_")
        fg, pa = opt[int(ch["option"])]
        cf = crossfit_h(z[f"aoo_{kk}"].astype(float), y, rec_t, fold, fg, pa, "aoo", np.arange(n))
        for q in ("mean", "L", "U"):
            worst = max(worst, float(np.abs(cf[q] - z[f"aoo_{q}_{kk}"]).max()))
        nm += 1
    return X.file_md5(fp) == rep["md5"]["calibrated"] and worst <= 1e-9 and nm > 0, f"{nm} methods, largest difference {worst:.1e}"


# ---------------------------------------------------------------- truth readers (phase 2)
def read_holes(bnm, fb, man):
    n = len(fb["row"])
    H = np.zeros((n, G, G), bool); sp = np.empty(n, dtype=object); pair_ok = True
    for i in np.unique(fb["file_index"]):
        k = np.nonzero(fb["file_index"] == i)[0]; r = fb["row"][k]
        row = man[str(fb["files"][i])]
        with np.load(bnm / row["input_file"], allow_pickle=False) as zi:
            pin = str(zi["pair_md5"])
        with np.load(bnm / row["truth_file"], allow_pickle=False) as zt:
            H[k] = X.unpack(zt["hole"])[r]; sp[k] = zt["species"][r]
            pair_ok &= str(zt["pair_md5"]) == pin
    if broken() == "holes_shift":
        H = np.roll(H, 1, 0)
    return H, sp.astype(str), pair_ok


def read_labels(bnm, fb, man):
    n = len(fb["row"])
    Ls = np.zeros((n, G, G), bool); Lp = np.zeros_like(Ls)
    Ro = np.zeros((n, W, G, G), bool); Eo = np.zeros_like(Ro)
    for i in np.unique(fb["file_index"]):
        k = np.nonzero(fb["file_index"] == i)[0]; r = fb["row"][k]
        with np.load(bnm / man[str(fb["files"][i])]["truth_file"], allow_pickle=False) as zt:
            Ls[k] = X.unpack(zt["label_strict"])[r]; Lp[k] = X.unpack(zt["label_pooled"])[r]
            Ro[k] = X.unpack(zt["rec_orig"])[r]; Eo[k] = X.unpack(zt["eff_orig"])[r]
    if broken() == "labels_shift":
        Ls, Lp = np.roll(Ls, 1, 0), np.roll(Lp, 1, 0)
    return Ls, Lp, Ro, Eo


# ---------------------------------------------------------------- run
def run(data="../data", bnm="../data/bnm_fit", h2a="../data/stage_h2_predict", runs=("../runs/f_seed1", "../runs/f_seed2"),
        g3="../data/stage_g3_calibration", out_dir="../data/stage_h2b_score", floor=FLOOR, out=print):
    data, bnm, h2a, g3, out_dir = Path(data), Path(bnm), Path(h2a), Path(g3), Path(out_dir)
    runs = [Path(r) for r in runs]
    o = out_dir.resolve()
    for p_ in (bnm, h2a):
        if o == p_.resolve() or p_.resolve() in o.parents:
            sys.exit(f"STOP: nothing is written inside {p_}; give another --out")
    if "real_eval" in str(o) or "_truth" in str(o):
        sys.exit("STOP: the output folder may not lie in a real_eval folder or carry '_truth' in its path")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    js1 = json.loads((bnm / "h1_report.json").read_text())
    if js1.get("half") != "fit":
        sys.exit("STOP: H2b scores the fit half only; test-half scoring is H6, after the H3 freeze")
    if len(runs) != 2:
        sys.exit("STOP: give the two runs (--runs seed1 seed2)")
    t0 = time.time()
    res, rep = {}, dict(floor=floor)

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage H2b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}")
        return res, rep

    out(f"stage_h2_bnm_score.py md5 {X.file_md5(__file__)}; stage_h2_bnm_predict.py {X.file_md5(HERE / 'stage_h2_bnm_predict.py')}; "
        f"stage_g_calibrate.py {X.file_md5(HERE / 'stage_g_calibrate.py')}; stage_f_check.py {X.file_md5(HERE / 'stage_f_check.py')}; "
        f"stage_d_headroom.py {X.file_md5(HERE / 'stage_d_headroom.py')}; rule floor {floor} distinct stacks")
    # ================================================================ phase 1, BNM truth closed
    with BnmGuard(bnm) as guard:
        try:
            # ---- Y0
            k0t = []
            fb, mf, js, ok_h1 = H2A.k0(SimpleNamespace(gate=lambda g, ok, text: k0t.append(text)), bnm)
            man_rows = D.rows_of(bnm)
            man = {r["input_file"]: r for r in man_rows}
            tbad = [r["truth_file"] for r in man_rows if X.file_md5(bnm / r["truth_file"]) != r["truth_md5"]]
            ha = json.loads((h2a / "h2a_report.json").read_text())
            need = ["baselines_bnm.npz", "maps_B3.npz", "maps_B4.npz", "maps_B5.npz"] + [f"samples_{r.name}.npz" for r in runs]
            missing = [f for f in need if f not in ha["files"]]
            fbad = [f for f, m in ha["files"].items() if X.file_md5(h2a / f) != m]
            parts_ok = all(all(v["gates"].values()) for v in ha["parts"].values())
            best = {r.name: X.file_md5(r / "best.pt") for r in runs}
            best_ok = all(ha["best_md5"].get(f"samples_{r.name}") == best[r.name] for r in runs)
            ok0 = ok_h1 and not tbad and js["half"] == "fit" and ha["half"] == "fit" and all(ha["gates"].values()) and \
                ha["truth_opened"] == 0 and parts_ok and not missing and not fbad and ha["frozen_md5"] == mf and best_ok
            gate("Y0", ok0, f"{k0t[0]}; truth files md5 = manifest {not tbad}; h2a_report.json ({ha['tool'][:8]}, {ha['written_utc']}): "
                 f"gates PASS {all(ha['gates'].values())}, truth opened {ha['truth_opened']}, every part PASS {parts_ok}, files md5 "
                 f"= list {not fbad}" + (f" (changed: {', '.join(fbad)})" if fbad else "") + (f", missing {missing}" if missing else "")
                 + f", frozen md5 = H1 {ha['frozen_md5'] == mf}; best.pt " + ", ".join(f"{k} {v[:8]}" for k, v in best.items())
                 + f" = H2a {best_ok}")
            if not ok0:
                return stop("Y0")
            # ---- Y1
            n = len(fb["row"])
            idx = dict(np.load(bnm / "bnm_stacks.npz"))
            V1, V2, Ef, _, _ = GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])
            V = V1 | V2
            del V1, V2
            rec_t = V[:, -1].reshape(n, -1).sum(1); rec6 = V.any(1).reshape(n, -1).sum(1)
            rcls, rbin, dcls = D.tidx(rec_t), D.bidx(rec6), det_class(rec6)
            winf = np.array([man[str(fb["files"][i])]["window"] for i in fb["file_index"]])
            repf = np.array([int(man[str(fb["files"][i])]["repeat"]) for i in fb["file_index"]])
            al = np.array_equal(rec_t, fb["rec_t"]) and np.array_equal(rec6, fb["rec6"]) and np.array_equal(rcls, fb["cls"]) \
                and len(idx["window"]) == n and np.array_equal(winf, idx["window"]) and np.array_equal(repf, idx["repeat"]) \
                and np.array_equal(fb["row"], idx["row"]) and bool((fb["blank"] == 1).all())
            maps, why1, ok1 = {}, [], True
            for m_, f, kexp in (("s1", f"samples_{runs[0].name}.npz", 50), ("s2", f"samples_{runs[1].name}.npz", 50),
                                ("B3", "maps_B3.npz", 50), ("B4", "maps_B4.npz", 50), ("B5", "maps_B5.npz", 1)):
                z = np.load(h2a / f)
                good = np.array_equal(z["sel"], np.arange(n)) and str(z["layout"]) == GS.LAYOUT and str(z["frozen_md5"]) == mf \
                    and int(z["K"]) == kexp and z["maps"].shape == (n, kexp, W * G * G // 8)
                if m_ in ("B3", "B4", "B5"):
                    good &= all(np.array_equal(GC.target_year(z["maps"][a:a + 512]).sum((2, 3)), z["aoo_t"][a:a + 512]) for a in range(0, n, 512))
                ok1 &= good; why1.append(f"{f} {good}")
                maps[m_] = z["maps"]
            bz = dict(np.load(h2a / "baselines_bnm.npz"))
            dist = GB.near_dist(V)
            bok = str(bz["frozen_md5"]) == mf and all(bz[k].shape == (n, G, G) for k in ("B1", "B1_exact_log", "B3_nohab", "B4_logit", "B4_dist", "B5")) \
                and np.array_equal(bz["B4_dist"], dist.astype(np.float32))
            gate("Y1", ok1 and al and bok, f"{n:,} stack-repeats (bnm_frozen {mf[:8]}); map files in frozen order with the G1 layout, "
                 f"K and frozen md5, B3-B5 AOO = stored: {', '.join(why1)}; kept records, classes, window, repeat, row and blank "
                 f"from the inputs = bnm_frozen and bnm_stacks {al}; baselines frozen md5, shapes and B4 distance = recomputed {bok}")
            if not (ok1 and al and bok):
                return stop("Y1")
            # ---- Y2
            a_f5, t_f5 = replica_f5(data, runs)
            a_g3, t_g3 = replica_g3(g3)
            gate("Y2", a_f5 and a_g3, f"on real_val: H-R1 group code with one item per stack = stage F step 3a F5 {a_f5} ({t_f5}); "
                 f"H-R2 correction code = g3_calibrated.npz corrected AOO {a_g3} ({t_g3})")
            if not (a_f5 and a_g3):
                return stop("Y2")
            # ---- count maps (no truth)
            cnt = {m_: count_maps(maps[m_]) for m_ in MAP_M}
            cnt["pooled"] = count_maps(maps["s1"], 25) + count_maps(maps["s2"], 25)
            K = 50
            smap = {"s1": cnt["s1"] / K, "s2": cnt["s2"] / K, "pooled": cnt["pooled"] / K, "B1": bz["B1"], "B1x": bz["B1_exact_log"],
                    "B3": bz["B3_nohab"], "B4": bz["B4_logit"], "B5": bz["B5"]}
            share = {"s1": smap["s1"], "s2": smap["s2"], "pooled": smap["pooled"], "B3": cnt["B3"] / K, "B4": cnt["B4"] / K,
                     "B5": cnt["B5"].astype(float)}
            out("   score maps from the H2a files, truth closed: " + ", ".join(f"{NAME[m_]} {arr_md5(smap[m_])[:8]}" for m_ in SCORE_M))
            if broken() == "p1_open":
                np.load(bnm / man_rows[0]["truth_file"])
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    if crashed:
        gate("Y5", False, f"order: {crashed}")
        return stop("Y5")
    p1_opened = len(guard.opened)
    stid = idx["stack_id"]; status = idx["status"].astype(str); wname = idx["window"].astype(str)
    fold = GC.make_folds(wname) if broken() != "fold_stack" else GC.make_folds(stid)
    # ================================================================ phase 2a, hole masks only
    with KeyLog() as kl:
        try:
            H, sp_t, pair_ok = read_holes(bnm, fb, man)
            nh = H.reshape(n, -1).sum(1)
            nobad = not (V & H[:, None]).any() and not (Ef & H[:, None]).any()
            sp_ok = np.array_equal(sp_t, idx["species"].astype(str))
            cnt_ok = np.array_equal(nh, idx["n_holes"]) and bool((nh >= 1).all())
            surv = Ef[:, -1] | H
            items, cells = np.nonzero(H.reshape(n, -1))
            ptr = np.r_[0, np.cumsum(nh)]
            pthr = np.array(DET)[dcls]
            occ_h = {m_: np.zeros((n, maps[m_].shape[1]), np.uint16) for m_ in MAP_M}
            thin_h = {m_: np.zeros_like(occ_h[m_]) for m_ in MAP_M}
            thin_s = {m_: np.zeros_like(occ_h[m_]) for m_ in MAP_M}
            eoo_s = {m_: np.zeros(occ_h[m_].shape, np.float32) for m_ in MAP_M}
            tc = time.time()

            def thin_block(a, b, eoo=True):
                Uc = uniforms(range(a, b), K)
                keep = Uc < (pthr[a:b] if broken() != "thin_p" else np.full(b - a, 0.9))[:, None, None, None]
                Hc, Sc = H[a:b, None], surv[a:b, None]
                res_ = {}
                for m_ in MAP_M:
                    occ = GC.target_year(maps[m_][a:b])
                    th = occ & keep[:, :occ.shape[1]]
                    ts = th & Sc
                    res_[m_] = ((occ & Hc).sum((2, 3)), (th & Hc).sum((2, 3)), ts.sum((2, 3)),
                                eoo_of(ts.reshape(-1, G, G)).reshape(b - a, -1) if eoo else None)
                return res_
            for a in range(0, n, CH):
                b = min(n, a + CH)
                rb = thin_block(a, b)
                for m_ in MAP_M:
                    occ_h[m_][a:b], thin_h[m_][a:b], thin_s[m_][a:b], eoo_s[m_][a:b] = rb[m_]
            # second pass, other chunks: same thinned counts
            same2, m2 = True, min(n, 200)
            for a in range(0, m2, 7):
                rb = thin_block(a, min(m2, a + 7), eoo=False)
                same2 &= all(np.array_equal(rb[m_][1], thin_h[m_][a:min(m2, a + 7)]) and np.array_equal(rb[m_][2], thin_s[m_][a:min(m2, a + 7)])
                             for m_ in MAP_M)
            ztxt, zok = [], True
            for c in range(3):
                O = sum(int(occ_h[m_][dcls == c].sum()) for m_ in ("s1", "s2", "B3", "B4"))
                T_ = sum(int(thin_h[m_][dcls == c].sum()) for m_ in ("s1", "s2", "B3", "B4"))
                if O == 0:
                    ztxt.append(f"{DLAB[c]} no occupied hole cell"); continue
                z_ = (T_ / O - DET[c]) / np.sqrt(DET[c] * (1 - DET[c]) / O)
                zok &= abs(z_) <= 4
                ztxt.append(f"{DLAB[c]} {T_ / O:.4f} (signed {DET[c]}, z {z_:+.2f}, {O:,} occupied hole cells)")
            gate("Y3", pair_ok and sp_ok and cnt_ok and nobad and zok and same2,
                 f"hole masks read with only {', '.join(HOLE_KEYS)} open: pair md5 = inputs {pair_ok}, species = bnm_stacks {sp_ok}, "
                 f"hole counts = bnm_stacks and at least 1 {cnt_ok} ({int(nh.sum()):,} hole tetrads); no kept record or input effort "
                 f"in a hole in any year {nobad}; thinned share in the holes by detection class: {'; '.join(ztxt)}; second pass in "
                 f"chunks of 7 gives the same thinned counts {same2} ({(time.time() - tc) / 60:.1f} min)")
            if not res["Y3"]:
                return stop("Y3")
            # ---- prediction-side values in the holes, written before any label array
            pred = dict(hole_ptr=ptr.astype(np.int64), hole_cell=cells.astype(np.int16), dist_h=dist.reshape(n, -1)[items, cells].astype(np.float32),
                        rec_t=rec_t.astype(np.int16), rec6=rec6.astype(np.int16), rcls=rcls.astype(np.int8), rbin=rbin.astype(np.int8),
                        dcls=dcls.astype(np.int8), fold=fold.astype(np.int8), stack_id=stid, frozen_md5=np.array(mf), key=np.int64(KEY))
            for m_ in SCORE_M:
                pred[f"score_{m_}"] = np.asarray(smap[m_]).reshape(n, -1)[items, cells].astype(np.float32)
            for m_ in R2_M:
                pred[f"share_{m_}"] = np.asarray(share[m_]).reshape(n, -1)[items, cells].astype(np.float32)
            for m_ in MAP_M:
                pred[f"occ_h_{m_}"], pred[f"thin_h_{m_}"], pred[f"thin_s_{m_}"], pred[f"eoo_s_{m_}"] = occ_h[m_], thin_h[m_], thin_s[m_], eoo_s[m_]
            out_dir.mkdir(parents=True, exist_ok=True)
            pf = out_dir / "h2b_predicted.npz"
            X.save_npz(pf, pred)
            mpred = X.file_md5(pf)
            out(f"   h2b_predicted.npz md5 {mpred}: scores, map shares and distances at {int(nh.sum()):,} hole tetrads, occupied and "
                f"thinned counts in the holes and over the 2019 survey, thinned EOO; written before any label array was read")
            kl.mode = "labels"
            if broken() == "label_early":
                kl.mode = "holes"
            # ================================================================ phase 2b, labels
            Ls, Lp, Ro, Eo = read_labels(bnm, fb, man)
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("Y5", False, f"order: {crashed}")
        return stop("Y5")
    early = [k for m_, k in kl.reads if m_ == "holes" and k not in HOLE_KEYS]
    # ---- Y4
    Hm = H[:, None]
    in_h = not (Ls & ~H).any() and not (Lp & ~H).any() and not (Ls & ~Lp).any()
    outside = np.array_equal(Ro & ~Hm, V) and np.array_equal(Eo & ~Hm, Ef)
    in_eff = not (H & ~Eo[:, -1]).any()
    lab_s = Ls.reshape(n, -1)[items, cells]; lab_p = Lp.reshape(n, -1)[items, cells]
    y = np.add.reduceat(lab_s.astype(int), ptr[:-1]).astype(float); yp = np.add.reduceat(lab_p.astype(int), ptr[:-1]).astype(float)
    rank = (y > 0) & (y < nh)
    sh_s, sh_p = y.sum() / nh.sum(), yp.sum() / nh.sum()
    h1_same = int(rank.sum()) == js1.get("rankable") and len(np.unique(stid[rank])) == js1.get("rankable_stacks") \
        and abs(sh_s - js1.get("hole_share_strict", -1)) < 1e-12 and abs(sh_p - js1.get("hole_share_pooled", -1)) < 1e-12
    rnd = X.rng(KEY, 9).random(len(lab_s))
    ar = item_aucs(ptr, rnd, lab_s)
    ar_m, ar_se = float(np.nanmean(ar)), float(np.nanstd(ar) / np.sqrt(max(1, np.isfinite(ar).sum())))
    tol = max(0.01, 4 * ar_se)
    gate("Y4", in_h and outside and in_eff and h1_same and abs(ar_m - 0.5) <= tol,
         f"recorded holes inside the holes, strict inside pooled {in_h}; outside the holes the truth records and effort = the inputs "
         f"in all {W} years {outside}; every hole inside the 2019 effort {in_eff}; rankable {int(rank.sum()):,} stack-repeats "
         f"({len(np.unique(stid[rank])):,} stacks), hole shares strict {sh_s:.3f}, pooled {sh_p:.3f} = h1_report.json {h1_same}; "
         f"random score AUC {ar_m:.3f} (0.50 within {tol:.3f})")
    if not res["Y4"]:
        return stop("Y4")
    # ================================================================ H-R1
    A = {m_: item_aucs(ptr, pred[f"score_{m_}"], lab_s) for m_ in SCORE_M}
    Ap = {m_: item_aucs(ptr, pred[f"score_{m_}"], lab_p) for m_ in SCORE_M}
    grp = {"recorded class": (rcls, TCLS), "records over 6 years": (rbin, BINS)}
    auc_g = {m_: {g: hr1_core(A[m_], stid, gi, wname, list(range(len(labs)))) for g, (gi, labs) in grp.items()} for m_ in SCORE_M}
    out("   AUC in the holes, stack-first mean by group (report):")
    for m_ in SCORE_M:
        out(f"      {NAME[m_]:<9} " + " | ".join("; ".join(f"{lab} {auc_g[m_][g][j]['mean']:.3f}" for j, lab in enumerate(labs))
                                             for g, (gi, labs) in grp.items()))

    def hr1(Adict, mods, refs, label_txt):
        outr = {}
        for m_ in mods:
            rr = {}
            for b in refs:
                rows = {g: hr1_rows(hr1_core(Adict[m_] - Adict[b], stid, gi, wname, list(range(len(labs)))), labs, floor)
                        for g, (gi, labs) in grp.items()}
                rr[b] = dict(rows=rows, verdict=overall(rows.values()))
            v = both(rr[refs[0]]["verdict"], rr[refs[-1]]["verdict"]) if len(refs) == 2 else rr[refs[0]]["verdict"]
            outr[m_] = dict(rr, verdict=v)
            out(f"      {label_txt} {NAME[m_]}:")
            for b in refs:
                for g, rows in rr[b]["rows"].items():
                    out(f"         vs {NAME[b]:<9} {g:<20} " + "; ".join(f"{r['group']} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                                                                          f"{r['verdict']} ({r['n']})" if r["n"] else f"{r['group']} none"
                                                                          for r in rows))
                out(f"         vs {NAME[b]}: {rr[b]['verdict']}")
            out(f"         {NAME[m_]} H-R1 {'(both B1 versions needed) ' if len(refs) == 2 and m_ in ('s1', 's2') else '(report) '}{v}")
        return outr
    out(f"   H-R1 ranking in the holes, model - B1, stack-first mean [95% interval, {len(np.unique(wname))} windows resampled] "
        f"verdict (distinct stacks; tested at {floor}+):")
    r1 = hr1(A, ("s1", "s2", "pooled"), ("B1", "B1x"), "strict truth")
    r1b = hr1(A, ("B3", "B4", "B5"), ("B1",), "strict truth (report)")
    out("   H-R1 with pooled truth (hole recorded in any of the 6 years; report):")
    r1p = hr1(Ap, ("s1", "s2"), ("B1", "B1x"), "pooled truth")
    # ================================================================ H-R2
    predc = {m_: pred[f"thin_h_{m_}"].astype(float) for m_ in MAP_M}
    predc["pooled"] = np.concatenate([predc["s1"][:, :25], predc["s2"][:, :25]], 1)
    zero = np.zeros(n)
    opt = {1: (rcls, np.full(n, -1)), 2: (rcls * len(BINS) + rbin, rcls)}
    r2, chosen = {}, {}
    for m_ in R2_M:
        per = {}
        for o_ in (1, 2):
            fg, pa = opt[o_]
            cf = crossfit_h(predc[m_], y, zero, fold, fg, pa, "aoo", stid)
            rows = evaluate_h(y, cf, grp, stid)
            ok, bad = rule_pass_h(rows, floor)
            per[o_] = dict(cf=cf, rows=rows, ok=ok, bad=bad)
        use = 1 if per[1]["ok"] else (2 if per[2]["ok"] else 1)
        chosen[m_] = dict(option=use, ok=per[use]["ok"], bad=per[use]["bad"], opt1_ok=per[1]["ok"], opt2_ok=per[2]["ok"])
        r2[m_] = per[use]
    out(f"   H-R2 recorded count in the holes after the cross-fitted correction (ratio / coverage / median width (distinct stacks); "
        f"rule: ratio {GC.BAND[0]}-{GC.BAND[1]}, coverage {GC.COV[0]}-{GC.COV[1]} in every group of {floor}+ stacks; * = reported):")
    for m_ in R2_M:
        ch = chosen[m_]
        out(f"      {NAME[m_]:<8} option {ch['option']} rule {'PASS' if ch['ok'] else 'FAIL'}" + ("" if ch["ok"] else f" ({', '.join(ch['bad'][:6])})")
            + f"; option 1 {ch['opt1_ok']}, option 2 {ch['opt2_ok']}")
        for g in grp:
            out(f"         {g:<20} " + "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}/{r['width']:.1f} ({r['stacks']})"
                                                + ("" if r["stacks"] >= floor else "*") for r in r2[m_]["rows"][g]))
    out("      stack-repeats per group: " + " | ".join("; ".join(f"{r['group']} {r['n']:,}" for r in r2["s1"]["rows"][g]) for g in grp)
        + f"; observed recorded holes per stack-repeat: mean {y.mean():.2f}, zero in {(y == 0).mean():.3f}")
    # ---- Y6
    one = all(len(np.unique(fold[wname == w_])) == 1 for w_ in np.unique(wname))
    nw = [len(np.unique(wname[fold == f_])) for f_ in (0, 1)]
    leak_ok = True
    fg, pa = opt[chosen["s1"]["option"]]
    for c in (0, 1):
        ix = np.nonzero(fold == c)[0]
        y2 = y.copy(); y2[ix] = y[X.rng(77, 7, c).permutation(ix)]
        a_ = crossfit_h(predc["s1"], y, zero, fold, fg, pa, "aoo", stid, only=c)
        b_ = crossfit_h(predc["s1"], y2, zero, fold, fg, pa, "aoo", stid, only=c)
        leak_ok &= all(np.array_equal(a_[q][ix], b_[q][ix]) for q in ("mean", "L", "U"))
    gate("Y6", one and abs(nw[0] - nw[1]) <= 1 and leak_ok, f"folds by window: {nw[0]} and {nw[1]} windows "
         f"({int((fold == 0).sum()):,} and {int((fold == 1).sum()):,} stack-repeats), every window in one fold {one}; each fold's "
         f"corrected seed-1 outputs unchanged when that fold's own labels are shuffled {leak_ok}")
    # ================================================================ reported only
    dbin = dist_bin(pred["dist_h"])
    rec_all = (Ro[:, -1]).reshape(n, -1).sum(1)
    sub10 = rec_all >= 10

    def recall(lab, mask_items=None):
        keep = np.ones(len(lab), bool) if mask_items is None else np.repeat(mask_items, nh)
        rows = {}
        for m_ in R2_M:
            s_ = pred[f"share_{m_}"]
            rows[m_] = [(float(s_[keep & lab & (dbin == j)].mean()) if (keep & lab & (dbin == j)).any() else float("nan"),
                         float(s_[keep & ~lab & (dbin == j)].mean()) if (keep & ~lab & (dbin == j)).any() else float("nan"))
                        for j in range(4)]
        ncell = [(int((keep & lab & (dbin == j)).sum()), int((keep & ~lab & (dbin == j)).sum())) for j in range(4)]
        return rows, ncell
    rc, ncs = recall(lab_s)
    out("   recall by distance to the nearest kept record (report; mean share of maps occupied at recorded / unrecorded hole tetrads; "
        "cells " + ", ".join(f"{DIST[j]} {ncs[j][0]:,}/{ncs[j][1]:,}" for j in range(4)) + "):")
    for m_ in R2_M:
        out(f"      {NAME[m_]:<8} " + "; ".join(f"{DIST[j]} {rc[m_][j][0]:.3f}/{rc[m_][j][1]:.3f}" for j in range(4)))
    rcp, ncp = recall(lab_p)
    out("      pooled truth: " + " | ".join(f"{NAME[m_]} " + "; ".join(f"{DIST[j]} {rcp[m_][j][0]:.3f}/{rcp[m_][j][1]:.3f}" for j in range(4))
                                         for m_ in ("s1", "s2", "B4")))
    obs_all = rec_all.astype(float)
    obs_eoo = eoo_of(Ro[:, -1])
    tmean = {m_: pred[f"thin_s_{m_}"].astype(float).mean(1) for m_ in MAP_M}
    tmean["pooled"] = np.concatenate([pred["thin_s_s1"][:, :25], pred["thin_s_s2"][:, :25]], 1).astype(float).mean(1)
    emean = {m_: pred[f"eoo_s_{m_}"].astype(float).mean(1) for m_ in MAP_M}
    emean["pooled"] = np.concatenate([pred["eoo_s_s1"][:, :25], pred["eoo_s_s2"][:, :25]], 1).astype(float).mean(1)
    ratio = lambda p_, o_, m: float(p_[m].mean() / o_[m].mean()) if m.any() and o_[m].mean() > 0 else float("nan")
    allrows = {m_: {g: [ratio(tmean[m_], obs_all, gi == j) for j in range(len(labs))] for g, (gi, labs) in grp.items()} for m_ in R2_M}
    out("   recorded count over every 2019 surveyed tetrad, thinned maps / all 2019 records, no correction (report): " + " || ".join(
        f"{NAME[m_]} " + " | ".join("; ".join(f"{labs[j]} {allrows[m_][g][j]:.2f}" for j in range(len(labs))) for g, (gi, labs) in grp.items())
        for m_ in ("s1", "s2", "B4", "B5")))
    cap = obs_eoo >= EOO_CAP
    eoorows = {m_: dict(below=ratio(emean[m_], obs_eoo, ~cap), cap=ratio(emean[m_], obs_eoo, cap),
                        by_class=[ratio(emean[m_], obs_eoo, rcls == j) for j in range(len(TCLS))]) for m_ in R2_M}
    out(f"   EOO of thinned recorded cells / EOO of all 2019 records (report; stack-repeats below the cap {int((~cap).sum()):,}, "
        f"at {EOO_CAP}+ {int(cap.sum()):,}): " + "; ".join(f"{NAME[m_]} {eoorows[m_]['below']:.2f} below, {eoorows[m_]['cap']:.2f} at cap"
                                                         for m_ in R2_M))
    status = np.where(status == "", "no status", status)
    stl = sorted(set(status.tolist()))
    sti = np.array([stl.index(s_) for s_ in status])
    by_status = {}
    for m_ in ("s1", "s2"):
        cb = hr1_core(A[m_] - A["B1"], stid, sti, wname, list(range(len(stl))))
        by_status[m_] = dict(hr1={stl[j]: dict(cb[j]) for j in range(len(stl))},
                             hr2={r_["group"]: r_ for r_ in evaluate_h(y, r2[m_]["cf"], {"s": (sti, stl)}, stid)["s"]})
    out("   GB Red List 2022 status (report; H-R1 seed - B1 stored, stack-first [interval] (stacks) | H-R2 ratio/coverage): " + " || ".join(
        f"{NAME[m_]} " + "; ".join(f"{s_} {by_status[m_]['hr1'][s_]['mean']:+.3f} [{by_status[m_]['hr1'][s_]['lo']:+.3f}, "
                                   f"{by_status[m_]['hr1'][s_]['hi']:+.3f}] ({by_status[m_]['hr1'][s_]['n']}) | "
                                   f"{by_status[m_]['hr2'][s_]['ratio']:.2f}/{by_status[m_]['hr2'][s_]['cov']:.2f}" for s_ in stl)
        for m_ in ("s1", "s2")))
    r10 = {}
    for m_ in ("s1", "s2"):
        r10[m_] = {b: hr1_rows(hr1_core(A[m_] - A[b], stid, np.where(sub10, 0, -1), wname, [0]), ["10+"], floor)[0] for b in ("B1", "B1x")}
    rc10, nc10 = recall(lab_s, sub10)
    out(f"   stacks with 10+ records in 2019, the 7 Nov version ({int(sub10.sum()):,} stack-repeats, {len(np.unique(stid[sub10])):,} "
        f"stacks; report): " + "; ".join(f"{NAME[m_]} - {NAME[b]} {r10[m_][b]['mean']:+.4f} [{r10[m_][b]['lo']:+.4f}, {r10[m_][b]['hi']:+.4f}]"
                                         for m_ in ("s1", "s2") for b in ("B1", "B1x"))
        + "; recall " + " | ".join(f"{NAME[m_]} " + "; ".join(f"{DIST[j]} {rc10[m_][j][0]:.3f}" for j in range(4)) for m_ in ("s1", "s2", "B4")))
    # ---- per window
    wl = sorted(set(wname.tolist()))
    wrows = []
    for w_ in wl:
        m = wname == w_
        r_ = dict(window=w_, fold=int(fold[m][0]), stack_repeats=int(m.sum()), stacks=len(np.unique(stid[m])), rankable=int((rank & m).sum()))
        for m_ in ("s1", "s2"):
            dd = (A[m_] - A["B1"])[m]
            ok_ = ~np.isnan(dd)
            if ok_.any():
                _, inv = np.unique(stid[m][ok_], return_inverse=True)
                r_[f"mean_d_{m_}_B1"] = float((np.bincount(np.asarray(inv).ravel(), weights=dd[ok_]) / np.bincount(np.asarray(inv).ravel())).mean())
            else:
                r_[f"mean_d_{m_}_B1"] = float("nan")
            cf = r2[m_]["cf"]
            r_[f"hr2_ratio_{m_}"] = float(cf["mean"][m].mean() / y[m].mean()) if y[m].mean() > 0 else float("nan")
            r_[f"hr2_cov_{m_}"] = float(((y[m] >= cf["L"][m]) & (y[m] <= cf["U"][m])).mean())
        wrows.append(r_)
    dneg = [r_["mean_d_s1_B1"] for r_ in wrows if np.isfinite(r_["mean_d_s1_B1"])]
    out(f"   per window (report, per_window_h2b.csv): {len(wl)} windows; seed 1 - B1 stored stack-first mean below 0 in "
        f"{sum(v < 0 for v in dneg)} of {len(dneg)} (range {min(dneg):+.3f} to {max(dneg):+.3f}); H-R2 seed 1 ratio range "
        f"{np.nanmin([r_['hr2_ratio_s1'] for r_ in wrows]):.2f} to {np.nanmax([r_['hr2_ratio_s1'] for r_ in wrows]):.2f}"
        if dneg else f"   per window (report): {len(wl)} windows, none rankable")
    # ---- Y5
    if broken() == "pred_changed":
        z_ = dict(np.load(pf)); z_["thin_h_s1"] = z_["thin_h_s1"] + 1; X.save_npz(pf, z_)
    same_p = X.file_md5(pf) == mpred
    gate("Y5", p1_opened == 0 and not early and same_p, f"order: BNM truth files opened in phase 1 {p1_opened}; label arrays read "
         f"before h2b_predicted.npz was written {len(early)}; h2b_predicted.npz unchanged after the labels were read {same_p}")
    # ================================================================ files
    sc = dict(y_strict=y, y_pooled=yp, rankable=rank, obs_all_2019=obs_all, obs_eoo_2019=obs_eoo, fold=fold, stack_id=stid,
              predicted_md5=np.array(mpred))
    for m_ in SCORE_M:
        sc[f"auc_strict_{m_}"] = A[m_]; sc[f"auc_pooled_{m_}"] = Ap[m_]
    for m_ in R2_M:
        for q in ("mean", "L", "U"):
            sc[f"hr2_{q}_{m_}"] = r2[m_]["cf"][q]
        sc[f"hr2_scale_q_{m_}"] = r2[m_]["cf"]["par"]
    X.save_npz(out_dir / "h2b_scores.npz", sc)
    with open(out_dir / "per_stack_h2b.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        head = ["item", "window", "species", "status", "repeat", "stack_id", "fold", "kept_rec_2019", "kept_rec_6y", "recorded_class",
                "records_bin", "detection_class", "holes", "recorded_holes_strict", "recorded_holes_pooled", "all_rec_2019"]
        head += [f"auc_{m_}" for m_ in SCORE_M] + [f"hr2_{q}_{m_}" for m_ in R2_M for q in ("mean", "lo", "hi")]
        wr.writerow(head)
        for i in range(n):
            row = [i, wname[i], sp_t[i], status[i], int(idx["repeat"][i]), int(stid[i]), int(fold[i]), int(rec_t[i]), int(rec6[i]),
                   TCLS[rcls[i]], BINS[rbin[i]], DLAB[dcls[i]], int(nh[i]), int(y[i]), int(yp[i]), int(rec_all[i])]
            row += ["" if np.isnan(A[m_][i]) else f"{A[m_][i]:.6f}" for m_ in SCORE_M]
            row += [f"{r2[m_]['cf'][q][i]:.3f}" for m_ in R2_M for q in ("mean", "L", "U")]
            wr.writerow(row)
    with open(out_dir / "per_window_h2b.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(wrows[0])); wr.writeheader(); wr.writerows(wrows)
    clean = lambda o_: json.loads(json.dumps(o_, default=lambda v: v.item() if hasattr(v, "item") else str(v)).replace("NaN", "null"))
    rep.update(gates=res, n=int(n), stacks=len(np.unique(stid)), rankable=int(rank.sum()), rankable_stacks=len(np.unique(stid[rank])),
               hole_share=dict(strict=sh_s, pooled=sh_p), floor=floor,
               md5=dict(tool=X.file_md5(__file__), predicted=mpred, scores=X.file_md5(out_dir / "h2b_scores.npz"),
                        h2a_report=X.file_md5(h2a / "h2a_report.json"), frozen=mf, h1_tool=js1["md5"]["tool"], best_pt=best),
               H_R1={m_: dict(verdict=v["verdict"], **{b: dict(verdict=v[b]["verdict"], rows=v[b]["rows"]) for b in ("B1", "B1x")}) for m_, v in r1.items()},
               H_R1_baselines={m_: dict(verdict=v["verdict"], rows=v["B1"]["rows"]) for m_, v in r1b.items()},
               H_R1_pooled_truth={m_: dict(verdict=v["verdict"], **{b: dict(verdict=v[b]["verdict"], rows=v[b]["rows"]) for b in ("B1", "B1x")}) for m_, v in r1p.items()},
               H_R2={m_: dict(chosen[m_], rows=r2[m_]["rows"]) for m_ in R2_M},
               recall=dict(strict={m_: rc[m_] for m_ in R2_M}, strict_cells=ncs, pooled={m_: rcp[m_] for m_ in R2_M}, pooled_cells=ncp,
                           bins=list(DIST)),
               survey_count_ratio=allrows, eoo=eoorows, red_list=by_status, seven_nov=dict(hr1=r10, recall={m_: rc10[m_] for m_ in R2_M}),
               minutes=round((time.time() - t0) / 60, 1), written_utc=now())
    (out_dir / "h2b_report.json").write_text(json.dumps(clean(rep), indent=1))
    n_ok = sum(res.values())
    out(f"   written h2b_scores.npz md5 {rep['md5']['scores']}, per_stack_h2b.csv, per_window_h2b.csv, h2b_report.json "
        f"({(time.time() - t0) / 60:.1f} min)")
    out(f"stage H2b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; H-R1 (both B1 versions) seed 1 {r1['s1']['verdict']}, seed 2 "
        f"{r1['s2']['verdict']}; H-R2 " + ", ".join(f"{NAME[m_]} {'PASS' if chosen[m_]['ok'] else 'FAIL'}" for m_ in R2_M))
    return res, rep


# ---------------------------------------------------------------- self-test
def build_world(tmp, tools):
    """G5c's planted pipeline (2 tiny runs with step 3a and G1, G2), G3 on its real_val, a planted H1 export (6 fit windows)
    and the three H2a parts plus the report."""
    quiet = lambda *a, **k: None
    tmp = Path(tmp)
    w = P5.build_world(tmp, tools)
    data, g2, runs = w["data"], w["g2"], w["runs"]
    r3, _ = GC.run(data, runs, g2, data / "stage_g3_calibration", out=quiet)
    H1 = _load("stage_h1_bnm_export")
    py, refpat = H1.world(tmp / "bw")
    bnm = data / "bnm_fit"
    r1, _ = H1.run(py, refpat, tools, bnm, expect=(6, 6), out=quiet)
    h2a = data / "stage_h2_predict"
    ra = [H2A.part_baselines(data, bnm, g2, h2a, out=quiet)] + [H2A.part_samples(data, bnm, r, h2a, "cpu", out=quiet) for r in runs]
    ra.append(H2A.part_report(bnm, runs, h2a, out=quiet))
    ok = all(r3.values()) and all(r1.values()) and all(all(x.values()) for x in ra)
    return dict(data=data, bnm=bnm, h2a=h2a, runs=runs, g3=data / "stage_g3_calibration", ok=ok,
                txt=f"G3 {all(r3.values())}, H1 {all(r1.values())}, H2a parts {[all(x.values()) for x in ra]}")


def unit_checks(t):
    g = np.random.default_rng(5)
    # U1 stack-first mean and floor
    stack = np.array([0] * 5 + [1]); grp = np.zeros(6, int); win = np.array(["a"] * 5 + ["b"])
    d = np.array([1.0] * 5 + [0.0])
    cb = hr1_core(d, stack, grp, win, [0])
    t("U1", abs(cb[0]["mean"] - 0.5) < 1e-12 and cb[0]["n"] == 2,
      f"stack-first mean: a stack with 5 repeats at 1 and a stack with 1 repeat at 0 give {cb[0]['mean']:.3f} over {cb[0]['n']} stacks "
      "(the stack-repeat mean would be 0.833)")
    # U2 F5 verdicts on planted d
    ns, nw = 120, 12
    stack = np.repeat(np.arange(ns), 5); win = (stack % nw).astype(str); grp = np.repeat(np.arange(ns) % 2, 5)
    up = hr1_rows(hr1_core(np.full(len(stack), 0.01), stack, grp, win, [0, 1]), ["a", "b"], FLOOR)
    dn = hr1_rows(hr1_core(np.full(len(stack), -0.01), stack, grp, win, [0, 1]), ["a", "b"], FLOOR)
    small = hr1_rows(hr1_core(np.full(len(stack), -0.01), stack, grp, win, [0, 1]), ["a", "b"], 61)
    t("U2", overall([up]) == "PASS" and overall([dn]) == "FAIL" and overall([small]) == "no group tested" and up[0]["n"] == 60,
      f"H-R1 rule: d +0.01 in every stack-repeat {overall([up])}, -0.01 {overall([dn])}; 60 stacks per group (300 stack-repeats) are "
      f"under a floor of 61: {overall([small])}")
    # U3 crossfit_h = G3 crossfit with one item per stack; distinct-stack fallback otherwise
    n = 600
    pred = g.poisson(5, (n, 50)).astype(float); y = g.poisson(5, n).astype(float)
    fold = g.integers(0, 2, n); rc = g.integers(0, 3, n); rb = g.integers(0, 5, n)
    fg, pa = rc * 5 + rb, rc
    a_ = GC.crossfit(pred, y, np.zeros(n), fold, fg, pa, "aoo")
    b_ = crossfit_h(pred, y, np.zeros(n), fold, fg, pa, "aoo", np.arange(n))
    same = all(np.array_equal(a_[q], b_[q]) for q in ("mean", "L", "U"))
    n2 = 2000                                             # 400 stacks x 5 repeats, folds by stack, one recorded class
    st = np.arange(n2) // 5
    pred2 = g.poisson(5, (n2, 50)).astype(float); y2 = g.poisson(6, n2).astype(float)
    fold2 = st % 2; rc2 = np.zeros(n2, int)
    rb2 = np.where(st < 16, 0, 1 + st % 4)               # cell 0: 8 stacks (40 stack-repeats) per fold; cells 1-4: 48 stacks
    c_ = crossfit_h(pred2, y2, np.zeros(n2), fold2, rc2 * 5 + rb2, rc2, "aoo", st)
    d_ = GC.crossfit(pred2, y2, np.zeros(n2), fold2, rc2 * 5 + rb2, rc2, "aoo")
    c0, rest = rb2 == 0, rb2 > 0
    par_cls = np.array([GC.fit_params(pred2, y2, np.nonzero(fold2 == 1 - f_)[0]) for f_ in fold2[c0]])
    falls = np.allclose(c_["par"][c0], par_cls, rtol=0, atol=0) and not np.array_equal(c_["par"][c0], d_["par"][c0]) \
        and all(np.array_equal(c_[q][rest], d_[q][rest]) for q in ("mean", "L", "U"))
    t("U3", same and falls, f"H-R2 correction: with one item per stack crossfit_h = stage_g_calibrate.crossfit exactly {same}; a fitting "
      f"cell of 40 stack-repeats from 8 stacks uses its class fit here (the item-count version keeps the cell), cells of 48 stacks "
      f"are the same in both {falls}")
    # U4 thinning
    U = uniforms(range(300), 50); U2 = np.concatenate([uniforms(range(0, 7), 50), uniforms(range(7, 300), 50)])
    occ = g.random((300, 50, G, G)) < 0.5
    p_ = np.array(DET)[np.arange(300) % 3]
    th = occ & (U < p_[:, None, None, None])
    zs = [((th[np.arange(300) % 3 == c].sum() / occ[np.arange(300) % 3 == c].sum()) - DET[c]) /
          np.sqrt(DET[c] * (1 - DET[c]) / occ[np.arange(300) % 3 == c].sum()) for c in range(3)]
    t("U4", np.array_equal(U, U2) and all(abs(z) <= 4 for z in zs), f"thinning uniforms keyed by stack-repeat give the same values in "
      f"other chunks {np.array_equal(U, U2)}; thinned share per detection class within 4 se of the signed value (z {', '.join(f'{z:+.2f}' for z in zs)})")
    # U5 distance bins
    db = dist_bin(np.array([1.0, 1.41, 2.0, 2.24, 3.49, 3.61, 6.4, 6.5, 7.0]))
    t("U5", db.tolist() == [0, 0, 1, 1, 1, 2, 2, 3, 3], f"distance rounded to the nearest whole cell: 1, 1.41 -> 1; 2, 2.24, 3.49 -> 2-3; "
      f"3.61, 6.4 -> 4-6; 6.5, 7 -> 7+ ({db.tolist()})")


def self_test(tools):
    quiet = lambda *a, **k: None
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    unit_checks(t)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        w = build_world(tmp, tools)
        t("S0", w["ok"], f"planted pipeline: {w['txt']}")
        kw = dict(data=w["data"], bnm=w["bnm"], h2a=w["h2a"], runs=w["runs"], g3=w["g3"], floor=5)
        lines = []
        r, rep = run(out_dir=tmp / "s1", out=lines.append, **kw)
        t("S1", all(r.values()) and set(r) == {f"Y{k}" for k in range(7)} and "H_R1" in rep and "H_R2" in rep,
          f"H2b on the planted export (floor 5 so that groups are tested): {r}")
        # S2 independent recomputation from the files
        fb = dict(np.load(w["bnm"] / "bnm_frozen.npz")); n = len(fb["row"])
        man = {r_["input_file"]: r_ for r_ in D.rows_of(w["bnm"])}
        H = np.zeros((n, G, G), bool); Ls = np.zeros_like(H)
        for i in np.unique(fb["file_index"]):
            k = np.nonzero(fb["file_index"] == i)[0]
            z = np.load(w["bnm"] / man[str(fb["files"][i])]["truth_file"])
            H[k] = X.unpack(z["hole"])[fb["row"][k]]; Ls[k] = X.unpack(z["label_strict"])[fb["row"][k]]
        sm = np.load(w["h2a"] / f"samples_{w['runs'][0].name}.npz")["maps"]
        occ = GC.target_year(sm)
        cnt = occ.sum(1) / 50
        sc = np.load(tmp / "s1" / "h2b_scores.npz")
        V1, V2 = GB.load_obs(w["bnm"], fb["files"], fb["file_index"], fb["row"])[:2]
        V = V1 | V2
        rec6 = V.any(1).reshape(n, -1).sum(1)
        p_ = np.array(DET)[det_class(rec6)]
        pr = np.load(tmp / "s1" / "h2b_predicted.npz")
        pick = np.sort(np.random.default_rng(3).choice(n, min(60, n), replace=False))
        auc_ok, y_ok, th_ok = True, True, True
        for i in pick:
            lab = Ls[i][H[i]]; s_ = cnt[i][H[i]]
            a_ = X.auc(s_, lab) if 0 < lab.sum() < lab.size else np.nan
            auc_ok &= (np.isnan(a_) and np.isnan(sc["auc_strict_s1"][i])) or abs(a_ - sc["auc_strict_s1"][i]) < 1e-12
            y_ok &= int(lab.sum()) == int(sc["y_strict"][i])
            U = X.rng(KEY, 1, int(i)).random((50, G * G)).reshape(50, G, G)
            th_ok &= np.array_equal((occ[i] & (U < p_[i]) & H[i]).sum((1, 2)), pr["thin_h_s1"][i])
        g0 = rep["H_R1"]["s1"]["B1"]["rows"]["recorded class"]
        stid = np.load(w["bnm"] / "bnm_stacks.npz")["stack_id"]
        dd = sc["auc_strict_s1"] - sc["auc_strict_B1"]
        rcls = D.tidx(V[:, -1].reshape(n, -1).sum(1))
        mean_ok = True
        for j, row in enumerate(g0):
            m = (rcls == j) & ~np.isnan(dd)
            if m.any():
                st = {}
                for s_, v_ in zip(stid[m], dd[m]):
                    st.setdefault(s_, []).append(v_)
                mean_ok &= abs(np.mean([np.mean(v) for v in st.values()]) - row["mean"]) < 1e-9 and len(st) == row["n"]
        t("S2", auc_ok and y_ok and th_ok and mean_ok, f"recomputed outside the tool from the truth and map files on {len(pick)} stack-repeats: "
          f"AUC seed 1 (export_obj2.auc) {auc_ok}, recorded holes {y_ok}, thinned counts {th_ok}; stack-first group means and stack "
          f"counts of seed 1 - B1 by recorded class {mean_ok}")
        # S3 refusals
        stop = []
        bt = tmp / "bnm_test"; shutil.copytree(w["bnm"], bt)
        j_ = json.loads((bt / "h1_report.json").read_text()); j_["half"] = "test"; (bt / "h1_report.json").write_text(json.dumps(j_))
        for f in (lambda: run(out_dir=tmp / "s1", out=quiet, **kw), lambda: run(out_dir=w["bnm"] / "x", out=quiet, **kw),
                  lambda: run(out_dir=w["h2a"] / "x", out=quiet, **kw), lambda: run(out_dir=tmp / "real_eval_x", out=quiet, **kw),
                  lambda: run(out_dir=tmp / "s3", out=quiet, **dict(kw, bnm=bt))):
            try:
                f(); stop.append(False)
            except SystemExit:
                stop.append(True)
        t("S3", all(stop), f"refused: output folder in use {stop[0]}, inside the H1 folder {stop[1]}, inside the H2a folder {stop[2]}, "
          f"a real_eval path {stop[3]}, a test-half H1 export {stop[4]}")
        # planted bugs
        for bug, g_ in (("p1_open", "Y5"), ("label_early", "Y5"), ("holes_shift", "Y3"), ("thin_p", "Y3"), ("corr_scale", "Y2"),
                        ("core_shift", "Y2"), ("labels_shift", "Y4"), ("leak", "Y6"), ("fold_stack", "Y6"), ("pred_changed", "Y5")):
            os.environ["STAGE_H2B_BROKEN"] = bug
            try:
                rb, _ = run(out_dir=tmp / f"b_{bug}", out=quiet, **kw)
            finally:
                os.environ.pop("STAGE_H2B_BROKEN", None)
            t(f"bug {bug}", rb.get(g_) is False, f"{g_} fails as it must ({rb})")
        o = tmp / "h2a_tamper"
        shutil.copytree(w["h2a"], o)
        rw = _load("stage_g5_export_check").rewrite
        rw(o / f"samples_{w['runs'][0].name}.npz", lambda a: a.__setitem__("maps", a["maps"] ^ np.uint8(1)))
        rt, _ = run(out_dir=tmp / "b_tamper", out=quiet, **dict(kw, h2a=o))
        t("bug tamper", rt.get("Y0") is False and len(rt) == 1, f"a samples file changed after H2a fails Y0 and stops ({rt})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--bnm", default="../data/bnm_fit")
    ap.add_argument("--h2a", default="../data/stage_h2_predict")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    ap.add_argument("--out", default="../data/stage_h2b_score")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    r, _ = run(a.data, a.bnm, a.h2a, a.runs, a.g3, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)