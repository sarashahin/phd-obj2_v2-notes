#!/usr/bin/env python3
"""
stage_f0_spread_check.py  (obj2_v2, fix step F0, gate A, 13 Nov 2026; Sara's yes on 12 Nov; report only)
Version 2 (13 Nov): --g2 and --g3 are arguments (version 1 took them as ../data/stage_g2_baselines_v2 and
../data/stage_g3_calibration, which holds on Sara's machine but not in the planted self-test, whose T1 then failed A0);
nothing else changed, so version 2 gives version 1's scores on the real data.

Question: on the BNM fit half (the development half), does a correction of the model's over-spread (R18) bring the
sparsely recorded groups up to B1 in H-R1, out of fold, while the well-recorded groups keep their gain? The answer decides
whether option A (thinning each sample's occupied cells by distance to the nearest kept record) is worth building. The
model, its samples, the simulation and every frozen file stay as they are. Nothing of the test half, real_eval or their
labels is read.

Correction
    score'(cell) = score(cell) x d^-gamma, with d the distance in cells from the hole tetrad to the nearest kept record
    (h2b_predicted.npz dist_h; at least 1) and gamma set per kept recorded class at 2019 (7 classes) and per seed, on the
    grid 0, 0.1, ..., 4.0. score' is the expected share of a sample map after its occupied cells are kept with chance
    d^-gamma (1 at d = 1): it adds no occupancy, and cells no sample occupies stay tied at 0.
    gamma of a class = the grid value with the highest stack-first mean AUC over the rankable stack-repeats of that class in
    one window fold (ties: the smallest gamma). Folds are H2b's (make_folds on the 44 window names, 22 / 22). The gamma
    fitted on one fold is applied to the other, so every scored AUC is out of fold. gamma 0 gives the H2b scores.

Rule (written before the run, 13 Nov)
    F0 gate A "option A worth building" = YES when, for seed 1 and seed 2, against B1 stored and B1 exact, with the
    out-of-fold scores and H2b's H-R1 code (stack-first means, 2,000 resamples of the 44 windows, seed 31, floor from the
    freeze): (i) no group of either grouping FAILs, and (ii) recorded classes 21-50 and 51+ and records 21+ keep a mean of 0
    or above. Otherwise NO. Report only: Sara decides. Reported beside it: the gamma table, the uncorrected rows, and per
    recorded class the share of predicted occupied hole cells the correction keeps, next to the frozen H3 scale.

Gates (if A0-A3 fail, nothing is printed past them and nothing is written)
    A0  inputs: h3_freeze.json md5 = signed (5516a6fc); every tool and data md5 listed in it = the files now (H6's
        freeze_check); stage_h6_bnm_test_score.py = 8b9a38dd (its freeze_check, H2b loader and row compare are used);
        h2b_report.json gates all PASS
    A1  alignment: recorded holes per stack-repeat from the fit-half truth files (H2b's read_labels) = h2b y_strict; AUC of
        seed 1, seed 2, B1 stored, B1 exact from the cell data = h2b_scores.npz (1e-12, same not-rankable set);
        dist_h = H2a's B4 distance at the holes; every distance 1 or more; each window in one fold, fold sizes 22 / 22
    A2  code check: with gamma 0 the H-R1 rows of both seeds against both B1 versions = h2b_report.json (1e-9, verdicts)
    A3  folds: the gamma applied to a fold is unchanged when every AUC of that fold is replaced by random values
    A4  np.load log: only files of bnm_fit, stage_h2_predict and stage_h2b_score; a path of the test half, H5, H6 or
        real_eval is refused

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_f0_spread_check.py --self-test 2>&1 | tail -n 1
    python3 -u ../tools/stage_f0_spread_check.py > ../logs/stage_f0_spread_check.txt 2>&1
Needs stage_h6_bnm_test_score.py and what it loads (stage_h2_bnm_score.py, stage_h3_freeze.py,
stage_h3b_sparse_diagnostic.py) in the same folder; the self-test lives in ../tests/test_25_f0a_spread_check.py and uses
../tests/test_23_h6_test_score.py.
"""
import argparse, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_f0_spread_check.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


H6 = _load("stage_h6_bnm_test_score")
S2, X, D = H6.S2, H6.X, H6.D
TCLS, BINS = S2.TCLS, S2.BINS
FREEZE_MD5 = H6.FREEZE_MD5                           # Sara's H3 run, 12 Nov
H6_MD5 = "8b9a38ddab5b23abdca92c3c9de527cc"         # stage_h6_bnm_test_score.py as run on 12 Nov
GRID = np.round(np.arange(0.0, 4.0001, 0.1), 1)
SEEDS, REFS = ("s1", "s2"), ("B1", "B1x")
NAME = dict(s1="seed 1", s2="seed 2", B1="B1 stored", B1x="B1 exact")
DENSE = {"recorded class": ("21-50", "51+"), "records over 6 years": ("21+",)}
ALLOWED = ("bnm_fit", "stage_h2_predict", "stage_h2b_score")
FORBID = ("bnm_test", "stage_h5_predict", "stage_h6_test_score", "real_eval", "STAGE_H6_SCORED", "BNM_TEST_EXPORTED")


def broken():
    return os.environ.get("STAGE_F0A_BROKEN", "")    # self-test only: plants one bug so its gate must fail


class LoadLog:
    """Every np.load path is logged; a path of the test half, H5, H6 or real_eval is refused before it is opened."""
    def __enter__(self):
        self.orig, self.paths = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        if isinstance(f, (str, Path)):
            p = str(Path(f).resolve())
            self.paths.append(p)
            if any(t in p for t in FORBID):
                raise RuntimeError(f"refused np.load of {p}")
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- pieces
def load_fit(bnm, h2a, h2b):
    """Fit-half holes: labels (H2b's read_labels), scores, shares, distance, folds, groups, windows."""
    fb = dict(np.load(bnm / "bnm_frozen.npz"))
    man = {r["input_file"]: r for r in D.rows_of(bnm)}
    n = len(fb["row"])
    with np.load(h2b / "h2b_predicted.npz") as P:
        ptr = P["hole_ptr"].astype(np.int64); cell = P["hole_cell"].astype(np.int64)
        F = dict(dist=P["dist_h"].astype(float), fold=P["fold"].astype(int), rcls=P["rcls"].astype(int), rbin=P["rbin"].astype(int),
                 stid=P["stack_id"].astype(np.int64), score={m: P[f"score_{m}"].astype(float) for m in SEEDS + REFS},
                 share={m: P[f"share_{m}"].astype(float) for m in SEEDS})
    item = np.repeat(np.arange(n), np.diff(ptr))
    Ls = S2.read_labels(bnm, fb, man)[0]
    if broken() == "labels_shift":
        Ls = np.roll(Ls, 1, 0)
    with np.load(h2a / "baselines_bnm.npz") as bz:
        b4d = np.asarray(bz["B4_dist"]).reshape(n, -1)[item, cell]
    with np.load(h2b / "h2b_scores.npz") as Sc:
        F.update(y_strict=Sc["y_strict"].astype(float), auc_ref={m: Sc[f"auc_strict_{m}"] for m in SEEDS + REFS})
    with np.load(bnm / "bnm_stacks.npz") as st:
        win = st["window"].astype(str)
    F.update(n=n, ptr=ptr, item=item, lab=Ls.reshape(n, -1)[item, cell], b4d=b4d.astype(float), win=win)
    return F


def sf_mean(a, stid, mask):
    """Stack-first mean (a stack's repeats averaged first) over the rankable items in mask."""
    ok = mask & ~np.isnan(a)
    if not ok.any():
        return float("nan")
    _, inv = np.unique(stid[ok], return_inverse=True)
    inv = np.asarray(inv).ravel()
    return float((np.bincount(inv, weights=a[ok]) / np.bincount(inv)).mean())


def grid_aucs(F, m):
    """Per stack-repeat AUC (H2b's item_aucs) of score x d^-gamma for every gamma of the grid: (len(GRID), n)."""
    ld = np.log(np.maximum(F["dist"], 1.0))
    return np.stack([S2.item_aucs(F["ptr"], F["score"][m] * np.exp(-g * ld), F["lab"]) for g in GRID])


def fit_gamma(Ag, F, fold_fit):
    """gamma per recorded class from the items of one fold: highest stack-first mean AUC, ties to the smallest gamma."""
    gam = np.zeros(len(TCLS))
    for c in range(len(TCLS)):
        m = (F["rcls"] == c) & ((F["fold"] == fold_fit) if broken() != "leak" else True)
        v = np.array([sf_mean(Ag[j], F["stid"], m) for j in range(len(GRID))])
        gam[c] = GRID[int(np.nanargmax(v))] if np.isfinite(v).any() else 0.0
    return gam


def oof(Ag, F, gams):
    """Out-of-fold AUC per item: fold f uses the gamma fitted on the other fold. Also the gamma used per item."""
    A, g_item = np.full(F["n"], np.nan), np.zeros(F["n"])
    for f in (0, 1):
        g = gams[1 - f]
        for c in range(len(TCLS)):
            m = (F["fold"] == f) & (F["rcls"] == c)
            A[m] = Ag[int(np.flatnonzero(GRID == g[c])[0])][m]
            g_item[m] = g[c]
    return A, g_item


def hr1(Am, Ab, F, floor):
    """H2b's H-R1 closure for one model - reference pair (stack-first, window resampling, H2b's rows and verdicts)."""
    grp = {"recorded class": (F["rcls"], TCLS), "records over 6 years": (F["rbin"], BINS)}
    d = Am - Ab + (1e-6 if broken() == "core_shift" else 0.0)
    rows = {g: S2.hr1_rows(S2.hr1_core(d, F["stid"], gi, F["win"], list(range(len(labs)))), labs, floor) for g, (gi, labs) in grp.items()}
    return dict(rows=rows, verdict=S2.overall(rows.values()))


def same_set(a, b):
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb)), float(np.max(np.abs(a[~na] - b[~na]))) if (~na).any() else 0.0


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- run
def run(bnm="../data/bnm_fit", h2a="../data/stage_h2_predict", h2b="../data/stage_h2b_score",
        freeze="../data/stage_h3_freeze/h3_freeze.json", runs=("../runs/f_seed1", "../runs/f_seed2"), obj1_tools="../../obj1_v2/tools",
        out_dir="../data/stage_f0_spread_check", g2="../data/stage_g2_baselines_v2", g3="../data/stage_g3_calibration",
        freeze_md5=FREEZE_MD5, h6_md5=H6_MD5, floor_override=None, out=print):
    bnm, h2a, h2b, freeze, out_dir = Path(bnm), Path(h2a), Path(h2b), Path(freeze), Path(out_dir)
    runs, data = [Path(r) for r in runs], bnm.parent
    o = out_dir.resolve()
    for p_ in (bnm, h2a, h2b):
        if o == p_.resolve() or p_.resolve() in o.parents:
            sys.exit(f"STOP: nothing is written inside {p_}; give another --out")
    if any(t in str(o) for t in FORBID):
        sys.exit("STOP: the output folder may not lie in a test-half, H5, H6 or real_eval folder")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    t0, res, rep = time.time(), {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage F0 gate A: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, rep

    out(f"stage_f0_spread_check.py md5 {X.file_md5(__file__)}; stage_h6_bnm_test_score.py {X.file_md5(HERE / 'stage_h6_bnm_test_score.py')}; "
        f"stage_h2_bnm_score.py {X.file_md5(HERE / 'stage_h2_bnm_score.py')}")
    # ---- A0
    fz = json.loads(freeze.read_text())
    fmd5 = X.file_md5(freeze)
    paths = dict(bnm_fit=bnm, stage_h2_predict=h2a, stage_h2b_score=h2b, stage_g2_baselines_v2=Path(g2), stage_g3_calibration=Path(g3))
    bad, nf = H6.freeze_check(fz, obj1_tools, paths, runs)
    h6ok = X.file_md5(HERE / "stage_h6_bnm_test_score.py") == h6_md5
    jb = json.loads((h2b / "h2b_report.json").read_text())
    jb_ok = bool(jb.get("gates")) and all(jb["gates"].values())
    floor = int(floor_override) if floor_override is not None else int(fz.get("settings", {}).get("floor", 50))
    gate("A0", fmd5 == freeze_md5 and not bad and h6ok and jb_ok,
         f"h3_freeze.json md5 {fmd5} = signed {fmd5 == freeze_md5}; {nf} md5s listed = the files now (differ: {bad}); "
         f"stage_h6_bnm_test_score.py = 8b9a38dd {h6ok}; h2b_report.json gates all PASS {jb_ok}; rule floor {floor} distinct stacks"
         + (" (override, self-test only)" if floor_override is not None else ""))
    if not res["A0"]:
        return stop("A0")
    with LoadLog() as log:
        try:
            if broken() == "test_path":
                np.load(data / "bnm_test" / "bnm_frozen.npz")
            F = load_fit(bnm, h2a, h2b)
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("A4", False, f"np.load log: {crashed}")
        return stop("A4")
    # ---- A1
    n = F["n"]
    y = np.add.reduceat(F["lab"].astype(int), F["ptr"][:-1]).astype(float)
    y_ok = np.array_equal(y, F["y_strict"])
    A0g = {m: S2.item_aucs(F["ptr"], F["score"][m], F["lab"]) for m in SEEDS + REFS}
    chk = {m: same_set(A0g[m], F["auc_ref"][m]) for m in A0g}
    auc_ok = all(s and w <= 1e-12 for s, w in chk.values())
    dist_ok = np.array_equal(F["dist"].astype(np.float32), F["b4d"].astype(np.float32)) and float(F["dist"].min()) >= 1.0
    wf = {w_: set(F["fold"][F["win"] == w_].tolist()) for w_ in np.unique(F["win"])}
    nw = [sum(1 for s_ in wf.values() if s_ == {f}) for f in (0, 1)]
    fold_ok = all(len(s_) == 1 for s_ in wf.values()) and abs(nw[0] - nw[1]) <= 1
    gate("A1", y_ok and auc_ok and dist_ok and fold_ok,
         f"{n:,} fit-half stack-repeats, {len(F['lab']):,} hole tetrads; recorded holes per stack-repeat = h2b y_strict {y_ok}; AUC of "
         + ", ".join(f"{NAME[m]} {chk[m][1]:.1e}" for m in chk) + f" (same not-rankable set {all(s for s, _ in chk.values())}); "
         f"dist_h = H2a B4 distance at the holes and 1 or more {dist_ok} (min {F['dist'].min():.2f}); each window in one fold, "
         f"{nw[0]} / {nw[1]} windows {fold_ok}")
    if not res["A1"]:
        return stop("A1")
    # ---- A2
    base = {m: {b: hr1(A0g[m], A0g[b], F, floor) for b in REFS} for m in SEEDS}
    worst, same_v = 0.0, True
    for m in SEEDS:
        for b in REFS:
            same_v &= base[m][b]["verdict"] == jb["H_R1"][m][b]["verdict"]
            for g in base[m][b]["rows"]:
                worst = max(worst, H6.rows_differ(base[m][b]["rows"][g], jb["H_R1"][m][b]["rows"][g]))
    gate("A2", worst <= 1e-9 and same_v, f"gamma 0: H-R1 rows of seed 1 and seed 2 against both B1 versions = h2b_report.json, "
         f"largest difference {worst:.1e}, verdicts same {same_v}")
    if not res["A2"]:
        return stop("A2")
    # ---- grid and fits
    Ag = {m: grid_aucs(F, m) for m in SEEDS}
    gams = {m: {f: fit_gamma(Ag[m], F, f) for f in (0, 1)} for m in SEEDS}
    # ---- A3
    rng = np.random.default_rng(20261113)
    a3 = True
    for m in SEEDS:
        for f in (0, 1):
            Ar = Ag[m].copy()
            other = F["fold"] != f
            Ar[:, other] = rng.random((len(GRID), int(other.sum())))
            a3 &= np.array_equal(fit_gamma(Ar, F, f), gams[m][f])
    gate("A3", a3, "the gamma fitted on each fold, for both seeds, is unchanged when every AUC of the other fold is replaced by random values")
    if not res["A3"]:
        return stop("A3")
    gate("A4", not any(any(t in p for t in FORBID) for p in log.paths) and all(any(f"/{a}/" in p for a in ALLOWED) for p in log.paths),
         f"np.load log: {len(log.paths)} files, all in bnm_fit, stage_h2_predict or stage_h2b_score; test half, H5, H6, real_eval 0")
    # ---- results
    edge = sum(int(g_ == GRID[-1]) for m in SEEDS for f in (0, 1) for g_ in gams[m][f])
    out(f"   gamma per kept recorded class at 2019 (fitted on fold 0 -> applied to fold 1 | fitted on fold 1 -> applied to fold 0; "
        f"at the grid edge {GRID[-1]:.1f}: {edge} of {4 * len(TCLS)}):")
    for m in SEEDS:
        out(f"      {NAME[m]:<7} " + "; ".join(f"{TCLS[c]} {gams[m][0][c]:.1f} | {gams[m][1][c]:.1f}" for c in range(len(TCLS))))
    Ao, gi = {}, {}
    for m in SEEDS:
        Ao[m], gi[m] = oof(Ag[m], F, gams[m])
    corr = {m: {b: hr1(Ao[m], A0g[b], F, floor) for b in REFS} for m in SEEDS}
    out(f"   H-R1 out of fold, corrected model - B1, stack-first mean [95% interval, {len(np.unique(F['win']))} windows resampled] verdict "
        f"(distinct stacks) {{uncorrected mean}}:")
    verdict = {}
    for m in SEEDS:
        for b in REFS:
            for g, rows in corr[m][b]["rows"].items():
                b0 = {r["group"]: r for r in base[m][b]["rows"][g]}
                out(f"      {NAME[m]} vs {NAME[b]:<9} {g:<20} " + "; ".join(
                    f"{r['group']} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['verdict']} ({r['n']}) {{{b0[r['group']]['mean']:+.4f}}}"
                    if r["n"] else f"{r['group']} none" for r in rows))
            out(f"      {NAME[m]} vs {NAME[b]}: {corr[m][b]['verdict']} (uncorrected {base[m][b]['verdict']})")
        verdict[m] = S2.both(corr[m]["B1"]["verdict"], corr[m]["B1x"]["verdict"])
    no_fail = all(corr[m][b]["verdict"] != "FAIL" for m in SEEDS for b in REFS)
    dense = all(r["mean"] >= 0 for m in SEEDS for b in REFS for g, labs in DENSE.items() for r in corr[m][b]["rows"][g]
                if r["group"] in labs and r["tested"])
    # ---- count report: share of predicted occupied hole cells kept, against the frozen H3 scale
    out("   count (report): share of predicted occupied hole cells kept by the correction (out of fold) | frozen H3 scale s, by recorded class:")
    keep = {}
    for m in SEEDS:
        k_cell = np.exp(-gi[m][F["item"]] * np.log(np.maximum(F["dist"], 1.0)))
        sh = F["share"][m]
        keep[m] = []
        for c in range(len(TCLS)):
            mc = F["rcls"][F["item"]] == c
            s_ = fz.get("fits", {}).get(m, {}).get("groups", {}).get(str(c), {}).get("s")
            keep[m].append(dict(group=TCLS[c], kept=float((sh[mc] * k_cell[mc]).sum() / sh[mc].sum()) if sh[mc].sum() > 0 else float("nan"),
                                scale=None if s_ is None else float(s_)))
        out(f"      {NAME[m]:<7} " + "; ".join(f"{r['group']} {r['kept']:.2f} | " + ("-" if r["scale"] is None else f"{r['scale']:.2f}") for r in keep[m]))
    yes = no_fail and dense
    out(f"   rule: (i) no group FAILs for both seeds against both B1 versions {no_fail}; (ii) classes 21-50, 51+ and records 21+ keep a mean "
        f"of 0 or above {dense}")
    # ---- files
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = dict(fold=F["fold"], stack_id=F["stid"], rcls=F["rcls"], rbin=F["rbin"], grid=GRID)
    for m in SEEDS:
        sc[f"auc_oof_{m}"], sc[f"gamma_item_{m}"] = Ao[m], gi[m]
        sc[f"gamma_fold0_{m}"], sc[f"gamma_fold1_{m}"] = gams[m][0], gams[m][1]
    X.save_npz(out_dir / "f0a_scores.npz", sc)
    clean = lambda o_: json.loads(json.dumps(o_, default=lambda v: v.item() if hasattr(v, "item") else str(v)).replace("NaN", "null"))
    rep.update(gates=res, floor=floor, n=int(n), grid=GRID.tolist(),
               gamma={m: {f"fit_fold{f}": gams[m][f].tolist() for f in (0, 1)} for m in SEEDS}, gamma_at_edge=edge,
               H_R1_oof={m: dict(verdict=verdict[m], **{b: corr[m][b] for b in REFS}) for m in SEEDS},
               H_R1_uncorrected={m: {b: base[m][b] for b in REFS} for m in SEEDS}, count_keep=keep,
               rule=dict(no_fail=no_fail, dense_kept=dense, worth_building=yes),
               md5=dict(tool=X.file_md5(__file__), freeze=fmd5, scores=X.file_md5(out_dir / "f0a_scores.npz")),
               minutes=round((time.time() - t0) / 60, 1), written_utc=now())
    (out_dir / "f0a_report.json").write_text(json.dumps(clean(rep), indent=1))
    n_ok = sum(res.values())
    out(f"   written f0a_scores.npz md5 {rep['md5']['scores']}, f0a_report.json ({rep['minutes']} min)")
    out(f"stage F0 gate A: gates {n_ok} PASS, {len(res) - n_ok} FAIL; H-R1 out of fold seed 1 {verdict['s1']}, seed 2 {verdict['s2']}; "
        f"option A worth building by the rule written before the run: {'YES' if yes else 'NO'}")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bnm", default="../data/bnm_fit")
    ap.add_argument("--h2a", default="../data/stage_h2_predict")
    ap.add_argument("--h2b", default="../data/stage_h2b_score")
    ap.add_argument("--freeze", default="../data/stage_h3_freeze/h3_freeze.json")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--out", default="../data/stage_f0_spread_check")
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_25_f0a_spread_check.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found")
        spec = importlib.util.spec_from_file_location("test_25", tp); t25 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t25)
        sys.exit(0 if t25.self_test(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.bnm, a.h2a, a.h2b, a.freeze, a.runs, a.obj1_tools, a.out, a.g2, a.g3, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)