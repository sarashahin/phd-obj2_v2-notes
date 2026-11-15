#!/usr/bin/env python3
"""
stage_h3_freeze.py  (obj2_v2, stage H step H3, 12 Nov 2026; H rules signed off by Sara on 11 Nov, H2b points and the
H3-before-diagnostic order on 12 Nov, claude/obj2_v2_H_rules_draft.md)

Question: before the test half is opened (H4), fix everything the test half will be scored with. Refit the H-R2
correction (G3 level correction + split conformal 90% interval) on all 44 fit-half windows at the option each method was
given on the fit half, and write one md5 list of the model, sampler settings, fits, hole rule, data and every tool.
Reads only fit-half outputs: no BNM grid, no input or truth file, nothing of the test half.

Order
    F0  inputs as logged: H1 export fit half (gates PASS), h2a_report.json (gates PASS, truth opened 0), h2b_report.json
        (gates PASS; h2b_predicted.npz and h2b_scores.npz md5 = report; stage_h2_bnm_score.py md5 = report; best.pt md5 =
        report); no test-half export yet (no data/BNM_TEST_EXPORTED, no data/bnm_test)
    F1  settings: K, noise seed, DDIM steps and batch of the H2a samples = G1's samples_k50.npz of each run, best.pt md5 =
        G1's; detection 0.643 / 0.642 / 0.800, thinning key 31201, holes (3n + 5) // 10 (none when n < 2), H1 key 31101,
        5 repeats, floor 50 distinct stacks, fit-size fallback 30, interval 5th-95th percentile, level 0.90, band
        0.8-1.25, coverage 0.85-0.95, read from the tools and equal to the signed values
    F2  same correction code: the fit resolver of this tool, given each fold's fitting stack-repeats, reproduces H2b's
        cross-fitted scale and Q of every stack-repeat and its corrected mean, lower and upper end, for all 6 methods at
        their H2b option (largest difference 1e-12)
    F3  refit on all 44 windows (7,463 stack-repeats): scale and Q for every possible fitting group of each method, G3's
        fallback counted in distinct stacks (under 30: the recorded class for option 2, then all); in-sample ratio and
        coverage reported (not a test: the same stack-repeats fitted and checked)
    F4  files passed to np.load: only H2a samples files (scalars), G1 samples_k50.npz (scalars), h2b_predicted.npz and
        h2b_scores.npz; none from the BNM export folder, no *_truth file
    F5  freeze file h3_freeze.json: md5 of every tool module loaded by the H1, H2a, H2b and H3 code (stage_h1_bnm_export.py
        = the md5 in h1_report.json, stage_h2_bnm_predict.py = h2a_report.json's, stage_h2_bnm_score.py = h2b_report.json's)
        and of bnm_detection_check.py; md5 of best.pt, G1 samples, g2_fit.json, g3_calibrated.npz and every H1, H2a and H2b
        output; the settings; the fits. Written once into a new folder; built twice in memory, same bytes
Apply rule for H6 (written into the json): group g = recorded class (option 1) or recorded class x 5 + records bin (option
2); s, q = fits[method][g]; scaled = s x thinned counts of the 50 maps (pooled: first 25 of each seed); lower = max(5th
percentile - q, 0), upper = max(95th percentile + q, 0), mean = mean of the scaled counts.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, under a minute)
    python3 ../tools/stage_h3_freeze.py --self-test | tail -1      (needs ../tests/test_20_h3_freeze.py)
    python3 -u ../tools/stage_h3_freeze.py > ../logs/stage_h3_freeze.txt 2>&1
Defaults: --data ../data, --bnm ../data/bnm_fit, --h2a ../data/stage_h2_predict, --h2b ../data/stage_h2b_score,
          --runs ../runs/f_seed1 ../runs/f_seed2, --g2 ../data/stage_g2_baselines_v2, --g3 ../data/stage_g3_calibration,
          --obj1-tools ../../obj1_v2/tools, --out ../data/stage_h3_freeze.
Needs stage_h2_bnm_score.py and stage_h1_bnm_export.py (with the stage tools they load) in the same folder.
"""
import argparse, importlib.util, json, os, sys, tempfile, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_h3_freeze.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


S2 = _load("stage_h2_bnm_score")
H1 = _load("stage_h1_bnm_export")
GC, X = S2.GC, S2.X
TCLS, BINS, R2_M = S2.TCLS, S2.BINS, S2.R2_M
NB = len(BINS)
SIGNED = dict(det=[0.643, 0.642, 0.8], thin_key=31201, h1_key=31101, repeats=5, floor=50, n_fit_min=30, qlo=5, qhi=95,
              level=0.9, band=[0.8, 1.25], cov=[0.85, 0.95], holes={"1": 0, "2": 1, "15": 5, "35": 11, "86": 26})
SAMPLER = dict(K=50, seed=11, ddim=50, batch=1024)


def broken():
    return os.environ.get("STAGE_H3_BROKEN", "")       # self-test only: plants one bug so its gate must fail


class LoadLog:
    def __enter__(self):
        self.orig, self.paths = np.load, []

        def rec(f, *a, **k):
            self.paths.append(str(Path(f).resolve()) if isinstance(f, (str, Path)) else str(f))
            return self.orig(f, *a, **k)
        np.load = rec
        return self

    def __exit__(self, *e):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- the correction fit (G3 code path of H2b)
def group_of(rcls, rbin, option):
    return np.asarray(rcls, np.int64) if option == 1 else np.asarray(rcls, np.int64) * NB + np.asarray(rbin, np.int64)


def parent_of(g, option):
    return -1 if option == 1 else int(g) // NB


def resolve(pred, y, fit, groups, fgrp, option, stack):
    """Scale and Q for each group in groups, fitted on the stack-repeats in fit, exactly as stage_h2_bnm_score.crossfit_h:
    a group with fewer than N_FIT_MIN distinct stacks uses its parent (recorded class, option 2), then all fit items."""
    nst = lambda ix: len(np.unique(stack[ix]))
    parent = fgrp // NB if option == 2 else np.full(len(fgrp), -1)
    cache, tab = {}, {}
    for g in groups:
        fg = fit[fgrp[fit] == g]
        key = ("g", int(g))
        if nst(fg) < GC.N_FIT_MIN:
            pg = parent_of(g, option)
            fg, key = (fit, ("all",)) if pg < 0 else (fit[parent[fit] == pg], ("p", pg))
            if nst(fg) < GC.N_FIT_MIN:
                fg, key = fit, ("all",)
        if key not in cache:
            s, q = GC.fit_params(pred, y, fg)
            cache[key] = (float(s) * (1.001 if broken() == "scale" else 1.0), float(q), len(fg), nst(fg))
        tab[int(g)] = dict(key=key[0] if key[0] != "p" else f"p{key[1]}", s=cache[key][0], q=cache[key][1],
                           n_items=cache[key][2], n_stacks=cache[key][3])
    return tab


def apply(pred, ix, s, q):
    sc = s * pred[ix]
    lo, hi = np.percentile(sc, GC.QLO, axis=1), np.percentile(sc, GC.QHI, axis=1)
    return sc.mean(1), np.maximum(lo - q, 0.0), np.maximum(hi + q, 0.0)


def preds(P):
    pc = {m: P[f"thin_h_{m}"].astype(float) for m in S2.MAP_M}
    pc["pooled"] = np.concatenate([pc["s1"][:, :25], pc["s2"][:, :25]], 1)
    return pc


# ---------------------------------------------------------------- md5 lists
def tool_md5s():
    """md5 of every tool module reachable from the H1, H2b (and so H2a, G3) code and this tool, by file name."""
    out, seen, todo = {Path(__file__).name: X.file_md5(Path(__file__))}, set(), [S2, H1]
    while todo:
        m = todo.pop()
        f = getattr(m, "__file__", None)
        if not f or id(m) in seen:
            continue
        seen.add(id(m))
        p = Path(f).resolve()
        if p.parent == HERE:
            out[p.name] = X.file_md5(p)
        todo += [v for v in vars(m).values() if isinstance(v, types.ModuleType) and getattr(v, "__file__", None)
                 and Path(v.__file__).resolve().parent == HERE]
    return dict(sorted(out.items()))


def run(data="../data", bnm="../data/bnm_fit", h2a="../data/stage_h2_predict", h2b="../data/stage_h2b_score",
        runs=("../runs/f_seed1", "../runs/f_seed2"), g2="../data/stage_g2_baselines_v2", g3="../data/stage_g3_calibration",
        obj1_tools="../../obj1_v2/tools", out_dir="../data/stage_h3_freeze", out=print):
    data, bnm, h2a, h2b, g2, g3, od = map(Path, (data, bnm, h2a, h2b, g2, g3, out_dir))
    runs = [Path(r) for r in runs]
    if od.exists():
        sys.exit(f"STOP: {od} exists; the freeze is written once into a new folder")
    for p_ in (bnm, h2a, h2b):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside {p_}")
    res = {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage H3: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, None
    out(f"stage_h3_freeze.py md5 {X.file_md5(Path(__file__))}; stage_h2_bnm_score.py {X.file_md5(HERE / 'stage_h2_bnm_score.py')}; "
        f"stage_h1_bnm_export.py {X.file_md5(HERE / 'stage_h1_bnm_export.py')}; stage_g_calibrate.py {X.file_md5(HERE / 'stage_g_calibrate.py')}")
    with LoadLog() as log:
        # ---- F0
        j1 = json.loads((bnm / "h1_report.json").read_text())
        ja = json.loads((h2a / "h2a_report.json").read_text())
        jb = json.loads((h2b / "h2b_report.json").read_text())
        mb = jb["md5"]
        best = {r.name: X.file_md5(r / "best.pt") for r in runs}
        h2a_files_ok = all(X.file_md5(h2a / f) == m for f, m in ja["files"].items())
        h1_files_ok = all(X.file_md5(bnm / f) == m for f, m in j1["md5"].items() if f not in ("tool", "bdc"))
        marker = (data / H1.MARKER).exists() or (data / "bnm_test").exists()
        if broken() == "marker":
            marker = True
        c0 = dict(h1=j1.get("half") == "fit" and all(j1["gates"].values()) and h1_files_ok,
                  h2a=ja.get("half") == "fit" and all(ja["gates"].values()) and ja.get("truth_opened") == 0 and h2a_files_ok
                  and ja.get("frozen_md5") == j1["md5"]["bnm_frozen.npz"],
                  h2b=all(jb["gates"].values()) and X.file_md5(h2b / "h2b_predicted.npz") == mb["predicted"]
                  and X.file_md5(h2b / "h2b_scores.npz") == mb["scores"] and X.file_md5(h2a / "h2a_report.json") == mb["h2a_report"]
                  and mb["frozen"] == j1["md5"]["bnm_frozen.npz"],
                  score_tool=X.file_md5(HERE / "stage_h2_bnm_score.py") == mb["tool"],
                  best_pt=all(mb["best_pt"].get(k) == v for k, v in best.items()), no_test_export=not marker)
        gate("F0", all(c0.values()), "inputs as logged: " + ", ".join(f"{k} {v}" for k, v in c0.items())
             + f"; best.pt " + ", ".join(f"{k} {v[:8]}" for k, v in best.items()))
        if not res["F0"]:
            return stop("F0")
        # ---- F1 settings (read before the fits; scalars only)
        st, sok = {}, True
        for r in runs:
            za, zg = np.load(h2a / f"samples_{r.name}.npz"), np.load(r / "check_g1" / "samples_k50.npz")
            a = {k: int(za[k]) for k in ("K", "seed", "ddim", "batch")}
            g = {k: int(zg[k]) for k in ("K", "seed", "ddim", "batch")}
            sok &= a == g == SAMPLER and str(za["best_md5"]) == str(zg["best_md5"]) == best[r.name]
            st[r.name] = dict(a, epoch=int(za["epoch"]), best_md5=best[r.name])
        det = [float(v) for v in S2.DET] if broken() != "det" else [0.643, 0.642, 0.79]
        got = dict(det=det, thin_key=int(S2.KEY), h1_key=int(H1.KEY), repeats=int(H1.REPEATS), floor=int(S2.FLOOR),
                   n_fit_min=int(GC.N_FIT_MIN), qlo=GC.QLO, qhi=GC.QHI, level=float(GC.LEVEL), band=list(GC.BAND),
                   cov=list(GC.COV), holes={k: int(H1.n_holes(int(k))) for k in SIGNED["holes"]})
        cok = got == SIGNED and int(H1.FLOOR) == SIGNED["floor"]
        gate("F1", sok and cok, f"sampler of the H2a samples = G1 (K, noise seed, DDIM, batch) and best.pt md5 = G1's {sok} "
             f"({', '.join(f'{k}: K {v['K']}, seed {v['seed']}, DDIM {v['ddim']}, batch {v['batch']}, epoch {v['epoch']}' for k, v in st.items())}); "
             f"detection {det}, thinning key {got['thin_key']}, H1 key {got['h1_key']}, {got['repeats']} repeats, floor {got['floor']}, "
             f"fallback {got['n_fit_min']}, interval {got['qlo']}-{got['qhi']}, level {got['level']}, band {got['band']}, coverage "
             f"{got['cov']}, holes for 1/2/15/35/86 tetrads {list(got['holes'].values())} = signed {cok}")
        if not res["F1"]:
            return stop("F1")
        # ---- F2 same code
        P, S = np.load(h2b / "h2b_predicted.npz"), np.load(h2b / "h2b_scores.npz")
        rcls, rbin, stack, fold = P["rcls"], P["rbin"], P["stack_id"], P["fold"]
        n = len(stack)
        y = S["y_strict"].astype(float)
        pc = preds(P)
        worst, ntab = 0.0, 0
        for m in R2_M:
            opt = int(jb["H_R2"][m]["option"])
            fg = group_of(rcls, rbin, opt)
            par = np.zeros((n, 2)); mu = np.zeros(n); L = np.zeros(n); U = np.zeros(n)
            for c in (0, 1):
                fit, chk = np.nonzero(fold == 1 - c)[0], np.nonzero(fold == c)[0]
                tab = resolve(pc[m], y, fit, np.unique(fg[chk]), fg, opt, stack)
                ntab += len(tab)
                for g, t in tab.items():
                    ci = chk[fg[chk] == g]
                    par[ci] = (t["s"], t["q"]); mu[ci], L[ci], U[ci] = apply(pc[m], ci, t["s"], t["q"])
            for a_, b_ in ((par, S[f"hr2_scale_q_{m}"]), (mu, S[f"hr2_mean_{m}"]), (L, S[f"hr2_L_{m}"]), (U, S[f"hr2_U_{m}"])):
                worst = max(worst, float(np.abs(a_ - b_).max()))
        gate("F2", worst <= 1e-12 and n == jb["n"], f"fold fits of this tool = H2b's cross-fit for {len(R2_M)} methods at their H2b "
             f"option ({ntab} group fits; scale, Q, corrected mean, lower and upper end of {n:,} stack-repeats): largest difference {worst:.1e}")
        if not res["F2"]:
            return stop("F2")
        # ---- F3 refit on all 44 windows
        allf = np.arange(n)
        fits, txt, ok2 = {}, [], True
        for m in R2_M:
            opt = int(jb["H_R2"][m]["option"])
            fg = group_of(rcls, rbin, opt)
            ng = len(TCLS) if opt == 1 else len(TCLS) * NB
            tab = resolve(pc[m], y, allf, range(ng), fg, opt, stack)
            ok2 &= len(tab) == ng and all(np.isfinite(t["s"]) and t["s"] > 0 and np.isfinite(t["q"]) for t in tab.values())
            ins = []
            for gname, gi, labs in (("recorded class", rcls, TCLS), ("records over 6 years", rbin, BINS)):
                for j, lab in enumerate(labs):
                    ix = np.nonzero(gi == j)[0]
                    if len(ix):
                        mus, los, his = [], [], []
                        for g in np.unique(fg[ix]):
                            k = ix[fg[ix] == g]; a_ = apply(pc[m], k, tab[int(g)]["s"], tab[int(g)]["q"])
                            mus.append((k, a_[0])); los.append(a_[1]); his.append(a_[2])
                        kk = np.concatenate([k for k, _ in mus]); mm = np.concatenate([v for _, v in mus])
                        lo_, hi_ = np.concatenate(los), np.concatenate(his)
                        ins.append(f"{lab} {mm.mean() / y[kk].mean() if y[kk].mean() > 0 else float('nan'):.2f}/"
                                   f"{((y[kk] >= lo_) & (y[kk] <= hi_)).mean():.2f}")
            labels = TCLS if opt == 1 else [f"{a} | {b}" for a in TCLS for b in BINS]
            fits[m] = dict(option=opt, groups={str(g): dict(label=labels[g], **t) for g, t in tab.items()})
            own = sum(t["key"] == "g" for t in tab.values())
            txt.append(f"{m} option {opt}: {ng} groups ({own} own fit, {ng - own} by fallback); scale "
                       f"{min(t['s'] for t in tab.values()):.3f}-{max(t['s'] for t in tab.values()):.3f}; in-sample ratio/coverage "
                       + "; ".join(ins))
        gate("F3", ok2 and len(np.unique(P["stack_id"])) == jb["stacks"], f"refit on all {n:,} fit-half stack-repeats "
             f"({len(np.unique(stack)):,} stacks): every possible group has a finite scale above 0 and a finite Q {ok2}")
        for t_ in txt:
            out("   " + t_)
        loaded = list(log.paths)
    # ---- F4
    allowed = {str((h2a / f"samples_{r.name}.npz").resolve()) for r in runs} | {str((r / "check_g1" / "samples_k50.npz").resolve())
                                                                                 for r in runs} | {str((h2b / f).resolve()) for f in ("h2b_predicted.npz", "h2b_scores.npz")}
    if broken() == "truth":
        loaded.append(str((bnm / "x_truth.npz").resolve()))
    bad = [p for p in loaded if p not in allowed or "_truth" in Path(p).name or str(bnm.resolve()) in p]
    gate("F4", not bad, f"files passed to np.load {len(loaded)}, all H2a / G1 samples (scalars) or H2b outputs; others {len(bad)}")
    # ---- F5 freeze file
    tools = tool_md5s()
    if broken() == "tool":
        tools["stage_h1_bnm_export.py"] = "0" * 32
    bdc = Path(obj1_tools) / "bnm_detection_check.py"
    dfiles = {f"{r.name}/best.pt": r / "best.pt" for r in runs} | {f"{r.name}/check_g1/samples_k50.npz": r / "check_g1" / "samples_k50.npz" for r in runs} \
        | {"stage_g2_baselines_v2/g2_fit.json": g2 / "g2_fit.json", "stage_g3_calibration/g3_calibrated.npz": g3 / "g3_calibrated.npz"} \
        | {f"bnm_fit/{f}": bnm / f for f in ("h1_report.json", "bnm_frozen.npz", "bnm_stacks.npz", "manifest.csv")} \
        | {f"stage_h2_predict/{f}": h2a / f for f in ["h2a_report.json"] + sorted(ja["files"])} \
        | {f"stage_h2b_score/{f}": h2b / f for f in ("h2b_predicted.npz", "h2b_scores.npz", "h2b_report.json", "per_stack_h2b.csv", "per_window_h2b.csv")}
    dmd5 = {k: X.file_md5(p) for k, p in dfiles.items()}
    same_tools = dict(h1=tools.get("stage_h1_bnm_export.py") == j1["md5"]["tool"], h2a=tools.get("stage_h2_bnm_predict.py") == ja.get("tool"),
                      h2b=tools.get("stage_h2_bnm_score.py") == mb["tool"], bdc=bdc.is_file() and X.file_md5(bdc) == j1["md5"]["bdc"])

    def build():
        return json.dumps(dict(stage="H3 freeze", half="fit", windows=44, stack_repeats=int(n), tools=tools,
                               obj1_tools={"bnm_detection_check.py": X.file_md5(bdc) if bdc.is_file() else None}, data=dmd5,
                               sampler=st, settings=got, options={m: fits[m]["option"] for m in R2_M}, fits=fits,
                               apply_rule="group = recorded class (option 1) or recorded class x 5 + records bin (option 2); scaled = s x "
                               "thinned counts of the maps (pooled: first 25 of each seed); lower = max(5th percentile - q, 0); upper = "
                               "max(95th percentile + q, 0); mean = mean of the scaled counts"), indent=1, sort_keys=False)
    t1, t2 = build(), build()
    od.mkdir(parents=True)
    (od / "h3_freeze.json").write_text(t1)
    fm = X.file_md5(od / "h3_freeze.json")
    gate("F5", all(same_tools.values()) and t1 == t2 and json.loads(t1)["tools"].get("stage_h1_bnm_export.py") == j1["md5"]["tool"],
         f"h3_freeze.json md5 {fm}: {len(tools)} tool modules, {len(dmd5)} data files, sampler, settings and fits of {len(fits)} methods; "
         "same tool as the run before: " + ", ".join(f"{k} {v}" for k, v in same_tools.items()) + f"; built twice, same bytes {t1 == t2}")
    n_ok = sum(res.values())
    out(f"stage H3: gates {n_ok} PASS, {len(res) - n_ok} FAIL; options " + ", ".join(f"{m} {fits[m]['option']}" for m in R2_M)
        + f"; h3_freeze.json {fm}")
    return res, fm


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--bnm", default="../data/bnm_fit")
    ap.add_argument("--h2a", default="../data/stage_h2_predict")
    ap.add_argument("--h2b", default="../data/stage_h2b_score")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--out", default="../data/stage_h3_freeze")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_20_h3_freeze.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_20_h3_freeze", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.self_test(Path(__file__)) else 1)
    r, _ = run(a.data, a.bnm, a.h2a, a.h2b, a.runs, a.g2, a.g3, a.obj1_tools, a.out)
    sys.exit(0 if r and all(r.values()) else 1)