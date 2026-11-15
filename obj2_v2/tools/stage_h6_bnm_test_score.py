#!/usr/bin/env python3
"""
stage_h6_bnm_test_score.py  (obj2_v2, stage H step H6, 12 Nov 2026; H rules signed off by Sara on 11 Nov; the H6
report-only D1 / D2 addition and the H6 points Q1-Q3 signed on 12 Nov, claude/obj2_v2_H_rules_draft.md)

Question: H6 of the signed H rules, the one reading of the BNM test half. Score the H5 predictions in the H4 holes with the
H2b code and the H-R2 correction frozen in H3 (no fit on the test half): H-R1 ranking, H-R2 recorded count, the
report-only items of H2b, and the diagnostic's D1 / D2 (report only). No overall verdict.

Order
    Phase 1, test labels closed (np.load of a *_truth file in the test export is refused):
        Z0 inputs: h3_freeze.json md5 = the signed one; every tool and data md5 listed in it = the files now; settings of
           this code = the frozen settings; H4 export as logged (gates PASS, half test, truth md5 = manifest, H1 tool md5 =
           frozen, marker BNM_TEST_EXPORTED with that md5); H5 report half test, gates PASS, truth opened 0, every part
           PASS, files md5 = list, frozen md5 = H4, H2a tool and best.pt md5 = frozen; diagnostic tool md5 = 22878cd5
        Z1 alignment from inputs (as H2b Y1): maps in frozen order, layout, K, frozen md5, sampler = frozen; kept records,
           classes, window, repeat, row and blank = bnm_frozen / bnm_stacks; B4 distance = recomputed
        Z2 code check on the fit half (fit-half labels may be read): (a) this tool's AUC and H-R1 code on h2b_predicted.npz
           and the fit-half labels = h2b_scores.npz AUCs (1e-12) and H2b's H-R1 rows and verdicts (1e-9); (b) H3's resolver
           on all fit-half stack-repeats = every frozen scale and Q (1e-12), and this tool's frozen-fit code = H3's apply
           (1e-12); in-sample ratio / coverage printed (they equal H3's F3 lines)
        Z3 holes and thinning on the test half (hole, pair_md5, species only): pair md5, species and hole counts = inputs;
           no kept record or effort in a hole; thinned share in the holes per method (seed 1, seed 2, B3, B4) and detection
           class = the frozen detection, |z| <= 4 (Sara, 12 Nov); cluster-robust z printed beside; a second pass in other
           chunks gives the same counts
        h6_predicted.npz written, md5 printed; then the marker data/STAGE_H6_SCORED (Sara, 12 Nov): a second run stops
           unless --rerun (a new folder; every output says rerun)
    Phase 2, labels, once:
        Z4 labels aligned (as Y4): recorded holes inside the holes, strict inside pooled, truth = inputs outside the holes,
           holes inside the 2019 effort; random score AUC 0.50
        H-R1 (both seeds, pooled; against B1 stored and exact; stack-first; 2,000 window resamples, seed 31), H-R2 with the
           frozen fits and options (coverage per stack-repeat), H2b's report-only items, D1 / D2 (report only)
        Z5 order: no test truth opened in phase 1; no label array before h6_predicted.npz; it is unchanged after; the
           marker holds its md5
        Z6 report-only code checks: D2 AUCs = the H-R1 AUCs (1e-12); D2 decomposition = a hand count of every pair on 200
           stack-repeats (1e-9)

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU)
    python3 ../tools/stage_h6_bnm_test_score.py --self-test | tail -1     (needs ../tests/test_23_h6_test_score.py; about 15 min)
    python3 -u ../tools/stage_h6_bnm_test_score.py --check-only           (phase 1 only: Z0-Z3, no label, nothing written)
    python3 -u ../tools/stage_h6_bnm_test_score.py > ../logs/stage_h6_test_score.txt 2>&1      (the one reading)
Needs stage_h2_bnm_score.py, stage_h3_freeze.py, stage_h3b_sparse_diagnostic.py and the stage tools they load.
"""
import argparse, csv, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_h6_bnm_test_score.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


S2 = _load("stage_h2_bnm_score")
H3 = _load("stage_h3_freeze")
DG = _load("stage_h3b_sparse_diagnostic")
H2A, GC, GB, GS, CK, HR, D, X = S2.H2A, S2.GC, S2.GB, S2.GS, S2.CK, S2.HR, S2.D, S2.X
G, W = X.G, X.WINDOW
TCLS, BINS, DIST, DLAB = S2.TCLS, S2.BINS, S2.DIST, S2.DLAB
MAP_M, SCORE_M, R2_M, NAME, HOLE_KEYS = S2.MAP_M, S2.SCORE_M, S2.R2_M, S2.NAME, S2.HOLE_KEYS
FREEZE_MD5 = "5516a6fc9e1cfcda3ce1a883aa5467f9"     # Sara's H3 run, 12 Nov
DIAG_MD5 = "22878cd5ad1b27d6c04faf865c2cc124"       # stage_h3b_sparse_diagnostic.py as run on 12 Nov
MARKER = "STAGE_H6_SCORED"
K = 50


def broken():
    return os.environ.get("STAGE_H6_BROKEN", "")    # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- pieces
def freeze_check(fz, obj1_tools, paths, runs):
    """Every tool, obj1 tool and data md5 listed in h3_freeze.json against the files now."""
    bad = [k for k, v in fz["tools"].items() if not (HERE / k).is_file() or X.file_md5(HERE / k) != v]
    for k, v in fz.get("obj1_tools", {}).items():
        p = Path(obj1_tools) / k
        if v is None or not p.is_file() or X.file_md5(p) != v:
            bad.append(f"obj1:{k}")
    rmap = {r.name: r for r in runs}
    for k, v in fz["data"].items():
        head, rest = k.split("/", 1)
        p = rmap[head] / rest if head in rmap else (Path(paths[head]) / rest if head in paths else None)
        if p is None or not p.is_file() or X.file_md5(p) != v:
            bad.append(k)
    return bad, len(fz["tools"]) + len(fz.get("obj1_tools", {})) + len(fz["data"])


def hr1_core(d, stack, grp, win, classes):
    return S2.hr1_core(d + (1e-6 if broken() == "core_shift" else 0.0), stack, grp, win, classes)


def hr1_block(A, mods, refs, stid, grp, wname, floor, label_txt=None, out=None):
    """H2b's H-R1 block (the closure hr1 in stage_h2_bnm_score.run), line for line; checked against H2b in Z2."""
    res = {}
    for m_ in mods:
        rr = {}
        for b in refs:
            rows = {g: S2.hr1_rows(hr1_core(A[m_] - A[b], stid, gi, wname, list(range(len(labs)))), labs, floor)
                    for g, (gi, labs) in grp.items()}
            rr[b] = dict(rows=rows, verdict=S2.overall(rows.values()))
        v = S2.both(rr[refs[0]]["verdict"], rr[refs[-1]]["verdict"]) if len(refs) == 2 else rr[refs[0]]["verdict"]
        res[m_] = dict(rr, verdict=v)
        if out:
            out(f"      {label_txt} {NAME[m_]}:")
            for b in refs:
                for g, rows in rr[b]["rows"].items():
                    out(f"         vs {NAME[b]:<9} {g:<20} " + "; ".join(f"{r['group']} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                                                                  f"{r['verdict']} ({r['n']})" if r["n"] else f"{r['group']} none" for r in rows))
                out(f"         vs {NAME[b]}: {rr[b]['verdict']}")
            out(f"         {NAME[m_]} H-R1 {'(both B1 versions needed) ' if len(refs) == 2 and m_ in ('s1', 's2') else '(report) '}{v}")
    return res


def apply_frozen(pred, fgrp, mfit):
    """The H3 apply rule, vectorised by group: scaled = s x counts; lower = max(5th pct - q, 0); upper = max(95th pct + q, 0)."""
    n = len(pred)
    mean, L, U, par = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, 2))
    for g in np.unique(fgrp):
        ix = np.nonzero(fgrp == g)[0]
        t = mfit["groups"][str(int(g))]
        s = float(t["s"]) * (1.001 if broken() == "apply_scale" else 1.0)
        sc = s * pred[ix]
        mean[ix] = sc.mean(1)
        L[ix] = np.maximum(np.percentile(sc, GC.QLO, axis=1) - float(t["q"]), 0.0)
        U[ix] = np.maximum(np.percentile(sc, GC.QHI, axis=1) + float(t["q"]), 0.0)
        par[ix] = (s, float(t["q"]))
    return dict(mean=mean, L=L, U=U, par=par)


def rows_differ(a, b):
    """Largest difference between two H-R1 row lists (mean, lo, hi; n and verdict must match)."""
    worst = 0.0
    for ra, rb in zip(a, b):
        if ra["n"] != rb["n"] or ra["verdict"] != rb["verdict"] or ra["group"] != rb["group"]:
            return float("inf")
        worst = max(worst, *(S2.dif(ra[q], rb[q]) for q in ("mean", "lo", "hi")))
    return worst if len(a) == len(b) else float("inf")


def thin_z(occ_h, thin_h, dcls, det):
    """Per method and detection class: binomial z of the thinned share, and a cluster-robust z (one cluster per stack-repeat
    and sample: residual = thinned - p x occupied, z = sum / sqrt(sum of squares)); pooled robust z over the 4 methods."""
    meth = ("s1", "s2", "B3", "B4")
    out = {}
    for c in range(3):
        m_ = dcls == c
        row = {}
        pooled = 0.0
        for k in meth:
            O, T_ = float(occ_h[k][m_].sum()), float(thin_h[k][m_].sum())
            r = thin_h[k][m_].astype(float) - det[c] * occ_h[k][m_].astype(float)
            pooled = pooled + r
            zb = (T_ / O - det[c]) / np.sqrt(det[c] * (1 - det[c]) / O) if O > 0 else 0.0
            zr = r.sum() / np.sqrt((r ** 2).sum()) if (r ** 2).sum() > 0 else 0.0
            row[k] = dict(O=int(O), share=T_ / O if O else float("nan"), z=float(zb), z_rob=float(zr))
        pr = np.asarray(pooled)
        row["pooled_z_rob"] = float(pr.sum() / np.sqrt((pr ** 2).sum())) if np.size(pr) and (pr ** 2).sum() > 0 else 0.0
        out[DLAB[c]] = row
    return out


# ---------------------------------------------------------------- run
def run(data="../data", bnm_test="../data/bnm_test", h5="../data/stage_h5_predict", bnm_fit="../data/bnm_fit",
        h2a="../data/stage_h2_predict", h2b="../data/stage_h2b_score", g2="../data/stage_g2_baselines_v2",
        g3="../data/stage_g3_calibration", freeze="../data/stage_h3_freeze/h3_freeze.json", runs=("../runs/f_seed1", "../runs/f_seed2"),
        obj1_tools="../../obj1_v2/tools", out_dir="../data/stage_h6_test_score", rerun=False, freeze_md5=FREEZE_MD5,
        diag_md5=DIAG_MD5, floor_override=None, check_only=False, out=print):
    data, bnm, h5, bnmf, h2a, h2b, g2, g3, freeze, od = map(Path, (data, bnm_test, h5, bnm_fit, h2a, h2b, g2, g3, freeze, out_dir))
    runs = [Path(r) for r in runs]
    mk = data / MARKER
    if mk.exists() and not rerun:
        sys.exit(f"STOP: {mk} exists: the test half was scored already (a second reading needs --rerun, into a new folder)")
    if od.exists():
        sys.exit(f"STOP: {od} exists; write into a new folder")
    for p_ in (bnm, h5, bnmf, h2b):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside {p_}")
    if json.loads((bnm / "h1_report.json").read_text()).get("half") != "test":
        sys.exit("STOP: H6 scores the test-half export only (--bnm-test)")
    t0 = time.time()
    res, rep = {}, dict(rerun=bool(rerun))
    tag = "RERUN " if rerun else ""

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"{tag}stage H6: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}" +
            ("; no label read, no marker written" if after in ("Z0", "Z1", "Z2", "Z3") else ""))
        return res, rep
    out(f"{tag}stage_h6_bnm_test_score.py md5 {X.file_md5(Path(__file__))}; stage_h2_bnm_score.py {X.file_md5(HERE / 'stage_h2_bnm_score.py')}; "
        f"stage_h3_freeze.py {X.file_md5(HERE / 'stage_h3_freeze.py')}; stage_h3b_sparse_diagnostic.py {X.file_md5(HERE / 'stage_h3b_sparse_diagnostic.py')}")
    fz = json.loads(freeze.read_text())
    floor = int(fz["settings"]["floor"]) if floor_override is None else int(floor_override)
    if floor_override is not None:
        out(f"   rule floor {floor} distinct stacks: override for the planted self-test only (the frozen floor is {fz['settings']['floor']})")
    # ================================================================ phase 1, test labels closed
    with S2.BnmGuard(bnm) as guard:
        try:
            # ---- Z0
            fmd5 = X.file_md5(freeze)
            bad_f, nf = freeze_check(fz, obj1_tools, dict(bnm_fit=bnmf, stage_h2_predict=h2a, stage_h2b_score=h2b,
                                                          stage_g2_baselines_v2=g2, stage_g3_calibration=g3), runs)
            st = fz["settings"]
            set_ok = list(S2.DET) == list(st["det"]) and int(S2.KEY) == int(st["thin_key"]) and int(S2.FLOOR) == int(st["floor"]) \
                and int(GC.N_FIT_MIN) == int(st["n_fit_min"]) and GC.QLO == st["qlo"] and GC.QHI == st["qhi"] \
                and list(GC.BAND) == list(st["band"]) and list(GC.COV) == list(st["cov"])
            k0t = []
            fb, mf, js, ok_h4 = H2A.k0(SimpleNamespace(gate=lambda g, ok, text: k0t.append(text)), bnm)
            man_rows = D.rows_of(bnm)
            man = {r["input_file"]: r for r in man_rows}
            tbad = [r["truth_file"] for r in man_rows if X.file_md5(bnm / r["truth_file"]) != r["truth_md5"]]
            h1m, h2am = fz["tools"].get("stage_h1_bnm_export.py"), fz["tools"].get("stage_h2_bnm_predict.py")
            mkj = json.loads((data / "BNM_TEST_EXPORTED").read_text().splitlines()[0]) if (data / "BNM_TEST_EXPORTED").is_file() else {}
            h4_ok = ok_h4 and not tbad and js["half"] == "test" and js["md5"]["tool"] == h1m and mkj.get("tool_md5") == h1m \
                and Path(mkj.get("out", "/nonexistent")).resolve() == bnm.resolve()
            ha = json.loads((h5 / "h2a_report.json").read_text())
            need = ["baselines_bnm.npz", "maps_B3.npz", "maps_B4.npz", "maps_B5.npz"] + [f"samples_{r.name}.npz" for r in runs]
            fbad = [f for f in need if f not in ha["files"] or X.file_md5(h5 / f) != ha["files"][f]]
            best = {r.name: X.file_md5(r / "best.pt") for r in runs}
            best_ok = all(ha["best_md5"].get(f"samples_{r.name}") == best[r.name] == fz["sampler"][r.name]["best_md5"] for r in runs)
            h5_ok = ha["half"] == "test" and all(ha["gates"].values()) and ha["truth_opened"] == 0 and not fbad and best_ok \
                and all(all(v["gates"].values()) for v in ha["parts"].values()) and ha["frozen_md5"] == mf and ha["tool"] == h2am
            dg_ok = X.file_md5(HERE / "stage_h3b_sparse_diagnostic.py") == diag_md5
            ok0 = fmd5 == freeze_md5 and not bad_f and set_ok and h4_ok and h5_ok and dg_ok
            gate("Z0", ok0, f"h3_freeze.json md5 {fmd5} = signed {fmd5 == freeze_md5}; {nf} md5s listed = the files now (differ: {bad_f}); "
                 f"settings = frozen {set_ok}; H4: {k0t[0]}, truth md5 = manifest {not tbad}, half {js['half']}, H1 tool = frozen "
                 f"{js['md5']['tool'] == h1m}, marker {mkj.get('utc')} with the frozen md5 {mkj.get('tool_md5') == h1m}; H5 "
                 f"(h2a_report {ha['written_utc']}): half {ha['half']}, gates and parts PASS, truth opened {ha['truth_opened']}, "
                 f"files = list {not fbad}, frozen md5 = H4 {ha['frozen_md5'] == mf}, H2a tool = frozen {ha['tool'] == h2am}, best.pt "
                 + ", ".join(f"{k} {v[:8]}" for k, v in best.items()) + f" = frozen {best_ok}; diagnostic tool = {diag_md5[:8]} {dg_ok}")
            if not ok0:
                return stop("Z0")
            # ---- Z1
            n = len(fb["row"])
            idx = dict(np.load(bnm / "bnm_stacks.npz"))
            V1, V2, Ef, _, _ = GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])
            V = V1 | V2
            del V1, V2
            rec_t = V[:, -1].reshape(n, -1).sum(1); rec6 = V.any(1).reshape(n, -1).sum(1)
            rcls, rbin, dcls = D.tidx(rec_t), D.bidx(rec6), S2.det_class(rec6)
            winf = np.array([man[str(fb["files"][i])]["window"] for i in fb["file_index"]])
            repf = np.array([int(man[str(fb["files"][i])]["repeat"]) for i in fb["file_index"]])
            al = np.array_equal(rec_t, fb["rec_t"]) and np.array_equal(rec6, fb["rec6"]) and np.array_equal(rcls, fb["cls"]) \
                and len(idx["window"]) == n and np.array_equal(winf, idx["window"]) and np.array_equal(repf, idx["repeat"]) \
                and np.array_equal(fb["row"], idx["row"]) and bool((fb["blank"] == 1).all())
            maps, why1, ok1 = {}, [], True
            for m_, f, kexp in (("s1", f"samples_{runs[0].name}.npz", K), ("s2", f"samples_{runs[1].name}.npz", K),
                                ("B3", "maps_B3.npz", K), ("B4", "maps_B4.npz", K), ("B5", "maps_B5.npz", 1)):
                z = np.load(h5 / f)
                good = np.array_equal(z["sel"], np.arange(n)) and str(z["layout"]) == GS.LAYOUT and str(z["frozen_md5"]) == mf \
                    and int(z["K"]) == kexp and z["maps"].shape == (n, kexp, W * G * G // 8)
                if m_ in ("s1", "s2"):
                    smp = fz["sampler"][runs[0 if m_ == "s1" else 1].name]
                    good &= all(int(z[k]) == int(smp[k]) for k in ("K", "seed", "ddim", "batch")) and str(z["best_md5"]) == smp["best_md5"]
                else:
                    good &= all(np.array_equal(GC.target_year(z["maps"][a:a + 512]).sum((2, 3)), z["aoo_t"][a:a + 512]) for a in range(0, n, 512))
                ok1 &= good; why1.append(f"{f} {good}")
                maps[m_] = z["maps"]
            bz = dict(np.load(h5 / "baselines_bnm.npz"))
            dist = GB.near_dist(V)
            bok = str(bz["frozen_md5"]) == mf and all(bz[k].shape == (n, G, G) for k in ("B1", "B1_exact_log", "B3_nohab", "B4_logit", "B4_dist", "B5")) \
                and np.array_equal(bz["B4_dist"], dist.astype(np.float32))
            gate("Z1", ok1 and al and bok, f"{n:,} test-half stack-repeats (bnm_frozen {mf[:8]}); map files in frozen order with the layout, "
                 f"K, frozen md5 and the frozen sampler: {', '.join(why1)}; kept records, classes, window, repeat, row and blank = "
                 f"bnm_frozen and bnm_stacks {al}; baselines frozen md5, shapes and B4 distance = recomputed {bok}")
            if not res["Z1"]:
                return stop("Z1")
            # ---- Z2 code check on the fit half
            P, Sc = np.load(h2b / "h2b_predicted.npz"), np.load(h2b / "h2b_scores.npz")
            jb = json.loads((h2b / "h2b_report.json").read_text())
            fbf = dict(np.load(bnmf / "bnm_frozen.npz")); manf = {r["input_file"]: r for r in D.rows_of(bnmf)}
            nfit = len(fbf["row"])
            Lsf = S2.read_labels(bnmf, fbf, manf)[0]
            ptrf, cellf = P["hole_ptr"].astype(np.int64), P["hole_cell"].astype(np.int64)
            itf = np.repeat(np.arange(nfit), np.diff(ptrf))
            labf = Lsf.reshape(nfit, -1)[itf, cellf]
            del Lsf
            Af = {m_: S2.item_aucs(ptrf, P[f"score_{m_}"], labf) for m_ in SCORE_M}
            wa, nan_same = 0.0, True
            for m_ in SCORE_M:
                a_, r_ = Af[m_], Sc[f"auc_strict_{m_}"]
                nan_same &= np.array_equal(np.isnan(a_), np.isnan(r_))
                okm = ~np.isnan(a_) & ~np.isnan(r_)
                wa = max(wa, float(np.abs(a_[okm] - r_[okm]).max()) if okm.any() else 0.0)
            wnf = np.load(bnmf / "bnm_stacks.npz")["window"].astype(str)
            grpf = {"recorded class": (P["rcls"], TCLS), "records over 6 years": (P["rbin"], BINS)}
            r1f = hr1_block(Af, ("s1", "s2", "pooled"), ("B1", "B1x"), P["stack_id"], grpf, wnf, int(jb["floor"]))
            wr, vsame = 0.0, True
            for m_ in ("s1", "s2", "pooled"):
                vsame &= r1f[m_]["verdict"] == jb["H_R1"][m_]["verdict"]
                for b in ("B1", "B1x"):
                    vsame &= r1f[m_][b]["verdict"] == jb["H_R1"][m_][b]["verdict"]
                    for g in grpf:
                        wr = max(wr, rows_differ(r1f[m_][b]["rows"][g], jb["H_R1"][m_][b]["rows"][g]))
            pcf, yf = H3.preds(P), Sc["y_strict"].astype(float)
            ws, wp, ins = 0.0, 0.0, []
            for m_ in R2_M:
                opt = int(fz["options"][m_])
                fgf = H3.group_of(P["rcls"], P["rbin"], opt)
                ng = len(TCLS) if opt == 1 else len(TCLS) * len(BINS)
                tab = H3.resolve(pcf[m_], yf, np.arange(nfit), range(ng), fgf, opt, P["stack_id"])
                for g, t in tab.items():
                    fr = fz["fits"][m_]["groups"][str(g)]
                    ws = max(ws, abs(t["s"] - fr["s"]), abs(t["q"] - fr["q"]))
                cf = apply_frozen(pcf[m_], fgf, fz["fits"][m_])
                for g in np.unique(fgf):
                    ix = np.nonzero(fgf == g)[0]; fr = fz["fits"][m_]["groups"][str(int(g))]
                    a_ = H3.apply(pcf[m_], ix, fr["s"], fr["q"])
                    wp = max(wp, float(np.abs(a_[0] - cf["mean"][ix]).max()), float(np.abs(a_[1] - cf["L"][ix]).max()),
                             float(np.abs(a_[2] - cf["U"][ix]).max()))
                rows = S2.evaluate_h(yf, cf, grpf, P["stack_id"])
                ins.append(f"{m_} option {opt}: " + " | ".join("; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}" for r in rows[g]) for g in grpf))
            ok2 = wa <= 1e-12 and nan_same and wr <= 1e-9 and vsame and ws <= 1e-12 and wp <= 1e-12 and \
                all(int(fz["options"][m_]) == int(jb["H_R2"][m_]["option"]) for m_ in R2_M)
            gate("Z2", ok2, f"code check on the fit half ({nfit:,} stack-repeats): AUC of {len(SCORE_M)} methods = h2b_scores.npz, largest "
                 f"difference {wa:.1e}, same not-rankable set {nan_same}; H-R1 rows of seed 1, seed 2, pooled vs both B1 versions = "
                 f"h2b_report.json, largest difference {wr:.1e}, verdicts same {vsame}; H3 resolver on all fit-half stack-repeats = frozen "
                 f"scale and Q, largest difference {ws:.1e}; frozen-fit code = H3 apply, largest difference {wp:.1e}; options = H2b's")
            for t_ in ins:
                out(f"   in-sample with the frozen fits (ratio/coverage; = H3 F3): {t_}")
            if not ok2:
                return stop("Z2")
            del labf, P
            cnt = {m_: S2.count_maps(maps[m_]) for m_ in MAP_M}
            cnt["pooled"] = S2.count_maps(maps["s1"], 25) + S2.count_maps(maps["s2"], 25)
            smap = {"s1": cnt["s1"] / K, "s2": cnt["s2"] / K, "pooled": cnt["pooled"] / K, "B1": bz["B1"], "B1x": bz["B1_exact_log"],
                    "B3": bz["B3_nohab"], "B4": bz["B4_logit"], "B5": bz["B5"]}
            share = {"s1": smap["s1"], "s2": smap["s2"], "pooled": smap["pooled"], "B3": cnt["B3"] / K, "B4": cnt["B4"] / K,
                     "B5": cnt["B5"].astype(float)}
            out("   score maps, truth closed: " + ", ".join(f"{NAME[m_]} {S2.arr_md5(smap[m_])[:8]}" for m_ in SCORE_M))
            if broken() == "p1_open":
                np.load(bnm / man_rows[0]["truth_file"])
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    if crashed:
        gate("Z5", False, f"order: {crashed}")
        return stop("Z5")
    p1_opened = len(guard.opened)
    stid, status, wname = idx["stack_id"], idx["status"].astype(str), idx["window"].astype(str)
    # ================================================================ holes, then labels
    with S2.KeyLog() as kl:
        try:
            H, sp_t, pair_ok = S2.read_holes(bnm, fb, man)
            nh = H.reshape(n, -1).sum(1)
            nobad = not (V & H[:, None]).any() and not (Ef & H[:, None]).any()
            sp_ok = np.array_equal(sp_t, idx["species"].astype(str))
            cnt_ok = np.array_equal(nh, idx["n_holes"]) and bool((nh >= 1).all())
            surv = Ef[:, -1] | H
            items, cells = np.nonzero(H.reshape(n, -1))
            ptr = np.r_[0, np.cumsum(nh)]
            det = [float(v) for v in fz["settings"]["det"]]
            pthr = np.array(det)[dcls] if broken() != "thin_p" else np.full(n, 0.9)
            occ_h = {m_: np.zeros((n, maps[m_].shape[1]), np.uint16) for m_ in MAP_M}
            thin_h = {m_: np.zeros_like(occ_h[m_]) for m_ in MAP_M}
            thin_s = {m_: np.zeros_like(occ_h[m_]) for m_ in MAP_M}
            eoo_s = {m_: np.zeros(occ_h[m_].shape, np.float32) for m_ in MAP_M}
            tc = time.time()

            def thin_block(a, b, eoo=True):
                keep = S2.uniforms(range(a, b), K) < pthr[a:b][:, None, None, None]
                Hc, Sc_ = H[a:b, None], surv[a:b, None]
                r_ = {}
                for m_ in MAP_M:
                    occ = GC.target_year(maps[m_][a:b])
                    th = occ & keep[:, :occ.shape[1]]
                    ts = th & Sc_
                    r_[m_] = ((occ & Hc).sum((2, 3)), (th & Hc).sum((2, 3)), ts.sum((2, 3)),
                              S2.eoo_of(ts.reshape(-1, G, G)).reshape(b - a, -1) if eoo else None)
                return r_
            for a in range(0, n, S2.CH):
                b = min(n, a + S2.CH)
                rb = thin_block(a, b)
                for m_ in MAP_M:
                    occ_h[m_][a:b], thin_h[m_][a:b], thin_s[m_][a:b], eoo_s[m_][a:b] = rb[m_]
            same2, m2 = True, min(n, 200)
            for a in range(0, m2, 7):
                rb = thin_block(a, min(m2, a + 7), eoo=False)
                same2 &= all(np.array_equal(rb[m_][1], thin_h[m_][a:min(m2, a + 7)]) and np.array_equal(rb[m_][2], thin_s[m_][a:min(m2, a + 7)])
                             for m_ in MAP_M)
            tz = thin_z(occ_h, thin_h, dcls, det)
            zok = all(abs(tz[c][k]["z"]) <= 4 for c in tz for k in ("s1", "s2", "B3", "B4") if tz[c][k]["O"] > 0)
            ztxt = "; ".join(f"{c}: " + ", ".join(f"{k} {tz[c][k]['share']:.4f} z {tz[c][k]['z']:+.2f} (robust {tz[c][k]['z_rob']:+.2f})"
                                                  for k in ("s1", "s2", "B3", "B4")) + f", 4 methods robust {tz[c]['pooled_z_rob']:+.2f}"
                             for c in tz)
            gate("Z3", pair_ok and sp_ok and cnt_ok and nobad and zok and same2,
                 f"hole masks read with only {', '.join(HOLE_KEYS)} open: pair md5 = inputs {pair_ok}, species = bnm_stacks {sp_ok}, hole "
                 f"counts = bnm_stacks and at least 1 {cnt_ok} ({int(nh.sum()):,} hole tetrads); no kept record or input effort in a hole "
                 f"{nobad}; thinned share per method against the frozen detection {det}, |z| <= 4 per method: {ztxt}; second pass in "
                 f"chunks of 7 the same {same2} ({(time.time() - tc) / 60:.1f} min)")
            if not res["Z3"]:
                return stop("Z3")
            if check_only:
                n_ok = sum(res.values())
                out(f"{tag}stage H6 check-only: gates {n_ok} PASS, {len(res) - n_ok} FAIL; phase 1 done (Z0-Z3), no label read, nothing "
                    f"written, no marker")
                return res, rep
            pred = dict(hole_ptr=ptr.astype(np.int64), hole_cell=cells.astype(np.int16), dist_h=dist.reshape(n, -1)[items, cells].astype(np.float32),
                        rec_t=rec_t.astype(np.int16), rec6=rec6.astype(np.int16), rcls=rcls.astype(np.int8), rbin=rbin.astype(np.int8),
                        dcls=dcls.astype(np.int8), stack_id=stid, frozen_md5=np.array(mf), key=np.int64(S2.KEY), freeze_md5=np.array(fmd5))
            for m_ in SCORE_M:
                pred[f"score_{m_}"] = np.asarray(smap[m_]).reshape(n, -1)[items, cells].astype(np.float32)
            for m_ in R2_M:
                pred[f"share_{m_}"] = np.asarray(share[m_]).reshape(n, -1)[items, cells].astype(np.float32)
            for m_ in MAP_M:
                pred[f"occ_h_{m_}"], pred[f"thin_h_{m_}"], pred[f"thin_s_{m_}"], pred[f"eoo_s_{m_}"] = occ_h[m_], thin_h[m_], thin_s[m_], eoo_s[m_]
            od.mkdir(parents=True)
            pf = od / "h6_predicted.npz"
            X.save_npz(pf, pred)
            mpred = X.file_md5(pf)
            out(f"   h6_predicted.npz md5 {mpred}: scores, map shares and distances at {int(nh.sum()):,} hole tetrads, occupied and thinned "
                f"counts in the holes and over the 2019 survey, thinned EOO; written before any label array was read")
            with open(mk, "a") as fh:
                fh.write(json.dumps(dict(utc=now(), tool_md5=X.file_md5(Path(__file__)), predicted_md5=mpred, out=str(od.resolve()),
                                         rerun=bool(rerun))) + "\n")
            out(f"   marker {mk.name} {'appended (rerun)' if rerun else 'written'}; now the test labels are read, once")
            kl.mode = "labels" if broken() != "label_early" else "holes"
            Ls, Lp, Ro, Eo = S2.read_labels(bnm, fb, man)
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("Z5", False, f"order: {crashed}")
        return stop("Z5")
    early = [k for m_, k in kl.reads if m_ == "holes" and k not in HOLE_KEYS]
    # ---- Z4
    Hm = H[:, None]
    in_h = not (Ls & ~H).any() and not (Lp & ~H).any() and not (Ls & ~Lp).any()
    outside = np.array_equal(Ro & ~Hm, V) and np.array_equal(Eo & ~Hm, Ef)
    in_eff = not (H & ~Eo[:, -1]).any()
    lab_s = Ls.reshape(n, -1)[items, cells]; lab_p = Lp.reshape(n, -1)[items, cells]
    y = np.add.reduceat(lab_s.astype(int), ptr[:-1]).astype(float); yp = np.add.reduceat(lab_p.astype(int), ptr[:-1]).astype(float)
    rank = (y > 0) & (y < nh)
    sh_s, sh_p = y.sum() / nh.sum(), yp.sum() / nh.sum()
    ar = S2.item_aucs(ptr, X.rng(S2.KEY, 9).random(len(lab_s)), lab_s)
    ar_m, ar_se = float(np.nanmean(ar)), float(np.nanstd(ar) / np.sqrt(max(1, np.isfinite(ar).sum())))
    tol = max(0.01, 4 * ar_se)
    gate("Z4", in_h and outside and in_eff and abs(ar_m - 0.5) <= tol,
         f"recorded holes inside the holes, strict inside pooled {in_h}; outside the holes the truth = the inputs in all {W} years {outside}; "
         f"every hole inside the 2019 effort {in_eff}; rankable {int(rank.sum()):,} stack-repeats ({len(np.unique(stid[rank])):,} stacks), "
         f"hole shares strict {sh_s:.3f}, pooled {sh_p:.3f}; random score AUC {ar_m:.3f} (0.50 within {tol:.3f})")
    if not res["Z4"]:
        return stop("Z4")
    # ================================================================ H-R1
    A = {m_: S2.item_aucs(ptr, pred[f"score_{m_}"], lab_s) for m_ in SCORE_M}
    Ap = {m_: S2.item_aucs(ptr, pred[f"score_{m_}"], lab_p) for m_ in SCORE_M}
    grp = {"recorded class": (rcls, TCLS), "records over 6 years": (rbin, BINS)}
    out("   rankable distinct stacks per group (floor %d): " % floor + " | ".join("; ".join(
        f"{lab} {len(np.unique(stid[rank & (gi == j)]))}" for j, lab in enumerate(labs)) for g, (gi, labs) in grp.items()))
    auc_g = {m_: {g: hr1_core(A[m_], stid, gi, wname, list(range(len(labs)))) for g, (gi, labs) in grp.items()} for m_ in SCORE_M}
    out("   AUC in the holes, stack-first mean by group (report):")
    for m_ in SCORE_M:
        out(f"      {NAME[m_]:<9} " + " | ".join("; ".join(f"{lab} {auc_g[m_][g][j]['mean']:.3f}" for j, lab in enumerate(labs))
                                             for g, (gi, labs) in grp.items()))
    out(f"   H-R1 ranking in the holes, model - B1, stack-first mean [95% interval, {len(np.unique(wname))} windows resampled] verdict "
        f"(distinct stacks; tested at {floor}+):")
    r1 = hr1_block(A, ("s1", "s2", "pooled"), ("B1", "B1x"), stid, grp, wname, floor, "strict truth", out)
    r1b = hr1_block(A, ("B3", "B4", "B5"), ("B1",), stid, grp, wname, floor, "strict truth (report)", out)
    out("   H-R1 with pooled truth (hole recorded in any of the 6 years; report):")
    r1p = hr1_block(Ap, ("s1", "s2"), ("B1", "B1x"), stid, grp, wname, floor, "pooled truth", out)
    # ================================================================ H-R2 with the frozen fits
    pc = {m_: pred[f"thin_h_{m_}"].astype(float) for m_ in MAP_M}
    pc["pooled"] = np.concatenate([pc["s1"][:, :25], pc["s2"][:, :25]], 1)
    r2, chosen = {}, {}
    for m_ in R2_M:
        opt = int(fz["options"][m_])
        cf = apply_frozen(pc[m_], H3.group_of(rcls, rbin, opt), fz["fits"][m_])
        rows = S2.evaluate_h(y, cf, grp, stid)
        ok, bad = S2.rule_pass_h(rows, floor)
        r2[m_] = dict(cf=cf, rows=rows); chosen[m_] = dict(option=opt, ok=ok, bad=bad)
    out(f"   H-R2 recorded count in the holes with the H3 fits (ratio / coverage / median width (distinct stacks); rule: ratio "
        f"{GC.BAND[0]}-{GC.BAND[1]}, coverage {GC.COV[0]}-{GC.COV[1]} in every group of {floor}+ stacks; * = reported):")
    for m_ in R2_M:
        ch = chosen[m_]
        out(f"      {NAME[m_]:<8} option {ch['option']} (frozen) rule {'PASS' if ch['ok'] else 'FAIL'}" + ("" if ch["ok"] else f" ({', '.join(ch['bad'][:6])})"))
        for g in grp:
            out(f"         {g:<20} " + "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}/{r['width']:.1f} ({r['stacks']})"
                                                + ("" if r["stacks"] >= floor else "*") for r in r2[m_]["rows"][g]))
    out(f"      observed recorded holes per stack-repeat: mean {y.mean():.2f}, zero in {(y == 0).mean():.3f}")
    # ================================================================ reported only (H2b's items)
    dbin = S2.dist_bin(pred["dist_h"])
    rec_all = Ro[:, -1].reshape(n, -1).sum(1)
    sub10 = rec_all >= 10

    def recall(lab, mask_items=None):
        keep = np.ones(len(lab), bool) if mask_items is None else np.repeat(mask_items, nh)
        rows = {m_: [(float(pred[f"share_{m_}"][keep & lab & (dbin == j)].mean()) if (keep & lab & (dbin == j)).any() else float("nan"),
                      float(pred[f"share_{m_}"][keep & ~lab & (dbin == j)].mean()) if (keep & ~lab & (dbin == j)).any() else float("nan"))
                     for j in range(4)] for m_ in R2_M}
        return rows, [(int((keep & lab & (dbin == j)).sum()), int((keep & ~lab & (dbin == j)).sum())) for j in range(4)]
    rc, ncs = recall(lab_s)
    out("   recall by distance to the nearest kept record (report; recorded / unrecorded hole tetrads; cells "
        + ", ".join(f"{DIST[j]} {ncs[j][0]:,}/{ncs[j][1]:,}" for j in range(4)) + "): " + " || ".join(
            f"{NAME[m_]} " + "; ".join(f"{DIST[j]} {rc[m_][j][0]:.3f}/{rc[m_][j][1]:.3f}" for j in range(4)) for m_ in R2_M))
    rcp, ncp = recall(lab_p)
    obs_all = rec_all.astype(float)
    obs_eoo = S2.eoo_of(Ro[:, -1])
    tmean = {m_: pred[f"thin_s_{m_}"].astype(float).mean(1) for m_ in MAP_M}
    tmean["pooled"] = np.concatenate([pred["thin_s_s1"][:, :25], pred["thin_s_s2"][:, :25]], 1).astype(float).mean(1)
    emean = {m_: pred[f"eoo_s_{m_}"].astype(float).mean(1) for m_ in MAP_M}
    emean["pooled"] = np.concatenate([pred["eoo_s_s1"][:, :25], pred["eoo_s_s2"][:, :25]], 1).astype(float).mean(1)
    ratio = lambda p_, o_, m: float(p_[m].mean() / o_[m].mean()) if m.any() and o_[m].mean() > 0 else float("nan")
    allrows = {m_: {g: [ratio(tmean[m_], obs_all, gi == j) for j in range(len(labs))] for g, (gi, labs) in grp.items()} for m_ in R2_M}
    out("   recorded count over every 2019 surveyed tetrad / all 2019 records, no correction (report): " + " || ".join(
        f"{NAME[m_]} " + " | ".join("; ".join(f"{labs[j]} {allrows[m_][g][j]:.2f}" for j in range(len(labs))) for g, (gi, labs) in grp.items())
        for m_ in ("s1", "s2", "B4", "B5")))
    cap = obs_eoo >= S2.EOO_CAP
    eoorows = {m_: dict(below=ratio(emean[m_], obs_eoo, ~cap), cap=ratio(emean[m_], obs_eoo, cap),
                        by_class=[ratio(emean[m_], obs_eoo, rcls == j) for j in range(len(TCLS))]) for m_ in R2_M}
    out(f"   EOO of thinned recorded cells / EOO of all 2019 records (report; below the cap {int((~cap).sum()):,}, at {S2.EOO_CAP}+ "
        f"{int(cap.sum()):,}): " + "; ".join(f"{NAME[m_]} {eoorows[m_]['below']:.2f} below, {eoorows[m_]['cap']:.2f} at cap" for m_ in R2_M))
    status = np.where(status == "", "no status", status)
    stl = sorted(set(status.tolist()))
    sti = np.array([stl.index(s_) for s_ in status])
    by_status = {}
    for m_ in ("s1", "s2"):
        cb = hr1_core(A[m_] - A["B1"], stid, sti, wname, list(range(len(stl))))
        by_status[m_] = dict(hr1={stl[j]: dict(cb[j]) for j in range(len(stl))},
                             hr2={r_["group"]: r_ for r_ in S2.evaluate_h(y, r2[m_]["cf"], {"s": (sti, stl)}, stid)["s"]})
    out("   GB Red List 2022 status (report; seed - B1 stored [interval] (stacks) | H-R2 ratio/coverage): " + " || ".join(
        f"{NAME[m_]} " + "; ".join(f"{s_} {by_status[m_]['hr1'][s_]['mean']:+.3f} [{by_status[m_]['hr1'][s_]['lo']:+.3f}, "
                                   f"{by_status[m_]['hr1'][s_]['hi']:+.3f}] ({by_status[m_]['hr1'][s_]['n']}) | "
                                   f"{by_status[m_]['hr2'][s_]['ratio']:.2f}/{by_status[m_]['hr2'][s_]['cov']:.2f}" for s_ in stl) for m_ in ("s1", "s2")))
    r10 = {m_: {b: S2.hr1_rows(hr1_core(A[m_] - A[b], stid, np.where(sub10, 0, -1), wname, [0]), ["10+"], floor)[0] for b in ("B1", "B1x")}
           for m_ in ("s1", "s2")}
    out(f"   stacks with 10+ records in 2019, the 7 Nov version ({int(sub10.sum()):,} stack-repeats, {len(np.unique(stid[sub10])):,} stacks; "
        f"report): " + "; ".join(f"{NAME[m_]} - {NAME[b]} {r10[m_][b]['mean']:+.4f} [{r10[m_][b]['lo']:+.4f}, {r10[m_][b]['hi']:+.4f}]"
                                 for m_ in ("s1", "s2") for b in ("B1", "B1x")))
    # ================================================================ D1 / D2 (report only; the diagnostic's code)
    sc_d = {m_: pred[f"score_{m_}"].astype(np.float64) for m_ in DG.METH}
    sc_d["B4p"] = 1 / (1 + np.exp(-np.clip(sc_d["B4"], -700, 700)))
    wu, wi = np.unique(wname, return_inverse=True)
    bd = DG.dist_bin(pred["dist_h"].astype(float))
    resD, decD, ptD, psD = DG.analyse(ptr, lab_s.astype(bool), bd, sc_d, stid, np.asarray(wi).ravel(), rcls, rbin, len(wu))
    wD, nanD = DG.check_auc(decD, {m_: A[m_] for m_ in DG.METH})
    bD = DG.brute_check(ptr, lab_s.astype(bool), bd, {m_: sc_d[m_] for m_ in DG.METH}, decD, psD)
    out("   == D1 / D2 on the test half (report only; label = recorded in 2019; holes)")
    for gname, g_ in resD.items():
        out(f"   {gname} ({g_['stack_repeats']:,} items)")
        out("      D1 bin: cells, labelled share | mean score s1, s2, B4 chance: " + "; ".join(
            f"{k} {v['cells']:,} {v['labelled']:.3f} | {v['s1']:.3f} {v['s2']:.3f} {v['B4p']:.3f}" for k, v in g_["d1"].items()))
        out("      D1 far/near ratio: labelled {:.3f}; ".format(g_["a"]["s1"]["ratio_obs"]) + "; ".join(
            f"{m_} {g_['a'][m_]['ratio']:.3f} (minus labelled {g_['a'][m_]['diff']:+.3f} [{g_['a'][m_]['lo']:+.3f}, {g_['a'][m_]['hi']:+.3f}])"
            for m_ in ("s1", "s2", "B4p")))
        d2 = g_["d2"]
        out(f"      D2 pairs in different distance bins {d2['pairs in different bins (share)']['mean']:.3f}; " + "; ".join(
            f"{k}: gap {v['gap']:+.4f} [{v['gap_lo']:+.4f}, {v['gap_hi']:+.4f}] = different-bin {v['diff_part']:+.4f} + same-bin "
            f"{v['same_part']:+.4f} ({v['stacks']} stacks)" for k, v in d2.items() if k.startswith("s")))
    rlD = DG.rule(resD)
    out("   the fit-half decision rule computed on the test half, as description only: " + "; ".join(
        f"{m_} (a) {rlD[m_]['a']}, (b) vs B1 stored {rlD[m_]['b_B1']}, vs B1 exact {rlD[m_]['b_B1x']}" for m_ in ("s1", "s2")))
    gate("Z6", wD <= 1e-12 and nanD and bD <= 1e-9, f"report-only code checks: D2 AUCs = H-R1 AUCs, largest difference {wD:.1e}, same "
         f"not-rankable set {nanD}; D2 decomposition = hand count of every pair on 200 stack-repeats, largest difference {bD:.1e}")
    # ---- Z5
    if broken() == "pred_changed":
        z_ = dict(np.load(pf)); z_["thin_h_s1"] = z_["thin_h_s1"] + 1; X.save_npz(pf, z_)
    mk_last = json.loads(mk.read_text().splitlines()[-1])
    same_p = X.file_md5(pf) == mpred
    gate("Z5", p1_opened == 0 and not early and same_p and mk_last.get("predicted_md5") == mpred,
         f"order: test truth files opened in phase 1 {p1_opened}; label arrays read before h6_predicted.npz was written {len(early)}; "
         f"h6_predicted.npz unchanged after the labels were read {same_p}; marker holds its md5 {mk_last.get('predicted_md5') == mpred}")
    # ================================================================ files
    scz = dict(y_strict=y, y_pooled=yp, rankable=rank, obs_all_2019=obs_all, obs_eoo_2019=obs_eoo, stack_id=stid,
               predicted_md5=np.array(mpred), rerun=np.bool_(rerun))
    for m_ in SCORE_M:
        scz[f"auc_strict_{m_}"] = A[m_]; scz[f"auc_pooled_{m_}"] = Ap[m_]
    for m_ in R2_M:
        for q in ("mean", "L", "U"):
            scz[f"hr2_{q}_{m_}"] = r2[m_]["cf"][q]
        scz[f"hr2_scale_q_{m_}"] = r2[m_]["cf"]["par"]
    X.save_npz(od / "h6_scores.npz", scz)
    with open(od / "per_stack_h6.csv", "w", newline="") as fh:
        wr_ = csv.writer(fh)
        wr_.writerow(["item", "window", "species", "status", "repeat", "stack_id", "kept_rec_2019", "kept_rec_6y", "recorded_class",
                      "records_bin", "detection_class", "holes", "recorded_holes_strict", "recorded_holes_pooled", "all_rec_2019"]
                     + [f"auc_{m_}" for m_ in SCORE_M] + [f"hr2_{q}_{m_}" for m_ in R2_M for q in ("mean", "lo", "hi")])
        for i in range(n):
            wr_.writerow([i, wname[i], sp_t[i], status[i], int(idx["repeat"][i]), int(stid[i]), int(rec_t[i]), int(rec6[i]), TCLS[rcls[i]],
                          BINS[rbin[i]], DLAB[dcls[i]], int(nh[i]), int(y[i]), int(yp[i]), int(rec_all[i])]
                         + ["" if np.isnan(A[m_][i]) else f"{A[m_][i]:.6f}" for m_ in SCORE_M]
                         + [f"{r2[m_]['cf'][q][i]:.3f}" for m_ in R2_M for q in ("mean", "L", "U")])
    wrows = []
    for w_ in sorted(set(wname.tolist())):
        m = wname == w_
        r_ = dict(window=w_, stack_repeats=int(m.sum()), stacks=len(np.unique(stid[m])), rankable=int((rank & m).sum()))
        for m_ in ("s1", "s2"):
            dd = (A[m_] - A["B1"])[m]; ok_ = ~np.isnan(dd)
            if ok_.any():
                _, inv = np.unique(stid[m][ok_], return_inverse=True); inv = np.asarray(inv).ravel()
                r_[f"mean_d_{m_}_B1"] = float((np.bincount(inv, weights=dd[ok_]) / np.bincount(inv)).mean())
            else:
                r_[f"mean_d_{m_}_B1"] = float("nan")
            cf = r2[m_]["cf"]
            r_[f"hr2_ratio_{m_}"] = float(cf["mean"][m].mean() / y[m].mean()) if y[m].mean() > 0 else float("nan")
            r_[f"hr2_cov_{m_}"] = float(((y[m] >= cf["L"][m]) & (y[m] <= cf["U"][m])).mean())
        wrows.append(r_)
    with open(od / "per_window_h6.csv", "w", newline="") as fh:
        wd = csv.DictWriter(fh, fieldnames=list(wrows[0])); wd.writeheader(); wd.writerows(wrows)
    clean = lambda o_: json.loads(json.dumps(o_, default=lambda v: v.item() if hasattr(v, "item") else str(v)).replace("-Infinity", "null")
                                  .replace("Infinity", "null").replace("NaN", "null"))
    rep.update(gates=res, n=int(n), stacks=len(np.unique(stid)), rankable=int(rank.sum()), rankable_stacks=len(np.unique(stid[rank])),
               hole_share=dict(strict=sh_s, pooled=sh_p), floor=floor, thinning=tz,
               md5=dict(tool=X.file_md5(Path(__file__)), freeze=fmd5, predicted=mpred, scores=X.file_md5(od / "h6_scores.npz"),
                        h5_report=X.file_md5(h5 / "h2a_report.json"), frozen=mf, best_pt=best),
               H_R1={m_: dict(verdict=v["verdict"], **{b: dict(verdict=v[b]["verdict"], rows=v[b]["rows"]) for b in ("B1", "B1x")}) for m_, v in r1.items()},
               H_R1_baselines={m_: dict(verdict=v["verdict"], rows=v["B1"]["rows"]) for m_, v in r1b.items()},
               H_R1_pooled_truth={m_: dict(verdict=v["verdict"], **{b: dict(verdict=v[b]["verdict"], rows=v[b]["rows"]) for b in ("B1", "B1x")}) for m_, v in r1p.items()},
               H_R2={m_: dict(chosen[m_], rows=r2[m_]["rows"]) for m_ in R2_M},
               recall=dict(strict={m_: rc[m_] for m_ in R2_M}, strict_cells=ncs, pooled={m_: rcp[m_] for m_ in R2_M}, pooled_cells=ncp, bins=list(DIST)),
               survey_count_ratio=allrows, eoo=eoorows, red_list=by_status, seven_nov=dict(hr1=r10), d1_d2=resD, d_rule_description=rlD,
               minutes=round((time.time() - t0) / 60, 1), written_utc=now())
    (od / "h6_report.json").write_text(json.dumps(clean(rep), indent=1))
    n_ok = sum(res.values())
    out(f"   written h6_scores.npz md5 {rep['md5']['scores']}, per_stack_h6.csv, per_window_h6.csv, h6_report.json ({(time.time() - t0) / 60:.1f} min)")
    out(f"{tag}stage H6: gates {n_ok} PASS, {len(res) - n_ok} FAIL; H-R1 (both B1 versions) seed 1 {r1['s1']['verdict']}, seed 2 "
        f"{r1['s2']['verdict']}, pooled {r1['pooled']['verdict']}; H-R2 " + ", ".join(f"{NAME[m_]} {'PASS' if chosen[m_]['ok'] else 'FAIL'}" for m_ in R2_M))
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--bnm-test", default="../data/bnm_test")
    ap.add_argument("--h5", default="../data/stage_h5_predict")
    ap.add_argument("--bnm-fit", default="../data/bnm_fit")
    ap.add_argument("--h2a", default="../data/stage_h2_predict")
    ap.add_argument("--h2b", default="../data/stage_h2b_score")
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    ap.add_argument("--freeze", default="../data/stage_h3_freeze/h3_freeze.json")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--out", default="../data/stage_h6_test_score")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="phase 1 only (Z0-Z3): no label read, nothing written, no marker")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_23_h6_test_score.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_23_h6_test_score", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.self_test(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.data, a.bnm_test, a.h5, a.bnm_fit, a.h2a, a.h2b, a.g2, a.g3, a.freeze, a.runs, a.obj1_tools, a.out, a.rerun,
               check_only=a.check_only, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)