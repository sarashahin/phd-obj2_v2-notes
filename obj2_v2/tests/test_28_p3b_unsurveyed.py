#!/usr/bin/env python3
"""
test_28_p3b_unsurveyed.py  (obj2_v2, products step P3b, 13 Nov 2026): self-test of tools/stage_p3b_unsurveyed_check.py.
Run through the tool, from PSD_Dispersal_pool/obj2_v2/work:
    python3 ../tools/stage_p3b_unsurveyed_check.py --self-test
It builds test_27's planted world in a temporary folder (planted P1 export with the real H1 and P1 tools, planted P2
samples, G5a fits, G5b record, H6 report and freeze file), runs the real P3 tool on it, then P3b. Nothing of ../data or
../runs is read or written. In the planted grids about a fifth of each window is never surveyed.
    U1  occ_counts and share_of by hand: 3 maps on a 20 x 20 window with known cells, shares 0, 1 and 0.5; a row whose
        maps are all empty gives NaN
    S0  the planted P1 export and the planted P3 run pass every gate
    T1  P3b on them: gates D0-D4 PASS
    T2  independent check, fresh code from the files (no tool function): surveyed tetrads of every window from the
        input files' effort bits, every share from the sample bits, the surveyed-only line from g5_fits.npz read with
        np.load = unsurveyed_bnm.npz (1e-12), unsurveyed_bnm.csv (its rounding) and per_window_unsurveyed.csv
    T3  meaning: the planted maps put occupied cells at random over the window, so the mean share must equal the mean
        never-surveyed fraction of the windows (within 0.02), and the surveyed-only line must be below P3's AOO
    T4  a second run into a new folder gives the same md5 for unsurveyed_bnm.csv, unsurveyed_bnm.npz and the per-window csv
    T5  refused: a run into an existing folder, into an input folder, a path with real_eval; D0 fails and nothing is
        written when products_bnm.npz is changed after P3
    bugs: effort of 2019 only (D1, the two paths differ); the scale of seed 1 (D2); the surveyed share reported (D2, the
        loop code differs); seed columns swapped in the csv (D3); a truth file opened (D4, nothing written)
"""
import csv, hashlib, importlib.util, json, os, shutil, tempfile
from pathlib import Path
import numpy as np

T_EDGES, B_EDGES = np.array([0, 1, 2, 6, 11, 21, 51]), np.array([1, 4, 7, 11, 21])


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def indep(out_dir, bnm, p2, fits_dir, p3_dir, runs):
    """Fresh code from the files: surveyed tetrads, shares and the surveyed-only line."""
    fb = np.load(bnm / "bnm_frozen.npz"); st = np.load(bnm / "bnm_stacks.npz")
    win = st["window"].astype(str); n = len(win)
    with open(bnm / "manifest.csv") as fh:
        man = {r["window"]: r for r in csv.DictReader(fh)}
    surv = {}
    for w, r in man.items():
        e = np.load(bnm / r["input_file"])["effort"]                          # (S, 6, 50) packed
        bits = np.unpackbits(e, axis=-1)[..., :400].reshape(e.shape[0], e.shape[1], 20, 20)
        surv[w] = bits.any((0, 1))
    U = np.stack([~surv[w] for w in win])
    occ = {}
    for m, r in zip(("s1", "s2"), runs):
        mp = np.load(p2 / f"samples_{Path(r).name}.npz")["maps"]
        occ[m] = np.unpackbits(mp[..., 250:], axis=-1)[..., :400].reshape(n, mp.shape[1], 20, 20).astype(bool)
    occ["pooled"] = np.concatenate([occ["s1"][:, :25], occ["s2"][:, :25]], 1)
    share = {}
    for m in ("pooled", "s1", "s2"):
        t = occ[m].sum((1, 2, 3)).astype(float); u = (occ[m] & U[:, None]).sum((1, 2, 3)).astype(float)
        share[m] = np.where(t > 0, u / np.maximum(t, 1), np.nan)
    rec, rec6 = fb["rec_t"].astype(int), fb["rec6"].astype(int)
    rcls = np.searchsorted(T_EDGES, rec, side="right") - 1
    rbin = np.clip(np.searchsorted(B_EDGES, rec6, side="right") - 1, 0, 4)
    fz = np.load(fits_dir / "g5_fits.npz")
    opt = int(fz["corr__pooled__aoo__option"])
    s = fz["corr__pooled__aoo__scale"][rcls if opt == 1 else rcls * 5 + rbin]
    nsurv = (~U).reshape(n, -1).sum(1)
    mean_s = (occ["pooled"] & ~U[:, None]).sum((2, 3)).mean(1)
    aoo_s = np.minimum(s * mean_s, nsurv)
    z = np.load(out_dir / "unsurveyed_bnm.npz")
    worst = max(float(np.nanmax(np.abs(share[m] - z[f"share_{m}"]))) for m in share)
    worst = max(worst, float(np.abs(aoo_s - z["aoo_surveyed_only_pooled"]).max()), float(np.abs(nsurv - z["surveyed"]).max()))
    nan_same = all(np.array_equal(np.isnan(share[m]), np.isnan(z[f"share_{m}"])) for m in share)
    with open(out_dir / "unsurveyed_bnm.csv") as fh:
        rows = list(csv.DictReader(fh))
    f = lambda c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows])
    ok_csv = len(rows) == n and all(np.nanmax(np.abs(f(f"share_never_surveyed_{m}") - share[m])) <= 5.01e-5 for m in share) and \
        float(np.abs(f("aoo_surveyed_only_pooled_report") - aoo_s).max()) <= 5.01e-4
    with open(out_dir / "per_window_unsurveyed.csv") as fh:
        pw = {r["window"]: r for r in csv.DictReader(fh)}
    ok_pw = set(pw) == set(surv) and all(int(pw[w]["never_surveyed"]) == int((~surv[w]).sum()) for w in surv)
    p3 = np.load(p3_dir / "products_bnm.npz")
    return dict(worst=worst, nan_same=nan_same, csv=ok_csv, pw=ok_pw, share=share, U=U, aoo_s=aoo_s, aoo_p3=p3["mean_pooled"],
                frac=np.array([(~surv[w]).mean() for w in win]))


def main(tool_path, tools):
    tool_path = Path(tool_path)
    P3B = load(tool_path, "stage_p3b_unsurveyed_check")
    P3 = P3B.P3
    T27 = load(tool_path.parent.parent / "tests" / "test_27_p3_products.py", "test_27_p3_products")
    P1 = load(tool_path.parent / "stage_p1_bnm_all_export.py", "stage_p1_bnm_all_export")
    H1, X = P1.H1, P3.X
    quiet = lambda *a, **k: None
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- U1 by hand
    occ = np.zeros((1, 3, 6, 20, 20), bool)
    occ[0, 0, -1, 0, :4] = True            # 4 cells, all surveyed
    occ[0, 1, -1, 19, :2] = True           # 2 cells, both never surveyed
    occ[0, 2, -1, 0, 0] = True; occ[0, 2, -1, 19, 0] = True       # 1 surveyed, 1 never surveyed
    Um = np.zeros((1, 20, 20), bool); Um[0, 19, :] = True
    pk = P3.GS.pack(occ)
    tot, inu = P3B.occ_counts(pk, Um)
    one = [P3B.share_of(tot[:, [k]], inu[:, [k]])[0] for k in range(3)]
    allm = P3B.share_of(tot, inu)[0]
    empty = P3B.share_of(np.zeros((1, 3), int), np.zeros((1, 3), int))[0]
    t("U1", list(tot[0]) == [4, 2, 2] and list(inu[0]) == [0, 2, 1] and np.allclose(one, [0, 1, 0.5]) and abs(allm - 3 / 8) < 1e-15
      and np.isnan(empty), f"by hand: occupied {[int(v) for v in tot[0]]}, never-surveyed {[int(v) for v in inu[0]]}, shares "
      f"{[float(v) for v in one]}, over the 3 maps "
      f"{allm} (= 3/8); all-empty maps give NaN")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = H1.world(tmp / "bw")
        data = tmp / "data"
        fit, test, al = data / "bnm_fit", data / "bnm_test", data / "bnm_all"
        rf, _ = H1.run(py, refpat, tools, fit, expect=(6, 6), out=quiet)
        fzh = tmp / "fz_h1.json"
        fzh.write_text(json.dumps(dict(tools={"stage_h1_bnm_export.py": X.file_md5(P1.H1_FILE)})))
        rt_, _ = H1.run(py, refpat, tools, test, expect=(6, 6), half="test", final_test=True, freeze=str(fzh), out=quiet)
        rp, _ = P1.run(py, refpat, tools, fit, test, al, expect=(6, 6), out=quiet)
        kw, h6 = T27.plant(tmp, al, P3)
        p3d = data / "p3"
        r3, _ = P3.run(out_dir=p3d, out=quiet, **kw)
        t("S0", all(rf.values()) and all(rt_.values()) and all(rp.values()) and all(r3.values()) and set(r3) == {"C0", "C1", "C2", "C3", "C4"},
          f"planted P1 export {rp}; planted P3 run {r3}")
        kb = dict(bnm=al, p2=kw["p2"], p3=p3d, fits_dir=kw["fits_dir"], runs=kw["runs"],
                  expect=dict(p1_report=kw["expect"]["p1_report"], p2_report=kw["expect"]["p2_report"], p3_report=md5(p3d / "p3_report.json")))
        lines = []
        out1 = data / "p3b"
        r, rep = P3B.run(out_dir=out1, out=lines.append, **kb)
        t("T1", set(r) == {"D0", "D1", "D2", "D3", "D4"} and all(r.values()), f"P3b on the planted inputs: {r}")
        if not all(r.values()):
            print("      " + "\n      ".join(lines))
        ind = indep(out1, al, Path(kw["p2"]), Path(kw["fits_dir"]), p3d, kw["runs"]) if out1.exists() else None
        t("T2", ind is not None and ind["worst"] <= 1e-12 and ind["nan_same"] and ind["csv"] and ind["pw"],
          f"independent recomputation from the files: npz largest difference {ind['worst'] if ind else float('nan'):.1e}, same "
          f"undefined rows {ind and ind['nan_same']}; csv = it within rounding {ind and ind['csv']}; per-window never-surveyed "
          f"counts {ind and ind['pw']}")
        if ind is not None:
            ms, mf = float(np.nanmean(ind["share"]["pooled"])), float(ind["frac"].mean())
            below = bool((ind["aoo_s"] <= ind["aoo_p3"] + 1e-9).all())
        else:
            ms, mf, below = float("nan"), float("nan"), False
        t("T3", abs(ms - mf) <= 0.02 and 0.05 < mf < 0.5 and below, f"maps at random over the window: mean share {ms:.3f} against "
          f"the mean never-surveyed fraction {mf:.3f}; surveyed-only line below P3's AOO in every row {below}")
        out2 = data / "p3b_again"
        P3B.run(out_dir=out2, out=quiet, **kb)
        same = all(X.file_md5(out1 / f) == X.file_md5(out2 / f) for f in ("unsurveyed_bnm.csv", "unsurveyed_bnm.npz", "per_window_unsurveyed.csv"))
        t("T4", same, "a second run into a new folder gives the same md5 for both csv files and the npz")
        stops = []
        for o in (out1, p3d / "inside", data / "real_eval_p3b"):
            try:
                P3B.run(out_dir=o, out=quiet, **kb); stops.append(False)
            except SystemExit:
                stops.append(True)
        p3b_ = tmp / "p3_tampered"
        shutil.copytree(p3d, p3b_)
        zz = dict(np.load(p3b_ / "products_bnm.npz")); zz["mean_pooled"] = zz["mean_pooled"] + 0.001
        X.save_npz(p3b_ / "products_bnm.npz", zz)
        rq, _ = P3B.run(out_dir=data / "p3b_tampered", out=quiet, **dict(kb, p3=p3b_))
        t("T5", all(stops) and rq.get("D0") is False and not (data / "p3b_tampered").exists(),
          f"refused: an existing folder {stops[0]}, a folder inside an input folder {stops[1]}, a real_eval path {stops[2]}; "
          f"products changed after P3 fail D0 and nothing is written {rq}")
        for bug, g_, written in (("one_year", "D1", False), ("scale_s1", "D2", False), ("share_surveyed", "D2", False),
                                 ("csv_swap", "D3", True), ("opentruth", "D4", False)):
            os.environ["STAGE_P3B_BROKEN"] = bug
            try:
                rb, _ = P3B.run(out_dir=data / f"bug_{bug}", out=quiet, **kb)
            finally:
                os.environ.pop("STAGE_P3B_BROKEN", None)
            ex = (data / f"bug_{bug}").exists()
            t(f"bug {bug}", rb.get(g_) is False and ex == written, f"{g_} fails as it must; written {ex} (expected {written}) ({rb})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)