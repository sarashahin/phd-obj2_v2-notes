#!/usr/bin/env python3
"""
test_23_h6_test_score.py  (obj2_v2, self-test of tools/stage_h6_bnm_test_score.py, 12 Nov 2026)

Run through the tool:  python3 ../tools/stage_h6_bnm_test_score.py --self-test | tail -1   (from obj2_v2/work)
Planted data only, in a temporary folder, with the real tools: the planted pipeline of H2a's self-test (P5.build_world: two
tiny runs with step 3a, G1, G2), G3 on its real_val, H1's planted grids exported as a fit half and (after the freeze) as a
test half, the three H2a parts on each, H2b (stage_h2_bnm_score.run, floor 5) on the fit half, and an h3_freeze.json built
here the way stage_h3_freeze.py writes it (tool md5s from H3.tool_md5s, the H3 data keys, fits from H3.resolve on all fit
items). Nothing of ../data is read or written.
    U1  thinning z: independent uniforms give per-method z and robust z near each other; the 4-method robust z counts
        shared uniforms once per stack-repeat and sample
    T0  --check-only: gates Z0-Z3 PASS, no output folder, no marker
    T1  H6 on the planted test half (floor 5 so groups are tested): gates Z0-Z6 PASS, marker written
    T2  recomputed outside the tool from the truth and map files on 60 stack-repeats: seed 1 AUC (export_obj2.auc); H-R2 mean,
        lower and upper end from the frozen fit and thinned counts rebuilt from the maps and the keyed uniforms
    T3  refused: a second reading without --rerun, an output inside the test export, the fit-half export as test; --rerun
        into a new folder runs and says rerun
    T4  planted bugs, each fails its gate: H-R1 code shifted (Z2), frozen-fit code off by 0.1% (Z2), thinning at 0.9 (Z3),
        a label read before the predictions are written (Z5), a test truth file opened in phase 1 (Z5), h6_predicted.npz
        changed after the labels (Z5), a test-half samples file changed (Z0), a frozen fit edited (Z2), another freeze file
        (Z0)
"""
import importlib.util, json, os, shutil, tempfile
from pathlib import Path
import numpy as np


def load_tool(tool_path):
    spec = importlib.util.spec_from_file_location("stage_h6_bnm_test_score_t", tool_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def build_freeze(T, data, bnm, h2a, h2b, g2, g3, runs, obj1, path):
    """h3_freeze.json as stage_h3_freeze.run writes it (F5 keys), for the planted fit half."""
    S2, H3, X = T.S2, T.H3, T.X
    ja, jb = json.loads((h2a / "h2a_report.json").read_text()), json.loads((h2b / "h2b_report.json").read_text())
    P, Sc = np.load(h2b / "h2b_predicted.npz"), np.load(h2b / "h2b_scores.npz")
    pc, y, n = H3.preds(P), Sc["y_strict"].astype(float), len(P["stack_id"])
    fits, opts = {}, {m: int(jb["H_R2"][m]["option"]) for m in S2.R2_M}
    for m, opt in opts.items():
        fg = H3.group_of(P["rcls"], P["rbin"], opt)
        ng = len(S2.TCLS) if opt == 1 else len(S2.TCLS) * len(S2.BINS)
        tab = H3.resolve(pc[m], y, np.arange(n), range(ng), fg, opt, P["stack_id"])
        fits[m] = dict(option=opt, groups={str(g): dict(label=str(g), **t) for g, t in tab.items()})
    sampler = {}
    for r in runs:
        z = np.load(h2a / f"samples_{r.name}.npz")
        sampler[r.name] = dict(K=int(z["K"]), seed=int(z["seed"]), ddim=int(z["ddim"]), batch=int(z["batch"]), epoch=int(z["epoch"]),
                               best_md5=X.file_md5(r / "best.pt"))
    dfiles = {f"{r.name}/best.pt": r / "best.pt" for r in runs} | {f"{r.name}/check_g1/samples_k50.npz": r / "check_g1" / "samples_k50.npz" for r in runs} \
        | {"stage_g2_baselines_v2/g2_fit.json": g2 / "g2_fit.json", "stage_g3_calibration/g3_calibrated.npz": g3 / "g3_calibrated.npz"} \
        | {f"bnm_fit/{f}": bnm / f for f in ("h1_report.json", "bnm_frozen.npz", "bnm_stacks.npz", "manifest.csv")} \
        | {f"stage_h2_predict/{f}": h2a / f for f in ["h2a_report.json"] + sorted(ja["files"])} \
        | {f"stage_h2b_score/{f}": h2b / f for f in ("h2b_predicted.npz", "h2b_scores.npz", "h2b_report.json", "per_stack_h2b.csv", "per_window_h2b.csv")}
    bdc = Path(obj1) / "bnm_detection_check.py"
    fz = dict(stage="H3 freeze", half="fit", tools=H3.tool_md5s(), obj1_tools={"bnm_detection_check.py": X.file_md5(bdc)},
              data={k: X.file_md5(p) for k, p in dfiles.items()}, sampler=sampler, settings=H3.SIGNED, options=opts, fits=fits)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(fz, indent=1))
    return path


def build_world(T, tmp, obj1):
    quiet = lambda *a, **k: None
    S2, H3, H2A, GC = T.S2, T.H3, T.H2A, T.GC
    H1, P5 = H3.H1, S2.P5
    w = P5.build_world(tmp, obj1)
    data, g2, runs = Path(w["data"]), Path(w["g2"]), [Path(r) for r in w["runs"]]
    g3 = data / "stage_g3_calibration"
    r3, _ = GC.run(data, runs, g2, g3, out=quiet)
    py, refpat = H1.world(Path(tmp) / "bw")
    bnm, h2a, h2b = data / "bnm_fit", data / "stage_h2_predict", data / "stage_h2b_score"
    r1, _ = H1.run(py, refpat, obj1, bnm, expect=(6, 6), out=quiet)
    ra = [H2A.part_baselines(data, bnm, g2, h2a, out=quiet)] + [H2A.part_samples(data, bnm, r, h2a, "cpu", out=quiet) for r in runs]
    ra.append(H2A.part_report(bnm, runs, h2a, out=quiet))
    rb, _ = S2.run(data, bnm, h2a, runs, g3, h2b, floor=5, out=quiet)
    fzp = build_freeze(T, data, bnm, h2a, h2b, g2, g3, runs, obj1, data / "stage_h3_freeze" / "h3_freeze.json")
    bt, h5 = data / "bnm_test", data / "stage_h5_predict"
    rt, _ = H1.run(py, refpat, obj1, bt, expect=(6, 6), half="test", final_test=True, freeze=str(fzp), out=quiet)
    r5 = [H2A.part_baselines(data, bt, g2, h5, out=quiet)] + [H2A.part_samples(data, bt, r, h5, "cpu", out=quiet) for r in runs]
    r5.append(H2A.part_report(bt, runs, h5, out=quiet))
    ok = all(r3.values()) and all(r1.values()) and all(all(x.values()) for x in ra) and all(rb.values()) and all(rt.values()) \
        and all(all(x.values()) for x in r5)
    kw = dict(data=data, bnm_test=bt, h5=h5, bnm_fit=bnm, h2a=h2a, h2b=h2b, g2=g2, g3=g3, freeze=fzp, runs=runs, obj1_tools=obj1,
              freeze_md5=T.X.file_md5(fzp), diag_md5=T.DIAG_MD5, floor_override=5)
    txt = (f"G3 {all(r3.values())}, H1 fit {all(r1.values())}, H2a fit parts {[all(x.values()) for x in ra]}, H2b {all(rb.values())}, "
           f"H1 test {all(rt.values())}, H2a test parts {[all(x.values()) for x in r5]}")
    return kw, ok, txt


def self_test(tool_path, obj1_tools="../../obj1_v2/tools"):
    T = load_tool(tool_path)
    S2, X, D, GC = T.S2, T.X, T.D, T.GC
    G, K = X.G, T.K
    quiet = lambda *a, **k: None
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)
    # U1 thinning z
    g = np.random.default_rng(5)
    n, det = 3000, [0.643, 0.642, 0.8]
    dcls = np.arange(n) % 3
    occ = {k: g.integers(0, 6, (n, K)).astype(np.uint16) for k in ("s1", "s2", "B3", "B4")}
    thin = {k: g.binomial(occ[k], np.array(det)[dcls][:, None]).astype(np.uint16) for k in occ}
    tz = T.thin_z(occ, thin, dcls, det)
    zs = [tz[c][k]["z"] for c in tz for k in occ]; zr = [tz[c][k]["z_rob"] for c in tz for k in occ]
    shared = {k: occ["s1"] for k in occ}; thin_sh = {k: thin["s1"] for k in occ}
    tz2 = T.thin_z(shared, thin_sh, dcls, det)
    t("U1", all(abs(a) <= 4 for a in zs) and max(abs(a - b) for a, b in zip(zs, zr)) < 0.5
      and all(abs(tz2[c]["pooled_z_rob"] - tz2[c]["s1"]["z_rob"]) < 1e-9 for c in tz2),
      f"thinning z: independent uniforms give per-method z {min(zs):+.2f} to {max(zs):+.2f}, robust z within 0.5 of them; 4 copies of one "
      f"method (shared uniforms) give a 4-method robust z equal to the one-method robust z (counted once per cluster)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kw, ok, txt = build_world(T, tmp, obj1_tools)
        t("S0", ok, f"planted pipeline: {txt}")
        rc, _ = T.run(out_dir=tmp / "o_check", out=quiet, check_only=True, **kw)
        t("T0", set(rc) == {"Z0", "Z1", "Z2", "Z3"} and all(rc.values()) and not (tmp / "o_check").exists() and not (kw["data"] / T.MARKER).exists(),
          f"--check-only: gates {rc}; nothing written, no marker")
        lines = []
        r, rep = T.run(out_dir=tmp / "o1", out=lines.append, **kw)
        t("T1", set(r) == {f"Z{i}" for i in range(7)} and all(r.values()) and (kw["data"] / T.MARKER).is_file(),
          f"H6 on the planted test half (floor 5): {r}; marker written")
        # T2 recomputed outside the tool
        sc = np.load(tmp / "o1" / "h6_scores.npz"); fz = json.loads(Path(kw["freeze"]).read_text())
        bnm = kw["bnm_test"]; fb = dict(np.load(bnm / "bnm_frozen.npz")); n = len(fb["row"])
        man = {r_["input_file"]: r_ for r_ in D.rows_of(bnm)}
        H = np.zeros((n, G, G), bool); Ls = np.zeros_like(H)
        for i in np.unique(fb["file_index"]):
            k = np.nonzero(fb["file_index"] == i)[0]
            z = np.load(bnm / man[str(fb["files"][i])]["truth_file"])
            H[k] = X.unpack(z["hole"])[fb["row"][k]]; Ls[k] = X.unpack(z["label_strict"])[fb["row"][k]]
        occ = GC.target_year(np.load(kw["h5"] / f"samples_{kw['runs'][0].name}.npz")["maps"])
        V1, V2 = S2.GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])[:2]
        V = V1 | V2
        rec_t, rec6 = V[:, -1].reshape(n, -1).sum(1), V.any(1).reshape(n, -1).sum(1)
        p_ = np.array(S2.DET)[S2.det_class(rec6)]; rc = D.tidx(rec_t)
        ok_a, ok_r = True, True
        for i in np.sort(np.random.default_rng(3).choice(n, min(60, n), replace=False)):
            lab = Ls[i][H[i]]; s_ = occ[i].sum(0)[H[i]] / K
            a_ = X.auc(s_, lab) if 0 < lab.sum() < lab.size else np.nan
            ok_a &= (np.isnan(a_) and np.isnan(sc["auc_strict_s1"][i])) or abs(a_ - sc["auc_strict_s1"][i]) < 1e-12
            U = X.rng(S2.KEY, 1, int(i)).random((K, G * G)).reshape(K, G, G)
            th = (occ[i] & (U < p_[i]) & H[i]).sum((1, 2)).astype(float)
            fr = fz["fits"]["s1"]["groups"][str(int(rc[i]))]
            scl = fr["s"] * th
            Lr, Ur = max(np.percentile(scl, GC.QLO) - fr["q"], 0.0), max(np.percentile(scl, GC.QHI) + fr["q"], 0.0)
            ok_r &= abs(Lr - sc["hr2_L_s1"][i]) < 1e-9 and abs(Ur - sc["hr2_U_s1"][i]) < 1e-9 and abs(scl.mean() - sc["hr2_mean_s1"][i]) < 1e-9
        t("T2", ok_a and ok_r, f"recomputed outside the tool on 60 stack-repeats: seed 1 AUC (export_obj2.auc) {ok_a}; H-R2 mean, lower and "
          f"upper end from the frozen fit and thinned counts rebuilt from the maps and the keyed uniforms {ok_r}")
        # T3 refusals and rerun
        stops = []
        for f in (lambda: T.run(out_dir=tmp / "o2", out=quiet, **kw),
                  lambda: T.run(out_dir=kw["bnm_test"] / "x", out=quiet, rerun=True, **kw),
                  lambda: T.run(out_dir=tmp / "o3", out=quiet, rerun=True, **dict(kw, bnm_test=kw["bnm_fit"]))):
            try:
                f(); stops.append(False)
            except SystemExit:
                stops.append(True)
        lr = []
        rr, rrep = T.run(out_dir=tmp / "o_rerun", out=lr.append, rerun=True, **kw)
        nmk = len((kw["data"] / T.MARKER).read_text().splitlines())
        t("T3", all(stops) and all(rr.values()) and rrep.get("rerun") is True and lr[-1].startswith("RERUN stage H6") and nmk == 2,
          f"refused: a second reading without --rerun {stops[0]}, an output inside the test export {stops[1]}, the fit-half export as test "
          f"{stops[2]}; --rerun into a new folder runs, says rerun in the json and the last line, marker lines {nmk}")
        # T4 planted bugs
        for bug, gate in (("core_shift", "Z2"), ("apply_scale", "Z2"), ("thin_p", "Z3"), ("label_early", "Z5"), ("p1_open", "Z5"),
                          ("pred_changed", "Z5")):
            os.environ["STAGE_H6_BROKEN"] = bug
            try:
                rb, _ = T.run(out_dir=tmp / f"b_{bug}", out=quiet, rerun=True, **kw)
            finally:
                os.environ.pop("STAGE_H6_BROKEN", None)
            t(f"bug {bug}", rb.get(gate) is False, f"{gate} fails as it must ({rb})")
        h5c = tmp / "h5_changed"; shutil.copytree(kw["h5"], h5c)
        rw = T.S2._load("stage_g5_export_check").rewrite
        rw(h5c / f"samples_{kw['runs'][0].name}.npz", lambda a: a.__setitem__("maps", a["maps"] ^ np.uint8(1)))
        rb, _ = T.run(out_dir=tmp / "b_h5", out=quiet, rerun=True, **dict(kw, h5=h5c))
        t("bug h5 changed", rb.get("Z0") is False and len(rb) == 1, f"a test-half samples file changed after H5 fails Z0 and stops ({rb})")
        fz2 = json.loads(Path(kw["freeze"]).read_text())
        g0 = next(iter(fz2["fits"]["s1"]["groups"]))
        fz2["fits"]["s1"]["groups"][g0]["s"] *= 1.01
        p2 = tmp / "fz_edit.json"; p2.write_text(json.dumps(fz2, indent=1))
        rb, _ = T.run(out_dir=tmp / "b_fit", out=quiet, rerun=True, **dict(kw, freeze=p2, freeze_md5=X.file_md5(p2)))
        t("bug fit edited", rb.get("Z2") is False, f"a frozen fit edited by 1% (its md5 passed as signed) fails Z2 ({rb})")
        rb, _ = T.run(out_dir=tmp / "b_fmd5", out=quiet, rerun=True, **dict(kw, freeze_md5="0" * 32))
        t("bug freeze md5", rb.get("Z0") is False and len(rb) == 1, f"a freeze file other than the signed one fails Z0 ({rb})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return n_ok == len(res)