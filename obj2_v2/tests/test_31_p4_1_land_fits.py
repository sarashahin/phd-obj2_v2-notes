#!/usr/bin/env python3
"""
test_31_p4_1_land_fits.py  (obj2_v2, P4 step P4-1, 14 Nov 2026): self-test of tools/stage_p4_1_land_fits.py.
Run through the tool, from PSD_Dispersal_pool/obj2_v2/work:
    python3 ../tools/stage_p4_1_land_fits.py --self-test
It builds test_30's planted world with 6 communities per split (grids, exports, frozen sets, G1 / G2 / G5c files, Land
Cover Map), runs the real P4-0 on it, then writes planted G3, G3b and G5a outputs with those tools' own functions and
G3's option loop (g3_calibrated.npz, g3b_recal.npz, g5_fits.npz and their reports), and runs P4-1. Nothing of ../data or
../runs is read or written.
    U1  G3's option rule: (True, any) -> 1; (False, True) -> 2; (False, False) -> 1 (and the method fails)
    U2  version 2: G3's crossfit and G5a's apply give crossed intervals (upper end below lower end) when Q < 0 and a stack's
        spread is under 2|Q| (8% of synthetic stacks with no spread, recorded floor 3 on some); apply_land counts them; with
        --crossed collapse both ends of each crossed interval = max(middle of the raw ends, recorded cells), every upper end
        is at least its lower end, and no other interval changes; with stop or empty the intervals are G5a's, unchanged
    S0  the planted P4-0 run passes every gate
    T1  P4-1 on it with --crossed stop (default) and collapse: gates Q0-Q6 PASS; the rule is stored in p4_fits.npz; the
        count of crossed intervals in the land-only cross-fit = a recount from p4_1_crossfit.npz (0 with the collapse)
    T2  independent check, fresh code (bits unpacked, plain sums): land-only AOO of seed 1's maps and of the truth; the
        option-1 scale of seed 1 per recorded class in fold 0 (sum of truth / sum of mean maps over the other fold);
        coverage and ratio per recorded class from the stored intervals = the report rows; each verdict = the rule applied
        by hand to the report rows
    T3  meaning: in all-land windows land-only truth = window truth; in coastal windows land-only <= window and smaller
        somewhere
    T4  a second run into a new folder gives the same p4_fits.npz md5
    T5  refused: an existing folder, a folder inside an input folder, a real_eval path; Q0 fails when g5_fits.npz changes
        after its report
    T6  crossed intervals planted in G5a's apply (every 37th stack's upper end put below its lower end): stop -> Q5 fails,
        nothing written, every Q5 part 0 but the crossed one; empty -> all gates PASS, the crossed count printed and stored,
        the rule stored as empty; collapse -> all gates PASS, no crossed interval left in-sample
    bugs (each must fail its gate, nothing written): map mask left out on one path (Q3); truth mask left out (Q3); windows
        shifted by one (Q3); fits on one fold only (Q5); an input off its record (Q0); a real_eval file opened (Q6); G3's
        crossfit fitted on all stacks (leak; Q2)
"""
import csv, hashlib, importlib.util, json, os, tempfile, warnings
from pathlib import Path
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def build(T30, tool4, tmp, obj1_tools, n_comm=6):
    """test_30's planted world with n_comm communities per split, then the real P4-0."""
    X, GS = tool4.X, tool4.GS
    g = np.random.default_rng(31)
    py, refpat = X.plant_grids(str(tmp))
    wins, _, _ = X.load_bnm(py, refpat, obj1_tools, out=lambda *a, **k: None)
    data, runs = tmp / "data", [tmp / "runs" / "f_seed1", tmp / "runs" / "f_seed2"]
    g2, g5c, lcm = data / "stage_g2_baselines_v2", data / "stage_g5_predict", tmp / "lcm"
    code = X.file_md5(tool4.HERE / "export_obj2.py")
    rv, tv = T30.plant_split(X, data / "real_val", "real_val", tuple(8101 + i for i in range(n_comm)), wins, code, g)
    re_, te = T30.plant_split(X, data / "real_eval", "real_eval", tuple(8201 + i for i in range(n_comm)), wins, code, g)
    fzv, fze = T30.plant_frozen(X, data / "real_val", rv, g), T30.plant_frozen(X, data / "real_eval", re_, g)
    X.save_npz(data / "stage_d_frozen_val.npz", fzv)
    g2.mkdir(parents=True); g5c.mkdir(parents=True)
    X.save_npz(g5c / "frozen_eval.npz", fze)
    mv, me_ = md5(data / "stage_d_frozen_val.npz"), md5(g5c / "frozen_eval.npz")
    parts, g2rep = {"frozen": {"frozen_eval.npz": me_}, "baselines": {}}, {}
    for b, f in (("B3", "maps_B3_halfblank.npz"), ("B4", "maps_B4.npz"), ("B5", "maps_B5.npz")):
        for dd, fz, sp, fm in ((g2, fzv, "real_val", mv), (g5c, fze, "real_eval", me_)):
            mp = T30.plant_maps(X, data / sp, fz, g, b)
            X.save_npz(dd / f, dict(maps=mp, sel=np.arange(len(fz["row"])), K=np.int64(mp.shape[1]), layout=np.array(GS.LAYOUT),
                                    frozen_md5=np.array(fm), aoo_t=np.zeros((len(mp), mp.shape[1]), np.uint16)))
        g2rep[f] = md5(g2 / f); parts["baselines"][f] = md5(g5c / f)
    (g2 / "g2_report.json").write_text(json.dumps(dict(md5=g2rep)))
    for r in runs:
        (r / "check_g1").mkdir(parents=True)
        mp = T30.plant_maps(X, data / "real_val", fzv, g, "model")
        X.save_npz(r / "check_g1" / "samples_k50.npz", dict(maps=mp, sel=np.arange(len(mp)), K=np.int64(50), layout=np.array(GS.LAYOUT)))
        (r / "check_g1" / "g1_report.json").write_text(json.dumps(dict(md5=dict(samples=md5(r / "check_g1" / "samples_k50.npz"), best_pt="b"))))
        mp = T30.plant_maps(X, data / "real_eval", fze, g, "model")
        X.save_npz(g5c / f"samples_{r.name}.npz", dict(maps=mp, sel=np.arange(len(mp)), K=np.int64(50), layout=np.array(GS.LAYOUT),
                                                          frozen_md5=np.array(me_)))
        parts[f"samples_{r.name}"] = {f"samples_{r.name}.npz": md5(g5c / f"samples_{r.name}.npz")}
    for p, fs in parts.items():
        (g5c / f"part_{p}.json").write_text(json.dumps(dict(gates={"R0": True, "R4": True}, files=fs)))
    T30.plant_map(wins, lcm)
    os.environ["STAGE_P4_0_BROKEN"] = ""
    r0, rep0 = tool4.run(data, runs, g2, g5c, lcm, py, refpat, obj1_tools, data / "stage_p4_0",
                         expect=dict(lcm=md5(lcm / "gb2019lcm1km_percentage_target.tif"), band_check=X.file_md5(tool4.HERE / "stage_p3c_band_check.py")),
                         out=lambda *a, **k: None)
    return dict(data=data, runs=runs, g2=g2, g5c=g5c, fzv=fzv, rv=rv, tv=tv, r0=r0)


def plant_g(tool, w, tmp):
    """G3 outputs by G3's own functions and option loop; G3b's recalibrate; G5a's fit functions and save_fits."""
    GC, GR, G5F, X, D = tool.GC, tool.GR, tool.G5F, tool.X, tool.D
    data, runs, fz = w["data"], w["runs"], w["fzv"]
    n = len(fz["row"]); fi, rw = fz["file_index"].astype(int), fz["row"].astype(int)
    man = {r["input_file"]: r for r in csv.DictReader(open(data / "real_val" / "manifest.csv"))}
    T = np.zeros((n, GC.G, GC.G), bool); V = np.zeros((n, GC.W, GC.G, GC.G), bool); comm = np.zeros(n, int)
    for f in sorted(set(fi.tolist())):
        k = np.nonzero(fi == f)[0]; nm = str(fz["files"][f])
        T[k] = X.unpack(np.load(data / "real_val" / man[nm]["truth_file"])["true_maps"][rw[k]])[:, -1]
        zi = np.load(data / "real_val" / nm)
        V[k] = X.unpack(zi["visit1"][rw[k]] | zi["visit2"][rw[k]]); comm[k] = int(man[nm]["sim_seed"])
    true_eoo, true_aoo = GC.eoo_aoo(T); true_aoo = true_aoo.astype(float)
    rec_t = V[:, -1].reshape(n, -1).sum(1)
    rcls, rbin = D.tidx(rec_t), D.bidx(V.any(1).reshape(n, -1).sum(1))
    fold = GC.make_folds(comm)
    maps = {f"model {r.name}": np.load(r / "check_g1" / "samples_k50.npz")["maps"] for r in runs}
    for lab, f in (("B3", "maps_B3_halfblank.npz"), ("B4", "maps_B4.npz"), ("B5", "maps_B5.npz")):
        maps[lab] = np.load(w["g2"] / f)["maps"]
    s1, s2 = [f"model {r.name}" for r in runs]
    order = [s1, s2, "pooled", "B3", "B4", "B5"]
    met = {k: GC.map_metrics(maps[k]) for k in order if k != "pooled"}
    met["pooled"] = tuple(np.concatenate([met[s1][q][:, :25], met[s2][q][:, :25]], 1) for q in (0, 1))
    groups = {"recorded class": (rcls, GC.TCLS), "records over 6 years": (rbin, GC.BINS), "true class": (D.tidx(true_aoo.astype(int)), GC.TRUE)}
    opt = {1: (rcls, np.full(n, -1)), 2: (rcls * len(GC.BINS) + rbin, rcls)}
    save = dict(fold=fold, comm=comm, rcls=rcls, rbin=rbin, rec_t=rec_t, true_class=D.tidx(true_aoo.astype(int)), true_aoo=true_aoo, true_eoo=true_eoo)
    chosen, pl = {}, {}
    for k in order:
        kk = k.replace(" ", "_")
        save[f"aoo_{kk}"] = met[k][0].astype(np.uint16); save[f"eoo_{kk}"] = met[k][1].astype(np.float32)
        for kind, pred, y in (("aoo", met[k][0].astype(float), true_aoo), ("eoo", met[k][1], true_eoo)):
            per = {}
            for o_ in (1, 2):
                cf = GC.crossfit(pred, y, rec_t, fold, *opt[o_], kind)
                ok, bad = GC.rule_pass(GC.evaluate(y, cf, groups))
                per[o_] = dict(cf=cf, ok=ok, bad=bad)
            use = 1 if per[1]["ok"] else (2 if per[2]["ok"] else 1)
            chosen[f"{k} {kind}"] = dict(option=use, ok=per[use]["ok"], bad=per[use]["bad"], opt1_ok=per[1]["ok"], opt2_ok=per[2]["ok"])
            cf = per[use]["cf"]
            for q in ("mean", "L", "U"):
                save[f"{kind}_{q}_{kk}"] = cf[q]
            save[f"{kind}_scale_q_{kk}"] = cf["par"]
            if kind == "aoo":
                save[f"p_le_{kk}"] = np.stack([GC.p_le(cf["scaled"], x) for x in GC.XS], 1)
                pc = GC.p_check(cf["scaled"], true_aoo, fold, rcls)
                pl[k] = {str(x): {q: pc[x][q] for q in ("brier", "bss", "gap")} for x in GC.XS}
    g3 = data / "stage_g3_calibration"; g3.mkdir()
    X.save_npz(g3 / "g3_calibrated.npz", save)
    m3 = md5(g3 / "g3_calibrated.npz")
    (g3 / "g3_report.json").write_text(json.dumps(dict(gates={"K1": True}, md5=dict(calibrated=m3), chosen=chosen, p_le=pl)))
    keys = [k.replace(" ", "_") for k in order]
    g3b = data / "stage_g3b_recalibration"; g3b.mkdir()
    sb = dict(fold=fold, comm=comm, rcls=rcls, rec_t=rec_t, true_aoo=true_aoo, xs=np.array(GC.XS))
    for kk in keys:
        sb[f"p_le_{kk}"] = save[f"p_le_{kk}"]; sb[f"p_recal_{kk}"] = GR.recalibrate(save[f"p_le_{kk}"], true_aoo, rec_t, rcls, fold)[0]
    X.save_npz(g3b / "g3b_recal.npz", sb)
    m3b = md5(g3b / "g3b_recal.npz")
    (g3b / "g3b_report.json").write_text(json.dumps(dict(gates={"L1": True}, md5=dict(input=m3, recal=m3b, tool="t"))))
    corr, recal = {}, {}
    for kk, k in zip(keys, order):
        for kind in ("aoo", "eoo"):
            corr[(kk, kind)] = G5F.fit_correction(save[f"{kind}_{kk}"].astype(float), true_aoo if kind == "aoo" else true_eoo.astype(float),
                                                  rcls, rbin, chosen[f"{k} {kind}"]["option"], np.ones(n, bool))
        recal[kk] = G5F.fit_recal(sb[f"p_le_{kk}"].astype(float), true_aoo, rec_t, rcls, np.ones(n, bool))
    br, ball, bn = G5F.base_rates(true_aoo, rcls)
    g5 = data / "stage_g5_fits"; g5.mkdir()
    G5F.save_fits(g5 / "g5_fits.npz", dict(methods=keys, corr=corr, recal=recal, base_rate=br, base_rate_all=ball, base_n=bn,
                                           communities=np.unique(comm), g3_md5=m3, g3b_md5=m3b))
    H = tool.HERE
    (g5 / "g5_fits_report.json").write_text(json.dumps(dict(gates={"A0": True, "A1": True}, md5=dict(
        tool=X.file_md5(H / "stage_g5_fits.py"), calibrate_tool=X.file_md5(H / "stage_g_calibrate.py"),
        recalibrate_tool=X.file_md5(H / "stage_g_recalibrate.py"), g3=m3, g3b=m3b, fits=md5(g5 / "g5_fits.npz")))))
    return dict(g3=g3, g3b=g3b, g5=g5, T=T, V=V, n=n, true_aoo=true_aoo)


def main(tool_path, obj1_tools="../../obj1_v2/tools"):
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    here = Path(tool_path).resolve().parent
    tool = load(tool_path, "stage_p4_1_land_fits")
    tool4 = load(here / "stage_p4_0_windows.py", "stage_p4_0_windows")
    T30 = load(here.parent / "tests" / "test_30_p4_0_windows.py", "test_30_p4_0_windows")
    X, GC = tool.X, tool.GC
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    t("U1", tool.choose_option(True, False) == 1 and tool.choose_option(True, True) == 1 and tool.choose_option(False, True) == 2
      and tool.choose_option(False, False) == 1, "G3's option rule: option 1 if it passes, else 2 if it passes, else 1 (method fails)")
    G5F, XS = tool.G5F, tool.XS
    g = np.random.default_rng(7); nn, K = 2000, 50
    cm = g.integers(0, 20, nn); fo = GC.make_folds(cm)
    yy = g.poisson(20, nn).astype(float); narrow = g.random(nn) < 0.08
    pr = yy[:, None] + g.uniform(-10, 10, (nn, K)); pr[narrow] = yy[narrow, None]
    z0, gr = np.zeros(nn, int), np.zeros(nn, int)
    rt = np.where(np.arange(nn) % 5 == 0, np.minimum(3, yy).astype(int), 0)          # a recorded floor of up to 3 on a fifth of stacks
    cfu = GC.crossfit(pr, yy, rt, fo, gr, np.full(nn, -1), "aoo")
    tab = G5F.fit_correction(pr, yy, gr, z0, 1, np.ones(nn, bool))
    fu = dict(corr={"m": tab}, recal={"m": G5F.fit_recal(np.stack([GC.p_le(pr, x) for x in XS], 1), yy, rt, gr, np.ones(nn, bool))},
              crossed="stop")
    a0, a1 = tool.apply_land(fu, "m", pr, rt, gr, z0), tool.apply_land(fu, "m", pr, rt, gr, z0, crossed="collapse")
    ae = tool.apply_land(fu, "m", pr, rt, gr, z0, crossed="empty")
    same_e = all(np.array_equal(ae["aoo"][q], a0["aoo"][q]) for q in ("L", "U", "scaled")) and np.array_equal(ae["p_b"], a0["p_b"])
    iv = a0["aoo"]["U"] < a0["aoo"]["L"]
    pt = np.maximum((a0["aoo"]["Lraw"] + a0["aoo"]["Uraw"]) / 2, rt)
    cf2 = {q: cfu[q].copy() for q in ("L", "U", "Lraw", "Uraw")}; n2 = tool.collapse(cf2, rt)
    t("U2", tab["q"][0] < 0 and (cfu["U"] < cfu["L"]).sum() > 0 and a0["inverted"] == int(iv.sum()) > 0 and a1["inverted"] == a0["inverted"]
      and bool((a1["aoo"]["U"] >= a1["aoo"]["L"]).all()) and np.array_equal(a1["aoo"]["U"][iv], pt[iv]) and np.array_equal(a1["aoo"]["L"][iv], pt[iv])
      and bool((a1["aoo"]["L"] >= rt).all()) and np.array_equal(a1["aoo"]["U"][~iv], a0["aoo"]["U"][~iv])
      and np.array_equal(a1["aoo"]["L"][~iv], a0["aoo"]["L"][~iv]) and n2 == int((cfu["U"] < cfu["L"]).sum()) and bool((cf2["U"] >= cf2["L"]).all())
      and same_e and np.array_equal(a1["p_b"], a0["p_b"]),
      f"Q {tab['q'][0]:.2f}: crossed in {int((cfu['U'] < cfu['L']).sum())} cross-fitted and {int(iv.sum())} applied stacks of "
      f"{int(narrow.sum())} with no spread; both ends set to the meeting point, not below the recorded cells; other intervals unchanged; empty = G5a's intervals; P unchanged")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        w = build(T30, tool4, tmp, obj1_tools)
        t("S0", w["r0"] and all(w["r0"].values()), f"planted world with {len(w['fzv']['row'])} real_val frozen stacks; P4-0 on it: {w['r0']}")
        gp = plant_g(tool, w, tmp)
        p40 = w["data"] / "stage_p4_0"
        expect = dict(p4_0_report=md5(p40 / "p4_0_report.json"), p4_windows=md5(p40 / "p4_windows.npz"), g3_prefix="", fits_prefix="")

        def go(out, env="", **kw):
            os.environ["STAGE_P4_1_BROKEN"] = env
            lines = []
            try:
                a = dict(data=w["data"], runs=w["runs"], g2=w["g2"], g3=gp["g3"], g3b=gp["g3b"], g5=gp["g5"], p4_0=p40, out_dir=out,
                         expect=expect, out=lines.append)
                a.update(kw)
                r, rep = tool.run(**a)
            finally:
                os.environ["STAGE_P4_1_BROKEN"] = ""
            return r, rep, lines

        r1, rep1, l1 = go(tmp / "out1")
        r1u, rep1u, l1u = go(tmp / "out1u", crossed="collapse")
        okr = r1 and all(r1.values()) and set(r1) == {f"Q{i}" for i in range(7)} and r1u and all(r1u.values())
        if not okr:
            t("T1", False, f"P4-1 on the planted world: {r1}; with the collapse {r1u}")
            print("\n".join("      " + s for s in l1 + l1u)); print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL"); return False
        cz = dict(np.load(tmp / "out1" / "p4_1_crossfit.npz")); czu = dict(np.load(tmp / "out1u" / "p4_1_crossfit.npz"))
        flg = tool.load_fits(tmp / "out1" / "p4_fits.npz")["crossed"] == "stop" and tool.load_fits(tmp / "out1u" / "p4_fits.npz")["crossed"] == "collapse"
        cnt = all(rep1["inverted"]["land_crossfit"][tool.rname(kk)][rep1["chosen"][tool.rname(kk)]["option"]] == int((cz[f"U_{kk}"] < cz[f"L_{kk}"]).sum())
                  for kk in ("model_f_seed1", "model_f_seed2", "pooled", "B3", "B4", "B5")) \
            and all(int((czu[f"U_{kk}"] < czu[f"L_{kk}"]).sum()) == 0 for kk in ("model_f_seed1", "pooled", "B4"))
        t("T1", okr and flg and cnt, f"P4-1 on the planted world: {r1}; with --crossed collapse all gates PASS too; rule stored {flg}; "
          f"crossed intervals = recount from p4_1_crossfit.npz, 0 with the collapse {cnt}")
        pz = dict(np.load(p40 / "p4_windows.npz"))
        # T2 independent check
        n, win = gp["n"], pz["val_window"].astype(int)
        L = pz["land"].astype(bool)[win]
        m1 = np.load(w["runs"][0] / "check_g1" / "samples_k50.npz")["maps"]
        occ = np.unpackbits(m1[:, :, 250:], axis=-1, count=400).reshape(n, m1.shape[1], 20, 20).astype(bool)
        a_ok = np.array_equal((occ & L[:, None]).sum((2, 3)), cz["aoo_land_model_f_seed1"].astype(int))
        y_ok = np.array_equal((gp["T"] & L).sum((1, 2)).astype(float), cz["true_land"])
        rcls, fold, yl = cz["rcls"], cz["fold"], cz["true_land"]
        pred = cz["aoo_land_model_f_seed1"].astype(float)
        ch = rep1["chosen"]["model f_seed1"]
        s_ok = True
        if ch["option"] == 1:
            fit = fold == 1
            for g_ in np.unique(rcls[fold == 0]):
                idx = fit & (rcls == g_)
                if idx.sum() < GC.N_FIT_MIN:
                    continue
                s_hand = yl[idx].sum() / pred[idx].mean(1).sum()
                s_ok &= bool(np.allclose(cz["scale_q_model_f_seed1"][(fold == 0) & (rcls == g_), 0], s_hand, rtol=0, atol=1e-12))
        r_ok, v_ok = True, True
        for k_, kk in (("model f_seed1", "model_f_seed1"), ("B4", "B4"), ("pooled", "pooled")):
            rows = rep1["rows"][k_]["recorded class"]
            cov = (yl >= cz[f"L_{kk}"]) & (yl <= cz[f"U_{kk}"])
            for r_ in rows:
                g_ = list(GC.TCLS).index(r_["group"]); m_ = rcls == g_
                r_ok &= abs(cov[m_].mean() - r_["cov"]) < 1e-12 and abs(cz[f"mean_{kk}"][m_].mean() / yl[m_].mean() - r_["ratio"]) < 1e-9
            hand = all(0.8 <= r_["ratio"] <= 1.25 and 0.85 <= r_["cov"] <= 0.95 for g_ in ("recorded class", "records over 6 years")
                       for r_ in rep1["rows"][k_][g_] if r_["n"] >= 50)
            v_ok &= hand == rep1["chosen"][k_]["ok"]
        t("T2", a_ok and y_ok and s_ok and r_ok and v_ok, f"fresh code: land-only AOO of seed 1's maps {a_ok} and of the truth {y_ok}; "
          f"seed 1 option-{ch['option']} scale per recorded class in fold 0 by hand {s_ok}; coverage and ratio per recorded class from "
          f"the stored intervals = report rows {r_ok}; verdicts = the rule applied by hand {v_ok}")
        # T3 meaning
        coastal = cz["coastal"].astype(bool)
        t("T3", np.array_equal(yl[~coastal], cz["true_window"][~coastal]) and bool((yl[coastal] <= cz["true_window"][coastal]).all())
          and bool((yl[coastal] < cz["true_window"][coastal]).any()),
          f"all-land windows: land-only truth = window truth; coastal ({int(coastal.sum())} stacks): land-only at most window and smaller "
          f"in {int((yl[coastal] < cz['true_window'][coastal]).sum())}")
        # T4 rerun
        go(tmp / "out2")
        t("T4", md5(tmp / "out1" / "p4_fits.npz") == md5(tmp / "out2" / "p4_fits.npz"), "a second run gives the same p4_fits.npz md5")
        # T5 refusals and a changed G5a file
        stops = []
        for out in (tmp / "out1", w["data"] / "real_val" / "o", tmp / "real_eval" / "o"):
            try:
                go(out); stops.append(False)
            except SystemExit:
                stops.append(True)
        f5 = gp["g5"] / "g5_fits.npz"; keep = f5.read_bytes()
        zz = dict(np.load(f5)); kq = next(k for k in zz if k.endswith("__aoo__scale")); zz[kq] = zz[kq] * 1.01; X.save_npz(f5, zz)
        r5, _, _ = go(tmp / "out5")
        f5.write_bytes(keep)
        t("T5", all(stops) and r5.get("Q0") is False and not (tmp / "out5").exists(),
          f"refused: existing folder {stops[0]}, inside an input folder {stops[1]}, a real_eval path {stops[2]}; g5_fits.npz changed "
          f"after its report: Q0 FAIL, nothing written")
        # T6 crossed intervals planted in G5a's apply, the three rules
        orig = tool.G5F.apply_correction

        def crossed_apply(pred, rec_t, rcls, rbin, tab, kind):
            r_ = orig(pred, rec_t, rcls, rbin, tab, kind)
            k_ = np.arange(len(pred)) % 37 == 0
            r_["Uraw"] = np.where(k_, r_["Lraw"] - 2.0, r_["Uraw"]); r_["U"] = np.where(k_, np.maximum(r_["Uraw"], np.maximum(rec_t, 0)), r_["U"])
            return r_
        tool.G5F.apply_correction = crossed_apply
        try:
            r6 = {c: go(tmp / f"out6_{c}", crossed=c) for c in ("stop", "empty", "collapse")}
        finally:
            tool.G5F.apply_correction = orig
        q5 = {c: next((s_ for s_ in v[2] if s_.startswith("Q5 ")), "") for c, v in r6.items()}

        def part(line, q):
            seg = line.split(f"; {q} " if q != "crossed" else "rule: crossed ", 1)[1].split(";", 1)[0]
            return [int(x.rsplit(" ", 1)[1]) for x in seg.split(", ")]
        others = ("lower below recorded", "P outside [0, 1]", "P falls with x", "P above 0 where recorded > x", "option B")
        try:
            st = r6["stop"][0].get("Q5") is False and not (tmp / "out6_stop").exists() and min(part(q5["stop"], "crossed")) > 0 \
                and all(max(part(q5["stop"], q)) == 0 for q in others)
            ins_e = r6["empty"][1].get("inverted", {}).get("p4_insample", {})
            em = all(r6["empty"][0].values()) and min(part(q5["empty"], "crossed")) > 0 and all(max(part(q5["empty"], q)) == 0 for q in others) \
                and part(q5["empty"], "crossed") == list(ins_e.values()) and tool.load_fits(tmp / "out6_empty" / "p4_fits.npz")["crossed"] == "empty"
            co = all(r6["collapse"][0].values()) and max(part(q5["collapse"], "crossed")) == 0 \
                and min(r6["collapse"][1]["inverted"]["p4_insample"].values()) > 0
        except (IndexError, ValueError, AttributeError) as e_:
            st = em = co = False; print("      T6 parse:", e_, q5)
        t("T6", st and em and co, f"crossed intervals planted in G5a's apply: stop -> Q5 FAIL, nothing written, only the crossed part above 0 "
          f"{st}; empty -> gates PASS, crossed {part(q5['empty'], 'crossed') if em else '?'} counted and stored, rule stored {em}; "
          f"collapse -> gates PASS, none left {co}")
        # planted bugs
        for i, (bug, g_) in enumerate((("no_mask", "Q3"), ("truth_nomask", "Q3"), ("mask_shift", "Q3"), ("fits_subset", "Q5"),
                                       ("changed_input", "Q0"), ("eval_open", "Q6"))):
            rb, _, _ = go(tmp / f"bugout{i}", env=bug)
            t(f"bug {bug}", rb.get(g_) is False and not (tmp / f"bugout{i}").exists(), f"{g_} fails and nothing is written")
        old = tool.GC.BROKEN; tool.GC.BROKEN = "leak"
        try:
            rb, _, _ = go(tmp / "bugout_leak")
        finally:
            tool.GC.BROKEN = old
        t("bug leak", rb.get("Q2") is False and not (tmp / "bugout_leak").exists(), "Q2 fails and nothing is written")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)


if __name__ == "__main__":
    import sys
    tp = Path(__file__).resolve().parent.parent / "tools" / "stage_p4_1_land_fits.py"
    sys.exit(0 if main(tp, sys.argv[1] if len(sys.argv) > 1 else "../../obj1_v2/tools") else 1)