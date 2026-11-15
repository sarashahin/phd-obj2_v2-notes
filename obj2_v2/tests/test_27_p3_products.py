#!/usr/bin/env python3
"""
test_27_p3_products.py  (obj2_v2, products step P3, 13 Nov 2026): self-test of tools/stage_p3_bnm_products.py.
Run through the tool, from PSD_Dispersal_pool/obj2_v2/work:
    python3 ../tools/stage_p3_bnm_products.py --self-test
Everything is planted in a temporary folder; nothing of ../data or ../runs is read or written. A planted P1 export is made
with the real stage_h1_bnm_export.py and stage_p1_bnm_all_export.py (as test_26); planted P2 samples, G1 real_val samples,
g3_calibrated.npz AOO arrays, G5a fits (made with stage_g5_fits.py's own fit functions and save_fits), a G5b record, an
H6 report and a freeze file are written in the layouts the tool reads.
    U1  six_p by hand: x 1, 5 = option B; x 10, 20 = option B, 0 above the recorded tetrads; P(2) = 0 above 2 recorded
        tetrads, held between P(1) and P(5); P(124) = 0 above 124 recorded tetrads, at or above P(20)
    U2  cap400 by hand: mean and both interval ends at most 400
    S0  the planted P1 export passes every P1 gate
    T1  P3 on the planted inputs: gates C0-C4 PASS
    T2  independent check, fresh code from the files (no tool function): AOO of every map from the packed bits, pooled =
        first 25 of each seed, the fits read from g5_fits.npz with np.load, scale, conformal interval, cap at 400, logistic
        recalibration, option B, the zero at x 10 and 20, x 2 and 124 rules: = products_bnm.npz (1e-12) and
        products_bnm.csv (its rounding) for
        pooled, seed 1 and seed 2; H-R2 columns = the planted h6 rows; the 1-cell note at classes 0-1 only
    T3  the report's zeroed and capped counts and seed-stability numbers = a recount from the npz; the planted data
        reach both rules (values set to 0 at x 10 and 20, rows capped at 400)
    T4  a second run into a new folder gives the same md5 for products_bnm.csv and products_bnm.npz
    T5  refused: a run into an existing folder, into an input folder, a path with real_eval; C0 fails and nothing is
        written when a samples file is changed after P2
    bugs: 2019 records off by one (C1); G1 maps shifted (C2); pooled from all 100 maps (C3, nothing written); seed columns
        swapped in the csv (C3); H-R2 rows of the next class (C3); a truth file opened (C4, nothing written); the zero at
        x 10 and 20 left out (C3, nothing written); the cap left out (C3, nothing written); seed 1 and seed 2 fits swapped
        (the gates pass; T2's independent check must catch it)
"""
import hashlib, importlib.util, json, os, shutil, tempfile
from pathlib import Path
import numpy as np

T_EDGES, B_EDGES = np.array([0, 1, 2, 6, 11, 21, 51]), np.array([1, 4, 7, 11, 21])
XS4 = (1, 5, 10, 20)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def maps_with(aoo, g, W, G, pack):
    """Packed maps whose target year holds exactly aoo cells (other years sparse random), aoo (n, K)."""
    n, K = aoo.shape
    occ = np.zeros((n, K, W, G, G), bool)
    occ[:, :, :-1] = g.random((n, K, W - 1, G, G), dtype=np.float32) < 0.05
    r = g.random((n * K, G * G)).argsort(1).argsort(1)
    occ[:, :, -1] = (r < aoo.reshape(-1, 1)).reshape(n, K, G, G)
    return pack(occ)


def plant(tmp, al, P3, seed=27):
    """Planted P2 samples, runs with G1 real_val samples, g3 AOO arrays, G5a fits, G5b record, H6 report, freeze file."""
    G5F, GC, X, GS = P3.G5F, P3.GC, P3.X, P3.GS
    G, W, TCLS = X.G, X.WINDOW, GC.TCLS
    g = np.random.default_rng(seed)
    fb = np.load(al / "bnm_frozen.npz"); n = len(fb["row"]); rec_t = fb["rec_t"].astype(int)
    fmd5 = md5(al / "bnm_frozen.npz")
    js1 = json.loads((al / "h1_report.json").read_text())
    runs = [tmp / "runs" / "f_seed1", tmp / "runs" / "f_seed2"]
    # ---- real_val (G1 samples, g3 AOO arrays) and the G5a fits
    nv = 800
    comm = np.repeat(np.arange(9001, 9021), nv // 20)
    cls = g.choice(7, nv, p=[0.25, 0.2, 0.2, 0.12, 0.1, 0.08, 0.05])
    rv = np.array([g.integers(T_EDGES[c], (T_EDGES[c + 1] if c < 6 else 90)) for c in cls])
    lam = rv * g.gamma(3, 0.6, nv) + g.gamma(1.5, 1.5, nv)
    yv = np.clip(np.maximum(g.poisson(lam), rv), 0, 400).astype(float)
    rcv, rbv = P3.D.tidx(rv), P3.D.bidx(rv + g.poisson(1 + 0.5 * rv))
    g3arr, metv = {}, {}
    for i, r in enumerate(runs):
        (r / "check_g1").mkdir(parents=True)
        (r / "best.pt").write_bytes(f"planted weights {i}".encode())
        av = np.clip(g.poisson(lam[:, None] * (1.3, 0.8)[i] * g.gamma(4, 0.25, (nv, 1)), (nv, 50)), 0, 400)   # seed 2: scale > 1
        pk = maps_with(av, g, W, G, GS.pack)
        X.save_npz(r / "check_g1" / "samples_k50.npz", dict(maps=pk, sel=np.arange(nv), layout=np.array(GS.LAYOUT)))
        (r / "check_g1" / "g1_report.json").write_text(json.dumps(dict(md5=dict(samples=md5(r / "check_g1" / "samples_k50.npz"),
                                                                                best_pt=md5(r / "best.pt")))))
        metv[f"model_{r.name}"] = GC.map_metrics(pk)
        g3arr[f"aoo_model_{r.name}"] = metv[f"model_{r.name}"][0].astype(np.uint16)
    k1, k2 = (f"model_{r.name}" for r in runs)
    metv["pooled"] = tuple(np.concatenate([metv[k1][q][:, :25], metv[k2][q][:, :25]], 1) for q in (0, 1))
    metv["B3"], metv["B4"], metv["B5"] = metv[k2], metv[k1], (rv[:, None].astype(float), rv[:, None].astype(float))
    g3arr["aoo_pooled"] = metv["pooled"][0].astype(np.uint16)
    (tmp / "g3").mkdir()
    X.save_npz(tmp / "g3" / "g3_calibrated.npz", dict(g3arr, rec_t=rv, comm=comm))
    g3md5 = md5(tmp / "g3" / "g3_calibrated.npz")
    ye = yv * 1.7
    corr, recal, allm = {}, {}, np.ones(nv, bool)
    for k in (k1, k2, "pooled", "B3", "B4", "B5"):
        for kind, pred, y in (("aoo", metv[k][0].astype(float), yv), ("eoo", metv[k][1].astype(float), ye)):
            corr[(k, kind)] = G5F.fit_correction(pred, y, rcv, rbv, 2 if (k, kind) == (k2, "aoo") else 1, allm)
        ca = G5F.apply_correction(metv[k][0].astype(float), rv, rcv, rbv, corr[(k, "aoo")], "aoo")
        recal[k] = G5F.fit_recal(np.stack([GC.p_le(ca["scaled"], x) for x in XS4], 1), yv, rv, rcv, allm)
    br, ball, bn = G5F.base_rates(yv, rcv)
    fits = dict(methods=[k1, k2, "pooled", "B3", "B4", "B5"], corr=corr, recal=recal, base_rate=br, base_rate_all=ball,
                base_n=bn, communities=np.unique(comm), g3_md5=g3md5, g3b_md5="planted")
    (tmp / "fits").mkdir(); (tmp / "g5b").mkdir()
    G5F.save_fits(tmp / "fits" / "g5_fits.npz", fits)
    mf = md5(tmp / "fits" / "g5_fits.npz")
    tools = Path(P3.__file__).resolve().parent
    (tmp / "fits" / "g5_fits_report.json").write_text(json.dumps(dict(gates={f"A{i}": True for i in range(5)}, md5=dict(
        tool=md5(tools / "stage_g5_fits.py"), calibrate_tool=md5(tools / "stage_g_calibrate.py"),
        recalibrate_tool=md5(tools / "stage_g_recalibrate.py"), g3=g3md5, fits=mf))))
    (tmp / "g5b" / "g5b_export_check.json").write_text(json.dumps(dict(gates={"E0": True, "E1": True}, md5=dict(fits=mf))))
    # ---- P2 samples on the planted P1 rows (some rows with 1-2 cell maps)
    p2 = tmp / "p2"; p2.mkdir()
    sampler, files, parts, bmd = {}, {}, {}, {}
    for i, r in enumerate(runs):
        lb = np.maximum(1.5 * rec_t, 0.6)[:, None] * g.gamma(4, 0.25, (n, 1)) * (1 + 0.15 * i)
        ab = np.clip(g.poisson(lb, (n, 50)), 0, 400)
        full = np.nonzero(rec_t >= 21)[0][::12]
        ab[full] = g.integers(380, 401, (len(full), 50))                  # near-full windows: the 400 cap is reached
        sampler[r.name] = dict(K=50, seed=11, ddim=50, batch=1024, epoch=179 + 20 * i, best_md5=md5(r / "best.pt"))
        f = f"samples_{r.name}.npz"
        X.save_npz(p2 / f, dict(maps=maps_with(ab, g, W, G, GS.pack), sel=np.arange(n), K=np.int64(50), seed=np.int64(11),
                                ddim=np.int64(50), batch=np.int64(1024), in_md5=np.array("planted inputs"),
                                epoch=np.int64(179 + 20 * i), best_md5=np.array(md5(r / "best.pt")),
                                layout=np.array(GS.LAYOUT), frozen_md5=np.array(fmd5)))
        files[f] = md5(p2 / f); bmd[f"samples_{r.name}"] = md5(r / "best.pt")
        parts[f"samples_{r.name}"] = dict(gates={"K0": True, "R3a": True, "R3b": True, "R4": True}, files={f: files[f]})
    fz = dict(tools={"stage_h2_bnm_predict.py": "a" * 32}, sampler=sampler,
              data={"stage_g3_calibration/g3_calibrated.npz": g3md5,
                    **{f"{r.name}/check_g1/samples_k50.npz": md5(r / "check_g1" / "samples_k50.npz") for r in runs}})
    (tmp / "h3").mkdir(); (tmp / "h6").mkdir()
    (tmp / "h3" / "h3_freeze.json").write_text(json.dumps(fz))
    (p2 / "h2a_report.json").write_text(json.dumps(dict(gates={"K0": True, "R5": True}, parts=parts, files=files, best_md5=bmd,
                                                        frozen_md5=fmd5, h1_tool=js1["md5"]["tool"], half="all", tool="a" * 32,
                                                        written_utc="2026-11-13T00:00:00+00:00", truth_opened=0)))
    h6 = {m: dict(option=1, ok=True, bad=[], rows={"recorded class": [dict(group=lab, n=100, stacks=50 + j, ratio=1.0 + 0.07 * j + 0.01 * q,
                                                                           cov=0.85 + 0.01 * j + 0.003 * q, width=1.0)
                                                                      for j, lab in enumerate(TCLS)]}) for q, m in enumerate(("s1", "s2", "pooled"))}
    (tmp / "h6" / "h6_report.json").write_text(json.dumps(dict(gates={f"Z{i}": True for i in range(7)}, rerun=False, H_R2=h6)))
    kw = dict(bnm=al, p2=p2, fits_dir=tmp / "fits", g5b=tmp / "g5b", g3=tmp / "g3", h6=tmp / "h6" / "h6_report.json", freeze=tmp / "h3" / "h3_freeze.json",
              runs=runs, expect=dict(p1_report=md5(al / "h1_report.json"), p2_report=md5(p2 / "h2a_report.json"),
                                     freeze=md5(tmp / "h3" / "h3_freeze.json"), h6_report=md5(tmp / "h6" / "h6_report.json"), fits_prefix=mf[:8]))
    return kw, h6


def indep(out_dir, kw, h6):
    """Every product recomputed with fresh code from the files; returns the largest npz difference, csv agreement, notes."""
    al = Path(kw["bnm"])
    fb = np.load(al / "bnm_frozen.npz")
    rec, rec6 = fb["rec_t"].astype(int), fb["rec6"].astype(int)
    rcls = np.searchsorted(T_EDGES, rec, side="right") - 1
    rbin = np.clip(np.searchsorted(B_EDGES, rec6, side="right") - 1, 0, 4)
    aoo = {}
    for m, r in zip(("s1", "s2"), kw["runs"]):
        mp = np.load(Path(kw["p2"]) / f"samples_{Path(r).name}.npz")["maps"]
        aoo[m] = np.unpackbits(mp[..., 250:], axis=-1).sum(-1).astype(float)
    aoo["pooled"] = np.concatenate([aoo["s1"][:, :25], aoo["s2"][:, :25]], 1)
    fz = np.load(Path(kw["fits_dir"]) / "g5_fits.npz")
    key = {"s1": f"model_{Path(kw['runs'][0]).name}", "s2": f"model_{Path(kw['runs'][1]).name}", "pooled": "pooled"}
    pr = np.load(Path(out_dir) / "products_bnm.npz")
    import csv
    with open(Path(out_dir) / "products_bnm.csv") as fh:
        rows = list(csv.DictReader(fh))
    worst, csv_ok, mine = 0.0, len(rows) == len(rec), {}
    for m in ("pooled", "s1", "s2"):
        kk = key[m]
        opt = int(fz[f"corr__{kk}__aoo__option"])
        grp = rcls if opt == 1 else rcls * 5 + rbin
        s, q = fz[f"corr__{kk}__aoo__scale"][grp], fz[f"corr__{kk}__aoo__q"][grp]
        sc = s[:, None] * aoo[m]
        mean = sc.mean(1)
        L = np.maximum(np.percentile(sc, 5, axis=1) - q, rec); U = np.maximum(np.percentile(sc, 95, axis=1) + q, rec)
        mean, L, U = np.minimum(mean, 400), np.minimum(L, 400), np.minimum(U, 400)
        pg = np.stack([(sc < x + 0.5).mean(1) for x in XS4], 1)
        w, src = fz[f"recal__{kk}__w"], fz[f"recal__{kk}__src"]
        prc = pg.copy()
        for j, x in enumerate(XS4):
            cz = np.clip(pg[:, j], 0.01, 0.99); z = np.log(cz / (1 - cz))
            for i in range(len(rec)):
                if rec[i] > x:
                    prc[i, j] = 0.0
                elif src[j, rcls[i]] != 2:
                    prc[i, j] = 1 / (1 + np.exp(-(w[j, rcls[i], 0] + w[j, rcls[i], 1] * z[i])))
        prc = np.maximum.accumulate(prc, 1)
        pb = np.maximum.accumulate(np.column_stack([prc[:, 0], prc[:, 1], pg[:, 2], pg[:, 3]]), 1)
        p10, p20 = np.where(rec > 10, 0.0, pb[:, 2]), np.where(rec > 20, 0.0, pb[:, 3])
        p2 = np.where(rec > 2, 0.0, (sc < 2.5).mean(1)); p2 = np.clip(p2, pb[:, 0], pb[:, 1])
        p124 = np.where(rec > 124, 0.0, (sc < 124.5).mean(1)); p124 = np.maximum(p124, p20)
        P = np.column_stack([pb[:, 0], p2, pb[:, 1], p10, p20, p124])
        mine[m] = P
        for a, b in ((mean, pr[f"mean_{m}"]), (L, pr[f"L_{m}"]), (U, pr[f"U_{m}"]), (P, pr[f"P_{m}"])):
            worst = max(worst, float(np.abs(a - b).max()) if a.shape == b.shape else float("inf"))
        for j, x in enumerate((1, 2, 5, 10, 20, 124)):
            csv_ok &= bool(np.all(np.abs(np.array([float(r[f"p_aoo_le_{x}_{m}"]) for r in rows]) - P[:, j]) <= 5.01e-5))
        for c, v in (("mean", mean), ("lo90", L), ("hi90", U)):
            csv_ok &= bool(np.all(np.abs(np.array([float(r[f"aoo_{c}_{m}"]) for r in rows]) - v) <= 5.01e-4))
        hrow = {d["group"]: d for d in h6[m]["rows"]["recorded class"]}
        csv_ok &= all(abs(float(r[f"h6_ratio_{m}"]) - hrow[r["recorded_class"]]["ratio"]) <= 5.01e-4 and
                      abs(float(r[f"h6_cov_{m}"]) - hrow[r["recorded_class"]]["cov"]) <= 5.01e-4 for r in rows)
    labs = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
    csv_ok &= [r["recorded_class"] for r in rows] == [labs[c] for c in rcls]
    notes = all((r["note"] != "") == (r["recorded_class"] in ("0", "1")) for r in rows)
    return worst, csv_ok, notes, mine, rec


def main(tool_path, tools):
    tool_path = Path(tool_path)
    P3 = load(tool_path, "stage_p3_bnm_products")
    P1 = load(tool_path.parent / "stage_p1_bnm_all_export.py", "stage_p1_bnm_all_export")
    H1, X = P1.H1, P3.X
    quiet = lambda *a, **k: None
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- U1 six_p by hand (x 2 and 124 from maps of 0, 1, 2, 3 ... cells)
    pb = np.array([[0.2, 0.5, 0.7, 0.9], [0.0, 0.3, 0.6, 0.8], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1], [0.1, 0.1, 0.2, 0.3],
                   [0.0, 0.0, 0.3, 0.5], [0.0, 0.0, 0.0, 0.4]])
    sc = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [1, 1, 1, 1], [200, 200, 200, 200], [2, 2, 2, 2], [20, 20, 20, 20],
                   [200, 200, 200, 200]], float)
    rt = np.array([0, 2, 3, 130, 1, 15, 25])
    got = P3.six_p(pb, sc, rt)
    exp = np.array([[0.2, 0.5, 0.5, 0.7, 0.9, 1.0],      # raw P(2) 0.75 held at P(5) 0.5; P(124) raw 1
                    [0.0, 0.3, 0.3, 0.6, 0.8, 1.0],      # raw P(2) 0.5 held at P(5) 0.3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],      # 3 recorded: P(2) = 0; raw P(124) 1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # 130 recorded: P(20) 0.1 set to 0, P(124) 0
                    [0.1, 0.1, 0.1, 0.2, 0.3, 1.0],      # raw P(2) 1 held at P(5) 0.1
                    [0.0, 0.0, 0.0, 0.0, 0.5, 1.0],      # 15 recorded: P(10) 0.3 set to 0, P(20) kept
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])     # 25 recorded: P(20) 0.4 set to 0; maps of 200 give P(124) 0
    t("U1", np.allclose(got, exp), "six_p by hand: x 1, 5 = option B; x 10, 20 = option B, 0 above the recorded tetrads; P(2) "
      "0 above 2 recorded tetrads and held between P(1) and P(5); P(124) 0 above 124 recorded tetrads, at or above P(20)")
    cm = P3.cap400(dict(mean=np.array([100.0, 450.0]), L=np.array([50.0, 420.0]), U=np.array([200.0, 500.0])))
    t("U2", all(np.array_equal(a, b) for a, b in zip(cm, ([100, 400], [50, 400], [200, 400]))),
      "cap400 by hand: (100, 50, 200) unchanged, (450, 420, 500) become 400")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = H1.world(tmp / "bw")
        data = tmp / "data"
        fit, test, al = data / "bnm_fit", data / "bnm_test", data / "bnm_all"
        rf, _ = H1.run(py, refpat, tools, fit, expect=(6, 6), out=quiet)
        fzh = tmp / "fz_h1.json"
        fzh.write_text(json.dumps(dict(tools={"stage_h1_bnm_export.py": X.file_md5(P1.H1_FILE)})))
        rt_, _ = H1.run(py, refpat, tools, test, expect=(6, 6), half="test", final_test=True, freeze=str(fzh), out=quiet)
        rp, rep1 = P1.run(py, refpat, tools, fit, test, al, expect=(6, 6), out=quiet)
        t("S0", all(rf.values()) and all(rt_.values()) and all(rp.values()), f"planted P1 export: {rp}, {rep1.get('stacks')} stacks")
        kw, h6 = plant(tmp, al, P3)
        lines = []
        out1 = data / "p3"
        r, rep = P3.run(out_dir=out1, out=lines.append, **kw)
        t("T1", set(r) == {"C0", "C1", "C2", "C3", "C4"} and all(r.values()), f"P3 on the planted inputs: {r}")
        if not all(r.values()):
            print("      " + "\n      ".join(lines))
        worst, csv_ok, notes, mine, rec = indep(out1, kw, h6) if out1.is_file() or out1.exists() else (9, False, False, {}, None)
        t("T2", worst <= 1e-12 and csv_ok and notes, f"independent recomputation from the files (pooled, seed 1, seed 2): npz "
          f"largest difference {worst:.1e}; csv = it within rounding, H-R2 columns = the planted rows {csv_ok}; 1-cell note at "
          f"classes 0-1 only {notes}")
        zb = np.load(out1 / "products_bnm.npz") if out1.exists() else None
        z0 = {m: {str(x): int(((zb[f"p_b_{m}"][:, j] > 0) & (rec > x)).sum()) for j, x in ((2, 10), (3, 20))} for m in ("pooled", "s1", "s2")} if zb is not None else {}
        c0 = {m: int(zb[f"capped_{m}"].sum()) for m in ("pooled", "s1", "s2")} if zb is not None else {}
        ok3 = bool(rep) and rep.get("zeroed") == z0 and rep.get("capped") == c0 \
            and abs(rep["seed_stability"]["2"]["mean_abs"] - float(np.abs(mine["s1"][:, 1] - mine["s2"][:, 1]).mean())) <= 1e-12 \
            and min(v for d in z0.values() for v in d.values()) > 0 and c0.get("s2", 0) > 0
        t("T3", ok3, f"zeroed {rep.get('zeroed')} and capped {rep.get('capped')} in p3_report.json = a recount; seed stability = a "
          f"recount; the planted data reach the zero and the cap")
        out2 = data / "p3_again"
        P3.run(out_dir=out2, out=quiet, **kw)
        same = all(X.file_md5(out1 / f) == X.file_md5(out2 / f) for f in ("products_bnm.csv", "products_bnm.npz"))
        t("T4", same, "a second run into a new folder gives the same md5 for products_bnm.csv and products_bnm.npz")
        stops = []
        for o in (out1, Path(kw["p2"]) / "inside", data / "real_eval_p3"):
            try:
                P3.run(out_dir=o, out=quiet, **kw); stops.append(False)
            except SystemExit:
                stops.append(True)
        p2b = tmp / "p2_tampered"
        shutil.copytree(kw["p2"], p2b)
        z = dict(np.load(p2b / "samples_f_seed2.npz")); z["maps"] = z["maps"].copy(); z["maps"][0, 0, -1] ^= 1
        X.save_npz(p2b / "samples_f_seed2.npz", z)
        rq, _ = P3.run(out_dir=data / "p3_tampered", out=quiet, **dict(kw, p2=p2b))
        t("T5", all(stops) and rq.get("C0") is False and not (data / "p3_tampered").exists(),
          f"refused: an existing folder {stops[0]}, a folder inside an input folder {stops[1]}, a real_eval path {stops[2]}; a "
          f"samples file changed after P2 fails C0 and nothing is written {rq}")
        for bug, g_, written in (("rec_shift", "C1", False), ("g1_shift", "C2", False), ("pool100", "C3", False),
                                 ("csv_swap", "C3", True), ("h6_class", "C3", True), ("opentruth", "C4", False),
                                 ("no_zero", "C3", False), ("no_cap", "C3", False)):
            os.environ["STAGE_P3_BROKEN"] = bug
            try:
                rb, _ = P3.run(out_dir=data / f"bug_{bug}", out=quiet, **kw)
            finally:
                os.environ.pop("STAGE_P3_BROKEN", None)
            ex = (data / f"bug_{bug}").exists()
            t(f"bug {bug}", rb.get(g_) is False and ex == written, f"{g_} fails as it must; written {ex} (expected {written}) ({rb})")
        os.environ["STAGE_P3_BROKEN"] = "fits_swap"
        try:
            rb, _ = P3.run(out_dir=data / "bug_fits_swap", out=quiet, **kw)
        finally:
            os.environ.pop("STAGE_P3_BROKEN", None)
        wb, cb, _, _, _ = indep(data / "bug_fits_swap", kw, h6)
        t("bug fits_swap", all(rb.values()) and (wb > 1e-6 or not cb), f"seed fits swapped: the gates pass ({rb}), the independent "
          f"check finds it (largest difference {wb:.2e}, csv agreement {cb})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)