#!/usr/bin/env python3
"""
test_33_p4_3_land_products.py  (obj2_v2, P4 step P4-3, 14 Nov 2026): self-test of tools/stage_p4_3_land_products.py.
Run through the tool, from PSD_Dispersal_pool/obj2_v2/work:
    python3 ../tools/stage_p4_3_land_products.py --self-test
It builds test_27's planted world with three planted changes, runs the real H1, P1, P3, P3b and P3c tools on it, plants
the P4-0 and P4-1 outputs, then runs P4-3. Nothing of ../data or ../runs is read or written.
Planted changes: SMALL (test half) keeps effort and records in tetrad row 0 only and every other tetrad is open sea, so
|L_w| <= 20; in half its P2 maps row 0 is fully occupied; every 2019 record is put into every P2 map (both after P2's
planting, its md5 record redone); EXTRA (fit half): a
species row with no record gets effort in 2016 at 6 east tetrads that no other row surveyed and the map calls sea, so L_w
puts them back; ALLLAND (fit half) is all land. Planted P4 fits = the planted G5a fits with the scale times 0.9 (seed 1),
1.1 (seed 2), 1.25 (pooled), and Q 6 below min(Q, 0) in recorded classes 1 and 2-5 for seed 2 and pooled (crossed rows).
p4_windows.npz is planted with fresh code (rasterio blocks and the per-year grids).
    U1  by hand: products() on three rows (an interval, a crossed one, one capped at |L_w| = 18 with P(20) 0.5 under keep
        and 1 under one)
    S0  the planted H1, P1, P3, P3b and P3c runs pass every gate
    T1  P4-3 with --above-land one: gates D0-D5 PASS
    T2  fresh code from the files (L_w from rasterio blocks and the per-year grids, sample bits, p4_fits.npz scale and Q with
        np.load): raw, mean, interval, crossed flags, P at x 2, 10, 20, 124 and the one rule = products_land_bnm.npz (1e-9);
        csv blanks and land tetrads
    T3  meaning: all-land rows: land-only AOO = P3's window AOO; coastal rows below or equal, some below; crossed rows in seed
        2 and pooled; capped rows; SMALL rows |L_w| <= 20 with P(20) = P(124) = 1, some set to 1; put-back tetrads = the
        6 planted ones; LCM land inside L_w
    T4  a second run gives the same md5 for the csv and the npz; --above-land keep differs from one only at x >= |L_w|
    T5  refused: an existing folder, a folder inside an input folder, a real_eval path; p4_fits.npz changed after P4-1 fails
        D0 and nothing is written; --above-land stop fails D2 (SMALL named) and nothing is written
    bugs: mask row-flipped, L_w without effort, cap at 400, crossed not flagged, seed fits swapped, G5a fits used, a truth
        file opened, rule one skipped, seed columns swapped in the csv, rows given the next window, pooled = 100 maps
        (each caught by a gate or by T2's fresh code)
"""
import csv, hashlib, importlib.util, json, os, shutil, tempfile
from pathlib import Path
import numpy as np

T_EDGES, B_EDGES = np.array([0, 1, 2, 6, 11, 21, 51]), np.array([1, 4, 7, 11, 21])
LANDCL = [c for c in range(1, 22) if c not in (13, 14, 17, 18)]
SMALL, EXTRA, ALLLAND = "bnm_e0160_n0000.npz", "bnm_e0040_n0040.npz", "bnm_e0080_n0000.npz"
XS6 = np.array([1, 2, 5, 10, 20, 124])
METH = ("pooled", "s1", "s2")
FK = {"pooled": "pooled", "s1": "model_f_seed1", "s2": "model_f_seed2"}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def en(nm):
    s = Path(nm).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])


def shape_world(py):
    """SMALL: effort and records in tetrad row 0 only; EXTRA: species row 3 (no record) gets 2016 effort at 6 east tetrads
    that no row surveyed in any year. Done before any export. Returns the 6 tetrads."""
    py = Path(py)
    ydirs = [d for _, d in sorted((int(p.name.rsplit("_y", 1)[1]), p) for p in py.glob("minlist8_y*"))]
    M = np.zeros((20, 20), bool); M[0] = True
    for d in ydirs:
        z = dict(np.load(d / SMALL))
        z["present_year"] = (z["present_year"].astype(bool) & M).astype(np.uint8)
        z["known_holdout"] = (z["known_holdout"].astype(bool) & M).astype(np.uint8)
        np.savez(d / SMALL, **z)
    a = dict(np.load(py / "minlist8" / SMALL))
    a["obs_mask"] = (a["obs_mask"].astype(bool) & M).astype(np.uint8); a["known_target"] = (a["known_target"].astype(bool) & M).astype(np.uint8)
    np.savez(py / "minlist8" / SMALL, **a)
    U = np.load(py / "minlist8" / EXTRA)["known_target"].astype(bool).any(0)
    for d in ydirs:
        U |= np.load(d / EXTRA)["known_holdout"].astype(bool).any(0)
    free = [(r, c) for r in range(20) for c in range(10, 20) if not U[r, c]][:6]
    z = dict(np.load(ydirs[2] / EXTRA))
    kh = z["known_holdout"].astype(bool)
    for r, c in free:
        kh[3, r, c] = True
    z["known_holdout"] = kh.astype(np.uint8)
    np.savez(ydirs[2] / EXTRA, **z)
    return free


def plant_map33(zb, d, T29, seed=9):
    """test_29's planted map, with SMALL all sea outside its surveyed tetrads and ALLLAND all land."""
    win = sorted(set(zb["window"].astype(str).tolist()))
    E = [en(w)[0] for w in win]; N = [en(w)[1] for w in win]
    west, north = (min(E) - 4) * 1000, (max(N) + 44) * 1000
    H, Wd = (max(N) + 44) - (min(N) - 4), (max(E) + 44) - (min(E) - 4)
    rng = np.random.default_rng(seed)
    T = np.zeros((21, H, Wd), np.uint8)
    for r in range(H):
        for c in range(Wd):
            k1 = (r * Wd + c) % 21 + 1
            k2, k3 = rng.choice([x for x in LANDCL if x != k1], 2, replace=False)
            T[[k1 - 1, k2 - 1, k3 - 1], r, c] = (60, 30, 10)
    for w in win:
        if w == ALLLAND:
            continue
        e, n = en(w); S = zb[f"surveyed_mask__{w}"].astype(bool)
        for tr_ in range(20):
            for tc in range(20):
                if S[tr_, tc]:
                    continue
                cells = [((north - (n + 2 * tr_ + dn + 1) * 1000) // 1000, ((e + 2 * tc + de) * 1000 - west) // 1000)
                         for dn in (0, 1) for de in (0, 1)]
                if tc >= 10 or w == SMALL:
                    for rr, cc in cells:
                        T[:, rr, cc] = 0
                elif tr_ % 3 == 0:
                    for rr, cc in cells:
                        T[:, rr, cc] = 0; T[17, rr, cc] = 60; T[12, rr, cc] = 20
                elif tr_ % 3 == 1:
                    for j, (rr, cc) in enumerate(cells):
                        T[:, rr, cc] = 0; T[12, rr, cc] = 50
                        if j == 0:
                            T[15, rr, cc] = 1
    T29.write_map(d, T, west, north)


def fresh_land(py, lcm, names):
    """Fresh code: LCM land per tetrad from rasterio blocks (row 0 south), surveyed = any effort of any row in any year."""
    import rasterio
    with rasterio.open(Path(lcm) / "gb2019lcm1km_percentage_target.tif") as s:
        T = s.read().astype(int); west, north = s.transform.c, s.transform.f
    LC = T[[c - 1 for c in LANDCL]].sum(0)
    py = Path(py); ydirs = sorted(py.glob("minlist8_y*"))
    out = {}
    for w in names:
        e, nn = en(w)
        r0, c0 = int((north - (nn + 40) * 1000) // 1000), int((e * 1000 - west) // 1000)
        ll = (LC[r0:r0 + 40, c0:c0 + 40][::-1].reshape(20, 2, 20, 2) > 0).any((1, 3))
        S = np.load(py / "minlist8" / w)["known_target"].astype(bool).any(0)
        for d in ydirs:
            S = S | np.load(d / w)["known_holdout"].astype(bool).any(0)
        out[w] = (ll, S, ll | S)
    return out


def adjust_p2(kw, al, X):
    """After P2's planting: every 2019 record put into every map (the model keeps the records it is given), and row 0
    fully occupied at 2019 in the first 25 maps of every SMALL row, both seeds; P2's md5 records redone."""
    p2 = Path(kw["p2"]); st = np.load(al / "bnm_stacks.npz"); rows = np.nonzero(st["window"].astype(str) == SMALL)[0]
    fb = np.load(al / "bnm_frozen.npz")
    rb = np.zeros((len(fb["row"]), 50), np.uint8)
    for i, (f, r) in enumerate(zip(fb["file_index"], fb["row"])):
        with np.load(al / str(fb["files"][f])) as z:
            rb[i] = z["visit1"][r, -1] | z["visit2"][r, -1]
    ha = json.loads((p2 / "h2a_report.json").read_text())
    for r in kw["runs"]:
        f = f"samples_{Path(r).name}.npz"
        z = dict(np.load(p2 / f))
        mp = z["maps"].copy()
        mp[:, :, 250:300] |= rb[:, None, :]
        tb = np.unpackbits(mp[rows, :25, 250:300], axis=-1, count=400).reshape(len(rows), 25, 20, 20).astype(bool)
        tb[:, :, 0, :] = True
        mp[rows, :25, 250:300] = np.packbits(tb.reshape(len(rows), 25, 400), axis=-1)
        z["maps"] = mp
        X.save_npz(p2 / f, z)
        ha["files"][f] = md5(p2 / f); ha["parts"][f"samples_{Path(r).name}"]["files"][f] = md5(p2 / f)
    (p2 / "h2a_report.json").write_text(json.dumps(ha))
    kw["expect"]["p2_report"] = md5(p2 / "h2a_report.json")
    return len(rows)


def plant_p4(tmp, al, py, lcm, kw, P43):
    X, G5F, P41 = P43.X, P43.G5F, P43.P41
    st = np.load(al / "bnm_stacks.npz")
    fitw = sorted(set(st["window"].astype(str)[st["half"].astype(str) == "fit"].tolist()))
    fl = fresh_land(py, lcm, fitw)
    p40 = tmp / "data" / "p4_0"; p40.mkdir(parents=True)
    X.save_npz(p40 / "p4_windows.npz", dict(window_names=np.array(fitw), land_lcm=np.stack([fl[w][0] for w in fitw]),
                                            surveyed=np.stack([fl[w][1] for w in fitw]), land=np.stack([fl[w][2] for w in fitw]),
                                            val_window=np.zeros(4, np.int16), eval_window=np.zeros(4, np.int16), code_md5=np.array("planted")))
    mw = md5(p40 / "p4_windows.npz")
    (p40 / "p4_0_report.json").write_text(json.dumps(dict(gates={f"W{i}": True for i in range(7)}, md5=dict(out_npz=mw))))
    f5 = G5F.load_fits(Path(kw["fits_dir"]) / "g5_fits.npz")
    corr, recal = {}, {}
    for k in f5["methods"]:
        t = {q: (np.array(v, copy=True) if q != "option" else v) for q, v in f5["corr"][(k, "aoo")].items()}
        t["scale"] = t["scale"] * {"model_f_seed1": 0.9, "model_f_seed2": 1.1, "pooled": 1.25}.get(k, 1.0)
        if k in ("model_f_seed2", "pooled"):
            g = np.arange(len(t["q"])); cl = g if int(t["option"]) == 1 else g // 5
            sel = (cl == 1) | (cl == 2)
            t["q"] = np.where(sel, np.minimum(t["q"], 0) - 6, t["q"])
        corr[k] = t; recal[k] = f5["recal"][k]
    fits = dict(methods=f5["methods"], corr=corr, recal=recal, base_rate=f5["base_rate"], base_rate_all=f5["base_rate_all"],
                base_n=f5["base_n"], communities=f5["communities"], g3_md5=f5["g3_md5"], g3b_md5=f5["g3b_md5"],
                p4_windows_md5=mw, crossed="empty")
    p41 = tmp / "data" / "p4_1"; p41.mkdir()
    P41.save_fits(p41 / "p4_fits.npz", fits)
    tools = Path(P43.__file__).resolve().parent
    (p41 / "p4_1_report.json").write_text(json.dumps(dict(gates={f"Q{i}": True for i in range(7)}, crossed="empty", md5=dict(
        fits=md5(p41 / "p4_fits.npz"), p4w=mw, p4_0=md5(p40 / "p4_0_report.json"), g5=md5(Path(kw["fits_dir"]) / "g5_fits.npz"),
        tools={nm: md5(tools / nm) for nm in ("stage_p4_1_land_fits.py", "stage_g5_fits.py", "stage_g_calibrate.py",
                                              "stage_g_recalibrate.py")}))))
    return p40, p41


def indep(out_dir, k4, above):
    """Fresh code from the files."""
    al, p2, py, lcm = Path(k4["bnm"]), Path(k4["p2"]), Path(k4["peryear"]), Path(k4["lcm"])
    st = np.load(al / "bnm_stacks.npz"); win = st["window"].astype(str); n = len(win)
    fl = fresh_land(py, lcm, sorted(set(win.tolist())))
    L = np.stack([fl[w][2] for w in win]); nl = L.reshape(n, -1).sum(1)
    fb = np.load(al / "bnm_frozen.npz"); rec = fb["rec_t"].astype(int)
    rcls = np.searchsorted(T_EDGES, rec, side="right") - 1
    rbin = np.clip(np.searchsorted(B_EDGES, fb["rec6"].astype(int), side="right") - 1, 0, 4)
    a = {}
    for m, s in (("s1", "f_seed1"), ("s2", "f_seed2")):
        mp = np.load(p2 / f"samples_{s}.npz")["maps"]
        occ = np.unpackbits(mp[..., 250:300], axis=-1, count=400).reshape(n, mp.shape[1], 20, 20).astype(bool)
        a[m] = (occ & L[:, None]).sum((2, 3)).astype(float)
    a["pooled"] = np.concatenate([a["s1"][:, :25], a["s2"][:, :25]], 1)
    fz = np.load(Path(k4["p4_1"]) / "p4_fits.npz"); z = np.load(out_dir / "products_land_bnm.npz")
    worst, cr = 0.0, {}
    for m in METH:
        k = FK[m]; opt = int(fz[f"corr__{k}__aoo__option"]); g = rcls if opt == 1 else rcls * 5 + rbin
        s, q = fz[f"corr__{k}__aoo__scale"][g], fz[f"corr__{k}__aoo__q"][g]
        sc = s[:, None] * a[m]
        Lr = np.maximum(np.percentile(sc, 5, axis=1) - q, rec); Ur = np.maximum(np.percentile(sc, 95, axis=1) + q, rec)
        c = Ur < Lr; cr[m] = c
        mean = np.minimum(sc.mean(1), nl)
        Lc, Uc = np.where(c, np.nan, np.minimum(Lr, nl)), np.where(c, np.nan, np.minimum(Ur, nl))
        pb = z[f"p_b_{m}"]
        p2_ = np.where(rec > 2, 0.0, (sc < 2.5).mean(1)); p2_ = np.minimum(np.maximum(p2_, pb[:, 0]), pb[:, 1])
        p20 = np.where(rec > 20, 0.0, pb[:, 3])
        p124 = np.maximum(np.where(rec > 124, 0.0, (sc < 124.5).mean(1)), p20)
        P = np.column_stack([pb[:, 0], p2_, pb[:, 1], np.where(rec > 10, 0.0, pb[:, 2]), p20, p124])
        if above == "one":
            P = np.where(XS6[None, :] >= nl[:, None], 1.0, P)
        for mine, theirs in ((a[m].mean(1), z[f"raw_{m}"]), (mean, z[f"mean_{m}"]), (Lc, z[f"L_{m}"]), (Uc, z[f"U_{m}"]),
                             (c.astype(float), z[f"crossed_{m}"].astype(float)), (P, z[f"P_{m}"]), (nl, z["land_n"])):
            mine, theirs = np.asarray(mine, float), np.asarray(theirs, float)
            if mine.shape != theirs.shape or not np.array_equal(np.isnan(mine), np.isnan(theirs)):
                worst = np.inf; continue
            ok_ = ~np.isnan(mine)
            worst = max(worst, float(np.abs(mine[ok_] - theirs[ok_]).max()) if ok_.any() else 0.0)
    with open(out_dir / "products_land_bnm.csv") as fh:
        rows = list(csv.DictReader(fh))
    ok_csv = len(rows) == n and all(int(r["land_tetrads"]) == int(nl[i]) and all((r[f"aoo_lo90_{m}"] == "") == bool(cr[m][i])
                                                                                    and (r[f"aoo_hi90_{m}"] == "") == bool(cr[m][i]) for m in METH)
                                    and all(abs(float(r[f"p_aoo_le_{x}_{m}"]) - float(z[f"P_{m}"][i, j])) <= 5.01e-5
                                            for m in METH for j, x in enumerate(XS6)) for i, r in enumerate(rows))
    return dict(worst=worst, csv=ok_csv, nl=nl, win=win, crossed=cr, fl=fl)


def main(tool_path, tools):
    tool_path = Path(tool_path)
    P43 = load(tool_path, "stage_p4_3_land_products")
    P3C, P3, P3B, X, GS, G5F = P43.P3C, P43.P3, P43.P3B, P43.X, P43.GS, P43.G5F
    tdir = tool_path.parent.parent / "tests"
    T27 = load(tdir / "test_27_p3_products.py", "test_27_p3_products")
    T29 = load(tdir / "test_29_p3c_land_line.py", "test_29_p3c_land_line")
    P1 = load(tool_path.parent / "stage_p1_bnm_all_export.py", "stage_p1_bnm_all_export")
    H1 = P1.H1
    quiet = lambda *a, **k: None
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- U1 by hand
    NC, NX = len(P43.TCLS), len(P43.GC.XS)
    hf = dict(corr={"m": dict(option=1, scale=np.full(NC, 2.0), q=np.where(np.arange(NC) == 1, -3.0, 0.5), src=np.zeros(NC, int),
                              nfit=np.full(NC, 99))},
              recal={"m": dict(w=np.zeros((NX, NC, 2)), w_all=np.zeros((NX, 2)), src=np.full((NX, NC), 2), nclass=np.zeros((NX, NC)),
                               nall=np.zeros(NX), events=np.zeros((NX, NC)))}, crossed="empty")
    aoo = np.array([[10, 10, 10, 10], [1, 1, 1, 1], [9, 10, 12, 15]], float)
    rec = np.array([5, 1, 3]); rc_ = np.searchsorted(T_EDGES, rec, side="right") - 1; rb_ = np.zeros(3, int); nl_ = np.array([30, 30, 18])
    pk, po = (P43.products(hf, "m", aoo, rec, rc_, rb_, nl_, ab) for ab in ("keep", "one"))
    exp_P = np.array([[0, 0, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0.5, 1]], float)
    u1 = np.allclose(pk["mean"], [20, 2, 18]) and np.allclose(pk["L"][[0, 2]], [19.5, 17.8]) and np.allclose(pk["U"][[0, 2]], [20.5, 18]) \
        and np.isnan(pk["L"][1]) and np.isnan(pk["U"][1]) and list(pk["crossed"]) == [False, True, False] and pk["inverted"] == 1 \
        and list(pk["capped"]) == [False, False, True] and np.allclose(pk["P"], exp_P) and np.allclose(po["P"][2], [0, 0, 0, 0, 1, 1]) \
        and np.allclose(po["P"][:2], exp_P[:2]) and int(po["set_one"].sum()) == 1
    t("U1", u1, f"by hand: means {pk['mean'].round(3).tolist()} (20, 2, 18 capped); intervals {np.round(pk['L'], 3).tolist()} / "
      f"{np.round(pk['U'], 3).tolist()} (19.5-20.5, crossed blank, 17.8-18 capped); crossed {pk['crossed'].tolist()}; capped "
      f"{pk['capped'].tolist()}; P row 3 keep {pk['P'][2].tolist()}, one {po['P'][2].tolist()} (P(20) 0.5 then 1)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- planted world and chain
        py, refpat = H1.world(tmp / "bw")
        added = shape_world(py)
        data = tmp / "data"
        fit, test, al = data / "bnm_fit", data / "bnm_test", data / "bnm_all"
        rf, _ = H1.run(py, refpat, tools, fit, expect=(6, 6), out=quiet)
        fzh = tmp / "fz_h1.json"
        fzh.write_text(json.dumps(dict(tools={"stage_h1_bnm_export.py": X.file_md5(P1.H1_FILE)})))
        rt_, _ = H1.run(py, refpat, tools, test, expect=(6, 6), half="test", final_test=True, freeze=str(fzh), out=quiet)
        rp, _ = P1.run(py, refpat, tools, fit, test, al, expect=(6, 6), out=quiet)
        kw, h6 = T27.plant(tmp, al, P3)
        n_small = adjust_p2(kw, al, X)
        p3d, p3b, p3c = data / "p3", data / "p3b", data / "p3c"
        r3, _ = P3.run(out_dir=p3d, out=quiet, **kw)
        rb, _ = P3B.run(bnm=al, p2=kw["p2"], p3=p3d, fits_dir=kw["fits_dir"], runs=kw["runs"], out_dir=p3b, out=quiet,
                        expect=dict(p1_report=kw["expect"]["p1_report"], p2_report=kw["expect"]["p2_report"], p3_report=md5(p3d / "p3_report.json")))
        zb = dict(np.load(p3b / "unsurveyed_bnm.npz"))
        lcm = tmp / "lcm"
        plant_map33(zb, lcm, T29)
        lnames = ("gb2019lcm1km_percentage_target.tif", "gb2019lcm1km_percentage_aggregate.tif", "gb2019lcm1km_dominant_target.tif")
        ex = dict(p3b_report=md5(p3b / "p3b_report.json"), band_check=X.file_md5(tool_path.parent / "stage_p3c_band_check.py"),
                  lcm={nm: md5(lcm / nm) for nm in lnames})
        rc, _ = P3C.run(p2=kw["p2"], p3=p3d, p3b=p3b, lcm=lcm, runs=kw["runs"], expect=ex, out_dir=p3c, out=quiet)
        t("S0", all(rf.values()) and all(rt_.values()) and all(rp.values()) and all(r3.values()) and all(rb.values()) and all(rc.values())
          and n_small > 0, f"planted H1 fit and test, P1 {rp}, P3 {r3}, P3b {rb}, P3c {rc}; SMALL rows {n_small}")
        p40, p41 = plant_p4(tmp, al, py, lcm, kw, P43)
        k4 = dict(bnm=al, p2=kw["p2"], p3=p3d, p3c=p3c, p4_0=p40, p4_1=p41, fits_dir=kw["fits_dir"], h6=kw["h6"], lcm=lcm, peryear=py,
                  seeds=("f_seed1", "f_seed2"),
                  expect=dict(p1_report=md5(al / "h1_report.json"), p2_report=kw["expect"]["p2_report"], p3_report=md5(p3d / "p3_report.json"),
                              p3c_report=md5(p3c / "p3c_report.json"), p4_0_report=md5(p40 / "p4_0_report.json"),
                              p4_windows=md5(p40 / "p4_windows.npz"), p4_1_report=md5(p41 / "p4_1_report.json"),
                              p4_fits=md5(p41 / "p4_fits.npz"), h6_report=md5(kw["h6"]), g5a_fits=md5(Path(kw["fits_dir"]) / "g5_fits.npz"),
                              p3c_fingerprint=P43.fingerprint(tool_path.parent / "stage_p3c_land_line.py"),
                              band_check=X.file_md5(tool_path.parent / "stage_p3c_band_check.py"), lcm=md5(lcm / lnames[0])))
        lines = []
        out1 = data / "p4_3"
        r, rep = P43.run(out_dir=out1, above_land="one", out=lines.append, **k4)
        t("T1", set(r) == {f"D{i}" for i in range(6)} and all(r.values()), f"P4-3 on the planted inputs, --above-land one: {r}")
        if not all(r.values()):
            print("      " + "\n      ".join(lines))
        ind = indep(out1, k4, "one") if out1.exists() else None
        t("T2", ind is not None and ind["worst"] <= 1e-9 and ind["csv"],
          f"fresh code from the files: largest difference {ind['worst'] if ind else float('nan'):.1e}; csv blanks, land tetrads and P "
          f"{ind and ind['csv']}")
        if ind is not None:
            z = np.load(out1 / "products_land_bnm.npz"); z3 = np.load(p3d / "products_bnm.npz")
            nl, win = ind["nl"], ind["win"]
            alln, small = nl == 400, win == SMALL
            eq_all = all(np.array_equal(z[f"raw_{m}"][alln], z3[f"raw_{m}"][alln]) for m in METH)
            below = all(bool((z[f"raw_{m}"][~alln] <= z3[f"raw_{m}"][~alln] + 1e-12).all()) and bool((z[f"raw_{m}"][~alln] < z3[f"raw_{m}"][~alln]).any())
                        for m in METH)
            ncr = {m: int(z[f"crossed_{m}"].sum()) for m in METH}
            ncap = int(z["capped_pooled"].sum())
            sm = bool(small.any()) and bool((nl[small] <= 20).all()) and all(bool((z[f"P_{m}"][small][:, 4:] == 1).all()) for m in METH)
            set1 = sum(v["set_to_1"] for m in METH for v in rep["above"][m].values())
            pb = {w["name"]: w["put_back"] for w in rep["windows"]}
            fe = ind["fl"][EXTRA]; pbt = sorted(map(tuple, np.argwhere(~fe[0] & fe[1]).tolist()))
            inside = all(bool((ind["fl"][w][0] <= ind["fl"][w][2]).all()) for w in ind["fl"])
            t("T3", alln.any() and eq_all and below and ncr["s2"] > 0 and ncr["pooled"] > 0 and ncap > 0 and sm and set1 > 0
              and pb.get(EXTRA) == 6 and pbt == sorted(map(tuple, added)) and inside and sum(pb.values()) == 6,
              f"all-land rows {int(alln.sum())}: land-only = P3 window AOO {eq_all}; coastal rows below or equal, some below {below}; "
              f"crossed {ncr}; capped pooled {ncap}; SMALL rows {int(small.sum())} with |L_w| {sorted(set(nl[small].tolist()))} and "
              f"P(20) = P(124) = 1 {sm}, values set to 1 {set1}; put back in EXTRA {pb.get(EXTRA)} = the planted tetrads "
              f"{pbt == sorted(map(tuple, added))}, elsewhere {sum(pb.values()) - pb.get(EXTRA, 0)}; LCM land inside L_w {inside}")
        else:
            t("T3", False, "no output to check")
        out2, out3 = data / "p4_3_again", data / "p4_3_keep"
        P43.run(out_dir=out2, above_land="one", out=quiet, **k4)
        same = out1.exists() and out2.exists() and all(X.file_md5(out1 / f) == X.file_md5(out2 / f)
                                                       for f in ("products_land_bnm.csv", "products_land_bnm.npz"))
        rk, _ = P43.run(out_dir=out3, above_land="keep", out=quiet, **k4)
        if out3.exists() and out1.exists():
            za, zk = np.load(out1 / "products_land_bnm.npz"), np.load(out3 / "products_land_bnm.npz")
            ab = XS6[None, :] >= za["land_n"][:, None]
            kd = all(np.array_equal(za[f"P_{m}"][~ab], zk[f"P_{m}"][~ab]) for m in METH) and \
                any(bool((za[f"P_{m}"][ab] != zk[f"P_{m}"][ab]).any()) for m in METH) and \
                all(P3.same(za[k_], zk[k_]) for k_ in za.files if not k_.startswith("P_") and k_ != "above_land")
        else:
            kd = False
        t("T4", same and all(rk.values()) and kd, f"a second run gives the same md5 for the csv and the npz {same}; --above-land keep "
          f"passes {rk} and differs from one only in P at x >= |L_w| {kd}")
        stops = []
        for o in (out1, p3d / "inside", data / "real_eval_p4_3"):
            try:
                P43.run(out_dir=o, above_land="one", out=quiet, **k4); stops.append(False)
            except SystemExit:
                stops.append(True)
        p41t = tmp / "p4_1_tampered"
        shutil.copytree(p41, p41t)
        zz = dict(np.load(p41t / "p4_fits.npz")); zz["corr__pooled__aoo__scale"] = zz["corr__pooled__aoo__scale"] * 1.001
        X.save_npz(p41t / "p4_fits.npz", zz)
        rq, _ = P43.run(out_dir=data / "p4_3_tampered", above_land="one", out=quiet, **dict(k4, p4_1=p41t))
        ls_ = []
        rs, _ = P43.run(out_dir=data / "p4_3_stop", above_land="stop", out=ls_.append, **k4)
        t("T5", all(stops) and rq.get("D0") is False and not (data / "p4_3_tampered").exists() and rs.get("D2") is False
          and not (data / "p4_3_stop").exists() and any(SMALL in l_ and "give --above-land" in l_ for l_ in ls_),
          f"refused: an existing folder {stops[0]}, a folder inside an input folder {stops[1]}, a real_eval path {stops[2]}; "
          f"p4_fits.npz changed after P4-1 fails D0, nothing written {rq}; --above-land stop fails D2 naming SMALL, nothing written {rs}")
        for bug in ("mask_flip", "no_effort_land", "cap400", "no_crossed", "fits_swap", "g5a_fits", "opentruth", "one_skip",
                    "csv_swap", "window_shift", "pool100"):
            os.environ["STAGE_P4_3_BROKEN"] = bug
            ob = data / f"bug_{bug}"
            try:
                rb_, _ = P43.run(out_dir=ob, above_land="one", out=quiet, **k4)
            finally:
                os.environ.pop("STAGE_P4_3_BROKEN", None)
            failed = [k_ for k_, v in rb_.items() if not v]
            ib = indep(ob, k4, "one") if ob.exists() and not failed else None
            caught = bool(failed) or (ib is not None and (ib["worst"] > 1e-9 or not ib["csv"]))
            t(f"bug {bug}", caught, f"caught: gates failing {failed or 'none'}" + (f", fresh code largest difference {ib['worst']:.1e}, "
                                                                                 f"csv {ib['csv']}" if ib else ""))
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)