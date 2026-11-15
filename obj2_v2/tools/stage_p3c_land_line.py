"""
stage_p3c_land_line.py  (obj2_v2, products step P3c, 13 Nov 2026; report only; Sara chose option (b) after the P3b result
and left the land rule to me; the rule is in claude/obj2_v2_H_rules_draft.md, written before any map file was read)
 
Question: how much of each P3 window AOO lies in tetrads with no land (open sea, open water, the shore between tides),
and what is the AOO over land only? Report only: the P3 products do not change, nothing is scored, no truth file is opened.
 
Rule (H rules doc, P3c)
    Map: UKCEH Land Cover Map 2019, 1 km percentage target class, GB file (21 bands, band k = class k, confirmed by P3c-2
        version 2). Land classes: all except 13 Saltwater, 14 Freshwater, 17 Littoral rock, 18 Littoral sediment.
    Land cover of a 1 km cell = the sum of its land-class percentages; a cell outside the raster has land cover 0.
    A tetrad is land if any of its four 1 km cells has land cover above 0; otherwise "no land". Window grid as step 10:
        row r from the south, column c from the west, 2 km per cell; tetrad (r, c) of window (e, n) km covers eastings
        e + 2c to e + 2c + 2 km and northings n + 2r to n + 2r + 2 km.
    Share of a species-window: of the occupied tetrads of its maps at 2019 (summed over the maps), the share in no-land
        tetrads; pooled (first 25 maps of each seed) and each seed.
    Report-only line, labelled: AOO over land = the row's G5a pooled scale (as P3 and P3b) x mean occupied land tetrads of
        the pooled maps, at most the window's land tetrads; no interval, no P.
 
Gates
    L0 inputs: P3b p3b_report.json md5 = Sara's run, gates PASS, its npz md5 = its record, stage_p3b_unsurveyed_check.py =
       the md5 P3b logged; P3 p3_report.json and products_bnm.npz md5 = P3b's record; P2 h2a_report.json md5 = P3b's
       record, sample files = its list; the three map files = block P3c-0; stage_p3c_band_check.py = version 2
    L1 band order: stage_p3c_band_check.py version 2 on the three map files says CONFIRMED (group sums and plurality
       both ways on classes 1-21, no dominant value outside 0-21); map 21 bands, uint8, EPSG 27700, 1 km pixels on whole km
    L2 alignment: products rows = P3b rows; samples in frozen order; window corners on whole 2 km; surveyed window-tetrads
       classed no land at most 1% of all surveyed window-tetrads, and none of the 7 other orientations of the window grid
       gives fewer
    L3 code: pooled map AOO recounted = P3b's mean (1e-12); scale x mean, capped at 400, = P3's corrected mean (1e-9);
       P3b's never-surveyed share recomputed from its masks = its file (1e-12); no-land share and land line by plain loop
       code on 200 rows = vectorised
    L4 no *_truth file and no real_eval path passed to np.load; np.load only on the P2, P3 and P3b folders
    L5 outputs: shares in [0, 1] (undefined only where a row's maps are all empty); land + no land = 400; occupied land +
       no land = occupied; land line <= the row's land tetrads and <= P3's uncapped mean; files read back = memory
    Refused: an output folder that exists, lies inside an input folder, or has real_eval or _truth in its path.
 
Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_p3c_land_line.py --self-test          (needs ../tests/test_29_p3c_land_line.py and test_27)
    python3 -u ../tools/stage_p3c_land_line.py > ../logs/stage_p3c_land_line.txt 2>&1
Needs stage_p3b_unsurveyed_check.py, stage_p3c_band_check.py and the tools they load; rasterio.
"""
import argparse, csv, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
 
HERE = Path(__file__).resolve().parent
 
 
def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_p3c_land_line.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
 
 
P3B = _load("stage_p3b_unsurveyed_check")
BC = _load("stage_p3c_band_check")
P3, GC, GS, X = P3B.P3, P3B.GC, P3B.GS, P3B.X
G, W, CELLS, K, HALF, TCLS = P3B.G, P3B.W, P3B.CELLS, P3B.K, P3B.HALF, P3B.TCLS
METH = P3B.METH
NOLAND = (13, 14, 17, 18)
EXPECT = dict(p3b_report="29b9c44f8548d391e85a90733f37457c", band_check="98faa2d95d1a96fa6680972c1100f3ca", lcm=dict(BC.MD5))
ORIENT = {"identity": lambda a: a, "row flip": lambda a: a[::-1], "column flip": lambda a: a[:, ::-1],
          "both flips": lambda a: a[::-1, ::-1], "transpose": lambda a: a.T, "transpose, row flip": lambda a: a.T[::-1],
          "transpose, column flip": lambda a: a.T[:, ::-1], "transpose, both flips": lambda a: a.T[::-1, ::-1]}
 
 
def broken():
    return os.environ.get("STAGE_P3C_BROKEN", "")   # self-test only: plants one bug so its gate must fail
 
 
def read_map(lcm):
    """Land cover per 1 km cell (sum of the land-class percentages) and the raster's transform."""
    import rasterio
    with rasterio.open(Path(lcm) / BC.PCT) as s:
        tr, epsg, ok = s.transform, (s.crs.to_epsg() if s.crs else None), s.count == 21 and s.dtypes[0] == "uint8"
        T = s.read().astype(np.int32)
    ok = ok and epsg == 27700 and tr.a == 1000 and tr.e == -1000 and tr.b == 0 and tr.d == 0 and tr.c % 1000 == 0 and tr.f % 1000 == 0
    keep = [c - 1 for c in range(1, 22) if c not in NOLAND]
    return T[keep].sum(0), tr, ok
 
 
def window_land(LC, tr, e_km, n_km):
    """(G, G) land flag and land share (sum of the four cells / 400) per tetrad, row from the south, column from the west,
    and the number of 1 km cells outside the raster."""
    H, Wd = LC.shape
    land, share, outside = np.zeros((G, G), bool), np.zeros((G, G)), 0
    for r in range(G):
        for c in range(G):
            v = []
            for dn in (0, 1):
                for de in (0, 1):
                    E, N = (e_km + 2 * c + de) * 1000, (n_km + 2 * r + dn) * 1000          # south-west corner of the cell
                    col, row = int((E - tr.c) // 1000), int((tr.f - (N + 1000)) // 1000)
                    if 0 <= row < H and 0 <= col < Wd:
                        v.append(int(LC[row, col]))
                    else:
                        v.append(0); outside += 1
            land[r, c] = max(v) > 0 if broken() != "half_land" else sum(v) >= 200
            share[r, c] = sum(v) / 400.0
    return land, share, outside
 
 
def loop_land(maps_row, land_row):
    """Plain loop code for one row (L3): mean occupied land tetrads over the maps."""
    s = 0
    for k in range(maps_row.shape[0]):
        bits = np.unpackbits(maps_row[k])[(W - 1) * CELLS:W * CELLS]
        s += sum(int(land_row[c // G, c % G]) for c in np.nonzero(bits)[0])
    return s / maps_row.shape[0]
 
 
# ---------------------------------------------------------------- run
def run(p2="../data/stage_p2_predict", p3="../data/stage_p3_products", p3b="../data/stage_p3b_unsurveyed",
        lcm="../data/lcm2019_1km/data", runs=("../runs/f_seed1", "../runs/f_seed2"), out_dir="../data/stage_p3c_land",
        expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    p2, p3, p3b, lcm, od = map(Path, (p2, p3, p3b, lcm, out_dir))
    runs = [Path(r) for r in runs]
    if od.exists():
        sys.exit(f"STOP: {od} exists; P3c writes into a new folder")
    if "real_eval" in str(od.resolve()) or "_truth" in str(od.resolve()):
        sys.exit("STOP: the output path may not contain real_eval or _truth")
    for p_ in (p2, p3, p3b, lcm):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    need = [p2 / "h2a_report.json", p3 / "p3_report.json", p3 / "products_bnm.npz", p3b / "p3b_report.json",
            p3b / "unsurveyed_bnm.npz"] + [lcm / f for f in (BC.PCT, BC.AGG, BC.DOM)]
    miss = [str(f) for f in need if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    t0, res, rep = time.time(), {}, {}
 
    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")
 
    def stop(after):
        n_ok = sum(res.values())
        out(f"stage P3c: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, rep
    me = X.file_md5(Path(__file__))
    tmd5 = {nm: X.file_md5(HERE / nm) for nm in ("stage_p3b_unsurveyed_check.py", "stage_p3c_band_check.py")}
    out(f"stage_p3c_land_line.py md5 {me}; " + ", ".join(f"{k} {v}" for k, v in tmd5.items()))
    with P3.LoadGuard() as lg:
        try:
            # ================================================================ L0 inputs
            rb, mb = json.loads((p3b / "p3b_report.json").read_text()), X.file_md5(p3b / "p3b_report.json")
            mz = X.file_md5(p3b / "unsurveyed_bnm.npz")
            r3, m3, mn = json.loads((p3 / "p3_report.json").read_text()), X.file_md5(p3 / "p3_report.json"), X.file_md5(p3 / "products_bnm.npz")
            ha, m2 = json.loads((p2 / "h2a_report.json").read_text()), X.file_md5(p2 / "h2a_report.json")
            fbad = [f for f in (f"samples_{r.name}.npz" for r in runs) if f not in ha["files"] or X.file_md5(p2 / f) != ha["files"][f]]
            ml = {n_: X.file_md5(lcm / n_) for n_ in (BC.PCT, BC.AGG, BC.DOM)}
            b_ok = mb == expect["p3b_report"] and all(rb["gates"].values()) and rb["md5"]["out_npz"] == mz and \
                rb["md5"]["tool"] == tmd5["stage_p3b_unsurveyed_check.py"]
            c_ok = rb["md5"]["p3_report"] == m3 and rb["md5"]["products_npz"] == mn and rb["md5"]["p2_report"] == m2 and not fbad
            l_ok = all(ml[k_] == expect["lcm"][k_] for k_ in ml) and tmd5["stage_p3c_band_check.py"] == expect["band_check"]
            gate("L0", b_ok and c_ok and l_ok,
                 f"P3b p3b_report.json {mb[:8]} = Sara's run {mb == expect['p3b_report']}, gates PASS, npz {mz[:8]} and tool = its record "
                 f"{b_ok}; P3 p3_report.json {m3[:8]}, products_bnm.npz {mn[:8]}, P2 h2a_report.json {m2[:8]} = P3b's record, sample "
                 f"files = P2's list (differ {len(fbad)}) {c_ok}; map files {', '.join(v[:8] for v in ml.values())} = block P3c-0 and "
                 f"stage_p3c_band_check.py = version 2 {l_ok}")
            if not res["L0"]:
                return stop("L0")
            # ================================================================ L1 band order and map
            bc = BC.check(lcm, want_md5=False, quiet=True) if broken() != "skip_bands" else dict(conf=True)
            LC, tr, m_ok = read_map(lcm)
            gate("L1", bc["conf"] and m_ok, f"band order by stage_p3c_band_check.py version 2: "
                 f"{'CONFIRMED' if bc['conf'] else 'NOT CONFIRMED'}; map 21 bands, uint8, EPSG 27700, 1 km pixels on whole km {m_ok}")
            if not res["L1"]:
                return stop("L1")
            # ================================================================ L2 alignment
            zb = dict(np.load(p3b / "unsurveyed_bnm.npz")); z3 = dict(np.load(p3 / "products_bnm.npz"))
            win = zb["window"].astype(str); n = len(win); wins = sorted(set(win.tolist()))
            al = np.array_equal(z3["window"].astype(str), win) and np.array_equal(z3["species"].astype(str), zb["species"].astype(str)) \
                and np.array_equal(z3["rec_2019"], zb["rec_2019"])
            maps, why, ok1 = {}, [], True
            for m_, r in zip(("s1", "s2"), runs):
                z = np.load(p2 / f"samples_{r.name}.npz")
                good = np.array_equal(z["sel"], np.arange(n)) and str(z["layout"]) == GS.LAYOUT and str(z["frozen_md5"]) == r3["md5"]["frozen"] \
                    and int(z["K"]) == K and z["maps"].shape == (n, K, W * CELLS // 8)
                ok1 &= good; why.append(f"{r.name} {good}"); maps[m_] = z["maps"]
            corner = all(e_ % 2 == 0 and n_ % 2 == 0 for e_, n_ in map(P3B.en, wins))
            surv = {w_: zb[f"surveyed_mask__{w_}"].astype(bool) for w_ in wins}
            land, lshare, outside = {}, {}, {}
            for w_ in wins:
                e_, n_ = P3B.en(w_)
                land[w_], lshare[w_], outside[w_] = window_land(LC, tr, e_, n_)
                if broken() == "orient":
                    land[w_] = land[w_][:, ::-1]
            n_surv_all = int(sum(surv[w_].sum() for w_ in wins))
            o_cnt = {k_: int(sum((surv[w_] & ~f_(land[w_])).sum() for w_ in wins)) for k_, f_ in ORIENT.items()}
            ident = o_cnt["identity"]
            l2 = al and ok1 and corner and ident <= 0.01 * n_surv_all and ident <= min(o_cnt.values())
            gate("L2", l2, f"{n:,} products rows = P3b rows {al}; samples in frozen order: {', '.join(why)}; window corners on whole 2 km "
                 f"{corner}; surveyed window-tetrads classed no land {ident:,} of {n_surv_all:,} ({ident / max(n_surv_all, 1):.4f}, at most 0.01); "
                 f"other orientations: " + ", ".join(f"{k_} {v:,}" for k_, v in o_cnt.items() if k_ != "identity"))
            if not res["L2"]:
                return stop("L2")
            # ================================================================ counts and L3 code
            NL_row = np.stack([~land[w_] for w_ in win]); U_row = np.stack([~surv[w_] for w_ in win])
            cnt = {m_: P3B.occ_counts(maps[m_], NL_row) for m_ in ("s1", "s2")}
            cnt["pooled"] = tuple(np.concatenate([cnt[m_][q][:, :HALF] for m_ in ("s1", "s2")], 1) for q in (0, 1))
            share = {m_: P3B.share_of(*cnt[m_]) for m_ in METH}
            if broken() == "share_land":
                share["pooled"] = 1 - share["pooled"]
            ul = {m_: P3B.occ_counts(maps[m_], U_row & ~NL_row)[1] for m_ in ("s1", "s2")}
            ul_p = np.concatenate([ul["s1"][:, :HALF], ul["s2"][:, :HALF]], 1)
            us = {m_: P3B.occ_counts(maps[m_], U_row)[1] for m_ in ("s1", "s2")}
            us_p = np.concatenate([us["s1"][:, :HALF], us["s2"][:, :HALF]], 1)
            tot_p, nl_p = cnt["pooled"]
            raw = tot_p.mean(1)
            s_row = zb["scale"] * (1.01 if broken() == "scale_bad" else 1.0)
            uncapped = s_row * raw
            n_land = np.array([land[w_].sum() for w_ in win])
            mean_l = (tot_p - nl_p).mean(1)
            aoo_l = np.minimum(s_row * mean_l, n_land)
            share_nsl = P3B.share_of(tot_p, ul_p)
            d_raw = float(np.abs(raw - zb["mean_occupied_pooled"]).max())
            d_mean = float(np.abs(np.minimum(uncapped, CELLS) - z3["mean_pooled"]).max())
            d_p3b = float(np.nanmax(np.abs(P3B.share_of(tot_p, us_p) - zb["share_pooled"])))
            pick = np.sort(X.rng(31, 8).choice(n, min(200, n), replace=False))
            pm = lambda i: np.concatenate([maps["s1"][i, :HALF], maps["s2"][i, :HALF]], 0)
            loop_s = np.array([P3B.loop_share(pm(i), NL_row[i]) for i in pick])
            loop_l = np.array([min(s_row[i] * loop_land(pm(i), ~NL_row[i]), n_land[i]) for i in pick])
            lo = P3.same(loop_s, share["pooled"][pick], 1e-12) and P3.same(loop_l, aoo_l[pick], 1e-9)
            gate("L3", d_raw <= 1e-12 and d_mean <= 1e-9 and d_p3b <= 1e-12 and lo,
                 f"pooled map AOO recounted = P3b's mean (largest difference {d_raw:.1e}); scale x mean, capped at {CELLS}, = P3's corrected "
                 f"mean ({d_mean:.1e}); P3b's never-surveyed share from its masks = its file ({d_p3b:.1e}); no-land share and land line by "
                 f"loop code on {len(pick)} rows = vectorised {lo}")
            if not res["L3"]:
                return stop("L3")
            empty = {m_: cnt[m_][0].sum(1) == 0 for m_ in METH}
            r_share = all(bool(((share[m_][~empty[m_]] >= 0) & (share[m_][~empty[m_]] <= 1)).all()) and
                          bool(np.isnan(share[m_][empty[m_]]).all()) for m_ in METH)
            r_sum = bool((n_land + NL_row.reshape(n, -1).sum(1) == CELLS).all()) and all(bool((cnt[m_][1] <= cnt[m_][0]).all()) for m_ in METH)
            r_line = bool((aoo_l <= n_land + 1e-9).all() and (aoo_l <= uncapped + 1e-9).all())
            if broken() == "opentruth":
                np.load(p3 / "bnm_e0000_n0000_truth.npz")
            crashed = None
        except RuntimeError as e:
            if not lg.refused:
                raise
            crashed = str(e)
    if crashed:
        gate("L4", False, f"np.load refused: {crashed[:160]}")
        return stop("L4")
    if not (r_share and r_sum and r_line):
        gate("L5", False, f"shares {r_share}; sums {r_sum}; land line {r_line}")
        return stop("L5")
    # ================================================================ reports (not tests)
    status, half = zb["status"].astype(str), z3["half"].astype(str)
    lab = np.array([TCLS[c] for c in zb["rcls"]])
    aoo_p, aoo_s, sh_b = z3["mean_pooled"], zb["aoo_surveyed_only_pooled"], zb["share_pooled"]
    q = lambda a: (float(np.nanmedian(a)), float(np.nanpercentile(a, 10)), float(np.nanpercentile(a, 90)))
    wrows = []
    for w_ in wins:
        m = win == w_; e_, n_ = P3B.en(w_)
        nl, ns = ~land[w_], ~surv[w_]
        nsl = lshare[w_][ns & land[w_]]
        wrows.append(dict(window=w_, half=half[m][0], e_km=e_, n_km=n_, species=int(m.sum()), land=int(land[w_].sum()),
                          no_land=int(nl.sum()), never_surveyed=int(ns.sum()), never_surveyed_no_land=int((ns & nl).sum()),
                          never_surveyed_land=int((ns & land[w_]).sum()), surveyed_no_land=int((~ns & nl).sum()),
                          ns_land_share_below_5=int((nsl < 0.05).sum()), ns_land_share_5_50=int(((nsl >= 0.05) & (nsl < 0.5)).sum()),
                          ns_land_share_50_up=int((nsl >= 0.5).sum()), cells_outside_map=int(outside[w_]),
                          median_share_no_land=float(np.nanmedian(share["pooled"][m])), max_aoo_p3=float(aoo_p[m].max()),
                          max_aoo_land_only=float(aoo_l[m].max())))
    tw = lambda k_: int(sum(r[k_] for r in wrows))
    out(f"   tetrads of the {len(wins)} windows: no land {tw('no_land'):,} of {len(wins) * CELLS:,}; never surveyed {tw('never_surveyed'):,}, "
        f"of which no land {tw('never_surveyed_no_land'):,} and land {tw('never_surveyed_land'):,}; never-surveyed land tetrads by land "
        f"share: below 5% {tw('ns_land_share_below_5'):,}, 5-50% {tw('ns_land_share_5_50'):,}, 50% or more {tw('ns_land_share_50_up'):,}; "
        f"1 km cells outside the map {tw('cells_outside_map'):,}")
    out("   share of the occupied tetrads in no-land tetrads, median [10th, 90th percentile]: " + "; ".join(
        f"{m_} {q(share[m_])[0]:.3f} [{q(share[m_])[1]:.3f}, {q(share[m_])[2]:.3f}]" for m_ in METH)
        + f"; in never-surveyed land tetrads (pooled) {q(share_nsl)[0]:.3f} [{q(share_nsl)[1]:.3f}, {q(share_nsl)[2]:.3f}]; P3b never-surveyed "
        f"{q(sh_b)[0]:.3f}")
    summ = {}
    for gname, gv, labs in (("recorded class", lab, TCLS), ("Red List 2022", status, sorted(set(status.tolist())))):
        summ[gname] = {}
        out(f"   by {gname} (rows; median share no land; median share never-surveyed land; median P3 AOO; median land-only line; "
            f"median surveyed-only line; report only):")
        for g_ in labs:
            m = gv == g_
            if not m.any():
                continue
            v = summ[gname][g_] = dict(rows=int(m.sum()), share_no_land=float(np.nanmedian(share["pooled"][m])),
                                       share_ns_land=float(np.nanmedian(share_nsl[m])), aoo_p3=float(np.median(aoo_p[m])),
                                       aoo_land_only=float(np.median(aoo_l[m])), aoo_surveyed_only=float(np.median(aoo_s[m])))
            out(f"      {g_:<6} {v['rows']:>5,}; {v['share_no_land']:.3f}; {v['share_ns_land']:.3f}; {v['aoo_p3']:.1f}; {v['aoo_land_only']:.1f}; "
                f"{v['aoo_surveyed_only']:.1f}")
    wq = sorted(wrows, key=lambda r: -r["no_land"])
    out("   windows with the most no-land tetrads (window: no land; never surveyed; surveyed no land; median species share no land): "
        + "; ".join(f"{r['window'][:-4]} {r['no_land']}; {r['never_surveyed']}; {r['surveyed_no_land']}; {r['median_share_no_land']:.2f}" for r in wq[:8]))
    # ================================================================ files
    od.mkdir(parents=True)
    sav = dict(window=win, species=zb["species"].astype(str), status=status, rcls=zb["rcls"], land=n_land.astype(np.int16),
               scale=s_row, mean_occupied_pooled=raw, mean_occupied_land_pooled=mean_l, aoo_land_only_pooled=aoo_l,
               share_never_surveyed_land_pooled=share_nsl, p3b_report_md5=np.array(mb))
    for m_ in METH:
        sav[f"share_no_land_{m_}"] = share[m_]
    for w_ in wins:
        sav[f"land_mask__{w_}"] = land[w_]; sav[f"land_share__{w_}"] = lshare[w_]
    X.save_npz(od / "land_bnm.npz", sav)
    head = ["window", "half", "species", "red_list_2022", "rec_2019", "recorded_class", "land_tetrads_window", "no_land_tetrads_window"] + \
        [f"share_no_land_{m_}" for m_ in METH] + ["share_never_surveyed_land_pooled", "share_never_surveyed_pooled_p3b",
                                                  "mean_occupied_land_pooled", "aoo_mean_pooled_p3", "aoo_surveyed_only_pooled_report",
                                                  "aoo_land_only_pooled_report"]
    shcols = list(METH) if broken() != "csv_swap" else ["pooled", "s2", "s1"]
    f4 = lambda v: "" if np.isnan(v) else f"{v:.4f}"
    with open(od / "land_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head)
        for i in range(n):
            w.writerow([win[i], half[i], sav["species"][i], status[i], int(zb["rec_2019"][i]), lab[i], int(n_land[i]), int(CELLS - n_land[i])]
                       + [f4(share[m_][i]) for m_ in shcols] + [f4(share_nsl[i]), f4(sh_b[i]), f"{mean_l[i]:.3f}", f"{aoo_p[i]:.3f}",
                                                                f"{aoo_s[i]:.3f}", f"{aoo_l[i]:.3f}"])
    with open(od / "per_window_land.csv", "w", newline="") as fh:
        wd = csv.DictWriter(fh, fieldnames=list(wrows[0])); wd.writeheader(); wd.writerows(wrows)
    zr = np.load(od / "land_bnm.npz")
    back_npz = all(P3.same(zr[k_], v) for k_, v in sav.items())
    with open(od / "land_bnm.csv") as fh:
        rc = list(csv.DictReader(fh))
    col = lambda c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rc])
    back_csv = len(rc) == n and [r["species"] for r in rc] == list(sav["species"]) and \
        all(P3.same(col(f"share_no_land_{m_}"), share[m_], 5.01e-5) for m_ in METH) and \
        P3.same(col("aoo_land_only_pooled_report"), aoo_l, 5.01e-4) and P3.same(col("land_tetrads_window"), n_land)
    gate("L5", r_share and r_sum and r_line and back_npz and back_csv,
         f"shares in [0, 1], undefined only for rows whose maps are all empty {r_share}; land + no land = {CELLS}, occupied parts add up "
         f"{r_sum}; land line <= the land tetrads and <= P3's uncapped mean {r_line}; land_bnm.npz read back = memory {back_npz}; "
         f"land_bnm.csv read back = memory (rounding) {back_csv}")
    other = [p for p in lg.paths if not any(Path(p).resolve().parent == a.resolve() for a in (p2, p3, p3b))]
    gate("L4", not lg.refused and not other, f"np.load refused {len(lg.refused)}; files passed to np.load {len(lg.paths)}, other than the "
         f"P2 samples, the P3 products and the P3b check {len(other)}; the map is read with rasterio")
    rep.update(gates=res, n=int(n), note="report only: nothing scored, nothing changed; the P3 products stay as signed",
               rule=dict(no_land_classes=list(NOLAND), land_tetrad="any of its four 1 km cells with land cover above 0",
                         grid="row from the south, column from the west, 2 km per cell (step 10)",
                         aoo_land_only="report only: G5a scale x mean occupied land tetrads (pooled), at most the window's land tetrads; no interval, no P"),
               orientation_counts=o_cnt, surveyed_window_tetrads=n_surv_all,
               md5=dict(tool=me, tools=tmd5, p3b_report=mb, p3_report=m3, products_npz=mn, p2_report=m2, lcm=ml,
                        out_npz=X.file_md5(od / "land_bnm.npz"), out_csv=X.file_md5(od / "land_bnm.csv"),
                        per_window_csv=X.file_md5(od / "per_window_land.csv")),
               share_no_land={m_: dict(median=q(share[m_])[0], p10=q(share[m_])[1], p90=q(share[m_])[2]) for m_ in METH},
               share_never_surveyed_land=dict(median=q(share_nsl)[0], p10=q(share_nsl)[1], p90=q(share_nsl)[2]),
               totals={k_: tw(k_) for k_ in ("land", "no_land", "never_surveyed", "never_surveyed_no_land", "never_surveyed_land",
                                             "surveyed_no_land", "ns_land_share_below_5", "ns_land_share_5_50", "ns_land_share_50_up",
                                             "cells_outside_map")},
               summaries=summ, windows=wrows, minutes=round((time.time() - t0) / 60, 2), written_utc=P3B.now())
    (od / "p3c_report.json").write_text(json.dumps(rep, indent=1, default=float))
    n_ok = sum(res.values())
    out(f"   written land_bnm.csv {rep['md5']['out_csv']}, land_bnm.npz {rep['md5']['out_npz']}, per_window_land.csv, p3c_report.json "
        f"({(time.time() - t0) / 60:.1f} min)")
    out(f"stage P3c: gates {n_ok} PASS, {len(res) - n_ok} FAIL; {n:,} species-windows, {len(wins)} windows; report only")
    return res, rep
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--p2", default="../data/stage_p2_predict")
    ap.add_argument("--p3", default="../data/stage_p3_products")
    ap.add_argument("--p3b", default="../data/stage_p3b_unsurveyed")
    ap.add_argument("--lcm", default="../data/lcm2019_1km/data")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--out", default="../data/stage_p3c_land")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_29_p3c_land_line.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_29_p3c_land_line", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.main(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.p2, a.p3, a.p3b, a.lcm, a.runs, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)
 