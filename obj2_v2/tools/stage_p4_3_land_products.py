#!/usr/bin/env python3
"""
stage_p4_3_land_products.py  (obj2_v2, P4 step P4-3, 14 Nov 2026; P4 plan signed by Sara on 13 Nov, rule E and the P4-3
rule for crossed rows on 14 Nov, her yes to P4-3 on 14 Nov; claude/obj2_v2_H_rules_draft.md, P4 section)

Question: for every species-window of the 101 BNM windows (P1 export, P2 samples), the land-only corrected AOO at 2019
with its 90% interval and P(land-only AOO <= x) at x 1, 2, 5, 10, 20, 124, from the P4 fits (p4_fits.npz 654da879)
applied unchanged. A new version of the P3 products; P3 stays as signed. Products, not a test: true AOO is unknown on
BNM and nothing is scored.

Rules (signed; nothing here changes them)
    L_w of a window: the P4-0 rule, now for all 101 windows: P3c's land (LCM2019 1 km, every class except 13, 14, 17, 18; a
        tetrad is land if any of its four 1 km cells is above 0; rows from the south) plus every tetrad with effort in any of
        the 6 years for any species row of the window in the BNM per-year grids. For the 44 fit-half windows this must be
        p4_windows.npz exactly.
    Land-only AOO of a map: occupied tetrads of L_w at 2019. Methods as P3: pooled = the first 25 maps of each seed is the
        product; seed 1 and seed 2 beside it.
    stage_p4_1_land_fits.apply_land with the P4 fits of the method, rule E (as P4-2): corrected mean, 90% interval (lower
        end at least the 2019 recorded tetrads), option B P at x 1, 5, 10, 20. Then P3's six_p, unchanged: P = 0 where the
        2019 recorded tetrads exceed x; x 2 and 124 from G3's p_le on the corrected maps.
    Cap: the corrected mean and both interval ends at most |L_w|, the land tetrads of the window.
    Crossed interval (upper end below lower end, before the cap; rule E): the mean and P are given, the interval is blank in
        the csv and NaN in the npz, the note says "no interval (crossed)"; counted by recorded class.
    x at or above |L_w| (--above-land, asked before the run): keep = P as six_p gives it; one = P = 1 at every x >= |L_w|
        (a land-only AOO cannot pass |L_w|); stop (default) = D2 fails if any row has such an x, before any AOO is made.
    Beside each row: |L_w|, all land (|L_w| = 400) or coastal, P3's pooled mean (window AOO) for comparison, H6's test-half
        H-R2 ratio and coverage of its recorded class as in P3 (H6 not reread: its holes are surveyed tetrads, all inside
        L_w), and for recorded classes 0-1 the 1-cell note with P4-2's land-only figure (true class 1: 4.31 seed 1, 4.46
        seed 2, 4.41 pooled).
    EOO is left out (as P3).

Gates
    D0 inputs = Sara's runs: P1 h1_report.json (gates PASS, half all, files = report, input files = manifest, H1 tool = the
       one P1 used); P2 h2a_report.json (half all, gates and parts PASS, truth opened 0, files = list, frozen md5 = P1);
       P3 p3_report.json (gates PASS, products_bnm.npz = report, made from these P1, P2, H6 and G5a files, P3 tool and G5a
       code = the files that made it); P3c p3c_report.json (gates PASS, land_bnm.npz = report, made from this P3);
       stage_p3c_land_line.py fingerprint, stage_p3c_band_check.py version 2, LCM file = block P3c-0, band order CONFIRMED;
       P4-0 p4_0_report.json and p4_windows.npz; P4-1 p4_1_report.json (gates PASS) with p4_fits.npz = 654da879, crossed
       rule empty, made from this p4_windows.npz, P4-0 report and G5a fits, P4-1 and G5a code = the files that made it;
       h6_report.json (gates PASS, first reading); G5a g5_fits.npz
    D1 alignment: rows = P1 stacks; 2019 recorded tetrads and 6-year records recounted from the input files = bnm_frozen = P3;
       classes, window, half, species and status = bnm_stacks = P3; samples in frozen order with the layout, K 50 and the
       frozen md5, both seeds from the same inputs; maps on the record grid (share of 2019 recorded tetrads occupied:
       identity above each of the 7 other orientations, both seeds)
    D2 land masks: windows = the P1 windows; per-year grids read for every window, species rows with a record = P1 rows;
       LCM part = P3c's land mask in every window; fit-half windows: LCM part, surveyed part and L_w = p4_windows.npz; every
       effort and record tetrad of every P1 row inside L_w; the --above-land rule given if any row has x >= |L_w|
    D3 code path: window AOO of every map = P3's raw means; land-only AOO by two paths equal, and = window AOO in all-land
       windows; apply_land with the G5a fits on window AOO, P3's 400 cap and six_p = P3's products (mean, interval, P,
       option B, G3's P, recalibrated P) within 1e-12; P4 fits: the six methods, crossed rule empty, p4_windows md5 stored
    D4 product rules: pooled = first 25 maps of each seed; interval not crossed: recorded tetrads <= lower <= upper <= |L_w|;
       crossed rows NaN, flagged by upper < lower before the cap, count = apply_land's; mean <= |L_w|; P in [0, 1], rising
       over x, 0 below the recorded tetrads, x 1 and 5 = option B, x 10 and 20 = option B or 0, the --above-land rule; the
       files read back = memory; H-R2 columns = h6_report.json; notes
    D5 no *_truth file and no file in a real_eval folder passed to np.load; every file loaded lies in an input folder
    Refused: an output folder that exists, lies inside an input folder, or has real_eval or _truth in its path.
Outputs (../data/stage_p4_3_products): products_land_bnm.csv, products_land_bnm.npz, p4_3_report.json.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, minutes)
    python3 ../tools/stage_p4_3_land_products.py --self-test         (needs ../tests/test_33_p4_3_land_products.py, test_27, test_29)
    python3 -u ../tools/stage_p4_3_land_products.py --above-land one > ../logs/stage_p4_3_products.txt 2>&1
Needs stage_p3_bnm_products.py, stage_p3c_land_line.py, stage_p4_1_land_fits.py, stage_h1_bnm_export.py and what they load,
rasterio, and the BNM per-year grids.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_p4_3_land_products.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P3C = _load("stage_p3c_land_line")
P3B, P3, BC = P3C.P3B, P3C.P3, P3C.BC
P41 = _load("stage_p4_1_land_fits")
H1 = _load("stage_h1_bnm_export")
G5F, GC, GB, GS, D, X = P3.G5F, P3.GC, P3.GB, P3.GS, P3.D, P3.X
G, W, CELLS, TCLS, BINS = P3.G, P3.W, P3.CELLS, P3.TCLS, P3.BINS
XS6, METH, K, HALF = P3.XS6, P3.METH, P3.K, P3.HALF
YB = CELLS // 8
TB0 = (W - 1) * YB
TOL = 1e-12
EXPECT = dict(p1_report="569c9a397574997ce818a14dbfd21eff", p2_report="0dc83e6b25d32ae8a1d726289fd8e0df",
              p3_report="517c98e52a611d92e1e69a866eed8c1e", p3c_report="6f0516765ecfebcb54eafad26a566e98",
              p4_0_report="842697b1acf4f8d6484e6d0f659d8b73", p4_windows="928cf96c9bcbe9a14bb0f639adb581eb",
              p4_1_report="5e980763bcd5935ad9bc31dbeb4bd0f1", p4_fits="654da879cf43a061d7bd685d5636b01e",
              h6_report="5142027a04e61aab88a89939bf91b4b0", g5a_fits="77ed594e4d311e6945bacea460673a8b",
              p3c_fingerprint="86af2eae6462c4972d6a7b6cca8d50cb", band_check="98faa2d95d1a96fa6680972c1100f3ca",
              lcm=BC.MD5[BC.PCT])
ABOVE = {"stop": "D2 fails if any row has an x at or above |L_w| (asked before the run)",
         "keep": "P as P3's six_p gives it, also at x >= |L_w|",
         "one": "P = 1 at every x >= |L_w| (a land-only AOO cannot pass |L_w|)"}
NOTE_1CELL = ("recorded class 0-1: if the true range is 1 tetrad, the land-only corrected AOO came out 4.3-4.5 times too "
              "large on simulation (P4-2 real_eval reading 2, true class 1, both seeds)")
NOTE_CROSSED = "no interval (crossed)"
ORIENT8 = {"identity": lambda a: a, "row flip": lambda a: a[:, ::-1], "column flip": lambda a: a[:, :, ::-1],
           "both flips": lambda a: a[:, ::-1, ::-1], "transpose": lambda a: a.transpose(0, 2, 1),
           "transpose, row flip": lambda a: a.transpose(0, 2, 1)[:, ::-1],
           "transpose, column flip": lambda a: a.transpose(0, 2, 1)[:, :, ::-1],
           "transpose, both flips": lambda a: a.transpose(0, 2, 1)[:, ::-1, ::-1]}


def broken():
    return os.environ.get("STAGE_P4_3_BROKEN", "")    # self-test only: plants one bug so a gate or the fresh code must catch it


def fingerprint(p):
    """md5 ignoring a first #! line and trailing spaces (a file copied as text still matches)."""
    b = Path(p).read_bytes().split(b"\n")
    return hashlib.md5(b"\n".join(l.rstrip() for l in b if l[:2] != b"#!").rstrip(b"\n")).hexdigest()


def g5a_as_p4(f5, methods):
    """G5a's AOO fits in P4-1's fits layout, rule E (G5a never collapsed an interval); as stage_p4_2_eval_reading.py."""
    return dict(corr={k: f5["corr"][(k, "aoo")] for k in methods}, recal={k: f5["recal"][k] for k in methods}, crossed="empty")


def maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf")
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return float("inf")
    return float(np.abs(a[~na] - b[~nb]).max()) if (~na).any() else 0.0


def grid_shares(maps, rec, ch=1000):
    """Share of 2019 recorded tetrads occupied in the maps, for the 8 orientations of the maps."""
    num, den = {k: 0 for k in ORIENT8}, 0
    for a in range(0, len(maps), ch):
        tb = maps[a:a + ch, :, TB0:TB0 + YB]
        C = np.unpackbits(tb, axis=-1, count=CELLS).reshape(len(tb), tb.shape[1], G, G).sum(1, dtype=np.int64)
        R = rec[a:a + ch]
        den += int(R.sum()) * tb.shape[1]
        for k, f in ORIENT8.items():
            num[k] += int((f(C) * R).sum())
    return {k: num[k] / max(den, 1) for k in num}


def products(fits, kk, aoo_l, rec_t, rcls, rbin, nl, above):
    """Land-only products of one method: apply_land (rule E of the fits), the crossed flag before the cap, the cap at |L_w|,
    P3's six_p and the --above-land rule."""
    a = P41.apply_land(fits, kk, aoo_l, rec_t, rcls, rbin)
    ca = a["aoo"]
    crossed = ca["U"] < ca["L"]
    if broken() == "no_crossed":
        crossed = np.zeros(len(crossed), bool)
    cap = nl.astype(float) if broken() != "cap400" else np.full(len(nl), float(CELLS))
    mean = np.minimum(ca["mean"], cap)
    L = np.where(crossed, np.nan, np.minimum(ca["L"], cap))
    U = np.where(crossed, np.nan, np.minimum(ca["U"], cap))
    P6 = P3.six_p(a["p_b"], ca["scaled"], rec_t)
    ab = np.array(XS6)[None, :] >= nl[:, None]
    one = above == "one" and broken() != "one_skip"
    P = np.where(ab, 1.0, P6) if one else P6.copy()
    return dict(raw=np.asarray(aoo_l, float).mean(1), mean=mean, L=L, U=U, L_unc=ca["L"], U_unc=ca["U"], crossed=crossed,
                capped=(ca["mean"] > nl) | (~crossed & ((ca["L"] > nl) | (ca["U"] > nl))), raised=ca["L"] > ca["Lraw"],
                P=P, P6=P6, above=ab, set_one=ab & (P6 < 1) if above == "one" else np.zeros_like(ab),
                p_b=a["p_b"], p_g3=a["p_g3"], p_recal=a["p_recal"], inverted=int(a["inverted"]))


def note_of(rcls, R, i):
    parts = [NOTE_1CELL] if rcls[i] <= 1 else []
    parts += [f"{NOTE_CROSSED}, {m_}" for m_ in METH if R[m_]["crossed"][i]]
    return "; ".join(parts)


# ---------------------------------------------------------------- run
def run(bnm="../data/bnm_all", p2="../data/stage_p2_predict", p3="../data/stage_p3_products", p3c="../data/stage_p3c_land",
        p4_0="../data/stage_p4_0", p4_1="../data/stage_p4_1", fits_dir="../data/stage_g5_fits",
        h6="../data/stage_h6_test_score/h6_report.json", lcm="../data/lcm2019_1km/data",
        peryear="../../AI_simulation/Empirical_data/bnm_grids_peryear", seeds=("f_seed1", "f_seed2"),
        out_dir="../data/stage_p4_3_products", above_land="stop", minlist=8, expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    bnm, p2, p3, p3c, p4_0, p4_1, fits_dir, h6, lcm, peryear, od = map(Path, (bnm, p2, p3, p3c, p4_0, p4_1, fits_dir, h6, lcm,
                                                                              peryear, out_dir))
    if above_land not in ABOVE:
        sys.exit(f"STOP: --above-land must be one of {list(ABOVE)}")
    if len(seeds) != 2:
        sys.exit("STOP: P4-3 needs the two seeds (--seeds f_seed1 f_seed2)")
    if od.exists():
        sys.exit(f"STOP: {od} exists; P4-3 writes into a new folder")
    if "real_eval" in str(od.resolve()) or "_truth" in str(od.resolve()):
        sys.exit("STOP: the output path may not contain real_eval or _truth")
    pooled = peryear / f"minlist{minlist}"
    ydirs = sorted((int(p.name.rsplit("_y", 1)[1]), p) for p in peryear.glob(f"minlist{minlist}_y*") if p.is_dir())
    ins = (bnm, p2, p3, p3c, p4_0, p4_1, fits_dir, h6.parent, lcm, pooled, *[d for _, d in ydirs])
    for p_ in ins:
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    need = [bnm / "h1_report.json", bnm / "bnm_frozen.npz", bnm / "bnm_stacks.npz", bnm / "manifest.csv", p2 / "h2a_report.json",
            *[p2 / f"samples_{s}.npz" for s in seeds], p3 / "p3_report.json", p3 / "products_bnm.npz", p3c / "p3c_report.json",
            p3c / "land_bnm.npz", p4_0 / "p4_0_report.json", p4_0 / "p4_windows.npz", p4_1 / "p4_1_report.json",
            p4_1 / "p4_fits.npz", fits_dir / "g5_fits.npz", h6, lcm / BC.PCT]
    miss = [str(f) for f in need if not f.is_file()] + ([] if pooled.is_dir() and ydirs else [f"{pooled} and its year folders"])
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    t0, res, rep = time.time(), {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage P4-3: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, rep
    me = X.file_md5(Path(__file__))
    tnames = ("stage_p3_bnm_products.py", "stage_p4_1_land_fits.py", "stage_g5_fits.py", "stage_g_calibrate.py",
              "stage_g_recalibrate.py", "stage_h1_bnm_export.py", "stage_p3c_band_check.py")
    tmd5 = {nm: X.file_md5(HERE / nm) for nm in tnames}
    tmd5["stage_p3c_land_line.py fingerprint"] = fingerprint(HERE / "stage_p3c_land_line.py")
    out(f"stage_p4_3_land_products.py md5 {me}; --above-land {above_land} ({ABOVE[above_land]}); "
        + ", ".join(f"{k} {v}" for k, v in tmd5.items()))
    k1, k2 = (f"model_{s}" for s in seeds)
    methods = [k1, k2, "pooled", "B3", "B4", "B5"]
    key0 = {"pooled": "pooled", "s1": k1, "s2": k2}
    key = key0 if broken() != "fits_swap" else {"pooled": "pooled", "s1": k2, "s2": k1}
    g5code = {nm: tmd5[nm] for nm in ("stage_g5_fits.py", "stage_g_calibrate.py", "stage_g_recalibrate.py")}
    with P3.LoadGuard() as lg:
        try:
            # ================================================================ D0 inputs
            jl = lambda p: (json.loads(Path(p).read_text()), X.file_md5(p))
            js1, m1 = jl(bnm / "h1_report.json")
            fmd5, smd5, mmd5 = (X.file_md5(bnm / f) for f in ("bnm_frozen.npz", "bnm_stacks.npz", "manifest.csv"))
            man_rows = D.rows_of(bnm)
            bad_in = [r["input_file"] for r in man_rows if X.file_md5(bnm / r["input_file"]) != r["input_md5"]]
            p1_ok = m1 == expect["p1_report"] and js1.get("tool_name") == "stage_p1_bnm_all_export.py" and js1["half"] == "all" \
                and all(js1["gates"].values()) and js1["md5"]["bnm_frozen.npz"] == fmd5 and js1["md5"]["bnm_stacks.npz"] == smd5 \
                and js1["md5"]["manifest.csv"] == mmd5 and not bad_in and js1["md5"]["h1_tool"] == tmd5["stage_h1_bnm_export.py"]
            ha, m2 = jl(p2 / "h2a_report.json")
            fbad = [f for f, m in ha["files"].items() if not (p2 / f).is_file() or X.file_md5(p2 / f) != m]
            p2_ok = m2 == expect["p2_report"] and ha["half"] == "all" and all(ha["gates"].values()) and ha["truth_opened"] == 0 \
                and all(all(v["gates"].values()) for v in ha["parts"].values()) and not fbad \
                and all(f"samples_{s}.npz" in ha["files"] for s in seeds) and ha["frozen_md5"] == fmd5
            r6, m6 = jl(h6)
            h6_ok = m6 == expect["h6_report"] and all(r6["gates"].values()) and r6.get("rerun") is False
            mg5 = X.file_md5(fits_dir / "g5_fits.npz")
            f5 = G5F.load_fits(fits_dir / "g5_fits.npz")
            g5_ok = mg5 == expect["g5a_fits"] and f5["methods"] == methods
            r3, m3 = jl(p3 / "p3_report.json")
            mp3 = X.file_md5(p3 / "products_bnm.npz")
            p3_ok = m3 == expect["p3_report"] and all(r3["gates"].values()) and r3["md5"]["products_npz"] == mp3 \
                and r3["md5"]["p1_report"] == m1 and r3["md5"]["p2_report"] == m2 and r3["md5"]["h6_report"] == m6 \
                and r3["md5"]["fits"] == mg5 and r3["md5"]["tool"] == tmd5["stage_p3_bnm_products.py"] and r3["md5"]["tools"] == g5code
            rc, mc = jl(p3c / "p3c_report.json")
            mlb = X.file_md5(p3c / "land_bnm.npz")
            lcm_md5 = X.file_md5(lcm / BC.PCT)
            bc = BC.check(lcm, want_md5=False, quiet=True)
            LC, tr, map_ok = P3C.read_map(lcm)
            p3c_ok = mc == expect["p3c_report"] and all(rc["gates"].values()) and rc["md5"]["out_npz"] == mlb \
                and rc["md5"]["p3_report"] == m3 and rc["md5"]["lcm"][BC.PCT] == lcm_md5 == expect["lcm"] and bc["conf"] and map_ok \
                and tmd5["stage_p3c_land_line.py fingerprint"] == expect["p3c_fingerprint"] \
                and tmd5["stage_p3c_band_check.py"] == expect["band_check"]
            r40, m40 = jl(p4_0 / "p4_0_report.json")
            mw = X.file_md5(p4_0 / "p4_windows.npz")
            p40_ok = m40 == expect["p4_0_report"] and all(r40["gates"].values()) and r40["md5"]["out_npz"] == mw == expect["p4_windows"]
            r41, m41 = jl(p4_1 / "p4_1_report.json")
            mf4 = X.file_md5(p4_1 / "p4_fits.npz")
            fits4 = P41.load_fits(p4_1 / "p4_fits.npz")
            want41 = dict(g5code, **{"stage_p4_1_land_fits.py": tmd5["stage_p4_1_land_fits.py"]})
            p41_ok = m41 == expect["p4_1_report"] and all(r41["gates"].values()) and r41["md5"]["fits"] == mf4 == expect["p4_fits"] \
                and r41["md5"]["p4w"] == mw and r41["md5"]["p4_0"] == m40 and r41["md5"]["g5"] == mg5 \
                and all(r41["md5"]["tools"].get(k_) == v for k_, v in want41.items()) and r41.get("crossed") == "empty" \
                and fits4["crossed"] == "empty" and fits4["p4_windows_md5"] == mw and fits4["methods"] == methods
            gate("D0", p1_ok and p2_ok and h6_ok and g5_ok and p3_ok and p3c_ok and p40_ok and p41_ok,
                 f"P1 h1_report.json {m1[:8]} = Sara's run, gates PASS, half all, files = report, input files = manifest (differ "
                 f"{len(bad_in)}), H1 tool = P1's {p1_ok}; P2 h2a_report.json {m2[:8]}, half all, gates and parts PASS, truth "
                 f"opened {ha['truth_opened']}, files = list (differ {len(fbad)}), frozen md5 = P1 {p2_ok}; h6_report.json {m6[:8]} "
                 f"{h6_ok}; G5a g5_fits.npz {mg5[:8]}, six methods {g5_ok}; P3 p3_report.json {m3[:8]}, gates PASS, "
                 f"products_bnm.npz {mp3[:8]} = report, made from these P1, P2, H6, G5a files and this code {p3_ok}; P3c "
                 f"p3c_report.json {mc[:8]}, gates PASS, land_bnm.npz {mlb[:8]} = report, made from this P3, map {lcm_md5[:8]} = "
                 f"block P3c-0, band order {'CONFIRMED' if bc['conf'] else 'NOT CONFIRMED'}, P3c tool fingerprint and band check "
                 f"version 2 {p3c_ok}; P4-0 p4_0_report.json {m40[:8]}, p4_windows.npz {mw[:8]} {p40_ok}; P4-1 p4_1_report.json "
                 f"{m41[:8]}, gates PASS, p4_fits.npz {mf4[:8]} = report, crossed rule {fits4['crossed']}, made from this P4-0 and "
                 f"G5a, P4-1 and G5a code = the files that made it {p41_ok}")
            if not res["D0"]:
                return stop("D0")
            # ================================================================ D1 alignment
            fb = dict(np.load(bnm / "bnm_frozen.npz")); n = len(fb["row"])
            idx = dict(np.load(bnm / "bnm_stacks.npz"))
            z3 = dict(np.load(p3 / "products_bnm.npz"))
            V1, V2, EF, _, _ = GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])
            V = V1 | V2
            del V1, V2
            rec_g = V[:, -1].reshape(n, G, G)
            rec_t, rec6 = rec_g.reshape(n, -1).sum(1), V.any(1).reshape(n, -1).sum(1)
            rec_any, eff_any = V.any(1).reshape(n, G, G), EF.any(1).reshape(n, G, G)
            del V, EF
            rcls, rbin = D.tidx(rec_t), D.bidx(rec6)
            status = np.array([s_ if s_ else "none" for s_ in idx["status"].astype(str)])
            win_s, half_s = idx["window"].astype(str), idx["half"].astype(str)
            al = n == int(js1["stacks"]) and np.array_equal(rec_t, fb["rec_t"]) and np.array_equal(rec6, fb["rec6"]) \
                and np.array_equal(rcls, fb["cls"]) and np.array_equal(rec_t, z3["rec_2019"]) and np.array_equal(rec6, z3["rec_6y"]) \
                and np.array_equal(rcls, z3["rcls"]) and np.array_equal(rbin, z3["rbin"]) and P3.same(win_s, z3["window"]) \
                and P3.same(half_s, z3["half"]) and P3.same(idx["species"].astype(str), z3["species"]) and P3.same(status, z3["status"])
            maps, why, ok1, inm, sh = {}, [], True, [], {}
            for m_, s in zip(("s1", "s2"), seeds):
                z = np.load(p2 / f"samples_{s}.npz")
                good = np.array_equal(z["sel"], np.arange(n)) and str(z["layout"]) == GS.LAYOUT and str(z["frozen_md5"]) == fmd5 \
                    and int(z["K"]) == K and z["maps"].shape == (n, K, W * YB)
                ok1 &= good; why.append(f"{s} {good}"); inm.append(str(z["in_md5"]))
                maps[m_] = z["maps"]
                sh[m_] = grid_shares(maps[m_], rec_g)
            orient = all(sh[m_]["identity"] > max(v for k_, v in sh[m_].items() if k_ != "identity") for m_ in sh)
            gate("D1", al and ok1 and inm[0] == inm[1] and orient,
                 f"{n:,} rows = P1 stacks; 2019 recorded tetrads and 6-year records recounted from the input files = bnm_frozen "
                 f"= P3, classes, window, half, species and status = bnm_stacks = P3 {al}; samples in frozen order with layout, "
                 f"K {K} and frozen md5: {', '.join(why)}; same inputs for both seeds {inm[0] == inm[1]}; share of 2019 recorded "
                 f"tetrads occupied, identity / best other orientation: " + ", ".join(
                     f"{m_} {sh[m_]['identity']:.4f} / {max(v for k_, v in sh[m_].items() if k_ != 'identity'):.4f}" for m_ in sh)
                 + f" {orient}")
            if not res["D1"]:
                return stop("D1")
            # ================================================================ D2 land masks
            wins = sorted(set(win_s.tolist()))
            nw = len(wins)
            miss_y = [f"{y}/{nm}" for y, d in ydirs for nm in wins if not (d / nm).is_file()] + \
                [nm for nm in wins if not (pooled / nm).is_file()]
            land_lcm, surv = np.zeros((nw, G, G), bool), np.zeros((nw, G, G), bool)
            rows_ok, outside_map, rec_out, rows_eq = not miss_y, 0, 0, True
            if not miss_y:
                for wi, nm in enumerate(wins):
                    R_, Ef_, sp_, _, ok_, o_ = H1.read_window(pooled, ydirs, nm)
                    rows_ok &= ok_
                    if not ok_:
                        continue
                    e_, n_ = P3B.en(nm)
                    ll, _, oc = P3C.window_land(LC, tr, e_, n_)
                    land_lcm[wi] = ll[::-1] if broken() == "mask_flip" else ll
                    surv[wi] = Ef_.any((0, 1)); outside_map += oc; rec_out += o_
                    rows_eq &= int(R_.any((0, 2, 3)).sum()) == int((win_s == nm).sum())
            land = land_lcm.copy() if broken() == "no_effort_land" else land_lcm | surv
            zc = np.load(p3c / "land_bnm.npz")
            lcm_ok = all(f"land_mask__{nm}" in zc.files and np.array_equal(land_lcm[wi], zc[f"land_mask__{nm}"]) for wi, nm in enumerate(wins)) \
                and P3.same(zc["window"], win_s) and P3.same(zc["species"], idx["species"].astype(str))
            zw = dict(np.load(p4_0 / "p4_windows.npz"))
            fitw = sorted(set(win_s[half_s == "fit"].tolist()))
            wix = {nm: i for i, nm in enumerate(wins)}
            names4 = [str(x) for x in zw["window_names"]]
            fit_ok = sorted(names4) == fitw and len(set(names4)) == len(names4) and all(np.array_equal(land_lcm[wix[nm]], zw["land_lcm"][j]) and np.array_equal(surv[wix[nm]], zw["surveyed"][j])
                                            and np.array_equal(land[wix[nm]], zw["land"][j]) for j, nm in enumerate(names4))
            win_row = np.array([wix[w_] for w_ in win_s])
            if broken() == "window_shift":
                win_row = (win_row + 1) % nw
            Lr = land[win_row]
            inside = not bool((eff_any & ~Lr).any()) and not bool((rec_any & ~Lr).any())
            nl_w = land.reshape(nw, -1).sum(1)
            nl = nl_w[win_row]
            ab_w = [wins[w_] for w_ in np.nonzero(nl_w <= max(XS6))[0]]
            n_ab = int((nl <= max(XS6)).sum())
            ab_ok = above_land != "stop" or n_ab == 0
            gate("D2", len(wins) == int(js1["windows"]["all"]) and rows_ok and rows_eq and lcm_ok and fit_ok and inside and ab_ok,
                 f"{nw} windows = P1's {js1['windows']['all']}; per-year grids read for every window ({len(ydirs)} year folders, "
                 f"missing {len(miss_y)}) {rows_ok}, species rows with a record = P1 rows in every window {rows_eq}; LCM part = "
                 f"P3c's land mask in every window, P3c rows = P1 rows {lcm_ok}; {len(fitw)} fit-half windows: LCM part, surveyed part and L_w = "
                 f"p4_windows.npz {fit_ok}; effort and record tetrads of every P1 row inside L_w {inside}; rows with an x at or "
                 f"above |L_w|: {n_ab:,} in {len(ab_w)} window(s) {', '.join(f'{w_} ({int(nl_w[wix[w_]])})' for w_ in ab_w)}, rule "
                 f"{above_land} {ab_ok}" + ("" if ab_ok else "; give --above-land keep or one (no AOO made)"))
            if not res["D2"]:
                return stop("D2")
            # ================================================================ D3 code path
            lb = np.packbits(land.reshape(nw, -1), axis=-1)[win_row]
            full, land_a, two = {}, {}, True
            for m_ in ("s1", "s2"):
                f_, a_, b_ = P41.land_counts(maps[m_][:, :, TB0:TB0 + YB], lb)
                full[m_], land_a[m_] = f_, a_
                two &= bool(np.array_equal(a_, b_))
            del maps
            pool = (lambda a1, a2: np.concatenate([a1, a2], 1)) if broken() == "pool100" else P3.pooled_of
            full["pooled"], land_a["pooled"] = pool(full["s1"], full["s2"]), pool(land_a["s1"], land_a["s2"])
            raw_ok = all(np.array_equal(full[m_].mean(1), z3[f"raw_{m_}"]) for m_ in METH)
            allland = nl == CELLS
            same_all = all(np.array_equal(land_a[m_][allland], full[m_][allland]) and bool((land_a[m_] <= full[m_]).all()) for m_ in METH)
            f5g = g5a_as_p4(f5, methods)
            dmax = 0.0
            for m_ in METH:
                a = P41.apply_land(f5g, key0[m_], full[m_], rec_t, rcls, rbin)
                mc_, Lc, Uc = P3.cap400(a["aoo"])
                P6 = P3.six_p(a["p_b"], a["aoo"]["scaled"], rec_t)
                for q, v in (("mean", mc_), ("L", Lc), ("U", Uc), ("P", P6), ("p_b", a["p_b"]), ("p_g3", a["p_g3"]), ("p_recal", a["p_recal"])):
                    dmax = max(dmax, maxdiff(v, z3[f"{q}_{m_}"]))
            fits_ok = fits4["methods"] == methods and fits4["crossed"] == "empty" and fits4["p4_windows_md5"] == mw
            gate("D3", raw_ok and two and same_all and dmax <= TOL and fits_ok,
                 f"window AOO of every map, mean over maps = P3's raw (pooled, seed 1, seed 2) {raw_ok}; land-only AOO by two "
                 f"paths equal {two}; = window AOO in all-land windows ({int(allland.sum()):,} rows) and never above it "
                 f"{same_all}; apply_land with the G5a fits on window AOO, P3's 400 cap and six_p against P3's products, largest "
                 f"difference {dmax:.1e}; P4 fits: six methods, crossed rule empty, p4_windows md5 stored {fits_ok}")
            if not res["D3"]:
                return stop("D3")
            # ================================================================ products
            use = f5g if broken() == "g5a_fits" else fits4
            R = {m_: products(use, key[m_], land_a[m_], rec_t, rcls, rbin, nl, above_land) for m_ in METH}
            h6r = {m_: P3.h6_rows(r6, m_) for m_ in METH}
            lab = np.array([TCLS[c] for c in rcls])
            H6 = {m_: (np.array([h6r[m_][g]["ratio"] if g in h6r[m_] else np.nan for g in lab]),
                       np.array([h6r[m_][g]["cov"] if g in h6r[m_] else np.nan for g in lab])) for m_ in METH}
            notes = np.array([note_of(rcls, R, i) for i in range(n)], dtype=object)
            wtype = np.where(allland, "all land", "coastal")
            # ================================================================ D4 product rules (in memory)
            shp = land_a["pooled"].shape == (n, 2 * HALF) and np.array_equal(land_a["pooled"][:, :HALF], land_a["s1"][:, :HALF]) \
                and np.array_equal(land_a["pooled"][:, HALF:], land_a["s2"][:, :HALF])
            rules, xs6 = {}, np.array(XS6)
            for m_ in METH:
                r = R[m_]; c = r["crossed"]; o = ~c; P, pb, P6 = r["P"], r["p_b"], r["P6"]
                rules[m_] = dict(
                    interval=bool((r["L"][o] >= rec_t[o]).all() and (r["U"][o] >= r["L"][o]).all() and (r["U"][o] <= nl[o]).all()
                                  and np.isnan(r["L"][c]).all() and np.isnan(r["U"][c]).all()
                                  and np.array_equal(c, r["U_unc"] < r["L_unc"]) and int(c.sum()) == r["inverted"]),
                    cap=bool(np.isfinite(r["mean"]).all() and (r["mean"] <= nl).all() and (r["mean"] >= 0).all()),
                    in01=bool(((P >= 0) & (P <= 1)).all()), rising=bool((np.diff(P, axis=1) >= 0).all()),
                    zeros=bool((P[rec_t[:, None] > xs6[None, :]] == 0).all()),
                    option_b=bool(np.array_equal(P6[:, [0, 2]], pb[:, :2]) and np.array_equal(P6[:, 3], np.where(rec_t > 10, 0.0, pb[:, 2]))
                                  and np.array_equal(P6[:, 4], np.where(rec_t > 20, 0.0, pb[:, 3]))),
                    above=bool(np.array_equal(P, np.where(r["above"], 1.0, P6) if above_land == "one" else P6)))
            d4_mem = shp and all(all(v.values()) for v in rules.values())
            if broken() == "opentruth":
                np.load(bnm / "bnm_e0000_n0000_truth.npz")
            crashed = None
        except RuntimeError as e:
            if not lg.refused:
                raise
            crashed = str(e)
    if crashed:
        gate("D5", False, f"np.load refused: {crashed[:160]}")
        return stop("D5")
    if not d4_mem:
        gate("D4", False, f"pooled = first 25 maps of each seed {shp}; rules per method {rules}")
        return stop("D4")
    # ================================================================ reports (not tests)
    coast_w = nl_w < CELLS
    put_back = (~land_lcm & surv).reshape(nw, -1).sum(1)
    out(f"   windows: {nw}; L_w all {CELLS} tetrads in {int((~coast_w).sum())}, coastal {int(coast_w.sum())}; tetrads no land by "
        f"the map {int((~land_lcm).sum()):,}, put back as land by effort {int(put_back.sum()):,}, outside L_w {int((~land).sum()):,}; "
        f"1 km cells outside the map {outside_map}; rows in coastal windows {int((~allland).sum()):,} of {n:,}")
    xc = {m_: {"all": int(R[m_]["crossed"].sum()), "by_class": {TCLS[c]: int((R[m_]["crossed"] & (rcls == c)).sum())
                                                               for c in range(len(TCLS))}} for m_ in METH}
    out("   crossed intervals (no interval, rule E) by recorded class: " + "; ".join(
        f"{m_} {v['all']} (" + ", ".join(f"{k_} {c_}" for k_, c_ in v["by_class"].items() if c_) + ")" for m_, v in xc.items()))
    over = {m_: int(R[m_]["capped"].sum()) for m_ in METH}
    flag = {m_: {str(x): int(((R[m_]["p_b"][:, j] > 0) & (rec_t > x)).sum()) for j, x in ((2, 10), (3, 20))} for m_ in METH}
    out(f"   rows whose corrected mean or interval end was capped at |L_w|: " + "; ".join(f"{m_} {over[m_]:,}" for m_ in METH)
        + "; lower end raised to the recorded tetrads: " + "; ".join(f"{m_} {R[m_]['raised'].mean():.3f}" for m_ in METH)
        + "; option B values above 0 set to 0 above the recorded tetrads: " + "; ".join(
            f"{m_} " + ", ".join(f"x {x} {v:,}" for x, v in flag[m_].items()) for m_ in METH))
    abv = {m_: {str(x): dict(rows=int(R[m_]["above"][:, j].sum()), p_below_1=int((R[m_]["above"][:, j] & (R[m_]["P6"][:, j] < 1)).sum()),
                             set_to_1=int(R[m_]["set_one"][:, j].sum())) for j, x in enumerate(XS6) if R[m_]["above"][:, j].any()} for m_ in METH}
    out(f"   x at or above |L_w| (rule {above_land}): " + ("; ".join(
        f"{m_} " + ", ".join(f"x {x}: {v['rows']} rows, six_p below 1 in {v['p_below_1']}, set to 1 {v['set_to_1']}" for x, v in abv[m_].items())
        for m_ in METH) if n_ab else "none"))
    stab = {str(x): dict(mean_abs=float(np.abs(R["s1"]["P"][:, j] - R["s2"]["P"][:, j]).mean()),
                         over_010=float((np.abs(R["s1"]["P"][:, j] - R["s2"]["P"][:, j]) > 0.10).mean())) for j, x in enumerate(XS6)}
    rat = R["s1"]["mean"] / np.maximum(R["s2"]["mean"], 1e-12)
    out("   seed 1 against seed 2 (stability, report): mean |P1 - P2| / share over 0.10: " + "; ".join(
        f"x {x} {v['mean_abs']:.3f} / {v['over_010']:.3f}" for x, v in stab.items()) + f"; corrected mean seed 1 / seed 2, "
        f"median {np.median(rat):.3f} (5-95%: {np.percentile(rat, 5):.3f}-{np.percentile(rat, 95):.3f})")
    p3m = z3["mean_pooled"]
    r3p = R["pooled"]["mean"] / np.maximum(p3m, 1e-12)
    rl = R["pooled"]["mean"] / np.maximum(zc["aoo_land_only_pooled"], 1e-12) if "aoo_land_only_pooled" in zc.files else np.full(n, np.nan)
    comp = {wt: dict(rows=int((wtype == wt).sum()), p4_over_p3_median=float(np.median(r3p[wtype == wt])) if (wtype == wt).any() else None,
                     p4_over_p3c_line_median=float(np.median(rl[wtype == wt])) if (wtype == wt).any() else None)
            for wt in ("all land", "coastal")}
    out("   pooled land-only mean against P3's pooled mean (window AOO) and P3c's land line, median ratio: " + "; ".join(
        f"{wt} ({v['rows']:,} rows) {v['p4_over_p3_median']:.3f} and {v['p4_over_p3c_line_median']:.3f}" for wt, v in comp.items() if v["rows"]))
    nmed = lambda v: float(np.median(v[~np.isnan(v)])) if (~np.isnan(v)).any() else float("nan")     # crossed rows have no ends
    summ = {}
    for gname, gv, labs in (("recorded class", lab, TCLS), ("Red List 2022", status, sorted(set(status.tolist()))),
                            ("window", wtype, ["all land", "coastal"])):
        summ[gname] = {}
        for g_ in labs:
            m = gv == g_
            if not m.any():
                continue
            P = R["pooled"]["P"][m]
            summ[gname][g_] = dict(rows=int(m.sum()), rec_2019_median=float(np.median(rec_t[m])),
                                   aoo_median=float(np.median(R["pooled"]["mean"][m])), lo_median=nmed(R["pooled"]["L"][m]),
                                   hi_median=nmed(R["pooled"]["U"][m]), p4_over_p3_median=float(np.median(r3p[m])),
                                   **{f"mean_p_{x}": float(P[:, j].mean()) for j, x in enumerate(XS6)},
                                   **{f"share_p_{x}_over_half": float((P[:, j] >= 0.5).mean()) for j, x in enumerate(XS6) if x in (2, 124)})
        out(f"   pooled land-only products by {gname} (rows; median 2019 records; median corrected AOO [lower, upper]; median "
            f"ratio to P3; mean P at x 2 | 124; share with P >= 0.5 at x 2 | 124):")
        for g_, v in summ[gname].items():
            out(f"      {g_:<8} {v['rows']:>5,}; {v['rec_2019_median']:.0f}; {v['aoo_median']:.1f} [{v['lo_median']:.1f}, "
                f"{v['hi_median']:.1f}]; {v['p4_over_p3_median']:.3f}; {v['mean_p_2']:.3f} | {v['mean_p_124']:.3f}; "
                f"{v['share_p_2_over_half']:.3f} | {v['share_p_124_over_half']:.3f}")
    # ================================================================ files
    od.mkdir(parents=True)
    sav = dict(window=win_s, half=half_s, species=idx["species"].astype(str), status=status, rec_2019=rec_t.astype(np.int16),
               rec_6y=rec6.astype(np.int16), rcls=rcls.astype(np.int8), rbin=rbin.astype(np.int8), land_n=nl.astype(np.int16),
               window_type=wtype, xs6=xs6, p3_mean_pooled=p3m, window_names=np.array(wins), land=land, land_lcm=land_lcm,
               surveyed=surv, above_land=np.array(above_land), frozen_md5=np.array(fmd5), p4_fits_md5=np.array(mf4),
               p3_products_md5=np.array(mp3), p2_report_md5=np.array(m2), h6_report_md5=np.array(m6), note=notes.astype(str))
    for m_ in METH:
        for q in ("raw", "mean", "L", "U", "L_unc", "U_unc", "crossed", "capped", "P", "p_b", "p_g3", "p_recal"):
            sav[f"{q}_{m_}"] = R[m_][q]
        sav[f"h6_ratio_{m_}"], sav[f"h6_cov_{m_}"] = H6[m_]
    X.save_npz(od / "products_land_bnm.npz", sav)
    head = ["window", "half", "species", "red_list_2022", "rec_2019", "rec_6y", "recorded_class", "records_bin", "land_tetrads",
            "window_type"]
    for m_ in METH:
        head += [f"aoo_mean_{m_}", f"aoo_lo90_{m_}", f"aoo_hi90_{m_}"] + [f"p_aoo_le_{x}_{m_}" for x in XS6]
    head += ["aoo_mean_pooled_p3"] + [f"h6_{q}_{m_}" for m_ in METH for q in ("ratio", "cov")] + ["note"]
    wcols = list(METH) if broken() != "csv_swap" else ["pooled", "s2", "s1"]
    f3 = lambda v: "" if np.isnan(v) else f"{v:.3f}"
    with open(od / "products_land_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for i in range(n):
            row = [win_s[i], half_s[i], sav["species"][i], status[i], int(rec_t[i]), int(rec6[i]), TCLS[rcls[i]], BINS[rbin[i]],
                   int(nl[i]), wtype[i]]
            for m_ in wcols:
                row += [f3(R[m_][q][i]) for q in ("mean", "L", "U")] + [f"{v:.4f}" for v in R[m_]["P"][i]]
            row += [f"{p3m[i]:.3f}"] + [f"{H6[m_][q][i]:.3f}" for m_ in METH for q in (0, 1)] + [notes[i]]
            w.writerow(row)
    # ================================================================ D4 read-back
    zb = np.load(od / "products_land_bnm.npz")
    back_npz = set(zb.files) == set(sav) and all(P3.same(zb[k_], v) for k_, v in sav.items())
    with open(od / "products_land_bnm.csv") as fh:
        rows_c = list(csv.DictReader(fh))
    col = lambda c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows_c])
    back_csv = len(rows_c) == n and [r["species"] for r in rows_c] == list(sav["species"]) and \
        [r["recorded_class"] for r in rows_c] == list(lab) and P3.same(col("land_tetrads"), nl) and \
        all(P3.same(col(f"p_aoo_le_{x}_{m_}"), R[m_]["P"][:, j], 5.01e-5) for m_ in METH for j, x in enumerate(XS6)) and \
        all(P3.same(col(f"aoo_{c}_{m_}"), R[m_][q], 5.01e-4) for m_ in METH for c, q in (("mean", "mean"), ("lo90", "L"), ("hi90", "U"))) \
        and P3.same(col("aoo_mean_pooled_p3"), p3m, 5.01e-4)
    h6back = len(rows_c) == n and all(
        P3.same(col(f"h6_{q}_{m_}"), [h6r[m_][r["recorded_class"]][q] if r["recorded_class"] in h6r[m_] else np.nan for r in rows_c], 5.01e-4)
        for m_ in METH for q in ("ratio", "cov"))
    notes_ok = len(rows_c) == n and all(
        (NOTE_1CELL in r["note"]) == (r["recorded_class"] in ("0", "1"))
        and all((f"{NOTE_CROSSED}, {m_}" in r["note"]) == bool(R[m_]["crossed"][i]) for m_ in METH)
        and all((r[f"aoo_lo90_{m_}"] == "") == bool(R[m_]["crossed"][i]) for m_ in METH) for i, r in enumerate(rows_c))
    gate("D4", d4_mem and back_npz and back_csv and h6back and notes_ok,
         f"pooled = first 25 maps of each seed {shp}; per method (interval order and floor, crossed rows blank and = apply_land's "
         f"count, mean and ends <= |L_w|, P in [0, 1], rising, 0 below the recorded tetrads, x 1 and 5 = option B, x 10 and 20 = "
         f"option B or 0, rule {above_land}): " + "; ".join(f"{m_} {all(v.values())}" for m_, v in rules.items())
         + f"; products_land_bnm.npz read back = memory {back_npz}; products_land_bnm.csv read back = memory (rounding) "
         f"{back_csv}; H-R2 columns = h6_report.json {h6back}; notes and blank crossed intervals {notes_ok}")
    # ================================================================ D5 load log
    allowed = [bnm, p2, p3, p3c, p4_0, p4_1, fits_dir, pooled, *[d for _, d in ydirs]]
    other = [p for p in lg.paths if not any(Path(p).resolve().parent == a.resolve() for a in allowed)]
    gate("D5", not lg.refused and not other, f"np.load refused {len(lg.refused)}; files passed to np.load {len(lg.paths)}, outside "
         f"the input folders {len(other)}")
    rep.update(gates=res, n=int(n), label="P4-3 land-only products: a new version of the P3 products; P3 stays as signed",
               note="products, not a test: true AOO is unknown on BNM; nothing is scored", above_land=above_land,
               rules=dict(land="L_w = P3c land (LCM2019) plus every tetrad with effort in the 6 years for any species row of the "
                                "window (the P4-0 rule, all 101 windows)", fits="p4_fits.npz applied by P4-1's apply_land, rule E",
                          six_p="P3's six_p unchanged (P = 0 above the recorded tetrads; x 2 and 124 from p_le)",
                          cap="corrected mean and interval ends at most |L_w|", crossed=f"interval blank, note '{NOTE_CROSSED}'",
                          above_land=ABOVE[above_land], note_1cell=NOTE_1CELL,
                          h6_columns="H6's test-half H-R2 ratio and coverage of the recorded class, as in P3 (H6 not reread)"),
               md5=dict(tool=me, tools=tmd5, p1_report=m1, p2_report=m2, p3_report=m3, p3_products=mp3, p3c_report=mc,
                        land_bnm=mlb, lcm=lcm_md5, p4_0_report=m40, p4_windows=mw, p4_1_report=m41, p4_fits=mf4, g5a_fits=mg5,
                        h6_report=m6, frozen=fmd5, out_npz=X.file_md5(od / "products_land_bnm.npz"),
                        out_csv=X.file_md5(od / "products_land_bnm.csv")),
               crossed=xc, capped=over, zeroed=flag, above=abv, raised={m_: float(R[m_]["raised"].mean()) for m_ in METH},
               seed_stability=stab, seed_mean_ratio=dict(median=float(np.median(rat)), p5=float(np.percentile(rat, 5)),
                                                         p95=float(np.percentile(rat, 95))),
               against_p3=comp, summaries=summ, grid_shares=sh,
               windows=[dict(name=wins[w_], half=str(half_s[win_s == wins[w_]][0]), land=int(nl_w[w_]),
                             land_lcm=int(land_lcm[w_].sum()), put_back=int(put_back[w_])) for w_ in range(nw)],
               minutes=round((time.time() - t0) / 60, 2), written_utc=P3.now())
    (od / "p4_3_report.json").write_text(json.dumps(rep, indent=1, default=float))
    n_ok = sum(res.values())
    out(f"   written products_land_bnm.csv {rep['md5']['out_csv']}, products_land_bnm.npz {rep['md5']['out_npz']}, p4_3_report.json "
        f"{X.file_md5(od / 'p4_3_report.json')} ({(time.time() - t0) / 60:.1f} min)")
    out(f"stage P4-3: gates {n_ok} PASS, {len(res) - n_ok} FAIL; {n:,} species-windows in {nw} windows ({int(coast_w.sum())} "
        f"coastal); crossed intervals pooled {xc['pooled']['all']}, seed 1 {xc['s1']['all']}, seed 2 {xc['s2']['all']}; rule "
        f"{above_land}; products, not a test")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--above-land", default="stop", choices=list(ABOVE))
    ap.add_argument("--bnm", default="../data/bnm_all")
    ap.add_argument("--p2", default="../data/stage_p2_predict")
    ap.add_argument("--p3", default="../data/stage_p3_products")
    ap.add_argument("--p3c", default="../data/stage_p3c_land")
    ap.add_argument("--p4-0", default="../data/stage_p4_0")
    ap.add_argument("--p4-1", default="../data/stage_p4_1")
    ap.add_argument("--fits", default="../data/stage_g5_fits")
    ap.add_argument("--h6", default="../data/stage_h6_test_score/h6_report.json")
    ap.add_argument("--lcm", default="../data/lcm2019_1km/data")
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--seeds", nargs=2, default=["f_seed1", "f_seed2"])
    ap.add_argument("--out", default="../data/stage_p4_3_products")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_33_p4_3_land_products.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_33_p4_3_land_products", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.main(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.bnm, a.p2, a.p3, a.p3c, a.p4_0, a.p4_1, a.fits, a.h6, a.lcm, a.peryear, a.seeds, a.out, a.above_land,
               out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)