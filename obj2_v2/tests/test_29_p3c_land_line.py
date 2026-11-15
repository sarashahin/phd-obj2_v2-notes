#!/usr/bin/env python3
"""
test_29_p3c_land_line.py  (obj2_v2, products step P3c, 13 Nov 2026): self-test of tools/stage_p3c_land_line.py.
Run through the tool, from PSD_Dispersal_pool/obj2_v2/work:
    python3 ../tools/stage_p3c_land_line.py --self-test
It builds test_27's planted world (real H1, P1 and P3 tools), runs the real P3b on it, writes a planted Land Cover Map
(21-band percentage, 10-band aggregate and dominant GeoTIFFs on the British National Grid around the planted windows),
then runs P3c. Nothing of ../data or ../runs is read or written.
Planted map: every 1 km cell of a surveyed tetrad holds land; never-surveyed tetrads in the east half of a window are open
sea (no cover); in the west half some are shore only (Littoral sediment 60 + Saltwater 20: no land), some hold a sliver
(Supralittoral sediment 1 + Saltwater 50: land), the rest land.
    U1  by hand: read_map drops classes 13, 14, 17, 18 (a shore-only cell has land cover 0, a 1% Supralittoral cell 1);
        window_land puts row 0 in the south and column 0 in the west; a window half outside the map has no land there
    S0  the planted P1, P3 and P3b runs pass every gate
    T1  P3c on them: gates L0-L5 PASS
    T2  independent check, fresh code (map blocks sliced with rasterio, sample bits, g5_fits.npz with np.load): land masks,
        no-land shares, never-surveyed land shares and the land line = land_bnm.npz (1e-12), land_bnm.csv (rounding),
        per_window_land.csv
    T3  meaning: no-land tetrads per window = the planted sea and shore tetrads; no surveyed tetrad is no land; every other
        orientation of the grid puts surveyed tetrads in no land; maps at random over the window give a mean no-land share
        equal to the mean no-land fraction (within 0.02)
    T4  a second run into a new folder gives the same md5 for both csv files and the npz
    T5  refused: an existing folder, a folder inside an input folder, a real_eval path; L0 fails and nothing is written
        when unsurveyed_bnm.npz is changed after P3b
    bugs: grid read column-flipped (L2); bands 13 and 3 swapped in the map (L1); scale off by 1% (L3); land share reported
        (L3); seed columns swapped in the csv (L5); a truth file opened (L4); half-land rule (caught by a gate or by T2's fresh code)
"""
import csv, hashlib, importlib.util, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
 
T_EDGES, B_EDGES = np.array([0, 1, 2, 6, 11, 21, 51]), np.array([1, 4, 7, 11, 21])
GROUPS = [[1], [2], [3], [4], [5, 6, 7, 8], [9, 10, 11, 12], [13], [14], [15, 16, 17, 18, 19], [20, 21]]
LANDCL = [c for c in range(1, 22) if c not in (13, 14, 17, 18)]
 
 
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
 
 
def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()
 
 
def en(nm):
    s = Path(nm).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])
 
 
def write_map(d, T, west, north, swap=None):
    """Write the three GB map files from a (21, H, W) uint8 array; aggregate = group sums; dominant from 22 values
    (unmapped share as 0, then the 21 bands), as Sara's file is made."""
    import rasterio
    from rasterio.transform import from_origin
    d.mkdir(parents=True, exist_ok=True)
    A = np.stack([T[[c - 1 for c in g]].astype(int).sum(0) for g in GROUPS]).astype(np.uint8)
    V = np.vstack([np.clip(100 - T.astype(int).sum(0), 0, None)[None], T.astype(int)])
    Dm = np.where(T.astype(int).sum(0) > 0, V.argmax(0), 0).astype(np.uint8)
    Tw = T.copy()
    if swap:
        a, b = swap; Tw[[a - 1, b - 1]] = Tw[[b - 1, a - 1]]
    kw = dict(driver="GTiff", height=T.shape[1], width=T.shape[2], crs="EPSG:27700", transform=from_origin(west, north, 1000, 1000), dtype="uint8")
    for nm, arr in (("gb2019lcm1km_percentage_target.tif", Tw), ("gb2019lcm1km_percentage_aggregate.tif", A),
                    ("gb2019lcm1km_dominant_target.tif", Dm[None])):
        with rasterio.open(d / nm, "w", count=arr.shape[0], **kw) as f:
            f.write(arr)
 
 
def plant_map(zb, d, swap=None, seed=9):
    """Planted map around the planted windows; returns the planted no-land tetrads per window."""
    win = sorted(set(zb["window"].astype(str).tolist()))
    E = [en(w)[0] for w in win]; N = [en(w)[1] for w in win]
    west, north = (min(E) - 4) * 1000, (max(N) + 44) * 1000
    H, Wd = (max(N) + 44) - (min(N) - 4), (max(E) + 44) - (min(E) - 4)
    rng = np.random.default_rng(seed)
    T = np.zeros((21, H, Wd), np.uint8)
    for r in range(H):                                   # land everywhere first, largest class cycling over all 21
        for c in range(Wd):
            k1 = (r * Wd + c) % 21 + 1
            k2, k3 = rng.choice([x for x in LANDCL if x != k1], 2, replace=False)
            T[[k1 - 1, k2 - 1, k3 - 1], r, c] = (60, 30, 10)
    noland = {}
    for w in win:
        e, n = en(w); S = zb[f"surveyed_mask__{w}"].astype(bool); nl = np.zeros((20, 20), bool)
        for tr_ in range(20):
            for tc in range(20):
                if S[tr_, tc]:
                    continue
                cells = [((north - (n + 2 * tr_ + dn + 1) * 1000) // 1000, ((e + 2 * tc + de) * 1000 - west) // 1000)
                         for dn in (0, 1) for de in (0, 1)]
                if tc >= 10:                                 # open sea: no cover
                    for rr, cc in cells:
                        T[:, rr, cc] = 0
                    nl[tr_, tc] = True
                elif tr_ % 3 == 0:                           # shore only: no land
                    for rr, cc in cells:
                        T[:, rr, cc] = 0; T[17, rr, cc] = 60; T[12, rr, cc] = 20
                    nl[tr_, tc] = True
                elif tr_ % 3 == 1:                           # a sliver of land in one cell
                    for j, (rr, cc) in enumerate(cells):
                        T[:, rr, cc] = 0; T[12, rr, cc] = 50
                        if j == 0:
                            T[15, rr, cc] = 1
        noland[w] = nl
    write_map(d, T, west, north, swap)
    return noland
 
 
def indep(out_dir, p2, p3b, fits_dir, p3_dir, runs, lcm):
    """Fresh code from the files."""
    import rasterio
    zb = np.load(p3b / "unsurveyed_bnm.npz"); win = zb["window"].astype(str); n = len(win)
    with rasterio.open(lcm / "gb2019lcm1km_percentage_target.tif") as s:
        T = s.read().astype(int); west, north = s.transform.c, s.transform.f
    LC = T[[c - 1 for c in LANDCL]].sum(0)
    land = {}
    for w in sorted(set(win.tolist())):
        e, nn = en(w)
        r0, c0 = int((north - (nn + 40) * 1000) // 1000), int((e * 1000 - west) // 1000)
        blk = LC[r0:r0 + 40, c0:c0 + 40][::-1]                              # row 0 = south
        land[w] = (blk.reshape(20, 2, 20, 2) > 0).any((1, 3))
    L = np.stack([land[w] for w in win]); U = np.stack([~zb[f"surveyed_mask__{w}"].astype(bool) for w in win])
    occ = {}
    for m, r in zip(("s1", "s2"), runs):
        mp = np.load(p2 / f"samples_{Path(r).name}.npz")["maps"]
        occ[m] = np.unpackbits(mp[..., 250:], axis=-1)[..., :400].reshape(n, mp.shape[1], 20, 20).astype(bool)
    occ["pooled"] = np.concatenate([occ["s1"][:, :25], occ["s2"][:, :25]], 1)
    sh = {}
    for m in ("pooled", "s1", "s2"):
        t = occ[m].sum((1, 2, 3)).astype(float); u = (occ[m] & ~L[:, None]).sum((1, 2, 3)).astype(float)
        sh[m] = np.where(t > 0, u / np.maximum(t, 1), np.nan)
    tp = occ["pooled"].sum((1, 2, 3)).astype(float)
    nsl = np.where(tp > 0, (occ["pooled"] & (U & L)[:, None]).sum((1, 2, 3)) / np.maximum(tp, 1), np.nan)
    fb = np.load(Path(p3b).parent / "bnm_all" / "bnm_frozen.npz")
    rcls = np.searchsorted(T_EDGES, fb["rec_t"].astype(int), side="right") - 1
    rbin = np.clip(np.searchsorted(B_EDGES, fb["rec6"].astype(int), side="right") - 1, 0, 4)
    fz = np.load(fits_dir / "g5_fits.npz"); opt = int(fz["corr__pooled__aoo__option"])
    s = fz["corr__pooled__aoo__scale"][rcls if opt == 1 else rcls * 5 + rbin]
    nl_ = L.reshape(n, -1).sum(1)
    aoo_l = np.minimum(s * (occ["pooled"] & L[:, None]).sum((2, 3)).mean(1), nl_)
    z = np.load(out_dir / "land_bnm.npz")
    worst = max([float(np.nanmax(np.abs(sh[m] - z[f"share_no_land_{m}"]))) for m in sh] +
                [float(np.nanmax(np.abs(nsl - z["share_never_surveyed_land_pooled"]))), float(np.abs(aoo_l - z["aoo_land_only_pooled"]).max()),
                 float(np.abs(nl_ - z["land"]).max())] + [float((land[w] != z[f"land_mask__{w}"]).sum()) for w in land])
    with open(out_dir / "land_bnm.csv") as fh:
        rows = list(csv.DictReader(fh))
    f = lambda c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows])
    ok_csv = len(rows) == n and all(np.nanmax(np.abs(f(f"share_no_land_{m}") - sh[m])) <= 5.01e-5 for m in sh) and \
        float(np.abs(f("aoo_land_only_pooled_report") - aoo_l).max()) <= 5.01e-4
    with open(out_dir / "per_window_land.csv") as fh:
        pw = {r["window"]: r for r in csv.DictReader(fh)}
    ok_pw = set(pw) == set(land) and all(int(pw[w]["no_land"]) == int((~land[w]).sum()) for w in land)
    p3 = np.load(p3_dir / "products_bnm.npz")
    return dict(worst=worst, csv=ok_csv, pw=ok_pw, share=sh["pooled"], frac=(~L).reshape(n, -1).mean(1), land=land,
                aoo_l=aoo_l, uncapped=s * p3["raw_pooled"])
 
 
def main(tool_path, tools):
    tool_path = Path(tool_path)
    P3C = load(tool_path, "stage_p3c_land_line")
    P3B, P3 = P3C.P3B, P3C.P3
    T27 = load(tool_path.parent.parent / "tests" / "test_27_p3_products.py", "test_27_p3_products")
    P1 = load(tool_path.parent / "stage_p1_bnm_all_export.py", "stage_p1_bnm_all_export")
    H1, X = P1.H1, P3.X
    quiet = lambda *a, **k: None
    res = []
 
    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)
 
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- U1 by hand
        T = np.zeros((21, 40, 40), np.uint8)
        T[2, 39, 2] = 50                                   # cell east 2-3 km, north 0-1 km: tetrad row 0, column 1
        T[3, 0, 0] = 40                                    # cell east 0-1 km, north 39-40 km: tetrad row 19, column 0
        T[17, 20, 20] = 60; T[12, 20, 20] = 20; T[16, 20, 21] = 70; T[13, 20, 22] = 90   # shore and water only: no land
        T[15, 30, 30] = 1                                  # 1% Supralittoral sediment: land
        write_map(tmp / "u1", T, 0, 40000)
        LC, tr_, ok = P3C.read_map(tmp / "u1")
        l0, _, o0 = P3C.window_land(LC, tr_, 0, 0)
        l1, _, o1 = P3C.window_land(LC, tr_, 20, 0)
        exp = np.zeros((20, 20), bool); exp[0, 1] = exp[19, 0] = True; exp[(40 - 30 - 1) // 2, 15] = True
        t("U1", ok and LC[20, 20] == 0 and LC[20, 21] == 0 and LC[20, 22] == 0 and LC[30, 30] == 1 and np.array_equal(l0, exp)
          and o0 == 0 and o1 == 800 and not l1[:, 10:].any(),
          f"shore-only and water-only cells land cover {int(LC[20, 20])}, {int(LC[20, 21])}, {int(LC[20, 22])}; 1% Supralittoral {int(LC[30, 30])}; "
          f"land tetrads at (row, column) {[tuple(map(int, x)) for x in np.argwhere(l0)]} (expected (0, 1), (4, 15), (19, 0)); window half "
          f"outside the map: {o1} cells outside, no land there {not l1[:, 10:].any()}")
        # ---- planted pipeline
        py, refpat = H1.world(tmp / "bw")
        data = tmp / "data"
        fit, test, al = data / "bnm_fit", data / "bnm_test", data / "bnm_all"
        rf, _ = H1.run(py, refpat, tools, fit, expect=(6, 6), out=quiet)
        fzh = tmp / "fz_h1.json"
        fzh.write_text(json.dumps(dict(tools={"stage_h1_bnm_export.py": X.file_md5(P1.H1_FILE)})))
        rt_, _ = H1.run(py, refpat, tools, test, expect=(6, 6), half="test", final_test=True, freeze=str(fzh), out=quiet)
        rp, _ = P1.run(py, refpat, tools, fit, test, al, expect=(6, 6), out=quiet)
        kw, h6 = T27.plant(tmp, al, P3)
        p3d, p3b = data / "p3", data / "p3b"
        r3, _ = P3.run(out_dir=p3d, out=quiet, **kw)
        rb, _ = P3B.run(bnm=al, p2=kw["p2"], p3=p3d, fits_dir=kw["fits_dir"], runs=kw["runs"], out_dir=p3b, out=quiet,
                        expect=dict(p1_report=kw["expect"]["p1_report"], p2_report=kw["expect"]["p2_report"], p3_report=md5(p3d / "p3_report.json")))
        t("S0", all(rf.values()) and all(rt_.values()) and all(rp.values()) and all(r3.values()) and all(rb.values()),
          f"planted P1 {rp}; P3 {r3}; P3b {rb}")
        zb = dict(np.load(p3b / "unsurveyed_bnm.npz"))
        lcm = tmp / "lcm"
        planted = plant_map(zb, lcm)
        ex = dict(p3b_report=md5(p3b / "p3b_report.json"), band_check=X.file_md5(tool_path.parent / "stage_p3c_band_check.py"),
                  lcm={nm: md5(lcm / nm) for nm in ("gb2019lcm1km_percentage_target.tif", "gb2019lcm1km_percentage_aggregate.tif",
                                                    "gb2019lcm1km_dominant_target.tif")})
        kc = dict(p2=kw["p2"], p3=p3d, p3b=p3b, lcm=lcm, runs=kw["runs"], expect=ex)
        lines = []
        out1 = data / "p3c"
        r, rep = P3C.run(out_dir=out1, out=lines.append, **kc)
        t("T1", set(r) == {"L0", "L1", "L2", "L3", "L4", "L5"} and all(r.values()), f"P3c on the planted inputs: {r}")
        if not all(r.values()):
            print("      " + "\n      ".join(lines))
        ind = indep(out1, Path(kw["p2"]), p3b, Path(kw["fits_dir"]), p3d, kw["runs"], lcm) if out1.exists() else None
        t("T2", ind is not None and ind["worst"] <= 1e-12 and ind["csv"] and ind["pw"],
          f"independent recomputation from the files: npz largest difference {ind['worst'] if ind else float('nan'):.1e}; csv within "
          f"rounding {ind and ind['csv']}; per-window no-land counts {ind and ind['pw']}")
        if ind is not None:
            match = all(np.array_equal(~ind["land"][w], planted[w]) for w in planted)
            oc = rep.get("orientation_counts", {})
            ms, mf = float(np.nanmean(ind["share"])), float(ind["frac"].mean())
            below = bool((ind["aoo_l"] <= ind["uncapped"] + 1e-9).all())
        else:
            match, oc, ms, mf, below = False, {}, float("nan"), float("nan"), False
        t("T3", match and oc.get("identity") == 0 and all(v > 0 for k_, v in oc.items() if k_ != "identity") and abs(ms - mf) <= 0.02
          and 0.02 < mf < 0.5 and below,
          f"no-land tetrads = the planted sea and shore tetrads in every window {match}; surveyed tetrads in no land {oc.get('identity')}, other "
          f"orientations {[v for k_, v in oc.items() if k_ != 'identity']}; mean no-land share {ms:.3f} against the mean no-land fraction "
          f"{mf:.3f}; land line below P3's uncapped AOO {below}")
        out2 = data / "p3c_again"
        P3C.run(out_dir=out2, out=quiet, **kc)
        same = all(X.file_md5(out1 / f) == X.file_md5(out2 / f) for f in ("land_bnm.csv", "land_bnm.npz", "per_window_land.csv"))
        t("T4", same, "a second run into a new folder gives the same md5 for both csv files and the npz")
        stops = []
        for o in (out1, p3b / "inside", data / "real_eval_p3c"):
            try:
                P3C.run(out_dir=o, out=quiet, **kc); stops.append(False)
            except SystemExit:
                stops.append(True)
        p3bt = tmp / "p3b_tampered"
        shutil.copytree(p3b, p3bt)
        zz = dict(np.load(p3bt / "unsurveyed_bnm.npz")); zz["scale"] = zz["scale"] * 1.001
        X.save_npz(p3bt / "unsurveyed_bnm.npz", zz)
        rq, _ = P3C.run(out_dir=data / "p3c_tampered", out=quiet, **dict(kc, p3b=p3bt))
        t("T5", all(stops) and rq.get("L0") is False and not (data / "p3c_tampered").exists(),
          f"refused: an existing folder {stops[0]}, a folder inside an input folder {stops[1]}, a real_eval path {stops[2]}; "
          f"unsurveyed_bnm.npz changed after P3b fails L0 and nothing is written {rq}")
        lcm_sw = tmp / "lcm_swap"
        plant_map(zb, lcm_sw, swap=(13, 3))
        ex_sw = dict(ex, lcm={nm: md5(lcm_sw / nm) for nm in ex["lcm"]})
        rs, _ = P3C.run(out_dir=data / "bug_bandswap", out=quiet, **dict(kc, lcm=lcm_sw, expect=ex_sw))
        t("bug bands 13/3", rs.get("L1") is False and not (data / "bug_bandswap").exists(), f"L1 fails as it must; written "
          f"{(data / 'bug_bandswap').exists()} ({rs})")
        for bug, g_, written in (("orient", "L2", False), ("scale_bad", "L3", False), ("share_land", "L3", False),
                                 ("csv_swap", "L5", True), ("opentruth", "L4", False)):
            os.environ["STAGE_P3C_BROKEN"] = bug
            try:
                rb_, _ = P3C.run(out_dir=data / f"bug_{bug}", out=quiet, **kc)
            finally:
                os.environ.pop("STAGE_P3C_BROKEN", None)
            exd = (data / f"bug_{bug}").exists()
            t(f"bug {bug}", rb_.get(g_) is False and exd == written, f"{g_} fails as it must; written {exd} (expected {written}) ({rb_})")
        os.environ["STAGE_P3C_BROKEN"] = "half_land"
        try:
            rh, _ = P3C.run(out_dir=data / "bug_half_land", out=quiet, **kc)
        finally:
            os.environ.pop("STAGE_P3C_BROKEN", None)
        ih = indep(data / "bug_half_land", Path(kw["p2"]), p3b, Path(kw["fits_dir"]), p3d, kw["runs"], lcm) if (data / "bug_half_land").exists() else None
        t("bug half_land", (ih is None and not all(rh.values())) or (ih is not None and ih["worst"] > 0),
          f"the half-land rule is caught (gates {rh}; fresh-code largest difference {ih['worst'] if ih else float('nan'):.1e})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)
 