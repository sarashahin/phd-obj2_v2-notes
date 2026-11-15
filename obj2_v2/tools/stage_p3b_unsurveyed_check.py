#!/usr/bin/env python3
"""
stage_p3b_unsurveyed_check.py  (obj2_v2, products step P3b, 13 Nov 2026; signed by Sara on 13 Nov after the P3 result:
a report-only check, the products stay as signed, the check's numbers are stated as a limit)

Question: how much of each P3 window AOO lies in tetrads where no butterfly survey passed the effort rule in any year of
2014-2019? The model cannot tell such a tetrad (unsurveyed land, or sea in a coastal window) from a surveyed tetrad
without records: both have zero effort and no record. Report only: nothing is scored, nothing is changed, no truth file
is opened.

Definitions (signed 13 Nov; written before the run)
    Surveyed tetrads of a window: tetrads with effort in any of the 6 years for any species row of the window in the P1
        input files (every species with a record in the window in 2014-2019; effort = the BNM effort rule, list length 8+
        in the species' flight months). Never-surveyed = the other tetrads of the 400. The union covers the species
        recorded in the window; a qualifying visit only in months when none of them flies would be missed.
    Share of a species-window: of the occupied tetrads of its maps at 2019 (summed over the maps), the share in
        never-surveyed tetrads. Pooled (first 25 maps of each seed, the product) and each seed beside it.
    Report-only line, labelled: AOO over surveyed tetrads = the row's G5a scale (pooled AOO fit, the scale P3 used) x the
        mean number of occupied surveyed tetrads of the pooled maps, at most the window's surveyed tetrads. No interval
        and no P is made for it; it assumes the G5a scale holds for the surveyed part of a window.

Gates
    D0 inputs: P1 h1_report.json, P2 h2a_report.json and P3 p3_report.json md5 = Sara's runs; P1 gates PASS, files = its
       report, input files = manifest; P2 half all, gates PASS, truth opened 0, files = its list, frozen md5 = P1; P3 gates
       PASS, products files, fits, P1, P2 and frozen md5 = its record; stage_p3_bnm_products.py and the three G5a apply
       files = the md5s P3 logged
    D1 alignment: products rows = P1 frozen rows (window, half, species, recorded tetrads, classes); 2019 records recounted
       from the input files; samples in frozen order (layout, K 50, frozen md5); surveyed tetrads per window by two paths
       (per frozen row through load_obs, and per input file) equal; frozen rows per window = manifest stacks; every record
       lies in a surveyed tetrad
    D2 code: AOO of every pooled map recounted = P3's raw mean; the G5a scale x that mean, capped at 400, = P3's corrected
       mean (1e-9); the share by vectorised code = by plain loop code on 200 rows
    D3 outputs: shares in [0, 1] or undefined only where a row's maps are all empty; surveyed + never-surveyed = 400;
       occupied surveyed + never-surveyed = occupied; the surveyed-only line <= the row's surveyed tetrads and <= P3's
       uncapped mean; files read back = memory
    D4 no *_truth file and no real_eval path passed to np.load; every file loaded is an input listed above
    Refused: an output folder that exists, lies inside an input folder, or has real_eval or _truth in its path.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_p3b_unsurveyed_check.py --self-test | tail -1     (needs ../tests/test_28_p3b_unsurveyed.py
                                                                             and ../tests/test_27_p3_products.py)
    python3 -u ../tools/stage_p3b_unsurveyed_check.py > ../logs/stage_p3b_unsurveyed.txt 2>&1
Needs stage_p3_bnm_products.py and the tools it loads.
"""
import argparse, csv, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_p3b_unsurveyed_check.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P3 = _load("stage_p3_bnm_products")
G5F, GC, GB, GS, D, X = P3.G5F, P3.GC, P3.GB, P3.GS, P3.D, P3.X
G, W, CELLS, K, HALF, TCLS = P3.G, P3.W, P3.CELLS, P3.K, P3.HALF, P3.TCLS
METH = ("pooled", "s1", "s2")
EXPECT = dict(p1_report=P3.EXPECT["p1_report"], p2_report=P3.EXPECT["p2_report"],
              p3_report="517c98e52a611d92e1e69a866eed8c1e")          # P3 p3_report.json, Sara's run 13 Nov


def broken():
    return os.environ.get("STAGE_P3B_BROKEN", "")   # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def en(nm):
    s = Path(str(nm)).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])


def occ_counts(maps, unsurveyed, ch=256):
    """Per map at 2019: occupied tetrads, and occupied never-surveyed tetrads. maps (n, k, 300), unsurveyed (n, G, G)."""
    n, k = maps.shape[:2]
    tot, inu = np.zeros((n, k), np.int64), np.zeros((n, k), np.int64)
    for a in range(0, n, ch):
        occ = GC.target_year(maps[a:a + ch])
        tot[a:a + ch] = occ.sum((2, 3))
        inu[a:a + ch] = (occ & unsurveyed[a:a + ch, None]).sum((2, 3))
    return tot, inu


def share_of(tot, inu):
    """Share of the occupied tetrads (summed over the maps) that lies in never-surveyed tetrads; NaN if all maps empty."""
    t, u = tot.sum(1).astype(float), inu.sum(1).astype(float)
    if broken() == "share_surveyed":
        u = t - u
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(t > 0, u / np.where(t > 0, t, 1), np.nan)


def loop_share(maps_row, unsurv_row):
    """Plain loop code for one row (D2): occupied cells as sets."""
    t = u = 0
    for k in range(maps_row.shape[0]):
        bits = np.unpackbits(maps_row[k])[(W - 1) * CELLS:W * CELLS]
        for c in np.nonzero(bits)[0]:
            t += 1
            u += int(unsurv_row[c // G, c % G])
    return (u / t) if t else float("nan")


def same(a, b, tol=0.0):
    return P3.same(a, b, tol)


# ---------------------------------------------------------------- run
def run(bnm="../data/bnm_all", p2="../data/stage_p2_predict", p3="../data/stage_p3_products", fits_dir="../data/stage_g5_fits",
        runs=("../runs/f_seed1", "../runs/f_seed2"), out_dir="../data/stage_p3b_unsurveyed", expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    bnm, p2, p3, fits_dir, od = map(Path, (bnm, p2, p3, fits_dir, out_dir))
    runs = [Path(r) for r in runs]
    if od.exists():
        sys.exit(f"STOP: {od} exists; P3b writes into a new folder")
    if "real_eval" in str(od.resolve()) or "_truth" in str(od.resolve()):
        sys.exit("STOP: the output path may not contain real_eval or _truth")
    for p_ in (bnm, p2, p3, fits_dir):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    need = [bnm / f for f in ("h1_report.json", "bnm_frozen.npz", "bnm_stacks.npz", "manifest.csv")] + \
        [p2 / "h2a_report.json", p3 / "p3_report.json", p3 / "products_bnm.npz", p3 / "products_bnm.csv", fits_dir / "g5_fits.npz"]
    miss = [str(f) for f in need if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    t0, res, rep = time.time(), {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage P3b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, rep
    tnames = ("stage_p3_bnm_products.py", "stage_g5_fits.py", "stage_g_calibrate.py", "stage_g_recalibrate.py")
    tmd5 = {nm: X.file_md5(HERE / nm) for nm in tnames}
    out(f"stage_p3b_unsurveyed_check.py md5 {X.file_md5(Path(__file__))}; " + ", ".join(f"{k} {v}" for k, v in tmd5.items()))
    with P3.LoadGuard() as lg:
        try:
            # ================================================================ D0 inputs
            js1, m1 = json.loads((bnm / "h1_report.json").read_text()), X.file_md5(bnm / "h1_report.json")
            fmd5, smd5, mmd5 = (X.file_md5(bnm / f) for f in ("bnm_frozen.npz", "bnm_stacks.npz", "manifest.csv"))
            man_rows = D.rows_of(bnm)
            bad_in = [r["input_file"] for r in man_rows if X.file_md5(bnm / r["input_file"]) != r["input_md5"]]
            p1_ok = m1 == expect["p1_report"] and js1["half"] == "all" and all(js1["gates"].values()) and \
                js1["md5"]["bnm_frozen.npz"] == fmd5 and js1["md5"]["bnm_stacks.npz"] == smd5 and js1["md5"]["manifest.csv"] == mmd5 and not bad_in
            ha, m2 = json.loads((p2 / "h2a_report.json").read_text()), X.file_md5(p2 / "h2a_report.json")
            fbad = [f for f, m in ha["files"].items() if not (p2 / f).is_file() or X.file_md5(p2 / f) != m]
            p2_ok = m2 == expect["p2_report"] and ha["half"] == "all" and all(ha["gates"].values()) and ha["truth_opened"] == 0 \
                and not fbad and ha["frozen_md5"] == fmd5 and all(f"samples_{r.name}.npz" in ha["files"] for r in runs)
            r3, m3 = json.loads((p3 / "p3_report.json").read_text()), X.file_md5(p3 / "p3_report.json")
            mn, mc, mf = X.file_md5(p3 / "products_bnm.npz"), X.file_md5(p3 / "products_bnm.csv"), X.file_md5(fits_dir / "g5_fits.npz")
            p3_ok = m3 == expect["p3_report"] and all(r3["gates"].values()) and r3["md5"]["products_npz"] == mn and \
                r3["md5"]["products_csv"] == mc and r3["md5"]["fits"] == mf and r3["md5"]["p1_report"] == m1 and \
                r3["md5"]["p2_report"] == m2 and r3["md5"]["frozen"] == fmd5
            code_ok = r3["md5"]["tool"] == tmd5["stage_p3_bnm_products.py"] and all(r3["md5"]["tools"][k] == tmd5[k] for k in tnames[1:])
            gate("D0", p1_ok and p2_ok and p3_ok and code_ok,
                 f"P1 h1_report.json {m1[:8]} = Sara's run {m1 == expect['p1_report']}, gates PASS, files = report, input files = "
                 f"manifest (differ {len(bad_in)}) {p1_ok}; P2 h2a_report.json {m2[:8]} = Sara's run {m2 == expect['p2_report']}, gates "
                 f"PASS, truth opened {ha['truth_opened']}, files = list (differ {len(fbad)}), frozen md5 = P1 {p2_ok}; P3 p3_report.json "
                 f"{m3[:8]} = Sara's run {m3 == expect['p3_report']}, gates PASS, products npz {mn[:8]} and csv {mc[:8]}, fits {mf[:8]}, "
                 f"P1, P2 and frozen md5 = its record {p3_ok}; stage_p3_bnm_products.py and the G5a apply files = the md5s P3 logged {code_ok}")
            if not res["D0"]:
                return stop("D0")
            # ================================================================ D1 alignment
            fb = dict(np.load(bnm / "bnm_frozen.npz")); n = len(fb["row"])
            idx = dict(np.load(bnm / "bnm_stacks.npz"))
            z3 = dict(np.load(p3 / "products_bnm.npz"))
            V1, V2, Ef, _, _ = GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])
            V = V1 | V2
            del V1, V2
            rec_t = V[:, -1].reshape(n, -1).sum(1)
            rcls, rbin = D.tidx(rec_t), D.bidx(V.any(1).reshape(n, -1).sum(1))
            win = idx["window"].astype(str)
            al = all(len(z3[k]) == n for k in ("window", "species", "rec_2019")) and np.array_equal(z3["window"].astype(str), win) \
                and np.array_equal(z3["species"].astype(str), idx["species"].astype(str)) and np.array_equal(z3["half"].astype(str), idx["half"].astype(str)) \
                and np.array_equal(z3["rec_2019"], rec_t) and np.array_equal(rec_t, fb["rec_t"]) and np.array_equal(z3["rcls"], rcls) \
                and np.array_equal(z3["rbin"], rbin)
            # surveyed tetrads per window, path A: per frozen row (load_obs), path B: per input file
            surv_row_eff = Ef.any(1)                                  # (n, G, G): this species' effort in any year
            if broken() == "one_year":
                surv_row_eff = Ef[:, -1]
            wins = sorted(set(win.tolist()))
            survA = {w_: np.logical_or.reduce(surv_row_eff[win == w_], axis=0) for w_ in wins}
            byfile = {r["window"]: r for r in man_rows}
            survB, stacks_ok = {}, True
            for w_ in wins:
                r = byfile[w_]
                with np.load(bnm / r["input_file"]) as zi:
                    E = X.unpack(zi["effort"])                         # (S, W, G, G)
                survB[w_] = E.any((0, 1))
                stacks_ok &= int(r["stacks"]) == int((win == w_).sum()) == len(E)
            two = all(np.array_equal(survA[w_], survB[w_]) for w_ in wins)
            S_row = np.stack([survA[w_] for w_ in win])                # (n, G, G) surveyed tetrads of the row's window
            rec_in = int((V.any(1) & ~S_row).sum()) == 0
            del Ef
            maps, why, ok1 = {}, [], True
            for m_, r in zip(("s1", "s2"), runs):
                z = np.load(p2 / f"samples_{r.name}.npz")
                good = np.array_equal(z["sel"], np.arange(n)) and str(z["layout"]) == GS.LAYOUT and str(z["frozen_md5"]) == fmd5 \
                    and int(z["K"]) == K and z["maps"].shape == (n, K, W * CELLS // 8)
                ok1 &= good; why.append(f"{r.name} {good}")
                maps[m_] = z["maps"]
            gate("D1", al and two and stacks_ok and rec_in and ok1,
                 f"{n:,} products rows = P1 frozen rows (window, half, species, recorded tetrads, classes; 2019 records recounted from "
                 f"the input files) {al}; surveyed tetrads of {len(wins)} windows by two paths equal {two}; frozen rows per window = "
                 f"manifest stacks {stacks_ok}; every record in a surveyed tetrad {rec_in}; samples in frozen order: {', '.join(why)}")
            if not res["D1"]:
                return stop("D1")
            # ================================================================ counts
            U_row = ~S_row
            cnt = {m_: occ_counts(maps[m_], U_row) for m_ in ("s1", "s2")}
            cnt["pooled"] = tuple(np.concatenate([cnt["s1"][q][:, :HALF], cnt["s2"][q][:, :HALF]], 1) for q in (0, 1))
            share = {m_: share_of(*cnt[m_]) for m_ in METH}
            tot_p, inu_p = cnt["pooled"]
            raw = tot_p.mean(1)
            fits = G5F.load_fits(fits_dir / "g5_fits.npz")
            tab = fits["corr"][("pooled" if broken() != "scale_s1" else f"model_{runs[0].name}", "aoo")]
            s_row = tab["scale"][G5F.group_of(rcls, rbin, tab["option"])]
            uncapped = s_row * raw
            n_surv = S_row.reshape(n, -1).sum(1)
            mean_s = (tot_p - inu_p).mean(1)
            aoo_s = np.minimum(s_row * mean_s, n_surv)
            # ================================================================ D2 code
            d_raw = float(np.abs(raw - z3["raw_pooled"]).max())
            d_mean = float(np.abs(np.minimum(uncapped, CELLS) - z3["mean_pooled"]).max())
            pick = np.sort(X.rng(31, 7).choice(n, min(200, n), replace=False))
            pooled_maps = lambda i: np.concatenate([maps["s1"][i, :HALF], maps["s2"][i, :HALF]], 0)
            loop = np.array([loop_share(pooled_maps(i), U_row[i]) for i in pick])
            loop_ok = same(loop, share["pooled"][pick], 1e-12)
            gate("D2", d_raw <= 1e-12 and d_mean <= 1e-9 and loop_ok,
                 f"pooled map AOO recounted = P3's raw mean (largest difference {d_raw:.1e}); G5a scale x that mean, capped at {CELLS}, "
                 f"= P3's corrected mean (largest difference {d_mean:.1e}); share by loop code on {len(pick)} rows = vectorised {loop_ok}")
            if not res["D2"]:
                return stop("D2")
            # ================================================================ D3 rules (in memory)
            empty = {m_: cnt[m_][0].sum(1) == 0 for m_ in METH}
            r_share = all(bool(((share[m_][~empty[m_]] >= 0) & (share[m_][~empty[m_]] <= 1)).all()) and
                          bool(np.isnan(share[m_][empty[m_]]).all()) and not np.isnan(share[m_][~empty[m_]]).any() for m_ in METH)
            r_sum = bool((n_surv + U_row.reshape(n, -1).sum(1) == CELLS).all()) and \
                all(bool((cnt[m_][1] <= cnt[m_][0]).all()) for m_ in METH)
            r_line = bool((aoo_s <= n_surv + 1e-9).all() and (aoo_s <= uncapped + 1e-9).all())
            if broken() == "opentruth":
                np.load(bnm / "bnm_e0000_n0000_truth.npz")
            crashed = None
        except RuntimeError as e:
            if not lg.refused:
                raise
            crashed = str(e)
    if crashed:
        gate("D4", False, f"np.load refused: {crashed[:160]}")
        return stop("D4")
    if not (r_share and r_sum and r_line):
        gate("D3", False, f"shares {r_share}; sums {r_sum}; surveyed-only line {r_line}")
        return stop("D3")
    # ================================================================ reports (not tests)
    status = z3["status"].astype(str)
    lab = np.array([TCLS[c] for c in rcls])
    aoo_p = z3["mean_pooled"]
    out(f"   never-surveyed tetrads per window (of {CELLS}): median {np.median([CELLS - survA[w_].sum() for w_ in wins]):.0f}, range "
        f"{min(CELLS - survA[w_].sum() for w_ in wins)}-{max(CELLS - survA[w_].sum() for w_ in wins)}; species-windows whose maps hold "
        f"no occupied tetrad: pooled {int(empty['pooled'].sum())}")
    q = lambda a: (float(np.nanmedian(a)), float(np.nanpercentile(a, 10)), float(np.nanpercentile(a, 90)))
    big = {t_: int((share["pooled"] > t_).sum()) for t_ in (0.25, 0.5)}
    out("   share of the occupied tetrads in never-surveyed tetrads, median [10th, 90th percentile]: " + "; ".join(
        f"{m_} {q(share[m_])[0]:.3f} [{q(share[m_])[1]:.3f}, {q(share[m_])[2]:.3f}]" for m_ in METH)
        + f"; pooled rows with a share above 0.25: {big[0.25]:,}, above 0.5: {big[0.5]:,} (of {n:,})")
    over = aoo_p > n_surv
    out(f"   rows whose P3 corrected AOO is above all the surveyed tetrads of their window: {int(over.sum()):,} of {n:,}")
    summ = {}
    for gname, gv, labs in (("recorded class", lab, TCLS), ("Red List 2022", status, sorted(set(status.tolist())))):
        summ[gname] = {}
        out(f"   by {gname} (rows; median share pooled [10th, 90th]; median P3 corrected AOO; median AOO over surveyed tetrads, report only):")
        for g_ in labs:
            m = gv == g_
            if not m.any():
                continue
            sq = q(share["pooled"][m])
            summ[gname][g_] = dict(rows=int(m.sum()), share_median=sq[0], share_p10=sq[1], share_p90=sq[2],
                                   aoo_median=float(np.median(aoo_p[m])), aoo_surveyed_median=float(np.median(aoo_s[m])),
                                   above_surveyed=int(over[m].sum()))
            v = summ[gname][g_]
            out(f"      {g_:<6} {v['rows']:>5,}; {v['share_median']:.3f} [{v['share_p10']:.3f}, {v['share_p90']:.3f}]; {v['aoo_median']:.1f}; "
                f"{v['aoo_surveyed_median']:.1f}")
    wrows = []
    for w_ in wins:
        m = win == w_
        e_, n_ = en(w_)
        ns = int(survA[w_].sum())
        wrows.append(dict(window=w_, half=str(z3["half"][m][0]), e_km=e_, n_km=n_, species=int(m.sum()), surveyed=ns,
                          never_surveyed=CELLS - ns, never_surveyed_share=(CELLS - ns) / CELLS,
                          median_share_pooled=float(np.nanmedian(share["pooled"][m])), max_aoo_p3=float(aoo_p[m].max()),
                          max_aoo_surveyed_only=float(aoo_s[m].max())))
    wq = sorted(wrows, key=lambda r: -r["never_surveyed"])
    out("   windows with the most never-surveyed tetrads (window: never-surveyed / 400; median species share; largest P3 AOO; largest "
        "AOO over surveyed tetrads): " + "; ".join(f"{r['window'][:-4]} {r['never_surveyed']}; {r['median_share_pooled']:.2f}; "
                                                    f"{r['max_aoo_p3']:.0f}; {r['max_aoo_surveyed_only']:.0f}" for r in wq[:8]))
    # ================================================================ files
    od.mkdir(parents=True)
    sav = dict(window=win, species=idx["species"].astype(str), status=status, rec_2019=rec_t.astype(np.int16), rcls=rcls.astype(np.int8),
               surveyed=n_surv.astype(np.int16), never_surveyed=(CELLS - n_surv).astype(np.int16), scale=s_row,
               mean_occupied_pooled=raw, mean_occupied_surveyed_pooled=mean_s, aoo_surveyed_only_pooled=aoo_s,
               products_md5=np.array(mn), p3_report_md5=np.array(m3))
    for m_ in METH:
        sav[f"share_{m_}"] = share[m_]
    for w_ in wins:
        sav[f"surveyed_mask__{w_}"] = survA[w_]
    X.save_npz(od / "unsurveyed_bnm.npz", sav)
    head = ["window", "half", "species", "red_list_2022", "rec_2019", "recorded_class", "surveyed_tetrads_window",
            "never_surveyed_tetrads_window"] + [f"share_never_surveyed_{m_}" for m_ in METH] + \
        ["mean_occupied_pooled", "mean_occupied_surveyed_pooled", "aoo_mean_pooled_p3", "aoo_surveyed_only_pooled_report"]
    shcols = list(METH) if broken() != "csv_swap" else ["pooled", "s2", "s1"]
    with open(od / "unsurveyed_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for i in range(n):
            w.writerow([win[i], str(z3["half"][i]), sav["species"][i], status[i], int(rec_t[i]), lab[i], int(n_surv[i]), int(CELLS - n_surv[i])]
                       + ["" if np.isnan(share[m_][i]) else f"{share[m_][i]:.4f}" for m_ in shcols]
                       + [f"{raw[i]:.3f}", f"{mean_s[i]:.3f}", f"{aoo_p[i]:.3f}", f"{aoo_s[i]:.3f}"])
    with open(od / "per_window_unsurveyed.csv", "w", newline="") as fh:
        wd = csv.DictWriter(fh, fieldnames=list(wrows[0])); wd.writeheader(); wd.writerows(wrows)
    zb = np.load(od / "unsurveyed_bnm.npz")
    back_npz = all(same(zb[k_], v) for k_, v in sav.items())
    with open(od / "unsurveyed_bnm.csv") as fh:
        rc = list(csv.DictReader(fh))
    col = lambda c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rc])
    back_csv = len(rc) == n and [r["species"] for r in rc] == list(sav["species"]) and \
        all(same(col(f"share_never_surveyed_{m_}"), share[m_], 5.01e-5) for m_ in METH) and \
        same(col("aoo_surveyed_only_pooled_report"), aoo_s, 5.01e-4) and same(col("surveyed_tetrads_window"), n_surv)
    gate("D3", r_share and r_sum and r_line and back_npz and back_csv,
         f"shares in [0, 1], undefined only for rows whose maps are all empty {r_share}; surveyed + never-surveyed = {CELLS}, occupied "
         f"parts add up {r_sum}; surveyed-only line <= the surveyed tetrads and <= P3's uncapped mean {r_line}; unsurveyed_bnm.npz "
         f"read back = memory {back_npz}; unsurveyed_bnm.csv read back = memory (rounding) {back_csv}")
    allowed = [bnm, p2, p3, fits_dir]
    other = [p for p in lg.paths if not any(Path(p).resolve().parent == a.resolve() for a in allowed)]
    gate("D4", not lg.refused and not other, f"np.load refused {len(lg.refused)}; files passed to np.load {len(lg.paths)}, other than the "
         f"P1 export, the P2 samples, the P3 products and the G5a fits {len(other)}")
    rep.update(gates=res, n=int(n), note="report only: nothing scored, nothing changed; the P3 products stay as signed",
               definitions=dict(surveyed="effort in any of 2014-2019 for any species row of the window in the P1 input files",
                                share="occupied never-surveyed tetrads / occupied tetrads, summed over the maps at 2019",
                                aoo_surveyed_only="report only: G5a scale x mean occupied surveyed tetrads (pooled), at most the "
                                                  "window's surveyed tetrads; no interval, no P; assumes the scale holds for the surveyed part"),
               md5=dict(tool=X.file_md5(Path(__file__)), tools=tmd5, p1_report=m1, p2_report=m2, p3_report=m3, products_npz=mn,
                        fits=mf, out_npz=X.file_md5(od / "unsurveyed_bnm.npz"), out_csv=X.file_md5(od / "unsurveyed_bnm.csv"),
                        per_window_csv=X.file_md5(od / "per_window_unsurveyed.csv")),
               share={m_: dict(median=q(share[m_])[0], p10=q(share[m_])[1], p90=q(share[m_])[2]) for m_ in METH},
               rows_share_above={str(k_): v for k_, v in big.items()}, above_surveyed=int(over.sum()), empty_rows={m_: int(empty[m_].sum()) for m_ in METH},
               never_surveyed_per_window=dict(median=float(np.median([r["never_surveyed"] for r in wrows])),
                                              min=int(min(r["never_surveyed"] for r in wrows)), max=int(max(r["never_surveyed"] for r in wrows))),
               summaries=summ, windows=wrows, minutes=round((time.time() - t0) / 60, 2), written_utc=now())
    (od / "p3b_report.json").write_text(json.dumps(rep, indent=1, default=float))
    n_ok = sum(res.values())
    out(f"   written unsurveyed_bnm.csv {rep['md5']['out_csv']}, unsurveyed_bnm.npz {rep['md5']['out_npz']}, per_window_unsurveyed.csv, "
        f"p3b_report.json ({(time.time() - t0) / 60:.1f} min)")
    out(f"stage P3b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; {n:,} species-windows, {len(wins)} windows; report only")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bnm", default="../data/bnm_all")
    ap.add_argument("--p2", default="../data/stage_p2_predict")
    ap.add_argument("--p3", default="../data/stage_p3_products")
    ap.add_argument("--fits", default="../data/stage_g5_fits")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--out", default="../data/stage_p3b_unsurveyed")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_28_p3b_unsurveyed.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_28_p3b_unsurveyed", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.main(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.bnm, a.p2, a.p3, a.fits, a.runs, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)