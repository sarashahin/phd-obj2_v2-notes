#!/usr/bin/env python3
"""stage_p3d_paper_tables.py  (obj2_v2, products step P3d, 13 Nov 2026; report only, presentation agreed by Sara on 13 Nov
after the P3c result, so every line chosen here is labelled as decided after the result)

What it writes: the paper's product table and summary table, from the csv files of P3, P3b and P3c only (no npz, no truth
file, no map). The P3 products stay as signed; nothing is scored or refitted.
    products_paper_bnm.csv  all 3,614 rows: the pooled P3 product (corrected AOO, 90% interval, P(AOO <= x) at x 1, 2, 5, 10,
                            20, 124, the H6 count ratio and coverage of the row's recorded class, the 1-cell note), and beside
                            it, report only: the share of the AOO in never-surveyed tetrads (P3b), the surveyed-only line
                            (P3b), the window's no-land tetrads, the share of the AOO in no-land tetrads and the land-only line
                            (P3c), "window_all_land" (the window holds no no-land tetrad) and "flag_aoo_above_land" (the
                            corrected AOO is larger than the window's land tetrads, values as printed in the csv files)
    summary_paper_bnm.csv   by recorded class and by GB Red List 2022, for three sets: all rows (as signed), the windows with
                            no no-land tetrad (chosen from the map alone; labelled post-result) and the windows with no-land
                            tetrads: rows; median corrected AOO, median lower and upper end; mean P(AOO <= 2) and (<= 124);
                            median share of the AOO in no-land tetrads; median land-only line; rows flagged
    p3d_report.json         md5s, gates, counts
Gates
    G0 inputs: md5 of the 7 input files = Sara's runs (P3 13 Nov, P3b 13 Nov, P3c 13 Nov)
    G1 rows: the three row files hold the same 3,614 (window, species) rows in the same order; the P3 columns copied into the
       P3b and P3c files equal products_bnm.csv; land tetrads per row = per_window_land.csv
    G2 code: the summary code on all rows reproduces p3_report.json (rows, median AOO and both ends within 0.001, mean P at
       x 2 and 124 within 1e-4, the csv rounding) and p3c_report.json (median share no land and land-only line, 0.0005)
    G3 counts: windows all land + with no land = 101, rows of both = 3,614; flagged rows recounted by a plain loop
    G4 files: output folder new and outside every input folder; both csv files read back = memory
Run (from PSD_Dispersal_pool/obj2_v2/work; seconds)
    python3 ../tools/stage_p3d_paper_tables.py --self-test
    python3 -u ../tools/stage_p3d_paper_tables.py > ../logs/stage_p3d_paper_tables.txt 2>&1
"""
import argparse, csv, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

EXPECT = {"p3/products_bnm.csv": "b409f660", "p3/p3_report.json": "517c98e5", "p3b/unsurveyed_bnm.csv": "4b6fde60",
          "p3b/p3b_report.json": "29b9c44f", "p3c/land_bnm.csv": "a569190d", "p3c/per_window_land.csv": "ba036312",
          "p3c/p3c_report.json": "6f051676"}
TCLS = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
XS = (1, 2, 5, 10, 20, 124)
P3COLS = ["window", "half", "species", "red_list_2022", "rec_2019", "rec_6y", "recorded_class", "aoo_mean_pooled",
          "aoo_lo90_pooled", "aoo_hi90_pooled"] + [f"p_aoo_le_{x}_pooled" for x in XS] + ["h6_ratio_pooled", "h6_cov_pooled", "note"]
SETS = [("all rows (as signed)", None), ("windows all land (post-result line)", True), ("windows with no-land tetrads", False)]


def broken():
    return os.environ.get("STAGE_P3D_BROKEN", "")        # self-test only


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def num(rs, c):
    return np.array([float(r[c]) if r[c] != "" else np.nan for r in rs])


def summarise(sel_rows, aoo, lo, hi, p2, p124, snl, al, flag):
    m = sel_rows
    if not m.any():
        return None
    return dict(rows=int(m.sum()), aoo_median=float(np.median(aoo[m])), lo_median=float(np.median(lo[m])),
                hi_median=float(np.median(hi[m])), mean_p_2=float(np.mean(p2[m])), mean_p_124=float(np.mean(p124[m])),
                share_no_land_median=float(np.nanmedian(snl[m])), aoo_land_only_median=float(np.median(al[m])),
                rows_flagged=int(flag[m].sum()))


def run(p3="../data/stage_p3_products", p3b="../data/stage_p3b_unsurveyed", p3c="../data/stage_p3c_land",
        out_dir="../data/stage_p3d_tables", expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    p3, p3b, p3c, od = map(Path, (p3, p3b, p3c, out_dir))
    if od.exists():
        sys.exit(f"STOP: {od} exists; P3d writes into a new folder")
    for p_ in (p3, p3b, p3c):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    files = {"p3/products_bnm.csv": p3 / "products_bnm.csv", "p3/p3_report.json": p3 / "p3_report.json",
             "p3b/unsurveyed_bnm.csv": p3b / "unsurveyed_bnm.csv", "p3b/p3b_report.json": p3b / "p3b_report.json",
             "p3c/land_bnm.csv": p3c / "land_bnm.csv", "p3c/per_window_land.csv": p3c / "per_window_land.csv",
             "p3c/p3c_report.json": p3c / "p3c_report.json"}
    miss = [str(f) for f in files.values() if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    t0, res = time.time(), {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        out(f"stage P3d: gates {sum(res.values())} PASS, {len(res) - sum(res.values())} FAIL; stopped after {after}, nothing written")
        return res, {}
    me = md5(Path(__file__))
    out(f"stage_p3d_paper_tables.py md5 {me}")
    m = {k: md5(f) for k, f in files.items()}
    gate("G0", all(m[k].startswith(expect[k]) for k in files),
         "input md5s = Sara's runs: " + ", ".join(f"{k} {m[k][:8]} {m[k].startswith(expect[k])}" for k in files))
    if not res["G0"]:
        return stop("G0")
    P, U, L = rows(files["p3/products_bnm.csv"]), rows(files["p3b/unsurveyed_bnm.csv"]), rows(files["p3c/land_bnm.csv"])
    PW = {r["window"]: r for r in rows(files["p3c/per_window_land.csv"])}
    r3, rc = json.loads(files["p3/p3_report.json"].read_text()), json.loads(files["p3c/p3c_report.json"].read_text())
    if broken() == "order":
        L = L[1:] + L[:1]
    n = len(P)
    key = lambda rs: [(r["window"], r["species"]) for r in rs]
    same_rows = len(U) == n and len(L) == n and key(P) == key(U) == key(L)
    copied = same_rows and float(np.nanmax(np.abs(num(P, "aoo_mean_pooled") - num(L, "aoo_mean_pooled_p3")))) == 0.0 and \
        float(np.nanmax(np.abs(num(P, "aoo_mean_pooled") - num(U, "aoo_mean_pooled_p3")))) == 0.0
    land_ok = same_rows and all(int(r["land_tetrads_window"]) == int(PW[r["window"]]["land"]) for r in L)
    gate("G1", same_rows and copied and land_ok, f"{n:,} rows, same (window, species) order in the P3, P3b and P3c files {same_rows}; "
         f"P3 AOO copied into the P3b and P3c files equal {copied}; land tetrads per row = per_window_land.csv {land_ok}")
    if not res["G1"]:
        return stop("G1")
    aoo, lo, hi = num(P, "aoo_mean_pooled"), num(P, "aoo_lo90_pooled"), num(P, "aoo_hi90_pooled")
    p2, p124 = num(P, "p_aoo_le_2_pooled"), num(P, "p_aoo_le_124_pooled")
    snl, al, land = num(L, "share_no_land_pooled"), num(L, "aoo_land_only_pooled_report"), num(L, "land_tetrads_window")
    flag = aoo > land if broken() != "flag_ge" else aoo >= land
    allland_w = {w for w, r in PW.items() if int(r["no_land"]) == 0}
    win_all = np.array([r["window"] in allland_w for r in P])
    cls, rl = np.array([r["recorded_class"] for r in P]), np.array([r["red_list_2022"] for r in P])
    groups = [("recorded class", cls, TCLS), ("Red List 2022", rl, sorted(set(rl.tolist())))]
    summ = {}
    for sname, sv in SETS:
        sm = np.ones(n, bool) if sv is None else (win_all if sv else ~win_all)
        summ[sname] = {g: {k: summarise(sm & (gv == k), aoo, lo, hi, p2, p124, snl, al, flag) for k in labs} for g, gv, labs in groups}
    # G2: all rows against the P3 and P3c reports
    worst_a, worst_p, worst_c, nrow = 0.0, 0.0, 0.0, True
    for g, _, labs in groups:
        for k in labs:
            a, b, c = summ[SETS[0][0]][g][k], r3["summaries"][g].get(k), rc["summaries"][g].get(k)
            if a is None and b is None and c is None:
                continue                                   # a level with no row in any file
            if a is None or b is None or c is None:
                nrow = False; continue
            nrow &= a["rows"] == b["rows"] == c["rows"]
            worst_a = max(worst_a, *(abs(a[x] - b[y]) for x, y in (("aoo_median", "aoo_median"), ("lo_median", "lo_median"), ("hi_median", "hi_median"))))
            worst_p = max(worst_p, abs(a["mean_p_2"] - b["mean_p_2"]), abs(a["mean_p_124"] - b["mean_p_124"]))
            worst_c = max(worst_c, abs(a["share_no_land_median"] - c["share_no_land"]), abs(a["aoo_land_only_median"] - c["aoo_land_only"]))
    gate("G2", nrow and worst_a <= 1e-3 and worst_p <= 1e-4 and worst_c <= 5e-4,
         f"summary code on all rows = p3_report.json (rows equal {nrow}; median AOO and ends largest difference {worst_a:.1e}, mean P at x 2 "
         f"and 124 {worst_p:.1e}) and = p3c_report.json (median share no land and land-only line {worst_c:.1e})")
    if not res["G2"]:
        return stop("G2")
    loop_flag = sum(1 for a_, l_ in zip(aoo, land) if a_ > l_)
    nw_all, nw_nl = len(allland_w), len(set(PW) - allland_w)
    gate("G3", nw_all + nw_nl == len(PW) and int(win_all.sum()) + int((~win_all).sum()) == n and int(flag.sum()) == loop_flag,
         f"windows all land {nw_all} + with no land {nw_nl} = {len(PW)}; rows {int(win_all.sum()):,} + {int((~win_all).sum()):,} = {n:,}; "
         f"rows with corrected AOO above the window's land tetrads {int(flag.sum()):,} (plain loop {loop_flag:,})")
    if not res["G3"]:
        return stop("G3")
    for sname, _ in SETS:
        out(f"   {sname} (rows; median corrected AOO [lower, upper]; mean P(AOO <= 2) | P(AOO <= 124); median share no land; median land-only line; flagged):")
        for g, _, labs in groups:
            for k in labs:
                v = summ[sname][g][k]
                if v is not None:
                    out(f"      {'class' if g == 'recorded class' else 'Red List'} {k:<6} {v['rows']:>5,}; {v['aoo_median']:.1f} [{v['lo_median']:.1f}, {v['hi_median']:.1f}]; "
                        f"{v['mean_p_2']:.2f} | {v['mean_p_124']:.2f}; {v['share_no_land_median']:.3f}; {v['aoo_land_only_median']:.1f}; {v['rows_flagged']}")
    od.mkdir(parents=True)
    head = P3COLS + ["share_never_surveyed_pooled_p3b", "aoo_surveyed_only_pooled_report", "no_land_tetrads_window",
                     "share_no_land_pooled", "aoo_land_only_pooled_report", "window_all_land", "flag_aoo_above_land"]
    with open(od / "products_paper_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head)
        for i in range(n):
            w.writerow([P[i][c] for c in P3COLS] + [U[i]["share_never_surveyed_pooled"], U[i]["aoo_surveyed_only_pooled_report"],
                                                    L[i]["no_land_tetrads_window"], L[i]["share_no_land_pooled"], L[i]["aoo_land_only_pooled_report"],
                                                    "yes" if win_all[i] else "no", "yes" if flag[i] else "no"])
    sh = ["set", "group", "level", "rows", "aoo_median", "aoo_lo90_median", "aoo_hi90_median", "mean_p_aoo_le_2", "mean_p_aoo_le_124",
          "share_no_land_median", "aoo_land_only_median", "rows_flagged"]
    with open(od / "summary_paper_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(sh)
        for sname, _ in SETS:
            for g, _, labs in groups:
                for k in labs:
                    v = summ[sname][g][k]
                    if v is not None:
                        w.writerow([sname, g, k, v["rows"], f"{v['aoo_median']:.3f}", f"{v['lo_median']:.3f}", f"{v['hi_median']:.3f}",
                                    f"{v['mean_p_2']:.4f}", f"{v['mean_p_124']:.4f}", f"{v['share_no_land_median']:.4f}",
                                    f"{v['aoo_land_only_median']:.3f}", v["rows_flagged"]])
    back = rows(od / "products_paper_bnm.csv")
    back_ok = len(back) == n and key(back) == key(P) and sum(r["flag_aoo_above_land"] == "yes" for r in back) == int(flag.sum()) and \
        sum(r["window_all_land"] == "yes" for r in back) == int(win_all.sum()) and all(back[i]["aoo_mean_pooled"] == P[i]["aoo_mean_pooled"] for i in range(n))
    bs = rows(od / "summary_paper_bnm.csv")
    nsum = sum(1 for sname, _ in SETS for g, _, labs in groups for k in labs if summ[sname][g][k] is not None)
    gate("G4", back_ok and len(bs) == nsum, f"products_paper_bnm.csv read back = memory {back_ok}; summary_paper_bnm.csv {len(bs)} lines = {nsum}")
    rep = dict(gates=res, n=n, note="report only; presentation agreed after the P3c result; the P3 products stay as signed",
               md5=dict(tool=me, inputs=m, products_paper=md5(od / "products_paper_bnm.csv"), summary_paper=md5(od / "summary_paper_bnm.csv")),
               windows_all_land=sorted(allland_w), rows_flagged=int(flag.sum()), summaries=summ,
               written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (od / "p3d_report.json").write_text(json.dumps(rep, indent=1))
    out(f"   written products_paper_bnm.csv {rep['md5']['products_paper']}, summary_paper_bnm.csv {rep['md5']['summary_paper']}, p3d_report.json")
    out(f"stage P3d: gates {sum(res.values())} PASS, {len(res) - sum(res.values())} FAIL; {n:,} rows, {len(PW)} windows "
        f"({nw_all} all land); {int(flag.sum())} rows flagged; report only")
    return res, rep


# ---------------------------------------------------------------- self-test on planted csv files
def plant(d):
    """Two windows (one all land, one with 100 no-land tetrads), 7 rows with known answers."""
    p3, p3b, p3c = d / "p3", d / "p3b", d / "p3c"
    for p in (p3, p3b, p3c):
        p.mkdir(parents=True)
    R = [("wA.npz", "sp1", "LC", 0, "0", 5.0, 0.0, 9.0, 0.4, 1.0), ("wA.npz", "sp2", "EN", 3, "2-5", 20.0, 6.0, 40.0, 0.0, 1.0),
         ("wA.npz", "sp3", "LC", 80, "51+", 400.0, 380.0, 400.0, 0.0, 0.0), ("wB.npz", "sp1", "LC", 0, "0", 8.0, 0.0, 15.0, 0.2, 1.0),
         ("wB.npz", "sp4", "VU", 1, "1", 12.0, 1.0, 30.0, 0.1, 1.0), ("wB.npz", "sp3", "LC", 60, "51+", 350.0, 320.0, 390.0, 0.0, 0.0),
         ("wB.npz", "sp5", "EN", 30, "21-50", 300.0, 250.0, 340.0, 0.0, 0.1)]
    land = {"wA.npz": 400, "wB.npz": 300}
    snl = [0.0, 0.0, 0.0, 0.6, 0.5, 0.2, 0.3]
    hp = ["window", "half", "species", "red_list_2022", "rec_2019", "rec_6y", "recorded_class", "records_bin", "aoo_mean_pooled",
          "aoo_lo90_pooled", "aoo_hi90_pooled"] + [f"p_aoo_le_{x}_pooled" for x in XS] + ["h6_ratio_pooled", "h6_cov_pooled", "note"]
    with open(p3 / "products_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hp)
        for (wn, sp, rl, rec, c, a, l_, h, p2_, p124_) in R:
            w.writerow([wn, "fit", sp, rl, rec, rec, c, "1-3", f"{a:.3f}", f"{l_:.3f}", f"{h:.3f}", "0.0000", f"{p2_:.4f}", "0.0000", "0.0000",
                        "0.0000", f"{p124_:.4f}", "1.000", "0.900", ""])
    with open(p3b / "unsurveyed_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["window", "species", "share_never_surveyed_pooled", "aoo_surveyed_only_pooled_report", "aoo_mean_pooled_p3"])
        for (wn, sp, *_r) in R:
            w.writerow([wn, sp, "0.5000", "1.000", f"{_r[3]:.3f}"])
    with open(p3c / "land_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["window", "species", "land_tetrads_window", "no_land_tetrads_window", "share_no_land_pooled",
                                        "aoo_mean_pooled_p3", "aoo_land_only_pooled_report"])
        for (wn, sp, *_r), s in zip(R, snl):
            w.writerow([wn, sp, land[wn], 400 - land[wn], f"{s:.4f}", f"{_r[3]:.3f}", f"{min(_r[3] * (1 - s), land[wn]):.3f}"])
    with open(p3c / "per_window_land.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["window", "land", "no_land"])
        for wn, v in land.items():
            w.writerow([wn, v, 400 - v])
    P, L = rows(p3 / "products_bnm.csv"), rows(p3c / "land_bnm.csv")
    cl, rl = np.array([r["recorded_class"] for r in P]), np.array([r["red_list_2022"] for r in P])
    sp3, spc = {}, {}
    for g, gv, labs in (("recorded class", cl, TCLS), ("Red List 2022", rl, sorted(set(rl.tolist())))):
        sp3[g], spc[g] = {}, {}
        for k in labs:
            mk = gv == k
            if mk.any():
                sp3[g][k] = dict(rows=int(mk.sum()), aoo_median=float(np.median(num(P, "aoo_mean_pooled")[mk])),
                                 lo_median=float(np.median(num(P, "aoo_lo90_pooled")[mk])), hi_median=float(np.median(num(P, "aoo_hi90_pooled")[mk])),
                                 mean_p_2=float(np.mean(num(P, "p_aoo_le_2_pooled")[mk])), mean_p_124=float(np.mean(num(P, "p_aoo_le_124_pooled")[mk])))
                spc[g][k] = dict(rows=int(mk.sum()), share_no_land=float(np.median(num(L, "share_no_land_pooled")[mk])),
                                 aoo_land_only=float(np.median(num(L, "aoo_land_only_pooled_report")[mk])))
    (p3 / "p3_report.json").write_text(json.dumps(dict(summaries=sp3)))
    (p3b / "p3b_report.json").write_text("{}")
    (p3c / "p3c_report.json").write_text(json.dumps(dict(summaries=spc)))
    ex = {k: md5(d / k) for k in EXPECT}
    return dict(p3=p3, p3b=p3b, p3c=p3c, expect=ex)


def self_test():
    res = []
    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")
    quiet = lambda *a: None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); kw = plant(tmp)
        r, rep = run(out_dir=tmp / "out", out=quiet, **kw)
        t("T1", r and all(r.values()), f"planted csv files: gates {r}")
        b = rows(tmp / "out" / "products_paper_bnm.csv") if (tmp / "out").exists() else []
        fl = [(x["window"], x["species"]) for x in b if x["flag_aoo_above_land"] == "yes"]
        wa = sorted({x["window"] for x in b if x["window_all_land"] == "yes"})
        t("T2", fl == [("wB.npz", "sp3")] and wa == ["wA.npz"] and len(b) == 7,
          f"by hand: flagged rows {fl} (expected only wB sp3, 350 above 300 land tetrads; wB sp5 at exactly 300 is not above); "
          f"all-land windows {wa} (expected wA)")
        s = rows(tmp / "out" / "summary_paper_bnm.csv") if (tmp / "out").exists() else []
        e = [x for x in s if x["set"].startswith("windows all land") and x["group"] == "recorded class" and x["level"] == "51+"]
        t("T3", len(e) == 1 and e[0]["rows"] == "1" and e[0]["aoo_median"] == "400.000" and e[0]["rows_flagged"] == "0",
          f"all-land 51+ line {e[0] if e else None}")
        stops = []
        for o in (tmp / "out", kw["p3c"] / "inside"):
            try:
                run(out_dir=o, out=quiet, **kw); stops.append(False)
            except SystemExit:
                stops.append(True)
        t("T4", all(stops), f"refused: an existing folder {stops[0]}, a folder inside an input folder {stops[1]}")
        for bug, g in (("order", "G1"), ("flag_ge", "G3")):
            os.environ["STAGE_P3D_BROKEN"] = bug
            try:
                rb, _ = run(out_dir=tmp / f"bug_{bug}", out=quiet, **kw)
            finally:
                os.environ.pop("STAGE_P3D_BROKEN", None)
            t(f"bug {bug}", rb.get(g) is False and not (tmp / f"bug_{bug}").exists(), f"{g} fails as it must, nothing written ({rb})")
        rw, _ = run(out_dir=tmp / "bad_md5", out=quiet, **dict(kw, expect=dict(kw["expect"], **{"p3c/land_bnm.csv": "00000000"})))
        t("bug md5", rw.get("G0") is False and not (tmp / "bad_md5").exists(), f"a changed input fails G0, nothing written ({rw})")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--p3", default="../data/stage_p3_products")
    ap.add_argument("--p3b", default="../data/stage_p3b_unsurveyed")
    ap.add_argument("--p3c", default="../data/stage_p3c_land")
    ap.add_argument("--out", default="../data/stage_p3d_tables")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    r, _ = run(a.p3, a.p3b, a.p3c, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)