#!/usr/bin/env python3
"""stage_p4d_paper_tables.py  (obj2_v2, P4 step P4d, 14 Nov 2026; report only; Sara's yes on 14 Nov after the P4-3 result, so
every presentation choice here is labelled as made after the result)

What it writes: the paper's tables for the land-only products, from csv and json files of P4-3, P3 and P3c only (no npz, no
map, no truth file). The P4-3 and P3 products stay as signed; nothing is scored, refitted or changed.
    products_land_paper_bnm.csv  all 3,614 rows: the pooled land-only product (corrected AOO, 90% interval, P(AOO <= x) at x 1,
                                 2, 5, 10, 20, 124) and beside it the P3 window product (mean, interval, P at x 2 and 124), the
                                 ratio of the means, the window's land tetrads and type (all land / coastal), "mean_at_land_cap"
                                 (printed mean = land tetrads), "no_interval" (crossed, rule E), the H6 columns, the P4-3 note,
                                 and, report only, "share_never_surveyed_land": the share of the maps' occupied land tetrads
                                 that nobody surveyed in 2014-2019, from P3c's columns as share_never_surveyed_land_pooled /
                                 (1 - share_no_land_pooled) (P3c's land; L_w adds 3 surveyed tetrads in 3 windows, which this
                                 share does not count in its denominator)
    summary_land_paper_bnm.csv   by recorded class and GB Red List 2022, for all rows, the 57 all-land windows and the 44 coastal
                                 windows: rows; median 2019 records; median land-only AOO [lower, upper] and P3's; median ratio;
                                 mean P(AOO <= 2) and (<= 124), land-only and P3; median width ratio; median never-surveyed land
                                 share; rows with no interval; rows with the mean at the land cap
    p4d_report.json              md5s, gates, counts, summaries
Gates
    G0 inputs: md5 of the 6 input files = Sara's runs; P4-3, P3 and P3c gates PASS and their csv md5 = their report
    G1 rows: the same 3,614 (window, species) rows in the same order in the three row files; P4-3's identity columns and its
       P3 mean column = products_bnm.csv; land tetrads = P4-3's window list = P3c land + put back
    G2 code: the summary code reproduces p4_3_report.json (by recorded class and Red List on all rows, and the all-land and
       coastal lines: rows, median 2019 records, median AOO and ends within 0.001, median ratio to P3 within 0.002, mean P at
       x 2 and 124 within 1e-4; the median ratios to P3 and to P3c's land line by window type)
    G3 counts: windows all land + coastal = 101, rows of both = 3,614; rows with no interval = P4-3's crossed count per class;
       mean-at-cap rows and the never-surveyed land share recounted by a plain loop; P = 1 at every x >= land tetrads
    G4 files: output folder new and outside every input folder; both csv files read back = memory
Run (from PSD_Dispersal_pool/obj2_v2/work; seconds)
    python3 ../tools/stage_p4d_paper_tables.py --self-test
    python3 -u ../tools/stage_p4d_paper_tables.py > ../logs/stage_p4d_paper_tables.txt 2>&1
"""
import argparse, csv, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

EXPECT = {"p4_3/products_land_bnm.csv": "af01b4ad", "p4_3/p4_3_report.json": "b3ccdf56", "p3/products_bnm.csv": "b409f660",
          "p3/p3_report.json": "517c98e5", "p3c/land_bnm.csv": "a569190d", "p3c/p3c_report.json": "6f051676"}
TCLS = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
XS = (1, 2, 5, 10, 20, 124)
ID = ["window", "half", "species", "red_list_2022", "rec_2019", "rec_6y", "recorded_class", "records_bin"]
SETS = [("all rows", None), ("all-land windows (post-result line)", "all land"), ("coastal windows", "coastal")]
NOTE_X = "no interval (crossed), pooled"


def broken():
    return os.environ.get("STAGE_P4D_BROKEN", "")        # self-test only


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def num(rs, c):
    return np.array([float(r[c]) if r[c] != "" else np.nan for r in rs])


def nmed(v):
    """Median over the values that exist (rows with no interval have none); NaN if none."""
    v = v[~np.isnan(v)] if broken() != "nan_median" else v
    return float(np.median(v)) if len(v) else float("nan")


def nsl_share(nsl, snl):
    """Share of the occupied land tetrads that nobody surveyed: never-surveyed land / (all occupied - occupied no land)."""
    if broken() == "share_raw":
        return nsl.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(nsl) & np.isfinite(snl) & (snl < 1), nsl / (1 - snl), np.nan)


def summarise(m, d):
    if not m.any():
        return None
    w = m & ~np.isnan(d["lo"]) & (d["hi3"] - d["lo3"] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        wr = (d["hi"] - d["lo"]) / (d["hi3"] - d["lo3"])
    return dict(rows=int(m.sum()), rec_2019_median=float(np.median(d["rec"][m])), aoo_median=float(np.median(d["mean"][m])),
                lo_median=nmed(d["lo"][m]), hi_median=nmed(d["hi"][m]), aoo_p3_median=float(np.median(d["mean3"][m])),
                lo_p3_median=float(np.median(d["lo3"][m])), hi_p3_median=float(np.median(d["hi3"][m])),
                ratio_median=float(np.median(d["ratio"][m])), mean_p_2=float(d["P"][m, 1].mean()), mean_p_124=float(d["P"][m, 5].mean()),
                mean_p_2_p3=float(d["p2_3"][m].mean()), mean_p_124_p3=float(d["p124_3"][m].mean()),
                width_ratio_median=float(np.median(wr[w])) if w.any() else float("nan"),
                share_nsl_median=nmed(d["nsl"][m]), rows_no_interval=int((m & np.isnan(d["lo"])).sum()), rows_at_cap=int((m & d["cap"]).sum()))


def run(p4_3="../data/stage_p4_3_products", p3="../data/stage_p3_products", p3c="../data/stage_p3c_land",
        out_dir="../data/stage_p4d_tables", expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    p4_3, p3, p3c, od = map(Path, (p4_3, p3, p3c, out_dir))
    if od.exists():
        sys.exit(f"STOP: {od} exists; P4d writes into a new folder")
    for p_ in (p4_3, p3, p3c):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    files = {"p4_3/products_land_bnm.csv": p4_3 / "products_land_bnm.csv", "p4_3/p4_3_report.json": p4_3 / "p4_3_report.json",
             "p3/products_bnm.csv": p3 / "products_bnm.csv", "p3/p3_report.json": p3 / "p3_report.json",
             "p3c/land_bnm.csv": p3c / "land_bnm.csv", "p3c/p3c_report.json": p3c / "p3c_report.json"}
    miss = [str(f) for f in files.values() if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    res = {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        out(f"stage P4d: gates {sum(res.values())} PASS, {len(res) - sum(res.values())} FAIL; stopped after {after}, nothing written")
        return res, {}
    me = md5(Path(__file__))
    out(f"stage_p4d_paper_tables.py md5 {me}")
    m = {k: md5(f) for k, f in files.items()}
    r4, r3, rc = (json.loads(files[k].read_text()) for k in ("p4_3/p4_3_report.json", "p3/p3_report.json", "p3c/p3c_report.json"))
    chain = all(r4["gates"].values()) and all(r3["gates"].values()) and all(rc["gates"].values()) \
        and r4["md5"]["out_csv"] == m["p4_3/products_land_bnm.csv"] and r3["md5"]["products_csv"] == m["p3/products_bnm.csv"] \
        and rc["md5"]["out_csv"] == m["p3c/land_bnm.csv"] and r4["md5"]["p3_report"] == m["p3/p3_report.json"] \
        and r4["md5"]["p3c_report"] == m["p3c/p3c_report.json"]
    gate("G0", all(m[k].startswith(expect[k]) for k in files) and chain,
         "input md5s = Sara's runs: " + ", ".join(f"{k} {m[k][:8]} {m[k].startswith(expect[k])}" for k in files)
         + f"; P4-3, P3 and P3c gates PASS, csv md5 = their reports, P4-3 made from this P3 and P3c {chain}")
    if not res["G0"]:
        return stop("G0")
    A, P, L = rows(files["p4_3/products_land_bnm.csv"]), rows(files["p3/products_bnm.csv"]), rows(files["p3c/land_bnm.csv"])
    if broken() == "order":
        L = L[1:] + L[:1]
    n = len(A)
    key = lambda rs: [(r["window"], r["species"]) for r in rs]
    same_rows = len(P) == n and len(L) == n and key(A) == key(P) == key(L)
    ident = same_rows and all(a[c] == b[c] for a, b in zip(A, P) for c in ID) and \
        all(a["aoo_mean_pooled_p3"] == b["aoo_mean_pooled"] for a, b in zip(A, P))
    wl = {w["name"]: w for w in r4["windows"]}
    land = num(A, "land_tetrads")
    land_ok = same_rows and all(int(a["land_tetrads"]) == wl[a["window"]]["land"] for a in A) and \
        all(int(l_["land_tetrads_window"]) + wl[l_["window"]]["put_back"] == int(a["land_tetrads"]) for a, l_ in zip(A, L)) and \
        all((a["window_type"] == "all land") == (a["land_tetrads"] == "400") for a in A)
    gate("G1", same_rows and ident and land_ok, f"{n:,} rows, same (window, species) order in the P4-3, P3 and P3c files {same_rows}; "
         f"identity columns and P3 mean copied into P4-3 = products_bnm.csv {ident}; land tetrads = P4-3's windows = P3c land + "
         f"put back, window type {land_ok}")
    if not res["G1"]:
        return stop("G1")
    d = dict(rec=num(A, "rec_2019"), mean=num(A, "aoo_mean_pooled"), lo=num(A, "aoo_lo90_pooled"), hi=num(A, "aoo_hi90_pooled"),
             P=np.column_stack([num(A, f"p_aoo_le_{x}_pooled") for x in XS]), mean3=num(P, "aoo_mean_pooled"),
             lo3=num(P, "aoo_lo90_pooled"), hi3=num(P, "aoo_hi90_pooled"), p2_3=num(P, "p_aoo_le_2_pooled"),
             p124_3=num(P, "p_aoo_le_124_pooled"))
    d["ratio"] = d["mean"] / np.maximum(d["mean3"], 1e-12)                    # as P4-3's report
    d["nsl"] = nsl_share(num(L, "share_never_surveyed_land_pooled"), num(L, "share_no_land_pooled"))
    d["cap"] = np.array([float(a["aoo_mean_pooled"]) == float(a["land_tetrads"]) for a in A])
    wt = np.array([a["window_type"] for a in A])
    cls, rl = np.array([a["recorded_class"] for a in A]), np.array([a["red_list_2022"] for a in A])
    groups = [("recorded class", cls, TCLS), ("Red List 2022", rl, sorted(set(rl.tolist())))]
    summ = {}
    for sname, sv in SETS:
        sm = np.ones(n, bool) if sv is None else wt == sv
        summ[sname] = {g: {k: summarise(sm & (gv == k), d) for k in labs} for g, gv, labs in groups}
        summ[sname]["all"] = summarise(sm, d)
    # G2: against p4_3_report.json
    keys_a = (("aoo_median", "aoo_median"), ("lo_median", "lo_median"), ("hi_median", "hi_median"))
    wa = wp = wr = 0.0
    nrow = True
    pairs = [(summ["all rows"][g][k], r4["summaries"][g].get(k)) for g, _, labs in groups for k in labs] + \
        [(summ[s]["all"], r4["summaries"]["window"].get(v)) for s, v in SETS[1:]]
    for a, b in pairs:
        if a is None and b is None:
            continue
        if a is None or b is None:
            nrow = False; continue
        nrow &= a["rows"] == b["rows"] and a["rec_2019_median"] == b["rec_2019_median"]
        for x, y in keys_a:
            u, v = a[x], b[y] if b[y] is not None else float("nan")
            wa = max(wa, 0.0 if (np.isnan(u) and np.isnan(v)) else (abs(u - v) if not (np.isnan(u) or np.isnan(v)) else np.inf))
        wr = max(wr, abs(a["ratio_median"] - b["p4_over_p3_median"]))
        wp = max(wp, abs(a["mean_p_2"] - b["mean_p_2"]), abs(a["mean_p_124"] - b["mean_p_124"]))
    lc = num(L, "aoo_land_only_pooled_report")
    agp = {v: (float(np.median(d["ratio"][wt == v])), float(np.median((d["mean"] / np.maximum(lc, 1e-12))[wt == v]))) for _, v in SETS[1:]}
    wg = max(max(abs(agp[v][0] - r4["against_p3"][v]["p4_over_p3_median"]), abs(agp[v][1] - r4["against_p3"][v]["p4_over_p3c_line_median"]))
             for _, v in SETS[1:])
    gate("G2", nrow and wa <= 1e-3 and wr <= 2e-3 and wp <= 1e-4 and wg <= 2e-3,
         f"summary code = p4_3_report.json by recorded class, Red List and window type (rows and median 2019 records equal {nrow}; "
         f"median AOO and ends largest difference {wa:.1e}; median ratio to P3 {wr:.1e}; mean P at x 2 and 124 {wp:.1e}); median "
         f"ratios to P3 and to P3c's land line by window type {wg:.1e}")
    if not res["G2"]:
        return stop("G2")
    xc = {c: int(((cls == c) & np.isnan(d["lo"])).sum()) for c in TCLS}
    x_ok = xc == r4["crossed"]["pooled"]["by_class"] and all((NOTE_X in a["note"]) == (a["aoo_lo90_pooled"] == "") for a in A)
    loop_cap = sum(1 for a in A if float(a["aoo_mean_pooled"]) == float(a["land_tetrads"]))
    loop_nsl = []
    for l_ in L:
        u, v = l_["share_never_surveyed_land_pooled"], l_["share_no_land_pooled"]
        loop_nsl.append(float(u) / (1 - float(v)) if u != "" and v != "" and float(v) < 1 else float("nan"))
    loop_nsl = np.array(loop_nsl)
    nsl_ok = bool(np.array_equal(np.isnan(loop_nsl), np.isnan(d["nsl"]))) and \
        float(np.nanmax(np.abs(loop_nsl - d["nsl"]))) <= 1e-12 and bool(((d["nsl"][~np.isnan(d["nsl"])] >= 0) & (d["nsl"][~np.isnan(d["nsl"])] <= 1 + 1e-9)).all())
    one_ok = bool((d["P"][np.array(XS)[None, :] >= land[:, None]] == 1).all())
    nwin = {w: wl[w]["land"] for w in wl}
    n_all, n_co = sum(v == 400 for v in nwin.values()), sum(v < 400 for v in nwin.values())
    gate("G3", n_all + n_co == len(nwin) and int((wt == "all land").sum()) + int((wt == "coastal").sum()) == n and x_ok
         and int(d["cap"].sum()) == loop_cap and nsl_ok and one_ok,
         f"windows all land {n_all} + coastal {n_co} = {len(nwin)}; rows {int((wt == 'all land').sum()):,} + {int((wt == 'coastal').sum()):,} = {n:,}; "
         f"rows with no interval by recorded class = P4-3's crossed count, notes match {x_ok}; mean at the land cap {int(d['cap'].sum())} "
         f"(plain loop {loop_cap}); never-surveyed land share by a plain loop, in [0, 1] {nsl_ok}; P = 1 at every x >= land tetrads {one_ok}")
    if not res["G3"]:
        return stop("G3")
    for sname, _ in SETS:
        out(f"   {sname} (rows; median land-only AOO [lower, upper] | P3; median ratio; mean P(AOO <= 2) land-only | P3; P(<= 124) "
            f"land-only | P3; median width ratio; median never-surveyed land share; no interval; mean at cap):")
        for g, _, labs in groups:
            for k in labs:
                v = summ[sname][g][k]
                if v is not None:
                    out(f"      {'class' if g == 'recorded class' else 'Red List'} {k:<6} {v['rows']:>5,}; {v['aoo_median']:.1f} [{v['lo_median']:.1f}, "
                        f"{v['hi_median']:.1f}] | {v['aoo_p3_median']:.1f} [{v['lo_p3_median']:.1f}, {v['hi_p3_median']:.1f}]; {v['ratio_median']:.3f}; "
                        f"{v['mean_p_2']:.2f} | {v['mean_p_2_p3']:.2f}; {v['mean_p_124']:.2f} | {v['mean_p_124_p3']:.2f}; {v['width_ratio_median']:.2f}; "
                        f"{v['share_nsl_median']:.3f}; {v['rows_no_interval']}; {v['rows_at_cap']}")
    od.mkdir(parents=True)
    f4 = lambda v: "" if np.isnan(v) else f"{v:.4f}"
    head = ID + ["land_tetrads", "window_type", "aoo_mean_land", "aoo_lo90_land", "aoo_hi90_land"] + [f"p_aoo_le_{x}_land" for x in XS] + \
        ["aoo_mean_p3", "aoo_lo90_p3", "aoo_hi90_p3", "p_aoo_le_2_p3", "p_aoo_le_124_p3", "ratio_mean_land_p3", "share_never_surveyed_land",
         "mean_at_land_cap", "no_interval", "h6_ratio_pooled", "h6_cov_pooled", "note"]
    with open(od / "products_land_paper_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head)
        for i in range(n):
            a, b = A[i], P[i]
            w.writerow([a[c] for c in ID] + [a["land_tetrads"], a["window_type"], a["aoo_mean_pooled"], a["aoo_lo90_pooled"], a["aoo_hi90_pooled"]]
                       + [a[f"p_aoo_le_{x}_pooled"] for x in XS] + [b["aoo_mean_pooled"], b["aoo_lo90_pooled"], b["aoo_hi90_pooled"],
                                                                     b["p_aoo_le_2_pooled"], b["p_aoo_le_124_pooled"],
                                                                     f4(d["mean"][i] / d["mean3"][i]) if d["mean3"][i] > 0 else "", f4(d["nsl"][i]),
                                                                     "yes" if d["cap"][i] else "no", "yes" if a["aoo_lo90_pooled"] == "" else "no",
                                                                     a["h6_ratio_pooled"], a["h6_cov_pooled"], a["note"]])
    sk = ["rows", "rec_2019_median", "aoo_median", "lo_median", "hi_median", "aoo_p3_median", "lo_p3_median", "hi_p3_median", "ratio_median",
          "mean_p_2", "mean_p_2_p3", "mean_p_124", "mean_p_124_p3", "width_ratio_median", "share_nsl_median", "rows_no_interval", "rows_at_cap"]
    fm = lambda k, v: str(v) if isinstance(v, int) else ("" if np.isnan(v) else (f"{v:.4f}" if k.startswith(("mean_p", "ratio", "width", "share")) else f"{v:.3f}"))
    nsum = 0
    with open(od / "summary_land_paper_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["set", "group", "level"] + sk)
        for sname, _ in SETS:
            for g, _, labs in groups + [("all", None, ["all"])]:
                for k in labs:
                    v = summ[sname]["all"] if g == "all" else summ[sname][g][k]
                    if v is not None:
                        w.writerow([sname, g, k] + [fm(q, v[q]) for q in sk]); nsum += 1
    back = rows(od / "products_land_paper_bnm.csv")
    back_ok = len(back) == n and key(back) == key(A) and all(back[i]["aoo_mean_land"] == A[i]["aoo_mean_pooled"] and
                                                             back[i]["aoo_mean_p3"] == P[i]["aoo_mean_pooled"] for i in range(n)) and \
        sum(r["mean_at_land_cap"] == "yes" for r in back) == int(d["cap"].sum()) and \
        sum(r["no_interval"] == "yes" for r in back) == int(np.isnan(d["lo"]).sum()) and \
        bool(np.array_equal(np.isnan(num(back, "share_never_surveyed_land")), np.isnan(d["nsl"])))
    bs = rows(od / "summary_land_paper_bnm.csv")
    gate("G4", back_ok and len(bs) == nsum, f"products_land_paper_bnm.csv read back = memory {back_ok}; summary_land_paper_bnm.csv {len(bs)} lines = {nsum}")
    rep = dict(gates=res, n=n, note="report only; presentation agreed after the P4-3 result; the P4-3 and P3 products stay as signed",
               share_never_surveyed_land="share_never_surveyed_land_pooled / (1 - share_no_land_pooled), P3c's columns (pooled maps, P3c land)",
               md5=dict(tool=me, inputs=m, products_paper=md5(od / "products_land_paper_bnm.csv"), summary_paper=md5(od / "summary_land_paper_bnm.csv")),
               windows=dict(all_land=n_all, coastal=n_co), rows_at_cap=int(d["cap"].sum()), rows_no_interval=int(np.isnan(d["lo"]).sum()),
               summaries=summ, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (od / "p4d_report.json").write_text(json.dumps(rep, indent=1))
    out(f"   written products_land_paper_bnm.csv {rep['md5']['products_paper']}, summary_land_paper_bnm.csv {rep['md5']['summary_paper']}, p4d_report.json")
    out(f"stage P4d: gates {sum(res.values())} PASS, {len(res) - sum(res.values())} FAIL; {n:,} rows, {len(nwin)} windows ({n_all} all land, "
        f"{n_co} coastal); report only")
    return res, rep


# ---------------------------------------------------------------- self-test on planted csv files
def plant(d):
    """Two windows (wA all land, wB coastal with 120 land tetrads), 6 rows with known answers."""
    p4, p3, p3c = d / "p4_3", d / "p3", d / "p3c"
    for p in (p4, p3, p3c):
        p.mkdir(parents=True)
    #     window  sp     RL    rec cls      land-only mean, lo, hi,     P2,  P124 | P3 mean, lo, hi, P2, P124 | snl, nsl
    R = [("wA.npz", "sp1", "LC", 0, "0", 4.0, 0.0, 9.0, 0.5, 1.0, 4.0, 0.0, 9.0, 0.5, 1.0, 0.0, 0.3),
         ("wA.npz", "sp2", "EN", 3, "2-5", 20.0, 8.0, 30.0, 0.0, 1.0, 20.2, 8.0, 31.0, 0.0, 1.0, 0.0, 0.1),
         ("wA.npz", "sp3", "LC", 90, "51+", 330.0, 300.0, 350.0, 0.0, 0.0, 331.0, 301.0, 351.0, 0.0, 0.0, 0.0, 0.0),
         ("wB.npz", "sp1", "LC", 0, "0", 2.0, 0.0, 6.0, 0.6, 1.0, 10.0, 0.0, 30.0, 0.3, 1.0, 0.6, 0.2),
         ("wB.npz", "sp4", "VU", 1, "1", 5.0, 1.0, 12.0, 0.3, 1.0, 16.0, 1.0, 40.0, 0.2, 1.0, 0.5, 0.25),
         ("wB.npz", "sp3", "LC", 70, "51+", 120.0, None, None, 0.0, 1.0, 340.0, 300.0, 380.0, 0.0, 0.0, 0.4, 0.3)]
    land = {"wA.npz": 400, "wB.npz": 120}
    ha = ID + ["land_tetrads", "window_type", "aoo_mean_pooled", "aoo_lo90_pooled", "aoo_hi90_pooled"] + [f"p_aoo_le_{x}_pooled" for x in XS] + \
        ["aoo_mean_pooled_p3", "h6_ratio_pooled", "h6_cov_pooled", "note"]
    hp = ID + ["aoo_mean_pooled", "aoo_lo90_pooled", "aoo_hi90_pooled"] + [f"p_aoo_le_{x}_pooled" for x in XS]
    f3 = lambda v: "" if v is None else f"{v:.3f}"
    with open(p4 / "products_land_bnm.csv", "w", newline="") as fa, open(p3 / "products_bnm.csv", "w", newline="") as fp, \
            open(p3c / "land_bnm.csv", "w", newline="") as fl:
        wa, wp, wl_ = csv.writer(fa), csv.writer(fp), csv.writer(fl)
        wa.writerow(ha); wp.writerow(hp)
        wl_.writerow(["window", "species", "land_tetrads_window", "share_no_land_pooled", "share_never_surveyed_land_pooled", "aoo_land_only_pooled_report"])
        for (wn, sp, rl, rec, c, m4, l4, h4, p24, p1244, m3, l3, h3, p23, p1243, snl, nsl) in R:
            idv = [wn, "fit", sp, rl, rec, rec, c, "1-3"]
            p124_4 = 1.0 if land[wn] <= 124 else p1244
            wa.writerow(idv + [land[wn], "all land" if land[wn] == 400 else "coastal", f3(m4), f3(l4), f3(h4), "0.0000", f"{p24:.4f}", "0.0000", "0.0000",
                               "1.0000" if land[wn] <= 20 else "0.0000", f"{p124_4:.4f}", f3(m3), "1.000", "0.900",
                               NOTE_X if l4 is None else ""])
            wp.writerow(idv + [f3(m3), f3(l3), f3(h3), "0.0000", f"{p23:.4f}", "0.0000", "0.0000", "0.0000", f"{p1243:.4f}"])
            wl_.writerow([wn, sp, land[wn], f"{snl:.4f}", f"{nsl:.4f}", f3(min(m3 * (1 - snl), land[wn]))])
    A, P, L = rows(p4 / "products_land_bnm.csv"), rows(p3 / "products_bnm.csv"), rows(p3c / "land_bnm.csv")
    mean, mean3, lo, hi = num(A, "aoo_mean_pooled"), num(P, "aoo_mean_pooled"), num(A, "aoo_lo90_pooled"), num(A, "aoo_hi90_pooled")
    ratio = mean / np.maximum(mean3, 1e-12)
    cl, rl = np.array([r["recorded_class"] for r in A]), np.array([r["red_list_2022"] for r in A])
    wt = np.array([r["window_type"] for r in A])
    p2, p124, rec = num(A, "p_aoo_le_2_pooled"), num(A, "p_aoo_le_124_pooled"), num(A, "rec_2019")
    def s(mk):
        q = lambda v: float(np.median(v[mk & ~np.isnan(v)])) if (mk & ~np.isnan(v)).any() else None
        return dict(rows=int(mk.sum()), rec_2019_median=float(np.median(rec[mk])), aoo_median=float(np.median(mean[mk])), lo_median=q(lo),
                    hi_median=q(hi), p4_over_p3_median=float(np.median(ratio[mk])), mean_p_2=float(p2[mk].mean()), mean_p_124=float(p124[mk].mean()))
    summ = {g: {k: s(gv == k) for k in labs if (gv == k).any()} for g, gv, labs in (("recorded class", cl, TCLS), ("Red List 2022", rl, sorted(set(rl.tolist()))))}
    summ["window"] = {v: s(wt == v) for v in ("all land", "coastal")}
    lc = num(L, "aoo_land_only_pooled_report")
    ag = {v: dict(p4_over_p3_median=float(np.median(ratio[wt == v])), p4_over_p3c_line_median=float(np.median((mean / np.maximum(lc, 1e-12))[wt == v])))
          for v in ("all land", "coastal")}
    (p3 / "p3_report.json").write_text(json.dumps(dict(gates={"C0": True}, md5=dict(products_csv=md5(p3 / "products_bnm.csv")))))
    (p3c / "p3c_report.json").write_text(json.dumps(dict(gates={"L0": True}, md5=dict(out_csv=md5(p3c / "land_bnm.csv")))))
    (p4 / "p4_3_report.json").write_text(json.dumps(dict(
        gates={"D0": True}, md5=dict(out_csv=md5(p4 / "products_land_bnm.csv"), p3_report=md5(p3 / "p3_report.json"),
                                     p3c_report=md5(p3c / "p3c_report.json")),
        windows=[dict(name=w, land=v, put_back=0) for w, v in land.items()], summaries=summ, against_p3=ag,
        crossed=dict(pooled=dict(all=1, by_class={c: int(c == "51+") for c in TCLS})))))
    return dict(p4_3=p4, p3=p3, p3c=p3c, expect={k: md5(d / k) for k in EXPECT})


def self_test():
    res = []
    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")
    quiet = lambda *a: None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); kw = plant(tmp)
        r, rep = run(out_dir=tmp / "out", out=quiet, **kw)
        t("T1", r and all(r.values()), f"planted csv files: gates {r}")
        b = rows(tmp / "out" / "products_land_paper_bnm.csv") if (tmp / "out").exists() else []
        cap = [(x["window"], x["species"]) for x in b if x["mean_at_land_cap"] == "yes"]
        nint = [(x["window"], x["species"]) for x in b if x["no_interval"] == "yes"]
        sh = {(x["window"], x["species"]): x["share_never_surveyed_land"] for x in b}
        rt = {(x["window"], x["species"]): x["ratio_mean_land_p3"] for x in b}
        t("T2", cap == [("wB.npz", "sp3")] and nint == [("wB.npz", "sp3")] and sh.get(("wB.npz", "sp1")) == "0.5000"
          and sh.get(("wA.npz", "sp1")) == "0.3000" and sh.get(("wB.npz", "sp3")) == "0.5000" and rt.get(("wB.npz", "sp1")) == "0.2000",
          f"by hand: mean at the land cap {cap} and no interval {nint} (expected wB sp3, 120 of 120); never-surveyed land share wB sp1 "
          f"{sh.get(('wB.npz', 'sp1'))} (0.2 / (1 - 0.6) = 0.5), wA sp1 {sh.get(('wA.npz', 'sp1'))} (0.3), wB sp3 {sh.get(('wB.npz', 'sp3'))} "
          f"(0.3 / 0.6); ratio wB sp1 {rt.get(('wB.npz', 'sp1'))} (2 / 10)")
        s = rows(tmp / "out" / "summary_land_paper_bnm.csv") if (tmp / "out").exists() else []
        e = [x for x in s if x["set"] == "coastal windows" and x["group"] == "recorded class" and x["level"] == "51+"]
        t("T3", len(e) == 1 and e[0]["rows"] == "1" and e[0]["aoo_median"] == "120.000" and e[0]["lo_median"] == "" and e[0]["rows_no_interval"] == "1"
          and e[0]["rows_at_cap"] == "1" and e[0]["aoo_p3_median"] == "340.000",
          f"coastal 51+ line by hand (1 row, 120 at the cap, no interval, P3 340): {e[0] if e else None}")
        stops = []
        for o in (tmp / "out", kw["p3c"] / "inside"):
            try:
                run(out_dir=o, out=quiet, **kw); stops.append(False)
            except SystemExit:
                stops.append(True)
        t("T4", all(stops), f"refused: an existing folder {stops[0]}, a folder inside an input folder {stops[1]}")
        for bug, g in (("order", "G1"), ("nan_median", "G2"), ("share_raw", "G3")):
            os.environ["STAGE_P4D_BROKEN"] = bug
            try:
                rb, _ = run(out_dir=tmp / f"bug_{bug}", out=quiet, **kw)
            finally:
                os.environ.pop("STAGE_P4D_BROKEN", None)
            t(f"bug {bug}", rb.get(g) is False and not (tmp / f"bug_{bug}").exists(), f"{g} fails as it must, nothing written ({rb})")
        rw, _ = run(out_dir=tmp / "bad_md5", out=quiet, **dict(kw, expect=dict(kw["expect"], **{"p3c/land_bnm.csv": "00000000"})))
        t("bug md5", rw.get("G0") is False and not (tmp / "bad_md5").exists(), f"a changed input fails G0, nothing written ({rw})")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--p4-3", default="../data/stage_p4_3_products")
    ap.add_argument("--p3", default="../data/stage_p3_products")
    ap.add_argument("--p3c", default="../data/stage_p3c_land")
    ap.add_argument("--out", default="../data/stage_p4d_tables")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    r, _ = run(a.p4_3, a.p3, a.p3c, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)