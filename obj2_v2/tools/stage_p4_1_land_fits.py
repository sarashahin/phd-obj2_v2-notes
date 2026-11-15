#!/usr/bin/env python3
"""
stage_p4_1_land_fits.py  (obj2_v2, P4 step P4-1, 14 Nov 2026; rule: claude/obj2_v2_H_rules_draft.md, last section,
signed by Sara on 13 Nov; option rule (b) signed by Sara on 14 Nov)
Version 2 (14 Nov, after Sara's version 1 run stopped at Q5 with no land-only number printed): Q5 now counts each
structural rule apart; a line before Q5 counts crossed intervals, whose upper end lies below the lower end (land-only
cross-fit, G3's stored window cross-fit, G5a's fits applied to the window AOO, the P4 fits in-sample). --crossed says
what happens to them (stored in p4_fits.npz, so P4-2 and P4-3 apply the same): stop (default; Q5 fails on any crossed
interval, as version 1), empty (kept as G3's code makes them, an empty interval that G3's evaluate counts as a miss;
counted and reported, not a Q5 failure), collapse (both ends set to the point where they meet). Only after Sara's yes.

Question: with sea and shore tetrads taken out (the land mask L_w of each stack's window, from P4-0), do the G3 level
correction and 90% interval, the G3b recalibration and option B still hold on real_val, cross-fitted on G3's community
folds; and what are the fits on all real_val stacks that P4-2 applies once to real_eval? Nothing of real_eval is opened.

Rules (signed; written before any land-only number)
    Land-only AOO of a map = occupied tetrads of L_w at the target year; the same for the true map. EOO is not used.
    Methods: model seed 1, model seed 2, pooled (first 25 maps of each seed), B3 half-blank, B4, B5: the G3 maps.
    Correction and interval: G3's crossfit, evaluate and rule_pass unchanged, on G3's folds; the AOO lower end at least
    the cells recorded at the target year (every record lies inside L_w, gate Q3). Grouping option by G3's own rule (Sara
    14 Nov, (b)): option 1 (recorded class) if it passes, else option 2 (recorded class x records bin) if it passes, else
    the method fails the rule and is reported with option 1. Rule: ratio 0.8-1.25 and coverage 0.85-0.95 in every
    recorded class and records bin with 50 or more stacks. A FAIL is reported as FAIL.
    P(AOO <= x): G3's p_le on the corrected maps; G3b's recalibrate (cross-fitted); option B (G5a's option_b: the
    recalibrated P at x 1 and 5, G3's P at x 10 and 20, running maximum); scored with G3b's evaluate (Brier and skill
    against the other fold's base rate per recorded class).
    Fits for real_eval: G5a's fit_correction (chosen option, all real_val stacks) and fit_recal (on the cross-fitted P, as
    G5a), the Brier reference = the real_val land-only share with AOO <= x per recorded class (G5a's base_rates).
    Reported, not tested: rows by land-only true class; all-land against coastal windows; widths over the model.
    Interval ends (version 2): G3's code raises the upper end to the floor but not to the lower end, so with Q < 0 an
    interval whose 5th-95th percentile spread is under 2|Q| is crossed (upper end below lower end). G3's rule says "so no
    interval is empty". --crossed empty keeps G3's and G5a's code unchanged: such an interval holds no value, G3's
    evaluate counts it as a miss and its width (U - L) as negative. --crossed collapse sets both ends to the point where
    the two shrinking ends meet, the middle of the scaled maps' 5th and 95th percentiles, not below the cells recorded at
    the target year. Either way no other interval changes, and the corrected mean and P do not change.

Gates
    Q0 inputs: P4-0 report and p4_windows.npz = Sara's run, made from this frozen set; G1 samples and G2 maps = their
       reports; g3_calibrated.npz, g3b_recal.npz and g5_fits.npz = their reports and each made from the one before; the
       G3, G3b and G5a tool files = the md5s G5a logged; input and truth files = the manifest
    Q1 alignment: target-year AOO of every map = G3's stored AOO (every method); true AOO from the truth files = G3's;
       rec_t, recorded class and folds = G3's
    Q2 code: on the window AOO this tool's calls give G3's stored corrected mean, interval ends, scale, Q and P, G3b's
       recalibrated P and G5a's AOO correction and recalibration fits (largest difference below 1e-9)
    Q3 land-only counts: two paths agree for every map and truth; land-only <= window AOO; equal in all-land windows;
       every target-year record inside L_w; true land-only AOO >= recorded cells
    Q4 leak: each fold's land-only corrected AOO and recalibrated P (model seed 1) unchanged when that fold's truth is
       shuffled
    Q5 fits: stack counts per fitting group = counts over all real_val stacks; applied in-sample: no crossed interval
       (allowed and counted with --crossed empty), lower end >= recorded cells, P in [0, 1], 0 where the recorded cells exceed x, non-decreasing, option B as stated;
       the file reloads equal
    Q6 np.load log: no real_eval file opened
Outputs (new folder): p4_fits.npz, p4_1_crossfit.npz (per stack), p4_1_report.json.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_p4_1_land_fits.py --self-test          (needs ../tests/test_31_p4_1_land_fits.py, test_30)
    python3 -u ../tools/stage_p4_1_land_fits.py --crossed empty|collapse > ../logs/stage_p4_1_land_fits.txt 2>&1
Needs stage_g5_fits.py, stage_g_calibrate.py, stage_g_recalibrate.py and the tools they load.
"""
import argparse, csv, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_p4_1_land_fits.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


G5F = _load("stage_g5_fits")
GC, GR = G5F.GC, G5F.GR
X, D = GC.X, GC.D
TCLS, BINS, TRUE, XS = GC.TCLS, GC.BINS, GC.TRUE, GC.XS
NC, NB, NX, TB0, G = len(TCLS), len(BINS), len(XS), GC.TB0, GC.G
TOL = 1e-9
CROSSED = {"stop": "Q5 fails on any crossed interval, as version 1",
           "empty": "kept as G3's code makes them: no value inside, counted as a miss, counted and reported",
           "collapse": "both ends set to max(middle of the raw ends, recorded cells)"}
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
EXPECT = dict(p4_0_report="842697b1acf4f8d6484e6d0f659d8b73", p4_windows="928cf96c9bcbe9a14bb0f639adb581eb",
              g3_prefix="b65c80ef", fits_prefix="77ed594e")          # Sara's P4-0 run; G3 and G5a outputs as logged


def broken():
    return os.environ.get("STAGE_P4_1_BROKEN", "")   # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rname(kk):
    return G5F.report_name(kk)


class Guard:
    """While active, every np.load path is logged; any file under a real_eval folder is refused."""
    def __enter__(self):
        self.orig, self.paths, self.refused = np.load, [], []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        p = str(Path(f).resolve()) if isinstance(f, (str, Path)) else str(f)
        if "real_eval" in p:
            self.refused.append(p)
            raise RuntimeError(f"refused np.load of {p}")
        self.paths.append(p)
        return self.orig(f, *a, **k)

    def __exit__(self, *e):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- pieces
def pc(b):
    return POP[b].sum(-1, dtype=np.int64)


def land_counts(tb, lb, ch=4000):
    """tb (n, K, 50) target-year bytes, lb (n, 50) land-mask bytes -> window AOO, land-only AOO by two paths (n, K)."""
    full, a_, b_ = (np.zeros(tb.shape[:2], np.int64) for _ in range(3))
    for a in range(0, len(tb), ch):
        t, l_ = tb[a:a + ch], lb[a:a + ch, None, :]
        full[a:a + ch] = pc(t)
        a_[a:a + ch] = pc(t & l_)
        b_[a:a + ch] = full[a:a + ch] - pc(t & ~l_)
    return full, a_, b_


def choose_option(ok1, ok2):
    """G3's grouping rule, word for word from stage_g_calibrate.run."""
    return 1 if ok1 else (2 if ok2 else 1)


def options(rcls, rbin):
    n = len(rcls)
    return {1: (rcls, np.full(n, -1)), 2: (rcls * NB + rbin, rcls)}


def p_chain(scaled, y, rec_t, rcls, fold):
    """G3's P, G3b's cross-fitted recalibration and option B, with G3b's scores."""
    p_g3 = np.stack([GC.p_le(scaled, x) for x in XS], 1)
    p_rec, _, raised = GR.recalibrate(p_g3, y, rec_t, rcls, fold)
    p_b = G5F.option_b(p_g3, p_rec)
    ev = {lab: {x: GR.evaluate(P[:, j], y, fold, rcls, x) for j, x in enumerate(XS)}
          for lab, P in (("g3", p_g3), ("recal", p_rec), ("b", p_b))}
    return p_g3, p_rec, p_b, ev, raised


def save_fits(path, fits):
    """G5a's save_fits layout, AOO only."""
    a = dict(methods=np.array(fits["methods"]), xs=np.array(XS), base_rate=fits["base_rate"], base_rate_all=fits["base_rate_all"],
             base_n=fits["base_n"], communities=fits["communities"], g3_md5=np.array(fits["g3_md5"]), g3b_md5=np.array(fits["g3b_md5"]),
             p4_windows_md5=np.array(fits["p4_windows_md5"]), crossed=np.array(str(fits["crossed"])))
    for kk, t in fits["corr"].items():
        for q in ("option", "scale", "q", "src", "nfit"):
            a[f"corr__{kk}__aoo__{q}"] = np.asarray(t[q])
    for kk, t in fits["recal"].items():
        for q in ("w", "w_all", "src", "nclass", "nall", "events"):
            a[f"recal__{kk}__{q}"] = np.asarray(t[q])
    X.save_npz(path, a)


def load_fits(path):
    z = np.load(path)
    methods = [str(m) for m in z["methods"]]
    corr = {kk: dict(option=int(z[f"corr__{kk}__aoo__option"]), **{q: z[f"corr__{kk}__aoo__{q}"] for q in ("scale", "q", "src", "nfit")})
            for kk in methods}
    recal = {kk: {q: z[f"recal__{kk}__{q}"] for q in ("w", "w_all", "src", "nclass", "nall", "events")} for kk in methods}
    return dict(methods=methods, corr=corr, recal=recal, base_rate=z["base_rate"], base_rate_all=z["base_rate_all"],
                base_n=z["base_n"], communities=z["communities"], g3_md5=str(z["g3_md5"]), g3b_md5=str(z["g3b_md5"]),
                p4_windows_md5=str(z["p4_windows_md5"]), crossed=str(z["crossed"]))


def same_fits(a, b):
    eq = lambda u, v: np.array_equal(np.asarray(u), np.asarray(v), equal_nan=np.asarray(u).dtype.kind == "f")
    return a["methods"] == b["methods"] and all(eq(a[k], b[k]) for k in ("base_rate", "base_rate_all", "base_n", "communities")) \
        and all(a[k] == b[k] for k in ("g3_md5", "g3b_md5", "p4_windows_md5", "crossed")) \
        and all(eq(a["corr"][kk][q], b["corr"][kk][q]) for kk in a["corr"] for q in ("option", "scale", "q", "src", "nfit")) \
        and all(eq(a["recal"][kk][q], b["recal"][kk][q]) for kk in a["recal"] for q in a["recal"][kk])


def collapse(iv, rec_t):
    """Crossed intervals (upper end below lower end): both ends set to the middle of the raw ends, (p5 - Q + p95 + Q) / 2 =
    (p5 + p95) / 2 of the scaled maps, not below the recorded cells (the AOO floor). Returns their number."""
    crossed = iv["U"] < iv["L"]
    point = np.maximum((iv["Lraw"] + iv["Uraw"]) / 2, np.maximum(rec_t, 0))
    iv["L"] = np.where(crossed, point, iv["L"]); iv["U"] = np.where(crossed, point, iv["U"])
    return int(crossed.sum())


def apply_land(fits, kk, aoo_land, rec_t, rcls, rbin, crossed=None):
    """What P4-2 and P4-3 apply: G5a's apply_correction (AOO), crossed intervals collapsed if the fits say so, G3's p_le,
    G5a's apply_recal and option B."""
    ca = G5F.apply_correction(np.asarray(aoo_land, float), rec_t, rcls, rbin, fits["corr"][kk], "aoo")
    cc = (fits["crossed"] if crossed is None else crossed) == "collapse"
    raised = int((ca["U"] < ca["L"]).sum())
    if cc:
        collapse(ca, rec_t)
    p_g3 = np.stack([GC.p_le(ca["scaled"], x) for x in XS], 1)
    p_rec = G5F.apply_recal(p_g3, rec_t, rcls, fits["recal"][kk])
    return dict(aoo=ca, p_g3=p_g3, p_recal=p_rec, p_b=G5F.option_b(p_g3, p_rec), inverted=raised)


def fmt_rows(rows):
    return "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}/{r['width']:.1f}" + ("" if r["n"] >= GC.N_TEST else "*") for r in rows)


# ---------------------------------------------------------------- run
def run(data="../data", runs=("../runs/f_seed1", "../runs/f_seed2"), g2="../data/stage_g2_baselines_v2",
        g3="../data/stage_g3_calibration", g3b="../data/stage_g3b_recalibration", g5="../data/stage_g5_fits",
        p4_0="../data/stage_p4_0", out_dir="../data/stage_p4_1", expect=None, crossed="stop", out=print):
    if crossed not in CROSSED:
        sys.exit(f"STOP: --crossed must be one of {list(CROSSED)}")
    expect = dict(EXPECT, **(expect or {}))
    data, g2, g3, g3b, g5, p4_0, od = map(Path, (data, g2, g3, g3b, g5, p4_0, out_dir))
    runs = [Path(r) for r in runs]
    val = data / "real_val"
    if od.exists():
        sys.exit(f"STOP: {od} exists; P4-1 writes into a new folder")
    if "real_eval" in str(od.resolve()) or "_truth" in str(od.resolve()):
        sys.exit("STOP: the output path may not contain real_eval or _truth")
    for p_ in (val, g2, g3, g3b, g5, p4_0, *runs):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    bf = {"B3": "maps_B3_halfblank.npz", "B4": "maps_B4.npz", "B5": "maps_B5.npz"}
    need = [data / "stage_d_frozen_val.npz", val / "manifest.csv", p4_0 / "p4_0_report.json", p4_0 / "p4_windows.npz",
            g2 / "g2_report.json", g3 / "g3_report.json", g3 / "g3_calibrated.npz", g3b / "g3b_report.json", g3b / "g3b_recal.npz",
            g5 / "g5_fits_report.json", g5 / "g5_fits.npz", *[g2 / f for f in bf.values()],
            *[r / "check_g1" / f for r in runs for f in ("samples_k50.npz", "g1_report.json")]]
    miss = [str(f) for f in need if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    t0, res, rep = time.time(), {}, dict(lines=[])

    def say(s):
        rep["lines"].append(s); out(s)

    def gate(name, ok, text):
        res[name] = bool(ok); say(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        first = next((k_ for k_, v_ in res.items() if not v_), after)
        say(f"stage P4-1: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped at {first}, nothing written")
        return res, rep
    tmd5 = {f: X.file_md5(HERE / f) for f in ("stage_p4_1_land_fits.py", "stage_g5_fits.py", "stage_g_calibrate.py", "stage_g_recalibrate.py")}
    say("; ".join(f"{k} md5 {v}" for k, v in tmd5.items()) + f"; crossed intervals: {crossed} ({CROSSED[crossed]})")
    with Guard() as lg:
        try:
            # ================================================================ Q0 inputs
            m = {k: X.file_md5(p) for k, p in (("frozen", data / "stage_d_frozen_val.npz"), ("p4_0", p4_0 / "p4_0_report.json"),
                                                 ("p4w", p4_0 / "p4_windows.npz"), ("g3", g3 / "g3_calibrated.npz"),
                                                 ("g3b", g3b / "g3b_recal.npz"), ("g5", g5 / "g5_fits.npz"))}
            r40, r3, r3b, r5 = (json.loads(p.read_text()) for p in (p4_0 / "p4_0_report.json", g3 / "g3_report.json",
                                                                     g3b / "g3b_report.json", g5 / "g5_fits_report.json"))
            pz = dict(np.load(p4_0 / "p4_windows.npz"))
            ok_p4 = m["p4_0"] == expect["p4_0_report"] and m["p4w"] == expect["p4_windows"] == r40["md5"]["out_npz"] \
                and all(r40["gates"].values()) and str(pz["val_frozen_md5"]) == m["frozen"]
            ok_g1 = all(X.file_md5(r / "check_g1" / "samples_k50.npz") == json.loads((r / "check_g1" / "g1_report.json").read_text())["md5"]["samples"]
                        for r in runs)
            j2 = json.loads((g2 / "g2_report.json").read_text())
            ok_g2 = all(X.file_md5(g2 / f) == j2["md5"][f] for f in bf.values())
            ok_g = m["g3"] == r3["md5"]["calibrated"] and m["g3"].startswith(expect["g3_prefix"]) and m["g3b"] == r3b["md5"]["recal"] \
                and r3b["md5"]["input"] == m["g3"] and m["g5"] == r5["md5"]["fits"] and m["g5"].startswith(expect["fits_prefix"]) \
                and r5["md5"]["g3"] == m["g3"] and r5["md5"]["g3b"] == m["g3b"] and all(r5["gates"].values())
            ok_t = tmd5["stage_g5_fits.py"] == r5["md5"]["tool"] and tmd5["stage_g_calibrate.py"] == r5["md5"]["calibrate_tool"] \
                and tmd5["stage_g_recalibrate.py"] == r5["md5"]["recalibrate_tool"]
            fz = dict(np.load(data / "stage_d_frozen_val.npz"))
            with open(val / "manifest.csv") as f_:
                man = {r_["input_file"]: r_ for r_ in csv.DictReader(f_)}
            used = sorted(set(int(x) for x in fz["file_index"]))
            bad_f = [str(fz["files"][f]) for f in used if X.file_md5(val / str(fz["files"][f])) != man[str(fz["files"][f])]["input_md5"]
                     or X.file_md5(val / man[str(fz["files"][f])]["truth_file"]) != man[str(fz["files"][f])]["truth_md5"]]
            if broken() == "changed_input":
                ok_g = False
            gate("Q0", ok_p4 and ok_g1 and ok_g2 and ok_g and ok_t and not bad_f,
                 f"P4-0 report {m['p4_0'][:8]} and p4_windows.npz {m['p4w'][:8]} = Sara's run, gates PASS, frozen set {m['frozen'][:8]} "
                 f"{ok_p4}; G1 samples and G2 maps = their reports {ok_g1 and ok_g2}; g3_calibrated.npz {m['g3'][:8]}, g3b_recal.npz "
                 f"{m['g3b'][:8]}, g5_fits.npz {m['g5'][:8]} = reports and chained {ok_g}; G3, G3b, G5a tools = G5a's record {ok_t}; "
                 f"input and truth files = manifest (differ {len(bad_f)})")
            if not res["Q0"]:
                return stop("Q0")
            # ================================================================ Q1 alignment
            z3, z3b, f5 = dict(np.load(g3 / "g3_calibrated.npz")), dict(np.load(g3b / "g3b_recal.npz")), G5F.load_fits(g5 / "g5_fits.npz")
            if broken() == "eval_open":
                np.load(sorted((data / "real_eval").glob("*.npz"))[0])
            keys = sorted(k[5:] for k in z3 if k.startswith("p_le_"))
            order = [k for k in keys if k.startswith("model_")] + [k for k in ("pooled", "B3", "B4", "B5") if k in keys]
            s1, s2 = f"model_{runs[0].name}", f"model_{runs[1].name}"
            n = len(fz["row"])
            rec_t, rcls, rbin, fold, comm = z3["rec_t"], z3["rcls"], z3["rbin"], z3["fold"], z3["comm"]
            ya = z3["true_aoo"].astype(float)
            tb = {}
            for kk, p in ((s1, runs[0] / "check_g1" / "samples_k50.npz"), (s2, runs[1] / "check_g1" / "samples_k50.npz"),
                          *[(b, g2 / f) for b, f in bf.items()]):
                zz = np.load(p)
                if not (np.array_equal(zz["sel"], np.arange(n)) and zz["maps"].shape[0] == n):
                    tb[kk] = None; continue
                tb[kk] = np.ascontiguousarray(zz["maps"][:, :, TB0:])
            ok_m = order == [s1, s2, "pooled", "B3", "B4", "B5"] and all(tb[k] is not None for k in tb)
            if ok_m:
                tb["pooled"] = np.concatenate([tb[s1][:, :25], tb[s2][:, :25]], 1)
            # truth and records of the target year, read from the files
            tt, rr = np.zeros((n, 50), np.uint8), np.zeros((n, 50), np.uint8)
            fi, rw = fz["file_index"].astype(int), fz["row"].astype(int)
            for f in used:
                k_ = np.nonzero(fi == f)[0]; nm = str(fz["files"][f])
                with np.load(val / man[nm]["truth_file"]) as zt:
                    tt[k_] = zt["true_maps"][rw[k_], -1]
                with np.load(val / nm) as zi:
                    rr[k_] = zi["visit1"][rw[k_], -1] | zi["visit2"][rw[k_], -1]
            ok_a = ok_m and all(np.array_equal(pc(tb[k]), z3[f"aoo_{k}"].astype(np.int64)) for k in order)
            ok_y = np.array_equal(pc(tt).astype(float), ya)
            ok_r = n == len(ya) and np.array_equal(rec_t, fz["rec_t"]) and np.array_equal(pc(rr), rec_t.astype(np.int64)) \
                and np.array_equal(rcls, D.tidx(rec_t)) and np.array_equal(fold, GC.make_folds(comm))
            gate("Q1", ok_a and ok_y and ok_r, f"{n:,} frozen stacks; methods {order} {ok_m}; target-year AOO of every map = G3's "
                 f"stored AOO {ok_a}; true AOO from the truth files = G3's {ok_y}; rec_t (frozen set and inputs), recorded class "
                 f"and folds = G3's {ok_r}")
            if not res["Q1"]:
                return stop("Q1")
            # ================================================================ Q2 code on the window AOO
            opt = options(rcls, rbin)
            worst, wtxt = 0.0, []
            for k in order:
                o_ = int(r3["chosen"][f"{rname(k)} aoo"]["option"])
                pred = z3[f"aoo_{k}"].astype(float)
                cf = GC.crossfit(pred, ya, rec_t, fold, *opt[o_], "aoo")
                pl = np.stack([GC.p_le(cf["scaled"], x) for x in XS], 1)
                d = max(*(float(np.abs(cf[q] - z3[f"aoo_{q}_{k}"]).max()) for q in ("mean", "L", "U")),
                        float(np.abs(cf["par"] - z3[f"aoo_scale_q_{k}"]).max()), float(np.abs(pl - z3[f"p_le_{k}"]).max()))
                p1, _, _ = GR.recalibrate(z3b[f"p_le_{k}"].astype(float), ya, rec_t, rcls, fold)
                d = max(d, float(np.abs(p1 - z3b[f"p_recal_{k}"]).max()))
                tc_ = G5F.fit_correction(pred, ya, rcls, rbin, o_, np.ones(n, bool))
                t5 = f5["corr"][(k, "aoo")]
                same_c = t5["option"] == o_ and all(np.array_equal(tc_[q], t5[q]) for q in ("src", "nfit"))
                d = max(d, float(np.abs(tc_["scale"] - t5["scale"]).max()), float(np.abs(tc_["q"] - t5["q"]).max()))
                tr_ = G5F.fit_recal(z3b[f"p_le_{k}"].astype(float), ya, rec_t, rcls, np.ones(n, bool))
                same_r = np.array_equal(tr_["src"], f5["recal"][k]["src"]) and np.allclose(tr_["w"], f5["recal"][k]["w"], rtol=0, atol=TOL, equal_nan=True)
                if not (same_c and same_r):
                    d = max(d, 1.0)
                worst = max(worst, d); wtxt.append(f"{rname(k)} {d:.1e}")
            gate("Q2", worst < TOL, "this tool's calls on the window AOO give G3's corrected mean, interval ends, scale, Q and P, G3b's "
                 f"recalibrated P and G5a's AOO correction and recalibration fits (largest difference {worst:.1e}; {'; '.join(wtxt)})")
            if not res["Q2"]:
                return stop("Q2")
            # ================================================================ Q3 land-only counts
            win = pz["val_window"].astype(int)
            if broken() == "mask_shift":
                win = (win + 1) % len(pz["window_names"])
            land = pz["land"].astype(bool)
            lbw = np.packbits(land.reshape(len(land), -1), axis=-1)
            lb = lbw[win]
            coastal = ~land.all((1, 2))[win]
            aoo_land, ok3, why3 = {}, True, []
            for k in order:
                full, a_, b_ = land_counts(tb[k], lb)
                if broken() == "no_mask" and k == s1:
                    a_ = full.copy()
                good = np.array_equal(a_, b_) and bool((a_ <= full).all()) and np.array_equal(a_[~coastal], full[~coastal])
                ok3 &= good; aoo_land[k] = a_
                why3.append(f"{rname(k)} {good}")
            yfull, yA, yB = land_counts(tt[:, None, :], lb)
            yA, yB = yA[:, 0], yB[:, 0]
            if broken() == "truth_nomask":
                yA = yfull[:, 0].copy()
            yl = yA.astype(float)
            rec_in = not bool((rr & ~lb).any())
            ok_t3 = np.array_equal(yA, yB) and bool((yA <= yfull[:, 0]).all()) and bool((yl >= rec_t).all()) and rec_in
            gate("Q3", ok3 and ok_t3, "land-only AOO by two paths equal, at most the window AOO, equal in all-land windows: "
                 + "; ".join(why3) + f"; truth the same and at least the recorded cells, every target-year record inside L_w {ok_t3}")
            if not res["Q3"]:
                return stop("Q3")
            # ================================================================ land-only cross-fit, option by G3's rule
            groups = {"recorded class": (rcls, TCLS), "records over 6 years": (rbin, BINS), "true class": (D.tidx(yl.astype(int)), TRUE)}
            per, chosen, pch = {}, {}, {}
            for k in order:
                pred = aoo_land[k].astype(float)
                tri = {}
                for o_ in (1, 2):
                    cf = GC.crossfit(pred, yl, rec_t, fold, *opt[o_], "aoo")
                    inv = int((cf["U"] < cf["L"]).sum())
                    if crossed == "collapse":
                        collapse(cf, rec_t)
                    rows = GC.evaluate(yl, cf, groups)
                    ok, bad = GC.rule_pass(rows)
                    tri[o_] = dict(cf=cf, rows=rows, ok=ok, bad=bad, inverted=inv)
                use = choose_option(tri[1]["ok"], tri[2]["ok"])
                chosen[k] = dict(option=use, ok=tri[use]["ok"], bad=tri[use]["bad"], opt1_ok=tri[1]["ok"], opt2_ok=tri[2]["ok"],
                                 inverted={1: tri[1]["inverted"], 2: tri[2]["inverted"]})
                per[k] = tri[use]
                pch[k] = p_chain(tri[use]["cf"]["scaled"], yl, rec_t, rcls, fold)
            # ================================================================ Q4 leak (model seed 1)
            leak_ok = True
            o1 = chosen[s1]["option"]
            for c in (0, 1):
                idx = np.nonzero(fold == c)[0]
                y2 = yl.copy(); y2[idx] = yl[X.rng(77, 41, c).permutation(idx)]
                a1 = GC.crossfit(aoo_land[s1].astype(float), yl, rec_t, fold, *opt[o1], "aoo", only=c)
                b1 = GC.crossfit(aoo_land[s1].astype(float), y2, rec_t, fold, *opt[o1], "aoo", only=c)
                leak_ok &= all(np.array_equal(a1[q][idx], b1[q][idx]) for q in ("mean", "L", "U"))
                pa, _, _ = GR.recalibrate(pch[s1][0], yl, rec_t, rcls, fold, only=c)
                pb_, _, _ = GR.recalibrate(pch[s1][0], y2, rec_t, rcls, fold, only=c)
                leak_ok &= np.array_equal(pa[idx], pb_[idx])
            gate("Q4", leak_ok, f"each fold's land-only corrected AOO and recalibrated P ({rname(s1)}) unchanged when that fold's own "
                 f"truth is shuffled")
            # ================================================================ fits on all real_val stacks
            use_m = np.ones(n, bool) if broken() != "fits_subset" else fold == 0
            corr = {k: G5F.fit_correction(aoo_land[k].astype(float), yl, rcls, rbin, chosen[k]["option"], use_m) for k in order}
            recal = {k: G5F.fit_recal(pch[k][0], yl, rec_t, rcls, use_m) for k in order}
            br, ball, bn = G5F.base_rates(yl, rcls)
            fits = dict(methods=list(order), corr=corr, recal=recal, base_rate=br, base_rate_all=ball, base_n=bn,
                        communities=np.unique(comm), g3_md5=m["g3"], g3b_md5=m["g3b"], p4_windows_md5=m["p4w"],
                        crossed=crossed)
            cnt_ok = True
            for k, t in corr.items():
                fg = G5F.group_of(rcls, rbin, t["option"])
                for g in range(len(t["scale"])):
                    exp = {0: int((fg == g).sum()), 1: int((rcls == g // NB).sum()), 2: n}[int(t["src"][g])]
                    cnt_ok &= int(t["nfit"][g]) == exp and np.isfinite(t["scale"][g]) and np.isfinite(t["q"][g])
            for k, t in recal.items():
                for j, x in enumerate(XS):
                    pos = rec_t <= x
                    cnt_ok &= int(t["nall"][j]) == int(pos.sum())
            ins, parts = {}, {q: {} for q in ("crossed", "lower below recorded", "P outside [0, 1]", "P falls with x",
                                              "P above 0 where recorded > x", "option B")}
            for k in order:
                r_ = apply_land(fits, k, aoo_land[k], rec_t, rcls, rbin); ins[k] = r_
                pb, pr, pg = r_["p_b"], r_["p_recal"], r_["p_g3"]
                parts["crossed"][k] = int((r_["aoo"]["U"] < r_["aoo"]["L"]).sum())
                parts["lower below recorded"][k] = int((r_["aoo"]["L"] < rec_t).sum())
                parts["P outside [0, 1]"][k] = int(sum(((P < 0) | (P > 1)).any(1).sum() for P in (pr, pb)))
                parts["P falls with x"][k] = int(sum((np.diff(P, axis=1) < 0).any(1).sum() for P in (pr, pb)))
                parts["P above 0 where recorded > x"][k] = int(((rec_t[:, None] > np.array(XS)[None, :]) & (pr != 0)).any(1).sum())
                parts["option B"][k] = int(((pb[:, 0] != pr[:, 0]) | (pb[:, 1] != pr[:, 1]) | (pb[:, 2] != np.maximum(pr[:, 1], pg[:, 2]))
                                            | (pb[:, 3] != np.maximum(pb[:, 2], pg[:, 3]))).sum())
            struct = all(v == 0 for q, d_ in parts.items() for v in d_.values() if q != "crossed" or crossed != "empty")
            # interval ends only (no truth): upper end below lower end, land-only cross-fit, G3's window cross-fit, G5a on the window
            win_ins = {k: G5F.apply_correction(z3[f"aoo_{k}"].astype(float), rec_t, rcls, rbin, f5["corr"][(k, "aoo")], "aoo") for k in order}
            say("   crossed intervals, upper end below lower end (stacks; interval ends only): land-only cross-fit, chosen option, "
                "before any collapse: " + ", ".join(f"{rname(k)} {chosen[k]['inverted'][chosen[k]['option']]}" for k in order)
                + "; G3's stored window cross-fit: " + ", ".join(f"{rname(k)} {int((z3[f'aoo_U_{k}'] < z3[f'aoo_L_{k}']).sum())}" for k in order)
                + "; G5a's fits applied to the window AOO: " + ", ".join(f"{rname(k)} {int((v['U'] < v['L']).sum())}" for k, v in win_ins.items())
                + "; P4 fits applied in-sample, before any collapse: " + ", ".join(f"{rname(k)} {ins[k]['inverted']}" for k in order))
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("Q6", False, f"np.load log: {crashed[:150]}")
        return stop("Q6")
    od.mkdir(parents=True)
    save_fits(od / "p4_fits.npz", fits)
    back = load_fits(od / "p4_fits.npz")
    mf = X.file_md5(od / "p4_fits.npz")
    gate("Q5", cnt_ok and struct and same_fits(fits, back), f"fitting-group counts = counts over all {n:,} real_val stacks {cnt_ok}; "
         f"applied in-sample, stacks breaking each rule: " + "; ".join(f"{q} " + ", ".join(f"{rname(k)} {v}" for k, v in d_.items())
                                                                 for q, d_ in parts.items()) + f"; p4_fits.npz {mf} reloads equal")
    gate("Q6", not lg.refused and all("real_eval" not in p for p in lg.paths),
         f"files passed to np.load {len(lg.paths)}; under real_eval 0 (refused {len(lg.refused)})")
    if not all(res.values()):
        for f_ in od.iterdir():
            f_.unlink()
        od.rmdir()
        return stop("Q6")
    # ================================================================ report (results, not gates)
    zero = lambda m_: float((yl[m_] == 0).mean()) if m_.any() else float("nan")
    say(f"   truth: land-only / window true AOO in coastal windows {yl[coastal].sum() / max(ya[coastal].sum(), 1):.3f} "
        f"({int(coastal.sum()):,} stacks); true land-only AOO 0 in {zero(np.ones(n, bool)):.3f} of all stacks, {zero(coastal):.3f} "
        f"in coastal windows (window AOO 0: {float((ya == 0).mean()):.3f}); map land-only / window AOO in coastal windows, seed 1 "
        f"{aoo_land[s1][coastal].sum() / max(pc(tb[s1][coastal]).sum(), 1):.3f}")
    say(f"   land-only AOO after correction (check folds; ratio / coverage / median width; rule ratio {GC.BAND[0]}-{GC.BAND[1]}, coverage "
        f"{GC.COV[0]}-{GC.COV[1]} in every group of {GC.N_TEST}+ stacks; option by G3's rule):")
    for k in order:
        ch = chosen[k]
        say(f"      {rname(k):<14} option {ch['option']} rule {'PASS' if ch['ok'] else 'FAIL'}" + ("" if ch["ok"] else f" ({', '.join(ch['bad'][:6])})")
            + f"; option 1 {ch['opt1_ok']}, option 2 {ch['opt2_ok']}")
        for gname in ("recorded class", "records over 6 years"):
            say(f"         {gname:<20} {fmt_rows(per[k]['rows'][gname])}")
    say("      stacks per group (* = under 50, reported, not tested): " + " | ".join(
        "; ".join(f"{r_['group']} {r_['n']}" for r_ in per[s1]["rows"][g_]) for g_ in ("recorded class", "records over 6 years")))
    say("   by land-only true class (report only; ratio / coverage):")
    for k in order:
        say(f"      {rname(k):<14} " + "; ".join(f"{r_['group']} {r_['ratio']:.2f}/{r_['cov']:.2f}" for r_ in per[k]["rows"]["true class"]))
    split = {}
    say("   all-land against coastal windows, by recorded class (report only; ratio / coverage, check folds):")
    for k in order:
        split[k] = {}
        for lab, msk in (("all land", ~coastal), ("coastal", coastal)):
            cf = per[k]["cf"]
            sub = {q: cf[q][msk] for q in ("mean", "L", "U")}
            rows = GC.evaluate(yl[msk], sub, {"recorded class": (rcls[msk], TCLS)})["recorded class"]
            split[k][lab] = rows
            say(f"      {rname(k):<14} {lab:<9} " + "; ".join(f"{r_['group']} {r_['ratio']:.2f}/{r_['cov']:.2f}" + ("" if r_["n"] >= GC.N_TEST else "*")
                                                       for r_ in rows))
    mw = {r_["group"]: (r_["width"] + q_["width"]) / 2 for r_, q_ in zip(per[s1]["rows"]["recorded class"], per[s2]["rows"]["recorded class"])}
    say("   width over the model (method median width / mean of the two seeds' median widths), by recorded class: " + " | ".join(
        f"{k} " + "; ".join(f"{r_['group']} {r_['width'] / mw[r_['group']]:.2f}" if mw.get(r_["group"], 0) > 0 else f"{r_['group']} n/a"
                            for r_ in per[k]["rows"]["recorded class"]) for k in ("pooled", "B3", "B4", "B5")))
    say("   P(AOO <= x), land-only, check folds (Brier / skill against the other fold's base rate per recorded class / largest gap); "
        "G3's P -> recalibrated -> option B:")
    for k in order:
        ev = pch[k][3]
        say(f"      {rname(k):<14} " + "; ".join(f"x {x}: " + " -> ".join(f"{ev[s_][x]['brier']:.4f} / {ev[s_][x]['bss']:+.3f} / {ev[s_][x]['gap']:.3f}"
                                                                         for s_ in ("g3", "recal", "b")) for x in XS))
    for k in (s1, s2):
        ev = pch[k][3]["b"]
        say(f"      {rname(k):<14} option B mean P vs share, recorded classes 0 and 1: " + "; ".join(
            f"x {x}: " + ", ".join(f"{g_} {ev[x]['per_class'][g_][0]:.3f} vs {ev[x]['per_class'][g_][1]:.3f}" for g_ in ("0", "1") if g_ in ev[x]["per_class"])
            for x in XS))
    say("   scale on all real_val stacks, land-only against G5a's window fit, option 1 methods by recorded class: " + " | ".join(
        f"{rname(k)} " + "; ".join(f"{TCLS[g_]} {corr[k]['scale'][g_]:.3f} vs {f5['corr'][(k, 'aoo')]['scale'][g_]:.3f}" for g_ in range(NC))
        for k in order if corr[k]["option"] == 1 and f5["corr"][(k, "aoo")]["option"] == 1))
    say("   in-sample check of the all-stack fits (the fitted stacks themselves, not a test; ratio / coverage by recorded class): " + " | ".join(
        f"{rname(k)} " + "; ".join(f"{TCLS[g_]} {ins[k]['aoo']['mean'][rcls == g_].mean() / max(yl[rcls == g_].mean(), 1e-12):.2f}/"
                                   f"{((yl >= ins[k]['aoo']['L']) & (yl <= ins[k]['aoo']['U']))[rcls == g_].mean():.2f}" for g_ in range(NC) if (rcls == g_).any())
        for k in order))
    # ================================================================ files
    cz = dict(true_land=yl, true_window=ya, rec_t=rec_t, rcls=rcls, rbin=rbin, fold=fold, comm=comm, coastal=coastal, window=win)
    for k in order:
        cz[f"aoo_land_{k}"] = aoo_land[k].astype(np.uint16)
        for q in ("mean", "L", "U"):
            cz[f"{q}_{k}"] = per[k]["cf"][q]
        cz[f"scale_q_{k}"] = per[k]["cf"]["par"]
        for lab, P in (("p_g3", pch[k][0]), ("p_recal", pch[k][1]), ("p_b", pch[k][2])):
            cz[f"{lab}_{k}"] = P
    X.save_npz(od / "p4_1_crossfit.npz", cz)
    mc = X.file_md5(od / "p4_1_crossfit.npz")
    verdict = ", ".join(f"{rname(k)} {'PASS' if chosen[k]['ok'] else 'FAIL'} (option {chosen[k]['option']})" for k in order)
    n_ok = sum(res.values())
    last = (f"stage P4-1: gates {n_ok} PASS, {len(res) - n_ok} FAIL; land-only rule on real_val (cross-fitted): {verdict}; crossed "
            f"intervals: {crossed}; p4_fits.npz {mf[:8]}; real_eval not opened")
    say(last)
    rep.update(gates=res, written_utc=now(), md5=dict(m, tools=tmd5, fits=mf, crossfit=mc), n=int(n), crossed=crossed,
               inverted=dict(land_crossfit={rname(k): chosen[k]["inverted"] for k in order},
                             g3_window_crossfit={rname(k): int((z3[f"aoo_U_{k}"] < z3[f"aoo_L_{k}"]).sum()) for k in order},
                             g5a_window={rname(k): int((v["U"] < v["L"]).sum()) for k, v in win_ins.items()},
                             p4_insample={rname(k): ins[k]["inverted"] for k in order}),
               chosen={rname(k): v for k, v in chosen.items()}, rows={rname(k): v["rows"] for k, v in per.items()},
               split={rname(k): v for k, v in split.items()},
               p_le={rname(k): {s_: {str(x): {q: v for q, v in d.items() if q != "table"} for x, d in pch[k][3][s_].items()}
                                for s_ in ("g3", "recal", "b")} for k in order},
               base_rate={str(x): {TCLS[g_]: float(br[j, g_]) for g_ in range(NC)} for j, x in enumerate(XS)},
               minutes=round((time.time() - t0) / 60, 2))
    (od / "p4_1_report.json").write_text(json.dumps(rep, indent=1))
    out(f"   p4_1_crossfit.npz md5 {mc}; p4_1_report.json md5 {X.file_md5(od / 'p4_1_report.json')}")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--runs", nargs=2, default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    ap.add_argument("--g3b", default="../data/stage_g3b_recalibration")
    ap.add_argument("--g5", default="../data/stage_g5_fits")
    ap.add_argument("--p4-0", default="../data/stage_p4_0")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--out", default="../data/stage_p4_1")
    ap.add_argument("--crossed", choices=list(CROSSED), default="stop", help="what happens to crossed intervals (Sara's choice needed)")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_31_p4_1_land_fits.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_31_p4_1_land_fits", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.main(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.data, a.runs, a.g2, a.g3, a.g3b, a.g5, a.p4_0, a.out, crossed=a.crossed, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)