#!/usr/bin/env python3
"""
stage_p3_bnm_products.py  (obj2_v2, products step P3, 13 Nov 2026; products points signed by Sara on 13 Nov, with her
answers of 13 Nov: pooled = the first 25 maps of each seed, EOO left out, P = 0 at x 10 and 20 above the recorded
tetrads, AOO capped at 400 tetrads; claude/obj2_v2_H_rules_draft.md)

Question: for every species-window of the 101 BNM windows (P1 export, all records, no holes; P2 samples), the corrected
window AOO at 2019 with its 90% interval and P(window AOO <= x) at x 1, 2, 5, 10, 20, 124, from the frozen G5a fits applied
unchanged. These are products, not a test: true AOO is unknown on BNM and nothing is scored.

Rules (signed; nothing here changes them)
    AOO of a map: occupied window tetrads at 2019 (stage_g_calibrate.map_metrics, the code G5d used).
    Methods: pooled = the first 25 maps of each seed (50 maps; as fitted in G5a and checked in G5d and H6) is the product;
        seed 1 and seed 2 (50 maps each) are given beside it as a stability check.
    stage_g5_fits.apply_all, unchanged, with the G5a fits of that method: corrected mean, 90% interval (lower end at least
        the 2019 recorded tetrads), option B P at x 1, 5, 10, 20, exactly as in G5d.
    Zero (Sara, 13 Nov): P = 0 where the 2019 recorded tetrads exceed x. Option B already has it at x 1 and 5; at x 10
        and 20 it is set on option B's value, which is otherwise unchanged.
    x 2 and 124 (IUCN B2 cut-offs inside one window: under 10 and under 500 km2 of 2 x 2 km tetrads; not checked on
        simulation): stage_g_calibrate.p_le on the corrected maps, 0 where the 2019 recorded tetrads exceed x; then P(2)
        is held between P(1) and P(5), and P(124) at or above P(20).
    Cap (Sara, 13 Nov): the corrected mean and both interval ends at most 400 tetrads (the window); P is unchanged.
    EOO is left out of the products (it failed G5d); a stated limit.
    Beside each row: H6's test-half H-R2 ratio and coverage of its recorded class, per method (recorded counts in the
        holes, classes by kept records; not true AOO); for recorded classes 0 and 1 the 1-cell note.
    Reported: values set to 0 at x 10 and 20, and rows whose mean or interval ends were capped at 400.

Gates
    C0 inputs: P1 h1_report.json, P2 h2a_report.json, h3_freeze.json and h6_report.json md5 = Sara's runs; P1 gates PASS,
       half all, file md5s = its report, input files = manifest; P2 half all, gates and parts PASS, truth opened 0, files =
       its list, frozen md5 = P1, P1 tool = P2's h1_tool, H2a tool and best.pt = frozen; g5_fits.npz md5 = G5a report =
       G5b record (prefix 77ed594e), G5a and G5b gates PASS, fit methods = the six G5d methods, fits made from
       g3_calibrated.npz = frozen md5; stage_g5_fits.py, stage_g_calibrate.py, stage_g_recalibrate.py = the files that
       wrote the fits; G1 real_val samples md5 = G1 report = frozen
    C1 alignment: rows = P1 stacks; 2019 recorded tetrads and 6-year records recounted from the input files = bnm_frozen;
       classes; window, half and row = bnm_stacks and manifest; every row blanked; samples in frozen order with the layout,
       K 50, frozen md5 and the frozen sampler (K, seed, DDIM, batch, epoch, best.pt); both seeds drawn from the same
       inputs; AOO from map_metrics = the plain target-year count, every map
    C2 code path on real_val: target-year AOO of the G1 samples = g3_calibrated.npz's AOO arrays for both seeds, and the
       first 25 of each = its pooled array (the arrays the G5a fits were fitted on)
    C3 product rules: pooled = first 25 maps of each seed (50); recorded tetrads <= lower end <= upper end <= 400, mean
       <= 400; P in [0, 1] and non-decreasing over the six x; P = 0 at every x below the recorded tetrads; the x 1 and 5
       columns = apply_all's option B, x 10 and 20 = option B or 0 above the recorded tetrads; the files read back = the
       arrays in memory, and the H-R2 columns = h6_report.json
    C4 no *_truth file and no file in a real_eval folder passed to np.load; every file loaded is an input listed above
    Refused: an output folder that exists, lies inside an input folder, or has real_eval or _truth in its path.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_p3_bnm_products.py --self-test | tail -1      (needs ../tests/test_27_p3_products.py)
    python3 -u ../tools/stage_p3_bnm_products.py > ../logs/stage_p3_products.txt 2>&1
Needs stage_g5_fits.py, stage_g_calibrate.py, stage_g_recalibrate.py and the stage tools they load; for the self-test also
stage_h1_bnm_export.py, stage_p1_bnm_all_export.py and --obj1-tools (bnm_detection_check.py).
"""
import argparse, csv, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_p3_bnm_products.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


G5F = _load("stage_g5_fits")
GC = G5F.GC
GB, GS, D, X = GC.GB, GC.GS, GC.D, GC.X
G, W = X.G, X.WINDOW
TCLS, BINS, XS = GC.TCLS, GC.BINS, GC.XS
XS6 = (1, 2, 5, 10, 20, 124)
K, HALF = 50, 25
CELLS = G * G
METH = ("pooled", "s1", "s2")
EXPECT = dict(p1_report="569c9a397574997ce818a14dbfd21eff",     # P1 h1_report.json, Sara's run 13 Nov
              p2_report="0dc83e6b25d32ae8a1d726289fd8e0df",     # P2 h2a_report.json, Sara's run 13 Nov
              freeze="5516a6fc9e1cfcda3ce1a883aa5467f9",        # H3 h3_freeze.json, Sara's run 12 Nov
              h6_report="5142027a04e61aab88a89939bf91b4b0",     # H6 h6_report.json, Sara's run 12 Nov
              fits_prefix="77ed594e")                           # G5a g5_fits.npz, Sara's run 11 Nov
NOTE_1CELL = ("recorded class 0-1: if the true range is 1 tetrad, the corrected AOO came out 7.1-7.4 times too large on "
              "simulation (G5d real_eval, true class 1, both seeds)")


def broken():
    return os.environ.get("STAGE_P3_BROKEN", "")    # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LoadGuard:
    """While active, every np.load path is logged; a *_truth file or a file in a real_eval folder is refused."""
    def __enter__(self):
        self.orig, self.paths, self.refused = np.load, [], []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        p = str(Path(f).resolve()) if isinstance(f, (str, Path)) else str(f)
        if "_truth" in Path(p).name or "real_eval" in p:
            self.refused.append(p)
            raise RuntimeError(f"refused np.load of {p}")
        self.paths.append(p)
        return self.orig(f, *a, **k)

    def __exit__(self, *e):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- products
def six_p(p_b, scaled, rec_t):
    """P(AOO <= x) at XS6. x 1, 5: option B. x 10, 20: option B, 0 where the recorded tetrads exceed x (Sara, 13 Nov).
    x 2, 124: G3's p_le on the corrected maps, 0 where the recorded tetrads exceed x, then P(2) held between P(1) and
    P(5) and P(124) at or above P(20)."""
    p10 = np.where(rec_t > 10, 0.0, p_b[:, 2]) if broken() != "no_zero" else p_b[:, 2]
    p20 = np.where(rec_t > 20, 0.0, p_b[:, 3]) if broken() != "no_zero" else p_b[:, 3]
    p2 = np.where(rec_t > 2, 0.0, GC.p_le(scaled, 2))
    p124 = np.where(rec_t > 124, 0.0, GC.p_le(scaled, 124))
    p2 = np.minimum(np.maximum(p2, p_b[:, 0]), p_b[:, 1])
    p124 = np.maximum(p124, p20)
    return np.column_stack([p_b[:, 0], p2, p_b[:, 1], p10, p20, p124])


def cap400(ca):
    """Corrected mean and interval ends at most the 400 tetrads of a window (Sara, 13 Nov)."""
    if broken() == "no_cap":
        return ca["mean"], ca["L"], ca["U"]
    return np.minimum(ca["mean"], CELLS), np.minimum(ca["L"], CELLS), np.minimum(ca["U"], CELLS)


def pooled_of(a1, a2):
    """G5d's pooled line: the first 25 maps of each seed."""
    if broken() == "pool100":
        return np.concatenate([a1, a2], 1)
    return np.concatenate([a1[:, :HALF], a2[:, :HALF]], 1)


def aoo_count(packed, ch=1024):
    """Plain target-year count of every map, (n, K, 300) -> (n, K)."""
    return np.concatenate([GC.target_year(packed[a:a + ch]).sum((2, 3)) for a in range(0, len(packed), ch)]) \
        if len(packed) else np.zeros((0, packed.shape[1]), int)


def h6_rows(r6, method):
    """H6's H-R2 rows by recorded class for one method; a missing number (null in the json) becomes NaN."""
    fl = lambda v: float("nan") if v is None else float(v)
    return {r["group"]: dict(ratio=fl(r["ratio"]), cov=fl(r["cov"]), stacks=int(r["stacks"]))
            for r in r6["H_R2"][method]["rows"]["recorded class"]}


def same(a, b, tol=0.0):
    """Equal arrays (strings exactly; numbers within tol, NaN equal to NaN)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "USO" or b.dtype.kind in "USO":
        return bool(np.array_equal(a.astype(str), b.astype(str)))
    a, b = a.astype(float), b.astype(float)
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb) and np.all(np.abs(a[~na] - b[~nb]) <= tol))


# ---------------------------------------------------------------- run
def run(bnm="../data/bnm_all", p2="../data/stage_p2_predict", fits_dir="../data/stage_g5_fits",
        g5b="../data/stage_g5b_export_check", g3="../data/stage_g3_calibration",
        h6="../data/stage_h6_test_score/h6_report.json", freeze="../data/stage_h3_freeze/h3_freeze.json",
        runs=("../runs/f_seed1", "../runs/f_seed2"), out_dir="../data/stage_p3_products", expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    bnm, p2, fits_dir, g5b, g3, h6, freeze, od = map(Path, (bnm, p2, fits_dir, g5b, g3, h6, freeze, out_dir))
    runs = [Path(r) for r in runs]
    if len(runs) != 2:
        sys.exit("STOP: P3 needs the two seeds (--runs seed1 seed2)")
    if od.exists():
        sys.exit(f"STOP: {od} exists; P3 writes into a new folder")
    if "real_eval" in str(od.resolve()) or "_truth" in str(od.resolve()):
        sys.exit("STOP: the output path may not contain real_eval or _truth")
    for p_ in (bnm, p2, fits_dir, g5b, g3, h6.parent, freeze.parent, *runs):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    need_f = [bnm / "h1_report.json", bnm / "bnm_frozen.npz", bnm / "bnm_stacks.npz", bnm / "manifest.csv", p2 / "h2a_report.json",
              fits_dir / "g5_fits.npz", fits_dir / "g5_fits_report.json", g5b / "g5b_export_check.json", g3 / "g3_calibrated.npz",
              h6, freeze] + [f for r in runs for f in (r / "best.pt", r / "check_g1" / "g1_report.json", r / "check_g1" / "samples_k50.npz")]
    miss = [str(f) for f in need_f if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    t0, res, rep = time.time(), {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage P3: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, rep
    names = ("stage_g5_fits.py", "stage_g_calibrate.py", "stage_g_recalibrate.py")
    tmd5 = {nm: X.file_md5(HERE / nm) for nm in names}
    out(f"stage_p3_bnm_products.py md5 {X.file_md5(Path(__file__))}; " + ", ".join(f"{k} {v}" for k, v in tmd5.items()))
    k1, k2 = (f"model_{r.name}" for r in runs)
    with LoadGuard() as lg:
        try:
            # ================================================================ C0 inputs
            js1, m1 = json.loads((bnm / "h1_report.json").read_text()), X.file_md5(bnm / "h1_report.json")
            fmd5, smd5, mmd5 = (X.file_md5(bnm / f) for f in ("bnm_frozen.npz", "bnm_stacks.npz", "manifest.csv"))
            man_rows = D.rows_of(bnm)
            bad_in = [r["input_file"] for r in man_rows if X.file_md5(bnm / r["input_file"]) != r["input_md5"]]
            p1_ok = m1 == expect["p1_report"] and js1.get("tool_name") == "stage_p1_bnm_all_export.py" and js1["half"] == "all" \
                and all(js1["gates"].values()) and js1["md5"]["bnm_frozen.npz"] == fmd5 and js1["md5"]["bnm_stacks.npz"] == smd5 \
                and js1["md5"]["manifest.csv"] == mmd5 and not bad_in
            fz, mfz = json.loads(freeze.read_text()), X.file_md5(freeze)
            ha, m2 = json.loads((p2 / "h2a_report.json").read_text()), X.file_md5(p2 / "h2a_report.json")
            need = [f"samples_{r.name}.npz" for r in runs]
            fbad = [f for f, m in ha["files"].items() if not (p2 / f).is_file() or X.file_md5(p2 / f) != m]
            best = {r.name: X.file_md5(r / "best.pt") for r in runs}
            best_ok = all(ha["best_md5"].get(f"samples_{r.name}") == best[r.name] == fz["sampler"][r.name]["best_md5"] for r in runs)
            p2_ok = m2 == expect["p2_report"] and ha["half"] == "all" and all(ha["gates"].values()) and ha["truth_opened"] == 0 \
                and all(all(v["gates"].values()) for v in ha["parts"].values()) and not fbad and all(f in ha["files"] for f in need) \
                and ha["frozen_md5"] == fmd5 and ha["h1_tool"] == js1["md5"]["tool"] \
                and ha["tool"] == fz["tools"]["stage_h2_bnm_predict.py"] and best_ok
            fr = json.loads((fits_dir / "g5_fits_report.json").read_text())
            g5bj = json.loads((g5b / "g5b_export_check.json").read_text())
            mf = X.file_md5(fits_dir / "g5_fits.npz")
            fits = G5F.load_fits(fits_dir / "g5_fits.npz")
            g3md5 = X.file_md5(g3 / "g3_calibrated.npz")
            methods = [k1, k2, "pooled", "B3", "B4", "B5"]
            code_ok = tmd5["stage_g5_fits.py"] == fr["md5"]["tool"] and tmd5["stage_g_calibrate.py"] == fr["md5"]["calibrate_tool"] \
                and tmd5["stage_g_recalibrate.py"] == fr["md5"]["recalibrate_tool"]
            fits_ok = mf == fr["md5"]["fits"] == g5bj["md5"].get("fits") and mf.startswith(expect["fits_prefix"]) \
                and all(fr["gates"].values()) and all(g5bj["gates"].values()) and fits["methods"] == methods \
                and fits["g3_md5"] == g3md5 == fz["data"]["stage_g3_calibration/g3_calibrated.npz"] and code_ok
            r6, m6 = json.loads(h6.read_text()), X.file_md5(h6)
            h6_ok = m6 == expect["h6_report"] and all(r6["gates"].values()) and r6.get("rerun") is False and \
                all(set(h6_rows(r6, m_)) <= set(TCLS) for m_ in METH)
            g1_ok, g1md5 = True, {}
            for r in runs:
                g1 = json.loads((r / "check_g1" / "g1_report.json").read_text())
                g1md5[r.name] = X.file_md5(r / "check_g1" / "samples_k50.npz")
                g1_ok &= g1md5[r.name] == g1["md5"]["samples"] == fz["data"][f"{r.name}/check_g1/samples_k50.npz"] \
                    and g1["md5"]["best_pt"] == best[r.name]
            gate("C0", p1_ok and mfz == expect["freeze"] and p2_ok and fits_ok and h6_ok and g1_ok,
                 f"P1 h1_report.json {m1[:8]} = Sara's run {m1 == expect['p1_report']}, gates PASS, half all, files = report, "
                 f"input files = manifest (differ {len(bad_in)}) {p1_ok}; h3_freeze.json {mfz[:8]} = signed {mfz == expect['freeze']}; "
                 f"P2 h2a_report.json {m2[:8]} = Sara's run {m2 == expect['p2_report']}, half all, gates and parts PASS, truth "
                 f"opened {ha['truth_opened']}, files = list (differ {len(fbad)}), frozen md5 = P1, P1 tool = h1_tool, H2a tool "
                 f"and best.pt = frozen {p2_ok}; g5_fits.npz {mf[:8]} = G5a report = G5b record, G5a and G5b gates PASS, "
                 f"methods {fits['methods']}, made from g3_calibrated.npz {g3md5[:8]} = frozen, apply code = the code that wrote "
                 f"the fits {code_ok}: {fits_ok}; h6_report.json {m6[:8]} = Sara's run, gates PASS, first reading {h6_ok}; G1 "
                 f"real_val samples = G1 report = frozen {g1_ok}")
            if not res["C0"]:
                return stop("C0")
            # ================================================================ C1 alignment
            fb = dict(np.load(bnm / "bnm_frozen.npz")); n = len(fb["row"])
            idx = dict(np.load(bnm / "bnm_stacks.npz"))
            man = {r["input_file"]: r for r in man_rows}
            V1, V2, _, _, _ = GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])
            V = V1 | V2
            del V1, V2
            rec_t = V[:, -1].reshape(n, -1).sum(1) + (1 if broken() == "rec_shift" else 0)
            rec6 = V.any(1).reshape(n, -1).sum(1)
            del V
            rcls, rbin = D.tidx(rec_t), D.bidx(rec6)
            winf = np.array([man[str(fb["files"][i])]["window"] for i in fb["file_index"]])
            halff = np.array([man[str(fb["files"][i])]["half"] for i in fb["file_index"]])
            al = n == int(js1["stacks"]) and np.array_equal(rec_t, fb["rec_t"]) and np.array_equal(rec6, fb["rec6"]) \
                and np.array_equal(rcls, fb["cls"]) and all(len(idx[k_]) == n for k_ in ("window", "half", "species", "status", "row")) \
                and np.array_equal(winf, idx["window"].astype(str)) and np.array_equal(halff, idx["half"].astype(str)) \
                and np.array_equal(fb["row"], idx["row"]) and bool((fb["blank"] == 1).all()) and bool((rec6 >= 1).all())
            maps, why, ok1, inm = {}, [], True, []
            for m_, r in zip(("s1", "s2"), runs):
                z = np.load(p2 / f"samples_{r.name}.npz")
                smp = fz["sampler"][r.name]
                good = np.array_equal(z["sel"], np.arange(n)) and str(z["layout"]) == GS.LAYOUT and str(z["frozen_md5"]) == fmd5 \
                    and int(z["K"]) == K and z["maps"].shape == (n, K, W * CELLS // 8) \
                    and all(int(z[k_]) == int(smp[k_]) for k_ in ("K", "seed", "ddim", "batch", "epoch")) and str(z["best_md5"]) == smp["best_md5"]
                ok1 &= good; why.append(f"{r.name} {good}"); inm.append(str(z["in_md5"]))
                maps[m_] = z["maps"]
            t1 = time.time()
            met = {m_: GC.map_metrics(maps[m_]) for m_ in ("s1", "s2")}
            plain = all(np.array_equal(aoo_count(maps[m_]), met[m_][0]) for m_ in ("s1", "s2"))
            del maps
            gate("C1", al and ok1 and inm[0] == inm[1] and plain,
                 f"{n:,} rows = P1 stacks; 2019 recorded tetrads and 6-year records recounted from the input files = bnm_frozen, "
                 f"classes, window, half and row = bnm_stacks and manifest, every row blanked, every row with a record {al}; "
                 f"samples in frozen order with layout, K {K}, frozen md5 and the frozen sampler: {', '.join(why)}; same inputs "
                 f"for both seeds {inm[0] == inm[1]} ({inm[0][:8]}); AOO from map_metrics = plain target-year count {plain} "
                 f"({(time.time() - t1) / 60:.1f} min)")
            if not res["C1"]:
                return stop("C1")
            # ================================================================ C2 code path on real_val
            zg = np.load(g3 / "g3_calibrated.npz")
            av, c2 = {}, []
            for r in runs:
                pk = np.load(r / "check_g1" / "samples_k50.npz")["maps"]
                if broken() == "g1_shift" and r == runs[0]:
                    pk = np.roll(pk, 1, axis=0)
                av[r.name] = aoo_count(pk)
                del pk
                c2.append(np.array_equal(av[r.name], zg[f"aoo_model_{r.name}"]))
            pv = np.concatenate([av[runs[0].name][:, :HALF], av[runs[1].name][:, :HALF]], 1)
            c2.append(np.array_equal(pv, zg["aoo_pooled"]))
            gate("C2", all(c2), f"real_val ({len(pv):,} stacks): target-year AOO of the G1 samples = g3_calibrated.npz "
                 f"aoo_{k1} {c2[0]}, aoo_{k2} {c2[1]}; first 25 of each seed = aoo_pooled {c2[2]}")
            del av, pv, zg
            if not res["C2"]:
                return stop("C2")
            # ================================================================ products
            met["pooled"] = tuple(pooled_of(met["s1"][q], met["s2"][q]) for q in (0, 1))
            key = {"pooled": "pooled", "s1": k1, "s2": k2}
            if broken() == "fits_swap":
                key = {"pooled": "pooled", "s1": k2, "s2": k1}
            R = {}
            for m_ in METH:
                a = G5F.apply_all(fits, key[m_], met[m_][0], met[m_][1], rec_t, rcls, rbin)
                ca = a["aoo"]
                mean_c, L_c, U_c = cap400(ca)
                R[m_] = dict(raw=met[m_][0].mean(1), mean=mean_c, L=L_c, U=U_c, raised=ca["L"] > ca["Lraw"],
                             capped=(ca["mean"] > CELLS) | (ca["L"] > CELLS) | (ca["U"] > CELLS),
                             P=six_p(a["p_b"], ca["scaled"], rec_t), p_b=a["p_b"], p_g3=a["p_g3"], p_recal=a["p_recal"])
            h6r = {m_: h6_rows(r6, m_) for m_ in METH}
            lab = np.array([TCLS[c] for c in rcls])
            lab_h6 = lab if broken() != "h6_class" else np.array([TCLS[min(c + 1, len(TCLS) - 1)] for c in rcls])
            H6 = {m_: (np.array([h6r[m_][g]["ratio"] if g in h6r[m_] else np.nan for g in lab_h6]),
                       np.array([h6r[m_][g]["cov"] if g in h6r[m_] else np.nan for g in lab_h6])) for m_ in METH}
            note = np.where(rcls <= 1, NOTE_1CELL, "")
            # ================================================================ C3 product rules (in memory)
            shp = met["pooled"][0].shape == (n, 2 * HALF) and np.array_equal(met["pooled"][0][:, :HALF], met["s1"][0][:, :HALF]) \
                and np.array_equal(met["pooled"][0][:, HALF:], met["s2"][0][:, :HALF])
            rules, xs6 = {}, np.array(XS6)
            for m_ in METH:
                P = R[m_]["P"]
                pb = R[m_]["p_b"]
                rules[m_] = dict(
                    interval=bool((R[m_]["L"] >= rec_t).all() and (R[m_]["U"] >= R[m_]["L"]).all() and np.isfinite(R[m_]["mean"]).all()),
                    cap=bool((R[m_]["mean"] <= CELLS).all() and (R[m_]["U"] <= CELLS).all()),
                    in01=bool(((P >= 0) & (P <= 1)).all()), rising=bool((np.diff(P, axis=1) >= 0).all()),
                    zeros=bool((P[rec_t[:, None] > xs6[None, :]] == 0).all()),
                    option_b=bool(np.array_equal(P[:, [0, 2]], pb[:, :2]) and np.array_equal(P[:, 3], np.where(rec_t > 10, 0.0, pb[:, 2]))
                                  and np.array_equal(P[:, 4], np.where(rec_t > 20, 0.0, pb[:, 3]))))
            c3_mem = shp and all(all(v.values()) for v in rules.values())
            if broken() == "opentruth":
                np.load(bnm / "bnm_e0000_n0000_truth.npz")
            crashed = None
        except RuntimeError as e:
            if not lg.refused:
                raise
            crashed = str(e)
    if crashed:
        gate("C4", False, f"np.load refused: {crashed[:160]}")
        return stop("C4")
    if not c3_mem:
        gate("C3", False, f"pooled = first 25 maps of each seed {shp}; rules per method {rules}")
        return stop("C3")
    # ================================================================ reports (not tests)
    flag = {m_: {str(x): int(((R[m_]["p_b"][:, j] > 0) & (rec_t > x)).sum()) for j, x in ((2, 10), (3, 20))} for m_ in METH}
    over = {m_: int(R[m_]["capped"].sum()) for m_ in METH}
    out(f"   option B values above 0 set to 0 because the 2019 recorded tetrads exceed x: " + "; ".join(
        f"{m_} " + ", ".join(f"x {x} {v:,}" for x, v in flag[m_].items()) for m_ in METH))
    out(f"   rows whose corrected mean or interval end was capped at {CELLS} tetrads: " + "; ".join(f"{m_} {over[m_]:,}" for m_ in METH)
        + "; lower end raised to the recorded tetrads: " + "; ".join(f"{m_} {R[m_]['raised'].mean():.3f}" for m_ in METH))
    stab = {str(x): dict(mean_abs=float(np.abs(R["s1"]["P"][:, j] - R["s2"]["P"][:, j]).mean()),
                         over_010=float((np.abs(R["s1"]["P"][:, j] - R["s2"]["P"][:, j]) > 0.10).mean())) for j, x in enumerate(XS6)}
    rat = R["s1"]["mean"] / np.maximum(R["s2"]["mean"], 1e-12)
    out("   seed 1 against seed 2 (stability, report): mean |P1 - P2| / share over 0.10: " + "; ".join(
        f"x {x} {v['mean_abs']:.3f} / {v['over_010']:.3f}" for x, v in stab.items()) + f"; corrected mean seed 1 / seed 2, "
        f"median {np.median(rat):.3f} (5-95%: {np.percentile(rat, 5):.3f}-{np.percentile(rat, 95):.3f})")
    status = np.array([s_ if s_ else "none" for s_ in idx["status"].astype(str)])
    summ = {}
    for gname, gv, labs in (("recorded class", lab, TCLS),
                            ("Red List 2022", status, sorted(set(status.tolist())))):
        summ[gname] = {}
        for g_ in labs:
            m = gv == g_
            if not m.any():
                continue
            P = R["pooled"]["P"][m]
            summ[gname][g_] = dict(rows=int(m.sum()), rec_2019_median=float(np.median(rec_t[m])),
                                   aoo_median=float(np.median(R["pooled"]["mean"][m])), lo_median=float(np.median(R["pooled"]["L"][m])),
                                   hi_median=float(np.median(R["pooled"]["U"][m])),
                                   **{f"mean_p_{x}": float(P[:, j].mean()) for j, x in enumerate(XS6)},
                                   **{f"share_p_{x}_over_half": float((P[:, j] >= 0.5).mean()) for j, x in enumerate(XS6) if x in (2, 124)})
        out(f"   pooled products by {gname} (rows; median 2019 records; median corrected AOO [lower, upper]; mean P at x 2 | 124; "
            f"share with P >= 0.5 at x 2 | 124):")
        for g_, v in summ[gname].items():
            out(f"      {g_:<6} {v['rows']:>5,}; {v['rec_2019_median']:.0f}; {v['aoo_median']:.1f} [{v['lo_median']:.1f}, "
                f"{v['hi_median']:.1f}]; {v['mean_p_2']:.3f} | {v['mean_p_124']:.3f}; {v['share_p_2_over_half']:.3f} | "
                f"{v['share_p_124_over_half']:.3f}")
    # ================================================================ files
    od.mkdir(parents=True)
    sav = dict(window=idx["window"].astype(str), half=idx["half"].astype(str), species=idx["species"].astype(str), status=status,
               rec_2019=rec_t.astype(np.int16), rec_6y=rec6.astype(np.int16), rcls=rcls.astype(np.int8), rbin=rbin.astype(np.int8),
               xs6=xs6, frozen_md5=np.array(fmd5), fits_md5=np.array(mf), p2_report_md5=np.array(m2), h6_report_md5=np.array(m6))
    for m_ in METH:
        for q in ("raw", "mean", "L", "U", "P", "p_b", "p_g3", "p_recal", "capped"):
            sav[f"{q}_{m_}"] = R[m_][q]
        sav[f"h6_ratio_{m_}"], sav[f"h6_cov_{m_}"] = H6[m_]
    X.save_npz(od / "products_bnm.npz", sav)
    head = ["window", "half", "species", "red_list_2022", "rec_2019", "rec_6y", "recorded_class", "records_bin"]
    for m_ in METH:
        head += [f"aoo_mean_{m_}", f"aoo_lo90_{m_}", f"aoo_hi90_{m_}"] + [f"p_aoo_le_{x}_{m_}" for x in XS6]
    head += [f"h6_{q}_{m_}" for m_ in METH for q in ("ratio", "cov")] + ["note"]
    wcols = list(METH) if broken() != "csv_swap" else ["pooled", "s2", "s1"]
    with open(od / "products_bnm.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for i in range(n):
            row = [sav["window"][i], sav["half"][i], sav["species"][i], status[i], int(rec_t[i]), int(rec6[i]), TCLS[rcls[i]], BINS[rbin[i]]]
            for m_ in wcols:
                row += [f"{R[m_][q][i]:.3f}" for q in ("mean", "L", "U")] + [f"{v:.4f}" for v in R[m_]["P"][i]]
            row += [f"{H6[m_][q][i]:.3f}" for m_ in METH for q in (0, 1)] + [note[i]]
            w.writerow(row)
    # ================================================================ C3 read-back
    zb = np.load(od / "products_bnm.npz")
    back_npz = all(same(zb[k_], v) for k_, v in sav.items())
    with open(od / "products_bnm.csv") as fh:
        rows_c = list(csv.DictReader(fh))
    col = lambda c: np.array([float(r[c]) for r in rows_c])
    back_csv = len(rows_c) == n and [r["species"] for r in rows_c] == list(sav["species"]) and \
        [r["recorded_class"] for r in rows_c] == list(lab) and \
        all(same(col(f"p_aoo_le_{x}_{m_}"), R[m_]["P"][:, j], 5.01e-5) for m_ in METH for j, x in enumerate(XS6)) and \
        all(same(col(f"aoo_{c}_{m_}"), R[m_][q], 5.01e-4) for m_ in METH for c, q in (("mean", "mean"), ("lo90", "L"), ("hi90", "U")))
    h6back = len(rows_c) == n and all(
        same(col(f"h6_{q}_{m_}"), [h6r[m_][r["recorded_class"]][q] if r["recorded_class"] in h6r[m_] else np.nan for r in rows_c], 5.01e-4)
        for m_ in METH for q in ("ratio", "cov"))
    notes_ok = all((r["note"] == NOTE_1CELL) == (r["recorded_class"] in ("0", "1")) for r in rows_c)
    gate("C3", c3_mem and back_npz and back_csv and h6back and notes_ok,
         f"pooled = first 25 maps of each seed (50) {shp}; per method (interval floor and order, mean and ends <= {CELLS}, P in "
         f"[0, 1], rising over x {', '.join(map(str, XS6))}, 0 at every x below the recorded tetrads, x 1 and 5 = option B, x 10 "
         f"and 20 = option B or 0): "
         + "; ".join(f"{m_} {all(v.values())}" for m_, v in rules.items()) + f"; products_bnm.npz read back = memory {back_npz}; "
         f"products_bnm.csv read back = memory (rounding) {back_csv}; H-R2 columns = h6_report.json {h6back}; 1-cell note at "
         f"classes 0-1 only {notes_ok}")
    # ================================================================ C4 load log
    allowed = [bnm, p2, fits_dir, g3, *[r / "check_g1" for r in runs]]
    other = [p for p in lg.paths if not any(Path(p).resolve().parent == a.resolve() for a in allowed)]
    gate("C4", not lg.refused and not other, f"np.load refused {len(lg.refused)}; files passed to np.load {len(lg.paths)}, other than "
         f"the P1 export, the P2 samples, the G5a fits, g3_calibrated.npz and the G1 samples {len(other)}")
    rep.update(gates=res, n=int(n), note="products, not a test: true AOO is unknown on BNM; nothing is scored",
               rules=dict(pooled="first 25 maps of each seed (Sara, 13 Nov)", eoo="left out (Sara, 13 Nov)",
                          zero="P = 0 where the 2019 recorded tetrads exceed x, at every x (Sara, 13 Nov)",
                          cap="corrected mean and interval ends at most 400 tetrads (Sara, 13 Nov)",
                          x2_x124="not checked on simulation; p_le on the corrected maps, 0 above the recorded tetrads; P(2) "
                                  "held between P(1) and P(5), P(124) at or above P(20)", note_1cell=NOTE_1CELL,
                          h6_columns="test-half H-R2 ratio and coverage of the recorded class (recorded counts in the holes, "
                                     "classes by kept records; not true AOO)"),
               md5=dict(tool=X.file_md5(Path(__file__)), tools=tmd5, p1_report=m1, p2_report=m2, freeze=mfz, h6_report=m6,
                        fits=mf, g3=g3md5, frozen=fmd5, g1=g1md5, best_pt=best, products_npz=X.file_md5(od / "products_bnm.npz"),
                        products_csv=X.file_md5(od / "products_bnm.csv")),
               zeroed=flag, capped=over, raised={m_: float(R[m_]["raised"].mean()) for m_ in METH}, seed_stability=stab,
               seed_mean_ratio=dict(median=float(np.median(rat)), p5=float(np.percentile(rat, 5)), p95=float(np.percentile(rat, 95))),
               summaries=summ, minutes=round((time.time() - t0) / 60, 2), written_utc=now())
    (od / "p3_report.json").write_text(json.dumps(rep, indent=1, default=float))
    n_ok = sum(res.values())
    out(f"   written products_bnm.csv {rep['md5']['products_csv']}, products_bnm.npz {rep['md5']['products_npz']}, p3_report.json "
        f"({(time.time() - t0) / 60:.1f} min)")
    out(f"stage P3: gates {n_ok} PASS, {len(res) - n_ok} FAIL; {n:,} species-windows; products, not a test")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bnm", default="../data/bnm_all")
    ap.add_argument("--p2", default="../data/stage_p2_predict")
    ap.add_argument("--fits", default="../data/stage_g5_fits")
    ap.add_argument("--g5b", default="../data/stage_g5b_export_check")
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    ap.add_argument("--h6", default="../data/stage_h6_test_score/h6_report.json")
    ap.add_argument("--freeze", default="../data/stage_h3_freeze/h3_freeze.json")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--out", default="../data/stage_p3_products")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_27_p3_products.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_27_p3_products", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.main(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.bnm, a.p2, a.fits, a.g5b, a.g3, a.h6, a.freeze, a.runs, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)