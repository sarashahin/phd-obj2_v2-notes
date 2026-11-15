#!/usr/bin/env python3
"""
stage_p4_2_eval_reading.py  (obj2_v2, P4 step P4-2, 14 Nov 2026; rule: claude/obj2_v2_H_rules_draft.md, P4 section,
signed by Sara on 13 Nov; option rule (b) 14 Nov; crossed intervals rule (E) 14 Nov; Sara's yes to P4-2 on 14 Nov)

Question: with the P4-1 fits written and md5-logged (p4_fits.npz 654da879), does the land-only AOO keep G3's rule on
real_eval, the 30 communities no fit has seen? This is the labelled second reading of the real_eval truth (G5d was the
first, 11 Nov). Nothing is fitted here and nothing changes after the reading; a FAIL is reported as FAIL.

What is read
    Before the marker (truth closed; any np.load of a real_eval truth file is refused): P4-0 and P4-1 outputs, the G5c
    prediction files, G5d's predicted file and report (window AOO, as read at the first reading), the real_eval manifest
    and input files. Land-only AOO = occupied tetrads of L_w (P4-0, window matched from the inputs) at the target year,
    for every map. The P4 fits are applied unchanged by P4-1's apply_land (G5a's apply_correction, G3's p_le, G5a's
    apply_recal and option B; crossed intervals kept empty, rule E).
    After the marker: from each real_eval truth file only true_maps at the target year, stack_round, round_ids,
    round_window and window_names. The truth window of each stack is checked against the window matched from the inputs
    (P4 rule). The marker line is appended to G5d's marker file, so one file lists every reading of the real_eval truth.

Rules (as signed; nothing here changes them)
    Land-only rule: G3's evaluate and rule_pass: ratio 0.8-1.25 and 90% interval coverage 0.85-0.95 in every recorded
    class and records bin with 50 or more stacks, per method (seed 1, seed 2, pooled first 25 + 25 maps, B3, B4, B5), at
    the option P4-1 chose. Every method is scored, whatever its P4-1 verdict. No cap at the land tetrads (a product rule).
    A crossed interval (rule E) holds no value and counts as a miss; the count is reported.
    Reported only: rows by land-only true class; all-land against coastal windows; widths over the model; P(AOO <= x) at
    x 1, 5, 10, 20 (G3's P, recalibrated, option B): Brier and skill against the real_val land-only share per recorded
    class (p4_fits base_rate; G5d's p_eval), G3b check rules R1-R3; G5d's window-AOO verdicts and option B skill beside.

Gates
    Phase 1, truth closed
    Z0 one labelled reading: --second-reading given; output folder new and outside real_eval; G5d's marker present, its
       first line = G5d's reading (predicted md5 = g5d_report.json = g5d_predicted.npz); no P4-2 line (or --rerun,
       reported as a rerun)
    Z1 inputs: p4_fits.npz, p4_1_report.json, p4_1_crossfit.npz, p4_windows.npz = Sara's runs; P4-1 gates PASS, rule E
       stored in the report and the fits; P4-1 tool and G3 / G3b / G5a tools = the md5s P4-1 logged; fit communities none
       in real_eval; g5_fits.npz = the G5a fits G5d read; G5c report gates PASS, truth opened 0, prediction files = its list = the md5s G5d read; real_eval
       manifest and every input and truth file = the G5b list (md5 only); frozen set = the one P4-0 matched
    Z2 alignment from inputs only: map files in frozen order (sel, K, frozen md5); window AOO of every map = G5d's stored
       AOO (six methods); recorded cells and records = frozen set = G5d; every frozen stack has a window; land-only AOO by
       two paths equal, at most the window AOO, equal in all-land windows; every target-year record inside L_w
    Z3 code: apply_land with G5a's fits on the window AOO = G5d's corrected mean, interval ends and P (all six methods,
       exactly); G3's evaluate and G5d's p_eval on p4_1_crossfit.npz = P4-1's report rows and P scores (1e-12)
    Z4 np.load log: no real_eval truth file opened in phase 1
    Phase 2, after p4_2_predicted.npz and the marker line are written
    Z5 truth aligned: truth window = matched window for every stack; target-year records inside the true maps; window
       true AOO = G5d's per_stack_g5d.csv; land-only truth by two paths, at most the window truth, at least the records
    Z6 leak: p4_2_predicted.npz unchanged after the truth opened
--check-only runs phase 1 and writes nothing. A second P4-2 reading is refused while a P4-2 marker line exists;
--rerun allows it and every line says rerun.

Outputs (new folder): p4_2_predicted.npz, per_stack_p4_2.csv, p4_2_report.json.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_p4_2_eval_reading.py --self-test      (needs ../tests/test_32_p4_2_eval_reading.py, test_31, test_30)
    python3 -u ../tools/stage_p4_2_eval_reading.py --second-reading --check-only > ../logs/stage_p4_2_check.txt 2>&1
    python3 -u ../tools/stage_p4_2_eval_reading.py --second-reading > ../logs/stage_p4_2_reading.txt 2>&1
Needs stage_p4_1_land_fits.py (and what it loads) in the same folder.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_p4_2_eval_reading.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P41 = _load("stage_p4_1_land_fits")
G5F, GC, GR, X, D = P41.G5F, P41.GC, P41.GR, P41.X, P41.D
TCLS, BINS, TRUE, XS, TB0, G, NB = P41.TCLS, P41.BINS, P41.TRUE, P41.XS, P41.TB0, P41.G, P41.NB
pc, land_counts, apply_land, rname = P41.pc, P41.land_counts, P41.apply_land, P41.rname
MARKER = "G5D_TRUTH_OPENED"                                  # stage_g5_predict.MARKER, in data/real_eval
LABEL = "P4-2 land-only AOO, labelled second reading of real_eval"
EXPECT = dict(fits="654da879cf43a061d7bd685d5636b01e", p4_1_report="5e980763bcd5935ad9bc31dbeb4bd0f1",
              crossfit="371f424a85688b574ba0613472633069", p4_windows="928cf96c9bcbe9a14bb0f639adb581eb",
              g5a_prefix="77ed594e")                         # Sara's P4-1 and P4-0 runs; G5a fits as logged
TOL = 1e-12


def broken():
    return os.environ.get("STAGE_P4_2_BROKEN", "")   # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def arr_md5(*arrays):
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class EvalGuard:
    """While active, every np.load path is logged; a truth file inside the real_eval folder is refused."""
    def __init__(self, ev_dir):
        self.dir = str(Path(ev_dir).resolve())

    def __enter__(self):
        self.orig, self.paths, self.refused = np.load, [], []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        p = str(Path(f).resolve()) if isinstance(f, (str, Path)) else str(f)
        if "_truth" in Path(p).name and p.startswith(self.dir):
            self.refused.append(p)
            raise RuntimeError(f"real_eval truth file opened before the marker: {p}")
        self.paths.append(p)
        return self.orig(f, *a, **k)

    def __exit__(self, *e):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- G5d's scoring pieces, word for word (stage_g5_score.py)
def fold_ref(y, fold, rcls, x):
    """G3 / G3b Brier reference: the other fold's share with AOO <= x per recorded class (for the Z5 replica)."""
    o = (y <= x).astype(float)
    ref = np.zeros(len(y))
    for c in (0, 1):
        fit, chk = fold == 1 - c, fold == c
        for g in np.unique(rcls[chk]):
            m = fit & (rcls == g)
            ref[chk & (rcls == g)] = o[m].mean() if m.any() else o[fit].mean()
    return ref


def p_eval(P, y, rcls, ref, x):
    """G3b's evaluate with the Brier reference given (real_eval: the real_val share per recorded class)."""
    o = (y <= x).astype(float)
    if broken() == "p_ref":
        ref = np.full(len(y), o.mean())
    brier, bref = float(((P - o) ** 2).mean()), float(((ref - o) ** 2).mean())
    b = np.minimum((P * 10).astype(int), 9)
    tab = [dict(bin=k, n=int((b == k).sum()), p=float(P[b == k].mean()) if (b == k).any() else float("nan"),
                obs=float(o[b == k].mean()) if (b == k).any() else float("nan")) for k in range(10)]
    big = [t for t in tab if t["n"] >= GC.N_TEST]
    gaps = [abs(t["p"] - t["obs"]) for t in big]
    zs = [abs(t["p"] - t["obs"]) / np.sqrt(max(t["p"] * (1 - t["p"]), 1e-4) / t["n"]) for t in big]
    gaps200 = [abs(t["p"] - t["obs"]) for t in tab if t["n"] >= GR.R3_BIN]
    per = {TCLS[g]: (float(P[rcls == g].mean()), float(o[rcls == g].mean()), int((rcls == g).sum())) for g in np.unique(rcls)}
    return dict(brier=brier, brier_ref=bref, bss=1 - brier / bref if bref > 0 else float("nan"),
                gap=max(gaps) if gaps else float("nan"), gap_z=float(max(zs)) if zs else float("nan"),
                gap200=max(gaps200) if gaps200 else float("nan"), mean_p=float(P.mean()), share=float(o.mean()),
                per_class=per, table=tab)


def r_rules(before, after):
    r1 = after["brier"] <= before["brier"] + 1e-15
    r2 = all(abs(after["per_class"][g][0] - after["per_class"][g][1]) <= GR.R2_TOL for g in ("0", "1") if g in after["per_class"])
    r3 = not (after["gap200"] > GR.R3_TOL)
    return dict(R1=bool(r1), R2=bool(r2), R3=bool(r3), ok=bool(r1 and r2 and r3))


def dif(a, b):
    """Absolute difference; 0 if both are missing (NaN or None), infinite if only one is."""
    na, nb = a is None or (isinstance(a, float) and np.isnan(a)), b is None or (isinstance(b, float) and np.isnan(b))
    return 0.0 if na and nb else (float("inf") if na or nb else abs(float(a) - float(b)))


def g5a_as_p4(f5, methods):
    """G5a's AOO fits in P4-1's fits layout, rule E (G5a never collapsed an interval)."""
    return dict(corr={k: f5["corr"][(k, "aoo")] for k in methods}, recal={k: f5["recal"][k] for k in methods}, crossed="empty")


def fmt(rows):
    return "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}/{r['width']:.1f}" + ("" if r["n"] >= GC.N_TEST else "*") for r in rows)


# ---------------------------------------------------------------- run
def run(data="../data", g5c="../data/stage_g5_predict", g5b="../data/stage_g5b_export_check", g5d="../data/stage_g5d_score",
        g5a="../data/stage_g5_fits", p4_0="../data/stage_p4_0", p4_1="../data/stage_p4_1", out_dir="../data/stage_p4_2",
        second_reading=False, check_only=False, rerun=False, expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    data, g5c, g5b, g5d, g5a, p4_0, p4_1, od = map(Path, (data, g5c, g5b, g5d, g5a, p4_0, p4_1, out_dir))
    ev = data / "real_eval"
    if not second_reading:
        sys.exit("STOP: P4-2 is the labelled second reading of real_eval; give --second-reading")
    if "real_eval" in str(od.resolve()) or "_truth" in str(od.resolve()):
        sys.exit("STOP: nothing is written inside a real_eval folder or a truth path; give another --out")
    if not check_only and od.exists():
        sys.exit(f"STOP: {od} exists; P4-2 writes into a new folder")
    for p_ in (g5c, g5b, g5d, g5a, p4_0, p4_1):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    need = [ev / MARKER, ev / "manifest.csv", g5c / "g5c_report.json", g5c / "frozen_eval.npz", g5b / "g5b_export_check.json",
            g5d / "g5d_report.json", g5d / "g5d_predicted.npz", g5d / "per_stack_g5d.csv", g5a / "g5_fits.npz", p4_0 / "p4_windows.npz",
            p4_1 / "p4_fits.npz", p4_1 / "p4_1_report.json", p4_1 / "p4_1_crossfit.npz"]
    miss = [str(f) for f in need if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    marker = ev / MARKER
    lines_m = [json.loads(s) for s in marker.read_text().splitlines() if s.strip()]
    ours = [l_ for l_ in lines_m if str(l_.get("label", "")).startswith("P4-2")]
    if ours and not rerun and not check_only:
        sys.exit(f"STOP: the marker holds a P4-2 reading already ({ours[0].get('opened_utc')}); another needs --rerun and is "
                 "reported as a rerun")
    if rerun and not ours:
        sys.exit("STOP: --rerun given but no P4-2 reading is on the marker; the P4-2 reading runs without --rerun")
    tag = f" (rerun {len(ours)})" if ours and not check_only else ""
    t0, res, rep = time.time(), {}, dict(lines=[])

    def say(s):
        rep["lines"].append(s); out(s)

    def gate(name, ok, text):
        res[name] = bool(ok); say(f"{name}{tag} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        first = next((k_ for k_, v_ in res.items() if not v_), after)
        say(f"stage P4-2{tag}: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped at {first}; no marker line written, no real_eval "
            "truth file opened, nothing written")
        return res, rep
    tmd5 = {f: X.file_md5(HERE / f) for f in ("stage_p4_2_eval_reading.py", "stage_p4_1_land_fits.py", "stage_g5_fits.py",
                                              "stage_g_calibrate.py", "stage_g_recalibrate.py")}
    say("; ".join(f"{k} md5 {v}" for k, v in tmd5.items()) + f"; {LABEL}{tag}")
    g5d_rep = json.loads((g5d / "g5d_report.json").read_text())
    m_pred5 = X.file_md5(g5d / "g5d_predicted.npz")
    first_ok = bool(lines_m) and lines_m[0].get("predicted_md5") == g5d_rep["md5"]["predicted"] == m_pred5 \
        and not str(lines_m[0].get("label", "")).startswith("P4-2")
    gate("Z0", first_ok, f"--second-reading given; output folder {od} new and outside real_eval; marker {marker.name}: {len(lines_m)} "
         f"line(s), the first = G5d's reading (predicted md5 {m_pred5[:8]} = g5d_report.json) {first_ok}; P4-2 lines {len(ours)}"
         + ("; check-only: nothing is written" if check_only else (f"; this reading is rerun {len(ours)}" if tag else "")))
    if not res["Z0"]:
        return stop("Z0")
    with EvalGuard(ev) as lg:
        try:
            # ================================================================ Z1 inputs
            m = {k: X.file_md5(p) for k, p in (("fits", p4_1 / "p4_fits.npz"), ("p4_1", p4_1 / "p4_1_report.json"),
                                                 ("cz", p4_1 / "p4_1_crossfit.npz"), ("p4w", p4_0 / "p4_windows.npz"),
                                                 ("frozen", g5c / "frozen_eval.npz"), ("manifest", ev / "manifest.csv"),
                                                 ("g5a", g5a / "g5_fits.npz"))}
            r41 = json.loads((p4_1 / "p4_1_report.json").read_text())
            fits = P41.load_fits(p4_1 / "p4_fits.npz")
            methods = fits["methods"]
            pz = dict(np.load(p4_0 / "p4_windows.npz"))
            ok_p4 = m["fits"] == expect["fits"] == r41["md5"]["fits"] and m["p4_1"] == expect["p4_1_report"] \
                and m["cz"] == expect["crossfit"] == r41["md5"]["crossfit"] and m["p4w"] == expect["p4_windows"] == r41["md5"]["p4w"] \
                and all(r41["gates"].values()) and r41.get("crossed") == "empty" == fits["crossed"] \
                and fits["p4_windows_md5"] == m["p4w"] and str(pz["eval_frozen_md5"]) == m["frozen"]
            t41 = r41["md5"]["tools"]
            ok_t = all(tmd5[f] == t41[f] for f in ("stage_p4_1_land_fits.py", "stage_g5_fits.py", "stage_g_calibrate.py", "stage_g_recalibrate.py"))
            with open(ev / "manifest.csv") as f_:
                man = list(csv.DictReader(f_))
            ev_seeds = {int(r_["sim_seed"]) for r_ in man}
            ok_c = not (set(int(c) for c in fits["communities"]) & ev_seeds) and len(fits["communities"]) > 0
            r5c = json.loads((g5c / "g5c_report.json").read_text())
            pfiles = ["frozen_eval.npz", "maps_B3_halfblank.npz", "maps_B4.npz", "maps_B5.npz"] + \
                [f"samples_{k[6:]}.npz" for k in methods if k.startswith("model_")]
            pm = {f: X.file_md5(g5c / f) if (g5c / f).is_file() else None for f in pfiles}
            ok_5c = all(r5c["gates"].values()) and r5c.get("truth_opened") == 0 and all(
                pm[f] is not None and pm[f] == r5c["files"].get(f) == g5d_rep["md5"]["predictions"].get(f) for f in pfiles)
            g5bj = json.loads((g5b / "g5b_export_check.json").read_text())
            bad_f = [f for r_ in man for f in (r_["input_file"], r_["truth_file"]) if X.file_md5(ev / f) != g5bj["md5"]["files"].get(f)]
            ok_5b = m["manifest"] == g5bj["md5"]["manifest"] and not bad_f and all(g5bj["gates"].values())
            ok_5a = m["g5a"] == g5d_rep["md5"]["fits"] and m["g5a"].startswith(expect["g5a_prefix"])
            if broken() == "changed_input":
                ok_p4 = False
            gate("Z1", ok_p4 and ok_t and ok_c and ok_5c and ok_5b and ok_5a,
                 f"p4_fits.npz {m['fits'][:8]}, p4_1_report.json {m['p4_1'][:8]}, p4_1_crossfit.npz {m['cz'][:8]}, p4_windows.npz "
                 f"{m['p4w'][:8]} = Sara's runs, P4-1 gates PASS, rule E in the report and the fits, frozen set = P4-0's {ok_p4}; P4-1 "
                 f"tool and G3 / G3b / G5a tools = P4-1's record {ok_t}; fit communities {len(fits['communities'])}, none of the "
                 f"{len(ev_seeds)} real_eval communities {ok_c}; G5c report gates PASS, truth opened 0, prediction files = its list = "
                 f"the md5s G5d read {ok_5c}; real_eval manifest and {2 * len(man)} input and truth files = the G5b list (md5 only; "
                 f"differ {len(bad_f)}) {ok_5b}; g5_fits.npz {m['g5a'][:8]} = the G5a fits G5d read {ok_5a}")
            if not res["Z1"]:
                return stop("Z1")
            # ================================================================ Z2 alignment from inputs only
            fe = dict(np.load(g5c / "frozen_eval.npz")); n = len(fe["row"])
            fi, rw = fe["file_index"].astype(int), fe["row"].astype(int)
            used = sorted(set(fi.tolist()))
            by_in = {r_["input_file"]: r_ for r_ in man}
            rr, r6, hab = np.zeros((n, 50), np.uint8), np.zeros(n, np.int64), np.zeros(n, np.int64)
            for f in used:
                k_ = np.nonzero(fi == f)[0]
                with np.load(ev / str(fe["files"][f])) as zi:
                    V = zi["visit1"][rw[k_]] | zi["visit2"][rw[k_]]
                    hab[k_] = zi["stack_hab"][rw[k_]]
                rr[k_] = V[:, -1]
                r6[k_] = pc(np.bitwise_or.reduce(V, axis=1))
            rec_t = pc(rr)
            rcls, rbin = D.tidx(rec_t), D.bidx(r6)
            zp = dict(np.load(g5d / "g5d_predicted.npz"))
            ok_r = np.array_equal(rec_t, fe["rec_t"].astype(np.int64)) and np.array_equal(r6, fe["rec6"].astype(np.int64)) \
                and np.array_equal(rec_t, zp["rec_t"].astype(np.int64)) and np.array_equal(rcls, zp["rcls"]) and np.array_equal(rbin, zp["rbin"])
            tb, why_m, ok_m = {}, [], True
            src = {k: (f"samples_{k[6:]}.npz", 50) for k in methods if k.startswith("model_")}
            src.update({"B3": ("maps_B3_halfblank.npz", 50), "B4": ("maps_B4.npz", 50), "B5": ("maps_B5.npz", 1)})
            for k, (f, kexp) in src.items():
                z = np.load(g5c / f)
                good = np.array_equal(z["sel"], np.arange(n)) and int(z["K"]) == kexp and str(z["frozen_md5"]) == m["frozen"] \
                    and z["maps"].shape[:2] == (n, kexp)
                tb[k] = np.ascontiguousarray(z["maps"][:, :, TB0:]) if good else None
                ok_m &= good; why_m.append(f"{f} {good}")
            s1, s2 = [k for k in methods if k.startswith("model_")]
            if ok_m:
                if broken() == "maps_shift":
                    tb[s1] = np.roll(tb[s1], 1, 0)
                tb["pooled"] = np.concatenate([tb[s1][:, :25], tb[s2][:, :25]], 1)
            ok_a = ok_m and all(np.array_equal(pc(tb[k]), zp[f"aoo_{k}"].astype(np.int64)) for k in methods)
            win = pz["eval_window"].astype(int)
            if broken() == "win_shift":
                win = (win + 1) % len(pz["window_names"])
            land = pz["land"].astype(bool)
            lb = np.packbits(land.reshape(len(land), -1), axis=-1)[win]
            coastal = ~land.all((1, 2))[win]
            aoo_land, ok3 = {}, ok_a
            for k in (methods if ok_a else []):
                full, a_, b_ = land_counts(tb[k], lb)
                ok3 &= np.array_equal(a_, b_) and bool((a_ <= full).all()) and np.array_equal(a_[~coastal], full[~coastal])
                aoo_land[k] = a_
            rec_in = len(win) == n and bool((win >= 0).all()) and not bool((rr & ~lb).any())
            gate("Z2", ok_r and ok_a and ok3 and rec_in,
                 f"{n:,} frozen real_eval stacks; recorded cells and records from the inputs = frozen set = G5d {ok_r}; map files in "
                 f"frozen order with K and frozen md5 ({', '.join(why_m)}); window AOO of every map = G5d's stored AOO, six methods "
                 f"{ok_a}; land-only AOO by two paths equal, at most the window AOO, equal in all-land windows {ok3}; every stack has "
                 f"a window and every target-year record lies inside L_w {rec_in}; coastal windows hold {int(coastal.sum()):,} stacks")
            if not res["Z2"]:
                return stop("Z2")
            # ================================================================ Z3 code
            f5w = g5a_as_p4(G5F.load_fits(g5a / "g5_fits.npz"), methods)
            worst = 0.0
            for k in methods:
                r_ = apply_land(f5w, k, pc(tb[k]).astype(float), rec_t, rcls, rbin)
                if broken() == "apply_code":
                    r_["aoo"]["mean"] = r_["aoo"]["mean"] * 1.001
                d = max(float(np.abs(r_["aoo"][q] - zp[f"aoo_{q}_{k}"]).max()) for q in ("mean", "L", "U"))
                d = max(d, *(float(np.abs(r_[q] - zp[f"{q}_{k}"]).max()) for q in ("p_g3", "p_recal", "p_b")))
                worst = max(worst, d)
            cz = dict(np.load(p4_1 / "p4_1_crossfit.npz"))
            yv, fv, rcv, rbv = cz["true_land"].astype(float), cz["fold"], cz["rcls"], cz["rbin"]
            grp_v = {"recorded class": (rcv, TCLS), "records over 6 years": (rbv, BINS), "true class": (D.tidx(yv.astype(int)), TRUE)}
            wv = 0.0
            for k in methods:
                rows = GC.evaluate(yv, {q: cz[f"{q}_{k}"] for q in ("mean", "L", "U")}, grp_v)
                for gname, rws in rows.items():
                    for a_, b_ in zip(rws, r41["rows"][rname(k)][gname]):
                        wv = max(wv, *(dif(a_[q], b_[q]) for q in ("ratio", "cov", "width")), float(a_["n"] != b_["n"]))
                for lab in ("g3", "recal", "b"):
                    P = cz[f"p_{lab}_{k}"]
                    for j, x in enumerate(XS):
                        e = p_eval(P[:, j].astype(float), yv, rcv, fold_ref(yv, fv, rcv, x), x)
                        rr_ = r41["p_le"][rname(k)][lab][str(x)]
                        wv = max(wv, *(dif(float(e[q]), rr_[q]) for q in ("brier", "brier_ref", "bss", "gap", "gap_z", "gap200", "mean_p", "share")))
            gate("Z3", worst <= TOL and wv <= TOL, f"P4-1's apply_land with G5a's fits on the window AOO = G5d's corrected mean, interval "
                 f"ends, G3's P, recalibrated P and option B (largest difference {worst:.1e}); G3's evaluate and G5d's p_eval on "
                 f"p4_1_crossfit.npz = P4-1's report rows and P scores (largest difference {wv:.1e})")
            if not res["Z3"]:
                return stop("Z3")
            # ---- predictions with the P4 fits (no truth)
            pred = {k: apply_land(fits, k, aoo_land[k].astype(float), rec_t, rcls, rbin) for k in methods}
            save = dict(rec_t=rec_t.astype(np.int16), rcls=rcls.astype(np.int8), rbin=rbin.astype(np.int8), window=win.astype(np.int16),
                        coastal=coastal, frozen_md5=np.array(m["frozen"]), fits_md5=np.array(m["fits"]), xs=np.array(XS))
            for k in methods:
                save[f"aoo_land_{k}"] = aoo_land[k].astype(np.uint16)
                for q in ("mean", "L", "U"):
                    save[f"{q}_{k}"] = pred[k]["aoo"][q]
                for q in ("p_g3", "p_recal", "p_b"):
                    save[f"{q}_{k}"] = pred[k][q]
            say("   predictions with the P4 fits (rule E): " + ", ".join(
                f"{rname(k)} land-only AOO {arr_md5(aoo_land[k])[:8]}, corrected {arr_md5(*[pred[k]['aoo'][q] for q in ('mean', 'L', 'U')])[:8]}, "
                f"P {arr_md5(pred[k]['p_b'])[:8]}, crossed {pred[k]['inverted']}" for k in methods))
            if broken() == "truth_early":
                np.load(ev / man[0]["truth_file"])
            crashed = None
        except RuntimeError as e:
            if not lg.refused:
                raise
            crashed = str(e)
    gate("Z4", not lg.refused and crashed is None, f"files passed to np.load {len(lg.paths)}; real_eval truth files refused "
         f"{len(lg.refused)}" + (f" ({crashed[:100]})" if crashed else ""))
    if not all(res.values()):
        return stop("Z4")
    if check_only:
        say(f"stage P4-2 check-only: gates {sum(res.values())} PASS, 0 FAIL; phase 1 done (Z0-Z4); no marker line, no real_eval "
            f"truth file opened, nothing written ({(time.time() - t0) / 60:.1f} min)")
        return res, rep
    # ================================================================ phase 2: predicted file, marker line, then the truth
    od.mkdir(parents=True)
    pfile = od / "p4_2_predicted.npz"
    X.save_npz(pfile, save)
    mpred = X.file_md5(pfile)
    opened = now()
    with open(marker, "a") as fh:
        fh.write(json.dumps(dict(opened_utc=opened, tool_md5=tmd5["stage_p4_2_eval_reading.py"], out=str(od.resolve()),
                                 predicted_md5=mpred, reading=len(lines_m) + 1, label=LABEL + (f", rerun {len(ours)}" if tag else ""),
                                 fits_md5=m["fits"])) + "\n")
    say(f"   p4_2_predicted.npz md5 {mpred} written; marker line written {opened} (reading {len(lines_m) + 1} of real_eval{tag}); "
        "the truth opens from here")
    tt, trw, tix = np.zeros((n, 50), np.uint8), np.empty(n, object), np.full(n, -1, np.int64)
    for f in used:
        k_ = np.nonzero(fi == f)[0]
        with np.load(ev / by_in[str(fe["files"][f])]["truth_file"], allow_pickle=False) as zt:
            tt[k_] = zt["true_maps"][rw[k_], -1]
            wn, rwin, rid = zt["window_names"].astype(str), zt["round_window"], zt["round_ids"]
            pos = {int(r_): j for j, r_ in enumerate(rid)}
            tix[k_] = [pos.get(int(s_), -1) for s_ in zt["stack_round"][rw[k_]]]      # round index from the truth's own round ids
            trw[k_] = [wn[rwin[j]] if j >= 0 else "" for j in tix[k_]]
    if broken() == "truth_shift":
        tt = np.roll(tt, 1, 0)
    with open(g5d / "per_stack_g5d.csv") as f_:
        ps = list(csv.DictReader(f_))
    ya_5d = np.array([int(r_["aoo_true"]) for r_ in ps])
    pos_5d = [(int(r_["file_index"]), int(r_["row"])) for r_ in ps]
    yfull, yA, yB = land_counts(tt[:, None, :], lb)
    yfull, yA, yB = yfull[:, 0], yA[:, 0], yB[:, 0]
    ok_w = bool((trw == pz["window_names"].astype(str)[win]).all()) and np.array_equal(tix, hab)
    ok_in = not bool((rr & ~tt).any())
    ok_5d = len(ps) == n and pos_5d == list(zip(fi.tolist(), rw.tolist())) and np.array_equal(yfull, ya_5d)
    ok_l = np.array_equal(yA, yB) and bool((yA <= yfull).all()) and bool((yA >= rec_t).all())
    gate("Z5", ok_w and ok_in and ok_5d and ok_l, f"truth window (round ids, round_window, window_names of the truth file) = window "
         f"matched from the inputs, and the truth's round = the input's stack_hab, for every stack {ok_w}; target-year "
         f"records inside the true maps {ok_in}; window true AOO = G5d's per_stack_g5d.csv, same stacks in the same order {ok_5d}; "
         f"land-only truth by two paths, at most the window truth, at least the records {ok_l}")
    if broken() == "leak":
        X.save_npz(pfile, dict(save, rec_t=save["rec_t"] + 1))
    gate("Z6", X.file_md5(pfile) == mpred, f"p4_2_predicted.npz unchanged after the truth opened ({X.file_md5(pfile)[:8]})")
    yl, ya = yA.astype(float), yfull.astype(float)
    # ================================================================ rules
    groups = {"recorded class": (rcls, TCLS), "records over 6 years": (rbin, BINS), "true class": (D.tidx(yA), TRUE)}
    say(f"   truth: land-only / window true AOO in coastal windows {yl[coastal].sum() / max(ya[coastal].sum(), 1):.3f} "
        f"({int(coastal.sum()):,} stacks); map land-only / window AOO in coastal windows, seed 1 "
        f"{aoo_land[s1][coastal].sum() / max(pc(tb[s1][coastal]).sum(), 1):.3f}")
    say(f"   land-only AOO on real_eval with the P4 fits (ratio / coverage / median width; rule ratio {GC.BAND[0]}-{GC.BAND[1]}, coverage "
        f"{GC.COV[0]}-{GC.COV[1]} in every group of {GC.N_TEST}+ stacks; option as chosen in P4-1):")
    rows_i, rule_i = {}, {}
    for k in methods:
        rows_i[k] = GC.evaluate(yl, pred[k]["aoo"], groups)
        ok, bad = GC.rule_pass(rows_i[k])
        rule_i[k] = dict(ok=ok, bad=bad, option=int(fits["corr"][k]["option"]), p4_1=bool(r41["chosen"][rname(k)]["ok"]))
        say(f"      {rname(k):<14} option {rule_i[k]['option']} rule {'PASS' if ok else 'FAIL'}" + ("" if ok else f" ({', '.join(bad[:6])})")
            + f" (P4-1 {'PASS' if rule_i[k]['p4_1'] else 'FAIL'})")
        for gname in ("recorded class", "records over 6 years"):
            say(f"         {gname:<20} {fmt(rows_i[k][gname])}")
    say("      stacks per group (* = under 50, reported, not tested): " + " | ".join(
        "; ".join(f"{r_['group']} {r_['n']}" for r_ in rows_i[s1][g_]) for g_ in ("recorded class", "records over 6 years")))
    crossed = {k: dict(all=int((pred[k]["aoo"]["U"] < pred[k]["aoo"]["L"]).sum()),
                       by_class={TCLS[g_]: int(((pred[k]["aoo"]["U"] < pred[k]["aoo"]["L"]) & (rcls == g_)).sum()) for g_ in range(len(TCLS))})
               for k in methods}
    say("   crossed intervals on real_eval (rule E: no value inside, counted as misses): " + "; ".join(
        f"{rname(k)} {v['all']}" + (" (" + ", ".join(f"{c} {x}" for c, x in v["by_class"].items() if x) + ")" if v["all"] else "")
        for k, v in crossed.items()))
    say("   by land-only true class (report only; ratio / coverage):")
    for k in methods:
        say(f"      {rname(k):<14} " + "; ".join(f"{r_['group']} {r_['ratio']:.2f}/{r_['cov']:.2f}" for r_ in rows_i[k]["true class"]))
    split = {}
    say("   all-land against coastal windows, by recorded class (report only; ratio / coverage):")
    for k in methods:
        split[k] = {}
        for lab, msk in (("all land", ~coastal), ("coastal", coastal)):
            if not msk.any():
                continue
            sub = {q: pred[k]["aoo"][q][msk] for q in ("mean", "L", "U")}
            split[k][lab] = GC.evaluate(yl[msk], sub, {"recorded class": (rcls[msk], TCLS)})["recorded class"]
            say(f"      {rname(k):<14} {lab:<9} " + "; ".join(f"{r_['group']} {r_['ratio']:.2f}/{r_['cov']:.2f}" + ("" if r_["n"] >= GC.N_TEST else "*")
                                                       for r_ in split[k][lab]))
    mw = {r_["group"]: (r_["width"] + q_["width"]) / 2 for r_, q_ in zip(rows_i[s1]["recorded class"], rows_i[s2]["recorded class"])}
    say("   width over the model (method median width / mean of the two seeds' median widths), by recorded class: " + " | ".join(
        f"{rname(k)} " + "; ".join(f"{r_['group']} {r_['width'] / mw[r_['group']]:.2f}" if mw.get(r_["group"], 0) > 0 else f"{r_['group']} n/a"
                                   for r_ in rows_i[k]["recorded class"]) for k in methods if k not in (s1, s2)))
    pe, prule = {}, {}
    say("   P(AOO <= x), land-only, on real_eval (Brier / skill against the real_val land-only share per recorded class / largest gap, "
        "bins 50+ | 200+ / mean P vs share); G3's P -> recalibrated -> option B:")
    for k in methods:
        for j, x in enumerate(XS):
            ref = fits["base_rate"][j][rcls]
            e = {s_: p_eval(pred[k][q][:, j], yl, rcls, ref, x) for s_, q in (("g3", "p_g3"), ("recal", "p_recal"), ("b", "p_b"))}
            pe[(k, x)] = e
            prule[(k, x)] = dict(option_b=r_rules(e["g3"], e["b"]), recal=r_rules(e["g3"], e["recal"]))
        say(f"      {rname(k):<14} " + "; ".join(f"x {x}: " + " -> ".join(
            f"{pe[(k, x)][s_]['brier']:.4f} / {pe[(k, x)][s_]['bss']:+.3f} / {pe[(k, x)][s_]['gap']:.3f} | {pe[(k, x)][s_]['gap200']:.3f} / "
            f"{pe[(k, x)][s_]['mean_p']:.3f} vs {pe[(k, x)][s_]['share']:.3f}" for s_ in ("g3", "recal", "b")) for x in XS))
    say("   G3b check rules (R1 Brier not worse than G3's P; R2 classes 0 and 1 within 0.02; R3 gap over bins of 200+ <= 0.05), "
        "reported; option B | recalibrated P:")
    for k in methods:
        say(f"      {rname(k):<14} " + "; ".join(f"x {x} " + " | ".join(
            ("PASS" if prule[(k, x)][f_]["ok"] else "FAIL (" + ", ".join(r_ for r_ in ("R1", "R2", "R3") if not prule[(k, x)][f_][r_]) + ")")
            for f_ in ("option_b", "recal")) for x in XS))
    wi = g5d_rep.get("intervals", {}).get("rule", {})
    wp = g5d_rep.get("p_le", {})
    say("   beside it, the window AOO at G5d's reading 1 (G5a fits; rule verdict; option B skill at x 1, 5, 10, 20), report only: " + "; ".join(
        f"{rname(k)} " + ({True: "PASS", False: "FAIL"}.get(wi.get(f"{rname(k)} aoo", {}).get("ok"), "n/a")) + ", " + "/".join(
            f"{wp.get(f'{rname(k)} x {x}', {}).get('scores', {}).get('b', {}).get('bss', float('nan')):+.2f}" for x in XS) for k in methods))
    # ================================================================ files
    with open(od / "per_stack_p4_2.csv", "w", newline="") as fh:
        w_ = csv.writer(fh)
        head = ["stack", "file_index", "row", "window", "coastal", "recorded_class", "records_6y_bin", "rec_t", "true_land", "true_window", "true_land_class"]
        for k in methods:
            kk = rname(k).replace(" ", "_")
            head += [f"aoo_land_raw_{kk}", f"mean_{kk}", f"lo_{kk}", f"hi_{kk}", f"crossed_{kk}"] + [f"p_le_{x}_{kk}" for x in XS]
        w_.writerow(head)
        names = pz["window_names"].astype(str)
        for i in range(n):
            row = [i, int(fi[i]), int(rw[i]), names[win[i]], int(coastal[i]), TCLS[rcls[i]], BINS[rbin[i]], int(rec_t[i]), int(yA[i]),
                   int(yfull[i]), TRUE[D.tidx(int(yA[i]))]]
            for k in methods:
                a_ = pred[k]["aoo"]
                row += [f"{aoo_land[k][i].mean():.3f}", f"{a_['mean'][i]:.3f}", f"{a_['L'][i]:.3f}", f"{a_['U'][i]:.3f}", int(a_["U"][i] < a_["L"][i])]
                row += [f"{v:.4f}" for v in pred[k]["p_b"][i]]
            w_.writerow(row)
    verdict = ", ".join(f"{rname(k)} {'PASS' if rule_i[k]['ok'] else 'FAIL'}" for k in methods)
    rep.update(gates=res, label=LABEL, rerun=len(ours), reading=len(lines_m) + 1, marker_utc=opened, n=int(n), crossed_rule="empty",
               md5=dict(m, tools=tmd5, predicted=mpred, g5d_predicted=m_pred5, csv=X.file_md5(od / "per_stack_p4_2.csv")),
               rule={rname(k): v for k, v in rule_i.items()}, rows={rname(k): v for k, v in rows_i.items()}, split={rname(k): v for k, v in split.items()},
               crossed={rname(k): v for k, v in crossed.items()},
               p_le={f"{rname(k)} x {x}": dict(scores={s_: {q: e[s_][q] for q in e[s_] if q != "table"} for s_ in e}, rules=prule[(k, x)])
                     for (k, x), e in pe.items()}, minutes=round((time.time() - t0) / 60, 2), written_utc=now())
    (od / "p4_2_report.json").write_text(json.dumps(rep, indent=1, default=float))
    n_ok = sum(res.values())
    say(f"stage P4-2{tag}: gates {n_ok} PASS, {len(res) - n_ok} FAIL; land-only rule on real_eval (reading {len(lines_m) + 1}, labelled): "
        f"{verdict}; p4_2_report.json {X.file_md5(od / 'p4_2_report.json')[:8]}; no overall verdict")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--second-reading", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--g5c", default="../data/stage_g5_predict")
    ap.add_argument("--g5b", default="../data/stage_g5b_export_check")
    ap.add_argument("--g5d", default="../data/stage_g5d_score")
    ap.add_argument("--g5a", default="../data/stage_g5_fits")
    ap.add_argument("--p4-0", default="../data/stage_p4_0")
    ap.add_argument("--p4-1", default="../data/stage_p4_1")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--out", default="../data/stage_p4_2")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_32_p4_2_eval_reading.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_32_p4_2_eval_reading", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.main(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.data, a.g5c, a.g5b, a.g5d, a.g5a, a.p4_0, a.p4_1, a.out, second_reading=a.second_reading, check_only=a.check_only,
               rerun=a.rerun, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)