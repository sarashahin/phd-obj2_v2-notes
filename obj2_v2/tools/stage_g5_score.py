#!/usr/bin/env python3
"""
stage_g5_score.py  (obj2_v2, stage G5 part G5d, 11 Nov 2026; G5 rules signed off by Sara on 11 Nov; Sara's answers of
11 Nov: export --report inside G5d after the predictions are md5-logged, a report-only EOO line split at the window cap)

Question: G5-5 of the signed G5 rules (claude/obj2_v2_G5_rules_draft.md). The one reading of real_eval. Every prediction
(G5c) and fit (G5a) was written and md5-logged before this tool runs. It checks them, writes a marker file, opens the
real_eval truth once and scores each rule on its own. There is no overall verdict.

Order
    Phase 1, truth closed (any np.load of a real_eval truth file or of an obj1 real_eval file is refused): gates Z0-Z6.
        The AOO and EOO of every map, the corrected intervals and P(AOO <= x) (G5a apply code, fits unchanged) and the
        F5 score maps are computed from the G5c files, and every md5 is printed. If any gate fails, the marker is not
        written, no truth file is opened, and the tool stops.
    Phase 2: g5d_predicted.npz is written and md5-logged, then the marker data/real_eval/G5D_TRUTH_OPENED. Then:
        export_obj2.py report on real_eval (Sara, 11 Nov), the G0 habitat and G4a niche checks on the real_eval obj1
        files (Z7), the stack ids and growth rates, the T1-T3 sample and control statistics written and md5-logged
        (G4b order), then the true maps; gates Z8-Z11 and the rules.
    --check-only runs phase 1 and writes nothing. A second reading is refused while the marker exists; --rerun allows
    it and every line of it says "rerun" (G5 rules: "Any rerun after that is reported as a rerun").

Rules (as signed; nothing here changes them)
    F5      stage_f_check.py rule: per true class 1 .. 51+, paired per stack, d = AUC(model, 50 samples) - AUC(B1);
            95% interval from resampling communities (2,000 draws, seed 31); PASS mean >= 0, FAIL upper end < 0, else
            NOT RESOLVED; the rule passes if every class passes. Run against both B1 versions (stored float32 and exact
            log); a seed passes only if it passes against both.
    AOO/EOO ratio 0.8-1.25 and 90% interval coverage 0.85-0.95 in every recorded class and records bin with 50 or more
            stacks, per method (G3 rule), with the G5a fits (G3 grouping options).
    P(AOO<=x) x = 1, 5, 10, 20, option B (G5a): Brier score, Brier skill against the real_val share per recorded
            class (G5a base_rate), 10-bin reliability table; G3b check rules R1 Brier not worse than G3's P, R2 classes 0
            and 1 within 0.02 of the observed share, R3 largest gap over bins of 200+ stacks <= 0.05, reported for option
            B and for the recalibrated P.
    T1-T3   G4b statistics (torus distance for T2, control: year t of control sample k is year t of sample (k + 7t) mod
            50, undefined-statistic rule). Binding rule per seed (Sara, 11 Nov): upper end of the 95% interval of the
            class-balanced (model error - control error) over true classes 1 .. 51+ below 0, and a gain of at least 10%.
            T3 is scored only if Z7 passes; otherwise T3 is dropped and stated.
    Reported only: true class 0, every class, the pooled line (first 25 samples of each seed), the mean of the seeds,
            B3-B5 in the AOO/EOO tables (B3 kept out of the printed T1-T3 tables, numbers in the json), T2 plain distance,
            the G4c split of T3, and the EOO line split by true EOO at the window cap (19 x 19 = 361 cells) or below.

Gates
    Z0  one reading: --final-eval given; output folder new and outside real_eval; no marker (or --rerun, reported)
    Z1  G5c: g5c_report.json R0 and R5 PASS, truth opened 0, written after every part file; every part's gates PASS;
        every prediction file md5 = the g5c list
    Z2  models and fits: best.pt md5 = the G5c and G1 logs for both runs; g5_fits.npz md5 = g5_fits_report.json =
        g5b_export_check.json (so byte-equal to the fits written before the export; G5b E0 checked the dates) and G5b
        gates all PASS; fit communities = the real_val communities, none in real_eval; fit methods = the methods scored
    Z3  real_eval inputs: manifest and every input and truth file md5 = the g5b list; one obj1 file per manifest row,
        md5 = manifest source_md5 (md5 only, not opened)
    Z4  alignment from inputs only: every map file covers the frozen stacks in order with the G1 layout and the frozen
        md5; K = 50 (model, B3, B4), 1 (B5); recorded class and records from the inputs = frozen_eval.npz; community and
        blank in baselines_eval.npz = inputs and frozen set; target-year AOO of the B3-B5 maps = their stored AOO
    Z5  scoring code = signed-off code, on real_val (no real_eval file): F5 from each run's check_3a counts = its
        report.json F5 (1e-9); P(AOO <= x) scores of g3b_recal.npz = g3b_report.json (1e-9); T1-T3 tests from
        per_stack_g4b.csv = g4b_report.json (csv has 6 digits: 1e-4, verdicts and counts exact); records lie inside the
        true maps in every year of every real_val frozen stack
    Z6  truth closed in phase 1: no real_eval truth file or obj1 real_eval file passed to np.load; every input and
        prediction md5 printed before the marker
    Z7  T3 inputs on real_eval: G0 H0-H3 on every obj1 file; G4a A0-A3 on every obj1 file and A4 on the real_eval truth
        species; G4b M3 (habitat of every used round, niches, ENV_r_field) in every community; G4b and G4c growth code agree
    Z8  control: every year's 50 maps unchanged; 6 different samples in every control sample (G4b M2)
    Z9  truth aligned: records inside the true maps in every year of every stack; target-year truth and true class
        from the two readers equal; no recorded target-year cell scored; a random score gives 0.50 (0.01 or 4 se)
    Z10 leak: g5d_predicted.npz and g5d_t_stats.npz unchanged after the truth opened; 300 stacks' statistics recomputed
        with every stack's true maps swapped for another's equal the logged ones (G4b M4)
    Z11 statistic code: vectorised T1, T2, T2 plain, T3 = loop code on 200 stacks (G4b M5)

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, about 30 min)
    python3 ../tools/stage_g5_score.py --self-test | tail -1
    python3 -u ../tools/stage_g5_score.py --final-eval --check-only > ../logs/stage_g5d_check.txt 2>&1
    python3 -u ../tools/stage_g5_score.py --final-eval > ../logs/stage_g5d_score.txt 2>&1
Defaults: --data ../data, --runs ../runs/f_seed1 ../runs/f_seed2, --g5c ../data/stage_g5_predict,
          --g5b ../data/stage_g5b_export_check, --fits ../data/stage_g5_fits, --g3b ../data/stage_g3b_recalibration,
          --g4b ../data/stage_g4b_process, --obj1-work ../../obj1_v2/work,
          --files "../../obj1_v2/work/results/stage2a/real_eval/*_training.npz", --out ../data/stage_g5d_score.
Needs the G tools (stage_g5_predict.py, stage_g5_fits.py, stage_g_process_tests.py, stage_g_t3_check.py and the tools
they load) in the same folder.
"""
import argparse, csv, glob, hashlib, importlib.util, inspect, json, os, subprocess, sys, tempfile, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g5_score.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P5 = _load("stage_g5_predict")
G5F = _load("stage_g5_fits")
PT = _load("stage_g_process_tests")
T3C = _load("stage_g_t3_check")
GB, GS, CK = P5.GB, P5.GS, P5.CK
HR, E, D, X = P5.HR, P5.E, P5.D, P5.X
GC, GR = G5F.GC, G5F.GR
HC, NC = PT.HC, PT.NC
G, W = X.G, X.WINDOW
TCLS, BINS, TRUE, XS = D.TCLS, D.BINS, D.TRUE, GC.XS
SCORE = CK.SCORE
MARKER = P5.MARKER
EOO_CAP = (G - 1) ** 2                                    # largest hull of cell centres in the 20 x 20 window
TESTS, TESTED, MIN_GAIN, K = PT.TESTS, PT.TESTED, PT.MIN_GAIN, PT.K
SEED_COLS = ("model_s1", "model_s2")
BROKEN = os.environ.get("STAGE_G5D_BROKEN", "")           # self-test only: plants one bug so its gate must fail
TOL_T = 1e-4


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def arr_md5(*arrays):
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class EvalGuard:
    """While active, np.load of a truth file inside the real_eval folder, or of an obj1 real_eval file, is refused."""
    def __init__(self, ev_dir, files):
        self.dir = str(Path(ev_dir).resolve())
        self.files = {str(Path(f).resolve()) for f in files}

    def __enter__(self):
        self.orig, self.opened = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        p = str(Path(f).resolve()) if isinstance(f, (str, Path)) else str(f)
        if ("_truth" in Path(p).name and p.startswith(self.dir)) or p in self.files:
            self.opened.append(p)
            raise RuntimeError(f"real_eval truth or obj1 file opened before the marker: {p}")
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- scoring pieces
def counts_t(maps, k=None):
    """Packed maps (n, K, 300) -> occupied-sample counts at the target year (n, G, G), first k samples if given."""
    n = len(maps)
    c = np.zeros((n, G, G), np.int32)
    for a in range(0, n, 512):
        pk = maps[a:a + 512] if k is None else maps[a:a + 512, :k]
        c[a:a + 512] = GC.target_year(pk).sum(1)
    return c


def f5_rule(A, ref, tc, comm, classes=SCORE):
    cb = CK.class_boot(A - ref, tc, comm, classes=classes)
    v = {c: CK.verdict(cb[c]["mean"], cb[c]["hi"]) for c in classes}
    overall = "FAIL" if "FAIL" in v.values() else ("PASS" if all(x == "PASS" for x in v.values()) else "NOT RESOLVED")
    return cb, v, overall


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
    if BROKEN == "p_ref":
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


def t_block(maps, fell, chunk=256, out=print):
    """G4b's sample and control statistics (stage_g_process_tests.run, same calls), for packed maps s1, s2, B3."""
    n = len(fell)
    IDX = PT.control_index()
    yr = np.arange(W)[None, :]
    two = all(len(set(IDX[q].tolist())) == W for q in range(K)) and all(len(set(IDX[:, t].tolist())) == K for t in range(W))
    methods = ["s1", "s2", "pool", "c1", "c2", "cpool", "B3"]
    S = {(m, v, q): np.full(n, np.nan) for m in methods for v in TESTS for q in ("mean", "p5", "p95")}
    S.update({(m, v, "n"): np.zeros(n, np.int16) for m in methods for v in TESTS})
    m2, t1 = True, time.time()
    for a in range(0, n, chunk):
        b = min(n, a + chunk); ix = np.arange(a, b)
        A = {m: PT.unpack6(maps[m][a:b]) for m in ("s1", "s2", "B3")}
        A["pool"] = np.concatenate([A["s1"][:, :K // 2], A["s2"][:, :K // 2]], 1)
        for src, cn in (("s1", "c1"), ("s2", "c2"), ("pool", "cpool")):
            A[cn] = A[src][:, IDX, yr]
            pm = np.packbits(A[src].reshape(b - a, K, W, -1), axis=-1)
            pc = np.packbits(A[cn].reshape(b - a, K, W, -1), axis=-1)
            for t in range(W):
                vm = np.sort(np.ascontiguousarray(pm[:, :, t]).view(f"V{pm.shape[-1]}")[..., 0], axis=1)
                vc = np.sort(np.ascontiguousarray(pc[:, :, t]).view(f"V{pc.shape[-1]}")[..., 0], axis=1)
                m2 &= bool(np.array_equal(vm, vc))
        for m in methods:
            st = PT.stats(A[m], fell[a:b], idx=ix)
            for v in TESTS:
                mean, cnt, p5, p95 = PT.agg(st[v])
                S[(m, v, "mean")][a:b], S[(m, v, "n")][a:b], S[(m, v, "p5")][a:b], S[(m, v, "p95")][a:b] = mean, cnt, p5, p95
        if a == 0 or (a // chunk) % 16 == 0:
            out(f"   statistics {b:,} of {n:,} stacks ({(time.time() - t1) / 60:.1f} min)")
    return S, m2 and two


def t_tests(S, tru, tc, comm, tested=TESTED):
    """G4b's tests (stage_g_process_tests.run, same rule); a test not in `tested` is 'dropped'."""
    defined = {v: ~np.isnan(tru[v]) for v in TESTS}
    rep, verdicts, verdicts_g, lines = {}, {}, {}, []
    for lab, mm, cc in (("seed 1", "s1", "c1"), ("seed 2", "s2", "c2"), ("pooled", "pool", "cpool")):
        for v in TESTS:
            em = np.abs(S[(mm, v, "mean")] - tru[v]); ec = np.abs(S[(cc, v, "mean")] - tru[v])
            eb = np.abs(S[("B3", v, "mean")] - tru[v])
            dm, dc = S[(mm, v, "n")] > 0, S[(cc, v, "n")] > 0
            enter = defined[v] & dm & dc
            if BROKEN == "t_err":
                em = S[(mm, v, "mean")] - tru[v]
            d = np.where(enter, em - ec, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cb = CK.class_boot(d, tc, comm, classes=SCORE)
                c0 = CK.class_boot(d, tc, comm, classes=[0])
            hascov = (mm, v, "p5") in S
            cov = lambda m: (tru[v] >= S[(m, v, "p5")]) & (tru[v] <= S[(m, v, "p95")])
            per = {}
            for c in [0] + SCORE:
                s = enter & (tc == c)
                sb = s & (S[("B3", v, "n")] > 0)
                row = dict(n=int(s.sum()), n_b3=int(sb.sum()))
                if s.any():
                    row.update(truth=float(tru[v][s].mean()), model=float(S[(mm, v, "mean")][s].mean()),
                               control=float(S[(cc, v, "mean")][s].mean()), err_model=float(em[s].mean()),
                               err_control=float(ec[s].mean()))
                    if hascov:
                        row.update(cov_model=float(cov(mm)[s].mean()), cov_control=float(cov(cc)[s].mean()))
                    src = cb[c] if c in cb else c0[c]
                    row.update(diff=src["mean"], lo=src["lo"], hi=src["hi"])
                if sb.any():
                    row.update(b3=float(S[("B3", v, "mean")][sb].mean()), err_b3=float(eb[sb].mean()))
                    if hascov:
                        row["cov_b3"] = float(cov("B3")[sb].mean())
                per[TRUE[c]] = row
            bal = cb["balanced"]
            em_c = [per[TRUE[c]]["err_model"] for c in SCORE if per[TRUE[c]]["n"]]
            ec_c = [per[TRUE[c]]["err_control"] for c in SCORE if per[TRUE[c]]["n"]]
            gain = 1 - float(np.mean(em_c)) / float(np.mean(ec_c)) if em_c and np.mean(ec_c) > 0 else float("nan")
            sb_all = enter & (S[("B3", v, "n")] > 0) & np.isin(tc, SCORE)
            dropped = dict(truth=int((~defined[v]).sum()), model=int((~dm).sum()), control=int((~dc).sum()), any=int((~enter).sum()))
            is_test = lab != "pooled" and v in TESTED
            if is_test and v not in tested:
                verdict = verdict_g = "dropped"
            elif is_test:
                verdict = "PASS" if (bal["hi"] < 0 and enter.any()) else "FAIL"
                verdict_g = "PASS" if (verdict == "PASS" and gain >= MIN_GAIN) else "FAIL"
            else:
                verdict = verdict_g = "report"
            rep[f"{lab} {v}"] = dict(entered=int(enter.sum()), dropped=dropped, balanced=bal, relative_gain=gain, classes=per,
                                     verdict=verdict, verdict_min_gain=verdict_g,
                                     b3_balanced_err=float(np.nanmean([per[TRUE[c]].get("err_b3", np.nan) for c in SCORE]))
                                     if sb_all.any() else None)
            if verdict != "report":
                verdicts[f"{lab} {v}"] = verdict; verdicts_g[f"{lab} {v}"] = verdict_g
            lines.append((lab, v, int(enter.sum()), dropped, bal, per, verdict, verdict_g, gain))
    for v in TESTS:
        b1, b2 = rep[f"seed 1 {v}"]["balanced"], rep[f"seed 2 {v}"]["balanced"]
        rep[f"mean of seeds {v}"] = dict(diff=(b1["mean"] + b2["mean"]) / 2)
    return rep, verdicts, verdicts_g, lines


def read_truth6(d, files, fidx, rows):
    """Six-year true maps (n, W, G, G), true class at the target year, from the truth files (G4b reader)."""
    man = {r["input_file"]: r for r in D.rows_of(d)}
    n = len(rows)
    T = np.zeros((n, W, G, G), bool); tc = np.zeros(n, int)
    for i in np.unique(fidx):
        k = np.nonzero(fidx == i)[0]
        with np.load(Path(d) / man[str(files[i])]["truth_file"], allow_pickle=False) as zt:
            T[k] = X.unpack(zt["true_maps"][rows[k]]); tc[k] = D.tidx(zt["true_cells"][rows[k]])
    return T, tc


def dif(a, b):
    """Absolute difference; 0 if both are missing (NaN or None), infinite if only one is."""
    na, nb = a is None or (isinstance(a, float) and np.isnan(a)), b is None or (isinstance(b, float) and np.isnan(b))
    return 0.0 if na and nb else (float("inf") if na or nb else abs(float(a) - float(b)))


def close(a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and (np.abs(a[~np.isnan(a)] - b[~np.isnan(b)]) <= tol).all())


# ---------------------------------------------------------------- Z5 replicas on real_val (no real_eval file)
def replica_f5(data, runs):
    fv = dict(np.load(data / "stage_d_frozen_val.npz"))
    hz = dict(np.load(data / "stage_d_headroom_pred.npz"))
    V, _, _, T, tc, comm = HR.load_stacks(data / "real_val", fv["files"], fv["file_index"], fv["row"], truth=True)
    T6, tc6 = read_truth6(data / "real_val", fv["files"], fv["file_index"], fv["row"])
    inside = int((V & ~T6).sum()) == 0 and np.array_equal(T6[:, -1], T) and np.array_equal(tc6, tc)
    Mk = ~V[:, W - 1] if BROKEN != "score_recorded" else np.ones_like(V[:, W - 1])
    voff, _ = E.frozen_draws(fv)
    a_b1 = HR.aucs(hz["B1"], T, Mk)
    worst, same_v, txt = 0.0, True, []
    for r in runs:
        pk = np.load(Path(r) / "check_3a" / "preds_k50.npz")
        s = CK.unplace(pk["counts"][:, W - 1][:, None].astype(np.float32), voff)[:, 0] / int(pk["K"])
        cb, v, overall = f5_rule(HR.aucs(s, T, Mk), a_b1, tc, comm)
        ref = json.loads((Path(r) / "check_3a" / "report.json").read_text())["F5"]
        d = max(max(dif(cb[c][q], ref["by_class"][TRUE[c]][q]) for c in SCORE for q in ("mean", "lo", "hi")),
                max(dif(cb["balanced"][q], ref["balanced"][q]) for q in ("mean", "lo", "hi")))
        sv = overall == ref["verdict"] and all(v[c] == ref["by_class"][TRUE[c]]["verdict"] for c in SCORE)
        worst = max(worst, d); same_v &= sv
        txt.append(f"{Path(r).name} F5 {overall} (report {ref['verdict']}), largest difference {d:.1e}")
    return worst <= 1e-9 and same_v, inside, "; ".join(txt)


def replica_p(g3b):
    rep = json.loads((g3b / "g3b_report.json").read_text())
    fp = g3b / "g3b_recal.npz"
    z = np.load(fp)
    y, rcls, fold = z["true_aoo"].astype(float), z["rcls"], z["fold"]
    keys = [k[8:] for k in z.files if k.startswith("p_recal_")]
    worst, nk = 0.0, 0
    for k in keys:
        for j, x in enumerate(XS):
            ref = fold_ref(y, fold, rcls, x)
            for stage, P in (("before", z[f"p_le_{k}"][:, j]), ("after", z[f"p_recal_{k}"][:, j])):
                ev = p_eval(P.astype(float), y, rcls, ref, x)
                rr = rep[stage][k][str(x)]
                for q in ("brier", "brier_ref", "bss", "gap", "gap_z", "gap200", "mean_p", "share"):
                    worst = max(worst, dif(float(ev[q]), rr[q]))
                nk += 1
    return X.file_md5(fp) == rep["md5"]["recal"] and worst <= 1e-9 and nk > 0, f"{nk} method-x-stage sets, largest difference {worst:.1e}"


def replica_t(g4b):
    rep = json.loads((g4b / "g4b_report.json").read_text())
    with open(g4b / "per_stack_g4b.csv") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    fl = lambda c: np.array([float(r[c]) for r in rows])
    tru = {v: fl(f"truth_{v}") for v in TESTS}
    tc = np.array([TRUE.index(r["true_class"]) for r in rows]); comm = np.array([int(r["comm"]) for r in rows])
    S = {}
    for m in ("s1", "s2", "pool", "c1", "c2", "cpool", "B3"):
        for v in TESTS:
            S[(m, v, "mean")] = fl(f"{m}_{v}_mean"); S[(m, v, "n")] = np.array([int(r[f"{m}_{v}_n"]) for r in rows])
    got, verd, verd_g, _ = t_tests(S, tru, tc, comm)
    worst, same = 0.0, verd == rep["verdicts"] and verd_g == rep["verdicts_min_gain"]
    for key, r in rep["tests"].items():
        if key.startswith("mean of seeds"):
            continue
        g = got[key]
        same &= g["entered"] == r["entered"] and g["dropped"] == r["dropped"] and g["verdict"] == r["verdict"]
        for q in ("mean", "lo", "hi"):
            worst = max(worst, dif(g["balanced"][q], r["balanced"][q]))
        worst = max(worst, dif(g["relative_gain"], r["relative_gain"]))
    return same and worst <= TOL_T, f"{n} stacks, verdicts and counts equal {same}, largest difference {worst:.1e}"


# ---------------------------------------------------------------- run
def run(data, runs, g5c, g5b, fits_dir, g3b, g4b, obj1_files, obj1_work, out_dir, final_eval, check_only=False,
        rerun=False, chunk=256, out=print):
    data, g5c, g5b, fits_dir, g3b, g4b, out_dir = (Path(p) for p in (data, g5c, g5b, fits_dir, g3b, g4b, out_dir))
    runs = [Path(r) for r in runs]
    obj1_files = sorted(obj1_files)
    ev = data / "real_eval"
    if not final_eval:
        sys.exit("STOP: G5d is the one reading of real_eval; give --final-eval")
    if "real_eval" in str(out_dir):
        sys.exit("STOP: nothing is written inside a real_eval folder; give another --out")
    if not check_only and out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    if len(runs) != 2:
        sys.exit("STOP: G5d needs the two seeds (--runs seed1 seed2)")
    marker = ev / MARKER
    earlier = marker.read_text().splitlines() if marker.exists() else []
    if earlier and not rerun and not check_only:
        sys.exit(f"STOP: {marker} exists, so the real_eval truth was opened already ({earlier[0][:120]}); a second reading "
                 "needs --rerun and is reported as a rerun")
    if rerun and not earlier:
        sys.exit("STOP: --rerun given but there is no marker; the first reading runs without --rerun")
    tag = f" (rerun {len(earlier)})" if earlier and not check_only else ""
    res, t0, info = {}, time.time(), {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name}{tag} {text}: {'PASS' if ok else 'FAIL'}")

    names = ("stage_g5_predict", "stage_g5_fits", "stage_g_calibrate", "stage_g_recalibrate", "stage_g_process_tests",
             "stage_g_t3_check", "stage_g_habitat_check", "stage_g_niche_check", "stage_f_check", "stage_d_headroom", "export_obj2")
    tools_md5 = {nm: X.file_md5(HERE / f"{nm}.py") for nm in names}
    out(f"stage_g5_score.py md5 {X.file_md5(__file__)}; " + ", ".join(f"{k}.py {v}" for k, v in tools_md5.items()))
    gate("Z0", True, f"--final-eval given; output folder {out_dir} new and outside real_eval; marker "
         + (f"present with --rerun: {len(earlier)} earlier reading(s), this run is reported as a rerun" if tag else
            ("present, check-only (no truth opened)" if earlier else "absent")) + ("; check-only: nothing is written" if check_only else ""))
    env, envp = PT.load_env(obj1_work)
    k1, k2 = (f"model_{r.name}" for r in runs)
    methods = [k1, k2, "pooled", "B3", "B4", "B5"]

    # ================================================================ phase 1: truth closed
    with EvalGuard(ev, obj1_files) as guard:
        try:
            # ---- Z1 G5c outputs
            rep5 = json.loads((g5c / "g5c_report.json").read_text())
            pnames = ["frozen", "baselines"] + [f"samples_{r.name}" for r in runs]
            pj, miss = {}, []
            for nm in pnames:
                f = g5c / f"part_{nm}.json"
                if f.is_file():
                    pj[nm] = json.loads(f.read_text())
                else:
                    miss.append(nm)
            when = datetime.fromisoformat(rep5["written_utc"])
            late = all(when >= datetime.fromisoformat(p["written_utc"]) for p in pj.values())
            need = ["frozen_eval.npz", "baselines_eval.npz", "maps_B3_halfblank.npz", "maps_B4.npz", "maps_B5.npz"] + \
                [f"samples_{r.name}.npz" for r in runs]
            md5s = {f: X.file_md5(g5c / f) if (g5c / f).is_file() else None for f in need}
            listed = rep5.get("files", {})
            bad = [f for f in need if md5s[f] is None or md5s[f] != listed.get(f) or
                   all(f not in p["files"] or p["files"][f] != md5s[f] for p in pj.values())]
            g_ok = rep5["gates"].get("R0") is True and rep5["gates"].get("R5") is True and all(rep5["gates"].values())
            parts_ok = all(all(p["gates"].values()) for p in pj.values())
            gate("Z1", g_ok and not miss and late and not bad and parts_ok and rep5.get("truth_opened") == 0,
                 f"g5c_report.json md5 {X.file_md5(g5c / 'g5c_report.json')} gates {rep5['gates']}, truth opened "
                 f"{rep5.get('truth_opened')}, written {rep5['written_utc']} after every part file {late}; parts present "
                 f"{sorted(pj)} (missing {miss}), their gates PASS {parts_ok}; prediction files = the g5c list "
                 f"{not bad}" + (f" (differ: {', '.join(bad)})" if bad else ""))
            out("   prediction files: " + ", ".join(f"{f} {m}" for f, m in md5s.items()))

            # ---- Z2 models and fits
            ok2, why2 = True, []
            for r in runs:
                mb = X.file_md5(r / "best.pt")
                g1 = json.loads((r / "check_g1" / "g1_report.json").read_text())
                good = mb == rep5.get("best_md5", {}).get(f"samples_{r.name}") == g1["md5"]["best_pt"]
                ok2 &= good; why2.append(f"{r.name} best.pt {mb} {good}")
            fr = json.loads((fits_dir / "g5_fits_report.json").read_text())
            mf = X.file_md5(fits_dir / "g5_fits.npz")
            g5bj = json.loads((g5b / "g5b_export_check.json").read_text())
            fits = G5F.load_fits(fits_dir / "g5_fits.npz")
            val_seeds = sorted(int(r["sim_seed"]) for r in D.rows_of(data / "real_val"))
            rows_ev = D.rows_of(ev)
            ev_seeds = sorted(int(r["sim_seed"]) for r in rows_ev)
            fit_c = sorted(int(c) for c in fits["communities"])
            ok_f = mf == fr["md5"]["fits"] == g5bj["md5"].get("fits") and all(g5bj["gates"].values())
            ok_c = fit_c == val_seeds and not set(fit_c) & set(ev_seeds)
            ok_m = fits["methods"] == methods
            gate("Z2", ok2 and ok_f and ok_c and ok_m,
                 "; ".join(why2) + f"; g5_fits.npz md5 {mf} = g5_fits_report.json {fr['md5']['fits'] == mf} = g5b_export_check.json "
                 f"{g5bj['md5'].get('fits') == mf} (fits written {fr.get('written_utc')}), G5b gates all PASS {all(g5bj['gates'].values())}; "
                 f"fit communities {len(fit_c)} = real_val {fit_c == val_seeds}, none in real_eval {not set(fit_c) & set(ev_seeds)}; "
                 f"fit methods {fits['methods']} = scored {ok_m}")

            # ---- Z3 real_eval inputs and obj1 files (md5 only)
            man = X.file_md5(ev / "manifest.csv")
            badf = [f for r in rows_ev for f in (r["input_file"], r["truth_file"])
                    if X.file_md5(ev / f) != g5bj["md5"]["files"].get(f)]
            by_name = {Path(f).name: f for f in obj1_files}
            bado = [r["source"] for r in rows_ev if r["source"] not in by_name or X.file_md5(by_name[r["source"]]) != r["source_md5"]]
            gate("Z3", man == g5bj["md5"]["manifest"] and not badf and not bado and len(obj1_files) == len(rows_ev),
                 f"real_eval manifest md5 {man} = g5b {man == g5bj['md5']['manifest']}; {2 * len(rows_ev)} input and truth files "
                 f"= the g5b list {not badf}" + (f" (differ: {', '.join(badf[:3])})" if badf else "") + f"; {len(obj1_files)} obj1 "
                 f"files for {len(rows_ev)} rows, md5 = manifest source_md5 {not bado}; environment.py md5 {X.file_md5(envp)}")

            # ---- Z4 alignment from inputs only
            fe = dict(np.load(g5c / "frozen_eval.npz")); fmd5 = md5s["frozen_eval.npz"]
            n = len(fe["row"])
            bz = dict(np.load(g5c / "baselines_eval.npz"))
            maps, why4, ok4 = {}, [], True
            for kk, f, kexp in ((k1, f"samples_{runs[0].name}.npz", K), (k2, f"samples_{runs[1].name}.npz", K),
                                ("B3", "maps_B3_halfblank.npz", K), ("B4", "maps_B4.npz", K), ("B5", "maps_B5.npz", 1)):
                z = np.load(g5c / f)
                good = np.array_equal(z["sel"], np.arange(n)) and str(z["layout"]) == GS.LAYOUT and str(z["frozen_md5"]) == fmd5 \
                    and int(z["K"]) == kexp and z["maps"].shape == (n, kexp, W * G * G // 8)
                if kk in ("B3", "B4", "B5"):
                    good &= all(np.array_equal(GC.target_year(z["maps"][a:a + 512]).sum((2, 3)), z["aoo_t"][a:a + 512])
                                for a in range(0, n, 512))
                ok4 &= good; why4.append(f"{f} {good}")
                maps[kk] = z["maps"]
            V, _, _, _, _, comm = HR.load_stacks(ev, fe["files"], fe["file_index"], fe["row"], truth=False)
            rec_t = V[:, -1].reshape(n, -1).sum(1)
            rec6 = V.any(1).reshape(n, -1).sum(1)
            rcls, rbin = D.tidx(rec_t), D.bidx(rec6)
            blank = fe["blank"].astype(bool)
            al = np.array_equal(rec_t, fe["rec_t"]) and np.array_equal(rec6, fe["rec6"]) and np.array_equal(rcls, fe["cls"]) \
                and np.array_equal(comm, bz["comm"]) and np.array_equal(bz["blank"].astype(bool), blank) \
                and str(bz["frozen_md5"]) == fmd5 and bz["B1"].shape == (n, G, G) and bz["B1_exact_log"].shape == (n, G, G)
            gate("Z4", ok4 and al, f"{n} frozen real_eval stacks (frozen md5 {fmd5}); map files in frozen order with the G1 "
                 f"layout, K and frozen md5, B3-B5 AOO = stored: {', '.join(why4)}; recorded cells, records, recorded class, "
                 f"community and blank from the inputs = frozen_eval.npz and baselines_eval.npz {al}")

            # ---- predictions derived from the G5c files (no truth)
            t1 = time.time()
            met = {k: GC.map_metrics(maps[k]) for k in (k1, k2, "B3", "B4", "B5")}
            met["pooled"] = tuple(np.concatenate([met[k1][q][:, :K // 2], met[k2][q][:, :K // 2]], 1) for q in (0, 1))
            pred = {k: G5F.apply_all(fits, k, met[k][0], met[k][1], rec_t, rcls, rbin) for k in methods}
            c1, c2 = counts_t(maps[k1]), counts_t(maps[k2])
            scores = {"model_s1": c1.astype(np.float32) / K, "model_s2": c2.astype(np.float32) / K,
                      "pooled": (counts_t(maps[k1], K // 2) + counts_t(maps[k2], K // 2)).astype(np.float32) / K,
                      "B1": bz["B1"], "B1_exact": bz["B1_exact_log"]}
            save = dict(rec_t=rec_t.astype(np.int16), rcls=rcls.astype(np.int8), rbin=rbin.astype(np.int8), comm=comm,
                        blank=blank.astype(np.uint8), frozen_md5=np.array(fmd5), fits_md5=np.array(mf), xs=np.array(XS))
            for k in methods:
                save[f"aoo_{k}"] = met[k][0].astype(np.uint16); save[f"eoo_{k}"] = met[k][1].astype(np.float32)
                for kind in ("aoo", "eoo"):
                    for q in ("mean", "L", "U"):
                        save[f"{kind}_{q}_{k}"] = pred[k][kind][q]
                for q in ("p_g3", "p_recal", "p_b"):
                    save[f"{q}_{k}"] = pred[k][q]
            for s in ("model_s1", "model_s2", "pooled"):
                save[f"score_md5_{s}"] = np.array(arr_md5(scores[s]))
            out(f"   predictions from the G5c files with the G5a fits ({(time.time() - t1) / 60:.1f} min): " + ", ".join(
                f"{k} AOO/EOO arrays md5 {arr_md5(met[k][0], met[k][1])[:8]}, corrected {arr_md5(*[pred[k][kd][q] for kd in ('aoo', 'eoo') for q in ('mean', 'L', 'U')])[:8]}, "
                f"P {arr_md5(pred[k]['p_b'])[:8]}" for k in methods) + "; F5 score maps " +
                ", ".join(f"{s} {arr_md5(scores[s])[:8]}" for s in scores))

            # ---- Z5 scoring code on real_val
            a_f5, a_in, t_f5 = replica_f5(data, runs)
            a_p, t_p = replica_p(g3b)
            a_t, t_t = replica_t(g4b)
            gate("Z5", a_f5 and a_in and a_p and a_t, f"on real_val: F5 code = stage F step 3a reports {a_f5} ({t_f5}); P(AOO <= x) "
                 f"code = g3b_report.json {a_p} ({t_p}); T1-T3 test code = g4b_report.json {a_t} ({t_t}); records inside the true "
                 f"maps in every year of every frozen stack {a_in}")
            if BROKEN == "truth_early":
                r0_ = rows_ev[0]
                np.load(ev / r0_["truth_file"])
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    gate("Z6", not guard.opened and crashed is None, f"no real_eval truth file and no obj1 real_eval file opened in phase 1 "
         f"(refused {len(guard.opened)}" + (f": {crashed[:120]}" if crashed else "") + "); every input and prediction md5 printed above")
    if not all(res.values()):
        out(f"STOP{tag}: phase 1 gates failed ({', '.join(k for k, v in res.items() if not v)}); the marker is not written and no "
            "real_eval truth file was opened")
        return res, info
    if check_only:
        out(f"stage G5d check-only: gates {sum(res.values())} PASS, 0 FAIL; nothing written, no truth opened "
            f"({(time.time() - t0) / 60:.1f} min)")
        return res, info

    # ================================================================ phase 2: marker, then the truth
    out_dir.mkdir(parents=True, exist_ok=True)
    pfile = out_dir / "g5d_predicted.npz"
    X.save_npz(pfile, save)
    mpred = X.file_md5(pfile)
    opened = now()
    with open(marker, "a") as fh:
        fh.write(json.dumps(dict(opened_utc=opened, tool_md5=X.file_md5(__file__), out=str(out_dir.resolve()),
                                 predicted_md5=mpred, reading=len(earlier) + 1)) + "\n")
    out(f"   g5d_predicted.npz md5 {mpred} written; marker {marker} written {opened} (reading {len(earlier) + 1}); the truth "
        "opens from here")

    # ---- export report on real_eval (Sara, 11 Nov)
    try:
        xr = X.report([str(ev)], out=out)[str(ev)]
        info["export_report"] = dict(stacks=xr["stacks"], recorded_class=xr["cls_t"], true_class=xr["cls_true"])
    except SystemExit as e:
        out(f"   export report stopped: {e}")
        info["export_report"] = dict(stopped=str(e))

    # ---- Z7 T3 inputs on the real_eval obj1 files
    t1 = time.time()
    files_ev = [by_name[r["source"]] for r in rows_ev]
    hab = [HC.check_file(f, env.habitat_layer, out=out) for f in files_ev]
    hab_ok = all(all(r[g] for g in ("H0", "H1", "H2", "H3")) for r in hab)
    nic = [NC.check_file(f, env.niche_traits, env.niche_r, out=out) for f in files_ev]
    nic_ok = all(all(r.get(g, False) for g in ("A0", "A1", "A2", "A3")) for r in nic)
    fidx, rows = fe["file_index"], fe["row"]
    mrow = {r["input_file"]: r for r in rows_ev}
    comm_of = {int(i): mrow[str(fe["files"][i])] for i in np.unique(fidx)}
    sp_id, rd = np.zeros(n, np.int64), np.zeros(n, np.int64)
    a4, early = True, None
    for i, row in comm_of.items():
        k = np.nonzero(fidx == i)[0]
        with np.load(ev / row["truth_file"], allow_pickle=False) as zt:
            sp_all = np.asarray(zt["stack_species"]).astype(np.int64)
            sp_id[k] = sp_all[rows[k]]; rd[k] = zt["stack_round"][rows[k]]
        z = np.load(by_name[row["source"]], allow_pickle=True)
        a4 &= bool(np.isin(sp_all, np.unique(np.asarray(z["ASM_F_ids"]))).all())
    fell = np.zeros((n, W - 1, G, G), bool); sink = np.zeros_like(fell)
    m3, agree, lines3 = True, True, []
    for i, row in comm_of.items():
        k = np.nonzero(fidx == i)[0]
        z = np.load(by_name[row["source"]], allow_pickle=True)
        pairs = sorted(set(zip(rd[k].tolist(), sp_id[k].tolist())))
        fl, inf = PT.growth_fell(z, env, pairs)
        if fl is None:
            m3 = False; lines3.append(f"{row['source']}: {inf['why']}"); continue
        gs, _ = T3C.growth(z, env, pairs)
        agree &= all(np.array_equal(fl[p], gs[p][0]) for p in pairs)
        for j in k:
            fell[j] = fl[(int(rd[j]), int(sp_id[j]))]; sink[j] = gs[(int(rd[j]), int(sp_id[j]))][1]
        m3 &= inf["ok"]
        lines3.append(f"{row['source'][-30:]}: rounds {inf['rounds']}, habitat diff {inf['hab_diff']:.1e}; saved niches "
                      f"{inf['matched']} of {inf['saved']}, r field diff {inf['r_diff']:.1e}; species {inf['species']}")
    for s in lines3:
        out("   " + s)
    t3_on = hab_ok and nic_ok and a4 and m3 and agree
    gate("Z7", t3_on, f"T3 inputs on {len(files_ev)} real_eval obj1 files: G0 H0-H3 {hab_ok}; G4a A0-A3 {nic_ok}, A4 truth "
         f"species among the piece ids {a4}; G4b M3 {m3}; G4b and G4c growth code agree {agree} ({(time.time() - t1) / 60:.1f} min)"
         + ("" if t3_on else "; T3 is dropped and stated (T1, T2 go on)"))

    # ---- T statistics before any true map (G4b order)
    if BROKEN == "leak":
        early, _ = read_truth6(ev, fe["files"], fidx, rows)
        PT._LEAK = early
    S, m2 = t_block({"s1": maps[k1], "s2": maps[k2], "B3": maps["B3"]}, fell, chunk, out)
    M4c = {}
    if t3_on:
        for si, kk in ((0, k1), (1, k2)):
            M4c[(si, "fell")] = np.full((n, W - 1), np.nan); M4c[(si, "sink")] = np.full((n, W - 1), np.nan)
            for a in range(0, n, chunk):
                df, ds = T3C.parts(PT.unpack6(maps[kk][a:a + chunk]), fell[a:a + chunk], sink[a:a + chunk])
                M4c[(si, "fell")][a:a + chunk], M4c[(si, "sink")][a:a + chunk] = PT._nanmean(df, 1), PT._nanmean(ds, 1)
    sfile = out_dir / "g5d_t_stats.npz"
    X.save_npz(sfile, {**{f"{m}__{v}__{q}": S[(m, v, q)] for (m, v, q) in S}, **{f"g4c_s{s + 1}_{q}": v for (s, q), v in M4c.items()}})
    smd5 = X.file_md5(sfile)
    out(f"   sample and control statistics written before any true map was opened: g5d_t_stats.npz md5 {smd5}")
    gate("Z8", m2, f"control (year t from sample (k + {PT.SHIFT} t) mod {K}): every year's 50 maps unchanged in every stack "
         "and 6 different samples in every control sample")

    # ---- truth
    T6, tc6 = read_truth6(ev, fe["files"], fidx, rows)
    _, _, _, Tt, tc, comm2 = HR.load_stacks(ev, fe["files"], fidx, rows, truth=True)
    if BROKEN == "shift_truth":
        T6, Tt, tc6, tc = np.roll(T6, 1, 0), np.roll(Tt, 1, 0), np.roll(tc6, 1), np.roll(tc, 1)
    Mk = ~V[:, W - 1]
    inside = int((V & ~T6).sum())
    rnd = X.rng(99, 41).random(Tt.shape).astype(np.float32)
    br = HR.balanced(HR.aucs(rnd, Tt, Mk), tc)
    tol = max(0.01, 4 * br[1])
    same_t = np.array_equal(T6[:, -1], Tt) and np.array_equal(tc6, tc) and np.array_equal(comm2, comm)
    gate("Z9", inside == 0 and same_t and int((Mk & V[:, W - 1]).sum()) == 0 and abs(br[0] - 0.5) <= tol,
         f"recorded cells outside the true maps {inside}; target-year truth, true class and community from the two readers "
         f"equal {same_t}; recorded target-year cells scored {int((Mk & V[:, W - 1]).sum())}; random score {br[0]:.3f} "
         f"(0.50 within {tol:.3f})")
    if BROKEN == "stats_changed":
        X.save_npz(sfile, {f"{m}__{v}__{q}": np.nan_to_num(S[(m, v, q)]) for (m, v, q) in S})
    tru = {v: np.zeros(n) for v in TESTS}
    for a in range(0, n, 2048):
        st = PT.stats(T6[a:a + 2048, None], fell[a:a + 2048])
        for v in TESTS:
            tru[v][a:a + 2048] = st[v][:, 0]
    g_ = X.rng(77, 9)
    pick = np.sort(g_.choice(n, min(PT.N_LEAK, n), replace=False))
    if PT._LEAK is not None:
        PT._LEAK = T6[np.roll(np.arange(n), 1)]
    st = PT.stats(PT.unpack6(maps[k1][pick]), fell[pick], idx=pick)
    eq = all(PT.same(PT.agg(st[v])[0], S[("s1", v, "mean")][pick]) for v in TESTS)
    PT._LEAK = None
    gate("Z10", X.file_md5(pfile) == mpred and X.file_md5(sfile) == smd5 and eq,
         f"g5d_predicted.npz unchanged after the truth opened {X.file_md5(pfile) == mpred}; g5d_t_stats.npz unchanged "
         f"{X.file_md5(sfile) == smd5}; {len(pick)} stacks recomputed with swapped true maps equal the logged values {eq}")
    pick = np.sort(X.rng(77, 10).choice(n, min(PT.N_LOOP, n), replace=False))
    A5 = PT.unpack6(maps[k1][pick, :5])
    fast, fast_t = PT.stats(A5, fell[pick]), PT.stats(T6[pick, None], fell[pick])
    ok11 = True
    for q, j in enumerate(pick):
        for kk in range(5):
            sl = PT.loop_stats(A5[q, kk], fell[j])
            ok11 &= all(PT.same(fast[v][q, kk], sl[v]) for v in TESTS)
        sl = PT.loop_stats(T6[j], fell[j])
        ok11 &= all(PT.same(fast_t[v][q, 0], sl[v]) for v in TESTS)
    gate("Z11", ok11, f"T1, T2 torus, T2 plain, T3 of the vectorised code = loop code on {len(pick)} stacks (5 samples of seed 1 "
         "and the truth)")

    # ================================================================ rules
    report = dict(rerun=len(earlier), marker_utc=opened, gates=res, n=int(n), md5=dict(
        tool=X.file_md5(__file__), tools=tools_md5, predictions=md5s, fits=mf, predicted=mpred, t_stats=smd5,
        best_pt={r.name: X.file_md5(r / "best.pt") for r in runs}, environment=X.file_md5(envp),
        g5c_report=X.file_md5(g5c / "g5c_report.json"), g5b=X.file_md5(g5b / "g5b_export_check.json")), **info)
    comm_n = len(np.unique(comm))
    # ---- F5
    A = {k: HR.aucs(v, Tt, Mk) for k, v in scores.items()}
    out(f"   AUC by true class at the target year (stacks scored; model seed 1 | seed 2 | pooled | B1 | B1 exact):")
    for c in SCORE:
        kq = (tc == c) & ~np.isnan(A["B1"])
        out(f"      {TRUE[c]:>6} {int(kq.sum()):>6}  " + "  ".join(f"{np.nanmean(A[s][kq]):.4f}" if kq.any() else "  -   "
                                                             for s in ("model_s1", "model_s2", "pooled", "B1", "B1_exact")))
    out("      balanced       " + "  ".join(f"{HR.balanced(A[s], tc)[0]:.4f}" for s in ("model_s1", "model_s2", "pooled", "B1", "B1_exact")))
    f5 = {}
    out(f"   F5 model - B1 by true class, paired per stack, 95% interval from resampling the {comm_n} communities (PASS mean >= 0, "
        "FAIL upper end < 0); against the stored float32 B1 | the exact B1:")
    for s, lab in (("model_s1", "seed 1"), ("model_s2", "seed 2"), ("pooled", "pooled (report)")):
        rr = {}
        for b in ("B1", "B1_exact"):
            cb, v, overall = f5_rule(A[s], A[b], tc, comm)
            rr[b] = dict(by_class={TRUE[c]: dict(cb[c], verdict=v[c]) for c in SCORE}, balanced=cb["balanced"], verdict=overall)
        both = "PASS" if rr["B1"]["verdict"] == rr["B1_exact"]["verdict"] == "PASS" else \
            ("FAIL" if "FAIL" in (rr["B1"]["verdict"], rr["B1_exact"]["verdict"]) else "NOT RESOLVED")
        f5[lab] = dict(rr, verdict=both)
        out(f"      {lab}:")
        for c in SCORE:
            a_, b_ = rr["B1"]["by_class"][TRUE[c]], rr["B1_exact"]["by_class"][TRUE[c]]
            out(f"         {TRUE[c]:>6} n {a_['n']:>6,}  {a_['mean']:+.4f} [{a_['lo']:+.4f}, {a_['hi']:+.4f}] {a_['verdict']:<12} | "
                f"{b_['mean']:+.4f} [{b_['lo']:+.4f}, {b_['hi']:+.4f}] {b_['verdict']}")
        a_, b_ = rr["B1"]["balanced"], rr["B1_exact"]["balanced"]
        out(f"         balanced {a_['mean']:+.4f} [{a_['lo']:+.4f}, {a_['hi']:+.4f}] | {b_['mean']:+.4f} [{b_['lo']:+.4f}, {b_['hi']:+.4f}]; "
            f"F5 {rr['B1']['verdict']} | {rr['B1_exact']['verdict']}; " + ("both needed: " if "seed" in lab and "pooled" not in lab
                                                                       else "report: ") + both)
    msd = (f5["seed 1"]["B1"]["balanced"]["mean"] + f5["seed 2"]["B1"]["balanced"]["mean"]) / 2
    out(f"      mean of the two seeds, balanced model - B1 (stored): {msd:+.4f} (report)")
    report["F5"] = dict(f5, mean_of_seeds_balanced=msd, auc_balanced={s: HR.balanced(A[s], tc)[0] for s in A})
    # ---- AOO and EOO intervals
    ya = Tt.reshape(n, -1).sum(1).astype(float)
    ye, _ = GC.eoo_aoo(Tt)
    groups = {"recorded class": (rcls, TCLS), "records over 6 years": (rbin, BINS), "true class": (tc, TRUE)}
    rows_i, rule_i = {}, {}
    for kind, y in (("aoo", ya), ("eoo", ye)):
        out(f"   {kind.upper()} on real_eval with the G5a fits (ratio / coverage / median width; rule: ratio {GC.BAND[0]}-{GC.BAND[1]}, "
            f"coverage {GC.COV[0]}-{GC.COV[1]} in every group of {GC.N_TEST}+ stacks):")
        for k in methods:
            rr = GC.evaluate(y, pred[k][kind], groups)
            ok, bad = GC.rule_pass(rr)
            rows_i[(k, kind)], rule_i[(k, kind)] = rr, dict(ok=ok, bad=bad, option=int(fits["corr"][(k, kind)]["option"]))
            out(f"      {G5F.report_name(k):<14} option {rule_i[(k, kind)]['option']} rule {'PASS' if ok else 'FAIL'}"
                + ("" if ok else f" ({', '.join(bad[:6])})"))
            for gname in ("recorded class", "records over 6 years"):
                out(f"         {gname:<20} " + "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}/{r['width']:.1f}"
                                                        + ("" if r["n"] >= GC.N_TEST else "*") for r in rr[gname]))
        ms = {g: [((a["ratio"] + b["ratio"]) / 2, (a["cov"] + b["cov"]) / 2) for a, b in
                  zip(rows_i[(k1, kind)][g], rows_i[(k2, kind)][g])] for g in ("recorded class", "records over 6 years")}
        out(f"      {'mean of seeds':<14} " + " | ".join("; ".join(f"{r['group']} {m[0]:.2f}/{m[1]:.2f}" for r, m in
                                                         zip(rows_i[(k1, kind)][g], ms[g])) for g in ms) + " (report)")
        out("      stacks per group (* = under 50, reported, not tested): " + " | ".join(
            "; ".join(f"{r['group']} {r['n']}" for r in rows_i[(k1, kind)][g]) for g in ("recorded class", "records over 6 years")))
        out(f"      by true class (report only; ratio / coverage):")
        for k in methods:
            out(f"         {G5F.report_name(k):<14} " + "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}" for r in rows_i[(k, kind)]["true class"]))
    cap = ye >= EOO_CAP - 1e-9
    split = {}
    out(f"   EOO split by true EOO at the window cap ({EOO_CAP} cells or more: {int(cap.sum())} stacks) or below ({int((~cap).sum())}); "
        "ratio / coverage by recorded class (report only, the rule above is unchanged):")
    for k in methods:
        split[k] = {}
        for lab, m in (("at cap", cap), ("below cap", ~cap)):
            if not m.any():
                continue
            oo = {q: pred[k]["eoo"][q][m] for q in ("mean", "L", "U")}
            rr = GC.evaluate(ye[m], oo, {"recorded class": (rcls[m], TCLS)})["recorded class"]
            split[k][lab] = rr
            out(f"      {G5F.report_name(k):<14} {lab:<9} " + "; ".join(f"{r['group']} n {r['n']} {r['ratio']:.2f}/{r['cov']:.2f}" for r in rr))
    out("   width over the model (method median width / mean of the two seeds' median widths), by recorded class (report):")
    for kind in ("aoo", "eoo"):
        mw = {r["group"]: (r["width"] + q["width"]) / 2 for r, q in zip(rows_i[(k1, kind)]["recorded class"], rows_i[(k2, kind)]["recorded class"])}
        for k in ("pooled", "B3", "B4", "B5"):
            out(f"      {kind.upper()} {k:<6} " + "; ".join(f"{r['group']} {r['width'] / mw[r['group']]:.2f}" if mw.get(r["group"], 0) > 0
                                                       else f"{r['group']} n/a" for r in rows_i[(k, kind)]["recorded class"]))
    report["intervals"] = dict(rule={f"{G5F.report_name(k)} {kind}": v for (k, kind), v in rule_i.items()},
                               rows={f"{G5F.report_name(k)} {kind}": v for (k, kind), v in rows_i.items()},
                               eoo_cap_split={G5F.report_name(k): v for k, v in split.items()}, eoo_cap=EOO_CAP,
                               at_cap=int(cap.sum()))
    # ---- P(AOO <= x)
    pe, prule, pdiff = {}, {}, {}
    out("   P(AOO <= x) on real_eval (Brier / skill against the real_val share per recorded class / largest gap, bins 50+ | 200+ / "
        "mean P vs share); G3's P -> recalibrated P -> option B:")
    for k in methods:
        for j, x in enumerate(XS):
            ref = fits["base_rate"][j][rcls]
            e = {st_: p_eval(pred[k][q][:, j], ya, rcls, ref, x) for st_, q in (("g3", "p_g3"), ("recal", "p_recal"), ("b", "p_b"))}
            pe[(k, x)] = e
            o = (ya <= x).astype(float)
            pdiff[(k, x)] = GR.boot_diff((pred[k]["p_b"][:, j] - o) ** 2 - (pred[k]["p_g3"][:, j] - o) ** 2, comm)
            prule[(k, x)] = dict(option_b=r_rules(e["g3"], e["b"]), recal=r_rules(e["g3"], e["recal"]))
            out(f"      {G5F.report_name(k):<14} x {x:>2}: " + " -> ".join(f"{e[s]['brier']:.4f} / {e[s]['bss']:+.3f} / {e[s]['gap']:.3f} | "
                                                                      f"{e[s]['gap200']:.3f} / {e[s]['mean_p']:.3f} vs {e[s]['share']:.3f}"
                                                                      for s in ("g3", "recal", "b"))
                + f"; option B - G3 Brier {pdiff[(k, x)][0]:+.4f} [{pdiff[(k, x)][1]:+.4f}, {pdiff[(k, x)][2]:+.4f}]")
    out("   mean P against the observed share in recorded classes 0 and 1, option B (model seeds):")
    for k in (k1, k2):
        out(f"      {G5F.report_name(k):<14} " + "; ".join(f"x {x}: " + ", ".join(
            f"class {g} {pe[(k, x)]['b']['per_class'][g][0]:.3f} vs {pe[(k, x)]['b']['per_class'][g][1]:.3f}"
            for g in ("0", "1") if g in pe[(k, x)]["b"]["per_class"]) for x in XS))
    out("   G3b check rules (R1 Brier not worse than G3's P; R2 classes 0 and 1 within 0.02; R3 gap over bins of 200+ <= 0.05), "
        "reported; option B | recalibrated P:")
    for k in methods:
        out(f"      {G5F.report_name(k):<14} " + "; ".join(
            f"x {x} " + " | ".join(("PASS" if prule[(k, x)][f]["ok"] else "FAIL (" + ", ".join(r for r in ("R1", "R2", "R3")
                                                                                  if not prule[(k, x)][f][r]) + ")")
                                   for f in ("option_b", "recal")) for x in XS))
    report["p_le"] = {f"{G5F.report_name(k)} x {x}": dict(scores=v, rules=prule[(k, x)], brier_b_minus_g3=pdiff[(k, x)])
                      for (k, x), v in pe.items()}
    # ---- T1-T3
    rep_t, vt, vt_g, lines = t_tests(S, tru, tc, comm, tested=TESTED if t3_on else ("T1", "T2"))
    out("   T1-T3 by true class, entered stacks (truth | model | control mean; error model | control; model - control error "
        "[95% interval]; coverage model | control); B3 numbers in the json only:")
    for lab, v, ne, dr, bal, per, verdict, verdict_g, gain in lines:
        if v == "T3" and not t3_on:
            out(f"   {lab:7s} T3 dropped: the T3 inputs failed Z7 (stated, not scored)")
            continue
        out(f"   {lab:7s} {v:7s} entered {ne:,}; dropped: truth undefined {dr['truth']:,}, model {dr['model']:,}, control "
            f"{dr['control']:,}; balanced (classes 1 .. 51+) {bal['mean']:+.4f} [{bal['lo']:+.4f}, {bal['hi']:+.4f}]; model error "
            f"{100 * gain:.1f}% below the control's; " + (f"interval only {verdict}, binding (with the 10% gain) {verdict_g}"
                                                           if verdict not in ("report", "dropped") else "report"))
        for c in [0] + SCORE:
            r = per[TRUE[c]]
            if r["n"] == 0:
                continue
            out(f"      class {TRUE[c]:>5s} n {r['n']:6,}: {r['truth']:.3f} | {r['model']:.3f} | {r['control']:.3f}; err "
                f"{r['err_model']:.3f} | {r['err_control']:.3f}; diff {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]; "
                f"cov {r['cov_model']:.2f} | {r['cov_control']:.2f}")
    out("   mean of the two seeds, balanced model - control error (report): " + "; ".join(
        f"{v} {rep_t[f'mean of seeds {v}']['diff']:+.4f}" for v in TESTS if t3_on or v != "T3"))
    report["T"] = dict(tests=rep_t, verdict_interval=vt, verdict_binding=vt_g, t3_scored=t3_on, min_gain=MIN_GAIN,
                       control=f"(k + {PT.SHIFT} t) mod {K}",
                       z7=dict(habitat=hab_ok, niche=nic_ok, a4=a4, m3=m3, agree=agree, lines=lines3))
    # ---- G4c split of T3 (report only)
    if t3_on:
        tf, ts = T3C.parts(T6[:, None], fell, sink)
        Ttr = {"fell": tf[:, 0], "sink": ts[:, 0]}
        tab, reading = {}, {}
        for si in (0, 1):
            for q, lab in (("fell", "T3 (growth rate fell)"), ("sink", "D2 (sink cells, r < 0)")):
                out(f"   G4c split, seed {si + 1}, {lab}: class-balanced mean over true classes 1 .. 51+ [95% interval]; truth | "
                    "model | model/truth (report only)")
                for gname, gm in (("shown", ~blank), ("blanked", blank)):
                    for t in range(W - 1):
                        tv = np.where(gm, Ttr[q][:, t], np.nan)
                        mv = np.where(gm & ~np.isnan(tv), M4c[(si, q)][:, t], np.nan)
                        tv = np.where(~np.isnan(mv), tv, np.nan)
                        bt, bm = T3C.bal(tv, tc, comm), T3C.bal(mv, tc, comm)
                        ratio = bm["mean"] / bt["mean"] if bt["mean"] else float("nan")
                        tab[f"seed {si + 1} {q} {gname} {T3C.CH[t]}"] = dict(truth=bt, model=bm, ratio=ratio)
                        out(f"      habitat {gname:7s} change {T3C.CH[t]}{' (into target year)' if t == W - 2 else ''}: n "
                            f"{bm['n']:6,}; {bt['mean']:+.4f} [{bt['lo']:+.4f}, {bt['hi']:+.4f}] | {bm['mean']:+.4f} "
                            f"[{bm['lo']:+.4f}, {bm['hi']:+.4f}] | {ratio:.2f}")
                kk_ = lambda g_, t: tab[f"seed {si + 1} {q} {g_} {T3C.CH[t]}"]
                last = kk_("shown", W - 2)
                r1 = (last["ratio"] >= 0.5 and last["model"]["lo"] > 0) if last["truth"]["lo"] > 0 else None
                below = lambda e_: (e_["model"]["mean"] < 0.25 * e_["truth"]["mean"]) if e_["truth"]["mean"] > 0 else None
                b2 = [below(kk_("shown", t)) for t in range(W - 2)]
                reading[f"seed {si + 1} {q}"] = dict(R1=r1, R2=None if None in b2 else all(b2), R3=below(kk_("blanked", W - 2)))
                out(f"      G4c reading ({q}): " + ", ".join(f"{a} {b}" for a, b in reading[f"seed {si + 1} {q}"].items()))
        report["G4c_split"] = dict(tables=tab, reading=reading)
    # ---- files
    with open(out_dir / "per_stack_g5d.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["stack", "community", "file_index", "row", "true_class", "recorded_class", "records_6y_bin", "blank", "aoo_true", "eoo_true"]
        head += [f"auc_{s}" for s in A]
        for k in methods:
            kk = G5F.report_name(k).replace(" ", "_")
            head += [f"{kind}_{q}_{kk}" for kind in ("aoo", "eoo") for q in ("mean", "lo", "hi")] + [f"p_aoo_le_{x}_{kk}" for x in XS]
        head += [f"truth_{v}" for v in TESTS] + [f"{m}_{v}_mean" for m in ("s1", "s2", "c1", "c2") for v in TESTS]
        w.writerow(head)
        for i in range(n):
            row = [i, int(comm[i]), int(fidx[i]), int(rows[i]), TRUE[tc[i]], TCLS[rcls[i]], BINS[rbin[i]], int(blank[i]), int(ya[i]), f"{ye[i]:.2f}"]
            row += [f"{A[s][i]:.5f}" for s in A]
            for k in methods:
                for kind in ("aoo", "eoo"):
                    row += [f"{pred[k][kind][q][i]:.3f}" for q in ("mean", "L", "U")]
                row += [f"{v:.4f}" for v in pred[k]["p_b"][i]]
            row += [f"{tru[v][i]:.6g}" for v in TESTS] + [f"{S[(m, v, 'mean')][i]:.6g}" for m in ("s1", "s2", "c1", "c2") for v in TESTS]
            w.writerow(row)
    report["minutes"] = round((time.time() - t0) / 60, 1)
    (out_dir / "g5d_report.json").write_text(json.dumps(report, indent=1, default=float))
    n_ok = sum(res.values())
    f5txt = ", ".join(f"{lab} {f5[lab]['verdict']}" for lab in ("seed 1", "seed 2"))
    itxt = ", ".join(f"{G5F.report_name(k)} {kind.upper()} {'PASS' if rule_i[(k, kind)]['ok'] else 'FAIL'}" for k in methods for kind in ("aoo", "eoo"))
    ttxt = ", ".join(f"{k} {v}" for k, v in vt_g.items())
    out(f"   written g5d_predicted.npz, g5d_t_stats.npz, per_stack_g5d.csv, g5d_report.json md5 {X.file_md5(out_dir / 'g5d_report.json')} "
        f"({report['minutes']} min)")
    out(f"stage G5d{tag}: gates {n_ok} PASS, {len(res) - n_ok} FAIL; F5 (both B1 versions) {f5txt}; AOO/EOO rule {itxt}; "
        f"T1-T3 binding rule {ttxt}" + ("" if t3_on else "; T3 dropped (Z7)") + "; no overall verdict")
    return res, dict(report=report, f5=f5, rule_i=rule_i, vt=vt, vt_g=vt_g, t3_on=t3_on)


# ---------------------------------------------------------------- self-test
def augment(path, env, hseed, tseed):
    """Give a planted community file the G0 habitat keys (assembly rule, stand-in habitat_layer) and G4a niche keys."""
    z = dict(np.load(path))
    rounds, M = len(z["ENV_habitat_t"]), int(np.asarray(z["HABITAT_substeps"]).item())
    Wst, tau, stau, contrast, ls, sd = 5000, 8000.0, 62500.0, 0.25, 3.0, 1.0
    rho_s, rho_ms = np.exp(-Wst / M / tau), np.exp(-Wst / M / stau)
    U = HC.toy_layer(G, G, ls, seed=[hseed, 1]); m = 0.0
    shift, Ht, Hs = {}, [], []
    for t in range(rounds):
        for j in range(M):
            if not (t == 0 and j == 0):
                U = rho_s * U + np.sqrt(1 - rho_s ** 2) * HC.toy_layer(G, G, ls, seed=[hseed, 104729, t, j])
                U = (U - U.mean()) / U.std()
                eps = float(np.random.default_rng([hseed, 104743, t, j]).standard_normal())
                m = rho_ms * m + np.sqrt(1 - rho_ms ** 2) * sd * eps
            shift[(t, j)] = m
        Ht.append(contrast * (U + m)); Hs.append(m)
    Ht = np.array(Ht, np.float32)
    ids = np.unique(np.asarray(z["ASM_F_ids"]))
    prm = dict(seed=tseed, opt_sd=0.4, w_min=0.2, w_max=1.0, peak_max=1.5, tradeoff=1.0)
    keep = ids[:min(10, len(ids))]
    tr = np.array([env.niche_traits(int(s), **prm) for s in keep]); cost = 0.1
    z.update(HABITAT_seed=np.int64(hseed), HABITAT_ls=np.float64(ls), HABITAT_contrast=np.float64(contrast),
             HABITAT_tau=np.int64(tau), HABITAT_substeps=np.int64(M), window_steps=np.int64(Wst),
             HABITAT_rho=np.float64(np.exp(-Wst / tau)), HABITAT_shift_sd=np.float64(sd), HABITAT_shift_tau=np.int64(stau),
             HABITAT_shift_rho=np.float64(np.exp(-Wst / stau)), ENV_habitat_t=Ht, ENV_habitat_shift=np.array(Hs),
             ASM_F_shift=np.array([shift[(int(a), int(b))] for a, b in zip(z["ASM_F_round"], z["ASM_F_piece"])]),
             TRAIT_seed=np.int64(tseed), NICHE_opt_sd=np.float64(0.4), NICHE_w_min=np.float64(0.2), NICHE_w_max=np.float64(1.0),
             NICHE_peak_max=np.float64(1.5), NICHE_tradeoff=np.float64(1.0), NICHE_cost=np.float64(cost),
             NICHE_opt=tr[:, 0].astype(np.float32), NICHE_w=tr[:, 1].astype(np.float32), NICHE_peak=tr[:, 2].astype(np.float32),
             ENV_r_field=np.stack([env.niche_r(Ht[-1].astype(np.float64), *tr[i], cost) for i in range(len(tr))]).astype(np.float32))
    np.savez(path, **z)


def build_world(tmp, tools, out=print):
    """Planted pipeline in G5 order: obj1-like files with habitat and niche keys; real_train and real_val exports, stage D,
    two tiny runs (step 3a, G1), G2, G3, G3b, G4b on real_val, G5a fits while real_eval does not exist; then the real_eval
    export, G5b and the four G5c parts."""
    quiet = lambda *a, **k: None
    tmp = Path(tmp)
    (tmp / "work").mkdir(parents=True)
    src = "import numpy as np\n" + inspect.getsource(HC.toy_layer).replace("def toy_layer", "def habitat_layer") + PT.ENV_EXTRA
    (tmp / "work" / "environment.py").write_text(src)
    env, _ = PT.load_env(tmp / "work")
    py, refpat = X.plant_grids(tmp / "g")
    sims = {}
    for sp, seeds in (("real_train", (7001, 7002, 7003)), ("real_val", (7101, 7102, 7103, 7104)), ("real_eval", (7201, 7202, 7203))):
        d = tmp / "sim" / sp; d.mkdir(parents=True)
        for s_ in seeds:
            X.plant_community(d / f"c{s_}_training.npz", s_)
            augment(d / f"c{s_}_training.npz", env, 105000 + s_, 200000 + s_)
        sims[sp] = str(d / "*_training.npz")
    base = tmp / "data"
    for sp in ("real_train", "real_val"):
        X.export(sorted(glob.glob(sims[sp])), sp, str(base / sp), 0.3, py, refpat, tools, out=quiet)
    D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
    hres, hinfo = HR.run(str(base), str(base / "real_train"), str(base / "real_val"), str(base), out=quiet)
    e1, e2 = round(hinfo["bal"]["B1"], 3), round(hinfo["bal"]["B2"], 3)
    runs = []
    for sd in (1, 2):
        cfg = dict(CK.F.CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=2, K=2, ddim=4,
                   val_set="full", val_chunk=64, lr=1e-3, seed=sd)
        rd = tmp / "runs" / f"run{sd}"
        CK.F.train(cfg, str(base), str(rd), "cpu", log=quiet)
        CK.run(rd, base, K=50, ddim=4, dev="cpu", expect_b1=e1, expect_b2=e2, out=quiet)
        GS.run_one(rd, base, K=50, ddim=4, dev="cpu", out=quiet)
        runs.append(rd)
    GB.run(base, base / "real_train", base / "real_val", tmp / "g2", runs=runs, K=50, out=quiet)
    r3, _ = GC.run(base, runs, tmp / "g2", base / "stage_g3_calibration", out=quiet)
    r3b, _ = GR.run(base / "stage_g3_calibration", base / "stage_g3b_recalibration", out=quiet)
    r4b, _, _ = PT.run(base, runs, tmp / "g2", sorted(glob.glob(sims["real_val"])), tmp / "work", base / "stage_g4b_process", out=quiet)
    r5a, _ = G5F.run(base / "stage_g3_calibration", base / "stage_g3b_recalibration", base, base / "stage_g5_fits", out=quiet)
    time.sleep(1.2)
    (base / "habitat_sigma.json").write_text(json.dumps(dict(sigma=0.3, ok=True, val=0.80)))
    X.export(sorted(glob.glob(sims["real_eval"])), "real_eval", str(base / "real_eval"), 0.3, py, refpat, tools, final_eval=True, out=quiet)
    XC = _load("stage_g5_export_check")
    r5b, _ = XC.post(sims["real_eval"], base, base / "stage_g5_fits", base / "habitat_sigma.json", base / "stage_g5b_export_check", out=quiet)
    g5c, g5b = base / "stage_g5_predict", base / "stage_g5b_export_check"
    rc = [P5.part_frozen(base, g5b, g5c, True, out=quiet), P5.part_baselines(base, tmp / "g2", g5b, g5c, True, out=quiet)]
    rc += [P5.part_samples(base, r, g5b, g5c, True, "cpu", out=quiet) for r in runs]
    rc.append(P5.part_report(base, runs, g5b, g5c, True, out=quiet))
    out(f"   planted world: G3 {all(r3.values())}, G3b {all(r3b.values())}, G4b {all(r4b.values())}, G5a {all(r5a.values())}, "
        f"G5b {all(r5b.values())}, G5c parts {[all(r.values()) for r in rc]}")
    return dict(data=base, runs=runs, g5c=g5c, g5b=g5b, fits=base / "stage_g5_fits", g3b=base / "stage_g3b_recalibration",
                g4b=base / "stage_g4b_process", files=sorted(glob.glob(sims["real_eval"])), work=tmp / "work",
                ok=all(r3.values()) and all(r3b.values()) and all(r4b.values()) and all(r5a.values()) and all(r5b.values())
                and all(all(r.values()) for r in rc))


def kw(w, out_dir, **extra):
    d = dict(data=w["data"], runs=w["runs"], g5c=w["g5c"], g5b=w["g5b"], fits_dir=w["fits"], g3b=w["g3b"], g4b=w["g4b"],
             obj1_files=w["files"], obj1_work=w["work"], out_dir=out_dir, final_eval=True)
    d.update(extra)
    return d


def sub(bug, w, out_dir, extra_env=None, **extra):
    """One G5d run with a planted bug, in a fresh process (the bug switch is read at import)."""
    env = dict(os.environ, STAGE_G5D_BROKEN=bug, **(extra_env or {}))
    args = {k: (str(v) if isinstance(v, Path) else ([str(x) for x in v] if isinstance(v, list) else v)) for k, v in kw(w, out_dir, **extra).items()}
    code = ("import importlib.util as u, json; s=u.spec_from_file_location('f', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
            "r, _ = m.run(out=lambda *a, **k: None, **json.loads(%r)); print('RES', json.dumps(r))" % (str(Path(__file__).resolve()), json.dumps(args)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    line = next((x for x in p.stdout.splitlines() if x.startswith("RES ")), None)
    return json.loads(line[4:]) if line else {"error": p.stderr.strip()[-300:]}


def self_test(tools, keep=None):
    quiet = lambda *a, **k: None
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    # ---- unit checks
    g = np.random.default_rng(5)
    n = 3000
    y = g.poisson(4, n).astype(float); rcls = g.integers(0, 7, n); fold = g.integers(0, 2, n); P = g.random(n)
    a = p_eval(P, y, rcls, fold_ref(y, fold, rcls, 5), 5)
    b = GR.evaluate(P, y, fold, rcls, 5)
    t("U1", all(close(a[q], b[q], 0) for q in ("brier", "brier_ref", "bss", "gap", "gap_z", "gap200", "mean_p", "share"))
      and a["per_class"] == b["per_class"], "p_eval with the fold reference = G3b's evaluate (every score, exactly)")
    tc = g.integers(0, 7, n); comm = g.integers(1, 6, n); A1 = g.random(n); A0 = A1 - 0.01
    cb, v, ov = f5_rule(A1, A0, tc, comm)
    t("U2", ov == "PASS" and all(x == "PASS" for x in v.values()) and f5_rule(A0, A1, tc, comm)[2] == "FAIL",
      "F5 rule: a model 0.01 above B1 in every stack passes every class; 0.01 below fails")
    # ---- planted world
    ctx = tempfile.TemporaryDirectory() if keep is None else None
    tmp = Path(ctx.name if ctx else keep)
    t0 = time.time()
    w = build_world(tmp, tools)
    t("S0", w["ok"], f"planted world built in G5 order (G3, G3b, G4b, G5a before the real_eval export; G5b; G5c) "
      f"({(time.time() - t0) / 60:.1f} min)")
    mk = w["data"] / "real_eval" / MARKER
    lines = []
    r, _ = run(**kw(w, tmp / "chk", check_only=True), out=lines.append)
    t("S1", all(r.values()) and set(r) == {f"Z{i}" for i in range(7)} and not mk.exists() and not (tmp / "chk").exists(),
      f"check-only: Z0-Z6 PASS, no marker, nothing written ({r})")
    lines.append("---- full run")
    r, info = run(**kw(w, tmp / "g5d"), out=lines.append)
    t("S2", all(r.values()) and set(r) == {f"Z{i}" for i in range(12)} and mk.exists() and info["t3_on"]
      and (tmp / "g5d" / "g5d_report.json").is_file(), f"full run: Z0-Z11 PASS, marker written, T3 scored, report written ({r})")
    rep = json.loads((tmp / "g5d" / "g5d_report.json").read_text())
    t("S3", rep["rerun"] == 0 and len(mk.read_text().splitlines()) == 1 and "verdict" in rep["F5"]["seed 1"]
      and "B1_exact" in rep["F5"]["seed 1"] and rep["intervals"]["eoo_cap"] == 361 and "G4c_split" in rep
      and set(rep["T"]["verdict_binding"]) == {f"seed {s} {v}" for s in (1, 2) for v in TESTED},
      "report: first reading, F5 against both B1 versions, EOO cap split at 361, binding T verdicts per seed, G4c split")
    stop = []
    for f in (lambda: run(**kw(w, tmp / "again"), out=quiet), lambda: run(**kw(w, tmp / "g5d", rerun=True), out=quiet),
              lambda: run(**kw(w, tmp / "x", final_eval=False), out=quiet),
              lambda: run(**kw(w, w["data"] / "real_eval" / "o", rerun=True), out=quiet)):
        try:
            f(); stop.append(False)
        except SystemExit:
            stop.append(True)
    t("S4", all(stop) and not (tmp / "again").exists() and len(mk.read_text().splitlines()) == 1,
      f"refused: a second reading without --rerun {stop[0]} (nothing written); an output folder that holds outputs {stop[1]}; "
      f"no --final-eval {stop[2]}; output inside real_eval {stop[3]}")
    r, info = run(**kw(w, tmp / "g5d_rerun", rerun=True), out=quiet)
    rep2 = json.loads((tmp / "g5d_rerun" / "g5d_report.json").read_text())
    t("S5", all(r.values()) and rep2["rerun"] == 1 and len(mk.read_text().splitlines()) == 2
      and rep2["md5"]["predicted"] == rep["md5"]["predicted"] and rep2["md5"]["t_stats"] == rep["md5"]["t_stats"],
      "a rerun with --rerun is reported as rerun 1, the marker holds two readings, and the outputs are the same")
    mk.unlink()
    try:
        run(**kw(w, tmp / "no_marker_rerun", rerun=True), out=quiet); s6 = False
    except SystemExit:
        s6 = True
    t("S6", s6, "--rerun without a marker is refused (the first reading runs without it)")
    # ---- planted bugs before the marker: the gate fails, no marker, no output
    def copy(src, name):
        d = tmp / name
        subprocess.run(["cp", "-r", str(src), str(d)], check=True)
        return d
    rw = _load("stage_g5_export_check").rewrite
    c = copy(w["g5c"], "b_tamper")
    rw(c / f"samples_{w['runs'][0].name}.npz", lambda a: a.__setitem__("maps", a["maps"] ^ np.uint8(1)))
    r = sub("", w, tmp / "o_tamper", g5c=c)
    t("bug tamper", r.get("Z1") is False and not mk.exists() and not (tmp / "o_tamper").exists(),
      f"a samples file changed after G5c fails Z1; no marker, nothing written ({r})")
    c = copy(w["g5c"], "b_early")
    j = json.loads((c / "g5c_report.json").read_text()); j["written_utc"] = "2000-01-01T00:00:00+00:00"
    (c / "g5c_report.json").write_text(json.dumps(j))
    r = sub("", w, tmp / "o_early", g5c=c)
    t("bug early_report", r.get("Z1") is False and not mk.exists(), f"a g5c report written before its parts fails Z1 ({r})")
    c = copy(w["fits"], "b_fits")
    rw(c / "g5_fits.npz", lambda a: a.__setitem__("base_rate", a["base_rate"] * 1.01))
    r = sub("", w, tmp / "o_fits", fits_dir=c)
    t("bug fits_changed", r.get("Z2") is False and not mk.exists(), f"fits changed after G5a fail Z2 ({r})")
    for bug, gate_ in (("truth_early", "Z6"), ("score_recorded", "Z5"), ("p_ref", "Z5"), ("t_err", "Z5")):
        r = sub(bug, w, tmp / f"o_{bug}")
        t(f"bug {bug}", r.get(gate_) is False and not mk.exists() and not (tmp / f"o_{bug}").exists(),
          f"{gate_} fails as it must; no marker, nothing written ({r})")
    # ---- planted bugs after the marker
    for bug, gate_, ex in (("shift_truth", "Z9", None), ("stats_changed", "Z10", None),
                           ("leak", "Z10", {"STAGE_G4B_BROKEN": "leak"}), ("", "Z7", {"STAGE_G4B_BROKEN": "hab_seed"})):
        r = sub(bug, w, tmp / f"o2_{bug or 'hab'}", extra_env=ex)
        ok = r.get(gate_) is False and mk.exists()
        if gate_ == "Z7":
            jj = json.loads((tmp / "o2_hab" / "g5d_report.json").read_text())
            ok &= jj["T"]["t3_scored"] is False and jj["T"]["verdict_binding"].get("seed 1 T3") == "dropped" \
                and jj["T"]["verdict_binding"].get("seed 1 T1") in ("PASS", "FAIL")
        if mk.exists():
            mk.unlink()
        t(f"bug {bug or 'hab_seed'}", ok, f"{gate_} fails as it must after the marker" +
          ("; T3 dropped and stated, T1 and T2 still scored" if gate_ == "Z7" else "") + f" ({r})")
    print("   planted run log (check-only, then the full run):", *lines, sep="\n      ")
    if ctx:
        ctx.cleanup()
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true"); ap.add_argument("--keep")
    ap.add_argument("--final-eval", action="store_true"); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--g5c", default="../data/stage_g5_predict")
    ap.add_argument("--g5b", default="../data/stage_g5b_export_check")
    ap.add_argument("--fits", default="../data/stage_g5_fits")
    ap.add_argument("--g3b", default="../data/stage_g3b_recalibration")
    ap.add_argument("--g4b", default="../data/stage_g4b_process")
    ap.add_argument("--obj1-work", default="../../obj1_v2/work")
    ap.add_argument("--files", default="../../obj1_v2/work/results/stage2a/real_eval/*_training.npz")
    ap.add_argument("--out", default="../data/stage_g5d_score")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools, a.keep) else 1)
    files = sorted(glob.glob(a.files))
    if not files:
        sys.exit(f"STOP: --files matches no file: {a.files}")
    r, _ = run(a.data, a.runs, a.g5c, a.g5b, a.fits, a.g3b, a.g4b, files, a.obj1_work, a.out, a.final_eval,
               check_only=a.check_only, rerun=a.rerun, chunk=a.chunk, out=lambda s: print(s, flush=True))
    sys.exit(0 if all(r.values()) else 1)