#!/usr/bin/env python3
"""
test_25_f0a_spread_check.py  (obj2_v2, self-test of tools/stage_f0_spread_check.py, 13 Nov 2026)

Run through the tool:  python3 ../tools/stage_f0_spread_check.py --self-test 2>&1 | tail -n 1   (from obj2_v2/work)
Planted data only, in a temporary folder, with the real tools: the planted pipeline of H2a's self-test (P5.build_world), G3
on its real_val, H1's planted grids exported as a fit half, the three H2a parts, H2b (stage_h2_bnm_score.run, floor 5),
and an h3_freeze.json built by test_23's build_freeze (the way stage_h3_freeze.py writes it). Nothing of ../data is read.
    U1  stack-first mean; a score too flat with distance gets gamma > 0 and a higher AUC; equal AUC on the whole grid gives
        gamma 0; out-of-fold scoring uses the other fold's gamma
    T1  the tool on the planted fit half (floor 5): gates A0-A4 PASS, report and scores written, the rule printed
    T2  recomputed outside the tool: out-of-fold AUC of 60 stack-repeats from score x d^-gamma (export_obj2.auc); the gamma
        of every class and fold by a brute grid search with an own stack-first mean
    T3  refused: a second run into the same folder; an output inside the H2b folder
    T4  planted bugs, each fails its gate: gamma fitted on both folds (A3), a test-half path loaded (A4), labels shifted
        (A1), H-R1 code shifted (A2), an H2b file changed after the freeze (A0)
"""
import importlib.util, os, shutil, tempfile
from pathlib import Path
import numpy as np


def load_tool(tool_path):
    spec = importlib.util.spec_from_file_location("stage_f0_spread_check_t", tool_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def build_fit(T, tmp, obj1, tool_path):
    """Planted fit half with the real tools (the fit-half part of test_23's build_world) and its freeze file."""
    quiet = lambda *a, **k: None
    H6 = T.H6
    S2, H3, H2A, GC = H6.S2, H6.H3, H6.H2A, H6.GC
    H1, P5 = H3.H1, S2.P5
    p23 = Path(tool_path).resolve().parent.parent / "tests" / "test_23_h6_test_score.py"
    spec = importlib.util.spec_from_file_location("test_23_t", p23); t23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t23)
    w = P5.build_world(tmp, obj1)
    data, g2, runs = Path(w["data"]), Path(w["g2"]), [Path(r) for r in w["runs"]]
    g3 = data / "stage_g3_calibration"
    r3, _ = GC.run(data, runs, g2, g3, out=quiet)
    py, refpat = H1.world(Path(tmp) / "bw")
    bnm, h2a, h2b = data / "bnm_fit", data / "stage_h2_predict", data / "stage_h2b_score"
    r1, _ = H1.run(py, refpat, obj1, bnm, expect=(6, 6), out=quiet)
    ra = [H2A.part_baselines(data, bnm, g2, h2a, out=quiet)] + [H2A.part_samples(data, bnm, r, h2a, "cpu", out=quiet) for r in runs]
    ra.append(H2A.part_report(bnm, runs, h2a, out=quiet))
    rb, _ = S2.run(data, bnm, h2a, runs, g3, h2b, floor=5, out=quiet)
    fzp = t23.build_freeze(H6, data, bnm, h2a, h2b, g2, g3, runs, obj1, data / "stage_h3_freeze" / "h3_freeze.json")
    ok = all(r3.values()) and all(r1.values()) and all(all(x.values()) for x in ra) and all(rb.values())
    return dict(bnm=bnm, h2a=h2a, h2b=h2b, g2=g2, g3=g3, freeze=fzp, runs=runs, obj1_tools=obj1, freeze_md5=H6.X.file_md5(fzp),
                h6_md5=H6.X.file_md5(Path(H6.__file__)), floor_override=5), ok


def unit_checks(T, t):
    S2 = T.S2
    stid = np.array([0] * 5 + [1]); a = np.array([1.0] * 5 + [0.0])
    m1 = T.sf_mean(a, stid, np.ones(6, bool))
    # planted holes: recorded share falls fast with distance, the model's score is flat beyond 2 cells
    g = np.random.default_rng(7)
    n, per = 120, 30
    dist = np.concatenate([g.uniform(1, 9, per) for _ in range(n)])
    lab = g.random(n * per) < 0.6 * np.exp(-(dist - 1))
    score = np.where(dist < 2, 0.5, 0.2) + 0.05 * g.random(n * per)
    ptr = np.r_[0, np.cumsum(np.full(n, per))]
    F = dict(n=n, ptr=ptr, lab=lab, dist=dist, score={"s1": score}, rcls=np.zeros(n, int), fold=np.arange(n) % 2,
             stid=np.arange(n))
    Ag = T.grid_aucs(F, "s1")
    gam = T.fit_gamma(Ag, F, 0)
    flat = dict(F, score={"s1": np.zeros(n * per)})                 # no sample occupies any hole: every gamma ties
    gflat = T.fit_gamma(T.grid_aucs(flat, "s1"), flat, 0)
    A, gi = T.oof(Ag, F, {0: np.full(7, 1.0), 1: np.full(7, 2.0)})
    j1, j2 = int(np.flatnonzero(T.GRID == 1.0)[0]), int(np.flatnonzero(T.GRID == 2.0)[0])
    oof_ok = np.allclose(A[F["fold"] == 0], Ag[j2][F["fold"] == 0], equal_nan=True) and np.allclose(A[F["fold"] == 1], Ag[j1][F["fold"] == 1], equal_nan=True)
    j = int(np.flatnonzero(T.GRID == gam[0])[0])
    better = T.sf_mean(Ag[j], F["stid"], F["fold"] == 0) > T.sf_mean(Ag[0], F["stid"], F["fold"] == 0)
    t("U1", abs(m1 - 0.5) < 1e-12 and gam[0] > 0 and better and gflat[0] == 0.0 and oof_ok,
      f"stack-first mean {m1:.3f} (stack-repeat mean 0.833); flat-with-distance score gets gamma {gam[0]:.1f} and a higher fold AUC {better}; "
      f"equal AUC over the grid gives gamma {gflat[0]:.1f}; fold 0 scored with fold 1's gamma and fold 1 with fold 0's {oof_ok}")


def checks(T, w, tmp, t):
    """T1-T4 on a planted fit half w (from build_fit on Sara's machine, or the sandbox stub world)."""
    quiet = lambda *a, **k: None
    tmp = Path(tmp)
    kw = dict(bnm=w["bnm"], h2a=w["h2a"], h2b=w["h2b"], g2=w["g2"], g3=w["g3"], freeze=w["freeze"], runs=w["runs"], obj1_tools=w["obj1_tools"],
              freeze_md5=w["freeze_md5"], h6_md5=w["h6_md5"], floor_override=w["floor_override"])
    lines = []
    r, rep = T.run(out_dir=tmp / "f0a", out=lines.append, **kw)
    t("T1", all(r.values()) and set(r) == {"A0", "A1", "A2", "A3", "A4"} and (tmp / "f0a" / "f0a_report.json").is_file()
      and "worth_building" in rep.get("rule", {}) and lines[-1].startswith("stage F0 gate A: gates 5 PASS"),
      f"planted fit half: {r}; last line: {lines[-1][:150]}")
    # T2: outside recomputation (reported as FAIL, without a crash, when T1 wrote nothing)
    if not (tmp / "f0a" / "f0a_scores.npz").is_file():
        t("T2", False, "skipped: T1 wrote no f0a_scores.npz")
        return lines
    F = T.load_fit(Path(w["bnm"]), Path(w["h2a"]), Path(w["h2b"]))
    sc = np.load(tmp / "f0a" / "f0a_scores.npz")
    X = T.X
    rk = np.flatnonzero(~np.isnan(sc["auc_oof_s1"]))
    pick = rk[np.random.default_rng(3).choice(len(rk), min(60, len(rk)), replace=False)]
    auc_ok = True
    for i in pick:
        a, b = F["ptr"][i], F["ptr"][i + 1]
        s_ = F["score"]["s1"][a:b] * np.maximum(F["dist"][a:b], 1.0) ** (-sc["gamma_item_s1"][i])
        auc_ok &= abs(X.auc(s_, F["lab"][a:b]) - sc["auc_oof_s1"][i]) < 1e-9
    gam_ok = True
    for m in T.SEEDS:
        for f in (0, 1):
            best = []
            for c in range(len(T.TCLS)):
                vals = []
                for gv in T.GRID:
                    per = {}
                    for i in np.flatnonzero((F["fold"] == f) & (F["rcls"] == c)):
                        a, b = F["ptr"][i], F["ptr"][i + 1]
                        lab = F["lab"][a:b]
                        if 0 < lab.sum() < len(lab):
                            s_ = F["score"][m][a:b] * np.maximum(F["dist"][a:b], 1.0) ** (-gv)
                            per.setdefault(int(F["stid"][i]), []).append(X.auc(s_, lab))
                    vals.append(np.mean([np.mean(v) for v in per.values()]) if per else -1.0)
                vals = np.array(vals)
                best.append(T.GRID[int(np.flatnonzero(vals >= vals.max() - 1e-12)[0])] if (vals > -1).any() else 0.0)
            gam_ok &= np.allclose(best, sc[f"gamma_fold{f}_{m}"])
    t("T2", auc_ok and gam_ok, f"recomputed outside the tool: out-of-fold AUC of {len(pick)} stack-repeats (export_obj2.auc) {auc_ok}; gamma of "
      f"every class, fold and seed by a brute grid search {gam_ok}")
    # T3: refusals
    stop = []
    for f_ in (lambda: T.run(out_dir=tmp / "f0a", out=quiet, **kw), lambda: T.run(out_dir=Path(w["h2b"]) / "x", out=quiet, **kw)):
        try:
            f_(); stop.append(False)
        except SystemExit:
            stop.append(True)
    t("T3", all(stop), f"refused: a second run into the same folder {stop[0]}, an output inside the H2b folder {stop[1]}")
    # T4: planted bugs
    for bug, g_ in (("leak", "A3"), ("test_path", "A4"), ("labels_shift", "A1"), ("core_shift", "A2")):
        os.environ["STAGE_F0A_BROKEN"] = bug
        try:
            rb, _ = T.run(out_dir=tmp / f"b_{bug}", out=quiet, **kw)
        finally:
            os.environ.pop("STAGE_F0A_BROKEN", None)
        t(f"bug {bug}", rb.get(g_) is False and not (tmp / f"b_{bug}").exists(), f"{g_} fails and nothing is written ({rb})")
    hx = tmp / "h2b_changed"
    shutil.copytree(w["h2b"], hx)
    z = dict(np.load(hx / "h2b_scores.npz")); z["y_strict"] = z["y_strict"] + 1
    T.X.save_npz(hx / "h2b_scores.npz", z)
    rb, _ = T.run(out_dir=tmp / "b_h2b", out=quiet, **dict(kw, h2b=hx))
    t("bug h2b_changed", rb.get("A0") is False and len(rb) == 1, f"an H2b file changed after the freeze fails A0 and stops ({rb})")
    return lines


def self_test(tool_path, obj1):
    T = load_tool(tool_path)
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    unit_checks(T, t)
    with tempfile.TemporaryDirectory() as tmp:
        w, ok = build_fit(T, tmp, obj1, tool_path)
        t("S0", ok, "planted pipeline: G3, H1 fit half, H2a parts, H2b (floor 5), freeze built as H3 writes it")
        lines = checks(T, w, tmp, t)
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)