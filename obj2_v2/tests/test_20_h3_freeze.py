#!/usr/bin/env python3
"""
test_20_h3_freeze.py  (obj2_v2, self-test of tools/stage_h3_freeze.py, 12 Nov 2026)

Run through the tool:  python3 ../tools/stage_h3_freeze.py --self-test | tail -1   (from obj2_v2/work)
Planted data only (a temporary folder); the real tools are used: stage_h2_bnm_score.crossfit_h makes the planted H2b
cross-fit, stage_g_calibrate.fit_params fits, stage_h1_bnm_export.mode_ok reads the freeze file.
    U1  the H3 fit resolver = crossfit_h, options 1 and 2, with groups under 30 distinct stacks (fallback): scale, Q, mean,
        lower and upper end equal
    U2  refit on all items: a group with 30+ stacks = fit_params on that group's items; a small group = its recorded class
        (option 2) or all items; a group with no item = parent or all
    U3  planted H1 / H2a / H2b folders: gates F0-F5 PASS; h3_freeze.json lists the H1 tool's md5, so H1's J0 (mode_ok)
        accepts it for the test half; the options are the H2b options
    U4  a run into the same folder stops; a run into a new folder writes the same h3_freeze.json bytes
    U5  planted bugs, each fails its gate: h2b_scores.npz changed after H2b (F0); a test-half marker (F0); a scale off by
        0.1% (F2); detection 0.79 (F1); a truth file in the np.load log (F4); a wrong H1 tool md5 in the list (F5)
"""
import importlib.util, json, os, shutil, tempfile
from pathlib import Path
import numpy as np


def load_tool(tool_path):
    spec = importlib.util.spec_from_file_location("stage_h3_freeze_t", tool_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def planted_arrays(g, ns=240, nw=8, R=5, K=50):
    stack = np.repeat(np.arange(ns), R); n = ns * R
    win = stack % nw; fold = (win % 2).astype(np.int8)
    rc_s = g.choice(7, ns, p=[0.04, 0.08, 0.18, 0.15, 0.15, 0.2, 0.2]); rb_s = g.integers(0, 5, ns)
    rcls, rbin = np.repeat(rc_s, R).astype(np.int8), np.repeat(rb_s, R).astype(np.int8)
    lam = np.repeat(g.uniform(1, 20, ns), R)
    thin = {m: g.poisson(lam[:, None] * g.uniform(0.6, 1.4), (n, K)).astype(np.uint16) for m in ("s1", "s2", "B3", "B4")}
    thin["B5"] = np.zeros((n, 1), np.uint16)
    y = g.poisson(lam).astype(float)
    return dict(stack=stack, win=win, fold=fold, rcls=rcls, rbin=rbin, thin=thin, y=y, n=n)


def planted_world(t, T, g):
    """data/bnm_fit, data/stage_h2_predict, data/stage_h2b_score, runs/f_seed1-2, g2, g3, obj1 tools; reports with md5s."""
    S2, X = T.S2, T.X
    d = Path(t) / "data"; bnm, h2a, h2b = d / "bnm_fit", d / "stage_h2_predict", d / "stage_h2b_score"
    g2, g3, o1 = d / "stage_g2_baselines_v2", d / "stage_g3_calibration", Path(t) / "obj1_tools"
    runs = [Path(t) / "runs" / "f_seed1", Path(t) / "runs" / "f_seed2"]
    for p in (bnm, h2a, h2b, g2, g3, o1, *[r / "check_g1" for r in runs]):
        p.mkdir(parents=True, exist_ok=True)
    A = planted_arrays(g)
    np.savez(bnm / "bnm_frozen.npz", row=np.arange(A["n"])); np.savez(bnm / "bnm_stacks.npz", stack_id=A["stack"])
    (bnm / "manifest.csv").write_text("input_file\nx_input.npz\n")
    (o1 / "bnm_detection_check.py").write_text("# planted\n")
    j1 = dict(half="fit", gates={f"J{i}": True for i in range(8)},
              md5={"tool": X.file_md5(T.HERE / "stage_h1_bnm_export.py"), "bdc": X.file_md5(o1 / "bnm_detection_check.py"),
                   **{f: X.file_md5(bnm / f) for f in ("bnm_frozen.npz", "bnm_stacks.npz", "manifest.csv")}})
    (bnm / "h1_report.json").write_text(json.dumps(j1))
    best = {}
    for i, r in enumerate(runs):
        (r / "best.pt").write_bytes(bytes([i + 1]) * 64); best[r.name] = X.file_md5(r / "best.pt")
        np.savez(r / "check_g1" / "samples_k50.npz", K=np.int64(50), seed=np.int64(11), ddim=np.int64(50), batch=np.int64(1024),
                 best_md5=np.array(best[r.name]))
        np.savez(h2a / f"samples_{r.name}.npz", K=np.int64(50), seed=np.int64(11), ddim=np.int64(50), batch=np.int64(1024),
                 epoch=np.int64(179 + 20 * i), best_md5=np.array(best[r.name]), maps=np.zeros((3, 50, 300), np.uint8))
    np.savez(h2a / "baselines_bnm.npz", a=np.arange(3))
    files = {f: X.file_md5(h2a / f) for f in ("baselines_bnm.npz", "samples_f_seed1.npz", "samples_f_seed2.npz")}
    ja = dict(half="fit", gates={"R5": True}, truth_opened=0, tool=X.file_md5(T.HERE / "stage_h2_bnm_predict.py"), files=files,
              frozen_md5=j1["md5"]["bnm_frozen.npz"], best_md5={f"samples_{k}": v for k, v in best.items()})
    (h2a / "h2a_report.json").write_text(json.dumps(ja))
    (g2 / "g2_fit.json").write_text("{}"); np.savez(g3 / "g3_calibrated.npz", a=np.arange(2))
    opts = dict(s1=1, s2=1, pooled=1, B3=2, B4=1, B5=1)
    P = dict(rcls=A["rcls"], rbin=A["rbin"], fold=A["fold"], stack_id=A["stack"], **{f"thin_h_{m}": v for m, v in A["thin"].items()})
    np.savez(h2b / "h2b_predicted.npz", **P)
    pc = T.preds(P)
    S = dict(y_strict=A["y"], fold=A["fold"], stack_id=A["stack"])
    for m in S2.R2_M:
        fg = T.group_of(A["rcls"], A["rbin"], opts[m]); pa = A["rcls"].astype(np.int64) if opts[m] == 2 else np.full(A["n"], -1)
        cf = S2.crossfit_h(pc[m], A["y"], np.zeros(A["n"]), A["fold"], fg, pa, "aoo", A["stack"])
        S.update({f"hr2_mean_{m}": cf["mean"], f"hr2_L_{m}": cf["L"], f"hr2_U_{m}": cf["U"], f"hr2_scale_q_{m}": cf["par"]})
    np.savez(h2b / "h2b_scores.npz", **S)
    for f in ("per_stack_h2b.csv", "per_window_h2b.csv"):
        (h2b / f).write_text("x\n")
    jb = dict(gates={f"Y{i}": True for i in range(7)}, n=int(A["n"]), stacks=int(len(np.unique(A["stack"]))),
              md5=dict(tool=X.file_md5(T.HERE / "stage_h2_bnm_score.py"), predicted=X.file_md5(h2b / "h2b_predicted.npz"),
                       scores=X.file_md5(h2b / "h2b_scores.npz"), h2a_report=X.file_md5(h2a / "h2a_report.json"),
                       frozen=j1["md5"]["bnm_frozen.npz"], best_pt=best),
              H_R2={m: dict(option=o) for m, o in opts.items()})
    (h2b / "h2b_report.json").write_text(json.dumps(jb))
    kw = dict(data=d, bnm=bnm, h2a=h2a, h2b=h2b, runs=runs, g2=g2, g3=g3, obj1_tools=o1)
    return kw, A, opts


def self_test(tool_path):
    T = load_tool(tool_path)
    S2, GC = T.S2, T.GC
    res = []
    quiet = lambda *a, **k: None

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    g = np.random.default_rng(20)
    # U1
    A = planted_arrays(g)
    pc = T.preds({f"thin_h_{m}": v for m, v in A["thin"].items()})
    worst, small = 0.0, 0
    for opt in (1, 2):
        fg = T.group_of(A["rcls"], A["rbin"], opt); pa = A["rcls"].astype(np.int64) if opt == 2 else np.full(A["n"], -1)
        cf = S2.crossfit_h(pc["s1"], A["y"], np.zeros(A["n"]), A["fold"], fg, pa, "aoo", A["stack"])
        par = np.zeros((A["n"], 2)); mu = np.zeros(A["n"]); L = np.zeros(A["n"]); U = np.zeros(A["n"])
        for c in (0, 1):
            fit, chk = np.nonzero(A["fold"] == 1 - c)[0], np.nonzero(A["fold"] == c)[0]
            tab = T.resolve(pc["s1"], A["y"], fit, np.unique(fg[chk]), fg, opt, A["stack"])
            small += sum(v["key"] != "g" for v in tab.values())
            for gg, v in tab.items():
                ci = chk[fg[chk] == gg]
                par[ci] = (v["s"], v["q"]); mu[ci], L[ci], U[ci] = T.apply(pc["s1"], ci, v["s"], v["q"])
        for a_, b_ in ((par, cf["par"]), (mu, cf["mean"]), (L, cf["L"]), (U, cf["U"])):
            worst = max(worst, float(np.abs(a_ - b_).max()))
    t("U1", worst == 0.0 and small > 0, f"fit resolver = stage_h2_bnm_score.crossfit_h, options 1 and 2: largest difference {worst:.1e}; "
      f"{small} group fits by fallback (under {GC.N_FIT_MIN} distinct stacks)")
    # U2
    allf = np.arange(A["n"]); fg = T.group_of(A["rcls"], A["rbin"], 2)
    tab = T.resolve(pc["s1"], A["y"], allf, range(35), fg, 2, A["stack"])
    own_ok = all(abs(v["s"] - GC.fit_params(pc["s1"], A["y"], allf[fg == int(k)])[0]) < 1e-15 and
                 abs(v["q"] - GC.fit_params(pc["s1"], A["y"], allf[fg == int(k)])[1]) < 1e-15 for k, v in tab.items() if v["key"] == "g")
    fb_ok = all((v["key"] == f"p{int(k) // 5}" and len(np.unique(A["stack"][fg == int(k)])) < 30) or
                (v["key"] == "all" and len(np.unique(A["stack"][A["rcls"] == int(k) // 5])) < 30)
                for k, v in tab.items() if v["key"] != "g")
    empty = [k for k in range(35) if not (fg == k).any()]
    t("U2", own_ok and fb_ok and len(tab) == 35 and empty and all(tab[k]["key"] != "g" for k in empty),
      f"refit on all items: own-group fits = fit_params on the group {own_ok}; fallback to the recorded class or all only under 30 "
      f"stacks {fb_ok}; {len(empty)} groups with no item take the parent or all")
    with tempfile.TemporaryDirectory() as tmp:
        kw, A2, opts = planted_world(tmp, T, g)
        o1 = Path(tmp) / "data" / "stage_h3_freeze"
        r, fm = T.run(out_dir=o1, out=quiet, **kw)
        js = json.loads((o1 / "h3_freeze.json").read_text()) if (o1 / "h3_freeze.json").is_file() else {}
        ok_j0, why = T.H1.mode_ok("test", True, str(o1 / "h3_freeze.json"), str(Path(tmp) / "data" / "bnm_test"))
        t("U3", r and all(r.values()) and set(r) == {"F0", "F2", "F3", "F1", "F5", "F4"} and ok_j0 and js.get("options") == opts
          and js["tools"].get("stage_h1_bnm_export.py") == T.X.file_md5(T.HERE / "stage_h1_bnm_export.py"),
          f"planted folders: {r}; H1 J0 accepts the freeze file for the test half: {ok_j0} ({why}); options = H2b's")
        try:
            T.run(out_dir=o1, out=quiet, **kw); again = False
        except SystemExit:
            again = True
        o2 = Path(tmp) / "data" / "stage_h3_freeze_b"
        r2, fm2 = T.run(out_dir=o2, out=quiet, **kw)
        t("U4", again and fm2 == fm, f"a run into the same folder stops {again}; a run into a new folder writes the same "
          f"h3_freeze.json {fm2 == fm}")
        bugs = []
        for bug, gname in (("marker", "F0"), ("scale", "F2"), ("det", "F1"), ("truth", "F4"), ("tool", "F5")):
            os.environ["STAGE_H3_BROKEN"] = bug
            try:
                rb, _ = T.run(out_dir=Path(tmp) / f"b_{bug}", out=quiet, **kw)
            finally:
                os.environ.pop("STAGE_H3_BROKEN", None)
            bugs.append((bug, rb.get(gname) is False))
        sp = kw["h2b"] / "h2b_scores.npz"; keep = Path(tmp) / "keep.npz"; shutil.copy(sp, keep)
        z = dict(np.load(sp)); z["hr2_mean_s1"] = z["hr2_mean_s1"] + 1e-9; np.savez(sp, **z)
        rb, _ = T.run(out_dir=Path(tmp) / "b_scores", out=quiet, **kw)
        bugs.append(("h2b_scores changed", rb.get("F0") is False)); shutil.copy(keep, sp)
        t("U5", all(b for _, b in bugs), "planted bugs fail their gate: " + ", ".join(f"{k} {v}" for k, v in bugs))
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)