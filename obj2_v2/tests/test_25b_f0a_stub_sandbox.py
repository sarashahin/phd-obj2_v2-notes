#!/usr/bin/env python3
"""
test_25b_f0a_stub_sandbox.py  (sandbox only, 13 Nov 2026): runs test_25's unit checks and T1-T4 on the stub world of
test_23b (stub stage_h2_bnm_score carrying H2b's own code for the functions used; real stage_h3_freeze.py,
stage_h3b_sparse_diagnostic.py and stage_h6_bnm_test_score.py). test_23b's planted H2b output lacks dist_h, fold and the
share keys, so they are added here the way H2b writes them (dist_h = the B4 distance at the holes, folds by window, share =
sample share) before the freeze file is built.
"""
import importlib.util, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


t23b = load("t23b", HERE / "test_23b_h6_stub_sandbox.py")
t25 = load("t25", HERE / "test_25_f0a_spread_check.py")
T = t25.load_tool(TOOLS / "stage_f0_spread_check.py")
X, S2 = T.X, T.S2


def stub_fit_world(tmp, g):
    t = Path(tmp); data = t / "data"; data.mkdir()
    runs = [t / "runs" / "run1", t / "runs" / "run2"]
    for i, r in enumerate(runs):
        r.mkdir(parents=True); (r / "best.pt").write_bytes(bytes([i + 1]) * 64)
    obj1 = t / "obj1"; obj1.mkdir(); (obj1 / "bnm_detection_check.py").write_text("# planted\n")
    g2, g3 = t / "g2", data / "stage_g3_calibration"                 # g2 outside data/, as P5.build_world puts it
    for d_, f in ((g2, "g2_fit.json"), (g3, "g3_calibrated.npz")):
        d_.mkdir(); (d_ / f).write_bytes(f.encode())
    h1m, h2am = t23b.md5(TOOLS / "stage_h1_bnm_export.py"), t23b.md5(TOOLS / "stage_h2_bnm_predict.py")
    names = [f"bnm_e{e:04d}_n0100.npz" for e in (100, 140, 180, 220, 260, 300)]
    bnm, h2a, h2b = data / "bnm_fit", data / "stage_h2_predict", data / "stage_h2b_score"
    t23b.export(bnm, names, g, "fit", h1m)
    t23b.predict(bnm, h2a, runs, g, h2am)
    opts = t23b.planted_h2b(bnm, h2a, h2b, runs, 5)
    # add the keys real H2b writes and the stub's planted H2b leaves out
    P = dict(np.load(h2b / "h2b_predicted.npz"))
    n = len(P["stack_id"]); ptr = P["hole_ptr"].astype(np.int64)
    item = np.repeat(np.arange(n), np.diff(ptr)); cell = P["hole_cell"].astype(np.int64)
    P["dist_h"] = np.load(h2a / "baselines_bnm.npz")["B4_dist"].reshape(n, -1)[item, cell].astype(np.float32)
    win = np.load(bnm / "bnm_stacks.npz")["window"].astype(str)
    wl = sorted(set(win.tolist()))
    P["fold"] = np.array([wl.index(w_) % 2 for w_ in win], np.int8)
    for m in ("s1", "s2"):
        P[f"share_{m}"] = P[f"score_{m}"]
    X.save_npz(h2b / "h2b_predicted.npz", P)
    for f in ("per_stack_h2b.csv", "per_window_h2b.csv"):
        (h2b / f).write_text("planted\n")
    paths = dict(bnm_fit=bnm, stage_h2_predict=h2a, stage_h2b_score=h2b, stage_g2_baselines_v2=g2, stage_g3_calibration=g3)
    fzp = t23b.freeze_file(t, paths, runs, obj1, opts)
    return dict(bnm=bnm, h2a=h2a, h2b=h2b, g2=g2, g3=g3, freeze=fzp, runs=runs, obj1_tools=obj1, freeze_md5=X.file_md5(fzp),
                h6_md5=X.file_md5(TOOLS / "stage_h6_bnm_test_score.py"), floor_override=5)


def main():
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    t25.unit_checks(T, t)
    with tempfile.TemporaryDirectory() as tmp:
        w = stub_fit_world(tmp, np.random.default_rng(11))
        lines = t25.checks(T, w, tmp, t)
        print("   planted run log:", *lines, sep="\n      ")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)