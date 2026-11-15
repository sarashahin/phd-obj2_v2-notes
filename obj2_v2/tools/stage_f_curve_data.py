#!/usr/bin/env python3
"""
stage_f_curve_data.py  (obj2_v2, stage F, gate F6 learning curve, 9 Nov 2026; agreed by Sara on 9 Nov)

Question: would more simulated communities still help? The plan (check 2) trains on 25, 50 and 100 of the 100
real_train communities and scores each on the same frozen real_val set. The 100-community points are seeds 1 and 2.
This tool makes the data folders for the 25 and 50 runs, so stage_f_train.py v4 runs on them unchanged.

What it does (inputs only; no truth file is opened)
    1. draws the communities once from a fixed seed: one random order of the 100 training input files, the first n
       taken, so the 25 communities are inside the 50 (nested)
    2. writes ../data_curve<n>/stage_d_train_sampler.npz: the stage D sampler cut to the stacks of those communities,
       file_index renumbered, every other array copied; the class weights are rescaled per recorded class so the
       weighted class shares stay exactly those of the full sampler (= the BNM fit half, gate D3), so the runs differ
       in the number of communities only
    3. links real_train, real_val, stage_d_frozen_val.npz and stage_d_headroom_pred.npz from ../data (relative links;
       nothing copied, the validation set is the same file)

Gates
    G1  nested and repeatable: the communities of the smaller n are inside the larger, and a rerun draws the same
    G2  every kept stack points to the same input file and row as in the full sampler
    G3  weighted recorded-class shares equal the full sampler's (within 1e-6); record-count bins reported, FLAG if a
        bin moves more than 2 points (the stage D raking rule)
    G4  the linked frozen set has the md5 of ../data/stage_d_frozen_val.npz; the output folder is new; real_eval refused

Run (from PSD_Dispersal_pool/obj2_v2/work; seconds)
    python3 ../tools/stage_f_curve_data.py --self-test 2>&1 | tail -1
    python3 ../tools/stage_f_curve_data.py --data ../data --n 25 50
then train each folder with stage_f_train.py v4, seed 1, as for the full runs:
    nohup python3 -u ../tools/stage_f_train.py --train --seed 1 --data ../data_curve25 --out ../runs/f_curve25 > ../logs/stage_f_curve25.txt 2>&1 &
Needs stage_d_sampler.py, export_obj2.py (and for the self-test stage_f_train.py, obj2_model.py, stage_e_inputs.py).
"""
import argparse, importlib.util, json, os, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 2026
LINKS = ("real_train", "real_val", "stage_d_frozen_val.npz", "stage_d_headroom_pred.npz")
BROKEN = os.environ.get("STAGE_F_CURVE_BROKEN", "")     # self-test only: plants one bug so its gate must fail


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_f_curve_data.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


D = _load("stage_d_sampler")
X = D.X if hasattr(D, "X") else _load("export_obj2")


def choose(n_files, n, seed=SEED):
    """Indices of the n communities: the first n of one fixed random order (so smaller n is inside larger n)."""
    order = X.rng(seed if BROKEN != "not_nested" else seed + n, 5).permutation(n_files)
    return np.sort(order[:n])


def cut(S, idx):
    """Sampler S cut to the input files idx; per-stack arrays filtered, file_index renumbered, weights rescaled per class."""
    nf, ns = len(S["files"]), len(S["row"])
    keep = np.isin(S["file_index"], idx)
    new = np.full(nf, -1, np.int64); new[idx] = np.arange(len(idx))
    out = {}
    for k, v in S.items():
        if v.ndim >= 1 and len(v) == ns and k != "files":
            out[k] = v[keep]
        elif v.ndim >= 1 and len(v) == nf and k in ("files", "file_md5"):
            out[k] = v[idx]
        else:
            out[k] = v
    fi = out["file_index"].astype(np.int64)
    out["file_index"] = (new[fi] if BROKEN != "bad_remap" else fi).astype(S["file_index"].dtype)
    w0, wf, cls, clsf = out["weight"].astype(np.float64), S["weight"].astype(np.float64), out["cls"], S["cls"]
    w = w0.copy()
    for c in np.unique(clsf):                           # one pass on the original subset weights
        m = cls == c
        sub = w0[m].sum() / w0.sum()
        if m.any() and sub > 0:
            w[m] = w0[m] * (wf[clsf == c].sum() / wf.sum()) / sub
    out["weight"] = (w / w.mean()).astype(S["weight"].dtype)
    return out, keep


def shares(w, g, labels):
    w = w.astype(np.float64)
    return np.array([w[g == j].sum() / w.sum() for j in range(len(labels))])


def make(data, ns, out_root=None, seed=SEED, out=print):
    data = Path(data)
    if "real_eval" in str(data) or (out_root and "real_eval" in str(out_root)):
        sys.exit("STOP: real_eval is read once, in stage G")
    out_root = Path(out_root) if out_root else data.parent
    sp = data / "stage_d_train_sampler.npz"
    S = dict(np.load(sp))
    nf = len(S["files"])
    ns = sorted(set(int(n) for n in ns))
    if not all(1 <= n < nf for n in ns):
        sys.exit(f"STOP: n must be between 1 and {nf - 1}")
    targets = [out_root / f"{data.name}_curve{n}" for n in ns]
    for t in targets:
        if t.exists() and any(t.iterdir()):
            sys.exit(f"STOP: {t} is not empty; move it aside for a new subset")
    res, info = {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    out(f"stage_f_curve_data.py md5 {X.file_md5(__file__)}; sampler {sp} md5 {X.file_md5(sp)}; {nf} communities, "
        f"{len(S['row']):,} stacks; seed {seed}")
    picks = {n: choose(nf, n, seed) for n in ns}
    nested = all(set(picks[a]) <= set(picks[b]) for a, b in zip(ns, ns[1:]))
    again = all(np.array_equal(picks[n], choose(nf, n, seed)) for n in ns)
    gate("G1", nested and again, f"communities nested ({' in '.join(map(str, ns))}) {nested}; a rerun draws the same {again}")
    fr_md5 = X.file_md5(data / "stage_d_frozen_val.npz")
    for n, t in zip(ns, targets):
        Sn, keep = cut(S, picks[n])
        fin = Sn["file_index"].astype(np.int64)
        ok2 = bool(len(fin) > 0 and fin.min() >= 0 and fin.max() < len(Sn["files"])
                   and np.array_equal(Sn["files"][fin], S["files"][S["file_index"][keep].astype(np.int64)])
                   and np.array_equal(Sn["row"], S["row"][keep]))
        a, b = shares(Sn["weight"], Sn["cls"], D.TCLS), shares(S["weight"], S["cls"], D.TCLS)
        ba, bb = shares(Sn["weight"], Sn["bin"], D.BINS), shares(S["weight"], S["bin"], D.BINS)
        ok3 = bool(np.abs(a - b).max() < 1e-6)
        flag = bool(np.abs(ba - bb).max() * 100 > 2)
        t.mkdir(parents=True, exist_ok=True)
        Sn.update(curve_n=np.int64(n), curve_seed=np.int64(seed), parent_md5=np.array(X.file_md5(sp)))
        X.save_npz(t / "stage_d_train_sampler.npz", Sn)
        for name in LINKS:
            src = data / name
            if src.exists() and not (t / name).exists():
                os.symlink(os.path.relpath(src.resolve(), t.resolve()), t / name)
        ok4 = X.file_md5(t / "stage_d_frozen_val.npz") == fr_md5
        comms = [str(f) for f in Sn["files"]]
        out(f"   n {n}: {len(comms)} communities, {int(keep.sum()):,} stacks ({100 * keep.mean():.1f}% of the full sampler); "
            f"folder {t}")
        out("      recorded class " + ", ".join(f"{D.TCLS[j]} {int((Sn['cls'] == j).sum())}" for j in range(len(D.TCLS)))
            + " stacks; weighted shares equal the full sampler's" + ("" if ok3 else " NOT"))
        out("      record-count bins, weighted % (this | full): " + ", ".join(f"{D.BINS[j]} {100 * ba[j]:.1f} | {100 * bb[j]:.1f}"
                                                                         for j in range(len(D.BINS)))
            + (" FLAG: a bin moved more than 2 points" if flag else ""))
        gate(f"G2 n{n}", ok2, "every kept stack points to the same input file and row as in the full sampler")
        gate(f"G3 n{n}", ok3, f"weighted recorded-class shares equal the full sampler's (largest gap {np.abs(a - b).max():.2e})")
        gate(f"G4 n{n}", ok4, f"the linked frozen set has the md5 of the original ({fr_md5})")
        info[n] = dict(folder=str(t), communities=comms, stacks=int(keep.sum()), bin_flag=flag,
                       sampler_md5=X.file_md5(t / "stage_d_train_sampler.npz"))
        (t / "curve.json").write_text(json.dumps(dict(n=n, seed=seed, parent=str(sp), parent_md5=X.file_md5(sp),
                                                      frozen_md5=fr_md5, **info[n]), indent=1))
    n_ok = sum(res.values())
    out(f"stage F curve data: {n_ok} PASS, {len(res) - n_ok} FAIL; " + "; ".join(
        f"n {n}: {info[n]['stacks']:,} stacks, sampler md5 {info[n]['sampler_md5'][:8]}" for n in ns))
    return res, info


# ---------------------------------------------------------------- self-test (CPU, planted data)
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    F = _load("stage_f_train")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = X.plant_grids(tmp / "g")
        st = tmp / "sim" / "real_train"; st.mkdir(parents=True); sv = tmp / "sim" / "real_val"; sv.mkdir(parents=True)
        for s_ in (7001, 7002, 7003, 7004):
            X.plant_community(st / f"c{s_}_training.npz", s_)
        for s_ in (7101, 7102):
            X.plant_community(sv / f"c{s_}_training.npz", s_)
        base = tmp / "w" / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.6, py, refpat, tools, out=quiet)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.6, py, refpat, tools, out=quiet)
        D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
        res, info = make(base, [1, 2, 3], out=quiet)
        t("S1", all(res.values()) and len(res) == 10, f"subsets of 1, 2 and 3 of 4 communities: {sum(res.values())} of {len(res)} gates PASS")
        S = dict(np.load(base / "stage_d_train_sampler.npz"))
        S2 = dict(np.load(base.parent / "data_curve2" / "stage_d_train_sampler.npz"))
        t("S2", set(info[1]["communities"]) <= set(info[2]["communities"]) <= set(info[3]["communities"])
          and len(S2["files"]) == 2 and int(S2["file_index"].max()) == 1 and S2["row"].dtype == S["row"].dtype,
          "files nested 1 in 2 in 3, file_index renumbered 0-1, dtypes kept")
        lines = []
        cfg = dict(F.CFG, base=8, emb=32, batch=32, draws=128, val_every=1, patience=50, cap=1, K=2, ddim=4,
                   val_set="full", val_chunk=64, lr=1e-3)
        r = F.train(cfg, str(base.parent / "data_curve2"), str(tmp / "run2"), "cpu", log=lines.append)
        f1 = next((l for l in lines if l.startswith("F1 files opened")), "")
        t("S3", r.get("F1") and r.get("ES4") and "F1 files opened: 4 for training" in f1,
          f"stage_f_train v4 trains on the 2-community folder unchanged: {f1[:60]}")
        stop = False
        try:
            make(base, [2], out=quiet)
        except SystemExit:
            stop = True
        t("S4", stop, "a second subset into a used folder is refused")
        for bug, gate in (("not_nested", "G1"), ("bad_remap", "G2 n2")):
            env = dict(os.environ); env["STAGE_F_CURVE_BROKEN"] = bug
            code = ("import importlib.util as u; s=u.spec_from_file_location('c', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r,_=m.make(%r, [1, 2], out_root=%r, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), str(base), str(tmp / f"b_{bug}")))
            p = __import__("subprocess").run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:120] or p.stderr.strip()[-200:]})")
        stop = False
        try:
            make(tmp / "real_eval", [2], out=quiet)
        except SystemExit:
            stop = True
        t("S5", stop, "a real_eval path is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--n", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    r, _ = make(a.data, a.n, out=lambda s: print(s, flush=True))
    sys.exit(0 if all(r.values()) else 1)