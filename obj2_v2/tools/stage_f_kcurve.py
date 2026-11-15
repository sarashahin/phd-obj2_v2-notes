#!/usr/bin/env python3
"""
stage_f_kcurve.py  (obj2_v2, stage F step 3b, 9 Nov 2026; agreed by Sara on 9 Nov)

Question: is 50 samples per stack enough? With 8 samples a cell's frequency takes only 9 values, and seed 1 scored
0.886 balanced with 8 samples against 0.931 with 50 (step 3a). If the score still rises past 50, stage G must use more.

What it does, on the frozen real_val set of stage D (16,347 stacks), from inputs only, under the truth guard
    1. K samples per stack (default 100, DDIM 50 steps, eta 0, own noise seed 17) from best.pt; keeps, for each k in
       ks (default 8, 16, 25, 50, 100), the target-year count of the FIRST k samples, so every k is a subset of the
       same draw; per-sample window AOO at the target year; written with its md5 before any truth file is opened
    2. truth read; per stack AUC at the target year over window cells with no record at the target year (the same
       cells and code as stage D, training validation and step 3a), for every k
    3. paired difference between the largest k and the next one (100 - 50) per true class, 95% interval by
       resampling communities (2,000 draws, stage_f_check.class_boot)

Rule (agreed by Sara, 9 Nov): if the class-balanced AUC with 100 samples is more than 0.005 above the one with 50,
stage G uses 100 samples; otherwise 50 is enough. If step 3a's report is in the run folder, its 50-sample score
(noise seed 11) is printed beside this one (noise seed 17): their difference is the sampling noise of a 50-sample score.

Gates
    C1  no truth file opened while the predictions are built; written with md5 before the truth is read
    N1  nested counts are consistent: for every k, the cells counted over the first k samples equal the summed
        per-sample AOO of those k samples, and counts never fall as k grows
    C3  the largest-k score equals the training code's score (stage_f_train.score) on the same frequencies
    C4  no cell recorded at the target year is scored; a random score gives 0.50 (within 0.01 or 4 standard errors)
    The output folder must not hold an earlier check. real_eval is refused.

Run (from PSD_Dispersal_pool/obj2_v2/work; about 2 h on the RTX 5090 for 100 samples)
    python3 ../tools/stage_f_kcurve.py --self-test 2>&1 | tail -1
    nohup python3 -u ../tools/stage_f_kcurve.py --run ../runs/f_seed1 --data ../data > ../logs/stage_f_kcurve_seed1.txt 2>&1 &
Needs stage_f_check.py, stage_f_train.py, stage_d_headroom.py, obj2_model.py, stage_e_inputs.py, stage_d_sampler.py and
export_obj2.py in the same folder.
"""
import argparse, hashlib, importlib.util, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_f_kcurve.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


CK = _load("stage_f_check")
F, HR, E, M, D, X = CK.F, CK.HR, CK.E, CK.M, CK.D, CK.X
G, W, C = CK.G, CK.W, CK.C
SCORE = CK.SCORE
KS = (8, 16, 25, 50, 100)
RULE = 0.005
BROKEN = os.environ.get("STAGE_F_KCURVE_BROKEN", "")    # self-test only: plants one bug so its gate must fail


def nested_counts(occ, ks):
    """occ (n, K, C, C) bool tensor -> (n, len(ks), C, C) uint8: cells occupied counted over the first k samples."""
    cs = occ.to(torch.int16).cumsum(1)
    K = occ.shape[1]
    idx = [min(k, K - 1) if BROKEN == "nested_wrong" and k < K else k - 1 for k in ks]
    return torch.stack([cs[:, i] for i in idx], 1).to(torch.uint8)


@torch.no_grad()
def sample_nested(model, diff, VA, frozen, sel, ks, ddim, dev, seed, batch=1000):
    """max(ks) samples per stack from inputs only. Returns nested target-year counts (n, len(ks), C, C) uint8,
    per-sample window AOO at the target year (n, K) uint16, and the md5 of the inputs."""
    K = max(ks)
    voff, vblank = E.frozen_draws(frozen)
    fi, rw = frozen["file_index"], frozen["row"]
    ch, n = max(1, batch // K), len(sel)
    cnt = np.zeros((n, len(ks), C, C), np.uint8)
    aoo = np.zeros((n, K), np.uint16)
    hin = hashlib.md5()
    model.eval()
    for j, a in enumerate(range(0, n, ch)):
        k = sel[a:a + ch]
        x, nums = VA.get(fi[k], rw[k], voff[k], vblank[k])
        if BROKEN == "truth_in_sample":
            E.targets(VA.d, VA.files, fi[k], rw[k], voff[k])
        hin.update(x.tobytes()); hin.update(nums.tobytes())
        cond = torch.from_numpy(x).to(dev).repeat_interleave(K, 0)
        nm = torch.from_numpy(nums).to(dev).repeat_interleave(K, 0)
        g = torch.Generator(device=dev).manual_seed(seed * 1000003 + j)
        noise = torch.randn((len(cond), W, C, C), generator=g, device=dev)
        x0 = diff.ddim(model, cond, nm, noise, steps=ddim)
        occ = ((x0[:, W - 1] > 0) & (cond[:, M.MASK_CH] > 0)).view(len(k), K, C, C)
        cnt[a:a + len(k)] = nested_counts(occ, ks).cpu().numpy()
        aoo[a:a + len(k)] = occ.sum((2, 3)).to(torch.int32).cpu().numpy().astype(np.uint16)
    return cnt, aoo, hin.hexdigest()


def run(run_dir, data, out_dir=None, ks=KS, ddim=50, dev="cpu", seed=17, out=print):
    run_dir, data = Path(run_dir), Path(data)
    out_dir = Path(out_dir) if out_dir else run_dir / "check_kcurve"
    for p in (run_dir, data, out_dir):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G")
    if (out_dir / "kcurve_preds.npz").exists() or (out_dir / "kcurve_report.json").exists():
        sys.exit(f"STOP: {out_dir} already holds a check; move it aside for a new one")
    ks = tuple(sorted(set(int(k) for k in ks)))
    if len(ks) < 2:
        sys.exit("STOP: at least two values of k are needed")
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = data / "real_val"
    res = {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    fp = data / "stage_d_frozen_val.npz"
    frozen = dict(np.load(fp))
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    out(f"stage_f_kcurve.py md5 {X.file_md5(__file__)}; stage_f_check.py {X.file_md5(HERE / 'stage_f_check.py')}; "
        f"stage_f_train.py {X.file_md5(HERE / 'stage_f_train.py')}; obj2_model.py {X.file_md5(HERE / 'obj2_model.py')}; "
        f"stage_e_inputs.py {X.file_md5(HERE / 'stage_e_inputs.py')}")
    out(f"   best.pt md5 {X.file_md5(run_dir / 'best.pt')} (epoch {ck['epoch']}, score {ck['score']:.5f}); frozen set md5 "
        f"{X.file_md5(fp)}; k {', '.join(map(str, ks))}; DDIM {ddim}; noise seed {seed}; device {dev}")
    model = M.UNet(cfg["base"], cfg["emb"]); model.load_state_dict(ck["ema"]); model = model.to(dev)
    diff = M.Diffusion(cfg["T"]).to(dev)
    sel = np.arange(len(frozen["row"]))
    voff, _ = E.frozen_draws(frozen)
    t0 = time.time()
    with D.TruthGuard() as guard:
        try:
            VA = E.Inputs(val_dir, frozen["files"])
            comm = np.array(VA.seed)[frozen["file_index"]]
            cnt, aoo, h_in = sample_nested(model, diff, VA, frozen, sel, ks, ddim, dev, seed)
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    if crashed:
        gate("C1", False, f"truth file opened while the predictions were built ({crashed[:120]})")
        return res
    X.save_npz(out_dir / "kcurve_preds.npz", dict(counts_t=cnt, aoo_t=aoo, ks=np.array(ks), sel=sel, seed=np.int64(seed),
                                                   in_md5=np.array(h_in), epoch=np.int64(ck["epoch"])))
    m1 = X.file_md5(out_dir / "kcurve_preds.npz")
    out(f"   predictions written before any truth file was opened: kcurve_preds.npz md5 {m1} "
        f"({(time.time() - t0) / 60:.1f} min of sampling)")
    gate("C1", not guard.opened, f"no truth file opened while the predictions were built (opened {len(guard.opened)})")
    cells = cnt.reshape(len(cnt), len(ks), -1).astype(np.int64).sum(2)
    n1 = all(np.array_equal(cells[:, q], aoo[:, :k].astype(np.int64).sum(1)) for q, k in enumerate(ks)) \
        and bool((np.diff(cnt.astype(np.int16), axis=1) >= 0).all())
    gate("N1", n1, "nested counts equal the summed per-sample AOO of the first k samples and never fall as k grows")

    # ---- truth, after the predictions are written
    Tc, tc = E.targets(val_dir, frozen["files"], frozen["file_index"], frozen["row"], voff)
    Tt = CK.unplace(Tc, voff)[:, W - 1] > 0
    V, _, _ = VA.arrays(frozen["file_index"], frozen["row"])
    Mk = ~V[:, W - 1] if BROKEN != "score_recorded" else np.ones_like(V[:, W - 1])
    win = lambda A: CK.unplace(A[:, None].astype(np.float32), voff)[:, 0]
    A = {k: HR.aucs(win(cnt[:, q]) / k, Tt, Mk) for q, k in enumerate(ks)}
    A["random"] = HR.aucs(X.rng(99, 41).random(Tt.shape).astype(np.float32), Tt, Mk)
    bal = {k: HR.balanced(v, tc) for k, v in A.items()}
    top, base = ks[-1], ks[-2]
    s_train = F.score(cnt[:, -1].astype(np.float32) / top, VA, frozen, sel, val_dir)["score"]
    gate("C3", abs(s_train - bal[top][0]) < 1e-9,
         f"the {top}-sample score by the training code {s_train:.6f} equals this tool's {bal[top][0]:.6f}")
    rec_scored = int((Mk & V[:, W - 1]).sum())
    tol = max(0.01, 4 * bal["random"][1])
    gate("C4", rec_scored == 0 and abs(bal["random"][0] - 0.5) <= tol,
         f"recorded cells among the scored cells {rec_scored}; random score {bal['random'][0]:.3f} (0.50 within {tol:.3f})")

    out("   AUC by true class at the target year, by number of samples (first k of the same draw):")
    out("      class  stacks  " + "  ".join(f"k {k:>3}" for k in ks))
    for c in SCORE:
        m = (tc == c) & ~np.isnan(A[top])
        out(f"      {D.TRUE[c]:>6} {int(m.sum()):>6}  " + "  ".join(f"{np.nanmean(A[k][m]):.4f}" for k in ks))
    out("      balanced       " + "  ".join(f"{bal[k][0]:.4f}" for k in ks))
    dd = CK.class_boot(A[top] - A[base], tc, comm)
    out(f"   {top} - {base} samples, paired per stack, 95% interval by resampling communities:")
    for c in SCORE:
        r = dd[c]
        out(f"      {D.TRUE[c]:>6}  {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    r = dd["balanced"]
    out(f"      balanced {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    noise = None
    rp = run_dir / "check_3a" / "report.json"
    if 50 in ks and rp.is_file():
        r3 = json.loads(rp.read_text().replace("NaN", "null"))
        if int(r3.get("K", 0)) == 50:
            noise = bal[50][0] - r3["auc_balanced"]["model"]
            out(f"   50 samples: balanced {bal[50][0]:.4f} here (noise seed {seed}) and {r3['auc_balanced']['model']:.4f} in step 3a "
                f"(noise seed 11): difference {noise:+.4f}, the sampling noise of a 50-sample score")
    gain = bal[top][0] - bal[base][0]
    use = top if gain > RULE else base
    out(f"   rule: {top} samples are used in stage G if they add more than {RULE} balanced over {base}; they add {gain:+.4f}")

    rep = dict(gates=res, ks=list(ks), auc_balanced={str(k): bal[k][0] for k in ks},
               auc_by_class={str(k): {D.TRUE[c]: float(np.nanmean(A[k][(tc == c)])) for c in SCORE} for k in ks},
               diff_top_base=dict({D.TRUE[c]: dd[c] for c in SCORE}, balanced=dd["balanced"]),
               gain=gain, rule=RULE, stage_g_samples=use, noise_vs_3a=noise,
               md5=dict(preds=m1, best_pt=X.file_md5(run_dir / "best.pt"), tool=X.file_md5(__file__)),
               ddim=ddim, seed=seed, best_epoch=int(ck["epoch"]))
    (out_dir / "kcurve_report.json").write_text(json.dumps(rep, indent=1))
    n_ok = sum(res.values())
    out(f"   written {out_dir}/kcurve_report.json")
    out(f"stage F K check: gates {n_ok} PASS, {len(res) - n_ok} FAIL; balanced AUC "
        + ", ".join(f"k {k} {bal[k][0]:.4f}" for k in ks) + f"; {top} - {base} {gain:+.4f}; stage G uses {use} samples")
    res.update(bal=bal, gain=gain, use=use, A=A)
    return res


# ---------------------------------------------------------------- self-test (CPU, tiny model, planted data)
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    occ = torch.from_numpy(np.random.default_rng(3).random((5, 8, C, C)) < 0.3)
    got = nested_counts(occ, (2, 5, 8)).numpy()
    ref = np.stack([occ.numpy()[:, :k].sum(1) for k in (2, 5, 8)], 1)
    t("U1", np.array_equal(got, ref), "nested counts equal the first-k sums computed apart")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = X.plant_grids(tmp / "g")
        st = tmp / "sim" / "real_train"; st.mkdir(parents=True); sv = tmp / "sim" / "real_val"; sv.mkdir(parents=True)
        for s_ in (7001, 7002):
            X.plant_community(st / f"c{s_}_training.npz", s_)
        for s_ in (7101, 7102):
            X.plant_community(sv / f"c{s_}_training.npz", s_)
        base = tmp / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.6, py, refpat, tools, out=quiet)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.6, py, refpat, tools, out=quiet)
        D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
        cfg = dict(F.CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=2, K=2, ddim=4,
                   val_set="full", val_chunk=64, lr=1e-3)
        F.train(cfg, str(base), str(tmp / "run"), "cpu", log=quiet)
        ks = (2, 4, 8)
        r = run(tmp / "run", base, ks=ks, ddim=4, dev="cpu", out=quiet)
        t("S1", all(r.get(g) for g in ("C1", "N1", "C3", "C4")) and (tmp / "run" / "check_kcurve" / "kcurve_report.json").is_file(),
          f"planted run end to end: C1 {r.get('C1')}, N1 {r.get('N1')}, C3 {r.get('C3')}, C4 {r.get('C4')}; "
          f"stage G uses {r.get('use')}")
        pz = np.load(tmp / "run" / "check_kcurve" / "kcurve_preds.npz")
        t("S2", int(pz["counts_t"][:, 0].max()) <= 2 and int(pz["counts_t"][:, -1].max()) <= 8
          and r["use"] == (8 if r["gain"] > RULE else 4), "counts stay within k, and the rule picks the right k")
        stop = False
        try:
            run(tmp / "run", base, ks=ks, ddim=4, dev="cpu", out=quiet)
        except SystemExit:
            stop = True
        t("S3", stop, "a second check into the same folder is refused")
        for bug, gate in (("truth_in_sample", "C1"), ("nested_wrong", "N1"), ("score_recorded", "C4")):
            env = dict(os.environ); env["STAGE_F_KCURVE_BROKEN"] = bug
            code = ("import importlib.util as u; s=u.spec_from_file_location('k', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r=m.run(%r, %r, out_dir=%r, ks=(2, 4, 8), ddim=4, dev='cpu', out=lambda *a, **k: None); "
                    "print('RES', {k: r.get(k) for k in ('C1','N1','C3','C4')})"
                    % (str(Path(__file__).resolve()), str(tmp / "run"), str(base), str(tmp / f"b_{bug}")))
            p = __import__("subprocess").run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        stop = False
        try:
            run(tmp / "run", tmp / "real_eval", ks=ks, ddim=4, out=quiet)
        except SystemExit:
            stop = True
        t("S4", stop, "a real_eval path is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", default="../runs/f_seed1"); ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ks", default=",".join(map(str, KS)), help="numbers of samples, comma-separated; the largest is drawn")
    ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        torch.use_deterministic_algorithms(True)
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    r = run(a.run, a.data, a.out, ks=[int(v) for v in a.ks.split(",")], ddim=a.ddim, dev=dev,
            out=lambda s: print(s, flush=True))
    sys.exit(0 if all(r.get(g) for g in ("C1", "N1", "C3", "C4")) else 1)