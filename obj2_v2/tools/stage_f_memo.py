#!/usr/bin/env python3
"""
stage_f_memo.py  (obj2_v2, stage F step 3b, gate F4, 9 Nov 2026; agreed by Sara on 9 Nov)

Question: does the model copy training maps? A model that memorised real_train would draw samples that sit closer to
some training map than the true map of the validation species does. Gate F4 (signed off 8 Nov): for validation
samples, the Jaccard to the nearest real_train truth stack against the same for the validation truth itself; samples
are no closer to training maps than the truth is (median gap 0.05 or less, by true class).

What it does, for each run given (best.pt), on the frozen real_val set of stage D (16,347 stacks)
    1. from inputs only, under the truth guard: S samples per stack (default 4, DDIM 50, eta 0, own noise seed 19)
       of all 6 years, kept as maps (window cells, packed); written with md5 before any truth file is opened
    2. truth read: the 6 true maps of each frozen stack; the 6 true maps of every real_train stack in the stage D
       sampler (the stacks the model could train on; duplicates merged)
    3. stacks drawn per true class (0, 1, 2-5, 6-10, 11-20, 21-50, 51+; up to 400 each, fixed seed; all when fewer)
    4. for each drawn stack: J_truth = largest Jaccard of its true 6-year stack with any training stack; J_sample =
       the same for each sample; gap = mean J_sample - J_truth. Jaccard over the 2,400 window cells of the 6 years.

Rule (8 Nov): F4 passes in a class if the median gap is 0.05 or less; F4 passes if every class passes. Reported beside
it: median J_truth and J_sample, the 90th percentile of the gap, and the share of truths and of samples equal to a
training stack.

Gates
    C1  no truth file opened while the samples are built; written with md5 before the truth is read
    N1  the search is exact: on 200 drawn queries, the pruned search (by stack size) equals a full scan of all
        training stacks
    F1  after sampling, files opened are only in real_train and real_val; real_eval is never opened
    The output folder must not hold an earlier check. real_eval is refused.

Run (from PSD_Dispersal_pool/obj2_v2/work; about 5 min of GPU per run, then the search on the CPU, up to about 1 h)
    python3 ../tools/stage_f_memo.py --self-test 2>&1 | tail -1
    nohup python3 -u ../tools/stage_f_memo.py --runs ../runs/f_seed1 ../runs/f_seed2 --data ../data > ../logs/stage_f_memo.txt 2>&1 &
Needs stage_f_check.py, stage_f_train.py, stage_d_headroom.py, obj2_model.py, stage_e_inputs.py, stage_d_sampler.py and
export_obj2.py in the same folder.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_f_memo.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


CK = _load("stage_f_check")
F, E, M, D, X = CK.F, CK.E, CK.M, CK.D, CK.X
G, W, C = CK.G, CK.W, CK.C
RULE, N_PER, N_CHECK = 0.05, 400, 200
LUT = np.array([bin(i).count("1") for i in range(256)], np.int32)
BROKEN = os.environ.get("STAGE_F_MEMO_BROKEN", "")      # self-test only: plants one bug so its gate must fail
FORCE_LUT = os.environ.get("STAGE_F_MEMO_LUT", "") == "1"


# ---------------------------------------------------------------- exact nearest-Jaccard search on packed stacks
def to64(P):
    """packed stacks (n, 6, 50) uint8 -> (n, 38) uint64 (300 bytes padded to 304)."""
    n = len(P)
    B = np.zeros((n, 304), np.uint8); B[:, :300] = P.reshape(n, 300)
    return B.view(np.uint64)


def popcount(A):
    """bits set per row of a (n, k) uint64 array."""
    if hasattr(np, "bitwise_count") and not FORCE_LUT:
        return np.bitwise_count(A).sum(1, dtype=np.int64)
    return LUT[A.view(np.uint8)].sum(1, dtype=np.int64)


class Index:
    """Training stacks, duplicates merged, sorted by size, for the largest Jaccard with a query stack."""
    def __init__(self, T64):
        U = np.unique(T64, axis=0)
        s = popcount(U)
        o = np.argsort(s, kind="stable")
        self.U, self.s = U[o], s[o]

    def best(self, q, full=False):
        qs = int(popcount(q[None])[0])
        if qs == 0:
            return 0.0
        if full:
            lo, hi = 0, len(self.s)
            return self._scan(q, qs, lo, hi)
        lo, hi = np.searchsorted(self.s, (qs + 1) // 2, "left"), np.searchsorted(self.s, 2 * qs, "right")
        b = self._scan(q, qs, lo, hi)
        if b >= 0.5:                                    # outside [qs/2, 2 qs] the Jaccard is below 0.5
            return b
        if b <= 0:
            return self._scan(q, qs, 0, len(self.s))
        lo, hi = np.searchsorted(self.s, int(np.floor(qs * b)), "left"), np.searchsorted(self.s, int(np.ceil(qs / b)), "right")
        if BROKEN == "prune_wrong":
            lo, hi = np.searchsorted(self.s, qs, "left"), np.searchsorted(self.s, qs, "right")
        return self._scan(q, qs, lo, hi)

    def _scan(self, q, qs, lo, hi):
        if hi <= lo:
            return 0.0
        inter = popcount(self.U[lo:hi] & q)
        return float((inter / (qs + self.s[lo:hi] - inter)).max())


def jaccard_sets(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


# ---------------------------------------------------------------- sampling (inputs only)
@torch.no_grad()
def sample_maps(model, diff, VA, frozen, S, ddim, dev, seed, batch=1000):
    """S samples per stack of all 6 years, window cells, packed (n, S, 6, 50) uint8; md5 of the inputs."""
    voff, vblank = E.frozen_draws(frozen)
    fi, rw = frozen["file_index"], frozen["row"]
    n = len(fi); ch = max(1, batch // S)
    out = np.zeros((n, S, W, G * G // 8), np.uint8)
    hin = hashlib.md5()
    model.eval()
    for j, a in enumerate(range(0, n, ch)):
        k = np.arange(a, min(n, a + ch))
        x, nums = VA.get(fi[k], rw[k], voff[k], vblank[k])
        if BROKEN == "truth_in_sample":
            E.targets(VA.d, VA.files, fi[k], rw[k], voff[k])
        hin.update(x.tobytes()); hin.update(nums.tobytes())
        cond = torch.from_numpy(x).to(dev).repeat_interleave(S, 0)
        nm = torch.from_numpy(nums).to(dev).repeat_interleave(S, 0)
        g = torch.Generator(device=dev).manual_seed(seed * 1000003 + j)
        noise = torch.randn((len(cond), W, C, C), generator=g, device=dev)
        x0 = diff.ddim(model, cond, nm, noise, steps=ddim)
        occ = ((x0 > 0) & (cond[:, M.MASK_CH:M.MASK_CH + 1] > 0)).cpu().numpy().reshape(len(k), S * W, C, C)
        win = CK.unplace(occ, voff[k])
        out[k] = X.pack(win.reshape(len(k), S, W, G, G))
    return out, hin.hexdigest()


class _Opened:
    def __enter__(self):
        self.orig, self.paths = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        self.paths.append(str(Path(f).resolve()))
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig


def train_truth(train_dir, sampler):
    """Packed true 6-year maps (n, 6, 50) of every stack in the stage D training sampler."""
    man = {r["input_file"]: r for r in D.rows_of(train_dir)}
    fi, rw = sampler["file_index"].astype(np.int64), sampler["row"].astype(np.int64)
    P = np.zeros((len(fi), W, G * G // 8), np.uint8)
    for i, f in enumerate(sampler["files"]):
        k = np.nonzero(fi == i)[0]
        if len(k) == 0:
            continue
        with np.load(Path(train_dir) / man[str(f)]["truth_file"], allow_pickle=False) as z:
            P[k] = z["true_maps"][rw[k]]
    return P


def draw(tc, n_per, seed):
    g = X.rng(seed, 7)
    out = []
    for c in range(len(D.TRUE)):
        idx = np.nonzero(tc == c)[0]
        out.append(np.sort(idx if len(idx) <= n_per else g.choice(idx, n_per, replace=False)))
    return np.concatenate(out)


def summary(jt, js, tc, sel):
    rows = {}
    gap = js.mean(1) - jt
    for c in range(len(D.TRUE)):
        m = tc[sel] == c
        if not m.any():
            continue
        g_ = gap[m]
        rows[D.TRUE[c]] = dict(n=int(m.sum()), j_truth=float(np.median(jt[m])), j_sample=float(np.median(js[m].mean(1))),
                               gap=float(np.median(g_)), gap_p90=float(np.percentile(g_, 90)),
                               copy_truth=float((jt[m] >= 1 - 1e-12).mean()), copy_sample=float((js[m] >= 1 - 1e-12).mean()),
                               verdict="PASS" if np.median(g_) <= RULE else "FAIL")
    return rows, gap


# ---------------------------------------------------------------- run
def run(runs, data, out_name="check_f4", S=4, ddim=50, dev="cpu", seed=19, n_per=N_PER, out=print):
    runs, data = [Path(r) for r in runs], Path(data)
    train_dir, val_dir = data / "real_train", data / "real_val"
    for p in runs + [data]:
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G")
    for r in runs:
        if (r / out_name).exists() and any((r / out_name).iterdir()):
            sys.exit(f"STOP: {r / out_name} already holds a check; move it aside for a new one")
    fp = data / "stage_d_frozen_val.npz"
    frozen = dict(np.load(fp))
    sampler = dict(np.load(data / "stage_d_train_sampler.npz"))
    voff, _ = E.frozen_draws(frozen)
    out(f"stage_f_memo.py md5 {X.file_md5(__file__)}; stage_f_check.py {X.file_md5(HERE / 'stage_f_check.py')}; "
        f"obj2_model.py {X.file_md5(HERE / 'obj2_model.py')}; stage_e_inputs.py {X.file_md5(HERE / 'stage_e_inputs.py')}; "
        f"frozen set md5 {X.file_md5(fp)}; {S} samples per stack, DDIM {ddim}, noise seed {seed}, device {dev}")
    res = {str(r): {} for r in runs}

    def gate(r, name, ok, text):
        res[str(r)][name] = bool(ok); out(f"{name} {r.name}: {text}: {'PASS' if ok else 'FAIL'}")

    samples = {}
    for r in runs:
        ck = torch.load(r / "best.pt", map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        model = M.UNet(cfg["base"], cfg["emb"]); model.load_state_dict(ck["ema"]); model = model.to(dev)
        diff = M.Diffusion(cfg["T"]).to(dev)
        t0 = time.time()
        with D.TruthGuard() as guard:
            try:
                VA = E.Inputs(val_dir, frozen["files"])
                smp, h_in = sample_maps(model, diff, VA, frozen, S, ddim, dev, seed)
                crashed = None
            except RuntimeError as e:
                if not guard.opened:
                    raise
                crashed = str(e)
        if crashed:
            gate(r, "C1", False, f"truth file opened while the samples were built ({crashed[:100]})")
            return res
        (r / out_name).mkdir(parents=True, exist_ok=True)
        X.save_npz(r / out_name / "samples.npz", dict(samples=smp, S=np.int64(S), seed=np.int64(seed), in_md5=np.array(h_in),
                                                       epoch=np.int64(ck["epoch"]), best_pt_md5=np.array(X.file_md5(r / "best.pt"))))
        out(f"   {r.name}: best.pt epoch {ck['epoch']} (md5 {X.file_md5(r / 'best.pt')}); samples written before any truth "
            f"file was opened: samples.npz md5 {X.file_md5(r / out_name / 'samples.npz')} ({(time.time() - t0) / 60:.1f} min)")
        gate(r, "C1", not guard.opened, f"no truth file opened while the samples were built (opened {len(guard.opened)})")
        samples[r] = smp

    # ---- truth, after every run's samples are written
    t0 = time.time()
    with _Opened() as op:
        Tc, tc = E.targets(val_dir, frozen["files"], frozen["file_index"], frozen["row"], voff)
        P = train_truth(train_dir, sampler)
    inside = lambda p, d: p.startswith(str(d.resolve()) + os.sep)
    f1 = all(inside(p, train_dir) or inside(p, val_dir) for p in op.paths) and not any("real_eval" in p for p in op.paths)
    Tw = X.pack(CK.unplace(Tc, voff) > 0)
    idx = Index(to64(P))
    out(f"   truth read: {len(Tw):,} frozen stacks; {len(P):,} training stacks ({len(idx.U):,} distinct) from "
        f"{len(set(op.paths))} files ({time.time() - t0:.0f} s)")
    sel = draw(tc, n_per, seed=seed)
    Q = to64(Tw[sel])
    t0 = time.time()
    jt = np.array([idx.best(q) for q in Q])
    out(f"   {len(sel):,} frozen stacks drawn (up to {n_per} per true class); nearest training stack of each truth "
        f"({time.time() - t0:.0f} s)")
    g = X.rng(seed, 11)
    chk = g.choice(len(sel), min(N_CHECK, len(sel)), replace=False)
    n1 = all(abs(idx.best(Q[i]) - idx.best(Q[i], full=True)) < 1e-12 for i in chk)

    reports = {}
    for r in runs:
        gate(r, "F1", f1, f"after sampling, {len(set(op.paths))} files opened, all in real_train or real_val, real_eval 0")
        smp = samples[r][sel]
        if BROKEN == "copy_train":                      # a model that returns the nearest training stack
            smp = np.repeat(np.stack([nearest_copy(idx, q) for q in Q])[:, None], S, 1)
        if BROKEN == "truth_as_sample":                 # control: a model that returns the true map
            smp = np.repeat(Tw[sel][:, None], S, 1)
        t0 = time.time()
        Qs = to64(smp.reshape(-1, W, G * G // 8))
        js = np.array([idx.best(q) for q in Qs]).reshape(len(sel), S)
        n1r = n1 and all(abs(idx.best(Qs[i]) - idx.best(Qs[i], full=True)) < 1e-12 for i in g.choice(len(Qs), min(N_CHECK, len(Qs)), replace=False))
        gate(r, "N1", n1r, f"pruned search equals a full scan on {2 * N_CHECK} queries (truths and samples)")
        rows, gap = summary(jt, js, tc, sel)
        f4 = all(v["verdict"] == "PASS" for v in rows.values())
        out(f"   {r.name}: nearest training stack, {S} samples per stack ({time.time() - t0:.0f} s); by true class: "
            "n | median J truth | median J sample | median gap | gap p90 | truths equal to a training stack | samples equal")
        for c, v in rows.items():
            out(f"      {c:>6} {v['n']:>4}  {v['j_truth']:.3f}  {v['j_sample']:.3f}  {v['gap']:+.3f}  {v['gap_p90']:+.3f}  "
                f"{100 * v['copy_truth']:.1f}%  {100 * v['copy_sample']:.1f}%  {v['verdict']}")
        out(f"F4 {r.name}: samples no closer to training maps than the truth (median gap <= {RULE} in every class): "
            f"{'PASS' if f4 else 'FAIL'}")
        with open(r / out_name / "f4_per_stack.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stack", "file_index", "row", "true_class", "j_truth"] + [f"j_sample{k}" for k in range(S)] + ["gap"])
            for a, i in enumerate(sel):
                w.writerow([int(i), int(frozen["file_index"][i]), int(frozen["row"][i]), D.TRUE[tc[i]], f"{jt[a]:.5f}"]
                           + [f"{v:.5f}" for v in js[a]] + [f"{gap[a]:+.5f}"])
        rep = dict(gates=res[str(r)], F4="PASS" if f4 else "FAIL", rule=RULE, by_class=rows, S=S, ddim=ddim, seed=seed,
                   n_per=n_per, n_train=int(len(P)), n_train_distinct=int(len(idx.U)),
                   md5=dict(samples=X.file_md5(r / out_name / "samples.npz"), tool=X.file_md5(__file__)))
        (r / out_name / "f4_report.json").write_text(json.dumps(rep, indent=1))
        n_ok = sum(res[str(r)].values())
        out(f"stage F F4 {r.name}: gates {n_ok} PASS, {len(res[str(r)]) - n_ok} FAIL; F4 {'PASS' if f4 else 'FAIL'}; median gap "
            + ", ".join(f"{c} {v['gap']:+.3f}" for c, v in rows.items()))
        reports[str(r)] = dict(rep, gap=gap, js=js)
    for k in res:
        res[k]["report"] = reports.get(k)
    res["jt"], res["sel"], res["tc"] = jt, sel, tc
    return res


def nearest_copy(idx, q):
    """The training stack (packed (6, 50)) with the largest Jaccard to q; self-test control only."""
    qs = int(popcount(q[None])[0])
    inter = popcount(idx.U & q)
    j = inter / np.maximum(qs + idx.s - inter, 1)
    return idx.U[int(np.argmax(j))].view(np.uint8)[:300].reshape(W, G * G // 8)


# ---------------------------------------------------------------- self-test
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    global FORCE_LUT
    rng = np.random.default_rng(5)
    Tb = rng.random((600, W, G, G)) < rng.random((600, 1, 1, 1)) * 0.3
    Tb[::7] = False; Tb[::7, 0, 3, 4] = True                          # tiny stacks: one cell in year 1
    Tp = X.pack(Tb)
    idx = Index(to64(Tp))
    sets = [set(np.flatnonzero(b.reshape(-1))) for b in Tb]
    qb = rng.random((40, W, G, G)) < rng.random((40, 1, 1, 1)) * 0.3
    qb[0] = False; qb[1] = Tb[10]; qb[2] = False; qb[2, 0, 3, 4] = True
    qp = to64(X.pack(qb))
    ref = [max(jaccard_sets(set(np.flatnonzero(b.reshape(-1))), s) for s in sets) if b.any() else 0.0 for b in qb]
    got = [idx.best(q) for q in qp]
    FORCE_LUT = True
    got_lut = [idx.best(q) for q in qp]
    FORCE_LUT = False
    t("U1", np.allclose(got, ref, atol=1e-12) and np.allclose(got_lut, ref, atol=1e-12) and got[0] == 0.0 and got[1] == 1.0
      and got[2] == 1.0, "pruned search equals the set-based Jaccard (both popcount paths); empty query 0; a copy 1")
    truthb = rng.random((60, W, G, G)) < 0.05
    tq = to64(X.pack(truthb))
    jt_ = np.array([idx.best(q) for q in tq])
    copy = np.stack([nearest_copy(idx, q) for q in tq])
    jc = np.array([idx.best(q) for q in to64(copy)])
    jself = np.array([idx.best(q) for q in tq])
    tcx = np.zeros(60, int) + 3
    rc, _ = summary(jt_, jc[:, None], tcx, np.arange(60))
    rs, _ = summary(jt_, jself[:, None], tcx, np.arange(60))
    t("U2", rc["6-10"]["verdict"] == "FAIL" and rc["6-10"]["copy_sample"] == 1.0 and rs["6-10"]["verdict"] == "PASS"
      and abs(rs["6-10"]["gap"]) < 1e-12, f"controls: samples that copy the nearest training stack fail (median gap "
      f"{rc['6-10']['gap']:+.3f}); samples equal to the truth pass (gap 0)")
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
        for nm_, sd in (("runA", 1), ("runB", 2)):
            F.train(dict(cfg, seed=sd), str(base), str(tmp / nm_), "cpu", log=quiet)
        r = run([tmp / "runA", tmp / "runB"], base, S=2, ddim=4, dev="cpu", n_per=50, out=quiet)
        ra, rb = r[str(tmp / "runA")], r[str(tmp / "runB")]
        t("S1", all(ra.get(k) for k in ("C1", "F1", "N1")) and all(rb.get(k) for k in ("C1", "F1", "N1"))
          and (tmp / "runA" / "check_f4" / "f4_report.json").is_file() and (tmp / "runB" / "check_f4" / "f4_per_stack.csv").is_file(),
          f"planted run, two seeds end to end: A {ra.get('C1')}, {ra.get('F1')}, {ra.get('N1')}; B {rb.get('C1')}, {rb.get('F1')}, "
          f"{rb.get('N1')}; F4 {ra['report']['F4']}, {rb['report']['F4']}")
        frozen = dict(np.load(base / "stage_d_frozen_val.npz"))
        voff, _ = E.frozen_draws(frozen)
        Tc, tc = E.targets(base / "real_val", frozen["files"], frozen["file_index"], frozen["row"], voff)
        Tw = X.pack(CK.unplace(Tc, voff) > 0)
        P = train_truth(base / "real_train", dict(np.load(base / "stage_d_train_sampler.npz")))
        sel = r["sel"]
        k0 = int(sel[0])
        ref0 = max(jaccard_sets(set(np.flatnonzero(X.unpack(Tw[k0]).reshape(-1))), set(np.flatnonzero(X.unpack(p).reshape(-1))))
                   for p in P)
        t("S2", abs(r["jt"][0] - ref0) < 1e-12 and np.array_equal(np.bincount(tc[sel], minlength=7) > 0, np.bincount(tc, minlength=7) > 0),
          "J truth of a drawn stack equals a set-based scan of every training stack; every true class present is drawn")
        sm = np.load(tmp / "runA" / "check_f4" / "samples.npz")["samples"]
        t("S3", sm.shape == (len(frozen["row"]), 2, W, G * G // 8), f"samples kept as packed 6-year window maps {sm.shape}")
        stop = False
        try:
            run([tmp / "runA"], base, S=2, ddim=4, dev="cpu", n_per=50, out=quiet)
        except SystemExit:
            stop = True
        t("S4", stop, "a second check into the same folder is refused")
        for bug, gate in (("truth_in_sample", "C1"), ("prune_wrong", "N1"), ("truth_as_sample", "F4 PASS"), ("copy_train", "F4")):
            env = dict(os.environ); env["STAGE_F_MEMO_BROKEN"] = bug
            code = ("import importlib.util as u; s=u.spec_from_file_location('m', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r=m.run([%r], %r, out_name=%r, S=2, ddim=4, dev='cpu', n_per=50, out=lambda *a, **k: None); "
                    "g=r[%r]; print('RES', {k: g.get(k) for k in ('C1','F1','N1')}, 'F4', (g.get('report') or {}).get('F4'))"
                    % (str(Path(__file__).resolve()), str(tmp / "runA"), str(base), f"b_{bug}", str(tmp / "runA")))
            p = __import__("subprocess").run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RES ")), "")
            if gate == "F4 PASS":
                t("control truth_as_sample", line.endswith("F4 PASS"), f"samples equal to the true maps pass F4 through the "
                  f"whole run, so the run itself adds no gap ({line[4:] or p.stderr.strip()[-200:]})")
                continue
            bad = (f"'{gate}': False" in line) if gate != "F4" else line.endswith("F4 FAIL")
            t(f"bug {bug}", bad, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        stop = False
        try:
            run([tmp / "runA"], tmp / "real_eval", S=2, ddim=4, out=quiet)
        except SystemExit:
            stop = True
        t("S5", stop, "a real_eval path is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--data", default="../data")
    ap.add_argument("--S", type=int, default=4); ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--n-per", type=int, default=N_PER)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        torch.use_deterministic_algorithms(True)
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    r = run(a.runs, a.data, S=a.S, ddim=a.ddim, dev=dev, n_per=a.n_per, out=lambda s: print(s, flush=True))
    ok = all(all(v.get(g) for g in ("C1", "F1", "N1")) for k, v in r.items() if isinstance(v, dict) and "C1" in v)
    sys.exit(0 if ok else 1)