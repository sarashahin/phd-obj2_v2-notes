#!/usr/bin/env python3
"""
stage_g_samples.py  (obj2_v2, stage G step G1, 10 Nov 2026; stage G plan signed off by Sara on 10 Nov)

Question: G3 (EOO and AOO of every sample, the calibrated intervals) and G4 (process tests) need each sample's own 6
yearly maps. Step 3a (stage_f_check.py) kept only the count per cell over the 50 samples and the AOO of each sample.
This draws the same 50 samples per frozen real_val stack again (same best.pt, DDIM 50, eta 0, noise seed 11, same
batches as stage_f_check.py) and keeps every sample's 6 window maps, packed. The truth is never read.

Output per run, in <run>/check_g1/ (the folder must be new)
    samples_k50.npz  maps uint8 (stacks, K, 300): each sample's 6 window maps after np.packbits, bit order year, row,
                     column (6 x 20 x 20 = 2,400 bits); sel, K, seed, ddim, batch, in_md5, epoch, best_md5, layout
    g1_report.json   gates, md5 of samples_k50.npz, agreement with step 3a
    About 245 MB per run for 16,347 stacks and K 50.

Gates (rules written before the run)
    S1  no truth file opened (the truth guard is on for the whole run)
    S2  the same inputs as step 3a: md5 of the inputs of this pass = in_md5 stored in check_3a/preds_k50.npz
    S3  inside this pass: the count per cell and the AOO of each sample, rebuilt from the packed maps, equal the
        count and AOO summed on the device (exact)
    S4  the same samples as step 3a: among (stack, year, cell) with a count above 0 in either pass, the share with
        equal counts >= 0.99; and the share of (stack, year, sample) with equal AOO >= 0.99. Exact on the CPU; on the GPU
        cuDNN is not bit-reproducible (stage F: epoch-8 loss 0.1030 against 0.1037), so a few cells near the 0
        threshold may flip; both shares are printed. Another noise seed or another checkpoint fails S4.
real_eval is refused.

Run (from PSD_Dispersal_pool/obj2_v2/work; about 60 min per run on the RTX 5090)
    python3 ../tools/stage_g_samples.py --self-test | tail -1
    nohup python3 -u ../tools/stage_g_samples.py --runs ../runs/f_seed1 ../runs/f_seed2 --data ../data \\
        > ../logs/stage_g1_samples.txt 2>&1 &
Needs stage_f_check.py, stage_f_train.py, stage_d_headroom.py, obj2_model.py, stage_e_inputs.py, stage_d_sampler.py and
export_obj2.py in the same folder; check_3a of each run from step 3a.
"""
import argparse, hashlib, importlib.util, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g_samples.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


CK = _load("stage_f_check")
F, E, M, D, X = CK.F, CK.E, CK.M, CK.D, CK.X
G, W, C, PAD = CK.G, CK.W, CK.C, CK.PAD
NB = W * G * G // 8                                    # 300 bytes per sample
SHARE = 0.99
LAYOUT = "np.packbits over (year 0-5, row 0-19, column 0-19) of the 20 x 20 window; year 5 = target year"
BROKEN = os.environ.get("STAGE_G1_BROKEN", "")         # self-test only: plants one bug so its gate must fail


def window(occ, off):
    """(n, K, W, C, C) bool -> (n, K, W, G, G): the window at each stack's offset."""
    out = np.zeros(occ.shape[:3] + (G, G), bool)
    for i in range(len(off)):
        oy, ox = int(off[i, 0]), int(off[i, 1])
        out[i] = occ[i, :, :, oy:oy + G, ox:ox + G]
    return out


def pack(maps):
    """(n, K, W, G, G) bool -> (n, K, 300) uint8."""
    if BROKEN == "transpose":
        maps = maps.swapaxes(-1, -2)
    return np.packbits(maps.reshape(maps.shape[:2] + (W * G * G,)), axis=-1)


def unpack(b):
    """(n, K, 300) uint8 -> (n, K, W, G, G) bool."""
    return np.unpackbits(b, axis=-1, count=W * G * G).astype(bool).reshape(b.shape[:2] + (W, G, G))


@torch.no_grad()
def sample_maps(model, diff, VA, frozen, sel, K, ddim, dev, seed, batch=1024):
    """Mirror of stage_f_check.sample_all (same chunks, generator seeds and noise) that also keeps each sample's maps.
    Returns maps (n, K, 300) uint8, counts (n, W, C, C) uint8, aoo (n, W, K) uint16, and the md5 of the inputs."""
    voff, vblank = E.frozen_draws(frozen)
    fi, rw = frozen["file_index"], frozen["row"]
    ch, n = max(1, batch // K), len(sel)
    maps = np.zeros((n, K, NB), np.uint8)
    counts = np.zeros((n, W, C, C), np.uint8); aoo = np.zeros((n, W, K), np.uint16)
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
        g = torch.Generator(device=dev).manual_seed((seed + (1 if BROKEN == "other_seed" else 0)) * 1000003 + j)
        noise = torch.randn((len(cond), W, C, C), generator=g, device=dev)
        x0 = diff.ddim(model, cond, nm, noise, steps=ddim)
        occ = ((x0 > 0) & (cond[:, M.MASK_CH:M.MASK_CH + 1] > 0)).view(len(k), K, W, C, C)
        counts[a:a + len(k)] = occ.sum(1).to(torch.uint8).cpu().numpy()
        aoo[a:a + len(k)] = occ.sum((3, 4)).permute(0, 2, 1).to(torch.int32).cpu().numpy().astype(np.uint16)
        maps[a:a + len(k)] = pack(window(occ.cpu().numpy(), voff[k]))
    return maps, counts, aoo, hin.hexdigest()


def run_one(run_dir, data, K=50, ddim=50, dev="cpu", seed=11, batch=1024, out=print):
    run_dir, data = Path(run_dir), Path(data)
    out_dir = run_dir / "check_g1"
    for p in (run_dir, data):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G5")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds samples; move it aside for a new draw")
    p3 = run_dir / "check_3a" / "preds_k50.npz"
    if not p3.is_file():
        sys.exit(f"STOP: {p3} not found; G1 is checked against step 3a of the same run")
    res = {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"   {name} {text}: {'PASS' if ok else 'FAIL'}")

    fp = data / "stage_d_frozen_val.npz"
    frozen = dict(np.load(fp))
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    bmd5 = X.file_md5(run_dir / "best.pt")
    out(f"== {run_dir.name}: best.pt md5 {bmd5} (epoch {ck['epoch']}, score {ck['score']:.5f}); frozen set md5 "
        f"{X.file_md5(fp)}; K {K}, DDIM {ddim}, noise seed {seed}, batch {batch}, device {dev}")
    model = M.UNet(cfg["base"], cfg["emb"]); model.load_state_dict(ck["ema"]); model = model.to(dev)
    diff = M.Diffusion(cfg["T"]).to(dev)
    sel = np.arange(len(frozen["row"]))
    t0 = time.time()
    with D.TruthGuard() as guard:
        try:
            VA = E.Inputs(data / "real_val", frozen["files"])
            maps, counts, aoo, h_in = sample_maps(model, diff, VA, frozen, sel, K, ddim, dev, seed, batch)
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    if crashed:
        gate("S1", False, f"truth file opened while sampling ({crashed[:120]})")
        return res
    out_dir.mkdir(parents=True, exist_ok=True)
    X.save_npz(out_dir / "samples_k50.npz", dict(maps=maps, sel=sel, K=np.int64(K), seed=np.int64(seed), ddim=np.int64(ddim),
                                                 batch=np.int64(batch), in_md5=np.array(h_in), epoch=np.int64(ck["epoch"]),
                                                 best_md5=np.array(bmd5), layout=np.array(LAYOUT)))
    m1 = X.file_md5(out_dir / "samples_k50.npz")
    out(f"   samples written: samples_k50.npz md5 {m1} ({(time.time() - t0) / 60:.1f} min)")
    gate("S1", not guard.opened, f"no truth file opened during the whole run (opened {len(guard.opened)})")
    z3 = np.load(p3)
    gate("S2", str(z3["in_md5"]) == h_in, f"inputs of this pass = inputs of step 3a (md5 {h_in[:8]} against {str(z3['in_md5'])[:8]})")
    voff, _ = E.frozen_draws(frozen)
    cw = CK.unplace(counts.reshape(len(sel), W, C, C), voff)   # (n, W, G, G) counts in the window
    same_c, same_a = True, True
    for a in range(0, len(sel), 1000):
        u = unpack(maps[a:a + 1000])
        same_c &= np.array_equal(u.sum(1), cw[a:a + 1000])
        same_a &= np.array_equal(u.sum((3, 4)).transpose(0, 2, 1), aoo[a:a + 1000])
    gate("S3", same_c and same_a, f"counts per cell and AOO per sample rebuilt from the packed maps equal the device sums "
                                  f"(counts {same_c}, AOO {same_a})")
    c3 = CK.unplace(z3["counts"].reshape(len(sel), W, C, C), voff)
    nz = (cw > 0) | (c3 > 0)
    share_c = float((cw[nz] == c3[nz]).mean()) if nz.any() else 1.0
    share_a = float((aoo == z3["aoo"]).mean())
    gate("S4", share_c >= SHARE and share_a >= SHARE and int(z3["K"]) == K and int(z3["seed"]) == seed,
         f"same samples as step 3a: equal counts in {share_c:.5f} of the cells occupied in either pass, equal AOO in "
         f"{share_a:.5f} of (stack, year, sample) (rule {SHARE} each)")
    rep = dict(gates=res, md5=dict(samples=m1, best_pt=bmd5, tool=X.file_md5(__file__)), share_counts=share_c,
               share_aoo=share_a, K=K, ddim=ddim, seed=seed, batch=batch, epoch=int(ck["epoch"]), n=int(len(sel)),
               minutes=round((time.time() - t0) / 60, 1))
    (out_dir / "g1_report.json").write_text(json.dumps(rep, indent=1))
    n_ok = sum(res.values())
    out(f"stage G1 {run_dir.name}: gates {n_ok} PASS, {len(res) - n_ok} FAIL")
    return res


def run(runs, data, **kw):
    allres = {}
    for r in runs:
        allres[str(r)] = run_one(r, data, **kw)
    return allres


# ---------------------------------------------------------------- self-test (CPU, tiny model, planted data)
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    rng = np.random.default_rng(3)
    mp = rng.random((3, 4, W, G, G)) < 0.3
    t("U1", np.array_equal(unpack(pack(mp)), mp), "pack and unpack are inverse (stacks, samples, years, rows, columns)")
    occ = rng.random((2, 3, W, C, C)) < 0.5
    off = np.array([[0, 4], [3, 1]])
    t("U2", np.array_equal(window(occ, off)[1, 2, 5], occ[1, 2, 5, 3:3 + G, 1:1 + G]) and
      np.array_equal(window(occ, off).sum(1), CK.unplace(occ.sum(1), off)), "window equals stage_f_check.unplace at each offset")
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
        hres, hinfo = CK.HR.run(str(base), str(base / "real_train"), str(base / "real_val"), str(base), out=quiet)
        e1, e2 = round(hinfo["bal"]["B1"], 3), round(hinfo["bal"]["B2"], 3)
        cfg = dict(F.CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=2, K=2, ddim=4,
                   val_set="full", val_chunk=64, lr=1e-3)
        F.train(cfg, str(base), str(tmp / "run"), "cpu", log=quiet)
        CK.run(tmp / "run", base, K=4, ddim=4, dev="cpu", expect_b1=e1, expect_b2=e2, out=quiet)
        r = run_one(tmp / "run", base, K=4, ddim=4, dev="cpu", out=quiet)
        rep = json.loads((tmp / "run" / "check_g1" / "g1_report.json").read_text())
        t("S1", all(r.get(g) for g in ("S1", "S2", "S3", "S4")) and rep["share_counts"] == 1.0 and rep["share_aoo"] == 1.0,
          f"planted run end to end: {r}; on the CPU the samples equal step 3a exactly "
          f"(counts {rep['share_counts']}, AOO {rep['share_aoo']})")
        z = np.load(tmp / "run" / "check_g1" / "samples_k50.npz")
        u = unpack(z["maps"])
        t("S2", z["maps"].shape == (len(z["sel"]), 4, NB) and u[:, :, :5].any() and u.dtype == bool and str(z["layout"]) == LAYOUT,
          f"samples_k50.npz holds (stacks, K, 300) packed maps of all 6 years ({z['maps'].shape})")
        stop = False
        try:
            run_one(tmp / "run", base, K=4, ddim=4, dev="cpu", out=quiet)
        except SystemExit:
            stop = True
        t("S3", stop, "a second draw into the same folder is refused")
        import shutil, subprocess
        for bug, gate in (("truth_in_sample", "S1"), ("transpose", "S3"), ("other_seed", "S4")):
            rb = tmp / f"run_{bug}"
            rb.mkdir()
            for f in ("best.pt", "run_info.json"):
                shutil.copy(tmp / "run" / f, rb / f)
            shutil.copytree(tmp / "run" / "check_3a", rb / "check_3a")
            env = dict(os.environ, STAGE_G1_BROKEN=bug)
            code = ("import importlib.util as u; s=u.spec_from_file_location('g', %r); m=u.module_from_spec(s); "
                    "s.loader.exec_module(m); r=m.run_one(%r, %r, K=4, ddim=4, dev='cpu', out=lambda *a, **k: None); "
                    "print('RES', {k: r.get(k) for k in ('S1','S2','S3','S4')})" % (str(Path(__file__).resolve()), str(rb), str(base)))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        stop = False
        try:
            run_one(tmp / "real_eval_run", base, K=4, ddim=4, out=quiet)
        except SystemExit:
            stop = True
        t("S4", stop, "a real_eval path is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--data", default="../data")
    ap.add_argument("--K", type=int, default=50); ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        torch.use_deterministic_algorithms(True)
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    allres = run(a.runs, a.data, K=a.K, ddim=a.ddim, dev=dev, out=lambda s: print(s, flush=True))
    sys.exit(0 if all(all(r.get(g) for g in ("S1", "S2", "S3", "S4")) for r in allres.values()) else 1)