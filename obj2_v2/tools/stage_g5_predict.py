#!/usr/bin/env python3
"""
stage_g5_predict.py  (obj2_v2, stage G5 part G5c, 11 Nov 2026; G5 rules signed off by Sara on 11 Nov)

Question: G5-2 and G5-3 of the signed G5 rules. Build the frozen real_eval set by the stage D rule, then write every
prediction with its md5: both seeds' 50 samples per stack, B1 as stage D stored it and exact, and the B3 half-blank, B4
and B5 maps with the G2 v2 parameters. Inputs only: no truth file of any split is opened, so the truth stays closed
until G5d.

Rules
    G5-2  stage_d_sampler.frozen_set unchanged (seed 99): every real_eval stack with 1+ record over 6 years in recorded
          classes 0 and 1, 1,500 per other class, exactly half of each class blanked, one shift per stack, a fast subset
          of 300 per class; from input files only.
    G5-3  model: the G1 code (stage_g_samples.sample_maps) with best.pt (EMA weights) and the K, noise seed, DDIM steps
          and batch stored in that run's check_g1/samples_k50.npz; B1 as stage D stored it (float32) and exact (log); B3
          half-blank, B4 and B5 by the G2 code with the parameters in g2_fit.json of the G2 v2 run (no refit); torch
          threads as in G2 (--threads 4).

Parts (run in this order; each refuses an output folder that already holds its outputs)
    --part frozen                frozen_eval.npz
    --part baselines             baselines_eval.npz (B1, B1 exact, B3, B3 no habitat, B4 logit and distance, B5 score),
                                 maps_B3_halfblank.npz, maps_B4.npz, maps_B5.npz  (CPU)
    --part samples --run <run>   samples_<run name>.npz  (GPU)
    --part report                g5c_report.json: every output md5 and every part's gates (the list G5d checks)

Gates
    R0  (every part) --final-eval given; no G5d marker in data/real_eval; g5b_export_check.json all PASS; real_eval
        manifest and every real_eval input file md5 = the g5b list
    R1  (frozen) the same code on the real_val inputs gives stage_d_frozen_val.npz exactly (files, stacks, classes,
        record counts, blanks, shifts, fast subset); real_eval counts per class = g5b's frozen_take; half of each class
        blanked
    R2  (baselines) g2_fit.json was written by the stage_g_baselines.py in tools (its code md5) with the stage D sigma;
        the same code on the real_val frozen set gives stage D's stored B1, g2_pred.npz and the three G2 v2 map files;
        real_eval maps: records on in every map, packed AOO = drawn AOO, mean AOO against the chances |z| <= 4 (G2 P5)
    R3  (samples) best.pt md5 and epoch logged; the same code on the first batch of the real_val frozen set gives G1's
        samples (equal counts in >= 0.99 of the cells occupied in either, equal AOO in >= 0.99: the G1 S4 rule); on
        real_eval the counts and AOO rebuilt from the packed maps = the device sums (G1 S3)
    R4  (every part) no truth file opened: stage D's TruthGuard over the whole part
    R5  (report) every part's gates PASS; every output md5 = its part log; one frozen set md5 in every part; samples
        of every run given
    Paths: data/real_eval is read only with --final-eval; nothing is written inside it.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_g5_predict.py --self-test | tail -1
    python3 -u ../tools/stage_g5_predict.py --final-eval --part frozen
    python3 -u ../tools/stage_g5_predict.py --final-eval --part samples --run ../runs/f_seed1   (then f_seed2; GPU)
    python3 -u ../tools/stage_g5_predict.py --final-eval --part baselines                     (CPU, can run meanwhile)
    python3 -u ../tools/stage_g5_predict.py --final-eval --part report --runs ../runs/f_seed1 ../runs/f_seed2
Defaults: --data ../data, --g2 ../data/stage_g2_baselines_v2, --g5b ../data/stage_g5b_export_check,
          --out ../data/stage_g5_predict. Needs torch, numpy and stage_g_baselines.py with the stage tools it loads.
"""
import argparse, glob, importlib.util, json, os, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g5_predict.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


GB = _load("stage_g_baselines")
GS, CK = GB.GS, GB.CK
HR, E, D, X, M = CK.HR, CK.E, CK.D, CK.X, CK.M
G, W, C = X.G, X.WINDOW, CK.C
BINS, TCLS = D.BINS, D.TCLS
MARKER = "G5D_TRUTH_OPENED"
SHARE = GS.SHARE
FROZEN_KEYS = ("files", "file_md5", "file_index", "row", "rec_t", "rec6", "cls", "blank", "shift", "fast")
BROKEN = os.environ.get("STAGE_G5C_BROKEN", "")   # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Part:
    """Gates, log lines and outputs of one part, written to part_<name>.json."""
    def __init__(self, out_dir, name, out=print):
        self.dir, self.name, self.out, self.res, self.lines, self.files = Path(out_dir), name, out, {}, [], {}
        self.json = self.dir / f"part_{name}.json"
        if self.json.exists():
            sys.exit(f"STOP: {self.json} exists; this part was run already (give a new --out to run it again)")

    def say(self, s):
        self.lines.append(s); self.out(s)

    def gate(self, g, ok, text):
        self.res[g] = bool(ok); self.say(f"{g} {text}: {'PASS' if ok else 'FAIL'}")

    def save(self, fname, arrays):
        X.save_npz(self.dir / fname, arrays)
        self.files[fname] = X.file_md5(self.dir / fname)

    def close(self, extra=None):
        n_ok = sum(self.res.values())
        self.say(f"stage G5c {self.name}: gates {n_ok} PASS, {len(self.res) - n_ok} FAIL" +
                 (f"; written {', '.join(f'{k} md5 {v}' for k, v in self.files.items())}" if self.files else ""))
        self.json.write_text(json.dumps(dict(part=self.name, gates=self.res, files=self.files, tool=X.file_md5(__file__),
                                             written_utc=now(), lines=self.lines, **(extra or {})), indent=1))
        return self.res


def paths_ok(out_dir, final_eval):
    """Checked before anything is created: --final-eval given, output folder outside any real_eval folder."""
    if not final_eval:
        sys.exit("STOP: real_eval is read once, for G5; give --final-eval")
    if "real_eval" in str(out_dir):
        sys.exit("STOP: nothing is written inside a real_eval folder; give another --out")


def r0(part, data, g5b, final_eval):
    """R0: may real_eval be read here? Returns (g5b json, ok); a part stops at once if R0 fails."""
    paths_ok(part.dir, final_eval)
    ev = Path(data) / "real_eval"
    js = json.loads((Path(g5b) / "g5b_export_check.json").read_text())
    mj = X.file_md5(Path(g5b) / "g5b_export_check.json")
    marker = (ev / MARKER).exists()
    man = X.file_md5(ev / "manifest.csv")
    bad = [r["input_file"] for r in D.rows_of(ev) if X.file_md5(ev / r["input_file"]) != js["md5"]["files"].get(r["input_file"])]
    ok = not marker and all(js["gates"].values()) and man == js["md5"]["manifest"] and not bad
    part.gate("R0", ok, f"--final-eval given; G5d marker absent {not marker}; g5b_export_check.json md5 {mj}, gates all PASS "
              f"{all(js['gates'].values())}; real_eval manifest md5 {man} = g5b {man == js['md5']['manifest']}; input files "
              f"md5 = g5b list {not bad}" + (f" (differ: {', '.join(bad[:3])})" if bad else ""))
    return js, ok


# ---------------------------------------------------------------- frozen set (G5-2)
def frozen_build(d):
    """The stage D frozen set of a split from its input files (stage_d_sampler.scan_inputs + frozen_set)."""
    rows, md5s, fi, rw, rt, r6 = D.scan_inputs(d)
    if BROKEN == "truth_frozen":
        with np.load(Path(d) / rows[0]["truth_file"]) as zt:
            rt = rt.copy()
    cls = D.tidx(rt)
    sel, blank, shift, fast = D.frozen_set(cls, r6)
    if BROKEN == "frozen_blank":
        blank = blank.copy(); blank[0] ^= 1
    return dict(files=np.array([r["input_file"] for r in rows]), file_md5=np.array(md5s), file_index=fi[sel].astype(np.int16),
                row=rw[sel].astype(np.int32), rec_t=rt[sel].astype(np.int16), rec6=r6[sel].astype(np.int16),
                cls=cls[sel].astype(np.int8), blank=blank, shift=shift, fast=fast,
                code_md5=np.array(X.file_md5(HERE / "stage_d_sampler.py")), classes=np.array(TCLS))


def part_frozen(data, g5b, out_dir, final_eval, out=print):
    data, out_dir = Path(data), Path(out_dir)
    paths_ok(out_dir, final_eval)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = Part(out_dir, "frozen", out)
    p.say(f"stage_g5_predict.py md5 {X.file_md5(__file__)}; stage_d_sampler.py {X.file_md5(HERE / 'stage_d_sampler.py')}; "
          f"export_obj2.py {X.file_md5(HERE / 'export_obj2.py')}")
    js, ok0 = r0(p, data, g5b, final_eval)
    if not ok0:
        return p.close()
    with D.TruthGuard() as guard:
        try:
            fv = frozen_build(data / "real_val")
            ref = dict(np.load(data / "stage_d_frozen_val.npz"))
            same = {k: np.array_equal(fv[k], ref[k]) for k in FROZEN_KEYS}
            fe = frozen_build(data / "real_eval")
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        p.gate("R4", False, f"truth closed: {crashed[:150]}")
        return p.close()
    p.gate("R1a", all(same.values()), f"the same code on the real_val inputs gives stage_d_frozen_val.npz (md5 "
           f"{X.file_md5(data / 'stage_d_frozen_val.npz')}) exactly: " + ", ".join(f"{k} {v}" for k, v in same.items()))
    take = {lab: int((fe["cls"] == c).sum()) for c, lab in enumerate(TCLS)}
    half = all(int(fe["blank"][fe["cls"] == c].sum()) == int((fe["cls"] == c).sum()) // 2 for c in range(len(TCLS)))
    p.gate("R1b", take == js["frozen_take"] and half, f"real_eval frozen set {len(fe['row'])} stacks, per class {take} = g5b "
           f"{take == js['frozen_take']}; half of each class blanked {half}; fast subset {int(fe['fast'].sum())}")
    p.save("frozen_eval.npz", fe)
    p.gate("R4", not guard.opened, f"truth closed: truth files opened {len(guard.opened)}")
    return p.close(dict(n=int(len(fe["row"])), per_class=take))


def load_frozen(out_dir):
    pj = json.loads((Path(out_dir) / "part_frozen.json").read_text())
    fp = Path(out_dir) / "frozen_eval.npz"
    m = X.file_md5(fp)
    if not all(pj["gates"].values()) or pj["files"].get("frozen_eval.npz") != m:
        sys.exit("STOP: the frozen part did not pass or frozen_eval.npz changed since; run --part frozen first")
    return dict(np.load(fp)), m


# ---------------------------------------------------------------- baselines (G5-3)
def diff_bits(a, b, step=500):
    """Number of map bits that differ between two packed map arrays, in chunks (memory)."""
    return int(sum(np.unpackbits(a[i:i + step] ^ b[i:i + step]).sum(dtype=np.int64) for i in range(0, len(a), step)))


def baselines(d, frozen, fits, b4, sig, K):
    """The G2 v2 prediction block (stage_g_baselines.run, 'predictions and maps') with given parameters."""
    vf, vfi, vrw = frozen["files"], frozen["file_index"], frozen["row"]
    blank = frozen["blank"].astype(bool)
    V1, V2, Ef, H, comm = GB.load_obs(d, vf, vfi, vrw)
    V = V1 | V2
    n = len(V)
    b1 = HR.b1_score(V, sig).astype(np.float32)
    bn = D.bidx(V.any(1).reshape(n, -1).sum(1))
    DH = GB.hab_dist(V, H)
    Y = V1.astype(np.int8) + V2.astype(np.int8)
    P3h = np.zeros((n, W, G, G), np.float32); P3n = np.zeros_like(P3h)
    S3h = np.zeros((n, G, G)); S3n = np.zeros_like(S3h)
    for b, lab in enumerate(BINS):
        k = np.nonzero(bn == b)[0]
        if len(k):
            f = fits[lab]
            P3h[k], S3h[k] = GB.b3_predict(f["hab"]["theta"], Y[k], Ef[k], DH[k], f["p"])
            P3n[k], S3n[k] = GB.b3_predict(f["nohab"]["theta"], Y[k], Ef[k], np.zeros_like(DH[k]), f["p"])
    b1x = GB.b1_log(V, sig)
    dist = GB.near_dist(V)
    P3b = np.where(blank[:, None, None, None], P3n, P3h)
    bb = dict(b4)
    if BROKEN == "b4_other":
        bb["b"] = bb["b"] * 1.01
    P4, z4 = GB.b4_chance(bb, V, x=dist)
    m3, a3 = GB.draw_maps(P3b, K, 1)
    m4, a4 = GB.draw_maps(P4, K, 2 if BROKEN != "map_key" else 3)
    M5 = GB.b5_maps(V)
    m5 = GS.pack(M5[:, None]); a5 = M5[:, W - 1].reshape(n, -1).sum(1).astype(np.uint16)[:, None]
    pred = dict(B1=b1, B1_exact_log=b1x, B3=S3h, B3_nohab=S3n, B4_logit=z4, B4_dist=dist.astype(np.float32),
                B5=GB.b5_score(V).astype(np.float32), comm=comm, blank=blank.astype(np.uint8))
    chk = dict(B3=GB.check_maps(m3, P3b, V, a3), B4=GB.check_maps(m4, P4, V, a4), B5=GB.check_maps(m5, M5.astype(np.float32), V, a5))
    return pred, dict(B3=(m3, a3), B4=(m4, a4), B5=(m5, a5)), chk


def part_baselines(data, g2, g5b, out_dir, final_eval, threads=4, out=print):
    data, g2, out_dir = Path(data), Path(g2), Path(out_dir)
    paths_ok(out_dir, final_eval)
    p = Part(out_dir, "baselines", out)
    torch.set_num_threads(threads)
    p.say(f"stage_g5_predict.py md5 {X.file_md5(__file__)}; stage_g_baselines.py {X.file_md5(HERE / 'stage_g_baselines.py')}; "
          f"torch threads {torch.get_num_threads()}")
    if not r0(p, data, g5b, final_eval)[1]:
        return p.close()
    fe, mf = load_frozen(out_dir)
    fj = json.loads((g2 / "g2_fit.json").read_text())
    hz = dict(np.load(data / "stage_d_headroom_pred.npz"))
    sig = float(hz["sigma"])
    fits, b4 = fj["B3"], fj["B4"]
    code_ok = fj["code_md5"] == X.file_md5(HERE / "stage_g_baselines.py") and float(fj["sigma"]) == sig
    p.say(f"   frozen_eval.npz md5 {mf}; g2_fit.json md5 {X.file_md5(g2 / 'g2_fit.json')} (written by stage_g_baselines.py "
          f"{fj['code_md5']}); B1 sigma {sig:g}; B4 logit = {b4['a']:+.4f} {b4['b']:+.4f} d")
    fv = dict(np.load(data / "stage_d_frozen_val.npz"))
    gz = dict(np.load(g2 / "g2_pred.npz"))
    K = int(np.load(g2 / "maps_B4.npz")["K"])
    with D.TruthGuard() as guard:
        try:
            pv, mv, _ = baselines(data / "real_val", fv, fits, b4, sig, K)
            eq = {"B1 stage D": np.array_equal(pv["B1"], hz["B1"])}
            for k in ("B1_exact_log", "B4_dist", "B4_logit", "B5"):
                eq[k] = np.array_equal(pv[k], gz[k])
            d3 = max(float(np.abs(pv["B3"] - gz["B3"]).max()), float(np.abs(pv["B3_nohab"] - gz["B3_nohab"]).max()))
            mapeq = {}
            for k, f in (("B3", "maps_B3_halfblank.npz"), ("B4", "maps_B4.npz"), ("B5", "maps_B5.npz")):
                ref = np.load(g2 / f)["maps"]
                mapeq[k] = diff_bits(mv[k][0], ref) if ref.shape == mv[k][0].shape else -1
            del pv, mv
            pe, me, chk = baselines(data / "real_eval", fe, fits, b4, sig, K)
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        p.gate("R4", False, f"truth closed: {crashed[:150]}")
        return p.close()
    nbits = K * W * G * G * len(fv["row"])
    ok_maps = mapeq["B4"] == 0 and mapeq["B5"] == 0 and 0 <= mapeq["B3"] <= 1e-6 * nbits
    p.gate("R2a", code_ok and all(eq.values()) and d3 <= 1e-9 and ok_maps,
           f"G2 v2 parameters from this stage_g_baselines.py {code_ok}; on the real_val frozen set: " + ", ".join(f"{k} equal {v}" for k, v in eq.items())
           + f"; B3 scores largest difference {d3:.1e}; map bits that differ from the G2 v2 files: B3 {mapeq['B3']}, B4 {mapeq['B4']}, B5 {mapeq['B5']}")
    ok_chk = all(c[0] and c[1] and (abs(c[2]) <= 4 or k == "B5") for k, c in chk.items())
    p.gate("R2b", ok_chk, "real_eval maps: " + "; ".join(f"{k} records on {c[0]}, packed AOO = drawn {c[1]}, mean AOO against "
           f"chances z {c[2]:+.2f}" for k, c in chk.items()) + f" ({len(fe['row'])} stacks, K {K} for B3 and B4, 1 for B5)")
    common = dict(sel=np.arange(len(fe["row"])), layout=np.array(GS.LAYOUT), frozen_md5=np.array(mf), code_md5=np.array(X.file_md5(__file__)))
    p.save("baselines_eval.npz", dict(pe, sigma=np.float64(sig), frozen_md5=np.array(mf), g2_fit_md5=np.array(X.file_md5(g2 / "g2_fit.json"))))
    for k, f, key, kk in (("B3", "maps_B3_halfblank.npz", 1, K), ("B4", "maps_B4.npz", 2, K), ("B5", "maps_B5.npz", 0, 1)):
        p.save(f, dict(common, maps=me[k][0], aoo_t=me[k][1], K=np.int64(kk), seed_key=np.int64(key)))
    p.gate("R4", not guard.opened, f"truth closed: truth files opened {len(guard.opened)}")
    return p.close(dict(frozen_md5=mf, g2_fit_md5=X.file_md5(g2 / "g2_fit.json")))


# ---------------------------------------------------------------- model samples (G5-3)
def window_counts(maps):
    u = GS.unpack(maps)
    return u.sum(1), u.sum((3, 4)).transpose(0, 2, 1)       # (n, W, G, G) counts, (n, W, K) AOO


def part_samples(data, run_dir, g5b, out_dir, final_eval, dev, out=print):
    data, run_dir, out_dir = Path(data), Path(run_dir), Path(out_dir)
    paths_ok(out_dir, final_eval)
    p = Part(out_dir, f"samples_{run_dir.name}", out)
    if not r0(p, data, g5b, final_eval)[1]:
        return p.close()
    fe, mf = load_frozen(out_dir)
    g1p = run_dir / "check_g1" / "samples_k50.npz"
    g1 = np.load(g1p)
    K, seed, ddim, batch = int(g1["K"]), int(g1["seed"]), int(g1["ddim"]), int(g1["batch"])
    if BROKEN == "other_noise":
        seed += 1
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    bmd5 = X.file_md5(run_dir / "best.pt")
    p.say(f"stage_g5_predict.py md5 {X.file_md5(__file__)}; stage_g_samples.py {X.file_md5(HERE / 'stage_g_samples.py')}; "
          f"{run_dir.name}: best.pt md5 {bmd5} (epoch {ck['epoch']}, score {ck['score']:.5f}); G1 samples md5 {X.file_md5(g1p)} "
          f"(best.pt {str(g1['best_md5'])[:8]}); K {K}, noise seed {seed}, DDIM {ddim}, batch {batch} (from G1); device {dev}")
    model = M.UNet(cfg["base"], cfg["emb"]); model.load_state_dict(ck["ema"]); model = model.to(dev)
    diff = M.Diffusion(cfg["T"]).to(dev)
    fv = dict(np.load(data / "stage_d_frozen_val.npz"))
    t0 = time.time()
    with D.TruthGuard() as guard:
        try:
            ch = max(1, batch // K)
            selv = np.arange(min(ch, len(fv["row"])))
            VAv = E.Inputs(data / "real_val", fv["files"])
            mv, _, _, _ = GS.sample_maps(model, diff, VAv, fv, selv, K, ddim, dev, seed, batch)
            del VAv
            VA = E.Inputs(data / "real_eval", fe["files"])
            sel = np.arange(len(fe["row"]))
            if BROKEN == "truth_samples":
                E.targets(VA.d, VA.files, fe["file_index"][:2], fe["row"][:2], np.zeros((2, 2), int))
            maps, counts, aoo, h_in = GS.sample_maps(model, diff, VA, fe, sel, K, ddim, dev, seed, batch)
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    if crashed:
        p.gate("R4", False, f"truth closed: {crashed[:150]}")
        return p.close()
    cv, av = window_counts(mv)
    cg, ag = window_counts(g1["maps"][:len(selv)])
    nz = (cv > 0) | (cg > 0)
    share_c = float((cv[nz] == cg[nz]).mean()) if nz.any() else 1.0
    share_a = float((av == ag).mean())
    p.gate("R3a", str(g1["best_md5"]) == bmd5 and share_c >= SHARE and share_a >= SHARE,
           f"G1 was drawn from this best.pt {str(g1['best_md5']) == bmd5}; the first real_val batch ({len(selv)} stacks x {K}) "
           f"drawn here equals G1: counts {share_c:.5f} of occupied cells, AOO {share_a:.5f} (rule {SHARE} each)")
    voff, _ = E.frozen_draws(fe)
    cw = CK.unplace(counts.reshape(len(sel), W, C, C), voff)
    same_c, same_a = True, True
    for a in range(0, len(sel), 1000):
        cc, aa = window_counts(maps[a:a + 1000])
        same_c &= np.array_equal(cc, cw[a:a + 1000]); same_a &= np.array_equal(aa, aoo[a:a + 1000])
    p.gate("R3b", same_c and same_a, f"real_eval: counts per cell and AOO per sample rebuilt from the packed maps equal the "
           f"device sums (counts {same_c}, AOO {same_a}); {len(sel)} stacks, {(time.time() - t0) / 60:.1f} min")
    p.save(f"samples_{run_dir.name}.npz", dict(maps=maps, sel=sel, K=np.int64(K), seed=np.int64(seed), ddim=np.int64(ddim),
                                               batch=np.int64(batch), in_md5=np.array(h_in), epoch=np.int64(ck["epoch"]),
                                               best_md5=np.array(bmd5), layout=np.array(GS.LAYOUT), frozen_md5=np.array(mf)))
    p.gate("R4", not guard.opened, f"truth closed: truth files opened {len(guard.opened)}")
    return p.close(dict(frozen_md5=mf, best_md5=bmd5, epoch=int(ck["epoch"]), in_md5=h_in, run=str(run_dir)))


# ---------------------------------------------------------------- report
def part_report(data, runs, g5b, out_dir, final_eval, out=print):
    data, out_dir = Path(data), Path(out_dir)
    paths_ok(out_dir, final_eval)
    if (out_dir / "g5c_report.json").exists():
        sys.exit(f"STOP: {out_dir / 'g5c_report.json'} exists")
    res, lines = {}, []

    def say(s):
        lines.append(s); out(s)

    def gate(g, ok, text):
        res[g] = bool(ok); say(f"{g} {text}: {'PASS' if ok else 'FAIL'}")

    from types import SimpleNamespace
    r0(SimpleNamespace(dir=out_dir, gate=gate, say=say), data, g5b, final_eval)
    names = ["frozen", "baselines"] + [f"samples_{Path(r).name}" for r in runs]
    parts, bad, missing = {}, [], []
    for nm in names:
        pj = out_dir / f"part_{nm}.json"
        if not pj.is_file():
            missing.append(nm); continue
        parts[nm] = json.loads(pj.read_text())
        for f, m in parts[nm]["files"].items():
            if X.file_md5(out_dir / f) != m:
                bad.append(f)
    allpass = all(all(p["gates"].values()) for p in parts.values())
    fm = {p.get("frozen_md5", p["files"].get("frozen_eval.npz")) for p in parts.values()}
    gate("R5", not missing and not bad and allpass and len(fm) == 1,
         f"parts present {sorted(parts)} (missing {missing}); every part's gates PASS {allpass}; every output md5 = its part log "
         f"{not bad}" + (f" (changed: {', '.join(bad)})" if bad else "") + f"; one frozen set md5 in every part {sorted(fm)}")
    files = {f: m for p in parts.values() for f, m in p["files"].items()}
    rep = dict(gates=res, parts={k: dict(gates=v["gates"], files=v["files"], written_utc=v["written_utc"]) for k, v in parts.items()},
               files=files, best_md5={k: v.get("best_md5") for k, v in parts.items() if k.startswith("samples_")},
               tool=X.file_md5(__file__), written_utc=now(), truth_opened=0)
    (out_dir / "g5c_report.json").write_text(json.dumps(rep, indent=1))
    say("   predictions written and md5-logged with the truth closed: " + ", ".join(f"{f} {m[:8]}" for f, m in files.items()))
    n_ok = sum(res.values())
    say(f"stage G5c report: gates {n_ok} PASS, {len(res) - n_ok} FAIL; g5c_report.json md5 {X.file_md5(out_dir / 'g5c_report.json')}")
    return res


# ---------------------------------------------------------------- self-test
def build_world(tmp, tools):
    """The G3 self-test's planted pipeline (real_train 3, real_val 4 communities, stage D, two tiny runs with step 3a
    and G1, G2), plus a real_eval of 3 communities exported with --final-eval after a G5a fits folder, and its G5b check."""
    quiet = lambda *a, **k: None
    tmp = Path(tmp)
    py, refpat = X.plant_grids(tmp / "g")
    sims = {}
    for sp, seeds in (("real_train", (7001, 7002, 7003)), ("real_val", (7101, 7102, 7103, 7104)), ("real_eval", (7201, 7202, 7203))):
        d = tmp / "sim" / sp; d.mkdir(parents=True)
        for s_ in seeds:
            X.plant_community(d / f"c{s_}_training.npz", s_)
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
        rd = tmp / f"run{sd}"
        CK.F.train(cfg, str(base), str(rd), "cpu", log=quiet)
        CK.run(rd, base, K=50, ddim=4, dev="cpu", expect_b1=e1, expect_b2=e2, out=quiet)
        GS.run_one(rd, base, K=50, ddim=4, dev="cpu", out=quiet)
        runs.append(rd)
    GB.run(base, base / "real_train", base / "real_val", tmp / "g2", runs=runs, K=50, out=quiet)
    fits = base / "stage_g5_fits"; fits.mkdir()
    X.save_npz(fits / "g5_fits.npz", dict(a=np.arange(3)))
    (fits / "g5_fits_report.json").write_text(json.dumps(dict(md5=dict(fits=X.file_md5(fits / "g5_fits.npz")), written_utc=now())))
    time.sleep(1.2)
    (base / "habitat_sigma.json").write_text(json.dumps(dict(sigma=0.3, ok=True, val=0.80)))
    X.export(sorted(glob.glob(sims["real_eval"])), "real_eval", str(base / "real_eval"), 0.3, py, refpat, tools, final_eval=True, out=quiet)
    XC = _load("stage_g5_export_check")
    r, _ = XC.post(sims["real_eval"], base, fits, base / "habitat_sigma.json", base / "stage_g5b_export_check", out=quiet)
    if not all(r.values()):
        raise RuntimeError(f"planted G5b check failed: {r}")
    return dict(data=base, runs=runs, g2=tmp / "g2", g5b=base / "stage_g5b_export_check", sims=sims)


def sub(bug, call):
    env = dict(os.environ, STAGE_G5C_BROKEN=bug)
    code = ("import importlib.util as u; s=u.spec_from_file_location('f', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
            "r=%s; print('RES', r)" % (str(Path(__file__).resolve()), call))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")[4:] or ("ERR " + p.stderr.strip()[-200:])


def self_test(tools):
    quiet = lambda *a, **k: None
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        w = build_world(tmp, tools)
        data, g2, g5b, runs = w["data"], w["g2"], w["g5b"], w["runs"]
        out = tmp / "pred"
        lines = []
        rf = part_frozen(data, g5b, out, True, out=lines.append)
        t("S1", all(rf.values()) and set(rf) == {"R0", "R1a", "R1b", "R4"}, f"frozen part on the planted data: {rf}")
        rb = part_baselines(data, g2, g5b, out, True, out=lines.append)
        t("S2", all(rb.values()) and set(rb) == {"R0", "R2a", "R2b", "R4"}, f"baselines part: {rb}")
        rs = [part_samples(data, r, g5b, out, True, "cpu", out=lines.append) for r in runs]
        t("S3", all(all(x.values()) and set(x) == {"R0", "R3a", "R3b", "R4"} for x in rs), f"samples parts, both runs: {rs}")
        rr = part_report(data, runs, g5b, out, True, out=lines.append)
        t("S4", all(rr.values()), f"report: {rr}")
        out2 = tmp / "pred2"
        part_frozen(data, g5b, out2, True, out=quiet)
        same = X.file_md5(out / "frozen_eval.npz") == X.file_md5(out2 / "frozen_eval.npz")
        t("U1", same, "the frozen part run twice gives the same frozen_eval.npz md5")
        stop = []
        for f in (lambda: part_frozen(data, g5b, out, True, out=quiet), lambda: part_frozen(data, g5b, tmp / "pred3", False, out=quiet),
                  lambda: part_frozen(data, g5b, data / "real_eval" / "pred", True, out=quiet)):
            try:
                f(); stop.append(False)
            except SystemExit:
                stop.append(True)
        t("S5", all(stop), f"refused: a part run again into its folder {stop[0]}; no --final-eval {stop[1]}; output inside real_eval {stop[2]}")
        (data / "real_eval" / MARKER).write_text("x")
        rm = part_frozen(data, g5b, tmp / "pred_m", True, out=quiet)
        (data / "real_eval" / MARKER).unlink()
        t("bug marker", rm["R0"] is False, f"with the G5d marker present R0 fails ({rm})")
        e0 = D.rows_of(data / "real_eval")[0]["input_file"]
        keep = (data / "real_eval" / e0).read_bytes()
        GB_rewrite = _load("stage_g5_export_check").rewrite
        GB_rewrite(data / "real_eval" / e0, lambda a: a.__setitem__("habitat_noisy", (a["habitat_noisy"] + np.float16(0.5)).astype(np.float16)))
        try:
            rc = part_frozen(data, g5b, tmp / "pred_c", True, out=quiet)
        except SystemExit as e:
            rc = {"R0": False, "stop": str(e)[:60]}
        (data / "real_eval" / e0).write_bytes(keep)
        t("bug changed_input", rc.get("R0") is False, f"a real_eval input file changed after G5b fails R0 ({rc})")
        args = lambda o: f"{str(data)!r}, {str(g5b)!r}, {str(tmp / o)!r}, True, out=lambda *a, **k: None"
        for bug, gate in (("frozen_blank", "R1a"), ("truth_frozen", "R4")):
            line = sub(bug, f"m.part_frozen({args('b_' + bug)})")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line})")
        for bug, gate in (("b4_other", "R2a"), ("map_key", "R2a")):
            o = tmp / f"b_{bug}"
            subprocess.run(["cp", "-r", str(out), str(o)], check=True)
            for f in ["part_baselines.json"] + [f for f in os.listdir(o) if f.startswith(("part_samples", "maps_", "baselines_", "samples_", "g5c_"))]:
                (o / f).unlink()
            line = sub(bug, f"m.part_baselines({str(data)!r}, {str(g2)!r}, {str(g5b)!r}, {str(o)!r}, True, out=lambda *a, **k: None)")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line})")
        for bug, gate in (("other_noise", "R3a"), ("truth_samples", "R4")):
            o = tmp / f"b_{bug}"
            subprocess.run(["cp", "-r", str(out), str(o)], check=True)
            for f in [f for f in os.listdir(o) if f.startswith(("part_samples", "samples_", "g5c_"))]:
                (o / f).unlink()
            line = sub(bug, f"m.part_samples({str(data)!r}, {str(runs[0])!r}, {str(g5b)!r}, {str(o)!r}, True, 'cpu', out=lambda *a, **k: None)")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line})")
        o = tmp / "b_tamper"
        subprocess.run(["cp", "-r", str(out), str(o)], check=True)
        (o / "g5c_report.json").unlink()
        sm = o / f"samples_{runs[0].name}.npz"
        GB_rewrite(sm, lambda a: a.__setitem__("maps", a["maps"] ^ np.uint8(1)))
        rt = part_report(data, runs, g5b, o, True, out=quiet)
        t("bug tamper", rt["R5"] is False, f"a samples file changed after its part fails R5 ({rt})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--final-eval", action="store_true")
    ap.add_argument("--part", choices=("frozen", "baselines", "samples", "report"))
    ap.add_argument("--run"); ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--data", default="../data")
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--g5b", default="../data/stage_g5b_export_check")
    ap.add_argument("--out", default="../data/stage_g5_predict")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    pr = lambda s: print(s, flush=True)
    if a.part == "frozen":
        r = part_frozen(a.data, a.g5b, a.out, a.final_eval, out=pr)
    elif a.part == "baselines":
        r = part_baselines(a.data, a.g2, a.g5b, a.out, a.final_eval, a.threads, out=pr)
    elif a.part == "samples":
        if not a.run:
            sys.exit("STOP: give --run")
        r = part_samples(a.data, a.run, a.g5b, a.out, a.final_eval, a.device, out=pr)
    elif a.part == "report":
        r = part_report(a.data, a.runs, a.g5b, a.out, a.final_eval, out=pr)
    else:
        ap.print_help(); sys.exit(1)
    sys.exit(0 if r and all(r.values()) else 1)