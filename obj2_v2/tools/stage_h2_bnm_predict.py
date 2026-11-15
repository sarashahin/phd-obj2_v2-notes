#!/usr/bin/env python3
"""
stage_h2_bnm_predict.py  (obj2_v2, stage H step H2a, 12 Nov 2026; H rules signed off by Sara on 11 Nov, H2 plan on 12 Nov,
claude/obj2_v2_H_rules_draft.md)

Question: H2a of the signed H rules. Write every BNM prediction with its md5, with the hole labels closed: both seeds'
50 samples per scored stack-repeat (the G1 sampler), B1 as stage D stored it and exact, and the B3 (no-habitat branch,
because BNM has no habitat map), B4 and B5 maps with the G2 v2 parameters. The same code later serves H5 on the test half.

Inputs: an H1 export folder (../data/bnm_fit): manifest.csv, *_input.npz, bnm_frozen.npz (stage D's frozen-set layout,
blank 1 everywhere) and h1_report.json. No *_truth.npz file is opened.

Parts (each refuses an output folder that already holds its outputs)
    --part baselines             baselines_bnm.npz (B1, B1 exact, B3, B3 no habitat, B4 logit and distance, B5 score),
                                 maps_B3.npz (no-habitat branch), maps_B4.npz, maps_B5.npz  (CPU)
    --part samples --run <run>   samples_<run name>.npz  (GPU)
    --part report                h2a_report.json: every output md5 and every part's gates (the list H2b checks)

Gates
    K0  (every part) the H1 export passed every gate (h1_report.json); its manifest, bnm_frozen.npz and every input file
        have the md5 H1 logged; the output folder lies outside the H1 folder and outside any real_eval folder
    R2a (baselines) g2_fit.json was written by the stage_g_baselines.py in tools with the stage D sigma; the same code on
        the real_val frozen set gives stage D's stored B1, g2_pred.npz and the three G2 v2 map files (as G5c R2a)
    R2b (baselines) BNM maps: records on in every map, packed AOO = drawn AOO, mean AOO against the chances |z| <= 4
        (G2 P5); every frozen row blanked, so B3 maps come from the no-habitat branch
    R3a (samples) best.pt md5 = the one G1 used; the same code on the first real_val batch gives G1's samples (the G1 S4
        rule: equal counts in >= 0.99 of occupied cells, equal AOO in >= 0.99)
    R3b (samples) BNM counts per cell and AOO per sample rebuilt from the packed maps = the device sums (G1 S3)
    R4  (every part) no truth file opened: stage D's TruthGuard over the whole part
    R5  (report) every part's gates PASS; every output md5 = its part log; one bnm_frozen.npz md5 in every part; samples
        of every run given

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_h2_bnm_predict.py --self-test | tail -1
    python3 -u ../tools/stage_h2_bnm_predict.py --part baselines                          (CPU)
    python3 -u ../tools/stage_h2_bnm_predict.py --part samples --run ../runs/f_seed1       (then f_seed2; GPU)
    python3 -u ../tools/stage_h2_bnm_predict.py --part report --runs ../runs/f_seed1 ../runs/f_seed2
Defaults: --data ../data, --bnm ../data/bnm_fit, --g2 ../data/stage_g2_baselines_v2, --out ../data/stage_h2_predict.
Needs stage_g5_predict.py and the stage tools it loads, and stage_h1_bnm_export.py (self-test only).
"""
import argparse, glob, importlib.util, json, os, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_h2_bnm_predict.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P5 = _load("stage_g5_predict")
GB, GS, CK = P5.GB, P5.GS, P5.CK
HR, E, D, X, M = P5.HR, P5.E, P5.D, P5.X, P5.M
G, W, C = P5.G, P5.W, P5.C
SHARE = P5.SHARE


def broken():
    return os.environ.get("STAGE_H2A_BROKEN", "")    # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Part:
    """Gates, log lines and outputs of one part, written to part_<name>.json."""
    def __init__(self, out_dir, name, out=print):
        self.dir, self.name, self.out, self.res, self.lines, self.files = Path(out_dir), name, out, {}, [], {}
        self.json = self.dir / f"part_{name}.json"
        if self.json.exists():
            sys.exit(f"STOP: {self.json} exists; this part was run already (give a new --out to run it again)")
        self.dir.mkdir(parents=True, exist_ok=True)

    def say(self, s):
        self.lines.append(s); self.out(s)

    def gate(self, g, ok, text):
        self.res[g] = bool(ok); self.say(f"{g} {text}: {'PASS' if ok else 'FAIL'}")

    def save(self, fname, arrays):
        X.save_npz(self.dir / fname, arrays)
        self.files[fname] = X.file_md5(self.dir / fname)

    def close(self, extra=None):
        n_ok = sum(self.res.values())
        self.say(f"stage H2a {self.name}: gates {n_ok} PASS, {len(self.res) - n_ok} FAIL" +
                 (f"; written {', '.join(f'{k} md5 {v}' for k, v in self.files.items())}" if self.files else ""))
        self.json.write_text(json.dumps(dict(part=self.name, gates=self.res, files=self.files, tool=X.file_md5(__file__),
                                             written_utc=now(), lines=self.lines, **(extra or {})), indent=1))
        return self.res


def paths_ok(out_dir, bnm):
    o, b = Path(out_dir).resolve(), Path(bnm).resolve()
    if o == b or b in o.parents:
        sys.exit("STOP: nothing is written inside the H1 export folder; give another --out")
    if "real_eval" in str(o) or "_truth" in str(o):
        sys.exit("STOP: the output folder may not lie in a real_eval folder or carry '_truth' in its path")


def k0(part, bnm):
    """K0: the H1 export as logged. Returns (frozen dict, frozen md5, h1 json, ok)."""
    bnm = Path(bnm)
    js = json.loads((bnm / "h1_report.json").read_text())
    man = X.file_md5(bnm / "manifest.csv")
    fm = X.file_md5(bnm / "bnm_frozen.npz")
    bad = [r["input_file"] for r in D.rows_of(bnm) if X.file_md5(bnm / r["input_file"]) != r["input_md5"]]
    ok = all(js["gates"].values()) and man == js["md5"]["manifest.csv"] and fm == js["md5"]["bnm_frozen.npz"] and not bad
    part.gate("K0", ok, f"H1 export {bnm.name} ({js['half']} half, stage_h1_bnm_export.py {js['md5']['tool'][:8]}): gates all "
              f"PASS {all(js['gates'].values())}; manifest md5 {man[:8]} = H1 {man == js['md5']['manifest.csv']}; bnm_frozen.npz "
              f"{fm[:8]} = H1 {fm == js['md5']['bnm_frozen.npz']}; input files md5 = manifest {not bad}"
              + (f" (differ: {', '.join(bad[:3])})" if bad else ""))
    return dict(np.load(bnm / "bnm_frozen.npz")), fm, js, ok


# ---------------------------------------------------------------- baselines
def part_baselines(data, bnm, g2, out_dir, threads=4, out=print):
    data, bnm, g2, out_dir = Path(data), Path(bnm), Path(g2), Path(out_dir)
    paths_ok(out_dir, bnm)
    p = Part(out_dir, "baselines", out)
    torch.set_num_threads(threads)
    p.say(f"stage_h2_bnm_predict.py md5 {X.file_md5(__file__)}; stage_g5_predict.py {X.file_md5(HERE / 'stage_g5_predict.py')}; "
          f"stage_g_baselines.py {X.file_md5(HERE / 'stage_g_baselines.py')}; torch threads {torch.get_num_threads()}")
    fb, mf, js, ok = k0(p, bnm)
    if not ok:
        return p.close()
    fj = json.loads((g2 / "g2_fit.json").read_text())
    hz = dict(np.load(data / "stage_d_headroom_pred.npz"))
    sig = float(hz["sigma"])
    fits, b4 = fj["B3"], fj["B4"]
    code_ok = fj["code_md5"] == X.file_md5(HERE / "stage_g_baselines.py") and float(fj["sigma"]) == sig
    p.say(f"   bnm_frozen.npz md5 {mf}, {len(fb['row']):,} stack-repeats; g2_fit.json md5 {X.file_md5(g2 / 'g2_fit.json')} "
          f"(written by stage_g_baselines.py {fj['code_md5']}); B1 sigma {sig:g}; B4 logit = {b4['a']:+.4f} {b4['b']:+.4f} d")
    fv = dict(np.load(data / "stage_d_frozen_val.npz"))
    gz = dict(np.load(g2 / "g2_pred.npz"))
    K = int(np.load(g2 / "maps_B4.npz")["K"])
    with D.TruthGuard() as guard:
        try:
            pv, mv, _ = P5.baselines(data / "real_val", fv, fits, b4, sig, K)
            eq = {"B1 stage D": np.array_equal(pv["B1"], hz["B1"])}
            for k in ("B1_exact_log", "B4_dist", "B4_logit", "B5"):
                eq[k] = np.array_equal(pv[k], gz[k])
            d3 = max(float(np.abs(pv["B3"] - gz["B3"]).max()), float(np.abs(pv["B3_nohab"] - gz["B3_nohab"]).max()))
            mapeq = {}
            for k, f in (("B3", "maps_B3_halfblank.npz"), ("B4", "maps_B4.npz"), ("B5", "maps_B5.npz")):
                ref = np.load(g2 / f)["maps"]
                mapeq[k] = P5.diff_bits(mv[k][0], ref) if ref.shape == mv[k][0].shape else -1
            del pv, mv
            pe, me, chk = P5.baselines(bnm, fb, fits, b4, sig, K)
            if broken() == "opentruth_base":
                np.load(sorted(bnm.glob("*_truth.npz"))[0])
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
    allblank = bool((fb["blank"] == 1).all())
    ok_chk = all(c[0] and c[1] and (abs(c[2]) <= 4 or k == "B5") for k, c in chk.items())
    p.gate("R2b", ok_chk and allblank, "BNM maps: " + "; ".join(f"{k} records on {c[0]}, packed AOO = drawn {c[1]}, mean AOO "
           f"against chances z {c[2]:+.2f}" for k, c in chk.items()) + f"; every row blanked (B3 no-habitat branch) {allblank} "
           f"({len(fb['row']):,} stack-repeats, K {K} for B3 and B4, 1 for B5)")
    common = dict(sel=np.arange(len(fb["row"])), layout=np.array(GS.LAYOUT), frozen_md5=np.array(mf), code_md5=np.array(X.file_md5(__file__)))
    p.save("baselines_bnm.npz", dict(pe, sigma=np.float64(sig), frozen_md5=np.array(mf), g2_fit_md5=np.array(X.file_md5(g2 / "g2_fit.json"))))
    for k, f, key, kk in (("B3", "maps_B3.npz", 1, K), ("B4", "maps_B4.npz", 2, K), ("B5", "maps_B5.npz", 0, 1)):
        p.save(f, dict(common, maps=me[k][0], aoo_t=me[k][1], K=np.int64(kk), seed_key=np.int64(key)))
    p.gate("R4", not guard.opened, f"truth closed: truth files opened {len(guard.opened)}")
    return p.close(dict(frozen_md5=mf, g2_fit_md5=X.file_md5(g2 / "g2_fit.json"), bnm=str(bnm.resolve()), half=js["half"]))


# ---------------------------------------------------------------- model samples
def part_samples(data, bnm, run_dir, out_dir, dev, out=print):
    data, bnm, run_dir, out_dir = Path(data), Path(bnm), Path(run_dir), Path(out_dir)
    paths_ok(out_dir, bnm)
    p = Part(out_dir, f"samples_{run_dir.name}", out)
    fb, mf, js, ok = k0(p, bnm)
    if not ok:
        return p.close()
    g1p = run_dir / "check_g1" / "samples_k50.npz"
    g1 = np.load(g1p)
    K, seed, ddim, batch = int(g1["K"]), int(g1["seed"]), int(g1["ddim"]), int(g1["batch"])
    if broken() == "other_noise":
        seed += 1
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    bmd5 = X.file_md5(run_dir / "best.pt")
    p.say(f"stage_h2_bnm_predict.py md5 {X.file_md5(__file__)}; stage_g_samples.py {X.file_md5(HERE / 'stage_g_samples.py')}; "
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
            VA = E.Inputs(bnm, fb["files"])
            sel = np.arange(len(fb["row"]))
            if broken() == "opentruth_samp":
                np.load(sorted(bnm.glob("*_truth.npz"))[0])
            maps, counts, aoo, h_in = GS.sample_maps(model, diff, VA, fb, sel, K, ddim, dev, seed, batch)
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    if crashed:
        p.gate("R4", False, f"truth closed: {crashed[:150]}")
        return p.close()
    cv, av = P5.window_counts(mv)
    cg, ag = P5.window_counts(g1["maps"][:len(selv)])
    nz = (cv > 0) | (cg > 0)
    share_c = float((cv[nz] == cg[nz]).mean()) if nz.any() else 1.0
    share_a = float((av == ag).mean())
    p.gate("R3a", str(g1["best_md5"]) == bmd5 and share_c >= SHARE and share_a >= SHARE,
           f"G1 was drawn from this best.pt {str(g1['best_md5']) == bmd5}; the first real_val batch ({len(selv)} stacks x {K}) "
           f"drawn here equals G1: counts {share_c:.5f} of occupied cells, AOO {share_a:.5f} (rule {SHARE} each)")
    voff, _ = E.frozen_draws(fb)
    cw = CK.unplace(counts.reshape(len(sel), W, C, C), voff)
    same_c, same_a = True, True
    for a in range(0, len(sel), 1000):
        cc, aa = P5.window_counts(maps[a:a + 1000])
        same_c &= np.array_equal(cc, cw[a:a + 1000]); same_a &= np.array_equal(aa, aoo[a:a + 1000])
    p.gate("R3b", same_c and same_a, f"BNM: counts per cell and AOO per sample rebuilt from the packed maps equal the device "
           f"sums (counts {same_c}, AOO {same_a}); {len(sel):,} stack-repeats, {(time.time() - t0) / 60:.1f} min")
    p.save(f"samples_{run_dir.name}.npz", dict(maps=maps, sel=sel, K=np.int64(K), seed=np.int64(seed), ddim=np.int64(ddim),
                                               batch=np.int64(batch), in_md5=np.array(h_in), epoch=np.int64(ck["epoch"]),
                                               best_md5=np.array(bmd5), layout=np.array(GS.LAYOUT), frozen_md5=np.array(mf)))
    p.gate("R4", not guard.opened, f"truth closed: truth files opened {len(guard.opened)}")
    return p.close(dict(frozen_md5=mf, best_md5=bmd5, epoch=int(ck["epoch"]), in_md5=h_in, run=str(run_dir), half=js["half"]))


# ---------------------------------------------------------------- report
def part_report(bnm, runs, out_dir, out=print):
    bnm, out_dir = Path(bnm), Path(out_dir)
    paths_ok(out_dir, bnm)
    if (out_dir / "h2a_report.json").exists():
        sys.exit(f"STOP: {out_dir / 'h2a_report.json'} exists")
    res, lines = {}, []

    def say(s):
        lines.append(s); out(s)

    def gate(g, ok, text):
        res[g] = bool(ok); say(f"{g} {text}: {'PASS' if ok else 'FAIL'}")

    from types import SimpleNamespace
    _, mf, js, _ = k0(SimpleNamespace(gate=gate), bnm)
    names = ["baselines"] + [f"samples_{Path(r).name}" for r in runs]
    parts, bad, missing = {}, [], []
    for nm in names:
        pj = out_dir / f"part_{nm}.json"
        if not pj.is_file():
            missing.append(nm); continue
        parts[nm] = json.loads(pj.read_text())
        for f, m in parts[nm]["files"].items():
            if X.file_md5(out_dir / f) != m:
                bad.append(f)
    allpass = all(all(q["gates"].values()) for q in parts.values())
    fm = {q.get("frozen_md5") for q in parts.values()}
    gate("R5", not missing and not bad and allpass and fm == {mf},
         f"parts present {sorted(parts)} (missing {missing}); every part's gates PASS {allpass}; every output md5 = its part log "
         f"{not bad}" + (f" (changed: {', '.join(bad)})" if bad else "") + f"; one bnm_frozen.npz md5 in every part, = H1 {fm == {mf}}")
    files = {f: m for q in parts.values() for f, m in q["files"].items()}
    rep = dict(gates=res, parts={k: dict(gates=v["gates"], files=v["files"], written_utc=v["written_utc"]) for k, v in parts.items()},
               files=files, best_md5={k: v.get("best_md5") for k, v in parts.items() if k.startswith("samples_")},
               frozen_md5=mf, h1_tool=js["md5"]["tool"], half=js["half"], tool=X.file_md5(__file__), written_utc=now(), truth_opened=0)
    (out_dir / "h2a_report.json").write_text(json.dumps(rep, indent=1))
    say("   predictions written and md5-logged with the hole labels closed: " + ", ".join(f"{f} {m[:8]}" for f, m in files.items()))
    n_ok = sum(res.values())
    say(f"stage H2a report: gates {n_ok} PASS, {len(res) - n_ok} FAIL; h2a_report.json md5 {X.file_md5(out_dir / 'h2a_report.json')}")
    return res


# ---------------------------------------------------------------- self-test
def sub(bug, call, env_name="STAGE_H2A_BROKEN"):
    env = dict(os.environ, **{env_name: bug})
    code = ("import importlib.util as u; s=u.spec_from_file_location('f', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
            "r=%s; print('RES', r)" % (str(Path(__file__).resolve()), call))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")[4:] or ("ERR " + p.stderr.strip()[-200:])


def self_test(tools):
    quiet = lambda *a, **k: None
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    H1 = _load("stage_h1_bnm_export")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        w = P5.build_world(tmp, tools)
        data, g2, runs = w["data"], w["g2"], w["runs"]
        py, refpat = H1.world(tmp / "bw")
        bnm = data / "bnm_fit"
        r1, _ = H1.run(py, refpat, tools, bnm, expect=(6, 6), out=quiet)
        t("S0", all(r1.values()), f"planted pipeline built (G5c's: 2 tiny runs with G1 and G2) and a planted H1 export: {r1}")
        out = tmp / "pred"
        lines = []
        rb = part_baselines(data, bnm, g2, out, out=lines.append)
        t("S1", all(rb.values()) and set(rb) == {"K0", "R2a", "R2b", "R4"}, f"baselines part on the planted BNM export: {rb}")
        rs = [part_samples(data, bnm, r, out, "cpu", out=lines.append) for r in runs]
        t("S2", all(all(x.values()) and set(x) == {"K0", "R3a", "R3b", "R4"} for x in rs), f"samples parts, both runs: {rs}")
        rr = part_report(bnm, runs, out, out=lines.append)
        t("S3", all(rr.values()) and set(rr) == {"K0", "R5"}, f"report: {rr}")
        # independent checks against the input and truth files, read here outside any part
        fb = dict(np.load(bnm / "bnm_frozen.npz"))
        V1, V2, Ef, H, _ = GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])
        V = V1 | V2
        z = np.load(out / "baselines_bnm.npz")
        b1_ok = np.array_equal(z["B1"], HR.b1_score(V, float(z["sigma"])).astype(np.float32))
        hole = np.zeros_like(V[:, 0])
        for i in np.unique(fb["file_index"]):
            k = np.nonzero(fb["file_index"] == i)[0]
            tz = np.load(bnm / str(fb["files"][i]).replace("_input.npz", "_truth.npz"))
            hole[k] = X.unpack(tz["hole"])[fb["row"][k]]
        m5 = GS.unpack(np.load(out / "maps_B5.npz")["maps"])[:, 0]
        b5_hole = int((m5[:, -1] & hole).sum())
        sm = np.load(out / f"samples_{runs[0].name}.npz")
        shape_ok = sm["maps"].shape[:2] == (len(fb["row"]), 50)
        t("S4", b1_ok and b5_hole == 0 and shape_ok and not (V[:, :, None] & hole[:, None, None]).any(),
          f"B1 recomputed from the input files equals the saved B1 {b1_ok}; B5 maps occupy no hole tetrad ({b5_hole}); "
          f"no record in a hole in any year; samples hold {sm['maps'].shape[0]} stack-repeats x {sm['maps'].shape[1]}")
        stop = []
        for f in (lambda: part_baselines(data, bnm, g2, out, out=quiet), lambda: part_baselines(data, bnm, g2, bnm / "pred", out=quiet)):
            try:
                f(); stop.append(False)
            except SystemExit:
                stop.append(True)
        t("S5", all(stop), f"refused: a part run again into its folder {stop[0]}; output inside the H1 folder {stop[1]}")
        e0 = D.rows_of(bnm)[0]["input_file"]
        keep = (bnm / e0).read_bytes()
        rw = _load("stage_g5_export_check").rewrite
        rw(bnm / e0, lambda a: a.__setitem__("visit1", a["visit1"] ^ np.uint8(1)))
        rc = part_baselines(data, bnm, g2, tmp / "pred_c", out=quiet)
        (bnm / e0).write_bytes(keep)
        t("bug changed_input", rc.get("K0") is False and len(rc) == 1, f"a BNM input file changed after H1 fails K0 and stops ({rc})")
        for bug, gate, env in (("b4_other", "R2a", "STAGE_G5C_BROKEN"), ("opentruth_base", "R4", "STAGE_H2A_BROKEN")):
            line = sub(bug, f"m.part_baselines({str(data)!r}, {str(bnm)!r}, {str(g2)!r}, {str(tmp / ('b_' + bug))!r}, out=lambda *a, **k: None)", env)
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line})")
        for bug, gate in (("other_noise", "R3a"), ("opentruth_samp", "R4")):
            line = sub(bug, f"m.part_samples({str(data)!r}, {str(bnm)!r}, {str(runs[0])!r}, {str(tmp / ('b_' + bug))!r}, 'cpu', out=lambda *a, **k: None)")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line})")
        o = tmp / "b_tamper"
        shutil.copytree(out, o)
        (o / "h2a_report.json").unlink()
        rw(o / f"samples_{runs[0].name}.npz", lambda a: a.__setitem__("maps", a["maps"] ^ np.uint8(1)))
        rt = part_report(bnm, runs, o, out=quiet)
        t("bug tamper", rt["R5"] is False, f"a samples file changed after its part fails R5 ({rt})")
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--part", choices=("baselines", "samples", "report"))
    ap.add_argument("--run"); ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--data", default="../data")
    ap.add_argument("--bnm", default="../data/bnm_fit")
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--out", default="../data/stage_h2_predict")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    pr = lambda s: print(s, flush=True)
    if a.part == "baselines":
        r = part_baselines(a.data, a.bnm, a.g2, a.out, a.threads, out=pr)
    elif a.part == "samples":
        if not a.run:
            sys.exit("STOP: give --run")
        r = part_samples(a.data, a.bnm, a.run, a.out, a.device, out=pr)
    elif a.part == "report":
        r = part_report(a.bnm, a.runs, a.out, out=pr)
    else:
        ap.print_help(); sys.exit(1)
    sys.exit(0 if r and all(r.values()) else 1)