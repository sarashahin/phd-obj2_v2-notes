#!/usr/bin/env python3
"""
stage_f_check.py  (obj2_v2, stage F step 3a, 9 Nov 2026; plan and rules signed off by Sara on 9 Nov)

Question: does the trained model (best.pt of one seed) keep at least the ranking of the records-pinned smoother B1 in
every true range class when it is sampled 50 times per stack (R13, gate F5)? Is its AOO level right for what a recorder
can see (R18, read by recorded class)? Does anything other than the records carry the answer (F3)? This decides
whether seed 2 is trained with the same design.

What it does, on the frozen real_val set of stage D (16,347 stacks), from inputs only, under the truth guard
    1. K samples per stack (default 50, DDIM 50 steps, eta 0, own noise seed 11) of all 6 years; stores per cell the
       number of samples occupied, per sample and year the occupied window cells (AOO), and the target-year counts of
       the first 8 samples; written with its md5 before any truth file is opened
    2. null-input control (F3): each stack gets the records of a stack of another community (stage D partner rule),
       its own effort, habitat, blank and offset, built by the stage E builder; 8 samples; written with its md5
    3. truth read; per stack AUC at the target year over window cells with no record at the target year
       (as stage D and the training validation) for: model (K samples), model (first 8), B1, B2 with the frozen
       half-blank habitat (B2e where the habitat is blanked, which is what the model saw), B2 with full habitat,
       null control; B1 and B2 come from stage_d_headroom_pred.npz (the same stacks, same cells)

Rules (written before the run; Sara, 9 Nov)
    F5  per true class, paired per stack, d = AUC(model, K samples) - AUC(B1); 95% interval by resampling communities
        (2,000 draws): PASS if the mean of d >= 0; FAIL if the upper end of the interval < 0; else NOT RESOLVED.
        F5 passes if every class passes, fails if any class fails. B2 (half-blank) is reported as the target, no gate.
    AOO expected AOO = mean over samples of occupied window cells at the target year; ratio = mean expected / mean true
        per group; band 0.8-1.25 by recorded class at the target year and by records over 6 years (observable groups);
        by true class reported only (it carries regression toward the prior: from the same few records a truly 1-cell
        species and a sparsely recorded widespread one look alike). A group outside the band is corrected in stage G.
        Raw 90% interval coverage of true AOO (5th-95th percentile of sample AOO) is reported per group as a preview.
    F3  null control: class-balanced AUC <= 0.55 and every class <= 0.60 -> PASS (the records carry the answer)

Gates
    C1  no truth file opened while either set of predictions is built; both written with md5 before the truth is read
    C2  alignment with stage D: stage_d_headroom_pred.npz was made from this frozen set (its stored frozen md5), and
        B1 and B2 class-balanced AUC recomputed here round to the stage D values (Sara's run: 0.848, 0.900)
    C3  scoring identical to training: best.pt epoch = best epoch in run_info.json, and the K-8 counts saved at that
        epoch (best_val_pred.npz) scored by the training code give exactly the best score in run_info.json, and the
        same value by this tool's window scoring
    C4  no cell recorded at the target year is scored; a random score gives 0.50 (within 0.01 or 4 standard errors)
    The output folder must not hold an earlier check (no mixing of runs). real_eval is refused.

Run (from PSD_Dispersal_pool/obj2_v2/work; about 70 min on the RTX 5090: 50 samples is 6.25 validations, plus 8 for F3)
    python3 ../tools/stage_f_check.py --self-test | tail -1
    nohup python3 -u ../tools/stage_f_check.py --run ../runs/f_seed1 --data ../data > ../logs/stage_f_check_seed1.txt 2>&1 &
Needs torch, numpy, and stage_f_train.py, stage_d_headroom.py, obj2_model.py, stage_e_inputs.py, stage_d_sampler.py,
export_obj2.py in the same folder; ../data/stage_d_headroom_pred.npz from stage D step 2.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_f_check.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


F = _load("stage_f_train")
HR = _load("stage_d_headroom")
E, M, D, X = F.E, F.M, F.D, F.X
G, W, C, PAD = X.G, X.WINDOW, E.CANVAS, E.PAD
SCORE = [1, 2, 3, 4, 5, 6]
TCLS, BINS = D.TCLS, D.BINS
EXPECT_B1, EXPECT_B2 = 0.848, 0.900                     # Sara's stage D step 2 run, printed to 3 decimals
N_BOOT, BAND = 2000, (0.8, 1.25)
NULL_BAL, NULL_CLASS = 0.55, 0.60
BROKEN = os.environ.get("STAGE_F_CHECK_BROKEN", "")     # self-test only: plants one bug so its gate must fail


# ---------------------------------------------------------------- helpers
def unplace(A, offset):
    """(n, k, CANVAS, CANVAS) -> (n, k, G, G): the window at each stack's offset (inverse of stage E place)."""
    n, k = A.shape[:2]
    out = np.zeros((n, k, G, G), A.dtype)
    for oy in range(PAD + 1):
        for ox in range(PAD + 1):
            m = (offset[:, 0] == oy) & (offset[:, 1] == ox)
            if m.any():
                out[m] = A[m][:, :, oy:oy + G, ox:ox + G]
    return out


def null_inputs(VA, frozen, comm):
    """F3 inputs: records of a stack of another community, own effort, habitat, offset and blank (stage E builder)."""
    voff, vblank = E.frozen_draws(frozen)
    fi, rw = frozen["file_index"], frozen["row"]
    part = HR.partner(comm)

    def get(k):
        V, Ef, H = VA.arrays(fi[k], rw[k])
        Vp, _, _ = VA.arrays(fi[part[k]], rw[part[k]])
        if BROKEN == "null_keeps_records":
            Vp = V
        return E.build(Vp, Ef, H, voff[k], vblank[k])
    return get, part


@torch.no_grad()
def sample_all(model, diff, VA, frozen, sel, K, ddim, dev, seed, get=None, batch=1024):
    """K samples per stack of all 6 years from inputs only. Returns counts (n, 6, C, C) uint8 (samples occupied),
    counts8 (n, C, C) uint8 (first 8 samples, target year), aoo (n, 6, K) uint16 (occupied window cells per sample),
    and the md5 of the inputs."""
    voff, vblank = E.frozen_draws(frozen)
    fi, rw = frozen["file_index"], frozen["row"]
    ch, n = max(1, batch // K), len(sel)
    counts = np.zeros((n, W, C, C), np.uint8); counts8 = np.zeros((n, C, C), np.uint8)
    aoo = np.zeros((n, W, K), np.uint16)
    hin = hashlib.md5()
    model.eval()
    for j, a in enumerate(range(0, n, ch)):
        k = sel[a:a + ch]
        x, nums = VA.get(fi[k], rw[k], voff[k], vblank[k]) if get is None else get(k)
        if BROKEN == "truth_in_sample":
            E.targets(VA.d, VA.files, fi[k], rw[k], voff[k])
        hin.update(x.tobytes()); hin.update(nums.tobytes())
        cond = torch.from_numpy(x).to(dev).repeat_interleave(K, 0)
        nm = torch.from_numpy(nums).to(dev).repeat_interleave(K, 0)
        g = torch.Generator(device=dev).manual_seed(seed * 1000003 + j)
        noise = torch.randn((len(cond), W, C, C), generator=g, device=dev)
        x0 = diff.ddim(model, cond, nm, noise, steps=ddim)
        occ = ((x0 > 0) & (cond[:, M.MASK_CH:M.MASK_CH + 1] > 0)).view(len(k), K, W, C, C)
        counts[a:a + len(k)] = occ.sum(1).to(torch.uint8).cpu().numpy()
        counts8[a:a + len(k)] = occ[:, :min(8, K), W - 1].sum(1).to(torch.uint8).cpu().numpy()
        aoo[a:a + len(k)] = occ.sum((3, 4)).permute(0, 2, 1).to(torch.int32).cpu().numpy().astype(np.uint16)
    return counts, counts8, aoo, hin.hexdigest()


def class_boot(d, tc, comm, classes=SCORE, n_boot=N_BOOT, seed=31):
    """Mean of d per class and its 95% interval by resampling communities; also the class-balanced mean."""
    ok = ~np.isnan(d)
    cu = np.unique(comm)
    ci = np.searchsorted(cu, comm)
    S = np.zeros((len(cu), len(classes))); N = np.zeros_like(S)
    for j, c in enumerate(classes):
        m = ok & (tc == c)
        np.add.at(S[:, j], ci[m], d[m]); np.add.at(N[:, j], ci[m], 1)
    g = X.rng(seed, 3)
    w = np.stack([np.bincount(g.integers(0, len(cu), len(cu)), minlength=len(cu)) for _ in range(n_boot)])
    with np.errstate(invalid="ignore", divide="ignore"):
        bm = (w @ S) / (w @ N)
        mean = S.sum(0) / N.sum(0)
    out = {}
    for j, c in enumerate(classes):
        lo, hi = np.nanpercentile(bm[:, j], [2.5, 97.5]) if N[:, j].sum() else (np.nan, np.nan)
        out[c] = dict(n=int(N[:, j].sum()), mean=float(mean[j]), lo=float(lo), hi=float(hi))
    bal = np.nanmean(bm, 1)
    out["balanced"] = dict(n=int(ok.sum()), mean=float(np.nanmean(mean)), lo=float(np.nanpercentile(bal, 2.5)),
                           hi=float(np.nanpercentile(bal, 97.5)))
    return out


def verdict(m, hi):
    return "PASS" if m >= 0 else ("FAIL" if hi < 0 else "NOT RESOLVED")


def aoo_table(exp, true, lo, hi, groups, labels):
    rows = []
    for j, lab in enumerate(labels):
        m = groups == j
        if not m.any():
            continue
        r = float(exp[m].mean() / true[m].mean()) if true[m].mean() > 0 else float("nan")
        cov = float(((true[m] >= lo[m]) & (true[m] <= hi[m])).mean())
        rows.append(dict(group=lab, n=int(m.sum()), exp=float(exp[m].mean()), true=float(true[m].mean()), ratio=r,
                         cov90=cov, band=bool(BAND[0] <= r <= BAND[1])))
    return rows


# ---------------------------------------------------------------- run
def run(run_dir, data, out_dir=None, K=50, ddim=50, dev="cpu", expect_b1=EXPECT_B1, expect_b2=EXPECT_B2, out=print):
    run_dir, data = Path(run_dir), Path(data)
    out_dir = Path(out_dir) if out_dir else run_dir / "check_3a"
    for p in (run_dir, data, out_dir):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G")
    if (out_dir / "preds_k50.npz").exists() or (out_dir / "report.json").exists():
        sys.exit(f"STOP: {out_dir} already holds a check; move it aside for a new one")
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = data / "real_val"
    res = {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    fp, hp = data / "stage_d_frozen_val.npz", data / "stage_d_headroom_pred.npz"
    frozen = dict(np.load(fp))
    info = json.loads((run_dir / "run_info.json").read_text())
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    out(f"stage_f_check.py md5 {X.file_md5(__file__)}; stage_f_train.py {X.file_md5(HERE / 'stage_f_train.py')}; "
        f"obj2_model.py {X.file_md5(HERE / 'obj2_model.py')}; stage_e_inputs.py {X.file_md5(HERE / 'stage_e_inputs.py')}")
    out(f"   best.pt md5 {X.file_md5(run_dir / 'best.pt')} (epoch {ck['epoch']}, score {ck['score']:.5f}); frozen set md5 "
        f"{X.file_md5(fp)}; headroom predictions md5 {X.file_md5(hp)}; K {K}, DDIM {ddim}, device {dev}")
    model = M.UNet(cfg["base"], cfg["emb"]); model.load_state_dict(ck["ema"]); model = model.to(dev)
    diff = M.Diffusion(cfg["T"]).to(dev)
    sel = np.arange(len(frozen["row"]))
    voff, vblank = E.frozen_draws(frozen)
    t0 = time.time()
    with D.TruthGuard() as guard:
        try:
            VA = E.Inputs(val_dir, frozen["files"])
            comm = np.array(VA.seed)[frozen["file_index"]]
            counts, counts8, aoo, h_in = sample_all(model, diff, VA, frozen, sel, K, ddim, dev, seed=11)
            get, part = null_inputs(VA, frozen, comm)
            ncounts, _, _, h_null = sample_all(model, diff, VA, frozen, sel, 8, ddim, dev, seed=13, get=get)
            crashed = None
        except RuntimeError as e:
            if not guard.opened:
                raise
            crashed = str(e)
    if crashed:
        gate("C1", False, f"truth file opened while the predictions were built ({crashed[:120]})")
        return res
    X.save_npz(out_dir / "preds_k50.npz", dict(counts=counts, counts8_t=counts8, aoo=aoo, sel=sel, K=np.int64(K),
                                                seed=np.int64(11), in_md5=np.array(h_in), epoch=np.int64(ck["epoch"])))
    X.save_npz(out_dir / "null_k8.npz", dict(counts_t=ncounts[:, W - 1], partner=part, seed=np.int64(13),
                                              in_md5=np.array(h_null)))
    m1, m2 = X.file_md5(out_dir / "preds_k50.npz"), X.file_md5(out_dir / "null_k8.npz")
    out(f"   predictions written before any truth file was opened: preds_k50.npz md5 {m1}, null_k8.npz md5 {m2} "
        f"({(time.time() - t0) / 60:.1f} min of sampling)")
    gate("C1", not guard.opened, f"no truth file opened while both sets of predictions were built (opened {len(guard.opened)})")

    # ---- truth, after the predictions are written
    Tc, tc = E.targets(val_dir, frozen["files"], frozen["file_index"], frozen["row"], voff)
    T6 = unplace(Tc, voff) > 0
    Tt = T6[:, W - 1]
    V, _, _ = VA.arrays(frozen["file_index"], frozen["row"])
    Mk = ~V[:, W - 1] if BROKEN != "score_recorded" else np.ones_like(V[:, W - 1])
    rec_t = V[:, W - 1].reshape(len(V), -1).sum(1)
    rec6 = V.any(1).reshape(len(V), -1).sum(1)
    rcls, rbin = D.tidx(rec_t), D.bidx(rec6)
    win = lambda A: unplace(A[:, None].astype(np.float32), voff)[:, 0]
    S = {"model": win(counts[:, W - 1]) / K, "model_k8": win(counts8) / min(8, K), "null": win(ncounts[:, W - 1]) / 8}
    hz = dict(np.load(hp))
    B1, B2, B2e = hz["B1"], hz["B2"], hz["B2e"]
    if BROKEN == "misalign_b1":
        B1, B2 = np.roll(B1, 1, 0), np.roll(B2, 1, 0)
    S["B1"], S["B2"] = B1, B2
    S["B2_halfblank"] = np.where(frozen["blank"].astype(bool)[:, None, None], B2e, B2)
    S["random"] = X.rng(99, 41).random(Tt.shape).astype(np.float32)
    A = {k: HR.aucs(v, Tt, Mk) for k, v in S.items()}
    bal = {k: HR.balanced(A[k], tc) for k in A}

    gate("C2", str(hz["frozen_md5"]) == X.file_md5(fp) and round(bal["B1"][0], 3) == expect_b1
         and round(bal["B2"][0], 3) == expect_b2,
         f"headroom predictions made from this frozen set {str(hz['frozen_md5']) == X.file_md5(fp)}; B1 {bal['B1'][0]:.4f} "
         f"and B2 {bal['B2'][0]:.4f} recomputed here (stage D {expect_b1:.3f}, {expect_b2:.3f})")
    bp = np.load(run_dir / "best_val_pred.npz")
    f8 = bp["counts"].astype(np.float32) / int(bp["K"])
    s_train = F.score(f8, VA, frozen, bp["sel"], val_dir)["score"]
    s_here = HR.balanced(HR.aucs(win(f8), Tt, Mk), tc)[0]
    best = info["end"]["best"]
    gate("C3", ck["epoch"] == info["end"]["best_epoch"] == int(bp["epoch"]) and s_train == best and abs(s_here - best) < 1e-9,
         f"best.pt epoch {ck['epoch']} = run_info best epoch {info['end']['best_epoch']}; its saved K-8 predictions score "
         f"{s_train:.6f} by the training code and {s_here:.6f} here (run_info {best:.6f})")
    rec_scored = int((Mk & V[:, W - 1]).sum())
    tol = max(0.01, 4 * bal["random"][1])
    gate("C4", rec_scored == 0 and abs(bal["random"][0] - 0.5) <= tol,
         f"recorded cells among the scored cells {rec_scored}; random score {bal['random'][0]:.3f} (0.50 within {tol:.3f})")

    # ---- F5
    out(f"   AUC by true class at the target year (stacks scored; model {K} samples | first 8 | B1 | B2 half-blank | B2 | null):")
    for c in SCORE:
        k = (tc == c) & ~np.isnan(A["B1"])
        out(f"      {D.TRUE[c]:>6} {int(k.sum()):>6}  " + "  ".join(f"{np.nanmean(A[n][k]):.4f}" for n in
                                                              ("model", "model_k8", "B1", "B2_halfblank", "B2", "null")))
    out("      balanced       " + "  ".join(f"{bal[n][0]:.4f}" for n in ("model", "model_k8", "B1", "B2_halfblank", "B2", "null")))
    f5 = class_boot(A["model"] - A["B1"], tc, comm)
    f5b2 = class_boot(A["model"] - A["B2_halfblank"], tc, comm)
    out("   F5 model - B1, paired per stack, 95% interval by resampling communities (rule: PASS mean >= 0, FAIL upper < 0):")
    v5 = {}
    for c in SCORE:
        r = f5[c]; v5[c] = verdict(r["mean"], r["hi"])
        out(f"      {D.TRUE[c]:>6}  {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]  {v5[c]}   "
            f"(model - B2 half-blank {f5b2[c]['mean']:+.4f} [{f5b2[c]['lo']:+.4f}, {f5b2[c]['hi']:+.4f}])")
    r = f5["balanced"]
    out(f"      balanced {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]   (model - B2 half-blank "
        f"{f5b2['balanced']['mean']:+.4f} [{f5b2['balanced']['lo']:+.4f}, {f5b2['balanced']['hi']:+.4f}])")
    f5v = "FAIL" if "FAIL" in v5.values() else ("PASS" if all(v == "PASS" for v in v5.values()) else "NOT RESOLVED")
    out(f"F5 model at least the smoother in every true class: {f5v}")

    # ---- AOO
    exp = aoo[:, W - 1].mean(1).astype(np.float64)
    true = Tt.reshape(len(Tt), -1).sum(1).astype(np.float64)
    lo, hi = np.percentile(aoo[:, W - 1], 5, axis=1), np.percentile(aoo[:, W - 1], 95, axis=1)
    tab = {"recorded class": aoo_table(exp, true, lo, hi, rcls, TCLS),
           "records over 6 years": aoo_table(exp, true, lo, hi, rbin, BINS),
           "true class (report only)": aoo_table(exp, true, lo, hi, tc, D.TRUE)}
    out(f"   AOO at the target year: ratio = mean expected / mean true; band {BAND[0]}-{BAND[1]} for the observable groups; "
        "raw 90% interval coverage (no calibration yet)")
    for name, rows in tab.items():
        out(f"      by {name}: " + "; ".join(f"{q['group']} n {q['n']} ratio {q['ratio']:.2f} cov {q['cov90']:.2f}"
                                           + ("" if name.startswith("true") else (" in" if q["band"] else " OUT")) for q in rows))
    outside = [f"{n}: {q['group']}" for n, rows in tab.items() if not n.startswith("true") for q in rows if not q["band"]]
    out(f"AOO groups outside the band (to stage G level correction): {', '.join(outside) if outside else 'none'}")

    # ---- F3
    nb = bal["null"][0]
    ncls = {D.TRUE[c]: float(np.nanmean(A["null"][(tc == c)])) for c in SCORE}
    f3 = nb <= NULL_BAL and all(v <= NULL_CLASS for v in ncls.values())
    out(f"F3 records of another community's stack: balanced AUC {nb:.4f} (rule <= {NULL_BAL}), by class "
        + ", ".join(f"{k} {v:.3f}" for k, v in ncls.items()) + f" (rule <= {NULL_CLASS}): {'PASS' if f3 else 'FAIL'}")

    # ---- files
    with open(out_dir / "per_stack.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stack", "community", "file_index", "row", "true_class", "recorded_class", "records_6y_bin", "records_6y",
                    "auc_model", "auc_model_k8", "auc_b1", "auc_b2_halfblank", "auc_b2", "auc_null", "aoo_expected",
                    "aoo_true", "aoo_p5", "aoo_p95", "blank"])
        for i in range(len(sel)):
            w.writerow([i, int(comm[i]), int(frozen["file_index"][i]), int(frozen["row"][i]), D.TRUE[tc[i]], TCLS[rcls[i]],
                        BINS[rbin[i]], int(rec6[i])] + [f"{A[n][i]:.5f}" for n in
                        ("model", "model_k8", "B1", "B2_halfblank", "B2", "null")]
                       + [f"{exp[i]:.3f}", int(true[i]), f"{lo[i]:.1f}", f"{hi[i]:.1f}", int(frozen["blank"][i])])
    rep = dict(gates=res, F5=dict(verdict=f5v, by_class={D.TRUE[c]: dict(f5[c], verdict=v5[c]) for c in SCORE},
                                  balanced=f5["balanced"], vs_b2_halfblank={D.TRUE[c]: f5b2[c] for c in SCORE}),
               auc_balanced={k: v[0] for k, v in bal.items()}, aoo=tab, aoo_outside=outside,
               F3=dict(verdict="PASS" if f3 else "FAIL", balanced=nb, by_class=ncls),
               md5=dict(preds=m1, null=m2, best_pt=X.file_md5(run_dir / "best.pt"), tool=X.file_md5(__file__)),
               K=K, ddim=ddim, best_epoch=int(ck["epoch"]))
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    n_ok = sum(res.values())
    out(f"   written {out_dir}/report.json and per_stack.csv")
    out(f"stage F step 3a: gates {n_ok} PASS, {len(res) - n_ok} FAIL; F5 {f5v}; F3 {'PASS' if f3 else 'FAIL'}; "
        f"AOO groups outside the band {len(outside)}")
    res.update(F5=f5v, F3=f3, outside=outside, bal=bal, v5=v5)
    return res


# ---------------------------------------------------------------- self-test (CPU, tiny model, planted data)
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    off = np.array([[0, 0], [4, 4], [2, 1], [3, 0]])
    Aw = np.random.default_rng(0).random((4, 2, G, G)).astype(np.float32)
    t("U1", np.array_equal(unplace(E.place(Aw, off), off), Aw), "unplace undoes the stage E placement at every offset")
    tc = np.repeat([1, 2, 3], 40); comm = np.tile(np.arange(10), 12)
    rng = np.random.default_rng(1)
    pos = class_boot(0.02 + 0.01 * rng.standard_normal(120), tc, comm, classes=[1, 2, 3])
    neg = class_boot(-0.05 + 0.01 * rng.standard_normal(120), tc, comm, classes=[1, 2, 3])
    mix = class_boot(-0.002 + 0.05 * rng.standard_normal(120), tc, comm, classes=[1, 2, 3])
    t("U2", all(verdict(pos[c]["mean"], pos[c]["hi"]) == "PASS" for c in (1, 2, 3))
      and all(verdict(neg[c]["mean"], neg[c]["hi"]) == "FAIL" for c in (1, 2, 3))
      and any(verdict(mix[c]["mean"], mix[c]["hi"]) == "NOT RESOLVED" for c in (1, 2, 3)),
      "F5 rule: a clear gain passes, a clear loss fails, a small noisy loss is not resolved")
    rows = aoo_table(np.array([2.0, 2.0, 10.0]), np.array([2.0, 2.0, 4.0]), np.zeros(3), np.full(3, 3.0),
                     np.array([0, 0, 1]), ["a", "b"])
    t("U3", rows[0]["ratio"] == 1.0 and rows[0]["band"] and rows[1]["ratio"] == 2.5 and not rows[1]["band"]
      and rows[0]["cov90"] == 1.0 and rows[1]["cov90"] == 0.0, "AOO ratio, band and interval coverage on hand-made numbers")
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
        hres, hinfo = HR.run(str(base), str(base / "real_train"), str(base / "real_val"), str(base), out=quiet)
        e1, e2 = round(hinfo["bal"]["B1"], 3), round(hinfo["bal"]["B2"], 3)
        cfg = dict(F.CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=2, K=2, ddim=4,
                   val_set="full", val_chunk=64, lr=1e-3)
        F.train(cfg, str(base), str(tmp / "run"), "cpu", log=quiet)
        r = run(tmp / "run", base, K=4, ddim=4, dev="cpu", expect_b1=e1, expect_b2=e2, out=quiet)
        t("S1", all(r.get(g) for g in ("C1", "C2", "C3", "C4")) and (tmp / "run" / "check_3a" / "report.json").is_file()
          and r["F5"] in ("PASS", "FAIL", "NOT RESOLVED"),
          f"planted run end to end: C1 {r.get('C1')}, C2 {r.get('C2')}, C3 {r.get('C3')}, C4 {r.get('C4')}; F5 {r.get('F5')}")
        t("S2", abs(r["bal"]["B1"][0] - hinfo["bal"]["B1"]) < 1e-12 and abs(r["bal"]["B2"][0] - hinfo["bal"]["B2"]) < 1e-12,
          "B1 and B2 recomputed here equal stage D's own values exactly (same stacks, same cells)")
        pz = np.load(tmp / "run" / "check_3a" / "preds_k50.npz")
        t("S3", pz["counts"].max() <= 4 and int(pz["aoo"][:, -1].sum()) == int(pz["counts"][:, -1].astype(np.int64).sum())
          and pz["counts8_t"].max() <= 4, "sample counts and per-sample AOO agree (cells occupied summed two ways)")
        stop = False
        try:
            run(tmp / "run", base, K=4, ddim=4, dev="cpu", expect_b1=e1, expect_b2=e2, out=quiet)
        except SystemExit:
            stop = True
        t("S4", stop, "a second check into the same folder is refused")
        frozen = dict(np.load(base / "stage_d_frozen_val.npz"))
        VA = E.Inputs(base / "real_val", frozen["files"])
        comm_ = np.array(VA.seed)[frozen["file_index"]]
        get, part = null_inputs(VA, frozen, comm_)
        k = np.arange(min(32, len(frozen["row"])))
        xn, _ = get(k)
        voff, vbl = E.frozen_draws(frozen)
        xr, _ = VA.get(frozen["file_index"][k], frozen["row"][k], voff[k], vbl[k])
        xp, _ = VA.get(frozen["file_index"][part[k]], frozen["row"][part[k]], voff[k], vbl[k])
        same_own = np.array_equal(xn[:, 12:18], xr[:, 12:18]) and np.array_equal(xn[:, 18:20], xr[:, 18:20]) \
            and np.array_equal(xn[:, 21], xr[:, 21])
        recs_partner = np.array_equal(xn[:, 0:12], xp[:, 0:12])
        t("S5", same_own and recs_partner and (comm_[part] != comm_).all(),
          "null inputs: records and halo of a stack of another community; own effort, habitat, blank flag and window")
        for bug, gate in (("truth_in_sample", "C1"), ("misalign_b1", "C2"), ("score_recorded", "C4"), ("null_keeps_records", "S5")):
            env = dict(os.environ); env["STAGE_F_CHECK_BROKEN"] = bug
            if gate == "S5":
                code = ("import importlib.util as u, numpy as np; s=u.spec_from_file_location('c', %r); m=u.module_from_spec(s); "
                        "s.loader.exec_module(m); fr=dict(np.load(%r)); VA=m.E.Inputs(%r, fr['files']); "
                        "cm=np.array(VA.seed)[fr['file_index']]; g,p=m.null_inputs(VA, fr, cm); k=np.arange(min(32,len(fr['row']))); "
                        "o,b=m.E.frozen_draws(fr); xn,_=g(k); xp,_=VA.get(fr['file_index'][p[k]], fr['row'][p[k]], o[k], b[k]); "
                        "print('RES', np.array_equal(xn[:,0:12], xp[:,0:12]))"
                        % (str(Path(__file__).resolve()), str(base / "stage_d_frozen_val.npz"), str(base / "real_val")))
                p = __import__("subprocess").run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
                t(f"bug {bug}", "RES False" in p.stdout, f"null inputs that keep the stack's own records fail S5 "
                  f"({p.stdout.strip()[-40:] or p.stderr.strip()[-200:]})")
                continue
            code = ("import importlib.util as u; s=u.spec_from_file_location('c', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r=m.run(%r, %r, out_dir=%r, K=4, ddim=4, dev='cpu', expect_b1=%r, expect_b2=%r, out=lambda *a, **k: None); "
                    "print('RES', {k: r.get(k) for k in ('C1','C2','C3','C4')})"
                    % (str(Path(__file__).resolve()), str(tmp / "run"), str(base), str(tmp / f"b_{bug}"), e1, e2))
            p = __import__("subprocess").run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        stop = False
        try:
            run(tmp / "run", tmp / "real_eval", K=4, ddim=4, out=quiet)
        except SystemExit:
            stop = True
        t("S6", stop, "a real_eval path is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", default="../runs/f_seed1"); ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--K", type=int, default=50); ap.add_argument("--ddim", type=int, default=50)
    ap.add_argument("--expect-b1", type=float, default=EXPECT_B1); ap.add_argument("--expect-b2", type=float, default=EXPECT_B2)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        torch.use_deterministic_algorithms(True)
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    r = run(a.run, a.data, a.out, K=a.K, ddim=a.ddim, dev=dev, expect_b1=a.expect_b1, expect_b2=a.expect_b2,
            out=lambda s: print(s, flush=True))
    sys.exit(0 if all(r.get(g) for g in ("C1", "C2", "C3", "C4")) else 1)