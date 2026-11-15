#!/usr/bin/env python3
"""
stage_d_sampler.py  (obj2_v2, stage D step 1, 8 Nov 2026)

What it does (claude/obj2_v2_plan.md, gates D1-D5, signed off by Sara on 8 Nov)
    1. Training sampler: every real_train species-stack with at least one record gets a weight, so the weighted recorded
       classes match the BNM fit half. Classes are the recorded cells at the target year (0 = records only in the 5 years
       before, 1, 2-5, 6-10, 11-20, 21-50, 51+), on a 6-year basis: BNM species-windows with at least one recorded tetrad
       in the 6 years, recomputed from the per-year grids. Weight = BNM share / training share of the class (Sara, 8 Nov:
       class weights first). If the class weights leave the share of stacks with 1-3 recorded cells over 6 years more than
       2 points from BNM, the weights are raked to both margins, recorded class and record-count bin (Sara, 8 Nov: record-
       count weights only if needed). Training stage F draws stacks with probability proportional to the weight.
    2. Frozen validation set from real_val: every stack with 0 or 1 recorded cells at the target year (and at least one
       record over 6 years), plus 1,500 drawn with a fixed seed from each other recorded class; one fixed habitat blank
       per stack (exactly half of each class blanked) and one fixed shift per stack, for stage E. Scoring in stage F uses
       every unrecorded cell (Sara, 8 Nov), so nothing is hidden here. A fixed fast subset (up to 300 stacks per recorded
       class) is marked for frequent validation if the full set proves too slow in stage F (timing measured there).
    Steps 1 and 2 read input files only; a guard stops the run if any truth file is opened while they run.

Gates (rules written before the run, 8 Nov)
    D1 no truth file opened while the sampler and the frozen set are built (guard); self-test: outputs byte-equal with
       every truth file removed
    D2 no stack without a record in the sampler or the frozen set
    D3 weighted recorded-class shares within 1 point of BNM (and record-count bins within 1 point if raked); largest
       weight and effective sample size per class reported, FLAG if a weight is above 5
    D4 frozen set: real_val input files only, no community seed of real_train, exactly half of each class blanked,
       md5 written; true classes reported (truth read after the build), FLAG if a true class of 1-20 cells has fewer
       than 100 stacks
    D5 (report) distinct species and distinct (species, 6 true maps) per class, so repeats count once

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_d_sampler.py --self-test | tail -1
    python3 ../tools/stage_d_sampler.py --train ../data/real_train --val ../data/real_val --out ../data | tee ../logs/stage_d_sampler_8nov.txt
Needs numpy and export_obj2.py in the same folder; reads the BNM per-year grids (fit half only).
"""
import argparse, csv, hashlib, importlib.util, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load_export():
    p = HERE / "export_obj2.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_d_sampler.py needs export_obj2.py in the same folder")
    spec = importlib.util.spec_from_file_location("export_obj2", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, m.file_md5(p)


X, XMD5 = _load_export()
G, NC, WINDOW = X.G, X.NC, X.WINDOW
TCLS = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
T_EDGES = np.array([0, 1, 2, 6, 11, 21, 51])
BINS = ["1-3", "4-6", "7-10", "11-20", "21+"]
B_EDGES = np.array([1, 4, 7, 11, 21])
TRUE = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
RARE_TRUE = ["1", "2-5", "6-10", "11-20"]
RAKE_GAP, W_FLAG, N_PER_CLASS, MIN_TRUE, N_FAST = 2.0, 5.0, 1500, 100, 300
SPARSE = (0, 1)                                      # recorded classes taken whole into the frozen set
BROKEN = os.environ.get("STAGE_D_BROKEN", "")       # self-test only: plants one bug so its gate must fail


def tidx(n):
    return np.searchsorted(T_EDGES, np.asarray(n), side="right") - 1


def bidx(n):
    return np.clip(np.searchsorted(B_EDGES, np.asarray(n), side="right") - 1, 0, len(BINS) - 1)


def rows_of(d):
    man = Path(d) / "manifest.csv"
    if not man.is_file():
        sys.exit(f"STOP: {man} not found")
    with open(man) as f:
        return list(csv.DictReader(f))


class TruthGuard:
    """While active, any np.load of a *_truth.npz file is refused and logged (gate D1)."""
    def __enter__(self):
        self.orig, self.opened = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        if "_truth" in str(f):
            self.opened.append(str(f))
            raise RuntimeError(f"truth file opened while the sampler runs: {f}")
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig
        return False


def scan_inputs(d):
    """Per stack, from input files only: file index, row, recorded cells at the target year, recorded cells over 6 years.
    Each input file's md5 is checked against the manifest."""
    rows = rows_of(d)
    fi, rw, rt, r6, md5s = [], [], [], [], []
    for i, row in enumerate(rows):
        p = Path(d) / row["input_file"]
        m = X.file_md5(p)
        if m != row["input_md5"]:
            sys.exit(f"STOP: {p} md5 {m} differs from the manifest {row['input_md5']}")
        md5s.append(m)
        with np.load(p, allow_pickle=False) as z:
            v = X.unpack(z["visit1"]) | X.unpack(z["visit2"])
        n = len(v)
        rec_t = v[:, -1].reshape(n, -1).sum(1)
        if BROKEN == "truth_class":
            with np.load(Path(d) / row["truth_file"]) as zt:
                rec_t = zt["true_cells"].astype(int)
        fi.append(np.full(n, i)); rw.append(np.arange(n)); rt.append(rec_t); r6.append(v.any(1).reshape(n, -1).sum(1))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, int)
    return rows, md5s, cat(fi), cat(rw), cat(rt), cat(r6)


def bnm_joint(peryear, split_ref, tools, minlist=8):
    """BNM fit-half species-windows with 1+ recorded tetrad over the 6 years: counts by recorded class at the target year
    (rows, TCLS) and recorded tetrads over 6 years (columns, BINS)."""
    bdc, _ = X.import_bdc(tools)
    pooled = Path(peryear) / f"minlist{minlist}"
    ydirs = sorted(p for p in Path(peryear).glob(f"minlist{minlist}_y*") if p.is_dir())
    names = sorted(f.name for f in pooled.glob("bnm_*.npz"))
    keep, _, _ = bdc.fit_half(names, bdc.split_ref(split_ref))
    J = np.zeros((len(TCLS), len(BINS)))
    for nm in keep:
        O = [np.asarray(X.bnm_load(bdc, pooled / nm)["obs_mask"]).astype(bool)]
        O += [np.asarray(X.bnm_load(bdc, d / nm)["present_year"]).astype(bool) for d in ydirs]
        nt = O[0].reshape(len(O[0]), -1).sum(1)
        n6 = np.any(np.stack(O), 0).reshape(len(O[0]), -1).sum(1)
        ok = n6 > 0
        np.add.at(J, (tidx(nt[ok]), bidx(n6[ok])), 1)
    return J, len(keep)


def rake(S, B, iters=5000, tol=1e-12):
    """Iterative proportional fitting: factors a (classes) and b (bins) so S * a b^T has B's two margins."""
    P = S / S.sum(); tc = B.sum(1) / B.sum(); tb = B.sum(0) / B.sum()
    a, b = np.ones(len(tc)), np.ones(len(tb)); err = np.inf
    for _ in range(iters):
        rc = (P * b[None]).sum(1); a = np.where(rc > 0, tc / np.where(rc > 0, rc, 1), 0)
        rb = (P * a[:, None]).sum(0); b = np.where(rb > 0, tb / np.where(rb > 0, rb, 1), 0)
        W = P * a[:, None] * b[None]
        err = abs(W.sum(1) - tc).max() + abs(W.sum(0) - tb).max()
        if err < tol:
            break
    return a[:, None] * b[None], err


def weights(cls, bins, Bj):
    """Stack weights (mean 1), the table used, the scheme, and the shares before and after."""
    S = np.zeros_like(Bj); np.add.at(S, (cls, bins), 1)
    sc, bc = S.sum(1) / S.sum(), Bj.sum(1) / Bj.sum()
    wc = np.where(sc > 0, bc / np.where(sc > 0, sc, 1), 0)
    Tclass = np.repeat(wc[:, None], len(BINS), 1)
    w = Tclass[cls, bins]
    bin_w = np.bincount(bins, weights=w, minlength=len(BINS)) / w.sum()
    gap13 = 100 * abs(bin_w[0] - Bj.sum(0)[0] / Bj.sum())
    scheme, T, err = "class", Tclass, 0.0
    if gap13 > RAKE_GAP:
        Tr, err = rake(S, Bj)
        if err < 1e-6:
            scheme, T = "raked", Tr
    w = T[cls, bins]
    if BROKEN == "inverted":
        w = np.where(w > 0, 1 / np.where(w > 0, w, 1), 0)
    w = w / w.mean()
    return w.astype(np.float64), T, scheme, gap13, err, S


def frozen_set(cls, r6):
    """Indices into the real_val stacks: all stacks of the sparse classes, 1,500 per other class (fixed seed); exact-half
    blank per class; one shift per stack."""
    elig = r6 >= 1 if BROKEN != "keep_zero" else r6 >= 0
    pick = []
    g = X.rng(99, 1)
    for c in range(len(TCLS)):
        idx = np.nonzero(elig & (cls == c))[0]
        if c in SPARSE or len(idx) <= N_PER_CLASS:
            pick.append(idx)
        else:
            pick.append(np.sort(g.choice(idx, N_PER_CLASS, replace=False)))
    sel = np.sort(np.concatenate(pick)) if pick else np.zeros(0, int)
    blank = np.zeros(len(sel), np.uint8)
    for c in range(len(TCLS)):
        k = np.nonzero(cls[sel] == c)[0]
        if BROKEN == "blank_by_class":
            blank[k] = 1 if c in SPARSE else 0
            continue
        perm = X.rng(99, 2, c).permutation(k)
        blank[perm[:len(k) // 2]] = 1
    shift = X.rng(99, 3).integers(0, G, size=(len(sel), 2)).astype(np.int8)
    fast = np.zeros(len(sel), np.uint8)
    for c in range(len(TCLS)):
        k = np.nonzero(cls[sel] == c)[0]
        fast[X.rng(99, 4, c).permutation(k)[:N_FAST]] = 1
    return sel, blank, shift, fast


def build(train_dir, val_dir, out_dir, peryear, split_ref, tools, out=print):
    for d in (train_dir, val_dir, out_dir):
        if "real_eval" in str(d):
            sys.exit("STOP: real_eval is read once, after training")
    Bj, n_win = bnm_joint(peryear, split_ref, tools)
    with TruthGuard() as guard:
        try:
            trows, tmd5, tfi, trw, trt, tr6 = scan_inputs(train_dir)
            vdir = val_dir if BROKEN != "val_from_train" else train_dir
            vrows, vmd5, vfi, vrw, vrt, vr6 = scan_inputs(vdir)
            elig = tr6 >= 1 if BROKEN != "keep_zero" else tr6 >= 0
            cls, bins = tidx(trt[elig]), bidx(tr6[elig])
            w, T, scheme, gap13, err, S = weights(cls, bins, Bj)
            vcls = tidx(vrt)
            sel, blank, shift, fast = frozen_set(vcls, vr6)
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        out(f"D1 {crashed}: FAIL")
        return dict(D1=False), None
    cmd5 = X.file_md5(__file__)
    os.makedirs(out_dir, exist_ok=True)
    sp, fp = Path(out_dir) / "stage_d_train_sampler.npz", Path(out_dir) / "stage_d_frozen_val.npz"
    X.save_npz(sp, dict(files=np.array([r["input_file"] for r in trows]), file_md5=np.array(tmd5),
                        file_index=tfi[elig].astype(np.int16), row=trw[elig].astype(np.int32), weight=w.astype(np.float32),
                        rec_t=trt[elig].astype(np.int16), rec6=tr6[elig].astype(np.int16), cls=cls.astype(np.int8),
                        bin=bins.astype(np.int8), table=T, scheme=np.array(scheme), bnm_joint=Bj, sim_joint=S,
                        code_md5=np.array(cmd5), classes=np.array(TCLS), bins_names=np.array(BINS)))
    X.save_npz(fp, dict(files=np.array([r["input_file"] for r in vrows]), file_md5=np.array(vmd5),
                        file_index=vfi[sel].astype(np.int16), row=vrw[sel].astype(np.int32), rec_t=vrt[sel].astype(np.int16),
                        rec6=vr6[sel].astype(np.int16), cls=vcls[sel].astype(np.int8), blank=blank, shift=shift, fast=fast,
                        code_md5=np.array(cmd5), classes=np.array(TCLS)))
    res = {}

    def check(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    out(f"stage_d_sampler.py md5 {cmd5}; export_obj2.py md5 {XMD5}; BNM fit half {n_win} windows, {int(Bj.sum())} "
        f"species-windows with 1+ record over 6 years")
    check("D1", not guard.opened, f"no truth file opened while the sampler and frozen set were built (opened {len(guard.opened)})")
    z_tr, z_va = int((tr6[elig] == 0).sum()), int((vr6[sel] == 0).sum())
    check("D2", z_tr == 0 and z_va == 0, f"stacks without a record: sampler {z_tr} of {int(elig.sum())}, frozen set {z_va} of {len(sel)}; "
          f"left out of training as having no record: {int((tr6 == 0).sum())}")
    bc, bb = Bj.sum(1) / Bj.sum(), Bj.sum(0) / Bj.sum()
    wc = np.bincount(cls, weights=w, minlength=len(TCLS)) / w.sum()
    wb = np.bincount(bins, weights=w, minlength=len(BINS)) / w.sum()
    rc = np.bincount(cls, minlength=len(TCLS)) / len(cls)
    rb = np.bincount(bins, minlength=len(BINS)) / len(bins)
    ok3 = np.all(abs(wc - bc) * 100 <= 1) and (scheme != "raked" or np.all(abs(wb - bb) * 100 <= 1))
    out(f"   class weights left the 1-3 record bin {gap13:.1f} points from BNM (rake above {RAKE_GAP}): weights used = {scheme}"
        + (f" (raking error {err:.1e})" if scheme == "raked" else ""))
    out("   recorded class at the target year   training   weighted   BNM     stacks    ESS     largest weight")
    for c, lab in enumerate(TCLS):
        k = cls == c
        ess = (w[k].sum() ** 2 / (w[k] ** 2).sum()) if k.any() and w[k].sum() > 0 else 0
        out(f"   {lab:>6}                             {100 * rc[c]:6.1f}%   {100 * wc[c]:6.1f}%   {100 * bc[c]:5.1f}%  {int(k.sum()):>8}  "
            f"{ess:>8.0f}  {w[k].max() if k.any() else 0:6.2f}")
    out("   recorded cells over 6 years          training   weighted   BNM")
    for b, lab in enumerate(BINS):
        out(f"   {lab:>6}                             {100 * rb[b]:6.1f}%   {100 * wb[b]:6.1f}%   {100 * bb[b]:5.1f}%")
    check("D3", ok3, "weighted recorded-class shares within 1 point of BNM" + (" and record-count bins too" if scheme == "raked" else "")
          + f"; largest weight {w.max():.2f}" + (" (FLAG above 5)" if w.max() > W_FLAG else ""))
    vfiles, tseeds = {r["input_file"] for r in rows_of(val_dir)}, {r["sim_seed"] for r in rows_of(train_dir)}
    fseeds = {vrows[int(i)]["sim_seed"] for i in np.unique(vfi[sel])}
    used_files = {vrows[int(i)]["input_file"] for i in np.unique(vfi[sel])}
    half_ok = all(int(blank[vcls[sel] == c].sum()) == int((vcls[sel] == c).sum()) // 2 for c in range(len(TCLS)))
    check("D4", used_files <= vfiles and not (fseeds & tseeds) and half_ok and "real_val" in str(val_dir),
          f"frozen set {len(sel)} stacks from {len(used_files)} real_val files, real_train seeds in it {len(fseeds & tseeds)}, "
          f"exactly half of each class blanked {half_ok}; " + ", ".join(f"{lab} {int((vcls[sel] == c).sum())}" for c, lab in enumerate(TCLS)))
    out(f"   written {sp} md5 {X.file_md5(sp)}")
    out(f"   written {fp} md5 {X.file_md5(fp)}")
    # reports that read the truth (after the build; they do not change the outputs)
    tc_f = np.concatenate([np.load(Path(vdir) / vrows[int(i)]["truth_file"])["true_cells"][vrw[sel][vfi[sel] == i]]
                           for i in np.unique(vfi[sel])]) if len(sel) else np.zeros(0, int)
    order = np.concatenate([np.nonzero(vfi[sel] == i)[0] for i in np.unique(vfi[sel])]) if len(sel) else np.zeros(0, int)
    tlab = np.array(TRUE, dtype=object)[np.searchsorted(T_EDGES, tc_f, side="right") - 1]
    counts = {k: int((tlab == k).sum()) for k in TRUE}
    fl = fast[order].astype(bool)
    out(f"   fast subset {int(fast.sum())} stacks; true class: " + ", ".join(f"{k} {int((tlab[fl] == k).sum())}" for k in TRUE))
    low = [k for k in RARE_TRUE if counts[k] < MIN_TRUE]
    out("D4 true class of the frozen-set stacks (truth read after the build): " + ", ".join(f"{k} {v}" for k, v in counts.items())
        + (f": FLAG, fewer than {MIN_TRUE} in {', '.join(low)}" if low else ""))
    dsp = {lab: set() for lab in TCLS}; dmap = {lab: set() for lab in TCLS}
    efi, erw = tfi[elig], trw[elig]
    for i in np.unique(efi):
        with np.load(Path(train_dir) / trows[int(i)]["truth_file"]) as zt:
            ss, tb, sd = zt["stack_species"], zt["true_maps"], int(zt["sim_seed"])
        k = np.nonzero(efi == i)[0]
        for j, r in zip(k, erw[k]):
            lab = TCLS[cls[j]]
            dsp[lab].add((sd, int(ss[r]))); dmap[lab].add((sd, int(ss[r]), hashlib.md5(tb[r].tobytes()).digest()))
    out("D5 per recorded class: stacks / distinct (species, 6 true maps) / distinct species: "
        + "; ".join(f"{lab} {int((cls == c).sum())} / {len(dmap[lab])} / {len(dsp[lab])}" for c, lab in enumerate(TCLS)))
    n_ok = sum(res.values())
    out(f"stage D: {n_ok} PASS, {len(res) - n_ok} FAIL; weights {scheme}; flags: "
        + ", ".join([f for f, on in (("D3 weight above 5", w.max() > W_FLAG), ("D4 thin true class", bool(low))) if on] or ["none"]))
    return res, dict(sampler=str(sp), frozen=str(fp), scheme=scheme, w=w, T=T, Bj=Bj, S=S)


# ---------------------------------------------------------------- self-test
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    # raking on a hand-made table: both margins reached
    S = np.array([[30., 10., 0.], [20., 20., 10.], [5., 25., 80.]]); B = np.array([[10., 5., 1.], [8., 8., 4.], [2., 6., 20.]])
    T, err = rake(S, B)
    W = S / S.sum() * T
    t("S1", err < 1e-9 and np.allclose(W.sum(1), B.sum(1) / B.sum()) and np.allclose(W.sum(0), B.sum(0) / B.sum()),
      f"raking reaches both margins on a hand-made table (error {err:.1e})")
    with tempfile.TemporaryDirectory() as tmp:
        py, refpat = X.plant_grids(tmp)
        st = Path(tmp) / "sim" / "real_train"; st.mkdir(parents=True); sv = Path(tmp) / "sim" / "real_val"; sv.mkdir(parents=True)
        for s in (7001, 7002):
            X.plant_community(st / f"c{s}_training.npz", s)
        for s in (7101, 7102):
            X.plant_community(sv / f"c{s}_training.npz", s)
        base = Path(tmp) / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.6, py, refpat, tools, out=quiet)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.6, py, refpat, tools, out=quiet)
        for split in ("real_train", "real_val"):              # planted: three stacks per split lose their records
            rows = rows_of(base / split); p = base / split / rows[0]["input_file"]
            z = dict(np.load(p))
            for k in ("visit1", "visit2"):
                z[k] = z[k].copy(); z[k][[1, 5, 9]] = 0
            z["n_rec"] = z["n_rec"].copy(); z["n_rec"][[1, 5, 9]] = 0
            X.save_npz(p, z); rows[0]["input_md5"] = X.file_md5(p)
            with open(base / split / "manifest.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        res, info = build(str(base / "real_train"), str(base / "real_val"), str(base / "o1"), py, refpat, tools, out=quiet)
        t("S2", all(res.values()) and len(res) == 4, f"clean planted export: D1-D4 all PASS {res}")
        res2, info2 = build(str(base / "real_train"), str(base / "real_val"), str(base / "o2"), py, refpat, tools, out=quiet)
        same = all(X.file_md5(Path(base / "o1" / n)) == X.file_md5(Path(base / "o2" / n))
                   for n in ("stage_d_train_sampler.npz", "stage_d_frozen_val.npz"))
        t("S3", same, "a rerun writes byte-equal sampler and frozen-set files")
        nt = Path(tmp) / "notruth"; shutil.copytree(base / "real_train", nt / "real_train"); shutil.copytree(base / "real_val", nt / "real_val")
        for p in nt.glob("*/*_truth.npz"):
            p.unlink()
        Bj = info["Bj"]
        with TruthGuard():
            r = scan_inputs(str(nt / "real_train")); v = scan_inputs(str(nt / "real_val"))
        el = r[5] >= 1
        w, *_ = weights(tidx(r[4][el]), bidx(r[5][el]), Bj)
        sel, blank, shift, fast = frozen_set(tidx(v[4]), v[5])
        z1 = np.load(base / "o1" / "stage_d_train_sampler.npz"); z2 = np.load(base / "o1" / "stage_d_frozen_val.npz")
        t("S4", np.array_equal(z1["weight"], w.astype(np.float32)) and np.array_equal(z2["row"], v[3][sel]) and np.array_equal(z2["blank"], blank),
          "with every truth file removed, the weights, the frozen rows and the blanks are the same (inputs only)")
        # forced raking path: a planted BNM table far from the training bins
        Bf = Bj.copy(); Bf[int(np.argmax(info["S"][:, 0])), 0] += 0.3 * Bj.sum()      # BNM given many 1-3 record stacks
        wr, Tr, scheme, gap, err, Sx = weights(tidx(r[4][el]), bidx(r[5][el]), Bf)
        cl, bn = tidx(r[4][el]), bidx(r[5][el])
        okr = scheme == "raked" and np.allclose(np.bincount(cl, wr, len(TCLS)) / wr.sum(), Bf.sum(1) / Bf.sum(), atol=1e-6) \
            and np.allclose(np.bincount(bn, wr, len(BINS)) / wr.sum(), Bf.sum(0) / Bf.sum(), atol=1e-6)
        t("S5", okr, f"a 1-3 bin gap of {gap:.1f} points (above {RAKE_GAP}) switches to raking, and both margins are met")
        # each planted bug must fail its own gate (fresh process per bug)
        want = {"truth_class": "D1", "keep_zero": "D2", "inverted": "D3", "val_from_train": "D4", "blank_by_class": "D4"}
        for bug, gate in want.items():
            env = dict(os.environ); env["STAGE_D_BROKEN"] = bug
            code = ("import sys; sys.argv=['x']; import importlib.util as u; s=u.spec_from_file_location('d', %r); "
                    "m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r,_=m.build(%r, %r, %r, %r, %r, %r, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), str(base / "real_train"), str(base / "real_val"), str(base / f"b_{bug}"),
                       py, refpat, tools))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RES ")), "")
            failed = f"'{gate}': False" in line
            t(f"bug {bug}", failed, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        stop = False
        try:
            build(str(base / "real_train"), str(Path(tmp) / "real_eval"), str(base / "o3"), py, refpat, tools, out=quiet)
        except SystemExit:
            stop = True
        t("S6", stop, "a real_eval folder is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--train", default="../data/real_train"); ap.add_argument("--val", default="../data/real_val")
    ap.add_argument("--out", default="../data")
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--split-ref", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    res, _ = build(a.train, a.val, a.out, a.peryear, a.split_ref, a.obj1_tools)
    sys.exit(0 if all(res.values()) else 1)