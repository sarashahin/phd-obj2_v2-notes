#!/usr/bin/env python3
"""
stage_h1_bnm_export.py  (obj2_v2, stage H step H1, 11 Nov 2026; stage H rules and the H1 plan signed off by Sara on
11 Nov, claude/obj2_v2_H_rules_draft.md)

BNM export into the stage E input format, with holes. One stack = one species in one window with at least one record in
the 6 years (2014-2019). For each stack and each of 5 repeats, holes = (3n + 5) // 10 of its n tetrads surveyed in 2019
(halves up; none when n < 2), drawn at random from effort only (fixed seed per window, species row and repeat). In the
holes, records and effort are removed in all 6 years. The habitat channel is blank (BNM has no habitat map).

Files (one input and one truth file per window and repeat, as export_obj2 writes them per community)
    *_input.npz   export_obj2's allow-list only: visit1 = kept records, visit2 = empty (the builder reads visit1 | visit2,
                  so the model sees the same records), effort with the holes removed, one zero habitat map, n_rec
    *_truth.npz   hole mask, original records and effort, label_strict (hole recorded in 2019), label_pooled (hole
                  recorded in any of the 6 years), species, GB Red List 2022 status
    bnm_frozen.npz  the scored stack-repeats in stage D's frozen-set layout (files, file_md5, file_index, row, rec_t, rec6,
                  cls, blank, shift, fast): every stack-repeat with at least one hole and at least one kept record (Q1,
                  11 Nov: zero kept records are left out and reported); blank = 1; shift fixed per stack-repeat
    bnm_stacks.npz  for each frozen row: window, species, status, repeat, row, stack id, 2019 surveyed tetrads, holes
    manifest.csv, h1_report.json

Gates
    J0 mode: fit half; the test half only with --half test --final-test, a freeze file (H3) listing this tool's md5, and
       no earlier test export (marker data/BNM_TEST_EXPORTED); the output folder is new
    J1 grids: 6 consecutive years ending at the pooled target year; fit / test windows = --expect; species rows and shapes
       equal across years; every record inside that year's effort; no window of the other half passed to np.load
    J2 holes: (3n + 5) // 10 per stack (none when n < 2), inside the 2019 effort, distinct
    J3 written inputs, read afresh: keys = the allow-list; md5 = manifest; records and effort 0 in the holes in all 6
       years; outside the holes equal to the grids bit for bit; visit2 empty; n_rec = the records
    J4 (fit half only) the hole mask carries no answer: mean per-stack AUC of hole against 2019 record, among 2019 surveyed
       tetrads, 0.48-0.52 (rule E5 / H2)
    J4b holes drawn again with every record set to 0 are the same (no record enters the draw; label-free, so it also
       serves the test half, where J4 would read hole labels)
    J5 truth apart: frozen set and index built from the input files under stage D's truth guard; no truth key in an input
       file
    J6 builder: tensors of every frozen row bit-identical by three paths (packed inputs in memory, each input file read
       afresh, the grids with holes in memory); the builder accepts every frozen row (no zero-record stack)
    J7 a second hole draw gives the same masks (md5)
Reported (fit half): stack-repeats scored, left out (no kept record; by records over 6 years) and without holes; by kept
recorded class at 2019 and kept records over 6 years: stack-repeats, distinct stacks, rankable stack-repeats (holes hold a
recorded and an unrecorded 2019 tetrad), rankable distinct stacks (rule floor 50, Q2); hole shares recorded (strict,
pooled); stacks with 10+ records in 2019 (7 Nov version); GB Red List status. Test half: input-side counts only.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_h1_bnm_export.py --self-test | tail -1
    python3 -u ../tools/stage_h1_bnm_export.py --out ../data/bnm_fit > ../logs/stage_h1_bnm_export.txt 2>&1
Defaults as stage_h0_bnm_check.py. Needs export_obj2.py, stage_d_sampler.py and stage_e_inputs.py in the same folder.
"""
import argparse, csv, glob, hashlib, importlib.util, json, os, shutil, sys, tempfile, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_h1_bnm_export.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


E = _load("stage_e_inputs")
D = E.D
X = D.X
G, W, NC = X.G, X.WINDOW, X.NC
TCLS, BINS = D.TCLS, D.BINS
FROZEN_KEYS = ("files", "file_md5", "file_index", "row", "rec_t", "rec6", "cls", "blank", "shift", "fast")
TRUTH_H = ("code_md5", "eff_orig", "hole", "label_pooled", "label_strict", "pair_md5", "rec_orig", "repeat", "species",
           "status", "window_name", "years")
KEY = 31101                                               # seed key of stage H1 (holes 1, shift 3)
REPEATS, FLOOR = 5, 50
MARKER = "BNM_TEST_EXPORTED"


def broken():
    return os.environ.get("STAGE_H1_BROKEN", "")         # self-test only: plants one bug so its gate must fail


def n_holes(n):
    if broken() == "round_even":
        return int(np.round(0.3 * n)) if n >= 2 else 0
    return (3 * n + 5) // 10 if n >= 2 else 0


def draw_holes(Et, Rt, e, n, s, r):
    """Et, Rt (G, G) bool: 2019 effort and records of one stack -> hole mask (G, G). Records are not used (J4b)."""
    idx = np.flatnonzero(Et)
    k = n_holes(len(idx))
    hole = np.zeros(NC, bool)
    if k:
        g = X.rng(KEY, 1, e, n, s, r)
        if broken() == "holes_on_records":
            w = 1.0 + 4.0 * Rt.ravel()[idx]
            pick = g.choice(idx, k, replace=False, p=w / w.sum())
        else:
            pick = g.choice(idx, k, replace=False)
        hole[pick] = True
    return hole.reshape(G, G)


def shift_of(e, n, s, r):
    return X.rng(KEY, 3, e, n, s, r).integers(0, G, size=2).astype(np.int8)


class LoadLog:
    def __enter__(self):
        self.orig, self.paths = np.load, []
        def rec(f, *a, **k):
            self.paths.append(str(f))
            return self.orig(f, *a, **k)
        np.load = rec
        return self

    def __exit__(self, *e):
        np.load = self.orig
        return False


def read_window(pooled, ydirs, nm):
    """-> R, Ef (6, S, G, G) bool, species, status, ok (rows and shapes equal), records outside effort"""
    a = np.load(pooled / nm, allow_pickle=True)
    R, Ef, sp = [], [], []
    for y, d in ydirs:
        z = np.load(d / nm, allow_pickle=True)
        R.append(np.asarray(z["present_year"]).astype(bool)); Ef.append(np.asarray(z["known_holdout"]).astype(bool))
        sp.append(np.asarray(z["species"]) if "species" in z.files else None)
    R.append(np.asarray(a["obs_mask"]).astype(bool)); Ef.append(np.asarray(a["known_target"]).astype(bool))
    sp.append(np.asarray(a["species"]) if "species" in a.files else None)
    shapes = {x.shape for x in R} | {x.shape for x in Ef}
    named = [s for s in sp if s is not None]
    ok = len(shapes) == 1 and next(iter(shapes))[1:] == (G, G) and len(named) == W and \
        all(np.array_equal(s, named[0]) for s in named[1:])
    if not ok:
        return None, None, None, None, False, 0
    R, Ef = np.stack(R), np.stack(Ef)
    S = R.shape[1]
    st = np.asarray(a["status"]).astype(str) if "status" in a.files else np.array([""] * S)
    return R, Ef, named[0].astype(str), st, True, int((R & ~Ef).sum())


def mode_ok(half, final_test, freeze, out_dir):
    if Path(out_dir).exists():
        return False, f"output folder {out_dir} exists (an export is written once into a new folder)"
    if half == "fit":
        return True, "fit half; output folder new"
    if not final_test:
        return False, "the test half needs --final-test (stage H4, once)"
    if not freeze or not Path(freeze).is_file():
        return False, "the test half needs --freeze (the stage H3 md5 list)"
    me = X.file_md5(__file__)
    listed = json.loads(Path(freeze).read_text()).get("tools", {}).get(Path(__file__).name)
    if listed != me:
        return False, f"this tool's md5 {me} differs from the frozen md5 {listed}"
    mk = Path(out_dir).resolve().parent / MARKER
    if mk.exists():
        return False, f"{mk} exists: the test half was exported already"
    return True, f"test half, final; tool md5 = frozen md5; no earlier test export; output folder new"


def run(peryear, split_ref, tools, out_dir, minlist=8, expect=(44, 57), half="fit", final_test=False, freeze=None,
        repeats=REPEATS, out=print):
    t0 = time.time()
    res, rep = {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    ok0, why = mode_ok(half, final_test, freeze, out_dir)
    if not ok0:
        sys.exit(f"STOP (J0): {why}")
    bdc, bmd5 = X.import_bdc(tools)
    me = X.file_md5(__file__)
    out(f"stage_h1_bnm_export.py md5 {me}; bnm_detection_check.py md5 {bmd5}; export_obj2.py {X.file_md5(HERE / 'export_obj2.py')}; "
        f"stage_e_inputs.py {X.file_md5(HERE / 'stage_e_inputs.py')}; half {half}; repeats {repeats}")
    gate("J0", True, why)
    pooled = Path(peryear) / f"minlist{minlist}"
    od = Path(out_dir)
    with LoadLog() as log:
        ref = bdc.split_ref(split_ref)
        names = sorted(f.name for f in pooled.glob("bnm_*.npz"))
        if not names:
            sys.exit(f"STOP: no window files in {pooled}")
        keep, east, overlap = bdc.fit_half(names, ref)
        fit_set = set(keep)
        test_set = {nm for nm in names if bdc.window_en(nm)[0] >= ref["cut"]}
        mine = keep if half == "fit" else sorted(test_set)
        if broken() == "testhalf":
            mine = names
        ydirs = sorted((int(p.name.rsplit("_y", 1)[1]), p) for p in Path(peryear).glob(f"minlist{minlist}_y*") if p.is_dir())
        a0 = np.load(pooled / mine[0], allow_pickle=True)
        ty = int(np.asarray(a0["target_year"]).item())
        years = [y for y, _ in ydirs] + [ty]
        consec = years == list(range(years[0], years[0] + len(years)))
        missing = [(y, nm) for y, d in ydirs for nm in mine if not (d / nm).is_file()]
        win = []                                          # (name, e, n, R, Ef, species, status)
        rows_ok, outside = True, 0
        if consec and len(years) == W and not missing:
            for nm in mine:
                R, Ef, sp, st, ok, o = read_window(pooled, ydirs, nm)
                rows_ok &= ok; outside += o
                if ok:
                    e, n = bdc.window_en(nm)
                    win.append((nm, e, n, R, Ef, sp, st))
    other = (test_set if half == "fit" else fit_set) | ({nm for nm in names} - fit_set - test_set)
    root = str(Path(peryear).resolve())
    touched = [p for p in log.paths if Path(p).name in other and str(Path(p).resolve()).startswith(root)]
    gate("J1", consec and len(years) == W and not missing and (len(keep), east) == tuple(expect) and overlap == 0 and rows_ok
         and outside == 0 and not touched,
         f"years {years}; {len(names)} windows: fit {len(keep)}, test {east}, sharing cells {overlap} (expected {expect[0]} and "
         f"{expect[1]}); {half} windows read {len(win)}; species rows and shapes equal {rows_ok}; records outside effort "
         f"{outside}; files passed to np.load {len(log.paths)}, of them windows of the other half {len(touched)}")
    if not win or not res["J1"]:
        n_ok = sum(res.values())
        out(f"stage H1: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after J1, nothing written")
        return res, rep
    # ---------------------------------------------------------------- holes and files
    od.mkdir(parents=True)
    manifest, mem = [], []                                # mem: per file (V, Ek, holes, keep6, idx) for J6 and gates
    j2_bad, j4b_bad, md5a = 0, 0, hashlib.md5()
    for wi, (nm, e, n, R, Ef, sp, st) in enumerate(win):
        k = np.nonzero(R.any((0, 2, 3)))[0]
        Rk, Ek0 = R[:, k].transpose(1, 0, 2, 3), Ef[:, k].transpose(1, 0, 2, 3)      # (m, 6, G, G)
        for r in range(repeats):
            H = np.stack([draw_holes(Ek0[j, -1], Rk[j, -1], e, n, int(s), r) for j, s in enumerate(k)]) if len(k) else \
                np.zeros((0, G, G), bool)
            Hz = np.stack([draw_holes(Ek0[j, -1], np.zeros((G, G), bool), e, n, int(s), r) for j, s in enumerate(k)]) \
                if len(k) else H
            j4b_bad += int((H != Hz).any((1, 2)).sum())
            md5a.update(np.packbits(H).tobytes())
            nsv = Ek0[:, -1].reshape(len(k), -1).sum(1)
            nh = H.reshape(len(k), -1).sum(1)
            j2_bad += int(((nh != np.array([(3 * int(v) + 5) // 10 if v >= 2 else 0 for v in nsv])) |
                           (H & ~Ek0[:, -1]).any((1, 2))).sum())
            H6 = H[:, None]
            V = Rk & ~H6 if broken() != "keep_earlier" else np.concatenate([Rk[:, :-1], Rk[:, -1:] & ~H6], 1)
            Ek = Ek0 & ~H6 if broken() != "keep_effort" else Ek0.copy()
            stem = f"{Path(nm).stem}_r{r + 1}"
            pair = hashlib.md5((nm + f"|{r + 1}|" + "|".join(sp[k])).encode()).hexdigest()
            inp = dict(code_md5=np.array(me), effort=X.pack(Ek), habitat_noisy=np.zeros((1, G, G), np.float16),
                       n_rec=V.reshape(len(k), W, -1).sum(2).astype(np.int16), pair_md5=np.array(pair),
                       stack_hab=np.zeros(len(k), np.int32), visit1=X.pack(V), visit2=X.pack(np.zeros_like(V)))
            if broken() == "hole_in_input":
                inp["hole"] = X.pack(H)
            tru = dict(code_md5=np.array(me), eff_orig=X.pack(Ek0), hole=X.pack(H), label_pooled=X.pack(H & Rk.any(1)),
                       label_strict=X.pack(H & Rk[:, -1]), pair_md5=np.array(pair), rec_orig=X.pack(Rk),
                       repeat=np.int32(r + 1), species=sp[k], status=st[k], window_name=np.array(nm),
                       years=np.array(years, np.int32))
            ip, tp = od / f"{stem}_input.npz", od / f"{stem}_truth.npz"
            X.save_npz(ip, inp); X.save_npz(tp, tru)
            manifest.append(dict(sim_seed=(e * 10000 + n) * 10 + r + 1, window=nm, repeat=r + 1, stacks=len(k),
                                 input_file=ip.name, input_md5=X.file_md5(ip), truth_file=tp.name, truth_md5=X.file_md5(tp)))
            mem.append(dict(wi=wi, r=r, k=k, V=V, Ek=Ek, H=H, Rk=Rk, Ek0=Ek0, nsv=nsv, nh=nh, e=e, n=n, sp=sp[k], st=st[k]))
    with open(od / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0])); wr.writeheader(); wr.writerows(manifest)
    hole_md5 = md5a.hexdigest()
    gate("J2", j2_bad == 0, f"{sum(len(m['k']) for m in mem):,} stack-repeats ({len(mem)} files); holes = (3n + 5) // 10 of the "
         f"2019 surveyed tetrads (none when n < 2), inside the 2019 effort: wrong {j2_bad}")
    # ---------------------------------------------------------------- J3 + J5: inputs read afresh under the truth guard
    j3_bad, keys_bad, fr = 0, 0, {k_: [] for k_ in ("file_index", "row", "rec_t", "rec6", "shift")}
    idx_rows = {k_: [] for k_ in ("window", "species", "status", "repeat", "row", "stack_id", "n_surv", "n_holes")}
    with D.TruthGuard() as guard:
        for fi, (row, m) in enumerate(zip(manifest, mem)):
            p = od / row["input_file"]
            if X.file_md5(p) != row["input_md5"]:
                j3_bad += 1
            with np.load(p, allow_pickle=False) as z:
                keys = set(z.files)
                v1, v2, ef, nr = X.unpack(z["visit1"]), X.unpack(z["visit2"]), X.unpack(z["effort"]), z["n_rec"]
            keys_bad += int(keys != set(X.ALLOW))
            H6 = m["H"][:, None]
            good = (not (v1 & H6).any()) and (not (ef & H6).any()) and np.array_equal(v1 & ~H6, m["Rk"] & ~H6) and \
                np.array_equal(ef & ~H6, m["Ek0"] & ~H6) and not v2.any() and \
                np.array_equal(nr, (v1 | v2).reshape(len(v1), W, -1).sum(2))
            j3_bad += int(not good)
            rec6 = (v1 | v2).any(1).reshape(len(v1), -1).sum(1)
            rec_t = (v1 | v2)[:, -1].reshape(len(v1), -1).sum(1)
            sel = np.nonzero((rec6 >= 1) & (m["nh"] >= 1))[0] if broken() != "zero_in_frozen" else np.nonzero(m["nh"] >= 1)[0]
            for j in sel:
                fr["file_index"].append(fi); fr["row"].append(j); fr["rec_t"].append(rec_t[j]); fr["rec6"].append(rec6[j])
                fr["shift"].append(shift_of(m["e"], m["n"], int(m["k"][j]), m["r"]))
                idx_rows["window"].append(win[m["wi"]][0]); idx_rows["species"].append(m["sp"][j])
                idx_rows["status"].append(m["st"][j]); idx_rows["repeat"].append(m["r"] + 1); idx_rows["row"].append(j)
                idx_rows["stack_id"].append(m["wi"] * 1000 + int(m["k"][j])); idx_rows["n_surv"].append(m["nsv"][j])
                idx_rows["n_holes"].append(m["nh"][j])
    gate("J3", j3_bad == 0 and keys_bad == 0, f"input files read afresh: md5 = manifest, records and effort 0 in the holes in "
         f"all {W} years, equal to the grids outside the holes, visit2 empty, n_rec = records: wrong files {j3_bad}; key set "
         f"other than the allow-list {keys_bad}")
    nfr = len(fr["row"])
    frozen = dict(files=np.array([r_["input_file"] for r_ in manifest]), file_md5=np.array([r_["input_md5"] for r_ in manifest]),
                  file_index=np.array(fr["file_index"], np.int64), row=np.array(fr["row"], np.int64),
                  rec_t=np.array(fr["rec_t"], np.int64), rec6=np.array(fr["rec6"], np.int64),
                  cls=D.tidx(np.array(fr["rec_t"], np.int64)).astype(np.int8), blank=np.ones(nfr, np.uint8),
                  shift=np.array(fr["shift"], np.int8).reshape(nfr, 2), fast=np.zeros(nfr, np.uint8))
    index = dict(window=np.array(idx_rows["window"]), species=np.array(idx_rows["species"]),
                 status=np.array(idx_rows["status"]), repeat=np.array(idx_rows["repeat"], np.int32),
                 row=np.array(idx_rows["row"], np.int64), stack_id=np.array(idx_rows["stack_id"], np.int64),
                 n_surv=np.array(idx_rows["n_surv"], np.int32), n_holes=np.array(idx_rows["n_holes"], np.int32))
    X.save_npz(od / "bnm_frozen.npz", frozen); X.save_npz(od / "bnm_stacks.npz", index)
    # ---------------------------------------------------------------- J4, J4b
    aucs = []
    if half == "fit":
        for m in mem:
            for j in range(len(m["k"])):
                s_ = m["Ek0"][j, -1]
                lab = m["Rk"][j, -1][s_]
                if 0 < lab.sum() < lab.size:
                    aucs.append(X.auc(m["H"][j][s_].astype(float), lab))
        ma = float(np.mean(aucs)) if aucs else float("nan")
        gate("J4", 0.48 <= ma <= 0.52, f"hole mask against 2019 records among 2019 surveyed tetrads: mean per-stack AUC "
             f"{ma:.4f} over {len(aucs):,} stack-repeats (rule 0.48-0.52)")
    gate("J4b", j4b_bad == 0, f"holes drawn again with every record set to 0: stack-repeats with a different mask {j4b_bad}")
    gate("J5", not guard.opened and keys_bad == 0 and all(r_["truth_file"].endswith("_truth.npz") for r_ in manifest),
         f"frozen set and index built from the input files with the truth guard on (truth files opened {len(guard.opened)}); "
         f"no truth key in an input file; truth files named *_truth.npz")
    # ---------------------------------------------------------------- J6 builder, three paths
    j6_bad, j6_err, n6 = 0, "", 0
    try:
        with D.TruthGuard() as g6:
            VA = E.Inputs(od, frozen["files"])
            off, blank = E.frozen_draws(frozen)
            fi_a, rw_a = frozen["file_index"], frozen["row"]
            for f_ in np.unique(fi_a):
                q = np.nonzero(fi_a == f_)[0]
                xa, na = VA.get(fi_a[q], rw_a[q], off[q], blank[q])
                xb, nb = E.build_from_file(od / frozen["files"][f_], rw_a[q], off[q], blank[q])
                m = mem[f_]
                Vc, Ec = m["Rk"][rw_a[q]] & ~m["H"][rw_a[q]][:, None], m["Ek0"][rw_a[q]] & ~m["H"][rw_a[q]][:, None]
                xc, nc = E.build(Vc, Ec, np.zeros((len(q), G, G), np.float32), off[q], blank[q])
                j6_bad += int(not (np.array_equal(xa, xb) and np.array_equal(xb, xc) and np.array_equal(na, nb)
                                   and np.array_equal(nb, nc)))
                n6 += len(q)
        j6_err = f"; truth files opened {len(g6.opened)}"
        j6_ok = j6_bad == 0 and not g6.opened
    except (ValueError, RuntimeError) as ex:
        j6_ok, j6_err = False, f"; builder refused: {ex}"
    gate("J6", j6_ok, f"tensors of {n6:,} frozen rows bit-identical by three paths (packed in memory, file read afresh, grids "
         f"with holes): files differing {j6_bad}{j6_err}")
    md5b = hashlib.md5()
    for m in mem:
        H2 = np.stack([draw_holes(m["Ek0"][j, -1], m["Rk"][j, -1], m["e"], m["n"], int(s), m["r"]) for j, s in enumerate(m["k"])]) \
            if len(m["k"]) else m["H"]
        md5b.update(np.packbits(H2).tobytes())
    gate("J7", md5b.hexdigest() == hole_md5, f"second hole draw: md5 {md5b.hexdigest()[:8]} = first {hole_md5[:8]}")
    # ---------------------------------------------------------------- report
    allk = [(m["wi"], int(s)) for m in mem for s in m["k"]]
    n_all = len(allk)
    kept6 = np.concatenate([m["V"].any(1).reshape(len(m["k"]), -1).sum(1) for m in mem])
    keptt = np.concatenate([m["V"][:, -1].reshape(len(m["k"]), -1).sum(1) for m in mem])
    nh_all = np.concatenate([m["nh"] for m in mem])
    sid = np.array([wi * 1000 + s for wi, s in allk])
    scored = (kept6 >= 1) & (nh_all >= 1)
    zero = (kept6 == 0) & (nh_all >= 1)
    nohole = nh_all == 0
    out(f"   stack-repeats {n_all:,} ({n_all // repeats:,} stacks x {repeats}); scored (a hole and a kept record) {int(scored.sum()):,}; "
        f"left out, no kept record {int(zero.sum()):,}; no hole (fewer than 2 surveyed tetrads in 2019) {int(nohole.sum()):,}")
    rep.update(stack_repeats=n_all, stacks=n_all // repeats, scored=int(scored.sum()), left_out_zero=int(zero.sum()),
               no_hole=int(nohole.sum()), frozen_rows=nfr, hole_md5=hole_md5)
    tab = {}
    if half == "fit":
        rec6_all = np.concatenate([m["Rk"].any(1).reshape(len(m["k"]), -1).sum(1) for m in mem])
        rec_t_all = np.concatenate([m["Rk"][:, -1].reshape(len(m["k"]), -1).sum(1) for m in mem])
        hr = np.concatenate([(m["H"] & m["Rk"][:, -1]).reshape(len(m["k"]), -1).sum(1) for m in mem])
        hu = np.concatenate([(m["H"] & ~m["Rk"][:, -1]).reshape(len(m["k"]), -1).sum(1) for m in mem])
        hp = np.concatenate([(m["H"] & m["Rk"].any(1)).reshape(len(m["k"]), -1).sum(1) for m in mem])
        rank = scored & (hr > 0) & (hu > 0)
        zb = Counter(D.bidx(rec6_all[zero]).tolist())
        out("   left out (no kept record), by records over 6 years (all records): " +
            "; ".join(f"{b} {zb.get(i, 0)}" for i, b in enumerate(BINS)))
        rep["left_out_by_records_6y"] = {b: int(zb.get(i, 0)) for i, b in enumerate(BINS)}
        for lab, gi, labels in (("kept recorded class at 2019", D.tidx(keptt), TCLS), ("kept records over 6 years", D.bidx(kept6), BINS)):
            rows_ = []
            for i, l in enumerate(labels):
                s_ = scored & (gi == i); r_ = rank & (gi == i)
                d_s, d_r = len(set(sid[s_])), len(set(sid[r_]))
                rows_.append(f"{l} {int(s_.sum()):,}/{d_s:,}/{int(r_.sum()):,}/{d_r:,}{'' if d_r >= FLOOR else ' (under 50)'}")
                tab.setdefault(lab, {})[l] = dict(scored=int(s_.sum()), scored_stacks=d_s, rankable=int(r_.sum()), rankable_stacks=d_r)
            out(f"   {lab} (scored / distinct stacks / rankable / rankable distinct stacks): " + "; ".join(rows_))
        sh = hr[scored].sum() / max(1, nh_all[scored].sum()); shp = hp[scored].sum() / max(1, nh_all[scored].sum())
        out(f"   rankable stack-repeats {int(rank.sum()):,} ({len(set(sid[rank])):,} distinct stacks); holes recorded in 2019 "
            f"(strict) {sh:.3f}, in any year (pooled) {shp:.3f}; stacks with 10+ records in 2019 (7 Nov version) "
            f"{len(set(sid[rec_t_all >= 10])):,}")
        rep.update(rankable=int(rank.sum()), rankable_stacks=len(set(sid[rank])), hole_share_strict=float(sh),
                   hole_share_pooled=float(shp), stacks_10plus_2019=len(set(sid[rec_t_all >= 10])), groups=tab,
                   j4_mean_auc=float(np.mean(aucs)) if aucs else None, j4_n=len(aucs))
    else:
        for lab, gi, labels in (("kept recorded class at 2019", D.tidx(keptt), TCLS), ("kept records over 6 years", D.bidx(kept6), BINS)):
            for i, l in enumerate(labels):
                s_ = scored & (gi == i)
                tab.setdefault(lab, {})[l] = dict(scored=int(s_.sum()), scored_stacks=len(set(sid[s_])))
            out(f"   {lab} (scored / distinct stacks): " + "; ".join(f"{l} {v['scored']:,}/{v['scored_stacks']:,}" for l, v in tab[lab].items()))
        rep.update(groups=tab)
    stat = np.concatenate([m["st"] for m in mem])
    spc = np.concatenate([m["sp"] for m in mem])
    st_s = {}
    for s_, sp_, i_ in zip(stat[scored], spc[scored], sid[scored]):
        st_s.setdefault(s_, [set(), set()]); st_s[s_][0].add(i_); st_s[s_][1].add(sp_)
    out("   GB Red List 2022 status of scored stacks (distinct stacks / species): " + "; ".join(
        f"{k_ or '(none)'} {len(v[0]):,} / {len(v[1])}" for k_, v in sorted(st_s.items(), key=lambda kv: -len(kv[1][0]))))
    for k_ in ("RE", "EX"):
        if k_ in st_s:
            ws = sorted({win[i_ // 1000][0] for i_ in st_s[k_][0]})
            out(f"   status {k_}: species {sorted(map(str, st_s[k_][1]))} in windows {ws}")
    rep["status"] = {str(k_): dict(stacks=len(v[0]), species=sorted(map(str, v[1]))) for k_, v in st_s.items()}
    if half == "test":
        (od.resolve().parent / MARKER).write_text(json.dumps(dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                                  tool_md5=me, out=str(od.resolve()))) + "\n")
    md5s = {nm_: X.file_md5(od / nm_) for nm_ in ("bnm_frozen.npz", "bnm_stacks.npz")}
    md5s["manifest.csv"] = X.file_md5(od / "manifest.csv")
    n_ok = sum(res.values())
    rep.update(gates=res, md5=dict(tool=me, bdc=bmd5, **md5s), half=half, years=years, minutes=round((time.time() - t0) / 60, 2))
    (od / "h1_report.json").write_text(json.dumps(rep, indent=1, default=int))
    out("   " + ", ".join(f"{k_} {v[:8]}" for k_, v in md5s.items()))
    out(f"stage H1 ({half} half): gates {n_ok} PASS, {len(res) - n_ok} FAIL; {len(win)} windows, {len(manifest)} files, "
        f"{nfr:,} frozen rows ({(time.time() - t0) / 60:.1f} min)")
    return res, rep


# ---------------------------------------------------------------- self-test
def world(tmp):
    """Planted grids (export_obj2.plant_grids: 12 windows, fit half 6) with GB Red List status and six planted species
    per window: sp00 one record (2019), sp01 one record (2016, a tetrad surveyed in 2016 and 2019), sp02 two records
    (2019), sp03 no record, sp04 one surveyed tetrad in 2019 and one record (2015), sp05 15 surveyed tetrads in 2019."""
    py, refpat = X.plant_grids(tmp)
    py = Path(py)
    ydirs = sorted((int(p.name.rsplit("_y", 1)[1]), p) for p in py.glob("minlist8_y*"))
    for f in sorted((py / "minlist8").glob("bnm_*.npz")):
        a = dict(np.load(f)); S = len(a["species"])
        Z = [dict(np.load(d / f.name)) for _, d in ydirs]
        R = [z["present_year"].astype(bool) for z in Z] + [a["obs_mask"].astype(bool)]
        Ef = [z["known_holdout"].astype(bool) for z in Z] + [a["known_target"].astype(bool)]
        for s in range(6):
            for t in range(W):
                R[t][s] = False
        both = np.flatnonzero(Ef[2][1] & Ef[5][1]); R[2][1].flat[both[0]] = True
        R[5][0].flat[np.flatnonzero(Ef[5][0])[0]] = True
        i2 = np.flatnonzero(Ef[5][2]); R[5][2].flat[i2[0]] = True; R[5][2].flat[i2[1]] = True
        one = np.flatnonzero(Ef[5][4])[0]; Ef[5][4][:] = False; Ef[5][4].flat[one] = True
        R[1][4].flat[np.flatnonzero(Ef[1][4])[0]] = True
        i5 = np.flatnonzero(Ef[5][5])[:15]; Ef[5][5][:] = False; Ef[5][5].flat[i5] = True; R[5][5].flat[i5[:4]] = True
        for t, (y, d) in enumerate(ydirs):
            Z[t]["present_year"] = R[t].astype(np.uint8); Z[t]["known_holdout"] = Ef[t].astype(np.uint8)
            np.savez(d / f.name, **Z[t])
        a["obs_mask"] = R[5].astype(np.uint8); a["known_target"] = Ef[5].astype(np.uint8)
        a["status"] = np.array(["LC", "NT", "VU", "EN", "RE", ""] * (S // 6 + 1))[:S]
        np.savez(f, **a)
    return str(py), refpat


def self_test(tools):
    res_all = []
    quiet = lambda *a, **k: None

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    def bug(name, tmp, py, refpat):
        os.environ["STAGE_H1_BROKEN"] = name
        try:
            r, _ = run(py, refpat, tools, Path(tmp) / f"bug_{name}", expect=(6, 6), out=quiet)
        finally:
            os.environ.pop("STAGE_H1_BROKEN", None)
        return r

    with tempfile.TemporaryDirectory() as tmp:
        py, refpat = world(Path(tmp) / "w")
        o1 = Path(tmp) / "data" / "bnm_fit"
        r, rep = run(py, refpat, tools, o1, expect=(6, 6), out=quiet)
        # independent counts from the planted files
        n_ind, holes_ok, zero_ind = 0, True, 0
        fitw = [f for f in sorted((Path(py) / "minlist8").glob("bnm_*.npz")) if int(f.stem.split("_e")[1][:4]) < 120]
        for f in fitw:
            Rs = np.stack([np.load(Path(py) / f"minlist8_y{y}" / f.name)["present_year"] for y in range(2014, 2019)]
                          + [np.load(f)["obs_mask"]]).astype(bool)
            k = np.nonzero(Rs.any((0, 2, 3)))[0]
            n_ind += len(k)
            nsv = np.load(f)["known_target"].astype(bool)[k].reshape(len(k), -1).sum(1)
            for rr in range(1, REPEATS + 1):
                z = np.load(o1 / f"{f.stem}_r{rr}_truth.npz")
                h = X.unpack(z["hole"]).reshape(len(k), -1).sum(1)
                holes_ok &= np.array_equal(h, [(3 * v + 5) // 10 if v >= 2 else 0 for v in nsv])
                ro = X.unpack(z["rec_orig"]); ho = X.unpack(z["hole"])
                zero_ind += int((~(ro & ~ho[:, None]).any((1, 2, 3))).sum())

        def holes_of(f, name):
            z = np.load(o1 / f"{f.stem}_r1_truth.npz")
            return int(X.unpack(z["hole"])[list(z["species"]).index(name)].sum())
        sp5 = [holes_of(f, "sp05") for f in fitw]
        sp4 = [holes_of(f, "sp04") for f in fitw]
        t("T1", all(r.values()) and set(r) == {"J0", "J1", "J2", "J3", "J4", "J4b", "J5", "J6", "J7"} and rep["stacks"] == n_ind
          and holes_ok, f"planted grids (6 fit-half windows): every gate PASS; stacks {rep['stacks']} = independent count {n_ind}; "
          f"holes = (3n + 5) // 10 in every stack-repeat by an independent count")
        t("T2", set(sp5) == {5} and set(sp4) == {0} and rep["no_hole"] == len(fitw) * REPEATS,
          f"15 surveyed tetrads give 5 holes (numpy's round would give 4): {sorted(set(sp5))}; 1 surveyed tetrad gives none and "
          f"is counted apart: {rep['no_hole']}")
        fz = np.load(o1 / "bnm_frozen.npz")
        t("T3", rep["left_out_zero"] == zero_ind and rep["left_out_zero"] > 0 and rep["frozen_rows"] == rep["scored"]
          and (fz["rec6"] >= 1).all() and set(fz.files) == set(FROZEN_KEYS) and (fz["blank"] == 1).all(),
          f"stack-repeats whose holes take every record: {rep['left_out_zero']} = independent count {zero_ind}, none in the "
          f"frozen set; frozen keys = stage D's; blank 1 everywhere")
        o2 = Path(tmp) / "data2" / "bnm_fit"
        run(py, refpat, tools, o2, expect=(6, 6), out=quiet)
        same = all(X.file_md5(p) == X.file_md5(o2 / p.name) for p in sorted(o1.glob("*.npz"))) and \
            X.file_md5(o1 / "manifest.csv") == X.file_md5(o2 / "manifest.csv")
        t("T4", same, "a second run into a new folder gives the same md5 for every file")
        try:
            run(py, refpat, tools, o1, expect=(6, 6), out=quiet); stopped = False
        except SystemExit:
            stopped = True
        t("T5", stopped, "a run into an existing folder stops (J0)")
        # the test half: refused without --final-test, without a matching freeze; runs once with them, no label read
        ot = Path(tmp) / "data" / "bnm_test"
        stops = []
        for kw in (dict(half="test"), dict(half="test", final_test=True),
                   dict(half="test", final_test=True, freeze=str(Path(tmp) / "fz_bad.json"))):
            Path(tmp, "fz_bad.json").write_text(json.dumps(dict(tools={Path(__file__).name: "0" * 32})))
            try:
                run(py, refpat, tools, ot, expect=(6, 6), out=quiet, **kw); stops.append(False)
            except SystemExit:
                stops.append(True)
        fzp = Path(tmp) / "fz.json"; fzp.write_text(json.dumps(dict(tools={Path(__file__).name: X.file_md5(__file__)})))
        rt, rept = run(py, refpat, tools, ot, expect=(6, 6), half="test", final_test=True, freeze=str(fzp), out=quiet)
        try:
            run(py, refpat, tools, Path(tmp) / "data" / "bnm_test2", expect=(6, 6), half="test", final_test=True,
                freeze=str(fzp), out=quiet); again = False
        except SystemExit:
            again = True
        t("T6", all(stops) and all(rt.values()) and "J4" not in rt and "rankable" not in rept and again
          and (Path(tmp) / "data" / MARKER).is_file(),
          "test half: refused without --final-test, without a freeze file and with a wrong md5; with them it passes J0-J3, "
          "J4b-J7, skips J4, reads no hole label for the report, writes the marker, and a second test export stops")
        rb = bug("holes_on_records", tmp, py, refpat)
        t("T7", rb["J4"] is False and rb["J4b"] is False, "holes drawn more often on recorded tetrads fail J4 and J4b")
        rb = bug("keep_effort", tmp, py, refpat)
        t("T8", rb["J3"] is False, "effort left in the holes fails J3")
        rb = bug("keep_earlier", tmp, py, refpat)
        t("T9", rb["J3"] is False, "records of 2014-2018 left in the holes fail J3")
        rb = bug("round_even", tmp, py, refpat)
        t("T10", rb["J2"] is False, "numpy's round to even (15 tetrads -> 4 holes) fails J2")
        rb = bug("testhalf", tmp, py, refpat)
        t("T11", rb["J1"] is False and not (Path(tmp) / "bug_testhalf").exists(), "a version that opens test-half windows fails J1 "
          "and writes nothing")
        rb = bug("hole_in_input", tmp, py, refpat)
        t("T12", rb["J3"] is False and rb["J5"] is False, "a hole mask written into the input file fails J3 and J5")
        rb = bug("zero_in_frozen", tmp, py, refpat)
        t("T13", rb["J6"] is False, "a stack-repeat with no kept record in the frozen set is refused by the builder (J6)")
        j = json.loads((o1 / "h1_report.json").read_text())
        t("T14", all(j["gates"].values()) and j["md5"]["tool"] == X.file_md5(__file__) and
          sum(v["scored"] for v in j["groups"]["kept recorded class at 2019"].values()) == j["scored"] and
          sum(v["stacks"] for v in j["status"].values()) == len(set(np.load(o1 / "bnm_stacks.npz")["stack_id"].tolist())),
          "the json holds the gates, the tool md5, group counts that add up to the scored count, and the Red List table")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--split-ref", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--minlist", type=int, default=8)
    ap.add_argument("--expect", type=int, nargs=2, default=[44, 57])
    ap.add_argument("--out", default="../data/bnm_fit")
    ap.add_argument("--half", choices=("fit", "test"), default="fit")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--freeze")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    r, _ = run(a.peryear, a.split_ref, a.obj1_tools, a.out, a.minlist, tuple(a.expect), a.half, a.final_test, a.freeze)
    sys.exit(0 if all(r.values()) else 1)