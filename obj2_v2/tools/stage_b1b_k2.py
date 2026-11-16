#!/usr/bin/env python3
"""
stage_b1b_k2.py  (obj2_v2, habitat check B-1b, 16 Nov 2026; rule in claude/obj2_v2_H_rules_part2.md, signed by Sara 16 Nov
(M3) after M1 and M2; report only)

Question: on the BNM fit half (development data), do hidden recorded tetrads sit closer on the B-1a habitat axis to the
species' kept records than hidden unrecorded tetrads do, and more so with the window's own map than with another window's
map? Nothing is fitted (no parameter, no folds). The model, its samples, h3_freeze.json, the P4 fits and every product stay
as they are. Nothing of the test half, H5, H6, real_eval or a real_val truth file is read.

Score (rule, as signed)
    Items: H2b's fit-half stack-repeats and hole tetrads; labels by H2b's read_labels (hole recorded in 2019); kept records
        from the inputs (visit1 | visit2, any of the 6 years); groups and windows as H2b.
    Far holes: H2a's distance to the nearest kept record >= 3 (tolerance 1e-4); near holes are not scored.
    Maps: own arm = B-1a z of the item's window; control arm = B-1a z of window (k + 22) mod 44 (b1a control_index).
    Support (M1): a far hole is kept when it is scored in both maps; the kept-record set is the kept-record tetrads scored in
        both maps (each tetrad once); both arms use the same holes and records. No kept record left: item dropped.
    s = -|z(hole) - mean z over the kept-record set| (minus the model's channel 20 built from the map). AUC per item over its
        kept far holes, recorded against unrecorded, ties half; d = AUC(own) - AUC(control).
    Groups: kept recorded class at 2019 and kept records over 6 years (H-R1); sparse set S = class 0-1 or 1-10 records.
        Stack-first means, 95% intervals from 2,000 resamples of the windows (seed 31; H2b's hr1_core), floor 50 stacks.
    Rule: B-1 PASS when, in S, the mean of d has its 95% interval above 0 (lower end > 0) with 50 or more distinct stacks.
        Otherwise FAIL (B stops). Sara decides either way.
    real_val reference (report only, inputs only): the habitat-shown stacks of stage_d_frozen_val.npz (blank 0); 5 repeats
        of H1's hole rule ((3n + 5) // 10 of the n tetrads surveyed at the target year, none when n < 2, from effort only,
        seed key 31401 with frozen row and repeat); records and effort removed in the holes in all 6 years; label = the input
        record at the target year; own map = stored noisy habitat, control = stage D's partner rule among the shown stacks;
        every cell scored; resampling over the communities.

Gates (if one fails, nothing past it is printed and nothing is written)
    W0  inputs: b1a_habitat.npz and b1a_report.json md5 = step 54, its gates PASS; h3_freeze.json = signed and its md5 list =
        the files now; h2b_report.json gates PASS; stage_f0_spread_check.py version 2; stage_d_frozen_val.npz md5 = P4-1's
        record; every real_val input md5 = the frozen set's record and the manifest
    W1  alignment: b1a windows = the fit-half windows of bnm_stacks; control index = (k + n/2) mod n, never the window itself;
        recorded holes per item = h2b y_strict; distance from the kept records = h2b dist_h (1e-5); kept-record tetrads off
        L_w at most 0.5% (the share with the map flipped north-south printed beside)
    W2  code: AUC by a pair loop on the items of 3 windows = the tool's (1e-12); a positive scale and shift of every map leaves
        every AUC unchanged (1e-12); control = own gives d = 0; a constant map gives 0.5 at every rankable item; real_val
        own-arm scores = minus stage_g_baselines hab_dist (1e-9)
    W3  support: both arms use the same holes and records on every item; every kept hole is scored in both maps and far
    W4  real_val holes: counts = (3n + 5) // 10, inside the target-year effort, distinct; the same with every record set to 0;
        no stack's partner in its own community
    W5  np.load log: only the BNM fit, H2a, H2b and B-1a folders, the frozen set and real_val input files
    W6  outputs written once into a new folder and read back equal

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_b1b_k2.py --self-test 2>&1 | tail -n 1
    python3 -u ../tools/stage_b1b_k2.py > ../logs/stage_b1b_k2.txt 2>&1
Needs stage_f0_spread_check.py (version 2) and what it loads in the same folder; the self-test lives in
../tests/test_37_b1b_k2.py and uses ../tests/test_25_f0a_spread_check.py and ../tests/test_23_h6_test_score.py.
"""
import argparse, hashlib, importlib.util, json, os, sys, time, warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_b1b_k2.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


F0 = _load("stage_f0_spread_check")
H6, S2, X, D = F0.H6, F0.S2, F0.X, F0.D
GB, HR = S2.GB, S2.HR
G = X.G
NC = G * G
TCLS, BINS = list(F0.TCLS), list(F0.BINS)
F0_FP = "3599c46ba15995431b8eaba743f07660"          # content fingerprint of stage_f0_spread_check.py version 2
B1A_MD5 = dict(habitat="fac95acff9353bbc583c081df90a9b43", report="71e2bedb034d72e3ef240330a9dcae23")   # Sara's run, step 54
KEY_RV, REPEATS = 31401, 5
FAR, FAR_TOL, OFF_MAX = 3.0, 1e-4, 0.005
FORBID = tuple(F0.FORBID)


def broken():
    return os.environ.get("STAGE_B1B_BROKEN", "")    # self-test only: plants one bug so its gate must fail


def fingerprint(p):
    """md5 of the file without a #! line, trailing spaces or final newlines (the block fingerprint)."""
    b = Path(p).read_bytes()
    return hashlib.md5(b"\n".join(l.rstrip() for l in b.split(b"\n") if l[:2] != b"#!").rstrip(b"\n")).hexdigest()


class LoadLog:
    """Every np.load path is logged; the test half, H5, H6, real_eval and real_val truth files are refused before opening."""
    def __init__(self):
        self.paths = []

    def __enter__(self):
        self.orig = np.load
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        if isinstance(f, (str, Path)):
            p = str(Path(f).resolve())
            self.paths.append(p)
            if any(t in p for t in FORBID) or ("/real_val/" in p and p.endswith("_truth.npz")):
                raise RuntimeError(f"refused np.load of {p}")
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- pieces
def far_of(dist):
    return dist >= (FAR - FAR_TOL if broken() != "near" else 0.0)


def k2_arms(ptr, cell, lab, far, R, Z, SC, ia, ib, support=True):
    """Both arms on every item. R (n, NC) kept records (any year); Z (M, NC) maps; SC (M, NC) scored; item i uses map ia[i]
    (own) and ib[i] (control). Returns per-item AUCs, per-hole scores and keep masks, and counts."""
    n = len(ptr) - 1
    o = dict(auc_a=np.full(n, np.nan), auc_b=np.full(n, np.nan), s_a=np.full(len(cell), np.nan), s_b=np.full(len(cell), np.nan),
             keep_a=np.zeros(len(cell), bool), keep_b=np.zeros(len(cell), bool), same_rec=np.ones(n, bool),
             n_far=np.zeros(n, int), n_keep=np.zeros(n, int), n_rec=R.sum(1).astype(int), n_rec_keep=np.zeros(n, int),
             dropped=np.zeros(n, bool))
    for i in range(n):
        a, b = ptr[i], ptr[i + 1]
        c, f, y = cell[a:b], far[a:b], lab[a:b]
        za, zb = Z[ia[i]], Z[ib[i]]
        sup = (SC[ia[i]] & SC[ib[i]]) if support else np.ones(NC, bool)
        sup_b = np.ones(NC, bool) if broken() == "one_arm" else sup
        ka, kb = f & sup[c], f & sup_b[c]
        ra, rb = R[i] & sup, R[i] & sup_b
        o["n_far"][i], o["n_keep"][i], o["n_rec_keep"][i] = int(f.sum()), int(ka.sum()), int(ra.sum())
        o["same_rec"][i] = bool(np.array_equal(ra, rb))
        if not ra.any() or not rb.any():
            o["dropped"][i] = True
            continue
        sa = -np.abs(za[c[ka]] - za[ra].mean())
        sb = -np.abs(zb[c[kb]] - zb[rb].mean())
        o["s_a"][a:b][ka], o["s_b"][a:b][kb] = sa, sb
        o["keep_a"][a:b], o["keep_b"][a:b] = ka, kb
        o["auc_a"][i] = HR.stack_auc(sa, y[ka])
        o["auc_b"][i] = HR.stack_auc(sb, y[kb])
    return o


def brute_auc(s, y):
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def group_rows(v, stid, gi, win, labels, floor):
    """Stack-first means and window-resampled 95% intervals per group (H2b's hr1_core / hr1_rows), without H-R1 verdicts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # empty groups: class_boot's balanced mean of nothing
        rows = S2.hr1_rows(S2.hr1_core(v, stid, gi, win, list(range(len(labels)))), labels, floor)
    return [{k: q[k] for k in ("group", "n", "mean", "lo", "hi", "tested")} for q in rows]


def s_row(v, stid, sparse, win, floor):
    return group_rows(v, stid, np.where(sparse, 0, -1), win, ["S"], floor)[0]


def fmt(q):
    return f"{q['group']} {q['mean']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}] ({q['n']})" if q["n"] else f"{q['group']} none"


def draw_holes(Et, Vt, frow):
    """real_val holes by H1's rule from effort only: per stack i and repeat r, (3n + 5) // 10 of the n tetrads surveyed at
    the target year (none when n < 2), seed (31401, frozen row, repeat). Returns a list over (i, r) of sorted cell arrays."""
    out = []
    for i in range(len(Et)):
        cells = np.flatnonzero(Et[i] & ~Vt[i]) if broken() == "rv_holes_rec" else np.flatnonzero(Et[i])
        k = (3 * len(cells) + 5) // 10 if len(cells) >= 2 else 0
        for r in range(1, REPEATS + 1):
            g = X.rng(KEY_RV, int(frow[i]), r)
            out.append(np.sort(g.choice(cells, k, replace=False)) if k else np.zeros(0, np.int64))
    return out


def realval_items(V, Ef, H, comm, frow):
    """Items of the real_val reference: holes, labels, kept records, groups, own and partner maps (z per stack as channel 20)."""
    ns = len(V)
    Vf, Ett = V.reshape(ns, V.shape[1], NC), Ef[:, -1].reshape(ns, NC)
    holes = draw_holes(Ett, Vf[:, -1], frow)
    Hf = H.astype(np.float32).astype(np.float64).reshape(ns, NC)
    mu, sd = Hf.mean(1), Hf.std(1); sd[sd == 0] = 1.0
    Z = (Hf - mu[:, None]) / sd[:, None]
    part = HR.partner(comm)
    keep, cells, labs, Rk, Vk_list, stk, rep = [], [], [], [], [], [], []
    for j, hc in enumerate(holes):
        i, r = divmod(j, REPEATS)
        if not len(hc):
            continue
        Vk = Vf[i].copy(); Vk[:, hc] = False
        if not Vk.any():
            continue
        cells.append(hc); labs.append(Vf[i, -1, hc]); Rk.append(Vk.any(0)); Vk_list.append(Vk); stk.append(i); rep.append(r + 1)
    n = len(cells)
    ptr = np.r_[0, np.cumsum([len(c) for c in cells])].astype(np.int64)
    Vk = np.stack(Vk_list).reshape(n, V.shape[1], G, G) if n else np.zeros((0, V.shape[1], G, G), bool)
    dist = np.concatenate([GB.near_dist(Vk[a:a + 2000]).reshape(-1, NC) for a in range(0, n, 2000)]) if n else np.zeros((0, NC))
    stk = np.array(stk, int)
    cell = np.concatenate(cells) if n else np.zeros(0, int)
    item = np.repeat(np.arange(n), np.diff(ptr))
    return dict(n=n, ptr=ptr, cell=cell, lab=np.concatenate(labs).astype(bool) if n else np.zeros(0, bool), R=np.array(Rk).reshape(n, NC),
                Vk=Vk, dist_h=dist[item, cell] if n else np.zeros(0), stk=stk, rep=np.array(rep, int), Z=Z, part=part,
                rcls=D.tidx(Vk[:, -1].reshape(n, -1).sum(1)), rbin=D.bidx(np.array(Rk).reshape(n, NC).sum(1)),
                holes=holes, Ett=Ett, Vt=Vf[:, -1], ns=ns, n_left=len(holes) - n)


# ---------------------------------------------------------------- run
def run(bnm="../data/bnm_fit", h2a="../data/stage_h2_predict", h2b="../data/stage_h2b_score", b1a="../data/stage_b1a_habitat",
        freeze="../data/stage_h3_freeze/h3_freeze.json", runs=("../runs/f_seed1", "../runs/f_seed2"), obj1_tools="../../obj1_v2/tools",
        g2="../data/stage_g2_baselines_v2", g3="../data/stage_g3_calibration", data="../data", p41="../data/stage_p4_1",
        out_dir="../data/stage_b1b_k2", freeze_md5=F0.FREEZE_MD5, h6_md5=F0.H6_MD5, f0_fp=F0_FP, b1a_md5=B1A_MD5,
        frozen_md5=None, floor_override=None, out=print):
    bnm, h2a, h2b, b1a, freeze, data, out_dir = (Path(p) for p in (bnm, h2a, h2b, b1a, freeze, data, out_dir))
    runs, rv, fzp = [Path(r) for r in runs], data / "real_val", data / "stage_d_frozen_val.npz"
    o = out_dir.resolve()
    for p_ in (bnm, h2a, h2b, b1a, rv):
        if o == p_.resolve() or p_.resolve() in o.parents:
            sys.exit(f"STOP: nothing is written inside {p_}; give another --out")
    if any(t in str(o) for t in FORBID):
        sys.exit("STOP: the output folder may not lie in a test-half, H5, H6 or real_eval folder")
    if out_dir.exists():
        sys.exit(f"STOP: {out_dir} exists; give a new folder")
    t0, res, rep = time.time(), {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage B-1b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, rep

    out(f"stage_b1b_k2.py md5 {X.file_md5(__file__)} (fingerprint {fingerprint(__file__)}); stage_f0_spread_check.py "
        f"{X.file_md5(HERE / 'stage_f0_spread_check.py')}; stage_h2_bnm_score.py {X.file_md5(HERE / 'stage_h2_bnm_score.py')}; "
        f"stage_g_baselines.py {X.file_md5(HERE / 'stage_g_baselines.py')}; stage_d_headroom.py {X.file_md5(HERE / 'stage_d_headroom.py')}")
    # ---- W0
    for f_ in (b1a / "b1a_habitat.npz", b1a / "b1a_report.json", freeze, h2b / "h2b_report.json", fzp):
        if not f_.is_file():
            sys.exit(f"STOP: {f_} not found")
    m_hab, m_rep = X.file_md5(b1a / "b1a_habitat.npz"), X.file_md5(b1a / "b1a_report.json")
    rj = json.loads((b1a / "b1a_report.json").read_text())
    b_ok = m_hab == b1a_md5["habitat"] and m_rep == b1a_md5["report"] and bool(rj.get("gates")) and all(rj["gates"].values()) \
        and rj.get("md5", {}).get("habitat") == m_hab
    fz = json.loads(freeze.read_text())
    fmd5 = X.file_md5(freeze)
    paths = dict(bnm_fit=bnm, stage_h2_predict=h2a, stage_h2b_score=h2b, stage_g2_baselines_v2=Path(g2), stage_g3_calibration=Path(g3))
    bad, nf = H6.freeze_check(fz, obj1_tools, paths, runs)
    jb = json.loads((h2b / "h2b_report.json").read_text())
    jb_ok = bool(jb.get("gates")) and all(jb["gates"].values())
    f0ok = fingerprint(HERE / "stage_f0_spread_check.py") == f0_fp
    if frozen_md5 is None:
        frozen_md5 = json.loads((Path(p41) / "p4_1_report.json").read_text())["md5"]["frozen"]
    m_fz = X.file_md5(fzp)
    with LoadLog() as log:
        fzv = dict(np.load(fzp))
    man = {r_["input_file"]: r_ for r_ in D.rows_of(rv)}
    rv_bad = [str(f_) for f_, m_ in zip(fzv["files"], fzv["file_md5"])
              if not (rv / str(f_)).is_file() or X.file_md5(rv / str(f_)) != m_ or man.get(str(f_), {}).get("input_md5") != m_]
    floor = int(floor_override) if floor_override is not None else int(fz.get("settings", {}).get("floor", 50))
    gate("W0", b_ok and fmd5 == freeze_md5 and not bad and jb_ok and f0ok and m_fz == frozen_md5 and not rv_bad,
         f"b1a_habitat.npz {m_hab[:8]}, b1a_report.json {m_rep[:8]} = step 54 and its gates PASS {b_ok}; h3_freeze.json = signed "
         f"{fmd5 == freeze_md5}, {nf} md5s listed = the files now (differ: {bad}); h2b_report.json gates PASS {jb_ok}; "
         f"stage_f0_spread_check.py = version 2 {f0ok}; stage_d_frozen_val.npz {m_fz[:8]} = P4-1's record {m_fz == frozen_md5}; "
         f"{len(fzv['files'])} real_val input files md5 = frozen set and manifest {not rv_bad}; rule floor {floor} distinct stacks"
         + (" (override, self-test only)" if floor_override is not None else ""))
    if not res["W0"]:
        return stop("W0")
    # ---- load (np.load logged)
    with log:
        try:
            if broken() == "test_path":
                np.load(data / "bnm_test" / "bnm_frozen.npz")
            if broken() == "rv_truth":
                np.load(rv / str(fzv["files"][0]).replace("_input", "_truth"))
            B = dict(np.load(b1a / "b1a_habitat.npz"))
            F = F0.load_fit(bnm, h2a, h2b)
            with np.load(h2b / "h2b_predicted.npz") as P_:
                cell = P_["hole_cell"].astype(np.int64)
            fb = dict(np.load(bnm / "bnm_frozen.npz"))
            V1, V2, _, _, _ = GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])
            shown = np.flatnonzero(fzv["blank"] == 0)
            RV1, RV2, REf, RH, Rcomm = GB.load_obs(rv, fzv["files"], fzv["file_index"][shown], fzv["row"][shown])
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("W5", False, f"np.load log: {crashed}")
        return stop("W5")
    if broken() == "labels_shift":
        F["lab"] = np.roll(F["lab"], 1)
    names = B["window_names"].astype(str)
    nw = len(names)
    Z, SC, L = B["z"].reshape(nw, NC).astype(np.float64), B["scored"].reshape(nw, NC).astype(bool), B["land"].reshape(nw, NC).astype(bool)
    ctrl = B["control_index"].astype(int)
    if broken() == "self_ctrl":
        ctrl = np.arange(nw)
    if broken() == "flip":
        Z, SC, L = (A.reshape(nw, G, G)[:, ::-1].reshape(nw, NC).copy() for A in (Z, SC, L))
    # ---- W1
    n, ptr, lab, win = F["n"], F["ptr"], F["lab"], F["win"]
    widx = np.searchsorted(names, win)
    win_ok = sorted(np.unique(win).tolist()) == names.tolist()
    ctrl_ok = np.array_equal(ctrl, (np.arange(nw) + nw // 2) % nw) and not (ctrl == np.arange(nw)).any()
    y = np.add.reduceat(lab.astype(int), ptr[:-1]).astype(float)
    y_ok = np.array_equal(y, F["y_strict"])
    V = V1 | V2
    del V1, V2
    R = V.any(1).reshape(n, NC)
    dist_full = GB.near_dist(V).reshape(n, NC)
    item = F["item"]
    dgap = float(np.abs(dist_full[item, cell] - F["dist"]).max())
    Lw = L[widx] if win_ok else np.zeros_like(R)
    off = float((R & ~Lw).sum() / max(R.sum(), 1))
    Lf = L.reshape(nw, G, G)[:, ::-1].reshape(nw, NC)
    off_flip = float((R & ~Lf[widx]).sum() / max(R.sum(), 1)) if win_ok else float("nan")
    gate("W1", win_ok and ctrl_ok and y_ok and dgap <= 1e-5 and off <= OFF_MAX,
         f"{n:,} fit-half stack-repeats, {len(lab):,} hole tetrads, {nw} windows; b1a windows = bnm_stacks windows {win_ok}; control "
         f"index = (k + {nw // 2}) mod {nw}, never itself {ctrl_ok}; recorded holes per item = h2b y_strict {y_ok}; distance from "
         f"the kept records = h2b dist_h, largest gap {dgap:.1e}; kept-record tetrads off L_w {off:.4%} (rule <= {OFF_MAX:.1%}; "
         f"map flipped north-south {off_flip:.2%})")
    if not res["W1"]:
        return stop("W1")
    # ---- BNM arms
    far = far_of(F["dist"])
    ia, ib = widx, ctrl[widx]
    A = k2_arms(ptr, cell, lab, far, R, Z, SC, ia, ib)
    d = A["auc_a"] - A["auc_b"]
    # ---- real_val items
    RV = RV1 | RV2
    del RV1, RV2
    frow = shown
    I = realval_items(RV, REf, RH, Rcomm, frow)
    far_r = far_of(I["dist_h"])
    allsc = np.ones((I["ns"], NC), bool)
    AR = k2_arms(I["ptr"], I["cell"], I["lab"], far_r, I["R"], I["Z"], allsc, I["stk"], I["part"][I["stk"]])
    dR = AR["auc_a"] - AR["auc_b"]
    # ---- W2
    pick = sorted({0, nw // 2, nw - 1})
    ok_loop, worst = True, 0.0
    for i in np.flatnonzero(np.isin(widx, pick)):
        a, b = ptr[i], ptr[i + 1]
        for k_, sk, kk_ in (("auc_a", "s_a", "keep_a"), ("auc_b", "s_b", "keep_b")):
            km = A[kk_][a:b]
            v_ = brute_auc(A[sk][a:b][km], lab[a:b][km]) if not A["dropped"][i] else np.nan
            if np.isnan(v_) != np.isnan(A[k_][i]):
                ok_loop = False
            elif not np.isnan(v_):
                worst = max(worst, abs(v_ - A[k_][i]))
    As = k2_arms(ptr, cell, lab, far, R, 2.5 * Z + 1.3, SC, ia, ib)
    sc_gap = max(float(np.nanmax(np.abs(As["auc_a"] - A["auc_a"]), initial=0)), float(np.nanmax(np.abs(As["auc_b"] - A["auc_b"]), initial=0)))
    sc_nan = np.array_equal(np.isnan(As["auc_a"]), np.isnan(A["auc_a"]))
    Ao = k2_arms(ptr, cell, lab, far, R, Z, SC, ia, ia)
    same0 = bool(np.array_equal(Ao["auc_a"], Ao["auc_b"], equal_nan=True))
    Ac = k2_arms(ptr, cell, lab, far, R, np.ones_like(Z), SC, ia, ib)
    rk_c = ~np.isnan(Ac["auc_a"])
    const_ok = bool(np.all(Ac["auc_a"][rk_c] == 0.5) and np.all(Ac["auc_b"][rk_c] == 0.5))
    m_ = min(I["n"], 500)
    hd = GB.hab_dist(I["Vk"][:m_], np.asarray(RH)[I["stk"][:m_]]).reshape(m_, NC) if m_ else np.zeros((0, NC))
    a_, b_ = I["ptr"][0], I["ptr"][m_]
    it_ = np.repeat(np.arange(m_), np.diff(I["ptr"][:m_ + 1]))
    kk = AR["keep_a"][a_:b_]
    hd_gap = float(np.abs(AR["s_a"][a_:b_][kk] + hd[it_, I["cell"][a_:b_]][kk]).max()) if kk.any() else 0.0
    gate("W2", ok_loop and worst <= 1e-12 and sc_gap <= 1e-12 and sc_nan and same0 and const_ok and hd_gap <= 1e-9,
         f"AUC by a pair loop on the items of windows {pick} = the tool's {ok_loop}, largest difference {worst:.1e}; maps x 2.5 + 1.3 "
         f"leave every AUC unchanged, largest difference {sc_gap:.1e}; control = own gives d = 0 {same0}; a constant map gives 0.5 "
         f"at every rankable item {const_ok} ({int(rk_c.sum()):,}); real_val own-arm scores = minus hab_dist on {m_} items, "
         f"largest difference {hd_gap:.1e}")
    if not res["W2"]:
        return stop("W2")
    # ---- W3
    same_holes = bool(np.array_equal(A["keep_a"], A["keep_b"])) and bool(A["same_rec"].all())
    kh = A["keep_a"]
    in_both = bool(np.all(SC[ia[item[kh]], cell[kh]] & SC[ib[item[kh]], cell[kh]]))
    far_ok = bool(np.all(dist_full[item[kh], cell[kh]] >= FAR - FAR_TOL))
    gate("W3", same_holes and in_both and far_ok,
         f"both arms use the same holes and records on every item {same_holes}; every kept hole scored in both maps {in_both} and "
         f"3 or more tetrads from every kept record {far_ok} ({int(kh.sum()):,} kept far holes)")
    if not res["W3"]:
        return stop("W3")
    # ---- W4
    nsv = I["Ett"].sum(1)
    want = np.repeat(np.where(nsv >= 2, (3 * nsv + 5) // 10, 0), REPEATS)
    cnt_ok = np.array_equal(np.array([len(h) for h in I["holes"]]), want)
    inside = all(I["Ett"][j // REPEATS][h].all() and len(np.unique(h)) == len(h) for j, h in enumerate(I["holes"]))
    again = draw_holes(I["Ett"], np.zeros_like(I["Vt"]), frow)
    same_draw = all(np.array_equal(p_, q_) for p_, q_ in zip(again, I["holes"]))
    no_self = bool(np.all(Rcomm[I["part"]] != Rcomm)) if len(np.unique(Rcomm)) > 1 else False
    gate("W4", cnt_ok and inside and same_draw and no_self,
         f"real_val: {I['ns']:,} habitat-shown stacks x {REPEATS} repeats, {I['n']:,} items ({I['n_left']:,} left out: no hole or no "
         f"kept record); hole counts = (3n + 5) // 10 {cnt_ok}; inside the target-year effort and distinct {inside}; drawn again "
         f"with every record set to 0, the same {same_draw}; no partner in its own community {no_self}")
    if not res["W4"]:
        return stop("W4")
    allowed = [str(p_.resolve()) for p_ in (bnm, h2a, h2b, b1a)]
    okp = all(any(p_.startswith(q_ + os.sep) for q_ in allowed) or p_ == str(fzp.resolve())
              or (p_.startswith(str(rv.resolve()) + os.sep) and p_.endswith("_input.npz")) for p_ in log.paths)
    nrv = sum(p_.startswith(str(rv.resolve()) + os.sep) for p_ in log.paths)
    gate("W5", okp, f"np.load log: {len(log.paths)} files, all in the BNM fit, H2a, H2b or B-1a folders, the frozen set or real_val "
         f"input files ({nrv}); test half, H5, H6, real_eval and real_val truth 0")
    if not res["W5"]:
        return stop("W5")
    # ---- results
    stid, rcls, rbin = F["stid"], F["rcls"], F["rbin"]
    sparse = (rcls <= 1) | (rbin <= 2)
    grp = {"recorded class": (rcls, TCLS), "records over 6 years": (rbin, BINS)}
    nwin = len(np.unique(win))
    vals = {"own - 0.5": A["auc_a"] - 0.5, "control - 0.5": A["auc_b"] - 0.5, "own - control": d}
    rows = {k: {g: group_rows(v, stid, gi, win, labs, floor) for g, (gi, labs) in grp.items()} for k, v in vals.items()}
    srow = {k: s_row(v, stid, sparse, win, floor) for k, v in vals.items()}
    out(f"   BNM fit half, far holes (3+ tetrads from every kept record), stack-first mean [95% interval, {nwin} windows resampled] "
        f"(distinct stacks):")
    for k in vals:
        for g in grp:
            out(f"      {k:<14} {g:<20} " + "; ".join(fmt(q) for q in rows[k][g]))
        out(f"      {k:<14} sparse set S         {fmt(srow[k])}")
    lost = A["n_far"] - A["n_keep"]
    out("   support (M1): far holes kept / dropped, items dropped for no kept record in the support, items that lost a kept record:")
    for g, (gi, labs) in grp.items():
        out(f"      {g:<20} " + "; ".join(f"{lb} {int(A['n_keep'][gi == j].sum()):,} / {int(lost[gi == j].sum()):,}, "
                                         f"{int(A['dropped'][gi == j].sum())}, {int((A['n_rec_keep'] < A['n_rec'])[gi == j].sum())}"
                                         for j, lb in enumerate(labs)))
    out(f"      sparse set S         {int(A['n_keep'][sparse].sum()):,} / {int(lost[sparse].sum()):,}, {int(A['dropped'][sparse].sum())}, "
        f"{int((A['n_rec_keep'] < A['n_rec'])[sparse].sum())}; rankable items in S {int((~np.isnan(d) & sparse).sum()):,}")
    # real_val
    sp_r = (I["rcls"] <= 1) | (I["rbin"] <= 2)
    win_r = Rcomm[I["stk"]].astype(str)
    stid_r = frow[I["stk"]]
    grp_r = {"recorded class": (I["rcls"], TCLS), "records over 6 years": (I["rbin"], BINS)}
    vals_r = {"own - 0.5": AR["auc_a"] - 0.5, "control - 0.5": AR["auc_b"] - 0.5, "own - control": dR}
    rows_r = {k: {g: group_rows(v, stid_r, gi, win_r, labs, floor) for g, (gi, labs) in grp_r.items()} for k, v in vals_r.items()}
    srow_r = {k: s_row(v, stid_r, sp_r, win_r, floor) for k, v in vals_r.items()}
    out(f"   real_val reference (simulated, habitat shown, report only), stack-first mean [95% interval, {len(np.unique(win_r))} "
        f"communities resampled] (distinct stacks); support drops {int((AR['n_far'] - AR['n_keep']).sum())} far holes:")
    for k in vals_r:
        for g in grp_r:
            out(f"      {k:<14} {g:<20} " + "; ".join(fmt(q) for q in rows_r[k][g]))
        out(f"      {k:<14} sparse set S         {fmt(srow_r[k])}")
    s = srow["own - control"]
    passed = bool(s["n"] >= floor and s["lo"] > 0)
    out(f"   rule: in S, AUC(own) - AUC(control) {fmt(s)}; lower end above 0 {s['lo'] > 0 if s['n'] else False}, {s['n']} distinct "
        f"stacks >= {floor} {s['n'] >= floor}")
    # ---- W6 files
    out_dir.mkdir(parents=True)
    sc = dict(stack_id=stid, window=win.astype(str), rcls=rcls, rbin=rbin, sparse=sparse, auc_own=A["auc_a"], auc_control=A["auc_b"],
              far_holes=A["n_far"], kept_holes=A["n_keep"], kept_records=A["n_rec"], support_records=A["n_rec_keep"], dropped=A["dropped"])
    rvz = dict(frozen_row=stid_r, repeat=I["rep"], community=Rcomm[I["stk"]], rcls=I["rcls"], rbin=I["rbin"], sparse=sp_r,
               auc_own=AR["auc_a"], auc_control=AR["auc_b"], far_holes=AR["n_far"], holes=np.diff(I["ptr"]))
    X.save_npz(out_dir / "b1b_scores.npz", sc)
    X.save_npz(out_dir / "b1b_realval.npz", rvz)
    back_ok = True
    for f_, arr in (("b1b_scores.npz", sc), ("b1b_realval.npz", rvz)):
        bk = dict(np.load(out_dir / f_))
        back_ok &= set(bk) == set(arr) and all(np.array_equal(bk[k], np.asarray(arr[k]), equal_nan=bk[k].dtype.kind == "f") for k in arr)
    mz = {f_: X.file_md5(out_dir / f_) for f_ in ("b1b_scores.npz", "b1b_realval.npz")}
    gate("W6", back_ok, f"b1b_scores.npz {mz['b1b_scores.npz']}, b1b_realval.npz {mz['b1b_realval.npz']} written once and read back "
         f"equal")
    clean = lambda o_: json.loads(json.dumps(o_, default=lambda v: v.item() if hasattr(v, "item") else str(v)).replace("NaN", "null"))
    rep.update(gates=res, floor=floor, n=int(n), n_realval=int(I["n"]), off_land=off, off_land_flipped=off_flip,
               bnm=dict(rows=rows, sparse_S=srow), realval=dict(rows=rows_r, sparse_S=srow_r),
               support=dict(far=int(A["n_far"].sum()), kept=int(A["n_keep"].sum()), items_dropped=int(A["dropped"].sum()),
                            items_lost_record=int((A["n_rec_keep"] < A["n_rec"]).sum())),
               rule=dict(sparse_d=s, passed=passed),
               md5=dict(tool=X.file_md5(__file__), fingerprint=fingerprint(__file__), freeze=fmd5, b1a_habitat=m_hab, frozen=m_fz, **mz),
               minutes=round((time.time() - t0) / 60, 1), written_utc=F0.now())
    (out_dir / "b1b_report.json").write_text(json.dumps(clean(rep), indent=1))
    n_ok = sum(res.values())
    out(f"   written b1b_report.json ({rep['minutes']} min)")
    out(f"stage B-1b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; S: AUC(own) - AUC(control) {s['mean']:+.4f} [{s['lo']:+.4f}, "
        f"{s['hi']:+.4f}] ({s['n']} stacks); B-1 by the rule written before the run: {'PASS' if passed else 'FAIL'}")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bnm", default="../data/bnm_fit")
    ap.add_argument("--h2a", default="../data/stage_h2_predict")
    ap.add_argument("--h2b", default="../data/stage_h2b_score")
    ap.add_argument("--b1a", default="../data/stage_b1a_habitat")
    ap.add_argument("--freeze", default="../data/stage_h3_freeze/h3_freeze.json")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--p41", default="../data/stage_p4_1")
    ap.add_argument("--out", default="../data/stage_b1b_k2")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_37_b1b_k2.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found")
        spec = importlib.util.spec_from_file_location("test_37", tp); t37 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t37)
        sys.exit(0 if t37.self_test(Path(__file__), a.obj1_tools) else 1)
    r_, _ = run(a.bnm, a.h2a, a.h2b, a.b1a, a.freeze, a.runs, a.obj1_tools, a.g2, a.g3, a.data, a.p41, a.out, out=lambda s: print(s, flush=True))
