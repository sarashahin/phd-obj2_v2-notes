#!/usr/bin/env python3
"""
stage_b1a_habitat_axis.py  (obj2_v2, habitat check B-1, part a, 15 Nov 2026; Sara's yes to B-1 before Figure 2 on 15 Nov;
rule in claude/obj2_v2_H_rules_part2.md of the notes repo, section "Habitat check B-1", written before any value; report only)

Question: build the one habitat axis the model can be given on BNM, from the UKCEH Land Cover Map only, for the 44 fit-half
windows: the first principal component of land-cover composition per 2 km tetrad, z-scored per window over its land mask,
the way channel 18 was built in training (one habitat axis per community, z-scored with the map's own mean and sd).
No BNM record, label, test-half or real_eval file is opened. B-1b then measures whether this axis carries signal.

Rule (as written before the run)
    Composition of a tetrad = per class, the sum of its four 1 km cells' percentages / the sum over all 21 classes; none when
        that sum is 0. Grid as step 10 and P3c: row from the south, column from the west, 2 km tetrads; a cell outside the
        raster counts 0.
    Fit set = tetrads of the fit-half windows in L_w (P4-0's land mask) with a composition, each window-tetrad once.
    Axis = PC1 of the 21-share vectors over the fit set (covariance, ddof 1, numpy eigh); the class with the largest
        absolute loading gets a positive loading. PC2 and PC3 reported only.
    Map value = (composition - fit-set mean) . PC1, z-scored per window over its scored L_w tetrads; 0 elsewhere.
    Control for B-1b: windows in name order, window k takes the map of window (k + n/2) mod n (n = 44: k + 22).

Gates (if one fails, nothing past it is printed and nothing is written)
    J0  inputs: map md5 = block P3c-0 (d66aac88); p4_windows.npz and p4_0_report.json md5 = Sara's P4-0 run, its gates PASS
    J1  grid: land from the composition by the P3c rule (any land class above 0 in any of the four cells) = P4-0's LCM land
        in every window (0 differences); 21 bands, uint8, EPSG 27700, 1 km pixels on whole km
    J2  code: PC1 by eigh = PC1 by SVD (|cos| and eigenvalue to 1e-9); sign rule held; compositions sum to 1 (1e-9); z mean 0
        and sd 1 over the scored L_w tetrads of every window (1e-9); a plain loop over the cells of 3 windows = the
        vectorised composition (1e-12)
    J3  np.load log: only p4_windows.npz; a path of BNM records or labels, the test half or real_eval is refused
    J4  outputs written once into a new folder and read back equal

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, under a minute)
    python3 ../tools/stage_b1a_habitat_axis.py --self-test 2>&1 | tail -n 1
    python3 -u ../tools/stage_b1a_habitat_axis.py > ../logs/stage_b1a_habitat_axis.txt 2>&1
Needs export_obj2.py and stage_p3c_band_check.py in the same folder, rasterio, and ../tests/test_36_b1a_habitat_axis.py for
the self-test.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_b1a_habitat_axis.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


X = _load("export_obj2")
BC = _load("stage_p3c_band_check")
G = X.G
NOLAND = (13, 14, 17, 18)
LAND = [c - 1 for c in range(1, 22) if c not in NOLAND]
NAMES = ("Deciduous woodland", "Coniferous woodland", "Arable", "Improved grassland", "Neutral grassland", "Calcareous grassland",
         "Acid grassland", "Fen, marsh, swamp", "Heather", "Heather grassland", "Bog", "Inland rock", "Saltwater", "Freshwater",
         "Supralittoral rock", "Supralittoral sediment", "Littoral rock", "Littoral sediment", "Saltmarsh", "Urban", "Suburban")
EXPECT = dict(lcm=BC.MD5[BC.PCT], p4_windows="928cf96c9bcbe9a14bb0f639adb581eb", p4_0_report="842697b1acf4f8d6484e6d0f659d8b73")
FORBID = ("bnm_fit", "bnm_test", "_truth", "real_eval", "stage_h2b_score", "stage_h5_predict", "stage_h6_test_score")


def broken():
    return os.environ.get("STAGE_B1A_BROKEN", "")      # self-test only: plants one bug so its gate must fail


def fingerprint(p):
    b = Path(p).read_bytes()
    return hashlib.md5(b"\n".join(l.rstrip() for l in b.split(b"\n") if l[:2] != b"#!").rstrip(b"\n")).hexdigest()


def en(nm):
    s = Path(str(nm)).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])


class LoadLog:
    def __enter__(self):
        self.orig, self.paths = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        if isinstance(f, (str, Path)):
            p = str(Path(f).resolve()); self.paths.append(p)
            if any(t in p for t in FORBID):
                raise RuntimeError(f"refused np.load of {p}")
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- pieces
def read_lcm(lcm):
    import rasterio
    with rasterio.open(Path(lcm) / BC.PCT) as s:
        tr, epsg = s.transform, (s.crs.to_epsg() if s.crs else None)
        ok = s.count == 21 and s.dtypes[0] == "uint8" and epsg == 27700 and tr.a == 1000 and tr.e == -1000 and tr.b == 0 \
            and tr.d == 0 and tr.c % 1000 == 0 and tr.f % 1000 == 0
        T = s.read().astype(np.int64)
    return T, tr, ok


def tetrad_sums(T, tr, e_km, n_km):
    """(G, G, 21) per-class sums of the four 1 km cells of each tetrad (row from the south, column from the west)."""
    _, H, Wd = T.shape
    out = np.zeros((G, G, 21), np.int64)
    r = np.arange(G)[:, None, None, None]; c = np.arange(G)[None, :, None, None]
    dn = np.array([0, 1])[None, None, :, None]; de = np.array([0, 1])[None, None, None, :]
    E = (e_km + 2 * c + de) * 1000 + 0 * r * dn; N = (n_km + 2 * r + dn) * 1000 + 0 * c * de
    col = ((E - tr.c) // 1000).astype(np.int64); row = ((tr.f - (N + 1000)) // 1000).astype(np.int64)
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < Wd)
    rr, cc = np.where(inside, row, 0), np.where(inside, col, 0)
    vals = T[:, rr, cc] * inside[None]                                     # (21, G, G, 2, 2)
    out[:] = np.moveaxis(vals.sum((3, 4)), 0, -1)
    if broken() == "flip":
        out = out[::-1]
    return out


def loop_sums(T, tr, e_km, n_km):
    """Plain loop version of tetrad_sums (J2)."""
    _, H, Wd = T.shape
    out = np.zeros((G, G, 21), np.int64)
    for r in range(G):
        for c in range(G):
            for dn in (0, 1):
                for de in (0, 1):
                    E, N = (e_km + 2 * c + de) * 1000, (n_km + 2 * r + dn) * 1000
                    col, row = int((E - tr.c) // 1000), int((tr.f - (N + 1000)) // 1000)
                    if 0 <= row < H and 0 <= col < Wd:
                        out[r, c] += T[:, row, col]
    return out


def composition(S):
    tot = S.sum(-1)
    C = np.zeros(S.shape)
    has = tot > 0
    C[has] = S[has] / (tot[has][:, None] if broken() != "no_norm" else 400.0)
    return C, has


def pca(Xf):
    mu = Xf.mean(0)
    cov = np.cov(Xf - mu, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    for j in range(3):
        k = int(np.argmax(np.abs(V[:, j])))
        if V[k, j] < 0:
            V[:, j] = -V[:, j]
    if broken() == "sign":
        V[:, 0] = -V[:, 0]
    return mu, w, V


def zmaps(C, has, L, mu, v1):
    s = np.where(has, (C - mu) @ v1, 0.0)
    z = np.zeros(s.shape); scored = has & L
    for k in range(len(s)):
        m = scored[k] if broken() != "z_all" else np.ones_like(scored[k])
        if m.sum() >= 2:
            z[k][m] = (s[k][m] - s[k][m].mean()) / s[k][m].std()
    z[~scored] = 0.0
    return z, scored


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- run
def run(lcm="../data/lcm2019_1km", p4="../data/stage_p4_0", out_dir="../data/stage_b1a_habitat", expect=EXPECT, out=print):
    lcm, p4, out_dir = Path(lcm), Path(p4), Path(out_dir)
    o = out_dir.resolve()
    if any(o == q.resolve() or q.resolve() in o.parents for q in (lcm, p4)):
        sys.exit("STOP: nothing is written inside an input folder; give another --out")
    if any(t in str(o) for t in FORBID):
        sys.exit("STOP: the output folder may not lie in a BNM, truth, test-half or real_eval folder")
    if out_dir.exists():
        sys.exit(f"STOP: {out_dir} exists; give a new folder")
    t0, res, rep = time.time(), {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage B-1a: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, rep

    out(f"stage_b1a_habitat_axis.py md5 {X.file_md5(__file__)} (fingerprint {fingerprint(__file__)}); export_obj2.py "
        f"{X.file_md5(HERE / 'export_obj2.py')}; stage_p3c_band_check.py {X.file_md5(HERE / 'stage_p3c_band_check.py')}")
    # ---- J0
    m_lcm = X.file_md5(lcm / BC.PCT)
    m_w, m_r = X.file_md5(p4 / "p4_windows.npz"), X.file_md5(p4 / "p4_0_report.json")
    rj = json.loads((p4 / "p4_0_report.json").read_text())
    g_ok = bool(rj.get("gates")) and all(rj["gates"].values())
    gate("J0", m_lcm == expect["lcm"] and m_w == expect["p4_windows"] and m_r == expect["p4_0_report"] and g_ok,
         f"map {BC.PCT} md5 {m_lcm} = block P3c-0 {m_lcm == expect['lcm']}; p4_windows.npz {m_w} = P4-0 {m_w == expect['p4_windows']}; "
         f"p4_0_report.json {m_r} = P4-0 {m_r == expect['p4_0_report']}, its gates PASS {g_ok}")
    if not res["J0"]:
        return stop("J0")
    with LoadLog() as log:
        try:
            if broken() == "test_path":
                np.load(p4.parent / "bnm_test" / "bnm_frozen.npz")
            P = dict(np.load(p4 / "p4_windows.npz"))
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("J3", False, f"np.load log: {crashed}")
        return stop("J3")
    # ---- J1
    T, tr, geo_ok = read_lcm(lcm)
    names0 = P["window_names"].astype(str)
    order = np.argsort(names0)
    names = names0[order]
    land_lcm, L = P["land_lcm"].astype(bool)[order], P["land"].astype(bool)[order]
    n = len(names)
    S = np.stack([tetrad_sums(T, tr, *en(nm)) for nm in names])
    land_c = S[..., LAND].sum(-1) > 0
    diff = int((land_c != land_lcm).sum())
    gate("J1", geo_ok and diff == 0 and S.shape == (n, G, G, 21),
         f"{n} fit-half windows; land from the composition by the P3c rule against P4-0's LCM land: {diff} tetrads differ; "
         f"map 21 bands, uint8, EPSG 27700, 1 km on whole km {geo_ok}")
    if not res["J1"]:
        return stop("J1")
    # ---- axis and maps
    C, has = composition(S)
    fit = L & has
    Xf = C[fit]
    mu, w, V = pca(Xf)
    v1 = V[:, 0]
    z, scored = zmaps(C, has, L, mu, v1)
    # ---- J2
    U, sv, Vt = np.linalg.svd(Xf - Xf.mean(0), full_matrices=False)
    w_svd = sv ** 2 / (len(Xf) - 1)
    cos = abs(float(Vt[0] @ v1))
    k1 = int(np.argmax(np.abs(v1)))
    sums = float(np.abs(C[has].sum(-1) - 1).max())
    zm = max(float(abs(z[k][scored[k]].mean())) for k in range(n) if scored[k].sum() >= 2)
    zs = max(float(abs(z[k][scored[k]].std() - 1)) for k in range(n) if scored[k].sum() >= 2)
    pick = sorted({0, n // 2, n - 1})
    loop_d = max(float(np.abs(loop_sums(T, tr, *en(names[k])) - S[k]).max()) for k in pick)
    ok2 = abs(cos - 1) <= 1e-9 and abs(w_svd[0] - w[0]) <= 1e-9 * max(1.0, w[0]) and v1[k1] > 0 and sums <= 1e-9 \
        and zm <= 1e-9 and zs <= 1e-9 and loop_d <= 1e-12
    gate("J2", ok2, f"PC1 by eigh against SVD |cos| {cos:.12f}, eigenvalue {w[0]:.6g} / {w_svd[0]:.6g}; largest loading "
         f"(class {k1 + 1}) positive {v1[k1] > 0}; compositions sum to 1 within {sums:.1e}; z per window mean within {zm:.1e}, "
         f"sd within {zs:.1e} of 1; plain loop on windows {pick} = vectorised, largest difference {loop_d:.1e}")
    if not res["J2"]:
        return stop("J2")
    gate("J3", len(log.paths) == 1 and log.paths[0].endswith("p4_windows.npz"),
         f"np.load log: {len(log.paths)} file ({Path(log.paths[0]).name}); no BNM record, label, test-half or real_eval file")
    # ---- report
    share = w / w.sum()
    out(f"   fit set: {int(fit.sum()):,} window-tetrads in L_w with a composition ({int((L & ~has).sum())} L_w tetrads with none, "
        f"given 0); variance share PC1 {share[0]:.3f}, PC2 {share[1]:.3f}, PC3 {share[2]:.3f}")
    for j in range(3):
        top = np.argsort(-np.abs(V[:, j]))[:6]
        out(f"   PC{j + 1} loadings (largest 6): " + "; ".join(f"{c + 1} {NAMES[c]} {V[c, j]:+.3f}" for c in top))
    out(f"   fit-set mean share (largest 6): " + "; ".join(f"{c + 1} {NAMES[c]} {mu[c]:.3f}" for c in np.argsort(-mu)[:6]))
    half = n // 2
    ctrl = (np.arange(n) + half) % n
    out(f"   control pairing (name order): window k takes window (k + {half}) mod {n}; e.g. {names[0]} <- {names[ctrl[0]]}")
    per = [dict(window=str(names[k]), scored=int(scored[k].sum()), zero=int((~scored[k]).sum()), land_lw=int(L[k].sum()))
           for k in range(n)]
    out(f"   scored tetrads per window: min {min(p['scored'] for p in per)}, median {int(np.median([p['scored'] for p in per]))}, "
        f"max {max(p['scored'] for p in per)}")
    # ---- J4 outputs
    out_dir.mkdir(parents=True)
    arrays = dict(window_names=names, z=z, scored=scored, land=L, composition=C.astype(np.float32), pc=V[:, :3], mean=mu,
                  eigvals=w, control_index=ctrl.astype(np.int64))
    X.save_npz(out_dir / "b1a_habitat.npz", arrays)
    back = dict(np.load(out_dir / "b1a_habitat.npz"))
    same = set(back) == set(arrays) and all(np.array_equal(back[k], np.asarray(arrays[k])) for k in arrays)
    mz = X.file_md5(out_dir / "b1a_habitat.npz")
    gate("J4", same, f"b1a_habitat.npz md5 {mz} written once and read back equal")
    clean = lambda o_: json.loads(json.dumps(o_, default=lambda v: v.item() if hasattr(v, "item") else str(v)).replace("NaN", "null"))
    rep.update(gates=res, windows=int(n), fit_tetrads=int(fit.sum()), variance_share=share[:3].tolist(),
               pc1={f"{c + 1} {NAMES[c]}": float(V[c, 0]) for c in range(21)}, per_window=per, control_offset=int(half),
               md5=dict(tool=X.file_md5(__file__), fingerprint=fingerprint(__file__), lcm=m_lcm, p4_windows=m_w, habitat=mz),
               minutes=round((time.time() - t0) / 60, 2), written_utc=now())
    (out_dir / "b1a_report.json").write_text(json.dumps(clean(rep), indent=1))
    n_ok = sum(res.values())
    out(f"stage B-1a: gates {n_ok} PASS, {len(res) - n_ok} FAIL; habitat axis PC1 ({share[0]:.3f} of the variance) for {n} "
        f"fit-half windows; b1a_habitat.npz {mz}; no BNM record or label opened")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--lcm", default="../data/lcm2019_1km")
    ap.add_argument("--p4", default="../data/stage_p4_0")
    ap.add_argument("--out", default="../data/stage_b1a_habitat")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_36_b1a_habitat_axis.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found")
        spec = importlib.util.spec_from_file_location("test_36", tp); t36 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t36)
        sys.exit(0 if t36.self_test(Path(__file__)) else 1)
    r_, _ = run(a.lcm, a.p4, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r_ and all(r_.values()) else 1)
