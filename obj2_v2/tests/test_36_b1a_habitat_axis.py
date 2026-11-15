#!/usr/bin/env python3
"""
test_36_b1a_habitat_axis.py  (obj2_v2, self-test of tools/stage_b1a_habitat_axis.py, 15 Nov 2026)

Run through the tool:  python3 ../tools/stage_b1a_habitat_axis.py --self-test 2>&1 | tail -n 1   (from obj2_v2/work)
Planted data only, in a temporary folder: a 21-band 1 km percentage raster (EPSG 27700) holding a west-to-east gradient from
woodland (class 1) to arable (class 3) with a little neutral grassland (class 5), open sea in one corner and part-covered
coastal cells; six planted windows of 20 x 20 tetrads; a planted p4_windows.npz whose LCM land is built here by its own loop
(the P3c rule) and whose surveyed part adds one sea tetrad. Nothing of ../data is read.
    U1  tetrad sums: vectorised = plain loop on random rasters and corners; composition sums to 1; window name parse
    U2  planted gradient: PC1 loads woodland and arable with opposite signs and the largest loading is positive; z follows
        easting inside each window (rank correlation above 0.95); control: a raster with no gradient gives a PC1 share well
        below the gradient raster's
    T1  the tool: gates J0-J4 PASS, files written, last line
    T2  recomputed outside the tool from the planted raster: composition, PC1 by SVD, z of every window (1e-9); the surveyed
        sea tetrad is 0; control index = (k + 3) mod 6
    T3  refused: an existing output folder; an output inside the P4-0 folder
    T4  planted bugs, each fails its gate and writes nothing: grid row-flipped (J1), composition not normalised (J2), z over all
        400 tetrads (J2), PC1 sign reversed (J2), a test-half path loaded (J3), p4_windows.npz changed (J0)
"""
import importlib.util, json, os, tempfile
from pathlib import Path
import numpy as np

G = 20


def load_tool(tool_path):
    spec = importlib.util.spec_from_file_location("stage_b1a_t", tool_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def write_raster(d, A, origin=(0, 80000)):
    import rasterio
    from rasterio.transform import from_origin
    d.mkdir(parents=True, exist_ok=True)
    tr = from_origin(origin[0], origin[1], 1000, 1000)
    with rasterio.open(d / "gb2019lcm1km_percentage_target.tif", "w", driver="GTiff", height=A.shape[1], width=A.shape[2],
                       count=21, crs="EPSG:27700", transform=tr, dtype="uint8") as f:
        f.write(A.astype(np.uint8))


def planted_raster(g, gradient=True):
    H, W = 80, 120                                               # northings 0-80 km, eastings 0-120 km (row 0 = north)
    A = np.zeros((21, H, W))
    for i in range(H):
        for j in range(W):
            n_km = H - 1 - i
            if n_km >= 70 and j >= 100:
                continue                                         # open sea: all classes 0
            a = j / (W - 1) if gradient else g.uniform(0, 1)
            a = float(np.clip(a + g.normal(0, 0.05), 0, 1))
            gr = g.uniform(0.02, 0.08)
            v = np.zeros(21); v[0] = (1 - a) * (1 - gr); v[2] = a * (1 - gr); v[4] = gr
            cover = 100 if not (n_km >= 66 and j >= 96) else g.uniform(30, 80)    # part-covered coastal cells
            A[:, i, j] = np.round(v * cover)
    return A


def land_loop(A, e_km, n_km, origin=(0, 80000)):
    """P3c rule by a plain loop: tetrad land if any of its four cells has land cover (classes except 13, 14, 17, 18) above 0."""
    land = np.zeros((G, G), bool); H, W = A.shape[1:]
    keep = [c - 1 for c in range(1, 22) if c not in (13, 14, 17, 18)]
    for r in range(G):
        for c in range(G):
            for dn in (0, 1):
                for de in (0, 1):
                    E, N = (e_km + 2 * c + de) * 1000, (n_km + 2 * r + dn) * 1000
                    col, row = (E - origin[0]) // 1000, (origin[1] - (N + 1000)) // 1000
                    if 0 <= row < H and 0 <= col < W and A[keep, row, col].sum() > 0:
                        land[r, c] = True
    return land


def world(tmp, g):
    tmp = Path(tmp)
    A = planted_raster(g)
    write_raster(tmp / "lcm", A)
    corners = [(0, 0), (40, 0), (80, 0), (0, 40), (40, 40), (80, 40)]
    names = np.array([f"bnm_e{e:04d}_n{n:04d}" for e, n in corners])
    perm = np.array([3, 0, 5, 1, 4, 2])                          # stored out of name order on purpose
    land_lcm = np.stack([land_loop(A, e, n) for e, n in corners])
    surv = g.random(land_lcm.shape) < 0.2
    sea = np.argwhere(~land_lcm[5])[0]
    surv[5, sea[0], sea[1]] = True                               # one surveyed tetrad with no land cover
    p4 = tmp / "stage_p4_0"; p4.mkdir()
    X = load_tool(Path(__file__).resolve().parent.parent / "tools" / "stage_b1a_habitat_axis.py").X
    X.save_npz(p4 / "p4_windows.npz", dict(window_names=names[perm], land_lcm=land_lcm[perm], surveyed=surv[perm], land=(land_lcm | surv)[perm]))
    (p4 / "p4_0_report.json").write_text(json.dumps(dict(gates={f"W{i}": True for i in range(7)})))
    expect = dict(lcm=X.file_md5(tmp / "lcm" / "gb2019lcm1km_percentage_target.tif"), p4_windows=X.file_md5(p4 / "p4_windows.npz"),
                  p4_0_report=X.file_md5(p4 / "p4_0_report.json"))
    return dict(lcm=tmp / "lcm", p4=p4, expect=expect, A=A, corners=corners, names=names, sea=(5, int(sea[0]), int(sea[1])))


def unit_checks(T, t, g):
    from rasterio.transform import from_origin
    ok = True
    for _ in range(5):
        A = g.integers(0, 6, (21, 50, 60)).astype(np.int64)
        tr = from_origin(10000, 90000, 1000, 1000)
        e, n = int(g.integers(0, 40)) * 2 + 4, int(g.integers(0, 20)) * 2 + 40
        ok &= np.array_equal(T.tetrad_sums(A, tr, e, n), T.loop_sums(A, tr, e, n))
    C, has = T.composition(T.loop_sums(A, tr, e, n))
    t("U1", ok and np.allclose(C[has].sum(-1), 1) and T.en("bnm_e0560_n0080.npz") == (560, 80),
      f"tetrad sums vectorised = plain loop on 5 random rasters and corners (some cells outside the raster) {ok}; compositions "
      f"sum to 1; window name parse {T.en('bnm_e0560_n0080.npz')}")


def checks(T, w, tmp, t):
    quiet = lambda *a, **k: None
    tmp = Path(tmp)
    lines = []
    r, rep = T.run(w["lcm"], w["p4"], tmp / "out", expect=w["expect"], out=lines.append)
    t("T1", all(r.values()) and set(r) == {"J0", "J1", "J2", "J3", "J4"} and (tmp / "out" / "b1a_report.json").is_file()
      and lines[-1].startswith("stage B-1a: gates 5 PASS"), f"planted windows: {r}; last line: {lines[-1][:120]}")
    if not (tmp / "out" / "b1a_habitat.npz").is_file():
        t("T2", False, "skipped: T1 wrote nothing"); return lines
    Z = np.load(tmp / "out" / "b1a_habitat.npz")
    A, corners, names = w["A"], w["corners"], w["names"]
    order = np.argsort(names)
    P = np.load(w["p4"] / "p4_windows.npz")
    pord = np.argsort(P["window_names"].astype(str))
    L = P["land"].astype(bool)[pord]
    comp = []
    for k in order:
        e, n = corners[k]; S = np.zeros((G, G, 21))
        for r_ in range(G):
            for c_ in range(G):
                for dn in (0, 1):
                    for de in (0, 1):
                        col, row = (e + 2 * c_ + de), 79 - (n + 2 * r_ + dn)
                        if 0 <= row < 80 and 0 <= col < 120:
                            S[r_, c_] += A[:, row, col]
        comp.append(S)
    S = np.stack(comp); tot = S.sum(-1); has = tot > 0
    C = np.where(has[..., None], S / np.maximum(tot, 1)[..., None], 0)
    fit = L & has
    Xf = C[fit]; mu = Xf.mean(0)
    v = np.linalg.svd(Xf - mu, full_matrices=False)[2][0]
    v = v if v[np.argmax(np.abs(v))] > 0 else -v
    s = np.where(has, (C - mu) @ v, 0)
    z = np.zeros_like(s)
    for k in range(len(s)):
        m = fit[k]
        z[k][m] = (s[k][m] - s[k][m].mean()) / s[k][m].std()
    k5 = int(np.flatnonzero(names[order] == names[w["sea"][0]])[0])
    ok_z = float(np.abs(z - Z["z"]).max()) <= 1e-9 and np.allclose(C, Z["composition"], atol=1e-6)
    sea0 = Z["z"][k5, w["sea"][1], w["sea"][2]] == 0 and bool(L[k5, w["sea"][1], w["sea"][2]]) and not bool(has[k5, w["sea"][1], w["sea"][2]])
    ctrl = np.array_equal(Z["control_index"], (np.arange(6) + 3) % 6)
    t("T2", ok_z and sea0 and ctrl, f"recomputed outside the tool from the raster: composition and z of all 6 windows {ok_z} "
      f"(largest z difference {float(np.abs(z - Z['z']).max()):.1e}); surveyed sea tetrad in L_w with no composition is 0 {sea0}; "
      f"control index (k + 3) mod 6 {ctrl}")
    stops = []
    for f_ in (lambda: T.run(w["lcm"], w["p4"], tmp / "out", expect=w["expect"], out=quiet),
               lambda: T.run(w["lcm"], w["p4"], w["p4"] / "x", expect=w["expect"], out=quiet)):
        try:
            f_(); stops.append(False)
        except SystemExit:
            stops.append(True)
    t("T3", all(stops), f"refused: an existing output folder {stops[0]}, an output inside the P4-0 folder {stops[1]}")
    for bug, g_ in (("flip", "J1"), ("no_norm", "J2"), ("z_all", "J2"), ("sign", "J2"), ("test_path", "J3")):
        os.environ["STAGE_B1A_BROKEN"] = bug
        try:
            rb, _ = T.run(w["lcm"], w["p4"], tmp / f"b_{bug}", expect=w["expect"], out=quiet)
        finally:
            os.environ.pop("STAGE_B1A_BROKEN", None)
        t(f"bug {bug}", rb.get(g_) is False and not (tmp / f"b_{bug}").exists(), f"{g_} fails and nothing is written ({rb})")
    e2 = dict(w["expect"], p4_windows="0" * 32)
    rb, _ = T.run(w["lcm"], w["p4"], tmp / "b_md5", expect=e2, out=quiet)
    t("bug p4_changed", rb.get("J0") is False and len(rb) == 1, f"a p4_windows.npz other than P4-0's fails J0 and stops ({rb})")
    return lines


def self_test(tool_path):
    T = load_tool(tool_path)
    g = np.random.default_rng(36)
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    unit_checks(T, t, g)
    with tempfile.TemporaryDirectory() as tmp:
        w = world(tmp, g)
        lines = checks(T, w, tmp, t)
        # U2: the axis on the planted gradient, and a control raster with no gradient
        Z = np.load(Path(tmp) / "out" / "b1a_habitat.npz")
        pc1 = Z["pc"][:, 0]
        opp = pc1[0] * pc1[2] < 0 and pc1[int(np.argmax(np.abs(pc1)))] > 0
        rho = []
        for k in range(6):
            m = Z["scored"][k]
            east = np.broadcast_to(np.arange(G)[None, :], (G, G))[m]
            a, b = np.argsort(np.argsort(Z["z"][k][m])), np.argsort(np.argsort(east))
            rho.append(abs(np.corrcoef(a, b)[0, 1]))
        share_g = Z["eigvals"][0] / Z["eigvals"].sum()
        w2 = Path(tmp) / "nograd"; A2 = planted_raster(np.random.default_rng(7), gradient=False)
        write_raster(w2 / "lcm", A2)
        S2 = np.stack([T.tetrad_sums(A2.astype(np.int64), T.read_lcm(w2 / "lcm")[1], e, n) for e, n in w["corners"]])
        C2, h2 = T.composition(S2)
        mu2, _, V2 = T.pca(C2[h2])
        z2, sc2 = T.zmaps(C2, h2, h2, mu2, V2[:, 0])
        rho2 = []
        for k in range(6):
            m = sc2[k]
            east = np.broadcast_to(np.arange(G)[None, :], (G, G))[m]
            a, b = np.argsort(np.argsort(z2[k][m])), np.argsort(np.argsort(east))
            rho2.append(abs(np.corrcoef(a, b)[0, 1]))
        t("U2", opp and min(rho) > 0.95 and max(rho2) < 0.3,
          f"PC1 loads woodland {pc1[0]:+.3f} and arable {pc1[2]:+.3f} with opposite signs, largest loading positive {opp}; z follows "
          f"easting inside every window (rank correlation min {min(rho):.3f}, PC1 share {share_g:.3f}); control raster with the same "
          f"classes in random places: z does not follow easting (rank correlation max {max(rho2):.3f})")
        print("   planted run log:", *lines, sep="\n      ")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)
