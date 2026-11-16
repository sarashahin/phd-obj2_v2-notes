#!/usr/bin/env python3
"""
test_37_b1b_k2.py  (obj2_v2, self-test of tools/stage_b1b_k2.py, 16 Nov 2026)

Run through the tool:  python3 ../tools/stage_b1b_k2.py --self-test 2>&1 | tail -n 1   (from obj2_v2/work)
Planted data only, in a temporary folder, with the real tools. Nothing of ../data is read.
    U1  the score and AUC on a hand example (one item, both arms, support drops a hole and a record)
    U2  planted windows (24, smooth maps): hidden records follow the own map -> S interval above 0 (PASS)
    U3  planted windows with gradient maps, hidden records follow distance to the records only, every tetrad land: the own
        AUC is above 0.55 (distance proxy) and the S interval of d holds 0 (FAIL)
    U4  M1 in action: the same distance-only world with sea over the south half of the 12 windows that are the controls of
        the 12 inland windows: without the support rule d is pushed above 0 there (a false PASS); with it, no PASS
    T1  the tool on test_25's planted fit half (floor 5) with a planted B-1a folder: gates W0-W6 PASS, files written, rule
        printed
    T2  recomputed outside the tool: AUC of both arms on 60 items by a pair loop with its own support rule; the S mean of d
        by an own stack-first loop; real_val holes, labels and kept records of 20 items by an own draw
    T3  refused: a second run into the same folder; an output inside the H2b folder; an output inside the B-1a folder
    T4  planted bugs, each fails its gate and writes nothing: B-1a file changed after its report (W0), grid flipped (W1),
        control = own window (W1), labels shifted (W1), support in one arm only (W3), near holes kept (W3), real_val holes
        drawn with the records (W4), a test-half path (W5), a real_val truth file (W5)
"""
import importlib.util, json, os, shutil, tempfile
from pathlib import Path
import numpy as np


def load_tool(tool_path):
    spec = importlib.util.spec_from_file_location("stage_b1b_k2_t", tool_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def smooth(g, G, ls=4.0):
    """A smooth random map (Gaussian filter of white noise by FFT), z-scored."""
    k = np.fft.fftfreq(G)
    kx, ky = np.meshgrid(k, k)
    f = np.real(np.fft.ifft2(np.fft.fft2(g.normal(size=(G, G))) * np.exp(-2 * (np.pi * ls) ** 2 * (kx ** 2 + ky ** 2))))
    return (f - f.mean()) / f.std()


def gradient(g, G):
    """A map that changes steadily across the window in a random direction, plus noise, z-scored (a strong distance proxy)."""
    th = g.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:G, 0:G]
    f = np.cos(th) * xx + np.sin(th) * yy + 0.5 * g.normal(size=(G, G))
    return (f - f.mean()) / f.std()


def plant_world(T, seed, mode, coast, kind="smooth", nw=24, nsp=14, reps=3):
    """Planted windows for U2-U4. mode 'habitat': occupancy falls with |z - optimum| on the own map; 'distance': occupancy
    falls with distance to a centre, no habitat. coast: windows nw/2.. have sea over rows 0-9 (unscored, z = 0). kind: the
    maps are smooth random fields or gradients."""
    G, NC = T.G, T.NC
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:G, 0:G]
    Z, SC = np.zeros((nw, NC)), np.ones((nw, NC), bool)
    for w in range(nw):
        z = smooth(g, G) if kind == "smooth" else gradient(g, G)
        land = np.ones((G, G), bool)
        if coast and w >= nw // 2:
            land[:10] = False
        z = np.where(land, (z - z[land].mean()) / z[land].std(), 0.0)
        Z[w], SC[w] = z.ravel(), land.ravel()
    ctrl = (np.arange(nw) + nw // 2) % nw
    ptr, cell, lab, far, R, ia, stid = [0], [], [], [], [], [], []
    for w in range(nw):
        land = SC[w].reshape(G, G)
        for sp in range(nsp):
            if mode == "habitat":
                p = 0.7 * np.exp(-((Z[w].reshape(G, G) - g.uniform(-1.2, 1.2)) / 0.45) ** 2)
            else:
                cy, cx = divmod(int(g.choice(np.flatnonzero(land.ravel()))), G)
                p = 0.8 * np.exp(-np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)
            occ = (g.random((G, G)) < p) & land
            surv = land & (g.random((G, G)) < 0.6)
            rec = (occ & surv).ravel()
            cells = np.flatnonzero(surv.ravel())
            if rec.sum() < 3:
                continue
            for r in range(reps):
                h = np.sort(g.choice(cells, (3 * len(cells) + 5) // 10, replace=False))
                kept = rec.copy(); kept[h] = False
                if not kept.any():
                    continue
                ky, kx = np.divmod(np.flatnonzero(kept), G)
                hy, hx = np.divmod(h, G)
                dist = np.sqrt(((hy[:, None] - ky[None]) ** 2 + (hx[:, None] - kx[None]) ** 2).min(1))
                cell.append(h); lab.append(rec[h]); far.append(dist >= 3); R.append(kept); ia.append(w); stid.append(w * 100 + sp)
                ptr.append(ptr[-1] + len(h))
    ia = np.array(ia)
    return dict(ptr=np.array(ptr), cell=np.concatenate(cell), lab=np.concatenate(lab), far=np.concatenate(far), R=np.array(R), Z=Z,
                SC=SC, ia=ia, ib=ctrl[ia], stid=np.array(stid), win=np.array([f"w{w:02d}" for w in ia]))


def world_s(T, W, support=True, mask=None):
    A = T.k2_arms(W["ptr"], W["cell"], W["lab"], W["far"], W["R"], W["Z"], W["SC"], W["ia"], W["ib"], support=support)
    d = A["auc_a"] - A["auc_b"]
    return T.s_row(d, W["stid"], np.ones(len(d), bool) if mask is None else mask, W["win"], 5), A


def unit_checks(T, t):
    NC = T.NC
    # U1 hand example: holes at cells 0-3 (far), records at cells 10 and 11; own map z, control map z'
    Z = np.zeros((2, NC)); SC = np.ones((2, NC), bool)
    Z[0, [0, 1, 2, 3, 10, 11]] = [1.0, 0.0, 3.0, 2.0, 1.0, 3.0]        # mean at the records 2.0
    Z[1, [0, 1, 2, 3, 10, 11]] = [0.0, 5.0, 1.0, 9.0, 1.0, 1.0]        # mean at the records 1.0
    R = np.zeros((1, NC), bool); R[0, [10, 11]] = True
    ptr, cell = np.array([0, 4]), np.array([0, 1, 2, 3])
    lab, far = np.array([True, False, True, False]), np.ones(4, bool)
    A = T.k2_arms(ptr, cell, lab, far, R, Z, SC, np.array([0]), np.array([1]))
    # own s = -|z - 2| = [-1, -2, -1, 0]: recorded {-1, -1} vs unrecorded {-2, 0}: pairs 1 + 0 + 1 + 0 = 2 of 4 -> 0.5
    # control s = -|z - 1| = [-1, -4, 0, -8]: recorded {-1, 0} vs {-4, -8}: 4 of 4 -> 1.0
    SC2 = SC.copy(); SC2[1, 3] = False; SC2[1, 11] = False                # control unscored at hole 3 and record 11
    A2 = T.k2_arms(ptr, cell, lab, far, R, Z, SC2, np.array([0]), np.array([1]))
    # support: holes 0-2, record 10 only: own s = -|z - 1| = [0, -1, -2]: recorded {0, -2} vs {-1}: 1 of 2 -> 0.5
    # control s = -|z - 1| = [-1, -4, 0]: recorded {-1, 0} vs {-4}: 2 of 2 -> 1.0
    ok = abs(A["auc_a"][0] - 0.5) < 1e-15 and abs(A["auc_b"][0] - 1.0) < 1e-15 and abs(A2["auc_a"][0] - 0.5) < 1e-15 \
        and abs(A2["auc_b"][0] - 1.0) < 1e-15 and A2["n_keep"][0] == 3 and A2["n_rec_keep"][0] == 1 and A2["same_rec"][0]
    t("U1", ok, f"hand example: own {A['auc_a'][0]:.3f} (0.5), control {A['auc_b'][0]:.3f} (1.0); with hole 3 and record 11 "
      f"unscored in the control: own {A2['auc_a'][0]:.3f} (0.5), control {A2['auc_b'][0]:.3f} (1.0), kept holes {A2['n_keep'][0]} (3), "
      f"kept records {A2['n_rec_keep'][0]} (1), same records in both arms {bool(A2['same_rec'][0])}")
    # U2-U4
    qh, _ = world_s(T, plant_world(T, 371, "habitat", False))
    t("U2", qh["lo"] > 0, f"hidden records follow the own map: S d {T.fmt(qh)}; lower end above 0 (PASS) {qh['lo'] > 0}")
    qd, Ad = world_s(T, plant_world(T, 372, "distance", False, kind="gradient"))
    own = float(np.nanmean(Ad["auc_a"]))
    t("U3", qd["lo"] <= 0 <= qd["hi"] and own > 0.55, f"hidden records follow distance only, all land, gradient maps: S d {T.fmt(qd)} "
      f"holds 0 (FAIL) {qd['lo'] <= 0 <= qd['hi']}, although the own AUC is {own:.3f} (a smooth map near the records acts as a "
      f"distance proxy; the control removes it)")
    Wc = plant_world(T, 374, "distance", True, kind="gradient", nsp=20)
    inl = Wc["ia"] < 12
    qn, An = world_s(T, Wc, support=False, mask=inl)
    qs, As = world_s(T, Wc, support=True, mask=inl)
    t("U4", qn["lo"] > 0 and qs["lo"] <= 0,
      f"distance only, gradient maps, sea over the south half of the 12 windows that are the controls of the 12 inland windows; "
      f"inland windows: without the support rule d {T.fmt(qn)}, a false PASS {qn['lo'] > 0} (control AUC "
      f"{np.nanmean(An['auc_b'][inl]):.3f} against own {np.nanmean(An['auc_a'][inl]):.3f}: sea ties pull the control to 0.5); with "
      f"it {T.fmt(qs)}, no PASS {qs['lo'] <= 0} (control {np.nanmean(As['auc_b'][inl]):.3f}, own {np.nanmean(As['auc_a'][inl]):.3f}; "
      f"{int((As['n_far'] - As['n_keep'])[inl].sum()):,} far holes dropped)")


def plant_b1a(T, bnm, d, seed=36):
    """A planted B-1a folder for the planted windows: land = tetrads surveyed in any year by any stack of the window, a smooth
    map z-scored over land and 0 elsewhere, control (k + n/2) mod n, a report with gates PASS."""
    G, NC = T.G, T.NC
    fb = dict(np.load(bnm / "bnm_frozen.npz"))
    _, _, Ef, _, _ = T.GB.load_obs(bnm, fb["files"], fb["file_index"], fb["row"])
    win = dict(np.load(bnm / "bnm_stacks.npz"))["window"].astype(str)
    names = np.array(sorted(set(win.tolist())))
    g = np.random.default_rng(seed)
    z, land = np.zeros((len(names), G, G)), np.zeros((len(names), G, G), bool)
    for k, nm in enumerate(names):
        land[k] = Ef[win == nm].any((0, 1))
        f = smooth(g, G)
        z[k] = np.where(land[k], (f - f[land[k]].mean()) / f[land[k]].std(), 0.0)
    d.mkdir(parents=True)
    arrays = dict(window_names=names, z=z, scored=land, land=land, control_index=((np.arange(len(names)) + len(names) // 2) % len(names)).astype(np.int64))
    T.X.save_npz(d / "b1a_habitat.npz", arrays)
    mh = T.X.file_md5(d / "b1a_habitat.npz")
    (d / "b1a_report.json").write_text(json.dumps(dict(gates={f"J{i}": True for i in range(5)}, md5=dict(habitat=mh)), indent=1))
    return dict(habitat=mh, report=T.X.file_md5(d / "b1a_report.json"))


def checks(T, w, tmp, t):
    """T1-T4 on a planted fit half w (test_25's build_fit)."""
    quiet = lambda *a, **k: None
    tmp = Path(tmp)
    data = Path(w["bnm"]).parent
    b1a = tmp / "stage_b1a_habitat"
    m1a = plant_b1a(T, Path(w["bnm"]), b1a)
    kw = dict(bnm=w["bnm"], h2a=w["h2a"], h2b=w["h2b"], b1a=b1a, g2=w["g2"], g3=w["g3"], freeze=w["freeze"], runs=w["runs"],
              obj1_tools=w["obj1_tools"], data=data, freeze_md5=w["freeze_md5"], h6_md5=w["h6_md5"], b1a_md5=m1a,
              frozen_md5=T.X.file_md5(data / "stage_d_frozen_val.npz"), floor_override=w["floor_override"])
    lines = []
    r, rep = T.run(out_dir=tmp / "b1b", out=lines.append, **kw)
    t("T1", all(r.values()) and set(r) == {f"W{i}" for i in range(7)} and (tmp / "b1b" / "b1b_report.json").is_file()
      and "passed" in rep.get("rule", {}) and lines[-1].startswith("stage B-1b: gates 7 PASS"),
      f"planted fit half: {r}; last line: {lines[-1][:160]}")
    if not (tmp / "b1b" / "b1b_scores.npz").is_file():
        t("T2", False, "skipped: T1 wrote no b1b_scores.npz")
        return lines
    # T2 BNM: own support rule and pair loop
    F = T.F0.load_fit(Path(w["bnm"]), Path(w["h2a"]), Path(w["h2b"]))
    with np.load(Path(w["h2b"]) / "h2b_predicted.npz") as P_:
        cell = P_["hole_cell"].astype(np.int64)
    fb = dict(np.load(Path(w["bnm"]) / "bnm_frozen.npz"))
    V1, V2, _, _, _ = T.GB.load_obs(Path(w["bnm"]), fb["files"], fb["file_index"], fb["row"])
    Rk = (V1 | V2).any(1).reshape(F["n"], -1)
    B = dict(np.load(b1a / "b1a_habitat.npz"))
    names = B["window_names"].astype(str)
    z, scd, ctl = B["z"].reshape(len(names), -1), B["scored"].reshape(len(names), -1), B["control_index"]
    sc = dict(np.load(tmp / "b1b" / "b1b_scores.npz"))
    rk = np.flatnonzero(~np.isnan(sc["auc_own"]))
    pick = rk[np.random.default_rng(5).choice(len(rk), min(60, len(rk)), replace=False)]
    ok_a = True
    for i in pick:
        a, b = F["ptr"][i], F["ptr"][i + 1]
        k = int(np.flatnonzero(names == F["win"][i])[0]); kc = int(ctl[k])
        sup = scd[k] & scd[kc]
        c = cell[a:b]; keep = (F["dist"][a:b] >= 3 - 1e-4) & sup[c]
        rr = Rk[i] & sup
        for zz, key in ((z[k], "auc_own"), (z[kc], "auc_control")):
            s_ = -np.abs(zz[c[keep]] - zz[rr].mean())
            ok_a &= abs(T.brute_auc(s_, F["lab"][a:b][keep]) - sc[key][i]) < 1e-12
    js = json.loads((tmp / "b1b" / "b1b_report.json").read_text())
    dd = sc["auc_own"] - sc["auc_control"]
    m = sc["sparse"].astype(bool) & ~np.isnan(dd)
    per = {}
    for s_, v_ in zip(sc["stack_id"][m], dd[m]):
        per.setdefault(int(s_), []).append(v_)
    own = float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")
    rep_m = js["rule"]["sparse_d"]["mean"]
    ok_s = (not per and rep_m is None) or abs(own - rep_m) < 1e-9
    # T2 real_val: own hole draw on the first 20 items
    fzv = dict(np.load(data / "stage_d_frozen_val.npz"))
    shown = np.flatnonzero(fzv["blank"] == 0)
    rvz = dict(np.load(tmp / "b1b" / "b1b_realval.npz"))
    RV1, RV2, REf, _, _ = T.GB.load_obs(data / "real_val", fzv["files"], fzv["file_index"][shown], fzv["row"][shown])
    RVv = (RV1 | RV2).reshape(len(shown), RV1.shape[1], -1)
    ok_r = True
    for j in range(min(20, len(rvz["frozen_row"]))):
        fr, rp = int(rvz["frozen_row"][j]), int(rvz["repeat"][j])
        i = int(np.flatnonzero(shown == fr)[0])
        eff = REf[i, -1].ravel(); cells = np.flatnonzero(eff)
        kh = (3 * len(cells) + 5) // 10
        h = np.sort(np.random.default_rng([20261108, 31401, fr, rp]).choice(cells, kh, replace=False))
        kept = RVv[i].copy(); kept[:, h] = False
        R6 = kept.any(0)
        ky, kx = np.divmod(np.flatnonzero(R6), T.G); hy, hx = np.divmod(h, T.G)
        dmin = np.sqrt(((hy[:, None] - ky[None]) ** 2 + (hx[:, None] - kx[None]) ** 2).min(1))
        ok_r &= len(h) == int(rvz["holes"][j]) and int((dmin >= 3).sum()) == int(rvz["far_holes"][j]) \
            and int(T.D.bidx(R6.sum())) == int(rvz["rbin"][j]) and int(T.D.tidx(kept[-1].sum())) == int(rvz["rcls"][j])
    t("T2", ok_a and ok_s and ok_r, f"recomputed outside the tool: AUC of both arms on {len(pick)} items by a pair loop with an own "
      f"support rule {ok_a}; S mean of d {own:+.6f} by an own stack-first loop = the report {ok_s}; real_val holes of 20 items "
      f"by an own draw (base seed 20261108), with their far holes and kept-record groups {ok_r}")
    stop = []
    for f_ in (lambda: T.run(out_dir=tmp / "b1b", out=quiet, **kw), lambda: T.run(out_dir=Path(w["h2b"]) / "x", out=quiet, **kw),
               lambda: T.run(out_dir=b1a / "x", out=quiet, **kw)):
        try:
            f_(); stop.append(False)
        except SystemExit:
            stop.append(True)
    t("T3", all(stop), f"refused: a second run into the same folder {stop[0]}, an output inside the H2b folder {stop[1]}, inside "
      f"the B-1a folder {stop[2]}")
    for bug, g_ in (("flip", "W1"), ("self_ctrl", "W1"), ("labels_shift", "W1"), ("one_arm", "W3"), ("near", "W3"),
                    ("rv_holes_rec", "W4"), ("test_path", "W5"), ("rv_truth", "W5")):
        os.environ["STAGE_B1B_BROKEN"] = bug
        try:
            rb, _ = T.run(out_dir=tmp / f"b_{bug}", out=quiet, **kw)
        finally:
            os.environ.pop("STAGE_B1B_BROKEN", None)
        t(f"bug {bug}", rb.get(g_) is False and not (tmp / f"b_{bug}").exists(), f"{g_} fails and nothing is written ({rb})")
    bx = tmp / "b1a_changed"
    shutil.copytree(b1a, bx)
    zz = dict(np.load(bx / "b1a_habitat.npz")); zz["z"] = zz["z"] * 1.01
    T.X.save_npz(bx / "b1a_habitat.npz", zz)
    rb, _ = T.run(out_dir=tmp / "b_b1a", out=quiet, **dict(kw, b1a=bx))
    t("bug b1a_changed", rb.get("W0") is False and len(rb) == 1, f"a B-1a file changed after its report fails W0 and stops ({rb})")
    return lines


def self_test(tool_path, obj1):
    T = load_tool(tool_path)
    p25 = Path(tool_path).resolve().parent.parent / "tests" / "test_25_f0a_spread_check.py"
    spec = importlib.util.spec_from_file_location("test_25_b", p25); t25 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t25)
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    unit_checks(T, t)
    with tempfile.TemporaryDirectory() as tmp:
        w, ok = t25.build_fit(T, tmp, obj1, tool_path)
        t("S0", ok, "planted pipeline: G3, H1 fit half, H2a parts, H2b (floor 5), freeze built as H3 writes it")
        lines = checks(T, w, tmp, t)
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return n_ok == len(res_all)
