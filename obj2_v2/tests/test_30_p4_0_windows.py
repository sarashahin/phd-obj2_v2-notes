#!/usr/bin/env python3
"""
test_30_p4_0_windows.py  (obj2_v2, P4 step P4-0, 13 Nov 2026): self-test of tools/stage_p4_0_windows.py.
Run through the tool, from PSD_Dispersal_pool/obj2_v2/work:
    python3 ../tools/stage_p4_0_windows.py --self-test
It builds a planted world in a temporary folder: BNM per-year grids (export_obj2.plant_grids, 6 fit-half windows), a
planted real_val and real_eval export made with export_obj2's survey rule (one window per community-round, each stack the
6-year known cells of one species row), frozen sets, model samples and B3-B5 maps in the G1 / G2 / G5c files, and a
planted Land Cover Map around the windows (open sea and shore-only tetrads among never-surveyed tetrads, one surveyed
tetrad shore-only). Nothing of ../data or ../runs is read or written.
    S0  the planted world: every window of a round holds the effort of its stacks; records inside effort
    T1  P4-0 on it: gates W0-W6 PASS
    T2  independent check, fresh code (map read with rasterio, grids read with np.load): LCM land, surveyed and L_w per
        window = p4_windows.npz; window of every frozen stack = the planted window (real_val and real_eval); the W3
        counts by a plain loop
    T3  meaning: windows with tetrads outside L_w = the windows given sea; the shore-only surveyed tetrad is put back as
        land; every frozen stack's effort lies inside L_w
    T4  a second run into a new folder gives the same p4_windows.npz md5
    T5  refused: an existing folder, a folder inside an input folder, a real_eval path; W0 fails and nothing is written when
        a G1 samples file is changed after its report
    bugs (each must fail its gate, nothing written): map grid column-flipped (W3); L_w without the surveyed part (W3);
        window shifted by one (W2); a real_eval truth file opened (W5); seed 1 maps transposed (W4); B5 rows shifted (W4);
        an input md5 not equal to its record (W0)
"""
import csv, hashlib, importlib.util, json, os, tempfile, warnings
from pathlib import Path
import numpy as np

GROUPS = [[1], [2], [3], [4], [5, 6, 7, 8], [9, 10, 11, 12], [13], [14], [15, 16, 17, 18, 19], [20, 21]]
LANDCL = [c for c in range(1, 22) if c not in (13, 14, 17, 18)]
G, W, K = 20, 6, 50


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def en(nm):
    s = Path(str(nm)).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])


def write_map(d, T, west, north):
    """The three GB map files from a (21, H, W) uint8 array (as test_29): aggregate = group sums; dominant from 22 values."""
    import rasterio
    from rasterio.transform import from_origin
    d.mkdir(parents=True, exist_ok=True)
    A = np.stack([T[[c - 1 for c in g]].astype(int).sum(0) for g in GROUPS]).astype(np.uint8)
    V = np.vstack([np.clip(100 - T.astype(int).sum(0), 0, None)[None], T.astype(int)])
    Dm = np.where(T.astype(int).sum(0) > 0, V.argmax(0), 0).astype(np.uint8)
    kw = dict(driver="GTiff", height=T.shape[1], width=T.shape[2], crs="EPSG:27700", transform=from_origin(west, north, 1000, 1000), dtype="uint8")
    for nm, arr in (("gb2019lcm1km_percentage_target.tif", T), ("gb2019lcm1km_percentage_aggregate.tif", A),
                    ("gb2019lcm1km_dominant_target.tif", Dm[None])):
        with rasterio.open(d / nm, "w", count=arr.shape[0], **kw) as f:
            f.write(arr)


def plant_map(wins, d, seed=9):
    """Land everywhere; in each window but the last, never-surveyed tetrads of columns 10+ are open sea, of rows 0, 3, ...
    shore only; in the first window the most surveyed tetrad is shore only; the last window is all land. Returns the
    planted LCM no-land per window and that tetrad."""
    E = [en(n)[0] for n, _ in wins]; N = [en(n)[1] for n, _ in wins]
    wk, nk = min(E) - 4, max(N) + 44
    H, Wd = nk - (min(N) - 4), (max(E) + 44) - wk
    rng = np.random.default_rng(seed)
    T = np.zeros((21, H, Wd), np.uint8)
    for r in range(H):
        for c in range(Wd):
            k1 = (r * Wd + c) % 21 + 1
            k2, k3 = rng.choice([x for x in LANDCL if x != k1], 2, replace=False)
            T[[k1 - 1, k2 - 1, k3 - 1], r, c] = (60, 30, 10)
    noland, shore = {}, None
    for wi, (nm, Ew) in enumerate(wins):
        e, n = en(nm); S = Ew.any((0, 1)); nl = np.zeros((G, G), bool)
        pick = np.unravel_index(np.argmax(Ew.sum((0, 1))), (G, G)) if wi == 0 else None
        for tr_ in range(G):
            for tc in range(G):
                cells = [(nk - (n + 2 * tr_ + dn + 1), e + 2 * tc + de - wk) for dn in (0, 1) for de in (0, 1)]
                if pick is not None and (tr_, tc) == tuple(pick):
                    for rr, cc in cells:
                        T[:, rr, cc] = 0; T[17, rr, cc] = 60; T[12, rr, cc] = 20
                    nl[tr_, tc] = True; shore = (wi, tr_, tc); continue
                if S[tr_, tc] or wi == len(wins) - 1:
                    continue
                if tc >= 10 or tr_ % 3 == 0:
                    for rr, cc in cells:
                        T[:, rr, cc] = 0
                        if tc < 10:
                            T[17, rr, cc] = 60; T[12, rr, cc] = 20
                    nl[tr_, tc] = True
        noland[nm] = nl
    write_map(d, T, wk * 1000, nk * 1000)
    return noland, shore, (wk, nk)


def plant_split(X, d, split, seeds, wins, code, g):
    """export_obj2's survey rule on planted ranges: per community-round one window, per stack one species row."""
    d.mkdir(parents=True)
    rows, truth_win = [], {}
    for s in seeds:
        rw, eff, v1, v2, hab, maps = [], [], [], [], [], []
        R = 12
        for i in range(R):
            w = int(g.integers(len(wins))); rw.append(w)
            Ew = wins[w][1]
            for _ in range(int(g.integers(8, 14))):
                bsp = int(g.integers(Ew.shape[1]))
                E = Ew[:, bsp]
                cy, cx, rad = g.integers(G), g.integers(G), g.uniform(0.8, 6)
                yy, xx = np.mgrid[0:G, 0:G]
                base = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
                M = np.stack([base & (g.random((G, G)) < 0.9) for _ in range(W)])
                PE = M & E
                v1.append(PE & (g.random(PE.shape) < 0.55)); v2.append(PE & (g.random(PE.shape) < 0.55))
                eff.append(E); hab.append(i); maps.append(M)
        eff, v1, v2, maps = map(np.array, (eff, v1, v2, maps))
        n = len(hab)
        inp = dict(code_md5=np.array(code), effort=X.pack(eff), habitat_noisy=np.zeros((R, G, G), np.float16),
                   n_rec=(v1 | v2).reshape(n, W, -1).sum(2).astype(np.int16), pair_md5=np.array("p"), stack_hab=np.array(hab, np.int32),
                   visit1=X.pack(v1), visit2=X.pack(v2))
        tru = dict(round_window=np.array(rw, np.int32), window_names=np.array([nm for nm, _ in wins]), stack_round=np.array(hab, np.int32),
                   true_maps=X.pack(maps), split=np.array(split))
        tp, ip = d / f"s{s}_truth.npz", d / f"s{s}_input.npz"
        X.save_npz(tp, tru); X.save_npz(ip, inp)
        rows.append(dict(sim_seed=s, source=f"c{s}", source_md5="m", rounds=R, skipped=0, stacks=n, truth_file=tp.name,
                         truth_md5=md5(tp), input_file=ip.name, input_md5=md5(ip), bytes=0))
        truth_win[ip.name] = (np.array(hab), np.array(rw))
    with open(d / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    return rows, truth_win


def plant_frozen(X, d, rows, g):
    files, fmd, fi, rw, rt, r6 = [], [], [], [], [], []
    for f, r in enumerate(rows):
        z = np.load(d / r["input_file"]); V = X.unpack(z["visit1"] | z["visit2"])
        files.append(r["input_file"]); fmd.append(md5(d / r["input_file"]))
        t, a = V[:, -1].reshape(len(V), -1).sum(1), V.any(1).reshape(len(V), -1).sum(1)
        for j in np.nonzero(a >= 1)[0]:
            if t[j] <= 1 or g.random() < 0.5:
                fi.append(f); rw.append(j); rt.append(t[j]); r6.append(a[j])
    n = len(rw)
    return dict(files=np.array(files), file_md5=np.array(fmd), file_index=np.array(fi, np.int16), row=np.array(rw, np.int32),
                rec_t=np.array(rt, np.int16), rec6=np.array(r6, np.int16), cls=np.zeros(n, np.int8), blank=np.zeros(n, np.uint8),
                shift=np.zeros((n, 2), np.int8), fast=np.zeros(n, np.uint8), code_md5=np.array("c"), classes=np.array(["0"]))


def plant_maps(X, d, fz, g, kind):
    """(n, K, 300) packed maps in the G1 layout; the target year only. Model: records kept 97%, the range 60%, noise;
    B3, B4: every record on plus noise; B5: the records."""
    n = len(fz["row"]); kk = 1 if kind == "B5" else K
    out = np.zeros((n, kk, W, G, G), bool)
    by = {}
    for f in sorted(set(fz["file_index"].tolist())):
        by[f] = np.load(d / str(fz["files"][f]))
    tz = {}
    for i, (f, r) in enumerate(zip(fz["file_index"], fz["row"])):
        z = by[int(f)]
        rec = X.unpack(z["visit1"][r] | z["visit2"][r])[-1]
        if kind == "model":
            tp = d / str(fz["files"][int(f)]).replace("_input", "_truth")
            tz.setdefault(int(f), np.load(tp)["true_maps"])
            tm = X.unpack(tz[int(f)][r])[-1]
            U = g.random((kk, G, G))
            out[i, :, -1] = (rec & (U < 0.97)) | (tm & (g.random((kk, G, G)) < 0.6)) | (g.random((kk, G, G)) < 0.02)
        elif kind == "B5":
            out[i, :, -1] = rec
        else:
            out[i, :, -1] = rec | (g.random((kk, G, G)) < 0.1)
    return np.packbits(out.reshape(n, kk, W * G * G), axis=-1)


def build(tool, tmp, obj1_tools):
    X, GS = tool.X, tool.GS
    g = np.random.default_rng(30)
    py, refpat = X.plant_grids(str(tmp))
    wins, _, _ = X.load_bnm(py, refpat, obj1_tools, out=lambda *a, **k: None)
    data, runs = tmp / "data", [tmp / "runs" / "f_seed1", tmp / "runs" / "f_seed2"]
    g2, g5c, lcm = data / "stage_g2_baselines_v2", data / "stage_g5_predict", tmp / "lcm"
    code = X.file_md5(tool.HERE / "export_obj2.py")
    rv, tv = plant_split(X, data / "real_val", "real_val", (8101, 8102), wins, code, g)
    re_, te = plant_split(X, data / "real_eval", "real_eval", (8201, 8202), wins, code, g)
    fzv, fze = plant_frozen(X, data / "real_val", rv, g), plant_frozen(X, data / "real_eval", re_, g)
    X.save_npz(data / "stage_d_frozen_val.npz", fzv)
    g2.mkdir(parents=True); g5c.mkdir(parents=True)
    X.save_npz(g5c / "frozen_eval.npz", fze)
    mv, me_ = md5(data / "stage_d_frozen_val.npz"), md5(g5c / "frozen_eval.npz")
    parts = {"frozen": {"frozen_eval.npz": me_}, "baselines": {}}
    g2rep = {}
    for b, f in (("B3", "maps_B3_halfblank.npz"), ("B4", "maps_B4.npz"), ("B5", "maps_B5.npz")):
        for dd, fz, sp, fm in ((g2, fzv, "real_val", mv), (g5c, fze, "real_eval", me_)):
            mp = plant_maps(X, data / sp, fz, g, b)
            X.save_npz(dd / f, dict(maps=mp, sel=np.arange(len(fz["row"])), K=np.int64(mp.shape[1]), layout=np.array(GS.LAYOUT),
                                    frozen_md5=np.array(fm), aoo_t=np.zeros((len(mp), mp.shape[1]), np.uint16)))
        g2rep[f] = md5(g2 / f); parts["baselines"][f] = md5(g5c / f)
    (g2 / "g2_report.json").write_text(json.dumps(dict(md5=g2rep)))
    for r in runs:
        (r / "check_g1").mkdir(parents=True)
        mp = plant_maps(X, data / "real_val", fzv, g, "model")
        X.save_npz(r / "check_g1" / "samples_k50.npz", dict(maps=mp, sel=np.arange(len(mp)), K=np.int64(K), layout=np.array(GS.LAYOUT)))
        (r / "check_g1" / "g1_report.json").write_text(json.dumps(dict(md5=dict(samples=md5(r / "check_g1" / "samples_k50.npz"), best_pt="b"))))
        mp = plant_maps(X, data / "real_eval", fze, g, "model")
        X.save_npz(g5c / f"samples_{r.name}.npz", dict(maps=mp, sel=np.arange(len(mp)), K=np.int64(K), layout=np.array(GS.LAYOUT),
                                                          frozen_md5=np.array(me_)))
        parts[f"samples_{r.name}"] = {f"samples_{r.name}.npz": md5(g5c / f"samples_{r.name}.npz")}
    for p, fs in parts.items():
        (g5c / f"part_{p}.json").write_text(json.dumps(dict(gates={"R0": True, "R4": True}, files=fs)))
    noland, shore, corner = plant_map(wins, lcm)
    return dict(py=py, refpat=refpat, wins=wins, data=data, runs=runs, g2=g2, g5c=g5c, lcm=lcm, fzv=fzv, fze=fze, tv=tv, te=te,
                noland=noland, shore=shore, corner=corner)


def main(tool_path, obj1_tools="../../obj1_v2/tools"):
    warnings.filterwarnings("ignore", message="Mean of empty slice")          # band check on a planted map with no mismatching cell
    warnings.filterwarnings("ignore", message="invalid value encountered")
    tool = load(tool_path, "stage_p4_0_windows")
    X = tool.X
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td); w = build(tool, tmp, obj1_tools)
        expect = dict(lcm=md5(w["lcm"] / "gb2019lcm1km_percentage_target.tif"), band_check=X.file_md5(tool.HERE / "stage_p3c_band_check.py"))
        names = [n for n, _ in w["wins"]]

        def go(out, env="", **kw):
            os.environ["STAGE_P4_0_BROKEN"] = env
            lines = []
            try:
                a = dict(data=w["data"], runs=w["runs"], g2=w["g2"], g5c=w["g5c"], lcm=w["lcm"], peryear=w["py"], split_ref=w["refpat"],
                         obj1_tools=obj1_tools, out_dir=out, expect=expect, out=lines.append)
                a.update(kw)
                r, rep = tool.run(**a)
            finally:
                os.environ["STAGE_P4_0_BROKEN"] = ""
            return r, rep, lines

        # S0 planted world
        ok0 = True
        for sp, tw in (("real_val", w["tv"]), ("real_eval", w["te"])):
            for f, (hab, rw) in tw.items():
                z = np.load(w["data"] / sp / f); E = X.unpack(z["effort"]); V = X.unpack(z["visit1"] | z["visit2"])
                ok0 &= not (V & ~E).any()
                for j in range(len(hab)):
                    Ew = w["wins"][rw[hab[j]]][1]
                    ok0 &= any(np.array_equal(E[j], Ew[:, s]) for s in range(Ew.shape[1]))
        t("S0", ok0 and len(w["fzv"]["row"]) > 50 and len(w["fze"]["row"]) > 50,
          f"planted world: records inside effort, every stack's effort = a species row of its round's window; frozen stacks "
          f"real_val {len(w['fzv']['row'])}, real_eval {len(w['fze']['row'])}")

        # T1
        r1, rep1, l1 = go(tmp / "out1")
        t("T1", r1 and all(r1.values()) and set(r1) == {f"W{i}" for i in range(7)}, f"P4-0 on the planted world: {r1}")
        if not (r1 and all(r1.values())):
            print("\n".join("      " + s for s in l1))
            print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL"); return False
        z = dict(np.load(tmp / "out1" / "p4_windows.npz"))

        # T2 independent check
        import rasterio
        with rasterio.open(w["lcm"] / "gb2019lcm1km_percentage_target.tif") as s_:
            A = s_.read().astype(int)
        lc = A[[c - 1 for c in LANDCL]].sum(0)
        wk, nk = w["corner"]
        ll, sv = np.zeros((len(names), G, G), bool), np.zeros((len(names), G, G), bool)
        for wi, nm in enumerate(names):
            e, n = en(nm)
            for r_ in range(G):
                for c_ in range(G):
                    ll[wi, r_, c_] = max(lc[nk - (n + 2 * r_ + dn + 1), e + 2 * c_ + de - wk] for dn in (0, 1) for de in (0, 1)) > 0
            yrs = sorted(Path(w["py"]).glob("minlist8_y*"))
            Es = [np.load(y / nm)["known_holdout"].astype(bool) for y in yrs] + [np.load(Path(w["py"]) / "minlist8" / nm)["known_target"].astype(bool)]
            sv[wi] = np.any([e_.any(0) for e_ in Es], 0)
        land_ok = np.array_equal(z["land_lcm"], ll) and np.array_equal(z["surveyed"], sv) and np.array_equal(z["land"], ll | sv) \
            and list(z["window_names"]) == names
        win_ok, cnt_ok = True, True
        for key, sp, fz, tw in (("val_window", "real_val", w["fzv"], w["tv"]), ("eval_window", "real_eval", w["fze"], w["te"])):
            want = [tw[str(fz["files"][f])][1][tw[str(fz["files"][f])][0][r]] for f, r in zip(fz["file_index"], fz["row"])]
            win_ok &= np.array_equal(z[key], np.array(want))
            tot = noland = 0
            for f, r, wi in zip(fz["file_index"], fz["row"], z[key]):
                Ez = X.unpack(np.load(w["data"] / sp / str(fz["files"][f]))["effort"][r]).any(0)
                for r_ in range(G):
                    for c_ in range(G):
                        if Ez[r_, c_]:
                            tot += 1; noland += int(not ll[wi, r_, c_])
            cnt_ok &= rep1["w3"][sp]["total"] == tot and rep1["w3"][sp]["orient"]["identity"] == noland
        t("T2", land_ok and win_ok and cnt_ok, f"fresh code: LCM land, surveyed and L_w per window equal {land_ok}; window of every "
          f"frozen stack = the planted window (both splits) {win_ok}; W3 effort and no-land counts by a plain loop {cnt_ok}")

        # T3 meaning
        planted = np.array([w["noland"][nm] for nm in names])
        wi, tr_, tc = w["shore"]
        coast = [not z["land"][i].all() for i in range(len(names))]
        t("T3", np.array_equal(~z["land_lcm"], planted) and not z["land_lcm"][wi, tr_, tc] and z["land"][wi, tr_, tc]
          and coast == [bool((planted[i] & ~sv[i]).any()) for i in range(len(names))]
          and rep1["w3"]["real_val"]["outside_land"] == 0 and rep1["w3"]["real_eval"]["outside_land"] == 0,
          f"no-land tetrads = the planted sea and shore; the shore-only surveyed tetrad ({tr_}, {tc}) of {names[wi]} is put back as "
          f"land; windows with tetrads outside L_w {sum(coast)}; effort outside L_w 0")

        # T4 rerun
        go(tmp / "out2")
        t("T4", md5(tmp / "out1" / "p4_windows.npz") == md5(tmp / "out2" / "p4_windows.npz"), "a second run gives the same p4_windows.npz md5")

        # T5 refusals and a changed input
        stops = []
        for out in (tmp / "out1", w["data"] / "real_val" / "o", tmp / "real_eval" / "o"):
            try:
                go(out); stops.append(False)
            except SystemExit:
                stops.append(True)
        sp_ = w["runs"][0] / "check_g1" / "samples_k50.npz"; keep = sp_.read_bytes()
        zz = dict(np.load(sp_)); zz["maps"] = zz["maps"].copy(); zz["maps"][0, 0, -1] ^= 1; X.save_npz(sp_, zz)
        r5, _, _ = go(tmp / "out5")
        sp_.write_bytes(keep)
        t("T5", all(stops) and r5.get("W0") is False and not (tmp / "out5").exists(),
          f"refused: existing folder {stops[0]}, inside an input folder {stops[1]}, a real_eval path {stops[2]}; a G1 samples file "
          f"changed after its report: W0 FAIL, nothing written")

        # planted bugs
        for i, (bug, g_) in enumerate((("orient", "W3"), ("no_effort_land", "W3"), ("win_shift", "W2"), ("eval_truth", "W5"),
                                       ("maps_transpose", "W4"), ("b5_roll", "W4"), ("changed_input", "W0"))):
            rb, _, _ = go(tmp / f"bugout{i}", env=bug)
            t(f"bug {bug}", rb.get(g_) is False and not (tmp / f"bugout{i}").exists(), f"{g_} fails and nothing is written")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)


if __name__ == "__main__":
    import sys
    tp = Path(__file__).resolve().parent.parent / "tools" / "stage_p4_0_windows.py"
    sys.exit(0 if main(tp, sys.argv[1] if len(sys.argv) > 1 else "../../obj1_v2/tools") else 1)