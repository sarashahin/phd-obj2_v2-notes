#!/usr/bin/env python3
"""
test_32_p4_2_eval_reading.py  (obj2_v2, P4 step P4-2, 14 Nov 2026): self-test of tools/stage_p4_2_eval_reading.py.
Run through the tool, from PSD_Dispersal_pool/obj2_v2/work:
    python3 ../tools/stage_p4_2_eval_reading.py --self-test
It builds test_31's planted world (test_30's grids, exports, frozen sets, G1 / G2 / G5c files and Land Cover Map; the real
P4-0; planted G3, G3b and G5a outputs from those tools' own functions), runs the real P4-1 with rule E, then plants what
G5d left on real_eval: g5c_report.json, g5b_export_check.json, g5d_predicted.npz (G5a's apply_all on the window AOO),
g5d_report.json, per_stack_g5d.csv and the marker with one line. The planted real_eval truth files get round ids that are
not the round indices (as in the real export), so a window read through the wrong key fails. Nothing of ../data or ../runs
is read or written.
    U1  G5d's p_eval with the fold reference = G3b's evaluate (every score, exactly)
    U2  apply_land with G5a's fits in P4-1's layout = G5a's apply_all, AOO part (exactly)
    S0  planted world: P4-0 and P4-1 gates PASS; G5d's outputs planted
    T1  check-only: Z0-Z4 PASS; the marker unchanged; nothing written
    T2  the reading: Z0-Z6 PASS; the marker gets one line, labelled P4-2, reading 2; report and csv written
    T3  independent check, fresh code (truth unpacked, plain sums): land-only truth and seed 1 land-only AOO; ratio and
        coverage per recorded class from the csv = the report rows; every verdict = the rule applied by hand; seed 1 option
        B Brier skill at x 5 from the csv = the report
    T4  refused: a second reading without --rerun (nothing written, marker unchanged), no --second-reading, an output inside
        real_eval, an existing folder; --rerun gives reading 3, labelled rerun 1, with the same predicted md5 and rows
    T5  meaning: window truth = G5d's per stack; in all-land windows land-only truth = window truth, smaller somewhere in
        coastal windows
    bugs before the marker (gate fails, no marker line, nothing written): input changed (Z1), seed 1 maps shifted (Z2),
        windows shifted (Z2), apply code changed (Z3), wrong Brier reference (Z3), a truth file opened early (Z4);
        after the marker (gate fails): truth rows shifted (Z5), predicted file changed after the truth opened (Z6)
"""
import csv, hashlib, importlib.util, json, os, tempfile, warnings
from pathlib import Path
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def plant_g5d(tool, w, gp):
    """What G5c, G5b and G5d left on real_eval, from G5a's own apply code."""
    P41, G5F, GC, D, X = tool.P41, tool.G5F, tool.GC, tool.D, tool.X
    data, g5c, ev = w["data"], w["g5c"], w["data"] / "real_eval"
    man = list(csv.DictReader(open(ev / "manifest.csv")))
    for r in man:                                             # round ids that are not the round indices, as in the real export
        tp = ev / r["truth_file"]
        z = dict(np.load(tp))
        R = len(z["round_window"]); rid = (100 + 7 * np.arange(R)).astype(np.int32)
        z["round_ids"] = rid; z["stack_round"] = rid[z["stack_round"]]
        X.save_npz(tp, z); r["truth_md5"] = md5(tp)
    with open(ev / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(man[0])); wr.writeheader(); wr.writerows(man)
    fe = dict(np.load(g5c / "frozen_eval.npz")); n = len(fe["row"])
    fi, rw = fe["file_index"].astype(int), fe["row"].astype(int)
    by = {r["input_file"]: r for r in man}
    V = np.zeros((n, 6, 20, 20), bool); T = np.zeros((n, 20, 20), bool)
    for f in sorted(set(fi.tolist())):
        k = np.nonzero(fi == f)[0]; nm = str(fe["files"][f])
        zi = np.load(ev / nm); V[k] = X.unpack(zi["visit1"][rw[k]] | zi["visit2"][rw[k]])
        T[k] = X.unpack(np.load(ev / by[nm]["truth_file"])["true_maps"][rw[k], -1])
    rec_t = V[:, -1].reshape(n, -1).sum(1); rcls, rbin = D.tidx(rec_t), D.bidx(V.any(1).reshape(n, -1).sum(1))
    f5 = G5F.load_fits(gp["g5"] / "g5_fits.npz")
    maps = {f"model_{r.name}": np.load(g5c / f"samples_{r.name}.npz")["maps"] for r in w["runs"]}
    for b, f in (("B3", "maps_B3_halfblank.npz"), ("B4", "maps_B4.npz"), ("B5", "maps_B5.npz")):
        maps[b] = np.load(g5c / f)["maps"]
    k1, k2 = [f"model_{r.name}" for r in w["runs"]]
    met = {k: GC.map_metrics(v) for k, v in maps.items()}
    met["pooled"] = tuple(np.concatenate([met[k1][q][:, :25], met[k2][q][:, :25]], 1) for q in (0, 1))
    save = dict(rec_t=rec_t.astype(np.int16), rcls=rcls.astype(np.int8), rbin=rbin.astype(np.int8))
    for k in (k1, k2, "pooled", "B3", "B4", "B5"):
        pr = G5F.apply_all(f5, k, met[k][0], met[k][1], rec_t, rcls, rbin)
        save[f"aoo_{k}"] = met[k][0].astype(np.uint16)
        for q in ("mean", "L", "U"):
            save[f"aoo_{q}_{k}"] = pr["aoo"][q]
        for q in ("p_g3", "p_recal", "p_b"):
            save[f"{q}_{k}"] = pr[q]
    g5d = data / "stage_g5d_score"; g5d.mkdir()
    X.save_npz(g5d / "g5d_predicted.npz", save)
    pfiles = ["frozen_eval.npz", "maps_B3_halfblank.npz", "maps_B4.npz", "maps_B5.npz"] + [f"samples_{r.name}.npz" for r in w["runs"]]
    pm = {f: md5(g5c / f) for f in pfiles}
    (g5c / "g5c_report.json").write_text(json.dumps(dict(gates={"R0": True, "R5": True}, truth_opened=0, files=pm)))
    g5b = data / "stage_g5b_export_check"; g5b.mkdir()
    (g5b / "g5b_export_check.json").write_text(json.dumps(dict(gates={"E0": True}, md5=dict(
        manifest=md5(ev / "manifest.csv"), files={f: md5(ev / f) for r in man for f in (r["input_file"], r["truth_file"])}))))
    mp = md5(g5d / "g5d_predicted.npz")
    (g5d / "g5d_report.json").write_text(json.dumps(dict(md5=dict(predicted=mp, fits=md5(gp["g5"] / "g5_fits.npz"), predictions=pm),
                                                         intervals=dict(rule={}), p_le={})))
    with open(g5d / "per_stack_g5d.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["stack", "file_index", "row", "aoo_true"])
        for i in range(n):
            wr.writerow([i, fi[i], rw[i], int(T[i].sum())])
    (ev / "G5D_TRUTH_OPENED").write_text(json.dumps(dict(opened_utc="2026-11-11T22:00:00+00:00", tool_md5="t", out=str(g5d),
                                                          predicted_md5=mp, reading=1)) + "\n")
    return dict(g5d=g5d, g5b=g5b, g5a=gp["g5"], T=T, rec_t=rec_t, rcls=rcls, n=n, fi=fi, rw=rw, man=man)


def main(tool_path, obj1_tools="../../obj1_v2/tools"):
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    here = Path(tool_path).resolve().parent
    tool = load(tool_path, "stage_p4_2_eval_reading")
    tool4 = load(here / "stage_p4_0_windows.py", "stage_p4_0_windows")
    T30 = load(here.parent / "tests" / "test_30_p4_0_windows.py", "test_30_p4_0_windows")
    T31 = load(here.parent / "tests" / "test_31_p4_1_land_fits.py", "test_31_p4_1_land_fits")
    P41, G5F, GC, GR, X, D, XS = tool.P41, tool.G5F, tool.GC, tool.GR, tool.X, tool.D, tool.XS
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    g = np.random.default_rng(5); nn = 3000
    y = g.poisson(4, nn).astype(float); rc = g.integers(0, 7, nn); fo = g.integers(0, 2, nn); P = g.random(nn)
    a = tool.p_eval(P, y, rc, tool.fold_ref(y, fo, rc, 5), 5); b = GR.evaluate(P, y, fo, rc, 5)
    t("U1", all(tool.dif(a[q], b[q]) == 0 for q in ("brier", "brier_ref", "bss", "gap", "gap_z", "gap200", "mean_p", "share"))
      and a["per_class"] == b["per_class"], "G5d's p_eval with the fold reference = G3b's evaluate (every score, exactly)")
    pr = y[:, None] + g.uniform(-4, 4, (nn, 50)); rt = np.minimum(g.integers(0, 3, nn), y).astype(int)
    rcl, rbn = D.tidx(rt), g.integers(0, len(D.BINS), nn)
    keys = ["m"]
    corr = {("m", kind): G5F.fit_correction(pr, y, rcl, rbn, 2, np.ones(nn, bool)) for kind in ("aoo", "eoo")}
    rec = {"m": G5F.fit_recal(np.stack([GC.p_le(pr, x) for x in XS], 1), y, rt, rcl, np.ones(nn, bool))}
    f5 = dict(corr=corr, recal=rec)
    A1 = P41.apply_land(tool.g5a_as_p4(f5, keys), "m", pr, rt, rcl, rbn)
    A2 = G5F.apply_all(f5, "m", pr, pr, rt, rcl, rbn)
    t("U2", all(np.array_equal(A1["aoo"][q], A2["aoo"][q]) for q in ("mean", "L", "U")) and all(np.array_equal(A1[q], A2[q])
      for q in ("p_g3", "p_recal", "p_b")), "apply_land with G5a's fits in P4-1's layout = G5a's apply_all, AOO part (exactly)")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        w = T31.build(T30, tool4, tmp, obj1_tools)
        gp = T31.plant_g(P41, w, tmp)
        p40 = w["data"] / "stage_p4_0"
        os.environ["STAGE_P4_1_BROKEN"] = ""
        r1, _ = P41.run(w["data"], w["runs"], w["g2"], gp["g3"], gp["g3b"], gp["g5"], p40, w["data"] / "stage_p4_1",
                        expect=dict(p4_0_report=md5(p40 / "p4_0_report.json"), p4_windows=md5(p40 / "p4_windows.npz"), g3_prefix="", fits_prefix=""),
                        crossed="empty", out=lambda *a_, **k_: None)
        pl = plant_g5d(tool, w, gp)
        p41 = w["data"] / "stage_p4_1"
        t("S0", w["r0"] and all(w["r0"].values()) and r1 and all(r1.values()), f"planted world: P4-0 {w['r0']}; P4-1 (rule E) {r1}; "
          f"G5d's outputs planted ({pl['n']} frozen real_eval stacks)")
        ex = dict(fits=md5(p41 / "p4_fits.npz"), p4_1_report=md5(p41 / "p4_1_report.json"), crossfit=md5(p41 / "p4_1_crossfit.npz"),
                  p4_windows=md5(p40 / "p4_windows.npz"), g5a_prefix="")
        mk = w["data"] / "real_eval" / "G5D_TRUTH_OPENED"; mk0 = mk.read_text()

        def go(out, env="", **kw):
            os.environ["STAGE_P4_2_BROKEN"] = env
            lines = []
            try:
                a_ = dict(data=w["data"], g5c=w["g5c"], g5b=pl["g5b"], g5d=pl["g5d"], g5a=pl["g5a"], p4_0=p40, p4_1=p41, out_dir=out,
                          second_reading=True, expect=ex, out=lines.append)
                a_.update(kw)
                r_, rep_ = tool.run(**a_)
            finally:
                os.environ["STAGE_P4_2_BROKEN"] = ""
            return r_, rep_, lines

        rc_, _, lc = go(tmp / "chk", check_only=True)
        t("T1", rc_ and all(rc_.values()) and set(rc_) == {f"Z{i}" for i in range(5)} and mk.read_text() == mk0 and not (tmp / "chk").exists(),
          f"check-only: {rc_}; marker unchanged, nothing written")
        if not (rc_ and all(rc_.values())):
            print("\n".join("      " + s for s in lc))
        for i, (bug, g_) in enumerate((("changed_input", "Z1"), ("maps_shift", "Z2"), ("win_shift", "Z2"), ("apply_code", "Z3"),
                                       ("p_ref", "Z3"), ("truth_early", "Z4"))):
            rb, _, _ = go(tmp / f"bugout{i}", env=bug)
            t(f"bug {bug}", rb.get(g_) is False and mk.read_text() == mk0 and not (tmp / f"bugout{i}").exists(),
              f"{g_} fails; no marker line, nothing written")
        r2, rep2, l2 = go(tmp / "out1")
        ml = [json.loads(s) for s in mk.read_text().splitlines()]
        okr = r2 and all(r2.values()) and set(r2) == {f"Z{i}" for i in range(7)}
        t("T2", okr and len(ml) == 2 and ml[1]["reading"] == 2 and ml[1]["label"].startswith("P4-2") and ml[1]["fits_md5"] == ex["fits"]
          and (tmp / "out1" / "p4_2_report.json").is_file() and (tmp / "out1" / "per_stack_p4_2.csv").is_file(),
          f"the reading: {r2}; marker line 2 labelled '{ml[-1].get('label', '')[:28]}...', report and csv written")
        if not okr:
            print("\n".join("      " + s for s in l2)); print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL"); return False
        # ---- T3 fresh code
        pz = dict(np.load(p40 / "p4_windows.npz")); win = pz["eval_window"].astype(int); L = pz["land"].astype(bool)[win]
        yl = (pl["T"] & L).sum((1, 2))
        m1 = np.load(w["g5c"] / f"samples_{w['runs'][0].name}.npz")["maps"]
        occ = np.unpackbits(m1[:, :, 250:], axis=-1, count=400).reshape(pl["n"], m1.shape[1], 20, 20).astype(bool)
        zq = dict(np.load(tmp / "out1" / "p4_2_predicted.npz"))
        a_ok = np.array_equal((occ & L[:, None]).sum((2, 3)), zq[f"aoo_land_model_{w['runs'][0].name}"].astype(int))
        rows = list(csv.DictReader(open(tmp / "out1" / "per_stack_p4_2.csv")))
        y_ok = [int(r_["true_land"]) for r_ in rows] == yl.tolist()
        s1n = f"model_{w['runs'][0].name}".replace("_", " ", 1).replace(" ", "_")
        k1 = f"model_{w['runs'][0].name}"
        mean, lo, hi = zq[f"mean_{k1}"], zq[f"L_{k1}"], zq[f"U_{k1}"]
        cls = np.array([r_["recorded_class"] for r_ in rows])
        rr = rep2["rows"][f"model {w['runs'][0].name}"]["recorded class"]
        r_ok = all(abs(mean[cls == q["group"]].mean() / yl[cls == q["group"]].mean() - q["ratio"]) < 1e-12
                   and abs(((yl >= lo) & (yl <= hi))[cls == q["group"]].mean() - q["cov"]) < 1e-12 and int((cls == q["group"]).sum()) == q["n"]
                   for q in rr if yl[cls == q["group"]].mean() > 0)
        v_ok = all(rep2["rule"][k]["ok"] == all(GC.BAND[0] <= q["ratio"] <= GC.BAND[1] and GC.COV[0] <= q["cov"] <= GC.COV[1]
                                                for gname in ("recorded class", "records over 6 years") for q in rep2["rows"][k][gname]
                                                if q["n"] >= GC.N_TEST) for k in rep2["rule"])
        rule_keys = sorted(rep2["rule"]); p4f = P41.load_fits(p41 / "p4_fits.npz")
        pb = np.array([float(r_[f"p_le_5_{s1n}"]) for r_ in rows]); o = (yl <= 5).astype(float)
        ref = p4f["base_rate"][XS.index(5)][pl["rcls"]]
        bss = 1 - ((pb - o) ** 2).mean() / ((ref - o) ** 2).mean()
        b_ok = abs(bss - rep2["p_le"][f"model {w['runs'][0].name} x 5"]["scores"]["b"]["bss"]) < 1e-3
        t("T3", a_ok and y_ok and r_ok and v_ok and b_ok, f"fresh code: seed 1 land-only AOO {a_ok}, land-only truth {y_ok}; ratio and "
          f"coverage per recorded class from the csv = report rows {r_ok}; verdicts by hand {v_ok} ({len(rule_keys)} methods); seed 1 "
          f"option B skill at x 5 from the csv = report {b_ok}")
        # ---- T4 refusals and the rerun
        stops = []
        for kw in (dict(out_dir=tmp / "again"), dict(out_dir=tmp / "x2", second_reading=False),
                   dict(out_dir=w["data"] / "real_eval" / "o", rerun=True), dict(out_dir=tmp / "out1", rerun=True)):
            try:
                go(**{k_: v_ for k_, v_ in kw.items() if k_ != "out_dir"}, out=kw["out_dir"]); stops.append(False)
            except SystemExit:
                stops.append(True)
        mk1 = mk.read_text()
        r3, rep3, _ = go(tmp / "out_rerun", rerun=True)
        ml3 = [json.loads(s) for s in mk.read_text().splitlines()]
        t("T4", all(stops) and not (tmp / "again").exists() and len(mk1.splitlines()) == 2 and r3 and all(r3.values())
          and rep3["rerun"] == 1 and rep3["reading"] == 3 and len(ml3) == 3 and "rerun 1" in ml3[-1]["label"]
          and rep3["md5"]["predicted"] == rep2["md5"]["predicted"]
          and json.dumps(rep3["rows"], default=float) == json.dumps(rep2["rows"], default=float),
          f"refused: second reading without --rerun {stops[0]} (nothing written, marker unchanged), no --second-reading {stops[1]}, "
          f"output inside real_eval {stops[2]}, existing folder {stops[3]}; --rerun: reading 3, rerun 1, same predicted md5 and rows")
        # ---- T5 meaning
        coastal = ~pz["land"].astype(bool).all((1, 2))[win]
        yw = np.array([int(r_["true_window"]) for r_ in rows])
        t("T5", np.array_equal(yw, pl["T"].sum((1, 2))) and np.array_equal(yl[~coastal], yw[~coastal]) and bool((yl[coastal] < yw[coastal]).any()),
          f"window truth = G5d's per stack; all-land windows: land-only = window truth; coastal ({int(coastal.sum())} stacks): smaller in "
          f"{int((yl[coastal] < yw[coastal]).sum())}")
        # ---- bugs after the marker
        for i, (bug, g_) in enumerate((("truth_shift", "Z5"), ("leak", "Z6"))):
            mk.write_text(mk0)
            rb, _, _ = go(tmp / f"bug2out{i}", env=bug)
            t(f"bug {bug}", rb.get(g_) is False and len(mk.read_text().splitlines()) == 2, f"{g_} fails after the marker line")
        mk.write_text(mk0)
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return n_ok == len(res)