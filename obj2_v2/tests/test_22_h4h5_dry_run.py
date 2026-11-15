#!/usr/bin/env python3
"""
test_22_h4h5_dry_run.py  (obj2_v2, stage H, before H4; 12 Nov 2026; Sara's go on 12 Nov)

Question: will H4 (stage_h1_bnm_export.py --half test --final-test --freeze h3_freeze.json) and H5
(stage_h2_bnm_predict.py on the test-half export) run with the frozen tools, unchanged, before the real test half is
opened? Both tools' md5s are in h3_freeze.json, so neither may change after H4, and a failure after the real opening
would force a second opening. H2a's own self-test ran only a fit-half export.

Planted data only, in a temporary folder: the planted BNM grids of H1's self-test (H1.world) and the planted pipeline of
H2a's self-test (P5.build_world: two tiny runs with G1 and G2). The real h3_freeze.json is read as text (its tool md5
list, and as H1's --freeze, which is what J0 checks). No real BNM grid, no file of ../data other than h3_freeze.json is
opened, and nothing is written outside the temporary folder.

Checks
    D0 the tools are the frozen ones: md5 of every tool listed in h3_freeze.json = the file in tools now; no test-half
       export in ../data (neither bnm_test nor BNM_TEST_EXPORTED)
    D1 H4 refusals with the frozen H1: the test half is refused without --final-test, without --freeze, and with a
       freeze file that lists another md5 for H1; nothing is written
    D2 H4 dry run with the frozen H1 and the real h3_freeze.json: gates J0-J3, J4b-J7 PASS (J4 skipped on the test
       half); h1_report.json says half "test", its tool md5 = the frozen md5, no rankable count (no hole label read);
       the marker is written in the temporary data folder; every window of the export is a test-half window; a second
       test export into a new folder stops
    D3 H5 dry run with the frozen H2a on that export: baselines (K0, R2a, R2b, R4), samples of both runs (K0, R3a,
       R3b, R4) and report (K0, R5) all PASS; h2a_report.json says half "test", truth opened 0, H1 tool = frozen md5
    D4 outputs checked outside the tools (planted data, so truth files may be read here): B1 recomputed from the
       test-half input files = the saved B1; B5 maps occupy no hole tetrad; no record in a hole in any year; each
       samples file holds every frozen row x K
    D5 a planted change: one test-half input file changed after the export makes H2a's K0 fail and the part stop
Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, about 10 min)
    python3 ../tests/test_22_h4h5_dry_run.py > ../logs/test_22_h4h5_dry_run.txt 2>&1; tail -n 1 ../logs/test_22_h4h5_dry_run.txt
"""
import argparse, importlib.util, json, sys, tempfile, time
from pathlib import Path
import numpy as np

H1_NAME, H2_NAME = "stage_h1_bnm_export.py", "stage_h2_bnm_predict.py"


def load(path, name):
    if not Path(path).is_file():
        sys.exit(f"STOP: {path} not found")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main(tools, obj1, freeze, data_real, out=print):
    t0 = time.time()
    tools, obj1, freeze, data_real = Path(tools), Path(obj1), Path(freeze), Path(data_real)
    res = {}

    def check(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    quiet = lambda *a, **k: None
    H2 = load(tools / H2_NAME, "stage_h2_bnm_predict")
    H1 = H2._load("stage_h1_bnm_export")
    X = H2.X
    fz = json.loads(freeze.read_text())
    ftools = fz.get("tools", {})
    bad = [k for k, v in ftools.items() if not (tools / k).is_file() or X.file_md5(tools / k) != v]
    h1m, h2m = X.file_md5(tools / H1_NAME), X.file_md5(tools / H2_NAME)
    opened = [p for p in (data_real / "bnm_test", data_real / H1.MARKER) if p.exists()]
    out(f"test_22: h3_freeze.json md5 {X.file_md5(freeze)}; {H1_NAME} {h1m}; {H2_NAME} {h2m}")
    check("D0", not bad and ftools.get(H1_NAME) == h1m and ftools.get(H2_NAME) == h2m and not opened,
          f"{len(ftools)} tool md5s in h3_freeze.json = the files in tools now (differ: {bad}); H1 and H2a listed "
          f"{ftools.get(H1_NAME) == h1m and ftools.get(H2_NAME) == h2m}; real test-half export present {[str(p) for p in opened]}")
    if not res["D0"]:
        out("test_22: stopped after D0"); return res
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        w = H2.P5.build_world(tmp, str(obj1))
        data, g2, runs = Path(w["data"]), Path(w["g2"]), [Path(r) for r in w["runs"]]
        py, refpat = H1.world(tmp / "bw")
        ot = data / "bnm_test"
        # D1 refusals
        badfz = tmp / "fz_other.json"
        badfz.write_text(json.dumps(dict(tools={H1_NAME: "0" * 32})))
        stops = []
        for kw in (dict(half="test"), dict(half="test", final_test=True), dict(half="test", final_test=True, freeze=str(badfz))):
            try:
                H1.run(py, refpat, str(obj1), ot, expect=(6, 6), out=quiet, **kw); stops.append(False)
            except SystemExit:
                stops.append(True)
        check("D1", all(stops) and not ot.exists(), f"test half refused without --final-test {stops[0]}, without --freeze "
              f"{stops[1]}, with a freeze listing another H1 md5 {stops[2]}; nothing written {not ot.exists()}")
        # D2 H4 dry run
        lines = []
        rt, rept = H1.run(py, refpat, str(obj1), ot, expect=(6, 6), half="test", final_test=True, freeze=str(freeze),
                          out=lines.append)
        js = json.loads((ot / "h1_report.json").read_text()) if (ot / "h1_report.json").is_file() else {}
        bdc, _ = H1.X.import_bdc(str(obj1))
        ref = bdc.split_ref(refpat)
        wins = sorted(set(np.load(ot / "bnm_stacks.npz")["window"].astype(str).tolist())) if (ot / "bnm_stacks.npz").is_file() else []
        all_test = bool(wins) and all(bdc.window_en(nm)[0] >= ref["cut"] for nm in wins)
        try:
            H1.run(py, refpat, str(obj1), data / "bnm_test2", expect=(6, 6), half="test", final_test=True,
                   freeze=str(freeze), out=quiet); again = False
        except SystemExit:
            again = True
        want = {"J0", "J1", "J2", "J3", "J4b", "J5", "J6", "J7"}
        check("D2", set(rt) == want and all(rt.values()) and js.get("half") == "test" and js.get("md5", {}).get("tool") == h1m
              and "rankable" not in js and (data / H1.MARKER).is_file() and all_test and again,
              f"H4 with the frozen H1 and the real h3_freeze.json: gates {rt}; h1_report half {js.get('half')}, tool md5 = "
              f"frozen {js.get('md5', {}).get('tool') == h1m}, rankable count absent {'rankable' not in js}; marker in the "
              f"temporary data folder {(data / H1.MARKER).is_file()}; {len(wins)} windows, all test half {all_test}; "
              f"{rept.get('frozen_rows', 0):,} frozen rows; a second test export stops {again}")
        out("   H4 dry-run log:")
        for s_ in lines:
            out("      " + s_)
        if not res["D2"]:
            out("test_22: stopped after D2"); return res
        # D3 H5 dry run
        o5 = tmp / "h5_predict"
        lines = []
        rb = H2.part_baselines(data, ot, g2, o5, out=lines.append)
        rs = [H2.part_samples(data, ot, r, o5, "cpu", out=lines.append) for r in runs]
        rr = H2.part_report(ot, runs, o5, out=lines.append)
        rep = json.loads((o5 / "h2a_report.json").read_text()) if (o5 / "h2a_report.json").is_file() else {}
        ok_b = set(rb) == {"K0", "R2a", "R2b", "R4"} and all(rb.values())
        ok_s = all(set(x) == {"K0", "R3a", "R3b", "R4"} and all(x.values()) for x in rs)
        ok_r = set(rr) == {"K0", "R5"} and all(rr.values())
        check("D3", ok_b and ok_s and ok_r and rep.get("half") == "test" and rep.get("truth_opened") == 0 and rep.get("h1_tool") == h1m,
              f"H5 with the frozen H2a on the test-half export: baselines {rb}; samples {rs}; report {rr}; h2a_report half "
              f"{rep.get('half')}, truth opened {rep.get('truth_opened')}, H1 tool = frozen {rep.get('h1_tool') == h1m}")
        out("   H5 dry-run log:")
        for s_ in lines:
            out("      " + s_)
        if not res["D3"]:
            out("test_22: stopped after D3"); return res
        # D4 outputs, checked here
        fb = dict(np.load(ot / "bnm_frozen.npz"))
        V1, V2, Ef, Hb, _ = H2.GB.load_obs(ot, fb["files"], fb["file_index"], fb["row"])
        V = V1 | V2
        z = np.load(o5 / "baselines_bnm.npz")
        b1_ok = np.array_equal(z["B1"], H2.HR.b1_score(V, float(z["sigma"])).astype(np.float32))
        hole = np.zeros_like(V[:, 0])
        for i in np.unique(fb["file_index"]):
            k = np.nonzero(fb["file_index"] == i)[0]
            tz = np.load(ot / str(fb["files"][i]).replace("_input.npz", "_truth.npz"))
            hole[k] = X.unpack(tz["hole"])[fb["row"][k]]
        m5 = H2.GS.unpack(np.load(o5 / "maps_B5.npz")["maps"])[:, 0]
        b5_hole = int((m5[:, -1] & hole).sum())
        rec_hole = bool((V[:, :, None] & hole[:, None, None]).any())
        shapes = []
        for r in runs:
            sm = np.load(o5 / f"samples_{r.name}.npz")
            shapes.append((tuple(sm["maps"].shape[:2]), int(sm["K"])))
        shape_ok = all(s == (len(fb["row"]), kk) for s, kk in shapes)
        check("D4", b1_ok and b5_hole == 0 and not rec_hole and shape_ok and int(hole.sum()) > 0,
              f"B1 recomputed from the test-half input files = saved B1 {b1_ok}; B5 maps in hole tetrads {b5_hole}; a record "
              f"in a hole in any year {rec_hole}; holes in the frozen rows {int(hole.sum()):,}; samples (rows, K) {shapes} for "
              f"{len(fb['row']):,} frozen rows")
        # D5 planted change after the export
        e0 = H2.D.rows_of(ot)[0]["input_file"]
        keep = (ot / e0).read_bytes()
        rw = H2._load("stage_g5_export_check").rewrite
        rw(ot / e0, lambda a: a.__setitem__("visit1", a["visit1"] ^ np.uint8(1)))
        try:
            rc = H2.part_baselines(data, ot, g2, tmp / "h5_changed", out=quiet)
        finally:
            (ot / e0).write_bytes(keep)
        check("D5", rc.get("K0") is False and len(rc) == 1 and X.file_md5(ot / e0) == H2.D.rows_of(ot)[0]["input_md5"],
              f"one test-half input file changed after the export: H2a K0 fails and the part stops ({rc}); file restored")
    n_ok = sum(res.values())
    out(f"test_22: {n_ok} PASS, {len(res) - n_ok} FAIL; H4 and H5 run with the frozen tools on a planted test half: "
        f"{'YES' if n_ok == len(res) == 6 else 'NO'} ({(time.time() - t0) / 60:.1f} min)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", default="../tools")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--freeze", default="../data/stage_h3_freeze/h3_freeze.json")
    ap.add_argument("--data", default="../data")
    a = ap.parse_args()
    r = main(a.tools, a.obj1_tools, a.freeze, a.data, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and len(r) == 6 and all(r.values()) else 1)