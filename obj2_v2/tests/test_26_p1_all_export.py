#!/usr/bin/env python3
"""
test_26_p1_all_export.py  (obj2_v2, products step P1, 13 Nov 2026): self-test of tools/stage_p1_bnm_all_export.py.
Run through the tool, from PSD_Dispersal_pool/obj2_v2/work:
    python3 ../tools/stage_p1_bnm_all_export.py --self-test
It builds planted per-year grids with stage H1's own world(), makes a planted fit export and a planted test export (the H4
path: --half test --final-test with a freeze file listing H1's md5) with the real stage_h1_bnm_export.py, all in a
temporary folder, and then checks P1 on them. Nothing of ../data is read or written.
    S0  the planted H1 fit and test exports pass every H1 gate
    T1  P1 on them: gates Q0-Q5 PASS
    T2  outside the tool, from the planted H1 truth files of repeat 1 (planted data only): every P1 input holds the
        original records and effort (rec_orig, eff_orig) of its window and the same species rows, in both halves; stacks
        = H1 fit + test stacks = frozen rows; frozen rec_t and rec6 = the original records
    T3  stage_h2_bnm_predict.py's own K0 accepts the P1 export
    T4  a second run into a new folder gives the same md5 for every npz file and the manifest
    T5  refused: a run into an existing folder; an output inside the H1 fit folder; Q0 fails and nothing is written when
        the fit export is given as the test export
    bugs: only the fit half read (Q1); stacks from 2019 records only (Q2); the repeat-2 shift (Q2); records in H1's
        repeat-1 holes left out (Q3, and K0 then refuses the export); a truth file opened (Q5); nothing written where the
        failing gate comes before the writing
"""
import hashlib, importlib.util, json, os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main(tool_path, tools):
    P1 = load(tool_path, "stage_p1_bnm_all_export")
    H1, X, D = P1.H1, P1.X, P1.D
    H2A = load(Path(tool_path).parent / "stage_h2_bnm_predict.py", "stage_h2_bnm_predict")
    quiet = lambda *a, **k: None
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    def k0(d):
        g = []
        ok = H2A.k0(SimpleNamespace(gate=lambda n_, ok_, txt: g.append(txt)), d)[3]
        return ok, (g[0] if g else "")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = H1.world(tmp / "bw")
        data = tmp / "data"
        fit, test = data / "bnm_fit", data / "bnm_test"
        rf, repf = H1.run(py, refpat, tools, fit, expect=(6, 6), out=quiet)
        fz = tmp / "fz.json"
        fz.write_text(json.dumps(dict(tools={"stage_h1_bnm_export.py": X.file_md5(P1.H1_FILE)})))
        rt, rept = H1.run(py, refpat, tools, test, expect=(6, 6), half="test", final_test=True, freeze=str(fz), out=quiet)
        t("S0", all(rf.values()) and all(rt.values()), f"planted H1 fit export {rf}; planted test export {rt}")

        def run(o, f_=fit, t_=test, out=quiet):
            return P1.run(py, refpat, tools, f_, t_, o, expect=(6, 6), out=out)

        al = data / "bnm_all"
        lines = []
        r, rep = run(al, out=lines.append)
        t("T1", set(r) == {"Q0", "Q1", "Q2", "Q3", "Q4", "Q5"} and all(r.values()), f"P1 on the planted exports: {r}")
        if not all(r.values()):
            print("      " + "\n      ".join(lines))
        # T2: independent check from the planted truth files (repeat 1 holds rec_orig and eff_orig of every stack)
        man = {row["window"]: row for row in D.rows_of(al)} if (al / "manifest.csv").is_file() else {}
        orig, ok2, nw = {}, bool(man), 0
        for d in (fit, test):
            for row in D.rows_of(d):
                if int(row["repeat"]) != 1:
                    continue
                tz = np.load(d / row["truth_file"])
                ro, eo, sp = X.unpack(tz["rec_orig"]), X.unpack(tz["eff_orig"]), [str(s) for s in tz["species"]]
                orig[row["window"]] = ro
                if row["window"] not in man:
                    ok2 = False; continue
                a = np.load(al / man[row["window"]]["input_file"])
                pair = hashlib.md5((row["window"] + "|0|" + "|".join(sp)).encode()).hexdigest()
                ok2 &= np.array_equal(X.unpack(a["visit1"]), ro) and np.array_equal(X.unpack(a["effort"]), eo) and \
                    not X.unpack(a["visit2"]).any() and str(a["pair_md5"]) == pair
                nw += 1
        if ok2:
            fr = np.load(al / "bnm_frozen.npz")
            files = [str(f) for f in fr["files"]]
            wname = {row["input_file"]: w for w, row in man.items()}
            rt_ok = rc_ok = True
            for fi, rw, rtv, r6v in zip(fr["file_index"], fr["row"], fr["rec_t"], fr["rec6"]):
                ro = orig[wname[files[fi]]][rw]
                rt_ok &= int(ro[-1].sum()) == int(rtv); rc_ok &= int(ro.any(0).sum()) == int(r6v)
            n_ok = rep["stacks"] == repf["stacks"] + rept["stacks"] == len(fr["row"])
            ok2 = ok2 and rt_ok and rc_ok and n_ok and nw == len(man)
        t("T2", ok2, f"outside the tool, from the planted truth files of both halves: {nw} windows, P1 records and effort = "
          f"rec_orig and eff_orig, species rows the same; stacks {rep.get('stacks')} = H1 fit {repf['stacks']} + test "
          f"{rept['stacks']} = frozen rows; frozen rec_t and rec6 = the original records")
        ok3, txt3 = k0(al) if man else (False, "no export")
        t("T3", ok3, f"stage_h2_bnm_predict.py's own K0 accepts the P1 export ({txt3})")
        al2 = data / "bnm_all2"
        run(al2)
        same = bool(man) and all(X.file_md5(p) == X.file_md5(al2 / p.name) for p in sorted(al.glob("*.npz"))) and \
            X.file_md5(al / "manifest.csv") == X.file_md5(al2 / "manifest.csv")
        t("T4", same, "a second run into a new folder gives the same md5 for every npz file and the manifest")
        stops = []
        for f in (lambda: run(al), lambda: run(fit / "inside")):
            try:
                f(); stops.append(False)
            except SystemExit:
                stops.append(True)
        rq, _ = run(data / "bnm_q0", t_=fit)
        t("T5", all(stops) and rq.get("Q0") is False and not (data / "bnm_q0").exists(),
          f"refused: a run into an existing folder {stops[0]}; an output inside the H1 fit folder {stops[1]}; Q0 fails and "
          f"nothing is written when the fit export is given as the test export {rq}")
        for bug, g_, written in (("fit_only", "Q1", False), ("rows_2019", "Q2", False), ("shift_r2", "Q2", False),
                                 ("keep_holes", "Q3", True), ("opentruth", "Q5", False)):
            os.environ["STAGE_P1_BROKEN"] = bug
            try:
                rb, _ = run(data / f"bug_{bug}")
            finally:
                os.environ.pop("STAGE_P1_BROKEN", None)
            ex = (data / f"bug_{bug}").exists()
            extra = ""
            ok_b = rb.get(g_) is False and ex == written
            if bug == "keep_holes":
                okk, _ = k0(data / f"bug_{bug}")
                ok_b &= not okk
                extra = f"; K0 refuses the export {not okk}"
            t(f"bug {bug}", ok_b, f"{g_} fails as it must; written {ex} (expected {written}){extra} ({rb})")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)