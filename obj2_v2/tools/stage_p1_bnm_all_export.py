#!/usr/bin/env python3
"""
stage_p1_bnm_all_export.py  (obj2_v2, products step P1, 13 Nov 2026; products plan signed off by Sara on 13 Nov: all six
x values with x 2 and 124 labelled "not checked on simulation", pooled seeds as the product with each seed beside it, all
101 windows stated as products, not a test)

The BNM export for the P(window AOO <= x) products: every species-window of all 101 windows (fit half 44, test half 57)
with at least one record in 2014-2019, with all its records and all its effort, and no holes. It uses
stage_h1_bnm_export.py's own functions (read_window, draw_holes, shift_of) and H1's export layout, so
stage_h2_bnm_predict.py runs on it unchanged (step P2). Nothing is scored here or later: true AOO is unknown on BNM. The
test-half labels were read once at H6; this tool opens no *_truth.npz file.

Files (one input file per window, in H1's input format)
    <window>_input.npz  export_obj2's allow-list only: visit1 = all records, visit2 = empty, effort = all effort, one zero
                        habitat map (BNM has no habitat map), n_rec
    bnm_frozen.npz      stage D's frozen-set layout, one row per stack; blank 1; shift = H1's shift of repeat 1
    bnm_stacks.npz      per row: window, half, species, GB Red List status, row, stack id, 2019 surveyed tetrads
    manifest.csv, h1_report.json (the file name stage_h2_bnm_predict.py's K0 reads; it says which tool wrote it, and its
                        md5 "tool" is this tool's md5)

Gates
    Q0 inputs: output folder new and outside the H1 folders; the stage_h1_bnm_export.py loaded here has the tool md5 of
       both H1 reports; both reports passed every gate, are the fit and the test half, and their manifest md5 = report
    Q1 grids: 6 consecutive years; windows fit + test = all (expect 44 + 57), none sharing cells; species rows and shapes
       equal across years in every window; every record inside that year's effort
    Q2 same grids, rows and draws as H1 and H4: H1's hole masks drawn again from the grids give each report's hole md5
       (J7); every H1 / H4 input file equals this tool's arrays with those holes applied (visit1, effort, n_rec, visit2
       empty, habitat 0, stack_hab, pair_md5), bit for bit; stacks = each report's count; H1's frozen shift of every
       repeat-1 row = this tool's shift
    Q3 written inputs, read afresh: keys = the allow-list; md5 = manifest; visit1 and effort = the grids in all 6 years,
       bit for bit; visit2 empty; n_rec = the records; no stack without a record
    Q4 builder: tensors of every frozen row bit-identical by three paths (packed in memory, file read afresh, grids);
       every row accepted
    Q5 truth closed: no *_truth.npz opened (stage D's guard over the whole run); np.load only on grid and reference
       files, H1 / H4 input and index files, and this export's own inputs
If Q0, Q1 or Q2 fails, or a truth file is touched before writing, nothing is written. If Q3-Q5 fail, h1_report.json
carries the failed gate and stage_h2_bnm_predict.py refuses the export (K0).

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_p1_bnm_all_export.py --self-test
    python3 -u ../tools/stage_p1_bnm_all_export.py > ../logs/stage_p1_bnm_all_export.txt 2>&1
Defaults: --fit-export ../data/bnm_fit, --test-export ../data/bnm_test, --out ../data/bnm_all; grids as stage H1.
Needs stage_h1_bnm_export.py and the stage tools it loads in the same folder; the self-test is
tests/test_26_p1_all_export.py.
"""
import argparse, csv, glob, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, folder=HERE):
    p = Path(folder) / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_p1_bnm_all_export.py needs it")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


H1 = _load("stage_h1_bnm_export")
E, D, X = H1.E, H1.D, H1.X
G, W, NC = X.G, X.WINDOW, X.NC
TCLS, BINS = D.TCLS, D.BINS
H1_FILE = HERE / "stage_h1_bnm_export.py"
FROZEN_KEYS = H1.FROZEN_KEYS


class Halt(Exception):
    pass


def broken():
    return os.environ.get("STAGE_P1_BROKEN", "")         # self-test only: plants one bug so its gate must fail


def p1_shift(e, n, s):
    """The frozen-set shift of a products stack: H1's shift of repeat 1 for this stack."""
    return H1.shift_of(e, n, s, 1 if broken() == "shift_r2" else 0)


def export_ok(d, half, h1md5):
    d = Path(d)
    if not (d / "h1_report.json").is_file() or not (d / "manifest.csv").is_file():
        return None, False, f"{d}: no h1_report.json or manifest.csv"
    js = json.loads((d / "h1_report.json").read_text())
    man = X.file_md5(d / "manifest.csv")
    gates = all(js.get("gates", {"none": False}).values())
    ok = js.get("half") == half and gates and man == js["md5"]["manifest.csv"] and js["md5"]["tool"] == h1md5 \
        and "hole_md5" in js
    return js, ok, (f"{d.name}: half {js.get('half')} (needed {half}), H1 gates all PASS {gates}, manifest = report "
                    f"{man == js['md5']['manifest.csv']}, written by stage_h1_bnm_export.py {js['md5']['tool'][:8]} = this one "
                    f"{js['md5']['tool'] == h1md5}")


def run(peryear, split_ref, tools, fit_export, test_export, out_dir, minlist=8, expect=(44, 57), out=print):
    t0 = time.time()
    res, rep = {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    od = Path(out_dir); o = od.resolve()
    if od.exists():
        sys.exit(f"STOP (Q0): output folder {out_dir} exists (the export is written once into a new folder)")
    for p_ in (fit_export, test_export):
        b = Path(p_).resolve()
        if o == b or b in o.parents:
            sys.exit(f"STOP (Q0): nothing is written inside {p_}; give another --out")
    if "real_eval" in str(o) or "_truth" in str(o):
        sys.exit("STOP (Q0): the output folder may not lie in a real_eval folder or carry '_truth' in its path")
    bdc, bmd5 = X.import_bdc(tools)
    me, h1md5 = X.file_md5(__file__), X.file_md5(H1_FILE)
    out(f"stage_p1_bnm_all_export.py md5 {me}; stage_h1_bnm_export.py {h1md5}; bnm_detection_check.py {bmd5}; "
        f"export_obj2.py {X.file_md5(HERE / 'export_obj2.py')}; stage_e_inputs.py {X.file_md5(HERE / 'stage_e_inputs.py')}")
    srcs, texts = {}, []
    for half, d in (("fit", fit_export), ("test", test_export)):
        js, ok, txt = export_ok(d, half, h1md5)
        srcs[half] = dict(dir=Path(d), js=js, ok=ok); texts.append(txt)
    gate("Q0", all(s_["ok"] for s_ in srcs.values()), "output folder new, outside the H1 folders; " + "; ".join(texts))
    if not res["Q0"]:
        out(f"stage P1: gates {sum(res.values())} PASS, {len(res) - sum(res.values())} FAIL; stopped after Q0, nothing written")
        return res, rep
    ref_files = {str(Path(p).resolve()) for p in glob.glob(split_ref)}
    halted, crashed, wrote, years = None, None, False, []
    win, manifest, fr, ix = [], [], {}, {}
    with D.TruthGuard() as guard, H1.LoadLog() as log:
        try:
            # ------------------------------------------------------------ Q1 grids, all 101 windows
            pooled = Path(peryear) / f"minlist{minlist}"
            ref = bdc.split_ref(split_ref)
            names = sorted(f.name for f in pooled.glob("bnm_*.npz"))
            if not names:
                sys.exit(f"STOP: no window files in {pooled}")
            keep, east, overlap = bdc.fit_half(names, ref)
            fit_set = set(keep)
            test_set = {nm for nm in names if bdc.window_en(nm)[0] >= ref["cut"]}
            mine = sorted(fit_set | test_set) if broken() != "fit_only" else sorted(fit_set)
            if broken() == "opentruth":
                np.load(sorted(Path(fit_export).glob("*_truth.npz"))[0])
            ydirs = sorted((int(p.name.rsplit("_y", 1)[1]), p) for p in Path(peryear).glob(f"minlist{minlist}_y*") if p.is_dir())
            a0 = np.load(pooled / mine[0], allow_pickle=True)
            years = [y for y, _ in ydirs] + [int(np.asarray(a0["target_year"]).item())]
            consec = years == list(range(years[0], years[0] + len(years)))
            missing = [(y, nm) for y, d in ydirs for nm in mine if not (d / nm).is_file()]
            rows_ok, outside = True, 0
            if consec and len(years) == W and not missing:
                for nm in mine:
                    R, Ef, sp, st, ok, o_ = H1.read_window(pooled, ydirs, nm)
                    rows_ok &= ok; outside += o_
                    if ok:
                        e, n = bdc.window_en(nm)
                        k = np.nonzero(R.any((0, 2, 3)))[0] if broken() != "rows_2019" else np.nonzero(R[-1].any((1, 2)))[0]
                        win.append(dict(nm=nm, e=e, n=n, half="fit" if nm in fit_set else "test", k=k, sp=sp[k], st=st[k],
                                        Rk=R[:, k].transpose(1, 0, 2, 3), Ek=Ef[:, k].transpose(1, 0, 2, 3)))
            nf, nt = sum(w["half"] == "fit" for w in win), sum(w["half"] == "test" for w in win)
            gate("Q1", consec and len(years) == W and not missing and (len(keep), east) == tuple(expect) and overlap == 0
                 and len(win) == len(names) and rows_ok and outside == 0,
                 f"years {years}; {len(names)} windows: fit {len(keep)}, test {east}, sharing cells {overlap} (expected "
                 f"{expect[0]} and {expect[1]}); read {len(win)} (fit {nf}, test {nt}); species rows and shapes equal {rows_ok}; "
                 f"records outside effort {outside}")
            if not res["Q1"]:
                raise Halt("Q1")
            # ------------------------------------------------------------ Q2 the same grids, rows and draws as H1 / H4
            holes, q2ok, q2txt = {}, True, []
            for half in ("fit", "test"):
                d, js = srcs[half]["dir"], srcs[half]["js"]
                wh = [w for w in win if w["half"] == half]
                man = D.rows_of(d)
                reps = sorted({int(r_["repeat"]) for r_ in man})
                md5a = hashlib.md5()
                for w in wh:
                    for r in range(len(reps)):
                        H = np.stack([H1.draw_holes(w["Ek"][j, -1], np.zeros((G, G), bool), w["e"], w["n"], int(s), r)
                                      for j, s in enumerate(w["k"])]) if len(w["k"]) else np.zeros((0, G, G), bool)
                        md5a.update(np.packbits(H).tobytes())
                        holes[(w["nm"], r + 1)] = H
                hole_ok = md5a.hexdigest() == js["hole_md5"] and reps == list(range(1, H1.REPEATS + 1))
                byname = {w["nm"]: w for w in wh}
                same_set = {(r_["window"], int(r_["repeat"])) for r_ in man} == {(w["nm"], r) for w in wh for r in reps}
                bad = 0
                for r_ in man:
                    w, rr = byname.get(r_["window"]), int(r_["repeat"])
                    H, p = holes.get((r_["window"], rr)), d / r_["input_file"]
                    good = w is not None and H is not None and X.file_md5(p) == r_["input_md5"] and int(r_["stacks"]) == len(w["k"])
                    if good:
                        with np.load(p, allow_pickle=False) as z:
                            a = {k_: z[k_] for k_ in z.files}
                        H6 = H[:, None]
                        V, Ek = w["Rk"] & ~H6, w["Ek"] & ~H6
                        pair = hashlib.md5((w["nm"] + f"|{rr}|" + "|".join(w["sp"])).encode()).hexdigest()
                        good = set(a) == set(X.ALLOW) and np.array_equal(a["visit1"], X.pack(V)) and \
                            np.array_equal(a["effort"], X.pack(Ek)) and a["visit2"].shape == a["visit1"].shape and \
                            not a["visit2"].any() and np.array_equal(a["n_rec"], V.reshape(len(V), W, -1).sum(2)) and \
                            np.array_equal(a["habitat_noisy"], np.zeros((1, G, G), np.float16)) and \
                            np.array_equal(a["stack_hab"], np.zeros(len(V), np.int32)) and str(a["pair_md5"]) == pair
                    bad += int(not good)
                n_st = sum(len(w["k"]) for w in wh)
                with np.load(d / "bnm_frozen.npz") as z:
                    fsh = z["shift"]
                with np.load(d / "bnm_stacks.npz") as z:
                    iw, irp, irw, isp = z["window"], z["repeat"], z["row"], z["species"]
                sh_bad, sh_n = 0, 0
                for i in np.nonzero(irp == 1)[0]:
                    w, j = byname.get(str(iw[i])), int(irw[i])
                    if w is None or j >= len(w["k"]) or str(w["sp"][j]) != str(isp[i]):
                        sh_bad += 1; continue
                    sh_bad += int(not np.array_equal(fsh[i], p1_shift(w["e"], w["n"], int(w["k"][j]))))
                    sh_n += 1
                ok_h = hole_ok and same_set and bad == 0 and n_st == js["stacks"] and sh_bad == 0 and sh_n > 0
                q2ok &= ok_h
                q2txt.append(f"{half}: hole md5 {md5a.hexdigest()[:8]} = H1 report {js['hole_md5'][:8]} {hole_ok}; "
                             f"{len(man)} input files compared, differing {bad}, same windows and repeats {same_set}; stacks "
                             f"{n_st:,} = H1 {js['stacks']:,} {n_st == js['stacks']}; repeat-1 shifts compared {sh_n:,}, "
                             f"differing {sh_bad}")
            gate("Q2", q2ok, "this export with H1's holes applied = the H1 / H4 input files, bit for bit: " + "; ".join(q2txt))
            if not res["Q2"]:
                raise Halt("Q2")
            # ------------------------------------------------------------ write
            od.mkdir(parents=True)
            for w in win:
                V, Ek = w["Rk"], w["Ek"]
                if broken() == "keep_holes":
                    V = V & ~holes[(w["nm"], 1)][:, None]
                pair = hashlib.md5((w["nm"] + "|0|" + "|".join(w["sp"])).encode()).hexdigest()
                inp = dict(code_md5=np.array(me), effort=X.pack(Ek), habitat_noisy=np.zeros((1, G, G), np.float16),
                           n_rec=V.reshape(len(V), W, -1).sum(2).astype(np.int16), pair_md5=np.array(pair),
                           stack_hab=np.zeros(len(V), np.int32), visit1=X.pack(V), visit2=X.pack(np.zeros_like(V)))
                ip = od / f"{Path(w['nm']).stem}_input.npz"
                X.save_npz(ip, inp)
                manifest.append(dict(sim_seed=(w["e"] * 10000 + w["n"]) * 10, window=w["nm"], half=w["half"], repeat=0,
                                     stacks=len(V), input_file=ip.name, input_md5=X.file_md5(ip)))
            with open(od / "manifest.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(manifest[0])); wr.writeheader(); wr.writerows(manifest)
            wrote = True
            # ------------------------------------------------------------ Q3 inputs read afresh; frozen set and index
            q3_bad, keys_bad, zero = 0, 0, 0
            fr = {k_: [] for k_ in ("file_index", "row", "rec_t", "rec6", "shift")}
            ix = {k_: [] for k_ in ("window", "half", "species", "status", "row", "stack_id", "n_surv")}
            for fi, (row, w) in enumerate(zip(manifest, win)):
                p = od / row["input_file"]
                md5ok = X.file_md5(p) == row["input_md5"]
                with np.load(p, allow_pickle=False) as z:
                    keys = set(z.files)
                    v1, v2, ef, nr = X.unpack(z["visit1"]), X.unpack(z["visit2"]), X.unpack(z["effort"]), z["n_rec"]
                keys_bad += int(keys != set(X.ALLOW))
                q3_bad += int(not (md5ok and np.array_equal(v1, w["Rk"]) and np.array_equal(ef, w["Ek"]) and not v2.any()
                                   and np.array_equal(nr, (v1 | v2).reshape(len(v1), W, -1).sum(2)) and not (v1 & ~ef).any()))
                rec6 = (v1 | v2).any(1).reshape(len(v1), -1).sum(1)
                rec_t = (v1 | v2)[:, -1].reshape(len(v1), -1).sum(1)
                zero += int((rec6 == 0).sum())
                for j in np.nonzero(rec6 >= 1)[0]:
                    fr["file_index"].append(fi); fr["row"].append(j); fr["rec_t"].append(rec_t[j]); fr["rec6"].append(rec6[j])
                    fr["shift"].append(p1_shift(w["e"], w["n"], int(w["k"][j])))
                    ix["window"].append(w["nm"]); ix["half"].append(w["half"]); ix["species"].append(w["sp"][j])
                    ix["status"].append(w["st"][j]); ix["row"].append(j); ix["stack_id"].append(fi * 1000 + int(w["k"][j]))
                    ix["n_surv"].append(int(w["Ek"][j, -1].sum()))
            gate("Q3", q3_bad == 0 and keys_bad == 0 and zero == 0,
                 f"input files read afresh: md5 = manifest, records and effort = the grids in all {W} years, visit2 empty, n_rec "
                 f"= records, records inside effort: wrong files {q3_bad} of {len(manifest)}; key set other than the "
                 f"allow-list {keys_bad}; stacks without a record {zero}")
            nfr = len(fr["row"])
            frozen = dict(files=np.array([r_["input_file"] for r_ in manifest]),
                          file_md5=np.array([r_["input_md5"] for r_ in manifest]),
                          file_index=np.array(fr["file_index"], np.int64), row=np.array(fr["row"], np.int64),
                          rec_t=np.array(fr["rec_t"], np.int64), rec6=np.array(fr["rec6"], np.int64),
                          cls=D.tidx(np.array(fr["rec_t"], np.int64)).astype(np.int8), blank=np.ones(nfr, np.uint8),
                          shift=np.array(fr["shift"], np.int8).reshape(nfr, 2), fast=np.zeros(nfr, np.uint8))
            index = dict(window=np.array(ix["window"]), half=np.array(ix["half"]), species=np.array(ix["species"]),
                         status=np.array(ix["status"]), row=np.array(ix["row"], np.int64),
                         stack_id=np.array(ix["stack_id"], np.int64), n_surv=np.array(ix["n_surv"], np.int32))
            X.save_npz(od / "bnm_frozen.npz", frozen); X.save_npz(od / "bnm_stacks.npz", index)
            # ------------------------------------------------------------ Q4 builder, three paths
            q4_bad, q4_err, n4 = 0, "", 0
            try:
                VA = E.Inputs(od, frozen["files"])
                off, blank = E.frozen_draws(frozen)
                fi_a, rw_a = frozen["file_index"], frozen["row"]
                for f_ in np.unique(fi_a):
                    q = np.nonzero(fi_a == f_)[0]
                    xa, na = VA.get(fi_a[q], rw_a[q], off[q], blank[q])
                    xb, nb = E.build_from_file(od / frozen["files"][f_], rw_a[q], off[q], blank[q])
                    w = win[f_]
                    xc, nc = E.build(w["Rk"][rw_a[q]], w["Ek"][rw_a[q]], np.zeros((len(q), G, G), np.float32), off[q], blank[q])
                    q4_bad += int(not (np.array_equal(xa, xb) and np.array_equal(xb, xc) and np.array_equal(na, nb)
                                       and np.array_equal(nb, nc)))
                    n4 += len(q)
                q4_ok = q4_bad == 0 and n4 == nfr
            except ValueError as ex:
                q4_ok, q4_err = False, f"; builder refused: {ex}"
            gate("Q4", q4_ok, f"tensors of {n4:,} of {nfr:,} frozen rows bit-identical by three paths (packed in memory, file "
                 f"read afresh, grids): files differing {q4_bad}{q4_err}")
        except Halt as h:
            halted = str(h)
        except RuntimeError as ex:
            crashed = str(ex)
    exp_dirs = [str(Path(p_).resolve()) for p_ in (fit_export, test_export)]
    root = str(Path(peryear).resolve())
    others = []
    for p in log.paths:
        rp = str(Path(p).resolve())
        if "_truth" in Path(p).name:
            others.append(p)
        elif rp in ref_files or (rp.startswith(root + os.sep) and Path(p).name.startswith("bnm_")):
            continue
        elif any(rp.startswith(d_ + os.sep) for d_ in exp_dirs) or rp.startswith(str(o) + os.sep):
            continue
        else:
            others.append(p)
    gate("Q5", not guard.opened and not crashed and not others,
         f"no *_truth.npz opened (stage D's guard over the whole run): opened {len(guard.opened)}"
         + (f" ({crashed[:120]})" if crashed else "") + f"; files passed to np.load {len(log.paths):,}, other than grid and "
         f"reference files, H1 / H4 inputs and index files and this export's inputs {len(others)}")
    n_ok = sum(res.values())
    if not wrote:
        out(f"stage P1: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {halted or 'a truth file was touched'}, "
            f"nothing written")
        return res, rep
    # ---------------------------------------------------------------- report
    half_a = np.array(ix.get("half", []))
    rt_a, r6_a = np.array(fr.get("rec_t", []), np.int64), np.array(fr.get("rec6", []), np.int64)
    groups = {}
    for lab, gi, labels in (("recorded class at 2019 (all records)", D.tidx(rt_a), TCLS),
                            ("records over 6 years (all records)", D.bidx(r6_a), BINS)):
        groups[lab] = {l: dict(fit=int(((gi == i) & (half_a == "fit")).sum()), test=int(((gi == i) & (half_a == "test")).sum()))
                       for i, l in enumerate(labels)}
        out(f"   {lab}, stacks fit / test: " + "; ".join(f"{l} {v['fit']:,} / {v['test']:,}" for l, v in groups[lab].items()))
    st_a, sp_a = np.array(ix.get("status", [])), np.array(ix.get("species", []))
    status = {str(s_): dict(fit=int(((st_a == s_) & (half_a == "fit")).sum()), test=int(((st_a == s_) & (half_a == "test")).sum()),
                            species=sorted({str(x) for x in sp_a[st_a == s_]})) for s_ in sorted(set(st_a.tolist()))}
    out("   GB Red List 2022 status, stacks fit / test (species): " + "; ".join(
        f"{k_ or '(none)'} {v['fit']:,} / {v['test']:,} ({len(v['species'])})" for k_, v in
        sorted(status.items(), key=lambda kv: -(kv[1]["fit"] + kv[1]["test"]))))
    nst = dict(fit=int((half_a == "fit").sum()), test=int((half_a == "test").sum()))
    md5s = {nm_: X.file_md5(od / nm_) for nm_ in ("bnm_frozen.npz", "bnm_stacks.npz", "manifest.csv")}
    rep.update(tool_name=Path(__file__).name, note="written by products step P1 in stage H1's export layout; md5.tool is this "
               "tool's md5; all records, no holes, nothing scored", half="all", years=years,
               windows=dict(all=len(win), fit=sum(w["half"] == "fit" for w in win), test=sum(w["half"] == "test" for w in win)),
               stacks=len(rt_a), stacks_by_half=nst, frozen_rows=len(rt_a), groups=groups, status=status,
               sources={h_: dict(h1_report_md5=X.file_md5(s_["dir"] / "h1_report.json"), hole_md5=s_["js"]["hole_md5"],
                                 stacks=s_["js"]["stacks"], manifest_md5=s_["js"]["md5"]["manifest.csv"])
                        for h_, s_ in srcs.items()},
               gates=res, md5=dict(tool=me, h1_tool=h1md5, bdc=bmd5, **md5s), minutes=round((time.time() - t0) / 60, 2))
    (od / "h1_report.json").write_text(json.dumps(rep, indent=1, default=int))
    out("   " + ", ".join(f"{k_} {v[:8]}" for k_, v in md5s.items()))
    out(f"stage P1 (all windows, no holes): gates {n_ok} PASS, {len(res) - n_ok} FAIL; {len(win)} windows (fit "
        f"{rep['windows']['fit']}, test {rep['windows']['test']}), {len(manifest)} files, {len(rt_a):,} stacks "
        f"(fit {nst['fit']:,}, test {nst['test']:,}) ({(time.time() - t0) / 60:.1f} min)")
    return res, rep


def self_test(tools):
    t = HERE.parent / "tests" / "test_26_p1_all_export.py"
    if not t.is_file():
        sys.exit(f"STOP: {t} not found; save test_26_p1_all_export.py into tests")
    return _load("test_26_p1_all_export", t.parent).main(Path(__file__).resolve(), tools)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--split-ref", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--minlist", type=int, default=8)
    ap.add_argument("--expect", type=int, nargs=2, default=[44, 57])
    ap.add_argument("--fit-export", default="../data/bnm_fit")
    ap.add_argument("--test-export", default="../data/bnm_test")
    ap.add_argument("--out", default="../data/bnm_all")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    r, _ = run(a.peryear, a.split_ref, a.obj1_tools, a.fit_export, a.test_export, a.out, a.minlist, tuple(a.expect))
    sys.exit(0 if r and all(r.values()) else 1)