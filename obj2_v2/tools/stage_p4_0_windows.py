#!/usr/bin/env python3
"""
stage_p4_0_windows.py  (obj2_v2, P4 step P4-0, 13 Nov 2026; P4 plan and rule details signed by Sara on 13 Nov,
claude/obj2_v2_H_rules_draft.md, last section)

Question: before any land-only AOO is computed, can every frozen stack of real_val and real_eval get the land mask of
the BNM fit-half window its survey footprint came from (for real_eval from the inputs only), and do the stored samples
and maps sit on the same window grid as that mask? Nothing is scored and no AOO is computed.

Land mask of a window, L_w (the signed rule): the P3c land rule (LCM2019 1 km; every class except 13, 14, 17, 18;
a tetrad is land if any of its four 1 km cells is above 0; rows from the south, columns from the west) plus every
tetrad with effort in any of the 6 years for any species row of the window in the BNM per-year grids. Every simulated
footprint is the effort of one species row of its window (export_obj2.survey_round), so no simulated effort or record
tetrad is ever masked. This surveyed part is P3b's surveyed set or more (P3b used the species recorded in a window).

Window of a stack: real_val read from the truth file (round_window, window_names) and also matched from the inputs;
real_eval matched from the inputs only: each stack's 6-year effort equals the per-year known cells of one species row
of its window, so the window of a community-round = the fit-half windows that hold the effort of every stack of that
round. No real_eval truth file is opened (np.load log, gate W5).

Gates
    W0 inputs: every input file md5 = its record (G1 g1_report.json, G2 v2 g2_report.json, G5c part_*.json with gates
       PASS, manifest input_md5 and the frozen set's file_md5); export_obj2.py md5 = code_md5 stored in every input file used; LCM file = block P3c-0;
       stage_p3c_band_check.py = version 2; stage_p3c_land_line.py content fingerprint = Sara's P3c run; band order
       CONFIRMED on the map
    W1 frozen sets and maps: recorded tetrads at the target year recounted from the inputs = the frozen rec_t; samples
       and maps hold one row per frozen stack, in order (sel), K 50 (B5 1), the G1 layout, made from that frozen set
    W2 windows: the fit-half windows read now = window_names in every real_val truth file used; real_val: the matched
       window = the truth window for every round that holds a frozen stack (one candidate each); real_eval: every such
       round has a candidate, and where it has several they share the same L_w
    W3 effort against the map: effort tetrads of the frozen stacks (any year) classed no land by the LCM part alone are
       at most 1% and no other of the 7 grid orientations gives fewer; effort tetrads outside L_w 0 (both splits)
    W4 samples against the inputs: share of the target-year recorded tetrads occupied in the maps: B3, B4, B5 exactly 1
       in the identity orientation (G2 / G5c put every record in every map); each seed: identity higher than each of
       the 7 other orientations (both splits)
    W5 np.load log: no real_eval truth file opened; real_val truth files opened only for round_window and window_names
    W6 outputs written once into a new folder and read back equal
Outputs: p4_windows.npz (fit-half window names; LCM land, surveyed and L_w per window; window index per frozen stack of
each split; md5 of both frozen sets) and p4_0_report.json.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_p4_0_windows.py --self-test          (needs ../tests/test_30_p4_0_windows.py)
    python3 -u ../tools/stage_p4_0_windows.py > ../logs/stage_p4_0_windows.txt 2>&1
Needs stage_p3c_land_line.py (and what it loads), export_obj2.py, rasterio, the BNM per-year grids.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_p4_0_windows.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P3C = _load("stage_p3c_land_line")
P3B, BC, GS, X = P3C.P3B, P3C.BC, P3C.GS, P3C.X
G, W, CELLS, TCLS = P3C.G, P3C.W, P3C.CELLS, P3C.TCLS
K = 50
YB = CELLS // 8                                             # 50 bytes per year in a packed map
T_EDGES = np.array([0, 1, 2, 6, 11, 21, 51])
EXPECT = dict(p3c_fingerprint="86af2eae6462c4972d6a7b6cca8d50cb",       # stage_p3c_land_line.py, Sara's P3c run
              band_check="98faa2d95d1a96fa6680972c1100f3ca", lcm=BC.MD5[BC.PCT])
ORIENT3 = {"identity": lambda a: a, "row flip": lambda a: a[:, ::-1], "column flip": lambda a: a[:, :, ::-1],
           "both flips": lambda a: a[:, ::-1, ::-1], "transpose": lambda a: a.transpose(0, 2, 1),
           "transpose, row flip": lambda a: a.transpose(0, 2, 1)[:, ::-1],
           "transpose, column flip": lambda a: a.transpose(0, 2, 1)[:, :, ::-1],
           "transpose, both flips": lambda a: a.transpose(0, 2, 1)[:, ::-1, ::-1]}   # same order and meaning as P3C.ORIENT


def broken():
    return os.environ.get("STAGE_P4_0_BROKEN", "")   # self-test only: plants one bug so its gate must fail


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fingerprint(p):
    """md5 ignoring a first #! line and trailing spaces (a file copied as text still matches)."""
    b = Path(p).read_bytes().split(b"\n")
    return hashlib.md5(b"\n".join(l.rstrip() for l in b if l[:2] != b"#!").rstrip(b"\n")).hexdigest()


def rows_of(d):
    with open(Path(d) / "manifest.csv") as f:
        return list(csv.DictReader(f))


def cls_of(n):
    return np.searchsorted(T_EDGES, np.asarray(n), side="right") - 1


class Guard:
    """While active, every np.load path is logged; a truth file in a real_eval folder is refused."""
    def __enter__(self):
        self.orig, self.paths, self.refused = np.load, [], []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        p = str(Path(f).resolve()) if isinstance(f, (str, Path)) else str(f)
        if "_truth" in Path(p).name and "real_eval" in p:
            self.refused.append(p)
            raise RuntimeError(f"refused np.load of {p}")
        self.paths.append(p)
        return self.orig(f, *a, **k)

    def __exit__(self, *e):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- pieces
def read_split(d, fz, man, use_truth):
    """Effort (packed), target-year records and round of every frozen stack; the effort keys of every stack of the
    rounds that hold a frozen stack; for real_val also the truth window of those rounds."""
    files = [str(s) for s in fz["files"]]
    fi, rw = fz["file_index"].astype(int), fz["row"].astype(int)
    m = len(rw)
    eff = np.zeros((m, W, YB), np.uint8); rec = np.zeros((m, G, G), bool); rkey = [None] * m
    rounds, codes, truthw, tnames = {}, {}, {}, {}
    by_in = {r["input_file"]: r for r in man}
    for f in sorted(set(fi.tolist())):
        nm, k = files[f], np.nonzero(fi == f)[0]
        with np.load(Path(d) / nm) as z:
            E, H = z["effort"], z["stack_hab"]
            V = z["visit1"][rw[k]] | z["visit2"][rw[k]]
            codes[nm] = str(z["code_md5"])
        eff[k] = E[rw[k]]
        rec[k] = X.unpack(V[:, W - 1])
        hs = sorted(set(int(H[r]) for r in rw[k]))
        for i, r in zip(k, rw[k]):
            rkey[i] = (f, int(H[r]))
        for h in hs:
            rounds[(f, h)] = [E[j].tobytes() for j in np.nonzero(H == h)[0]]
        if use_truth:
            with np.load(Path(d) / by_in[nm]["truth_file"]) as zt:
                rwin, wn = zt["round_window"], zt["window_names"].astype(str)
            tnames[nm] = wn
            for h in hs:
                truthw[(f, h)] = str(wn[int(rwin[h])])
        elif broken() == "eval_truth":
            np.load(Path(d) / by_in[nm]["truth_file"])
    return dict(eff=eff, rec=rec, rkey=rkey, rounds=rounds, codes=codes, truthw=truthw, tnames=tnames, files=files)


def key_map(wins):
    """Packed 6-year effort of every species row of every fit-half window -> set of window indices."""
    k2w = defaultdict(set)
    for wi, (_, E) in enumerate(wins):
        P = X.pack(E.transpose(1, 0, 2, 3))                 # (S, 6, 50), as export_obj2 packs a stack's effort
        for s in range(P.shape[0]):
            k2w[P[s].tobytes()].add(wi)
    return k2w


def match(rounds, k2w, land):
    """Window of each round = the windows holding every stack's effort; -1 if none, or if several differ in L_w."""
    out, amb, none = {}, 0, 0
    for key, ks in rounds.items():
        c = None
        for b in ks:
            c = set(k2w.get(b, ())) if c is None else c & k2w.get(b, set())
            if not c:
                break
        c = sorted(c or [])
        if broken() == "win_shift" and c:
            c = [(c[0] + 1) % len(land)]
        same = len(c) >= 1 and all(np.array_equal(land[c[0]], land[x]) for x in c)
        amb += len(c) > 1; none += not c
        out[key] = c[0] if same else -1
    return out, amb, none


def effort_counts(eff, win, nw):
    """(nw, G, G): number of frozen stacks with effort in each tetrad (any year), summed per window."""
    cnt = np.zeros((nw, G, G), np.int64)
    np.add.at(cnt, win, X.unpack(eff).any(1).astype(np.int64))
    return cnt


def shares(maps, rec, transpose=False, ch=2000):
    """Share of target-year recorded tetrads occupied in the maps, for each of the 8 orientations of the maps."""
    num, den = {k: 0 for k in ORIENT3}, 0
    for a in range(0, len(maps), ch):
        tb = maps[a:a + ch, :, (W - 1) * YB:W * YB]
        C = np.unpackbits(tb, axis=-1, count=CELLS).reshape(len(tb), tb.shape[1], G, G).sum(1, dtype=np.int64)
        if transpose:
            C = C.transpose(0, 2, 1)
        R = rec[a:a + ch]
        den += int(R.sum()) * tb.shape[1]
        for k, f in ORIENT3.items():
            num[k] += int((f(C) * R).sum())
    return {k: num[k] / max(den, 1) for k in num}


# ---------------------------------------------------------------- run
def run(data="../data", runs=("../runs/f_seed1", "../runs/f_seed2"), g2="../data/stage_g2_baselines_v2",
        g5c="../data/stage_g5_predict", lcm="../data/lcm2019_1km/data",
        peryear="../../AI_simulation/Empirical_data/bnm_grids_peryear",
        split_ref="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz", obj1_tools="../../obj1_v2/tools",
        out_dir="../data/stage_p4_0", expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    data, g2, g5c, lcm, od = map(Path, (data, g2, g5c, lcm, out_dir))
    runs = [Path(r) for r in runs]
    val, ev = data / "real_val", data / "real_eval"
    if od.exists():
        sys.exit(f"STOP: {od} exists; P4-0 writes into a new folder")
    if "real_eval" in str(od.resolve()) or "_truth" in str(od.resolve()):
        sys.exit("STOP: the output path may not contain real_eval or _truth")
    for p_ in (val, ev, g2, g5c, lcm, *runs):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    fz_val_p, fz_ev_p = data / "stage_d_frozen_val.npz", g5c / "frozen_eval.npz"
    smp = {"s1": runs[0] / "check_g1" / "samples_k50.npz", "s2": runs[1] / "check_g1" / "samples_k50.npz"}
    smp_e = {"s1": g5c / f"samples_{runs[0].name}.npz", "s2": g5c / f"samples_{runs[1].name}.npz"}
    bf = {"B3": "maps_B3_halfblank.npz", "B4": "maps_B4.npz", "B5": "maps_B5.npz"}
    need = [fz_val_p, fz_ev_p, val / "manifest.csv", ev / "manifest.csv", g2 / "g2_report.json", lcm / BC.PCT,
            *smp.values(), *smp_e.values(), *[g2 / f for f in bf.values()], *[g5c / f for f in bf.values()],
            *[r / "check_g1" / "g1_report.json" for r in runs],
            *[g5c / f"part_{p}.json" for p in ("frozen", "baselines", f"samples_{runs[0].name}", f"samples_{runs[1].name}")]]
    miss = [str(f) for f in need if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    t0, res, rep = time.time(), {}, dict(lines=[])

    def say(s):
        rep["lines"].append(s); out(s)

    def gate(name, ok, text):
        res[name] = bool(ok); say(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        first = next((k_ for k_, v_ in res.items() if not v_), after)
        say(f"stage P4-0: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped at {first}, nothing written")
        return res, rep
    me = X.file_md5(Path(__file__))
    tool = dict(p4_0=me, export_obj2=X.file_md5(HERE / "export_obj2.py"), p3c_fingerprint=fingerprint(HERE / "stage_p3c_land_line.py"),
                band_check=X.file_md5(HERE / "stage_p3c_band_check.py"))
    say(f"stage_p4_0_windows.py md5 {me}; export_obj2.py {tool['export_obj2']}; stage_p3c_land_line.py fingerprint "
        f"{tool['p3c_fingerprint']}; stage_p3c_band_check.py {tool['band_check']}")
    with Guard() as lg:
        try:
            # ================================================================ W0 inputs
            md5 = {}
            ok_g1 = True
            for m_, r in zip(("s1", "s2"), runs):
                j = json.loads((r / "check_g1" / "g1_report.json").read_text())
                md5[f"{r.name}/check_g1/samples_k50.npz"] = X.file_md5(smp[m_])
                ok_g1 &= md5[f"{r.name}/check_g1/samples_k50.npz"] == j["md5"]["samples"]
            j2 = json.loads((g2 / "g2_report.json").read_text())
            ok_g2 = True
            for f in bf.values():
                md5[f"{g2.name}/{f}"] = X.file_md5(g2 / f); ok_g2 &= md5[f"{g2.name}/{f}"] == j2["md5"][f]
            ok_g5 = True
            for p, fs in (("frozen", ["frozen_eval.npz"]), ("baselines", list(bf.values())),
                          (f"samples_{runs[0].name}", [smp_e["s1"].name]), (f"samples_{runs[1].name}", [smp_e["s2"].name])):
                jp = json.loads((g5c / f"part_{p}.json").read_text())
                ok_g5 &= all(jp["gates"].values())
                for f in fs:
                    md5[f"{g5c.name}/{f}"] = X.file_md5(g5c / f); ok_g5 &= md5[f"{g5c.name}/{f}"] == jp["files"].get(f)
            md5["stage_d_frozen_val.npz"] = X.file_md5(fz_val_p)
            if broken() == "changed_input":
                md5[f"{runs[0].name}/check_g1/samples_k50.npz"] = "0" * 32; ok_g1 = False
            fz = {"real_val": dict(np.load(fz_val_p)), "real_eval": dict(np.load(fz_ev_p))}
            man = {"real_val": rows_of(val), "real_eval": rows_of(ev)}
            bad_in, code_bad = [], []
            for sp, d in (("real_val", val), ("real_eval", ev)):
                byi = {r["input_file"]: r for r in man[sp]}
                for f in sorted(set(int(x) for x in fz[sp]["file_index"])):
                    nm = str(fz[sp]["files"][f]); mi = X.file_md5(d / nm)
                    if nm not in byi or mi != byi[nm]["input_md5"] or mi != str(fz[sp]["file_md5"][f]):
                        bad_in.append(f"{sp}/{nm}")
            lcm_md5 = X.file_md5(lcm / BC.PCT)
            bc = BC.check(lcm, want_md5=False, quiet=True)
            LC, tr, m_ok = P3C.read_map(lcm)
            t_ok = tool["p3c_fingerprint"] == expect["p3c_fingerprint"] and tool["band_check"] == expect["band_check"]
            gate("W0", ok_g1 and ok_g2 and ok_g5 and not bad_in and lcm_md5 == expect["lcm"] and bc["conf"] and m_ok and t_ok,
                 f"G1 samples = g1_report.json {ok_g1}; G2 v2 maps = g2_report.json {ok_g2}; G5c frozen, maps and samples = "
                 f"part files, parts PASS {ok_g5}; input files used = manifest and frozen-set md5 (differ {len(bad_in)}); map "
                 f"{lcm_md5[:8]} = block P3c-0 {lcm_md5 == expect['lcm']}, band order {'CONFIRMED' if bc['conf'] else 'NOT CONFIRMED'}, "
                 f"21 bands on whole km {m_ok}; P3c tool fingerprint and band check version 2 {t_ok}")
            if not res["W0"]:
                return stop("W0")
            # ================================================================ fit-half windows and L_w
            wins, years, keep = X.load_bnm(peryear, split_ref, obj1_tools, out=lambda s: say("   " + s))
            names = np.array([n_ for n_, _ in wins])
            nw = len(wins)
            land_lcm = np.zeros((nw, G, G), bool); surv = np.zeros((nw, G, G), bool); outside = 0
            for wi, (nm, E) in enumerate(wins):
                e_, n_ = P3B.en(nm)
                ll, _, o_ = P3C.window_land(LC, tr, e_, n_)
                land_lcm[wi] = ll[:, ::-1] if broken() == "orient" else ll
                surv[wi] = E.any((0, 1)); outside += o_
            land = land_lcm.copy() if broken() == "no_effort_land" else land_lcm | surv
            k2w = key_map(wins)
            # ================================================================ read both splits
            S = {"real_val": read_split(val, fz["real_val"], man["real_val"], True),
                 "real_eval": read_split(ev, fz["real_eval"], man["real_eval"], False)}
            for sp in S:
                code_bad += [f"{sp}/{n_}" for n_, c in S[sp]["codes"].items() if c != tool["export_obj2"]]
            # ================================================================ W1 frozen sets and maps
            fmd5 = {"real_val": md5["stage_d_frozen_val.npz"], "real_eval": md5[f"{g5c.name}/frozen_eval.npz"]}
            ok1, why1, maps_at = True, [], {}
            for sp in S:
                n = len(fz[sp]["row"])
                rc = np.array_equal(S[sp]["rec"].reshape(n, -1).sum(1), fz[sp]["rec_t"].astype(np.int64))
                ok1 &= rc; why1.append(f"{sp} {n:,} frozen stacks, rec_t recount {rc}")
                sm = smp if sp == "real_val" else smp_e                         # G1 samples carry no frozen_md5, G5c samples do
                srcs = [(m_, sm[m_], K, sp == "real_eval") for m_ in ("s1", "s2")]
                srcs += [(b, (g2 if sp == "real_val" else g5c) / f, 1 if b == "B5" else K, True) for b, f in bf.items()]
                maps_at[sp] = srcs
                for lab, p, kk, has_f in srcs:
                    z = np.load(p)
                    good = np.array_equal(z["sel"], np.arange(n)) and str(z["layout"]) == GS.LAYOUT and int(z["K"]) == kk \
                        and z["maps"].shape == (n, kk, W * YB) and (not has_f or str(z["frozen_md5"]) == fmd5[sp])
                    ok1 &= good; why1.append(f"{sp} {lab} {good}")
            gate("W1", ok1 and not code_bad, "; ".join(why1) + f"; inputs made by this export_obj2.py (differ {len(code_bad)})")
            if not res["W1"]:
                return stop("W1")
            # ================================================================ W2 windows
            names_ok = all(np.array_equal(wn, names) for wn in S["real_val"]["tnames"].values())
            win, amb, none = {}, {}, {}
            for sp in S:
                mt, amb[sp], none[sp] = match(S[sp]["rounds"], k2w, land)
                win[sp] = np.array([mt[k] for k in S[sp]["rkey"]], np.int16)
                S[sp]["match"] = mt
            tw = S["real_val"]["truthw"]
            val_ok = names_ok and all(mt_ >= 0 and names[mt_] == tw[k] for k, mt_ in S["real_val"]["match"].items()) \
                and amb["real_val"] == 0
            ev_ok = (win["real_eval"] >= 0).all()
            gate("W2", val_ok and ev_ok,
                 f"{nw} fit-half windows read now = window_names in every real_val truth file used {names_ok}; real_val: "
                 f"{len(tw):,} rounds, matched window = truth window in every round {val_ok} (several candidates "
                 f"{amb['real_val']}, none {none['real_val']}); real_eval from inputs only: {len(S['real_eval']['match']):,} rounds, "
                 f"every frozen stack has a window {bool(ev_ok)} (several candidates {amb['real_eval']}, none {none['real_eval']})")
            if not res["W2"]:
                return stop("W2")
            # ================================================================ W3 effort against the map
            ok3, why3, w3 = True, [], {}
            for sp in S:
                cnt = effort_counts(S[sp]["eff"], win[sp], nw)
                tot = int(cnt.sum())
                o = {k_: int(sum((cnt[w_] * ~f_(land_lcm[w_])).sum() for w_ in range(nw))) for k_, f_ in P3C.ORIENT.items()}
                outl = int((cnt * ~land).sum())
                good = o["identity"] <= 0.01 * tot and o["identity"] < min(v for k_, v in o.items() if k_ != "identity") \
                    and outl == 0
                ok3 &= good; w3[sp] = dict(total=tot, orient=o, outside_land=outl)
                why3.append(f"{sp}: effort tetrads of frozen stacks {tot:,}, classed no land by the map {o['identity']:,} "
                            f"({o['identity'] / max(tot, 1):.4f}); other orientations "
                            + ", ".join(f"{k_} {v:,}" for k_, v in o.items() if k_ != "identity") + f"; outside L_w {outl}")
            gate("W3", ok3, "; ".join(why3))
            # ================================================================ W4 samples against the inputs
            ok4, why4, w4 = True, [], {}
            for sp in S:
                for lab, p, kk, _ in maps_at[sp]:
                    mp = np.load(p)["maps"]
                    if broken() == "b5_roll" and lab == "B5":
                        mp = np.roll(mp, 1, axis=0)
                    sh = shares(mp, S[sp]["rec"], transpose=broken() == "maps_transpose" and lab == "s1")
                    del mp
                    idn, oth = sh["identity"], max(v for k_, v in sh.items() if k_ != "identity")
                    good = idn == 1.0 if lab in bf else idn > oth
                    ok4 &= good; w4[f"{sp} {lab}"] = sh
                    why4.append(f"{sp} {lab} identity {idn:.4f}, best other {oth:.4f} {good}")
            gate("W4", ok4, "share of target-year recorded tetrads occupied in the maps: " + "; ".join(why4))
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    loaded = lg.paths
    t_val = [p for p in loaded if "_truth" in Path(p).name]
    gate("W5", crashed is None and not lg.refused and all("real_eval" not in p for p in t_val),
         f"files passed to np.load {len(loaded)}; real_eval truth files opened 0 (refused {len(lg.refused)}); real_val truth "
         f"files opened {len(t_val)} (round_window and window_names read)" + (f"; stopped: {crashed[:120]}" if crashed else ""))
    if crashed or not all(res.values()):
        return stop("W5" if crashed else "W4")
    # ================================================================ report (no AOO)
    coast = np.array([not land[w_].all() for w_ in range(nw)])
    say(f"   windows: {nw} fit-half; L_w all 400 tetrads in {int((~coast).sum())}; with tetrads outside L_w in {int(coast.sum())} "
        f"(tetrads no land by the map {int((~land_lcm).sum()):,}, of them put back as land by effort {int((~land_lcm & surv).sum()):,}; "
        f"outside L_w {int((~land).sum()):,}); 1 km cells outside the map {outside}")
    say("   windows with tetrads outside L_w (window: land tetrads of 400): "
        + ", ".join(f"{names[w_]} {int(land[w_].sum())}" for w_ in np.nonzero(coast)[0]))
    per = {}
    for sp in S:
        c_ = coast[win[sp]]; cl = cls_of(fz[sp]["rec_t"])
        per[sp] = dict(n=int(len(c_)), coastal=int(c_.sum()), ambiguous_rounds=int(amb[sp]),
                       by_class={TCLS[c]: [int((c_ & (cl == c)).sum()), int((cl == c).sum())] for c in range(len(TCLS))})
        say(f"   {sp}: frozen stacks in windows with tetrads outside L_w {per[sp]['coastal']:,} of {per[sp]['n']:,}; by recorded "
            f"class (outside-L_w windows / all): " + ", ".join(f"{k_} {v[0]}/{v[1]}" for k_, v in per[sp]["by_class"].items()))
    # ================================================================ W6 outputs
    od.mkdir(parents=True)
    arrays = dict(window_names=names, land_lcm=land_lcm, surveyed=surv, land=land, val_window=win["real_val"],
                  eval_window=win["real_eval"], val_frozen_md5=np.array(fmd5["real_val"]), eval_frozen_md5=np.array(fmd5["real_eval"]),
                  code_md5=np.array(me))
    X.save_npz(od / "p4_windows.npz", arrays)
    zr = dict(np.load(od / "p4_windows.npz"))
    back = set(zr) == set(arrays) and all(np.array_equal(zr[k_], np.asarray(arrays[k_])) for k_ in arrays)
    mo = X.file_md5(od / "p4_windows.npz")
    gate("W6", back, f"p4_windows.npz md5 {mo} written and read back equal")
    n_ok = sum(res.values())
    last = (f"stage P4-0: gates {n_ok} PASS, {len(res) - n_ok} FAIL; {nw} fit-half windows ({int(coast.sum())} with tetrads outside "
            f"L_w); real_val {per['real_val']['n']:,} and real_eval {per['real_eval']['n']:,} frozen stacks given a window; no AOO computed")
    say(last)
    rep.update(gates=res, written_utc=now(), md5=dict(md5, tool=tool, lcm=lcm_md5, out_npz=mo), years=years,
               windows=[dict(name=str(names[w_]), land=int(land[w_].sum()), no_land_map=int((~land_lcm[w_]).sum()),
                             put_back=int((~land_lcm[w_] & surv[w_]).sum())) for w_ in range(nw)],
               splits=per, w3=w3, w4=w4, minutes=round((time.time() - t0) / 60, 2))
    (od / "p4_0_report.json").write_text(json.dumps(rep, indent=1))
    out(f"   p4_0_report.json md5 {X.file_md5(od / 'p4_0_report.json')}")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--runs", nargs=2, default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--g5c", default="../data/stage_g5_predict")
    ap.add_argument("--lcm", default="../data/lcm2019_1km/data")
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--split-ref", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--out", default="../data/stage_p4_0")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_30_p4_0_windows.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_30_p4_0_windows", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.main(Path(__file__), a.obj1_tools) else 1)
    r, _ = run(a.data, a.runs, a.g2, a.g5c, a.lcm, a.peryear, a.split_ref, a.obj1_tools, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)