#!/usr/bin/env python3
"""
stage_h0_bnm_check.py  (obj2_v2, stage H step H0, 11 Nov 2026; stage H rules signed off by Sara on 11 Nov,
claude/obj2_v2_H_rules_draft.md)

Read-only checks on the BNM per-year grids before any stage H tool is built. Fit-half windows are opened; test-half
windows are seen by file name only. Nothing is written except the optional json report (--out).

Facts it checks (from step10_bnm_gridder.py, patch_step10_peryear.py and bnm_detection_check.py)
    the pooled folder minlist8 holds 2019 (obs_mask, known_target, species, status); minlist8_y<year> holds each earlier
    year (present_year, known_holdout); the split is bnm_detection_check's fit_half on the minlist8 reference windows,
    imported, never re-written; Sara's 30 Oct run read 44 fit-half windows and left 57 test-half windows unopened.

Gates
    H0a years: one folder per earlier year, years consecutive, 6 years ending at the pooled target year; every window
        name (fit and test half) present in every year folder (names only)
    H0b split: fit-half windows = 44 and test-half windows = 57 (--expect), none left out for sharing cells
    H0c fit half: species rows and array shapes equal in all 6 years; every record inside that year's effort
        (present_year within known_holdout, obs_mask within known_target)
    H0d fit half: at least one stack (a species with a record in the 6 years) per window
    H0e no test-half file passed to np.load (every load is logged)
Reported: stacks by recorded class at 2019, by records over 6 years, by detection class (1-5, 6-20, 21+ recorded
tetrads over 6 years); 2019 effort-passing tetrads per stack and the expected holes (30%); stacks with no 2019 effort
(no hole possible); stacks with a recorded and an unrecorded 2019 effort-passing tetrad (needed to be scored for
ranking); stacks with 10+ records in 2019 (the 7 Nov version); GB Red List 2022 status of the stacks and species.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_h0_bnm_check.py --self-test | tail -1
    python3 ../tools/stage_h0_bnm_check.py --out ../data/stage_h0_bnm_check.json | tee ../logs/stage_h0_bnm_check.txt
Defaults: --peryear ../../AI_simulation/Empirical_data/bnm_grids_peryear --minlist 8
          --split-ref "../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz" --obj1-tools ../../obj1_v2/tools
Needs export_obj2.py and stage_d_sampler.py in the same folder.
"""
import argparse, importlib.util, json, os, shutil, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_h0_bnm_check.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


D = _load("stage_d_sampler")
X = D.X
G, W = X.G, X.WINDOW
TCLS, BINS = D.TCLS, D.BINS
DCLS = ["1-5", "6-20", "21+"]                           # bnm_detection_check classes (recorded tetrads, any year)
HOLE = 0.30
BROKEN = os.environ.get("STAGE_H0_BROKEN", "")         # self-test only


def dcls(n):
    return np.where(n <= 5, 0, np.where(n <= 20, 1, 2))


class LoadLog:
    """Logs every np.load path while active."""
    def __enter__(self):
        self.orig, self.paths = np.load, []
        def rec(f, *a, **k):
            self.paths.append(str(f))
            return self.orig(f, *a, **k)
        np.load = rec
        return self

    def __exit__(self, *e):
        np.load = self.orig
        return False


def q(a):
    a = np.asarray(a)
    return "none" if not len(a) else f"median {np.median(a):.0f} (min {a.min()}, 5% {np.percentile(a, 5):.0f}, 95% {np.percentile(a, 95):.0f}, max {a.max()})"


def run(peryear, split_ref, tools, minlist=8, expect=(44, 57), out_json=None, out=print):
    res, info = {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    bdc, bmd5 = X.import_bdc(tools)
    out(f"stage_h0_bnm_check.py md5 {X.file_md5(__file__)}; bnm_detection_check.py md5 {bmd5}")
    pooled = Path(peryear) / f"minlist{minlist}"
    with LoadLog() as log:
        ref = bdc.split_ref(split_ref)
        names = sorted(f.name for f in pooled.glob("bnm_*.npz"))
        if not names:
            sys.exit(f"STOP: no window files in {pooled}")
        keep, east, overlap = bdc.fit_half(names, ref)
        if BROKEN == "testhalf":
            keep = names
        if not keep:
            sys.exit("STOP: no fit-half window in the per-year grids")
        ref_names = sorted(Path(f).name for f in __import__("glob").glob(split_ref))
        # ---- H0a years and names
        ydirs = sorted((int(p.name.rsplit("_y", 1)[1]), p) for p in Path(peryear).glob(f"minlist{minlist}_y*") if p.is_dir())
        a0 = np.load(pooled / keep[0], allow_pickle=True)
        ty = int(np.asarray(a0["target_year"]).item())
        years = [y for y, _ in ydirs] + [ty]
        consec = years == list(range(years[0], years[0] + len(years)))
        missing = [(y, nm) for y, d in ydirs for nm in names if not (d / nm).is_file()]
        gate("H0a", consec and len(years) == W and not missing,
             f"years {years} ({len(years)}, consecutive {consec}, ending at the pooled target year {ty}); window names missing "
             f"from a year folder {len(missing)}" + (f" (e.g. {missing[0]})" if missing else ""))
        gate("H0b", (len(keep), east) == tuple(expect) and overlap == 0,
             f"{len(names)} windows in {pooled}: fit half {len(keep)}, test half {east}, left out for sharing cells {overlap} "
             f"(expected {expect[0]} and {expect[1]}); cut at easting {ref['cut']} km, window side {ref['side']:.0f} km; "
             f"reference windows {len(ref_names)}, per-year names not in the reference {len(set(names) - set(ref_names))}")
        # ---- fit half
        rows_ok, inside, per_win, outside = True, True, [], []
        S = dict(rec_t=[], rec6=[], surv_t=[], unrec_t=[], status=[], species=[], window=[])
        for nm in keep:
            a = np.load(pooled / nm, allow_pickle=True)
            R, E, sp = [], [], []
            for y, d in ydirs:
                z = np.load(d / nm, allow_pickle=True)
                R.append(np.asarray(z["present_year"]).astype(bool)); E.append(np.asarray(z["known_holdout"]).astype(bool))
                sp.append(np.asarray(z["species"]) if "species" in z.files else None)
            R.append(np.asarray(a["obs_mask"]).astype(bool)); E.append(np.asarray(a["known_target"]).astype(bool))
            sp.append(np.asarray(a["species"]) if "species" in a.files else None)
            shapes = {r.shape for r in R} | {e.shape for e in E}
            named = [s for s in sp if s is not None]
            good = len(shapes) == 1 and next(iter(shapes))[1:] == (G, G) and len(named) == W and \
                all(np.array_equal(s, named[0]) for s in named[1:])
            rows_ok &= good
            if not good:
                per_win.append((nm, 0)); continue
            R, E = np.stack(R), np.stack(E)
            o = int((R & ~E).sum())
            if o:
                inside = False; outside.append((nm, o))
            k = np.nonzero(R.any((0, 2, 3)))[0]
            per_win.append((nm, len(k)))
            st = np.asarray(a["status"]) if "status" in a.files else np.array([""] * R.shape[1])
            S["rec_t"] += R[-1, k].sum((1, 2)).tolist(); S["rec6"] += R[:, k].any(0).sum((1, 2)).tolist()
            S["surv_t"] += E[-1, k].sum((1, 2)).tolist(); S["unrec_t"] += (E[-1, k] & ~R[-1, k]).sum((1, 2)).tolist()
            S["status"] += [str(st[i]) for i in k]; S["species"] += [str(named[0][i]) for i in k]; S["window"] += [nm] * len(k)
        gate("H0c", rows_ok and inside, f"fit half: species rows and shapes equal in all {W} years {rows_ok}; records outside "
             f"that year's effort {sum(o for _, o in outside)}" + (f" ({outside[:3]})" if outside else ""))
        gate("H0d", all(n > 0 for _, n in per_win), f"stacks per fit-half window {q([n for _, n in per_win])}; windows with "
             f"no stack {sum(n == 0 for _, n in per_win)}")
    fit_set = {nm for nm in keep} if BROKEN != "testhalf" else set(bdc.fit_half(names, ref)[0])
    touched_test = [p for p in log.paths if Path(p).name in set(names) - fit_set]
    gate("H0e", not touched_test, f"files passed to np.load {len(log.paths)}, of them test-half windows {len(touched_test)}")
    # ---- reported
    A = {k: np.array(v) for k, v in S.items()}
    n = len(A["rec_t"])
    if n:
        out(f"   fit-half stacks (a species with a record in {years[0]}-{years[-1]}): {n:,} in {len(keep)} windows, "
            f"{len(set(A['species'])):,} species")
        for lab, idx, labels in (("recorded class at 2019", D.tidx(A["rec_t"]), TCLS),
                                 ("records over 6 years", D.bidx(A["rec6"]), BINS),
                                 ("detection class (recorded tetrads over 6 years)", dcls(A["rec6"]), DCLS)):
            c = Counter(idx.tolist())
            out(f"   stacks by {lab}: " + "; ".join(f"{l} {c.get(i, 0):,}" for i, l in enumerate(labels)))
        holes = np.round(HOLE * A["surv_t"]).astype(int)
        score = (A["rec_t"] > 0) & (A["unrec_t"] > 0)
        out(f"   2019 effort-passing tetrads per stack: {q(A['surv_t'])}; expected holes (30%): {q(holes)}")
        out(f"   stacks with no 2019 effort-passing tetrad (no hole possible): {int((A['surv_t'] == 0).sum()):,}; with a recorded and "
            f"an unrecorded 2019 effort-passing tetrad (needed to be scored for ranking): {int(score.sum()):,}; with 10+ records in "
            f"2019 (7 Nov version): {int((A['rec_t'] >= 10).sum()):,}")
        st = Counter(A["status"].tolist())
        sps = {}
        for s_, sp_ in zip(A["status"], A["species"]):
            sps.setdefault(s_, set()).add(sp_)
        out("   GB Red List 2022 status (stacks / species): " + "; ".join(
            f"{k or '(none)'} {v:,} / {len(sps[k])}" for k, v in sorted(st.items(), key=lambda kv: -kv[1])))
        info = dict(stacks=n, species=len(set(A["species"])), windows_fit=len(keep), windows_test=east,
                    by_recorded_class={l: int((D.tidx(A["rec_t"]) == i).sum()) for i, l in enumerate(TCLS)},
                    by_records_6y={l: int((D.bidx(A["rec6"]) == i).sum()) for i, l in enumerate(BINS)},
                    by_detection_class={l: int((dcls(A["rec6"]) == i).sum()) for i, l in enumerate(DCLS)},
                    no_2019_effort=int((A["surv_t"] == 0).sum()), rankable=int(score.sum()), rec_2019_10plus=int((A["rec_t"] >= 10).sum()),
                    status={k: dict(stacks=v, species=len(sps[k])) for k, v in st.items()}, years=years)
    n_ok = sum(res.values())
    out(f"stage H0: gates {n_ok} PASS, {len(res) - n_ok} FAIL; fit half {len(keep)} windows read, test half {east} by name only")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(dict(gates=res, md5=dict(tool=X.file_md5(__file__), bdc=bmd5), **info), indent=1))
    return res, info


# ---------------------------------------------------------------- self-test
def self_test(tools):
    import subprocess
    res_all = []
    quiet = lambda *a, **k: None

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    def world(tmp, **kw):
        py, refpat = X.plant_grids(tmp, **kw)
        for f in sorted((Path(py) / "minlist8").glob("bnm_*.npz")):           # add GB Red List status as step 10 writes it
            z = dict(np.load(f)); S_ = len(z["species"])
            z["status"] = np.array(["LC", "NT", "VU", "EN", "CR", ""] * (S_ // 6 + 1))[:S_]
            np.savez(f, **z)
        return py, refpat

    with tempfile.TemporaryDirectory() as tmp:
        py, refpat = world(Path(tmp) / "a")
        r, info = run(py, refpat, tools, expect=(6, 6), out=quiet)
        # independent count of the planted stacks: species with a record in any of the 6 years, fit-half windows (eastings 0-80)
        n_ind, inside = 0, True
        for f in sorted((Path(py) / "minlist8").glob("bnm_*.npz")):
            if int(f.stem.split("_e")[1][:4]) >= 120:
                continue
            Rs = [np.load(Path(py) / f"minlist8_y{y}" / f.name)["present_year"] for y in range(2014, 2019)] + [np.load(f)["obs_mask"]]
            n_ind += int(np.stack(Rs).astype(bool).any((0, 2, 3)).sum())
        t("T1", all(r.values()) and set(r) == {"H0a", "H0b", "H0c", "H0d", "H0e"} and info["stacks"] == n_ind and info["windows_fit"] == 6,
          f"planted grids (12 windows, 6 fit half): every gate PASS, stacks {info.get('stacks')} = independent count {n_ind}")
        t("T2", sum(info["status"][k]["stacks"] for k in info["status"]) == info["stacks"] and "" in info["status"]
          and sum(info["by_recorded_class"].values()) == info["stacks"],
          "Red List table and class tables add up to the stack count; a species without status is shown as (none)")
        r, _ = run(py, refpat, tools, expect=(44, 57), out=quiet)
        t("T3", r["H0b"] is False and r["H0a"] and r["H0c"], "a split that does not give the expected 44 / 57 fails H0b")
        # record outside effort in a fit-half window
        py2, ref2 = world(Path(tmp) / "b")
        f = Path(py2) / "minlist8_y2016" / "bnm_e0040_n0000.npz"
        z = dict(np.load(f)); s0, y0, x0 = np.argwhere(z["known_holdout"] == 0)[0]
        z["present_year"] = z["present_year"].copy(); z["present_year"][s0, y0, x0] = 1; np.savez(f, **z)
        r, _ = run(py2, ref2, tools, expect=(6, 6), out=quiet)
        t("T4", r["H0c"] is False and r["H0e"], "one record outside its year's effort fails H0c")
        # a test-half window missing from a year folder (seen by name only)
        py3, ref3 = world(Path(tmp) / "c")
        os.remove(Path(py3) / "minlist8_y2015" / "bnm_e0200_n0040.npz")
        r, _ = run(py3, ref3, tools, expect=(6, 6), out=quiet)
        t("T5", r["H0a"] is False and r["H0e"], "a test-half window missing from one year folder fails H0a, found by name without opening it")
        # species rows differ between years
        py4, ref4 = X.plant_grids(Path(tmp) / "d", bad="rows")
        r, _ = run(py4, ref4, tools, expect=(6, 6), out=quiet)
        t("T6", r["H0c"] is False, "species rows that differ between years fail H0c")
        # a year folder missing
        py5, ref5 = world(Path(tmp) / "e")
        shutil.rmtree(Path(py5) / "minlist8_y2014")
        r, _ = run(py5, ref5, tools, expect=(6, 6), out=quiet)
        t("T7", r["H0a"] is False, "a missing year folder fails H0a (5 years)")
        # the tool reading test-half windows
        code = ("import importlib.util as u; s=u.spec_from_file_location('h', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                "r, _ = m.run(%r, %r, %r, expect=(6, 6), out=lambda *a, **k: None); print('RES', r['H0e'])"
                % (str(Path(__file__).resolve()), py, refpat, tools))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ, STAGE_H0_BROKEN="testhalf"))
        t("T8", "RES False" in p.stdout, "a version that opens the test-half windows fails H0e")
        out_j = Path(tmp) / "o" / "h0.json"
        run(py, refpat, tools, expect=(6, 6), out_json=out_j, out=quiet)
        j = json.loads(out_j.read_text())
        t("T9", all(j["gates"].values()) and j["stacks"] == n_ind and j["md5"]["tool"] == X.file_md5(__file__),
          "the json report holds the gates, the counts and the tool md5")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--split-ref", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--minlist", type=int, default=8)
    ap.add_argument("--expect", type=int, nargs=2, default=[44, 57])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    r, _ = run(a.peryear, a.split_ref, a.obj1_tools, a.minlist, tuple(a.expect), a.out)
    sys.exit(0 if all(r.values()) else 1)