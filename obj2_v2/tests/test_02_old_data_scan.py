#!/usr/bin/env python3
"""
test_02_old_data_scan.py  (obj2_v2, stage A, read-only, 8 Nov 2026)

What it does
    Reads every old training file the FIXAB run could use (PSD_Dispersal_pool/results/data/*_training.npz, the
    loader's own pattern) through Sara's own data_preprocessing.py (md5 checked first): get_world_metadata_fast for
    the loader's 20 x 20 filter, then SimulationWorld for what training saw. Prints, over the 20 x 20 files:
      D1 files, read errors, grid shapes, files kept (the loader keeps 20 x 20 only)
      D2 F1: species feature 4 (prevalence_final) against the true occupied share of the target frame
         (P_last_final, the training target), and whether it was stored in the file or computed from the target
      D3 species feature 5 = prevalence_any, and prevalence_any against the share of cells occupied in any saved frame
      D4 F2: range sizes at the target frame: share of species with 1, 5 or fewer, 10 or fewer, 20 or fewer cells
         (whole range given to the model in fixed-K batches with K 5, 10, 20)
      D5 F7: files with a species above 30 cells (the only files where the cap can fire)
      D6 the last saved frame P_t[-1] against the target frame (history records drawn from the target year?)
      D7 F3: biomass scale; history_B is log1p biomass divided by the 99.5th percentile of log1p biomass
         (normalize_biomass); the percentile is taken here over all kept files, training took it from a random
         sample of 30 training files, so the implied Poisson recording shares are approximate
    Writes nothing except what it prints.

Labels: PASS/FAIL = the code does what the audit says; FLAG = it works against the aim; INFO = a fact.

Run (from PSD_Dispersal_pool/obj2_v2/work, lvmeta environment)
    python3 ../tests/test_02_old_data_scan.py --self-test --stage2 ../original/stage2 2>/dev/null | tail -1
    python3 ../tests/test_02_old_data_scan.py --stage2 ../original/stage2 --old "../../results/data/*_training.npz" \
        2>/dev/null | tee ../logs/test_02_old_data_scan_8nov.txt
"""
import argparse, glob, hashlib, importlib.util, logging, math, os, sys, tempfile

import numpy as np

sys.dont_write_bytecode = True
EXPECT_MD5 = "f966d6276af61f94ea59935850899a9b"   # data_preprocessing.py in obj2_v2/original/stage2 (8 Nov)
BODY_MASS, P_OBS = 1e-4, (1e-4, 5e-4, 1e-3)       # training_v2_patch hook defaults


class Count(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.n = 0

    def emit(self, record):
        self.n += 1


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scan(mod, files):
    warn = Count()
    logging.getLogger(mod.__name__).addHandler(warn)
    rng = np.random.default_rng(0)
    r = dict(files=len(files), errors=[], grids={}, used=0, stored=0, computed=0, f1_eq=0, col4_ok=0, col5_ok=0,
             any_eq=0, last_eq=0, big=0, cells=[], zero=0, bmed=[], logs=[])
    for f in files:
        try:
            g = tuple(mod.get_world_metadata_fast(f)["grid_shape"])
        except Exception as e:
            r["errors"].append(f"{os.path.basename(f)}: {e}")
            continue
        r["grids"][g] = r["grids"].get(g, 0) + 1
        if g != (20, 20):
            continue
        try:
            w = mod.SimulationWorld(f)
        except Exception as e:
            r["errors"].append(f"{os.path.basename(f)}: {e}")
            continue
        r["used"] += 1
        n_cells = w.grid_y * w.grid_x
        P = np.asarray(w.P_last_final) > 0
        cells = P.reshape(P.shape[0], -1).sum(1)
        share = cells / n_cells
        keys = set(w.available_keys)
        r["stored" if keys & {"prevalence_final", "prevalence"} else "computed"] += 1
        pf = np.asarray(w.prevalence_final, dtype=float)
        r["f1_eq"] += pf.shape == share.shape and float(np.abs(pf - share).max()) < 1e-6
        feat = np.asarray(w.get_species_features(), dtype=float)
        r["col4_ok"] += feat.shape[0] == pf.shape[0] and float(np.abs(feat[:, 4] - pf).max()) < 1e-6
        pa0 = np.asarray(w.prevalence_any, dtype=float)
        r["col5_ok"] += feat.shape[0] == pa0.shape[0] and float(np.abs(feat[:, 5] - pa0).max()) < 1e-6
        anyshare = (np.asarray(w.P_t) > 0).any(axis=0).reshape(P.shape[0], -1).sum(1) / n_cells
        pa = np.asarray(w.prevalence_any, dtype=float)
        r["any_eq"] += pa.shape == anyshare.shape and float(np.abs(pa - anyshare).max()) < 1e-6
        r["last_eq"] += bool(np.array_equal(np.asarray(w.P_t)[-1] > 0, P))
        r["big"] += bool((cells > 30).any())
        r["zero"] += int((cells == 0).sum())
        r["cells"].extend(int(c) for c in cells if c > 0)
        B = np.asarray(w.B_last, dtype=float)
        pos = B[B > 0]
        if pos.size:
            r["bmed"].append(float(np.median(pos)))
            allpos = np.asarray(w.IBM_B, dtype=float)
            allpos = allpos[allpos > 0]
            r["logs"].extend(np.log1p(rng.choice(allpos, min(5000, allpos.size), replace=False)).tolist())
    r["warnings"] = warn.n
    return r


def report(r, out=print):
    rows = []

    def put(name, label, what):
        rows.append(label); out(f"{name} {what}: {label}")
    m = r["used"]
    out(f"D1 files {r['files']}, read errors {len(r['errors'])}" + (f" (first: {r['errors'][0][:120]})" if r["errors"] else "")
        + f", grid shapes {dict(sorted(r['grids'].items()))}, 20 x 20 files kept {m}, loader warnings {r['warnings']}")
    if not m:
        out("STOP: no 20 x 20 file read"); return rows
    out(f"D2 prevalence_final stored in the file in {r['stored']} files, computed from the target frame in {r['computed']}")
    put("D2a", "PASS" if r["col4_ok"] == m else "FAIL", f"species feature 4 = prevalence_final in {r['col4_ok']} of {m} files")
    put("D2b", "FLAG" if r["f1_eq"] else "PASS",
        f"prevalence_final = true occupied share of the target frame in {r['f1_eq']} of {m} files (F1)")
    put("D3a", "PASS" if r["col5_ok"] == m else "FAIL", f"species feature 5 = prevalence_any in {r['col5_ok']} of {m} files")
    put("D3b", "FLAG" if r["any_eq"] else "PASS",
        f"species feature 5 (prevalence_any) = share of cells occupied in any saved frame in {r['any_eq']} of {m} files")
    c = np.array(r["cells"]) if r["cells"] else np.array([0])
    n = len(r["cells"])
    out(f"D4 species with at least 1 cell at the target frame: {n} (and {r['zero']} with none); range size median "
        f"{int(np.median(c))}, max {int(c.max())}")
    for k in (1, 5, 10, 20):
        out(f"D4 share with {'1 cell' if k == 1 else f'{k} cells or fewer'}: {np.mean(c <= k):.3f}")
    put("D4", "FLAG" if n and np.mean(c <= 5) > 0 else "PASS",
        f"species given their whole range in fixed-K batches: {np.mean(c <= 5):.3f} at K 5, {np.mean(c <= 10):.3f} "
        f"at K 10, {np.mean(c <= 20):.3f} at K 20 (F2)")
    out(f"D5 files with a species above 30 cells (the cap can fire only there): {r['big']} of {m} (F7)")
    out(f"D6 last saved frame P_t[-1] = target frame in {r['last_eq']} of {m} files")
    if r["bmed"] and r["logs"]:
        bm = float(np.median(r["bmed"])); lmax = float(np.percentile(r["logs"], 99.5))
        norm = min(math.log1p(bm) / lmax, 1.0)
        ps = ", ".join(f"p {p:g}: {1 - math.exp(-norm / BODY_MASS * p):.2f}" for p in P_OBS)
        out(f"D7 biomass at the target frame: median of file medians {bm:.4g}; 99.5th percentile of log1p biomass "
            f"{lmax:.4g}; normalised median {norm:.3f}; implied Poisson recording share per occupied cell "
            f"(body mass {BODY_MASS:g}): {ps} (approximate; F3)")
    return rows


# ---------------------------------------------------------------- self-test
def planted(tmp):
    rng = np.random.default_rng(3)

    def ranges(sizes, T=3, shift_last=False, first_differs=True):
        S = len(sizes)
        P = np.zeros((T, S, 20, 20), np.int8)
        for s, k in enumerate(sizes):
            idx = rng.choice(400, k, replace=False)
            P[:, s].reshape(T, -1)[:, idx] = 1
            if first_differs:
                P[0, s].reshape(-1)[idx[0]] = 0
        target = P[-1].copy()
        if shift_last:
            P[-1, 0].reshape(-1)[np.flatnonzero(P[-1, 0].reshape(-1) == 0)[0]] = 1
        return P, target
    P, t = ranges([1, 3, 8, 25], first_differs=False)
    np.savez(f"{tmp}/a_training.npz", IBM_B=P * 0.4, P_t=P, P_last_final=t, prevalence_final=t.reshape(4, -1).sum(1) / 400)
    P, t = ranges([2, 40, 6])
    np.savez(f"{tmp}/b_training.npz", IBM_B=P * 0.4, P_t=P, P_last=t)
    P, t = ranges([4, 10], shift_last=True)
    np.savez(f"{tmp}/c_training.npz", IBM_B=P * 0.4, P_t=P, P_last_final=t, prevalence_final=t.reshape(2, -1).sum(1) / 800)
    Q = np.zeros((2, 2, 3, 3), np.int8); Q[:, :, 0, 0] = 1
    np.savez(f"{tmp}/d_training.npz", IBM_B=Q * 0.4, P_t=Q)
    P, t = ranges([5])
    np.savez(f"{tmp}/e_training.npz", P_t=P)
    return sorted(glob.glob(f"{tmp}/*_training.npz"))


def self_test(stage2):
    res = []

    def t(name, cond, what):
        res.append(bool(cond)); print(f"{name} {what}: {'PASS' if cond else 'FAIL'}")
    src = open(os.path.join(stage2, "data_preprocessing.py")).read()
    with tempfile.TemporaryDirectory() as tmp:
        files = planted(tmp)
        r = scan(load_module(os.path.join(stage2, "data_preprocessing.py"), "dp_selftest"), files)
        lines = []
        rows = report(r, out=lines.append)
        txt = "\n".join(lines)
        t("S1", r["files"] == 5 and len(r["errors"]) == 1 and r["grids"].get((3, 3)) == 1 and r["used"] == 3,
          "5 files: 1 read error (no biomass), 1 grid 3 x 3 left out, 3 kept")
        t("S2", r["stored"] == 2 and r["computed"] == 1 and r["f1_eq"] == 2 and r["col4_ok"] == 3,
          "prevalence stored in 2, computed in 1; equal to the true share in 2 of 3 (a half-size stored value is caught)")
        t("S3", len(r["cells"]) == 9 and "share with 5 cells or fewer: 0.444" in txt and "share with 20 cells or fewer: 0.778" in txt
          and "max 40" in txt, "range sizes 1, 3, 8, 25, 2, 40, 6, 4, 10: 4 of 9 at 5 cells or fewer, 7 of 9 at 20 or fewer, max 40")
        t("S4", r["big"] == 1 and r["last_eq"] == 2 and r["any_eq"] == 3,
          "1 file above 30 cells; last frame = target in 2 of 3 (one planted different); prevalence_any = any-frame share in 3")
        t("S5", rows.count("FLAG") == 3 and rows.count("PASS") == 2 and rows.count("FAIL") == 0, "labels: D2b, D3b, D4 FLAG; D2a, D3a PASS")
        key = "self.P_last_final.sum(axis=(1, 2))"
        if key in src:
            with open(f"{tmp}/data_preprocessing.py", "w") as fh:
                fh.write(src.replace(key, "self.P_t[0].sum(axis=(1, 2))"))
            rb = scan(load_module(f"{tmp}/data_preprocessing.py", "dp_broken"), files)
            t("S6", rb["f1_eq"] == 1, f"broken copy (prevalence from the first frame): equal in {rb['f1_eq']} of 3, drops from 2 to 1")
        else:
            t("S6", False, f"text {key!r} not found in data_preprocessing.py")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--old", default=None, help="quoted glob of old training files")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.stage2) else 1)
    path = os.path.join(a.stage2, "data_preprocessing.py")
    got = hashlib.md5(open(path, "rb").read()).hexdigest()
    if got != EXPECT_MD5:
        sys.exit(f"STOP: data_preprocessing.py md5 {got}, expected {EXPECT_MD5}")
    print(f"md5 data_preprocessing.py {got} OK")
    files = sorted(glob.glob(a.old or ""))
    if not files:
        sys.exit(f"STOP: no file matches {a.old}")
    rows = report(scan(load_module(path, "old_data_preprocessing"), files))
    print(f"{rows.count('PASS')} PASS, {rows.count('FAIL')} FAIL, {rows.count('FLAG')} FLAG")