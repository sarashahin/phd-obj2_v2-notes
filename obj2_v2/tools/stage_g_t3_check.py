#!/usr/bin/env python3
"""
stage_g_t3_check.py  (obj2_v2, stage G step G4c, 11 Nov 2026; agreed by Sara 11 Nov as a report-only check on real_val
before G5, after G4b found T3 FAIL for both seeds)

Question: why do the model's maps miss the true extra loss where the growth rate fell (T3)? The model's only habitat
input is the noisy habitat of the target year (stage E channel 18), blanked in half the frozen stacks. If that is the
cause, the model's T3 should appear, if anywhere, in the last change (year 5 to 6, into the target year) and only where
the habitat was shown.

What it reports (real_val frozen stacks, both seeds; no pass rule, nothing here changes a verdict)
    T3 by change: for each of the 5 year-to-year changes, among cells occupied in year t, the share lost where the
        growth rate fell minus the share lost where it rose or stayed (as in G4b, per change instead of averaged);
        per stack, the model value is the mean over the samples that define it.
    D2 loss in sink cells: the same split by the growth rate of year t+1 itself, r < 0 (the species declines there)
        against r >= 0. This separates "losses follow how poor the habitat is" (level, which the target-year habitat
        could show) from "losses follow where the habitat got worse" (change, which needs the earlier years).
    Each split by habitat shown and habitat blanked (frozen set "blank"). Class-balanced means over true classes 1 .. 51+
    with 95% intervals from resampling the 20 communities (stage_f_check.class_boot), truth and model side by side.

Reading, written before the run
    R1  with habitat shown, the model's T3 in the last change is at least half the truth's, with its interval above 0;
    R2  with habitat shown, the model's T3 in each of changes 1-4 is below a quarter of the truth's;
    R3  with habitat blanked, the model's T3 in the last change is below a quarter of the truth's.
    Each rule is read only where the truth's mean is above 0 (R1: its interval); otherwise it prints None.
    R1, R2 and R3 together: the model uses the target-year habitat for losses and lacks the earlier years' habitat, so
    giving it each year's habitat is the likely fix (later work). Model T3 near 0 everywhere, R1 failing: the model does
    not use habitat for losses at all, and yearly habitat alone may not fix it. D2 is read the same way for habitat level.

Gates
    C1  inputs: samples_k50.npz = g1_report.json for both runs; frozen set = the md5 in g4b_report.json; truth files =
        manifest truth_md5; obj1 files = manifest source_md5; G4b report and per-stack file present
    C2  T3 inputs rebuilt as in G4b (habitat of every used round = ENV_habitat_t to 1e-5; saved niches and ENV_r_field
        reproduced to 1e-6)
    C3  tie to G4b: per stack, T3 averaged over changes and samples as in G4b equals per_stack_g4b.csv (truth, seed 1,
        seed 2) to its 6 significant digits; the model statistics are written and md5-logged before the true maps open
    real_eval is refused; the output folder must be new.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_g_t3_check.py --self-test | tail -1
    python3 -u ../tools/stage_g_t3_check.py --data ../data --runs ../runs/f_seed1 ../runs/f_seed2 \\
        --obj1-work ../../obj1_v2/work --files "../../obj1_v2/work/results/stage2a/real_val/*_training.npz" \\
        --g4b ../data/stage_g4b_process --out ../data/stage_g4c_t3check > ../logs/stage_g4c_t3check.txt 2>&1
Needs stage_g_process_tests.py (G4b) and the tools it loads in the same folder.
"""
import argparse, csv, glob, importlib.util, json, os, sys, tempfile, time, warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g_t3_check.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


PT = _load("stage_g_process_tests")
CK, HC, NC, X, D = PT.CK, PT.HC, PT.NC, PT.X, PT.D
G, W, K, TRUE, SCORE = PT.G, PT.W, PT.K, PT.TRUE, PT.SCORE
BROKEN = os.environ.get("STAGE_G4C_BROKEN", "")       # self-test only
CH = [f"{t + 1}-{t + 2}" for t in range(W - 1)]       # changes between years; year 6 is the target year


def parts(A, fell, sink):
    """A (c, k, 6, G, G); fell, sink (c, 5, G, G) -> d_fell, d_sink (c, k, 5), nan where a group is empty."""
    prev, nxt = A[:, :, :-1], A[:, :, 1:]
    lost = prev & ~nxt
    out = []
    for F in (fell, sink):
        F = F[:, None]
        nf, lf = (prev & F).sum((-1, -2)), (lost & F).sum((-1, -2))
        nr, lr = (prev & ~F).sum((-1, -2)), (lost & ~F).sum((-1, -2))
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where((nf > 0) & (nr > 0), lf / np.maximum(nf, 1) - lr / np.maximum(nr, 1), np.nan))
    return out


def growth(z, env, pairs):
    """Per (round, species): fell (r_t+1 < r_t) and sink (r_t+1 < 0), each (5, G, G); and the C2 checks (G4b rules)."""
    p = HC.params(z)
    kw_t, _, miss_t = NC.fill(env.niche_traits, z, 1)
    kw_r, _, miss_r = NC.fill(env.niche_r, z, 4)
    if miss_t or miss_r:
        return None, dict(ok=False, why=f"no saved value or default for {miss_t + miss_r}")
    Ht = np.asarray(z["ENV_habitat_t"], np.float64); Hs = np.asarray(z["ENV_habitat_shift"], np.float64)
    hab, dmax = {}, 0.0
    for r in sorted({int(r) for r, _ in pairs}):
        maps, _ = HC.rebuild_round(Ht[r - 1], Hs[r - 1], r, p, env.habitat_layer)
        dmax = max(dmax, float(np.abs(maps[-1] - Ht[r]).max())); hab[r] = maps[p["M"] - W:]
    saved = np.stack([np.asarray(z[k], np.float64) for k in ("NICHE_opt", "NICHE_w", "NICHE_peak")], 1)
    found = NC.match_saved(env.niche_traits, kw_t, saved, z["ASM_F_ids"])
    a1 = all(len(f) == 1 for f in found)
    dr = float("inf")
    if a1:
        tr = NC.traits(env.niche_traits, [f[0] for f in found], kw_t)
        R = np.asarray(z["ENV_r_field"], np.float64)
        if R.shape[0] == len(tr):
            dr = float(np.abs(NC.r_fields(env.niche_r, Ht[-1], tr, kw_r) - R).max())
    niche, res = {}, {}
    for r, s in pairs:
        if s not in niche:
            niche[s] = tuple(float(v) for v in env.niche_traits(int(s), **kw_t)[:3])
        rr = np.stack([np.asarray(env.niche_r(hab[r][t], *niche[s], **kw_r), np.float64) for t in range(W)])
        fell = rr[1:] < rr[:-1]
        if BROKEN == "shift_t":
            fell = np.roll(fell, 1, axis=0)
        res[(r, s)] = (fell, rr[1:] < 0)
    return res, dict(ok=dmax <= PT.TOL_HAB and a1 and dr <= PT.TOL_R, hab_diff=dmax, r_diff=dr, rounds=len(hab))


def bal(v, tc, comm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return CK.class_boot(v, tc, comm, classes=SCORE)["balanced"]


def run(data, runs, obj1_files, obj1_work, g4b, out_dir, chunk=256, out=print):
    data, g4b, out_dir = Path(data), Path(g4b), Path(out_dir)
    runs = [Path(r) for r in runs]; val_dir = data / "real_val"
    for p in [data, g4b, out_dir, *runs, *obj1_files, obj1_work]:
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G5")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    res, t0 = {}, time.time()

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    env, envp = PT.load_env(obj1_work)
    fp = data / "stage_d_frozen_val.npz"
    frozen = dict(np.load(fp)); fmd5 = X.file_md5(fp); n = len(frozen["row"])
    rep4 = json.loads((g4b / "g4b_report.json").read_text())
    out(f"stage_g_t3_check.py md5 {X.file_md5(__file__)}; stage_g_process_tests.py {X.file_md5(HERE / 'stage_g_process_tests.py')}; "
        f"environment.py {X.file_md5(envp)}; frozen set {fmd5} ({n} stacks); G4b report by tool {rep4['md5']['tool'][:8]}")
    # C1
    ok1, why, maps = True, [], {}
    for j, r in enumerate(runs):
        g1 = json.loads((r / "check_g1" / "g1_report.json").read_text())
        ms = X.file_md5(r / "check_g1" / "samples_k50.npz"); z = np.load(r / "check_g1" / "samples_k50.npz")
        good = ms == g1["md5"]["samples"] and np.array_equal(z["sel"], np.arange(n)); ok1 &= good
        why.append(f"{r.name} samples {ms[:8]} {good}"); maps[j] = z["maps"]
    good = fmd5 == rep4["md5"]["frozen"]; ok1 &= good; why.append(f"frozen set = g4b_report.json {good}")
    man = {row["input_file"]: row for row in D.rows_of(val_dir)}
    by_name = {Path(f).name: f for f in obj1_files}
    files, fidx, rows, blank = frozen["files"], frozen["file_index"], frozen["row"], frozen["blank"].astype(bool)
    comm_of, t_ok, o_ok = {}, True, True
    for i in np.unique(fidx):
        row = man[str(files[i])]; comm_of[int(i)] = row
        t_ok &= X.file_md5(val_dir / row["truth_file"]) == row["truth_md5"]
        o_ok &= row["source"] in by_name and X.file_md5(by_name.get(row["source"], fp)) == row["source_md5"]
    g4csv = g4b / "per_stack_g4b.csv"
    gate("C1", ok1 and t_ok and o_ok and g4csv.is_file(), "; ".join(why) + f"; truth files {t_ok}; obj1 files {o_ok}; "
         f"habitat blanked in {blank.mean():.3f} of stacks")
    # ids and C2
    sp = np.zeros(n, np.int64); rd = np.zeros(n, np.int64); comm = np.zeros(n, np.int64)
    for i, row in comm_of.items():
        k = np.nonzero(fidx == i)[0]
        with np.load(val_dir / row["truth_file"]) as zt:
            sp[k] = zt["stack_species"][rows[k]]; rd[k] = zt["stack_round"][rows[k]]
        comm[k] = int(row["sim_seed"])
    fell = np.zeros((n, W - 1, G, G), bool); sink = np.zeros_like(fell); ok2, worst = True, [0.0, 0.0]
    for i, row in comm_of.items():
        k = np.nonzero(fidx == i)[0]
        r_, info = growth(np.load(by_name[row["source"]], allow_pickle=True), env, sorted(set(zip(rd[k].tolist(), sp[k].tolist()))))
        if r_ is None:
            ok2 = False; out(f"   {row['source']}: {info['why']}"); continue
        for j in k:
            fell[j], sink[j] = r_[(int(rd[j]), int(sp[j]))]
        ok2 &= info["ok"]; worst = [max(worst[0], info["hab_diff"]), max(worst[1], info["r_diff"])]
    gate("C2", ok2, f"T3 inputs rebuilt as in G4b in {len(comm_of)} communities (largest habitat diff {worst[0]:.1e}, r field diff {worst[1]:.1e})")
    out(f"   growth rate fell in {fell.mean():.3f} of cells per change; sink cells (r < 0 in year t+1) {sink.mean():.3f}")
    # model statistics before any true map
    M = {(s, q): np.full((n, W - 1), np.nan) for s in (0, 1) for q in ("fell", "sink")}
    Mall = {s: np.full(n, np.nan) for s in (0, 1)}
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        for s in (0, 1):
            df, ds = parts(PT.unpack6(maps[s][a:b]), fell[a:b], sink[a:b])
            M[(s, "fell")][a:b], M[(s, "sink")][a:b] = PT._nanmean(df, 1), PT._nanmean(ds, 1)
            Mall[s][a:b] = PT._nanmean(PT._nanmean(df, -1), 1)
    sfile = out_dir / "g4c_model_stats.npz"
    np.savez(sfile, **{f"s{s + 1}_{q}": v for (s, q), v in M.items()}, **{f"s{s + 1}_all": v for s, v in Mall.items()})
    smd5 = X.file_md5(sfile)
    out(f"   model statistics written before any true map was opened: {sfile.name} md5 {smd5} ({(time.time() - t0) / 60:.1f} min)")
    # truth
    T = np.zeros((n, W, G, G), bool); tc = np.zeros(n, int)
    for i, row in comm_of.items():
        k = np.nonzero(fidx == i)[0]
        with np.load(val_dir / row["truth_file"]) as zt:
            T[k] = X.unpack(zt["true_maps"][rows[k]]); tc[k] = D.tidx(zt["true_cells"][rows[k]])
    if BROKEN == "stats_changed":
        M[(0, "fell")][:] = np.nan_to_num(M[(0, "fell")]); Mall[0][:] = 0.0
    tf, ts = parts(T[:, None], fell, sink)
    Tt = {"fell": tf[:, 0], "sink": ts[:, 0]}
    Tall = PT._nanmean(tf[:, 0], -1)
    # C3
    with open(g4csv) as fh:
        g = list(csv.DictReader(fh))
    ref = {c: np.array([float(r[c]) for r in g]) for c in ("truth_T3", "s1_T3_mean", "s2_T3_mean")}

    def close(a, b):
        return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=2e-5, atol=1e-9))
    tie = [close(Tall, ref["truth_T3"]), close(Mall[0], ref["s1_T3_mean"]), close(Mall[1], ref["s2_T3_mean"])]
    gate("C3", all(tie) and X.file_md5(sfile) == smd5 and len(g) == n,
         f"T3 per stack averaged as in G4b = per_stack_g4b.csv (truth {tie[0]}, seed 1 {tie[1]}, seed 2 {tie[2]}); model "
         f"statistics file unchanged after the truth was opened")
    # report
    report = dict(gates=res, md5=dict(model_stats=smd5, frozen=fmd5, tool=X.file_md5(__file__), environment=X.file_md5(envp)),
                  tables={}, reading={})
    groups = (("shown", ~blank), ("blanked", blank))
    for s in (0, 1):
        for q, lab in (("fell", "T3 (growth rate fell)"), ("sink", "D2 (sink cells, r < 0)")):
            out(f"   seed {s + 1}, {lab}: class-balanced mean over true classes 1 .. 51+ [95% interval]; truth | model | model/truth")
            for gname, gm in groups:
                for t in range(W - 1):
                    tv = np.where(gm, Tt[q][:, t], np.nan); mv = np.where(gm & ~np.isnan(tv), M[(s, q)][:, t], np.nan)
                    tv = np.where(~np.isnan(mv), tv, np.nan)
                    bt, bm = bal(tv, tc, comm), bal(mv, tc, comm)
                    ratio = bm["mean"] / bt["mean"] if bt["mean"] else float("nan")
                    report["tables"][f"seed {s + 1} {q} {gname} {CH[t]}"] = dict(truth=bt, model=bm, ratio=ratio)
                    out(f"      habitat {gname:7s} change {CH[t]}{' (into target year)' if t == W - 2 else ''}: n {bm['n']:6,}; "
                        f"{bt['mean']:+.4f} [{bt['lo']:+.4f}, {bt['hi']:+.4f}] | {bm['mean']:+.4f} [{bm['lo']:+.4f}, {bm['hi']:+.4f}] | {ratio:.2f}")
            tb = report["tables"]
            k = lambda g_, t: tb[f"seed {s + 1} {q} {g_} {CH[t]}"]
            # each rule is read only where the truth's mean is above 0 (by its interval for R1); else None (not applicable)
            last = k("shown", W - 2)
            r1 = (last["ratio"] >= 0.5 and last["model"]["lo"] > 0) if last["truth"]["lo"] > 0 else None
            below = lambda e: (e["model"]["mean"] < 0.25 * e["truth"]["mean"]) if e["truth"]["mean"] > 0 else None
            b2 = [below(k("shown", t)) for t in range(W - 2)]
            r2 = None if None in b2 else all(b2)
            r3 = below(k("blanked", W - 2))
            report["reading"][f"seed {s + 1} {q}"] = dict(R1=r1, R2=r2, R3=r3)
            out(f"      reading ({q}): R1 {r1}, R2 {r2}, R3 {r3}")
    np.savez(out_dir / "g4c_truth_stats.npz", fell=Tt["fell"], sink=Tt["sink"], all=Tall, true_class=tc, comm=comm, blank=blank)
    (out_dir / "g4c_report.json").write_text(json.dumps(report, indent=1, default=float))
    out(f"stage G4c T3 check: gates {sum(res.values())} PASS, {len(res) - sum(res.values())} FAIL; "
        + "; ".join(f"{k} R1 {v['R1']} R2 {v['R2']} R3 {v['R3']}" for k, v in report["reading"].items())
        + f" ({(time.time() - t0) / 60:.1f} min)")
    return res, report


# ---------------------------------------------------------------- self-test
def self_test():
    out_ = []

    def t(name, ok, what):
        out_.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    A = np.zeros((1, 1, W, G, G), bool); A[0, 0, :, 0, 0:4] = True; A[0, 0, 5, 0, 0] = False
    fl = np.zeros((1, W - 1, G, G), bool); fl[0, :, 0, 0:2] = True
    sk = np.zeros_like(fl); sk[0, :, 0, 0] = True
    df, ds = parts(A, fl, sk)
    t("U1", np.allclose(df[0, 0], [0, 0, 0, 0, 0.5]) and np.allclose(ds[0, 0], [0, 0, 0, 0, 1.0]),
      f"a cell lost only in the last change: T3 by change {df[0, 0].tolist()}, sink split {ds[0, 0].tolist()}")
    A2 = np.zeros((1, 1, W, G, G), bool); A2[0, 0, :, 1, 1] = True
    df, ds = parts(A2, fl, sk)
    t("U2", np.all(np.isnan(df)) and np.all(np.isnan(ds)), "only one kind of cell occupied: undefined in every change")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); P = PT.plant(tmp / "p")
        fz = P["data"] / "stage_d_frozen_val.npz"; z = dict(np.load(fz))
        z["blank"] = (np.arange(len(z["row"])) % 2).astype(np.uint8); np.savez(fz, **z)
        b3 = P["g2"] / "maps_B3_halfblank.npz"; zb = dict(np.load(b3)); zb["frozen_md5"] = np.array(X.file_md5(fz)); np.savez(b3, **zb)
        (P["g2"] / "g2_report.json").write_text(json.dumps(dict(md5={"maps_B3_halfblank.npz": X.file_md5(b3)})))
        PT.run(P["data"], P["runs"], P["g2"], P["files"], P["work"], tmp / "g4b", out=lambda *a, **k: None)
        lines = []
        g, rep = run(P["data"], P["runs"], P["files"], P["work"], tmp / "g4b", tmp / "o1", chunk=64, out=lines.append)
        t("S1", all(g.values()), f"planted communities pass C1-C3 {g}")
        tb = rep["tables"]
        r1 = [tb[f"seed 1 fell {gm} {c}"]["ratio"] for gm in ("shown", "blanked") for c in CH]
        t("S2", all(0.7 <= r <= 1.3 for r in r1), "samples from the truth's own process match the truth's T3 in every change "
          f"and both habitat groups (model/truth {min(r1):.2f}-{max(r1):.2f})")
        r2 = [tb[f"seed 2 fell {gm} {c}"]["ratio"] for gm in ("shown", "blanked") for c in CH]
        t("S3", np.mean(r2) < 0.75 * np.mean(r1), f"static samples (years from different chains) show less on average: "
          f"model/truth {np.mean(r2):.2f} against {np.mean(r1):.2f} (by change {min(r2):.2f}-{max(r2):.2f}; a static sample still "
          f"loses more where the habitat fell in year t+1, so part of T3 survives)")
        print("   planted run log:", *lines, sep="\n      ")
        for j, (bug, gate_) in enumerate((("shift_t", "C3"), ("stats_changed", "C3"))):
            env = dict(os.environ, STAGE_G4C_BROKEN=bug)
            code = ("import importlib.util as u; s=u.spec_from_file_location('c', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "g,_=m.run(%r, [%r, %r], %r, %r, %r, %r, out=lambda *a, **k: None); print('RES', g)"
                    % (str(Path(__file__).resolve()), str(P["data"]), str(P["runs"][0]), str(P["runs"][1]), P["files"], str(P["work"]),
                       str(tmp / "g4b"), str(tmp / f"bug{j}")))
            import subprocess
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate_}': False" in line, f"{gate_} fails ({line[4:] or p.stderr.strip()[-300:]})")
        zf = dict(np.load(fz)); zf["blank"] = 1 - zf["blank"]; np.savez(fz, **zf)
        try:
            g, _ = run(P["data"], P["runs"], P["files"], P["work"], tmp / "g4b", tmp / "o3", out=lambda *a, **k: None)
            t("bug frozen_changed", not g["C1"], f"C1 fails when the frozen set differs from the one G4b used {g}")
        except SystemExit as e:
            t("bug frozen_changed", False, f"stopped: {e}")
        try:
            run(P["data"], P["runs"], P["files"], P["work"], tmp / "g4b", tmp / "o1", out=lambda *a, **k: None); ok = False
        except SystemExit:
            ok = True
        t("S4", ok, "an output folder that holds outputs is refused")
    print(f"{sum(out_)} PASS, {len(out_) - sum(out_)} FAIL")
    return all(out_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--runs", nargs=2, default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--obj1-work", default="../../obj1_v2/work")
    ap.add_argument("--files", default="../../obj1_v2/work/results/stage2a/real_val/*_training.npz")
    ap.add_argument("--g4b", default="../data/stage_g4b_process")
    ap.add_argument("--out", default="../data/stage_g4c_t3check")
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    g, _ = run(a.data, a.runs, sorted(glob.glob(a.files)), a.obj1_work, a.g4b, a.out, chunk=a.chunk, out=lambda s: print(s, flush=True))
    sys.exit(0 if all(g.values()) else 1)