#!/usr/bin/env python3
"""
stage_g5_export_check.py  (obj2_v2, stage G5 part G5b, 11 Nov 2026; G5 rules signed off by Sara on 11 Nov)

Question: G5-1 of the signed G5 rules. Is real_eval exported by the same code, with the same habitat noise, survey
windows and years as real_train and real_val, complete (each of the obj1 real_eval files once), and with no community
or input stack shared with the other splits? It answers from manifests, input files and settings only. No true map,
true cell count, exact habitat, species id or detection value is read, so the truth stays closed until G5d.

Two modes
    --pre   before the export, nothing written to data: export_obj2.py md5 = the export code md5 stored in every
            real_train and real_val input file; habitat_sigma.json = the sigma in real_val; the G5a fits match their
            report; data/real_eval does not exist; no obj1 real_eval file (md5 or seed) is in the real_train or real_val
            manifest. If any check fails, do not export.
    (post)  after export_obj2.py --export --split real_eval --final-eval, gates E0-E6 below; writes
            g5b_export_check.json with the md5 of every real_eval file, the list G5c and G5d check against.

Truth files: only these keys are read, through a reader that refuses any other key: code_md5, habitat_sigma, pair_md5,
pieces, round_window, sim_seed, source_md5, split, window_names, years (settings of the export, no answer). All other
work runs under stage D's TruthGuard, which refuses any np.load of a *_truth.npz file.

Gates (post)
    E0  order: g5_fits.npz md5 = its report, written before the oldest real_eval file; no G5d marker in data/real_eval
    E1  complete: one manifest row per obj1 real_eval file, source md5 and seed equal; input and truth file md5 = the
        manifest; every truth file says split real_eval
    E2  same export: one export code md5 in every real_train, real_val and real_eval file; habitat sigma in every
        real_eval truth file = habitat_sigma.json = real_val's; years, pieces and survey window names = real_val's;
        every round's window index inside that list
    E3  split: no community seed or source file in two of the three folders
    E4  no copied input stack: no real_eval input stack with 5 or more recorded cells over the 6 years whose effort,
        visit 1 and visit 2 equal a real_train or real_val input stack (stage C audit rule L6); repeats inside
        real_eval are reported
    E5  pairing: input keys = the export allow-list; each input file's pair md5 = its truth file's; every per-stack
        array has the manifest's stack count; records per year = the visits; habitat index inside the file
    E6  truth closed: no truth file passed to np.load; only the keys above read from truth files
    Reported: stacks with 1+ record over the 6 years by recorded class at the target year (inputs only) and what the
    stage D rule (G5-2) will take: every stack of classes 0 and 1, 1,500 per other class, all of a class if fewer.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_g5_export_check.py --self-test | tail -1
    python3 ../tools/stage_g5_export_check.py --pre --sim "../../obj1_v2/work/results/stage2a/real_eval/*_training.npz"
    (export, only if --pre passes)
    python3 ../tools/stage_g5_export_check.py --sim "../../obj1_v2/work/results/stage2a/real_eval/*_training.npz" \\
        --out ../data/stage_g5b_export_check
Defaults: --data ../data, --fits ../data/stage_g5_fits, --sigma-file ../data/habitat_sigma.json,
          --export-tool ../tools/export_obj2.py. Needs numpy, export_obj2.py and stage_d_sampler.py in the same folder.
"""
import argparse, csv, glob, hashlib, importlib.util, json, os, subprocess, sys, tempfile, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_g5_export_check.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


X = _load("export_obj2")
D = _load("stage_d_sampler")
SPLITS = ("real_train", "real_val", "real_eval")
META = ("code_md5", "habitat_sigma", "pair_md5", "pieces", "round_window", "sim_seed", "source_md5", "split", "window_names",
        "years")
MARKER = "G5D_TRUTH_OPENED"
MIN_REC6 = 5
BROKEN = os.environ.get("STAGE_G5B_BROKEN", "")   # self-test only: plants one bug so its gate must fail
META_READ = []                                    # (file, key) of every truth key read


def truth_meta(path, keys=META):
    """Settings keys of a truth file, read from the zip members one by one; any key outside META is refused."""
    if BROKEN == "meta_all":
        keys = tuple(keys) + ("true_cells",)
    out = {}
    with zipfile.ZipFile(path) as zf:
        for k in keys:
            if k not in META and BROKEN != "meta_all":
                raise KeyError(f"truth key {k} is not a settings key; refused")
            META_READ.append((Path(path).name, k))
            with zf.open(k + ".npy") as f:
                out[k] = np.lib.format.read_array(f, allow_pickle=False)
    return out


def member(path, key):
    """One array of an obj1 file or input file, without np.load (used for SIM_SEED of the obj1 files)."""
    with zipfile.ZipFile(path) as zf, zf.open(key + ".npy") as f:
        return np.lib.format.read_array(f, allow_pickle=False)


def rows_of(d):
    p = Path(d) / "manifest.csv"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found")
    with open(p) as f:
        return list(csv.DictReader(f))


def input_stacks(d, row):
    p = Path(d) / row["input_file"]
    with np.load(p, allow_pickle=False) as z:
        ii = {k: z[k] for k in z.files}
    return ii


def stack_hashes(ii):
    v = X.unpack(ii["visit1"]) | X.unpack(ii["visit2"])
    n = len(v)
    rec6 = v.any(1).reshape(n, -1).sum(1)
    out = {}
    for j in np.nonzero(rec6 >= MIN_REC6)[0]:
        h = hashlib.md5(ii["effort"][j].tobytes() + ii["visit1"][j].tobytes() + ii["visit2"][j].tobytes()).digest()
        out.setdefault(h, []).append(int(j))
    return out, v, rec6


def fits_info(fits_dir):
    fp, rp = Path(fits_dir) / "g5_fits.npz", Path(fits_dir) / "g5_fits_report.json"
    if not fp.is_file() or not rp.is_file():
        return None, None, None
    rep = json.loads(rp.read_text())
    return X.file_md5(fp), rep["md5"]["fits"], datetime.fromisoformat(rep["written_utc"])


# ---------------------------------------------------------------- before the export
def pre(sim, data, fits, sigma_file, export_tool, out=print):
    res = {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    data = Path(data)
    sims = sorted(glob.glob(sim))
    out(f"stage_g5_export_check.py md5 {X.file_md5(__file__)}; mode --pre (nothing written)")
    tool_md5 = X.file_md5(export_tool)
    stored = set()
    for sp in ("real_train", "real_val"):
        for row in rows_of(data / sp):
            with np.load(data / sp / row["input_file"], allow_pickle=False) as z:
                stored.add(str(z["code_md5"]))
    gate("P1", stored == {tool_md5}, f"export_obj2.py md5 {tool_md5} = the export code md5 in every real_train and real_val input "
         f"file {sorted(stored)}")
    js = json.loads(Path(sigma_file).read_text())
    vs = {float(truth_meta(data / "real_val" / row["truth_file"], ("habitat_sigma",))["habitat_sigma"]) for row in rows_of(data / "real_val")}
    gate("P2", bool(js.get("ok")) and vs == {float(js["sigma"])}, f"habitat_sigma.json sigma {js.get('sigma')} (real_val check ok "
         f"{js.get('ok')}) = the sigma in every real_val truth file {sorted(vs)}")
    m, mr, when = fits_info(fits)
    gate("P3", m is not None and m == mr, f"G5a fits g5_fits.npz md5 {m} = its report {mr}, written {when}")
    seen = {}
    for sp in ("real_train", "real_val"):
        for row in rows_of(data / sp):
            seen[row["source_md5"]] = sp; seen["seed " + row["sim_seed"]] = sp
    hits = []
    for p in sims:
        s = str(int(np.asarray(member(p, "SIM_SEED")).item()))
        if seen.get(X.file_md5(p)) or seen.get("seed " + s):
            hits.append(Path(p).name)
    ev = data / "real_eval"
    gate("P4", not ev.exists() and len(sims) > 0 and not hits and all("real_eval" in p for p in sims),
         f"{ev} does not exist {not ev.exists()}; {len(sims)} obj1 real_eval files, none in the real_train or real_val manifest "
         f"(by md5 or seed) {not hits}" + (f" ({', '.join(hits[:5])})" if hits else ""))
    n_ok = sum(res.values())
    out(f"stage G5b pre-check: {n_ok} PASS, {len(res) - n_ok} FAIL; " + ("export may run" if n_ok == len(res) else "do not export"))
    return res


# ---------------------------------------------------------------- after the export
def post(sim, data, fits, sigma_file, out_dir, out=print):
    data, out_dir = Path(data), Path(out_dir)
    if "real_eval" in str(out_dir):
        sys.exit("STOP: the check writes nothing into a real_eval folder; give another --out")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    res, t0 = {}, time.time()
    del META_READ[:]

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    out(f"stage_g5_export_check.py md5 {X.file_md5(__file__)}; export_obj2.py {X.file_md5(HERE / 'export_obj2.py')}; "
        f"stage_d_sampler.py {X.file_md5(HERE / 'stage_d_sampler.py')}")
    ev = data / "real_eval"
    rows = {sp: rows_of(data / sp) for sp in SPLITS}
    sims = sorted(glob.glob(sim))
    info = {}
    with D.TruthGuard() as guard:
        try:
            # ---- E0 order
            m, mr, when = fits_info(fits)
            files_ev = [ev / r[k] for r in rows["real_eval"] for k in ("truth_file", "input_file")] + [ev / "manifest.csv"]
            oldest = min(datetime.fromtimestamp(os.path.getmtime(f), timezone.utc) for f in files_ev if f.is_file())
            marker = (ev / MARKER).exists()
            gate("E0", m is not None and m == mr and when < oldest and not marker,
                 f"g5_fits.npz md5 {m} = its report {mr}; fits written {when.isoformat() if when else None}, oldest real_eval file "
                 f"{oldest.isoformat(timespec='seconds')} (fits first {bool(when and when < oldest)}); G5d marker absent {not marker}")
            # ---- E1 complete
            ev_rows = rows["real_eval"]
            by_src = {r["source"]: r for r in ev_rows}
            bad1 = []
            for p in sims:
                r = by_src.get(Path(p).name)
                if r is None:
                    bad1.append(f"{Path(p).name} not exported"); continue
                if r["source_md5"] != X.file_md5(p) or r["sim_seed"] != str(int(np.asarray(member(p, "SIM_SEED")).item())):
                    bad1.append(f"{Path(p).name} source md5 or seed differs")
            if len(ev_rows) != len(sims):
                bad1.append(f"{len(ev_rows)} manifest rows for {len(sims)} obj1 files")
            meta_ev, md5_ev = {}, {}
            for r in ev_rows:
                for k, k5 in (("truth_file", "truth_md5"), ("input_file", "input_md5")):
                    f = ev / r[k]
                    md5_ev[r[k]] = X.file_md5(f) if f.is_file() else None
                    if md5_ev[r[k]] != r[k5]:
                        bad1.append(f"{r[k]} md5 differs from the manifest")
                meta_ev[r["truth_file"]] = mt = truth_meta(ev / r["truth_file"])
                if str(mt["split"]) != "real_eval" or str(int(mt["sim_seed"])) != r["sim_seed"] or str(mt["source_md5"]) != r["source_md5"]:
                    bad1.append(f"{r['truth_file']} split, seed or source differs")
            gate("E1", not bad1 and len(sims) > 0, f"{len(sims)} obj1 real_eval files, {len(ev_rows)} exported communities, "
                 f"{sum(int(r['stacks']) for r in ev_rows)} species-stacks; each file once with its md5 and seed, input and truth "
                 f"md5 = manifest, split real_eval" + (f"; problems: {'; '.join(bad1[:5])}" if bad1 else ""))
            # ---- E2 same export
            codes = {}
            inputs = {}
            for sp in SPLITS:
                for r in rows[sp]:
                    with np.load(data / sp / r["input_file"], allow_pickle=False) as z:
                        codes.setdefault(str(z["code_md5"]), set()).add(sp)
            for r in ev_rows:
                codes.setdefault(str(meta_ev[r["truth_file"]]["code_md5"]), set()).add("real_eval truth")
            js = json.loads(Path(sigma_file).read_text())
            val_meta = [truth_meta(data / "real_val" / r["truth_file"], ("habitat_sigma", "years", "pieces", "window_names"))
                        for r in rows["real_val"]]
            sig_ev = {float(mt["habitat_sigma"]) for mt in meta_ev.values()}
            sig_val = {float(mt["habitat_sigma"]) for mt in val_meta}
            ref = val_meta[0]
            same_y = all(np.array_equal(mt["years"], ref["years"]) and np.array_equal(mt["pieces"], ref["pieces"])
                         and np.array_equal(mt["window_names"], ref["window_names"]) for mt in list(meta_ev.values()) + val_meta)
            win_ok = all(((mt["round_window"] >= 0) & (mt["round_window"] < len(mt["window_names"]))).all() for mt in meta_ev.values())
            ok2 = len(codes) == 1 and sig_ev == sig_val == {float(js["sigma"])} and same_y and win_ok
            gate("E2", ok2, f"export code md5 {sorted(codes)} in {sorted(set().union(*codes.values()))}; habitat sigma real_eval "
                 f"{sorted(sig_ev)}, real_val {sorted(sig_val)}, habitat_sigma.json {js['sigma']}; years "
                 f"{[int(y) for y in ref['years']]}, pieces {[int(p) for p in ref['pieces']]} and {len(ref['window_names'])} survey "
                 f"windows equal to real_val {same_y}; every round window inside the list {win_ok}")
            # ---- E3 split
            seeds = {}
            for sp in SPLITS:
                for r in rows[sp]:
                    seeds.setdefault(("seed", r["sim_seed"]), set()).add(sp)
                    seeds.setdefault(("md5", r["source_md5"]), set()).add(sp)
            both = sorted(f"{k[0]} {k[1][:12]} in {sorted(v)}" for k, v in seeds.items() if len(v) > 1)
            gate("E3", not both, f"no community seed or source file in two of the three folders ({len(rows['real_train'])}, "
                 f"{len(rows['real_val'])}, {len(ev_rows)} communities)" + (f"; shared: {'; '.join(both[:5])}" if both else ""))
            # ---- E4 copy and E5 pairing
            ref_h = set()
            for sp in ("real_train", "real_val"):
                for r in rows[sp]:
                    h, _, _ = stack_hashes(input_stacks(data / sp, r))
                    ref_h |= set(h)
            hits, dup_in, n_inf, bad5 = [], 0, 0, []
            ev_h = {}
            rec_t_all, rec6_all = [], []
            for r in ev_rows:
                ii = input_stacks(ev, r)
                h, v, rec6 = stack_hashes(ii)
                n_inf += sum(len(j) for j in h.values())
                for k, js_ in h.items():
                    if k in ref_h:
                        hits.append(f"seed {r['sim_seed']} row {js_[0]}")
                    if k in ev_h:
                        dup_in += len(js_)
                    ev_h.setdefault(k, []).extend(js_)
                n = int(r["stacks"])
                rec_t_all.append(v[:, -1].reshape(len(v), -1).sum(1)); rec6_all.append(rec6)
                mt = meta_ev[r["truth_file"]]
                keys_ok = sorted(ii) == sorted(X.ALLOW)
                rows_ok = all(len(ii[k]) == n for k in ("effort", "visit1", "visit2", "n_rec", "stack_hab"))
                nrec_ok = rows_ok and np.array_equal(v.reshape(n, X.WINDOW, -1).sum(2), ii["n_rec"])
                hab_ok = rows_ok and bool(((ii["stack_hab"] >= 0) & (ii["stack_hab"] < len(ii["habitat_noisy"]))).all())
                pair_ok = str(ii["pair_md5"]) == str(mt["pair_md5"])
                if BROKEN == "truth_load":
                    np.load(ev / r["truth_file"])
                if not (keys_ok and rows_ok and nrec_ok and hab_ok and pair_ok):
                    bad5.append(f"seed {r['sim_seed']}: keys {keys_ok}, rows {rows_ok}, records {nrec_ok}, habitat index {hab_ok}, "
                                f"pair {pair_ok}")
            gate("E4", not hits, f"real_eval input stacks with {MIN_REC6}+ recorded cells over 6 years {n_inf}; equal to a real_train "
                 f"or real_val input stack {len(hits)}" + (f" ({'; '.join(hits[:5])})" if hits else "")
                 + f"; repeats inside real_eval (reported) {dup_in}")
            gate("E5", not bad5, f"input keys = allow-list, pair md5 = truth file, per-stack arrays = manifest count, records per year "
                 f"= visits, habitat index inside the file, in all {len(ev_rows)} input files" + (f"; problems: {'; '.join(bad5[:3])}" if bad5 else ""))
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("E6", False, f"truth closed: {crashed}")
        return res, None
    other = sorted({k for _, k in META_READ if k not in META})
    gate("E6", not guard.opened and not other, f"truth closed: truth files passed to np.load {len(guard.opened)}; keys read from truth "
         f"files {sorted({k for _, k in META_READ})} (settings only {not other})")
    # ---- report: recorded classes (inputs only)
    rec_t, rec6 = np.concatenate(rec_t_all), np.concatenate(rec6_all)
    cls = D.tidx(rec_t)
    elig = rec6 >= 1
    out(f"   stacks: {len(rec_t)}; with 1+ record over 6 years {int(elig.sum())} (no record {int((~elig).sum())}, never used)")
    take, short = [], []
    for c, lab in enumerate(D.TCLS):
        n_c = int((elig & (cls == c)).sum())
        t_c = n_c if (c in D.SPARSE or n_c <= D.N_PER_CLASS) else D.N_PER_CLASS
        if c not in D.SPARSE and n_c < D.N_PER_CLASS:
            short.append(f"{lab} ({n_c})")
        take.append(t_c)
        out(f"   recorded class {lab:>6}: {n_c:>8} stacks with 1+ record; the stage D rule takes {t_c}")
    out(f"   frozen real_eval set by the stage D rule: {sum(take)} stacks" + (f"; classes short of 1,500: {', '.join(short)}" if short else
                                                                            "; no class short of 1,500"))
    # ---- write
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = dict(gates=res, md5=dict(tool=X.file_md5(__file__), export_tool=X.file_md5(HERE / "export_obj2.py"),
                                   manifest=X.file_md5(ev / "manifest.csv"), files=md5_ev, fits=m),
               code_md5=sorted(codes), habitat_sigma=sorted(sig_ev), communities=[int(r["sim_seed"]) for r in ev_rows],
               stacks=int(len(rec_t)), stacks_with_record=int(elig.sum()),
               recorded_class={lab: int((elig & (cls == c)).sum()) for c, lab in enumerate(D.TCLS)},
               frozen_take={lab: int(t) for lab, t in zip(D.TCLS, take)}, truth_keys_read=sorted({k for _, k in META_READ}),
               checked_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), minutes=round((time.time() - t0) / 60, 2))
    (out_dir / "g5b_export_check.json").write_text(json.dumps(rep, indent=1))
    mj = X.file_md5(out_dir / "g5b_export_check.json")
    n_ok = sum(res.values())
    out(f"   written g5b_export_check.json md5 {mj} ({(time.time() - t0) / 60:.1f} min)")
    out(f"stage G5b export check: gates {n_ok} PASS, {len(res) - n_ok} FAIL; real_eval manifest md5 {rep['md5']['manifest']}")
    return res, rep


# ---------------------------------------------------------------- self-test
def rewrite(path, fn):
    """Load an npz, change it, save it with the export's writer."""
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    fn(a)
    X.save_npz(path, a)


def fix_manifest(d):
    """Make the manifest md5s match the files again (to isolate a gate other than E1)."""
    rows = rows_of(d)
    for r in rows:
        r["truth_md5"] = X.file_md5(Path(d) / r["truth_file"]); r["input_md5"] = X.file_md5(Path(d) / r["input_file"])
    with open(Path(d) / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def plant(root, tools, fits_late=False, eval_sigma=0.3, n_eval=3):
    """real_train (2), real_val (2) and real_eval (n_eval) exported by export_obj2.py from planted communities; the
    G5a fits folder written before the real_eval export (or after it, fits_late)."""
    root = Path(root)
    quiet = lambda *a, **k: None
    py, refpat = X.plant_grids(root / "g")
    sims = {}
    for sp, seeds in (("real_train", (8001, 8002)), ("real_val", (8101, 8102)), ("real_eval", (8201, 8202, 8203))):
        d = root / "sim" / sp; d.mkdir(parents=True)
        for s in seeds:
            X.plant_community(d / f"c{s}_training.npz", s, n_species=10, rounds=20)
        sims[sp] = str(d / "*_training.npz")
    data = root / "data"
    for sp in ("real_train", "real_val"):
        X.export(sorted(glob.glob(sims[sp])), sp, str(data / sp), 0.3, py, refpat, tools, out=quiet)
    (data / "habitat_sigma.json").write_text(json.dumps(dict(sigma=0.3, ok=True, val=0.80)))
    fits = root / "fits"
    def write_fits():
        fits.mkdir()
        X.save_npz(fits / "g5_fits.npz", dict(a=np.arange(3)))
        (fits / "g5_fits_report.json").write_text(json.dumps(dict(md5=dict(fits=X.file_md5(fits / "g5_fits.npz")),
                                                                  written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))))
    if not fits_late:
        write_fits(); time.sleep(1.2)
    ev_paths = sorted(glob.glob(sims["real_eval"]))[:n_eval]
    X.export(ev_paths, "real_eval", str(data / "real_eval"), eval_sigma, py, refpat, tools, final_eval=True, out=quiet)
    if fits_late:
        time.sleep(1.2); write_fits()
    return dict(data=data, fits=fits, sim=sims["real_eval"], py=py, refpat=refpat, sims=sims)


def self_test(tools):
    quiet = lambda *a, **k: None
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tmp / "ok"
        # pre-check on the clean state, before the real_eval export
        p0 = plant(base, tools)
        ev_dir = p0["data"] / "real_eval"
        hold = tmp / "hold_eval"
        os.rename(ev_dir, hold)                                   # as before the export
        lines = []
        rp = pre(p0["sim"], p0["data"], p0["fits"], p0["data"] / "habitat_sigma.json", HERE / "export_obj2.py", out=lines.append)
        t("S1", all(rp.values()) and len(rp) == 4, f"pre-check on the planted state before the export: {rp}")
        os.rename(hold, ev_dir)
        rp2 = pre(p0["sim"], p0["data"], p0["fits"], p0["data"] / "habitat_sigma.json", HERE / "export_obj2.py", out=quiet)
        t("S2", rp2["P4"] is False, f"pre-check after the export fails P4 (real_eval exists): {rp2}")
        # post on the clean export
        plines = []
        r, rep = post(p0["sim"], p0["data"], p0["fits"], p0["data"] / "habitat_sigma.json", tmp / "out_ok", out=plines.append)
        t("S3", all(r.values()) and set(r) == {"E0", "E1", "E2", "E3", "E4", "E5", "E6"} and rep["stacks"] > 0
          and rep["truth_keys_read"] and set(rep["truth_keys_read"]) <= set(META),
          f"planted export end to end: {r}; {rep['stacks']} stacks, truth keys read {rep['truth_keys_read']}")
        u1 = False
        try:
            truth_meta(ev_dir / rows_of(ev_dir)[0]["truth_file"], ("true_cells",))
        except KeyError:
            u1 = True
        t("U1", u1, "the truth reader refuses a key outside the settings list (true_cells)")
        stop = []
        for od in (tmp / "out_ok", ev_dir / "check"):
            try:
                post(p0["sim"], p0["data"], p0["fits"], p0["data"] / "habitat_sigma.json", od, out=quiet); stop.append(False)
            except SystemExit:
                stop.append(True)
        t("S4", all(stop), f"an output folder that holds outputs is refused {stop[0]}; an output folder inside real_eval is refused {stop[1]}")
        # planted pre-check bug: a different export_obj2.py
        other_tool = tmp / "export_obj2_other.py"; other_tool.write_text((HERE / "export_obj2.py").read_text() + "\n# changed\n")
        os.rename(ev_dir, hold)
        rpb = pre(p0["sim"], p0["data"], p0["fits"], p0["data"] / "habitat_sigma.json", other_tool, out=quiet)
        os.rename(hold, ev_dir)
        t("bug pre_other_tool", rpb["P1"] is False, f"a changed export_obj2.py fails P1 ({rpb})")

        def case(name, gate, setup, fresh=None):
            d = tmp / name
            if fresh is None:
                subprocess.run(["cp", "-r", str(base), str(d)], check=True)
                p = dict(p0, data=d / "data", fits=d / "fits", sim=str(d / "sim" / "real_eval" / "*_training.npz"))
            else:
                p = fresh(d)
            setup(p)
            try:
                rr, _ = post(p["sim"], p["data"], p["fits"], p["data"] / "habitat_sigma.json", d / "out", out=quiet)
            except SystemExit as e:
                rr = {"STOP": str(e)}
            t(f"bug {name}", rr.get(gate) is False, f"{gate} fails as it must ({rr})")

        ev = lambda p: p["data"] / "real_eval"
        case("marker", "E0", lambda p: (ev(p) / MARKER).write_text("x"))
        case("late_fits", "E0", lambda p: None, fresh=lambda d: plant(d, tools, fits_late=True))
        case("missing", "E1", lambda p: None, fresh=lambda d: plant(d, tools, n_eval=2))
        case("changed_input", "E1", lambda p: rewrite(ev(p) / rows_of(ev(p))[0]["input_file"], lambda a: a.__setitem__(
            "habitat_noisy", (a["habitat_noisy"] + np.float16(0.5)).astype(np.float16))))

        def other_code(p):
            r0 = rows_of(ev(p))[0]
            rewrite(ev(p) / r0["input_file"], lambda a: a.__setitem__("code_md5", np.array("0" * 32)))
            fix_manifest(ev(p))
        case("other_code", "E2", other_code)
        case("other_sigma", "E2", lambda p: None, fresh=lambda d: plant(d, tools, eval_sigma=0.5))

        def overlap(p):
            rv = rows_of(p["data"] / "real_val")[0]
            for k in ("truth_file", "input_file"):
                (ev(p) / rv[k].replace("s81", "s91")).write_bytes((p["data"] / "real_val" / rv[k]).read_bytes())
            rows = rows_of(ev(p))
            new = dict(rv); new["truth_file"] = rv["truth_file"].replace("s81", "s91"); new["input_file"] = rv["input_file"].replace("s81", "s91")
            rows.append(new)
            with open(ev(p) / "manifest.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        case("overlap", "E3", overlap)

        def copy_stack(p):
            rt = rows_of(p["data"] / "real_train")[0]
            with np.load(p["data"] / "real_train" / rt["input_file"]) as z:
                src = {k: z[k] for k in z.files}
            h, _, _ = stack_hashes(src)
            j = next(iter(h.values()))[0]
            def put(a):
                for k in ("effort", "visit1", "visit2", "n_rec"):
                    a[k] = a[k].copy(); a[k][0] = src[k][j]
            rewrite(ev(p) / rows_of(ev(p))[0]["input_file"], put)
            fix_manifest(ev(p))
        case("copy_stack", "E4", copy_stack)

        def pair(p):
            rewrite(ev(p) / rows_of(ev(p))[0]["input_file"], lambda a: a.__setitem__("pair_md5", np.array("1" * 32)))
            fix_manifest(ev(p))
        case("pair", "E5", pair)
        for bug in ("truth_load", "meta_all"):
            env = dict(os.environ, STAGE_G5B_BROKEN=bug)
            code = ("import importlib.util as u; s=u.spec_from_file_location('f', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "from pathlib import Path; d=Path(%r); r,_=m.post(%r, d, %r, d / 'habitat_sigma.json', %r, out=lambda *a, **k: None); "
                    "print('RES', r)" % (str(Path(__file__).resolve()), str(p0["data"]), p0["sim"], str(p0["fits"]), str(tmp / f"b_{bug}")))
            pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((x for x in pr.stdout.splitlines() if x.startswith("RES ")), "")
            t(f"bug {bug}", "'E6': False" in line, f"E6 fails as it must ({line[4:] or pr.stderr.strip()[-200:]})")
        print("   planted pre-check log:", *lines, sep="\n      ")
        print("   planted post-check log:", *plines, sep="\n      ")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pre", action="store_true")
    ap.add_argument("--sim", default="../../obj1_v2/work/results/stage2a/real_eval/*_training.npz")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--fits", default="../data/stage_g5_fits")
    ap.add_argument("--sigma-file", default="../data/habitat_sigma.json")
    ap.add_argument("--export-tool", default="../tools/export_obj2.py")
    ap.add_argument("--out", default="../data/stage_g5b_export_check")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    if a.pre:
        r = pre(a.sim, a.data, a.fits, a.sigma_file, a.export_tool, out=lambda s: print(s, flush=True))
        sys.exit(0 if r and all(r.values()) else 1)
    r, _ = post(a.sim, a.data, a.fits, a.sigma_file, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)