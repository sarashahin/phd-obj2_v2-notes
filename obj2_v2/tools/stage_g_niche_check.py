#!/usr/bin/env python3
"""
stage_g_niche_check.py  (obj2_v2, stage G step G4a, read-only, 11 Nov 2026; G4 plan signed off by Sara on 11 Nov)

Question: can the niche of any species in a real_val community be rebuilt exactly? Process test T3 of G4b (losses fall
where the habitat fell) needs each stack species' growth rate in each of the 6 years. The obj1 files save the niche of
only 32 species each (NICHE_opt, NICHE_w, NICHE_peak) and one growth field (ENV_r_field, 32 x 20 x 20). If this check
fails, T3 is dropped and stated as a limit; T1 and T2 go on.

How. Sara's own niche_traits and niche_r are imported from obj1_v2/work/environment.py. Their parameters are read from
the function signatures and filled from the settings saved in each file (TRAIT_seed, NICHE_opt_sd, NICHE_w_min,
NICHE_w_max, NICHE_peak_max, NICHE_tradeoff, NICHE_cost); a parameter with neither a saved value nor a default stops the
check. Nothing about which species the 32 are, or which habitat ENV_r_field used, is assumed: both are searched.

Gates (rules written before the run, claude/obj2_v2_results_log_G_part2.md)
    A0  niche_traits and niche_r found; every parameter has a saved value or a default; the call is printed
    A1  in every file, each of the 32 saved species is matched by exactly one species id (searched first among the ids of
        the file's saved pieces, ASM_F_ids, then over 0 .. the largest id) whose rebuilt optimum, width and peak equal the
        saved values within 1e-6
    A2  niche_r with the rebuilt niches, on the habitat of one saved round (ENV_habitat_t, every round tried), equals
        ENV_r_field within 1e-6; the round is printed
    A3  controls that must fail: trait seed + 1 matches none of the 32 species; cost + 0.01 misses ENV_r_field by more
        than 1e-3
    A4  for every real_val community in the obj2 manifest: the obj1 file is the one exported (md5 = manifest source_md5)
        and every stack species id in its truth file is among the file's ASM_F_ids (only stack_species and stack_round are
        read from the truth files; no true map is opened)
real_eval is refused. Writes one json summary; nothing is written in the obj1 folders.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, minutes)
    python3 ../tools/stage_g_niche_check.py --self-test | tail -1
    USE_GPU=0 python3 ../tools/stage_g_niche_check.py --obj1-work ../../obj1_v2/work \\
        --files "../../obj1_v2/work/results/stage2a/real_val/*_training.npz" --data ../data --out ../logs/stage_g4a_niche.json
"""
import argparse, csv, glob, hashlib, importlib.util, inspect, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

TOL, CTRL_MIN, ID_CAP = 1e-6, 1e-3, 500000
SAVED = dict(seed="TRAIT_seed", opt_sd="NICHE_opt_sd", w_min="NICHE_w_min", w_max="NICHE_w_max",
             peak_max="NICHE_peak_max", tradeoff="NICHE_tradeoff", cost="NICHE_cost")
BROKEN = os.environ.get("STAGE_G4A_BROKEN", "")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scalar(z, k):
    v = np.asarray(z[k])
    return v.item() if v.ndim == 0 else v.ravel()[0]


def load_env(obj1_work):
    p = Path(obj1_work) / "environment.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found")
    sys.path.insert(0, str(Path(obj1_work).resolve()))      # environment.py may import its neighbours (as in G0)
    spec = importlib.util.spec_from_file_location("environment_obj1", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for f in ("niche_traits", "niche_r"):
        if not hasattr(m, f):
            sys.exit(f"STOP: {p} has no {f}")
    return m.niche_traits, m.niche_r, p


def fill(fn, z, n_data):
    """Keyword values for fn's parameters after the first n_data ones: saved value, else the default."""
    kw, src, missing = {}, {}, []
    for prm in list(inspect.signature(fn).parameters.values())[n_data:]:
        key = SAVED.get(prm.name)
        if key and key in z.files:
            v = scalar(z, key)
            kw[prm.name] = int(v) if prm.name == "seed" else float(v)
            src[prm.name] = key
        elif prm.default is not inspect.Parameter.empty:
            src[prm.name] = f"default {prm.default}"
        else:
            missing.append(prm.name)
    return kw, src, missing


def traits(fn, ids, kw):
    return np.array([fn(int(s), **kw)[:3] for s in ids], dtype=np.float64).reshape(-1, 3)


def match_saved(fn, kw, saved, ids0):
    """Species id per saved row: rows whose rebuilt (opt, w, peak) equal the saved ones within TOL."""
    ids = np.unique(np.asarray(ids0, np.int64))
    T = traits(fn, ids, kw)
    found = [[] for _ in range(len(saved))]
    for i, s in enumerate(saved):
        hit = np.nonzero(np.abs(T - s).max(1) <= TOL)[0]
        found[i] = ids[hit].tolist()
    if any(len(f) == 0 for f in found):                      # search the rest of the id range
        rest = np.setdiff1d(np.arange(min(int(ids.max()) + 1, ID_CAP)), ids)
        T2 = traits(fn, rest, kw)
        for i, s in enumerate(saved):
            if not found[i]:
                found[i] = rest[np.nonzero(np.abs(T2 - s).max(1) <= TOL)[0]].tolist()
    return found


def r_fields(fn_r, H, tr, kw_r):
    """niche_r of every species row on habitat H (G, G) -> (rows, G, G)."""
    return np.stack([np.asarray(fn_r(H, tr[i, 0], tr[i, 1], tr[i, 2], **kw_r), dtype=np.float64) for i in range(len(tr))])


def check_file(path, fn_t, fn_r, out=print):
    z = np.load(path, allow_pickle=True)
    name = Path(path).name
    need = ("NICHE_opt", "NICHE_w", "NICHE_peak", "ENV_r_field", "ENV_habitat_t", "ASM_F_ids")
    miss = [k for k in need if k not in z.files]
    if miss:
        out(f"   {name}: keys missing {miss}")
        return dict(file=name, A0=False, A1=False, A2=False, A3=False, missing=miss)
    kw_t, src_t, miss_t = fill(fn_t, z, 1)
    kw_r, src_r, miss_r = fill(fn_r, z, 4)
    a0 = not miss_t and not miss_r
    res = dict(file=name, A0=a0, call_traits=src_t, call_r=src_r, missing_params=miss_t + miss_r)
    if not a0:
        out(f"   {name}: A0 FAIL, no saved value or default for {miss_t + miss_r}")
        return dict(res, A1=False, A2=False, A3=False)
    saved = np.stack([np.asarray(z[k], np.float64) for k in ("NICHE_opt", "NICHE_w", "NICHE_peak")], 1)
    kw_use = dict(kw_t)
    if BROKEN == "seed_off":
        kw_use["seed"] = kw_use["seed"] + 1
    found = match_saved(fn_t, kw_use, saved, z["ASM_F_ids"])
    ids = [f[0] if len(f) == 1 else None for f in found]
    a1 = all(len(f) == 1 for f in found)
    tr = traits(fn_t, [i for i in ids if i is not None], kw_use) if a1 else None
    dtr = float(np.abs(tr - saved).max()) if a1 else float("nan")
    # A2: which saved round's habitat gives ENV_r_field
    R = np.asarray(z["ENV_r_field"], np.float64)
    Ht = np.asarray(z["ENV_habitat_t"], np.float64)
    best_t, best_d = None, float("inf")
    if a1 and R.shape[0] == len(tr):
        for t in range(len(Ht) - 1, -1, -1):
            d = float(np.abs(r_fields(fn_r, Ht[t], tr, kw_r) - R).max())
            if d < best_d:
                best_t, best_d = t, d
            if d <= TOL:
                break
    a2 = best_d <= TOL
    # A3 controls
    kw_c = dict(kw_t); kw_c["seed"] = kw_t["seed"] + 1
    T_c = traits(fn_t, np.unique(np.asarray(z["ASM_F_ids"])), kw_c)
    ctrl_seed = int(sum(bool((np.abs(T_c - s).max(1) <= TOL).any()) for s in saved))
    ctrl_cost = float("nan")
    if a2:
        kw_rc = dict(kw_r)
        if "cost" in kw_rc:
            kw_rc["cost"] = kw_rc["cost"] + 0.01
            ctrl_cost = float(np.abs(r_fields(fn_r, Ht[best_t], tr, kw_rc) - R).max())
    a3 = ctrl_seed == 0 and (not a2 or ctrl_cost > CTRL_MIN)
    res.update(A1=a1, A2=a2, A3=a3, n_saved=len(saved), species_ids=ids, n_matches=[len(f) for f in found],
               trait_max_diff=dtr, r_round=best_t, r_rounds=len(Ht), r_max_diff=best_d, ctrl_seed_matches=ctrl_seed,
               ctrl_cost_diff=ctrl_cost, n_piece_ids=int(len(np.unique(z["ASM_F_ids"]))))
    got = [i for i in ids if i is not None]
    out(f"   {name[-52:]}: {len(saved)} saved species matched {len(got)} (ids {min(got) if got else '-'}"
        f"-{max(got) if got else '-'}), trait diff {dtr:.1e}; r field = round {best_t} of {len(Ht)} "
        f"(diff {best_d:.1e}); controls: seed+1 matches {ctrl_seed}, cost+0.01 diff {ctrl_cost:.3f}")
    return res


def check_truth(data, files_by_name, out=print):
    """A4: obj1 file md5 = manifest source_md5; every stack species of the truth file is among the file's ASM_F_ids."""
    man = Path(data) / "real_val" / "manifest.csv"
    if not man.is_file():
        sys.exit(f"STOP: {man} not found")
    rows = list(csv.DictReader(open(man)))
    ok, lines, n_st, n_sp = True, [], 0, 0
    for r in rows:
        src = r["source"]
        if src not in files_by_name:
            ok = False; lines.append(f"{src} not among --files"); continue
        p = files_by_name[src]
        m_ok = md5(p) == r["source_md5"]
        z = np.load(p, allow_pickle=True)
        ids = set(np.unique(np.asarray(z["ASM_F_ids"])).tolist())
        with np.load(Path(data) / "real_val" / r["truth_file"], allow_pickle=False) as zt:
            sp = np.asarray(zt["stack_species"]).astype(np.int64)
            if BROKEN == "foreign_species":
                sp = sp.copy(); sp[0] = max(ids) + 7
        inside = bool(np.isin(sp, list(ids)).all())
        ok &= m_ok and inside
        n_st += len(sp); n_sp += len(np.unique(sp))
        if not (m_ok and inside):
            lines.append(f"{src}: md5 {m_ok}, species inside {inside}")
    out(f"   A4: {len(rows)} communities, {n_st:,} stacks, {n_sp:,} distinct (community, species) pairs; "
        + ("all obj1 files as exported and every stack species among the file's piece ids" if ok else "; ".join(lines[:5])))
    return ok, dict(communities=len(rows), stacks=n_st, species=n_sp, problems=lines)


def run(files, obj1_work, data, out_json=None, out=print):
    for p in list(files) + [str(data)]:
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, in stage G5")
    if not files:
        sys.exit("STOP: no files match --files")
    fn_t, fn_r, envp = load_env(obj1_work)
    out(f"stage_g_niche_check.py md5 {md5(__file__)}; niche functions from {envp} (md5 {md5(envp)})")
    out(f"   niche_traits{inspect.signature(fn_t)}; niche_r{inspect.signature(fn_r)}")
    res, per = {}, []
    t0 = time.time()
    for f in sorted(files):
        per.append(check_file(f, fn_t, fn_r, out=out))
    if per and per[0].get("call_traits"):
        out(f"   call: niche_traits(id, {per[0]['call_traits']}); niche_r(H, opt, w, peak, {per[0]['call_r']})")
    for g in ("A0", "A1", "A2", "A3"):
        res[g] = all(r.get(g, False) for r in per)
    a4, info4 = check_truth(data, {Path(f).name: f for f in files}, out=out)
    res["A4"] = a4
    texts = dict(A0="niche functions found; every parameter has a saved value or a default",
                 A1="each of the 32 saved species per file matched by exactly one species id (optimum, width, peak within 1e-6)",
                 A2="niche_r on one saved round's habitat equals ENV_r_field within 1e-6 in every file",
                 A3="controls fail as they must (trait seed + 1 matches none; cost + 0.01 misses by more than 1e-3)",
                 A4="obj1 files as exported; every truth-file stack species among the file's piece ids")
    for g in ("A0", "A1", "A2", "A3", "A4"):
        out(f"{g} {texts[g]}: {'PASS' if res[g] else 'FAIL'}")
    rounds = sorted({(r.get('r_round'), r.get('r_rounds')) for r in per if r.get('r_round') is not None})
    out(f"   ENV_r_field matched the habitat of round (index, rounds in file): {rounds[:6]}{' ...' if len(rounds) > 6 else ''}; "
        f"{time.time() - t0:.0f} s")
    if out_json:
        Path(out_json).write_text(json.dumps(dict(gates=res, files=per, truth=info4, tool=md5(__file__), env=str(envp),
                                                  env_md5=md5(envp)), indent=1, default=str))
    n_ok = sum(res.values())
    t3 = "T3 can run" if all(res[g] for g in ("A0", "A1", "A2", "A3", "A4")) else "T3 is dropped and stated (T1, T2 go on)"
    out(f"stage G4a niche check: {n_ok} PASS, {len(res) - n_ok} FAIL; {t3}")
    return res, per


# ---------------------------------------------------------------- self-test (stand-in environment and planted files)
ENV_PY = '''
import numpy as np
def niche_traits(sid, *, seed, opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=1.5, tradeoff=0.5{extra}):
    rng = np.random.default_rng([int(seed), 7919, int(sid)])
    opt = float(rng.normal(0.0, opt_sd))
    w = float(np.exp(rng.uniform(np.log(w_min), np.log(w_max))))
    return opt, w, float(peak_max * (w_min / w) ** tradeoff)
def niche_r(H, opt, w, peak, cost):
    return peak * np.exp(-(np.asarray(H) - opt) ** 2 / (2 * w * w)) - cost
'''


def plant(tmp, bug=None):
    """obj1-like real_val files and an obj2 data folder (manifest + truth files with stack_species)."""
    spec = importlib.util.spec_from_file_location("env_t", tmp / "work" / "environment.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    files = []
    (tmp / "obj1").mkdir(); (tmp / "data" / "real_val").mkdir(parents=True)
    rows = []
    for c, seed in enumerate((5101, 5102, 5103)):
        g = np.random.default_rng(seed)
        ids = np.unique(g.integers(0, 3000, 300))
        tseed = 200000 + seed
        keep = np.sort(g.choice(ids, 32, replace=False))
        prm = dict(seed=tseed, opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=1.5, tradeoff=0.5)
        tr = np.array([env.niche_traits(int(s), **prm) for s in keep])
        Ht = g.normal(0, 0.3, (40, 20, 20))
        cost = 0.1
        R = np.stack([env.niche_r(Ht[-1], *tr[i], cost + (0.05 if bug == "r_cost" else 0.0)) for i in range(32)])
        p = tmp / "obj1" / f"x_s2a_k1_f1_s{seed}_training.npz"
        np.savez(p, NICHE_opt=tr[:, 0].astype(np.float32), NICHE_w=tr[:, 1].astype(np.float32), NICHE_peak=tr[:, 2].astype(np.float32),
                 ENV_r_field=R.astype(np.float32), ENV_habitat_t=Ht.astype(np.float32), ASM_F_ids=np.repeat(ids, 3),
                 TRAIT_seed=np.int64(tseed + (1 if bug == "trait_seed" else 0)), NICHE_opt_sd=np.float64(1.5),
                 NICHE_w_min=np.float64(0.2), NICHE_w_max=np.float64(1.0), NICHE_peak_max=np.float64(1.5),
                 NICHE_tradeoff=np.float64(0.5), NICHE_cost=np.float64(cost), SIM_SEED=np.int64(seed))
        files.append(str(p))
        sp = g.choice(ids, 50)
        np.savez(tmp / "data" / "real_val" / f"s{seed}_truth.npz", stack_species=sp.astype(np.int32),
                 stack_round=np.full(50, 30, np.int32))
        rows.append(dict(sim_seed=seed, source=p.name, source_md5=md5(p) if bug != "md5" else "0" * 32,
                         truth_file=f"s{seed}_truth.npz", input_file=f"s{seed}_input.npz"))
    with open(tmp / "data" / "real_val" / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return files


def self_test():
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    def setup(tmp, bug=None, extra=""):
        (tmp / "work").mkdir(parents=True)
        (tmp / "work" / "environment.py").write_text(ENV_PY.format(extra=extra))
        return plant(tmp, bug)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        files = setup(tmp)
        lines = []
        r, per = run(files, tmp / "work", tmp / "data", out_json=tmp / "g4a.json", out=lines.append)
        t("S1", all(r.values()), f"planted files built by the rule pass A0-A4: {r}")
        t("S2", all(p["r_round"] == p["r_rounds"] - 1 for p in per) and all(p["trait_max_diff"] < 1e-6 for p in per),
          f"ENV_r_field found on the last round's habitat; trait differences {max(p['trait_max_diff'] for p in per):.1e} (float32 saving)")
        t("S3", all(p["ctrl_seed_matches"] == 0 and p["ctrl_cost_diff"] > 1e-3 for p in per),
          f"controls: seed+1 matches {[p['ctrl_seed_matches'] for p in per]}, cost+0.01 diff {[round(p['ctrl_cost_diff'], 3) for p in per]}")
        try:
            run([str(tmp / "real_eval_x.npz")], tmp / "work", tmp / "data", out=lambda *a, **k: None); refused = False
        except SystemExit:
            refused = True
        t("S4", refused, "a real_eval path is refused")
        print("   planted run log:", *lines, sep="\n      ")
    for bug, gate, extra in (("trait_seed", "A1", ""), ("r_cost", "A2", ""), ("md5", "A4", ""), (None, "A0", ", *, kappa"),
                             ("env_foreign", "A4", ""), ("env_seed_off", "A1", "")):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plant_bug = bug if bug in ("trait_seed", "r_cost", "md5") else None
            if extra:                                       # plant with the normal function, then add a parameter
                files = setup(tmp, None)
                (tmp / "work" / "environment.py").write_text(ENV_PY.format(extra="").replace(
                    "tradeoff=0.5):", "tradeoff=0.5, kappa_needed):"))
            else:
                files = setup(tmp, plant_bug)
            env = dict(os.environ, STAGE_G4A_BROKEN={"env_foreign": "foreign_species", "env_seed_off": "seed_off"}.get(bug, ""))
            code = ("import importlib.util as u; s=u.spec_from_file_location('n', %r); m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r,_=m.run(%r, %r, %r, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), files, str(tmp / "work"), str(tmp / "data")))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((x for x in p.stdout.splitlines() if x.startswith("RES ")), "")
            label = {"trait_seed": "file's TRAIT_seed not the one that drew the niches", "r_cost": "ENV_r_field made with another cost",
                     "md5": "obj1 file changed since the export", None: "a function parameter with no saved value or default",
                     "env_foreign": "a truth-file species not in the community", "env_seed_off": "traits rebuilt with the wrong seed"}[bug]
            t(f"bug {bug or 'missing_param'}", f"'{gate}': False" in line, f"{label}: {gate} fails ({line[4:] or p.stderr.strip()[-200:]})")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--obj1-work", default="../../obj1_v2/work")
    ap.add_argument("--files", default="../../obj1_v2/work/results/stage2a/real_val/*_training.npz")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../logs/stage_g4a_niche.json")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    res, _ = run(glob.glob(a.files), a.obj1_work, a.data, out_json=a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if all(res.values()) else 1)