#!/usr/bin/env python3
"""
test_01_old_sparsifiers.py  (obj2_v2, stage A, read-only, 8 Nov 2026)

What it does
    Runs Sara's own copies of the old observation code (obj2_v2/original/stage2, md5 checked first) on planted
    inputs, and, if given one old training file, checks two facts in it. Writes nothing except what it prints.
      T1 F2  training_v2_patch._sparsify_all_frames_fixed_K with K 5, 10, 20 (the FIXAB log: hist sparsify K
             [5, 10, 20]): are ranges of K cells or fewer given whole?
      T2 F7  EcoDiffusionSpatial._ensure_sparse: if one species has more than the threshold, is every species cut
             to the cap?
      T3 F3  training_v2_patch._sparsify_all_frames_poisson: recorded share per occupied cell = 1 - exp(-N p);
             can a cell with zero presence be recorded?
      T4 F6  EcoDiffusionSpatial._obs_decay: nearest record (max) or sum? across the edge?
      T5 F9  which files use history_B
      T6 F1  (needs --old) is the stored prevalence_final the true occupied share of the final frame? how many
             species hold more than the cap threshold, so the cap can fire at all?
    _ensure_sparse and _obs_decay are methods; their text is read from the file with ast and run with the
    threshold, cap and sigma read from EcoDiffusionSpatial.__init__ defaults, so no model is built. The decay
    kernel is rebuilt with the formula of __init__ on the 20 x 20 grid.

Labels (as in test_00): PASS/FAIL = the old code does what the audit finding says; FLAG = it works against the
aim; INFO = a fact to read.

Run (from PSD_Dispersal_pool/obj2_v2/work, lvmeta environment, needs torch and numpy)
    python3 ../tests/test_01_old_sparsifiers.py --self-test --stage2 ../original/stage2 | tail -1
    grep -m1 "simulation dir" ../../stage2_outputs_SPATIAL_FIXAB_partial/logs/train.log
    python3 ../tests/test_01_old_sparsifiers.py --stage2 ../original/stage2 --old "<simulation dir>/*.npz" \
        | tee ../logs/test_01_old_sparsifiers_8nov.txt
"""
import argparse, ast, glob, hashlib, importlib.util, math, os, shutil, sys, tempfile, textwrap, types

sys.dont_write_bytecode = True   # original/ is read-only; never write __pycache__ there

EXPECT = {"training_v2_patch.py": "4cee2fc8dd1b33813323510fc6dd91a9",
          "models/ecodiffusion_spatial_cond.py": "30d7997a0e8a44dbb58a50fbdbe90d28"}
G = 20  # old grid, 20 x 20


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class Log:
    def __init__(self, quiet=False):
        self.rows, self.quiet = [], quiet

    def put(self, name, label, what):
        self.rows.append((name, label))
        if not self.quiet:
            print(f"{name} {what}: {label}")

    def info(self, text):
        if not self.quiet:
            print(text)

    def label(self, name):
        return [l for n, l in self.rows if n == name]


def load_fixed_and_poisson(path):
    spec = importlib.util.spec_from_file_location("old_training_v2_patch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # imports torch, logging, sys; auto_install only warns (training not imported)
    return mod._sparsify_all_frames_fixed_K, mod._sparsify_all_frames_poisson


def load_methods(path, torch):
    src = open(path).read()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "EcoDiffusionSpatial")
    fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    init = fns["__init__"].args
    defaults = {a.arg: ast.literal_eval(d) for a, d in zip(init.kwonlyargs, init.kw_defaults) if d is not None}
    ns = {"torch": torch, "F": torch.nn.functional, "Dict": dict, "Optional": object, "Tuple": tuple}
    out = {}
    for name in ("_ensure_sparse", "_obs_decay"):
        code = textwrap.dedent(ast.get_source_segment(src, fns[name], padded=True))
        exec(code, ns)
        out[name] = ns[name]
    return out, defaults


def kernel(torch, sigma):
    yy, xx = torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij")
    c = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return torch.exp(-d2 / (sigma ** 2))


def planted_ranges(torch, sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.zeros(1, 1, len(sizes), G, G)
    for s, n in enumerate(sizes):
        idx = torch.randperm(G * G, generator=g)[:n]
        h[0, 0, s].view(-1)[idx] = 1.0
    return h


def run_checks(stage2, log, old=None, check_md5=True):
    import torch
    if check_md5:
        for rel, want in EXPECT.items():
            got = md5(os.path.join(stage2, rel))
            if got != want:
                sys.exit(f"STOP: {rel} md5 {got}, expected {want} (the copy is not the one audited)")
            log.info(f"md5 {rel} {got} OK")
    fixed_k, poisson = load_fixed_and_poisson(os.path.join(stage2, "training_v2_patch.py"))
    meth, d = load_methods(os.path.join(stage2, "models/ecodiffusion_spatial_cond.py"), torch)
    thr, cap, sigma = d["dense_obs_threshold"], d["K_obs_cap"], d["obs_decay_sigma"]
    log.info(f"EcoDiffusionSpatial defaults: dense_obs_threshold {thr}, K_obs_cap {cap}, obs_decay_sigma {sigma}")

    # T1 (F2)
    sizes = (3, 5, 10, 20, 50, 200)
    h = planted_ranges(torch, sizes)
    g = torch.Generator().manual_seed(1)
    inside, exact, small, full = True, True, 0, 0
    for K in (5, 10, 20):
        out = fixed_k(h, K, generator=g)
        inside &= bool((out <= h).all())
        for s, n in enumerate(sizes):
            r = int(out[0, 0, s].sum())
            exact &= r == min(n, K)
            if n <= K:
                small += 1
                full += r == n
            log.info(f"T1 range {n:3d} cells, K {K:2d}: records {r:3d}, share of range given {r / n:.2f}")
    log.put("T1a", "PASS" if inside else "FAIL", "records lie inside the range")
    log.put("T1b", "PASS" if exact else "FAIL", "records = min(K, range) in every case")
    log.put("T1c", "FLAG" if full else "PASS", f"ranges of K cells or fewer given whole: {full} of {small} cases")

    # T2 (F7)
    ens = meth["_ensure_sparse"]
    me = types.SimpleNamespace(dense_obs_threshold=thr, K_obs_cap=cap)
    m = planted_ranges(torch, (120, thr + 6, 5))[:, 0]
    r = [int(x) for x in ens(me, m).reshape(3, -1).sum(-1)]
    log.info(f"T2 dense frame (120, {thr + 6}, 5 cells): records after the cap {r}")
    log.put("T2a", "PASS" if r == [min(cap, 120), min(cap, thr + 6), min(cap, 5)] else "FAIL",
            f"one species above {thr} -> every species cut to {cap}")
    m = planted_ranges(torch, (thr, 20, 5), seed=2)[:, 0]
    log.put("T2b", "PASS" if torch.equal(ens(me, m), m) else "FAIL", f"a frame with at most {thr} per species is left as it is")
    m = planted_ranges(torch, (25, thr + 1), seed=3)[:, 0]
    r = [int(x) for x in ens(me, m).reshape(2, -1).sum(-1)]
    log.put("T2c", "PASS" if r[0] == min(cap, 25) else "FAIL",
            f"a 25-cell species is cut to {r[0]} because another species has {thr + 1}")
    log.put("T2d", "FLAG", f"a 120-cell species gives the model {min(cap, 120)} of 120 records once the cap fires")

    # T3 (F3)
    bm, n_ind = 1e-4, 3910
    hp = torch.ones(1, 1, 50, G, G)
    ok = True
    for p in (1e-4, 5e-4, 1e-3):
        out = poisson(hp, hp * (n_ind * bm), bm, p, generator=torch.Generator().manual_seed(4))
        got, want = float(out.mean()), 1 - math.exp(-n_ind * p)
        ok &= abs(got - want) < 0.02
        log.info(f"T3 p_obs {p:g}, {n_ind} individuals per occupied cell: recorded share {got:.3f} (formula {want:.3f})")
    log.put("T3a", "PASS" if ok else "FAIL", "recorded share per occupied cell = 1 - exp(-N p) within 0.02")
    hz = torch.zeros(1, 1, 1, G, G)
    bz = torch.zeros(1, 1, 1, G, G); bz[0, 0, 0, 0, 0] = 1.0
    out = poisson(hz, bz, bm, 1e-3, generator=torch.Generator().manual_seed(5))
    log.put("T3b", "FLAG" if float(out[0, 0, 0, 0, 0]) == 1.0 else "PASS",
            "a cell with presence 0 and biomass > 0 can be recorded (output not masked by presence)")

    # T4 (F6)
    dec = meth["_obs_decay"]
    me = types.SimpleNamespace(obs_decay_kernel=kernel(torch, sigma))
    m = torch.zeros(1, 1, G, G); m[0, 0, 5, 5] = 1; m[0, 0, 5, 7] = 1
    v = float(dec(me, m)[0, 0, 5, 6]); k1 = math.exp(-1 / sigma ** 2)
    log.info(f"T4 two records 1 cell away: halo {v:.4f} (max {k1:.4f}, sum {2 * k1:.4f})")
    log.put("T4a", "PASS" if abs(v - k1) < 1e-5 else "FAIL", "halo = nearest record (max), not the sum")
    m = torch.zeros(1, 1, G, G); m[0, 0, 0, 0] = 1
    v = float(dec(me, m)[0, 0, 0, G - 1])
    log.info(f"T4b record at (0,0), halo at (0,{G - 1}): {v:.2e} (torus neighbour would give {k1:.4f}); "
             "no wrap, as in the agreed edge rule C1: INFO")

    # T5 (F9)
    hits = {}
    for f in sorted(glob.glob(os.path.join(stage2, "**", "*.py"), recursive=True)):
        n = sum(1 for line in open(f, errors="replace") if "history_B" in line.split("#")[0])
        if n:
            hits[os.path.relpath(f, stage2)] = n
    log.info("T5 files using history_B (code lines): " + (", ".join(f"{k} {v}" for k, v in hits.items()) or "none"))

    # T6 (F1 and cap reach), one old training file
    if old:
        files = sorted(glob.glob(old))
        if not files:
            log.info(f"T6 no file matches {old}")
            return
        import numpy as np
        z = np.load(files[0], allow_pickle=True)
        log.info(f"T6 file {files[0]}: {len(z.files)} keys")
        key = next((k for k in ("P_last_final", "P_last", "P_t", "P") if k in z.files), None)
        if key is None:
            log.info("T6 no presence key (P_last_final, P_last, P_t, P); keys: " + ", ".join(z.files))
            return
        P = np.asarray(z[key])
        P = P[-1] if P.ndim == 4 else P
        cells = (P > 0.5).reshape(P.shape[0], -1).sum(1)
        share = cells / float(P.shape[-1] * P.shape[-2])
        log.info(f"T6 {key} final frame: {P.shape[0]} species, range sizes min {cells.min()}, median "
                 f"{int(np.median(cells))}, max {cells.max()}; species above {thr} cells: {(cells > thr).sum()}")
        if "prevalence_final" in z.files:
            pf = np.asarray(z["prevalence_final"], dtype=float).reshape(-1)
            if pf.shape[0] == share.shape[0]:
                diff = float(np.abs(pf - share).max())
                log.put("T6a", "PASS" if diff < 1e-6 else "FAIL",
                        f"prevalence_final = occupied share of the final frame (max difference {diff:.2e})")
                if diff < 1e-6:
                    log.put("T6b", "FLAG", "the old species features were built from the true range size of the target")
            else:
                log.info(f"T6 prevalence_final has {pf.shape[0]} values, {key} has {share.shape[0]} species")
        else:
            log.info("T6 no prevalence_final key")
        for bk in ("IBM_B", "B_last", "B_final"):
            if bk in z.files:
                B = np.asarray(z[bk], dtype=float)
                B = B[-1] if B.ndim == 4 else B
                pos = B[B > 0]
                if pos.size:
                    log.info(f"T6 {bk} final frame, raw stored values: median of positive cells {np.median(pos):.4g}")
                break


# ---------------------------------------------------------------- self-test
BROKEN = [  # (file, old text, new text, check that must fail)
    ("training_v2_patch.py", "sparse_flat = sparse_flat * (flat > 0.5).float()", "pass", "T1a"),
    ("models/ecodiffusion_spatial_cond.py", ".amax(dim=1)", ".sum(dim=1)", "T4a"),
    ("models/ecodiffusion_spatial_cond.py", "<= self.dense_obs_threshold:", "<= 10 ** 9:", "T2a"),
]


def self_test(stage2):
    import numpy as np
    res = []

    def t(name, cond, what):
        res.append(bool(cond)); print(f"{name} {what}: {'PASS' if cond else 'FAIL'}")
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good")
        for rel in EXPECT:
            os.makedirs(os.path.dirname(os.path.join(good, rel)), exist_ok=True)
            shutil.copyfile(os.path.join(stage2, rel), os.path.join(good, rel))   # copyfile: no read-only bits
        S = 6
        P = np.zeros((S, G, G), np.int8)
        for s in range(S):
            P[s].reshape(-1)[: 3 + 10 * s] = 1
        share = P.reshape(S, -1).sum(1) / 400.0
        np.savez(os.path.join(tmp, "old_ok.npz"), P_last_final=P, prevalence_final=share)
        np.savez(os.path.join(tmp, "old_bad.npz"), P_last_final=P, prevalence_final=share * 0.5)
        log = Log(quiet=True)
        run_checks(good, log, old=os.path.join(tmp, "old_ok.npz"), check_md5=False)
        lab = dict(log.rows)
        t("S1", all(lab[k] == "PASS" for k in ("T1a", "T1b", "T2a", "T2b", "T2c", "T3a", "T4a", "T6a")),
          "the copy given reproduces every mechanism (T1a, T1b, T2a-c, T3a, T4a, T6a PASS)")
        t("S2", lab["T1c"] == "FLAG" and lab["T6b"] == "FLAG", "full reveal and the prevalence leak are flagged")
        log = Log(quiet=True)
        run_checks(good, log, old=os.path.join(tmp, "old_bad.npz"), check_md5=False)
        t("S3", dict(log.rows)["T6a"] == "FAIL", "a prevalence_final that is not the true share fails T6a")
        for i, (rel, a, b, must) in enumerate(BROKEN):
            bad = os.path.join(tmp, f"bad{i}")
            shutil.copytree(good, bad, copy_function=shutil.copyfile)
            src = open(os.path.join(bad, rel)).read()
            if a not in src:
                t(f"S{4 + i}", False, f"broken version {i}: text '{a}' not found in {rel}")
                continue
            open(os.path.join(bad, rel), "w").write(src.replace(a, b, 1))
            log = Log(quiet=True)
            run_checks(bad, log, check_md5=False)
            t(f"S{4 + i}", dict(log.rows)[must] == "FAIL", f"broken version {i} ({rel}: {a!r} -> {b!r}) fails {must}")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", required=True, help="folder holding training_v2_patch.py and models/")
    ap.add_argument("--old", default=None, help="glob of old training .npz files; the first is read")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.stage2) else 1)
    log = Log()
    run_checks(a.stage2, log, old=a.old)
    labels = [l for _, l in log.rows]
    print(f"{labels.count('PASS')} PASS, {labels.count('FAIL')} FAIL, {labels.count('FLAG')} FLAG")