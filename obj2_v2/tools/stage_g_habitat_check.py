#!/usr/bin/env python3
"""
stage_g_habitat_check.py  (obj2_v2, stage G step G0, read-only, 10 Nov 2026; stage G plan signed off by Sara on 10 Nov)

Question: is the habitat of the 5 years before each target year known exactly? The export kept only the target year's
habitat (ENV_habitat_t, the map of the last piece of a round). Process test 3 of G4 (losses fall where the habitat fell)
needs the habitat of all 6 years. If this check fails, test 3 is dropped and stated as a limit.

How. Assembly v2.6 moves the habitat before every piece j of round t (t >= 1, j = 0 .. M-1):
    U <- rho_s U + sqrt(1 - rho_s^2) Z,  Z = habitat_layer(20, 20, ls, seed=[habitat seed, 104729, t, j]),  U standardised
    m <- rho_ms m + sqrt(1 - rho_ms^2) shift_sd eps,  eps = default_rng([habitat seed, 104743, t, j]).standard_normal()
    H = contrast (U + m),  rho_s = exp(-W / M / tau),  rho_ms = exp(-W / M / shift_tau)
(obj1_v2/new_code/assembly_stepwise_ibm.py, habitat_step; obj1_v2 test_20 U3). Starting from the saved map and shift of
round t-1, the M pieces of round t are rebuilt with Sara's own habitat_layer (imported from obj1_v2/work/environment.py).

Gates (rules written before the run)
    H0  the keys are in the file; HABITAT_rho = exp(-W / tau) and HABITAT_shift_rho = exp(-W / shift_tau) (1e-12)
    H1  rebuilt shift of every piece = ASM_F_shift (1e-9)
    H2  rebuilt last piece of every round = ENV_habitat_t of that round (largest absolute difference 1e-5)
    H3  controls that must fail H2 by far (largest difference above 1e-3, 100 times the H2 rule, on the first 5
        rounds): habitat seed + 1; pieces counted from 1; the round rho in place of rho_s. They show H2 can fail.
Report only: keys, shapes and dtypes of the niche entries (NICHE_*, TRAIT_*, ENV_r_field, ENV_seed_field), for the G4
design; nothing is assumed about them here.
real_eval is refused (read once, in G5). Writes one json summary; nothing is written in the obj1 folders.

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU)
    python3 ../tools/stage_g_habitat_check.py --self-test | tail -1
    python3 ../tools/stage_g_habitat_check.py --obj1-work ../../obj1_v2/work \\
        --files "../../obj1_v2/work/results/stage2a/real_val/*_training.npz" --out ../logs/stage_g0_habitat.json
"""
import argparse, glob, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

G = 20
NEED = ("HABITAT_seed", "HABITAT_ls", "HABITAT_contrast", "HABITAT_tau", "HABITAT_substeps", "window_steps",
        "HABITAT_rho", "HABITAT_shift_sd", "HABITAT_shift_tau", "HABITAT_shift_rho", "ENV_habitat_t",
        "ENV_habitat_shift", "ASM_F_round", "ASM_F_piece", "ASM_F_shift")
NICHE_PREFIX = ("NICHE_", "TRAIT_", "ENV_r_field", "ENV_seed_field")
TOL_SHIFT, TOL_MAP, CTRL_MIN = 1e-9, 1e-5, 1e-3
BROKEN = os.environ.get("STAGE_G0_BROKEN", "")          # self-test only: plants one bug so its gate must fail


def scalar(z, k):
    v = np.asarray(z[k])
    return v.item() if v.ndim == 0 else v.ravel()[0]


def params(z):
    p = dict(seed=int(scalar(z, "HABITAT_seed")), ls=float(scalar(z, "HABITAT_ls")),
             contrast=float(scalar(z, "HABITAT_contrast")), tau=float(scalar(z, "HABITAT_tau")),
             M=int(scalar(z, "HABITAT_substeps")), W=float(scalar(z, "window_steps")),
             shift_sd=float(scalar(z, "HABITAT_shift_sd")), shift_tau=float(scalar(z, "HABITAT_shift_tau")))
    p["rho"] = float(np.exp(-p["W"] / p["tau"]))
    p["rho_s"] = float(np.exp(-p["W"] / p["M"] / p["tau"]))
    p["rho_m"] = float(np.exp(-p["W"] / p["shift_tau"]))
    p["rho_ms"] = float(np.exp(-p["W"] / p["M"] / p["shift_tau"]))
    return p


def rebuild_round(H_prev, m_prev, t, p, layer, seed_off=0, j0=0, rho=None):
    """The M pieces of round t from the saved map and shift of round t-1: maps (M, G, G) and shifts (M,)."""
    rs = p["rho_s"] if rho is None else rho
    U = np.asarray(H_prev, float) / p["contrast"] - float(m_prev)
    m = float(m_prev)
    maps, shifts = [], []
    for j in range(p["M"]):
        jj = j + j0
        Z = np.asarray(layer(G, G, p["ls"], seed=[p["seed"] + seed_off, 104729, int(t), int(jj)]), dtype=float)
        U = rs * U + np.sqrt(1.0 - rs ** 2) * Z
        if BROKEN != "no_standardise":
            U = (U - U.mean()) / U.std()
        eps = float(np.random.default_rng([p["seed"] + seed_off, 104743, int(t), int(jj)]).standard_normal())
        m = p["rho_ms"] * m + np.sqrt(1.0 - p["rho_ms"] ** 2) * p["shift_sd"] * eps
        maps.append(p["contrast"] * (U + m)); shifts.append(m)
    return np.array(maps), np.array(shifts)


def check_file(path, layer, out=print):
    z = np.load(path, allow_pickle=True)
    miss = [k for k in NEED if k not in z.files]
    r = dict(file=Path(path).name, missing=miss)
    if miss:
        out(f"   {Path(path).name}: H0 FAIL, keys missing: {', '.join(miss)}")
        r.update(H0=False, H1=False, H2=False, H3=False)
        return r
    p = params(z)
    h0 = abs(float(scalar(z, "HABITAT_rho")) - p["rho"]) < 1e-12 and abs(float(scalar(z, "HABITAT_shift_rho")) - p["rho_m"]) < 1e-12
    Ht = np.asarray(z["ENV_habitat_t"], dtype=np.float64)
    Hs = np.asarray(z["ENV_habitat_shift"], dtype=np.float64)
    fr, fp, fs = (np.asarray(z[k]) for k in ("ASM_F_round", "ASM_F_piece", "ASM_F_shift"))
    rounds = [int(t) for t in np.unique(fr) if t >= 1 and np.array_equal(np.sort(fp[fr == t]), np.arange(p["M"]))]
    t0 = time.time()
    d_shift, d_map, ctrl = 0.0, 0.0, {"seed+1": 0.0, "pieces from 1": 0.0, "round rho": 0.0}
    for i, t in enumerate(rounds):
        maps, shifts = rebuild_round(Ht[t - 1], Hs[t - 1], t, p, layer)
        k = fr == t
        saved_shift = fs[k][np.argsort(fp[k])].astype(np.float64)
        d_shift = max(d_shift, float(np.abs(shifts - saved_shift).max()))
        d_map = max(d_map, float(np.abs(maps[-1] - Ht[t]).max()))
        if i < 5:
            for name, kw in (("seed+1", dict(seed_off=1)), ("pieces from 1", dict(j0=1)), ("round rho", dict(rho=p["rho"]))):
                mc, _ = rebuild_round(Ht[t - 1], Hs[t - 1], t, p, layer, **kw)
                ctrl[name] = max(ctrl[name], float(np.abs(mc[-1] - Ht[t]).max()))
    sec = time.time() - t0
    h1 = bool(rounds) and d_shift <= TOL_SHIFT
    h2 = bool(rounds) and d_map <= TOL_MAP
    h3 = bool(rounds) and all(v > CTRL_MIN for v in ctrl.values())
    niche = {k: [list(np.asarray(z[k]).shape), str(np.asarray(z[k]).dtype)] for k in z.files if k.startswith(NICHE_PREFIX)}
    r.update(H0=h0, H1=h1, H2=h2, H3=h3, rounds=len(rounds), first_round=rounds[0] if rounds else None,
             last_round=rounds[-1] if rounds else None, max_shift_diff=d_shift, max_map_diff=d_map, controls=ctrl,
             seconds=round(sec, 1), params={k: p[k] for k in ("seed", "ls", "contrast", "tau", "M", "W", "shift_sd",
                                                                "shift_tau", "rho_s", "rho_ms")}, niche_keys=niche)
    out(f"   {Path(path).name[-48:]}: rounds {len(rounds)} ({r['first_round']}-{r['last_round']}); H0 {h0}; "
        f"shift diff {d_shift:.1e}; map diff {d_map:.1e}; controls " + ", ".join(f"{a} {b:.3f}" for a, b in ctrl.items())
        + f"; {sec:.1f} s")
    return r


def run(files, layer, out_json=None, out=print):
    files = sorted(files)
    if not files:
        sys.exit("STOP: no file matches --files")
    if any("real_eval" in f for f in files):
        sys.exit("STOP: real_eval is read once, in stage G5")
    out(f"stage_g_habitat_check: {len(files)} files")
    rows = [check_file(f, layer, out=out) for f in files]
    res = {g: all(r[g] for r in rows) for g in ("H0", "H1", "H2", "H3")}
    for g, text in (("H0", "keys present and rho values as the settings give"),
                    ("H1", f"rebuilt shift of every piece = ASM_F_shift (largest {max(r.get('max_shift_diff', np.inf) for r in rows):.1e}, rule {TOL_SHIFT:g})"),
                    ("H2", f"rebuilt last piece = saved round map (largest {max(r.get('max_map_diff', np.inf) for r in rows):.1e}, rule {TOL_MAP:g})"),
                    ("H3", f"the three controls fail H2 (smallest control difference "
                           f"{min(min(r['controls'].values()) for r in rows if 'controls' in r) if any('controls' in r for r in rows) else float('nan'):.3f}, rule > {CTRL_MIN})")):
        out(f"{g} {text}: {'PASS' if res[g] else 'FAIL'}")
    keys = rows[0].get("niche_keys", {})
    out("   niche entries in the first file (for the G4 design, report only): "
        + ("; ".join(f"{k} {v[0]} {v[1]}" for k, v in sorted(keys.items())) if keys else "none"))
    ok = all(res.values())
    out(f"stage G0 habitat check: {sum(res.values())} PASS, {4 - sum(res.values())} FAIL; "
        + ("the habitat of every piece is rebuilt exactly: process test 3 can run" if ok
           else "the habitat of earlier years is not rebuilt exactly: process test 3 is dropped unless the cause is found"))
    if out_json:
        Path(out_json).write_text(json.dumps(dict(gates=res, files=rows, tool=__file__), indent=1, default=float))
        out(f"   written {out_json}")
    return res, rows


# ---------------------------------------------------------------- self-test (planted files, stand-in habitat_layer)
def toy_layer(ny, nx, length_scale, *, seed=None, path=None):
    """Stand-in for obj1's habitat_layer: periodic smooth Gaussian field, mean 0, sd 1, from the seed only."""
    z = np.random.default_rng(seed).standard_normal((ny, nx))
    k2 = np.fft.fftfreq(ny)[:, None] ** 2 + np.fft.fftfreq(nx)[None, :] ** 2
    f = np.real(np.fft.ifft2(np.fft.fft2(z) * np.exp(-2.0 * (np.pi * length_scale) ** 2 * k2)))
    return (f - f.mean()) / f.std()


def plant_file(path, rounds=12, M=5, W=5000, tau=62500.0, shift_tau=62500.0, seed=105101, bug=None, drop=()):
    """A file built by the assembly rule (round 0 moves pieces 1 .. M-1, later rounds 0 .. M-1); pieces saved for the
    last half of the rounds."""
    contrast, ls, sd = 0.25, 3.0, 1.0
    rho_s, rho_ms = np.exp(-W / M / tau), np.exp(-W / M / shift_tau)
    U = toy_layer(G, G, ls, seed=[seed, 1]); m = 0.0
    Ht, Hs, Fr, Fp, Fs, pieces = [], [], [], [], [], {}
    for t in range(rounds):
        for j in range(M):
            if not (t == 0 and j == 0):
                key = [seed, 104729, t] if bug == "no_piece_in_seed" else [seed, 104729, t, j]
                Z = toy_layer(G, G, ls, seed=key)
                U = rho_s * U + np.sqrt(1 - rho_s ** 2) * Z
                U = (U - U.mean()) / U.std()
                eps = float(np.random.default_rng([seed, 104743, t, j]).standard_normal())
                m = rho_ms * m + np.sqrt(1 - rho_ms ** 2) * sd * eps
            pieces[(t, j)] = contrast * (U + m)
            if t >= rounds // 2:
                Fr.append(t); Fp.append(j); Fs.append(m)
        Ht.append(contrast * (U + m)); Hs.append(m)
    keys = dict(HABITAT_seed=np.int64(seed), HABITAT_ls=np.float64(ls), HABITAT_contrast=np.float64(contrast),
                HABITAT_tau=np.int64(tau), HABITAT_substeps=np.int64(M), window_steps=np.int64(W),
                HABITAT_rho=np.float64(np.exp(-W / tau)), HABITAT_shift_sd=np.float64(sd),
                HABITAT_shift_tau=np.int64(shift_tau), HABITAT_shift_rho=np.float64(np.exp(-W / shift_tau)),
                ENV_habitat_t=np.array(Ht, np.float32), ENV_habitat_shift=np.array(Hs), ASM_F_round=np.array(Fr),
                ASM_F_piece=np.array(Fp), ASM_F_shift=np.array(Fs), NICHE_opt=np.zeros(7), TRAIT_seed=np.int64(205101))
    for k in drop:
        keys.pop(k)
    np.savez(path, **keys)
    return pieces


def self_test():
    quiet = lambda *a, **k: None
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        a = tmp / "real_val" / "a_training.npz"; a.parent.mkdir()
        pieces = plant_file(a)
        g, rows = run([str(a)], toy_layer, out=quiet)
        r = rows[0]
        t("T1", all(g.values()) and r["max_map_diff"] < 1e-6 and r["max_shift_diff"] < 1e-12 and r["rounds"] == 6,
          f"a file built by the rule passes H0-H3 (rounds {r['rounds']}, map diff {r['max_map_diff']:.1e}, "
          f"shift diff {r['max_shift_diff']:.1e}, controls {min(r['controls'].values()):.3f} and up)")
        # T2: every piece of a round (the 5 earlier years too) equals the habitat the planted assembly used
        z = np.load(a)
        p = params(z)
        maps, _ = rebuild_round(z["ENV_habitat_t"][6].astype(float), z["ENV_habitat_shift"][6], 7, p, toy_layer)
        dif = max(float(np.abs(maps[j] - pieces[(7, j)]).max()) for j in range(5))
        t("T2", maps.shape == (5, G, G) and dif < 1e-6 and not np.allclose(maps[0], maps[-1]),
          f"every rebuilt piece of a round equals the habitat used for that piece (largest difference {dif:.1e})")
        b = tmp / "real_val" / "b_training.npz"
        plant_file(b, bug="no_piece_in_seed")
        g, _ = run([str(b)], toy_layer, out=quiet)
        t("T3", not g["H2"], "a file whose habitat did not follow the rule (piece missing from the seed) fails H2")
        c = tmp / "real_val" / "c_training.npz"
        plant_file(c, drop=("ASM_F_shift",))
        g, rows = run([str(c)], toy_layer, out=quiet)
        t("T4", not g["H0"] and rows[0]["missing"] == ["ASM_F_shift"], "a file without ASM_F_shift fails H0 and names the key")
        d = tmp / "real_val" / "d_training.npz"
        plant_file(d, M=25, rounds=6)
        g, rows = run([str(d)], toy_layer, out=quiet)
        t("T5", all(g.values()) and rows[0]["rounds"] == 3, f"25 pieces per round as in the real run: H0-H3 PASS")
        stop = False
        try:
            run([str(tmp / "real_eval" / "x_training.npz")], toy_layer, out=quiet)
        except SystemExit:
            stop = True
        t("T6", stop, "a real_eval path is refused")
        import subprocess
        env = dict(os.environ, STAGE_G0_BROKEN="no_standardise")
        code = ("import importlib.util as u; s=u.spec_from_file_location('h', %r); m=u.module_from_spec(s); "
                "s.loader.exec_module(m); g,_=m.run([%r], m.toy_layer, out=lambda *a, **k: None); print('RES', g['H2'])"
                % (str(Path(__file__).resolve()), str(a)))
        p_ = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        t("bug no_standardise", "RES False" in p_.stdout, f"a rebuild without the standardising step fails H2 "
          f"({p_.stdout.strip()[-20:] or p_.stderr.strip()[-200:]})")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--obj1-work", default="../../obj1_v2/work")
    ap.add_argument("--files", default="../../obj1_v2/work/results/stage2a/real_val/*_training.npz")
    ap.add_argument("--out", default="../logs/stage_g0_habitat.json")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    env_py = Path(a.obj1_work) / "environment.py"
    if not env_py.is_file():
        sys.exit(f"STOP: {env_py} not found; give --obj1-work, the folder that holds obj1's environment.py")
    sys.path.insert(0, str(Path(a.obj1_work).resolve()))
    from environment import habitat_layer
    print(f"habitat_layer from {env_py.resolve()}")
    g, _ = run(glob.glob(a.files), habitat_layer, out_json=a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if all(g.values()) else 1)