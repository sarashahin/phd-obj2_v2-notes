"""
test_04_saved_worlds.py  (obj1_v2, written 24 Oct 2026)

Question: what do the saved world files really contain, and does it make ecological
sense for rare-species maps? Reads every *_training.npz that matches the pattern (one
per distinct final map) and checks history, environment, interactions and records.

History      H1 bookkeeping (last frame = final map, frame times, richness, attempts;
                the last round may be capped)
             H2 invaders dropped in the last window  H3 empty final maps
             H4 maps that never change in the final run  H5 'first occupancy' = first frame
             H6 when final-run losses happen  H7 losses that are only gaps
Environment  E1 saved fields rebuilt (by list position in old files, by species id in new)
             E2-E3 do ranges sit on better cells of their own field than of other fields;
                in old files also of fields held earlier in assembly (positions further on)
             E4 own field at the cells held when assembly ended: lost against kept species
Interactions I1 saved partner weights, competitors per species, partner position
Records      O1 records valid  O2 nested across budgets  O3 cells under 2 tau  O4 individuals
             at cells that switch state
PASS/FAIL = the file matches the code that wrote it. FLAG = it does, but it works
against the aim. Ecological reading of each line: obj1_v2/tests/README.md.
Run from obj1_v2/work:
    USE_GPU=0 python3 ../tests/test_04_saved_worlds.py "../../results/data/data_eval_unseen/*_training.npz"
About 3 minutes for the 25 evaluation communities.
Results 24 Oct (25 evaluation communities, old code): H2 166 dropped, H3 9.6% empty,
H4 57.0% static, H5 83.7%, H6 quartiles 0.18/0.44/0.72, H7 0%, E3 own +0.049 +- 0.007,
earlier positions +0.0025 +- 0.0021, E4 lost -0.288 +- 0.022 against kept +0.031 +- 0.007,
O2 17%, O3 0.0% (median 3,911 individuals per occupied cell), O4 median 1,366.
"""
import os
import sys, glob, hashlib, warnings
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import environment
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "../../results/data/data_eval_unseen/*_training.npz"))
if not files: sys.exit("no files matched")
res = lambda c: "PASS" if c else "FAIL"
flag = lambda c: "FLAG" if c else "ok"
FRAC, KMAX, KNEG = 0.05, 200, 20                 # invasion fraction of the sweeps; field offsets tested
seen, out = set(), {k: [] for k in ("chk", "empty", "static", "tfirst", "lost_early", "lost_t", "disc", "e_own", "e_past1",
                                     "e_past2", "e_ctrl", "lost_f", "kept_f", "regen", "w_ok", "w0", "deg", "pos",
                                     "mask_ok", "nest", "lowN", "Nocc", "Nflip", "back")}
for f in files:
    z = np.load(f, allow_pickle=True); k = set(z.files)
    P = z["P_last_final"].astype(bool)
    h = hashlib.md5(P.tobytes()).hexdigest()
    if h in seen: continue
    seen.add(h)
    S = P.shape[0]; R = P.reshape(S, -1).sum(1); thr = float(z["detection_threshold"]); BODY = float(z["BODY_MASS"])
    Pt = z["P_t"].astype(bool); t = z["IBM_t"]
    # ---------------- history ----------------
    ok = [np.array_equal(Pt[-1], P), np.all(np.diff(t) == t[0]), np.array_equal(z["t_lastK"], t[-4:]),
          int(z["gamma"]) == S == int(z["ASM_gamma"][-1])]
    g, a = z["ASM_gamma"].astype(int), z["ASM_attempts"].astype(int)
    step, want = np.diff(np.r_[0, a]), 1 + (FRAC * np.r_[1, g[:-1]]).astype(int)
    ok.append(np.array_equal(step[:-1], want[:-1]) and 1 <= step[-1] <= want[-1])     # last round may be capped
    out["chk"].append(ok)
    out["disc"].append(int(z["attempts_total"]) - int(a[-1]))
    out["empty"].append((R == 0).mean())
    any_t = Pt.reshape(len(t), S, -1).any(2).any(0)
    same = (Pt == Pt[-1][None]).all(axis=(0, 2, 3))
    out["static"].append(same[any_t].mean())
    tf = z["T_first_occ"][P]
    out["tfirst"].append((tf == t[0]).mean())
    ext = z["IBM_ext_step"]; gone = ext >= 0; lost = gone & (R == 0)       # lost = absent at the end, not just a gap
    out["back"].append((R[gone] > 0).mean() if gone.any() else np.nan)
    out["lost_t"].extend((ext[lost] / t[-1]).tolist()); out["lost_early"].append((ext[lost] <= 0.1 * t[-1]).mean() if lost.any() else np.nan)
    # ---------------- environment ----------------
    if "ENV_r_field" in k:
        E = z["ENV_r_field"].astype(np.float64)
        by_id = "species_id" in k                       # new files: one field per species, keyed by its id
        if by_id:
            import assembly_stepwise_ibm as asm
            F = np.stack([asm.species_field(int(i), length_scale=float(z["ENV_length_scale"]), var_r=float(z["ENV_var_r"]),
                                            seed_field=int(z["ENV_seed_field"])) for i in z["species_id"]])
            F = np.concatenate([F, F[np.random.default_rng(0).integers(0, S, KMAX)]])      # KMAX filler rows keep shapes; not used below
        else:
            F = environment.generate_spatial_r(S + KMAX, P.shape[1], P.shape[2], float(z["ENV_length_scale"]),
                                               np.ones(S + KMAX), float(z["ENV_var_r"]), seed=int(z["ENV_seed_field"]))
        out["regen"].append(np.abs(F[:S] - E).max())
        Fz = F.reshape(S + KMAX, -1); Fz = (Fz - Fz.mean(1, keepdims=True)) / Fz.std(1, keepdims=True)
        Pf = P.reshape(S, -1).astype(np.float64)
        sel = np.flatnonzero((np.arange(S) >= KNEG) & (R > 0) & (R < Pf.shape[1]))
        d = lambda kk: (Pf[sel] * Fz[sel + kk]).sum(1) / R[sel]
        out["e_own"].extend(d(0).tolist())
        if by_id:                                       # control: fields of 20 other species of the same community
            prm = [np.random.default_rng(q).permutation(S) for q in range(20)]
            ctl = np.mean([(Pf[sel] * Fz[pm[sel]]).sum(1) / R[sel] for pm in prm], 0)
            out["e_ctrl"].extend(ctl.tolist()); out["e_past1"].extend([np.nan] * len(sel)); out["e_past2"].extend([np.nan] * len(sel))
        else:
            out["e_past1"].extend(np.mean([d(kk) for kk in range(1, 21)], 0).tolist())
            out["e_past2"].extend(np.mean([d(kk) for kk in range(21, KMAX)], 0).tolist())
            out["e_ctrl"].extend(np.mean([d(-kk) for kk in range(1, KNEG + 1)], 0).tolist())
        Pi = z["P_init_assembly"].astype(bool).reshape(S, -1); Ri = Pi.sum(1)
        own_init = np.where(Ri > 0, (Pi * Fz[:S]).sum(1) / np.maximum(Ri, 1), np.nan)
        out["lost_f"].extend(own_init[lost & (Ri > 0)].tolist()); out["kept_f"].extend(own_init[~lost & (Ri > 0)].tolist())
    # ---------------- interactions ----------------
    w = z["C_topk_w"]; idx = z["C_topk_idx"]
    uw = np.unique(w.astype(np.float64)); out["w_ok"].append(bool(np.all(np.isclose(uw, 0.0) | np.isclose(uw, 0.4))))
    out["w0"].append((w == 0).mean())
    out["deg"].append(np.median(z["deg_in"]) / (S - 1)); out["pos"].append((idx / S).mean())
    # ---------------- observations ----------------
    mk = sorted(int(x.split("_")[-1]) for x in k if x.startswith("obs_mask_"))
    mok = True
    for b in mk:
        M = z[f"obs_mask_{b}"].astype(bool)
        mok &= bool((M <= P).all()) and np.array_equal(M.reshape(S, -1).sum(1), np.minimum(b, R))
    out["mask_ok"].append(mok)
    if 5 in mk and 10 in mk:
        m5, m10 = z["obs_mask_5"].astype(bool), z["obs_mask_10"].astype(bool)
        big = R > 10
        out["nest"].append(np.mean([(m5[s] <= m10[s]).all() for s in np.flatnonzero(big)]) if big.any() else np.nan)
    N = np.rint(z["B_last"].astype(np.float64) / BODY)
    out["lowN"].append((N[P] < 2 * thr).mean()); out["Nocc"].append(np.median(N[P]))
    if "IBM_B" in k:
        flipN = []
        IB = z["IBM_B"]
        for q in range(len(t) - 1):
            fl = Pt[q] != Pt[q + 1]
            if fl.any():
                n0 = np.rint(IB[q][fl].astype(np.float64) / BODY); n1 = np.rint(IB[q + 1][fl].astype(np.float64) / BODY)
                flipN.extend(np.maximum(n0, n1).tolist())
        out["Nflip"].append(np.median(flipN) if flipN else np.nan)
    print(f"{f.split('/')[-1][:58]}: S {S}, tau {thr:g}, frames {len(t)}, lost in final run {lost.mean():.1%}, "
          f"empty final maps {(R == 0).mean():.1%}")

n = len(out["chk"]); chk = np.array(out["chk"])
mse = lambda v: (np.nanmean(v), 2 * np.nanstd(v) / np.sqrt(np.sum(~np.isnan(np.asarray(v, float)))))
print(f"\n{n} distinct communities from {len(files)} files")
print(f"H1 last frame = final map {res(chk[:, 0].all())}; even frame times {res(chk[:, 1].all())}; last-4 times {res(chk[:, 2].all())}; "
      f"richness = species rows = last assembly richness {res(chk[:, 3].all())}; attempts follow 1 + floor(0.05 richness) {res(chk[:, 4].all())}")
print(f"H2 invaders simulated in the last window and dropped: median {np.median(out['disc']):.0f} {flag(np.median(out['disc']) > 0)}")
print(f"H3 species rows with an empty final map: median {np.median(out['empty']):.1%} of the community {flag(np.median(out['empty']) > 0)}")
print(f"H4 species whose map never changes over the final run: median {np.median(out['static']):.1%} {flag(np.median(out['static']) > 0.5)}")
print(f"H5 occupied cells whose 'first occupancy time' is the first frame: median {np.median(out['tfirst']):.1%} "
      f"(the run starts from the assembled state, so this is not colonisation time) {flag(np.median(out['tfirst']) > 0.5)}")
lt = np.array(out["lost_t"])
if lt.size:
    print(f"H6 when final-run losses happen (share of the run): quartiles {np.quantile(lt, [0.25, 0.5, 0.75]).round(2).tolist()}; "
          f"in the first 10% {np.mean(lt <= 0.1):.0%} {flag(np.mean(lt <= 0.1) > 0.2)}")
print(f"H7 species counted as lost in the final run (IBM_ext_step >= 0) that are present again in the last frame: "
      f"median {np.nanmedian(out['back']):.1%} {flag(np.nanmedian(out['back']) > 0)} (these are gaps, not extinctions)")
if out["regen"]:
    print(f"E1 saved fields rebuilt from seed, l and variance: max difference {max(out['regen']):.1e} {res(max(out['regen']) < 1e-5)}")
    keyed = np.all(np.isnan(np.array(out["e_past1"], float)))
    rows = ([("own field", "e_own"), ("fields of 20 other species of the community (control)", "e_ctrl")] if keyed else
            [("own field (k = 0)", "e_own"), ("fields of positions 1-20 later (held earlier in assembly)", "e_past1"),
             ("positions 21-199 later", "e_past2"), ("positions 1-20 earlier (never held, control)", "e_ctrl")])
    for lab, key in rows:
        m_, s_ = mse(out[key]); print(f"E2 occupied cells against {lab}: mean z {m_:+.4f} +- {s_:.4f}")
    own, past, ctrl = (np.array(out[x]) for x in ("e_own", "e_past1", "e_ctrl"))      # same species, same order
    d1, s1 = mse(own - ctrl); d2, s2 = mse(past - ctrl)
    past_txt = ("not applicable (fields keyed by species id)" if np.all(np.isnan(past)) else
                f"{d2:+.4f} +- {s2:.4f}: ranges carry an imprint of fields held earlier in assembly {'yes' if d2 > s2 else 'no'} {flag(d2 > s2)}")
    print(f"E3 own field minus control {d1:+.4f} +- {s1:.4f}: ranges carry an imprint of their saved field "
          f"{'yes' if d1 > s1 else 'no'} {flag(d1 <= s1)}; earlier-position fields minus control {past_txt}")
    if len(out["lost_f"]) > 1:
        ml, sl = mse(out["lost_f"]); mk_, sk = mse(out["kept_f"])
        print(f"E4 own final field at the cells held when assembly ended: lost species {ml:+.4f} +- {sl:.4f}, kept {mk_:+.4f} +- {sk:.4f} "
              f"{flag(mk_ - ml > np.hypot(sl, sk))} (FLAG = lost species held cells that are poor under the field they got in the final run)")
    else:
        print("E4 not tested: no species lost in the final run")
print(f"I1 saved partner weights all 0 or 0.4 {res(all(out['w_ok']))}; zero-weight slots median {np.median(out['w0']):.1%}; "
      f"competitors per species = {np.median(out['deg']):.3f} of the community; mean partner position / S {np.median(out['pos']):.2f}")
print(f"O1 records inside the true range and = min(budget, range) {res(all(out['mask_ok']))}")
if out["nest"]:
    print(f"O2 5-record set inside the 10-record set: median {np.nanmedian(out['nest']):.0%} of species {flag(np.nanmedian(out['nest']) < 1)}")
print(f"O3 occupied cells with fewer than 2 x tau individuals: median {np.median(out['lowN']):.1%}; median individuals per occupied cell "
      f"{np.median(out['Nocc']):.0f}")
if out["Nflip"]:
    print(f"O4 cells that switch state between frames hold a median of {np.nanmedian(out['Nflip']):.0f} individuals on the occupied side "
          f"(threshold noise if close to tau)")
