#!/usr/bin/env python3
"""
species_compare.py  (obj1_v2, 30 Oct 2026, Sara's yes; version 2 the same day; version 3: w_min in K1, K2 at w_min 0.2, K6;
version 4: the last line names what the reading rule is applied to)

Question: best founding (LVMCM test, n33, n37-n40) keeps the sparse species BNM has, but holds
6-10 times BNM's species per window. Which kind of species are the extra ones, and so which
single dial should change next? Random-founding worlds (n29-n32) are the comparison.
Reads saved worlds only (nothing is simulated, no BNM file is opened); writes nothing.

Per kept species (final community; true maps, no survey; presence = P_last_final, 1 individual):
  w        niche width (NICHE_w)
  |opt|    distance of the niche optimum from 0 (NICHE_opt), as in README sketch 22
  F        cells where the species can grow alone: ENV_r_field > 0
  d        its dispersal rate (SP_disp_rate)
  range    occupied cells
  N/cell   median individuals per occupied cell: B_last / BODY_MASS
  sink     share of its occupied cells where it cannot grow alone (r <= 0)

Parts:
  K0  check that the saved growth field is the niche formula: r = peak exp(-(H - opt)^2 / 2w^2) - cost,
      H = ENV_habitat, which run_all_rps saves as the layer the species see (contrast x standardised
      layer; patch_run_all_rps_contrast.py, point 4), so the contrast is not applied again. If it holds,
      F can also be computed for every species that ever entered (life table LT_*), so the entry
      filter can be read (K4). Version 2 (30 Oct): version 1 applied the contrast a second time and
      failed K0 on every real world (max difference 1.3-1.4); K1-K3 and K5 never used the formula.
  K1  one line per world: species, and medians of every trait.
  K2  best founding at strength 0.4 against random founding: AUC per trait = the chance that a
      best-founding species has the larger value (0.5 = no difference), from all kept species; then the
      same for one seed at a time where both rules share it (same habitat map and niche draws).
      n33 and n40 share their first 1,000 newcomers, so their species are not independent: K2 describes,
      it does not test.
  K3  best founding: species with true ranges of 1-5 cells against the rest (what the sparse ones are).
  K4  entry filter from the life table: F of entered species and of kept species, both founding rules.
  K5  strength series (best founding): medians of every trait at each strength.
  K6  minimum niche width series (best founding, strength 0.4), when worlds with another NICHE_w_min are given
      (plan niche_w). K2 keeps comparing best founding at w_min 0.2 with random founding, as before.

Reading rule (written before the run, 30 Oct). Main trait = the K2 trait with the largest |AUC - 0.5|
among w, F, d, N/cell; it counts only if |AUC - 0.5| >= 0.2.
  w lower in best founding        -> narrow niches pack the habitat: next test a higher minimum niche width
  F lower, w not counting          -> specialists of rare habitat: next test habitat contrast or optimum spread
  d lower                          -> slow dispersers held in place: next test the lower end of the d range
  N/cell higher                    -> populations too large for drift to remove: next test fewer
                                      individuals per cell (body mass)
  no trait counts                  -> the extra species are not one kind; the founding rule itself
                                      (how many newcomers pass the test) is the next dial
Every next test is one change, on the GPU, with the BNM bands, after Sara's yes.
Version 4 (30 Oct): the rule reads K2 only, so with niche_w worlds given it still prints the 30 Oct answer
about n33 against random founding; the last line now says so. The niche_w result is K6, read by hand.

Run from obj1_v2/work:
  python3 ../tools/species_compare.py --self-test | tee ../logs/species_compare_selftest_30oct.txt
  python3 ../tools/species_compare.py \\
      --world "results/stage2a/niche_d/*_training.npz" \\
      --world "results/stage2a/niche_b/*_training.npz" \\
      --world "results/stage2a/niche_s/*_training.npz" | tee ../logs/species_compare_30oct.txt
"""
import argparse, glob, sys, tempfile
from pathlib import Path
import numpy as np

TRAITS = ["w", "|opt|", "F", "d", "range", "N/cell", "sink"]
RULE_TRAITS = ["w", "F", "d", "N/cell"]
NEED = ["P_last_final", "B_last", "BODY_MASS", "ENV_r_field", "NICHE_w", "NICHE_opt", "SP_disp_rate"]


def scalar(z, k, default=None):
    if k not in z.files:
        return default
    v = z[k]
    return v.item() if v.ndim == 0 else (str(v[0]) if v.dtype.kind in "UO" else v[0].item())


def niche_r(H, c, opt, w, peak, cost):
    """Growth field of species with niche (opt, w, peak) on habitat H seen at contrast c."""
    return peak[:, None, None] * np.exp(-(c * H[None] - opt[:, None, None]) ** 2 / (2 * w[:, None, None] ** 2)) - cost


def load_world(f):
    z = np.load(f, allow_pickle=True)
    miss = [k for k in NEED if k not in z.files]
    if miss:
        return None, f"skipped {Path(f).parent.name}/{Path(f).name}: missing {miss}"
    fb = scalar(z, "FOUND_rule")
    rule = fb if fb is not None else ("best" if "_fb" in Path(f).name else "random")
    P = z["P_last_final"].astype(bool)
    r = z["ENV_r_field"].astype(np.float64)
    N = np.rint(z["B_last"].astype(np.float64) / float(z["BODY_MASS"]))
    rng_ = P.reshape(len(P), -1).sum(1)
    ncell = np.array([np.median(N[s][P[s]]) if rng_[s] else 0.0 for s in range(len(P))])
    sink = np.array([np.mean(r[s][P[s]] <= 0) if rng_[s] else np.nan for s in range(len(P))])
    t = {"w": z["NICHE_w"].astype(float), "|opt|": np.abs(z["NICHE_opt"].astype(float)),
         "F": (r > 0).reshape(len(r), -1).sum(1).astype(float), "d": z["SP_disp_rate"].astype(float),
         "range": rng_.astype(float), "N/cell": ncell, "sink": sink}
    w = dict(file=f, label=f"{Path(f).parent.name}/{Path(f).name[-40:]}", rule=rule,
             strength=float(scalar(z, "INTERACTION_STRENGTH", 0.4)), seed=scalar(z, "SIM_SEED"),
             w_min=float(scalar(z, "NICHE_w_min", 0.2)),
             attempts=scalar(z, "attempts_total"), S=len(P), t=t, z=z)
    return w, None


def formula_check(w):
    """K0: max |saved field - niche formula| over kept species; None if a key is missing."""
    z = w["z"]
    need = ["ENV_habitat", "NICHE_opt", "NICHE_w", "NICHE_peak", "NICHE_cost"]
    if any(k not in z.files for k in need):
        return None
    rr = niche_r(z["ENV_habitat"].astype(float), 1.0, z["NICHE_opt"].astype(float), z["NICHE_w"].astype(float),
                 z["NICHE_peak"].astype(float), float(z["NICHE_cost"]))
    return float(np.abs(rr - z["ENV_r_field"].astype(float)).max())


def lt_F(w):
    """F for every species that entered (life table), from the niche formula; also who was kept."""
    z = w["z"]
    need = ["LT_opt", "LT_w", "LT_peak", "LT_id", "ENV_habitat", "NICHE_cost", "species_id"]
    if any(k not in z.files for k in need):
        return None
    rr = niche_r(z["ENV_habitat"].astype(float), 1.0, z["LT_opt"].astype(float), z["LT_w"].astype(float),
                 z["LT_peak"].astype(float), float(z["NICHE_cost"]))
    F = (rr > 0).reshape(len(rr), -1).sum(1)
    kept = np.isin(z["LT_id"], z["species_id"])
    return F, kept


def auc(a, b):
    """P(a > b) + 0.5 P(a = b) over all pairs (Mann-Whitney), NaNs dropped."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                                   # mid-ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.median(x) if len(x) else np.nan


def fmt_traits(t, idx=None):
    g = lambda k: t[k] if idx is None else t[k][idx]
    return (f"w {med(g('w')):.2f}, |opt| {med(g('|opt|')):.2f}, F {med(g('F')):.0f}, d {med(g('d')):.1e}, "
            f"range {med(g('range')):.0f}, N/cell {med(g('N/cell')):.0f}, sink {med(g('sink')):.2f}")


def pool(ws):
    return {k: np.concatenate([w["t"][k] for w in ws]) for k in TRAITS}


def decide(A):
    """Reading rule on the K2 AUCs (dict trait -> AUC). Returns (main trait or None, text)."""
    sep = {k: abs(A[k] - 0.5) for k in RULE_TRAITS if np.isfinite(A.get(k, np.nan))}
    if not sep:
        return None, "not read: no AUC"
    main = max(sep, key=sep.get)
    if sep[main] < 0.2:
        return None, "no trait counts -> the founding rule itself is the next dial"
    up = A[main] > 0.5
    if main == "w" and not up:
        return "w", "narrow niches pack the habitat -> next: a higher minimum niche width"
    if main == "F" and not up and sep.get("w", 0) < 0.2:
        return "F", "specialists of rare habitat -> next: habitat contrast or optimum spread"
    if main == "d" and not up:
        return "d", "slow dispersers held in place -> next: the lower end of the d range"
    if main == "N/cell" and up:
        return "N/cell", "populations too large for drift -> next: fewer individuals per cell (body mass)"
    return main, f"main trait {main} goes the other way ({'higher' if up else 'lower'} in best founding); not in the rule -> discuss"


def run(patterns):
    worlds, seen = [], set()
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"no file matched {pat}")
        for f in files:
            if f in seen:
                continue
            seen.add(f)
            w, msg = load_world(f)
            print(msg) if msg else worlds.append(w)
    if not worlds:
        sys.exit("STOP: no world read")

    print("\nK0 saved growth field = niche formula (max difference over kept species)")
    ok_formula = True
    for w in worlds:
        e = formula_check(w)
        ok_formula &= (e is not None and e < 1e-3)
        print(f"  {w['label']}: {'keys missing' if e is None else f'{e:.2e}'}")
    print(f"  -> {'PASS: F of entered species is computed from the life table' if ok_formula else 'FAIL: K4 not read'}")

    print("\nK1 kept species per world (medians; true maps, presence at 1 individual)")
    for w in worlds:
        print(f"  {w['rule']:6s} s {w['strength']:.1f} w_min {w['w_min']:.1f} seed {w['seed']} attempts {w['attempts']}  {w['label']}: "
              f"{w['S']} species; {fmt_traits(w['t'])}")

    rnd = [w for w in worlds if w["rule"] == "random"]
    best4 = [w for w in worlds if w["rule"] == "best" and abs(w["strength"] - 0.4) < 1e-6 and abs(w["w_min"] - 0.2) < 1e-6]
    A = {}
    if rnd and best4:
        a, b = pool(best4), pool(rnd)
        A = {k: auc(a[k], b[k]) for k in TRAITS}
        print(f"\nK2 best founding at strength 0.4 ({len(best4)} worlds, {len(a['w'])} species) against random founding "
              f"({len(rnd)} worlds, {len(b['w'])} species): AUC = chance a best-founding species has the larger value")
        print("  " + ", ".join(f"{k} {A[k]:.2f}" for k in TRAITS))
        print(f"  medians, best:   {fmt_traits(a)}")
        print(f"  medians, random: {fmt_traits(b)}")
        seeds = sorted({w["seed"] for w in best4} & {w["seed"] for w in rnd})
        for sd in seeds:                         # same habitat map and niche draws: founding is the only difference
            a1 = pool([w for w in best4 if w["seed"] == sd]); b1 = pool([w for w in rnd if w["seed"] == sd])
            print(f"  same seed {sd} only (same habitat map): " + ", ".join(f"{k} {auc(a1[k], b1[k]):.2f}" for k in TRAITS))
    else:
        print("\nK2 not read: needs random-founding worlds and best-founding worlds at strength 0.4")

    best = [w for w in worlds if w["rule"] == "best"]
    if best:
        p = pool(best); small = (p["range"] >= 1) & (p["range"] <= 5)
        print(f"\nK3 best founding ({len(best)} worlds): {small.sum()} species with true ranges of 1-5 cells, "
              f"{(~small).sum()} with more")
        print(f"  1-5 cells: {fmt_traits(p, small)}")
        print(f"  6+ cells:  {fmt_traits(p, ~small)}")

    if ok_formula:
        print("\nK4 entry filter (life table): F of species that entered and of species kept")
        for w in worlds:
            x = lt_F(w)
            if x is None:
                print(f"  {w['label']}: life-table keys missing"); continue
            F, kept = x
            print(f"  {w['rule']:6s} s {w['strength']:.1f} {w['label']}: entered {len(F)}, kept {kept.sum()}; "
                  f"F median entered {med(F):.0f}, kept {med(F[kept]):.0f}; share with F <= 100: "
                  f"entered {np.mean(F <= 100):.0%}, kept {np.mean(F[kept] <= 100) if kept.any() else np.nan:.0%}")

    ser = sorted({w["strength"] for w in best})
    if len(ser) > 1:
        print("\nK5 strength series, best founding (medians over the worlds at each strength)")
        for s in ser:
            ws = [w for w in best if abs(w["strength"] - s) < 1e-6]
            print(f"  strength {s:.1f} ({len(ws)} worlds, {sum(w['S'] for w in ws)} species): {fmt_traits(pool(ws))}")

    bw = [w for w in worlds if w["rule"] == "best" and abs(w["strength"] - 0.4) < 1e-6]
    wmins = sorted({round(w["w_min"], 6) for w in bw})
    if len(wmins) > 1:
        print("\nK6 minimum niche width series, best founding at strength 0.4 (medians over the worlds at each w_min)")
        for v in wmins:
            ws = [w for w in bw if abs(w["w_min"] - v) < 1e-6]
            print(f"  w_min {v:.1f} ({len(ws)} worlds, species per world {', '.join(str(w['S']) for w in ws)}): {fmt_traits(pool(ws))}")

    main, text = decide(A) if A else (None, "not read: K2 missing")
    print(f"\nReading rule, K2 only: best founding at strength 0.4 and w_min 0.2 against random founding; other strengths "
          f"and widths do not enter it (main trait = largest |AUC - 0.5| among {', '.join(RULE_TRAITS)}; counts at 0.2 or more):")
    if A:
        print("  AUC - 0.5: " + ", ".join(f"{k} {A[k] - 0.5:+.2f}" for k in RULE_TRAITS))
    print(f"  -> {text}")


# ---------------------------------------------------------------- self-test
def self_test():
    rng = np.random.default_rng(0); G = 20; ok = True

    def check(name, cond, text=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name} {text} {'PASS' if cond else 'FAIL'}")

    check("T1a", abs(auc([1, 2, 3], [1, 2, 3]) - 0.5) < 1e-12 and auc([4, 5], [1, 2]) == 1.0 and auc([1, 2], [4, 5]) == 0.0,
          "AUC: same values 0.5, all larger 1, all smaller 0")
    a, b = rng.normal(0.5, 1, 400), rng.normal(0, 1, 500)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    check("T1b", abs(auc(a, b) - brute) < 1e-12, f"AUC equals a direct count over pairs ({brute:.4f})")
    check("T1c", abs(auc([1, 1, 2], [1, 2, 2]) - np.mean([(x > y) + 0.5 * (x == y) for x in [1, 1, 2] for y in [1, 2, 2]])) < 1e-12,
          "AUC with ties uses half credit")

    H = rng.normal(0, 1, (G, G))

    def world(path, rule, strength, S, w_rng, d_rng, ncell, seed, mutate=False, contrast=0.25, keep_found=True, w_min=None):
        opt = rng.normal(0, 1.5, S); w = rng.uniform(*w_rng, S); peak = 1.5 * (0.2 / w) ** 0.5; cost = 0.1
        r = niche_r(H, contrast, opt, w, peak, cost)
        if mutate:
            r = r + 0.05
        P = (r > 0) & (rng.random((S, G, G)) < 0.6)
        for s in range(S):
            if not P[s].any():
                P[s].flat[int(np.argmax(r[s]))] = True
        B = (P * ncell * 1e-4).astype(np.float16)
        lt_n = S + 30
        lt_opt = np.r_[opt, rng.normal(0, 1.5, 30)]; lt_w = np.r_[w, rng.uniform(*w_rng, 30)]
        extra = dict(FOUND_rule=np.array([rule])) if keep_found else {}
        if w_min is not None:
            extra["NICHE_w_min"] = np.float32(w_min)
        np.savez(path, P_last_final=P.astype(np.uint8), B_last=B, BODY_MASS=np.float32(1e-4),
                 ENV_r_field=r.astype(np.float32), ENV_habitat=(contrast * H).astype(np.float32), HABITAT_contrast=np.float32(contrast),
                 NICHE_w=w.astype(np.float32), NICHE_opt=opt.astype(np.float32), NICHE_peak=peak.astype(np.float32),
                 NICHE_cost=np.float32(cost), SP_disp_rate=np.exp(rng.uniform(np.log(d_rng[0]), np.log(d_rng[1]), S)),
                 INTERACTION_STRENGTH=np.float32(strength), SIM_SEED=np.int64(seed), attempts_total=np.int64(1000),
                 species_id=np.arange(S), LT_id=np.arange(lt_n), LT_opt=lt_opt.astype(np.float32),
                 LT_w=lt_w.astype(np.float32), LT_peak=(1.5 * (0.2 / lt_w) ** 0.5).astype(np.float32), **extra)

    with tempfile.TemporaryDirectory() as tmp:
        for k in range(4):
            world(f"{tmp}/rnd{k}_f1_s{2303 + k}_training.npz", "random", 0.4, 30, (0.6, 1.0), (1e-6, 1e-2), 3000, 2303 + k)
        Path(f"{tmp}/b").mkdir()
        world(f"{tmp}/b/x_fb_s2303_training.npz", "best", 0.4, 200, (0.2, 0.35), (1e-6, 1e-2), 3000, 2303)
        world(f"{tmp}/b/z_fb_w0p4_s2303_training.npz", "best", 0.4, 60, (0.4, 1.0), (1e-6, 1e-2), 3000, 2303, w_min=0.4)
        world(f"{tmp}/b/y_fb_is0p8_s2303_training.npz", "best", 0.8, 150, (0.2, 0.35), (1e-6, 1e-2), 3000, 2303,
              keep_found=False)
        ws = [load_world(f)[0] for f in sorted(glob.glob(f"{tmp}/**/*_training.npz", recursive=True))]
        check("T2a", all(w is not None for w in ws) and sorted(w["rule"] for w in ws).count("best") == 3,
              "founding rule from FOUND_rule, else from '_fb' in the name")
        w0 = ws[0]; z = w0["z"]
        N = np.rint(z["B_last"].astype(float) / 1e-4); P = z["P_last_final"].astype(bool)
        good = np.allclose(w0["t"]["N/cell"], [np.median(N[s][P[s]]) for s in range(len(P))]) and \
            np.allclose(w0["t"]["F"], (z["ENV_r_field"] > 0).reshape(len(P), -1).sum(1))
        check("T2b", good and np.isclose(med(w0["t"]["N/cell"]), 3000, rtol=2e-3),
              f"individuals per cell from B_last / BODY_MASS (float16 kept within 0.2%), F from the field")
        check("T3a", all(formula_check(w) < 1e-3 for w in ws), "saved field equals the niche formula on ENV_habitat as saved (contrast x H)")
        z0 = ws[0]["z"]
        twice = niche_r(z0["ENV_habitat"].astype(float), float(z0["HABITAT_contrast"]), z0["NICHE_opt"].astype(float),
                        z0["NICHE_w"].astype(float), z0["NICHE_peak"].astype(float), float(z0["NICHE_cost"]))
        e2 = float(np.abs(twice - z0["ENV_r_field"].astype(float)).max())
        check("T3d", e2 > 0.1, f"negative control: applying the contrast a second time (version 1's error) is caught (max difference {e2:.2f})")
        world(f"{tmp}/mut_f1_s9_training.npz", "random", 0.4, 10, (0.6, 1.0), (1e-6, 1e-2), 3000, 9, mutate=True)
        check("T3b", formula_check(load_world(f"{tmp}/mut_f1_s9_training.npz")[0]) > 0.04,
              "negative control: a field off the formula is caught")
        Fk = lt_F(ws[0])
        check("T3c", Fk is not None and Fk[1].sum() == ws[0]["S"] and len(Fk[0]) == ws[0]["S"] + 30
              and np.array_equal(Fk[0][Fk[1]], ws[0]["t"]["F"].astype(int)),
              "life-table F of kept species equals F from the saved field")
        check("T2c", sorted(round(w["w_min"], 6) for w in ws if w["rule"] == "best") == [0.2, 0.2, 0.4],
              "w_min read from NICHE_w_min, 0.2 when absent")
        A = {k: auc(pool([w for w in ws if w["rule"] == "best" and abs(w["strength"] - 0.4) < 1e-6 and abs(w["w_min"] - 0.2) < 1e-6])[k],
                    pool([w for w in ws if w["rule"] == "random"])[k]) for k in TRAITS}
        main, text = decide(A)
        check("T4a", main == "w", f"planted narrow niches in best founding are found: w AUC {A['w']:.2f} -> {text}")
        check("T4b", decide({"w": 0.55, "F": 0.2, "d": 0.5, "N/cell": 0.5})[0] == "F"
              and decide({"w": 0.5, "F": 0.5, "d": 0.25, "N/cell": 0.5})[0] == "d"
              and decide({"w": 0.5, "F": 0.5, "d": 0.5, "N/cell": 0.85})[0] == "N/cell"
              and decide({"w": 0.6, "F": 0.45, "d": 0.52, "N/cell": 0.4})[0] is None
              and "other way" in decide({"w": 0.9, "F": 0.5, "d": 0.5, "N/cell": 0.5})[1],
              "every branch of the reading rule, and a trait going the other way")
        check("T4c", decide({"w": 0.25, "F": 0.2, "d": 0.5, "N/cell": 0.5})[0] == "F"
              and "rare habitat" not in decide({"w": 0.27, "F": 0.25, "d": 0.5, "N/cell": 0.5})[1],
              "F counts as rare habitat only when w does not count")
        out = Path(f"{tmp}/bad_training.npz"); np.savez(out, P_last_final=np.zeros((2, G, G), np.uint8))
        w, msg = load_world(out)
        check("T5", w is None and "missing" in msg, "a file without the needed keys is skipped with a message")
    return ok


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--world", action="append", default=[])
ap.add_argument("--self-test", action="store_true")
a = ap.parse_args()
if a.self_test:
    sys.exit(0 if self_test() else 1)
if not a.world:
    sys.exit("give --world PATTERN (one or more)")
run(a.world)