#!/usr/bin/env python3
"""
rare_end_check.py  (obj1_v2, gate 5 rare end, first fix round for the 1-5 cell share, the 1 cell share and rare clumping;
2 Nov 2026, Sara's yes)

Why: in the gate 5 setting (best founding, habitat tau 62,500, shift sd 1.0, 25 sub-steps; n70, n73-n75) the rare end
fails on the 80 window-end maps: 1-5 cell share 22/80 inside (58 below, median 13.2% against BNM 23.3%), 1 cell share
2.9% against 8.3%, rare clumping 13/80 (too high at 2-3 cells, too low at 4). footprint_export (2 Nov) found every
species can grow in 78-92% of the world (footprint median 312-366 of 400 cells), so no species is rare by habitat: every
small range comes from competition. Earlier worlds suggest a cause to test: without the regional shift (n59) the 1-5
share and rare clumping passed, with shift 0.5 (n65) rare clumping was lost, and in static worlds the rare species had
extreme habitat optima (species_compare K3: |opt| 0.57 against 0.29). The shift moves the whole habitat layer, so a
species of rare habitat may lose all of it when the shift goes the other way.
Reads saved worlds only (window-end maps, life table, habitat maps); nothing is simulated, no BNM file is opened.

Per world, over the last half of the recorded rounds (resident-round = a species present at the end of a round):
  M1 footprint: cells where the species can grow (r > 0) on that round's map (ENV_habitat_t[i], the map the round's
     window ends on; static worlds ENV_habitat), from its niche in the life table (LT_opt, LT_w, LT_peak, NICHE_cost).
     Share of resident-rounds with a small footprint, 40 cells or fewer (10% of the world: a true range of about 36 cells
     gives 5 recorded cells at the median survey of 86 known cells and detection 0.64).
  M2 loss hazard per 1,000 resident-rounds by footprint (40 or fewer, 41-200, 201+), 95% exact interval; lost = absent
     at the end of the next round.
  M3 small ranges (36 true cells or fewer): share with a small footprint, and share young (age 0 or 1 round since
     entry, LT_round_in), so a small range is told apart as habitat-rare, a newcomer still spreading, or squeezed.
  M4 species per window-end map against the regional shift m of that round (ENV_habitat_shift): Spearman correlation
     over the last half (item 2 of the 2 Nov plan, species per window on maps 52/80).
Checks before reading (a world that fails is not read):
  C0 the final fields equal the niche formula on the final map (ENV_r_field vs NICHE_*, 1e-3), and every final species'
     niche (NICHE_opt, w, peak) is found in the life table: the footprints of M1 use the simulator's own rule.
  C1 the life table agrees with the presence history (as resident_loss_check C1).
  C2 moving worlds: one map per recorded round.

Reading rule (written 2 Nov, before the build; claude/n33_result_29oct.md):
  groups: static (no habitat change: n33), moving without shift (n59), shift 0.5 (n65), shift 1.0 (n66-n69, n70,
  n73-n75), each pooled over its worlds.
  A shift removes habitat specialists: small-footprint share in the moving-without-shift group 5% or more and 3 times or
    more the shift 1.0 group's -> the shift that balances richness removes species of rare habitat; the fix goes to
    the shift's design. The hazard of small-footprint residents against the rest (M2) is reported beside it.
  B no specialists anywhere: small-footprint share below 5% in the static and the moving-without-shift groups -> every
    species can grow almost everywhere in this niche and habitat setup (contrast 0.25); the fix goes to that setup.
  C otherwise, from M3 in the shift 1.0 group: half or more of the small-range resident-rounds young -> small ranges are
    newcomers still spreading (set by entry and spread); else squeezed by competition (row 25, the stated limit).
  M4: |Spearman| 0.5 or more in 3 or more of the shift 1.0 worlds with sub-steps -> the swing of species per window on
    the maps follows the shift (reported for the decision on item 2).
  Expected (written now): A.

Self-test (--self-test, planted worlds, 20 x 20): T1 footprint and age of every resident-round equal a direct loop on
that round's map; a version reading the final map for every round and a version with age off by one both differ;
T2 planted generalists (static and moving) read B; T3 planted specialists without shift, and a shift world that loses
small-footprint species, read A, with the small-footprint hazard 3 times or more the rest (stable over 6 seeds); T4 with
the footprint shares set between the A and B cuts, newcomers small for two rounds read C transient and old species
squeezed into small ranges read C squeezed; T5 Spearman recovers
a planted dependence on m and reads about 0 for none; T6 C1 catches a life table off by one round, C0 catches fields in
another species order.

Run from obj1_v2/work:
    python3 ../tools/rare_end_check.py --self-test | tail -1
    python3 ../tools/rare_end_check.py "results/stage2a/niche_b/*_training.npz" "results/stage2a/niche_t2/*_ht62500_*_training.npz" "results/stage2a/niche_h/*_fb_ht62500_hs*_training.npz" "results/stage2a/niche_h2/*_training.npz" "results/stage2a/niche_*[p5]/*_cn1p0_*_fb_ht62500_hs1p0t62500_hss25_*_training.npz" | tee ../logs/rare_end_2nov.txt
"""
import glob, sys, tempfile
from pathlib import Path
import numpy as np
from scipy.stats import chi2, spearmanr

NEED = ["ASM_P_round", "ASM_P_offsets", "ASM_P_ids", "ASM_P_bits", "LT_id", "LT_round_in", "LT_round_out",
        "LT_opt", "LT_w", "LT_peak", "NICHE_cost", "ENV_habitat"]
SMALL_FP, SMALL_RANGE = 40, 36
FP_CLS = ((0, 40, "<=40"), (41, 200, "41-200"), (201, 10 ** 9, "201+"))
A_MIN, A_RATIO, YOUNG = 0.05, 3.0, 1


def scalar(z, k, d=None):
    if k not in z.files:
        return d
    v = z[k]
    return v.item() if v.ndim == 0 else (v[0].item() if hasattr(v[0], "item") else v[0])


def niche_r(H, opt, w, peak, cost):
    H = np.asarray(H, float)
    return peak[:, None, None] * np.exp(-(H[None] - opt[:, None, None]) ** 2 / (2 * w[:, None, None] ** 2)) - cost


def footprint(H, opt, w, peak, cost):
    return (niche_r(H, opt, w, peak, cost) > 0).reshape(len(opt), -1).sum(1)


def poisson_ci(k, e):
    lo = chi2.ppf(0.025, 2 * k) / 2 / e if k > 0 else 0.0
    return lo, chi2.ppf(0.975, 2 * k + 2) / 2 / e


def world(z, final_map_for_all=False, age_shift=0):
    """Resident-rounds of the last half: rows (cells, footprint, age, lost or -1), plus richness and shift per round."""
    miss = [k for k in NEED if k not in z.files]
    if miss:
        return None, f"missing {miss}"
    H = z["ENV_habitat"].astype(float); n = H.size
    rnd = z["ASM_P_round"].astype(int); off = z["ASM_P_offsets"].astype(int); pid = z["ASM_P_ids"].astype(int)
    cells = np.unpackbits(z["ASM_P_bits"], axis=1, count=n).sum(1)
    tau = int(scalar(z, "HABITAT_tau", -1))
    moving = tau > 0 and "ENV_habitat_t" in z.files
    maps = z["ENV_habitat_t"].astype(float) if moving else None
    if moving and len(maps) != len(rnd):
        return None, f"C2 FAIL: {len(maps)} maps for {len(rnd)} recorded rounds"
    cost = float(scalar(z, "NICHE_cost"))
    lt = {int(i): k for k, i in enumerate(z["LT_id"])}
    rin, rout = z["LT_round_in"].astype(int), z["LT_round_out"].astype(int)
    opt, w, pk = (z[k].astype(float) for k in ("LT_opt", "LT_w", "LT_peak"))
    c0 = None
    if all(k in z.files for k in ("ENV_r_field", "NICHE_opt", "NICHE_w", "NICHE_peak")):
        no, nw, npk = (z[k].astype(float).ravel() for k in ("NICHE_opt", "NICHE_w", "NICHE_peak"))
        R = z["ENV_r_field"].astype(float)
        if R.shape[0] != len(no):
            return None, f"C0 FAIL: {R.shape[0]} fields for {len(no)} niches"
        c0 = float(np.abs(niche_r(H, no, nw, npk, cost) - R).max()) if len(no) else 0.0
        lt_keys = np.stack([opt, w, pk], 1)
        found = sum(bool(np.any(np.all(np.abs(lt_keys - np.array([a, b, c])) <= 1e-5 * np.maximum(1, np.abs([a, b, c])), axis=1)))
                    for a, b, c in zip(no, nw, npk))
        if c0 > 1e-3 or found != len(no):
            return None, f"C0 FAIL: fields vs formula {c0:.1e}; {found} of {len(no)} final niches found in the life table"
    sets = [pid[off[i]:off[i + 1]] for i in range(len(rnd))]
    cel = [cells[off[i]:off[i + 1]] for i in range(len(rnd))]
    sset = [set(x.tolist()) for x in sets]
    bad = 0
    for i in range(len(rnd) - 1):
        for s in sset[i + 1] - sset[i]:
            bad += int(rin[lt[s]] != rnd[i + 1])
        for s in sset[i] - sset[i + 1]:
            bad += int(rout[lt[s]] != rnd[i + 1])
    if bad:
        return None, f"C1 FAIL: {bad} species whose life table disagrees with the presence history"
    shift_t = z["ENV_habitat_shift"].astype(float) if "ENV_habitat_shift" in z.files else None
    rows, rich, mm = [], [], []
    for i in range(len(rnd) // 2, len(rnd)):
        Hi = H if (not moving or final_map_for_all) else maps[i]
        ix = np.array([lt[s] for s in sets[i].tolist()], int)
        rich.append(len(ix))
        if shift_t is not None and len(shift_t) == len(rnd):
            mm.append(shift_t[i])
        if not len(ix):
            continue
        fp = footprint(Hi, opt[ix], w[ix], pk[ix], cost)
        age = rnd[i] - rin[ix] + age_shift
        lost = (np.array([s not in sset[i + 1] for s in sets[i].tolist()], float) if i + 1 < len(rnd)
                else np.full(len(ix), -1.0))
        rows.append(np.stack([cel[i].astype(float), fp.astype(float), age.astype(float), lost], 1))
    E = np.vstack(rows) if rows else np.zeros((0, 4))
    return dict(E=E, rich=np.array(rich, float), m=np.array(mm, float) if mm else None, moving=moving, tau=tau,
                shift=float(scalar(z, "HABITAT_shift_sd", 0.0)), sub=int(scalar(z, "HABITAT_substeps", 0) or 0),
                rounds=len(rnd), c0=c0), None


def summarise(E):
    c, fp, age, lost = E.T
    out = dict(n=len(E), small_fp=float(np.mean(fp <= SMALL_FP)) if len(E) else np.nan,
               fp_med=float(np.median(fp)) if len(E) else np.nan, fp_p10=float(np.percentile(fp, 10)) if len(E) else np.nan)
    known = lost >= 0
    hz = []
    for a, b, nm in FP_CLS:
        m = known & (fp >= a) & (fp <= b)
        e, k = int(m.sum()), int(lost[m].sum())
        lo, hi = poisson_ci(k, e) if e else (np.nan, np.nan)
        hz.append(dict(name=nm, e=e, k=k, h=1000 * k / e if e else np.nan, lo=1000 * lo, hi=1000 * hi))
    out["hz"] = hz
    rest = known & (fp > SMALL_FP)
    h_small = hz[0]["h"]; h_rest = 1000 * lost[rest].sum() / rest.sum() if rest.any() else np.nan
    out["hratio"] = h_small / h_rest if (np.isfinite(h_small) and h_rest > 0) else np.nan
    sr = c <= SMALL_RANGE
    out["small_range"] = int(sr.sum())
    out["sr_share"] = float(np.mean(sr)) if len(E) else np.nan
    out["sr_fp"] = float(np.mean(fp[sr] <= SMALL_FP)) if sr.any() else np.nan
    out["sr_young"] = float(np.mean(age[sr] <= YOUNG)) if sr.any() else np.nan
    return out


def group_of(o):
    if not o["moving"]:
        return "static"
    if o["shift"] <= 0:
        return "moving, no shift"
    return f"shift {o['shift']:g}"


def reading(G, worlds_m4):
    """G: group name -> summary. Returns (verdict text, letter)."""
    st, ns, s1 = G.get("static"), G.get("moving, no shift"), G.get("shift 1")
    if ns is None or s1 is None:
        return "needs the moving-without-shift and the shift 1.0 groups", "-"
    if ns["small_fp"] >= A_MIN and ns["small_fp"] >= A_RATIO * s1["small_fp"]:
        return (f"A shift removes habitat specialists: small footprints {100 * ns['small_fp']:.1f}% without shift against "
                f"{100 * s1['small_fp']:.1f}% at shift 1.0 (hazard of small-footprint residents x{s1['hratio']:.1f} the rest)"), "A"
    if ns["small_fp"] < A_MIN and (st is None or st["small_fp"] < A_MIN):
        return (f"B no specialists anywhere: small footprints {100 * ns['small_fp']:.1f}% without shift"
                + (f", {100 * st['small_fp']:.1f}% static" if st else "") + "; the niche and habitat setup gives every species a large footprint"), "B"
    if np.isfinite(s1["sr_young"]) and s1["sr_young"] >= 0.5:
        return f"C transient: {100 * s1['sr_young']:.0f}% of small-range resident-rounds at shift 1.0 are newcomers still spreading", "C-transient"
    return (f"C squeezed: {100 * s1['sr_young']:.0f}% of small-range resident-rounds at shift 1.0 are young; small ranges are "
            "old species squeezed by competition (row 25)"), "C-squeezed"


def report(files):
    G, rows4 = {}, []
    for f in files:
        z = np.load(f, allow_pickle=True)
        o, why = world(z)
        short = f"{Path(f).parent.name}/...{Path(f).name[-56:]}"
        if o is None:
            print(f"{short}: not read ({why})"); continue
        s = summarise(o["E"]); g = group_of(o)
        rho = np.nan
        if o["m"] is not None and len(o["m"]) == len(o["rich"]) and np.std(o["m"]) > 0:
            rho = float(spearmanr(o["rich"], o["m"]).correlation)
        if g == "shift 1" and o["sub"] > 0:
            rows4.append(rho)
        c0 = "-" if o["c0"] is None else f"{o['c0']:.1e}"
        print(f"{short}: {g}{', sub-steps ' + str(o['sub']) if o['sub'] else ''}; {o['rounds']} rounds; C0 {c0}, C1"
              + (", C2" if o["moving"] else "") + " PASS")
        print(f"   footprint median {s['fp_med']:.0f} cells (10th percentile {s['fp_p10']:.0f}); small footprint (<= {SMALL_FP}) "
              f"{100 * s['small_fp']:.1f}% of {s['n']} resident-rounds; small ranges (<= {SMALL_RANGE} cells) {100 * s['sr_share']:.1f}%, "
              f"of which small footprint {100 * s['sr_fp']:.0f}%, young {100 * s['sr_young']:.0f}%"
              + (f"; species per map vs shift: Spearman {rho:+.2f}" if np.isfinite(rho) else ""))
        G.setdefault(g, []).append(o["E"])
    if not G:
        return
    S = {g: summarise(np.vstack(v)) for g, v in G.items()}
    print("\ngroups (pooled last halves): resident-rounds, small footprint share, footprint median, hazard per 1,000 "
          "by footprint <=40 / 41-200 / 201+ (95%), small ranges: share, small footprint, young")
    order = ["static", "moving, no shift"] + sorted(k for k in S if k.startswith("shift"))
    for g in [k for k in order if k in S]:
        s = S[g]
        hz = "; ".join(f"{x['name']} {x['h']:.1f} ({x['lo']:.1f}-{x['hi']:.1f}, {x['k']}/{x['e']})" if x["e"] else f"{x['name']} -"
                       for x in s["hz"])
        print(f"  {g:17s} {s['n']:7d}  {100 * s['small_fp']:5.1f}%  {s['fp_med']:5.0f}   {hz}   "
              f"{100 * s['sr_share']:.1f}%, {100 * s['sr_fp']:.0f}%, {100 * s['sr_young']:.0f}%")
    v, _ = reading(S, rows4)
    print(f"\nreading rule (written 2 Nov, before the build): {v}")
    if rows4:
        k = sum(abs(r) >= 0.5 for r in rows4 if np.isfinite(r))
        print(f"M4 species per map vs shift, shift 1.0 worlds with sub-steps: Spearman {', '.join(f'{r:+.2f}' for r in rows4)} -> "
              + ("the swing of species per window follows the shift" if k >= 3 else "not shown to follow the shift")
              + f" ({k} of {len(rows4)} with |rho| >= 0.5)")


# ------------------------------------------------------------------ self-test
def grf(rng, G=20, ls=3.0):
    k = np.fft.fftfreq(G) * G
    k2 = k[:, None] ** 2 + k[None] ** 2
    U = np.real(np.fft.ifft2(np.exp(-k2 * (np.pi * ls / G) ** 2 / 2) * np.fft.fft2(rng.normal(size=(G, G)))))
    return (U - U.mean()) / U.std()


def planted(path, rng, *, rounds=160, S0=30, new=1, opt_sd=0.1, w=(0.4, 0.9), static=False, shift_sd=0.0,
            loss="random", young_small=False, squeeze=False, rho=0.9):
    """A world with known footprints. Present species occupy a random part of their footprint (half; one to three
    cells in their first two rounds with young_small; one to ten cells for every third species with squeeze).
    loss: 'random' (2% per round), 'small_fp' (90% per round when the footprint is 40 cells or fewer, else 1%), 'none'."""
    G, contrast, cost, peak = 20, 0.25, 0.1, 1.0
    U = grf(rng); m = 0.0; maps, shifts = [], []
    trait, rin, rout, alive = {}, {}, {}, set()
    nid = [0]

    def add(t):
        s = nid[0]; nid[0] += 1
        trait[s] = (rng.normal(0, opt_sd), rng.uniform(*w), peak); rin[s] = t; rout[s] = -1; alive.add(s)
    for _ in range(S0):
        add(-1)
    ids_rows, bits, offs, rlist = [], [], [0], []
    prev = set()
    for t in range(rounds):
        if not static and t > 0:
            U = rho * U + np.sqrt(1 - rho ** 2) * grf(rng); U = (U - U.mean()) / U.std()
            m = rho * m + np.sqrt(1 - rho ** 2) * shift_sd * rng.normal()
        H = contrast * (U + m); maps.append(H.astype(np.float32)); shifts.append(m)
        if t > 0:
            for _ in range(new):
                add(t)
        P_rows, ids_t = [], []
        for s in sorted(alive):
            o, ww, p = trait[s]
            fp = (niche_r(H, np.array([o]), np.array([ww]), np.array([p]), cost)[0] > 0).ravel()
            nfp = int(fp.sum())
            gone = nfp == 0
            if not gone and s in prev:
                pl = (0.9 if nfp <= SMALL_FP else 0.01) if loss == "small_fp" else (0.0 if loss == "none" else 0.02)
                gone = rng.random() < pl
            if gone:
                if s in prev:
                    rout[s] = t
                alive.discard(s)
                if s not in prev:
                    rin[s] = -2                                  # never present: not in the history
                continue
            cand = np.flatnonzero(fp)
            if young_small and t - rin[s] <= 1:
                k = int(rng.integers(1, 4))
            elif squeeze and s % 3 == 0:
                k = int(rng.integers(1, 11))
            else:
                k = max(1, nfp // 2)
            row = np.zeros(G * G, bool); row[rng.choice(cand, min(k, nfp), replace=False)] = True
            P_rows.append(row); ids_t.append(s)
        P = np.array(P_rows, bool).reshape(-1, G * G)
        ids_rows += ids_t; bits.append(np.packbits(P, axis=1) if len(P) else np.zeros((0, 50), np.uint8))
        offs.append(offs[-1] + len(ids_t)); rlist.append(t); prev = set(ids_t)
    lid = np.array([s for s in sorted(trait) if rin[s] != -2], np.int64)
    keys = dict(ASM_P_round=np.array(rlist, np.int64), ASM_P_offsets=np.array(offs, np.int64),
                ASM_P_ids=np.array(ids_rows, np.int64), ASM_P_bits=np.vstack(bits),
                LT_id=lid, LT_round_in=np.array([rin[s] for s in lid], np.int64), LT_round_out=np.array([rout[s] for s in lid], np.int64),
                LT_opt=np.array([trait[s][0] for s in lid], np.float32), LT_w=np.array([trait[s][1] for s in lid], np.float32),
                LT_peak=np.array([trait[s][2] for s in lid], np.float32), NICHE_cost=np.float32(cost), ENV_habitat=maps[-1])
    fin = [s for s in ids_t]
    if fin:
        no = np.array([trait[s][0] for s in fin], np.float32); nw = np.array([trait[s][1] for s in fin], np.float32)
        npk = np.array([trait[s][2] for s in fin], np.float32)
        keys.update(NICHE_opt=no, NICHE_w=nw, NICHE_peak=npk,
                    ENV_r_field=niche_r(maps[-1], no.astype(float), nw.astype(float), npk.astype(float), cost).astype(np.float32))
    if not static:
        keys.update(HABITAT_tau=np.int64(62500), ENV_habitat_t=np.array(maps, np.float32))
        if shift_sd > 0:
            keys.update(HABITAT_shift_sd=np.float64(shift_sd), ENV_habitat_shift=np.array(shifts, np.float64))
    np.savez(path, **keys)
    return np.load(path, allow_pickle=True)


def self_test():
    rng = np.random.default_rng(7); res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        z = planted(f"{tmp}/t1.npz", rng, opt_sd=0.25, w=(0.08, 0.5), rounds=60)
        o, why = world(z)
        rnd = z["ASM_P_round"]; off = z["ASM_P_offsets"]; pid = z["ASM_P_ids"]; maps = z["ENV_habitat_t"].astype(float)
        lt = {int(i): k for k, i in enumerate(z["LT_id"])}; cost = float(z["NICHE_cost"])
        exp = []
        for i in range(len(rnd) // 2, len(rnd)):
            for s in pid[off[i]:off[i + 1]]:
                k = lt[int(s)]
                r = z["LT_peak"][k] * np.exp(-(maps[i] - z["LT_opt"][k]) ** 2 / (2 * z["LT_w"][k] ** 2)) - cost
                exp.append(((r > 0).sum(), rnd[i] - z["LT_round_in"][k]))
        exp = np.array(exp, float)
        ob, _ = world(z, final_map_for_all=True); oa, _ = world(z, age_shift=1)
        good = why is None and np.array_equal(o["E"][:, 1], exp[:, 0]) and np.array_equal(o["E"][:, 2], exp[:, 1])
        good &= not np.array_equal(ob["E"][:, 1], exp[:, 0]) and not np.array_equal(oa["E"][:, 2], exp[:, 1])
        check("T1", good, f"footprint and age of {len(exp)} resident-rounds equal a direct loop on each round's map; "
                          "final-map and age-off-by-one versions differ")
        # T2 generalists everywhere -> B
        Gs = {"static": summarise(world(planted(f"{tmp}/g1.npz", rng, static=True))[0]["E"]),
              "moving, no shift": summarise(world(planted(f"{tmp}/g2.npz", rng))[0]["E"]),
              "shift 1": summarise(world(planted(f"{tmp}/g3.npz", rng, shift_sd=1.0))[0]["E"])}
        v2, l2 = reading(Gs, [])
        check("T2", l2 == "B", f"planted generalists (optimum sd 0.1, width 0.4-0.9): {v2[:90]}")
        # T3 specialists (extreme optima, narrow niches) without shift; a shift world that loses small-footprint species -> A
        sp = dict(opt_sd=0.45, w=(0.03, 0.15), S0=60, new=2)
        Gs3 = {"static": summarise(world(planted(f"{tmp}/s1.npz", rng, static=True, **sp))[0]["E"]),
               "moving, no shift": summarise(world(planted(f"{tmp}/s2.npz", rng, **sp))[0]["E"]),
               "shift 1": summarise(world(planted(f"{tmp}/s3.npz", rng, shift_sd=1.0, loss="small_fp", **sp))[0]["E"])}
        v3, l3 = reading(Gs3, [])
        check("T3", l3 == "A" and Gs3["shift 1"]["hratio"] >= 3,
              f"planted specialists, lost under the shift: {v3[:120]}")
        # T4 the C branch: footprint shares set between the A and B cuts (no-shift 2 x shift, both 5% or more), so the
        # reading falls to M3, which comes from planted worlds: newcomers small for two rounds, or old species squeezed
        tr = summarise(world(planted(f"{tmp}/c1.npz", rng, shift_sd=1.0, new=3, young_small=True))[0]["E"])
        sq = summarise(world(planted(f"{tmp}/c2.npz", rng, shift_sd=1.0, squeeze=True, new=0, loss="none"))[0]["E"])
        l4 = []
        for x in (tr, sq):
            s1 = dict(x); s1["small_fp"] = 0.05
            l4.append(reading({"moving, no shift": dict(x, small_fp=0.10), "shift 1": s1}, [])[1])
        check("T4", l4 == ["C-transient", "C-squeezed"],
              f"newcomers small for two rounds -> {l4[0]} (young {100 * tr['sr_young']:.0f}% of {tr['small_range']}); "
              f"old species squeezed -> {l4[1]} (young {100 * sq['sr_young']:.0f}% of {sq['small_range']})")
        # T5 Spearman on richness vs m
        m = rng.normal(size=300); rich = 30 - 6 * m + rng.normal(size=300); rich0 = 30 + rng.normal(size=300)
        r1, r0 = spearmanr(rich, m).correlation, spearmanr(rich0, m).correlation
        check("T5", r1 < -0.8 and abs(r0) < 0.2, f"planted richness = 30 - 6 m + noise: {r1:+.2f}; independent: {r0:+.2f}")
        # T6 C1 and C0
        d = dict(z); lo = d["LT_round_out"].copy(); j = int(np.flatnonzero(lo >= 0)[0]); lo[j] += 1; d["LT_round_out"] = lo
        np.savez(f"{tmp}/bad1.npz", **d); _, w1 = world(np.load(f"{tmp}/bad1.npz"))
        d = dict(z); d["ENV_r_field"] = d["ENV_r_field"][::-1].copy(); np.savez(f"{tmp}/bad2.npz", **d)
        _, w2 = world(np.load(f"{tmp}/bad2.npz"))
        check("T6", str(w1).startswith("C1 FAIL") and str(w2).startswith("C0 FAIL"),
              f"life table off by one round -> {str(w1)[:40]}; fields in reversed order -> {str(w2)[:40]}")
    print(f"\n{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    files = sorted({f for pat in sys.argv[1:] for f in glob.glob(pat)})
    if not files:
        sys.exit("no files: give one or more quoted patterns of _training.npz files")
    report(files)