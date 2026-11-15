"""
sink_growth.py  (obj1_v2, stage 2d-b step 1, 29 Oct 2026; version 2, 30 Oct 2026: closed-cell check)

Part E (version 1, unchanged): growth rate of each species in the cells where it is absent at the end of
the final run, for connectance-1 worlds (there every off-diagonal C_ij is the strength s, so C is known
exactly from the saved metadata). g_i(x) = r_i(x) - B_i(x) - s * sum_{j != i} B_j(x), r = ENV_r_field,
B = B_last. A share of empty cells with g > 0 means species could grow there but have not arrived.

Version 2 (30 Oct, Sara's yes): best founding holds about 3.5 x BNM's species per cell (niche_w). Are the
extra species kept alive by arrivals from other cells? Dispersal costs the source nothing (propagule mode;
LVMCM_src has the same default, minus_e false), so a species of rate d gains up to d growth per step from it.
  O1 per cell (true maps): species present (P_last_final, 1 individual); species that persist in the
     closed cell, from the species present there (no arrivals: the exact Lotka-Volterra equilibrium of the
     cell, closed form for equal competition s, counting species with 1 individual or more); species that
     persist in the closed cell from every species of the world (the cell's own capacity). Medians over cells.
  O2 presences held by arrivals: present, but not in the closed-cell set of the species present there;
     their share, by thirds of the species' d (slow, middle, fast).
  O3 species held everywhere by arrivals: present somewhere, but in no closed-cell set where present.
Calibration first (stop if either fails):
  C1 inputs: large populations (time-mean 1,000 individuals or more) sit near zero mean growth over the
     second half of the final run: median |g| 0.01 or less and 95th percentile 0.05 or less. A wrong s or C
     breaks this.
  C2 balance: over the second half of the final run the world is stationary, so mean local change plus
     mean arrivals is about 0: |sum(L + I)| / sum(I) 0.25 or less over species with d 1e-4 or more (B is saved
     as float16, so slower species' arrivals are smaller than the rounding), with L = mean N (e^g - 1) and
     I = d x kernel x mean N (2Dt, Gaussian or four-neighbour kernel as dispersal.py v2 builds it, plus the
     long-distance share). A wrong d or kernel breaks this.
Reading rule (written before the run, 30 Oct): R = median over best-founding worlds of the median closed
  species per cell (from the species present), against the median over random-founding worlds of the
  median species present per cell, per best-founding setting (strength, minimum width).
    R / random 1.25 or less -> arrivals hold the excess -> next change on the dispersal side
                               (lower upper d, or emigration loss)
    R / random 2 or more    -> local coexistence holds the excess -> next on the founding and competition side
    between                 -> both; shares reported
Reads saved files only; writes nothing.

Run from obj1_v2/work:
    python3 ../tools/sink_growth.py --self-test | tee ../logs/sink_growth_selftest_30oct.txt
    python3 ../tools/sink_growth.py "results/stage2a/niche_d/*_training.npz" "results/stage2a/niche_b/*_training.npz" \\
        "results/stage2a/niche_w/*_training.npz" | tee ../logs/sink_growth_v2_30oct.txt
Version 1's call (one pattern, part E only for those files) still works:
    python3 ../tools/sink_growth.py "results/stage2a/niche_f/*cn1p0*_training.npz"
"""
import glob, os, sys, tempfile
import numpy as np


C2_D_MIN = 1e-4   # C2 uses species with d >= this: saved B is float16 (spacing 5e-4 near B 1), so the local change of a
                  # large population carries rounding of about N x 5e-4 per step, larger than the arrivals of slower species


# ---------------------------------------------------------------- model pieces
def growth(r, B, s):
    """g_i(x) = r_i(x) - B_i(x) - s * sum_{j != i} B_j(x); B (..., S, G, G)."""
    return r - B - s * (B.sum(-3, keepdims=True) - B)


def kernel_disp(G, kind, a, b, images):
    """Weight of landing at displacement (dy, dx) mod G from a source, as dispersal.py v2 builds it: rows sum
    to 1, the source cell gets 0; 2dt (1 + r^2/a^2)^-b and gauss exp(-r^2 / 2a^2) summed over periodic images."""
    kind = str(kind)
    K = np.zeros((G, G))
    if kind == "nn4":
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            K[dy % G, dx % G] += 0.25
    elif kind in ("2dt", "gauss"):
        dy, dx = np.mgrid[0:G, 0:G]
        for i in range(-images, images + 1):
            for j in range(-images, images + 1):
                r2 = (dy + i * G) ** 2 + (dx + j * G) ** 2
                w = (1 + r2 / a ** 2) ** (-b) if kind == "2dt" else np.exp(-r2 / (2 * a ** 2))
                K += w
    else:
        raise ValueError(f"kernel {kind} not known")
    K[0, 0] = 0.0
    return K / K.sum()


def arrivals(N, d, Kd, p):
    """Expected arrivals per step (individuals): d_i [(1 - p) (N_i * K) + p sum(N_i) / G^2]; N (S, G, G)."""
    G = N.shape[-1]
    conv = np.real(np.fft.ifft2(np.fft.fft2(N) * np.fft.fft2(Kd)[None]))
    return d[:, None, None] * ((1 - p) * conv + p * N.sum((1, 2), keepdims=True) / G ** 2)


def closed_count(r, s, thr):
    """Species that persist in a closed cell (no arrivals) with equal competition s (diagonal 1): the exact
    equilibrium is the top-n set by r, B_k = (r_k - s T_n) / (1 - s), T_n = R_n / (1 - s + s n); B of the
    lowest-r member falls as n grows, so the answer is the largest n whose lowest member has B >= thr.
    r (S, M) for M cells -> boolean (S, M) of persisting species."""
    S, M = r.shape
    order = np.argsort(-r, axis=0, kind="mergesort")
    rs = np.take_along_axis(r, order, 0)
    n = np.arange(1, S + 1)[:, None]
    T = np.cumsum(rs, 0) / (1 - s + s * n)
    Bn = (rs - s * T) / (1 - s)
    ok = Bn >= thr
    k = np.where(ok.any(0), S - np.argmax(ok[::-1], 0), 0)            # largest n with B_(n) >= thr
    keep_sorted = n <= k[None]
    out = np.zeros((S, M), bool)
    np.put_along_axis(out, order, keep_sorted, 0)
    return out


def closed_from(present, r, s, thr):
    """Closed-cell set per cell from the given candidates (present species): r of other species set to -inf."""
    S, G, _ = r.shape
    rr = np.where(present, r, -np.inf).reshape(S, -1)
    rr = np.where(np.isfinite(rr), rr, -1e9)                             # never in a set (r below any threshold)
    return closed_count(rr, s, thr).reshape(S, G, G) & present


# ---------------------------------------------------------------- one world
def scalar(z, k, default=None):
    if k not in z.files:
        return default
    v = z[k]
    return v.item() if v.ndim == 0 else (str(v[0]) if v.dtype.kind in "UO" else v[0].item())


def world(f):
    z = np.load(f, allow_pickle=True)
    if "CONNECTANCE" not in z.files or abs(float(z["CONNECTANCE"]) - 1.0) > 1e-6:
        return None, f"skipped (connectance not 1): {os.path.basename(f)[:80]}"
    need = ["INTERACTION_STRENGTH", "B_last", "ENV_r_field", "P_last_final", "IBM_B", "BODY_MASS", "DISPERSAL_KERNEL"]
    miss = [k for k in need if k not in z.files]
    if miss:
        return None, f"skipped (missing {miss}): {os.path.basename(f)[:80]}"
    s = float(z["INTERACTION_STRENGTH"])
    if not 0 <= s < 1:
        return None, f"skipped (strength {s} outside 0-1; the closed form needs s < 1): {os.path.basename(f)[:80]}"
    body = float(z["BODY_MASS"])
    r = np.asarray(z["ENV_r_field"], float)
    S, G, _ = r.shape
    per = int(z["DISP_per_species"]) if "DISP_per_species" in z.files else 0
    d = np.asarray(z["SP_disp_rate"], float) if per == 1 else np.full(S, float(z["DISPERSAL_RATE"]))
    Kd = kernel_disp(G, scalar(z, "DISPERSAL_KERNEL"), float(scalar(z, "DISPERSAL_A", 1.0)),
                     float(scalar(z, "DISPERSAL_B", 1.75)), int(scalar(z, "DISPERSAL_IMAGES", 10)))
    p = float(scalar(z, "LONG_DISTANCE_PROB", 0.0))
    frames = np.asarray(z["IBM_B"], float)
    T = len(frames)
    Bf = frames[T // 2:]                                                  # second half of the final run
    Nf = Bf / body
    gf = growth(r[None], Bf, s)
    Nbar = Nf.mean(0)
    gbar = growth(r, Bf.mean(0), s)                                       # linear in B: the time-mean growth
    L = (Nf * np.expm1(gf)).mean(0)                                       # mean local change per step
    I = arrivals(Nbar, d, Kd, p)                                          # mean arrivals per step (linear in N)
    P = np.asarray(z["P_last_final"]) > 0
    big = Nbar >= 1000
    c1_med = float(np.median(np.abs(gbar[big]))) if big.any() else np.nan
    c1_p95 = float(np.percentile(np.abs(gbar[big]), 95)) if big.any() else np.nan
    fastsp = d >= C2_D_MIN                                                # float16 B: below this d the balance is lost in rounding
    c2 = float((L[fastsp] + I[fastsp]).sum() / I[fastsp].sum()) if fastsp.any() and I[fastsp].sum() > 0 else np.nan
    cp = closed_from(P, r, s, body)                                       # B >= 1 individual
    ca = closed_count(r.reshape(S, -1), s, body).reshape(S, G, G)
    held = P & ~cp
    lg = np.log10(np.maximum(d, 1e-300))
    third = np.digitize(lg, np.quantile(lg, [1 / 3, 2 / 3])) if np.ptp(lg) > 0 else np.ones(S, int)
    thirds = []
    for t in range(3):
        m = third == t
        nP = int(P[m].sum()); nH = int(held[m].sum())
        thirds.append((int(m.sum()), nP, nH))
    sp_present = P.reshape(S, -1).any(1)
    sp_everywhere = sp_present & ~cp.reshape(S, -1).any(1)
    w = dict(file=f, name=os.path.basename(f), S=S, s=s, rule=scalar(z, "FOUND_rule", "random"),
             w_min=float(scalar(z, "NICHE_w_min", 0.2)), seed=scalar(z, "SIM_SEED"), frames=(T, len(Bf)),
             c1=(c1_med, c1_p95, int(big.sum())), c2=c2, c2n=int(fastsp.sum()),
             pres=float(np.median(P.sum(0))), clp=float(np.median(cp.sum(0))), cla=float(np.median(ca.sum(0))),
             held=(int(held.sum()), int(P.sum())), thirds=thirds,
             every=(int(sp_everywhere.sum()), int(sp_present.sum())),
             every_third=[int((sp_everywhere & (third == t)).sum()) for t in range(3)],
             gE=growth(r, np.asarray(z["B_last"], float), s)[~P], gO=growth(r, np.asarray(z["B_last"], float), s)[P])
    w["c1ok"] = bool(np.isfinite(c1_med) and c1_med <= 0.01 and c1_p95 <= 0.05)
    w["c2ok"] = bool(abs(c2) <= 0.25) if np.isfinite(c2) else True     # not testable without species of d >= C2_D_MIN
    return w, None


def label(w):
    return f"{w['rule']} founding, strength {w['s']:.1f}, w_min {w['w_min']:.1f}"


def verdict(ratio):
    if not np.isfinite(ratio):
        return "not read"
    if ratio <= 1.25:
        return "arrivals hold the excess -> next change on the dispersal side"
    if ratio >= 2:
        return "local coexistence holds the excess -> next change on the founding and competition side"
    return "both; read the shares"


def run(patterns):
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    if not files:
        sys.exit("STOP: no files matched")
    ws = []
    for f in files:
        w, msg = world(f)
        print(msg) if msg else ws.append(w)
    if not ws:
        sys.exit("STOP: no connectance-1 world with the needed keys")

    print("\nE  empty cells (version 1): g from B_last")
    okE = True
    for w in ws:
        ge, go = w["gE"], w["gO"]
        good = abs(np.median(go)) < 0.02; okE &= good
        print(f"   {w['name'][-34:]}: empty-cell g quartiles {np.percentile(ge, 25):.3f} / {np.median(ge):.3f} / "
              f"{np.percentile(ge, 75):.3f}; share g > 0 {np.mean(ge > 0):.1%}; occupied median g {np.median(go):+.4f} "
              f"{'PASS' if good else 'FAIL'}")

    print("\nC  calibration (second half of the final run)")
    for w in ws:
        m, p95, nb = w["c1"]
        c2 = (f"C2 sum(L + I) / sum(I), {w['c2n']} species with d >= {C2_D_MIN:g}: {w['c2']:+.3f} {'PASS' if w['c2ok'] else 'FAIL'}"
              if np.isfinite(w["c2"]) else f"C2 not testable (no species with d >= {C2_D_MIN:g})")
        print(f"   {w['name'][-34:]}: frames {w['frames'][1]} of {w['frames'][0]}; C1 large populations ({nb}) |g| median "
              f"{m:.4f}, 95th pct {p95:.4f} {'PASS' if w['c1ok'] else 'FAIL'}; {c2}")
    if not all(w["c1ok"] and w["c2ok"] for w in ws):
        print("\nSTOP: a calibration failed; inputs are not read as the simulator uses them. Nothing below is read.")
        return False

    print("\nO  per world (true maps; medians over cells)")
    print("   world                                 rule    s    w_min  species  present/cell  closed (present)  closed (all)  held by arrivals  held, slow/mid/fast d third  species held everywhere (slow/mid/fast)")
    for w in ws:
        h, n = w["held"]
        tt = " / ".join(f"{(c[2] / c[1] if c[1] else 0):.0%}" for c in w["thirds"])
        e, sp = w["every"]
        print(f"   {w['name'][-36:]:36s}  {w['rule']:6s}  {w['s']:.1f}  {w['w_min']:.1f}    {w['S']:4d}     {w['pres']:6.1f}        "
              f"{w['clp']:6.1f}            {w['cla']:6.1f}       {h / n:6.1%}            {tt:22s}      {e} of {sp} "
              f"({' / '.join(str(x) for x in w['every_third'])})")

    rnd = [w for w in ws if w["rule"] == "random"]
    best = [w for w in ws if w["rule"] == "best"]
    print("\nReading rule (written before the run): median closed species per cell (from the species present) in best-"
          "founding worlds, against median species present per cell in random-founding worlds")
    if not rnd or not best:
        print("   not read: needs random-founding and best-founding worlds")
        return True
    ref = float(np.median([w["pres"] for w in rnd]))
    print(f"   random founding ({len(rnd)} worlds): species present per cell {ref:.1f}")
    for key in sorted({(w["s"], w["w_min"]) for w in best}):
        g = [w for w in best if (w["s"], w["w_min"]) == key]
        R = float(np.median([w["clp"] for w in g]))
        pres = float(np.median([w["pres"] for w in g]))
        print(f"   best founding, strength {key[0]:.1f}, w_min {key[1]:.1f} ({len(g)} worlds): present {pres:.1f}, closed {R:.1f} "
              f"-> x{R / ref:.2f} of random -> {verdict(R / ref)}")
    return True


# ---------------------------------------------------------------- self-test
def self_test():
    rng = np.random.default_rng(0); ok = True

    def check(name, cond, text=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name} {text} {'PASS' if cond else 'FAIL'}")

    # T1 closed form = brute-force equilibrium (every subset; the one with B > 0 inside and g <= 0 outside)
    import itertools
    bad = 0
    for t in range(150):
        S = int(rng.integers(1, 8)); s = float(rng.choice([0.2, 0.4, 0.7])); r = rng.normal(0.5, 0.4, S)
        C = s * np.ones((S, S)) + (1 - s) * np.eye(S); want = None
        for k in range(S + 1):
            for sub in itertools.combinations(range(S), k):
                B = np.zeros(S)
                if k:
                    ix = list(sub); B[ix] = np.linalg.solve(C[np.ix_(ix, ix)], r[ix])
                    if (B[ix] <= 0).any():
                        continue
                if np.all((r - C @ B)[[i for i in range(S) if i not in sub]] <= 1e-12):
                    want = B > 0
        bad += not np.array_equal(closed_count(r[:, None], s, 0.0)[:, 0], want)
    check("T1", bad == 0, f"closed form equals the brute-force equilibrium in 150 random cells ({bad} differ)")
    r = np.array([[1.0], [0.99], [0.98], [0.5]]); s = 0.4
    got = closed_count(r, s, 0.0)[:, 0]; got_t = closed_count(r, s, 0.02)[:, 0]
    T3 = r[:3, 0].sum() / (1 - s + 3 * s); B3 = (r[:3, 0] - s * T3) / (1 - s)
    check("T2", got.tolist() == [True, True, True, False] and got_t.sum() <= 3 and np.all(B3 > 0.02) == bool(got_t[:3].all()),
          f"three near-equal species coexist and a weak one is excluded; a threshold drops members below it (B {np.round(B3, 3).tolist()})")

    # T3 kernel as dispersal.py v2 (the audit's shares for 2dt a 1, b 1.75) and arrivals = a direct loop
    K = kernel_disp(20, "2dt", 1.0, 1.75, 10)
    dy = np.minimum(np.arange(20), 20 - np.arange(20)); ch = np.maximum(dy[:, None], dy[None, :])
    sh = (K[ch == 1].sum(), K[ch > 2].sum(), K[ch > 5].sum())
    check("T3a", np.isclose(K.sum(), 1) and K[0, 0] == 0 and abs(sh[0] - 0.546) < 0.0015 and abs(sh[1] - 0.247) < 0.0015
          and abs(sh[2] - 0.075) < 0.0015, f"2dt a 1 b 1.75: 8 neighbours {sh[0]:.4f}, beyond 2 {sh[1]:.4f}, beyond 5 {sh[2]:.4f}")
    G = 8; N = rng.integers(0, 50, (3, G, G)).astype(float); dd = np.array([1e-3, 1e-2, 0.1]); Kd = kernel_disp(G, "2dt", 1.0, 3.0, 10)
    direct = np.zeros_like(N)
    for i in range(3):
        for y in range(G):
            for x in range(G):
                for yy in range(G):
                    for xx in range(G):
                        direct[i, yy, xx] += dd[i] * (0.8 * N[i, y, x] * Kd[(yy - y) % G, (xx - x) % G] + 0.2 * N[i, y, x] / G ** 2)
    check("T3b", np.allclose(arrivals(N, dd, Kd, 0.2), direct) and np.allclose(arrivals(N, dd, Kd, 0.2).sum((1, 2)), dd * N.sum((1, 2))),
          "arrivals equal a direct loop over sources, with a 20% long-distance share; totals = d x N")

    # T4 planted worlds: deterministic stationary state of N' = N e^g + I (the IBM's expected update)
    def make(path, d, s_saved=0.4, s_true=0.4, steps=4000):
        G, S, body = 10, 6, 1e-4
        x = np.arange(G)[None, :] * np.ones((G, 1))
        left = (x < G / 2).astype(float)
        r = np.zeros((S, G, G))
        for i in range(3):
            r[i] = np.where(left > 0, 0.9 + 0.05 * i, -0.2)
            r[3 + i] = np.where(left > 0, -0.2, 0.9 + 0.05 * i)
        Kd = kernel_disp(G, "2dt", 1.0, 3.0, 10); N = np.full((S, G, G), 100.0); fr = []
        for t in range(steps):
            B = N * body
            N = N * np.exp(growth(r, B, s_true)) + arrivals(N, d, Kd, 0.0)
            if t >= steps - 50:
                fr.append(N * body)
        B = np.array(fr)
        np.savez(path, CONNECTANCE=np.float32(1), INTERACTION_STRENGTH=np.float32(s_saved), B_last=B[-1].astype(np.float16),
                 ENV_r_field=r.astype(np.float32), P_last_final=(B[-1] / body >= 1).astype(np.uint8), IBM_B=B.astype(np.float16),
                 BODY_MASS=np.float32(body), DISPERSAL_KERNEL=np.array(["2dt"]), DISPERSAL_A=np.float32(1), DISPERSAL_B=np.float32(3),
                 DISPERSAL_IMAGES=np.int64(10), LONG_DISTANCE_PROB=np.float32(0), DISP_per_species=np.int8(1),
                 SP_disp_rate=d, FOUND_rule=np.array(["best"]), SIM_SEED=np.int64(1))

    with tempfile.TemporaryDirectory() as tmp:
        dhi = np.array([1e-6, 1e-4, 3e-2, 1e-6, 1e-4, 3e-2])
        make(f"{tmp}/hi.npz", dhi)
        w, msg = world(f"{tmp}/hi.npz")
        check("T4a", w is not None and w["c1ok"] and w["c2ok"],
              f"stationary planted world: C1 {w['c1'][0]:.4f} / {w['c1'][1]:.4f}, C2 {w['c2']:+.3f}")
        h, n = w["held"]
        fast = w["thirds"][2]
        check("T4b", h > 0 and fast[2] == h and w["clp"] <= w["pres"],
              f"fast dispersers present on the wrong side are held by arrivals ({h} of {n} presences, all in the fast third)")
        make(f"{tmp}/lo.npz", np.full(6, 1e-6))
        w2, _ = world(f"{tmp}/lo.npz")
        check("T4c", w2["c1ok"] and not np.isfinite(w2["c2"]) and w2["held"][0] == 0 and w2["every"][0] == 0,
              f"slow dispersal: no presence held by arrivals ({w2['held'][0]} of {w2['held'][1]}); C2 not testable, as it should be")
        z = dict(np.load(f"{tmp}/hi.npz", allow_pickle=True)); z["SP_disp_rate"] = dhi * 10
        np.savez(f"{tmp}/wrong_d.npz", **z); w3, _ = world(f"{tmp}/wrong_d.npz")
        check("T5a", not w3["c2ok"], f"negative control: d read 10 x too high fails C2 ({w3['c2']:+.2f})")
        make(f"{tmp}/wrong_s.npz", dhi, s_saved=0.7, s_true=0.4)
        w4, _ = world(f"{tmp}/wrong_s.npz")
        check("T5b", not w4["c1ok"], f"negative control: strength read as 0.7 (true 0.4) fails C1 (median |g| {w4['c1'][0]:.3f})")
        z = dict(np.load(f"{tmp}/hi.npz", allow_pickle=True)); z["CONNECTANCE"] = np.float32(0.4)
        np.savez(f"{tmp}/c04.npz", **z); w5, msg5 = world(f"{tmp}/c04.npz")
        check("T6", w5 is None and "connectance not 1" in msg5, "a connectance-0.4 world is skipped (C not known exactly)")
    check("T7", verdict(1.1).startswith("arrivals") and verdict(2.5).startswith("local") and verdict(1.6).startswith("both")
          and verdict(np.nan) == "not read", "every branch of the reading rule")
    return ok


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    pats = [a for a in sys.argv[1:] if not a.startswith("--")] or ["results/stage2a/niche_f/*cn1p0*_training.npz"]
    sys.exit(0 if run(pats) else 1)