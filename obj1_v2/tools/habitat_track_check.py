#!/usr/bin/env python3
"""
habitat_track_check.py  (obj1_v2, root cause 23, fix B, reading rule 2; 31 Oct 2026, Sara's yes; v2 the same day: a world with
a regional shift (HABITAT_shift_sd, assembly v2.5, plan niche_h) is its own setting, not pooled with its twin)

Question: once the habitat changes through time (assembly v2.4, runner --habitat-tau, plan niche_t), do species'
ranges still follow the habitat of the time? This is the guard of the fix B reading rule (written 30 Oct, before
the run): per world, the median over species of the AUC of the final suitability for occupied cells is 0.8 or more.
Reads saved worlds only (nothing is simulated, no BNM file is opened); writes nothing.

Per world, per species of the final community (true maps, no survey):
  present   cells where the species is present at the end of the final run (P_last_final, 1 individual)
  AUC now   score = its growth field on the final map (ENV_r_field, the fields of the final run); AUC = chance that
            a present cell has a higher score than an absent cell (ties count half)
  AUC start the same presence scored on the map at round 0 (ENV_habitat_start; static worlds: the one map, so equal)
  AUC -k    the same presence scored on the map of k rounds before the last (ENV_habitat_t), k = 10 and 50: how far
            back the ranges still match (description only)
  Species with no present or no absent cell are skipped and counted.
Fields on other maps use the niche formula r = peak exp(-(H - opt)^2 / (2 w^2)) - cost with the saved niche of each
species (NICHE_opt, NICHE_w, NICHE_peak, NICHE_cost); H is saved as the layer the species see (contrast x U).

Checks before reading (a world that fails is not read):
  C0 the saved fields equal the formula on the saved final map (ENV_habitat), largest difference 1e-3 or less
     (float32); catches fields saved from another map or in another species order.
  C1 moving worlds: the saved final map equals the last map of ENV_habitat_t (1e-6), or the run stopped early
     (best founding saturated; then the final map is one step past the last map, reported).

Reading rule (written 31 Oct, before the run):
  guard       per world: median AUC now >= 0.8 -> PASS; a setting keeps its guard when every world passes.
  control     moving worlds: median of (AUC now - AUC start) over species > 0, and more than half of the species
              have now > start -> "ranges follow the moving habitat"; otherwise "ranges stuck to the start map"
              (the species do not see the change, which would undercut fix B). Static worlds: difference 0 exactly.
  calibration static twins (niche_d, niche_b) are run with the same tool: if their median AUC now is below 0.9,
              the 0.8 guard is discussed again before any moving world is read.
  expected    static worlds about 0.99 (30 Oct note); moving worlds a little lower; AUC start near 0.5 in moving worlds
              (the start map is renewed many times over 600-950 rounds), give or take about 0.1: one map is compared, and
              its patches are shared by all species (a planted world with an unrelated start map read 0.60). AUC -10 and
              -50 are description only, for the same reason.

Self-test (planted worlds, 20 x 20; --self-test): T1 auc equals a direct count over pairs with ties; T2 C0 passes
on fields from the final map and fails on fields from the start map; T3 a world whose ranges follow a moving map:
guard PASS, "follow", AUC start lower; T4 ranges left on the start map: "stuck"; T5 static world: AUC start = AUC
now exactly; T6 random presence: median AUC about 0.5, guard FAIL; T7 species with all or no cells are skipped and
counted; T8 ranges planted on the map of 10 rounds before the last score highest at -10 (the lag maps are indexed
right).

Run from obj1_v2/work:
    python3 ../tools/habitat_track_check.py --self-test | tee ../logs/habitat_track_selftest_31oct.txt | tail -1   # "8 PASS, 0 FAIL" (grep -c PASS gives 9: 8 lines + this one)
    python3 ../tools/habitat_track_check.py "results/stage2a/niche_d/*_training.npz" "results/stage2a/niche_b/*_training.npz" "results/stage2a/niche_t/*_training.npz" | tee ../logs/habitat_track_31oct.txt
"""
import glob, sys, tempfile
from pathlib import Path
import numpy as np

NEED = ["P_last_final", "ENV_r_field", "ENV_habitat", "NICHE_opt", "NICHE_w", "NICHE_peak", "NICHE_cost"]
LAGS = (10, 50)
GUARD, CALIB = 0.8, 0.9


def scalar(z, k, d=None):
    if k not in z.files:
        return d
    v = z[k]
    return v.item() if v.ndim == 0 else (v[0].item() if hasattr(v[0], "item") else v[0])


def niche_r(H, opt, w, peak, cost):
    """Growth fields (S, Ny, Nx) of species with niches (opt, w, peak) on the layer H they see."""
    H = np.asarray(H, float)
    return peak[:, None, None] * np.exp(-(H[None] - opt[:, None, None]) ** 2 / (2 * w[:, None, None] ** 2)) - cost


def auc(score, pos):
    """P(score of a present cell > score of an absent cell) + 0.5 P(equal), by mid-ranks (Mann-Whitney)."""
    s = np.asarray(score, float).ravel(); p = np.asarray(pos, bool).ravel()
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    rank = (start + (cnt + 1) / 2.0)[inv]                 # mid-rank of each value, 1-based
    return float((rank[p].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def per_species_auc(fields, P):
    return np.array([auc(fields[i], P[i]) for i in range(len(P))])


def analyse(z, name=""):
    """Numbers of one world (dict) or (None, reason)."""
    miss = [k for k in NEED if k not in z.files]
    if miss:
        return None, f"missing {miss}"
    P = z["P_last_final"].astype(bool)
    r_now = z["ENV_r_field"].astype(float)
    H = z["ENV_habitat"].astype(float)
    tr = [z[k].astype(float).ravel() for k in ("NICHE_opt", "NICHE_w", "NICHE_peak")]
    cost = float(scalar(z, "NICHE_cost"))
    if not (len(P) == len(r_now) == len(tr[0])):
        return None, f"species counts differ: presence {len(P)}, fields {len(r_now)}, niches {len(tr[0])}"
    c0 = float(np.abs(niche_r(H, *tr, cost) - r_now).max()) if len(P) else 0.0
    tau = int(scalar(z, "HABITAT_tau", -1))
    moving = tau > 0 and "ENV_habitat_start" in z.files
    H0 = z["ENV_habitat_start"].astype(float) if moving else H
    maps = z["ENV_habitat_t"].astype(float) if (moving and "ENV_habitat_t" in z.files) else None
    c1 = None
    if moving and maps is not None and len(maps):
        c1 = float(np.abs(maps[-1] - H).max())
    shift = float(scalar(z, "HABITAT_shift_sd", 0.0))       # v2: the regional shift (assembly v2.5) names its own setting
    out = dict(name=name, tau=tau, moving=moving, S=len(P), c0=c0, c1=c1, rounds=None if maps is None else len(maps), shift=shift)
    if c0 > 1e-3:
        return out, "C0"
    a_now = per_species_auc(r_now, P)
    a_start = per_species_auc(niche_r(H0, *tr, cost), P) if moving else a_now.copy()
    keep = np.isfinite(a_now)
    out.update(used=int(keep.sum()), skipped=int((~keep).sum()), a_now=a_now[keep], a_start=a_start[keep])
    lag = {}
    if maps is not None:
        for k in LAGS:
            if len(maps) > k:
                lag[k] = float(np.median(per_species_auc(niche_r(maps[-1 - k], *tr, cost), P)[keep]))
    out["lag"] = lag
    return out, None


def verdict(o):
    if not o["used"]:
        return dict(med_now=np.nan, med_start=np.nan, d=np.nan, frac=np.nan, guard="no species", control="-")
    med_now, med_start = float(np.median(o["a_now"])), float(np.median(o["a_start"]))
    d = o["a_now"] - o["a_start"]
    frac = float(np.mean(d > 0))
    guard = "PASS" if med_now >= GUARD else "FAIL"
    if not o["moving"]:
        control = "static (start = now)" if np.all(d == 0) else "STATIC BUT START != NOW (check the tool)"
    else:
        control = "ranges follow the moving habitat" if (np.median(d) > 0 and frac > 0.5) else "ranges stuck to the start map"
    return dict(med_now=med_now, med_start=med_start, d=float(np.median(d)), frac=frac, guard=guard, control=control)


def rule_of(z, f):
    fr = scalar(z, "FOUND_rule")
    return str(fr) if fr is not None else ("best" if "_fb" in Path(f).name else "random")


def report(files):
    settings = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        o, why = analyse(z, Path(f).name)
        short = f"{Path(f).parent.name}/...{Path(f).name[-60:]}"
        if o is None:
            print(f"{short}: skipped ({why})"); continue
        rule = rule_of(z, f)
        head = (f"{short}\n  {rule} founding, habitat tau {o['tau'] if o['moving'] else 'fixed'}"
                + (f", shift sd {o['shift']:g}" if o["shift"] > 0 else "") + f", {o['S']} species; "
                f"C0 largest field difference {o['c0']:.1e}")
        if o["c1"] is not None:
            head += f"; C1 final map vs last of {o['rounds']} maps {o['c1']:.1e}" + ("" if o["c1"] <= 1e-6 else " (run stopped early: one step past)")
        print(head)
        if why == "C0":
            print("  C0 FAIL: saved fields are not the niche formula on the saved final map; world not read"); continue
        v = verdict(o)
        lag = "".join(f", AUC -{k} {x:.3f}" for k, x in o["lag"].items())
        print(f"  {o['used']} species used, {o['skipped']} skipped (all or no cells); median AUC now {v['med_now']:.3f}, "
              f"start {v['med_start']:.3f}{lag}")
        print(f"  now - start: median {v['d']:+.3f}, species with now > start {100 * v['frac']:.0f}%  ->  guard {v['guard']}; {v['control']}")
        key = (rule, o["tau"] if o["moving"] else -1, o["shift"])
        settings.setdefault(key, []).append(v)
    if settings:
        print("\nsettings (founding rule, habitat tau, shift sd): worlds, median AUC now per world, guard, control")
        for (rule, tau, sh), vs in sorted(settings.items()):
            meds = ", ".join(f"{v['med_now']:.3f}" for v in vs)
            g = "PASS" if all(v["guard"] == "PASS" for v in vs) else "FAIL"
            ctl = sorted({v["control"] for v in vs})
            print(f"  {rule}, {'fixed' if tau < 0 else tau}{f', shift sd {sh:g}' if sh > 0 else ''}: {len(vs)} worlds; {meds}; guard {g}; {' / '.join(ctl)}")
        stat = [v["med_now"] for (r, t, sh), vs in settings.items() if t < 0 for v in vs]
        if stat:
            low = min(stat)
            print(f"\ncalibration: static worlds' median AUC now {low:.3f} at the lowest "
                  + ("(0.9 or more: the 0.8 guard stands)" if low >= CALIB else "(below 0.9: discuss the 0.8 guard before reading the moving worlds)"))


# ------------------------------------------------------------------ self-test
def grf(rng, G=20, ls=3.0):
    k = np.fft.fftfreq(G) * G
    k2 = k[:, None] ** 2 + k[None] ** 2
    U = np.real(np.fft.ifft2(np.exp(-k2 * (np.pi * ls / G) ** 2 / 2) * np.fft.fft2(rng.normal(size=(G, G)))))
    return (U - U.mean()) / U.std()


def planted(path, rng, *, rounds=40, rho=0.9, follow="now", moving=True, S=40, noise=0.3, extremes=False):
    contrast, cost = 0.25, 0.1
    U = grf(rng); U0 = U.copy(); maps = []
    if moving:
        for _ in range(rounds):
            U = rho * U + np.sqrt(1 - rho ** 2) * grf(rng); U = (U - U.mean()) / U.std(); maps.append(contrast * U)
    H0, H = contrast * U0, contrast * U
    opt = rng.normal(0, 0.25, S); w = rng.uniform(0.2, 1.0, S); peak = 1.5 * (0.2 / w) ** 0.5
    r_now = niche_r(H, opt, w, peak, cost)
    base = {"now": r_now, "start": niche_r(H0, opt, w, peak, cost), "random": rng.normal(size=r_now.shape),
            "lag10": niche_r(maps[-11], opt, w, peak, cost) if len(maps) > 10 else None}[follow]
    P = np.zeros(r_now.shape, bool)
    for i in range(S):
        s = base[i] + noise * base[i].std() * rng.normal(size=base[i].shape)
        P[i] = s > np.quantile(s, 1 - rng.uniform(0.1, 0.6))
    if extremes:
        P[0] = True; P[1] = False
    keys = dict(P_last_final=P.astype(np.uint8), ENV_r_field=r_now.astype(np.float32), ENV_habitat=H.astype(np.float32),
                NICHE_opt=opt, NICHE_w=w, NICHE_peak=peak, NICHE_cost=np.float64(cost))
    if moving:
        keys.update(HABITAT_tau=np.int64(5000), ENV_habitat_start=H0.astype(np.float32),
                    ENV_habitat_t=np.array(maps, np.float32))
    np.savez(path, **keys)
    return np.load(path, allow_pickle=True)


def self_test():
    rng = np.random.default_rng(31)
    res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    a, p = rng.integers(0, 4, 30).astype(float), rng.random(30) < 0.4
    direct = np.mean([(x > y) + 0.5 * (x == y) for x in a[p] for y in a[~p]])
    check("T1", abs(auc(a, p) - direct) < 1e-12, f"auc with ties {auc(a, p):.6f} = direct count over pairs {direct:.6f}")
    with tempfile.TemporaryDirectory() as tmp:
        z = planted(f"{tmp}/a.npz", rng)
        o, why = analyse(z)
        d = dict(z); d["ENV_r_field"] = niche_r(z["ENV_habitat_start"].astype(float), z["NICHE_opt"], z["NICHE_w"],
                                                z["NICHE_peak"], float(z["NICHE_cost"])).astype(np.float32)
        np.savez(f"{tmp}/b.npz", **d); ob, whyb = analyse(np.load(f"{tmp}/b.npz"))
        check("T2", why is None and o["c0"] <= 1e-3 and whyb == "C0",
              f"C0 {o['c0']:.1e} on fields from the final map; fields from the start map caught ({ob['c0']:.2f})")
        v = verdict(o)
        check("T3", v["guard"] == "PASS" and v["control"] == "ranges follow the moving habitat" and v["med_start"] < v["med_now"] - 0.15,
              f"ranges follow a moving map: AUC now {v['med_now']:.3f}, start {v['med_start']:.3f}, {v['control']}")
        z4 = planted(f"{tmp}/c.npz", rng, follow="start"); v4 = verdict(analyse(z4)[0])
        check("T4", v4["control"] == "ranges stuck to the start map" and v4["med_start"] > v4["med_now"],
              f"ranges left on the start map: AUC now {v4['med_now']:.3f}, start {v4['med_start']:.3f}, {v4['control']}")
        z5 = planted(f"{tmp}/d.npz", rng, moving=False); o5 = analyse(z5)[0]; v5 = verdict(o5)
        check("T5", np.array_equal(o5["a_now"], o5["a_start"]) and v5["control"].startswith("static (")
              and v5["guard"] == "PASS", f"static world: AUC start = now exactly ({v5['med_now']:.3f}), {v5['control']}")
        z6 = planted(f"{tmp}/e.npz", rng, follow="random"); v6 = verdict(analyse(z6)[0])
        check("T6", abs(v6["med_now"] - 0.5) < 0.06 and v6["guard"] == "FAIL", f"random presence: median AUC now {v6['med_now']:.3f}, guard {v6['guard']}")
        z7 = planted(f"{tmp}/f.npz", rng, extremes=True); o7 = analyse(z7)[0]
        check("T7", o7["skipped"] == 2 and o7["used"] == 38, f"a species everywhere and one nowhere skipped: {o7['skipped']} skipped, {o7['used']} used")
        z8 = planted(f"{tmp}/g.npz", rng, follow="lag10", rounds=60); o8 = analyse(z8)[0]; v8 = verdict(o8); l8 = o8["lag"]
        check("T8", l8.get(10, 0) > max(v8["med_now"], l8.get(50, 1), v8["med_start"]) and l8[10] >= 0.9,
              f"ranges planted on the map of 10 rounds before the last: AUC -10 {l8.get(10, np.nan):.3f} highest "
              f"(now {v8['med_now']:.3f}, -50 {l8.get(50, np.nan):.3f}, start {v8['med_start']:.3f})")
    print(f"\n{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    files = sorted({f for pat in sys.argv[1:] for f in glob.glob(pat)})
    if not files:
        sys.exit("no files: give one or more quoted patterns of _training.npz files")
    report(files)