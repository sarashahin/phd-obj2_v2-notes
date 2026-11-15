#!/usr/bin/env python3
"""
resident_loss_check.py  (obj1_v2, root cause 23, gate 1; 31 Oct 2026, Sara's yes; v2 the same day: a world with a regional
shift (HABITAT_shift_sd, assembly v2.5, plan niche_h) is its own setting, not pooled with its twin)

Question: why are random-founding residents almost never lost, even when the habitat moves and their ranges follow it
(pilot niche_t: losses per resident x0.70 and x0.74 of the static twins; habitat_track_check: AUC now 0.95, start
0.5)? Best founding, whose residents are lost about twice as often under the moving habitat, is the comparison.
Reads saved worlds only (the window-end presence history and the life table); nothing is simulated, no BNM file is
opened; writes nothing.

Per setting (founding rule, habitat tau), pooled over its worlds. A resident at round t = a species present (1
individual) at the end of round t; it is lost at round t+1 when it is absent at the end of round t+1. Exposure is
resident-rounds; hazard = losses per 1,000 resident-rounds, 95% interval exact (Poisson).
  L1 range size: hazard by the species' cells at the end of round t (1-5, 6-20, 21-50, 51+, the classes of the BNM
     bands), and the share of resident-rounds spent in each class.
  L2 dispersal: hazard by the decade of the species' dispersal rate d (LT_d): 1e-6, 1e-5, 1e-4, 1e-3.
  L3 habitat amount: the cells where a species can grow (r > 0) on the map of each round, relative to the species'
     own median over its resident rounds; for each loss, the amount on the map of the window in which it was lost.
     Static worlds: exactly 1 (control).
Checks before reading (a world that fails is not read):
  C1 every species lost between two rounds has LT_round_out = the later round, and every species new at a round has
     LT_round_in = that round (the history and the life table agree).
  C2 moving worlds: one saved map per recorded round (ENV_habitat_t), so each window's map is known.

Reading rule (written 31 Oct, before the run):
  R1 range protects: the hazard at 1-20 cells is 10 or more times the hazard at 51+ cells (upper end of its interval
     when 51+ has no loss), and a setting's residents spend 90% or more of resident-rounds at 21+ cells -> its
     residents are rarely lost because they are rarely small.
  R2 tracking: in moving worlds, slow dispersers (d below 1e-4) carry twice or more their share of resident-rounds
     among the losses -> losses come from failing to keep up with the habitat.
  R3 habitat amount: in moving worlds, half or more of the losses happen in a window where the species' habitat
     amount is below half its own median -> losses follow habitat shrinking; otherwise they come from competition or
     tracking, with the habitat amount kept.
  Expected (written now): R1 holds for both founding rules, and random-founding residents spend most rounds at 51+
  cells; R2 holds for best founding in moving worlds; R3 open (the map is standardised every round, so the amount of
  each habitat value is kept on average, but species with rare-habitat optima may lose most of theirs in some rounds).

Self-test (--self-test, planted histories, 20 x 20): T1 a planted hazard by range size is recovered (chi-square fit
within the 0.1% level, each printed hazard inside its own interval) and the 1-20 / 51+ ratio is 10 or more; T2 a flat hazard reads R1 "no"; T3 C1 catches a life table that
disagrees with the history; T4 a static map gives habitat amount exactly 1; T5 losses planted only when the habitat
amount is below half read R3 "yes", losses planted at random read "no"; T6 losses planted only in slow dispersers
read R2 "yes".

Run from obj1_v2/work:
    python3 ../tools/resident_loss_check.py --self-test | tee ../logs/resident_loss_selftest_31oct.txt | tail -1   # "6 PASS, 0 FAIL"
    python3 ../tools/resident_loss_check.py "results/stage2a/niche_d/*_training.npz" "results/stage2a/niche_b/*_training.npz" "results/stage2a/niche_t/*_training.npz" | tee ../logs/resident_loss_31oct.txt
"""
import glob, sys, tempfile
from pathlib import Path
import numpy as np
from scipy.stats import chi2

NEED = ["ASM_P_round", "ASM_P_offsets", "ASM_P_ids", "ASM_P_bits", "LT_id", "LT_round_in", "LT_round_out", "LT_d",
        "LT_opt", "LT_w", "LT_peak", "NICHE_cost", "ENV_habitat"]
RCLS = ((1, 5), (6, 20), (21, 50), (51, 10 ** 9))
RNAME = ("1-5", "6-20", "21-50", "51+")
DDEC = (1e-6, 1e-5, 1e-4, 1e-3)


def scalar(z, k, d=None):
    if k not in z.files:
        return d
    v = z[k]
    return v.item() if v.ndim == 0 else (v[0].item() if hasattr(v[0], "item") else v[0])


def poisson_ci(k, e):
    """Exact 95% interval of k events over exposure e (rate per unit exposure)."""
    lo = chi2.ppf(0.025, 2 * k) / 2 / e if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * k + 2) / 2 / e
    return lo, hi


def amount(H, opt, w, peak, cost):
    """Cells where each species can grow (r > 0) on map H."""
    r = peak[:, None, None] * np.exp(-(H[None] - opt[:, None, None]) ** 2 / (2 * w[:, None, None] ** 2)) - cost
    return (r > 0).reshape(len(opt), -1).sum(1)


def world_events(z):
    """Resident-rounds and losses of one world: list of (cells, d, rel_amount_now, lost, rel_amount_at_loss), checks."""
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
    lt = {int(i): k for k, i in enumerate(z["LT_id"])}
    rin, rout = z["LT_round_in"].astype(int), z["LT_round_out"].astype(int)
    d, opt, w, pk = (z[k].astype(float) for k in ("LT_d", "LT_opt", "LT_w", "LT_peak"))
    cost = float(scalar(z, "NICHE_cost"))
    sets = [pid[off[i]:off[i + 1]] for i in range(len(rnd))]
    cel = [cells[off[i]:off[i + 1]] for i in range(len(rnd))]
    sset = [set(x.tolist()) for x in sets]
    # habitat amount of every species present at round i, on map i (and of residents on map i+1, for losses)
    amt = {}
    for i in range(len(rnd)):
        Hi = maps[i] if moving else H
        need = sset[i] | (sset[i - 1] if i else set())
        ks = np.array(sorted(need), int)
        if len(ks):
            ix = np.array([lt[s] for s in ks])
            a = amount(Hi, opt[ix], w[ix], pk[ix], cost)
            for s, v in zip(ks.tolist(), a.tolist()):
                amt[(i, s)] = v
    med = {}
    for s in set(pid.tolist()):
        v = [amt[(i, s)] for i in range(len(rnd)) if s in sset[i]]
        med[s] = float(np.median(v)) if v else np.nan
    bad, ev = 0, []
    for i in range(len(rnd) - 1):
        now, nxt = sets[i], sset[i + 1]
        for s in nxt - sset[i]:
            bad += int(rin[lt[s]] != rnd[i + 1])
        for s, c in zip(now.tolist(), cel[i].tolist()):
            lost = s not in nxt
            if lost:
                bad += int(rout[lt[s]] != rnd[i + 1])
            m = med[s]
            rel_now = amt[(i, s)] / m if m > 0 else np.nan
            rel_loss = amt[(i + 1, s)] / m if (lost and m > 0) else np.nan
            ev.append((c, d[lt[s]], rel_now, lost, rel_loss))
    if bad:
        return None, f"C1 FAIL: {bad} species whose life table disagrees with the presence history"
    shift = float(scalar(z, "HABITAT_shift_sd", 0.0))       # v2: the regional shift (assembly v2.5) names its own setting
    return dict(ev=np.array(ev, float).reshape(-1, 5), moving=moving, tau=tau, rounds=len(rnd), shift=shift), None


def summarise(E, moving):
    """Numbers and readings of pooled events E (rows: cells, d, rel_now, lost, rel_loss)."""
    c, d, reln, lost, rell = E.T
    lost = lost.astype(bool)
    out = dict(n=len(E), k=int(lost.sum()))
    cls = []
    for (a, b), nm in zip(RCLS, RNAME):
        m = (c >= a) & (c <= b)
        e, k = int(m.sum()), int(lost[m].sum())
        lo, hi = poisson_ci(k, e) if e else (np.nan, np.nan)
        cls.append(dict(name=nm, e=e, k=k, h=1000 * k / e if e else np.nan, lo=1000 * lo, hi=1000 * hi, share=e / max(len(E), 1)))
    out["cls"] = cls
    small = (c <= 20); e_s, k_s = int(small.sum()), int(lost[small].sum())
    big = cls[3]
    h_small = 1000 * k_s / e_s if e_s else np.nan
    h_big = big["h"] if big["k"] > 0 else big["hi"]
    share21 = float(np.mean(c >= 21)) if len(E) else np.nan
    out["R1"] = dict(h_small=h_small, h_big=h_big, big_has_loss=big["k"] > 0, share21=share21,
                     ratio=h_small / h_big if h_big > 0 else np.inf,
                     yes=bool(e_s and h_big > 0 and h_small >= 10 * h_big and share21 >= 0.9) or
                         bool(e_s and h_big == 0 and k_s > 0 and share21 >= 0.9))
    dd = []
    for j, lo_d in enumerate(DDEC):
        hi_d = DDEC[j + 1] if j + 1 < len(DDEC) else 1.0 + 1e-12
        m = (d >= lo_d * (1 - 1e-9)) & (d < hi_d)
        e, k = int(m.sum()), int(lost[m].sum())
        dd.append(dict(name=f"{lo_d:.0e}", e=e, k=k, h=1000 * k / e if e else np.nan))
    out["dd"] = dd
    slow = d < 1e-4
    share_slow_res = float(np.mean(slow)) if len(E) else np.nan
    share_slow_loss = float(np.mean(slow[lost])) if lost.any() else np.nan
    out["R2"] = dict(res=share_slow_res, loss=share_slow_loss,
                     yes=bool(moving and lost.any() and share_slow_res > 0 and share_slow_loss >= 2 * share_slow_res))
    rl = rell[lost]; rl = rl[np.isfinite(rl)]
    out["R3"] = dict(n=len(rl), med=float(np.median(rl)) if len(rl) else np.nan,
                     below=float(np.mean(rl < 0.5)) if len(rl) else np.nan,
                     res_below=float(np.nanmean(reln < 0.5)) if len(E) else np.nan,
                     yes=bool(moving and len(rl) and np.mean(rl < 0.5) >= 0.5))
    out["static_amount_one"] = bool(np.all(reln[np.isfinite(reln)] == 1.0)) if not moving else None
    return out


def rule_of(z, f):
    fr = scalar(z, "FOUND_rule")
    return str(fr) if fr is not None else ("best" if "_fb" in Path(f).name else "random")


def report(files):
    groups = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        o, why = world_events(z)
        short = f"{Path(f).parent.name}/...{Path(f).name[-58:]}"
        if o is None:
            print(f"{short}: not read ({why})"); continue
        k = int(o["ev"][:, 3].sum())
        print(f"{short}: {o['rounds']} rounds, {len(o['ev'])} resident-rounds, {k} losses; C1 and C2 PASS")
        key = (rule_of(z, f), o["tau"] if o["moving"] else -1, o["shift"])
        g = groups.setdefault(key, dict(E=[], moving=o["moving"], worlds=0))
        g["E"].append(o["ev"]); g["worlds"] += 1
    for (rule, tau, sh), g in sorted(groups.items()):
        s = summarise(np.vstack(g["E"]), g["moving"])
        print(f"\n=== {rule} founding, habitat tau {'fixed' if tau < 0 else tau}{f', shift sd {sh:g}' if sh > 0 else ''}: "
              f"{g['worlds']} worlds, {s['n']} resident-rounds, {s['k']} losses")
        print("  L1 range size at the previous window end: class, resident-rounds (share), losses, hazard per 1,000 (95%)")
        for c in s["cls"]:
            print(f"     {c['name']:>5}  {c['e']:7d} ({100 * c['share']:5.1f}%)  {c['k']:4d}  "
                  + (f"{c['h']:7.2f} ({c['lo']:.2f}-{c['hi']:.2f})" if c["e"] else "      -"))
        r1 = s["R1"]
        print(f"  R1 hazard at 1-20 cells {r1['h_small']:.2f}, at 51+ {r1['h_big']:.2f}{'' if r1['big_has_loss'] else ' (upper end, no loss)'}"
              f" -> x{r1['ratio']:.1f}; resident-rounds at 21+ cells {100 * r1['share21']:.1f}%  ->  "
              + ("YES: residents rarely lost because they are rarely small" if r1["yes"] else "no"))
        print("  L2 dispersal decade: resident-rounds, losses, hazard per 1,000")
        print("     " + "; ".join(f"d {x['name']}: {x['e']}, {x['k']}, " + (f"{x['h']:.2f}" if x['e'] else "-") for x in s["dd"]))
        r2 = s["R2"]
        if g["moving"]:
            print(f"  R2 slow dispersers (d < 1e-4): {100 * r2['res']:.1f}% of resident-rounds, {100 * r2['loss']:.1f}% of losses  ->  "
                  + ("YES: losses come from failing to keep up with the habitat" if r2["yes"] else "no"))
            r3 = s["R3"]
            print(f"  L3 habitat amount (own median = 1): resident-rounds below 0.5: {100 * r3['res_below']:.1f}%; "
                  f"losses: median {r3['med']:.2f}, below 0.5 in {100 * r3['below']:.0f}% of {r3['n']}")
            print("  R3 " + ("YES: losses follow habitat shrinking" if r3["yes"] else "no: losses come with the habitat amount kept (competition or tracking)"))
        else:
            print(f"  L3 static map: habitat amount exactly 1 in every resident-round: {'PASS' if s['static_amount_one'] else 'FAIL'}")


# ------------------------------------------------------------------ self-test
def grf(rng, G=20, ls=3.0):
    k = np.fft.fftfreq(G) * G
    k2 = k[:, None] ** 2 + k[None] ** 2
    U = np.real(np.fft.ifft2(np.exp(-k2 * (np.pi * ls / G) ** 2 / 2) * np.fft.fft2(rng.normal(size=(G, G)))))
    return (U - U.mean()) / U.std()


def planted(path, rng, *, rounds=400, hazard=None, moving=False, loss_rule="range", rho=0.9, S0=30, new_per_round=1,
            opt_sd=0.15, w_range=(0.05, 0.6)):
    """A history with known losses. hazard(cells) gives the loss chance per round for loss_rule 'range'; 'amount'
    loses a species (chance 0.3) only in windows where its amount is below half its median; 'slow' loses only species
    with d < 1e-4 (chance 0.02); 'random' loses any resident with chance 0.01."""
    G, contrast, cost = 20, 0.25, 0.1
    U = grf(rng); maps = []
    nid = 0; alive = {}; trait = {}
    lt_in, lt_out = {}, {}

    def new_species(r_in):
        nonlocal nid
        s = nid; nid += 1
        trait[s] = (rng.normal(0, opt_sd), rng.uniform(*w_range), 1.5, 10 ** rng.uniform(-6, -2))
        alive[s] = int(rng.integers(1, 120)); lt_in[s] = r_in; lt_out[s] = -1
        return s
    for _ in range(S0):
        new_species(-1)
    rows_ids, rows_cells, offs, rlist = [], [], [0], []
    med_amt = {}
    amt_hist = {}
    for t in range(rounds):
        if moving and t > 0:
            U = rho * U + np.sqrt(1 - rho ** 2) * grf(rng); U = (U - U.mean()) / U.std()
        H = contrast * U; maps.append(H.copy())
        ids = sorted(alive)
        if ids:
            a = amount(H, *(np.array([trait[s][j] for s in ids]) for j in range(3)), cost)
            for s, v in zip(ids, a.tolist()):
                amt_hist.setdefault(s, []).append(v)
        if t > 0:                                       # losses during window t (map t)
            for s in list(alive):
                if s not in prev:
                    continue
                c = prev_cells[s]
                if loss_rule == "range":
                    p = hazard(c)
                elif loss_rule == "amount":
                    hist = amt_hist[s][:-1] if len(amt_hist[s]) > 1 else amt_hist[s]
                    m = np.median(hist) if hist else 0
                    p = 0.3 if (m > 0 and amt_hist[s][-1] / m < 0.5) else 0.0
                elif loss_rule == "slow":
                    p = 0.02 if trait[s][3] < 1e-4 else 0.0
                else:
                    p = 0.01
                if rng.random() < p:
                    del alive[s]; lt_out[s] = t
            for _ in range(new_per_round):
                new_species(t)
        for s in alive:                                 # range sizes change a little each round
            alive[s] = int(np.clip(alive[s] + rng.integers(-6, 7), 1, 400))
        ids = sorted(alive)
        P = np.zeros((len(ids), G * G), bool)
        for k, s in enumerate(ids):
            P[k, rng.choice(G * G, alive[s], replace=False)] = True
        rows_ids += ids; rows_cells.append(np.packbits(P, axis=1)); offs.append(offs[-1] + len(ids)); rlist.append(t)
        prev = set(ids); prev_cells = dict(alive)
    lid = np.array(sorted(trait), np.int64)
    keys = dict(ASM_P_round=np.array(rlist, np.int64), ASM_P_offsets=np.array(offs, np.int64), ASM_P_ids=np.array(rows_ids, np.int64),
                ASM_P_bits=np.vstack(rows_cells), LT_id=lid, LT_round_in=np.array([lt_in[s] for s in lid], np.int64),
                LT_round_out=np.array([lt_out[s] for s in lid], np.int64), LT_d=np.array([trait[s][3] for s in lid]),
                LT_opt=np.array([trait[s][0] for s in lid], np.float32), LT_w=np.array([trait[s][1] for s in lid], np.float32),
                LT_peak=np.array([trait[s][2] for s in lid], np.float32), NICHE_cost=np.float32(cost),
                ENV_habitat=maps[-1].astype(np.float32))
    if moving:
        keys.update(HABITAT_tau=np.int64(50000), ENV_habitat_t=np.array(maps, np.float32))
    np.savez(path, **keys)
    return np.load(path, allow_pickle=True)


def self_test():
    rng = np.random.default_rng(31)
    res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    true = {"1-5": 0.05, "6-20": 0.02, "21-50": 0.004, "51+": 0.0005}

    def hz(c):
        return true["1-5"] if c <= 5 else true["6-20"] if c <= 20 else true["21-50"] if c <= 50 else true["51+"]
    with tempfile.TemporaryDirectory() as tmp:
        Es = []
        for j in range(4):
            o, why = world_events(planted(f"{tmp}/a{j}.npz", rng, hazard=hz, S0=60, new_per_round=0, rounds=500))
            Es.append(o["ev"])
        s = summarise(np.vstack(Es), False)
        x2 = sum((c["k"] - c["e"] * true[c["name"]]) ** 2 / (c["e"] * true[c["name"]]) for c in s["cls"] if c["e"])
        fits = x2 <= chi2.ppf(0.999, 4) and all(c["lo"] <= c["h"] <= c["hi"] for c in s["cls"] if c["e"])
        check("T1", fits and s["R1"]["ratio"] >= 10,
              "planted hazard by range size recovered: " + ", ".join(f"{c['name']} {c['h']:.1f} (true {1000 * true[c['name']]:.1f})" for c in s["cls"])
              + f"; fit chi-square {x2:.1f} (0.1% level {chi2.ppf(0.999, 4):.1f}); R1 ratio x{s['R1']['ratio']:.0f}")
        Ef = np.vstack([world_events(planted(f"{tmp}/b{j}.npz", rng, loss_rule="random", S0=60, new_per_round=0, rounds=300))[0]["ev"]
                        for j in range(2)])
        sf = summarise(Ef, False)
        check("T2", not sf["R1"]["yes"] and sf["R1"]["ratio"] < 3,
              f"flat hazard: ratio 1-20 / 51+ x{sf['R1']['ratio']:.1f}, R1 {'yes' if sf['R1']['yes'] else 'no'}")
        z = planted(f"{tmp}/c.npz", rng, hazard=hz, rounds=150)
        d = dict(z); lo = d["LT_round_out"].copy(); j = int(np.flatnonzero(lo >= 0)[0]); lo[j] += 1; d["LT_round_out"] = lo
        np.savez(f"{tmp}/c2.npz", **d); o3, why3 = world_events(np.load(f"{tmp}/c2.npz"))
        o3ok, _ = world_events(z)
        check("T3", o3 is None and why3.startswith("C1 FAIL") and o3ok is not None, f"life table off by one round for one species caught ({why3})")
        s4 = summarise(o3ok["ev"], False)
        check("T4", s4["static_amount_one"], "static map: habitat amount exactly 1 in every resident-round")
        Ea = np.vstack([world_events(planted(f"{tmp}/e{j}.npz", rng, loss_rule="amount", moving=True, rounds=300, S0=40,
                                             opt_sd=0.35, w_range=(0.03, 0.12)))[0]["ev"]
                        for j in range(2)])
        Er = np.vstack([world_events(planted(f"{tmp}/r{j}.npz", rng, loss_rule="random", moving=True, rounds=300, S0=40,
                                             opt_sd=0.35, w_range=(0.03, 0.12)))[0]["ev"]
                        for j in range(2)])
        sa, sr = summarise(Ea, True), summarise(Er, True)
        check("T5", sa["R3"]["yes"] and not sr["R3"]["yes"] and sa["R3"]["n"] >= 20,
              f"losses planted at low habitat amount: below 0.5 in {100 * sa['R3']['below']:.0f}% of {sa['R3']['n']} -> "
              f"{'yes' if sa['R3']['yes'] else 'no'}; at random: {100 * sr['R3']['below']:.0f}% of {sr['R3']['n']} -> "
              f"{'yes' if sr['R3']['yes'] else 'no'}")
        Esl = np.vstack([world_events(planted(f"{tmp}/s{j}.npz", rng, loss_rule="slow", moving=True, rounds=300, S0=40))[0]["ev"]
                         for j in range(2)])
        ss = summarise(Esl, True)
        check("T6", ss["R2"]["yes"] and not sr["R2"]["yes"],
              f"losses planted in slow dispersers: {100 * ss['R2']['loss']:.0f}% of losses against {100 * ss['R2']['res']:.0f}% of "
              f"resident-rounds -> yes; random losses: {100 * sr['R2']['loss']:.0f}% against {100 * sr['R2']['res']:.0f}% -> no")
    print(f"\n{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    files = sorted({f for pat in sys.argv[1:] for f in glob.glob(pat)})
    if not files:
        sys.exit("no files: give one or more quoted patterns of _training.npz files")
    report(files)