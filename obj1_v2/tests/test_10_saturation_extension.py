"""
test_10_saturation_extension.py  (obj1_v2, 26 Oct 2026; X4 turnover added 27 Oct)

Question: is assembly saturated after 1,000 attempts? Each world of plan "extend" (2,000
attempts) is compared with its 1,000-attempt twin of plan "settle" (same setting and seed).
test_09 showed that one assembly history cannot answer this at about 30 species, so each
world is compared with itself after twice the attempts.

X1  Assembly repeats: the 2,000-attempt history starts with the 1,000-attempt history
    (richness, attempts and newcomers kept in every round but the twin's last, which the
    attempt cap cuts short). If not, the pair are two replicates (noted).
X2  Change from 1,000 to 2,000 attempts, per world, in five numbers: species in the final
    map; assembly richness, mean of the last 200 rounds; and, seen through BNM survey effort
    on the geographic fit half (world_vs_bnm.py version 3 functions, test half never opened),
    range-size distance to BNM (TV), rare-case co-occupancy at 1-4 cells relative to BNM
    (mean of the four ratios), species per window relative to BNM. Printed with the
    difference between the two seeds of the same setting at 1,000 attempts, as a guide only.
X4  Turnover (added 27 Oct): species ids count invasion attempts (id = attempt at arrival), so
    the saved ids show who replaced whom. Share of the 1,000-attempt community still present
    after 2,000 attempts, and share of the 2,000-attempt community that arrived after attempt
    1,000; per world, the share of the community that arrived in the last 20% of its attempts.
    Description only, no pass rule.
X3  Verdict: the changes of all paired worlds are pooled; for each number a one-sample
    t-test asks whether the mean change differs from 0 (two-sided, 5% shared over the five
    numbers). Saturated if no number moves.
Why X3 and not "every change within the seed difference" (agreed first on 26 Oct):
--calibrate shows that rule fails about two thirds of saturated worlds, because with two
seeds the yardstick is a single noisy difference; the pooled test fails about 2% (set to 5% at most).
Run from obj1_v2/work:
  python3 ../tests/test_10_saturation_extension.py --short "results/stage2a/settle/*_training.npz" \
      --long "results/stage2a/extend/*_training.npz"
  python3 ../tests/test_10_saturation_extension.py --calibrate     # pure calculation, about 30 seconds
Result of --calibrate (26 Oct, 4 worlds, 1,000 trials, immigration-loss process as in test_09):
  saturated (30 or 50 species): seed-difference rule fails 66-68% of worlds, pooled test 1.7-2.0%;
  drift of about +8 species: pooled test 17.2%; about +21 species: 57.7%. With 4 worlds of about
  30 species, only a drift of about 20 species or more is likely to be seen.
"""
import argparse, glob, os, sys
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--short", help="1,000-attempt worlds (plan settle)")
ap.add_argument("--long", help="2,000-attempt worlds (plan extend)")
ap.add_argument("--bnm", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
ap.add_argument("--split-ref", default=None, help="windows that set the geographic split (default: --bnm)")
ap.add_argument("--wvb", default="../tools/world_vs_bnm.py", help="world_vs_bnm.py version 3 (fit half only)")
ap.add_argument("--last-rounds", type=int, default=200)
ap.add_argument("--calibrate", action="store_true")
a = ap.parse_args()
N_NUM, ALPHA = 5, 0.05
CRIT = lambda n: stats.t.ppf(1 - ALPHA / (2 * N_NUM), n - 1)          # two-sided, shared over the five numbers


def pooled(d):
    """t for the mean change of each number over worlds (rows); inf when every change is equal and not 0."""
    m, s = d.mean(0), d.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(s > 0, m / (s / np.sqrt(len(d))), np.where(m == 0, 0.0, np.inf))
    return m, t


if a.calibrate:
    rng = np.random.default_rng(1)
    def world(lam, S, g0):
        mu, x, g = lam / S, (rng.poisson(S) if g0 is None else g0), np.empty(1000)
        for t in range(1000):
            x = x + rng.poisson(lam) - rng.binomial(x, mu); g[t] = x
        return np.array([[g[529], g[330:530].mean()], [g[999], g[800:1000].mean()]])   # 1,000 and 2,000 attempts
    for lab, lam, S, g0 in (("saturated, 30 species", 0.3, 30, None), ("saturated, 50 species", 0.3, 50, None),
                            ("drifting about +8 species", 0.3, 80, 5), ("drifting about +21 species", 0.3, 120, 5)):
        old = new = 0
        for _ in range(1000):
            W = [world(lam, S, g0) for _ in range(4)]
            d = np.array([w[1] - w[0] for w in W]); sp = np.array([abs(W[0][0] - W[1][0])] * 2 + [abs(W[2][0] - W[3][0])] * 2)
            old += np.mean(~np.all(np.abs(d) <= sp, axis=1)); new += bool(np.any(np.abs(pooled(d)[1]) > CRIT(4)))
        print(f"{lab:<28} seed-difference rule fails {old / 10:.0f}% of worlds; pooled test fails {new / 10:.1f}% of sets of 4")
    sys.exit(0)
if not (a.short and a.long): ap.error("--short and --long are required (or --calibrate)")

# world_vs_bnm.py functions, without running its command line (the file is read up to its parser)
src = open(a.wvb).read(); cut = "\nap = argparse.ArgumentParser()"
if src.count(cut) != 1 or "def bnm_glob" not in src:
    sys.exit(f"STOP: {a.wvb} is not world_vs_bnm.py version 3 (fit half only)")
W = {"__name__": "world_vs_bnm_functions"}
exec(compile(src[: src.index(cut)], a.wvb, "exec"), W)
W["REF"] = W["split_ref"](a.split_ref or a.bnm)
bfiles, bcases = W["load_bnm"](a.bnm)
if not bcases: sys.exit("STOP: no BNM fit-half windows read")
target = W["summarise"](bcases, wrap=False)
masks = W["survey_masks"](a.bnm)
NAMES = ["species in final map", f"assembly richness, mean of last {a.last_rounds} rounds", "range-size TV to BNM",
         "rare co-occupancy / BNM, mean 1-4 cells", "species per window / BNM"]


def load(f):
    z = np.load(f, allow_pickle=True)
    P = z["P_last_final"].astype(bool)
    co, tv, rr, _ = W["score"](W["summarise"](W["observe"](P, masks), wrap=False), target)
    g = z["ASM_gamma"].astype(float)
    key = (round(float(z["DISPERSAL_RATE"]), 12), str(z["DISPERSAL_KERNEL"][0]),
           round(float(z["DISPERSAL_A"]), 6) if "DISPERSAL_A" in z.files else None,
           round(float(z["DISPERSAL_B"]), 6) if "DISPERSAL_B" in z.files else None,
           int(z["seed_size"]), int(z["window_steps"]))
    return dict(file=os.path.basename(f), seed=int(z["SIM_SEED"]), key=key, z=z,
                stats=np.array([float(P.any(axis=(1, 2)).sum()), float(g[-a.last_rounds:].mean()), tv, float(np.nanmean(co)), rr]))


short = [load(f) for f in sorted(glob.glob(a.short))]
long_ = [load(f) for f in sorted(glob.glob(a.long))]
if not short or not long_: sys.exit(f"STOP: {len(short)} short and {len(long_)} long worlds matched")
print(f"BNM fit half: {len(bfiles)} windows; short worlds {len(short)}, long worlds {len(long_)}")
by_seed = {w["seed"]: w for w in short}
fmt = lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
changes, all_rep = [], True
for L in long_:
    S = by_seed.get(L["seed"])
    print(f"\nseed {L['seed']}: {L['file'][:60]}")
    if S is None or S["key"] != L["key"]:
        print("  no 1,000-attempt twin with this seed and setting: not compared"); continue
    gs, gl = S["z"]["ASM_gamma"], L["z"]["ASM_gamma"]; n = len(gs) - 1
    rep = n > 0 and len(gl) > n and all(np.array_equal(S["z"][k][:n], L["z"][k][:n]) for k in ("ASM_gamma", "ASM_attempts", "ASM_established"))
    all_rep &= rep
    print(f"  X1 assembly repeats for the first {n} rounds (the twin's last round is cut by the cap): "
          f"{'PASS' if rep else 'FAIL (the two runs are replicates, not one run extended)'}")
    print(f"     rounds {len(gs)} -> {len(gl)}, attempts {int(S['z']['ASM_attempts'][-1])} -> {int(L['z']['ASM_attempts'][-1])}")
    mates = [w for w in short if w["key"] == L["key"] and w["seed"] != L["seed"]]
    change = L["stats"] - S["stats"]; changes.append(change)
    if "species_id" in S["z"].files and "species_id" in L["z"].files:
        ids_s, ids_l = set(S["z"]["species_id"].tolist()), set(L["z"]["species_id"].tolist())
        A_s, A_l = int(S["z"]["ASM_attempts"][-1]), int(L["z"]["ASM_attempts"][-1])
        young = lambda ids, A: np.mean(np.array(sorted(ids)) > 0.8 * A)
        print(f"  X4 turnover: {len(ids_s & ids_l) / len(ids_s):.0%} of the {len(ids_s)} species at {A_s} attempts are still present at {A_l}; "
              f"{np.mean(np.array(sorted(ids_l)) > A_s):.0%} of the {len(ids_l)} species at {A_l} arrived after attempt {A_s}")
        print(f"     arrived in the last 20% of attempts: {young(ids_s, A_s):.0%} of the community at {A_s}, {young(ids_l, A_l):.0%} at {A_l}")
    else:
        print("  X4 turnover: no species_id in the files")
    for i, nm in enumerate(NAMES):
        guide = f"  seed difference {fmt(abs(S['stats'][i] - mates[0]['stats'][i])):>7} (seed {mates[0]['seed']}, guide only)" if len(mates) == 1 else ""
        print(f"  X2 {nm:<42} 1,000: {fmt(S['stats'][i]):>7}  2,000: {fmt(L['stats'][i]):>7}  change {fmt(change[i]):>7}{guide}")
D = np.array(changes)
print(f"\nX3 pooled over {len(D)} paired worlds (needs at least 3):")
if len(D) < 3:
    print("  too few pairs: no verdict"); sys.exit(0)
m, t = pooled(D); crit = CRIT(len(D)); moved = []
for i, nm in enumerate(NAMES):
    mv = abs(t[i]) > crit; moved += [nm] if mv else []
    print(f"  {nm:<45} mean change {fmt(m[i]):>7}  t {t[i]:7.2f}  (moves if |t| > {crit:.2f})  {'MOVED' if mv else 'ok'}")
print(f"  verdict: {'saturated at 1,000 attempts (PASS)' if not moved else 'not saturated at 1,000 attempts (FAIL): ' + '; '.join(moved)}"
      + ("" if all_rep else "  [some pairs are replicates, not extensions]"))