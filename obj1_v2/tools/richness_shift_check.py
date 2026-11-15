#!/usr/bin/env python3
"""
richness_shift_check.py  (obj1_v2, gate 1, 31 Oct 2026, Sara's yes)

Question: in a world whose whole habitat layer shifts (assembly v2.5), does richness level off and swing with the
shift, or does it keep falling or rising? n66 (best founding, habitat tau 62,500, shift sd 1.0) was in balance by
richness_balance_check (kept share 0.49, 0.45-0.54) but its richness fell from 50 to 34 over the last 20% of attempts.
Reads saved worlds only (round history and the saved shift); nothing is simulated; writes nothing.

Per world, over the last half of the rounds:
  trend   the slope of richness (ASM_gamma) against round, in species per 100 rounds and as the change over the half in %
          of its mean; 95% interval from 5 block means (about 46 rounds each in n66, close to 4 shift time scales), with
          a t interval on 3 degrees of freedom, because the swings make neighbouring rounds alike.
  shift   correlation of richness with the shift m and with |m| some rounds earlier (lag 0-30), the strongest of each;
          the verdict uses the slope with m and |m| at those lags held fixed (the trend beyond what the shift explains),
          because swings with the shift hide a trend (planted: a 45% fall over the half was found in 15 of 20 draws
          without this, see T2). Worlds without a shift use the plain slope.
Verdict per world: "no trend" (interval holds 0), "falling" or "rising".

Reading rule (written 31 Oct, before the run):
  a setting levels off when 3 or more of its 4 worlds read "no trend" and its pooled kept share holds 0.5
  (richness_balance_check); "falling" or "rising" in 2 or more worlds -> gate 1 is not closed for that setting.
  Calibration on Sara's worlds: n33, n57, n58 and n59 grew 11-17% over the last 20% of attempts; they must read
  "rising", or the check lacks power and is not read.
  Expected: n66-n69 "no trend", richness swinging with the shift.

Self-test (--self-test, planted series with a shift like n66's, 460 rounds): T1 a stationary series swinging with the
shift reads "no trend" in 33 or more of 40 draws (measured over 300 draws: 7% read a trend); T2 a fall of 0.08 species
per round (about 45% over the half) reads "falling" in 36 or more of 40 (measured 98%); T3 a rise of 0.08 per round
reads "rising" in 36 or more; T4 the planted lag (3 rounds) and sign of the shift
effect are found; T5 the slope equals numpy's fit to the block means; T6 a world without a shift gets a trend and no shift
line.

Run from obj1_v2/work:
    python3 ../tools/richness_shift_check.py --self-test | tee ../logs/richness_shift_selftest_31oct.txt | tail -1   # "6 PASS, 0 FAIL"
    python3 ../tools/richness_shift_check.py "results/stage2a/niche_b/*_training.npz" "results/stage2a/niche_t/*_fb_*_training.npz" "results/stage2a/niche_t2/*_training.npz" "results/stage2a/niche_h/*_fb_*_training.npz" "results/stage2a/niche_h2/*_training.npz" | tee ../logs/richness_shift_31oct.txt
"""
import glob, sys
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist

LAGS = range(0, 31)
K_BLOCKS = 5


def scalar(z, k, d=None):
    if k not in z.files:
        return d
    v = z[k]
    return v.item() if v.ndim == 0 else (v[0].item() if hasattr(v[0], "item") else v[0])


def trend(S, K=K_BLOCKS, extra=None):
    """Slope of S against its index with a 95% interval, from block means. The columns of extra (the shift terms) are
    fitted together with a straight line and their part is taken out; the rest is averaged over K equal blocks, and the
    block means are regressed on block time, with a t interval on K - 2 degrees of freedom. Block means of 40-50 rounds
    are close to independent; a bootstrap of single-round residuals was too narrow (planted series without a trend read
    'trend' in 13-29% of draws), the block means 6-7%."""
    S = np.asarray(S, float); n = len(S); t = np.arange(n, dtype=float)
    X = np.c_[np.ones(n), t] if extra is None else np.c_[np.ones(n), t, np.asarray(extra, float).reshape(n, -1)]
    beta = np.linalg.lstsq(X, S, rcond=None)[0]
    part = S - (X[:, 2:] @ beta[2:] if extra is not None else 0.0)
    edges = np.linspace(0, n, K + 1).astype(int)
    bm = np.array([part[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    bt = np.array([t[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    A = np.c_[np.ones(K), bt]
    c = np.linalg.lstsq(A, bm, rcond=None)[0]
    res = bm - A @ c
    se = float(np.sqrt(res @ res / (K - 2) / ((bt - bt.mean()) ** 2).sum()))
    q = float(tdist.ppf(0.975, K - 2))
    return float(c[1]), float(c[1] - q * se), float(c[1] + q * se)


def lag_corr(S, m, h):
    """(lag, r) of the strongest correlation of S[i] (round h + i) with m at round h + i - lag, lag 0-30; m is the
    shift of every round, so earlier rounds are used for the lag."""
    best = (0, 0.0)
    for L in LAGS:
        if h - L < 0:
            break
        a, b = S, m[h - L:h - L + len(S)]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if abs(r) > abs(best[1]):
            best = (L, r)
    return best


def analyse(z, rng):
    if "ASM_gamma" not in z.files or "ASM_round" not in z.files:
        return None, "missing ASM_gamma or ASM_round"
    S = z["ASM_gamma"].astype(float); n = len(S)
    if n < 80:
        return None, f"only {n} rounds (needs 80)"
    h = n // 2
    Sh = S[h:]
    raw = trend(Sh)
    out = dict(n=n, half=len(Sh), mean=float(Sh.mean()), raw=raw, B=len(Sh) // K_BLOCKS,
               s80=float(S[int(0.8 * (n - 1))]), send=float(S[-1]), adjusted=False)
    slope, lo, hi = raw
    if "ENV_habitat_shift" in z.files and len(z["ENV_habitat_shift"]) == n:
        m = z["ENV_habitat_shift"].astype(float)
        out["m"] = lag_corr(Sh, m, h); out["am"] = lag_corr(Sh, np.abs(m), h)
        Lm, La = out["m"][0], out["am"][0]
        extra = np.c_[m[h - Lm:h - Lm + len(Sh)], np.abs(m)[h - La:h - La + len(Sh)]]
        slope, lo, hi = trend(Sh, extra=extra)                  # the trend beyond what the shift explains
        out["adjusted"] = True
    out.update(slope=slope, lo=lo, hi=hi, change=100 * slope * (len(Sh) - 1) / Sh.mean() if Sh.mean() > 0 else np.nan,
               verdict="no trend" if lo <= 0 <= hi else ("falling" if hi < 0 else "rising"))
    return out, None


def rule_of(z, f):
    fr = scalar(z, "FOUND_rule")
    return str(fr) if fr is not None else ("best" if "_fb" in Path(f).name else "random")


def report(files):
    rng = np.random.default_rng(31)
    groups = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        o, why = analyse(z, rng)
        short = f"{Path(f).parent.name}/...{Path(f).name[-62:]}"
        if o is None:
            print(f"{short}: not read ({why})"); continue
        tau = int(scalar(z, "HABITAT_tau", -1)); sh = float(scalar(z, "HABITAT_shift_sd", 0.0))
        r = o["raw"]
        print(f"{short}\n  {o['n']} rounds; richness at 80% of rounds {o['s80']:.0f}, at the end {o['send']:.0f}; last half mean "
              f"{o['mean']:.1f}; slope per 100 rounds {100 * r[0]:+.2f} ({100 * r[1]:+.2f} to {100 * r[2]:+.2f}), {K_BLOCKS} blocks of about {o['B']} rounds")
        if "m" in o:
            print(f"  shift: strongest correlation with m {o['m'][1]:+.2f} at lag {o['m'][0]} rounds, with |m| {o['am'][1]:+.2f} at lag "
                  f"{o['am'][0]}; slope with the shift held fixed {100 * o['slope']:+.2f} ({100 * o['lo']:+.2f} to {100 * o['hi']:+.2f})")
        print(f"  change over the half {o['change']:+.0f}% of the mean  ->  {o['verdict']}")
        key = (rule_of(z, f), tau, sh)
        groups.setdefault(key, []).append(o["verdict"])
    if groups:
        print("\nsettings (founding rule, habitat tau, shift sd): worlds, verdicts")
        for (rule, tau, sh), vs in sorted(groups.items()):
            n_ok = sum(v == "no trend" for v in vs)
            lab = f"{rule}, {'fixed' if tau < 0 else tau}" + (f", shift sd {sh:g}" if sh > 0 else "")
            read = ("levels off (with the kept share holding 0.5)" if len(vs) >= 4 and n_ok >= 3 else
                    "not closed" if sum(v != "no trend" for v in vs) >= 2 else "too few worlds to read")
            print(f"  {lab}: {len(vs)} worlds; {', '.join(vs)}  ->  {read}")


# ------------------------------------------------------------------ self-test
def planted(rng, *, n=460, drift=0.0, lag=3, coef=6.0, rho_m=np.exp(-1 / 12.5)):
    m = np.zeros(n)
    for t in range(1, n):
        m[t] = rho_m * m[t - 1] + np.sqrt(1 - rho_m ** 2) * rng.normal()
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + 2.0 * rng.normal()
    S = 40 + coef * np.r_[np.zeros(lag), m[:n - lag]] + e
    S[n // 2:] += drift * np.arange(n - n // 2)
    return np.clip(np.round(S), 1, None), m


def self_test():
    rng = np.random.default_rng(31)
    res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    class Z(dict):
        files = property(lambda self: list(self.keys()))

    def world(S, m=None):
        d = dict(ASM_round=np.arange(len(S)), ASM_gamma=np.asarray(S), window_steps=np.int64(5000))
        if m is not None:
            d.update(ENV_habitat_shift=np.asarray(m), HABITAT_shift_tau=np.int64(62500))
        return Z({k: np.asarray(v) for k, v in d.items()})

    def verdicts(drift):
        out = []
        for _ in range(40):
            S, m = planted(rng, drift=drift)
            o, _ = analyse(world(S, m), np.random.default_rng(rng.integers(1 << 30)))
            out.append(o["verdict"])
        return out
    v0 = verdicts(0.0); v1 = verdicts(-0.08); v2 = verdicts(0.08)
    check("T1", v0.count("no trend") >= 33, f"stationary series swinging with the shift: 'no trend' in {v0.count('no trend')} of 40")
    check("T2", v1.count("falling") >= 36, f"fall of 0.08 species per round: 'falling' in {v1.count('falling')} of 40")
    check("T3", v2.count("rising") >= 36, f"rise of 0.08 species per round: 'rising' in {v2.count('rising')} of 40")
    S, m = planted(rng, coef=6.0, lag=3)
    L, r = lag_corr(S[230:], m, 230)
    check("T4", abs(L - 3) <= 1 and r > 0.3, f"planted shift effect +6 at lag 3: found lag {L}, correlation {r:+.2f}")
    Sh = S[230:]
    s, lo, hi = trend(Sh)
    edges = np.linspace(0, len(Sh), K_BLOCKS + 1).astype(int)
    ref = float(np.polyfit([np.arange(a, b).mean() for a, b in zip(edges[:-1], edges[1:])],
                           [Sh[a:b].mean() for a, b in zip(edges[:-1], edges[1:])], 1)[0])
    check("T5", abs(s - ref) < 1e-10 and lo <= s <= hi, f"slope {s:.6f} = numpy fit to the {K_BLOCKS} block means {ref:.6f}, inside its interval")

    o, why = analyse(world(np.round(40 + np.random.default_rng(1).normal(0, 2, 300))), rng)
    check("T6", o is not None and "m" not in o and not o["adjusted"], f"world without a shift: trend read ({o['verdict'] if o else why}), no shift line")
    print(f"\n{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    files = sorted({f for pat in sys.argv[1:] for f in glob.glob(pat)})
    if not files:
        sys.exit("no files: give one or more quoted patterns of _training.npz files")
    report(files)