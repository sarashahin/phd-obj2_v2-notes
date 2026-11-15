"""
test_09_saturation_rule_calibration.py  (obj1_v2, 26 Oct 2026)

Question: can a saturation rule read from ONE assembly history tell a saturated community
from one still growing, at the richness of the settled worlds (about 30 species)?

Rule tested (agreed 26 Oct, before this test): over the last 40% of rounds,
  trend    no trend in richness: slope of 8 block means, 95% interval includes 0
  balance  newcomers kept equal to residents removed within counting noise,
           |K - R| <= 2 sqrt(K + R)
Process: newcomers kept per round ~ Poisson(lam), each resident lost per round with
probability mu (immigration and loss). Saturated richness is lam / mu; a community
relaxes towards it over about 1 / mu rounds. Rates from the settle worlds: kept about
0.15-0.40 per round, about 30-50 species, about 450-550 rounds (test_05, 26 Oct).
A rule is usable only if it rarely fails saturated worlds (about 5%) AND usually fails
growing ones. Pure calculation, no files; about 15 seconds.
Run from obj1_v2/work:  python3 ../tests/test_09_saturation_rule_calibration.py
Result 26 Oct: saturated worlds fail the trend part 37-46% of the time (false alarms); the
balance part fails 0.1-0.9%, but growing worlds fail it only 0.5-2.3% and the trend part
44-51%, no more than saturated ones. Positive control: a world gaining about 26 species in
the segment (towards 150, relaxation 500 rounds) is caught by the trend part 97% and the
balance part 79% of the time, so the code can see real growth. Over about 500 rounds a 30-species world relaxes in 86-167 rounds, so the growth
left in the last 40% is a few species, the size of the fluctuation: one history cannot tell
them apart.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
N_SIM, ROUNDS = 1000, 520


def series(lam, mu, n, g0):
    g = np.empty(n, int); kept = np.empty(n, int); rem = np.empty(n, int); x = g0
    for t in range(n):
        k = rng.poisson(lam); r = rng.binomial(x, mu); x = x + k - r
        g[t], kept[t], rem[t] = x, k, r
    return g, kept, rem


def trend_fail(g, nb=8):
    seg = g[int(0.6 * len(g)):]; B = np.array_split(seg, nb)
    y = np.array([b.mean() for b in B]); start = np.cumsum([0] + [len(b) for b in B[:-1]])
    x = np.array([s + (len(b) - 1) / 2 for s, b in zip(start, B)])
    fit = stats.linregress(x, y)
    return abs(fit.slope) > stats.t.ppf(0.975, nb - 2) * fit.stderr


def balance_fail(kept, rem):
    s = slice(int(0.6 * len(kept)), None); K, R = kept[s].sum(), rem[s].sum()
    return abs(K - R) > 2 * np.sqrt(K + R)


def rate(lam, S, g0):
    mu = lam / S; ft = fb = 0
    for _ in range(N_SIM):
        g, k, r = series(lam, mu, ROUNDS, rng.poisson(S) if g0 is None else g0)
        ft += trend_fail(g); fb += balance_fail(k, r)
    return 100 * ft / N_SIM, 100 * fb / N_SIM, 1 / mu


for lam, S in ((0.25, 30), (0.35, 30), (0.30, 50)):
    ft, fb, tau = rate(lam, S, None)
    print(f"saturated, kept {lam}/round, richness {S}, relaxation {tau:.0f} rounds: trend part fails {ft:.1f}%, "
          f"balance part fails {fb:.1f}% (should be about 5% or less)")
for lam, S, g0 in ((0.30, 52, 30), (0.30, 52, 40), (0.30, 52, 5), (0.30, 150, 5)):   # last: positive control, +26 species in the segment
    ft, fb, tau = rate(lam, S, g0)
    print(f"still growing from {g0} towards {S} (relaxation {tau:.0f} rounds): trend part fails {ft:.1f}%, "
          f"balance part fails {fb:.1f}% (should be high)")