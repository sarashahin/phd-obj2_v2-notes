#!/usr/bin/env python3
"""
richness_balance_check.py  (obj1_v2, gate 1 and root cause 13d, 30 Oct 2026, Sara's yes)

Question: at what richness would each saved world level off, if assembly went on at the same rates? Gate 1 (richness
levels off) must pass before the real run, and the answer decides which founding rule the next fix starts from.
Best founding keeps 2.0-2.5 species per round and loses 0.5-0.6 late on (niche_w, 30 Oct), so it is still growing;
random founding is closer to a balance.

Reads saved training files only, through effort_check.py (same reading and the same bookkeeping checks C1-C4; a world
that fails any of them is reported and left out). Files saved before the life table (assembly v1, e.g. the settle
worlds) are read from the round history alone, with C1 and C4 where the keys exist (version 2, 30 Oct). Settings are
grouped by folder and file name with the seed parts replaced by * (version 2; version 1 grouped by founding rule,
strength and width only, which pooled n33 with n40, whose first 206 rounds are the same run). Nothing is simulated;
writes nothing.

Chain mode (--chain, version 2, Sara's yes 30 Oct): each pattern is one link of a chain of twin plans, in order, each
link changing one thing. Per link, the last halves of its worlds are pooled: kept, removed, kept share = kept /
(kept + removed) with a 95% Wilson interval, losses per 1,000 residents per round, kept per entered.
Why: in all 17 niche worlds about three newcomers are kept per resident lost (kept share 0.75-0.77), while in the
field-mode settle worlds of 27 Oct (README sketch 15) gains and losses balanced. The chain shows where that changed.
Reading rule (written before the run, 30 Oct): balanced = interval holds 0.5; broken = interval above 0.5 and share
2/3 or more. The first place where a balanced link is followed by a broken one is the change that made residents
outlast newcomers. If the first link is not balanced, the lead does not hold on these files; if no single link breaks
it, the two largest rises are reported. The kept share is used rather than losses per resident, because losses per
resident also move with how many newcomers get in. Expected (written now): the break is at settle -> niche (n1-n4).

Per round: S = richness at the start of the round, newcomers entered, kept (passed their first window) and removed.
Both founding rules bring in 1 + floor(0.05 S) newcomers per round, so gains grow with S in steps of one newcomer per
20 species; a world can level off only if losses grow faster.
  rates    over the last half of the run (the final round, which may be capped, left out): kept / removed per round;
           f = kept per entered; mu = removed per resident per round.
  ceiling  at these rates the expected change per round is E(S) = (1 + floor(0.05 S)) f (1 - mu) - mu S. The ceiling
           is one above the largest S with E(S) > 0: above it gains never beat losses. Below it a world can sit for a
           while at a lower level held by the newcomer step (E falls below 0 inside a band of 20 species and jumps up
           at the next band). No ceiling if E is still positive at 20,000 species.
  95% intervals from 1,000 moving-block bootstrap resamples of the rounds (blocks of 20 rounds).
  table    five blocks of rounds: mean S, entered, kept and removed per round, kept per entered, kept / removed.

Verdict per world (S_end = richness at the end):
  kept / removed above 1 (interval)       -> growing: "no level at today's rates" (no ceiling) or "toward a level above"
  kept / removed below 1 (interval)       -> falling toward its level
  balanced, ceiling interval above 1.25 S_end -> balanced now, but today's rates lead higher (held by the newcomer step)
  balanced, ceiling interval below 0.75 S_end -> falling toward its level
  otherwise                               -> at its level now

Reading rule (written before the run, 30 Oct), per setting (3 of 4 worlds, or all if fewer than 4):
  random founding "at its level now"      -> gate 1 is within reach for random founding (a longer run confirms);
                                             the rare-species fix starts from random founding (cause 13)
  best founding "growing"                 -> still growing above a richness already several times BNM's: best
                                             founding cannot be the base of the fix
  anything else                           -> reported as it is, discussed before any change
Expected (written now): random founding at its level near 27-33 species; best founding growing, no level near.

Calibration (self-test; worlds made by the counting process of the assembly rule, as effort_check.plant):
  T1  20 random-like worlds at a ceiling of 26: "at its level now" in 17 or more; interval holds 26 in 18 or more
  T2  20 random-like worlds with no ceiling (growing): no ceiling found in 18 or more; "at its level now" in 1 or fewer
  T3  20 best-like growing worlds (kept per entered 19%, removed 2.5 per 1,000 per round, 200 rounds, the rates of
      niche_w): "growing, no level at today's rates" in 19 or more
  T4  rounds() on a file made by effort_check.plant: each round starts where the last ended; final round dropped
  T5  a planted bookkeeping error (a removal not counted): the world is left out with a FAIL line
  T6  every branch of the verdict and of the reading rule
  T7  a file without life table, ASM_pruned, attempts_total and gamma (as assembly v1) gives the same rounds as the
      full file, with removed derived from the history
  T8  setting key: two seeds of one setting share it; another contrast or another folder does not
  T9  every branch of the chain reading;  T10  a planted chain (balanced, then growing, then no files): break at link 2
Broken versions caught: removed derived with the wrong sign (T7), seed left in the key (T8), "broken" read as above
0.5 only (T9), links read in reverse (T9, T10); and from version 1: mu per round, newcomer rule ignored, ratio
inverted, first half of the run.
  D1, D2 (descriptions of power, no pass line): near 30 species a world held by the newcomer step cannot be told
      from one at its ceiling, so for random founding the verdict is a screen and a longer run (test_10) decides;
      a best-like world at its ceiling is not called growing.

Run from obj1_v2/work (needs ../tools/effort_check.py, 29 Oct):
    python3 ../tools/richness_balance_check.py --self-test | tee ../logs/richness_balance_selftest_30oct.txt
    python3 ../tools/richness_balance_check.py "results/stage2a/niche_d/*_training.npz" "results/stage2a/niche_b/*_training.npz" "results/stage2a/niche_s/*_training.npz" "results/stage2a/niche_w/*_training.npz" | tee ../logs/richness_balance_30oct.txt
    python3 ../tools/richness_balance_check.py --chain "results/stage2a/settle/*_s230[3478]_training.npz" "results/stage2a/niche/*_training.npz" "results/stage2a/niche_k/*_training.npz" "results/stage2a/niche_c/*hc0p5_*_training.npz" "results/stage2a/niche_c/*hc0p25_*_training.npz" "results/stage2a/niche_f/*ls3p0*_cn1p0_*_training.npz" "results/stage2a/niche_d/*_training.npz" "results/stage2a/niche_b/*_training.npz" | tee ../logs/richness_chain_30oct.txt
(each command is one line; paste it as it is)
Chain links: settle s3, s4, s7, s8 (field mode) -> niche n1-n4 -> k 0.5 n5-n8 -> contrast 0.5 n9-n12 -> contrast 0.25
n13-n16 -> connectance 1 n21-n24 -> d per species n29-n32 -> best founding n33. Link 1 must list 4 files.
"""
import argparse, contextlib, glob, importlib.util, io, os, re, sys, tempfile
from pathlib import Path
import numpy as np

_spec = importlib.util.spec_from_file_location("effort_check", Path(__file__).with_name("effort_check.py"))
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)
for _f in ("read", "checks", "plant", "scalar", "rule_of", "NEED"):
    if not hasattr(ec, _f):
        sys.exit(f"STOP: ../tools/effort_check.py has no {_f}(); take the 29 Oct version from the project")

N_BOOT, BLOCK = 1000, 20


def rounds(w):
    """Per round: S at the start, entered, kept, removed. The final round (may be capped) is dropped."""
    G = w["ASM_gamma"]; start = 0 if w["rule"] == "best" else 1
    S0 = np.r_[start, G[:-1]].astype(float)
    ent = np.diff(np.r_[0, w["ASM_attempts"]]).astype(float)
    kept = np.diff(np.r_[0, w["ASM_established"]]).astype(float)
    rem = w["ASM_pruned"].astype(float)
    return S0[:-1], ent[:-1], kept[:-1], rem[:-1]


S_GRID = np.arange(0, 20001)
N_GRID = 1 + np.floor(0.05 * S_GRID)


def ceiling(f, mu):
    """Highest richness the expected change can reach at these rates. Expected change per round at S:
    E(S) = (1 + floor(0.05 S)) f (1 - mu) - mu S. The ceiling is one above the largest S with E(S) > 0; above it
    gains never beat losses. Below it the newcomer steps can hold a world for a while (E falls below 0 inside a
    band of 20 species and jumps up at the next band). None if E is still positive at 20,000 species."""
    if mu <= 0:
        return np.inf
    E = N_GRID * f * (1 - mu) - mu * S_GRID
    pos = np.nonzero(E > 0)[0]
    if not len(pos):
        return 0.0
    return np.inf if pos[-1] >= S_GRID[-1] - 20 else float(pos[-1] + 1)


def stats(x, e, k, r):
    """kept / removed per round, f = kept per entered, mu = removed per resident per round, and the ceiling."""
    f = k.sum() / max(e.sum(), 1e-9); mu = r.sum() / max((x + k).sum(), 1e-9)
    return dict(ratio=(k.sum() / r.sum() if r.sum() > 0 else np.inf), f=f, mu=mu, s_inf=ceiling(f, mu),
                k_sum=float(k.sum()), r_sum=float(r.sum()), e_sum=float(e.sum()), x_sum=float((x + k).sum()))


def analyse(S0, ent, kept, rem, frac=0.5, seed=0):
    """Last frac of the rounds, with a moving-block bootstrap (blocks of BLOCK rounds)."""
    n = len(S0); k0 = max(n - max(int(frac * n), 2 * BLOCK), 0)
    x, e, k, r = S0[k0:], ent[k0:], kept[k0:], rem[k0:]
    A = stats(x, e, k, r)
    rng = np.random.default_rng(seed); m = len(x); L = min(BLOCK, m); nb = int(np.ceil(m / L))
    B = []
    for _ in range(N_BOOT):
        idx = (rng.integers(0, m - L + 1, nb)[:, None] + np.arange(L)).ravel()[:m]
        B.append(stats(x[idx], e[idx], k[idx], r[idx]))
    q = lambda key, p, meth: float(np.quantile([d[key] for d in B], p, method=meth))
    A.update(n=m, r_lo=q("ratio", 0.025, "lower"), r_hi=q("ratio", 0.975, "higher"),
             s_lo=q("s_inf", 0.025, "lower"), s_hi=q("s_inf", 0.975, "higher"))
    return A


def verdict(A, s_end):
    if A["r_lo"] > 1:
        return "growing, no level at today's rates" if not np.isfinite(A["s_inf"]) else "growing toward a level above today"
    if A["r_hi"] < 1:
        return "falling toward its level"
    if A["s_lo"] > 1.25 * s_end:
        return "balanced now, but today's rates lead higher (held by the newcomer step)"
    if A["s_hi"] < 0.75 * s_end:
        return "falling toward its level"
    return "at its level now"


LT_KEYS = ("LT_id", "LT_round_in", "LT_established")


def load_world(f):
    """(world, check lines) or (None, reason). Files with a life table go through effort_check (checks C1-C4);
    files saved before it (assembly v1, e.g. the settle worlds) are read from the round history alone: C1 when
    attempts_total and gamma are saved, C4 when ASM_pruned is saved; otherwise removed is derived from the history
    (previous + kept - richness), as test_05 does, and C4 cannot be a check."""
    z = np.load(f, allow_pickle=True); keys = set(z.files)
    if all(k in keys for k in ec.NEED) and all(k in keys for k in LT_KEYS):
        w = ec.read(f); w["light"] = False
        return w, ec.checks(w)
    miss = [k for k in ("ASM_attempts", "ASM_gamma", "ASM_established") if k not in keys]
    if miss:
        return None, f"lacks {miss}"
    w = dict(path=f, name=os.path.basename(f), rule=ec.rule_of(z), seed=int(ec.scalar(z, "SIM_SEED", -1)),
             strength=float(ec.scalar(z, "INTERACTION_STRENGTH", 0.4)), attempts=int(ec.scalar(z, "attempts_total", -1)),
             gamma=int(ec.scalar(z, "gamma", -1)), light=True, resumed=int(ec.scalar(z, "ASM_resumed_from_round", -1)) >= 0)
    for k in ("ASM_attempts", "ASM_gamma", "ASM_established"):
        w[k] = np.asarray(z[k]).astype(np.int64).ravel()
    G, E = w["ASM_gamma"], w["ASM_established"]
    prev = np.r_[0 if w["rule"] == "best" else 1, G[:-1]]
    derived = "ASM_pruned" not in keys
    w["ASM_pruned"] = prev + np.diff(np.r_[0, E]) - G if derived else np.asarray(z["ASM_pruned"]).astype(np.int64).ravel()
    res = []
    if w["attempts"] >= 0 and w["gamma"] >= 0:
        res.append(("C1", int(w["ASM_attempts"][-1]) == w["attempts"] and int(G[-1]) == w["gamma"],
                    f"last attempts {int(w['ASM_attempts'][-1])} = attempts_total {w['attempts']}; last richness {int(G[-1])} = gamma {w['gamma']}"))
    if derived:
        res.append(("C4", bool((w["ASM_pruned"] >= 0).all()), "removed derived from the history (no ASM_pruned): none negative"))
    else:
        res.append(("C4", bool(np.all(G == prev + np.diff(np.r_[0, E]) - w["ASM_pruned"])) and not w["resumed"],
                    "richness each round = previous + kept - removed"))
    return w, res


def setting_key(w):
    """Folder and file name with the seed-dependent parts replaced by *, so worlds of one setting share a key."""
    nm = os.path.basename(w["path"])
    for pat, rep_ in ((r"env\d+", "env*"), (r"_dr\d+em\d+", "_dr*"), (r"nicheh\d+", "nicheh*"), (r"_s\d+_training", "_s*_training")):
        nm = re.sub(pat, rep_, nm)
    return f"{Path(w['path']).parent.name}/{nm}"


def setting_of(w):
    z = np.load(w["path"], allow_pickle=True)
    wm = float(ec.scalar(z, "NICHE_w_min", 0.2))
    parts = [f"{w['rule']} founding"]
    if abs(w["strength"] - 0.4) > 1e-6:
        parts.append(f"strength {w['strength']:g}")
    if abs(wm - 0.2) > 1e-6:
        parts.append(f"w_min {wm:g}")
    return ", ".join(parts)


def rule_reading(rule, verdicts):
    need = min(3, len(verdicts))
    if rule == "random" and sum(v == "at its level now" for v in verdicts) >= need:
        return ("at its level: gate 1 is within reach for random founding (a longer run confirms); "
                "the rare-species fix starts from random founding (cause 13)")
    if rule == "best" and sum(v.startswith("growing") for v in verdicts) >= need:
        return ("still growing above a richness already several times BNM's: "
                "best founding cannot be the base of the fix")
    return "neither branch of the rule: reported as it is, to discuss before any change"


def wilson(k, n, z=1.96):
    """95% Wilson interval for a share k / n."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt_level(A):
    f = lambda v: "none" if not np.isfinite(v) else f"{v:.0f}"
    return f"{f(A['s_inf'])} ({f(A['s_lo'])}-{f(A['s_hi'])})"


def describe(A, v):
    return (f"kept / removed per round {A['ratio']:.2f} ({A['r_lo']:.2f}-{A['r_hi']:.2f}); kept per entered {A['f']:.1%}; "
            f"removed per resident {1000 * A['mu']:.2f} per 1,000 per round; ceiling at these rates {fmt_level(A)} "
            f"-> {v}")


def report(files, out=print, summary=True):
    rows = []
    for f in files:
        w, res = load_world(f)
        if w is None:
            out(f"\n{os.path.basename(f)}: left out, {res}")
            continue
        bad = [f"{n} {t}" for n, g, t in res if not g]
        if bad:
            out(f"\n{w['name']}: bookkeeping FAIL, left out: {'; '.join(bad)}")
            continue
        S0, ent, kept, rem = rounds(w)
        if len(S0) < 4 * BLOCK:
            out(f"\n{w['name']}: only {len(S0)} rounds, too few (needs {4 * BLOCK}); left out")
            continue
        A = analyse(S0, ent, kept, rem)
        s_end = int(w["ASM_gamma"][-1]); v = verdict(A, s_end); st = setting_of(w); key = setting_key(w)
        rows.append(dict(key=key, label=st, rule=w["rule"], v=v, seed=w["seed"], s_end=s_end, **{q: A[q] for q in ("k_sum", "r_sum", "e_sum", "x_sum")}))
        note = "; no life table: read from the round history" + (", removed derived" if "ASM_pruned" not in np.load(f, allow_pickle=True).files else "") if w["light"] else ""
        out(f"\n{w['name']}  ({st}, seed {w['seed']}{note}): {len(w['ASM_gamma'])} rounds, {s_end} species at the end")
        out(f"  last half ({A['n']} rounds): {describe(A, v)}")
        out("  block     mean S   entered/round   kept/round   removed/round   kept per entered   kept / removed")
        for i, idx in enumerate(np.array_split(np.arange(len(S0)), 5)):
            e, k, r = ent[idx].mean(), kept[idx].mean(), rem[idx].mean()
            out(f"  {i + 1}/5    {S0[idx].mean():8.1f}   {e:13.2f}   {k:10.2f}   {r:13.2f}   {k / max(e, 1e-9):16.1%}   "
                f"{(k / r if r > 0 else np.inf):14.2f}")
    if summary:
        out("\nsettings (folder and file name, seed parts as *; reading rule written before the run, 30 Oct)")
        for key in sorted({r["key"] for r in rows}):
            g = [r for r in rows if r["key"] == key]; vs = [r["v"] for r in g]
            counts = ", ".join(f"{vs.count(x)} {x}" for x in dict.fromkeys(vs))
            P = pool(g)
            out(f"  {key}\n    {g[0]['label']}: {len(g)} world{'s' if len(g) != 1 else ''}: {counts}\n"
                f"    pooled last halves: {P['K']:.0f} kept, {P['R']:.0f} removed; kept share {P['share']:.2f} "
                f"(95% {P['lo']:.2f}-{P['hi']:.2f}; above 0.5 = gains beat losses)\n    -> {rule_reading(g[0]['rule'], vs)}")
    return rows


def pool(g):
    K = sum(r["k_sum"] for r in g); R = sum(r["r_sum"] for r in g)
    E = sum(r["e_sum"] for r in g); X = sum(r["x_sum"] for r in g)
    lo, hi = wilson(K, K + R)
    return dict(K=K, R=R, share=K / max(K + R, 1e-9), lo=lo, hi=hi, mu=R / max(X, 1e-9), f=K / max(E, 1e-9), n=len(g))


def chain_reading(links):
    """links: list of (name, pooled dict or None), in chain order. Reading rule written before the run (30 Oct)."""
    L = [(n, P) for n, P in links if P is not None and P["K"] + P["R"] > 0]
    if len(L) < 2:
        return "fewer than two links with events: no reading"
    bal = lambda P: P["lo"] <= 0.5 <= P["hi"]
    brk = lambda P: P["lo"] > 0.5 and P["share"] >= 2 / 3
    if not bal(L[0][1]):
        return (f"the first link ({L[0][0]}) is not balanced (kept share {L[0][1]['share']:.2f}): the lead from README "
                "sketch 15 does not hold on these files; discuss")
    for (n0, P0), (n1, P1) in zip(L, L[1:]):
        if bal(P0) and brk(P1):
            return (f"balance broken between {n0} and {n1} (kept share {P0['share']:.2f} -> {P1['share']:.2f}; losses per "
                    f"1,000 residents per round {1000 * P0['mu']:.2f} -> {1000 * P1['mu']:.2f}): the change made at this link "
                    "is where residents began to outlast newcomers")
    rises = sorted(((P1["share"] - P0["share"], n0, n1) for (n0, P0), (n1, P1) in zip(L, L[1:])), reverse=True)[:2]
    return ("no single link breaks the balance; largest rises in kept share: "
            + "; ".join(f"{a} -> {b} ({d:+.2f})" for d, a, b in rises))


def chain(patterns, out=print):
    links = []
    for i, pat in enumerate(patterns, 1):
        files = sorted(glob.glob(pat))
        out(f"\n=== link {i}: {pat} ({len(files)} files)")
        if not files:
            links.append((f"link {i}", None)); continue
        rows = report(files, out=out, summary=False)
        links.append((f"link {i}", pool(rows) if rows else None))
        if rows:
            out("  per world (seed: kept share of last-half events): " +
                ", ".join(f"{r['seed']}: {r['k_sum']:.0f}/{r['k_sum'] + r['r_sum']:.0f}" for r in rows))
    out("\nchain (pooled last halves of each link; kept share = kept / (kept + removed))")
    out("  link   worlds   kept   removed   kept share (95%)     lost per 1,000 residents per round   kept per entered   pattern")
    for (n, P), pat in zip(links, patterns):
        if P is None:
            out(f"  {n:6s}  no files or no usable world   {pat}")
        else:
            out(f"  {n:6s} {P['n']:6d}   {P['K']:4.0f}   {P['R']:7.0f}   {P['share']:.2f} ({P['lo']:.2f}-{P['hi']:.2f})   "
                f"{1000 * P['mu']:34.2f}   {P['f']:16.1%}   {pat}")
    out("\nreading rule (written before the run, 30 Oct): balanced = interval holds 0.5; broken = interval above 0.5 and share "
        "2/3 or more; the first link where a balanced link is followed by a broken one is the change that made residents "
        "outlast newcomers")
    out(f"  -> {chain_reading(links)}")
    return links


# ---------------------------------------------------------------- self-test
def sim(f, mu, s_init, n_rounds, seed):
    """The assembly rule as a counting process: 1 + floor(0.05 S) newcomers, each kept with chance f, then each
    resident removed with chance mu (the order of effort_check.plant). Returns S at the start of each round, entered,
    kept, removed."""
    rng = np.random.default_rng(seed); S = s_init; out = []
    for _ in range(n_rounds):
        n = 1 + int(0.05 * S); k = int(rng.binomial(n, f)); r = int(rng.binomial(S + k, mu))
        out.append((S, n, k, r)); S = S + k - r
    return [np.array(c, float) for c in zip(*out)]


def self_test():
    ok = True

    def check(name, cond, text):
        nonlocal ok
        ok &= bool(cond)
        print(f"{name} {text}: {'PASS' if cond else 'FAIL'}")

    def one(f, mu, s0, R, seed):
        S0, e, k, r = sim(f, mu, s0, R, seed)
        A = analyse(S0, e, k, r, seed=seed)
        s_end = int(S0[-1] + k[-1] - r[-1])
        return A, s_end, verdict(A, s_end)

    f1, mu1 = 0.15, 0.15 / 13.15                                          # c = f (1 - mu) / mu = 13: ceiling 26
    L1 = ceiling(f1, mu1)
    res = [one(f1, mu1, int(L1), 500, 100 + i) for i in range(20)]
    n_at = sum(v == "at its level now" for _, _, v in res); cov = sum(A["s_lo"] <= L1 <= A["s_hi"] for A, _, _ in res)
    wid = np.median([A["s_hi"] / max(A["s_lo"], 1) for A, _, _ in res])
    check("T1", n_at >= 17 and cov >= 18, f"20 random-like worlds at a ceiling of {L1:.0f} species (500 rounds): 'at its level now' "
          f"in {n_at}; interval holds {L1:.0f} in {cov}; median interval top / bottom x{wid:.1f}")
    res = [one(0.2, 0.008, 20, 500, 300 + i) for i in range(20)]
    n_none = sum(not np.isfinite(A["s_inf"]) for A, _, _ in res); wrong = sum(v == "at its level now" for _, _, v in res)
    check("T2", n_none >= 18 and wrong <= 1,
          f"20 replicates of random-like growth (no ceiling; ends at {min(s for _, s, _ in res)}-{max(s for _, s, _ in res)} "
          f"species): no ceiling found in {n_none}; 'at its level now' in {wrong}")
    res = [one(0.19, 0.0025, 120, 200, 400 + i) for i in range(20)]
    n_g = sum(v == "growing, no level at today's rates" for _, _, v in res)
    check("T3", n_g >= 19, f"20 replicates of best-like growth (kept per entered 19%, removed 2.5 per 1,000 per round, "
          f"200 rounds): 'growing, no level at today's rates' in {n_g}")
    f3, mu3 = 0.15, 0.15 / 15.15                                          # c = 15: ceiling 45; held at 30 by the step
    L3 = ceiling(f3, mu3)
    res = [one(f3, mu3, 30, 300, 200 + i) for i in range(20)]
    low = [v for A, s, v in res if s < 40]
    print(f"D1 power, a description: 20 replicates held at about 30 by the newcomer step under a ceiling of {L3:.0f} "
          f"(300 rounds): {len(low)} still below 40 at the end; of those, called 'at its level now' in "
          f"{low.count('at its level now')} and 'held by the newcomer step' in {sum(v.startswith('balanced now') for v in low)}. "
          f"Near 30 species the rates cannot tell a step from the ceiling; a longer run (test_10) decides")
    L6 = ceiling(0.19, 0.0105)
    res = [one(0.19, 0.0105, int(L6), 200, 500 + i) for i in range(20)]
    print(f"D2 power, a description: 20 replicates of a best-like world at its ceiling ({L6:.0f} species, 200 rounds): "
          f"interval holds it in {sum(A['s_lo'] <= L6 <= A['s_hi'] for A, _, _ in res)}; 'at its level now' in "
          f"{sum(v == 'at its level now' for _, _, v in res)}; 'growing' in {sum(v.startswith('growing') for _, _, v in res)}")
    with tempfile.TemporaryDirectory() as t:
        d = ec.plant(f"{t}/p.npz", "random", 2303, 1000, p_keep=0.1, p_loss=0.01, rng_seed=5)
        w = ec.read(f"{t}/p.npz"); S0, e, k, r = rounds(w)
        good = (np.array_equal(S0[1:], (S0 + k - r)[:-1]) and int(e.sum()) == int(d["ASM_attempts"][-2])
                and S0[0] == 1 and len(S0) == len(d["ASM_gamma"]) - 1)
        check("T4", good, "rounds() on a planted file: each round's start = previous start + kept - removed; final round dropped")
        np.savez(f"{t}/m.npz", **{**d, "ASM_pruned": np.r_[d["ASM_pruned"][:-1], d["ASM_pruned"][-1] + 1]})
        buf = []
        rows = report([f"{t}/m.npz"], out=buf.append)
        check("T5", not rows and any("bookkeeping FAIL, left out" in x for x in buf), "a removal not counted: world left out")
    base = dict(r_lo=0.8, r_hi=1.2, s_inf=60.0, s_lo=50.0, s_hi=70.0)
    t9 = (verdict(base, 60) == "at its level now"
          and verdict(dict(base, s_inf=100.0, s_lo=90.0, s_hi=120.0), 60).startswith("balanced now, but")
          and verdict(dict(base, s_inf=30.0, s_lo=20.0, s_hi=40.0), 60).startswith("falling")
          and verdict(dict(base, r_lo=1.1, s_inf=np.inf), 60) == "growing, no level at today's rates"
          and verdict(dict(base, r_lo=1.1), 40) == "growing toward a level above today"
          and verdict(dict(base, r_hi=0.9), 60).startswith("falling")
          and rule_reading("random", ["at its level now"] * 3 + ["growing toward a level above today"]).startswith("at its level")
          and rule_reading("best", ["growing, no level at today's rates"] * 4).startswith("still growing")
          and rule_reading("random", ["balanced now, but today's rates lead higher (held by the newcomer step)"] * 4).startswith("neither"))
    check("T6", t9, "every branch of the verdict and of the reading rule")
    with tempfile.TemporaryDirectory() as t:
        d = ec.plant(f"{t}/v2.npz", "random", 2303, 1000, p_keep=0.1, p_loss=0.01, rng_seed=7)
        old = {k: v for k, v in d.items() if not k.startswith("LT_") and k not in ("ASM_pruned", "attempts_total", "gamma")}
        np.savez(f"{t}/v1.npz", **old)
        w1, r1 = load_world(f"{t}/v1.npz"); w2, r2 = load_world(f"{t}/v2.npz")
        good = (w1["light"] and not w2["light"] and np.array_equal(w1["ASM_pruned"], w2["ASM_pruned"])
                and all(g for _, g, _ in r1) and all(g for _, g, _ in r2)
                and all(np.array_equal(a_, b_) for a_, b_ in zip(rounds(w1), rounds(w2))))
        check("T7", good, "a file without life table, ASM_pruned, attempts_total or gamma (as assembly v1) is read from the "
              "round history: removed derived equals the saved one, same rounds as the full file")
    names = ["results/stage2a/niche_d/pool22510000_s2a_lsna_vrna_thr1p0_env202303_grid20x20_dr1em06_ld0p0_nicheh102303ls3p0o1p5w0p2-1p0c0p1p1p5k0p5hc0p25_cn1p0_dps1em06-0p01_k2dta1p0b3p0_f1_s2303_training.npz",
             "results/stage2a/niche_d/pool22510000_s2a_lsna_vrna_thr1p0_env202308_grid20x20_dr3em06_ld0p0_nicheh102308ls3p0o1p5w0p2-1p0c0p1p1p5k0p5hc0p25_cn1p0_dps1em06-0p01_k2dta1p0b3p0_f1_s2308_training.npz",
             "results/stage2a/niche_c/pool22510000_s2a_lsna_vrna_thr1p0_env202303_grid20x20_dr1em06_ld0p0_nicheh102303ls3p0o1p5w0p2-1p0c0p1p1p5k0p5hc0p5_k2dta1p0b3p0_f1_s2303_training.npz",
             "results/stage2a/niche_c/pool22510000_s2a_lsna_vrna_thr1p0_env202303_grid20x20_dr1em06_ld0p0_nicheh102303ls3p0o1p5w0p2-1p0c0p1p1p5k0p5hc0p25_k2dta1p0b3p0_f1_s2303_training.npz"]
    k = [setting_key(dict(path=n)) for n in names]
    check("T8", k[0] == k[1] and len({k[0], k[2], k[3]}) == 3,
          "setting key: two seeds of one setting share it; another contrast or another folder does not")
    P = lambda K, R: dict(K=K, R=R, share=K / (K + R), lo=wilson(K, K + R)[0], hi=wilson(K, K + R)[1], mu=0.001, f=0.1)
    r1 = chain_reading([("a", P(50, 50)), ("b", P(52, 48)), ("c", P(80, 20)), ("d", P(90, 10))])
    r2 = chain_reading([("a", P(80, 20)), ("b", P(80, 20))])
    r3 = chain_reading([("a", P(50, 50)), ("b", P(58, 42)), ("c", P(62, 38))])
    r4 = chain_reading([("a", P(50, 50)), ("b", None)])
    check("T9", r1.startswith("balance broken between b and c") and r2.startswith("the first link (a) is not balanced")
          and r3.startswith("no single link") and r4.startswith("fewer than two"), "every branch of the chain reading")
    with tempfile.TemporaryDirectory() as t:
        for i in range(2):
            ec.plant(f"{t}/bal_{i}.npz", "random", 2303 + i, 1500, p_keep=0.3, p_loss=0.02, rng_seed=20 + i)
            ec.plant(f"{t}/grow_{i}.npz", "random", 2303 + i, 1500, p_keep=0.3, p_loss=0.005, rng_seed=30 + i)
        buf = []
        links = chain([f"{t}/bal_*.npz", f"{t}/grow_*.npz", f"{t}/none_*.npz"], out=buf.append)
        good = (any("balance broken between link 1 and link 2" in x for x in buf) and links[2][1] is None
                and links[0][1]["lo"] <= 0.5 <= links[0][1]["hi"] and links[1][1]["lo"] > 0.5)
        check("T10", good, f"planted chain (balanced worlds, then growing worlds, then a pattern with no files): kept share "
              f"{links[0][1]['share']:.2f} -> {links[1][1]['share']:.2f}, break found at link 2, empty link reported")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*")
    ap.add_argument("--chain", action="store_true", help="each pattern is one link of a chain of twin plans, in order")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.chain:
        chain(a.patterns)
    else:
        files = sorted({f for pat in a.patterns for f in glob.glob(pat)})
        if not files:
            sys.exit("STOP: no files matched")
        report(files)