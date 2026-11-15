"""
effort_check.py  (obj1_v2, 29 Oct 2026; read-only, no new run)

Question (Sara's yes, 29 Oct): does n33 hold 8 times more species than its twin n29 because best founding
packs more species into the same effort, or because it simply tried about 10 times more candidates
(LVMCM counting: only entered newcomers count toward the 1,000 attempts)? Neither world had stopped
growing, so the answer decides what the strength scan is compared against.

Reads saved training files only (assembly history ASM_*, life table LT_*, FOUND_*). For every file:
  C1-C4  bookkeeping checks: the history agrees with attempts_total, gamma, the life table and FOUND_*
         (PASS/FAIL; a FAIL means the file is not read as this tool assumes, so nothing below counts).
  effort candidates tested (best founding: ASM_tested; random founding: every attempt is a candidate),
         entered, kept (passed their first window), final species, kept per candidate.
  curve  richness at 250, 500, 1,000, 2,000, 5,000, 10,000 and 20,000 candidates, and at rounds 50, 100,
         200, 400 and 700 (equal model time: every round is one window).
  settle richness at 50 / 80 / 100% of attempts and the change over the last 20% (as test_05); kept and
         removed per round in the last 20% of rounds; best founding: pass rate (entered / tested) in the
         first and last 20% of rounds.
Pairs, matched by SIM_SEED:
  random against best founding: richness of the best-founding world at the candidate count where the
         random world ended, and at its last round (equal time).
  best against best with the same settings but more attempts (n33 against n40): the shorter history must
         equal the start of the longer one (the GPU repeats runs exactly); the first round that differs is shown.
Reading rule, written before the run (29 Oct): at the candidate count where n29 ended (1,000), if n33 holds
at most 1.5 times n29's species, the 8-fold gap is mostly extra tries; at 2 times or more, best founding itself
packs more species into the same tries; in between, both. A description, not a gate.

Run from obj1_v2/work:
    python3 ../tools/effort_check.py "results/stage2a/niche_b/*_training.npz" "results/stage2a/niche_d/*_training.npz" \\
        | tee ../logs/effort_check_29oct.txt
    python3 ../tools/effort_check.py --self-test
"""
import glob, os, sys, tempfile
import numpy as np

NEED = ("ASM_round", "ASM_attempts", "ASM_gamma", "ASM_established", "ASM_pruned", "LT_id", "LT_round_in",
        "LT_established", "attempts_total", "gamma", "SIM_SEED")
CANDS = (250, 500, 1000, 2000, 5000, 10000, 20000)
ROUNDS = (50, 100, 200, 400, 700)


def rule_of(z):
    return str(np.asarray(z["FOUND_rule"]).ravel()[0]) if "FOUND_rule" in z.files else "random"


def scalar(z, k, default):
    return np.asarray(z[k]).ravel()[0].item() if k in z.files else default


def read(path):
    z = np.load(path, allow_pickle=True)
    miss = [k for k in NEED if k not in z.files]
    if miss:
        sys.exit(f"STOP: {os.path.basename(path)} lacks {miss}")
    w = dict(path=path, name=os.path.basename(path), rule=rule_of(z), seed=int(scalar(z, "SIM_SEED", -1)),
             strength=float(scalar(z, "INTERACTION_STRENGTH", 0.4)), attempts=int(scalar(z, "attempts_total", -1)),
             gamma=int(scalar(z, "gamma", -1)))
    for k in ("ASM_round", "ASM_attempts", "ASM_gamma", "ASM_established", "ASM_pruned"):
        w[k] = np.asarray(z[k]).astype(np.int64).ravel()
    w["lt_in"] = np.asarray(z["LT_round_in"]).astype(np.int64)
    w["lt_est"] = np.asarray(z["LT_established"]).astype(np.int64)
    w["n_lt"] = len(np.asarray(z["LT_id"]))
    best = w["rule"] == "best"
    if best:
        for k in ("ASM_tested", "LT_entered", "FOUND_tested_total", "FOUND_saturated"):
            if k not in z.files:
                sys.exit(f"STOP: best-founding file {w['name']} lacks {k}")
        w["cand"] = np.asarray(z["ASM_tested"]).astype(np.int64).ravel()
        w["lt_ent"] = np.asarray(z["LT_entered"]).astype(np.int64)
        w["tested_total"] = int(scalar(z, "FOUND_tested_total", -1))
        w["saturated"] = int(scalar(z, "FOUND_saturated", 0))
    else:
        w["cand"] = w["ASM_attempts"]
    w["resumed"] = int(scalar(z, "ASM_resumed_from_round", -1)) >= 0
    return w


def checks(w):
    best = w["rule"] == "best"
    A, G, E, P = w["ASM_attempts"], w["ASM_gamma"], w["ASM_established"], w["ASM_pruned"]
    out = []
    out.append(("C1", int(A[-1]) == w["attempts"] and int(G[-1]) == w["gamma"],
                f"last attempts {int(A[-1])} = attempts_total {w['attempts']}; last richness {int(G[-1])} = gamma {w['gamma']}"))
    newc = w["lt_in"] >= 0
    if best:
        ent = w["lt_ent"] == 1
        good = (int(w["cand"][-1]) == w["tested_total"] == w["n_lt"] and int(ent.sum()) == w["attempts"])
        out.append(("C2", good, f"tested {int(w['cand'][-1])} = FOUND_tested_total {w['tested_total']} = life-table rows "
                                f"{w['n_lt']}; entered {int(ent.sum())} = attempts {w['attempts']}"))
        kept_lt = int((w["lt_est"][ent & newc] == 1).sum())
    else:
        good = int(newc.sum()) == w["attempts"]
        out.append(("C2", good, f"newcomer rows {int(newc.sum())} = attempts {w['attempts']}"))
        kept_lt = int((w["lt_est"][newc] == 1).sum())
    out.append(("C3", int(E[-1]) == kept_lt, f"kept (history) {int(E[-1])} = kept (life table) {kept_lt}"))
    est = np.diff(np.r_[0, E])
    start = 0 if best else 1
    prev = np.r_[start, G[:-1]]
    good = bool(np.all(G == prev + est - P)) and not w["resumed"]
    out.append(("C4", good, "richness each round = previous + kept - removed" + (" (resumed run: not checked)" if w["resumed"] else "")))
    return out


def at_cand(w, c):
    cand = w["cand"]
    if cand[-1] < c:
        return None
    k = int(np.searchsorted(cand, c, side="right")) - 1
    return int(w["ASM_gamma"][k]) if k >= 0 else (0 if w["rule"] == "best" else 1)


def at_round(w, r):
    return int(w["ASM_gamma"][r - 1]) if len(w["ASM_gamma"]) >= r else None


def settle(w):
    A, G = w["ASM_attempts"], w["ASM_gamma"]
    tot = A[-1]
    g = lambda f: int(G[min(int(np.searchsorted(A, f * tot, side="left")), len(G) - 1)])
    n = len(G); a = int(0.8 * n)
    est = np.diff(np.r_[0, w["ASM_established"]])
    s = dict(g50=g(0.5), g80=g(0.8), gend=int(G[-1]))
    s["change"] = (s["gend"] - s["g80"]) / max(s["g80"], 1)
    s["kept_pr"], s["removed_pr"] = float(est[a:].mean()), float(w["ASM_pruned"][a:].mean())
    if w["rule"] == "best":
        dt, da = np.diff(np.r_[0, w["cand"]]), np.diff(np.r_[0, A])
        b = max(1, int(0.2 * n))
        s["pass_first"] = da[:b].sum() / max(dt[:b].sum(), 1)
        s["pass_last"] = da[a:].sum() / max(dt[a:].sum(), 1)
    return s


def label(w):
    s = f", strength {w['strength']:g}" if abs(w["strength"] - 0.4) > 1e-6 else ""
    return f"seed {w['seed']}, {w['rule']} founding{s}, {w['attempts']} attempts"


def report(files):
    W = [read(f) for f in files]
    ok = True
    for w in W:
        print(f"\n{w['name']}\n  {label(w)}")
        for name, good, text in checks(w):
            ok &= bool(good)
            print(f"  {name} {text}: {'PASS' if good else 'FAIL'}")
        kept = int(w["ASM_established"][-1]); cand = int(w["cand"][-1])
        line = (f"  effort: {len(w['ASM_gamma'])} rounds, {cand} candidates, {w['attempts']} entered, {kept} kept, "
                f"{w['gamma']} species at the end; kept per candidate {kept / max(cand, 1):.1%}")
        if w["rule"] == "best":
            line += (f"; entered per candidate {w['attempts'] / max(cand, 1):.1%}, kept per entered {kept / max(w['attempts'], 1):.1%}"
                     f"; stopped saturated {w['saturated']}")
        print(line)
        print("  richness at candidates  " + "  ".join(f"{c}: {at_cand(w, c) if at_cand(w, c) is not None else '-'}" for c in CANDS))
        print("  richness at rounds      " + "  ".join(f"{r}: {at_round(w, r) if at_round(w, r) is not None else '-'}" for r in ROUNDS))
        s = settle(w)
        line = (f"  settle: richness at 50 / 80 / 100% of attempts {s['g50']} / {s['g80']} / {s['gend']}, change over the last 20% "
                f"{s['change']:+.1%}; per round in the last 20%: kept {s['kept_pr']:.2f}, removed {s['removed_pr']:.2f}")
        if w["rule"] == "best":
            line += f"; pass rate first / last 20% of rounds {s['pass_first']:.1%} / {s['pass_last']:.1%}"
        print(line)
    print("\npairs (same sim seed)")
    seen = False
    for wr in [w for w in W if w["rule"] == "random"]:
        for wb in [w for w in W if w["rule"] == "best" and w["seed"] == wr["seed"]]:
            seen = True
            c = int(wr["cand"][-1]); gb = at_cand(wb, c); gr = int(wr["ASM_gamma"][-1])
            nr = len(wr["ASM_gamma"]); nb = len(wb["ASM_gamma"])
            ratio = gb / max(gr, 1) if gb is not None else float("nan")
            read_ = ("mostly extra tries" if ratio <= 1.5 else "best founding packs more species into the same tries"
                     if ratio >= 2 else "both") if gb is not None else "not comparable"
            print(f"  {label(wr)} against {label(wb)}:")
            print(f"    at {c} candidates (where the random world ended): random {gr}, best {gb} species (x{ratio:.2f}) -> {read_}")
            t = min(nr, nb)
            print(f"    at round {t} (equal model time): random {at_round(wr, t)}, best {at_round(wb, t)} species; "
                  f"rounds run: random {nr}, best {nb}")
    B = [w for w in W if w["rule"] == "best"]
    for i, w1 in enumerate(B):
        for w2 in B[i + 1:]:
            if w1["seed"] != w2["seed"] or abs(w1["strength"] - w2["strength"]) > 1e-6 or w1["attempts"] == w2["attempts"]:
                continue
            seen = True
            short, long_ = (w1, w2) if w1["attempts"] < w2["attempts"] else (w2, w1)
            n = len(short["ASM_gamma"]) - 1                       # the last round of the shorter run may be capped
            keys = ("ASM_gamma", "ASM_attempts", "ASM_established", "ASM_pruned", "cand")
            diff = [r for r in range(n) if any(short[k][r] != long_[k][r] for k in keys)] if len(long_["ASM_gamma"]) > n else [0]
            good = not diff
            ok &= good
            print(f"  {label(short)} against {label(long_)}: history identical for the first {n} rounds "
                  f"{'PASS' if good else 'FAIL (first difference at round ' + str(diff[0]) + ')'}")
    if not seen:
        print("  none")
    print("\nall checks passed" if ok else "\nCHECK FAILED")
    return ok


# ---------------------------------------------------------------- self-test on planted files
def plant(path, rule, seed, attempts, p_pass=0.1, p_keep=0.3, p_loss=0.02, strength=0.4, rng_seed=0):
    """Emulates the bookkeeping of assembly v2.3 (not the ecology): one round adds 1 + floor(0.05 S) newcomers."""
    rng = np.random.default_rng(rng_seed)
    best = rule == "best"
    S = 0 if best else 1
    lt_in, lt_est, lt_ent = ([], [], []) if best else ([-1], [1], [])
    R, A, G, E, P, T = [], [], [], [], [], []
    att = tested = est_cum = rnd = 0
    while att < attempts:
        need = min(1 + int(0.05 * S), attempts - att)
        entered = 0
        if best:
            while entered < need:
                tested += 1
                if rng.random() < p_pass:
                    entered += 1; ok = rng.random() < p_keep
                    lt_in.append(rnd); lt_est.append(int(ok)); lt_ent.append(1); est_cum += ok; S += ok
                else:
                    lt_in.append(rnd); lt_est.append(0); lt_ent.append(0)
        else:
            for _ in range(need):
                entered += 1; tested += 1; ok = rng.random() < p_keep
                lt_in.append(rnd); lt_est.append(int(ok)); est_cum += ok; S += ok
        att += entered
        pruned = int(rng.binomial(S, p_loss)); S -= pruned
        R.append(rnd); A.append(att); G.append(S); E.append(est_cum); P.append(pruned); T.append(tested)
        rnd += 1
    d = dict(ASM_round=np.array(R), ASM_attempts=np.array(A), ASM_gamma=np.array(G), ASM_established=np.array(E),
             ASM_pruned=np.array(P), LT_id=np.arange(len(lt_in)), LT_round_in=np.array(lt_in), LT_established=np.array(lt_est),
             attempts_total=np.int64(att), gamma=np.int64(S), SIM_SEED=np.int64(seed), FOUND_rule=np.array([rule]),
             INTERACTION_STRENGTH=np.float32(strength))
    if best:
        d.update(ASM_tested=np.array(T), LT_entered=np.array(lt_ent), FOUND_tested_total=np.int64(tested),
                 FOUND_saturated=np.int64(0))
    np.savez(path, **d)
    return d


def self_test():
    import contextlib, io
    ok = True
    with tempfile.TemporaryDirectory() as t:
        dr = plant(f"{t}/r.npz", "random", 2303, 1000, rng_seed=1)
        db = plant(f"{t}/b.npz", "best", 2303, 1000, rng_seed=2)
        db2 = plant(f"{t}/b2.npz", "best", 2303, 2000, rng_seed=2)       # same stream: its start equals b
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            good = report([f"{t}/r.npz", f"{t}/b.npz", f"{t}/b2.npz"])
        out = buf.getvalue()
        k = int(np.searchsorted(db["ASM_tested"], 1000, side="right")) - 1
        want = f"at 1000 candidates (where the random world ended): random {int(dr['gamma'])}, best {int(db['ASM_gamma'][k])} species"
        t1 = good and out.count("PASS") == 13 and want in out and "history identical for the first" in out
        print(f"S1 planted random and best worlds: all bookkeeping checks PASS, richness at 1,000 candidates read at the right round, "
              f"start of the longer run equals the shorter run: {'PASS' if t1 else 'FAIL'}")
        ok &= t1
        w = read(f"{t}/b.npz")
        t2 = (at_cand(w, int(w["cand"][-1]) + 1) is None and at_cand(w, int(w["cand"][0])) == int(w["ASM_gamma"][0])
              and at_round(w, 1) == int(w["ASM_gamma"][0]) and at_round(w, len(w["ASM_gamma"]) + 1) is None)
        print(f"S2 values at the ends of the curve: {'PASS' if t2 else 'FAIL'}")
        ok &= t2
        for name, change in (("attempts_total off by one", dict(attempts_total=np.int64(int(dr["attempts_total"]) + 1))),
                             ("one kept species missing from the life table", dict(LT_established=np.r_[dr["LT_established"][:-1], 1 - dr["LT_established"][-1]])),
                             ("a removal not counted", dict(ASM_pruned=np.r_[dr["ASM_pruned"][:-1], dr["ASM_pruned"][-1] + 1]))):
            np.savez(f"{t}/m.npz", **{**dr, **change})
            with contextlib.redirect_stdout(io.StringIO()) as b2:
                g = report([f"{t}/m.npz"])
            caught = not g and "FAIL" in b2.getvalue()
            print(f"S3 planted error ({name}) caught: {'PASS' if caught else 'FAIL'}")
            ok &= caught
        np.savez(f"{t}/m.npz", **{**db, "FOUND_tested_total": np.int64(int(db["FOUND_tested_total"]) + 1)})
        with contextlib.redirect_stdout(io.StringIO()) as b2:
            g = report([f"{t}/m.npz"])
        caught = not g
        print(f"S3 planted error (FOUND_tested_total off by one) caught: {'PASS' if caught else 'FAIL'}")
        ok &= caught
        plant(f"{t}/b3.npz", "best", 2303, 2000, rng_seed=3)             # another stream: start differs
        with contextlib.redirect_stdout(io.StringIO()) as b2:
            g = report([f"{t}/b.npz", f"{t}/b3.npz"])
        caught = not g and "first difference at round" in b2.getvalue()
        print(f"S4 two runs that are not repeats are told apart: {'PASS' if caught else 'FAIL'}")
        ok &= caught
    print("all passed" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    files = sorted({f for pat in sys.argv[1:] for f in glob.glob(pat)})
    if not files:
        sys.exit("no files matched")
    sys.exit(0 if report(files) else 1)