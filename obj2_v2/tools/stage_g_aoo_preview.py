"""Exploratory preview (real_val only, no gate): can a level correction plus split conformal (CQR)
fitted on half the real_val communities bring AOO into the band with 90% coverage on the other half?
Inputs: per_stack.csv of stage F step 3a (aoo_expected, aoo_true, aoo_p5, aoo_p95). Two folds by community, swapped."""
import sys, numpy as np, pandas as pd
RC = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
RB = ["1-3", "4-6", "7-10", "11-20", "21+"]
TC = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]

def run(path, label, seed=2026):
    d = pd.read_csv(path, dtype={"true_class": str, "recorded_class": str, "records_6y_bin": str})
    comms = np.array(sorted(d.community.unique()))
    perm = np.random.default_rng(seed).permutation(comms)
    folds = [set(perm[:10]), set(perm[10:])]
    out = []
    for f in (0, 1):
        fit = d[d.community.isin(folds[f])]; chk = d[d.community.isin(folds[1 - f])].copy()
        s = (fit.groupby("recorded_class").aoo_true.sum() / fit.groupby("recorded_class").aoo_expected.sum())
        def apply(x):
            k = x.recorded_class.map(s)
            return x.aoo_expected * k, x.aoo_p5 * k, x.aoo_p95 * k
        e, lo, hi = apply(fit)
        score = np.maximum(lo - fit.aoo_true, fit.aoo_true - hi)
        q = {}
        for g, sc in score.groupby(fit.recorded_class):
            n = len(sc); q[g] = np.sort(sc.to_numpy())[min(n - 1, int(np.ceil((n + 1) * 0.90)) - 1)]
        e, lo, hi = apply(chk)
        qq = chk.recorded_class.map(q)
        chk["exp_c"] = e; chk["lo_c"] = np.maximum(0, lo - qq); chk["hi_c"] = hi + qq
        out.append(chk)
    c = pd.concat(out)
    c["cov_raw"] = (c.aoo_true >= c.aoo_p5) & (c.aoo_true <= c.aoo_p95)
    c["cov_c"] = (c.aoo_true >= c.lo_c) & (c.aoo_true <= c.hi_c)
    c["w_raw"] = c.aoo_p95 - c.aoo_p5; c["w_c"] = c.hi_c - c.lo_c
    print(f"== {label}: {len(c)} stacks, 20 communities, 2 folds of 10 (fit on one, checked on the other, swapped)")
    for col, order, name in (("recorded_class", RC, "recorded class"), ("records_6y_bin", RB, "records over 6 years"),
                             ("true_class", TC, "true class (report only)")):
        print(f"   by {name}: group n | ratio raw -> corrected | 90% coverage raw -> corrected | median width raw -> corrected")
        for g in order:
            x = c[c[col] == g]
            if len(x) == 0: continue
            rr = x.aoo_expected.sum() / x.aoo_true.sum() if x.aoo_true.sum() > 0 else float("nan")
            rc = x.exp_c.sum() / x.aoo_true.sum() if x.aoo_true.sum() > 0 else float("nan")
            print(f"     {g:6s} {len(x):5d} | {rr:5.2f} -> {rc:5.2f} | {x.cov_raw.mean():.3f} -> {x.cov_c.mean():.3f} | "
                  f"{x.w_raw.median():6.1f} -> {x.w_c.median():6.1f}")
    return c

if __name__ == "__main__":
    for p, l in zip(sys.argv[1::2], sys.argv[2::2]):
        run(p, l)