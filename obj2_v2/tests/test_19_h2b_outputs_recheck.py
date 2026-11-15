#!/usr/bin/env python3
"""
test_19_h2b_outputs_recheck.py, version 3 (obj2_v2, stage H2b check, 12 Nov 2026)

Question: do Sara's H2b outputs hold when recomputed with separate code on her own machine, and does the Y3 thinning
check still pass once the shared random numbers of the four methods are allowed for?

Reads only ../data/stage_h2b_score (fit half; no BNM truth file, no test-half file). numpy and the standard library only.
Checks, each PASS or FAIL:
  C1  md5 of h2b_predicted.npz and h2b_scores.npz = h2b_report.json; item order of the csv = the npz files (item, stack, classes,
      recorded holes, AUC of 8 methods within 1e-6: the csv keeps 6 decimals and an AUC m/(2 n1 n0) can sit exactly halfway,
      an error of exactly 5e-7 that float arithmetic can push a hair above it; version 2 used 5e-7 and failed on Sara's files)
  C2  counts from per_stack_h2b.csv (7,463 rows, 1,516 stacks, rankable rows and stacks) = h2b_report.json
  C3  H-R1 stack-first means and stack counts from the csv (AUC to 6 decimals) = h2b_report.json (1e-5):
      seed 1, seed 2, pooled against B1 stored and exact; B3, B4, B5 against B1 stored
  C4  the 7 Nov 10+ version and the Red List means = h2b_report.json (1e-5)
  C5  per-window stack-first means = per_window_h2b.csv (1e-5)
  C6  H-R2 ratio, coverage, median width and stacks per group, full precision from h2b_scores.npz = h2b_report.json (1e-9)
  C7  Y3 per detection class: the pooled share and binomial z as the tool prints them (= the log line, with --log);
      binomial z per method (s1, s2, B3, B4); the pooled z with a cluster-robust se. The thinning uniform of a cell is
      the same for the 4 methods (stack-repeat, sample, cell), so a stack-repeat x sample is one cluster; clusters are
      independent. Rule: |z| <= 4 per method and for the robust z.
  Report only: correlation of seed 1 and seed 2 thinning residuals at the same sample index, and at the next index as
  the control (near 0); the variance ratio robust / binomial.
Run (from obj2_v2/work; save this file in obj2_v2/tests):
  python3 ../tests/test_19_h2b_outputs_recheck.py --self-test
  python3 ../tests/test_19_h2b_outputs_recheck.py ../data/stage_h2b_score --log ../logs/stage_h2b_score.txt
Self-test (planted data, a few seconds): U1 stack-first mean; U2 on planted maps with shared uniforms the pooled binomial
z is too wide (sd above 1.5) while the robust and per-method z have sd near 1; U3 a detection of 0.78 against a stated 0.80
fails; U4 H-R2 rows = a plain loop; U5 a planted H2b folder (with AUC values exactly halfway at the 7th decimal) passes C1-C7, and a changed H-R2 ratio, two swapped csv rows
and a wrong z in the log line each fail their check; U6 md5 of a changed file.
"""
import csv, hashlib, json, re, sys, tempfile
from pathlib import Path
import numpy as np

TCLS = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
BINS = ["1-3", "4-6", "7-10", "11-20", "21+"]
DET, DLAB = (0.643, 0.642, 0.800), ("1-5", "6-20", "21+")
Y3_M = ("s1", "s2", "B3", "B4")
RES = []
QUIET = [False]


def check(name, ok, text):
    RES.append(bool(ok))
    if not QUIET[0]:
        print(f"{name} {text}: {'PASS' if ok else 'FAIL'}", flush=True)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def sf_mean(d, stack, mask):
    """Stack-first mean: each stack's items averaged, then the mean over stacks; n = distinct stacks."""
    ok = mask & ~np.isnan(d)
    if not ok.any():
        return float("nan"), 0
    u, inv = np.unique(stack[ok], return_inverse=True)
    return float((np.bincount(inv, weights=d[ok]) / np.bincount(inv)).mean()), len(u)


def read_csv(p):
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    f = lambda k: np.array([float(v) if v != "" else np.nan for v in cols[k]])
    return cols, f


def hr2_rows(y, mean, L, U, gi, labels, stack):
    cov = (y >= L) & (y <= U)
    out = []
    for j, lab in enumerate(labels):
        m = gi == j
        if m.any():
            out.append(dict(group=lab, n=int(m.sum()), stacks=len(np.unique(stack[m])), ratio=float(mean[m].mean() / y[m].mean()),
                            cov=float(cov[m].mean()), width=float(np.median((U - L)[m]))))
    return out


def y3(occ, thin, dcls, methods=Y3_M):
    """occ, thin: dict method -> (n, K) counts in the holes; returns per class the tool's pooled z, per-method z, robust z."""
    out = []
    for c in range(3):
        m = dcls == c
        p = DET[c]
        O = sum(occ[k][m].sum() for k in methods); T = sum(thin[k][m].sum() for k in methods)
        if O == 0 or any(occ[k][m].sum() == 0 for k in methods):
            out.append(None); continue
        zb = (T / O - p) / np.sqrt(p * (1 - p) / O)
        zm = {k: (thin[k][m].sum() / occ[k][m].sum() - p) / np.sqrt(p * (1 - p) / occ[k][m].sum()) for k in methods}
        r = {k: thin[k][m].astype(float) - p * occ[k][m] for k in methods}
        R = sum(r.values())
        zr = R.sum() / np.sqrt((R ** 2).sum())
        vr = (R ** 2).sum() / (p * (1 - p) * O)
        a, b = r[methods[0]].ravel(), r[methods[1]].ravel()
        same = float(np.corrcoef(a, b)[0, 1]); shift = float(np.corrcoef(a, np.roll(r[methods[1]], 1, axis=1).ravel())[0, 1])
        out.append(dict(share=T / O, zb=zb, zm=zm, zr=zr, vr=vr, same=same, shift=shift, O=int(O)))
    return out


def run(d, log=None):
    d = Path(d)
    rep = json.loads((d / "h2b_report.json").read_text())
    cols, f = read_csv(d / "per_stack_h2b.csv")
    stack = np.array(cols["stack_id"], dtype=int); n = len(stack)
    rc = np.array([TCLS.index(v) for v in cols["recorded_class"]]); rb = np.array([BINS.index(v) for v in cols["records_bin"]])
    A = {k: f("auc_" + k) for k in ("s1", "s2", "pooled", "B1", "B1x", "B3", "B4", "B5")}
    # C1
    pf, sf = d / "h2b_predicted.npz", d / "h2b_scores.npz"
    have = pf.is_file() and sf.is_file()
    if have:
        P, S = np.load(pf), np.load(sf)
        ok = md5(pf) == rep["md5"]["predicted"] and md5(sf) == rep["md5"]["scores"]
        item = np.array(cols["item"], dtype=int)
        yc = np.array(cols["recorded_holes_strict"], dtype=int)
        da = max(float(np.nanmax(np.abs(A[k] - S[f"auc_strict_{k}"]))) for k in A)
        nan_same = all(np.array_equal(np.isnan(A[k]), np.isnan(S[f"auc_strict_{k}"])) for k in A)
        parts = dict(item=np.array_equal(item, np.arange(n)), stack=np.array_equal(P["stack_id"], stack) and np.array_equal(S["stack_id"], stack),
                     classes=np.array_equal(P["rcls"], rc) and np.array_equal(P["rbin"], rb), holes=np.array_equal(S["y_strict"].astype(int), yc),
                     auc=da <= 1e-6 and nan_same)
        al = all(parts.values())
        check("C1", ok and al, f"h2b_predicted.npz {md5(pf)[:8]}, h2b_scores.npz {md5(sf)[:8]} = report {ok}; csv rows in npz order: "
              + ", ".join(f"{k} {v}" for k, v in parts.items()) + f" (AUC of 8 methods, csv 6 decimals, largest difference {da:.2e}; "
              f"limit 1e-6, since an AUC m/(2 n1 n0) can sit exactly halfway at the 7th decimal)")
    else:
        print("C1, C6, C7 SKIPPED: h2b_predicted.npz / h2b_scores.npz not in the folder")
    # C2
    rk = ~np.isnan(A["s1"])
    ok = n == rep["n"] and len(np.unique(stack)) == rep["stacks"] and int(rk.sum()) == rep["rankable"] and len(np.unique(stack[rk])) == rep["rankable_stacks"]
    check("C2", ok, f"{n:,} rows, {len(np.unique(stack)):,} stacks, rankable {int(rk.sum()):,} ({len(np.unique(stack[rk])):,} stacks) = report")
    # C3
    worst, nbad, ncmp = 0.0, 0, 0
    pairs = [(m, b, rep["H_R1"][m][b]["rows"]) for m in ("s1", "s2", "pooled") for b in ("B1", "B1x")] + \
            [(m, "B1", rep["H_R1_baselines"][m]["rows"]) for m in ("B3", "B4", "B5")]
    for m, b, rows in pairs:
        dd = A[m] - A[b]
        for g, gi, labs in (("recorded class", rc, TCLS), ("records over 6 years", rb, BINS)):
            for r in rows[g]:
                mu, ns = sf_mean(dd, stack, gi == labs.index(r["group"]))
                ncmp += 1; nbad += ns != r["n"]; worst = max(worst, abs(mu - r["mean"]))
    check("C3", worst <= 1e-5 and nbad == 0, f"{ncmp} H-R1 group rows: largest mean difference {worst:.1e}, stack counts different {nbad}")
    # C4
    w4 = 0.0; b4 = 0
    t10 = np.array(cols["all_rec_2019"], dtype=int) >= 10
    for m in ("s1", "s2"):
        for b in ("B1", "B1x"):
            mu, ns = sf_mean(A[m] - A[b], stack, t10); r = rep["seven_nov"]["hr1"][m][b]
            w4 = max(w4, abs(mu - r["mean"])); b4 += ns != r["n"]
        st = np.array(cols["status"])
        for s, r in rep["red_list"][m]["hr1"].items():
            mu, ns = sf_mean(A[m] - A["B1"], stack, st == s)
            w4 = max(w4, abs(mu - r["mean"])); b4 += ns != r["n"]
    check("C4", w4 <= 1e-5 and b4 == 0, f"10+ version and Red List: largest mean difference {w4:.1e}, stack counts different {b4}")
    # C5
    wc, wf = read_csv(d / "per_window_h2b.csv")
    win = np.array(cols["window"]); w5 = 0.0
    for i, w in enumerate(wc["window"]):
        for m in ("s1", "s2"):
            w5 = max(w5, abs(sf_mean(A[m] - A["B1"], stack, win == w)[0] - wf(f"mean_d_{m}_B1")[i]))
    check("C5", w5 <= 1e-5, f"{len(wc['window'])} windows: per-window stack-first means, largest difference {w5:.1e}")
    if not have:
        return
    # C6
    y = S["y_strict"].astype(float); w6 = 0.0; nb = 0
    for m in ("s1", "s2", "pooled", "B3", "B4", "B5"):
        for g, gi, labs in (("recorded class", P["rcls"], TCLS), ("records over 6 years", P["rbin"], BINS)):
            mine = hr2_rows(y, S[f"hr2_mean_{m}"], S[f"hr2_L_{m}"], S[f"hr2_U_{m}"], gi, labs, stack)
            for a, r in zip(mine, rep["H_R2"][m]["rows"][g]):
                nb += a["group"] != r["group"] or a["n"] != r["n"] or a["stacks"] != r["stacks"]
                w6 = max(w6, *(abs(a[q] - r[q]) for q in ("ratio", "cov", "width")))
    check("C6", w6 <= 1e-9 and nb == 0, f"H-R2 from h2b_scores.npz (6 methods x 12 groups): largest difference {w6:.1e}, "
          f"groups or counts different {nb}")
    # C7
    occ = {k: P[f"occ_h_{k}"] for k in Y3_M}; thin = {k: P[f"thin_h_{k}"] for k in Y3_M}
    res = y3(occ, thin, P["dcls"])
    logok, logtxt = True, "no log given"
    if log:
        line = next((l for l in Path(log).read_text().splitlines() if l.startswith("Y3 ")), "")
        got = re.findall(r"(1-5|6-20|21\+) ([0-9.]+) \(signed [0-9.]+, z ([+-][0-9.]+)", line)
        rr = [r for r in res if r is not None]
        logok = len(got) == len(rr) > 0 and all(abs(float(s) - r["share"]) < 5e-5 and abs(float(z) - r["zb"]) < 5e-3 for (_, s, z), r in zip(got, rr))
        logtxt = f"pooled share and z = the log's Y3 line {logok}"
    zok = all(r is None or (abs(r["zr"]) <= 4 and all(abs(v) <= 4 for v in r["zm"].values())) for r in res)
    txt = "; ".join(f"{DLAB[c]}: no occupied hole cell" if r is None else f"{DLAB[c]}: pooled {r['share']:.4f} z {r['zb']:+.2f} (binomial, as the tool), robust z {r['zr']:+.2f}, variance ratio "
                    f"{r['vr']:.2f}; per method " + ", ".join(f"{k} {v:+.2f}" for k, v in r["zm"].items())
                    + f"; corr s1/s2 residuals same sample {r['same']:+.3f}, next sample {r['shift']:+.3f}" for c, r in enumerate(res))
    check("C7", logok and zok, f"Y3 thinning ({logtxt}): {txt}")


def planted_folder(t, g):
    """A small H2b output folder whose report, per-window csv and Y3 log line are written with plain Python loops."""
    ns, R, wins = 60, 5, [f"w{i}" for i in range(6)]
    n = ns * R
    stack = np.repeat(np.arange(ns), R); win = np.array([wins[s % 6] for s in stack])
    rc = np.repeat(g.integers(0, 7, ns), R); rb = np.repeat(g.integers(0, 5, ns), R); dc = np.repeat(g.integers(0, 3, ns), R)
    status = np.repeat(np.array(["LC", "VU", "EN"])[g.integers(0, 3, ns)], R); rec19 = np.repeat(g.integers(0, 30, ns), R)
    A = {k: np.round(g.uniform(0.3, 0.9, n), 6) for k in ("s1", "s2", "pooled", "B1", "B1x", "B3", "B4", "B5")}
    A["s1"][:3] = [0.0234375, 0.0390625, 0.0546875]         # m / 128: exactly halfway at the 7th decimal, float error just above 5e-7, as on Sara's files
    blank = g.random(n) < 0.15
    blank[:3] = False
    for k in A:
        A[k][blank] = np.nan
    y = g.poisson(4, n).astype(float)
    head = ["item", "window", "status", "stack_id", "recorded_class", "records_bin", "recorded_holes_strict", "all_rec_2019"] + [f"auc_{k}" for k in A]
    with open(t / "per_stack_h2b.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head)
        for i in range(n):
            w.writerow([i, win[i], status[i], stack[i], TCLS[rc[i]], BINS[rb[i]], int(y[i]), rec19[i]] + ["" if np.isnan(A[k][i]) else f"{A[k][i]:.6f}" for k in A])
    def loop_sf(d, keep):
        acc = {}
        for i in range(n):
            if keep[i] and not np.isnan(d[i]):
                acc.setdefault(stack[i], []).append(d[i])
        return (sum(sum(v) / len(v) for v in acc.values()) / len(acc), len(acc)) if acc else (float("nan"), 0)
    def rows_for(d):
        out = {}
        for g_, gi, labs in (("recorded class", rc, TCLS), ("records over 6 years", rb, BINS)):
            out[g_] = []
            for j, lab in enumerate(labs):
                mu, k = loop_sf(d, gi == j)
                if k:
                    out[g_].append(dict(group=lab, n=k, mean=mu))
        return out
    K = 20; C = 49
    base = g.random((n, 1, C)) < 0.6
    om = {k: base & (g.random((n, K, C)) < 0.9) for k in Y3_M}; U = g.random((n, K, C)); p = np.array(DET)[dc][:, None, None]
    P = dict(stack_id=stack, rcls=rc, rbin=rb, dcls=dc)
    for k in Y3_M:
        P[f"occ_h_{k}"] = om[k].sum(2).astype(np.uint16); P[f"thin_h_{k}"] = (om[k] & (U < p)).sum(2).astype(np.uint16)
    S = dict(y_strict=y, stack_id=stack, **{f"auc_strict_{k}": A[k] for k in A})
    H2 = {}
    for m in ("s1", "s2", "pooled", "B3", "B4", "B5"):
        mn = y * g.uniform(0.7, 1.3, n); L = np.maximum(y - g.integers(-1, 4, n), 0); Uu = y + g.integers(-1, 4, n)
        S[f"hr2_mean_{m}"], S[f"hr2_L_{m}"], S[f"hr2_U_{m}"] = mn, L.astype(float), Uu.astype(float)
        H2[m] = {"rows": {}}
        for g_, gi, labs in (("recorded class", rc, TCLS), ("records over 6 years", rb, BINS)):
            H2[m]["rows"][g_] = []
            for j, lab in enumerate(labs):
                ix = [i for i in range(n) if gi[i] == j]
                if ix:
                    H2[m]["rows"][g_].append(dict(group=lab, n=len(ix), stacks=len(set(stack[ix].tolist())),
                                                  ratio=sum(mn[ix]) / sum(y[ix]), cov=sum(1 for i in ix if L[i] <= y[i] <= Uu[i]) / len(ix),
                                                  width=float(np.median([Uu[i] - L[i] for i in ix]))))
    np.savez(t / "h2b_predicted.npz", **P); np.savez(t / "h2b_scores.npz", **S)
    rk = ~np.isnan(A["s1"])
    rep = dict(n=n, stacks=ns, rankable=int(rk.sum()), rankable_stacks=len(set(stack[rk].tolist())),
               md5=dict(predicted=md5(t / "h2b_predicted.npz"), scores=md5(t / "h2b_scores.npz")),
               H_R1={m: {b: {"rows": rows_for(A[m] - A[b])} for b in ("B1", "B1x")} for m in ("s1", "s2", "pooled")},
               H_R1_baselines={m: {"rows": rows_for(A[m] - A["B1"])} for m in ("B3", "B4", "B5")},
               seven_nov={"hr1": {m: {b: dict(zip(("mean", "n"), loop_sf(A[m] - A[b], rec19 >= 10))) for b in ("B1", "B1x")} for m in ("s1", "s2")}},
               red_list={m: {"hr1": {s: dict(zip(("mean", "n"), loop_sf(A[m] - A["B1"], status == s))) for s in ("LC", "VU", "EN")}} for m in ("s1", "s2")},
               H_R2=H2)
    (t / "h2b_report.json").write_text(json.dumps(rep))
    with open(t / "per_window_h2b.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["window", "mean_d_s1_B1", "mean_d_s2_B1"])
        for wn in wins:
            w.writerow([wn] + [loop_sf(A[m] - A["B1"], win == wn)[0] for m in ("s1", "s2")])
    parts = []
    for c in range(3):
        O = T = 0
        for k in Y3_M:
            for i in range(n):
                if dc[i] == c:
                    O += int(P[f"occ_h_{k}"][i].sum()); T += int(P[f"thin_h_{k}"][i].sum())
        parts.append(f"{DLAB[c]} {T / O:.4f} (signed {DET[c]}, z {(T / O - DET[c]) / (DET[c] * (1 - DET[c]) / O) ** 0.5:+.2f}, {O:,} occupied hole cells)")
    (t / "log.txt").write_text("Y0 ...\nY3 hole masks ...: thinned share in the holes by detection class: " + "; ".join(parts) + ": PASS\n")
    return True


def self_test():
    g = np.random.default_rng(7)
    # U1 stack-first mean
    mu, ns = sf_mean(np.array([1.0] * 5 + [0.0]), np.array([0] * 5 + [1]), np.ones(6, bool))
    check("U1", abs(mu - 0.5) < 1e-12 and ns == 2, f"stack-first mean of 5 repeats at 1 and 1 at 0 = {mu:.3f} over {ns} stacks (item mean 0.833)")
    # U2 Y3 calibration on planted maps: 4 methods with overlapping maps, one shared uniform per (item, sample, cell)
    n, K, C = 60, 20, 64
    zb, zr, zm = [], [], []
    for rep_ in range(150):
        dc = np.arange(n) % 3
        base = g.random((n, 1, C)) < 0.6
        occ_maps = {k: base & (g.random((n, K, C)) < 0.9) for k in Y3_M}
        U = g.random((n, K, C))
        p = np.array(DET)[dc][:, None, None]
        occ = {k: occ_maps[k].sum(2) for k in Y3_M}; thin = {k: (occ_maps[k] & (U < p)).sum(2) for k in Y3_M}
        r = y3(occ, thin, dc)
        zb += [x["zb"] for x in r]; zr += [x["zr"] for x in r]; zm += [x["zm"]["s1"] for x in r]
    sb, sr, sm = np.std(zb), np.std(zr), np.std(zm)
    check("U2", sb > 1.5 and 0.8 <= sr <= 1.2 and 0.8 <= sm <= 1.2, f"planted overlapping maps, shared uniforms, 450 class tests: sd of the "
          f"pooled binomial z {sb:.2f} (inflated), robust z {sr:.2f}, per-method z {sm:.2f} (both near 1)")
    # U3 a thinning bias is found: kept with 0.78 where 0.80 is stated
    dc = np.full(400, 2); base = g.random((400, 1, 64)) < 0.6
    om = {k: base & (g.random((400, 50, 64)) < 0.9) for k in Y3_M}; U = g.random((400, 50, 64))
    r = y3({k: om[k].sum(2) for k in Y3_M}, {k: (om[k] & (U < 0.78)).sum(2) for k in Y3_M}, dc)[2]
    check("U3", abs(r["zr"]) > 4 and r["same"] > 0.2 and abs(r["shift"]) < 0.05, f"detection 0.78 against a stated 0.80: robust z {r['zr']:+.1f} "
          f"(fails as it must); residual corr same sample {r['same']:+.2f}, next sample {r['shift']:+.3f}")
    # U4 H-R2 group summary = a plain loop; a changed ratio is seen
    yy = g.poisson(5, 300).astype(float); L = np.maximum(yy - g.integers(0, 4, 300), 0); U_ = yy + g.integers(-1, 4, 300)
    mn = yy * g.uniform(0.8, 1.2, 300); gi = g.integers(0, 3, 300); st = np.arange(300) // 3
    rows = hr2_rows(yy, mn, L, U_, gi, ["a", "b", "c"], st)
    loop_ok = all(abs(r["cov"] - np.mean([(yy[i] >= L[i]) and (yy[i] <= U_[i]) for i in range(300) if gi[i] == j])) < 1e-12 and
                  abs(r["ratio"] - sum(mn[i] for i in range(300) if gi[i] == j) / sum(yy[i] for i in range(300) if gi[i] == j)) < 1e-12
                  for j, r in enumerate(rows))
    rows2 = hr2_rows(yy, mn * (1 + 1e-6), L, U_, gi, ["a", "b", "c"], st)
    seen = max(abs(a["ratio"] - b["ratio"]) for a, b in zip(rows, rows2)) > 1e-9
    check("U4", loop_ok and seen, f"H-R2 rows = a plain loop {loop_ok}; a ratio changed by 1e-6 is above the 1e-9 tolerance {seen}")
    # U5 planted H2b folder end to end (report written by plain loops), then three tampers
    with tempfile.TemporaryDirectory() as t:
        t = Path(t); ok6 = planted_folder(t, g)
        QUIET[0] = True
        n0 = len(RES); run(t, t / "log.txt"); clean = all(RES[n0:]) and len(RES) - n0 == 7
        del RES[n0:]
        rep = json.loads((t / "h2b_report.json").read_text())
        rep["H_R2"]["s2"]["rows"]["records over 6 years"][1]["ratio"] += 1e-6; (t / "h2b_report.json").write_text(json.dumps(rep))
        n0 = len(RES); run(t, t / "log.txt"); c6 = RES[n0 + 5] is False; del RES[n0:]
        rep["H_R2"]["s2"]["rows"]["records over 6 years"][1]["ratio"] -= 1e-6; (t / "h2b_report.json").write_text(json.dumps(rep))
        lines = (t / "per_stack_h2b.csv").read_text().splitlines(); lines[1], lines[2] = lines[2], lines[1]
        (t / "per_stack_h2b.csv").write_text("\n".join(lines) + "\n")
        n0 = len(RES); run(t, t / "log.txt"); c1 = RES[n0] is False; del RES[n0:]
        lines[1], lines[2] = lines[2], lines[1]; (t / "per_stack_h2b.csv").write_text("\n".join(lines) + "\n")
        (t / "log.txt").write_text((t / "log.txt").read_text().replace("z ", "z +9", 1))
        n0 = len(RES); run(t, t / "log.txt"); c7 = RES[n0 + 6] is False; del RES[n0:]
        QUIET[0] = False
        check("U5", ok6 and clean and c6 and c1 and c7, f"planted H2b folder: C1-C7 all PASS {clean}; an H-R2 ratio changed by 1e-6 fails C6 {c6}; "
              f"two csv rows swapped fail C1 {c1}; a wrong z in the log line fails C7 {c7}")
    # U6 md5 of a changed file differs
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "x.npz"; np.savez(p, a=np.arange(5)); m1 = md5(p)
        b = bytearray(p.read_bytes()); b[-1] ^= 1; p.write_bytes(bytes(b))
        check("U6", md5(p) != m1, "one byte changed in a file changes its md5")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        log = sys.argv[sys.argv.index("--log") + 1] if "--log" in sys.argv else None
        args = [a for a in args if a != log]
        run(args[0] if args else "../data/stage_h2b_score", log)
    print(f"test_19: {sum(RES)} PASS, {len(RES) - sum(RES)} FAIL")
    sys.exit(0 if RES and all(RES) else 1)