#!/usr/bin/env python3
"""stage_p3c_band_check.py  (obj2_v2, 13 Nov 2026), block P3c-2, read-only. Version 2 (block P3c-2b).

Why: in P3c-1 the largest band matched the dominant-class file in 0.96666 of cells, below the 0.99 gate, so
the band order (band k = class k) is not yet shown. This check finds out why, using only the three GB LCM2019
1 km files (percentage target, percentage aggregate, dominant target). No BNM file is opened; nothing is written.

Readings written before this tool was run (H rules doc, P3c-2):
 (A) group sums: each aggregate band j must equal the sum of the target bands of its group (documentation v1.6,
     Table 1: 1 {1}, 2 {2}, 3 {3}, 4 {4}, 5 {5-8}, 6 {9-12}, 7 {13}, 8 {14}, 9 {15-19}, 10 {20, 21}) within
     rounding, |difference| <= (n + 1) // 2 for a group of n classes, in at least 99% of covered cells.
     Required for groups 7, 8 and 9 (they hold the four no-land classes 13, 14, 17, 18); the others are printed.
 (B) plurality both ways, every class with 20 or more cells each way: among cells whose dominant class is c,
     band c is the most common largest band; among cells whose largest band is c, class c is the most common
     dominant class. Classes 15-19 must each have 20 or more cells both ways.
 Band order confirmed for the land rule = (A) and (B). Otherwise P3c stops and the tables go to Sara.
Version 2 (13 Nov, after the version 1 result, labelled; Sara asked for a fix if one exists): version 1 counted
 the dominant file's value 0 as if it were a class in (B)'s reverse half. The documentation's classes are 1-21,
 and (B) speaks of the "most common dominant class", so version 2 counts dominant values 1-21 only and prints
 the 0s apart. New gate: no dominant value outside 0-21 in covered cells. New report line (no gate): the
 dominant made again from 22 values, the unmapped share (100 - sum of the 21 bands, at least 0) as value 0 plus
 the 21 bands, and its agreement with the dominant file, to test whether the 0s are unmapped-share cells.

Run from obj2_v2/work:
  python3 ../tools/stage_p3c_band_check.py --self-test
  python3 ../tools/stage_p3c_band_check.py --dir ../data/lcm2019_1km/data
"""
import argparse, hashlib, sys, tempfile
from pathlib import Path
import numpy as np

PCT, AGG, DOM = ("gb2019lcm1km_percentage_target.tif", "gb2019lcm1km_percentage_aggregate.tif",
                 "gb2019lcm1km_dominant_target.tif")
MD5 = {PCT: "d66aac8836c89b317e9b6b0a09262c66", AGG: "cf99e45ec1f6240e99e90797dec6aae0",
       DOM: "27a89c103b2f3562fba5acc66e1423b8"}                     # Sara's block P3c-0
GROUPS = {1: [1], 2: [2], 3: [3], 4: [4], 5: [5, 6, 7, 8], 6: [9, 10, 11, 12], 7: [13], 8: [14],
          9: [15, 16, 17, 18, 19], 10: [20, 21]}
NEED_A, MIN_N, OPENED = (7, 8, 9), 20, []


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(d):
    import rasterio
    out = []
    for n in (PCT, AGG, DOM):
        OPENED.append(n)
        with rasterio.open(d / n) as s:
            if s.nodata is not None:
                raise SystemExit(f"STOP: {n} declares no-data {s.nodata}; this check assumes none (P3c-1 found none)")
            out.append((s.read().astype(np.int32), s.transform, s.shape))
    (T, tr, sh), (A, tra, sha), (D, trd, shd) = out
    if not (T.shape[0] == 21 and A.shape[0] == 10 and D.shape[0] == 1 and tr == tra == trd and sh == sha == shd):
        raise SystemExit("STOP: the three files are not 21 / 10 / 1 bands on one grid")
    return T, A, D[0]


def check(d, want_md5=True, quiet=False):
    say = (lambda *a: None) if quiet else print
    d = Path(d)
    if want_md5:
        m = {n: md5(d / n) for n in MD5}
        ok = all(m[n] == MD5[n] for n in m)
        say(f"md5 {', '.join(f'{n} {m[n][:8]}' for n in m)}; = block P3c-0 {ok}")
        if not ok:
            raise SystemExit("STOP: a map file differs from block P3c-0")
    T, A, D = read(d)
    tot = T.sum(0); cov = tot > 0; ncov = int(cov.sum())
    say(f"covered cells {ncov:,}; coastal (21 bands sum below 100) {int((cov & (tot < 100)).sum()):,}")
    T, A, D, tot = T[:, cov], A[:, cov], D[cov], tot[cov]                 # covered cells only, (bands, cells)
    cov = np.ones(ncov, bool)
    # (A) group sums
    a_ok = {}
    say("(A) aggregate band j - sum of its target bands, covered cells:")
    for j, cl in GROUPS.items():
        diff = (A[j - 1] - T[[c - 1 for c in cl]].sum(0))[cov]
        lim = (len(cl) + 1) // 2; share = float((np.abs(diff) <= lim).mean())
        a_ok[j] = share >= 0.99
        say(f"  group {j:>2} classes {cl}: exact {float((diff == 0).mean()):.4f}, within {lim} {share:.4f}, "
            f"mean |diff| {float(np.abs(diff).mean()):.3f}, max |diff| {int(np.abs(diff).max())}"
            + ("  (required)" if j in NEED_A else ""))
    A_pass = all(a_ok[j] for j in NEED_A)
    # (B) dominant class against largest band
    top = T.max(0); one = cov & ((T == top[None]).sum(0) == 1)
    arg = T.argmax(0) + 1; dom = D
    a1, d1 = arg[one], dom[one]
    srt = np.sort(T, axis=0); margin = (srt[-1] - srt[-2])[one]
    agree = a1 == d1
    sec = (np.argsort(T, axis=0)[-2] + 1)[one]
    coast = (tot < 100)[one]
    say(f"(B) cells with one largest band {int(one.sum()):,}; dominant = largest band {float(agree.mean()):.5f}")
    say(f"  where they differ ({int((~agree).sum()):,} cells): dominant = second-largest band {float((sec[~agree] == d1[~agree]).mean()):.3f}; "
        f"median margin largest - second {float(np.median(margin[~agree])) if (~agree).any() else float('nan'):.0f} "
        f"(where they agree {float(np.median(margin[agree])):.0f}); coastal share {float(coast[~agree].mean()) if (~agree).any() else 0:.3f} "
        f"(all cells {float(coast.mean()):.3f})")
    bad = int(((d1 < 0) | (d1 > 21)).sum()) + int(((D < 0) | (D > 21)).sum() - ((d1 < 0) | (d1 > 21)).sum())
    say(f"  dominant values outside 0-21 in covered cells: {bad}")
    b_ok, few = bad == 0, []
    say("  class: cells dominant c | share with largest band c | most common largest band || cells largest band c with a dominant class 1-21 | share dominant c | most common dominant class | cells with dominant 0 (apart)")
    for c in range(1, 22):
        md, mb = d1 == c, (a1 == c) & (d1 >= 1) & (d1 <= 21)                 # reverse half: classes 1-21 only
        nd, nb = int(md.sum()), int(mb.sum()); n0 = int(((a1 == c) & (d1 == 0)).sum())
        pd_ = np.bincount(a1[md], minlength=22) if nd else np.zeros(22, int)
        pb = np.bincount(d1[mb], minlength=22) if nb else np.zeros(22, int)
        top_b = int(pd_.argmax()) if nd else -1; top_d = int(pb.argmax()) if nb else -1
        if nd >= MIN_N and nb >= MIN_N:
            b_ok &= (top_b == c and top_d == c)
        else:
            few.append(c)
        say(f"  {c:>2}: {nd:>7,} | {pd_[c] / nd if nd else float('nan'):.3f} | {top_b:>2} || {nb:>7,} | "
            f"{pb[c] / nb if nb else float('nan'):.3f} | {top_d:>2} | dominant 0: {n0:,}")
    coastal_ok = not any(c in few for c in (15, 16, 17, 18, 19))
    U = np.clip(100 - tot, 0, None)[one]                                   # unmapped share, report only
    V = np.vstack([U[None], T[:, one]]); vt = V.max(0); v1 = (V == vt[None]).sum(0) == 1
    d22 = V.argmax(0)
    say(f"  report only, dominant made from 22 values (unmapped share as 0, then the 21 bands), cells with one largest "
        f"of the 22 ({int(v1.sum()):,}): = dominant file {float((d22[v1] == d1[v1]).mean()):.5f}; in the "
        f"{int((~agree).sum()):,} mismatching cells: dominant file 0 {float((d1[~agree] == 0).mean()):.3f}, unmapped share "
        f">= largest band {float((U[~agree] >= top[one][~agree]).mean()) if (~agree).any() else float('nan'):.3f}")
    B_pass = b_ok and coastal_ok
    say(f"  classes with fewer than {MIN_N} cells one way (plurality not tested): {few}")
    say(f"(A) groups 7, 8, 9: {'PASS' if A_pass else 'FAIL'}; (B) plurality both ways: {'PASS' if b_ok else 'FAIL'}; "
        f"classes 15-19 with {MIN_N}+ cells both ways: {coastal_ok}")
    conf = A_pass and B_pass
    say(f"stage P3c-2: band order for the land rule {'CONFIRMED' if conf else 'NOT CONFIRMED'}; read-only, nothing written")
    return dict(A=A_pass, B=B_pass, conf=conf, agree=float(agree.mean()))


def write_planted(d, swap=None, noise=0.03, seed=5, zero_coast=True, bad_value=False, coast_bias=False):
    import rasterio
    from rasterio.transform import from_origin
    rng = np.random.default_rng(seed); H, W = 60, 50
    X = np.zeros((21, H, W))
    for i in range(H):
        for j in range(W):
            if j >= 44:
                continue                                          # unmapped sea
            k = rng.choice(21, size=4, replace=False)
            X[k, i, j] = rng.dirichlet([3, 1.5, 1, 0.5]) * (100 if j < 40 else rng.uniform(20, 90))
            if coast_bias and j >= 36:                                 # mostly-sea cells led by a shore class
                k = np.r_[rng.choice([14, 16]), rng.choice([c for c in range(21) if c not in (14, 16)], 3, replace=False)]
                X[:, i, j] = 0; X[k, i, j] = rng.dirichlet([6, 1, 1, 1]) * rng.uniform(10, 40)
    T = np.round(X).astype(np.uint8)
    Ag = np.stack([np.round(X[[c - 1 for c in cl]].sum(0)) for cl in GROUPS.values()]).astype(np.uint8)
    Xn = X + rng.normal(0, noise * 100, X.shape) * (X > 0)          # dominant made from slightly different shares
    D = np.where(X.sum(0) > 0, Xn.argmax(0) + 1, 0).astype(np.uint8)
    if zero_coast:                                                   # unmapped share counted as value 0
        D = np.where((X.sum(0) > 0) & (100 - X.sum(0) > X.max(0)), 0, D).astype(np.uint8)
    if bad_value:
        D[0, 0] = 25
    if swap:
        a, b = swap; T[[a - 1, b - 1]] = T[[b - 1, a - 1]]
    tr = from_origin(400000, 300000, 1000, 1000)
    kw = dict(driver="GTiff", height=H, width=W, crs="EPSG:27700", transform=tr, dtype="uint8")
    for n, arr in ((PCT, T), (AGG, Ag), (DOM, D[None])):
        with rasterio.open(d / n, "w", count=arr.shape[0], **kw) as f:
            f.write(arr)


def self_test():
    res = []
    def case(name, ok):
        res.append(ok); print(f"{name}: {'PASS' if ok else 'FAIL'}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); out = {}
        for k, sw in [("ok", None), ("s17_15", (17, 15)), ("s13_3", (13, 3)), ("s20_21", (20, 21)), ("s18_4", (18, 4))]:
            dd = tmp / k; dd.mkdir(); write_planted(dd, swap=sw); out[k] = check(dd, want_md5=False, quiet=True)
        dd = tmp / "bad"; dd.mkdir(); write_planted(dd, bad_value=True); out["bad"] = check(dd, want_md5=False, quiet=True)
        dd = tmp / "coast"; dd.mkdir(); write_planted(dd, coast_bias=True); out["coast"] = check(dd, want_md5=False, quiet=True)
        import rasterio
        with rasterio.open(dd / PCT) as f:
            Tc = f.read().astype(int)
        with rasterio.open(dd / DOM) as f:
            Dc = f.read(1).astype(int)
        cv = Tc.sum(0) > 0; lb = Tc.argmax(0) + 1
        zero_first = [c for c in (15, 17) if np.bincount(Dc[cv & (lb == c)], minlength=22).argmax() == 0]
        dd = tmp / "coast_swap"; dd.mkdir(); write_planted(dd, coast_bias=True, swap=(17, 15))
        out["coast_swap"] = check(dd, want_md5=False, quiet=True)
        case(f"T1 right order, dominant from slightly different shares and 0 where the unmapped share is largest (agreement {out['ok']['agree']:.3f}): CONFIRMED",
             out["ok"]["conf"] and out["ok"]["agree"] < 0.99)
        case("T2 Littoral rock and Supralittoral rock swapped (same group): (A) passes, (B) fails, NOT CONFIRMED",
             out["s17_15"]["A"] and not out["s17_15"]["B"] and not out["s17_15"]["conf"])
        case("T3 Saltwater and Arable swapped: (A) fails, NOT CONFIRMED", not out["s13_3"]["A"] and not out["s13_3"]["conf"])
        case("T4 Urban and Suburban swapped: (B) fails, NOT CONFIRMED", not out["s20_21"]["B"] and not out["s20_21"]["conf"])
        case("T5 Littoral sediment and Improved grassland swapped: NOT CONFIRMED", not out["s18_4"]["conf"])
        case("T6 only the three map files opened", set(OPENED) <= {PCT, AGG, DOM})
        case("T7 a dominant value 25 in a covered cell: NOT CONFIRMED", not out["bad"]["B"] and not out["bad"]["conf"])
        case(f"T8 shore classes 15 and 17 lead mostly-sea cells, the dominant file says 0 there (0 first for bands {zero_first}, "
             f"as in Sara's file; version 1 counted 0 and fails here): CONFIRMED", zero_first == [15, 17] and out["coast"]["conf"])
        case("T9 the same world with Littoral rock and Supralittoral rock swapped: NOT CONFIRMED", not out["coast_swap"]["conf"])
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../data/lcm2019_1km/data")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    r = check(a.dir)
    sys.exit(0 if r["conf"] else 1)