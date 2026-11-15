#!/usr/bin/env python3
"""stage_p3c_lcm_header.py  (obj2_v2, 13 Nov 2026), block P3c-1, read-only.

Reads the two GB LCM2019 1 km files (percentage target, dominant target) and prints: the header (bands,
data type, coordinate system, pixel size, origin, no-data value, band names), the value range after no-data
masking, how unmapped cells (open sea) are coded, and the band-order check added to the P3c rule before any
value was read (the product documentation v1.6 gives no band list for the 1 km rasters): among cells with
any cover and a single largest band, the class number of that band (band k = class k) equals the class in
the dominant target file in at least 99% of cells.
No BNM file is opened; nothing is written.

Run from obj2_v2/work:
  python3 ../tools/stage_p3c_lcm_header.py --self-test
  python3 ../tools/stage_p3c_lcm_header.py --dir ../data/lcm2019_1km/data
"""
import argparse, hashlib, sys, tempfile
from pathlib import Path
import numpy as np

PCT, DOM = "gb2019lcm1km_percentage_target.tif", "gb2019lcm1km_dominant_target.tif"
MD5 = {PCT: "d66aac8836c89b317e9b6b0a09262c66", DOM: "27a89c103b2f3562fba5acc66e1423b8"}   # Sara's block P3c-0
AGREE_MIN = 0.99
OPENED = []


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(d, want_md5=True, quiet=False):
    import rasterio
    say = (lambda *a: None) if quiet else print
    d = Path(d); g = {}
    m = {n: md5(d / n) for n in (PCT, DOM)}
    g["L0 md5"] = (not want_md5) or all(m[n] == MD5[n] for n in m)
    say(f"L0 md5 {PCT} {m[PCT]}, {DOM} {m[DOM]}; = block P3c-0 {g['L0 md5']}")
    OPENED.extend([str(d / PCT), str(d / DOM)])
    with rasterio.open(d / PCT) as s, rasterio.open(d / DOM) as t:
        tr = s.transform; epsg = s.crs.to_epsg() if s.crs else None
        say(f"header {PCT}: bands {s.count}, type {s.dtypes[0]}, EPSG {epsg}, size {s.width} x {s.height}, "
            f"pixel {tr.a} x {tr.e}, origin (west {tr.c}, north {tr.f}), no-data {s.nodata}")
        say(f"  band names {list(s.descriptions)}")
        say(f"  file tags {dict(s.tags())}; band 1 tags {dict(s.tags(1))}")
        say(f"header {DOM}: bands {t.count}, type {t.dtypes[0]}, EPSG {t.crs.to_epsg() if t.crs else None}, "
            f"size {t.width} x {t.height}, no-data {t.nodata}, same grid {t.transform == tr and t.shape == s.shape}")
        g["L1 header"] = (s.count == 21 and s.dtypes[0] == "uint8" and epsg == 27700 and tr.a == 1000 and tr.e == -1000
                          and tr.b == 0 and tr.d == 0 and tr.c % 1000 == 0 and tr.f % 1000 == 0
                          and t.count == 1 and t.transform == tr and t.shape == s.shape)
        A = s.read().astype(np.int32); D = t.read(1).astype(np.int32); nod = s.nodata
    nodm = np.zeros(A.shape, bool) if nod is None else (A == int(nod))
    Am = np.where(nodm, 0, A)
    tot = Am.sum(0); cover = tot > 0
    g["L1 values"] = int(Am.max()) <= 100
    say(f"values after no-data masking: max {int(Am.max())}; cells all no-data {int(nodm.all(0).sum()):,}; "
        f"cells with some no-data {int(nodm.any(0).sum()):,}; cells with any cover {int(cover.sum()):,} of {cover.size:,}")
    if cover.any():
        q = np.percentile(tot[cover], [0, 1, 50, 99, 100])
        say(f"  sum of the 21 bands in covered cells: min {q[0]:.0f}, 1% {q[1]:.0f}, median {q[2]:.0f}, 99% {q[3]:.0f}, "
            f"max {q[4]:.0f}; cells summing above 100: {int((tot > 100).sum()):,}; below 100: {int((cover & (tot < 100)).sum()):,}")
    v, c = np.unique(D[~cover], return_counts=True)
    say(f"  dominant file in cells with no cover (unmapped, e.g. open sea): values {dict(zip(v.tolist()[:6], c.tolist()[:6]))}")
    top = Am.max(0); one = cover & ((Am == top[None]).sum(0) == 1)
    arg = Am.argmax(0) + 1
    agree = float((arg[one] == D[one]).mean()) if one.any() else 0.0
    g["L1 band order"] = agree >= AGREE_MIN
    say(f"band order (band k = class k): dominant class = largest band in {agree:.5f} of {int(one.sum()):,} covered cells "
        f"with a single largest band (needs {AGREE_MIN})")
    for k, ok in g.items():
        say(f"{k}: {'PASS' if ok else 'FAIL'}")
    say(f"stage P3c-1: gates {sum(g.values())} PASS, {len(g) - sum(g.values())} FAIL; read-only, nothing written")
    return g


def write_planted(d, order=None, nodata=None, pixel=1000, put255=False, seed=3):
    import rasterio
    from rasterio.transform import from_origin
    rng = np.random.default_rng(seed); H, W = 30, 20
    A = np.zeros((21, H, W), np.uint8); D = np.zeros((H, W), np.uint8)
    for i in range(H):
        for j in range(W):
            if j >= 15:                                        # unmapped sea: no cover, dominant 0
                continue
            k = rng.choice(21, size=3, replace=False); p = np.sort(rng.dirichlet([1, 1, 1]))[::-1]
            v = np.round(p * (100 if j < 13 else 60)).astype(int)
            if v[0] == v[1]:
                v[0] += 1
            A[k, i, j] = v; D[i, j] = k[0] + 1
    if put255:
        A[:, 0, 16] = 255
    if order is not None:
        A = A[order]
    tr = from_origin(400000, 300000, pixel, pixel)
    kw = dict(driver="GTiff", height=H, width=W, crs="EPSG:27700", transform=tr, dtype="uint8")
    with rasterio.open(d / PCT, "w", count=21, nodata=nodata, **kw) as f:
        f.write(A)
    with rasterio.open(d / DOM, "w", count=1, **kw) as f:
        f.write(D, 1)


def self_test():
    res = []
    def case(name, ok):
        res.append(ok); print(f"{name}: {'PASS' if ok else 'FAIL'}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, kw in enumerate([{}, dict(order=np.arange(21)[::-1]), dict(nodata=255, put255=True), dict(put255=True),
                                dict(pixel=500)]):
            d = tmp / f"c{k}"; d.mkdir(); write_planted(d, **kw)
            g = check(d, want_md5=False, quiet=True)
            if k == 0:
                case("T1 planted map, right band order: every gate PASS", all(g.values()))
            if k == 1:
                case("T2 bands in reverse order: band-order gate FAILs", not g["L1 band order"] and g["L1 header"])
            if k == 2:
                case("T3 no-data 255 declared and masked: values gate PASS", g["L1 values"] and g["L1 band order"])
            if k == 3:
                case("T4 255 not declared as no-data: values gate FAILs", not g["L1 values"])
            if k == 4:
                case("T5 500 m pixels: header gate FAILs", not g["L1 header"])
        case("T6 only the two map files opened", all(Path(p).name in (PCT, DOM) for p in OPENED))
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../data/lcm2019_1km/data")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    g = check(a.dir)
    sys.exit(0 if all(g.values()) else 1)