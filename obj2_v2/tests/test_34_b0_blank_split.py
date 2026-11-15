#!/usr/bin/env python3
"""
test_34_b0_blank_split.py  (obj2_v2, self-test of tools/stage_b0_blank_split.py, 14 Nov 2026)

Planted P4-1 outputs (p4_1_crossfit.npz, p4_1_report.json) and a planted frozen set with the blank flag. The planted
report is made with the project's own G3 evaluate / rule_pass (stage_g_calibrate.py) and G3b evaluate
(stage_g_recalibrate.py) loaded from the tool's folder, so gate C2 compares the tool's copies with them (on Sara's
machine: her real tools). Three planted communities sets: A, habitat shown shrinks the true 1-cell excess of the model
methods; B, no habitat effect; C, as A, but the blanked half of the model methods is 35% high at recorded class 2-5.

Checks
    U1 the tool's evaluate and rule_pass = G3's on random data with ties and a group whose truth is 0
    U2 the tool's class index = an explicit if-chain for counts 0-400
    U3 the community resampling = the same draws applied by repeating stacks
    T1 world A: gates C0-C5 PASS; R0a "shrinks" (pooled below 0 and below B4 and B5); R0b PASS for every method
    T2 world B: R0a "not expected to shrink"
    T3 world C: R0b FAIL for the model methods and B3 (blanked half), PASS for B4 and B5; shown half PASS everywhere
    T4 fresh code: every pooled ratio and coverage by half and recorded class recomputed by plain loops = b0_rows.csv;
       a planted swap of the halves is caught this way
    T5 refusals: an existing output folder stops; a crossfit file changed after the report fails C0; one extra blanked
       stack fails C1; nothing written in either case
    B1-B4 planted bugs: coverage with strict inequalities (C2), recorded classes shifted (C2), resampling without a seed
       (C4), an extra file opened (C5); nothing written
Run: python3 ../tools/stage_b0_blank_split.py --self-test
"""
import csv, importlib.util, json, os, sys, tempfile
from pathlib import Path
import numpy as np


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def cls_chain(v):
    v = int(v)
    if v == 0:
        return 0
    if v == 1:
        return 1
    if v <= 5:
        return 2
    if v <= 10:
        return 3
    if v <= 20:
        return 4
    if v <= 50:
        return 5
    return 6


def world(base, B, GC, GR, effect=True, blank_off=False, seed=0):
    """Planted P4-1 folder, frozen set and report. Returns (p41, data)."""
    g = np.random.default_rng(seed)
    nc, per = 20, 600
    n = nc * per
    comm = np.repeat(np.arange(201, 201 + nc), per)
    cu = np.unique(comm); first = set(cu[g.permutation(nc)[:nc // 2]].tolist())
    fold = np.array([0 if c in first else 1 for c in comm])
    rcls = g.choice(7, n, p=[0.12, 0.18, 0.16, 0.12, 0.12, 0.15, 0.15])
    lo, hi = np.array([0, 1, 2, 6, 11, 21, 51]), np.array([0, 1, 5, 10, 20, 50, 120])
    rec_t = lo[rcls] + np.floor(g.random(n) * (hi[rcls] - lo[rcls] + 1)).astype(int)
    u = g.random(n)
    extra = np.where(rcls == 0, np.where(u < 0.15, 0, g.geometric(0.3, n)),
                     np.where((rcls == 1) & (u < 0.25), 0, np.round(rec_t * g.gamma(2.0, 0.4, n)) + g.geometric(0.2, n)))
    y = np.minimum(rec_t + extra, 400).astype(float)
    rec6 = np.maximum(rec_t, 1) + g.integers(0, 30, n)
    rbin = np.searchsorted([3, 6, 10, 20], rec6, side="left")
    tcl = np.array([cls_chain(v) for v in y])
    blank = np.zeros(n, np.uint8)
    for c in range(7):
        idx = np.nonzero(rcls == c)[0]
        blank[g.permutation(idx)[:len(idx) // 2]] = 1
    sh = blank == 0
    idx = np.arange(n)
    covd = (idx % 10) != 3
    L = np.where(covd, np.where(idx % 2 == 0, y, np.maximum(y - 2, 0)), y + 1)
    U = np.where(covd, y + 3, y + 4)
    methods = ["model_f_seed1", "model_f_seed2", "pooled", "B3", "B4", "B5"]
    z = dict(true_land=y, true_window=y, rec_t=rec_t, rcls=rcls, rbin=rbin, fold=fold, comm=comm, coastal=np.zeros(n, bool),
             window=np.zeros(n, int))
    for j, k in enumerate(methods):
        habitat = k not in ("B4", "B5")
        noise = np.exp(0.12 * g.standard_normal(n))
        one = 3.0 * np.exp(0.2 * g.standard_normal(n))
        if habitat and effect:
            one = np.where(sh, 0.5 * one, one)
        mean = np.where(tcl == 1, one, np.maximum(y, 0.3) * noise)
        for c in range(7):
            mc = rcls == c
            mean[mc] *= y[mc].sum() / mean[mc].sum()
        if blank_off and habitat:
            mean = np.where((rcls == 2) & ~sh, 1.35 * mean, mean)
        p1 = np.where(y <= 1, 0.5, 0.05)
        if habitat and effect:
            p1 = np.where(tcl == 1, np.where(sh, 0.6, 0.3), p1)
        p5 = np.maximum(p1, np.where(y <= 5, 0.7, 0.1)); p10 = np.maximum(p5, np.where(y <= 10, 0.8, 0.15))
        p20 = np.maximum(p10, np.where(y <= 20, 0.9, 0.2))
        pb = np.stack([p1, p5, p10, p20], 1)
        z.update({f"aoo_land_{k}": np.zeros((n, 2), np.uint16), f"mean_{k}": mean, f"L_{k}": L.astype(float), f"U_{k}": U.astype(float),
                  f"scale_q_{k}": np.ones((n, 2)), f"p_g3_{k}": pb, f"p_recal_{k}": pb, f"p_b_{k}": pb})
    p41, data = Path(base) / "stage_p4_1", Path(base) / "data"
    p41.mkdir(parents=True); data.mkdir(parents=True)
    np.savez(p41 / "p4_1_crossfit.npz", **z)
    np.savez(p41 / "p4_fits.npz", dummy=np.zeros(1))
    np.savez(data / "stage_d_frozen_val.npz", row=np.arange(n), rec_t=rec_t, cls=rcls.astype(np.int8), blank=blank,
             file_index=np.zeros(n, np.int16))
    groups = {"recorded class": (rcls, GC.TCLS), "records over 6 years": (rbin, GC.BINS), "true class": (tcl, GC.TRUE)}
    rows, chosen, ple = {}, {}, {}
    for k in methods:
        rr = GC.evaluate(y, dict(mean=z[f"mean_{k}"], L=z[f"L_{k}"], U=z[f"U_{k}"]), groups)
        ok, bad = GC.rule_pass(rr)
        nm = B.rname(k)
        rows[nm], chosen[nm] = rr, dict(option=1, ok=ok, bad=bad)
        ple[nm] = {"b": {str(x): {q: v for q, v in GR.evaluate(z[f"p_b_{k}"][:, j], y, fold, rcls, x).items() if q != "table"}
                         for j, x in enumerate(GC.XS)}}
    rep = dict(gates=dict(Q0=True, Q1=True, Q2=True, Q3=True, Q4=True, Q5=True, Q6=True), crossed="empty",
               md5=dict(frozen=B.md5(data / "stage_d_frozen_val.npz"), crossfit=B.md5(p41 / "p4_1_crossfit.npz")),
               rows=rows, chosen=chosen, p_le=ple)
    (p41 / "p4_1_report.json").write_text(json.dumps(rep, indent=1))
    return p41, data


def expect_of(p41, B):
    return dict(report=B.md5(p41 / "p4_1_report.json")[:8], crossfit=B.md5(p41 / "p4_1_crossfit.npz")[:8])


def main(tool_path):
    tool_path = Path(tool_path).resolve()
    tools = tool_path.parent
    sys.path.insert(0, str(tools))
    B = _load(tool_path, "stage_b0_blank_split")
    GC = _load(tools / "stage_g_calibrate.py", "stage_g_calibrate")
    GR = _load(tools / "stage_g_recalibrate.py", "stage_g_recalibrate")
    quiet = lambda *a, **k: None
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    def bug(name, envv, fn):
        os.environ["STAGE_B0_BROKEN"] = envv
        try:
            return fn()
        finally:
            os.environ.pop("STAGE_B0_BROKEN", None)

    # ---- U1-U3
    g = np.random.default_rng(3)
    n = 4000
    y = g.integers(0, 8, n).astype(float); y[:300] = 0
    gi = g.integers(0, 7, n); gi[:300] = 6
    out = dict(mean=y + g.normal(0, 1, n), L=np.where(g.random(n) < 0.5, y, y - 1), U=y + g.integers(-1, 3, n))
    grp = {"recorded class": (gi, GC.TCLS), "records over 6 years": (g.integers(0, 5, n), GC.BINS)}
    a, b = B.evaluate(y, out, grp), GC.evaluate(y, out, grp)
    t("U1", B.rows_equal(a, b) == 0.0 and B.rule_pass(a) == GC.rule_pass(b), "the tool's evaluate and rule_pass = G3's on random data "
      "with ties at both interval ends and a group whose truth is 0 (ratio NaN)")
    t("U2", all(int(B.tidx(v)) == cls_chain(v) for v in range(401)), "the tool's class index = an explicit if-chain for counts 0-400")
    comm = np.repeat([5, 9, 12], [4, 3, 5]); sh = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0], bool)
    num, den = g.random(12) * 5, g.random(12) + 0.5
    bres, raw = B.boot({"s": (num, den, sh)}, comm, draws=50, seed=7, return_draws=True)
    rng = np.random.default_rng(7); cu = np.unique(comm); ok3 = True
    for d in range(50):
        pick = cu[rng.integers(0, 3, 3)]
        rep_i = np.concatenate([np.nonzero(comm == c)[0] for c in pick])
        s_, b_ = rep_i[sh[rep_i]], rep_i[~sh[rep_i]]
        ok3 &= abs((num[s_].sum() / den[s_].sum() - num[b_].sum() / den[b_].sum()) - raw["s"][d]) < 1e-12
    t("U3", ok3, "community resampling = the same 50 draws applied by repeating the stacks of the drawn communities")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- T1 world A
        p41, data = world(tmp / "A", B, GC, GR, effect=True, seed=1)
        ea = expect_of(p41, B)
        r, rep = B.run(p41, data, tmp / "A" / "out", expect=ea, out=quiet)
        rd = rep.get("readings", {})
        bt = rep.get("boot", {})
        t("T1", r and all(r.values()) and len(r) == 6 and rd.get("R0a", {}).get("shrinks") is True
          and rd.get("R0a", {}).get("beyond_B4_B5") is True and all(rd.get("R0b", {}).values()),
          f"world A: gates {r}; R0a {rd.get('R0a')}; pooled true-1 ratio shown - blanked {bt.get('pooled', {}).get('ratio true 1')}; "
          f"B4 {bt.get('B4', {}).get('ratio true 1')}; R0b {rd.get('R0b')}")
        # ---- T2 world B
        p41b, datab = world(tmp / "B", B, GC, GR, effect=False, seed=2)
        r, rep = B.run(p41b, datab, tmp / "B" / "out", expect=expect_of(p41b, B), out=quiet)
        rd = rep.get("readings", {})
        t("T2", r and all(r.values()) and rd.get("R0a", {}).get("shrinks") is False,
          f"world B (no habitat effect): R0a {rd.get('R0a')}; pooled {rep.get('boot', {}).get('pooled', {}).get('ratio true 1')}")
        # ---- T3 world C
        p41c, datac = world(tmp / "C", B, GC, GR, effect=True, blank_off=True, seed=3)
        r, rep = B.run(p41c, datac, tmp / "C" / "out", expect=expect_of(p41c, B), out=quiet)
        rb = rep.get("readings", {}).get("R0b", {})
        shown_ok = all(v["shown"]["ok"] for v in rep.get("rule", {}).values())
        t("T3", r and all(r.values()) and [rb.get(B.rname(k)) for k in ("model_f_seed1", "model_f_seed2", "pooled", "B3", "B4", "B5")]
          == [False, False, False, False, True, True] and shown_ok,
          f"world C (blanked half 35% high at recorded class 2-5): R0b blanked half {rb}; shown half all PASS {shown_ok}")
        # ---- T4 fresh code on world A's csv; the halves swap is caught the same way
        def fresh_ok(out_dir):
            z = np.load(p41 / "p4_1_crossfit.npz"); fz = np.load(data / "stage_d_frozen_val.npz")
            yy, rc, mean, L, U, bl = z["true_land"], z["rcls"], z["mean_pooled"], z["L_pooled"], z["U_pooled"], fz["blank"]
            got = {(r_["half"], r_["group"]): r_ for r_ in csv.DictReader(open(out_dir / "b0_rows.csv"))
                   if r_["method"] == "pooled" and r_["grouping"] == "recorded class"}
            worst, cnt = 0.0, 0
            for half, hv in (("blanked", 1), ("shown", 0)):
                for c in range(7):
                    sm, sy, cv, nn = 0.0, 0.0, 0, 0
                    for i in range(len(yy)):
                        if bl[i] == hv and rc[i] == c:
                            sm += mean[i]; sy += yy[i]; cv += int(L[i] <= yy[i] <= U[i]); nn += 1
                    r_ = got.get((half, GC.TCLS[c]))
                    if r_ is None or int(r_["n"]) != nn:
                        return False, float("inf"), cnt
                    worst = max(worst, abs(float(r_["ratio"]) - sm / sy), abs(float(r_["coverage"]) - cv / nn)); cnt += 1
            return worst < 1e-9, worst, cnt
        ok4, w4, c4 = fresh_ok(tmp / "A" / "out")
        bug("halves_swap", "halves_swap", lambda: B.run(p41, data, tmp / "A" / "out_swap", expect=ea, out=quiet))
        ok4s, _, _ = fresh_ok(tmp / "A" / "out_swap")
        t("T4", ok4 and not ok4s, f"pooled ratio and coverage of {c4} half-class rows by plain loops = b0_rows.csv (largest difference "
          f"{w4:.1e}); a planted swap of the halves is caught by the same recomputation {not ok4s}")
        # ---- T5 refusals
        try:
            B.run(p41, data, tmp / "A" / "out", expect=ea, out=quiet); st = False
        except SystemExit:
            st = True
        p41x, datax = world(tmp / "X", B, GC, GR, seed=1)
        ex = expect_of(p41x, B)
        zz = dict(np.load(p41x / "p4_1_crossfit.npz")); zz["mean_pooled"] = zz["mean_pooled"] * 1.0001
        np.savez(p41x / "p4_1_crossfit.npz", **zz)
        r0, _ = B.run(p41x, datax, tmp / "X" / "out", expect=ex, out=quiet)
        p41y, datay = world(tmp / "Y", B, GC, GR, seed=1)
        fy = dict(np.load(datay / "stage_d_frozen_val.npz"))
        k_ = np.nonzero((fy["cls"] == 2) & (fy["blank"] == 0))[0][0]; fy["blank"][k_] = 1
        np.savez(datay / "stage_d_frozen_val.npz", **fy)
        rj = json.loads((p41y / "p4_1_report.json").read_text()); rj["md5"]["frozen"] = B.md5(datay / "stage_d_frozen_val.npz")
        (p41y / "p4_1_report.json").write_text(json.dumps(rj, indent=1))
        r1, _ = B.run(p41y, datay, tmp / "Y" / "out", expect=expect_of(p41y, B), out=quiet)
        t("T5", st and r0.get("C0") is False and not (tmp / "X" / "out").exists() and r1.get("C1") is False and not (tmp / "Y" / "out").exists(),
          f"existing output folder stops {st}; crossfit changed after the report: C0 {r0.get('C0')}; one extra blanked stack in recorded "
          f"class 2-5: C1 {r1.get('C1')}; nothing written in either case")
        # ---- planted bugs
        for nm, envv, gate_ in (("B1", "cov_strict", "C2"), ("B2", "rows_shift", "C2"), ("B3", "boot_noseed", "C4"), ("B4", "extra_load", "C5")):
            od = tmp / "A" / f"out_{envv}"
            rr, _ = bug(nm, envv, lambda: B.run(p41, data, od, expect=ea, out=quiet))
            t(nm, rr.get(gate_) is False and not od.exists(), f"planted bug {envv}: {gate_} {rr.get(gate_)}, nothing written {not od.exists()}")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)


if __name__ == "__main__":
    tp = Path(__file__).resolve().parent.parent / "tools" / "stage_b0_blank_split.py"
    sys.exit(0 if main(tp) else 1)