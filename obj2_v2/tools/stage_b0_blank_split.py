#!/usr/bin/env python3
"""
stage_b0_blank_split.py  (obj2_v2, habitat step B-0, 14 Nov 2026; Sara's yes on 14 Nov; rule in the H rules doc, section
"Habitat checks B"; report only)

Question: on real_val (development data), how do the cross-fitted land-only AOO results of P4-1 differ between the
stacks whose habitat channel was shown and the stacks where it was blanked, as every BNM stack is? Two readings: (1) does
a habitat map of training strength (K2 0.80) shrink the excess at truly 1-cell ranges, the CR end? If it does not, a real
habitat map (steps B-1, B-2) is not expected to shrink it either. (2) Does the blanked half alone keep G3's rule? The P4
fits were fitted on both halves together, and every BNM product uses the blank mode. Nothing is refitted or changed. No
truth file is opened (P4-1 stored the land-only truth); nothing of real_eval or BNM is read.

Inputs (read only)
    ../data/stage_p4_1/p4_1_crossfit.npz and p4_1_report.json (Sara's P4-1 version 2 run, 14 Nov: 371f424a, 5e980763)
    ../data/stage_d_frozen_val.npz (blank flag of each frozen stack; its md5 = the one P4-1 checked)

Rule (written before the run)
    Halves: blank = 1 (habitat channel blanked) and blank = 0 (shown); the frozen set blanks exactly half of each recorded
    class (stage D). Methods: seed 1, seed 2, pooled, B3, B4, B5, as stored by P4-1 (rule E: a crossed interval is a miss).
    Per method and half: G3's evaluate (ratio, coverage, median width) by recorded class, records over 6 years and
    land-only true class; G3's rule_pass on the half alone. Option B mean P(AOO <= x), x 1, 5, 10, 20, against the observed
    share at recorded classes 0 and 1; mean option B P(AOO <= 1) and P(AOO <= 5) at land-only true class 1 and 2-5.
    Differences shown minus blanked, 95% interval from 2,000 resamples of the real_val communities (numpy seed 31), the same
    draws for every statistic: ratio at true class 1 and 2-5, ratio at recorded class 0 and 1, P(AOO <= 1) and P(AOO <= 5)
    at true class 1.
    R0a (pooled, the product line; seeds beside): "habitat of training strength shrinks the true 1-cell excess" when the
    pooled shown-minus-blanked ratio at true class 1 has its interval below 0 and is also below B4's and below B5's (no
    habitat input; pooled minus B4 and pooled minus B5 on the same draws, both intervals below 0); otherwise "not expected
    to shrink it". (Changed in the sandbox before any real value was read: the first wording asked B4's and B5's own
    intervals to hold 0, which failed by chance in 3 of 20 planted sets of communities with no habitat effect.)
    R0b: per method, the blanked half alone passes or fails G3's rule (ratio 0.8-1.25, coverage 0.85-0.95 in every
    recorded class and records bin with 50 or more stacks). Report only; Sara decides what follows.

Gates (if any fails, nothing is written)
    C0 inputs: p4_1_report.json md5 starts 5e980763, its gates all PASS, crossed rule "empty"; p4_1_crossfit.npz md5 = the
       report's and starts 371f424a; stage_d_frozen_val.npz md5 = the report's frozen md5
    C1 alignment: frozen rec_t = stored rec_t, frozen class = stored recorded class, same stack count; blank is 0 or 1;
       blanked stacks in each recorded class = its stacks // 2
    C2 code: on all stacks, this tool's evaluate rows (three groupings) = the report's rows for every method (labels, n;
       ratio, coverage, width within 1e-12), its rule_pass = the report's verdict; option B Brier, mean P and share per
       recorded class at every x = the report's (1e-12)
    C3 halves: per method, the sums of the corrected mean, of covered stacks and of stacks over the two halves = all stacks
    C4 resampling: a second run of the resampling gives the same intervals
    C5 np.load log: only the two npz inputs were opened

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, under a minute; numpy only)
    python3 ../tools/stage_b0_blank_split.py --self-test 2>&1 | tail -n 1
    python3 -u ../tools/stage_b0_blank_split.py > ../logs/stage_b0_blank_split.txt 2>&1
The self-test lives in ../tests/test_34_b0_blank_split.py and loads stage_g_calibrate.py and stage_g_recalibrate.py.
"""
import argparse, csv, hashlib, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TCLS = ("0", "1", "2-5", "6-10", "11-20", "21-50", "51+")        # stage D classes (recorded and true)
BINS = ("1-3", "4-6", "7-10", "11-20", "21+")                    # records over 6 years
UPPER = np.array([0, 1, 5, 10, 20, 50])                          # class upper bounds; 51+ is the last class
BAND, COV, N_TEST, XS = (0.8, 1.25), (0.85, 0.95), 50, (1, 5, 10, 20)
DRAWS, SEED, TOL = 2000, 31, 1e-12
EXPECT = dict(report="5e980763", crossfit="371f424a")            # Sara's P4-1 version 2 run, 14 Nov
ORDER_HINT = ("pooled", "B3", "B4", "B5")


def broken():
    return os.environ.get("STAGE_B0_BROKEN", "")    # self-test only: plants one bug so its check must fail


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tidx(v):
    """Stage D class index of a count: 0, 1, 2-5, 6-10, 11-20, 21-50, 51+."""
    return np.searchsorted(UPPER, np.asarray(v), side="left")


def rname(k):
    """G5a's report_name."""
    return k.replace("_", " ", 1) if k.startswith("model_") else k


class LoadLog:
    def __enter__(self):
        self.orig, self.paths = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        self.paths.append(str(Path(f).resolve()) if isinstance(f, (str, Path)) else str(f))
        return self.orig(f, *a, **k)

    def __exit__(self, *e):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- G3's code, copied word for word (stage_g_calibrate.py)
def evaluate(y, out, groups):
    """Ratio, coverage and median width per evaluation group."""
    cov = (y >= out["L"]) & (y <= out["U"])
    if broken() == "cov_strict":
        cov = (y > out["L"]) & (y < out["U"])
    wid = out["U"] - out["L"]
    rows = {}
    for gname, (gi, labels) in groups.items():
        rows[gname] = []
        for j, lab in enumerate(labels):
            m = gi == j
            if m.any():
                rows[gname].append(dict(group=lab, n=int(m.sum()), ratio=float(out["mean"][m].mean() / y[m].mean()) if y[m].mean() > 0
                                        else float("nan"), cov=float(cov[m].mean()), width=float(np.median(wid[m]))))
    return rows


def rule_pass(rows):
    bad = [f"{g} {r['group']}" for g in ("recorded class", "records over 6 years") for r in rows[g] if r["n"] >= N_TEST and
           not (BAND[0] <= r["ratio"] <= BAND[1] and COV[0] <= r["cov"] <= COV[1])]
    return len(bad) == 0, bad


# ---------------------------------------------------------------- pieces
def close(a, b, tol=TOL):
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= tol


def rows_equal(mine, ref):
    """Largest difference between two evaluate outputs (inf if labels or counts differ)."""
    worst = 0.0
    for g in mine:
        if g not in ref or len(mine[g]) != len(ref[g]):
            return float("inf")
        for a, b in zip(mine[g], ref[g]):
            if a["group"] != b["group"] or a["n"] != b["n"]:
                return float("inf")
            for q in ("ratio", "cov", "width"):
                if not close(a[q], b[q], 0.0):
                    worst = max(worst, abs(float(a[q]) - float(b[q])) if np.isfinite(float(a[q]) - float(b[q])) else float("inf"))
    return worst


def boot(parts, comm, draws=DRAWS, seed=SEED, return_draws=False):
    """Differences shown minus blanked of ratio statistics, resampling communities. parts: name -> (num, den, shown) with
    num, den per stack (den 0 outside the statistic's group) and shown the half mask. The same draws serve every
    statistic. Returns name -> point, lo, hi (2.5 and 97.5 percentiles)."""
    cu, ci = np.unique(comm, return_inverse=True)
    rng = np.random.default_rng(None if broken() == "boot_noseed" else seed)
    Wt = np.stack([np.bincount(rng.integers(0, len(cu), len(cu)), minlength=len(cu)) for _ in range(draws)]).astype(float)
    out, raw = {}, {}
    for k, (num, den, shown) in parts.items():
        s = [np.bincount(ci, weights=np.where(h, v, 0.0), minlength=len(cu)) for h in (shown, ~shown) for v in (num, den)]
        with np.errstate(invalid="ignore", divide="ignore"):
            pt = s[0].sum() / s[1].sum() - s[2].sum() / s[3].sum()
            d = (Wt @ s[0]) / (Wt @ s[1]) - (Wt @ s[2]) / (Wt @ s[3])
        ok = d[np.isfinite(d)]
        lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
        out[k] = dict(point=float(pt), lo=float(lo), hi=float(hi), draws=int(len(ok)))
        raw[k] = d
    return (out, raw) if return_draws else out


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- run
def run(p41="../data/stage_p4_1", data="../data", out_dir="../data/stage_b0_blank_split", expect=None, out=print):
    expect = dict(EXPECT, **(expect or {}))
    p41, data, od = Path(p41), Path(data), Path(out_dir)
    if od.exists():
        sys.exit(f"STOP: {od} exists; B-0 writes into a new folder")
    if any(t in str(od.resolve()) for t in ("real_eval", "_truth", "bnm_")):
        sys.exit("STOP: the output path may not contain real_eval, _truth or bnm_")
    for p_ in (p41, data / "real_val"):
        if od.resolve() == p_.resolve() or p_.resolve() in od.resolve().parents:
            sys.exit(f"STOP: nothing is written inside the input folder {p_}")
    f_cf, f_rep, f_fz = p41 / "p4_1_crossfit.npz", p41 / "p4_1_report.json", data / "stage_d_frozen_val.npz"
    miss = [str(f) for f in (f_cf, f_rep, f_fz) if not f.is_file()]
    if miss:
        sys.exit("STOP: input files not found: " + ", ".join(miss))
    t0, res, rep = time.time(), {}, dict(lines=[])

    def say(s):
        rep["lines"].append(s); out(s)

    def gate(name, ok, text):
        res[name] = bool(ok); say(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(at):
        n_ok = sum(res.values())
        say(f"stage B-0: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped at {at}, nothing written")
        return res, rep
    say(f"stage_b0_blank_split.py md5 {md5(__file__)}")
    # ================================================================ C0 inputs
    m = dict(report=md5(f_rep), crossfit=md5(f_cf), frozen=md5(f_fz))
    r4 = json.loads(f_rep.read_text())
    ok0 = m["report"].startswith(expect["report"]) and bool(r4.get("gates")) and all(r4["gates"].values()) \
        and r4.get("crossed") == "empty" and m["crossfit"] == r4["md5"]["crossfit"] and m["crossfit"].startswith(expect["crossfit"]) \
        and m["frozen"] == r4["md5"]["frozen"]
    gate("C0", ok0, f"p4_1_report.json {m['report'][:8]} (expected {expect['report']}), gates all PASS "
         f"{bool(r4.get('gates')) and all(r4['gates'].values())}, crossed rule {r4.get('crossed')}; p4_1_crossfit.npz {m['crossfit'][:8]} = "
         f"report {m['crossfit'] == r4['md5']['crossfit']}; stage_d_frozen_val.npz {m['frozen'][:8]} = report {m['frozen'] == r4['md5']['frozen']}")
    if not res["C0"]:
        return stop("C0")
    with LoadLog() as lg:
        z = dict(np.load(f_cf))
        fz = dict(np.load(f_fz))
        if broken() == "extra_load":
            np.load(p41 / "p4_fits.npz")
    # ================================================================ C1 alignment
    keys = sorted(k[5:] for k in z if k.startswith("mean_"))
    order = [k for k in keys if k.startswith("model_")] + [k for k in ORDER_HINT if k in keys]
    yl, rec_t, rcls, rbin, comm = (z[q] for q in ("true_land", "rec_t", "rcls", "rbin", "comm"))
    yl = yl.astype(float)
    n = len(yl)
    blank = fz["blank"].astype(int)
    per_cls = [(int((rcls == c).sum()), int(blank[rcls == c].sum())) for c in range(len(TCLS))]
    ok1 = len(fz["row"]) == n and np.array_equal(fz["rec_t"], rec_t) and np.array_equal(fz["cls"].astype(int), np.asarray(z["rcls"]).astype(int)) \
        and set(np.unique(blank).tolist()) <= {0, 1} and all(b == c // 2 for c, b in per_cls) and len(order) == 6
    gate("C1", ok1, f"{n:,} frozen stacks; methods {order}; frozen rec_t and class = P4-1's; blanked {int(blank.sum()):,}, shown "
         f"{int((blank == 0).sum()):,}; blanked per recorded class = its stacks // 2 {all(b == c // 2 for c, b in per_cls)} "
         f"({'; '.join(f'{TCLS[c]} {b}/{t}' for c, (t, b) in enumerate(per_cls))})")
    if not res["C1"]:
        return stop("C1")
    tcl = tidx(yl.astype(int))
    if broken() == "rows_shift":
        rcls = np.roll(rcls, 1)
    groups = lambda msk: {"recorded class": (rcls[msk], TCLS), "records over 6 years": (rbin[msk], BINS), "true class": (tcl[msk], TCLS)}
    cf = {k: dict(mean=z[f"mean_{k}"].astype(float), L=z[f"L_{k}"].astype(float), U=z[f"U_{k}"].astype(float), pb=z[f"p_b_{k}"].astype(float))
          for k in order}
    allm = np.ones(n, bool)
    # ================================================================ C2 code against P4-1's report
    worst, why, pworst = 0.0, [], 0.0
    for k in order:
        mine = evaluate(yl, {q: cf[k][q] for q in ("mean", "L", "U")}, groups(allm))
        d = rows_equal(mine, r4["rows"][rname(k)])
        okr = rule_pass(mine)[0] == bool(r4["chosen"][rname(k)]["ok"])
        if not okr:
            d = float("inf")
        worst = max(worst, d); why.append(f"{rname(k)} {d:.1e}")
        for j, x in enumerate(XS):
            P, o = cf[k]["pb"][:, j], (yl <= x).astype(float)
            ref = r4["p_le"][rname(k)]["b"][str(x)]
            pworst = max(pworst, abs(float(((P - o) ** 2).mean()) - float(ref["brier"])))
            for g in np.unique(rcls):
                pc = ref["per_class"].get(TCLS[g])
                cur = (float(P[rcls == g].mean()), float(o[rcls == g].mean()), int((rcls == g).sum()))
                if pc is None or int(pc[2]) != cur[2]:
                    pworst = float("inf"); continue
                pworst = max(pworst, abs(cur[0] - float(pc[0])), abs(cur[1] - float(pc[1])))
    gate("C2", worst <= TOL and pworst <= TOL, f"on all stacks this tool's G3 evaluate and rule_pass = p4_1_report.json rows and verdicts "
         f"(largest difference {worst:.1e}; {'; '.join(why)}); option B Brier, mean P and share per recorded class at x "
         f"{', '.join(map(str, XS))} = the report (largest difference {pworst:.1e})")
    if not res["C2"]:
        return stop("C2")
    # ================================================================ halves
    halves = {"blanked": blank == 1, "shown": blank == 0}
    if broken() == "halves_swap":
        halves = {"blanked": blank == 0, "shown": blank == 1}
    rows, rules, pcls, ptrue, ok3 = {}, {}, {}, {}, True
    for k in order:
        rows[k], rules[k], pcls[k], ptrue[k] = {}, {}, {}, {}
        cov_all = (yl >= cf[k]["L"]) & (yl <= cf[k]["U"])
        sums = np.zeros(3)
        for h, msk in halves.items():
            sub = {q: cf[k][q][msk] for q in ("mean", "L", "U")}
            rows[k][h] = evaluate(yl[msk], sub, groups(msk))
            okh, bad = rule_pass(rows[k][h])
            rules[k][h] = dict(ok=okh, bad=bad)
            sums += [cf[k]["mean"][msk].sum(), cov_all[msk].sum(), msk.sum()]
            pcls[k][h] = {str(x): {TCLS[g]: [float(cf[k]["pb"][msk & (rcls == g), j].mean()), float((yl[msk & (rcls == g)] <= x).mean()),
                                             int((msk & (rcls == g)).sum())] for g in (0, 1) if (msk & (rcls == g)).any()}
                          for j, x in enumerate(XS)}
            ptrue[k][h] = {lab: dict(n=int((msk & (tcl == c)).sum()),
                                     **{f"x{x}": float(cf[k]["pb"][msk & (tcl == c), j].mean()) if (msk & (tcl == c)).any() else float("nan")
                                        for j, x in ((0, 1), (1, 5))})
                           for c, lab in ((1, "1"), (2, "2-5"))}
        ok3 &= abs(sums[0] - cf[k]["mean"].sum()) <= 1e-6 * max(1.0, abs(cf[k]["mean"].sum())) and sums[1] == cov_all.sum() and sums[2] == n
    gate("C3", ok3, "per method, corrected mean, covered stacks and stacks summed over the two halves = all stacks")
    # ================================================================ shown minus blanked, communities resampled
    shown = halves["shown"]
    one = np.ones(n)

    def parts(k):
        mm = cf[k]["mean"]
        p = {f"ratio true {lab}": (np.where(tcl == c, mm, 0.0), np.where(tcl == c, yl, 0.0), shown) for c, lab in ((1, "1"), (2, "2-5"))}
        p.update({f"ratio recorded {TCLS[c]}": (np.where(rcls == c, mm, 0.0), np.where(rcls == c, yl, 0.0), shown) for c in (0, 1)})
        p.update({f"P(AOO<={x}) true 1": (np.where(tcl == 1, cf[k]["pb"][:, j], 0.0), np.where(tcl == 1, one, 0.0), shown) for j, x in ((0, 1), (1, 5))})
        return p
    bt, raw = {}, {}
    for k in order:
        bt[k], raw[k] = boot(parts(k), comm, return_draws=True)
    bt2 = {k: boot(parts(k), comm) for k in order}
    ok4 = all(bt[k][s][q] == bt2[k][s][q] or (np.isnan(bt[k][s][q]) and np.isnan(bt2[k][s][q])) for k in order for s in bt[k]
              for q in ("point", "lo", "hi"))
    gate("C4", ok4, f"a second run of the resampling ({DRAWS:,} draws of {len(np.unique(comm))} communities, seed {SEED}) gives the same intervals")
    allowed = {str(f_cf.resolve()), str(f_fz.resolve())}
    gate("C5", set(lg.paths) <= allowed and len(lg.paths) == 2, f"np.load log: {len(lg.paths)} files, all of them the two inputs "
         f"{set(lg.paths) <= allowed}; no truth, real_eval or BNM file")
    if not all(res.values()):
        return stop(next(k_ for k_, v_ in res.items() if not v_))
    # ================================================================ readings
    s1 = order[0]
    r0a = {k: bt[k]["ratio true 1"]["hi"] < 0 for k in order}
    did = {}
    for k in order[:3]:
        for c in ("B4", "B5"):
            d = raw[k]["ratio true 1"] - raw[c]["ratio true 1"]
            d = d[np.isfinite(d)]
            lo_, hi_ = (np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan))
            did[(k, c)] = dict(point=bt[k]["ratio true 1"]["point"] - bt[c]["ratio true 1"]["point"], lo=float(lo_), hi=float(hi_))
    ctrl = did[("pooled", "B4")]["hi"] < 0 and did[("pooled", "B5")]["hi"] < 0
    shrink = r0a["pooled"] and ctrl
    # ================================================================ report lines
    say(f"   halves: blanked {int(halves['blanked'].sum()):,}, shown {int(halves['shown'].sum()):,}; land-only true class 1: blanked "
        f"{int((halves['blanked'] & (tcl == 1)).sum())}, shown {int((halves['shown'] & (tcl == 1)).sum())}")
    fmt = lambda rr: "; ".join(f"{r['group']} {r['ratio']:.2f}/{r['cov']:.2f}" + ("" if r["n"] >= N_TEST else "*") for r in rr)
    say("   ratio / coverage by half (cross-fitted, P4 fits; * = under 50 stacks):")
    for k in order:
        for h in halves:
            say(f"      {rname(k):<14} {h:<7} recorded class {fmt(rows[k][h]['recorded class'])} | records {fmt(rows[k][h]['records over 6 years'])}")
            say(f"      {'':<14} {'':<7} true class {fmt(rows[k][h]['true class'])}")
    say("   R0b, G3's rule on each half alone: " + "; ".join(f"{rname(k)} blanked {'PASS' if rules[k]['blanked']['ok'] else 'FAIL'}"
        + ("" if rules[k]["blanked"]["ok"] else f" ({', '.join(rules[k]['blanked']['bad'][:4])})") + f", shown {'PASS' if rules[k]['shown']['ok'] else 'FAIL'}"
        for k in order))
    say("   option B mean P vs observed share, recorded classes 0 and 1 (blanked | shown):")
    for k in order[:3]:
        say(f"      {rname(k):<14} " + "; ".join(f"x {x}: " + ", ".join(
            f"{g} {pcls[k]['blanked'][str(x)][g][0]:.3f} vs {pcls[k]['blanked'][str(x)][g][1]:.3f} | {pcls[k]['shown'][str(x)][g][0]:.3f} vs "
            f"{pcls[k]['shown'][str(x)][g][1]:.3f}" for g in ("0", "1") if g in pcls[k]["blanked"][str(x)] and g in pcls[k]["shown"][str(x)]) for x in XS))
    say("   option B mean P at land-only true class 1 and 2-5 (blanked | shown): " + "; ".join(
        f"{rname(k)} " + ", ".join(f"true {lab} P(<=1) {ptrue[k]['blanked'][lab]['x1']:.3f} | {ptrue[k]['shown'][lab]['x1']:.3f}, P(<=5) "
                                   f"{ptrue[k]['blanked'][lab]['x5']:.3f} | {ptrue[k]['shown'][lab]['x5']:.3f}" for lab in ("1", "2-5")) for k in order))
    say("   shown minus blanked [95% interval, communities resampled]:")
    for k in order:
        say(f"      {rname(k):<14} " + "; ".join(f"{s} {b['point']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]" for s, b in bt[k].items()))
    say("   difference beyond the methods without habitat input (ratio at true class 1, shown minus blanked, minus B4's or B5's on "
        "the same draws): " + "; ".join(f"{rname(k)} - {c} {v['point']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" for (k, c), v in did.items()))
    say(f"   R0a: pooled ratio at true class 1, shown minus blanked, interval below 0 {r0a['pooled']} (seed 1 {r0a[s1]}, seed 2 "
        f"{r0a[order[1]]}); below B4's and B5's {ctrl} -> habitat of training strength "
        f"{'SHRINKS the true 1-cell excess' if shrink else 'is NOT expected to shrink the true 1-cell excess'}")
    # ================================================================ files
    od.mkdir(parents=True)
    with open(od / "b0_rows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "half", "grouping", "group", "n", "ratio", "coverage", "median_width"])
        for k in order:
            for h in halves:
                for g, rr in rows[k][h].items():
                    for r in rr:
                        w.writerow([rname(k), h, g, r["group"], r["n"], f"{r['ratio']:.12g}", f"{r['cov']:.12g}", f"{r['width']:.12g}"])
    n_ok = sum(res.values())
    last = (f"stage B-0: gates {n_ok} PASS, {len(res) - n_ok} FAIL; R0a: habitat of training strength "
            f"{'shrinks' if shrink else 'is not expected to shrink'} the true 1-cell excess; R0b blanked half alone: "
            + ", ".join(f"{rname(k)} {'PASS' if rules[k]['blanked']['ok'] else 'FAIL'}" for k in order))
    say(last)
    rep.update(gates=res, written_utc=now(), md5=dict(m, tool=md5(__file__), rows_csv=md5(od / "b0_rows.csv")), n=int(n),
               halves={h: int(v.sum()) for h, v in halves.items()},
               rows={rname(k): v for k, v in rows.items()}, rule={rname(k): v for k, v in rules.items()},
               p_class={rname(k): v for k, v in pcls.items()}, p_true={rname(k): v for k, v in ptrue.items()},
               boot={rname(k): v for k, v in bt.items()}, readings=dict(R0a=dict(pooled=r0a["pooled"], seeds=[r0a[s1], r0a[order[1]]],
               beyond_B4_B5=ctrl, shrinks=shrink, did={f"{rname(k)} - {c}": v for (k, c), v in did.items()}), R0b={rname(k): rules[k]["blanked"]["ok"] for k in order}),
               minutes=round((time.time() - t0) / 60, 2))
    (od / "b0_report.json").write_text(json.dumps(rep, indent=1))
    out(f"   b0_rows.csv md5 {rep['md5']['rows_csv']}; b0_report.json md5 {md5(od / 'b0_report.json')}")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--p41", default="../data/stage_p4_1")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../data/stage_b0_blank_split")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_34_b0_blank_split.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_34_b0_blank_split", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.main(Path(__file__)) else 1)
    r, _ = run(a.p41, a.data, a.out, out=lambda s: print(s, flush=True))
    sys.exit(0 if r and all(r.values()) else 1)