#!/usr/bin/env python3
"""
stage_t_tiebreak.py  (obj2_v2, rare-end check T, 15 Nov 2026; Sara's yes to T on 15 Nov; rule in claude/obj2_v2_H_rules_part2.md;
report only)

Question: on the BNM fit half (development data), when the model's hole ranking keeps its own order and B1 exact only orders
the holes the model scores the same, do the sparsely recorded groups reach B1 in H-R1 while the well-recorded groups keep
their gain? Nothing is fitted (no parameter, no folds). The model, its samples, h3_freeze.json, the P4 fits and every product
stay as they are: T changes the order of cells in a map, never an AOO, interval or P. Nothing of the test half, H5, H6 or
real_eval is read.

Score
    score_T = model score + r / (2 x 50). r = rank of B1 exact among the stack-repeat's hole tetrads (0-based, ties share
    their mean rank) divided by the number of holes, so 0 <= r < 1 and the added term is below 1/100. The model score is the
    share of 50 samples occupied (a multiple of 1/50, stored as float32; smallest stored gap 0.0199999), so the added term
    never changes the model's own order. Holes equal on both stay tied.

Rule (written 15 Nov before any T value; the F0 shape)
    "T worth using" = YES when, for seed 1 and seed 2, against B1 stored and B1 exact, with H2b's H-R1 code (stack-first means,
    2,000 resamples of the 44 windows, seed 31, floor 50 distinct stacks): (i) no group of either grouping FAILs, and (ii)
    recorded classes 21-50 and 51+ and records 21+ keep a mean of 0 or above. Otherwise NO. Sara decides either way.
    Reported beside it: the unchanged H2b rows; per group the mean of AUC(T) - AUC(model) with its interval; per group the
    share of recorded / unrecorded hole pairs the model ties (the room T has); H3b's sparse set S pooled (kept recorded class
    0-1 in 2019, or 1-10 kept records over 6 years). Known limit: against B1 exact, T gains only on pairs the model ties.

Gates (if one fails, nothing past it is printed and nothing is written)
    V0  inputs: h3_freeze.json md5 = signed (5516a6fc), every md5 listed in it = the files now (H6's freeze_check);
        stage_h6_bnm_test_score.py = 8b9a38dd; stage_f0_spread_check.py content = version 2 (download form 75ce6f24; its loader,
        H-R1 closure and np.load log are used); h2b_report.json gates all PASS
    V1  alignment (F0's A1 checks): recorded holes per stack-repeat from the truth files = h2b y_strict; AUC of seed 1, seed 2,
        B1 stored, B1 exact from the cell data = h2b_scores.npz (1e-12, same not-rankable set); every seed score x 50 a whole
        number (1e-5: float32 storage gives up to 1.4e-6); B1 exact never NaN at a hole
    V2  code: with the added term times 0 the H-R1 rows of both seeds against both B1 versions = h2b_report.json (1e-9, verdicts)
    V3  order: on every stack-repeat two holes with different model scores keep their order and do not tie; AUC(T) = AUC(model)
        on every stack-repeat with no tied recorded / unrecorded pair; with B1 exact set to a constant, AUC(T) = AUC(model)
    V4  np.load log: only files of bnm_fit, stage_h2_predict and stage_h2b_score; a test-half, H5, H6 or real_eval path refused

Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_t_tiebreak.py --self-test 2>&1 | tail -n 1
    python3 -u ../tools/stage_t_tiebreak.py > ../logs/stage_t_tiebreak.txt 2>&1
Needs stage_f0_spread_check.py (version 2) and what it loads in the same folder; the self-test lives in
../tests/test_35_t_tiebreak.py and uses ../tests/test_25_f0a_spread_check.py and ../tests/test_23_h6_test_score.py.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_t_tiebreak.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


F0 = _load("stage_f0_spread_check")
H6, S2, X = F0.H6, F0.S2, F0.X
TCLS, BINS, SEEDS, REFS, NAME, DENSE = F0.TCLS, F0.BINS, F0.SEEDS, F0.REFS, F0.NAME, F0.DENSE
F0_FP = "3599c46ba15995431b8eaba743f07660"          # content fingerprint of stage_f0_spread_check.py version 2 (download form 75ce6f24)
K = 50
WHOLE_TOL = 1e-5


def broken():
    return os.environ.get("STAGE_T_BROKEN", "")      # self-test only: plants one bug so its gate must fail


def fingerprint(p):
    """md5 of the file without a #! line, trailing spaces or final newlines (the block fingerprint)."""
    b = Path(p).read_bytes()
    return hashlib.md5(b"\n".join(l.rstrip() for l in b.split(b"\n") if l[:2] != b"#!").rstrip(b"\n")).hexdigest()


# ---------------------------------------------------------------- pieces
def rank01(ptr, v):
    """Per stack-repeat rank of v among its holes, 0-based, ties at their mean rank, divided by the number of holes."""
    nh = np.diff(ptr)
    item = np.repeat(np.arange(len(nh)), nh)
    o = np.lexsort((v, item))
    vs, it = v[o], item[o]
    pos = np.arange(len(v)) - ptr[it]
    new = np.r_[True, (it[1:] != it[:-1]) | (vs[1:] != vs[:-1])] if broken() != "rank_noties" else np.ones(len(v), bool)
    gid = np.cumsum(new) - 1
    r = np.empty(len(v))
    r[o] = (np.bincount(gid, weights=pos) / np.bincount(gid))[gid]
    return r / nh[item]


def score_t(s, r, mult=1.0):
    return s + mult * r / (2.0 * K if broken() != "big_term" else 10.0)


def tie_share(ptr, s, lab):
    """Per stack-repeat share of recorded / unrecorded hole pairs with equal score (NaN when not rankable)."""
    out = np.full(len(ptr) - 1, np.nan)
    for i in range(len(out)):
        a, b = ptr[i], ptr[i + 1]
        pos, neg = s[a:b][lab[a:b]], np.sort(s[a:b][~lab[a:b]])
        if len(pos) and len(neg):
            out[i] = (np.searchsorted(neg, pos, "right") - np.searchsorted(neg, pos, "left")).sum() / (len(pos) * len(neg))
    return out


def order_kept(ptr, s, st):
    """Along each stack-repeat's holes sorted by score_T the model score never falls, and equal score_T means equal model score."""
    item = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    o = np.lexsort((st, item))
    it, m, t = item[o], s[o], st[o]
    same = it[1:] == it[:-1]
    tie = same & (t[1:] == t[:-1])
    return bool(np.all(m[1:][same] >= m[:-1][same]) and np.all(m[1:][tie] == m[:-1][tie]))


def hr1(Am, Ab, F, floor):
    return F0.hr1(Am + (1e-6 if broken() == "core_shift" else 0.0), Ab, F, floor)


def one_group(d, F, mask, floor, label="S"):
    """Stack-first mean and window-resampled interval over the items in mask, as H2b's 10+ row."""
    return S2.hr1_rows(S2.hr1_core(d, F["stid"], np.where(mask, 0, -1), F["win"], [0]), [label], floor)[0]


def bare(rows):
    """Report-only rows: mean, interval and stack count, without a rule verdict."""
    return [{k: v for k, v in q.items() if k not in ("verdict", "tested")} for q in rows]


def fmt(r):
    return f"{r['group']} {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['verdict']} ({r['n']})" if r["n"] else f"{r['group']} none"


# ---------------------------------------------------------------- run
def run(bnm="../data/bnm_fit", h2a="../data/stage_h2_predict", h2b="../data/stage_h2b_score",
        freeze="../data/stage_h3_freeze/h3_freeze.json", runs=("../runs/f_seed1", "../runs/f_seed2"), obj1_tools="../../obj1_v2/tools",
        out_dir="../data/stage_t_tiebreak", g2="../data/stage_g2_baselines_v2", g3="../data/stage_g3_calibration",
        freeze_md5=F0.FREEZE_MD5, h6_md5=F0.H6_MD5, f0_fp=F0_FP, floor_override=None, out=print):
    bnm, h2a, h2b, freeze, out_dir = Path(bnm), Path(h2a), Path(h2b), Path(freeze), Path(out_dir)
    runs, data = [Path(r) for r in runs], bnm.parent
    o = out_dir.resolve()
    for p_ in (bnm, h2a, h2b):
        if o == p_.resolve() or p_.resolve() in o.parents:
            sys.exit(f"STOP: nothing is written inside {p_}; give another --out")
    if any(t in str(o) for t in F0.FORBID):
        sys.exit("STOP: the output folder may not lie in a test-half, H5, H6 or real_eval folder")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"STOP: {out_dir} already holds outputs; give a new folder")
    t0, res, rep = time.time(), {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def stop(after):
        n_ok = sum(res.values())
        out(f"stage T: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped after {after}, nothing written")
        return res, rep

    out(f"stage_t_tiebreak.py md5 {X.file_md5(__file__)} (fingerprint {fingerprint(__file__)}); stage_f0_spread_check.py "
        f"{X.file_md5(HERE / 'stage_f0_spread_check.py')}; stage_h6_bnm_test_score.py {X.file_md5(HERE / 'stage_h6_bnm_test_score.py')}; "
        f"stage_h2_bnm_score.py {X.file_md5(HERE / 'stage_h2_bnm_score.py')}")
    # ---- V0
    fz = json.loads(freeze.read_text())
    fmd5 = X.file_md5(freeze)
    paths = dict(bnm_fit=bnm, stage_h2_predict=h2a, stage_h2b_score=h2b, stage_g2_baselines_v2=Path(g2), stage_g3_calibration=Path(g3))
    bad, nf = H6.freeze_check(fz, obj1_tools, paths, runs)
    h6ok = X.file_md5(HERE / "stage_h6_bnm_test_score.py") == h6_md5
    f0ok = fingerprint(HERE / "stage_f0_spread_check.py") == f0_fp
    jb = json.loads((h2b / "h2b_report.json").read_text())
    jb_ok = bool(jb.get("gates")) and all(jb["gates"].values())
    floor = int(floor_override) if floor_override is not None else int(fz.get("settings", {}).get("floor", 50))
    gate("V0", fmd5 == freeze_md5 and not bad and h6ok and f0ok and jb_ok,
         f"h3_freeze.json md5 {fmd5} = signed {fmd5 == freeze_md5}; {nf} md5s listed = the files now (differ: {bad}); "
         f"stage_h6_bnm_test_score.py = 8b9a38dd {h6ok}; stage_f0_spread_check.py = version 2 {f0ok}; h2b_report.json gates all "
         f"PASS {jb_ok}; rule floor {floor} distinct stacks" + (" (override, self-test only)" if floor_override is not None else ""))
    if not res["V0"]:
        return stop("V0")
    with F0.LoadLog() as log:
        try:
            if broken() == "test_path":
                np.load(data / "bnm_test" / "bnm_frozen.npz")
            F = F0.load_fit(bnm, h2a, h2b)
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("V4", False, f"np.load log: {crashed}")
        return stop("V4")
    if broken() == "labels_shift":
        F["lab"] = np.roll(F["lab"], 1)
    # ---- V1
    n, ptr, lab = F["n"], F["ptr"], F["lab"]
    y = np.add.reduceat(lab.astype(int), ptr[:-1]).astype(float)
    y_ok = np.array_equal(y, F["y_strict"])
    A0 = {m: S2.item_aucs(ptr, F["score"][m], lab) for m in SEEDS + REFS}
    chk = {m: F0.same_set(A0[m], F["auc_ref"][m]) for m in A0}
    auc_ok = all(s and w <= 1e-12 for s, w in chk.values())
    whole = max(float(np.abs(F["score"][m] * K - np.round(F["score"][m] * K)).max()) for m in SEEDS)
    b1x_ok = not np.isnan(F["score"]["B1x"]).any()
    gate("V1", y_ok and auc_ok and whole <= WHOLE_TOL and b1x_ok,
         f"{n:,} fit-half stack-repeats, {len(lab):,} hole tetrads; recorded holes per stack-repeat = h2b y_strict {y_ok}; AUC of "
         + ", ".join(f"{NAME[m]} {chk[m][1]:.1e}" for m in chk) + f" (same not-rankable set {all(s for s, _ in chk.values())}); "
         f"seed scores x 50 whole numbers, largest gap {whole:.1e} (tolerance {WHOLE_TOL:.0e}); B1 exact never NaN at a hole {b1x_ok}")
    if not res["V1"]:
        return stop("V1")
    # ---- V2
    r = rank01(ptr, F["score"]["B1x"])
    Az = {m: S2.item_aucs(ptr, score_t(F["score"][m], r, 0.0), lab) for m in SEEDS}
    base = {m: {b: hr1(Az[m], A0[b], F, floor) for b in REFS} for m in SEEDS}
    worst, same_v = 0.0, True
    for m in SEEDS:
        for b in REFS:
            same_v &= base[m][b]["verdict"] == jb["H_R1"][m][b]["verdict"]
            for g in base[m][b]["rows"]:
                worst = max(worst, H6.rows_differ(base[m][b]["rows"][g], jb["H_R1"][m][b]["rows"][g]))
    gate("V2", worst <= 1e-9 and same_v, f"added term x 0: H-R1 rows of seed 1 and seed 2 against both B1 versions = h2b_report.json, "
         f"largest difference {worst:.1e}, verdicts same {same_v}")
    if not res["V2"]:
        return stop("V2")
    # ---- T scores and V3
    ST = {m: score_t(F["score"][m], r) for m in SEEDS}
    AT = {m: S2.item_aucs(ptr, ST[m], lab) for m in SEEDS}
    tie = {m: tie_share(ptr, F["score"][m], lab) for m in SEEDS}
    rc = rank01(ptr, np.zeros(len(lab)))
    order = {m: order_kept(ptr, F["score"][m], ST[m]) for m in SEEDS}
    notie = {m: bool(np.array_equal(AT[m][tie[m] == 0], A0[m][tie[m] == 0])) for m in SEEDS}
    const = {m: bool(np.array_equal(S2.item_aucs(ptr, score_t(F["score"][m], rc), lab), A0[m], equal_nan=True)) for m in SEEDS}
    gate("V3", all(order.values()) and all(notie.values()) and all(const.values()),
         "order of holes with different model scores kept on every stack-repeat " + ", ".join(f"{NAME[m]} {order[m]}" for m in SEEDS)
         + "; AUC(T) = AUC(model) where the model ties no recorded / unrecorded pair " + ", ".join(
             f"{NAME[m]} {notie[m]} ({int((tie[m] == 0).sum()):,} stack-repeats)" for m in SEEDS)
         + "; B1 exact set to a constant gives AUC(model) " + ", ".join(f"{NAME[m]} {const[m]}" for m in SEEDS))
    if not res["V3"]:
        return stop("V3")
    gate("V4", not any(any(t in p for t in F0.FORBID) for p in log.paths) and all(any(f"/{a}/" in p for a in F0.ALLOWED) for p in log.paths),
         f"np.load log: {len(log.paths)} files, all in bnm_fit, stage_h2_predict or stage_h2b_score; test half, H5, H6, real_eval 0")
    # ---- results
    grp = {"recorded class": (F["rcls"], TCLS), "records over 6 years": (F["rbin"], BINS)}
    nwin = len(np.unique(F["win"]))
    out(f"   share of recorded / unrecorded hole pairs the model ties (stack-first mean over rankable stack-repeats; the room T has):")
    tie_rows = {m: {g: S2.hr1_rows(S2.hr1_core(tie[m], F["stid"], gi, F["win"], list(range(len(labs)))), labs, floor)
                    for g, (gi, labs) in grp.items()} for m in SEEDS}
    for m in SEEDS:
        out(f"      {NAME[m]:<7} " + " | ".join("; ".join(f"{q['group']} {q['mean']:.3f}" for q in tie_rows[m][g]) for g in grp))
    gain = {m: {g: S2.hr1_rows(S2.hr1_core(AT[m] - A0[m], F["stid"], gi, F["win"], list(range(len(labs)))), labs, floor)
                for g, (gi, labs) in grp.items()} for m in SEEDS}
    out(f"   AUC(T) - AUC(model), stack-first mean [95% interval, {nwin} windows resampled] (distinct stacks):")
    for m in SEEDS:
        for g in grp:
            out(f"      {NAME[m]:<7} {g:<20} " + "; ".join(f"{q['group']} {q['mean']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}] ({q['n']})"
                                                      if q["n"] else f"{q['group']} none" for q in gain[m][g]))
    tb = {m: {b: hr1(AT[m], A0[b], F, floor) for b in REFS} for m in SEEDS}
    out(f"   H-R1 with T, T - B1, stack-first mean [95% interval] verdict (distinct stacks) {{unchanged H2b mean, model - B1}}:")
    verdict = {}
    for m in SEEDS:
        for b in REFS:
            for g, rows in tb[m][b]["rows"].items():
                b0 = {q["group"]: q for q in base[m][b]["rows"][g]}
                out(f"      {NAME[m]} vs {NAME[b]:<9} {g:<20} " + "; ".join(f"{fmt(q)} {{{b0[q['group']]['mean']:+.4f}}}" if q["n"] else fmt(q)
                                                                       for q in rows))
            out(f"      {NAME[m]} vs {NAME[b]}: {tb[m][b]['verdict']} (unchanged {base[m][b]['verdict']})")
        verdict[m] = S2.both(tb[m]["B1"]["verdict"], tb[m]["B1x"]["verdict"])
    sparse = (F["rcls"] <= 1) | (F["rbin"] <= 2)
    srows = {m: dict(model_B1=one_group(A0[m] - A0["B1"], F, sparse, floor), model_B1x=one_group(A0[m] - A0["B1x"], F, sparse, floor),
                     T_B1=one_group(AT[m] - A0["B1"], F, sparse, floor), T_B1x=one_group(AT[m] - A0["B1x"], F, sparse, floor),
                     T_model=one_group(AT[m] - A0[m], F, sparse, floor), tie=one_group(tie[m], F, sparse, floor)) for m in SEEDS}
    out(f"   sparse set S of H3b, pooled ({int(sparse.sum()):,} stack-repeats): " + " || ".join(
        f"{NAME[m]}: model - B1 stored {fmt(srows[m]['model_B1'])[2:]}, - B1 exact {fmt(srows[m]['model_B1x'])[2:]}; T - B1 stored "
        f"{fmt(srows[m]['T_B1'])[2:]}, - B1 exact {fmt(srows[m]['T_B1x'])[2:]}; T - model {srows[m]['T_model']['mean']:+.4f} "
        f"[{srows[m]['T_model']['lo']:+.4f}, {srows[m]['T_model']['hi']:+.4f}]; tied pairs {srows[m]['tie']['mean']:.3f}" for m in SEEDS))
    no_fail = all(tb[m][b]["verdict"] != "FAIL" for m in SEEDS for b in REFS)
    dense = all(q["mean"] >= 0 for m in SEEDS for b in REFS for g, labs in DENSE.items() for q in tb[m][b]["rows"][g]
                if q["group"] in labs and q["tested"])
    yes = no_fail and dense
    out(f"   rule: (i) no group FAILs for both seeds against both B1 versions {no_fail}; (ii) classes 21-50, 51+ and records 21+ keep a mean "
        f"of 0 or above {dense}")
    # ---- files
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = dict(stack_id=F["stid"], rcls=F["rcls"], rbin=F["rbin"], sparse=sparse, rank_b1x=r.astype(np.float32))
    for m in SEEDS:
        sc[f"auc_T_{m}"], sc[f"tie_{m}"] = AT[m], tie[m]
    X.save_npz(out_dir / "t_scores.npz", sc)
    clean = lambda o_: json.loads(json.dumps(o_, default=lambda v: v.item() if hasattr(v, "item") else str(v)).replace("NaN", "null"))
    rep.update(gates=res, floor=floor, n=int(n), whole_gap=whole,
               H_R1_T={m: dict(verdict=verdict[m], **{b: tb[m][b] for b in REFS}) for m in SEEDS},
               H_R1_unchanged={m: {b: base[m][b] for b in REFS} for m in SEEDS},
               gain={m: {g: bare(v) for g, v in gain[m].items()} for m in SEEDS},
               tie_share={m: {g: bare(v) for g, v in tie_rows[m].items()} for m in SEEDS},
               sparse_S={m: {k: (v if k in ('model_B1', 'model_B1x', 'T_B1', 'T_B1x') else bare([v])[0]) for k, v in srows[m].items()} for m in SEEDS},
               rule=dict(no_fail=no_fail, dense_kept=dense, worth_using=yes),
               md5=dict(tool=X.file_md5(__file__), fingerprint=fingerprint(__file__), freeze=fmd5, scores=X.file_md5(out_dir / "t_scores.npz")),
               minutes=round((time.time() - t0) / 60, 1), written_utc=F0.now())
    (out_dir / "t_report.json").write_text(json.dumps(clean(rep), indent=1))
    n_ok = sum(res.values())
    out(f"   written t_scores.npz md5 {rep['md5']['scores']}, t_report.json ({rep['minutes']} min)")
    out(f"stage T: gates {n_ok} PASS, {len(res) - n_ok} FAIL; H-R1 with T seed 1 {verdict['s1']}, seed 2 {verdict['s2']}; "
        f"T worth using by the rule written before the run: {'YES' if yes else 'NO'}")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bnm", default="../data/bnm_fit")
    ap.add_argument("--h2a", default="../data/stage_h2_predict")
    ap.add_argument("--h2b", default="../data/stage_h2b_score")
    ap.add_argument("--freeze", default="../data/stage_h3_freeze/h3_freeze.json")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--out", default="../data/stage_t_tiebreak")
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--g3", default="../data/stage_g3_calibration")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_35_t_tiebreak.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found")
        spec = importlib.util.spec_from_file_location("test_35", tp); t35 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t35)
        sys.exit(0 if t35.self_test(Path(__file__), a.obj1_tools) else 1)
    r_, _ = run(a.bnm, a.h2a, a.h2b, a.freeze, a.runs, a.obj1_tools, a.out, a.g2, a.g3, out=lambda s: print(s, flush=True))
    sys.exit(0 if r_ and all(r_.values()) else 1)
