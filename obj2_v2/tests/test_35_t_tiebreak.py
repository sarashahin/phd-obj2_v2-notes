#!/usr/bin/env python3
"""
test_35_t_tiebreak.py  (obj2_v2, self-test of tools/stage_t_tiebreak.py, 15 Nov 2026)

Run through the tool:  python3 ../tools/stage_t_tiebreak.py --self-test 2>&1 | tail -n 1   (from obj2_v2/work)
Planted data only, in a temporary folder, with the real tools: the planted fit half of test_25 (build_fit: the planted
pipeline of H2a's self-test, G3, H1's planted grids, the three H2a parts, H2b with floor 5, h3_freeze.json as H3 writes it).
Nothing of ../data is read.
    U1  rank: ties share their mean rank, divided by the number of holes, per stack-repeat (hand example)
    U2  order: on 2,000 planted stack-repeats with float32 scores k/50 the model order is kept and model ties are broken by
        B1 exact; a model that ties everything gives AUC(T) = AUC(B1 exact); a model with no tie gives AUC(T) = AUC(model);
        order_kept catches a swapped pair; tie share by hand
    U3  control: holes whose recorded share falls with distance, a model flat at 0-1 of 50 beyond 2 cells: B1 exact as
        -distance lifts the mean AUC; a B1 exact of pure noise leaves the mean change within 4 standard errors of 0
    T1  the tool on the planted fit half (floor 5): gates V0-V4 PASS, report and scores written, the rule printed
    T2  recomputed outside the tool on 60 stack-repeats: B1 exact ranks by counting, AUC(T) and the tie share by a loop
        over every recorded / unrecorded pair; the sparse-set mean of T - model by an own stack-first loop
    T3  refused: a second run into the same folder; an output inside the H2b folder
    T4  planted bugs, each fails its gate and writes nothing: labels shifted (V1), H-R1 code shifted (V2), added term too
        large (V3), ties ranked in order of appearance (V3), a test-half path loaded (V4), an H2b file changed after the
        freeze (V0)
"""
import importlib.util, json, os, shutil, tempfile
from pathlib import Path
import numpy as np


def load_tool(tool_path):
    spec = importlib.util.spec_from_file_location("stage_t_tiebreak_t", tool_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def brute_auc(s, lab):
    pos, neg = s[lab], s[~lab]
    if not len(pos) or not len(neg):
        return np.nan, np.nan
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size), float((d == 0).sum() / d.size)


def brute_rank01(v):
    return np.array([((v < x).sum() + ((v == x).sum() - 1) / 2) / len(v) for x in v])


def unit_checks(T, t):
    # U1
    ptr = np.array([0, 4, 7]); v = np.array([3.0, 1, 1, 2, 5, 5, 5])
    r = T.rank01(ptr, v)
    want = np.array([0.75, 0.125, 0.125, 0.5, 1 / 3, 1 / 3, 1 / 3])
    t("U1", np.allclose(r, want, atol=1e-15), f"ranks of [3, 1, 1, 2 | 5, 5, 5] / holes = {np.round(r, 4).tolist()}")
    # U2
    g = np.random.default_rng(35)
    n, per = 2000, 25
    ptr = np.r_[0, np.cumsum(np.full(n, per))]
    s = (g.integers(0, 51, n * per) / 50).astype(np.float32).astype(float)
    b = g.normal(size=n * per)
    lab = g.random(n * per) < 0.3
    st = T.score_t(s, T.rank01(ptr, b))
    ok_order = T.order_kept(ptr, s, st)
    flat = np.full(n * per, 0.02)
    a_flat, a_b = T.S2.item_aucs(ptr, T.score_t(flat, T.rank01(ptr, b)), lab), T.S2.item_aucs(ptr, b, lab)
    strict = np.concatenate([g.permutation(per) / 50 for _ in range(n)]).astype(np.float32).astype(float)
    a_s, a_st = T.S2.item_aucs(ptr, strict, lab), T.S2.item_aucs(ptr, T.score_t(strict, T.rank01(ptr, b)), lab)
    sw = st.copy(); i0 = int(np.flatnonzero(np.diff(s[:per]) != 0)[0]); sw[[i0, i0 + 1]] = sw[[i0 + 1, i0]]
    caught = not T.order_kept(ptr, s, sw)
    ts = T.tie_share(np.array([0, 5]), np.array([1.0, 1, 2, 1, 3]), np.array([True, False, False, True, False]))
    t("U2", ok_order and np.array_equal(a_flat, a_b, equal_nan=True) and np.array_equal(a_s, a_st, equal_nan=True) and caught
      and abs(ts[0] - 2 / 6) < 1e-15,
      f"model order kept on {n:,} planted stack-repeats {ok_order}; all-tied model gives AUC(B1 exact) exactly "
      f"{np.array_equal(a_flat, a_b, equal_nan=True)}; untied model keeps its AUC {np.array_equal(a_s, a_st, equal_nan=True)}; swapped pair "
      f"caught {caught}; tie share of [1, 1 | 1, 2, 3] = {ts[0]:.4f} (2/6)")
    # U3
    n, per = 600, 40
    ptr = np.r_[0, np.cumsum(np.full(n, per))]
    dist = g.uniform(1, 10, n * per)
    lab = g.random(n * per) < 0.5 * np.exp(-(dist - 1) / 1.5)
    s = np.where(dist < 2, g.integers(10, 30, n * per), g.integers(0, 2, n * per)) / 50
    a0 = T.S2.item_aucs(ptr, s, lab)
    d_inf = T.S2.item_aucs(ptr, T.score_t(s, T.rank01(ptr, -dist)), lab) - a0
    d_noise = T.S2.item_aucs(ptr, T.score_t(s, T.rank01(ptr, g.normal(size=n * per))), lab) - a0
    m1, m2 = np.nanmean(d_inf), np.nanmean(d_noise)
    se2 = np.nanstd(d_noise) / np.sqrt(np.isfinite(d_noise).sum())
    t("U3", m1 > 0.02 and abs(m2) <= 4 * se2, f"distance as tie-breaker lifts the mean AUC by {m1:+.4f}; noise as tie-breaker "
      f"{m2:+.4f} (4 se {4 * se2:.4f})")


def checks(T, w, tmp, t):
    """T1-T4 on a planted fit half w (test_25's build_fit on Sara's machine, or the sandbox stub world)."""
    quiet = lambda *a, **k: None
    tmp = Path(tmp)
    kw = dict(bnm=w["bnm"], h2a=w["h2a"], h2b=w["h2b"], g2=w["g2"], g3=w["g3"], freeze=w["freeze"], runs=w["runs"], obj1_tools=w["obj1_tools"],
              freeze_md5=w["freeze_md5"], h6_md5=w["h6_md5"], floor_override=w["floor_override"])
    lines = []
    r, rep = T.run(out_dir=tmp / "t", out=lines.append, **kw)
    t("T1", all(r.values()) and set(r) == {"V0", "V1", "V2", "V3", "V4"} and (tmp / "t" / "t_report.json").is_file()
      and "worth_using" in rep.get("rule", {}) and lines[-1].startswith("stage T: gates 5 PASS"),
      f"planted fit half: {r}; last line: {lines[-1][:150]}")
    if not (tmp / "t" / "t_scores.npz").is_file():
        t("T2", False, "skipped: T1 wrote no t_scores.npz")
        return lines
    F = T.F0.load_fit(Path(w["bnm"]), Path(w["h2a"]), Path(w["h2b"]))
    sc = np.load(tmp / "t" / "t_scores.npz")
    rk = np.flatnonzero(~np.isnan(sc["auc_T_s1"]))
    pick = rk[np.random.default_rng(3).choice(len(rk), min(60, len(rk)), replace=False)]
    ok_a, ok_t = True, True
    for i in pick:
        a, b = F["ptr"][i], F["ptr"][i + 1]
        lab = F["lab"][a:b]
        for m in T.SEEDS:
            s_ = F["score"][m][a:b] + brute_rank01(F["score"]["B1x"][a:b]) / 100.0
            aa, _ = brute_auc(s_, lab)
            _, tt = brute_auc(F["score"][m][a:b], lab)
            ok_a &= abs(aa - sc[f"auc_T_{m}"][i]) < 1e-12
            ok_t &= abs(tt - sc[f"tie_{m}"][i]) < 1e-12
    js = json.loads((tmp / "t" / "t_report.json").read_text())
    A0 = T.S2.item_aucs(F["ptr"], F["score"]["s1"], F["lab"])
    d = sc["auc_T_s1"] - A0
    m = sc["sparse"].astype(bool) & ~np.isnan(d)
    per = {}
    for s_, v_ in zip(sc["stack_id"][m], d[m]):
        per.setdefault(int(s_), []).append(v_)
    own = float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")
    rep_m = js["sparse_S"]["s1"]["T_model"]["mean"]
    ok_s = (not per and rep_m is None) or abs(own - rep_m) < 1e-9
    t("T2", ok_a and ok_t and ok_s, f"recomputed outside the tool on {len(pick)} stack-repeats: AUC(T) of both seeds by a pair loop {ok_a}, "
      f"tie share {ok_t}; sparse-set mean of T - model (seed 1) {own:+.6f} against the report {ok_s}")
    stop = []
    for f_ in (lambda: T.run(out_dir=tmp / "t", out=quiet, **kw), lambda: T.run(out_dir=Path(w["h2b"]) / "x", out=quiet, **kw)):
        try:
            f_(); stop.append(False)
        except SystemExit:
            stop.append(True)
    t("T3", all(stop), f"refused: a second run into the same folder {stop[0]}, an output inside the H2b folder {stop[1]}")
    for bug, g_ in (("labels_shift", "V1"), ("core_shift", "V2"), ("big_term", "V3"), ("rank_noties", "V3"), ("test_path", "V4")):
        os.environ["STAGE_T_BROKEN"] = bug
        try:
            rb, _ = T.run(out_dir=tmp / f"b_{bug}", out=quiet, **kw)
        finally:
            os.environ.pop("STAGE_T_BROKEN", None)
        t(f"bug {bug}", rb.get(g_) is False and not (tmp / f"b_{bug}").exists(), f"{g_} fails and nothing is written ({rb})")
    hx = tmp / "h2b_changed"
    shutil.copytree(w["h2b"], hx)
    z = dict(np.load(hx / "h2b_scores.npz")); z["y_strict"] = z["y_strict"] + 1
    T.X.save_npz(hx / "h2b_scores.npz", z)
    rb, _ = T.run(out_dir=tmp / "b_h2b", out=quiet, **dict(kw, h2b=hx))
    t("bug h2b_changed", rb.get("V0") is False and len(rb) == 1, f"an H2b file changed after the freeze fails V0 and stops ({rb})")
    return lines


def self_test(tool_path, obj1):
    T = load_tool(tool_path)
    p25 = Path(tool_path).resolve().parent.parent / "tests" / "test_25_f0a_spread_check.py"
    spec = importlib.util.spec_from_file_location("test_25_t", p25); t25 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t25)
    res_all = []

    def t(name, ok, what):
        res_all.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    unit_checks(T, t)
    with tempfile.TemporaryDirectory() as tmp:
        w, ok = t25.build_fit(T, tmp, obj1, tool_path)
        t("S0", ok, "planted pipeline: G3, H1 fit half, H2a parts, H2b (floor 5), freeze built as H3 writes it")
        lines = checks(T, w, tmp, t)
        print("   planted run log:", *lines, sep="\n      ")
    n_ok = sum(res_all)
    print(f"{n_ok} PASS, {len(res_all) - n_ok} FAIL")
    return all(res_all)