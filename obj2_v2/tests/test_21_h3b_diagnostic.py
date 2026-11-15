#!/usr/bin/env python3
"""
test_21_h3b_diagnostic.py  (obj2_v2, self-test of tools/stage_h3b_sparse_diagnostic.py, 12 Nov 2026)

Run through the tool:  python3 ../tools/stage_h3b_sparse_diagnostic.py --self-test | tail -1   (from obj2_v2/work)
Planted data only. The data loaders are checked in the real run by gates K1 and K3 (labels, AUCs = H2b and stage D).
    U1  U statistic AUC (ties half) = brute-force pair count, 200 random stacks with ties
    U2  decomposition: different-bin and same-bin concordance = brute-force typed pairs; parts add to the AUC
    U3  rule (a) on planted decay: a score that falls off more slowly than the labelled share gives (a) True; a score
        proportional to the labelled share gives (a) False
    U4  rule (b) on planted stacks: a model that misorders only across distance bins gives (b) True; one that misorders
        only inside bins gives (b) False (its different-bin part is 0)
    U5  analyse + rule end to end: both seeds with distance errors -> fix cycle worth considering YES; seed 2 with
        within-bin errors -> NO
    U6  h3_freeze check: unchanged files pass; one byte changed is named
    U7  distance bins at the edges; the K2 hand count agrees with the decomposition and catches a planted decomposition bug
"""
import importlib.util, json, os, tempfile
from pathlib import Path
import numpy as np


def load_tool(p):
    spec = importlib.util.spec_from_file_location("stage_h3b_t", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def brute(s, y):
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def planted(g, T, n_st=300, R=2, n_cl=20, cells=40, mode1="dist", mode2="dist"):
    """Stacks with cells at distances 1-10; label ~ logistic(2 - 0.8 d + 0.3 z). B1 = -d + 0.01 z (orders by distance, then
    z). Model 'dist': -0.1 d + z (slow decay; across bins z overrides distance, inside a bin the order of B1); model 'within':
    B1's order across bins, z reversed inside a bin."""
    n = n_st * R
    stack = np.repeat(np.arange(n_st), R); clus = stack % n_cl
    rcls = np.repeat(g.integers(0, 2, n_st), R); rbin = np.repeat(g.integers(0, 3, n_st), R)
    ptr = np.r_[0, np.cumsum(np.full(n, cells))]
    d = g.integers(1, 11, ptr[-1]).astype(float); z = g.random(ptr[-1])
    y = g.random(ptr[-1]) < 1 / (1 + np.exp(-(2 - 0.8 * d + 0.3 * z)))
    sc = {"B1": -d + 0.01 * z, "B1x": -d + 0.01 * z + 1e-6 * z, "B4": 1.5 - 0.6 * d}
    for m, mode in (("s1", mode1), ("s2", mode2)):
        sc[m] = 1 / (1 + np.exp(-(-0.1 * d + z))) if mode == "dist" else 1 / (1 + np.exp(-(-d - 0.01 * z)))
    sc["B4p"] = 1 / (1 + np.exp(-sc["B4"]))
    return dict(ptr=ptr, y=y, b=T.dist_bin(d), scores=sc, stack=stack, clus=clus, rcls=rcls, rbin=rbin, nclus=n_cl)


def self_test(tool_path):
    T = load_tool(tool_path)
    g = np.random.default_rng(21)
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}", flush=True)

    # U1
    worst = 0.0
    for _ in range(200):
        k = g.integers(2, 40); s = g.integers(0, 6, k).astype(float); y = g.random(k) < 0.4
        u, p = T.u_stat(s, y); b = brute(s, y)
        worst = max(worst, 0.0 if (p == 0 and np.isnan(b)) else abs(u / p - b) if p else 1.0)
    t("U1", worst < 1e-12, f"U statistic AUC = brute-force pairs with ties counted half: largest difference {worst:.1e}")
    # U2
    w2 = 0.0
    for _ in range(100):
        k = g.integers(4, 40); s = g.integers(0, 5, k).astype(float); y = g.random(k) < 0.5; b = g.integers(1, 5, k)
        ptr = np.array([0, k]); dec, pt, ps = T.decompose(ptr, y, b, {"m": s})
        pos, neg = np.nonzero(y)[0], np.nonzero(~y)[0]
        same = [(i, j) for i in pos for j in neg if b[i] == b[j]]; diff = [(i, j) for i in pos for j in neg if b[i] != b[j]]
        c = lambda pr: sum(1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0 for i, j in pr)
        if pt[0] > 0:
            w2 = max(w2, abs(dec["m"]["u_same"][0] - c(same)), abs(dec["m"]["u_diff"][0] - c(diff)), abs(ps[0] - len(same)),
                     abs((dec["m"]["u_same"][0] + dec["m"]["u_diff"][0]) / pt[0] - dec["m"]["auc"][0]))
    t("U2", w2 < 1e-9, f"same-bin and different-bin concordance = brute-force typed pairs, parts add to the AUC: largest difference {w2:.1e}")
    # U3
    n, cells = 400, 40
    ptr = np.r_[0, np.cumsum(np.full(n, cells))]; d = g.integers(1, 11, ptr[-1]).astype(float)
    p_obs = 0.6 * np.exp(-(d - 1) / 1.5); y = g.random(ptr[-1]) < p_obs
    clus = np.arange(n) % 20; C = T.draws_matrix(20)
    _, st = T.decay(ptr, y, T.dist_bin(d), {"slow": 0.6 * np.exp(-(d - 1) / 6), "match": 0.3 * np.exp(-(d - 1) / 1.5)},
                    np.ones(n, bool), clus, C)
    t("U3", st["slow"]["lo"] > 0 and st["match"]["lo"] <= 0 <= st["match"]["hi"],
      f"(a): slower decay far/near {st['slow']['ratio']:.3f} against labelled {st['slow']['ratio_obs']:.3f}, interval of the "
      f"difference [{st['slow']['lo']:+.3f}, {st['slow']['hi']:+.3f}] -> True; proportional score [{st['match']['lo']:+.3f}, "
      f"{st['match']['hi']:+.3f}] -> False")
    # U4
    P = planted(g, T, mode1="dist", mode2="within")
    dec, pt, ps = T.decompose(P["ptr"], P["y"], P["b"], {m: P["scores"][m] for m in T.METH})
    C = T.draws_matrix(P["nclus"]); allm = np.ones(len(pt), bool)
    rows = T.d2_table(dec, pt, ps, P["stack"], P["clus"], allm, C)
    r1, r2 = rows["s1 - B1x"], rows["s2 - B1x"]
    b1 = r1["gap_hi"] < 0 and r1["rule_b_hi"] < 0
    b2 = r2["gap_hi"] < 0 and r2["rule_b_hi"] < 0
    t("U4", b1 and not b2 and abs(r2["diff_part"]) < 1e-12, f"(b): distance errors gap {r1['gap']:+.4f}, different-bin share "
      f"{r1['diff_share']:.2f} -> {b1}; within-bin errors gap {r2['gap']:+.4f}, different-bin part {r2['diff_part']:+.1e} -> {b2}")
    # U5
    P1 = planted(g, T, mode1="dist", mode2="dist"); P2 = planted(g, T, mode1="dist", mode2="within")
    ra = T.analyse(P1["ptr"], P1["y"], P1["b"], P1["scores"], P1["stack"], P1["clus"], P1["rcls"], P1["rbin"], P1["nclus"])[0]
    rb = T.analyse(P2["ptr"], P2["y"], P2["b"], P2["scores"], P2["stack"], P2["clus"], P2["rcls"], P2["rbin"], P2["nclus"])[0]
    ya, yb = T.rule(ra), T.rule(rb)
    t("U5", ya["fix cycle worth considering"] and not yb["fix cycle worth considering"] and yb["s1"]["both"] and not yb["s2"]["both"],
      f"end to end: both seeds with distance errors -> {ya['fix cycle worth considering']} ({ya['s1']}, {ya['s2']}); seed 2 with "
      f"within-bin errors -> {yb['fix cycle worth considering']} (seed 2 {yb['s2']})")
    # U6
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); (tmp / "tools").mkdir(); (tmp / "data" / "x").mkdir(parents=True); (tmp / "runs" / "f_seed1").mkdir(parents=True)
        (tmp / "tools" / "a.py").write_text("a\n"); (tmp / "data" / "x" / "f.npz").write_bytes(b"12345"); (tmp / "runs" / "f_seed1" / "best.pt").write_bytes(b"w")
        md5 = lambda p: __import__("hashlib").md5(Path(p).read_bytes()).hexdigest()
        fz = tmp / "fz.json"
        fz.write_text(json.dumps(dict(tools={"a.py": md5(tmp / "tools" / "a.py")}, data={"x/f.npz": md5(tmp / "data" / "x" / "f.npz"),
                                                                                       "f_seed1/best.pt": md5(tmp / "runs" / "f_seed1" / "best.pt")})))
        bad1, nf = T.freeze_check(fz, tmp / "data", tmp / "runs", tmp / "tools", md5)
        (tmp / "data" / "x" / "f.npz").write_bytes(b"12346")
        bad2, _ = T.freeze_check(fz, tmp / "data", tmp / "runs", tmp / "tools", md5)
    t("U6", bad1 == [] and bad2 == ["x/f.npz"] and nf == 3, f"freeze check: unchanged {bad1}; one byte changed names {bad2}")
    # U7
    db = T.dist_bin(np.array([0.0, 1.0, 1.41, 1.5, 2.24, 3.0, 3.61, 5.4, 5.5, 8.4, 8.5, np.inf]))
    sub = {m: P["scores"][m] for m in T.METH}
    dec0, _, ps0 = T.decompose(P["ptr"][:51], P["y"], P["b"], sub)
    ok0 = T.brute_check(P["ptr"][:51], P["y"], P["b"], sub, dec0, ps0, k=50)
    os.environ["STAGE_H3B_BROKEN"] = "decomp"
    try:
        decb, _, psb = T.decompose(P["ptr"][:51], P["y"], P["b"], sub)
    finally:
        os.environ.pop("STAGE_H3B_BROKEN", None)
    idb = T.brute_check(P["ptr"][:51], P["y"], P["b"], sub, decb, psb, k=50)
    t("U7", db.tolist() == [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6] and ok0 < 1e-9 and idb > 1e-6, f"bins 0, 1, 1.41, 1.5, 2.24, 3, 3.61, "
      f"5.4, 5.5, 8.4, 8.5, inf -> {db.tolist()}; K2 hand count on the correct decomposition {ok0:.1e}, on a planted "
      f"decomposition bug {idb:.1e} (fails as it must)")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)