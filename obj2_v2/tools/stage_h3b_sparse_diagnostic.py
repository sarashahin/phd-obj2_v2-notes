#!/usr/bin/env python3
"""
stage_h3b_sparse_diagnostic.py  (obj2_v2, stage H, diagnostic between H3 and H4, 12 Nov 2026; report only; plan, real_val
part and decision rule signed off by Sara on 12 Nov, claude/obj2_v2_H_rules_draft.md)

Question: where does the model lose to B1 for sparsely recorded stacks on the BNM fit half (H2b, H-R1), and is the same
pattern already there on simulation (real_val, habitat blanked, the BNM condition)? Nothing here is a rule of stage H and
nothing frozen in H3 changes; the result feeds Sara's decision before H4.

Data
    BNM fit half: h2b_predicted.npz (scores of seed 1, seed 2, B1 stored, B1 exact, B4 at every hole tetrad; distance to the
    nearest kept record; groups), the hole labels (recorded in 2019) from the fit-half truth files through H2b's own
    read_labels, h2b_scores.npz (for the alignment check). real_val: frozen set, step 3a samples of each run (preds_k50.npz),
    B1 stored (stage_d_headroom_pred.npz), B1 exact and B4 (g2_pred.npz), records and truth through stage D's load_stacks;
    habitat-blanked stacks only; scored cells = cells with no record at the target year; label = occupied at the target year.
Distance bins (distance rounded to the nearest cell, as H2b): 0 (recorded in an earlier year only; real_val), 1, 2, 3, 4-5,
6-8, 9+. Near = 1, far = 4+.
Sparse set S: kept recorded class 0 or 1 at the target year, or kept records over 6 years 1-10 (Sara's groups, pooled);
each of those groups and records 21+ (contrast) reported too.

    D1 distance decay: per group and bin, cells, observed (recorded / occupied) share, mean score of seed 1, seed 2, B4 (as a
       chance); far / near ratio of each
    D2 where the AUC gap sits: each stack-repeat's labelled / unlabelled pairs split into pairs in different distance bins
       and pairs in the same bin; AUC = w_diff C_diff + w_same C_same; gap to B1 = different-bin part + same-bin part,
       stack-first means (each stack's repeats in the group averaged, then over stacks)
    D3 D1 and D2 on real_val, habitat-blanked stacks, groups from the records; intervals resample communities

Decision rule (signed by Sara on 12 Nov, before the run): a fix cycle is worth considering only if both hold for both seeds
in the sparse set S on BNM:
    (a) the model's score falls off with distance more slowly than the recorded share: far/near ratio of the mean score
        minus far/near ratio of the recorded share > 0, the lower end of its 95% interval above 0
    (b) more than half of the AUC gap to B1 comes from pairs in different distance bins: the gap G < 0 (upper end of its
        interval below 0) and G_diff - 0.5 G < 0 (upper end below 0); against both B1 versions, as H-R1
    Intervals: 2,000 window resamples (BNM) or community resamples (real_val), numpy seed 31. Otherwise v2 is reported
    as it is. Meeting the rule does not decide the fix; Sara does.
Gates
    K0 every tool and data md5 in h3_freeze.json equals the file now (before) and after the run (nothing frozen changes)
    K1 BNM labels aligned: recorded holes per stack-repeat = h2b_scores y_strict; this tool's AUC of seed 1, seed 2, B1,
       B1 exact, B4 per stack-repeat = h2b_scores (1e-12, same not-rankable set)
    K2 decomposition = every labelled / unlabelled pair counted by hand, by same or different bin, on 200 BNM stack-repeats
       and 200 real_val stacks (1e-9)
    K3 real_val aligned: blank of the frozen set = g2_pred.npz's; this tool's AUC of every method per stack = stage D's
       HR.aucs (1e-12); no scored cell holds a target-year record
    K4 nothing of the test half or real_eval passed to np.load; no test-half export; output folder new
    If K0-K4 fail, the tool stops before any result or rule is printed and writes nothing.
Run (from PSD_Dispersal_pool/obj2_v2/work; CPU, a few minutes)
    python3 ../tools/stage_h3b_sparse_diagnostic.py --self-test | tail -1     (needs ../tests/test_21_h3b_diagnostic.py)
    python3 -u ../tools/stage_h3b_sparse_diagnostic.py > ../logs/stage_h3b_diagnostic.txt 2>&1
Needs stage_h2_bnm_score.py (and the stage tools it loads) in the same folder.
"""
import argparse, importlib.util, json, os, sys, warnings
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DBIN = ("0", "1", "2", "3", "4-5", "6-8", "9+")
NEAR, FAR = (1,), (4, 5, 6)
METH = ("s1", "s2", "B1", "B1x", "B4")
NDRAW, BOOT_SEED = 2000, 31


def broken():
    return os.environ.get("STAGE_H3B_BROKEN", "")      # self-test only: plants one bug so its check must fail


# ---------------------------------------------------------------- core (numpy only)
def pct(x, q):
    """Percentiles that ignore failed draws (NaN); NaN when every draw failed."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float("nan")] * len(q) if len(x) == 0 else [float(v) for v in np.percentile(x, q)]


def dist_bin(d):
    r = np.floor(np.where(np.isfinite(d), d, 1e9) + 0.5)
    return np.select([r <= 0, r <= 1, r == 2, r == 3, r <= 5, r <= 8], [0, 1, 2, 3, 4, 5], 6).astype(np.int8)


def u_stat(s, y):
    """Mann-Whitney U of the labelled cells (ties count half) and the number of pairs."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.0, 0
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[np.asarray(inv).ravel()]
    return float(r[y].sum() - n1 * (n1 + 1) / 2.0), n1 * n0


def decompose(ptr, y, b, scores):
    """Per item (cells ptr[i]:ptr[i+1]): AUC and U / pairs over all pairs and over same-bin pairs, for every method."""
    n = len(ptr) - 1
    out = {m: dict(auc=np.full(n, np.nan), u_same=np.zeros(n), u_diff=np.zeros(n)) for m in scores}
    p_tot, p_same = np.zeros(n), np.zeros(n)
    for i in range(n):
        a, z = ptr[i], ptr[i + 1]
        yi, bi = y[a:z], b[a:z]
        bins = np.unique(bi)
        for m, s in scores.items():
            si = s[a:z]
            u, p = u_stat(si, yi)
            if p == 0:
                continue
            us, ps = 0.0, 0
            for k in bins:
                q = bi == k
                u_, p_ = u_stat(si[q], yi[q])
                us += u_; ps += p_
            if broken() == "decomp":
                us *= 1.01
            out[m]["auc"][i] = u / p; out[m]["u_same"][i] = us; out[m]["u_diff"][i] = u - us
            p_tot[i], p_same[i] = p, ps
    return out, p_tot, p_same


def gap_parts(dec, p_tot, m, ref):
    ok = p_tot > 0
    g = np.where(ok, dec[m]["auc"] - dec[ref]["auc"], np.nan)
    gd = np.where(ok, (dec[m]["u_diff"] - dec[ref]["u_diff"]) / np.where(ok, p_tot, 1), np.nan)
    gs = np.where(ok, (dec[m]["u_same"] - dec[ref]["u_same"]) / np.where(ok, p_tot, 1), np.nan)
    return g, gd, gs


def draws_matrix(nclus, seed=BOOT_SEED, ndraw=NDRAW):
    g = np.random.default_rng(seed)
    pick = g.integers(0, nclus, size=(ndraw, nclus))
    return np.stack([np.bincount(r, minlength=nclus) for r in pick]).astype(float)


def stackfirst(vals, stack, clus, mask, C):
    """Stack-first means of each array in vals over the items in mask (finite in all), with cluster-resampled draws."""
    ok = mask & np.all([np.isfinite(v) for v in vals], 0)
    if not ok.any():
        return [float("nan")] * len(vals), [np.full(len(C), np.nan)] * len(vals), 0
    us, inv = np.unique(stack[ok], return_inverse=True)
    inv = np.asarray(inv).ravel(); cnt = np.bincount(inv)
    cl = np.zeros(len(us), np.int64); cl[inv] = clus[ok]
    full, dr = [], []
    for v in vals:
        sm = np.bincount(inv, weights=v[ok]) / cnt
        full.append(float(sm.mean()))
        S = np.bincount(cl, weights=sm, minlength=C.shape[1]); N = np.bincount(cl, minlength=C.shape[1]).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            dr.append((C @ S) / (C @ N))
    return full, dr, len(us)


def decay(ptr, y, b, scores, mask_items, clus, C):
    """D1 table and the (a) statistic: far/near ratio of each score minus far/near ratio of the labelled share."""
    n = len(ptr) - 1
    item = np.repeat(np.arange(n), np.diff(ptr))
    keep = mask_items[item]
    tab = {}
    for k in range(len(DBIN)):
        q = keep & (b == k)
        if q.any():
            tab[DBIN[k]] = dict(cells=int(q.sum()), labelled=float(y[q].mean()), **{m: float(s[q].mean()) for m, s in scores.items()})
    ncl = C.shape[1]
    cl = clus[item]
    def sums(v, sel):
        return np.bincount(cl[keep & sel], weights=v[keep & sel].astype(float), minlength=ncl)
    near, far = np.isin(b, NEAR), np.isin(b, FAR)
    nn, nf = sums(np.ones(len(y)), near), sums(np.ones(len(y)), far)
    yn, yf = sums(y, near), sums(y, far)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = (yf.sum() / nf.sum()) / (yn.sum() / nn.sum())
        r_obs_d = ((C @ yf) / (C @ nf)) / ((C @ yn) / (C @ nn))
        stat = {}
        for m, s in scores.items():
            sn, sf_ = sums(s, near), sums(s, far)
            r = (sf_.sum() / nf.sum()) / (sn.sum() / nn.sum())
            rd = ((C @ sf_) / (C @ nf)) / ((C @ sn) / (C @ nn))
            dd = rd - r_obs_d
            lo, hi = pct(dd, [2.5, 97.5])
            stat[m] = dict(ratio=float(r), ratio_obs=float(r_obs), diff=float(r - r_obs), lo=float(lo), hi=float(hi))
    return tab, stat


def d2_table(dec, p_tot, p_same, stack, clus, mask, C, models=("s1", "s2"), refs=("B1", "B1x")):
    rows = {}
    w_diff = np.where(p_tot > 0, 1 - p_same / np.where(p_tot > 0, p_tot, 1), np.nan)
    (wd,), _, ns = stackfirst([w_diff], stack, clus, mask, C)
    rows["pairs in different bins (share)"] = dict(mean=wd, stacks=ns)
    for m in models:
        for ref in refs:
            g, gd, gs = gap_parts(dec, p_tot, m, ref)
            (G, Gd, Gs, H), (dG, dGd, _, dH), n_ = stackfirst([g, gd, gs, gd - 0.5 * g], stack, clus, mask, C)
            glo, ghi = pct(dG, [2.5, 97.5])
            rows[f"{m} - {ref}"] = dict(gap=G, gap_lo=glo, gap_hi=ghi, diff_part=Gd, same_part=Gs,
                                        diff_share=(Gd / G if G != 0 else float("nan")), rule_b_stat=H, rule_b_hi=pct(dH, [97.5])[0],
                                        stacks=n_)
    return rows


def analyse(ptr, y, b, scores, stack, clus, rcls, rbin, nclus):
    """D1 and D2 for the sparse set S, its groups and the contrast group; rule statistics on S."""
    C = draws_matrix(nclus)
    dec, p_tot, p_same = decompose(ptr, y, b, {m: scores[m] for m in METH})
    groups = {"sparse S": (rcls <= 1) | (rbin <= 2), "recorded class 0": rcls == 0, "recorded class 1": rcls == 1,
              "records 1-3": rbin == 0, "records 4-6": rbin == 1, "records 7-10": rbin == 2, "records 21+ (contrast)": rbin == 4}
    out = {}
    for gname, mask in groups.items():
        tab, stat = decay(ptr, y, b, {m: scores[m] for m in ("s1", "s2", "B4p")}, mask, clus, C)
        out[gname] = dict(stack_repeats=int(mask.sum()), d1=tab, a=stat, d2=d2_table(dec, p_tot, p_same, stack, clus, mask, C))
    return out, dec, p_tot, p_same


def rule(res):
    S = res["sparse S"]
    r = {}
    for m in ("s1", "s2"):
        a = S["a"][m]["lo"] > 0
        bb = {ref: S["d2"][f"{m} - {ref}"]["gap_hi"] < 0 and S["d2"][f"{m} - {ref}"]["rule_b_hi"] < 0 for ref in ("B1", "B1x")}
        r[m] = dict(a=bool(a), b_B1=bool(bb["B1"]), b_B1x=bool(bb["B1x"]), both=bool(a and all(bb.values())))
    r["fix cycle worth considering"] = bool(r["s1"]["both"] and r["s2"]["both"])
    return r


# ---------------------------------------------------------------- loaders (Sara's tools)
def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_h3b_sparse_diagnostic.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class LoadLog:
    def __enter__(self):
        self.orig, self.paths = np.load, []
        def rec(f, *a, **k):
            self.paths.append(str(Path(f).resolve()) if isinstance(f, (str, Path)) else str(f))
            return self.orig(f, *a, **k)
        np.load = rec
        return self

    def __exit__(self, *e):
        np.load = self.orig
        return False


def freeze_check(freeze, data, runs_parent, tools_dir, file_md5):
    j = json.loads(Path(freeze).read_text())
    bad = [k for k, v in j["tools"].items() if not (tools_dir / k).is_file() or file_md5(tools_dir / k) != v]
    for k, v in j["data"].items():
        p = runs_parent / k if k.split("/")[0].startswith("f_seed") else data / k
        if not p.is_file() or file_md5(p) != v:
            bad.append(k)
    return bad, len(j["tools"]) + len(j["data"])


def load_bnm(S2, data):
    bnm, h2b = data / "bnm_fit", data / "stage_h2b_score"
    P, Sc = np.load(h2b / "h2b_predicted.npz"), np.load(h2b / "h2b_scores.npz")
    fb = dict(np.load(bnm / "bnm_frozen.npz"))
    man = {r["input_file"]: r for r in S2.D.rows_of(bnm)}
    Ls = S2.read_labels(bnm, fb, man)[0]
    n = len(fb["row"])
    ptr = P["hole_ptr"].astype(np.int64)
    item = np.repeat(np.arange(n), np.diff(ptr))
    y = Ls.reshape(n, -1)[item, P["hole_cell"].astype(np.int64)]
    sc = {"s1": P["score_s1"], "s2": P["score_s2"], "B1": P["score_B1"], "B1x": P["score_B1x"], "B4": P["score_B4"]}
    win = np.load(bnm / "bnm_stacks.npz")["window"].astype(str)
    wu, wi = np.unique(win, return_inverse=True)
    return dict(ptr=ptr, y=y.astype(bool), b=dist_bin(P["dist_h"].astype(float)), scores=sc, stack=P["stack_id"].astype(np.int64),
                clus=np.asarray(wi).ravel(), nclus=len(wu), rcls=P["rcls"].astype(int), rbin=P["rbin"].astype(int),
                y_strict=Sc["y_strict"], auc_ref={m: Sc[f"auc_strict_{m}"] for m in METH})


def load_realval(S2, data, runs, g2):
    HR, CK, E, D, W = S2.HR, S2.CK, S2.E, S2.D, S2.W
    fv = dict(np.load(data / "stage_d_frozen_val.npz"))
    hz, gz = np.load(data / "stage_d_headroom_pred.npz"), np.load(g2 / "g2_pred.npz")
    V, _, _, T, tc, comm = HR.load_stacks(data / "real_val", fv["files"], fv["file_index"], fv["row"], truth=True)
    voff, blank = E.frozen_draws(fv)
    full = {"B1": np.asarray(hz["B1"]), "B1x": np.asarray(gz["B1_exact_log"]), "B4": np.asarray(gz["B4_logit"])}
    for r, m in zip(runs, ("s1", "s2")):
        pk = np.load(Path(r) / "check_3a" / "preds_k50.npz")
        full[m] = CK.unplace(pk["counts"][:, W - 1][:, None].astype(np.float32), voff)[:, 0] / int(pk["K"])
    Mk = ~V[:, W - 1]
    ref = {m: HR.aucs(full[m], T, Mk) for m in METH}
    blank_ok = np.array_equal(np.asarray(blank).astype(bool), np.asarray(gz["blank"]).astype(bool))
    sel = np.nonzero(np.asarray(blank).astype(bool))[0]
    n = len(sel)
    rec_t = V[sel, W - 1].reshape(n, -1).sum(1); rec6 = V[sel].any(1).reshape(n, -1).sum(1)
    dist = np.asarray(gz["B4_dist"])[sel].astype(float)
    Mks = Mk[sel].reshape(n, -1)
    cnt = Mks.sum(1); ptr = np.r_[0, np.cumsum(cnt)].astype(np.int64)
    flat = lambda a: np.asarray(a)[sel].reshape(n, -1)[Mks]
    sc = {m: flat(full[m]).astype(np.float64) for m in METH}
    cu, ci = np.unique(np.asarray(comm)[sel], return_inverse=True)
    return dict(ptr=ptr, y=flat(T).astype(bool), b=dist_bin(dist.reshape(n, -1)[Mks]), scores=sc, stack=np.arange(n),
                clus=np.asarray(ci).ravel(), nclus=len(cu), rcls=np.asarray(D.tidx(rec_t)).astype(int),
                rbin=np.asarray(D.bidx(rec6)).astype(int), auc_ref={m: ref[m][sel] for m in METH}, blank_ok=blank_ok,
                rec_scored=int((V[:, W - 1] & Mk).sum()), n_all=len(fv["row"]))


def check_auc(dec, ref):
    worst, same_nan = 0.0, True
    for m in METH:
        a, r = dec[m]["auc"], np.asarray(ref[m], float)
        same_nan &= np.array_equal(np.isnan(a), np.isnan(r))
        ok = ~np.isnan(a) & ~np.isnan(r)
        if ok.any():
            worst = max(worst, float(np.abs(a[ok] - r[ok]).max()))
    return worst, same_nan


def brute_check(ptr, y, b, scores, dec, p_same, k=200, seed=7):
    """K2: on k stack-repeats (fixed seed) count every labelled / unlabelled pair by hand, split by same or different
    distance bin, and compare with the decomposition (U same, U different, same-bin pairs)."""
    n = len(ptr) - 1
    pick = np.random.default_rng(seed).choice(n, min(k, n), replace=False)
    worst = 0.0
    for i in pick:
        a, z = ptr[i], ptr[i + 1]
        yi, bi = y[a:z], b[a:z]
        pos, neg = np.nonzero(yi)[0], np.nonzero(~yi)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        same = bi[pos][:, None] == bi[neg][None, :]
        worst = max(worst, abs(float(same.sum()) - p_same[i]))
        for m, s in scores.items():
            si = s[a:z]
            dd = si[pos][:, None] - si[neg][None, :]
            c = (dd > 0) + 0.5 * (dd == 0)
            worst = max(worst, abs(float(c[same].sum()) - dec[m]["u_same"][i]), abs(float(c[~same].sum()) - dec[m]["u_diff"][i]))
    return worst


def run(data="../data", runs=("../runs/f_seed1", "../runs/f_seed2"), g2="../data/stage_g2_baselines_v2",
        freeze="../data/stage_h3_freeze/h3_freeze.json", out_dir="../data/stage_h3b_diagnostic", out=print):
    data, g2, od = Path(data), Path(g2), Path(out_dir)
    runs = [Path(r) for r in runs]
    if od.exists():
        sys.exit(f"STOP: {od} exists; write into a new folder")
    S2 = _load("stage_h2_bnm_score")
    fm = S2.X.file_md5
    res, rep = {}, {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")
    out(f"stage_h3b_sparse_diagnostic.py md5 {fm(Path(__file__))}; stage_h2_bnm_score.py {fm(HERE / 'stage_h2_bnm_score.py')}; "
        f"h3_freeze.json {fm(freeze)}")
    bad0, nf = freeze_check(freeze, data, runs[0].parent, HERE, fm)
    gate("K0", not bad0, f"before the run: {nf} md5s listed in h3_freeze.json = the files now; changed {bad0}")
    if bad0:
        out("stage H3b: stopped after K0"); return res, rep
    with LoadLog() as log:
        B = load_bnm(S2, data)
        R = load_realval(S2, data, runs, g2)
    for d_ in (B, R):
        d_["scores"]["B4p"] = 1 / (1 + np.exp(-np.clip(d_["scores"]["B4"].astype(float), -700, 700)))
    if broken() == "labels":
        B["y"] = np.roll(B["y"], 1)
    resB, decB, ptB, psB = analyse(B["ptr"], B["y"], B["b"], B["scores"], B["stack"], B["clus"], B["rcls"], B["rbin"], B["nclus"])
    yk = np.add.reduceat(B["y"].astype(int), B["ptr"][:-1])
    wB, nanB = check_auc(decB, B["auc_ref"])
    gate("K1", np.array_equal(yk, B["y_strict"].astype(int)) and wB <= 1e-12 and nanB,
         f"BNM: recorded holes per stack-repeat = h2b y_strict {np.array_equal(yk, B['y_strict'].astype(int))}; AUC of "
         f"{', '.join(METH)} per stack-repeat = h2b_scores, largest difference {wB:.1e}, same not-rankable set {nanB}")
    resR, decR, ptR, psR = analyse(R["ptr"], R["y"], R["b"], R["scores"], R["stack"], R["clus"], R["rcls"], R["rbin"], R["nclus"])
    idw = max(brute_check(B["ptr"], B["y"], B["b"], {m: B["scores"][m] for m in METH}, decB, psB),
              brute_check(R["ptr"], R["y"], R["b"], {m: R["scores"][m] for m in METH}, decR, psR))
    gate("K2", idw <= 1e-9, f"decomposition = every pair counted by hand on 200 stack-repeats (BNM) and 200 stacks (real_val): "
         f"same-bin and different-bin concordance of {len(METH)} methods and same-bin pair counts, largest difference {idw:.1e}")
    wR, nanR = check_auc(decR, R["auc_ref"])
    gate("K3", R["blank_ok"] and wR <= 1e-12 and nanR and R["rec_scored"] == 0,
         f"real_val: blank = g2_pred.npz {R['blank_ok']}; {len(R['ptr']) - 1:,} habitat-blanked stacks of {R['n_all']:,}; AUC of "
         f"every method = stage D's HR.aucs, largest difference {wR:.1e}, same not-rankable set {nanR}; target-year records "
         f"among scored cells {R['rec_scored']}")
    bad_paths = [p for p in log.paths if "bnm_test" in p or "real_eval" in p or "bnm_grids" in p]
    if broken() == "testhalf":
        bad_paths.append("bnm_test/x")
    marker = (data / "BNM_TEST_EXPORTED").exists() or (data / "bnm_test").exists()
    gate("K4", not bad_paths and not marker, f"files passed to np.load {len(log.paths)}, of the test half, real_eval or the BNM "
         f"grids {len(bad_paths)}; test-half export present {marker}")
    if not all(res.values()):
        n_ok = sum(res.values())
        out(f"stage H3b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; stopped before any result is printed, nothing written")
        return res, rep
    for title, rr in (("BNM fit half (label = recorded in 2019; holes)", resB), ("real_val, habitat blanked (label = occupied; unrecorded cells)", resR)):
        out(f"   == {title}")
        for gname, g in rr.items():
            out(f"   {gname} ({g['stack_repeats']:,} items)")
            out("      D1 bin: cells, labelled share | mean score s1, s2, B4 chance: " + "; ".join(
                f"{k} {v['cells']:,} {v['labelled']:.3f} | {v['s1']:.3f} {v['s2']:.3f} {v['B4p']:.3f}" for k, v in g["d1"].items()))
            out("      D1 far/near ratio: labelled {:.3f}; ".format(g["a"]["s1"]["ratio_obs"]) + "; ".join(
                f"{m} {g['a'][m]['ratio']:.3f} (minus labelled {g['a'][m]['diff']:+.3f} [{g['a'][m]['lo']:+.3f}, {g['a'][m]['hi']:+.3f}])"
                for m in ("s1", "s2", "B4p")))
            d2 = g["d2"]
            out(f"      D2 pairs in different distance bins {d2['pairs in different bins (share)']['mean']:.3f}; " + "; ".join(
                f"{k}: gap {v['gap']:+.4f} [{v['gap_lo']:+.4f}, {v['gap_hi']:+.4f}] = different-bin {v['diff_part']:+.4f} + same-bin "
                f"{v['same_part']:+.4f} (different-bin share {v['diff_share']:.2f}) ({v['stacks']} stacks)"
                for k, v in d2.items() if k.startswith("s")))
    rl = rule(resB)
    K0b = freeze_check(freeze, data, runs[0].parent, HERE, fm)[0]
    gate("K0b", not K0b, f"after the run: h3_freeze.json files unchanged; changed {K0b}")
    od.mkdir(parents=True)
    rep.update(gates=res, bnm=resB, real_val=resR, rule=rl, bins=list(DBIN), near=list(NEAR), far=list(FAR),
               md5=dict(tool=fm(Path(__file__)), freeze=fm(freeze)))
    clean = lambda o_: json.loads(json.dumps(o_, default=lambda v: v.item() if hasattr(v, "item") else str(v)).replace("-Infinity", "null").replace("Infinity", "null").replace("NaN", "null"))
    (od / "h3b_report.json").write_text(json.dumps(clean(rep), indent=1))
    n_ok = sum(res.values())
    out(f"   decision rule (signed 12 Nov), BNM sparse set: " + "; ".join(
        f"{m} (a) {rl[m]['a']}, (b) vs B1 stored {rl[m]['b_B1']}, vs B1 exact {rl[m]['b_B1x']}" for m in ("s1", "s2")))
    out(f"stage H3b: gates {n_ok} PASS, {len(res) - n_ok} FAIL; fix cycle worth considering by the signed rule: "
        f"{'YES' if rl['fix cycle worth considering'] else 'NO'}; h3b_report.json {fm(od / 'h3b_report.json')}")
    return res, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--runs", nargs="+", default=["../runs/f_seed1", "../runs/f_seed2"])
    ap.add_argument("--g2", default="../data/stage_g2_baselines_v2")
    ap.add_argument("--freeze", default="../data/stage_h3_freeze/h3_freeze.json")
    ap.add_argument("--out", default="../data/stage_h3b_diagnostic")
    a = ap.parse_args()
    if a.self_test:
        tp = HERE.parent / "tests" / "test_21_h3b_diagnostic.py"
        if not tp.is_file():
            sys.exit(f"STOP: {tp} not found (the self-test of this tool lives there)")
        spec = importlib.util.spec_from_file_location("test_21_h3b_diagnostic", tp)
        T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
        sys.exit(0 if T.self_test(Path(__file__)) else 1)
    r, _ = run(a.data, a.runs, a.g2, a.freeze, a.out)
    sys.exit(0 if r and all(r.values()) else 1)