"""
test_24_h6_outputs_recheck.py  (obj2_v2, stage H6, 12 Nov 2026; read-only, after the one reading)

Recomputes the H6 numbers from per_stack_h6.csv, per_window_h6.csv and h6_report.json, outside the tool: no tool import,
pandas and numpy only. class_boot is copied from stage_f_check.py (window resampling, numpy rng [20261108, 31, 3], 2,000
draws); hr1_core from stage_h2_bnm_score.py (stack-first means, every window kept as a NaN placeholder).
Checks: T1 totals, T2 hole shares, T3 rankable set; R1 every H-R1 row (strict truth, seeds, pooled, B3-B5); R2 AUC group
means; R3 every H-R2 row (ratio, coverage, median width, counts); R4 Red List; R5 the 10+ version; R6 per window.
The csv files round AUCs to 6 decimals and interval ends to 3, so the tolerances are 2e-6 (means), 5e-6 (interval ends),
2e-4 (ratios), 0.01 (coverage). Pooled truth and D1 / D2 need cell-level data and are not rechecked here (Z6 checked D2).

Run (from PSD_Dispersal_pool/obj2_v2/work):
    python3 ../tests/test_24_h6_outputs_recheck.py ../data/stage_h6_test_score
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(sys.argv[1])
ps = pd.read_csv(next(D.glob("*per_stack_h6.csv")), dtype={"recorded_class": str, "records_bin": str, "status": str},
                 keep_default_na=False, na_values=[""])
pw = pd.read_csv(next(D.glob("*per_window_h6.csv")))
rep = json.loads(next(D.glob("*h6_report.json")).read_text())
TCLS = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
BINS = ["1-3", "4-6", "7-10", "11-20", "21+"]
GRP = {"recorded class": ("recorded_class", TCLS), "records over 6 years": ("records_bin", BINS)}
fails = []
N_CHECK = [0]


def check(name, ok, txt):
    N_CHECK[0] += 1
    print(f"{name}: {'OK' if ok else 'DIFFERS'}  {txt}")
    if not ok:
        fails.append(name)


def class_boot(d, tc, comm, classes, n_boot=2000):
    ok = ~np.isnan(d)
    cu = np.unique(comm)
    ci = np.searchsorted(cu, comm)
    S = np.zeros((len(cu), len(classes))); N = np.zeros_like(S)
    for j, c in enumerate(classes):
        m = ok & (tc == c)
        np.add.at(S[:, j], ci[m], d[m]); np.add.at(N[:, j], ci[m], 1)
    g = np.random.default_rng([20261108, 31, 3])
    w = np.stack([np.bincount(g.integers(0, len(cu), len(cu)), minlength=len(cu)) for _ in range(n_boot)])
    with np.errstate(invalid="ignore", divide="ignore"):
        bm = (w @ S) / (w @ N)
        mean = S.sum(0) / N.sum(0)
    out = {}
    for j, c in enumerate(classes):
        lo, hi = np.nanpercentile(bm[:, j], [2.5, 97.5]) if N[:, j].sum() else (np.nan, np.nan)
        out[c] = dict(n=int(N[:, j].sum()), mean=float(mean[j]), lo=float(lo), hi=float(hi))
    return out


win_all = ps["window"].to_numpy().astype(str)
stid = ps["stack_id"].to_numpy()


def hr1_core(d, gi, ncls):
    ok = ~np.isnan(d)
    wu = np.unique(win_all)
    t = pd.DataFrame(dict(s=stid[ok], g=gi[ok], w=win_all[ok], d=d[ok]))
    a = t.groupby(["s", "g"], sort=True).agg(d=("d", "mean"), w=("w", "first")).reset_index()
    dd = np.concatenate([a["d"].to_numpy(), np.full(len(wu), np.nan)])
    gg = np.concatenate([a["g"].to_numpy(), np.zeros(len(wu), int)])
    ww = np.concatenate([a["w"].to_numpy().astype(str), wu])
    return class_boot(dd, gg, ww, list(range(ncls)))


def verdict(m, hi):
    return "PASS" if m >= 0 else ("FAIL" if hi < 0 else "NOT RESOLVED")


A = {m: ps[f"auc_{m}"].to_numpy(float) for m in ("s1", "s2", "pooled", "B1", "B1x", "B3", "B4", "B5")}
gidx = {g: (ps[col].map({l: i for i, l in enumerate(labs)}).to_numpy(), labs) for g, (col, labs) in GRP.items()}

# ---- totals
y = ps["recorded_holes_strict"].to_numpy(float); nh = ps["holes"].to_numpy(float)
rank = ~np.isnan(A["s1"])
check("T1 totals", len(ps) == rep["n"] and ps["stack_id"].nunique() == rep["stacks"] and int(rank.sum()) == rep["rankable"]
      and ps["stack_id"][rank].nunique() == rep["rankable_stacks"] and int(nh.sum()) == 481676,
      f"rows {len(ps)}, stacks {ps['stack_id'].nunique()}, rankable {int(rank.sum())} ({ps['stack_id'][rank].nunique()} stacks), "
      f"holes {int(nh.sum()):,}, windows {ps['window'].nunique()}")
sh_s, sh_p = y.sum() / nh.sum(), ps["recorded_holes_pooled"].sum() / nh.sum()
check("T2 hole shares", abs(sh_s - rep["hole_share"]["strict"]) < 1e-12 and abs(sh_p - rep["hole_share"]["pooled"]) < 1e-12,
      f"strict {sh_s:.4f}, pooled {sh_p:.4f}")
rk2 = (y > 0) & (y < nh)
check("T3 rankable = 0 < recorded < holes", np.array_equal(rk2, rank), f"{int(rk2.sum())}")

# ---- H-R1 strict
worst = dict(mean=0.0, lohi=0.0); vsame = True; nsame = True
def cmp_rows(rows_rep, cb, labs):
    global vsame, nsame
    for j, lab in enumerate(labs):
        r = rows_rep[j]; c = cb[j]
        worst["mean"] = max(worst["mean"], abs(r["mean"] - c["mean"]))
        worst["lohi"] = max(worst["lohi"], abs(r["lo"] - c["lo"]), abs(r["hi"] - c["hi"]))
        nsame &= r["n"] == c["n"] and r["group"] == lab
        v = verdict(c["mean"], c["hi"]) if c["n"] >= 50 else "reported"
        vsame &= v == r["verdict"]
for m in ("s1", "s2", "pooled"):
    for b in ("B1", "B1x"):
        for g, (gi, labs) in gidx.items():
            cmp_rows(rep["H_R1"][m][b]["rows"][g], hr1_core(A[m] - A[b], gi, len(labs)), labs)
for m in ("B3", "B4", "B5"):
    for g, (gi, labs) in gidx.items():
        cmp_rows(rep["H_R1_baselines"][m]["rows"][g], hr1_core(A[m] - A["B1"], gi, len(labs)), labs)
check("R1 H-R1 strict, 9 comparisons x 12 groups", worst["mean"] < 2e-6 and worst["lohi"] < 5e-6 and vsame and nsame,
      f"largest difference: mean {worst['mean']:.1e}, interval ends {worst['lohi']:.1e}; stack counts and verdicts same {vsame and nsame} "
      f"(csv AUCs carry 6 decimals)")

# ---- AUC by group (report table): spot values from the log
log_auc = {("s1", "recorded class", "0"): 0.639, ("s1", "records over 6 years", "21+"): 0.613, ("B1", "recorded class", "1"): 0.709,
           ("B1x", "recorded class", "1"): 0.731, ("B4", "recorded class", "51+"): 0.527, ("s2", "records over 6 years", "7-10"): 0.788}
ok_auc = True
for (m, g, lab), v in log_auc.items():
    gi, labs = gidx[g]
    c = hr1_core(A[m], gi, len(labs))[labs.index(lab)]["mean"]
    ok_auc &= round(c, 3) == v
check("R2 AUC group means (6 log values)", ok_auc, "as printed to 3 decimals")

# ---- H-R2
cols = {"s1": "s1", "s2": "s2", "pooled": "pooled", "B3": "B3", "B4": "B4", "B5": "B5"}
wr = dict(ratio=0.0, width=0.0); cov_d = []; st_same = True
for m in cols:
    mean = ps[f"hr2_mean_{m}"].to_numpy(float); lo = ps[f"hr2_lo_{m}"].to_numpy(float); hi = ps[f"hr2_hi_{m}"].to_numpy(float)
    cov = (y >= lo) & (y <= hi)
    for g, (gi, labs) in gidx.items():
        for j, lab in enumerate(labs):
            mm = gi == j
            r = next(x for x in rep["H_R2"][m]["rows"][g] if x["group"] == lab)
            ratio = mean[mm].mean() / y[mm].mean() if y[mm].mean() > 0 else np.nan
            if not (np.isnan(ratio) and r["ratio"] is None):
                wr["ratio"] = max(wr["ratio"], abs(ratio - r["ratio"]))
            wr["width"] = max(wr["width"], abs(np.median(hi[mm] - lo[mm]) - r["width"]))
            cov_d.append((abs(cov[mm].mean() - r["cov"]), m, g, lab))
            st_same &= ps["stack_id"][mm].nunique() == r["stacks"] and int(mm.sum()) == r["n"]
cd = max(cov_d)
check("R3 H-R2 ratio, width, coverage, counts (6 methods x 12 groups)", wr["ratio"] < 2e-4 and wr["width"] < 2e-3 and cd[0] < 0.01 and st_same,
      f"largest difference: ratio {wr['ratio']:.1e}, median width {wr['width']:.1e}, coverage {cd[0]:.4f} ({cd[1]} {cd[3]}); "
      f"stack and stack-repeat counts same {st_same} (csv ends carry 3 decimals)")
print("   observed recorded holes per stack-repeat by recorded class: " + "; ".join(
    f"{lab} {y[gidx['recorded class'][0] == j].mean():.2f}" for j, lab in enumerate(TCLS)) + f"; all {y.mean():.2f}, zero {(y == 0).mean():.3f}")
print("   H-R2 seed 1 mean corrected count by recorded class: " + "; ".join(
    f"{lab} {ps['hr2_mean_s1'][gidx['recorded class'][0] == j].mean():.2f}" for j, lab in enumerate(TCLS)))

# ---- Red List
st = ps["status"].fillna("").replace("", "no status").to_numpy().astype(str)
stl = sorted(set(st.tolist())); sti = np.array([stl.index(s) for s in st])
rl_w = 0.0; rl_n = True
for m in ("s1", "s2"):
    cb = hr1_core(A[m] - A["B1"], sti, len(stl))
    for j, s in enumerate(stl):
        r = rep["red_list"][m]["hr1"].get(s)
        if r is None:
            rl_n = False; continue
        rl_n &= r["n"] == cb[j]["n"]
        if cb[j]["n"]:
            rl_w = max(rl_w, abs(r["mean"] - cb[j]["mean"]), abs(r["lo"] - cb[j]["lo"]), abs(r["hi"] - cb[j]["hi"]))
check("R4 Red List H-R1 (seed - B1 stored)", rl_w < 5e-6 and rl_n, f"statuses {stl}; largest difference {rl_w:.1e}")
print("   distinct scored stacks / rankable stacks by status: " + "; ".join(
    f"{s} {ps['stack_id'][st == s].nunique()}/{ps['stack_id'][(st == s) & rank].nunique()}" for s in stl))

# ---- 7 Nov version
sub = ps["all_rec_2019"].to_numpy() >= 10
w7 = 0.0
for m in ("s1", "s2"):
    for b in ("B1", "B1x"):
        cb = hr1_core(A[m] - A[b], np.where(sub, 0, -1), 1)[0]
        r = rep["seven_nov"]["hr1"][m][b]
        w7 = max(w7, abs(r["mean"] - cb["mean"]), abs(r["lo"] - cb["lo"]), abs(r["hi"] - cb["hi"]))
check("R5 10+ records in 2019", w7 < 5e-6 and int(sub.sum()) == 7845 and ps["stack_id"][sub].nunique() == 1569,
      f"{int(sub.sum())} stack-repeats, {ps['stack_id'][sub].nunique()} stacks; largest difference {w7:.1e}")

# ---- per window
pwr = []
for w in sorted(ps["window"].unique()):
    t = ps[ps["window"] == w]; yy = t["recorded_holes_strict"].to_numpy(float)
    row = dict(window=w, stack_repeats=len(t), stacks=t["stack_id"].nunique(), rankable=int(t["auc_s1"].notna().sum()))
    for m in ("s1", "s2"):
        d = (t[f"auc_{m}"] - t["auc_B1"]).to_numpy(float); ok = ~np.isnan(d)
        row[f"mean_d_{m}_B1"] = pd.Series(d[ok]).groupby(t["stack_id"].to_numpy()[ok]).mean().mean() if ok.any() else np.nan
        row[f"hr2_ratio_{m}"] = t[f"hr2_mean_{m}"].mean() / yy.mean()
        row[f"hr2_cov_{m}"] = ((yy >= t[f"hr2_lo_{m}"]) & (yy <= t[f"hr2_hi_{m}"])).mean()
    pwr.append(row)
pwr = pd.DataFrame(pwr).set_index("window"); pwx = pw.set_index("window").loc[pwr.index]
dif = {c: float(np.nanmax(np.abs(pwr[c].to_numpy(float) - pwx[c].to_numpy(float)))) for c in pwr.columns}
check("R6 per window", len(pw) == 57 and dif["stack_repeats"] == 0 and dif["stacks"] == 0 and dif["rankable"] == 0
      and dif["mean_d_s1_B1"] < 2e-6 and dif["hr2_ratio_s1"] < 2e-4 and dif["hr2_cov_s1"] < 0.02 and pw["stack_repeats"].sum() == 10366,
      f"{len(pw)} windows; largest difference " + ", ".join(f"{k} {v:.1e}" for k, v in dif.items()))
for m in ("s1", "s2"):
    v = pw[f"mean_d_{m}_B1"]
    print(f"   {m} - B1 stored per window: below 0 in {(v < 0).sum()} of {len(v)} (range {v.min():+.3f} to {v.max():+.3f}); "
          f"H-R2 ratio range {pw[f'hr2_ratio_{m}'].min():.2f} to {pw[f'hr2_ratio_{m}'].max():.2f}, coverage {pw[f'hr2_cov_{m}'].min():.2f} "
          f"to {pw[f'hr2_cov_{m}'].max():.2f}")
print(f"\nrecheck: {N_CHECK[0] - len(fails)} of {N_CHECK[0]} OK" + (f"; differs: {fails}" if fails else ""))
sys.exit(1 if fails else 0)