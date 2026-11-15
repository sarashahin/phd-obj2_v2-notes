"""Stage F step 3a: recompute the report from per_stack.csv alone (independent of stage_f_check.py).

Checks that the class means, the balanced AUC and the F5 means in report.json follow from the per-stack rows,
then prints two readings the report does not hold: how many communities have the model below B1 per class,
and the 1-cell comparison split by habitat blanking (exploratory, no gate).

Run from obj2_v2/work:
    python3 ../tools/stage_f_recompute.py --run ../runs/f_seed1
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]
COLS = {"model": "auc_model", "model_k8": "auc_model_k8", "B1": "auc_b1",
        "B2_halfblank": "auc_b2_halfblank", "B2": "auc_b2", "null": "auc_null"}


def boot(x, n_boot=2000, seed=31):
    """95% interval of the pooled per-stack mean over resampled communities (columns community, d)."""
    pc = x.groupby("community").d.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    s, c = pc["sum"].to_numpy(), pc["count"].to_numpy()
    m = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(pc), len(pc)) for _ in range(n_boot))]
    return np.percentile(m, [2.5, 97.5])


def main(run, tol=1e-4):
    d0 = pd.read_csv(Path(run) / "check_3a" / "per_stack.csv", dtype={"true_class": str})
    rep = json.loads((Path(run) / "check_3a" / "report.json").read_text().replace("NaN", "null"))
    d = d0.dropna(subset=["auc_model"])
    print(f"rows {len(d0)}, scored {len(d)}, communities {d0.community.nunique()}")
    bad = 0
    m = d.groupby("true_class")[list(COLS.values())].mean().loc[CLASSES]
    for k, c in COLS.items():
        b = m[c].mean()
        ok = abs(b - rep["auc_balanced"][k]) < tol
        bad += not ok
        print(f"   balanced {k:13s} {b:.4f} (report {rep['auc_balanced'][k]:.4f}) {'ok' if ok else 'DIFFERS'}")
    d = d.assign(d=d.auc_model - d.auc_b1)
    print("   F5 model - B1 per class: mean (report) | communities with model below B1")
    for c in CLASSES:
        x = d[d.true_class == c]
        r = rep["F5"]["by_class"][c]
        ok = abs(x.d.mean() - r["mean"]) < tol and len(x) == r["n"]
        bad += not ok
        pc = x.groupby("community").d.mean()
        print(f"   {c:6s} n {len(x):5d} {x.d.mean():+.4f} ({r['mean']:+.4f}) {'ok' if ok else 'DIFFERS'} | "
              f"{int((pc < 0).sum())} of {len(pc)}, range {pc.min():+.3f} to {pc.max():+.3f}")
    print("   model - B2 full habitat per class (not in the report): mean [95% interval over communities]")
    for c in CLASSES:
        x = d[d.true_class == c].assign(d=lambda z: z.auc_model - z.auc_b2)
        lo, hi = boot(x)
        print(f"   {c:6s} {x.d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("   1 cell split by habitat blanking (exploratory, no gate): model - B1, 95% interval over communities")
    for b in (0, 1):
        x = d[(d.true_class == "1") & (d.blank == b)]
        lo, hi = boot(x)
        print(f"   blank {b}: n {len(x)}, model {x.auc_model.mean():.3f}, B1 {x.auc_b1.mean():.3f}, "
              f"B2 full {x.auc_b2.mean():.3f}, model - B1 {x.d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"stage F recompute: {'report reproduced' if bad == 0 else f'{bad} values differ'}")
    return bad


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run folder holding check_3a/per_stack.csv and report.json")
    raise SystemExit(main(p.parse_args().run))