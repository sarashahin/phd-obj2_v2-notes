"""
test_06_rarity_order.py  (obj1_v2, written 23 Oct 2026)

Question: are the rare species ecologically rare, or just recent invaders still sitting
in their founding cells? Assembly appends newcomers and pruning keeps the order, so a
species' list position is its arrival order (new files also save species_id, which
increases with arrival).
Prints, over distinct communities: Spearman correlation of position and range size;
share of species with ranges of 5 cells or fewer among the oldest and newest 20%;
median range in each; where the species lost in the final run sit in the list.
Ecological reading: if small ranges and losses pile up at the newest end, rarity is
recency and the final run must last until transient species resolve. If small ranges
are spread over all ages, the cause is the world itself (frozen ranges, founding rule).
Run from obj1_v2/work:
    python3 ../tests/test_06_rarity_order.py "../../results/data/data_eval_unseen/*_training.npz"
Results 23 Oct (25 evaluation communities, old code): Spearman -0.178 (-0.195 to -0.139);
ranges of 5 cells or fewer 87.3% of the oldest 20% and 99.2% of the newest; median range
2 and 1 cells; losses at median position 0.564, 24.8% of them in the newest 20%.
"""
import sys, glob
import numpy as np
from scipy.stats import spearmanr
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "../../results/data/data_eval_unseen/*_training.npz"))
if not files: sys.exit("no files matched")
seen, out = set(), []
for f in files:
    z = np.load(f, allow_pickle=True)
    P = z["P_last_final"].astype(bool)
    h = hash(P.tobytes())
    if h in seen: continue
    seen.add(h)
    S = P.shape[0]; R = P.sum(axis=(1, 2)); pos = np.arange(S) / S          # list position = arrival order (0 oldest)
    ext = z["IBM_ext_step"] if "IBM_ext_step" in z.files else np.full(S, -1.0)
    lost = (ext >= 0) & (R == 0)
    alive = R > 0; rare = alive & (R <= 5)
    old, new = pos < 0.2, pos >= 0.8
    out.append(dict(S=S, rho=spearmanr(pos[alive], R[alive])[0],
                    rare_old=rare[old].sum() / max(1, alive[old].sum()), rare_new=rare[new].sum() / max(1, alive[new].sum()),
                    med_R_old=np.median(R[old & alive]), med_R_new=np.median(R[new & alive]),
                    ext_pos_median=np.median(pos[lost]) if lost.any() else np.nan,
                    ext_share_new=(lost & new).sum() / max(1, lost.sum()),
                    target_share_new=(new & (R > 5)).sum() / max(1, (R > 5).sum())))
print(f"{len(out)} distinct communities; list position 0 = first arrival, 1 = last")
for k, lab in [("rho", "Spearman(position, range size), living species"),
               ("rare_old", "share of living species with range <= 5 cells, oldest 20%"),
               ("rare_new", "share of living species with range <= 5 cells, newest 20%"),
               ("med_R_old", "median range, oldest 20%"), ("med_R_new", "median range, newest 20%"),
               ("ext_pos_median", "median list position of species lost in the final run"),
               ("ext_share_new", "share of final-run losses that are in the newest 20%"),
               ("target_share_new", "share of species with range > 5 cells (the paper's analysed set) in the newest 20%")]:
    v = np.array([o[k] for o in out], float); v = v[~np.isnan(v)]
    print(f"  {lab:<88} " + (f"median {np.median(v):.3f}  (range {v.min():.3f} to {v.max():.3f})" if v.size else "none"))
