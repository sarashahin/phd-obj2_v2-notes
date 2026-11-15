"""
test_07_shared_seed.py  (obj1_v2, written 23 Oct 2026)

Question: do different communities secretly share species? The old run_all_rps.py gave
every community assembly seed 1, so early arrivals could get the same interactions and
founding cells everywhere (test_03 B2 showed this in the code). If so, training and
evaluation communities are not independent.
Compares each evaluation community with up to 12 others: Jaccard overlap of species at
the same list position, and the best match within +-50 positions (pruning shifts
positions), each against the same community moved to a random place on the torus.
Ecological reading: unrelated communities give observed = shifted. Higher observed
values, mostly for the first arrivals, mean shared species (data leakage across the
train/evaluation split).
Run from obj1_v2/work:
    python3 ../tests/test_07_shared_seed.py "../../results/data/data_eval_unseen/*_training.npz" "../../results/data/*_training.npz"
24 Oct fix: files without P_last_final (assembly-pool and default-run files) are skipped
and listed; before, the first such file stopped the test (KeyError).
Tested on mocks 23 Oct: shared founding gives 0.091 against 0.002 (first 50 arrivals,
same position) and 0.338 against 0.187 (best within 50); independent founding gives
0.005 against 0.004 and 0.187 against 0.186. Not yet run on Sara's files.
"""
import sys, glob
import numpy as np
EVAL = sys.argv[1] if len(sys.argv) > 1 else "../../results/data/data_eval_unseen/*_training.npz"
OTHER = sys.argv[2] if len(sys.argv) > 2 else "../../results/data/*_training.npz"
MAX_OTHER, N_MAX, W = 12, 1500, 50          # communities to compare against, species per list, position window
rng = np.random.default_rng(0)

skipped = []
def load(pattern, limit=None):
    out, seen = {}, set()
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if "P_last_final" not in z.files:                # assembly-pool or default-run files hold no final map
            skipped.append(f.split("/")[-1]); continue
        P = z["P_last_final"].astype(bool)
        h = hash(P.tobytes())
        if h in seen: continue
        seen.add(h); out[f] = P
        if limit and len(out) >= limit: break
    return out

def jac(X, Y):                                   # full Jaccard matrix between rows of X and rows of Y
    X = X.reshape(len(X), -1).astype(np.float32); Y = Y.reshape(len(Y), -1).astype(np.float32)
    inter = X @ Y.T
    return inter / np.maximum(X.sum(1)[:, None] + Y.sum(1)[None, :] - inter, 1)

ev, ot = load(EVAL), load(OTHER, MAX_OTHER)
if not ev: sys.exit("no evaluation files matched")
label = OTHER if ot and set(ot) != set(ev) else "the evaluation communities themselves"
if not ot: ot = ev
print(f"evaluation communities: {len(ev)}; compared with {len(ot)} communities from {label}")
if skipped: print(f"skipped {len(set(skipped))} files without P_last_final: {sorted(set(skipped))}")
res = {k: [] for k in ("same", "same_null", "win", "win_null")}
age, posl = [], []
for fa, A in ev.items():
    for fb, B in ot.items():
        if fa == fb: continue
        n = min(len(A), len(B), N_MAX)
        a, b = A[:n], B[:n]
        Bs = np.roll(b, tuple(rng.integers(1, 20, 2)), axis=(1, 2))      # same community, moved on the torus
        alive = a.sum(axis=(1, 2)) > 0
        i, j = np.indices((n, n)); near = np.abs(i - j) <= W
        for key, M in (("win", jac(a, b)), ("win_null", jac(a, Bs))):
            res[key].append(np.where(near, M, 0).max(1)[alive])
            if key == "win": res["same"].append(np.diag(M)[alive])
            else: res["same_null"].append(np.diag(M)[alive])
        age.append((np.arange(n) / n)[alive]); posl.append(np.arange(n)[alive])
res = {k: np.concatenate(v) for k, v in res.items()}; age = np.concatenate(age); pos = np.concatenate(posl)
print(f"species compared: {res['win'].size}; positions are places in the arrival-ordered list (0 = first arrival), first {N_MAX} species")
for lab, m in (("pos 0-49", pos < 50), ("pos 0-299", pos < 300), ("pos 600-899", (pos >= 600) & (pos < 900)), ("pos 1200+", pos >= 1200), ("all", pos >= 0)):
    if not m.any(): continue
    print(f"  {lab:<11} same position: mean Jaccard {res['same'][m].mean():.4f} (torus-shifted {res['same_null'][m].mean():.4f})"
          f" | best within +-{W} positions: {res['win'][m].mean():.4f} (torus-shifted {res['win_null'][m].mean():.4f})")