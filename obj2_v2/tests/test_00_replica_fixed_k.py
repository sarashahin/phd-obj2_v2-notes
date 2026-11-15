#!/usr/bin/env python3
"""
test_00_replica_fixed_k.py  (obj2_v2, 7 Nov 2026)

What it does
    Re-implements, line for line in numpy, _sparsify_all_frames_fixed_K from the project copy of
    training_v2_patch.py (13 May 2026), the function that made the "K = 5 or 10 records" in the old
    training (phases 3 and 4, every history frame), and prints how much of each range it reveals.
    Reads no files, writes nothing. It is a replica: the real function is tested on Sara's own copy
    in stage A (test_01_old_sparsifiers.py, to be written after her yes).

Why
    The old code draws K random occupied cells per species and frame, then multiplies by the
    occupancy mask. If a species holds K cells or fewer, every occupied cell is kept, so the
    "sparse" input is the whole range. In the old communities 87.3% of the oldest and 99.2% of the
    newest species held 5 cells or fewer (rarity check, 23 Oct), so most species were trained with
    their full range as input.

Labels (as in obj1_v2/tests): PASS/FAIL = the replica does what the old code does; FLAG = the old
code does what it says but works against the aim.

Run (from PSD_Dispersal_pool/obj2_v2/work, numpy only)
    python3 ../tests/test_00_replica_fixed_k.py | tee ../logs/test_00_replica_fixed_k_7nov.txt

Expected output (the same on any machine: records = min(K, range) whatever cells are drawn)
    12 lines, e.g. "range   5 cells, K  5: records   5, share of range revealed 1.00"
    check 1 records lie inside the range: PASS
    check 2 records = min(K, range) in every case: PASS
    check 3 ranges of K cells or fewer fully revealed: 5 of 5 cases -> FLAG
"""
import numpy as np

rng = np.random.default_rng(0)


def fixed_k(flat, K):
    scores = np.where(flat > 0.5, rng.random(flat.shape), -np.inf)   # torch.where(flat > 0.5, rand, -inf)
    top = np.argsort(-scores, axis=-1)[..., :K]                       # scores.topk(k=K)
    out = np.zeros_like(flat)
    np.put_along_axis(out, top, 1.0, axis=-1)                         # scatter_(-1, top_idx, 1.0)
    return out * (flat > 0.5)                                         # "* (flat > 0.5).float()"


G = 400
rows, inside = [], True
for n in (3, 5, 10, 12, 50, 200):
    for K in (5, 10):
        f = np.zeros((1, G))
        f[0, rng.choice(G, n, replace=False)] = 1
        s = fixed_k(f, K)
        inside &= bool((s <= f).all())
        rows.append((n, K, int(s.sum())))
        print(f"range {n:3d} cells, K {K:2d}: records {int(s.sum()):3d}, share of range revealed {s.sum() / n:.2f}")

small = [(n, K, r) for n, K, r in rows if n <= K]
full = sum(r == n for n, K, r in small)
print(f"check 1 records lie inside the range: {'PASS' if inside else 'FAIL'}")
print(f"check 2 records = min(K, range) in every case: {'PASS' if all(r == min(n, K) for n, K, r in rows) else 'FAIL'}")
print(f"check 3 ranges of K cells or fewer fully revealed: {full} of {len(small)} cases -> "
      f"{'FLAG (the old rule gives the whole range as input)' if full else 'ok'}")