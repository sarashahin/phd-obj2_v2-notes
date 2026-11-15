#!/usr/bin/env python3
"""
range_class_count.py  (obj2_v2, stage D step 1, read-only, 8 Nov 2026)

What it does
    Counts, over the real-run training and validation communities, how many species-stacks fall in each true range
    class, before any export or training, to answer "is the simulated data enough at the rare end?"
    (claude/obj2_v2_plan.md, "Is the simulated data enough?", check 1).
      A candidate target is the last piece (year) of a round in the last half of the assembly that has all M pieces;
      its stack is the target year and the 5 years before it (pieces M-6 .. M-1), as agreed for P1.
      A species-stack is one species present in at least one of those 6 years. Its class is its true range at the
      target year: 1, 2-5, 6-10, 11-20, 21-50, 51+ cells, or 0 (present earlier in the window, gone by the target).
    Per split and class it prints: species-stacks over all candidate rounds; communities giving at least one; distinct
    species (community, species id) ever in the class; the median per community and how many communities give none;
    and the same stacks for the planned design of 10 rounds spread evenly over the candidates.
    Recorded classes (after the survey) are not counted here; they need the recording rule (stage C export).
    real_eval is refused. Writes nothing except what it prints. Needs numpy only.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/range_class_count.py --self-test | tail -1
    python3 ../tools/range_class_count.py --train "../../obj1_v2/work/results/stage2a/real_train/*_training.npz" \
        --val "../../obj1_v2/work/results/stage2a/real_val/*_training.npz" | tee ../logs/range_class_count_8nov.txt
"""
import argparse, glob, os, sys, tempfile
import numpy as np

G = 20
CLASSES = (("0", 0, 0), ("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), ("51+", 51, G * G))
WINDOW, N_DESIGN = 6, 10


def cls(n):
    return next(lab for lab, lo, hi in CLASSES if lo <= n <= hi)


def community(path):
    """Per candidate round: {species id: target-year cells} for species present in the 6-year window."""
    z = np.load(path, allow_pickle=True)
    M = int(np.asarray(z["HABITAT_substeps"]).item())
    n_rounds = len(np.asarray(z["ENV_habitat_t"]))
    rnd, pc, off = (np.asarray(z[k]) for k in ("ASM_F_round", "ASM_F_piece", "ASM_F_offsets"))
    ids, bits = np.asarray(z["ASM_F_ids"]), np.asarray(z["ASM_F_bits"])
    where = {(int(r), int(p)): (int(off[k]), int(off[k + 1])) for k, (r, p) in enumerate(zip(rnd, pc))}
    seed = int(np.asarray(z["SIM_SEED"]).item()) if "SIM_SEED" in z.files else -1
    out = []
    for r in range(n_rounds // 2, n_rounds):
        if any((r, j) not in where for j in range(M)):
            continue
        window = {}
        for j in range(M - WINDOW, M):
            a, b = where[(r, j)]
            cells = np.unpackbits(bits[a:b], axis=1, count=G * G).sum(1)
            for sid, n in zip(ids[a:b].tolist(), cells.tolist()):
                if n > 0:
                    window.setdefault(sid, 0)
                    if j == M - 1:
                        window[sid] = n
        out.append((r, window))
    return dict(seed=seed, M=M, n_rounds=n_rounds, rounds=out)


def design(rounds):
    if len(rounds) <= N_DESIGN:
        return set(range(len(rounds)))
    return set(np.round(np.linspace(0, len(rounds) - 1, N_DESIGN)).astype(int).tolist())


def count(paths, label, out=print):
    labs = [c[0] for c in CLASSES]
    stacks = {c: 0 for c in labs}; dstacks = {c: 0 for c in labs}
    comm = {c: set() for c in labs}; dcomm = {c: set() for c in labs}; distinct = {c: set() for c in labs}
    per = {c: [] for c in labs}; sizes, seeds, n_cand, Ms = [], [], [], set()
    for p in paths:
        w = community(p)
        seeds.append(w["seed"]); n_cand.append(len(w["rounds"])); Ms.add(w["M"])
        mine = {c: 0 for c in labs}
        pick = design(w["rounds"])
        for i, (r, window) in enumerate(w["rounds"]):
            sizes.append(len(window))
            for sid, n in window.items():
                c = cls(n)
                stacks[c] += 1; mine[c] += 1; comm[c].add(p); distinct[c].add((p, sid))
                if i in pick:
                    dstacks[c] += 1; dcomm[c].add(p)
        for c in labs:
            per[c].append(mine[c])
    n = len(paths)
    out(f"== {label}: {n} communities (seeds {min(seeds)}-{max(seeds)}), pieces per round {sorted(Ms)}, candidate rounds per "
        f"community min {min(n_cand)}, max {max(n_cand)}; species per stack median {int(np.median(sizes))}, "
        f"min {min(sizes)}, max {max(sizes)}")
    out(f"{'class':>6} {'stacks':>9} {'share':>6} {'comms':>6} {'distinct':>9} {'median/comm':>12} {'comms 0':>8} "
        f"{'stacks 10-round':>16} {'comms':>6}")
    tot = sum(stacks.values()) or 1
    for c in labs:
        out(f"{c:>6} {stacks[c]:>9} {stacks[c] / tot:>6.3f} {len(comm[c]):>6} {len(distinct[c]):>9} "
            f"{int(np.median(per[c])):>12} {sum(1 for v in per[c] if v == 0):>8} {dstacks[c]:>16} {len(dcomm[c]):>6}")
    return dict(stacks=stacks, dstacks=dstacks, comm={c: len(v) for c, v in comm.items()},
                distinct={c: len(v) for c, v in distinct.items()}, per=per, n=n)


def files(pattern, name):
    fs = sorted(glob.glob(pattern))
    if any("real_eval" in f for f in fs):
        sys.exit(f"STOP: {name} pattern reaches real_eval, which is read once after training")
    if not fs:
        sys.exit(f"STOP: no file matches {pattern}")
    return fs


# ---------------------------------------------------------------- self-test
def planted(path, sizes, M=25, rounds=12, seed=1, gone=None, drop=None, early=None):
    """Species s has sizes[s] cells at every piece; gone: species absent at the last piece only;
    drop: (round, piece) left out; early: (species, cells) at pieces before the window, to test the window."""
    rng = np.random.default_rng(seed)
    Fr, Fp, Fo, Fi, Fb = [], [], [0], [], []
    for r in range(rounds):
        for j in range(M):
            if drop == (r, j):
                continue
            ids, B = [], []
            for s, k in enumerate(sizes):
                kk = k
                if early and s == early[0] and j < M - WINDOW:
                    kk = early[1]
                if gone is not None and s == gone and j == M - 1:
                    kk = 0
                if kk == 0:
                    continue
                m = np.zeros(G * G, bool); m[rng.choice(G * G, kk, replace=False)] = True
                ids.append(s); B.append(m)
            Fr.append(r); Fp.append(j); Fi += ids; Fo.append(Fo[-1] + len(ids))
            Fb.append(np.packbits(np.array(B), axis=1) if B else np.zeros((0, 50), np.uint8))
    np.savez(path, ASM_F_round=np.array(Fr), ASM_F_piece=np.array(Fp), ASM_F_offsets=np.array(Fo), ASM_F_ids=np.array(Fi),
             ASM_F_bits=np.concatenate(Fb), ENV_habitat_t=np.zeros((rounds, G, G), np.float32),
             HABITAT_substeps=np.int64(M), SIM_SEED=np.int64(seed))


def self_test():
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(f"{tmp}/real_train"); os.makedirs(f"{tmp}/real_eval")
        a, b = f"{tmp}/real_train/a_training.npz", f"{tmp}/real_train/b_training.npz"
        planted(a, [1, 3, 8, 15, 30, 60], seed=1)
        planted(b, [2, 4, 70], seed=2, gone=1, drop=(9, 4), early=(0, 40))
        planted(f"{tmp}/real_eval/c_training.npz", [1], seed=3)
        lines = []
        r = count([a, b], "planted", out=lines.append)
        s = r["stacks"]
        t("T1", s == {"0": 5, "1": 6, "2-5": 11, "6-10": 6, "11-20": 6, "21-50": 6, "51+": 11},
          f"stacks per class from known ranges, last half of 12 rounds = 6 candidates, b loses round 9: {s}")
        t("T2", r["comm"]["2-5"] == 2 and r["comm"]["1"] == 1 and r["distinct"]["2-5"] == 2 and r["comm"]["0"] == 1,
          "communities and distinct species per class (2-5: 11 stacks from 2 species in 2 communities; 0: only b)")
        t("T3", s["0"] == 5 and r["per"]["0"] == [0, 5], "a species gone at the target year but present earlier in the window counts as class 0")
        t("T4", r["stacks"]["21-50"] == 6, "40 cells before the window do not move b's 2-cell species into 21-50: the window is 6 pieces")
        t("T5", r["dstacks"]["1"] == 6 and sum(r["dstacks"].values()) == sum(s.values()),
          "with 6 candidates (fewer than 10) the 10-round design keeps them all")
        planted(f"{tmp}/real_train/d_training.npz", [5], rounds=60, seed=4)
        r2 = count([f"{tmp}/real_train/d_training.npz"], "planted", out=lambda x: None)
        t("T6", r2["stacks"]["2-5"] == 30 and r2["dstacks"]["2-5"] == 10, "30 candidates: all counted, 10 in the design")
        import subprocess
        p = subprocess.run([sys.executable, __file__, "--train", f"{tmp}/real_*/*_training.npz", "--val", f"{tmp}/real_train/a*"],
                           capture_output=True, text=True)
        t("T7", p.returncode != 0 and "real_eval" in p.stderr, "a pattern that reaches real_eval is refused")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.train:
        sys.exit("STOP: give --train (and --val)")
    tr = files(a.train, "--train")
    va = files(a.val, "--val") if a.val else []
    count(tr, "real_train")
    if va:
        count(va, "real_val")