#!/usr/bin/env python3
"""
check_training_file.py  (obj2_v2, stage A step 2, read-only, 7 Nov 2026)

What it does
    Opens one real-run training file (obj1_v2 real run, assembly v2.6 with habitat substeps) and checks the facts the
    Objective 2 export needs, before any export code is written (claude/obj2_data_plan.md, facts a and b):
      C1 the keys the export reads are there
      C2 every round of the last half of the assembly has all M yearly piece maps (M = HABITAT_substeps)
      C3 enough pieces for 10 target years per community, each with the 5 years before it inside the same round
      C4 habitat at a target's own time: the saved round map (ENV_habitat_t) belongs to the last piece of its round,
         checked through the shift (ASM_F_shift of the last piece = ENV_habitat_shift of the round)
      C5 the last piece map of each round equals that round's window-end map (ASM_P), ids and cells
      C6 every recorded species in a piece map has at least one cell, and no id appears twice in one map
      C7 truth keys in the file (true niches, growth fields, dispersal rates, life table): listed, so the export
         leaves every one of them out of the input files
      C8 true range classes at the last piece (1, 2-5, 6-10, 11-20, 21-50, 51+ cells), for the range weights
    Writes nothing except what it prints.

Why
    The export takes target years at the last piece of a round (habitat known exactly) with records from the 5 pieces
    before it. If pieces are missing, or the saved habitat does not belong to the last piece, the design changes
    before any code is written.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/check_training_file.py --self-test | tail -1
    python3 ../tools/check_training_file.py "../../obj1_v2/work/results/stage2a/real_train/*_training.npz" \
        | tee ../logs/check_training_file_7nov.txt
    (the first file in sorted order is checked; give one exact path to check another)

Labels: PASS/FAIL = the fact holds or not; INFO = a number for the plan.
"""
import argparse, glob, os, sys, tempfile
import numpy as np

G = 20
NEED = ("ASM_F_round", "ASM_F_piece", "ASM_F_offsets", "ASM_F_ids", "ASM_F_bits", "ENV_habitat_t", "ENV_habitat_shift",
        "HABITAT_substeps", "ASM_P_round", "ASM_P_offsets", "ASM_P_ids", "ASM_P_bits", "P_last_final", "SIM_SEED")
TRUTH_PREFIX = ("ENV_r_field", "NICHE_", "SP_disp", "LT_", "C_", "A_", "IBM_B", "B_last", "r_", "deg_", "prevalence")
CLASSES = ((1, 1, "1"), (2, 5, "2-5"), (6, 10, "6-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, 400, "51+"))


def maps(z, pre):
    """Per saved map: (round, piece or -1, {species id: boolean 400-cell map})."""
    rnd = np.asarray(z[f"{pre}_round"]); off = np.asarray(z[f"{pre}_offsets"]); ids = np.asarray(z[f"{pre}_ids"])
    bits = np.asarray(z[f"{pre}_bits"]); piece = np.asarray(z["ASM_F_piece"]) if pre == "ASM_F" else np.full(len(rnd), -1)
    out = []
    for k in range(len(rnd)):
        a, b = int(off[k]), int(off[k + 1])
        P = np.unpackbits(bits[a:b], axis=1, count=G * G).astype(bool)
        out.append((int(rnd[k]), int(piece[k]), ids[a:b].tolist(), P))
    return out


def check(path):
    z = np.load(path, allow_pickle=True)
    res = []

    def say(tag, ok, text):
        if ok is None:
            print(f"{tag} {text}: INFO")
        else:
            res.append(bool(ok)); print(f"{tag} {text}: {'PASS' if ok else 'FAIL'}")

    print(f"file {os.path.basename(path)}; seed {int(np.asarray(z['SIM_SEED']).item()) if 'SIM_SEED' in z.files else '?'}")
    miss = [k for k in NEED if k not in z.files]
    say("C1", not miss, "keys the export reads are present" + (f" (missing {miss})" if miss else ""))
    if miss:
        print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL"); return False
    M = int(np.asarray(z["HABITAT_substeps"]).item())
    F = maps(z, "ASM_F")
    rounds = sorted({r for r, _, _, _ in F})
    n_rounds = len(np.asarray(z["ENV_habitat_t"]))
    last_half = [r for r in range(n_rounds // 2, n_rounds)]
    by_round = {}
    for r, j, _, _ in F:
        by_round.setdefault(r, []).append(j)
    full = [r for r in last_half if sorted(by_round.get(r, [])) == list(range(M))]
    say("C2", len(full) == len(last_half),
        f"M = {M} pieces per round; rounds {n_rounds} (habitat maps), rounds with piece maps {len(rounds)} "
        f"({rounds[0]}-{rounds[-1]}); last half {len(last_half)} rounds, {len(full)} with all {M} pieces")
    say("C3", M >= 6 and len(full) >= 10,
        f"targets at the last piece of a round with the 5 pieces before it in the same round: {len(full)} rounds "
        f"available in the last half (10 needed), {M - 1} earlier pieces per round (5 needed)")
    sh_r = np.asarray(z["ENV_habitat_shift"], float)
    if "ASM_F_shift" in z.files:
        sh_f = np.asarray(z["ASM_F_shift"], float); fr = np.asarray(z["ASM_F_round"]); fp = np.asarray(z["ASM_F_piece"])
        last = [(int(r), float(sh_f[k])) for k, (r, j) in enumerate(zip(fr, fp)) if j == M - 1 and r < len(sh_r)]
        dev = max(abs(s - sh_r[r]) for r, s in last) if last else np.inf
        say("C4", dev < 1e-9, f"the round's saved habitat belongs to its last piece: shift of the last piece = round shift "
            f"in {len(last)} rounds (largest difference {dev:.2e})")
    else:
        say("C4", None, "ASM_F_shift absent: the shift check cannot run; habitat of the last piece taken from ENV_habitat_t "
            "by the v2.6 rule (test_20 U3)")
    P = {r: (ids, B) for r, _, ids, B in maps(z, "ASM_P")}
    eq = [r for r, j, ids, B in F if j == M - 1 and r in P and ids == P[r][0] and np.array_equal(B, P[r][1])]
    n_last = sum(1 for r, j, _, _ in F if j == M - 1 and r in P)
    say("C5", n_last > 0 and len(eq) == n_last, f"last piece map = window-end map (ids and cells) in {len(eq)} of {n_last} rounds")
    empty = sum(int((B.sum(1) == 0).sum()) for _, _, _, B in F)
    dup = sum(len(ids) - len(set(ids)) for _, _, ids, _ in F)
    say("C6", empty == 0 and dup == 0, f"recorded species without a cell {empty}; ids repeated within a map {dup}")
    truth = sorted(k for k in z.files if k.startswith(TRUTH_PREFIX))
    say("C7", None, f"truth keys to keep out of every input file ({len(truth)}): {', '.join(truth)}")
    r_last, j_last, _, B_last = max(F, key=lambda t: (t[0], t[1]))
    n = B_last.sum(1)
    counts = [int(((n >= lo) & (n <= hi)).sum()) for lo, hi, _ in CLASSES]
    say("C8", None, f"true range classes at round {r_last} piece {j_last} ({len(n)} species): "
        + ", ".join(f"{lab} {c}" for (_, _, lab), c in zip(CLASSES, counts)))
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


# ---------------------------------------------------------------- self-test
def planted(path, M=25, rounds=24, S=6, seed=5001, drop_piece=False, bad_shift=False, last_differs=False, empty_sp=False):
    rng = np.random.default_rng(seed)
    shift_r = rng.normal(size=rounds)
    Fr, Fp, Fo, Fi, Fb, Fs, Pr, Po, Pi, Pb = [], [], [0], [], [], [], [], [0], [], []
    for r in range(rounds):
        for j in range(M):
            if drop_piece and r == rounds - 1 and j == 3:
                continue
            ids = list(range(r, r + S))
            B = rng.random((S, G * G)) < 0.1
            B[:, 0] = True
            if empty_sp and r == rounds - 2 and j == 0:
                B[1] = False
            Fr.append(r); Fp.append(j); Fi += ids; Fb.append(np.packbits(B, axis=1)); Fo.append(Fo[-1] + S)
            Fs.append(shift_r[r] + (0.5 if bad_shift and j == M - 1 else 0.0) if j == M - 1 else rng.normal())
            if j == M - 1:
                Bp = B.copy()
                if last_differs and r == 5:
                    Bp[0, 1] = ~Bp[0, 1]
                Pr.append(r); Pi += ids; Pb.append(np.packbits(Bp, axis=1)); Po.append(Po[-1] + S)
    np.savez(path, ASM_F_round=np.array(Fr), ASM_F_piece=np.array(Fp), ASM_F_offsets=np.array(Fo), ASM_F_ids=np.array(Fi),
             ASM_F_bits=np.concatenate(Fb), ASM_F_shift=np.array(Fs), ENV_habitat_t=rng.normal(size=(rounds, G, G)).astype(np.float32),
             ENV_habitat_shift=shift_r, HABITAT_substeps=np.int64(M), ASM_P_round=np.array(Pr), ASM_P_offsets=np.array(Po),
             ASM_P_ids=np.array(Pi), ASM_P_bits=np.concatenate(Pb), P_last_final=np.zeros((S, G, G), np.uint8),
             SIM_SEED=np.int64(seed), ENV_r_field=np.zeros((S, G, G)), NICHE_opt=np.zeros(S), SP_disp_rate=np.zeros(S))


def self_test():
    res = []

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")
    import contextlib, io
    with tempfile.TemporaryDirectory() as tmp:
        def run(**kw):
            p = f"{tmp}/w_{len(os.listdir(tmp))}_training.npz"; planted(p, **kw)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ok = check(p)
            return ok, buf.getvalue()
        ok, out = run()
        t("T1", ok and "C8" in out and "ENV_r_field, NICHE_opt, SP_disp_rate" in out, "a complete planted file passes and lists its truth keys")
        ok, out = run(drop_piece=True)
        t("T2", not ok and "FAIL" in [l for l in out.splitlines() if l.startswith("C2")][0],
          "broken version caught: one missing piece in the last round fails C2")
        ok, out = run(bad_shift=True)
        t("T3", not ok and "FAIL" in [l for l in out.splitlines() if l.startswith("C4")][0], "broken version caught: last-piece shift off the round shift fails C4")
        ok, out = run(last_differs=True)
        t("T4", not ok and "FAIL" in [l for l in out.splitlines() if l.startswith("C5")][0], "broken version caught: one cell differs from the window-end map fails C5")
        ok, out = run(empty_sp=True)
        t("T5", not ok and "FAIL" in [l for l in out.splitlines() if l.startswith("C6")][0], "broken version caught: a recorded species with no cell fails C6")
        ok, out = run(M=5)
        t("T6", not ok and "FAIL" in [l for l in out.splitlines() if l.startswith("C3")][0], "too few pieces per round (5) fails C3")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.pattern:
        sys.exit("STOP: give a training file or a pattern")
    files = sorted(glob.glob(a.pattern))
    if not files:
        sys.exit(f"STOP: no file matches {a.pattern}")
    sys.exit(0 if check(files[0]) else 1)