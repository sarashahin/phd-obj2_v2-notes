#!/usr/bin/env python3
"""
frame_bands_export.py  (obj1_v2, gate 7 and the band rule, 1 Nov 2026, Sara's yes)

Why: world_vs_bnm scores one map per world, P_last_final (the end of the 10,000-step final run with no newcomers).
In the shift worlds (niche_h, niche_h2) richness swings with the regional shift (end richness 22-74 species in one
setting; within n69 the block means fall from 34 to 18), so one end map gives a band value that depends on where the
shift is at that moment, and species per window failed on both sides (x0.62 and x2.09). Objective 2 will train on the
window-end maps during invasions (root cause 21, gate 7), not on the final-run map. Those maps are already saved in
every training file since assembly v2 (ASM_P_*: each species' presence at the end of every round, 1 individual =
the observation tau 1), so no new run is needed.

What it does: for each world file, takes F window-end maps evenly spaced over the last half of the assembly rounds
(default 20) and writes, per map, a copy of the training file in which P_last_final is that map. Every other key is
copied unchanged, so world_vs_bnm (unchanged, --tau 1, which reads presence from P_last_final only) gives each copy
the same setting as its world and scores it exactly as it scores a world.
Control copy (--control): one more file per world whose P_last_final is the world's own final map; world_vs_bnm must
print for it exactly the numbers of the end-map run (logs/world_vs_bnm_niche_h2_1nov.txt), which shows the copy path
changes nothing.

Reading rule (written 1 Nov, before any frame is scored): world_vs_bnm's own setting verdict on the frame copies of
the 4 worlds of one setting (80 maps): a band passes when the median over maps is inside and at least 75% of maps are
inside (the 28 Oct rule, applied to maps instead of worlds). The end-map verdict (1 of 8) stays in the record.
Frame copies are scored in their own world_vs_bnm run, never together with the training files (same setting keys).

Run from obj1_v2/work:
    python3 ../tools/frame_bands_export.py --self-test | tail -1
    python3 ../tools/frame_bands_export.py --out results/frames_h2 "results/stage2a/niche_h/*_fb_ht62500_hs1p0t62500_*_training.npz" "results/stage2a/niche_h2/*_training.npz"
Writes only under --out; reads the training files.
"""
import argparse, glob, sys, tempfile
from pathlib import Path
import numpy as np

NEED = ["P_last_final", "ASM_P_round", "ASM_P_offsets", "ASM_P_ids", "ASM_P_bits"]


def frame(z, i, shape):
    """(ids, P) of the window-end map of recorded round index i; P (S, Ny, Nx) uint8."""
    o = np.asarray(z["ASM_P_offsets"]); a, b = int(o[i]), int(o[i + 1])
    n = shape[0] * shape[1]
    P = np.unpackbits(np.asarray(z["ASM_P_bits"])[a:b], axis=1, count=n).reshape(b - a, *shape)
    return np.asarray(z["ASM_P_ids"])[a:b], P.astype(np.uint8)


def pick(n_rounds, n_frames):
    """Indices of n_frames maps evenly spaced over the last half of n_rounds recorded rounds (no repeats)."""
    lo = n_rounds // 2
    return np.unique(np.linspace(lo, n_rounds - 1, n_frames).round().astype(int))


def export(f, out, n_frames, control):
    z = np.load(f, allow_pickle=True)
    miss = [k for k in NEED if k not in z.files]
    if miss:
        return None, f"skipped {Path(f).name}: missing {miss}"
    rounds = np.asarray(z["ASM_P_round"])
    if len(rounds) < 4 or not np.all(np.diff(rounds) == 1):
        return None, f"skipped {Path(f).name}: recorded rounds are not consecutive"
    shape = tuple(np.asarray(z["P_last_final"]).shape[1:])
    keep = {k: z[k] for k in z.files}
    stem = Path(f).name[: -len("_training.npz")] if Path(f).name.endswith("_training.npz") else Path(f).stem
    written, S = [], []
    jobs = [(int(i), f"_fr{int(rounds[i]):04d}") for i in pick(len(rounds), n_frames)]
    for i, tag in jobs:
        ids, P = frame(z, i, shape)
        if P.shape[0] == 0 or not P.reshape(len(P), -1).any(1).all():
            return None, f"skipped {Path(f).name}: round {rounds[i]} has an empty map"
        keep["P_last_final"] = P
        p = Path(out) / f"{stem}{tag}_training.npz"
        np.savez_compressed(p, **keep); written.append(p.name); S.append(len(ids))
    if control:
        keep["P_last_final"] = np.asarray(z["P_last_final"])
        p = Path(out) / "control" / f"{stem}_frend_training.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **keep)
    return dict(name=Path(f).name, rounds=len(rounds), frames=[r for _, r in jobs], S=S), None


# ------------------------------------------------ self-test
def planted(path, rng, n_rounds=40, S0=6, G=20, gap=False, empty=False):
    """A training file with known window-end maps: round t holds species ids t..t+S0-1, species k in cells
    k*(t+1) % (G*G) and the next 3 cells (so every map differs and is known)."""
    rounds, offs, ids, bits = [], [0], [], []
    for t in range(n_rounds):
        r_ = t + (5 if gap and t > n_rounds // 2 else 0)
        sid = np.arange(t, t + S0)
        P = np.zeros((S0, G * G), bool)
        for k in range(S0):
            c = (k * (t + 1) + np.arange(4)) % (G * G)
            P[k, c] = True
        if empty and t == n_rounds - 1:
            P[0] = False
        rounds.append(r_); ids.append(sid); bits.append(np.packbits(P, axis=1)); offs.append(offs[-1] + S0)
    final = rng.random((S0 + 1, G, G)) < 0.2
    np.savez(path, P_last_final=final.astype(np.uint8), B_last=(final * 1e-4).astype(np.float16), BODY_MASS=np.float32(1e-4),
             ASM_P_round=np.array(rounds), ASM_P_offsets=np.array(offs), ASM_P_ids=np.concatenate(ids),
             ASM_P_bits=np.concatenate(bits), HABITAT_shift_sd=np.float64(1.0), SIM_SEED=np.int64(2303),
             NICHE_w=np.full(S0 + 1, 0.3))


def self_test():
    rng = np.random.default_rng(1); res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    check("T1", list(pick(40, 20)) == list(range(20, 40)) and list(pick(462, 20))[0] == 231 and list(pick(462, 20))[-1] == 461
          and len(pick(462, 20)) == 20 and len(pick(10, 20)) == 5,
          "maps evenly spaced over the last half: 40 rounds -> rounds 20-39; 462 -> 231..461, 20 maps; 10 rounds -> 5 maps")
    with tempfile.TemporaryDirectory() as tmp:
        planted(f"{tmp}/w_training.npz", rng)
        out = Path(tmp) / "out"; out.mkdir()
        info, why = export(f"{tmp}/w_training.npz", out, 20, control=True)
        z0 = np.load(f"{tmp}/w_training.npz")
        good = why is None and len(info["frames"]) == 20
        for t in range(20, 40):
            c = np.load(out / f"w_fr{t:04d}_training.npz")
            P = c["P_last_final"].reshape(6, -1).astype(bool)
            exp = np.zeros((6, 400), bool)
            for k in range(6):
                exp[k, (k * (t + 1) + np.arange(4)) % 400] = True
            good &= np.array_equal(P, exp)
        check("T2", good, "each copy holds exactly the planted map of its round (20 maps, rounds 20-39)")
        c = np.load(out / "w_fr0025_training.npz")
        same = all(np.array_equal(c[k], z0[k]) for k in z0.files if k != "P_last_final") and set(c.files) == set(z0.files)
        check("T3", same, "every other key copied unchanged (setting keys, B_last, history), no key added or lost")
        c = np.load(out / "control" / "w_frend_training.npz")
        check("T4", all(np.array_equal(c[k], z0[k]) for k in z0.files) and set(c.files) == set(z0.files),
              "control copy equals the training file key for key")
        planted(f"{tmp}/g_training.npz", rng, gap=True); planted(f"{tmp}/e_training.npz", rng, empty=True)
        np.savez(f"{tmp}/m_training.npz", P_last_final=np.zeros((2, 20, 20), np.uint8))
        r = [export(f"{tmp}/{n}_training.npz", out, 20, False)[1] or "" for n in ("g", "e", "m")]
        check("T5", "not consecutive" in r[0] and "empty map" in r[1] and "missing" in r[2],
              "refused: rounds with a gap, a map with an empty species, a file without window-end maps")
        # mutant: map taken from the wrong round (off by one) must not pass T2's comparison
        ids, P = frame(z0, 25, (20, 20)); ids2, P2 = frame(z0, 26, (20, 20))
        check("T6", not np.array_equal(P, P2) and list(ids) == list(range(25, 31)),
              "mutant check: neighbouring rounds give different maps, ids follow the round")
    print(f"{sum(res)} of {len(res)} PASS" if all(res) else f"FAIL: {len(res) - sum(res)} of {len(res)}")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.out or not a.files:
        sys.exit("STOP: give --out DIR and one or more training-file patterns")
    files = sorted({f for p in a.files for f in glob.glob(p)})
    if not files:
        sys.exit("STOP: no file matched")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if any(Path(a.out).glob("*_training.npz")):
        sys.exit(f"STOP: {a.out} already holds copies; use an empty folder")
    for f in files:
        info, why = export(f, a.out, a.frames, not a.no_control)
        if why:
            print(why); continue
        print(f"{info['name'][-60:]}: {info['rounds']} rounds; {len(info['frames'])} maps at rounds "
              f"{info['frames'][0][3:]}..{info['frames'][-1][3:]}; species per map {min(info['S'])}-{max(info['S'])} "
              f"(median {int(np.median(info['S']))})")
    print(f"\nwritten to {a.out} (control copies in {a.out}/control)")