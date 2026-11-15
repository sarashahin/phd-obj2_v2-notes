#!/usr/bin/env python3
"""
obj1_figure_data.py  (obj1_v2, Objective 1 figures, data export; 7 Nov 2026, Sara's yes to figures by route (a))

Why: the Objective 1 figures (PDF and PNG) are drawn from the 150 real-run worlds. The training files are about 11 MB
each and stay on Sara's machine; this tool copies only what the figures need into one small file, which Sara attaches.
It reads the real-run training files only. It never opens a BNM file: the BNM numbers for the figures come from the
world_vs_bnm and frame_change_check logs, which print the BNM fit half only.

Per world (every file given): split (from the folder: real_train, real_val, real_eval), seed, richness per round
(ASM_gamma), kept per round (from ASM_established), removed per round (ASM_pruned), the regional shift per round
(ENV_habitat_shift), richness per final-run frame (gamma_t, IBM_t), final range sizes (cells per species in P_last_final),
the final species' niche traits (NICHE_opt, NICHE_w, NICHE_peak) and dispersal rates (SP_disp_rate), and the life table
(LT_round_in, LT_round_out, LT_established, LT_entered, LT_peak_cells).
Example worlds (--examples, default 5001 5002 5003, training only): start and final habitat maps, the habitat map every
5th round, the final presence and growth fields of every species, and the last 100 yearly piece maps (packed bits as saved).

Guard: every file must be a real-run world (seed 5001-5120 or 6001-6030, in the folder of its split), or the tool stops;
a seed seen twice stops it.

Run from obj1_v2/work:
    python3 ../tools/obj1_figure_data.py --self-test | tail -1
    python3 ../tools/obj1_figure_data.py --out ../logs/obj1_figure_data.npz "results/stage2a/real_train/*_training.npz" "results/stage2a/real_val/*_training.npz" "results/stage2a/real_eval/*_training.npz"
"""
import argparse, glob, os, sys, tempfile
from pathlib import Path
import numpy as np

SPLITS = {"real_train": (5001, 5100), "real_val": (5101, 5120), "real_eval": (6001, 6030)}
PER_WORLD = ("ASM_gamma", "ASM_established", "ASM_pruned", "ASM_round", "ENV_habitat_shift", "gamma_t", "IBM_t",
             "P_last_final", "NICHE_opt", "NICHE_w", "NICHE_peak", "SP_disp_rate", "LT_round_in", "LT_round_out",
             "LT_established", "LT_entered", "LT_peak_cells", "SIM_SEED")
EXAMPLE = ("ENV_habitat_start", "ENV_habitat", "ENV_habitat_t", "ENV_r_field", "ASM_F_round", "ASM_F_piece",
           "ASM_F_offsets", "ASM_F_ids", "ASM_F_bits", "species_id")
LAST_FRAMES = 100


def split_of(path):
    parts = Path(path).resolve().parts
    hits = [p for p in parts if p in SPLITS]
    return hits[-1] if hits else None


def world(z):
    seed = int(np.asarray(z["SIM_SEED"]).item())
    est = np.asarray(z["ASM_established"]).astype(np.int64)
    P = np.asarray(z["P_last_final"]).astype(bool)
    return seed, dict(
        gamma=np.asarray(z["ASM_gamma"]).astype(np.int16), kept=np.diff(np.r_[0, est]).astype(np.int16),
        removed=np.asarray(z["ASM_pruned"]).astype(np.int16), shift=np.asarray(z["ENV_habitat_shift"]).astype(np.float32),
        final_gamma=np.asarray(z["gamma_t"]).astype(np.int16), final_t=np.asarray(z["IBM_t"]).astype(np.int32),
        range_cells=P.reshape(len(P), -1).sum(1).astype(np.int16),
        opt=np.asarray(z["NICHE_opt"]).astype(np.float32), w=np.asarray(z["NICHE_w"]).astype(np.float32),
        peak=np.asarray(z["NICHE_peak"]).astype(np.float32), d=np.asarray(z["SP_disp_rate"]).astype(np.float32),
        lt_in=np.asarray(z["LT_round_in"]).astype(np.int32), lt_out=np.asarray(z["LT_round_out"]).astype(np.int32),
        lt_est=np.asarray(z["LT_established"]).astype(np.uint8), lt_entered=np.asarray(z["LT_entered"]).astype(np.uint8),
        lt_peak=np.asarray(z["LT_peak_cells"]).astype(np.int16))


def example(z):
    fr = np.asarray(z["ASM_F_round"]); off = np.asarray(z["ASM_F_offsets"]); n = len(fr)
    k0 = max(0, n - LAST_FRAMES); a, b = int(off[k0]), int(off[n])
    H_t = np.asarray(z["ENV_habitat_t"]).astype(np.float32)
    return dict(hab_start=np.asarray(z["ENV_habitat_start"]).astype(np.float32), hab_end=np.asarray(z["ENV_habitat"]).astype(np.float32),
                hab_every5=H_t[::5], hab_every5_round=np.arange(len(H_t))[::5].astype(np.int32),
                presence=np.asarray(z["P_last_final"]).astype(np.uint8), r_field=np.asarray(z["ENV_r_field"]).astype(np.float32),
                species_id=np.asarray(z["species_id"]).astype(np.int64),
                F_round=fr[k0:].astype(np.int32), F_piece=np.asarray(z["ASM_F_piece"])[k0:].astype(np.int16),
                F_offsets=(off[k0:n + 1] - a).astype(np.int64), F_ids=np.asarray(z["ASM_F_ids"])[a:b].astype(np.int64),
                F_bits=np.asarray(z["ASM_F_bits"])[a:b])


def export(files, out, examples):
    data, seen, n_ex = {}, {}, 0
    for f in files:
        sp = split_of(f)
        if sp is None:
            sys.exit(f"STOP: {f} is not in a real-run folder (real_train, real_val, real_eval)")
        z = np.load(f, allow_pickle=True)
        missing = [k for k in PER_WORLD if k not in z.files]
        if missing:
            sys.exit(f"STOP: {os.path.basename(f)} lacks {missing}")
        seed, w = world(z)
        lo, hi = SPLITS[sp]
        if not lo <= seed <= hi:
            sys.exit(f"STOP: seed {seed} in {sp} is outside {lo}-{hi}: not a real-run world of that split")
        if seed in seen:
            sys.exit(f"STOP: seed {seed} found twice ({seen[seed]} and {f})")
        seen[seed] = f
        for k, v in w.items():
            data[f"w{seed}_{k}"] = v
        data[f"w{seed}_split"] = np.array(sp)
        if seed in examples and sp == "real_train":
            miss = [k for k in EXAMPLE if k not in z.files]
            if miss:
                sys.exit(f"STOP: example world {seed} lacks {miss}")
            for k, v in example(z).items():
                data[f"ex{seed}_{k}"] = v
            n_ex += 1
    data["seeds"] = np.array(sorted(seen), dtype=np.int64)
    np.savez_compressed(out, **data)
    counts = {s: sum(1 for x in seen.values() if split_of(x) == s) for s in SPLITS}
    print(f"written {out} ({os.path.getsize(out) / 1e6:.1f} MB): {len(seen)} worlds ({counts}); {n_ex} example worlds")
    return data


# ---------------------------------------------------------------- self-test
def planted(path, seed, S=4, rounds=12, Y=20):
    rng = np.random.default_rng(seed)
    P = rng.random((S, Y, Y)) < 0.3
    est = np.cumsum(rng.integers(0, 3, rounds))
    nF = 30; ids = np.arange(S); bits = np.packbits(rng.random((nF * S, Y * Y)) < 0.2, axis=1)
    np.savez(path, ASM_gamma=np.arange(rounds) + 1, ASM_established=est, ASM_pruned=rng.integers(0, 2, rounds),
             ASM_round=np.arange(rounds), ENV_habitat_shift=rng.normal(size=rounds), gamma_t=np.full(50, S), IBM_t=np.arange(1, 51) * 200,
             P_last_final=P.astype(np.uint8), NICHE_opt=rng.normal(size=S), NICHE_w=rng.uniform(0.2, 1, S), NICHE_peak=rng.uniform(1, 1.5, S),
             SP_disp_rate=rng.uniform(1e-6, 1e-2, S), LT_round_in=np.arange(S + 2), LT_round_out=np.full(S + 2, -1),
             LT_established=np.ones(S + 2), LT_entered=np.ones(S + 2), LT_peak_cells=np.full(S + 2, 50), SIM_SEED=np.int64(seed),
             ENV_habitat_start=rng.normal(size=(Y, Y)), ENV_habitat=rng.normal(size=(Y, Y)), ENV_habitat_t=rng.normal(size=(rounds, Y, Y)),
             ENV_r_field=rng.normal(size=(S, Y, Y)), ASM_F_round=np.repeat(np.arange(6), 5), ASM_F_piece=np.tile(np.arange(5), 6),
             ASM_F_offsets=np.arange(nF + 1) * S, ASM_F_ids=np.tile(ids, nF), ASM_F_bits=bits, species_id=ids)
    return P, est


def self_test():
    res = []
    def check(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("real_train", "real_val", "real_eval", "niche_p"):
            os.makedirs(f"{tmp}/{d}")
        P1, est1 = planted(f"{tmp}/real_train/a_s5001_training.npz", 5001)
        planted(f"{tmp}/real_val/b_s5101_training.npz", 5101)
        planted(f"{tmp}/real_eval/c_s6001_training.npz", 6001)
        global LAST_FRAMES
        keep = LAST_FRAMES; LAST_FRAMES = 10
        try:
            d = export(sorted(glob.glob(f"{tmp}/real_*/*.npz")), f"{tmp}/o.npz", {5001})
        finally:
            LAST_FRAMES = keep
        z = np.load(f"{tmp}/o.npz")
        check("T1", list(z["seeds"]) == [5001, 5101, 6001] and str(z["w5101_split"]) == "real_val", "three worlds, split read from the folder")
        check("T2", np.array_equal(z["w5001_range_cells"], P1.reshape(4, -1).sum(1)) and np.array_equal(z["w5001_kept"], np.diff(np.r_[0, est1])),
              "range sizes and kept per round copied exactly")
        src = np.load(f"{tmp}/real_train/a_s5001_training.npz")
        ok3 = (len(z["ex5001_F_round"]) == 10 and z["ex5001_F_offsets"][0] == 0
               and np.array_equal(z["ex5001_F_bits"], src["ASM_F_bits"][20 * 4:30 * 4]) and z["ex5001_hab_every5"].shape == (3, 20, 20))
        check("T3", ok3, "example world: last frames' bits and offsets cut correctly, habitat every 5th round")
        check("T4", "ex5101_hab_start" not in z.files and "ex6001_hab_start" not in z.files, "no example maps from validation or evaluation")
        planted(f"{tmp}/niche_p/t_s2303_training.npz", 2303)
        planted(f"{tmp}/real_val/x_s6002_training.npz", 6002)
        def stops(files):
            try:
                export(files, f"{tmp}/bad.npz", set()); return False
            except SystemExit:
                return True
        check("T5", stops([f"{tmp}/niche_p/t_s2303_training.npz"]), "a test world outside the real-run folders is refused")
        check("T6", stops([f"{tmp}/real_val/x_s6002_training.npz"]), "an evaluation seed inside the validation folder is refused")
        os.link(f"{tmp}/real_train/a_s5001_training.npz", f"{tmp}/real_train/a2_s5001_training.npz")
        check("T7", stops([f"{tmp}/real_train/a_s5001_training.npz", f"{tmp}/real_train/a2_s5001_training.npz"]),
              "the same seed twice is refused")
        with open(__file__) as fh:
            code = fh.read().split("def world")[1].split("# ---------------------------------------------------------------- self-test")[0]
        check("T8", "bnm" not in code.lower(), "the export code never reads a BNM file")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patterns", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--examples", nargs="*", type=int, default=[5001, 5002, 5003])
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.out or not a.patterns:
        sys.exit("STOP: give --out FILE and the training-file patterns")
    files = sorted({f for p in a.patterns for f in glob.glob(p)})
    if not files:
        sys.exit("STOP: no file matched")
    export(files, a.out, set(a.examples))