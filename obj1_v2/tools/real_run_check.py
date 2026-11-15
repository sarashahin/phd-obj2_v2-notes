#!/usr/bin/env python3
"""
real_run_check.py  (obj1_v2, the real IBM run: settings, seeds and leak check; 2 Nov 2026, Sara's yes)

Reads the training files of the three real-run splits (results/stage2a/real_train, real_val, real_eval) and checks,
before any file is used by Objective 2:
  R1 counts: every seed of the split is there once (train 5001-5100, val 5101-5120, eval 6001-6030); none missing, none
     extra, none in the wrong folder.
  R2 setting: every file carries n70's setting exactly (the gate 5 setting): habitat seed 100000 + seed, trait seed
     200000 + seed, w_min 0.2, w_max 1.0, optimum sd 1.5, cost 0.1, peak max 1.5, trade-off 0.5, contrast 0.25, habitat
     length scale 3, habitat tau 62,500, shift sd 1.0 and tau 62,500, 25 sub-steps, strength 0.4, connectance 1, d per
     species 1e-6 to 1e-2, 2Dt a 1 b 3, best founding, founded in 1 cell, detection threshold 1, 5,000-step windows,
     1,000 entered newcomers reached, final run to 10,000, not resumed.
  R3 leaks: no seed, start habitat map, final habitat map or final presence map shared by two worlds, across all splits
     and against the test worlds (--tests, default the stage 2a niche plans).
Exit code 0 only when R1-R3 pass. --smoke checks smoke files (window 300, 120 attempts, final run 1,000) and does not
require every seed of a split (smoke runs only a few named worlds).

Run from obj1_v2/work:
    python3 ../tools/real_run_check.py --self-test | tail -1
    python3 ../tools/real_run_check.py --smoke --train results/stage2a/real_train_smoke --val results/stage2a/real_val_smoke --eval results/stage2a/real_eval_smoke
    python3 ../tools/real_run_check.py | tee ../logs/real_run_check.txt
"""
import argparse, glob, hashlib, sys, tempfile
from pathlib import Path
import numpy as np

SPLITS = {"train": (5001, 5100), "val": (5101, 5120), "eval": (6001, 6030)}
EXPECT = dict(NICHE_w_min=0.2, NICHE_w_max=1.0, NICHE_opt_sd=1.5, NICHE_cost=0.1, NICHE_peak_max=1.5, NICHE_tradeoff=0.5,
              HABITAT_contrast=0.25, HABITAT_ls=3.0, HABITAT_tau=62500, HABITAT_shift_sd=1.0, HABITAT_shift_tau=62500,
              HABITAT_substeps=25, INTERACTION_STRENGTH=0.4, CONNECTANCE=1.0, DISP_d_min=1e-6, DISP_d_max=1e-2,
              DISPERSAL_A=1.0, DISPERSAL_B=3.0, seed_size=1, detection_threshold=1.0, ASM_resumed_from_round=-1)
TEXT = dict(FOUND_rule="best", DISPERSAL_KERNEL="2dt", ENV_mode="niche")


def val(z, k):
    v = np.asarray(z[k])
    return v.item() if v.ndim == 0 else v.ravel()[0]


def h(a):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def check_file(z, split, smoke):
    """List of problems of one file (empty = R2 pass), and its seed."""
    bad = []
    if "SIM_SEED" not in z.files:
        return ["no SIM_SEED"], None
    s = int(val(z, "SIM_SEED"))
    lo, hi = SPLITS[split]
    if not lo <= s <= hi:
        bad.append(f"seed {s} outside {split} ({lo}-{hi})")
    for k, want in (("HABITAT_seed", 100000 + s), ("TRAIT_seed", 200000 + s), ("ENV_seed_field", 200000 + s)):
        if k not in z.files or int(val(z, k)) != want:
            bad.append(f"{k} {val(z, k) if k in z.files else 'missing'} (want {want})")
    for k, want in EXPECT.items():
        if k not in z.files:
            bad.append(f"{k} missing"); continue
        got = float(val(z, k))
        if abs(got - want) > 1e-6 * max(1.0, abs(want)):
            bad.append(f"{k} {got:g} (want {want:g})")
    for k, want in TEXT.items():
        if k not in z.files or str(val(z, k)) != want:
            bad.append(f"{k} {val(z, k) if k in z.files else 'missing'} (want {want})")
    win, att, tmax = (300, 120, 1000) if smoke else (5000, 1000, 10000)
    if "window_steps" not in z.files or int(val(z, "window_steps")) != win:
        bad.append(f"window_steps {val(z, 'window_steps') if 'window_steps' in z.files else 'missing'} (want {win})")
    if "ASM_attempts" not in z.files or int(np.asarray(z["ASM_attempts"])[-1]) < att:
        bad.append(f"entered newcomers {int(np.asarray(z['ASM_attempts'])[-1]) if 'ASM_attempts' in z.files else 'missing'} (want {att})")
    if "IBM_t" not in z.files or abs(float(np.asarray(z["IBM_t"])[-1]) - tmax) > 1e-6:
        bad.append(f"final run to {float(np.asarray(z['IBM_t'])[-1]) if 'IBM_t' in z.files else 'missing'} (want {tmax})")
    return bad, s


def run(dirs, tests, smoke, out=print):
    ok = True; seen = {}; maps = {}
    for split, d in dirs.items():
        files = sorted(glob.glob(str(Path(d) / "*_training.npz")))
        lo, hi = SPLITS[split]
        seeds, n_bad = [], 0
        for f in files:
            z = np.load(f, allow_pickle=True)
            bad, s = check_file(z, split, smoke)
            if bad:
                n_bad += 1; ok = False
                out(f"  R2 FAIL {split} {Path(f).name[-40:]}: " + "; ".join(bad))
            if s is not None:
                seeds.append(s); seen.setdefault(s, []).append(split)
                for key in ("ENV_habitat_start", "ENV_habitat", "P_last_final"):
                    if key in z.files:
                        maps.setdefault((key, h(z[key])), []).append(f"{split}:{s}")
        want = set(range(lo, hi + 1)); have = set(seeds)
        missing = sorted(want - have); extra = sorted(have - want)
        dup = sorted({s for s in seeds if seeds.count(s) > 1})
        r1 = (smoke or not missing) and not extra and not dup     # smoke: a few worlds per split, completeness not asked
        ok &= r1
        out(f"{split} ({d}): {len(files)} files; R1 {'PASS' if r1 else 'FAIL'}"
            + (f"; missing {len(missing)} ({missing[:5]}{'...' if len(missing) > 5 else ''})" + (" (smoke: not counted)" if smoke else "")
               if missing else "")
            + (f"; extra {extra[:5]}" if extra else "") + (f"; twice {dup}" if dup else "")
            + f"; R2 {'PASS' if n_bad == 0 else f'FAIL in {n_bad}'}")
    shared_seed = {s: v for s, v in seen.items() if len(v) > 1}
    shared_map = {k: v for k, v in maps.items() if len(v) > 1}
    test_hits = []
    tfiles = sorted({f for p in tests for f in glob.glob(p)})
    for f in tfiles:
        z = np.load(f, allow_pickle=True)
        for key in ("ENV_habitat_start", "ENV_habitat", "P_last_final"):
            if key in z.files and (key, h(z[key])) in maps:
                test_hits.append((Path(f).name[-40:], key, maps[(key, h(z[key]))]))
    r3 = not shared_seed and not shared_map and not test_hits
    ok &= r3
    out(f"R3 leaks: {'PASS' if r3 else 'FAIL'}: seeds shared {len(shared_seed)}, maps shared {len(shared_map)}"
        f" (start habitat, final habitat, final presence), maps equal to a test world {len(test_hits)} of {len(tfiles)} test files read")
    for k, v in list(shared_map.items())[:5]:
        out(f"   shared {k[0]}: {v}")
    for t in test_hits[:5]:
        out(f"   equal to test world {t[0]} ({t[1]}): {t[2]}")
    out(("ALL PASS (smoke files): the plan is ready for the real run" if smoke else "ALL PASS: the three splits can go to Objective 2")
        if ok else "NOT PASSED: fix or report before any use")
    return ok


# ------------------------------------------------------------------ self-test
def planted(path, s, rng, **over):
    keys = dict(SIM_SEED=np.int64(s), HABITAT_seed=np.int64(100000 + s), TRAIT_seed=np.int64(200000 + s),
                ENV_seed_field=np.int64(200000 + s), window_steps=np.int64(5000), ASM_attempts=np.array([500, 1000]),
                IBM_t=np.array([0.0, 10000.0]), FOUND_rule=np.array(["best"]), DISPERSAL_KERNEL=np.array(["2dt"]),
                ENV_mode=np.array(["niche"]), ENV_habitat_start=rng.normal(size=(20, 20)).astype(np.float32),
                ENV_habitat=rng.normal(size=(20, 20)).astype(np.float32),
                P_last_final=(rng.random((5, 20, 20)) < 0.3).astype(np.uint8))
    keys.update({k: np.float64(v) for k, v in EXPECT.items()})
    keys.update(over)
    np.savez(path, **keys)


def self_test():
    rng = np.random.default_rng(11); res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    def build(tmp, mutate=None):
        dirs = {}
        for split, (lo, hi) in SPLITS.items():
            d = Path(tmp) / split; d.mkdir(parents=True, exist_ok=True); dirs[split] = str(d)
            for s in range(lo, hi + 1):
                over = mutate(split, s) if mutate else {}
                if over is None:
                    continue
                planted(d / f"w_s{s}_training.npz", s, rng, **over)
        return dirs

    quiet = lambda *a, **k: None
    with tempfile.TemporaryDirectory() as tmp:
        dirs = build(tmp + "/a")
        check("T1", run(dirs, [], False, out=quiet), "150 planted worlds with the setting and seeds: all pass")
        lines = []
        dirs = build(tmp + "/b", lambda sp, s: {"NICHE_w_min": np.float64(0.1)} if s == 5007 else {})
        okb = run(dirs, [], False, out=lines.append)
        check("T2", not okb and any("NICHE_w_min 0.1" in l for l in lines), "one world at width 0.1 is caught (R2)")
        dirs = build(tmp + "/c", lambda sp, s: None if s == 6012 else {})
        lines = []; okc = run(dirs, [], False, out=lines.append)
        check("T3", not okc and any("missing 1 ([6012])" in l for l in lines), "a missing evaluation world is caught (R1)")
        H = rng.normal(size=(20, 20)).astype(np.float32)
        dirs = build(tmp + "/d", lambda sp, s: {"ENV_habitat_start": H} if s in (5050, 6003) else {})
        lines = []; okd = run(dirs, [], False, out=lines.append)
        check("T4", not okd and any("shared ENV_habitat_start" in l and "train:5050" in l and "eval:6003" in l for l in lines),
              "a habitat map shared by a training and an evaluation world is caught (R3)")
        dirs = build(tmp + "/e")
        Path(dirs["val"], "w_s5003_training.npz").write_bytes(Path(dirs["train"], "w_s5003_training.npz").read_bytes())
        lines = []; oke = run(dirs, [], False, out=lines.append)
        check("T5", not oke and any("outside val" in l for l in lines) and any("seeds shared 1" in l for l in lines),
              "a training world copied into the validation folder is caught (R1, R2, R3)")
        dirs = build(tmp + "/f", lambda sp, s: {"ASM_attempts": np.array([400, 800])} if s == 5100 else
                     ({"ASM_resumed_from_round": np.float64(120)} if s == 5110 else {}))
        lines = []; okf = run(dirs, [], False, out=lines.append)
        check("T6", not okf and any("entered newcomers 800" in l for l in lines) and any("ASM_resumed_from_round 120" in l for l in lines),
              "a world short of 1,000 newcomers and a resumed world are caught (R2)")
        dirs = build(tmp + "/g")
        t = Path(tmp) / "tests"; t.mkdir()
        z = np.load(Path(dirs["eval"], "w_s6020_training.npz"))
        np.savez(t / "test_training.npz", ENV_habitat_start=z["ENV_habitat_start"])
        lines = []; okg = run(dirs, [str(t / "*_training.npz")], False, out=lines.append)
        check("T7", not okg and any("equal to test world" in l for l in lines), "an evaluation world equal to a test world is caught (R3)")
        dirs = build(tmp + "/h", lambda sp, s: {"HABITAT_seed": np.int64(100000 + s + 1)} if s == 5101 else {})
        lines = []; okh = run(dirs, [], False, out=lines.append)
        check("T8", not okh and any("HABITAT_seed" in l for l in lines), "a habitat seed not equal to 100000 + seed is caught (R2)")
    print(f"\n{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="results/stage2a/real_train")
    ap.add_argument("--val", default="results/stage2a/real_val")
    ap.add_argument("--eval", default="results/stage2a/real_eval")
    ap.add_argument("--tests", nargs="*", default=["results/stage2a/niche*/*_training.npz"])
    ap.add_argument("--smoke", action="store_true", help="smoke files: 300-step windows, 120 attempts, final run 1,000; "
                    "completeness of each split is reported, not required")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    sys.exit(0 if run({"train": a.train, "val": a.val, "eval": a.eval}, a.tests, a.smoke) else 1)