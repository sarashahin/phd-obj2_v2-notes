"""
test_03_assembly_mini.py  (obj1_v2, written 23-24 Oct 2026)

Question: does the infinite-pool assembly (assembly_stepwise_ibm.py) keep its books
right, give every community and every window its own random numbers, and let each
species keep its own environment while other species come and go?

It runs the real assembly code on a small setting (300-step windows, 120 invasion
attempts, tau 1, l 5, variance 0.002) five times, with a "spy" that records what each
window hands to IBMModel. Run A checks every round; a repeat checks reproducibility;
run B changes only the field seed; run B3 changes the assembly seed; run T tags species
ids and removes one early species on purpose.

PASS/FAIL here are requirements, ecological ones included:
  A1 per-round bookkeeping          A2 saved history = rounds seen
  A3 no invaders dropped at the end A5 every window gets its own demographic seed
  B1 same settings, same result     B3 another assembly seed gives unrelated founders
  C1 a species keeps its own field when another species is removed
  C2 fields are returned and depend only on the species id
B2 and A6 are for reading: B2 shows the assembly seed alone fixes interactions and
founders (so each community needs its own seed); A6 is the own-field imprint of ranges.
Results: 24 Oct on Sara's original code: A3, A5 and C1 FLAG (now FAIL), C2 absent.
Run from obj1_v2/work (new code) or obj1_v2/original (old code):
    USE_GPU=0 python3 ../tests/test_03_assembly_mini.py
About 2 minutes on a CPU.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import hashlib, inspect, sys, time
import numpy as np
import config, dispersal, environment, assembly_utils
import assembly_stepwise_ibm as asm
host = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
res = lambda c: "PASS" if c else "FAIL"
flag = lambda c: "FLAG" if c else "ok"
A_C, FRAC, THR, LS, VR, MAXA = config.INTERACTION_STRENGTH, 0.05, 1.0, 5.0, 0.002, 120
print("assembly_stepwise_ibm.py md5", hashlib.md5(open(asm.__file__, "rb").read()).hexdigest())
print("signature", inspect.signature(asm.stepwise_assembly_ibm))

IBM0, ERC0 = asm.IBMModel, asm.expand_RC
calls, tagging = [], {"on": False, "next": 0}
class Spy(IBM0):                                       # records what each assembly window is given
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        calls.append(dict(seed=kw.get("seed", "not passed (default 0)"), r=np.array(host(a[0]), float),
                          field=host(self.r_field).copy()))
    def run(self):
        out = super().run()
        if tagging["on"] and len(calls) == 11:         # run T only: species at position 1 dies in window 11
            self.N[1] = 0
        return out
def tag_expand(r, C, r_new, row, col):                 # identity tag for run T only: r_new = 1 + id * 1e-9
    if tagging["on"]:
        tagging["next"] += 1; r_new = r_new + tagging["next"] * 1e-9
    return ERC0(r, C, r_new, row, col)
asm.IBMModel, asm.expand_RC = Spy, tag_expand

def run(seed_field, tag=False, aseed=1):
    calls.clear(); states = []; tagging.update(on=tag, next=0)
    keep = lambda s: states.append({k: (host(v).copy() if hasattr(v, "shape") else v) for k, v in s.items()})
    t0 = time.time()
    r, C, N, extra = asm.stepwise_assembly_ibm(
        window_steps=300, record_step=100, seed_size=5, F_sat=None, max_rounds=None, max_attempts=MAXA,
        richness_cap=None, frac_multi=FRAC, seed=aseed, detection_threshold=THR, checkpoint_fn=keep,
        length_scale=LS, var_r=VR, seed_field=seed_field)
    return host(r), host(C), host(N), extra, states, list(calls), time.time() - t0

r, C, N, extra, st, cl, sec = run(11)
if not cl: sys.exit("STOP: the spy did not reach IBMModel inside assembly_stepwise_ibm.py; send me this output")
print(f"run A (field seed 11): {len(st)} rounds, final richness {len(r)}, {sec:.0f} s")
off = ~np.eye(len(r), dtype=bool)
inv = []
for k, s in enumerate(st):
    S = len(s["r"]); g_prev = 1 if k == 0 else st[k - 1]["gamma"]; a_prev = 0 if k == 0 else st[k - 1]["attempts"]
    inv.append([s["C"].shape == (S, S) and s["N"].shape[0] == S == s["gamma"],
                np.all(np.diag(s["C"]) == 1.0),
                set(np.unique(s["C"][~np.eye(S, dtype=bool)]).tolist()) <= {0.0, A_C},
                bool((s["N"] >= s["thr"]).any(axis=(1, 2)).all()),
                np.all(s["r"] == 1.0),
                s["attempts"] - a_prev == min(1 + int(FRAC * g_prev), MAXA - a_prev)])         # the last round may be capped
inv = np.array(inv)
names = ["shapes agree", "C diagonal 1", "C entries 0 or 0.4", "every kept species present", "r = 1", "attempts step = 1 + floor(0.05 richness), last round capped"]
print("A1 every round: " + "; ".join(f"{n} {res(inv[:, i].all())}" for i, n in enumerate(names)))
print(f"A2 history saved = rounds seen: richness {res(np.array_equal(extra['ASM_gamma'], [s['gamma'] for s in st]))}, "
      f"attempts {res(np.array_equal(extra['ASM_attempts'], [s['attempts'] for s in st]))}; returned state = last round "
      f"{res(np.array_equal(C, st[-1]['C']) and np.array_equal(N, st[-1]['N']))}")
lost = int(extra["attempts_total"]) - int(extra["ASM_attempts"][-1])
print(f"A3 attempts_total {int(extra['attempts_total'])}, last recorded {int(extra['ASM_attempts'][-1])}: "
      f"{lost} invaders simulated in the last window and then dropped {res(lost == 0)}")
print(f"A4 final connectance {(C[off] > 0).mean():.3f} (0.4 expected; selection by survival can move it)")
seeds = sorted({str(c["seed"]) for c in cl})
print(f"A5 IBMModel got {len(seeds)} different seeds in {len(cl)} windows {res(len(seeds) == len(cl))} "
      f"(one value = the same demographic random numbers in every window; first: {seeds[:3]})")

r2, C2, N2, *_ = run(11)
print(f"B1 same settings again: identical richness, interactions and counts {res(np.array_equal(C, C2) and np.array_equal(N, N2))}")
rb, Cb, Nb, eb, stb, *_ = run(22)
m = min(len(r), len(rb), 20)
same_c = (C[:m, :m] == Cb[:m, :m]).mean()
f0a, f0b = np.argwhere(st[0]["N"][0] > 0).tolist(), np.argwhere(stb[0]["N"][0] > 0).tolist()
print(f"B2 field seed 22, same assembly seed: interactions among the first {m} species identical for {same_c:.0%} of pairs; "
      f"founder cell {f0a} vs {f0b} (100% = fully shared, about 52% = chance; the founder cell is set by the assembly seed alone)")
r3, C3, N3, e3, st3, *_ = run(11, aseed=2)
m3 = min(len(r), len(r3), 20); same3 = (C[:m3, :m3] == C3[:m3, :m3]).mean()
f03 = np.argwhere(st3[0]["N"][0] > 0).tolist()
print(f"B3 assembly seed 2: interactions among the first {m3} species identical for {same3:.0%} of pairs (52% by chance), "
      f"founder cell {f03} vs {f0a} {res(same3 < 0.7 and f03 != f0a)}")

est = np.diff(np.r_[0, extra.get("ASM_established", np.zeros(len(st)))])
pr = [st[k - 1]["gamma"] + int(est[k]) - st[k]["gamma"] for k in range(1, len(st))]
print(f"C0 natural pruning in run A: {sum(pr)} species removed over {len(st)} rounds (the full assemblies remove about 88 per round late on)")
r_t, C_t, N_t, ex_t, st_t, cl_t, _ = run(11, tag=True)
idf = lambda rv: np.rint((rv - 1.0) / 1e-9).astype(int)         # 0 = founder
def kept_share(w0, w1):
    p0 = {i: j for j, i in enumerate(idf(w0["r"]))}; out = []
    for j, i in enumerate(idf(w1["r"])):
        if i in p0 and j >= 1:
            out.append(np.allclose(w0["field"][p0[i]] - w0["r"][p0[i]], w1["field"][j] - w1["r"][j], atol=1e-12))
    return np.mean(out), len(out)
if len(set(idf(r_t).tolist())) != len(r_t) or len(cl_t) < 12:
    sys.exit("C1 not tested: the identity tag did not reach expand_RC, or fewer than 12 windows ran")
k_before, n_before = kept_share(cl_t[9], cl_t[10])
k_after, n_after = kept_share(cl_t[10], cl_t[11])
print(f"C1 run T, one early species (position 1) removed in window 11: later species keep their field {k_before:.0%} of {n_before} "
      f"from window 10 to 11, and {k_after:.0%} of {n_after} from window 11 to 12 {res(k_after == 1.0)}")
if "r_field" in extra and "species_id" in extra:
    rebuilt = np.stack([asm.species_field(int(i), length_scale=LS, var_r=VR, seed_field=11) for i in extra["species_id"]])
    print(f"C2 fields returned with species ids, each equal to the field rebuilt from its id {res(np.abs(rebuilt - extra['r_field']).max() < 1e-12)}")
    Ffin = extra["r_field"]
else:
    print("C2 fields returned with species ids FAIL (this assembly does not return them; the final run draws new fields)")
    Ffin = environment.generate_spatial_r(len(r), config.NUM_PATCHES_Y, config.NUM_PATCHES_X, LS, np.ones(len(r)), VR, seed=11)
z = (Ffin.reshape(len(r), -1) - Ffin.reshape(len(r), -1).mean(1, keepdims=True)) / Ffin.reshape(len(r), -1).std(1, keepdims=True)
occ = (N >= THR).reshape(len(r), -1)
imp = np.array([z[i][occ[i]].mean() for i in range(len(r)) if occ[i].any()])
print(f"A6 own-field imprint at the end of assembly (field used in the final run): mean z {imp.mean():+.3f} +- {2 * imp.std() / np.sqrt(len(imp)):.3f} "
      f"(above 0 = ranges sit on better cells of their own field)")
