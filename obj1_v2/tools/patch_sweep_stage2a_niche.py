"""
patch_sweep_stage2a_niche.py  (obj1_v2, stage 2d-a part B, 27 Oct 2026)

Adds plan "niche" to sweep_stage2a.py: the first niche worlds, paired with field-mode twins.

  n1  setting and sim seed of settle s3 (d 1e-6, 2Dt a 1 b 3, 1 cell, window 5,000, seed 2303)
  n2  as s4 (seed 2304);  n3  as s7 (d 3e-6, seed 2307);  n4  as s8 (seed 2308)
  all in niche mode: habitat length scale 3 cells, optimum sd 1.5, width 0.2-1.0, cost 0.1
  (Sara, 27 Oct); habitat seed 100000 + sim seed, trait seed 200000 + sim seed, so every
  world has its own habitat and niches. 1,000 attempts, final run 10,000 steps.
Each world differs from its settle twin only in the environment, so the twins are the
control for every effect claimed for the niche model. Seed rule for later plans: training and
evaluation worlds take sim seeds from separate ranges, and habitat and trait seeds follow from
the sim seed, so the two sets never share a habitat or a niche draw.

Also: rows with niche settings get their saved mode, habitat seed, trait seed and niche
settings checked (a world that does not carry them goes to rejected/); older rows must not be
niche worlds. Needs run_all_rps.py with patch_run_all_rps_niche.py for this plan only.

Run from obj1_v2/work:
    python3 ../tools/patch_sweep_stage2a_niche.py            # dry run, prints the diff
    python3 ../tools/patch_sweep_stage2a_niche.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/sweep_stage2a.py"); MARK = "# obj1_v2 niche"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: the niche plan is already in sweep_stage2a.py")
if "# obj1_v2 extend" not in src:
    sys.exit("STOP: apply patch_sweep_stage2a_extend.py first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "2308) with 2,000 attempts. test_10 compares each with its 1,000-attempt twin.", "extend docstring end")
edits.append((i + 1, i + 1, [
    "  niche  4 worlds (added 27 Oct): settings and sim seeds of settle s3, s4, s7, s8 in niche mode (habitat",
    "         scale 3, optimum sd 1.5, width 0.2-1.0, cost 0.1; habitat seed 100000 + sim seed, trait seed",
    "         200000 + sim seed). Each settle twin is the control for the niche effect."]))
i = find(lambda s: s.strip() == 'if plan == "extend":   # obj1_v2 extend', "extend plan")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if plan == "niche":   {MARK}',
    f'{p}    return [dict(name=f"n{{k}}", d=d, kernel="2dt_a1_b3", f=1, seed=s, attempts=1000, window=5000, niche=True,',
    f'{p}                 habitat_seed=100000 + s, trait_seed=200000 + s)',
    f'{p}            for k, (d, s) in enumerate((("1e-6", 2303), ("1e-6", 2304), ("3e-6", 2307), ("3e-6", 2308)), 1)]']))
i = find(lambda s: s.strip() == "def command(r, smoke=False):", "def command")
edits.append((i, i, [
    f"NICHE_SET = dict(HABITAT_ls=3.0, NICHE_opt_sd=1.5, NICHE_w_min=0.2, NICHE_w_max=1.0, NICHE_cost=0.1)   {MARK}",
    "",
    "",
    f"def niche_ok(z, r, smoke):   {MARK}",
    '    """Niche rows: the file is a niche world with this row\'s habitat seed, trait seed and settings.',
    '    Other rows: the file is not a niche world."""',
    '    mode = str(z["ENV_mode"][0]) if "ENV_mode" in z.files else "field"',
    '    if not r.get("niche"):',
    '        return mode == "field"',
    '    return (mode == "niche" and int(z["HABITAT_seed"]) == r["habitat_seed"] and int(z["TRAIT_seed"]) == r["trait_seed"]',
    '            and all(abs(float(z[k]) - v) < 1e-6 for k, v in NICHE_SET.items()))',
    "",
    ""]))
i = find(lambda s: s.strip() == 'if r.get("tmax"): base[base.index("--tmax") + 1] = str(r["tmax"])   # obj1_v2 long', "tmax line in command")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f'{p}if r.get("niche"):   {MARK}: field-mode environment options out, niche options in',
    f'{p}    for k in ("--env-length-scale", "--env-var-r", "--env-seed-field"):',
    f'{p}        j = base.index(k); del base[j:j + 2]',
    f'{p}    base += ["--env-mode", "niche", "--habitat-seed", str(r["habitat_seed"]), "--env-seed-field", str(r["trait_seed"]),',
    f'{p}             "--habitat-ls", "3", "--niche-opt-sd", "1.5", "--niche-w-min", "0.2", "--niche-w-max", "1.0",',
    f'{p}             "--niche-cost", "0.1"]']))
i = find(lambda s: 'choices=["pilot", "full", "settle", "long", "extend"]' in s, "--plan choices")
edits.append((i, i + 1, [L[i].replace('choices=["pilot", "full", "settle", "long", "extend"]',
                                      'choices=["pilot", "full", "settle", "long", "extend", "niche"]') + f"   {MARK}"]))
i = find(lambda s: s.strip() == "rows = plan_rows(a.plan, a.reps)", "plan_rows call")
p = ind(L[i])
edits.append((i, i, [
    f'{p}if a.plan == "niche" and "--env-mode" not in src:   {MARK}',
    f'{p}    sys.exit("STOP: run_all_rps.py lacks --env-mode; apply patch_run_all_rps_niche.py first")']))
i = find(lambda s: s.strip().startswith("ok_meta = meta_ok(z, r, a.smoke) and tmax_ok(z, r, a.smoke) and attempts_ok(z, r, a.smoke)"), "ok_meta call")
edits.append((i, i + 1, [f"{ind(L[i])}ok_meta = (meta_ok(z, r, a.smoke) and tmax_ok(z, r, a.smoke) and attempts_ok(z, r, a.smoke)"
                         f" and niche_ok(z, r, a.smoke))   {MARK}"]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "sweep_stage2a.py (now)", "sweep_stage2a.py (patched)", lineterm="", n=1)))
compile(out, "sweep_stage2a.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: ../tools/sweep_stage2a.py")
else:
    print("dry run: nothing written (add --apply to write)")