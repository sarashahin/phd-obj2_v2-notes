"""
patch_world_vs_bnm_habitat_shift.py  (obj1_v2, root cause 23, fix B part 2, 31 Oct 2026, Sara's yes)

world_vs_bnm v5.10: niche worlds whose habitat layer also shifts as a whole (saved HABITAT_shift_sd > 0, runner
--habitat-shift-sd, assembly v2.5) get ", shift sd <s> tau <steps>" right after their habitat tau in the setting key,
so plan niche_h (n61-n64 shift sd 1.0 on n49-n52; n65, n66 shift sd 0.5 and 1.0 on n59) is not pooled with its twins.
Worlds without the shift keep their label exactly. Adds one self-test line (31 -> 32 PASS lines). Nothing else changes.
Needs world_vs_bnm v5.9 (patch_world_vs_bnm_habitat_tau.py).

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_habitat_shift.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_habitat_shift.py --apply
    python3 ../tools/world_vs_bnm.py --self-test | grep -c PASS          # 32
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
world_vs_bnm.before_habitat_shift.py.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 habitat-shift"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has the habitat shift label")
if "# obj1_v2 habitat-tau" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_habitat_tau.py (v5.9) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "in their setting key, so plan niche_t is not pooled with its static twins; fixed-habitat worlds keep their label.",
         "v5.9 docstring line")
edits.append((i + 1, i + 1, [
    "Version 5.10 (31 Oct): niche worlds whose whole habitat layer also shifts (HABITAT_shift_sd > 0) get \", shift sd <s>",
    "  tau <steps>\" after the habitat tau, so plan niche_h is not pooled with its twins; other worlds keep their label."]))
i = find(lambda s: s.strip() == "ht = f\", habitat tau {ht}\" if (mode == 'niche' and ht > 0) else \"\"", "habitat tau label line")
p = ind(L[i])
edits.append((i + 1, i + 1, [
    f"{p}hs = float(g('HABITAT_shift_sd', 0.0))   {MARK}",
    f"{p}ht += f\", shift sd {{hs:g}} tau {{int(g('HABITAT_shift_tau', 0))}}\" if (mode == 'niche' and hs > 0) else \"\""]))
i = find(lambda s: s.strip() == "print(f\"self-test, a habitat changing through time gives its own setting (tau named); fixed (-1 or not saved) or \"",
         "habitat tau self-test print")
if L[i + 1].strip() != "f\"field mode keeps its label: {'PASS' if good else 'FAIL'}\")":
    sys.exit("STOP: the habitat tau self-test print is not the v5.9 text; nothing written")
p = ind(L[i])
edits.append((i + 2, i + 2, [
    f"{p}with _tf.TemporaryDirectory() as tmp:   {MARK}: a shifting habitat layer gives its own setting",
    f"{p}    bs = dict({{k: v for k, v in base.items() if k not in ('HABITAT_tau', 'HABITAT_shift_sd', 'HABITAT_shift_tau')}},",
    f"{p}              HABITAT_tau=np.int64(62500))",
    f"{p}    for n, extra in dict(h0={{}}, h5=dict(HABITAT_shift_sd=np.float64(0.5), HABITAT_shift_tau=np.int64(62500)),",
    f"{p}                         h1=dict(HABITAT_shift_sd=np.float64(1.0), HABITAT_shift_tau=np.int64(62500))).items():",
    f"{p}        np.savez(f\"{{tmp}}/{{n}}.npz\", **bs, **extra); key[n] = setting_key(f\"{{tmp}}/{{n}}.npz\")",
    f"{p}    bf = dict({{k: v for k, v in bs.items() if k != 'ENV_mode'}}, ENV_mode=np.array(['field']), HABITAT_shift_sd=np.float64(1.0),",
    f"{p}              HABITAT_shift_tau=np.int64(62500))",
    f"{p}    np.savez(f\"{{tmp}}/hf.npz\", **bf); key['hf'] = setting_key(f\"{{tmp}}/hf.npz\")",
    f"{p}good = ('shift' not in key['h0'] and len({{key['h0'], key['h5'], key['h1']}}) == 3",
    f"{p}        and ', habitat tau 62500, shift sd 0.5 tau 62500, connectance' in key['h5']",
    f"{p}        and ', habitat tau 62500, shift sd 1 tau 62500, connectance' in key['h1'] and 'shift' not in key['hf']); ok &= good",
    f"{p}print(f\"self-test, a shifting habitat layer gives its own setting (shift named); no shift or field mode keeps its label: \"",
    f"{p}      f\"{{'PASS' if good else 'FAIL'}}\")"]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "world_vs_bnm.py (now)", "world_vs_bnm.py (patched)", lineterm="", n=1)))
compile(out, "world_vs_bnm.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    Path("../tools/world_vs_bnm.before_habitat_shift.py").write_text(src)
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py (the version before is kept as world_vs_bnm.before_habitat_shift.py)")
else:
    print("dry run: nothing written (add --apply to write)")