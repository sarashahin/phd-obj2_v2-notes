"""
patch_world_vs_bnm_habitat_tau.py  (obj1_v2, root cause 23, fix B, 31 Oct 2026, Sara's yes)

world_vs_bnm v5.9: niche worlds whose habitat changes through time (saved HABITAT_tau > 0, runner --habitat-tau) get
", habitat tau <steps>" after the habitat and width parts of their setting key. Without it the static twins
(n29-n32, n33) and the moving worlds of plan niche_t (n49-n52 at 1,000,000 steps, n53-n56 at 250,000, n57, n58)
would be pooled into one setting verdict each founding rule; the same problem w_min had (fixed by v5.6).
Fixed-habitat worlds (HABITAT_tau -1), files without the key and field-mode worlds keep their label exactly.
Adds one self-test line (30 -> 31 PASS lines). Nothing else changes; score() and observe() are untouched.
Needs world_vs_bnm v5.8 (patch_world_vs_bnm_detect_size.py) and v5.6 (patch_world_vs_bnm_wmin.py).

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_habitat_tau.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_habitat_tau.py --apply
    python3 ../tools/world_vs_bnm.py --self-test | grep -c PASS          # 31
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
world_vs_bnm.before_habitat_tau.py.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 habitat-tau"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has the habitat tau label")
if "# obj1_v2 detect-size" not in src or "# obj1_v2 width" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_wmin.py (v5.6) and patch_world_vs_bnm_detect_size.py (v5.8) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip() == "cells in a window is recorded with p_s, others with --detect (BNM fit half: 0.64 for 1-20 tetrads, 0.80 for 21+).",
         "v5.8 docstring line")
edits.append((i + 1, i + 1, [
    "Version 5.9 (31 Oct): niche worlds whose habitat changes through time (HABITAT_tau > 0) get \", habitat tau <steps>\"",
    "  in their setting key, so plan niche_t is not pooled with its static twins; fixed-habitat worlds keep their label."]))
j = find(lambda s: s.strip() == "wm = f\", w_min {wm:g}\" if (mode == 'niche' and abs(wm - 0.2) > 1e-6) else \"\"", "w_min label line")
i = find(lambda s: s.strip() == "return (f\"{mode}, {env}{wm}, connectance {float(g('CONNECTANCE', 0.4)):g}{st}{dps}, founded {fnd}, \"",
         "setting key return line (v5.6)")
if i != j + 1:
    sys.exit("STOP: the setting key return line is not right after the w_min label; nothing written")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}ht = int(g('HABITAT_tau', -1))   {MARK}",
    f"{p}ht = f\", habitat tau {{ht}}\" if (mode == 'niche' and ht > 0) else \"\"",
    f"{p}return (f\"{{mode}}, {{env}}{{wm}}{{ht}}, connectance {{float(g('CONNECTANCE', 0.4)):g}}{{st}}{{dps}}, founded {{fnd}}, \""]))
i = find(lambda s: s.strip() == "print(f\"self-test, minimum niche width other than 0.2 gives its own setting; 0.2, not saved or field mode keeps its label: \"",
         "w_min self-test print")
if L[i + 1].strip() != "f\"{'PASS' if good else 'FAIL'}\")":
    sys.exit("STOP: the w_min self-test print is not the v5.6 text; nothing written")
p = ind(L[i])
edits.append((i + 2, i + 2, [
    f"{p}with _tf.TemporaryDirectory() as tmp:   {MARK}: a habitat changing through time gives its own setting",
    f"{p}    bt = {{k: v for k, v in base.items() if k not in ('HABITAT_tau', 'NICHE_w_min')}}",
    f"{p}    for n, extra in dict(t0={{}}, t1=dict(HABITAT_tau=np.int64(-1)), t6=dict(HABITAT_tau=np.int64(1000000)),",
    f"{p}                         t2=dict(HABITAT_tau=np.int64(250000)),",
    f"{p}                         tw=dict(HABITAT_tau=np.int64(250000), NICHE_w_min=np.float32(0.3))).items():",
    f"{p}        np.savez(f\"{{tmp}}/{{n}}.npz\", **bt, **extra); key[n] = setting_key(f\"{{tmp}}/{{n}}.npz\")",
    f"{p}    bf = dict({{k: v for k, v in bt.items() if k != 'ENV_mode'}}, ENV_mode=np.array(['field']), HABITAT_tau=np.int64(5000))",
    f"{p}    np.savez(f\"{{tmp}}/tf.npz\", **bf); key['tf'] = setting_key(f\"{{tmp}}/tf.npz\")",
    f"{p}good = (key['t0'] == key['t1'] and 'habitat tau' not in key['t0'] and len({{key['t0'], key['t6'], key['t2']}}) == 3",
    f"{p}        and ', habitat tau 1000000, connectance' in key['t6'] and ', w_min 0.3, habitat tau 250000, connectance' in key['tw']",
    f"{p}        and 'habitat tau' not in key['tf']); ok &= good",
    f"{p}print(f\"self-test, a habitat changing through time gives its own setting (tau named); fixed (-1 or not saved) or \"",
    f"{p}      f\"field mode keeps its label: {{'PASS' if good else 'FAIL'}}\")"]))

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
    Path("../tools/world_vs_bnm.before_habitat_tau.py").write_text(src)
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py (the version before is kept as world_vs_bnm.before_habitat_tau.py)")
else:
    print("dry run: nothing written (add --apply to write)")