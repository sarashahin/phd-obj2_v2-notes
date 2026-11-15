"""
patch_world_vs_bnm_wmin.py  (obj1_v2, minimum niche width, 30 Oct 2026)

world_vs_bnm v5.6: niche worlds whose saved NICHE_w_min differs from 0.2 get ", w_min <w>" after the habitat part
of their setting key, so the worlds of plan niche_w (w_min 0.3 and 0.4) are not pooled with each other or with
n33 (0.2). On 30 Oct the setting summary pooled all 9 worlds as one setting. Worlds at 0.2, files without the key,
and field-mode worlds keep their label exactly. Adds one self-test line (27 -> 28 PASS lines). Nothing else
changes; score() is untouched.
Needs world_vs_bnm v5.5 (patch_world_vs_bnm_tau.py).

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_wmin.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_wmin.py --apply
    python3 ../tools/world_vs_bnm.py --self-test | grep -c PASS          # 28
Every anchor must be found exactly once, or nothing is written. --apply keeps the file before as
world_vs_bnm.before_wmin.py.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 width"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has the width label")
if "# obj1_v2 tau" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_tau.py (v5.5) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("the last frame (B_last), for observation and scoring only; the assembly is unchanged."),
         "v5.5 docstring line")
edits.append((i + 1, i + 1, [
    "Version 5.6 (30 Oct): niche worlds whose saved minimum niche width is not 0.2 get \", w_min <w>\" in their setting",
    "  key, so plan niche_w is not pooled; worlds at 0.2, files without NICHE_w_min and field worlds keep their label."]))
j = find(lambda s: s.strip() == "st = f\", strength {st:g}\" if abs(st - 0.4) > 1e-6 else \"\"", "strength label line")
i = find(lambda s: s.strip() == "return (f\"{mode}, {env}, connectance {float(g('CONNECTANCE', 0.4)):g}{st}{dps}, founded {fnd}, \"",
         "setting key return line")
if i != j + 1:
    sys.exit("STOP: the setting key return line is not right after the strength label; nothing written")
p = ind(L[i])
edits.append((i, i + 1, [
    f"{p}wm = float(g('NICHE_w_min', 0.2))   {MARK}",
    f"{p}wm = f\", w_min {{wm:g}}\" if (mode == 'niche' and abs(wm - 0.2) > 1e-6) else \"\"",
    f"{p}return (f\"{{mode}}, {{env}}{{wm}}, connectance {{float(g('CONNECTANCE', 0.4)):g}}{{st}}{{dps}}, founded {{fnd}}, \""]))
i = find(lambda s: s.strip() == "print(f\"self-test, competition strength other than 0.4 gives its own setting; 0.4 (or not saved) keeps its label: \"",
         "strength self-test print")
if L[i + 1].strip() != "f\"{'PASS' if good else 'FAIL'}\")":
    sys.exit("STOP: the strength self-test print is not the v5.4 text; nothing written")
p = ind(L[i])
edits.append((i + 2, i + 2, [
    f"{p}with _tf.TemporaryDirectory() as tmp:   {MARK}: a minimum niche width other than 0.2 gives its own setting",
    f"{p}    bn = dict({{k: v for k, v in base.items() if k != 'NICHE_w_min'}}, ENV_mode=np.array(['niche']))",
    f"{p}    for n, extra in dict(w0={{}}, w2=dict(NICHE_w_min=np.float32(0.2)), w3=dict(NICHE_w_min=np.float32(0.3)),",
    f"{p}                         w4=dict(NICHE_w_min=np.float32(0.4))).items():",
    f"{p}        np.savez(f\"{{tmp}}/{{n}}.npz\", **bn, **extra); key[n] = setting_key(f\"{{tmp}}/{{n}}.npz\")",
    f"{p}    bf = dict({{k: v for k, v in base.items() if k not in ('NICHE_w_min', 'ENV_mode')}}, ENV_mode=np.array(['field']),",
    f"{p}              NICHE_w_min=np.float32(0.3))",
    f"{p}    np.savez(f\"{{tmp}}/f3.npz\", **bf); key['f3'] = setting_key(f\"{{tmp}}/f3.npz\")",
    f"{p}good = (key['w0'] == key['w2'] and 'w_min' not in key['w0'] and len({{key['w0'], key['w3'], key['w4']}}) == 3",
    f"{p}        and ', w_min 0.3, connectance' in key['w3'] and ', w_min 0.4, connectance' in key['w4'] and 'w_min' not in key['f3']); ok &= good",
    f"{p}print(f\"self-test, minimum niche width other than 0.2 gives its own setting; 0.2, not saved or field mode keeps its label: \"",
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
    Path("../tools/world_vs_bnm.before_wmin.py").write_text(src)
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py (the version before is kept as ../tools/world_vs_bnm.before_wmin.py)")
else:
    print("dry run: nothing written (add --apply to write)")