"""
patch_world_vs_bnm_setting.py  (obj1_v2, stage 2d-a, 28 Oct 2026)

world_vs_bnm.py version 5 groups worlds into settings by mode, trade-off, contrast, founding,
kernel and window. The 2 x 2 run (plan niche_f) varies habitat length scale and connectance, which
the key does not name, so worlds of different settings would be pooled into one verdict.
This patch (version 5.1) adds both to the setting key:
  - habitat scale (HABITAT_ls) for niche worlds; files without it count as 3;
  - connectance (CONNECTANCE) for every world; files without it count as 0.4 (the model value
    before --connectance existed).
Worlds made so far keep their grouping; only the printed label gets the two parts.
A self-test line is added: files that differ only in scale or connectance get their own setting;
a file without the keys joins the one that saves the default values.

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_setting.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_setting.py --apply
    python3 ../tools/world_vs_bnm.py --self-test
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 setting"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has the version 5.1 setting key")
if "# obj1_v2 bands" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_bands.py (version 5) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


edits = []
i = find(lambda s: s.strip() == "co-occupancy at 1-4 cells; PASS/FAIL per world and per setting (median inside, 3 of 4 worlds inside).", "version 5 docstring")
edits.append((i + 1, i + 1, [
    "Version 5.1 (28 Oct): the setting key also names habitat length scale (niche worlds; 3 if not saved) and",
    "  connectance (0.4 if not saved), so the 2 x 2 worlds of plan niche_f are not pooled."]))
i = find(lambda s: s.strip() == "def setting_key(f):", "def setting_key")
want = ['    """Worlds of one setting share mode, trade-off, contrast, founding, kernel and window (d and seeds may differ)."""',
        "    z = np.load(f, allow_pickle=True)",
        "    g = lambda k, d: (z[k].item() if z[k].ndim == 0 else str(z[k][0])) if k in z.files else d",
        "    mode = str(g('ENV_mode', 'field'))",
        "    env = (f\"k {float(g('NICHE_tradeoff', 1.0)):g}, contrast {float(g('HABITAT_contrast', 1.0)):g}\" if mode == 'niche'",
        "           else f\"field var {float(g('ENV_var_r', -1)):g}\")",
        "    return f\"{mode}, {env}, founded in {g('seed_size', '?')}, kernel {g('DISPERSAL_KERNEL', '?')}, window {g('window_steps', '?')}\""]
if L[i + 1:i + 8] != want:
    sys.exit("STOP: setting_key body is not the version 5 text; nothing written")
edits.append((i, i + 8, [
    f"def setting_key(f):   {MARK}",
    '    """Worlds of one setting share mode, trade-off, contrast, habitat scale, connectance, founding, kernel and window',
    '    (d and seeds may differ). Files saved before these keys existed: habitat scale 3, connectance 0.4."""',
    "    z = np.load(f, allow_pickle=True)",
    "    g = lambda k, d: (z[k].item() if z[k].ndim == 0 else str(z[k][0])) if k in z.files else d",
    "    mode = str(g('ENV_mode', 'field'))",
    "    env = (f\"k {float(g('NICHE_tradeoff', 1.0)):g}, contrast {float(g('HABITAT_contrast', 1.0)):g}, \"",
    "           f\"habitat scale {float(g('HABITAT_ls', 3.0)):g}\" if mode == 'niche'",
    "           else f\"field var {float(g('ENV_var_r', -1)):g}\")",
    "    return (f\"{mode}, {env}, connectance {float(g('CONNECTANCE', 0.4)):g}, founded in {g('seed_size', '?')}, \"",
    "            f\"kernel {g('DISPERSAL_KERNEL', '?')}, window {g('window_steps', '?')}\")"]))
i = find(lambda s: s.strip() == "print(f\"self-test, a setting passes with 3 of 4 worlds inside and fails with 2 of 4: {'PASS' if good else 'FAIL'}\")", "setting self-test line")
edits.append((i + 1, i + 1, [
    f"    import tempfile as _tf   {MARK}: scale and connectance split settings; missing keys = defaults",
    "    with _tf.TemporaryDirectory() as tmp:",
    "        base = dict(ENV_mode=np.array(['niche']), NICHE_tradeoff=np.float32(0.5), HABITAT_contrast=np.float32(0.25),",
    "                    seed_size=np.int64(1), DISPERSAL_KERNEL=np.array(['2dt']), window_steps=np.int64(5000))",
    "        sets = dict(a={}, b=dict(HABITAT_ls=np.float32(3.0), CONNECTANCE=np.float32(0.4)), c=dict(CONNECTANCE=np.float32(1.0)),",
    "                    d=dict(HABITAT_ls=np.float32(1.0)), e=dict(HABITAT_ls=np.float32(1.0), CONNECTANCE=np.float32(1.0)))",
    "        key = {}",
    "        for n, extra in sets.items():",
    "            np.savez(f\"{tmp}/{n}.npz\", **base, **extra); key[n] = setting_key(f\"{tmp}/{n}.npz\")",
    "    good = (key['a'] == key['b'] and len({key[n] for n in 'acde'}) == 4 and 'habitat scale 3, connectance 0.4' in key['a']",
    "            and 'habitat scale 1, connectance 1,' in key['e']); ok &= good",
    "    print(f\"self-test, habitat scale and connectance give their own settings; files without them = scale 3, \"",
    "          f\"connectance 0.4: {'PASS' if good else 'FAIL'}\")"]))

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
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py")
else:
    print("dry run: nothing written (add --apply to write)")