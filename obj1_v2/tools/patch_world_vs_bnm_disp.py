"""
patch_world_vs_bnm_disp.py  (obj1_v2, stage 2d-b, 29 Oct 2026)

world_vs_bnm.py v5.1 groups worlds into settings by mode, trade-off, contrast, habitat scale,
connectance, founding, kernel and window. Plan niche_d (n29-n32) differs from its twins n21-n24
only in the per-species dispersal rate, so without this patch the 8 worlds would be pooled.
Version 5.2 adds ", d per species <DMIN>-<DMAX>" to the setting key of worlds saved with
DISP_per_species 1; worlds with shared d (DISP_per_species 0 or not saved) keep their label as it is.
A self-test line is added.

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_disp.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_disp.py --apply
    python3 ../tools/world_vs_bnm.py --self-test
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 per-species d"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has version 5.2")
if "# obj1_v2 setting" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_setting.py (version 5.1) first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


edits = []
i = find(lambda s: s.strip() == "connectance (0.4 if not saved), so the 2 x 2 worlds of plan niche_f are not pooled.", "version 5.1 docstring")
edits.append((i + 1, i + 1, [
    "Version 5.2 (29 Oct): worlds with a dispersal rate per species (DISP_per_species 1) get \", d per species",
    "  <DMIN>-<DMAX>\" in their setting key, so plan niche_d is not pooled with its twins."]))
i = find(lambda s: s.strip() == "return (f\"{mode}, {env}, connectance {float(g('CONNECTANCE', 0.4)):g}, founded in {g('seed_size', '?')}, \"",
         "setting_key return")
if L[i + 1].strip() != "f\"kernel {g('DISPERSAL_KERNEL', '?')}, window {g('window_steps', '?')}\")":
    sys.exit("STOP: setting_key return is not the version 5.1 text; nothing written")
edits.append((i, i + 2, [
    f"    dps = (f\", d per species {{float(g('DISP_d_min', 0)):g}}-{{float(g('DISP_d_max', 0)):g}}\"   {MARK}",
    "           if int(g('DISP_per_species', 0)) else \"\")",
    "    return (f\"{mode}, {env}, connectance {float(g('CONNECTANCE', 0.4)):g}{dps}, founded in {g('seed_size', '?')}, \"",
    "            f\"kernel {g('DISPERSAL_KERNEL', '?')}, window {g('window_steps', '?')}\")"]))
i = find(lambda s: s.strip() == "f\"connectance 0.4: {'PASS' if good else 'FAIL'}\")", "version 5.1 self-test print end")
edits.append((i + 1, i + 1, [
    f"    with _tf.TemporaryDirectory() as tmp:   {MARK}: per-species d splits settings; shared d keeps its label",
    "        for n, extra in dict(a={}, b=dict(DISP_per_species=np.int8(0)),",
    "                             c=dict(DISP_per_species=np.int8(1), DISP_d_min=np.float64(1e-6), DISP_d_max=np.float64(1e-2))).items():",
    "            np.savez(f\"{tmp}/{n}.npz\", **base, **extra); key[n] = setting_key(f\"{tmp}/{n}.npz\")",
    "    good = (key['a'] == key['b'] and 'per species' not in key['a'] and key['c'] != key['a']",
    "            and 'connectance 0.4, d per species 1e-06-0.01, founded' in key['c']); ok &= good",
    "    print(f\"self-test, per-species d gives its own setting; shared d (0 or not saved) keeps its label: {'PASS' if good else 'FAIL'}\")"]))

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