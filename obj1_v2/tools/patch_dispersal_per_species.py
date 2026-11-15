"""
patch_dispersal_per_species.py  (obj1_v2, stage 2d-b, 29 Oct 2026)

Adds a dispersal rate per species to work/dispersal.py (v2, md5 b2035f4e). Nothing changes
unless a caller switches it on, so every world made so far is reproduced exactly.
  1. species_dispersal_rate(sid, *, seed, d_min, d_max): the d of species sid, log-uniform in
     [d_min, d_max], drawn from (seed, sid) only, on a random stream of its own (the niche
     traits use default_rng([seed, 7919, sid]); this uses default_rng([seed, 7933, sid])), so a
     species keeps its d whatever its place in the list and its niche does not change.
  2. set_species_rates(arr): the rates of the species of the next IBM run, in list order
     (shape S); None switches back to the shared DISPERSAL_RATE. Kept on the active backend
     (CuPy on the GPU), set once per run, like set_invasion_pressure.
  3. compute_dispersal(B): with rates set, species i emits d_i * B_i, landing by the same unit
     kernel (and the same long-distance share); without, the code runs exactly as before.
     A rates vector of the wrong length stops the run.
Why (sketch 33-34, test_14): at d 1e-6 a sink cell next to a source holds 0.004 individuals,
so no species can be held where it cannot grow; LVMCM's example rate 1e-2 holds 28-72.
Sara's yes (29 Oct): per-species d, log-uniform 1e-6 to 1e-2, independent of the niche.

Run from obj1_v2/work:
    python3 ../tools/patch_dispersal_per_species.py            # dry run, prints the diff
    python3 ../tools/patch_dispersal_per_species.py --apply
Anchors are matched on the code part of a line (a trailing # comment is ignored); every anchor
must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("dispersal.py"); MARK = "# obj1_v2 per-species d"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: dispersal.py already has per-species d")
L = src.split("\n")
code = lambda s: s.split("#")[0].strip()


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


edits = []
i = find(lambda s: code(s) == "_KERNEL_CACHE = {}", "_KERNEL_CACHE")
edits.append((i + 1, i + 1, [f"_SPECIES_RATES = None   {MARK}: rates of the species of the current IBM run, or None"]))

i = find(lambda s: code(s) == "def set_invasion_pressure(arr):", "def set_invasion_pressure")
j = i + 1
while j < len(L) and (L[j].startswith(" ") or not L[j].strip()):
    j += 1
while not L[j - 1].strip():
    j -= 1
edits.append((j, j, [
    "",
    "",
    f"def set_species_rates(arr):   {MARK}",
    "    \"\"\"arr: dispersal rate of each species of the next IBM run, in list order (shape S);",
    "    None switches back to the shared DISPERSAL_RATE.\"\"\"",
    "    global _SPECIES_RATES",
    "    if arr is None:",
    "        _SPECIES_RATES = None",
    "    else:",
    "        a = np.asarray(arr, float).reshape(-1)",
    "        if not bool((a >= 0).all()):",
    "            raise ValueError(\"species dispersal rates must be >= 0\")",
    "        _SPECIES_RATES = a",
    "",
    "",
    f"def species_dispersal_rate(sid, *, seed, d_min, d_max):   {MARK}",
    "    \"\"\"d of species sid: log-uniform in [d_min, d_max], from (seed, sid) only, on its own random",
    "    stream (niche traits use default_rng([seed, 7919, sid])), so it never depends on list position.\"\"\"",
    "    if not 0 < d_min <= d_max:",
    "        raise ValueError(f\"need 0 < d_min <= d_max, got {d_min}, {d_max}\")",
    "    if d_min == d_max:",
    "        return float(d_min)",
    "    u = _np.random.default_rng([int(seed), 7933, int(sid)]).random()",
    "    return float(_np.exp(_np.log(d_min) + u * (_np.log(d_max) - _np.log(d_min))))"]))

i = find(lambda s: code(s) == "M = DISPERSAL_RATE * unit_kernel_matrix()", "M = DISPERSAL_RATE * unit_kernel_matrix()")
if code(L[i + 1]) != "dispersal_flux = (M.T @ B2.T).T":
    sys.exit("STOP: the line after 'M = ...' is not 'dispersal_flux = (M.T @ B2.T).T'; nothing written")
p = L[i][: len(L[i]) - len(L[i].lstrip())]
edits.append((i, i + 2, [
    f"{p}if _SPECIES_RATES is None:   {MARK}: shared d, exactly as before",
    "    " + L[i],
    "    " + L[i + 1],
    f"{p}else:                          # species i emits d_i x B_i by the same kernel",
    f"{p}    if _SPECIES_RATES.shape[0] != S:",
    f"{p}        raise ValueError(f\"species dispersal rates: {{_SPECIES_RATES.shape[0]}} given for {{S}} species\")",
    f"{p}    dispersal_flux = (unit_kernel_matrix().T @ B2.T).T * _SPECIES_RATES[:, None]"]))

starts = sorted(e[0] for e in edits)
if len(set(starts)) != len(starts):
    sys.exit("STOP: two edits touch the same line; nothing written")
new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "dispersal.py (now)", "dispersal.py (patched)", lineterm="", n=1)))
compile(out, "dispersal.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: dispersal.py")
else:
    print("dry run: nothing written (add --apply to write)")