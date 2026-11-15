"""
patch_world_vs_bnm_bands.py  (obj1_v2, 28 Oct 2026)

world_vs_bnm v5: pass bands written before the next run, so every world and every setting gets
PASS or FAIL against BNM on the whole community pattern. Replaces the PASS/FAIL of the 24 Oct
bands (x1.3, TV 0.15, x0.5-2) in the printout; score() itself is unchanged (test_10 uses it).

Band = BNM value (minlist 8, fit half) +- (D + S):
  D  definition spread: largest distance of BNM minlist 4 or 12 from minlist 8 (Sara's runs, 28 Oct)
  S  seed spread: half the range over the 4 worlds of one set, the larger of n1-n4 and n5-n8
     (each set mixes d 1e-6 and 3e-6, so S is on the generous side)
  number                                   D        S        band
  species per window (relative)            0.118    0.089    +-0.207
  species per cell, alpha (relative)       0.270    0.085    +-0.355
  turnover, beta (relative)                0.248    0.060    +-0.308
  similarity decay, 1 minus 5-10 cells     0.000    0.045    +-0.045 around BNM's 0.04
  range-size TV to BNM                     0.065    0.051    <= 0.116
  share of cases in 1-5 cells              0.028    0.053    +-0.081
  share of cases in 51+ cells              0.032    0.016    +-0.048
  rare co-occupancy / BNM at 1, 2, 3, 4    0.121 0.087 0.057 0.027 + 0.075 0.145 0.080 0.050
                                           -> within 1 +- 0.196, 0.232, 0.137, 0.077
A setting (worlds with the same mode, trade-off, contrast, founding, kernel and window; d and
seeds vary) passes a number when its median is inside the band and at least 3 of 4 worlds
(75%) are; it passes when it passes every number. Rare-species numbers are part of the set.

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_bands.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_bands.py --apply
    python3 ../tools/world_vs_bnm.py --self-test            # 19 earlier checks + 3 new = 22 PASS
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 bands"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has the 28 Oct bands")
if "# obj1_v2 alpha" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_alpha.py first (world_vs_bnm v4)")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


edits = []
i = find(lambda s: s.strip() == "(geometric means over windows, so gamma = alpha x beta exactly). Description only.", "v4 docstring end")
edits.append((i + 1, i + 1, [
    "Version 5 (28 Oct): pass bands (BNM minlist 8 +- definition spread + seed spread) for species per",
    "  window, species per cell, turnover, similarity decay, range-size TV, 1-5 and 51+ shares and rare",
    "  co-occupancy at 1-4 cells; PASS/FAIL per world and per setting (median inside, 3 of 4 worlds inside)."]))
i = find(lambda s: s.strip() == "def report(name, s):", "def report")
edits.append((i, i, [
    f"BANDS = dict(species=0.207, alpha=0.355, beta=0.308, decay=0.045, tv=0.116, r15=0.081, r51=0.048,   {MARK}",
    "             co=(0.196, 0.232, 0.137, 0.077))",
    "",
    "",
    "def band_values(s, b):",
    "    \"\"\"The numbers judged against BNM summary b.\"\"\"",
    "    co, tv, rr, _ = score(s, b)",
    "    return dict(species=s['rich_med'], alpha=s['alpha_gm'], beta=s['beta_gm'], decay=float(s['sor'][0] - s['sor'][3]),",
    "                tv=tv, r15=s['bins'][0] + s['bins'][1], r51=s['bins'][5], co=np.asarray(co, float))",
    "",
    "",
    "def band_check(v, b):",
    "    \"\"\"PASS flags of the values v (band_values) against BNM summary b.\"\"\"",
    "    rel = lambda x, t, w: bool(abs(x / t - 1) <= w)",
    "    return dict(species=rel(v['species'], b['rich_med'], BANDS['species']), alpha=rel(v['alpha'], b['alpha_gm'], BANDS['alpha']),",
    "                beta=rel(v['beta'], b['beta_gm'], BANDS['beta']),",
    "                decay=bool(abs(v['decay'] - (b['sor'][0] - b['sor'][3])) <= BANDS['decay']), tv=bool(v['tv'] <= BANDS['tv']),",
    "                r15=bool(abs(v['r15'] - (b['bins'][0] + b['bins'][1])) <= BANDS['r15']),",
    "                r51=bool(abs(v['r51'] - b['bins'][5]) <= BANDS['r51']),",
    "                co=bool(np.all(np.abs(v['co'] - 1) <= np.array(BANDS['co']))))",
    "",
    "",
    "BAND_NAMES = dict(species='species per window', alpha='species per cell', beta='turnover', decay='similarity decay',",
    "                  tv='range-size TV', r15='1-5 cell share', r51='51+ share', co='rare clumping 1-4 cells')",
    "",
    "",
    "def setting_key(f):",
    "    \"\"\"Worlds of one setting share mode, trade-off, contrast, founding, kernel and window (d and seeds may differ).\"\"\"",
    "    z = np.load(f, allow_pickle=True)",
    "    g = lambda k, d: (z[k].item() if z[k].ndim == 0 else str(z[k][0])) if k in z.files else d",
    "    mode = str(g('ENV_mode', 'field'))",
    "    env = (f\"k {float(g('NICHE_tradeoff', 1.0)):g}, contrast {float(g('HABITAT_contrast', 1.0)):g}\" if mode == 'niche'",
    "           else f\"field var {float(g('ENV_var_r', -1)):g}\")",
    "    return f\"{mode}, {env}, founded in {g('seed_size', '?')}, kernel {g('DISPERSAL_KERNEL', '?')}, window {g('window_steps', '?')}\"",
    "",
    "",
    "def setting_verdict(rows, b):",
    "    \"\"\"rows: list of band_values of the worlds of one setting. PASS per number: median inside and 75% of worlds inside.\"\"\"",
    "    need = int(np.ceil(0.75 * len(rows))); out = {}",
    "    for k in BANDS:",
    "        med = dict(rows[0]); med[k] = np.median(np.array([r[k] for r in rows]), axis=0)",
    "        n_in = sum(band_check(r, b)[k] for r in rows)",
    "        out[k] = (band_check(med, b)[k] and n_in >= need, n_in, med[k])",
    "    return out",
    "",
    ""]))
i = find(lambda s: s.strip().startswith("print(f\"  score against BNM: rare co-occupancy / BNM at 1-4 cells {np.round(co, 2).tolist()} \""), "old score print")
if not L[i + 2].strip().startswith("f\"species per window x{rr:.2f} {'PASS' if ok['species'] else 'FAIL'}\")"):
    sys.exit("STOP: old score print is not three lines; nothing written")
edits.append((i, i + 3, [
    f"                v = band_values(s, target); ck = band_check(v, target); rows.setdefault(setting_key(f), []).append(v)   {MARK}",
    "                print(f\"  against BNM: rare co-occupancy / BNM at 1-4 cells {np.round(co, 2).tolist()}; range-size distance {tv:.3f}; \"",
    "                      f\"species per window x{rr:.2f}; similarity decay {v['decay']:.2f} (BNM {target['sor'][0] - target['sor'][3]:.2f})\")",
    "                print(\"  bands (28 Oct): \" + \", \".join(f\"{BAND_NAMES[k]} {'PASS' if ck[k] else 'FAIL'}\" for k in BANDS)",
    "                      + f\"  ->  {sum(ck.values())} of {len(ck)} pass\")"]))
i = find(lambda s: s.strip() == "for pat in a.sim:", "sim loop")
edits.append((i, i, [f"rows = {{}}   {MARK}: band values per setting"]))
edits.append((len(L), len(L), [
    f"if rows and target is not None:   {MARK}",
    "    print(\"\\nsettings against the 28 Oct bands (PASS: median inside and at least 75% of worlds inside)\")",
    "    for key, rs in rows.items():",
    "        sv = setting_verdict(rs, target)",
    "        print(f\"  {key}: {len(rs)} worlds -> {'PASSES' if all(x[0] for x in sv.values()) else 'fails'} \"",
    "              f\"({sum(x[0] for x in sv.values())} of {len(sv)} numbers)\")",
    "        print(\"    \" + \"; \".join(f\"{BAND_NAMES[k]} {'PASS' if x[0] else 'FAIL'} ({x[1]}/{len(rs)})\" for k, x in sv.items()))"]))
i = find(lambda s: s.strip().startswith("import tempfile") and "obj1_v2 fit-half" in s, "self-test tempfile line")
edits.append((i, i, [
    f"    b0 = summarise(observe(rng.random((60, G, G)) < rng.uniform(0.01, 0.12, (60, 1, 1)), [np.ones((30, G, G), bool)]), wrap=False)   {MARK}: bands",
    "    v0 = band_values(b0, b0); ck = band_check(v0, b0)",
    "    good = all(ck.values()); ok &= good",
    "    print(f\"self-test, BNM against itself passes every band: {'PASS' if good else 'FAIL'}\")",
    "    v2 = dict(v0); v2['species'] = 2 * v0['species']; v3 = dict(v0); v3['r51'] = v0['r51'] + 0.06; v4 = dict(v0); v4['co'] = v0['co'] * 1.25",
    "    v5 = dict(v0); v5['co'] = v0['co'] * 0.7",
    "    c2, c3, c4, c5 = band_check(v2, b0), band_check(v3, b0), band_check(v4, b0), band_check(v5, b0)",
    "    good = all(not c[k] and sum(c.values()) == 7 for c, k in ((c2, 'species'), (c3, 'r51'), (c4, 'co'), (c5, 'co'))); ok &= good",
    "    print(f\"self-test, twice the species, 6 points more 51+ cases, rare clumping x1.25 or x0.7: each fails its own band only: {'PASS' if good else 'FAIL'}\")",
    "    sv3 = setting_verdict([v0, v0, v0, v2], b0); sv2 = setting_verdict([v0, v0, v2, v2], b0)",
    "    good = sv3['species'][0] and sv3['species'][1] == 3 and not sv2['species'][0] and all(x[0] for k, x in sv2.items() if k != 'species'); ok &= good",
    "    print(f\"self-test, a setting passes with 3 of 4 worlds inside and fails with 2 of 4: {'PASS' if good else 'FAIL'}\")"]))

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