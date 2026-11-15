"""
patch_world_vs_bnm_alpha.py  (obj1_v2, 28 Oct 2026)

world_vs_bnm v4: where do the extra species of the niche worlds sit? Adds, for BNM windows and
for simulated worlds seen through BNM effort (same windows, same masks):
  alpha     species recorded per cell (cells with at least one record), median and mean
  gamma     species recorded in the window (as before: species per window)
  beta      gamma / mean alpha of the window (Whittaker turnover); geometric means over
            windows, so gamma = alpha x beta holds exactly for them
  Sorensen  mean similarity of the species lists of two recorded cells, by Chebyshev distance
            1, 2, 3-4, 5-10 (high = cells alike, steep fall = patchy composition)
and, with --bnm, splits the species excess: gamma / BNM = (alpha / BNM) x (beta / BNM).
Reading rule (set before any result, 28 Oct): if the per-cell part carries more than half of
log(gamma / BNM), the excess is local (competition too weak); otherwise it is turnover between
cells (habitat too patchy, or dispersal too short). With an excess (gamma / BNM > 1) this is the
same as asking which factor is larger, per cell or turnover. Description only, no pass/fail.

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_alpha.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_alpha.py --apply
    python3 ../tools/world_vs_bnm.py --self-test            # 13 old checks + 6 new
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 alpha"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already has the alpha-beta report")
if "# obj1_v2 fit-half" not in src:
    sys.exit("STOP: apply patch_world_vs_bnm_fithalf.py first (world_vs_bnm v3)")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


edits = []
i = find(lambda s: s.strip() == "windows sharing cells with them are never opened. Pass bands are to be set again from the fit half.", "v3 docstring end")
edits.append((i + 1, i + 1, [
    "Version 4 (28 Oct): species per recorded cell (alpha), turnover gamma / mean alpha (beta), Sorensen",
    "  similarity of cells by distance, and the species excess split into a per-cell and a turnover part",
    "  (geometric means over windows, so gamma = alpha x beta exactly). Description only."]))
i = find(lambda s: s.strip() == "def load_bnm(pattern):", "def load_bnm")
edits.append((i, i, [
    f"SOR_DIST = [(1, 1, \"1\"), (2, 2, \"2\"), (3, 4, \"3-4\"), (5, 10, \"5-10\")]   {MARK}",
    "",
    "",
    "def alpha_beta(cases, wrap):",
    "    \"\"\"Species per recorded cell (alpha), gamma / mean alpha per window (beta), and the mean Sorensen",
    "    similarity of two recorded cells by Chebyshev distance, pooled over windows.\"\"\"",
    "    alphas, gam, abar, sor_sum, sor_n = [], [], [], np.zeros(len(SOR_DIST)), np.zeros(len(SOR_DIST))",
    "    for o, _ in cases:",
    "        S, G0, G1 = o.shape",
    "        X = o.reshape(S, -1).T.astype(float)                  # cells x species",
    "        n = X.sum(1); idx = np.flatnonzero(n > 0)",
    "        g = int((o.reshape(S, -1).sum(1) > 0).sum())",
    "        if g == 0 or len(idx) == 0:",
    "            continue",
    "        alphas.append(n[idx]); gam.append(g); abar.append(n[idx].mean())",
    "        Xi = X[idx]; inter = Xi @ Xi.T; sor = 2.0 * inter / (n[idx][:, None] + n[idx][None, :])",
    "        yy, xx = np.divmod(idx, G1)",
    "        dy, dx = np.abs(yy[:, None] - yy[None, :]), np.abs(xx[:, None] - xx[None, :])",
    "        if wrap:",
    "            dy, dx = np.minimum(dy, G0 - dy), np.minimum(dx, G1 - dx)",
    "        d = np.maximum(dy, dx); up = np.triu(np.ones_like(d, dtype=bool), 1)",
    "        for j, (lo, hi, _) in enumerate(SOR_DIST):",
    "            m = up & (d >= lo) & (d <= hi)",
    "            sor_sum[j] += sor[m].sum(); sor_n[j] += m.sum()",
    "    if not gam:",
    "        return dict(alpha_med=np.nan, alpha_mean=np.nan, alpha_gm=np.nan, beta_gm=np.nan, gamma_gm=np.nan,",
    "                    sor=np.full(len(SOR_DIST), np.nan))",
    "    a = np.concatenate(alphas); gam, abar = np.array(gam, float), np.array(abar, float)",
    "    return dict(alpha_med=float(np.median(a)), alpha_mean=float(a.mean()),",
    "                alpha_gm=float(np.exp(np.log(abar).mean())), beta_gm=float(np.exp(np.log(gam / abar).mean())),",
    "                gamma_gm=float(np.exp(np.log(gam).mean())),",
    "                sor=np.where(sor_n > 0, sor_sum / np.maximum(sor_n, 1), np.nan))",
    "",
    ""]))
i = find(lambda s: s.strip().startswith("return dict(n_units=len(cases), n_cases=len(R),"), "summarise return")
edits.append((i, i, [f"    ab = alpha_beta(cases, wrap)   {MARK}"]))
i = find(lambda s: s.strip() == "ratio_sparse=np.where(ex2 > 0, ob2 / np.maximum(ex2, 1e-12), np.nan))", "summarise return end")
edits.append((i, i + 1, [L[i].replace("np.nan))", "np.nan), **ab)") + f"   {MARK}"]))
i = find(lambda s: s.strip() == "\", \".join(f\"{lab} {v:.2f}\" for (_, _, lab), v in zip(DIST, s[\"ratio_sparse\"])))", "report last line")
edits.append((i + 1, i + 1, [
    f"    print(f\"  species per recorded cell (alpha): median {{s['alpha_med']:.0f}}, mean {{s['alpha_mean']:.1f}}; \"   {MARK}",
    "          f\"geometric means over windows: gamma {s['gamma_gm']:.1f} = alpha {s['alpha_gm']:.1f} x turnover (beta) {s['beta_gm']:.2f}\")",
    "    print(\"  similarity of two recorded cells (Sorensen) by distance: \" +",
    "          \", \".join(f\"{lab} {v:.2f}\" for (_, _, lab), v in zip(SOR_DIST, s[\"sor\"])))"]))
i = find(lambda s: s.strip() == "f\"species per window x{rr:.2f} {'PASS' if ok['species'] else 'FAIL'}\")", "score print end")
p = L[i][: len(L[i]) - len(L[i].lstrip())]
edits.append((i + 1, i + 1, [
    f"                ga, aa, bb = s['gamma_gm'] / target['gamma_gm'], s['alpha_gm'] / target['alpha_gm'], s['beta_gm'] / target['beta_gm']   {MARK}",
    "                why = ('no species excess' if ga <= 1 else 'mostly per cell: local competition' if aa > bb",
    "                       else 'mostly turnover between cells: habitat or dispersal')",
    "                print(f\"  species against BNM (geometric means) x{ga:.2f} = per cell x{aa:.2f} x turnover x{bb:.2f} ({why})\")"]))
i = find(lambda s: s.strip().startswith("import tempfile") and "obj1_v2 fit-half" in s, "self-test tempfile line")
edits.append((i, i, [
    f"    full = np.ones((5, G, G), bool)   {MARK}: alpha-beta known answers",
    "    ab = alpha_beta([(full, full)], wrap=False)",
    "    good = ab['alpha_med'] == 5 and abs(ab['beta_gm'] - 1) < 1e-12 and np.allclose(ab['sor'], 1); ok &= good",
    "    print(f\"self-test, 5 species in every cell: alpha 5, turnover 1, similarity 1 at every distance: {'PASS' if good else 'FAIL'}\")",
    "    quad = np.zeros((4, G, G), bool); quad[0, :10, :10] = quad[1, :10, 10:] = quad[2, 10:, :10] = quad[3, 10:, 10:] = True",
    "    ab = alpha_beta([(quad, quad)], wrap=False)",
    "    good = ab['alpha_med'] == 1 and abs(ab['gamma_gm'] - 4) < 1e-12 and abs(ab['beta_gm'] - 4) < 1e-12; ok &= good",
    "    print(f\"self-test, 4 species in 4 quadrants: alpha 1, gamma 4, turnover 4: {'PASS' if good else 'FAIL'}\")",
    "    rw = [(rng.random((8, G, G)) < 0.1, None) for _ in range(2)]",
    "    for wrap in (False, True):",
    "        ab = alpha_beta(rw, wrap)",
    "        num, den = np.zeros(len(SOR_DIST)), np.zeros(len(SOR_DIST))",
    "        for o, _ in rw:",
    "            cells = [(y, x) for y in range(G) for x in range(G) if o[:, y, x].any()]",
    "            for a_ in range(len(cells)):",
    "                for b_ in range(a_ + 1, len(cells)):",
    "                    (y1, x1), (y2, x2) = cells[a_], cells[b_]",
    "                    dy, dx = abs(y1 - y2), abs(x1 - x2)",
    "                    if wrap: dy, dx = min(dy, G - dy), min(dx, G - dx)",
    "                    dd = max(dy, dx); A, B = o[:, y1, x1], o[:, y2, x2]",
    "                    for j, (lo, hi, _) in enumerate(SOR_DIST):",
    "                        if lo <= dd <= hi: num[j] += 2 * (A & B).sum() / (A.sum() + B.sum()); den[j] += 1",
    "        good = np.allclose(ab['sor'], num / den); ok &= good",
    "        print(f\"self-test, Sorensen by distance against a direct loop over cell pairs ({'torus' if wrap else 'window'}): {'PASS' if good else 'FAIL'}\")",
    "    good = abs(ab['gamma_gm'] - ab['alpha_gm'] * ab['beta_gm']) < 1e-9 * ab['gamma_gm']; ok &= good",
    "    print(f\"self-test, gamma = alpha x turnover for the geometric means: {'PASS' if good else 'FAIL'}\")",
    "    Pz = rng.random((40, G, G)) < 0.1",
    "    a1 = alpha_beta(observe(Pz, [np.ones((30, G, G), bool)]), wrap=False); a2 = alpha_beta([(Pz, np.ones_like(Pz))], wrap=False)",
    "    good = a1['alpha_med'] == a2['alpha_med'] and np.allclose(a1['sor'], a2['sor']) and a1['beta_gm'] == a2['beta_gm']; ok &= good",
    "    print(f\"self-test, a survey with every cell known leaves alpha, turnover and similarity unchanged: {'PASS' if good else 'FAIL'}\")"]))

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