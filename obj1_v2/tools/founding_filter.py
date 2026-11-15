"""
founding_filter.py  (obj1_v2, stage 2d-b follow-up, 29 Oct 2026; v2 29 Oct 2026: worlds with LVMCM founding)

Question: does founding each newcomer in one random cell filter out the species whose habitat is
rare (the would-be rare species), before competition has any say?

For every species that arrived in a niche world (LT_id), its fundamental range F is the number of
cells where its growth rate r = peak exp(-(Hs - opt)^2 / (2 w^2)) - cost is above 0, with Hs the
habitat as the species see it (ENV_habitat, contrast x H) and its saved niche (LT_opt, LT_w, LT_peak,
NICHE_cost). A newcomer founded in one random cell can only start growing if that cell has r > 0,
so its chance of landing well is F / 400 (the "landing chance"). The tool reports, by F:
arrivals, newcomers kept, the kept share against the landing chance, and the final community.

v2: worlds with LVMCM founding (FOUND_rule "best", assembly v2.3) have no founder, and every tested
candidate is in the life table. For them "arrivals" are the tested candidates, and two columns are
added: entered (passed the growth test, founded in the best cell) and kept share of the entered.
Newcomers are the rows with LT_round_in >= 0 (the founder of a random-founding world has -1), which
gives the same rows as before for random-founding worlds.

Checks (of the reconstruction, not of the ecology):
  F1 the saved growth-rate fields of the final species equal the reconstruction (max |diff| < 1e-4),
     so F is computed on the habitat and niches the run used;
  F2 no newcomer with F = 0 (no cell where it can grow) was kept;
  F3 (best founding) no candidate with F = 0 entered: its g = r - competition is below 0 everywhere.
Ecological reading: if species with small F (20 cells or fewer: specialists of rare habitat) make up
a fair share of arrivals but are almost never kept, random founding is removing the would-be rare
species. LVMCM founds every invader in the cell where it grows best, after a growth test (sketch 34).

Run from obj1_v2/work:
    python3 ../tools/founding_filter.py "results/stage2a/niche_b/*_training.npz" "results/stage2a/niche_d/*_training.npz" \
        | tee ../logs/founding_filter_niche_b.txt
"""
import glob, os, sys
import numpy as np

BINS = [(0, 0, "0"), (1, 20, "1-20"), (21, 100, "21-100"), (101, 200, "101-200"), (201, 300, "201-300"), (301, 10**9, "301+")]


def is_best(z):
    return "FOUND_rule" in z.files and str(z["FOUND_rule"][0]) == "best"


def label(z):
    out = f"contrast {float(z['HABITAT_contrast']) if 'HABITAT_contrast' in z.files else 1.0:g}"
    out += f", connectance {float(z['CONNECTANCE']) if 'CONNECTANCE' in z.files else 0.4:g}"
    if "DISP_per_species" in z.files and int(z["DISP_per_species"]):
        out += f", d per species {float(z['DISP_d_min']):g}-{float(z['DISP_d_max']):g}"
    if is_best(z):
        out += f", best founding (cap {int(z['FOUND_cap'])})"
    return out


patterns = sys.argv[1:] or ["results/stage2a/niche_d/*_training.npz"]
files = [f for p in patterns for f in sorted(glob.glob(p))]
if not files:
    sys.exit(f"STOP: no files matched {patterns}")
ok, pooled = True, {}
for f in files:
    z = np.load(f, allow_pickle=True)
    if "LT_opt" not in z.files or "ENV_habitat" not in z.files:
        print(f"skipped (not a niche world with a life table): {os.path.basename(f)[:80]}"); continue
    best = is_best(z)
    Hs = np.asarray(z["ENV_habitat"], float); cost = float(z["NICHE_cost"]); ncell = Hs.size
    ids, kept = np.asarray(z["LT_id"]), np.asarray(z["LT_established"]).astype(bool)
    entered = np.asarray(z["LT_entered"]).astype(bool) if best else np.ones(len(ids), bool)
    opt, w, pk = (np.asarray(z[k], float) for k in ("LT_opt", "LT_w", "LT_peak"))
    r = pk[:, None, None] * np.exp(-(Hs[None] - opt[:, None, None]) ** 2 / (2 * w[:, None, None] ** 2)) - cost
    F = (r > 0).reshape(len(ids), -1).sum(1)
    pos = {int(i): k for k, i in enumerate(ids)}
    fin = np.array([pos[int(i)] for i in np.asarray(z["species_id"])], dtype=int)
    diff = float(np.abs(r[fin] - np.asarray(z["ENV_r_field"], float)).max()) if len(fin) else 0.0
    new = np.asarray(z["LT_round_in"]) >= 0              # newcomers; the founder of a random-founding world has round -1
    f1, f2 = diff < 1e-4, not bool((kept & new & (F == 0)).any())
    f3 = not bool((entered & new & (F == 0)).any()) if best else True
    ok &= f1 and f2 and f3
    print(f"\nseed {int(z['SIM_SEED'])} ({label(z)}): {new.sum()} {'candidates tested' if best else 'newcomers'}, "
          + (f"{entered[new].sum()} entered, " if best else "") + f"{kept[new].sum()} kept, {len(fin)} in the final community")
    print(f"  F1 final fields = reconstruction (max |diff| {diff:.1e}) {'PASS' if f1 else 'FAIL'}; "
          f"F2 no newcomer without any suitable cell was kept {'PASS' if f2 else 'FAIL'}"
          + (f"; F3 no candidate without any suitable cell entered {'PASS' if f3 else 'FAIL'}" if best else ""))
    if best:
        print("  fundamental range F (cells)   tested   entered   entered share   kept   kept share   kept of entered   final community")
    else:
        print("  fundamental range F (cells)   arrivals   kept   kept share   landing chance F/400   final community")
    for lo, hi, lab in BINS:
        m = new & (F >= lo) & (F <= hi)
        mf = (F[fin] >= lo) & (F[fin] <= hi)
        nan = float("nan")
        if best:
            me = m & entered
            print(f"  {lab:>27}   {m.sum():6d}   {me.sum():7d}   {entered[m].mean() if m.any() else nan:13.1%}   {kept[m].sum():4d}   "
                  f"{kept[m].mean() if m.any() else nan:10.1%}   {kept[me].mean() if me.any() else nan:15.1%}   "
                  f"{mf.sum():3d} ({mf.mean() if len(fin) else nan:.0%})")
        else:
            print(f"  {lab:>27}   {m.sum():8d}   {kept[m].sum():4d}   {kept[m].mean() if m.any() else nan:10.1%}   "
                  f"{(F[m] / ncell).mean() if m.any() else nan:20.1%}   {mf.sum():3d} ({mf.mean() if len(fin) else nan:.0%})")
        acc = pooled.setdefault(label(z), {}).setdefault(lab, [0, 0, 0.0, 0, 0])
        acc[0] += int(m.sum()); acc[1] += int(kept[m].sum()); acc[2] += float((F[m] / ncell).sum()); acc[3] += int(mf.sum())
        acc[4] += int((m & entered).sum())
for lab_s, rows in pooled.items():
    n_all = sum(v[0] for v in rows.values()); f_all = sum(v[3] for v in rows.values())
    if "best founding" in lab_s:
        print(f"\npooled, {lab_s}: candidates by F, entered share, kept share (kept of entered)")
        print("  " + "; ".join(f"F {k}: {v[0] / max(n_all, 1):.0%} of candidates, entered {v[4] / max(v[0], 1):.1%}, kept {v[1] / max(v[0], 1):.1%} "
                               f"({v[1] / max(v[4], 1):.1%} of entered), {v[3] / max(f_all, 1):.0%} of final" for k, v in rows.items()))
    else:
        print(f"\npooled, {lab_s}: arrivals by F and kept share (landing chance)")
        print("  " + "; ".join(f"F {k}: {v[0] / max(n_all, 1):.0%} of arrivals, kept {v[1] / max(v[0], 1):.1%} ({v[2] / max(v[0], 1):.1%}), "
                               f"{v[3] / max(f_all, 1):.0%} of final" for k, v in rows.items()))
print(f"\n{'all passed' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)