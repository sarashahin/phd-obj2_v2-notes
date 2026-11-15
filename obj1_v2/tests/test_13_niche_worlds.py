"""
test_13_niche_worlds.py  (obj1_v2, stage 2d-a, 27 Oct 2026; any twin set, trade-off and habitat contrast printed, 28 Oct;
every file kept, habitat scale and connectance printed, 28 Oct; K3b added 29 Oct; per-species d and K7 added 29 Oct;
worlds with LVMCM founding read and labelled, 29 Oct)

Question: do niche worlds (plan niche) differ from their field-mode twins (same sim seed,
same dispersal) in the ways the niche model is meant to give, and does the shared habitat
carry information about far cells that records alone do not show?
Reads saved training files only. BNM range-size shares through BNM effort stay with
world_vs_bnm.py, final-run losses with test_05; this test does not repeat them.

K1 true range sizes (P_last_final): shares of species in 1, 2-5, 6-10, 11-20, 21-50, 51+
   cells, and species filling all cells; niche world beside its field twin. Description.
K2 habitat points to far cells (headroom, environment part). Per species with at least 2
   occupied and 2 empty cells: records = 10% of occupied cells (at least 1, 20 random
   draws); far cells = 3 or more cells (Chebyshev, torus) from every record; score of a
   far cell = -|H(cell) - mean H at the records| (the optimum guessed from records only);
   AUC of presence among far cells. Control: the same with the habitat shuffled over cells.
   PASS when the median AUC is at least 0.10 above the shuffled median and the shuffled
   median is within 0.45-0.55. Field twins: the species' own field as score, for reference
   only (no ecologist has it; the old model was given it).
K3 species live where their habitat fits: share of occupied cells with own r > 0 (90% or
   more), shuffled-habitat control; realised against fundamental range (cells with r > 0).
   K3 can separate only when the shuffled share is 90% or less; above that (contrast 0.25:
   kept species have r > 0 in 92-97% of all cells) K3 is printed with "no power" and K3b decides.
K3b species sit in their better cells (added 29 Oct, Sara's yes): per species with at least 2
   occupied and 2 empty cells, AUC of its own r, occupied against empty cells; control: its r
   shuffled over cells (10 shuffles). PASS when the median AUC is at least 0.10 above the
   shuffled median and the shuffled median is within 0.45-0.55 (the K2 rule). Printed as a
   description when K3 has power.
K4 who gets in: niche width of all arrivals, of species kept, of the final community.
K7 dispersal rate per species (worlds with SP_disp_rate, plan niche_d; description, the twin's K1 is
   the control): d of all arrivals, of species kept and of the final community (median); true range
   size by thirds of d in the final community; share of ranges of 20 cells or fewer in the lowest and
   highest third; Spearman(d, range). Expected (sketch 33): high-d species widespread, low-d species small.
K5 life table: newcomers kept (all, last 20% of attempts); stay of removed species in rounds
   and steps; share of the final community that arrived in the last 20% of attempts;
   share of the community at 50% and 80% of the run still there at the end.
K6 who took the cells of a removed species: in its removal round, the species newly present
   in most of its last cells; newcomer of that round, resident, or nobody.
Worlds with LVMCM founding (FOUND_rule "best", plan niche_b) have no founder and every tested candidate in
the life table. Newcomers are therefore the rows with LT_round_in >= 0 (the same rows as before in other
worlds); K4 adds the entered candidates; K5 counts kept among entered newcomers and reads attempts from
LT_attempt (other worlds: the id is the attempt).
Files: every file of --niche is read, also when two share a sim seed (each is labelled by its
settings). Twins are paired by sim seed; when --field has more than one file with that seed,
the twin lines are skipped with a note, so no world is compared with the wrong twin. Narrow
--field to one setting, e.g. "results/stage2a/niche_c/*hc0p25_*_training.npz".
Ecological reading: niche worlds should hold more small ranges than their twins, rare
species should sit on narrow niches or rare habitat, and the habitat should rank far cells
well above chance. K5 and K6 say how long species stay and who replaces them.

Run from obj1_v2/work:
    python3 ../tests/test_13_niche_worlds.py --niche "results/stage2a/niche/*_training.npz" \
        --field "results/stage2a/settle/*_training.npz" | tee ../logs/test_13_niche_worlds_<date>.txt
"""
import argparse, glob, os, sys
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--niche", required=True)
ap.add_argument("--field", default=None, help="twin worlds paired by sim seed: field-mode worlds, or niche worlds with other settings")
ap.add_argument("--draws", type=int, default=20); ap.add_argument("--frac", type=float, default=0.10)
a = ap.parse_args()
res = lambda c: "PASS" if c else "FAIL"
fails = []
BINS = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]


def load(pattern):
    """every file with P_last_final, as (sim seed, file name, data); none is dropped when seeds repeat."""
    out = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if "P_last_final" not in z.files:
            print(f"skipped (no P_last_final): {os.path.basename(f)}"); continue
        out.append((int(z["SIM_SEED"]), os.path.basename(f), z))
    return out


def setting(z):
    """settings that tell niche worlds apart, as printed beside a file."""
    parts = [(k, lab) for k, lab in (("NICHE_tradeoff", "trade-off k"), ("HABITAT_contrast", "habitat contrast"),
                                     ("HABITAT_ls", "habitat scale"), ("CONNECTANCE", "connectance")) if k in z.files]
    out = ", ".join(f"{lab} {float(z[k]):g}" for k, lab in parts)
    if "DISP_per_species" in z.files and int(z["DISP_per_species"]):
        out += f", d per species {float(z['DISP_d_min']):g}-{float(z['DISP_d_max']):g}"
    if "FOUND_rule" in z.files and str(z["FOUND_rule"][0]) == "best":
        out += f", best founding (cap {int(z['FOUND_cap'])})"
    return out


def far_mask(rec, G):
    """cells 3 or more (Chebyshev, torus) from every record."""
    near = np.zeros((G, G), bool)
    for y, x in zip(*np.nonzero(rec)):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                near[(y + dy) % G, (x + dx) % G] = True
    return ~near


def k3b(P, F, seed, shuffles=10):
    """AUC of each species' own r, occupied against empty cells, and the same with its r shuffled
    over cells (median of the shuffles); species with fewer than 2 occupied or 2 empty cells left out."""
    S, G, _ = P.shape
    rng = np.random.default_rng(seed + 20)
    real, ctl = [], []
    for s in range(S):
        lab, f = P[s].ravel(), F[s].ravel()
        if lab.sum() < 2 or (~lab).sum() < 2:
            continue
        real.append(auc(f, lab))
        ctl.append(np.median([auc(f[rng.permutation(G * G)], lab) for _ in range(shuffles)]))
    return np.array(real), np.array(ctl)


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg))


def far_auc(P, score_fn, rng, draws, frac):
    """median AUC over species and draws; score_fn(s, records) gives the score map of species s."""
    S, G, _ = P.shape
    out = []
    for s in range(S):
        occ = P[s].ravel().nonzero()[0]
        if len(occ) < 2 or (P[s] == 0).sum() < 2:
            continue
        k = max(1, int(np.ceil(frac * len(occ))))
        for _ in range(draws):
            rec = np.zeros(G * G, bool); rec[rng.choice(occ, k, replace=False)] = True; rec = rec.reshape(G, G)
            far = far_mask(rec, G)
            v = auc(score_fn(s, rec)[far], P[s][far] > 0)
            if np.isfinite(v):
                out.append(v)
    return np.array(out)


niche = load(a.niche)
field = {}
for sd, nm, zz in (load(a.field) if a.field else []):
    field.setdefault(sd, []).append((nm, zz))
if not niche:
    sys.exit(f"STOP: no niche files match {a.niche}")
nseed = {}
for sd, _, _ in niche:
    nseed[sd] = nseed.get(sd, 0) + 1
for idx, (seed, name, z) in enumerate(niche):
    if "ENV_habitat" not in z.files:
        sys.exit(f"STOP: {name} is not a niche world (no ENV_habitat)")
    lab = f"{seed}" if nseed[seed] == 1 else f"{seed}, file {idx + 1}"
    print(f"\nseed {seed}: {name[:90]}" + (f" ({setting(z)})" if setting(z) else ""))
    twins = field.get(seed, [])
    if len(twins) > 1:
        print(f"     note: {len(twins)} --field files share sim seed {seed}, twin lines skipped: " + "; ".join(f"{nm[:60]}... ({setting(tz)})" for nm, tz in twins))
    twin = twins[0][1] if len(twins) == 1 else None
    P = (np.asarray(z["P_last_final"]) > 0); S, G, _ = P.shape
    H = np.asarray(z["ENV_habitat"], dtype=float)
    F = np.asarray(z["ENV_r_field"], dtype=float) if "ENV_r_field" in z.files else None
    cells = P.reshape(S, -1).sum(axis=1)
    sh = [np.mean((cells >= lo) & (cells <= hi)) for lo, hi in BINS]
    line = f"  K1 niche: {S} species, median range {np.median(cells):.0f} cells; shares 1 / 2-5 / 6-10 / 11-20 / 21-50 / 51+: " + " ".join(f"{v:.0%}" for v in sh) + f"; all {G * G} cells {np.mean(cells == G * G):.0%}"
    print(line)
    if twin is not None:
        fz = twin; Pf = np.asarray(fz["P_last_final"]) > 0; cf = Pf.reshape(len(Pf), -1).sum(axis=1)
        tw = str(fz["ENV_mode"][0]) if "ENV_mode" in fz.files else "field"
        tw += f", {setting(fz)}" if setting(fz) else ""
        print(f"     twin ({tw}): {len(cf)} species, median range {np.median(cf):.0f} cells; shares " + " ".join(f"{np.mean((cf >= lo) & (cf <= hi)):.0%}" for lo, hi in BINS) + f"; all cells {np.mean(cf == G * G):.0%}")
    rng = np.random.default_rng(seed)
    score_H = lambda s, rec, Hm=H: -np.abs(Hm - Hm[rec].mean())
    A = far_auc(P, score_H, rng, a.draws, a.frac)
    perm = np.random.default_rng(seed + 1).permutation(G * G)
    Hs = H.ravel()[perm].reshape(G, G)
    As = far_auc(P, lambda s, rec: -np.abs(Hs - Hs[rec].mean()), np.random.default_rng(seed), a.draws, a.frac)
    small = np.array([c <= 20 for c in cells])
    ok = np.median(A) - np.median(As) >= 0.10 and 0.45 <= np.median(As) <= 0.55
    print(f"  K2 far cells ranked by habitat: median AUC {np.median(A):.3f} (IQR {np.percentile(A, 25):.2f}-{np.percentile(A, 75):.2f}, {len(A)} species x draws); "
          f"shuffled habitat {np.median(As):.3f} {res(ok)}"); fails.extend([] if ok else [f"K2[{lab}]"])
    if twin is not None and "ENV_r_field" in twin.files:
        Ff = np.asarray(twin["ENV_r_field"], dtype=float)
        Af = far_auc(Pf, lambda s, rec: Ff[s], np.random.default_rng(seed), a.draws, a.frac)
        print(f"     twin, its own growth-rate field as score (reference only): median AUC {np.median(Af):.3f} ({len(Af)})")
    if F is not None:
        occ_pos = (F[P] > 0).mean()
        ctrl = np.mean([(np.stack([F[s].ravel()[np.random.default_rng(seed + 10 + k).permutation(G * G)].reshape(G, G) for s in range(S)])[P] > 0).mean() for k in range(10)])
        fund = (F > 0).reshape(S, -1).sum(axis=1)
        rho = stats.spearmanr(cells, fund).correlation if S > 2 else np.nan
        ok3 = occ_pos >= 0.90 and ctrl < occ_pos - 0.1
        power = ctrl <= 0.90                          # K3 separates only when r > 0 in 90% or less of shuffled cells
        print(f"  K3 occupied cells with own r > 0: {occ_pos:.1%} (shuffled {ctrl:.1%}) "
              f"{res(ok3) if power else 'no power (shuffled above 90%), K3b decides'}; realised / fundamental median {np.median(cells / np.maximum(fund, 1)):.2f}; "
              f"Spearman(realised, fundamental) {rho:.2f}")
        b3, b3s = k3b(P, F, seed)
        if len(b3):
            ok3b = np.median(b3) - np.median(b3s) >= 0.10 and 0.45 <= np.median(b3s) <= 0.55
            print(f"  K3b occupied against empty cells ranked by own r: median AUC {np.median(b3):.3f} (IQR {np.percentile(b3, 25):.2f}-{np.percentile(b3, 75):.2f}, "
                  f"{len(b3)} species); shuffled r {np.median(b3s):.3f} " + (f"(description; K3 decides) {res(ok3b)}" if power else res(ok3b)))
        else:
            ok3b = False
            print("  K3b no species with 2 or more occupied and 2 or more empty cells: FAIL")
        fails.extend([] if (ok3 if power else ok3b) else [f"K3{'' if power else 'b'}[{lab}]"])
    lt_w, lt_est = np.asarray(z["LT_w"]), np.asarray(z["LT_established"]).astype(bool)
    best = "FOUND_rule" in z.files and str(z["FOUND_rule"][0]) == "best"
    new = np.asarray(z["LT_round_in"]) >= 0             # newcomers (the founder of a random-founding world has round -1)
    entd = np.asarray(z["LT_entered"]).astype(bool) if best else np.ones(len(lt_w), bool)
    print(f"  K4 niche width, median: all {'candidates' if best else 'arrivals'} {np.median(lt_w[new]):.2f}, "
          + (f"entered {np.median(lt_w[new & entd]):.2f}, " if best else "")
          + f"kept {np.median(lt_w[new & lt_est]):.2f}, final community {np.median(np.asarray(z['NICHE_w'])):.2f} "
          f"(width {float(z['NICHE_w_min']):.1f}-{float(z['NICHE_w_max']):.1f}); small ranges (20 cells or fewer) median width "
          f"{np.median(np.asarray(z['NICHE_w'])[small]) if small.any() else float('nan'):.2f}, |optimum| {np.median(np.abs(np.asarray(z['NICHE_opt'])[small])) if small.any() else float('nan'):.2f} "
          f"against {np.median(np.abs(np.asarray(z['NICHE_opt'])[~small])) if (~small).any() else float('nan'):.2f}")
    if "SP_disp_rate" in z.files:
        dsp, ltd = np.asarray(z["SP_disp_rate"], float), np.asarray(z["LT_d"], float)
        q = np.quantile(np.log10(dsp), [1 / 3, 2 / 3]) if len(dsp) >= 3 else None
        if q is not None:
            lo, hi = np.log10(dsp) <= q[0], np.log10(dsp) > q[1]; mid = ~lo & ~hi
            rho_d = stats.spearmanr(dsp, cells).correlation if len(dsp) > 2 else np.nan
            print(f"  K7 d, median: all arrivals {np.median(ltd[new]):.1e}, kept {np.median(ltd[new & lt_est]):.1e}, final community {np.median(dsp):.1e}; "
                  f"median true range by thirds of d (low / mid / high) {np.median(cells[lo]):.0f} / {np.median(cells[mid]) if mid.any() else float('nan'):.0f} / "
                  f"{np.median(cells[hi]):.0f} cells; ranges of 20 cells or fewer: lowest third {np.mean(cells[lo] <= 20):.0%}, highest {np.mean(cells[hi] <= 20):.0%}; "
                  f"Spearman(d, range) {rho_d:.2f}")
    ids, rin, rout = np.asarray(z["LT_id"]), np.asarray(z["LT_round_in"]), np.asarray(z["LT_round_out"])
    A_tot = int(z["attempts_total"])
    att = np.asarray(z["LT_attempt"]) if "LT_attempt" in z.files else ids   # random founding: the id is the attempt
    last = att > 0.8 * A_tot; newc = new & entd
    rounds = np.asarray(z["ASM_round"]); W = int(z["window_steps"])
    gone = lt_est & (rout >= 0)
    stay = rout[gone] - np.maximum(rin[gone], 0)
    fin = np.asarray(z["species_id"])
    att_of = dict(zip(ids.tolist(), att.tolist())); fin_att = np.array([att_of[int(i)] for i in fin])
    def members(k):
        o = np.asarray(z["ASM_P_offsets"]); return set(np.asarray(z["ASM_P_ids"])[o[k]:o[k + 1]].tolist())
    end = set(fin.tolist())
    still = [len(members(int(f * (len(rounds) - 1))) & end) / max(1, len(members(int(f * (len(rounds) - 1))))) for f in (0.5, 0.8)]
    print(f"  K5 kept: {lt_est[newc].mean():.0%} of {'entered newcomers' if best else 'newcomers'} ({lt_est[newc & last].mean():.0%} in the last 20% of attempts); removed species stayed a median "
          f"{np.median(stay) if len(stay) else float('nan'):.0f} rounds ({np.median(stay) * W if len(stay) else float('nan'):,.0f} steps, IQR {np.percentile(stay, 25) if len(stay) else float('nan'):.0f}-{np.percentile(stay, 75) if len(stay) else float('nan'):.0f} rounds, {len(stay)} removed); "
          f"final community arrived in the last 20% of attempts {np.mean(fin_att > 0.8 * A_tot):.0%}; still there at the end: {still[0]:.0%} of the community at 50% of rounds, {still[1]:.0%} at 80%")
    o = np.asarray(z["ASM_P_offsets"]); pid = np.asarray(z["ASM_P_ids"]); bits = np.asarray(z["ASM_P_bits"])
    pos = {int(r): k for k, r in enumerate(rounds)}
    def maps(k):
        return pid[o[k]:o[k + 1]], np.unpackbits(bits[o[k]:o[k + 1]], axis=1, count=G * G).astype(bool)
    kinds = {"newcomer": 0, "resident": 0, "nobody": 0}
    for i in ids[gone].tolist():
        ro = int(rout[ids == i][0])
        if ro not in pos or pos[ro] == 0:
            continue
        ib, Pb = maps(pos[ro] - 1); ia, Pa = maps(pos[ro])
        mine = Pb[list(ib).index(i)]
        before = {j: Pb[k] for k, j in enumerate(ib.tolist())}
        gain = {j: int((Pa[k] & mine & ~before.get(j, np.zeros(G * G, bool))).sum()) for k, j in enumerate(ia.tolist())}
        j = max(gain, key=gain.get) if gain else None
        if j is None or gain[j] == 0:
            kinds["nobody"] += 1
        else:
            kinds["newcomer" if int(rin[ids == j][0]) == ro else "resident"] += 1
    n = sum(kinds.values())
    print(f"  K6 cells of {n} removed species taken mostly by: a newcomer of that round {kinds['newcomer'] / max(n, 1):.0%}, "
          f"a resident {kinds['resident'] / max(n, 1):.0%}, nobody new {kinds['nobody'] / max(n, 1):.0%}")

print(f"\n{'all passed' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)