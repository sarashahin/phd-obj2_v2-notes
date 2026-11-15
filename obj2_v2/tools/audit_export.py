#!/usr/bin/env python3
"""
audit_export.py  (obj2_v2, gate between stage C and stage D, 8 Nov 2026; read-only; version 2 the same day: L6 counts
only maps with enough arrangements to be copied, and lists every match with its real_train twin)

Question: before any training, do the exported files (export_obj2.py) hold any leak of the truth into the inputs, any
copying between splits, or any bias that would hide the rare, sparsely recorded species the model is for?
It reads ../data/real_train and ../data/real_val only (real_eval refused) and the BNM fit-half per-year grids.

Leak and copy checks (PASS or FAIL; rules written before the run, 8 Nov):
  L1 input files hold exactly the allow-list keys, with the expected types and shapes; each input row pairs with its
     truth row (pair md5, row counts, round of every row)
  L2 every record lies inside the true map and the surveyed cells of its year
  L3 recorded share per occupied surveyed cell-year = d (0.80 and 0.64, within 0.005); both visits = d1^2 (visits drawn
     apart, neither copied from the other)
  L4 effort carries no truth: P(surveyed | occupied) / P(surveyed | empty) inside the spread of the same ratio after the
     truth is shifted on the torus (10 shifts; real within 4 sd of the shift mean, or 0.01); Spearman between surveyed
     cells and true cells per stack within 0.05 of 0
  L5 the habitat input is noisy: median correlation of noisy and exact map within 0.02 of 1/sqrt(1 + sigma^2), no map
     above 0.99, median noise sd / map sd within 0.02 of sigma; K2 recomputed from the stored real_val maps within 0.01
     of the calibrate value and inside 0.78-0.82
  L6 no copying between splits: no community seed or source file in both; no real_val stack with 11-389 true cells at
     the target year whose 6 true maps equal a real_train stack; no shared community-round; no real_val input stack
     (5 or more records) equal to a real_train input stack. Version 1 counted every stack of 11+ cells and failed on
     Sara's run (14 matches, 8 Nov): a map with nearly all 400 cells occupied has as few arrangements as a map with
     nearly none, so identical maps there do not show copying. Matches with 1-10 or 390-400 cells are listed for
     reference, each with its real_train twin and cells per year
  L7 the same code (one export md5) and sigma in every file; years 2014-2019 consecutive, pieces 19-24; every survey
     window is a BNM fit-half window
  L8 records per year in n_rec equal the visits; stacks with no record can be found from the input alone
  L9 (FLAG only) row order inside a round against true range size: if Spearman is above 0.05, the stage E loader must
     shuffle rows and never use their position (rows are written in species-id order)

Bias and rare-species checks (reported, with the rule that raises a FLAG):
  B1 recorded class at the target year against the BNM fit half (shares recomputed from the per-year grids, and the
     constants of export_obj2 beside them); stage D sampler weight per class = BNM share / training share; FLAG if a
     weight is above 5 or a recomputed BNM share differs from the constant by more than 1 point
  B2 recorded class against true class (how many sparsely recorded stacks are truly rare)
  B3 truly rare stacks (true 1-20 cells at the target year): share with a record, records over 6 years, distinct
     species with a record; FLAG if a rare class has fewer than 500 distinct species with a record
  B4 real_train against real_val recorded shares; FLAG if any class differs by more than 1 point
  B5 recorded cells over 6 years per stack (bins 1-3, 4-6, 7-10, 11-20, 21+) against BNM fit-half species-windows
  B6 the same species with the same 6 true maps in more than one round of a community (repeats, not new examples)

Run (from PSD_Dispersal_pool/obj2_v2/work):
    python3 ../tools/audit_export.py --self-test | tail -1
    python3 ../tools/audit_export.py --train ../data/real_train --val ../data/real_val | tee ../logs/audit_export_8nov.txt
Needs numpy and export_obj2.py in the same folder.
"""
import argparse, csv, hashlib, importlib.util, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load_export():
    p = HERE / "export_obj2.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; audit_export.py needs export_obj2.py in the same folder")
    spec = importlib.util.spec_from_file_location("export_obj2", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, m.file_md5(p)


X, XMD5 = _load_export()
G, NC, WINDOW = X.G, X.NC, X.WINDOW
REC = ["1", "2-5", "6-10", "11-20", "21-50", "51+"]
TRUE = ["0"] + REC
RARE = ["1", "2-5", "6-10", "11-20"]
BINS = [(1, 3, "1-3"), (4, 6, "4-6"), (7, 10, "7-10"), (11, 20, "11-20"), (21, 10 ** 6, "21+")]
N_NULL = 10
INFO_LO, INFO_HI = 11, NC - 11          # L6 copy rule: maps with at least 11 occupied and 11 empty cells at the target year


def rankdata(x):
    _, inv, cnt = np.unique(np.asarray(x), return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return ((cum - cnt + 1 + cum) / 2.0)[inv]


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def bin_of(n):
    return next(lab for lo, hi, lab in BINS if lo <= n <= hi)


EDGES = np.array([lo for _, lo, _ in X.CLASSES])


def labels(x):
    """Class label of every count in x (0, 1, 2-5, 6-10, 11-20, 21-50, 51+), vectorised X.cls."""
    return np.array(TRUE, dtype=object)[np.searchsorted(EDGES, np.asarray(x), side="right") - 1]


def rows_of(d):
    man = Path(d) / "manifest.csv"
    if not man.is_file():
        sys.exit(f"STOP: {man} not found")
    with open(man) as f:
        return list(csv.DictReader(f))


def load_pair(d, row):
    """Both files read once into memory (an npz key read again would be decompressed again)."""
    with np.load(Path(d) / row["truth_file"], allow_pickle=False) as z:
        ti = {k: z[k] for k in z.files}
    with np.load(Path(d) / row["input_file"], allow_pickle=False) as z:
        ii = {k: z[k] for k in z.files}
    return ti, ii


def bnm_side(peryear, split_ref, tools, minlist=8):
    """Fit-half window names; BNM recorded class shares at the target year and 6-year record bins (species with 1+
    record in a window)."""
    bdc, _ = X.import_bdc(tools)
    pooled = Path(peryear) / f"minlist{minlist}"
    ydirs = sorted(p for p in Path(peryear).glob(f"minlist{minlist}_y*") if p.is_dir())
    names = sorted(f.name for f in pooled.glob("bnm_*.npz"))
    keep, _, _ = bdc.fit_half(names, bdc.split_ref(split_ref))
    cls_t, bins6 = {k: 0 for k in REC}, {lab: 0 for _, _, lab in BINS}
    for nm in keep:
        a = X.bnm_load(bdc, pooled / nm)
        O = [np.asarray(a["obs_mask"]).astype(bool)] + [np.asarray(X.bnm_load(bdc, d / nm)["present_year"]).astype(bool) for d in ydirs]
        nt = O[0].reshape(len(O[0]), -1).sum(1)
        n6 = np.any(np.stack(O), 0).reshape(len(O[0]), -1).sum(1)
        for x in nt[nt > 0]:
            cls_t[X.cls(int(x))] += 1
        for x in n6[n6 > 0]:
            bins6[bin_of(int(x))] += 1
    return set(keep), cls_t, bins6


def k2_stored(d, n_comm, n_rounds, draws):
    """K2 as export_obj2.calibrate, with the noisy maps read back from the input files (same communities, rounds,
    species and record draws as calibrate, so the value must equal calibrate's up to float16 rounding)."""
    rows = rows_of(d)
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, min(n_comm, len(rows)))).astype(int))
    vals = []
    for i in idx:
        ti, ii = load_pair(d, rows[i])
        seed = int(ti["sim_seed"]); rids = ti["round_ids"]; st_r = ti["stack_round"]; st_s = ti["stack_species"]
        T = X.unpack(ti["true_maps"][:, -1])
        Hn_all = ii["habitat_noisy"].astype(np.float64)
        pick = np.unique(np.round(np.linspace(0, len(rids) - 1, min(n_rounds, len(rids)))).astype(int)) if len(rids) else []
        for k in pick:
            r = int(rids[k]); Hn = Hn_all[k]
            for j in np.nonzero(st_r == r)[0]:
                P = T[j]; occ = P.ravel().nonzero()[0]
                if len(occ) < 2 or (~P).sum() < 2:
                    continue
                kk = max(1, int(np.ceil(0.10 * len(occ))))
                g = X.rng(seed, r, int(st_s[j]), 6)
                for _ in range(draws):
                    rec = np.zeros(NC, bool); rec[g.choice(occ, kk, replace=False)] = True; rec = rec.reshape(G, G)
                    far = X.far_mask(rec)
                    if P[far].all() or not P[far].any():
                        continue
                    vals.append(X.auc(-np.abs(Hn - Hn[rec].mean())[far], P[far]))
    v = np.array(vals)
    return float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan, len(v)


class Split:
    def __init__(self):
        self.l1, self.l7 = [], []
        self.l2 = 0; self.l8 = 0; self.zero = 0; self.n = 0
        self.den = {X.D_BIG: 0, X.D_SMALL: 0}; self.num = {X.D_BIG: np.zeros(3), X.D_SMALL: np.zeros(3)}
        self.A = self.B = self.C = self.D = 0.0; self.A0 = np.zeros(N_NULL); self.B0 = np.zeros(N_NULL)
        self.eff, self.tru, self.corr, self.sdr = [], [], [], []
        self.l9x = []; self.l9y = []
        self.rec_t, self.rec6, self.true_t = [], [], []
        self.distinct = {k: set() for k in RARE}
        self.dup_tot = {k: 0 for k in TRUE}; self.dup_distinct = {k: 0 for k in TRUE}
        self.code, self.sigma, self.seeds, self.srcs = set(), set(), set(), set()
        self.stack_h, self.round_h, self.input_h = set(), set(), set()
        self.hit_stack = 0; self.hit_small = {k: 0 for k in ["1", "2-5", "6-10"]}; self.hit_round = 0; self.hit_input = 0
        self.round_dup_within = 0
        self.stack_h_small, self.stack_h_full = set(), set(); self.hit_full = 0; self.hits = []
        self.full6 = 0; self.full_t = 0


def scan_split(d, fit_names, ref=None):
    S = Split()
    for row in rows_of(d):
        ti, ii = load_pair(d, row)
        name = row["truth_file"]
        seed = int(ti["sim_seed"]); S.seeds.add(seed); S.srcs.add(row["source_md5"])
        n = len(ti["stack_round"]); nr = len(ti["round_ids"]); S.n += n
        # L1
        keys = set(ii)
        if keys != set(X.ALLOW):
            S.l1.append(f"{row['input_file']}: keys {sorted(keys ^ set(X.ALLOW))} differ from the allow-list")
        want = {"effort": ((n, WINDOW, NC // 8), np.uint8), "visit1": ((n, WINDOW, NC // 8), np.uint8),
                "visit2": ((n, WINDOW, NC // 8), np.uint8), "n_rec": ((n, WINDOW), np.int16),
                "habitat_noisy": ((nr, G, G), np.float16), "stack_hab": ((n,), np.int32)}
        for k, (sh, dt) in want.items():
            if k in ii and (ii[k].shape != sh or ii[k].dtype != dt):
                S.l1.append(f"{row['input_file']}: {k} is {ii[k].dtype} {ii[k].shape}, expected {np.dtype(dt)} {sh}")
        if str(ti["pair_md5"]) != str(ii["pair_md5"]):
            S.l1.append(f"{name}: pair md5 differs from its input file")
        sh_ = ii["stack_hab"] if "stack_hab" in ii else np.zeros(0, int)
        if len(sh_) != n or (n and (sh_.min() < 0 or sh_.max() >= nr or not np.array_equal(ti["round_ids"][sh_], ti["stack_round"]))):
            S.l1.append(f"{name}: input rows do not point to their own rounds")
        if any(k not in ii for k in ("effort", "visit1", "visit2", "n_rec", "habitat_noisy")):
            continue
        T = X.unpack(ti["true_maps"]); E = X.unpack(ii["effort"]); V1 = X.unpack(ii["visit1"]); V2 = X.unpack(ii["visit2"])
        V = V1 | V2
        # L2
        S.l2 += int((V & ~T).any((1, 2, 3)).sum() + (V & ~E).any((1, 2, 3)).sum())
        # L3
        d_ = ti["stack_d"]; PE = T & E
        for dv in (X.D_BIG, X.D_SMALL):
            k = np.isclose(d_, dv)
            pe = PE[k]
            S.den[dv] += int(pe.sum())
            S.num[dv] += [(V1[k] & pe).sum(), (V1[k] & V2[k] & pe).sum(), (V[k] & pe).sum()]
        # L4
        S.A += float((E & T).sum()); S.B += float((E & ~T).sum()); S.C += float(T.sum()); S.D += float((~T).sum())
        g = np.random.default_rng([7, seed])
        for q in range(N_NULL):
            dy, dx = (int(v) for v in g.integers(1, G, size=2))
            Ts = np.roll(np.roll(T, dy, 2), dx, 3)
            S.A0[q] += float((E & Ts).sum()); S.B0[q] += float((E & ~Ts).sum())
        S.eff.append(E.any(1).reshape(n, -1).sum(1)); S.tru.append(ti["true_cells"].astype(int))
        # L5
        He = ti["habitat_exact"].astype(np.float64); Hn = ii["habitat_noisy"].astype(np.float64)
        for a, b in zip(He, Hn):
            if a.std() > 0:
                S.corr.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1])); S.sdr.append(float((b - a).std() / a.std()))
        # L7
        S.code.add(str(ti["code_md5"])); S.code.add(str(ii["code_md5"])); S.sigma.add(float(ti["habitat_sigma"]))
        yrs, pcs = list(ti["years"]), list(ti["pieces"])
        if len(yrs) != WINDOW or yrs != list(range(yrs[0], yrs[0] + WINDOW)):
            S.l7.append(f"{name}: years {yrs}")
        if pcs != list(range(pcs[0], pcs[0] + WINDOW)) or pcs[-1] != 24:
            S.l7.append(f"{name}: pieces {pcs}")
        wn = [str(w) for w in ti["window_names"]]
        bad_w = sorted({wn[int(w)] for w in ti["round_window"]} - fit_names)
        if bad_w:
            S.l7.append(f"{name}: survey windows outside the fit half {bad_w[:3]}")
        # L8
        cnt = V.reshape(n, WINDOW, -1).sum(2)
        S.l8 += int((cnt != ii["n_rec"]).any(1).sum())
        S.zero += int((ii["n_rec"].sum(1) == 0).sum())
        # L9, B, L6, B6
        st_r, st_s, tc = ti["stack_round"], ti["stack_species"], ti["true_cells"].astype(int)
        rec_t = V[:, -1].reshape(n, -1).sum(1); rec6 = V.any(1).reshape(n, -1).sum(1)
        S.rec_t.append(rec_t); S.rec6.append(rec6); S.true_t.append(tc)
        seen_c = {}
        tb = ti["true_maps"]
        for r in np.unique(st_r):
            idx = np.nonzero(st_r == r)[0]
            m = len(idx)
            if m >= 3:
                S.l9x.append((np.arange(m) + 0.5) / m); S.l9y.append((rankdata(tc[idx]) - 0.5) / m)
            fp = hashlib.md5(b"".join(sorted(tb[j][-1].tobytes() for j in idx))).digest()
            if ref is not None and fp in ref.round_h:
                S.hit_round += 1
            if fp in S.round_h:
                S.round_dup_within += 1
            S.round_h.add(fp)
        for j in range(n):
            c = X.cls(int(tc[j]))
            h = hashlib.md5(tb[j].tobytes()).digest()
            key = (int(st_s[j]), h)
            S.dup_tot[c] += 1
            if key not in seen_c:
                seen_c[key] = 1; S.dup_distinct[c] += 1
            if rec6[j] > 0 and c in S.distinct:
                S.distinct[c].add((seed, int(st_s[j])))
            if INFO_LO <= tc[j] <= INFO_HI:
                if ref is not None and h in ref.stack_h:
                    S.hit_stack += 1; S.hits.append(("informative", h, seed, int(st_r[j]), int(st_s[j]), cells_year(tb[j])))
                S.stack_h.add(h)
            elif tc[j] > INFO_HI:
                if ref is not None and h in ref.stack_h_full:
                    S.hit_full += 1; S.hits.append(("390-400 cells", h, seed, int(st_r[j]), int(st_s[j]), cells_year(tb[j])))
                S.stack_h_full.add(h)
            elif ref is not None and c in S.hit_small and h in ref.stack_h_small:
                S.hit_small[c] += 1
            if rec6[j] >= 5:
                hi = hashlib.md5(ii["effort"][j].tobytes() + ii["visit1"][j].tobytes() + ii["visit2"][j].tobytes()).digest()
                if ref is not None and hi in ref.input_h:
                    S.hit_input += 1
                S.input_h.add(hi)
        if ref is None:
            for j in np.nonzero((tc >= 1) & (tc < INFO_LO))[0]:
                S.stack_h_small.add(hashlib.md5(tb[j].tobytes()).digest())
        per_year = np.unpackbits(tb, axis=-1, count=NC).sum(-1)
        S.full6 += int((per_year == NC).all(1).sum()); S.full_t += int((tc > INFO_HI).sum())
    for k in ("eff", "tru", "rec_t", "rec6", "true_t"):
        setattr(S, k, np.concatenate(getattr(S, k)) if getattr(S, k) else np.zeros(0, int))
    return S


def cells_year(packed):
    return [int(v) for v in np.unpackbits(packed, axis=-1, count=NC).sum(-1)]


def locate(d, hashes, limit=3):
    """Where each hash occurs in a split: {hash: [(seed, round, species), ...]} (truth files only, first `limit`)."""
    found = {h: [] for h in hashes}
    if not hashes:
        return found
    for row in rows_of(d):
        with np.load(Path(d) / row["truth_file"], allow_pickle=False) as z:
            tb, st_r, st_s, seed = z["true_maps"], z["stack_round"], z["stack_species"], int(z["sim_seed"])
        for j in range(len(tb)):
            h = hashlib.md5(tb[j].tobytes()).digest()
            if h in found and len(found[h]) < limit:
                found[h].append((seed, int(st_r[j]), int(st_s[j])))
    return found


def audit(train_dir, val_dir, sigma_file, peryear, split_ref, tools, n_comm=20, n_rounds=10, draws=5, out=print):
    res, flags = {}, []

    def check(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    def flag(name, raised, text):
        out(f"{name} {text}" + (": FLAG" if raised else ""))
        if raised and name not in flags:
            flags.append(name)

    for d in (train_dir, val_dir):
        if "real_eval" in str(d):
            sys.exit("STOP: real_eval is read once, after training")
    js = json.load(open(sigma_file)) if sigma_file and os.path.isfile(sigma_file) else {}
    fit_names, bnm_t, bnm6 = bnm_side(peryear, split_ref, tools)
    out(f"audit_export.py reads export_obj2.py md5 {XMD5}; BNM fit half: {len(fit_names)} windows")
    tr = scan_split(train_dir, fit_names)
    va = scan_split(val_dir, fit_names, ref=tr)
    both = (("real_train", tr), ("real_val", va))

    # L1
    msgs = tr.l1 + va.l1
    check("L1", not msgs, f"input keys = allow-list {list(X.ALLOW)}, types and shapes, pairing with truth rows"
          + (f"; problems {len(msgs)}, first: {msgs[0]}" if msgs else f" ({tr.n} + {va.n} stacks)"))
    # L2
    check("L2", tr.l2 + va.l2 == 0, f"stacks with a record outside the true map or outside the surveyed cells: {tr.l2 + va.l2}")
    # L3
    ok3, txt = True, []
    for lab, S in both:
        for dv in (X.D_BIG, X.D_SMALL):
            sh = S.num[dv] / max(S.den[dv], 1); d1 = 1 - np.sqrt(1 - dv)
            tol = max(0.005, 4 * np.sqrt(0.25 / max(S.den[dv], 1)))      # 4 binomial sd at most; 0.005 on large samples
            ok3 &= S.den[dv] > 0 and abs(sh[0] - d1) <= tol and abs(sh[1] - d1 ** 2) <= tol and abs(sh[2] - dv) <= tol
            txt.append(f"{lab} d {dv}: {S.den[dv]} cell-years, one visit {sh[0]:.3f} ({d1:.3f}), both {sh[1]:.3f} ({d1 ** 2:.3f}), "
                       f"at least once {sh[2]:.3f} (tolerance {tol:.3f})")
    check("L3", ok3, "recorded share per occupied surveyed cell-year; " + "; ".join(txt))
    # L4
    ok4, txt = True, []
    for lab, S in both:
        real = (S.A / S.C) / (S.B / S.D)
        null = (S.A0 / S.C) / (S.B0 / S.D)
        tol = max(4 * null.std(), 0.01)
        rho = spearman(S.eff, S.tru)
        ok4 &= abs(real - null.mean()) <= tol and abs(rho) <= 0.05
        txt.append(f"{lab}: P(surveyed | occupied) / P(surveyed | empty) {real:.4f}, after torus shifts {null.mean():.4f} "
                   f"(sd {null.std():.4f}); Spearman(surveyed cells, true cells) {rho:+.3f}")
    check("L4", ok4, "effort carries no truth; " + "; ".join(txt))
    # L5
    sig = float(js.get("sigma", next(iter(tr.sigma)) if tr.sigma else np.nan))
    expc = 1 / np.sqrt(1 + sig ** 2)
    ok5, txt = True, []
    for lab, S in both:
        c, s = np.array(S.corr), np.array(S.sdr)
        ok5 &= len(c) > 0 and abs(np.median(c) - expc) <= 0.02 and c.max() < 0.99 and abs(np.median(s) - sig) <= 0.02
        txt.append(f"{lab}: noisy against exact map correlation median {np.median(c):.3f} (expected {expc:.3f}), max {c.max():.3f}; "
                   f"noise sd / map sd median {np.median(s):.3f} (sigma {sig:g})")
    k2, nk2 = k2_stored(val_dir, n_comm, n_rounds, draws)
    ref_v = js.get("val", np.nan)
    ok5 &= X.K2_LO <= k2 <= X.K2_HI and (not np.isfinite(ref_v) or abs(k2 - ref_v) <= 0.01)
    check("L5", ok5, "habitat input is noisy; " + "; ".join(txt) + f"; K2 from stored real_val maps {k2:.3f} ({nk2} cases; "
          f"calibrate {ref_v:.3f}, band {X.K2_LO}-{X.K2_HI})")
    # L6
    sd_both, src_both = tr.seeds & va.seeds, tr.srcs & va.srcs
    ok6 = not sd_both and not src_both and va.hit_stack == 0 and va.hit_round == 0 and va.hit_input == 0
    check("L6", ok6, f"no copying between splits: seeds in both {len(sd_both)}, source files in both {len(src_both)}; real_val "
          f"stacks of {INFO_LO}-{INFO_HI} true cells equal to a real_train stack {va.hit_stack}; community-rounds in both "
          f"{va.hit_round}; real_val input stacks (5+ records) equal to a real_train input stack {va.hit_input}")
    out("   for reference, low-information real_val stacks whose 6 true maps also occur in real_train (few arrangements, "
        "chance is expected): " + ", ".join(f"{k} cells {v}" for k, v in va.hit_small.items()) + f", 390-400 cells {va.hit_full}")
    if va.hits:
        twins = locate(train_dir, {h for _, h, *_ in va.hits})
        out(f"   matches of 11+ cells, real_val stack -> real_train twin (cells per year, years 1-6), first 20 of {len(va.hits)}:")
        for kind, h, sd, r, s, cy in va.hits[:20]:
            tw = "; ".join(f"seed {a} round {b} species {c}" for a, b, c in twins.get(h, []))
            out(f"   {kind}: seed {sd} round {r} species {s} {cy} -> {tw}")
    out(f"   stacks with 390+ true cells at the target year: real_train {tr.full_t}, real_val {va.full_t}; all {NC} cells in all "
        f"6 years: real_train {tr.full6}, real_val {va.full6}")
    # L7
    msgs = tr.l7 + va.l7
    codes = tr.code | va.code; sigs = tr.sigma | va.sigma
    ok7 = not msgs and len(codes) == 1 and len(sigs) == 1 and (not js or abs(next(iter(sigs)) - float(js["sigma"])) < 1e-12)
    check("L7", ok7, f"one export md5 in every file ({sorted(codes)}), one sigma ({sorted(sigs)}), years and pieces, fit-half "
          f"windows only" + (f"; problems {len(msgs)}, first: {msgs[0]}" if msgs else ""))
    # L8
    check("L8", tr.l8 + va.l8 == 0, f"n_rec equals the visits in every stack and year (mismatches {tr.l8 + va.l8}); stacks with no "
          f"record, found from the input alone: real_train {tr.zero} ({tr.zero / max(tr.n, 1):.1%}), real_val {va.zero} "
          f"({va.zero / max(va.n, 1):.1%})")
    # L9
    for lab, S in both:
        x = np.concatenate(S.l9x) if S.l9x else np.zeros(0); y = np.concatenate(S.l9y) if S.l9y else np.zeros(0)
        rho = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
        flag("L9", abs(rho) > 0.05, f"{lab}: row order inside a round against true range size, Spearman {rho:+.3f} "
             f"(rows are in species-id order; above 0.05 the stage E loader must shuffle rows and never use their position)")

    # B1
    tot_b = sum(bnm_t.values()) or 1
    out("B1 recorded class at the target year (stacks with 1+ record); BNM fit half recomputed from the per-year grids:")
    shares = {}
    for lab, S in both:
        lt = labels(S.rec_t[S.rec_t > 0])
        shares[lab] = {k: 100 * np.mean(lt == k) if len(lt) else 0 for k in REC}
    w_max, const_off = 0, 0
    for k in REC:
        b = 100 * bnm_t[k] / tot_b
        w = b / shares["real_train"][k] if shares["real_train"][k] > 0 else np.inf
        w_max = max(w_max, w); const_off = max(const_off, abs(b - X.BNM_SHARES[k]))
        out(f"   {k:>6}  real_train {shares['real_train'][k]:5.1f}%  real_val {shares['real_val'][k]:5.1f}%  BNM {b:5.1f}% "
            f"(constant {X.BNM_SHARES[k]:4.1f}%)  sampler weight {w:.2f}")
    flag("B1", w_max > 5 or const_off > 1, f"largest sampler weight {w_max:.2f} (FLAG above 5); largest gap between recomputed BNM "
         f"share and constant {const_off:.1f} points (FLAG above 1)")
    # B2
    out("B2 true class (columns) of stacks by recorded class at the target year (rows), real_train, % of the row:")
    out("   recorded " + " ".join(f"{k:>6}" for k in TRUE))
    lrec, ltrue = labels(tr.rec_t), labels(tr.true_t)
    for k in REC:
        sel = (lrec == k) & (tr.rec_t > 0)
        n = sel.sum()
        row = [100 * np.mean(ltrue[sel] == q) if n else 0 for q in TRUE]
        out(f"   {k:>8} " + " ".join(f"{v:6.1f}" for v in row) + f"   ({n} stacks)")
    # B3
    out("B3 truly rare stacks (true cells at the target year), real_train:")
    low = []
    for k in RARE:
        sel = ltrue == k
        n = sel.sum(); r6 = tr.rec6[sel]; rt = tr.rec_t[sel]
        bins = {lab: int(np.sum((r6 >= lo) & (r6 <= hi))) for lo, hi, lab in BINS}
        nd = len(tr.distinct[k])
        if nd < 500:
            low.append(k)
        out(f"   {k:>6}: {n} stacks; with a record over 6 years {np.mean(r6 > 0) if n else 0:.1%}, at the target year "
            f"{np.mean(rt > 0) if n else 0:.1%}; records over 6 years " + ", ".join(f"{b} {c}" for b, c in bins.items())
            + f"; distinct species with a record {nd}")
    flag("B3", bool(low), "rare classes with fewer than 500 distinct species with a record: " + (", ".join(low) if low else "none"))
    # B4
    diff = max(abs(shares["real_train"][k] - shares["real_val"][k]) for k in REC)
    flag("B4", diff > 1, f"largest gap between real_train and real_val recorded shares {diff:.1f} points (FLAG above 1)")
    # B5
    tb = sum(bnm6.values()) or 1
    out("B5 recorded cells over 6 years per stack (stacks with 1+ record) against BNM fit-half species-windows:")
    for lo, hi, lab in BINS:
        s_tr = 100 * np.mean((tr.rec6 >= lo) & (tr.rec6 <= hi)) / max(np.mean(tr.rec6 > 0), 1e-12)
        out(f"   {lab:>6}  real_train {s_tr:5.1f}%   BNM {100 * bnm6[lab] / tb:5.1f}%")
    # B6
    out("B6 repeats inside real_train (same community, species and 6 true maps in more than one round), by true class:")
    out("   " + ", ".join(f"{k}: {tr.dup_tot[k] - tr.dup_distinct[k]} of {tr.dup_tot[k]}" for k in TRUE)
        + f"; community-rounds repeated inside real_train {tr.round_dup_within}")
    n_ok = sum(res.values())
    out(f"audit: {n_ok} PASS, {len(res) - n_ok} FAIL, flags: {', '.join(flags) if flags else 'none'}")
    return res, flags, dict(hit_stack=va.hit_stack, hit_full=va.hit_full, hits=va.hits)


# ---------------------------------------------------------------- self-test
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        py, refpat = X.plant_grids(tmp)
        sim_t = Path(tmp) / "sim" / "real_train"; sim_t.mkdir(parents=True)
        sim_v = Path(tmp) / "sim" / "real_val"; sim_v.mkdir(parents=True)
        for s in (7001, 7002):
            X.plant_community(sim_t / f"c{s}_training.npz", s)
        for s in (7101, 7102):
            X.plant_community(sim_v / f"c{s}_training.npz", s)
        tp = sorted(str(p) for p in sim_t.glob("*.npz")); vp = sorted(str(p) for p in sim_v.glob("*.npz"))
        cal = X.calibrate(vp, vp, out=quiet)                     # K2 on val by construction, so L5 tests the read-back
        sig_file = Path(tmp) / "data" / "habitat_sigma.json"; sig_file.parent.mkdir()
        json.dump(dict(cal), open(sig_file, "w"))
        base = Path(tmp) / "data"
        X.export(tp, "real_train", str(base / "real_train"), cal["sigma"], py, refpat, tools, out=quiet)
        X.export(vp, "real_val", str(base / "real_val"), cal["sigma"], py, refpat, tools, out=quiet)

        def run(root, sigma_file=str(sig_file), info=False):
            r = audit(str(root / "real_train"), str(root / "real_val"), sigma_file, py, refpat, tools, out=quiet)
            return (r[0], r[2]) if info else r[0]

        good = run(base)
        t("A1", all(good.values()) and len(good) == 8, f"clean planted export: L1-L8 all PASS {good}")

        def broken(name, fn, must_fail):
            root = Path(tmp) / f"b_{name}"
            shutil.copytree(base, root)
            fn(root)
            r = run(root)
            ok = not r[must_fail] and all(v for k, v in r.items() if k != must_fail and k not in KNOCK.get(name, ()))
            t(f"bug {name}", ok, f"{must_fail} FAILS and no other check is wrongly blamed ({r})")

        def edit_input(root, split, fn, first_only=True):
            rows = rows_of(root / split)
            for row in rows[:1] if first_only else rows:
                p = root / split / row["input_file"]
                z = dict(np.load(p)); fn(z, np.load(root / split / row["truth_file"])); X.save_npz(p, z)

        def edit_truth(root, split, fn):
            row = rows_of(root / split)[0]
            p = root / split / row["truth_file"]
            z = dict(np.load(p)); fn(z); X.save_npz(p, z)

        def leak_key(z, ti):
            z["true_cells"] = ti["true_cells"]

        def outside(z, ti):
            E = X.unpack(z["effort"]); V = X.unpack(z["visit1"]); k = int(np.argmax(E.sum((1, 2, 3)) < NC * WINDOW))
            off = ~E[k]; V[k][off] = True; z["visit1"] = X.pack(V)

        def copy_visit(z, ti):
            z["visit2"] = z["visit1"].copy()

        def effort_truth(z, ti):
            z["effort"] = X.pack(X.unpack(z["effort"]) | X.unpack(ti["true_maps"]))

        def effort_by_size(z, ti):
            """subtle leak: larger true ranges get the larger survey footprints (records kept inside the new effort)"""
            E = X.unpack(z["effort"]); n = len(E)
            order_e = np.argsort(E.any(1).reshape(n, -1).sum(1), kind="stable")
            order_t = np.argsort(ti["true_cells"], kind="stable")
            En = np.empty_like(E); En[order_t] = E[order_e]
            V1 = X.unpack(z["visit1"]) & En; V2 = X.unpack(z["visit2"]) & En
            z["effort"] = X.pack(En); z["visit1"] = X.pack(V1); z["visit2"] = X.pack(V2)
            z["n_rec"] = (V1 | V2).reshape(n, WINDOW, -1).sum(2).astype(np.int16)

        def no_noise(z, ti):
            z["habitat_noisy"] = ti["habitat_exact"].astype(np.float16)

        def nrec(z, ti):
            z["n_rec"] = z["n_rec"].copy(); z["n_rec"][0, 0] += 1

        def copy_comm(root):
            row = rows_of(root / "real_train")[0]
            for k in ("truth_file", "input_file"):
                shutil.copy(root / "real_train" / row[k], root / "real_val" / row[k].replace("s700", "s990"))
            rows = rows_of(root / "real_val")
            new = dict(row); new["sim_seed"] = "9901"; new["source_md5"] = "x" * 32
            new["truth_file"] = row["truth_file"].replace("s700", "s990"); new["input_file"] = row["input_file"].replace("s700", "s990")
            rows.append(new)
            with open(root / "real_val" / "manifest.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

        def years(z):
            z["years"] = np.array([2013, 2014, 2015, 2016, 2017, 2019], np.int32)

        def set_stack(root, split, pick, new_map):
            """Replace the 6 true maps of one stack (picked by its target-year cells) and keep its records inside them."""
            row = rows_of(root / split)[0]
            tp_, ip_ = root / split / row["truth_file"], root / split / row["input_file"]
            ti_, ii_ = dict(np.load(tp_)), dict(np.load(ip_))
            k = int(np.argmax(pick(ti_["true_cells"])))
            Tm = X.unpack(ti_["true_maps"]); M = new_map(Tm[k]); Tm[k] = M
            ti_["true_maps"] = X.pack(Tm); ti_["true_cells"] = ti_["true_cells"].copy(); ti_["true_cells"][k] = int(M[-1].sum())
            V1 = X.unpack(ii_["visit1"]); V2 = X.unpack(ii_["visit2"]); V1[k] &= M; V2[k] &= M
            ii_["visit1"], ii_["visit2"] = X.pack(V1), X.pack(V2)
            ii_["n_rec"] = (V1 | V2).reshape(len(V1), WINDOW, -1).sum(2).astype(np.int16)
            X.save_npz(tp_, ti_); X.save_npz(ip_, ii_)
            return Tm[k]

        def copy_stack(root):
            """one informative real_train stack's 6 true maps written into a real_val stack"""
            src = set_stack(root, "real_train", lambda c: (c >= 30) & (c <= 300), lambda m: m)
            set_stack(root, "real_val", lambda c: c >= 1, lambda m: src)

        def full_twins(root):
            set_stack(root, "real_train", lambda c: c >= 1, lambda m: np.ones_like(m))
            set_stack(root, "real_val", lambda c: c >= 1, lambda m: np.ones_like(m))

        KNOCK = {"outside": ("L8",), "copy_visit": ("L8",), "effort_truth": ("L3",), "effort_by_size": ("L3",), "nrec": (),
                 "copy_comm": ("L4", "L5", "L7"), "copy_stack": ("L3", "L4")}
        broken("leak_key", lambda r: edit_input(r, "real_train", leak_key), "L1")
        broken("outside", lambda r: edit_input(r, "real_train", outside), "L2")
        broken("copy_visit", lambda r: edit_input(r, "real_train", copy_visit, first_only=False), "L3")
        broken("effort_truth", lambda r: edit_input(r, "real_train", effort_truth, first_only=False), "L4")
        broken("effort_by_size", lambda r: edit_input(r, "real_train", effort_by_size, first_only=False), "L4")
        broken("no_noise", lambda r: edit_input(r, "real_val", no_noise, first_only=False), "L5")
        broken("copy_comm", copy_comm, "L6")
        broken("copy_stack", copy_stack, "L6")
        root = Path(tmp) / "b_full_twins"; shutil.copytree(base, root); full_twins(root)
        r, info = run(root, info=True)
        t("A4", r["L6"] and info["hit_full"] >= 1 and info["hit_stack"] == 0,
          f"two identical all-cell stacks in real_train and real_val do not count as copying: L6 {r['L6']}, listed as 390-400 "
          f"cells {info['hit_full']}, informative matches {info['hit_stack']}")
        broken("years", lambda r: edit_truth(r, "real_train", years), "L7")
        broken("nrec", lambda r: edit_input(r, "real_train", nrec), "L8")

        # helpers on known answers
        t("A2", abs(spearman(np.arange(10), np.arange(10)) - 1) < 1e-12 and abs(spearman([1, 1, 2, 2], [1, 2, 1, 2])) < 1e-12,
          "Spearman with ties: 1 for the same order, 0 for no relation")
        stop = False
        try:
            audit(str(base / "real_train"), str(Path(tmp) / "real_eval"), str(sig_file), py, refpat, tools, out=quiet)
        except SystemExit:
            stop = True
        t("A3", stop, "a real_eval folder is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--train", default="../data/real_train"); ap.add_argument("--val", default="../data/real_val")
    ap.add_argument("--sigma-file", default="../data/habitat_sigma.json")
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--split-ref", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--n-comm", type=int, default=20); ap.add_argument("--n-rounds", type=int, default=10)
    ap.add_argument("--draws", type=int, default=5)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    res, flags, _ = audit(a.train, a.val, a.sigma_file, a.peryear, a.split_ref, a.obj1_tools, a.n_comm, a.n_rounds, a.draws)
    sys.exit(0 if all(res.values()) else 1)