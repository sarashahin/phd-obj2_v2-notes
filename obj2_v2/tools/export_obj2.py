#!/usr/bin/env python3
"""
export_obj2.py  (obj2_v2, stage C, 8 Nov 2026)

What it does
    Turns each real-run community into two files for Objective 2 training (claude/obj2_data_plan.md, "Stage C plan"):
      s<seed>_truth.npz  the answer, for scoring only: per species-stack the 6 true yearly maps, the true cells at the
                         target year, the survey draw (BNM window, BNM species, detection d); per round the exact
                         habitat of the target year.
      s<seed>_input.npz  what a survey could give, written from an allow-list only: per species-stack the surveyed
                         cells of each year (effort), two visits per surveyed cell-year, and records per year; per
                         round the noisy habitat of the target year.
    A species-stack is one species present in at least one of the 6 years of a candidate round (target year = last
    piece of a round in the last half with all 25 pieces; the 5 pieces before it are the 5 years before). Every
    candidate round is written (Sara, 8 Nov); a stage D sampler sets the class mix later.

Recording (Sara's agreed rule of 7 Nov, with q = 1 and the per-year footprints of 8 Nov)
    1. one BNM fit-half window per community-round (as world_vs_bnm v5.8 observe(): all species of a community are
       seen through the same window); fit half imported from obj1_v2/tools/bnm_detection_check.py, never re-written;
    2. each species gets the per-year known cells (effort rule, list length 8, focal species left out) of one BNM
       species of that window drawn at random; simulated year t = BNM year t of the 6 consecutive per-year grids
       (target piece = BNM target year);
    3. d = 0.64 if the stack has 20 or fewer present surveyed cells over the 6 years, else 0.80 (bnm_detection_check:
       0.643 / 0.642 / 0.800; world_vs_bnm v5.8 class rule);
    4. two visits per surveyed cell-year, each records an occupied cell with d1 = 1 - sqrt(1 - d), so "recorded at
       least once" = d exactly. No q: the per-year footprints carry the change in effort and d was measured inside
       them, so a q would count effort twice.
    Every draw has its own stream [20261108, community seed, round, (species,) purpose], so a rerun is bit-identical
    and one community or stack never changes another.

Habitat noise
    noisy map = exact map + sigma x sd(map) x N(0, 1), one draw per community-round, stored as float16. sigma is set
    once by --calibrate on real_train so the median K2 AUC (test_13: records 10% of the target-year occupied cells,
    far cells 3 or more cells from every record on the torus, score -|H - mean H at the records|) is 0.80, then
    checked on real_val (pass 0.78-0.82). Blanking is done later, when inputs are built (stage E).

Data rules
    real_eval is refused unless --final-eval is given (once, after training). A community already exported to a
    sibling split folder is refused. BNM test-half windows are never opened.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/export_obj2.py --self-test | tail -1
    python3 ../tools/export_obj2.py --calibrate --train "../../obj1_v2/work/results/stage2a/real_train/*_training.npz" \
        --val "../../obj1_v2/work/results/stage2a/real_val/*_training.npz" | tee ../logs/export_calibrate_8nov.txt
    python3 ../tools/export_obj2.py --export --split real_train \
        --sim "../../obj1_v2/work/results/stage2a/real_train/*_training.npz" --out ../data/real_train
    (--calibrate writes ../data/habitat_sigma.json; --export reads sigma from it and refuses it if the real_val check
    failed; --habitat-sigma overrides it)
    python3 ../tools/export_obj2.py --report --data ../data/real_train ../data/real_val
Defaults: --peryear ../../AI_simulation/Empirical_data/bnm_grids_peryear --minlist 8
          --split-ref "../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz" --obj1-tools ../../obj1_v2/tools
Needs numpy only.
"""
import argparse, csv, glob, hashlib, importlib.util, io, json, os, subprocess, sys, tempfile, zipfile
from pathlib import Path
import numpy as np

G = 20
NC = G * G
WINDOW = 6
D_BIG, D_SMALL, SMALL = 0.80, 0.64, 20
BASE = 20261108
CLASSES = (("0", 0, 0), ("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50), ("51+", 51, NC))
BNM_SHARES = {"1": 8.3, "2-5": 15.0, "6-10": 9.3, "11-20": 11.0, "21-50": 25.7, "51+": 30.7}
K2_TARGET, K2_LO, K2_HI = 0.80, 0.78, 0.82
ALLOW = ("code_md5", "effort", "habitat_noisy", "n_rec", "pair_md5", "stack_hab", "visit1", "visit2")
TRUTH_KEYS = ("code_md5", "habitat_exact", "habitat_sigma", "pair_md5", "pieces", "round_ids", "round_window", "sim_seed",
              "source_md5", "split", "stack_bnm_species", "stack_d", "stack_round", "stack_species", "true_cells",
              "true_maps", "window_names", "years")
BROKEN = os.environ.get("EXPORT_OBJ2_BROKEN", "")    # self-test only: plants one bug so its test must fail
OPENED = []
_SEQ = np.random.default_rng(BASE)                   # used only by the planted bug "sequential"


# ---------------------------------------------------------------- small helpers
def cls(n):
    return next(lab for lab, lo, hi in CLASSES if lo <= n <= hi)


def rng(*keys):
    if BROKEN == "sequential":
        return _SEQ
    return np.random.default_rng([BASE, *[int(k) for k in keys]])


def pack(m):
    """bool (..., G, G) -> uint8 (..., 50)"""
    return np.packbits(m.reshape(m.shape[:-2] + (NC,)), axis=-1)


def unpack(b):
    return np.unpackbits(b, axis=-1, count=NC).astype(bool).reshape(b.shape[:-1] + (G, G))


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_npz(path, arrays):
    """npz with fixed zip timestamps, so the same arrays always give the same file md5; written to a temporary
    name first, so a stopped run never leaves half a file."""
    tmp = str(path) + ".part"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[k]), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)


def auc(score, lab):
    """Mann-Whitney AUC with average ranks for ties (test_13 uses scipy's mannwhitneyu, the same value)."""
    pos = int(lab.sum()); neg = lab.size - pos
    if pos == 0 or neg == 0:
        return np.nan
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    ranks = ((cum - cnt + 1 + cum) / 2.0)[inv]
    return (ranks[lab].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


def far_mask(rec):
    """Cells 3 or more cells (Chebyshev, torus) from every record, as test_13 far_mask."""
    near = np.zeros_like(rec)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            near |= np.roll(np.roll(rec, dy, 0), dx, 1)
    return ~near


def code_md5():
    return file_md5(__file__)


# ---------------------------------------------------------------- simulated communities
def read_community(path):
    """Candidate rounds: (round, species ids (sorted), maps (n, 6, G, G) bool) for species present in any of the
    6 years, plus the exact round habitat ENV_habitat_t."""
    z = np.load(path, allow_pickle=True)
    need = ("ASM_F_round", "ASM_F_piece", "ASM_F_offsets", "ASM_F_ids", "ASM_F_bits", "ENV_habitat_t", "HABITAT_substeps", "SIM_SEED")
    miss = [k for k in need if k not in z.files]
    if miss:
        sys.exit(f"STOP: {path} lacks {miss}")
    M = int(np.asarray(z["HABITAT_substeps"]).item())
    if M < WINDOW:
        sys.exit(f"STOP: {path} has {M} pieces per round, fewer than {WINDOW}")
    H = np.asarray(z["ENV_habitat_t"], dtype=np.float64)
    if H.shape[1:] != (G, G):
        sys.exit(f"STOP: {path} habitat maps are {H.shape[1:]}, not {G} x {G}")
    rnd, pc, off = (np.asarray(z[k]) for k in ("ASM_F_round", "ASM_F_piece", "ASM_F_offsets"))
    ids, bits = np.asarray(z["ASM_F_ids"]), np.asarray(z["ASM_F_bits"])
    where = {(int(r), int(p)): (int(off[k]), int(off[k + 1])) for k, (r, p) in enumerate(zip(rnd, pc))}
    n_rounds = len(H)
    rounds, skipped = [], 0
    for r in range(n_rounds // 2, n_rounds):
        if any((r, j) not in where for j in range(M)):
            skipped += 1
            continue
        maps = {}
        for t, j in enumerate(range(M - WINDOW, M)):
            a, b = where[(r, j)]
            if b == a:
                continue
            B = unpack(bits[a:b])
            for sid, m in zip(ids[a:b].tolist(), B):
                if not m.any():
                    continue
                cur = maps.setdefault(int(sid), np.zeros((WINDOW, G, G), bool))
                if cur[t].any():
                    sys.exit(f"STOP: {path} round {r} piece {j}: species {sid} appears twice")
                cur[t] = m
        sids = np.array(sorted(maps), np.int64)
        st = np.stack([maps[s] for s in sids]) if len(sids) else np.zeros((0, WINDOW, G, G), bool)
        rounds.append((r, sids, st))
    return dict(seed=int(np.asarray(z["SIM_SEED"]).item()), M=M, n_rounds=n_rounds, rounds=rounds, skipped=skipped, H=H)


def noisy_habitat(H, sigma, seed, r):
    if BROKEN == "no_noise":
        return H.copy()
    sd = float(H.std()) or 1.0
    return H + sigma * sd * rng(seed, r, 4).standard_normal((G, G))


# ---------------------------------------------------------------- BNM per-year grids, fit half only
def import_bdc(tools):
    p = Path(tools) / "bnm_detection_check.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; give --obj1-tools (the folder that holds bnm_detection_check.py)")
    spec = importlib.util.spec_from_file_location("bnm_detection_check", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, file_md5(p)


def bnm_load(bdc, f):
    OPENED.append(str(f))
    return bdc.load(f)


def load_bnm(peryear, split_ref_pattern, tools, minlist=8, out=print):
    """Fit-half windows of the per-year grids: list of (name, effort (6, S, G, G) bool), the 6 years."""
    bdc, bmd5 = import_bdc(tools)
    ref = bdc.split_ref(split_ref_pattern)
    pooled = Path(peryear) / f"minlist{minlist}"
    ydirs = sorted((int(p.name.rsplit("_y", 1)[1]), p) for p in Path(peryear).glob(f"minlist{minlist}_y*") if p.is_dir())
    names = sorted(f.name for f in pooled.glob("bnm_*.npz"))
    if not names or not ydirs:
        sys.exit(f"STOP: no per-year grids under {peryear} (step 10 with --per-year)")
    keep, east, overlap = bdc.fit_half(names, ref)
    if BROKEN == "testhalf":
        keep = names
    if not keep:
        sys.exit("STOP: no fit-half window in the per-year grids")
    wins, years0 = [], None
    for nm in keep:
        a = bnm_load(bdc, pooled / nm)
        years = [y for y, _ in ydirs] + [int(np.asarray(a["target_year"]).item())]
        if years != list(range(years[0], years[0] + len(years))):
            sys.exit(f"STOP: years are not consecutive: {years}")
        if len(years) != WINDOW:
            sys.exit(f"STOP: {len(years)} years in the per-year grids, the stack needs {WINDOW}")
        if years0 is not None and years != years0:
            sys.exit(f"STOP: {nm} has years {years}, other windows {years0}")
        years0 = years
        E, spc = [], []
        for y, d in ydirs:
            f = d / nm
            if not f.is_file():
                sys.exit(f"STOP: {nm} has no file for year {y}")
            z = bnm_load(bdc, f)
            E.append(np.asarray(z["known_holdout"]).astype(bool))
            spc.append(np.asarray(z["species"]) if "species" in z.files else None)
        E.append(np.asarray(a["known_target"]).astype(bool))
        spc.append(np.asarray(a["species"]) if "species" in a.files else None)
        shapes = {e.shape for e in E}
        if len(shapes) != 1 or next(iter(shapes))[1:] != (G, G):
            sys.exit(f"STOP: {nm}: effort arrays differ between years or are not S x {G} x {G}: {sorted(shapes)}")
        named = [s for s in spc if s is not None]
        if any(not np.array_equal(s, named[0]) for s in named[1:]):
            sys.exit(f"STOP: {nm}: the species rows differ between years")
        wins.append((nm, np.stack(E)))
    out(f"BNM per-year grids {pooled}: {len(names)} windows; {len(wins)} fit-half windows read; {east} test-half and "
        f"{overlap} sharing cells left unopened; years {years0}; species rows {wins[0][1].shape[1]}; "
        f"bnm_detection_check.py md5 {bmd5}")
    return wins, years0, keep


# ---------------------------------------------------------------- survey of one community-round
def survey_round(seed, r, sids, maps, wins):
    w = int(rng(seed, r, 1).integers(len(wins)))
    Ew = wins[w][1]
    S = Ew.shape[1]
    bsp = np.array([int(rng(seed, r, s, 2).integers(S)) for s in sids], np.int64)
    E = Ew[:, bsp].transpose(1, 0, 2, 3) if len(sids) else np.zeros((0, WINDOW, G, G), bool)
    PE = maps & E if BROKEN != "outside" else maps.copy()
    nps = (maps & E).any(1).reshape(len(sids), -1).sum(1)
    d = np.where(nps <= SMALL, D_SMALL, D_BIG)
    d1 = d if BROKEN == "detect" else 1 - np.sqrt(1 - d)
    V1 = np.zeros_like(PE); V2 = np.zeros_like(PE)
    for k, s in enumerate(sids):
        U = rng(seed, r, s, 3).random((2, WINDOW, G, G))
        V1[k] = PE[k] & (U[0] < d1[k]); V2[k] = PE[k] & (U[1] < d1[k])
    if BROKEN == "effort_from_truth":
        E = E | maps
    return w, bsp, E, d, V1, V2


def refuse_eval(paths, out_dir, final_eval):
    if final_eval:
        return
    bad = [p for p in list(paths) + [str(out_dir)] if "real_eval" in str(p)]
    if bad and BROKEN != "eval":
        sys.exit("STOP: real_eval is read once, after training; give --final-eval only then. Refused: " + ", ".join(bad[:3]))


def sibling_seeds(out_dir):
    """Community seeds and source md5s already exported to other split folders next to out_dir."""
    seen = {}
    for man in Path(out_dir).resolve().parent.glob("*/manifest.csv"):
        if man.parent.resolve() == Path(out_dir).resolve():
            continue
        with open(man) as f:
            for row in csv.DictReader(f):
                seen[("seed", row["sim_seed"])] = man.parent.name
                seen[("md5", row["source_md5"])] = man.parent.name
    return seen


def export_community(path, split, wins, win_names, years, sigma, out_dir, cmd5):
    c = read_community(path)
    seed = c["seed"]
    rows_r, rows_s, maps_all, cells, rwin, bsps, ds = [], [], [], [], [], [], []
    effs, v1s, v2s, hab_ex, hab_no, stack_hab, rids = [], [], [], [], [], [], []
    for i, (r, sids, maps) in enumerate(c["rounds"]):
        w, bsp, E, d, V1, V2 = survey_round(seed, r, sids, maps, wins)
        rids.append(r); rwin.append(w)
        hab_ex.append(c["H"][r]); hab_no.append(noisy_habitat(c["H"][r], sigma, seed, r))
        rows_r.append(np.full(len(sids), r)); rows_s.append(sids); maps_all.append(maps)
        cells.append(maps[:, -1].reshape(len(sids), -1).sum(1)); bsps.append(bsp); ds.append(d)
        effs.append(E); v1s.append(V1); v2s.append(V2); stack_hab.append(np.full(len(sids), i))
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    st_r, st_s = cat(rows_r, np.int32), cat(rows_s, np.int64)
    pair = hashlib.md5(np.stack([st_r.astype(np.int64), st_s]).tobytes()).hexdigest()
    maps_c = np.concatenate(maps_all) if maps_all else np.zeros((0, WINDOW, G, G), bool)
    V1c = np.concatenate(v1s) if v1s else np.zeros_like(maps_c); V2c = np.concatenate(v2s) if v2s else np.zeros_like(maps_c)
    Ec = np.concatenate(effs) if effs else np.zeros_like(maps_c)
    src = file_md5(path)
    truth = dict(code_md5=np.array(cmd5), habitat_exact=np.array(hab_ex, np.float32).reshape(-1, G, G),
                 habitat_sigma=np.float64(sigma), pair_md5=np.array(pair), pieces=np.arange(c["M"] - WINDOW, c["M"]),
                 round_ids=np.array(rids, np.int32), round_window=np.array(rwin, np.int32), sim_seed=np.int64(seed),
                 source_md5=np.array(src), split=np.array(split), stack_bnm_species=cat(bsps, np.int32),
                 stack_d=cat(ds, np.float32), stack_round=st_r, stack_species=st_s, true_cells=cat(cells, np.int16),
                 true_maps=pack(maps_c), window_names=np.array(win_names), years=np.array(years, np.int32))
    inp = dict(code_md5=np.array(cmd5), effort=pack(Ec), habitat_noisy=np.array(hab_no, np.float16).reshape(-1, G, G),
               n_rec=(V1c | V2c).reshape(len(st_r), WINDOW, -1).sum(2).astype(np.int16), pair_md5=np.array(pair),
               stack_hab=cat(stack_hab, np.int32), visit1=pack(V1c), visit2=pack(V2c))
    if BROKEN == "leak_d":
        inp["stack_d"] = truth["stack_d"]
    tp, ip = Path(out_dir) / f"s{seed}_truth.npz", Path(out_dir) / f"s{seed}_input.npz"
    save_npz(tp, truth); save_npz(ip, inp)
    return dict(sim_seed=seed, source=os.path.basename(path), source_md5=src, rounds=len(c["rounds"]), skipped=c["skipped"],
                stacks=len(st_r), truth_file=tp.name, truth_md5=file_md5(tp), input_file=ip.name, input_md5=file_md5(ip),
                bytes=os.path.getsize(tp) + os.path.getsize(ip))


def export(paths, split, out_dir, sigma, peryear, split_ref, tools, final_eval=False, resume=False, out=print):
    refuse_eval(paths, out_dir, final_eval)
    if sigma is None or sigma < 0:
        sys.exit("STOP: give --habitat-sigma from --calibrate (0 or more)")
    os.makedirs(out_dir, exist_ok=True)
    seen = sibling_seeds(out_dir)
    wins, years, keep = load_bnm(peryear, split_ref, tools, out=out)
    names = [n for n, _ in wins]
    cmd5 = code_md5()
    man_path = Path(out_dir) / "manifest.csv"
    done = {}
    if resume and man_path.is_file():
        with open(man_path) as f:
            done = {row["source"]: row for row in csv.DictReader(f)}
    rows = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        sd = str(int(np.asarray(z["SIM_SEED"]).item())) if "SIM_SEED" in z.files else "?"
        hit = seen.get(("seed", sd)) or seen.get(("md5", file_md5(p)))
        if hit and BROKEN != "twosplits":
            sys.exit(f"STOP: community {os.path.basename(p)} (seed {sd}) is already exported in split {hit}")
        b = os.path.basename(p)
        if b in done and all((Path(out_dir) / done[b][k]).is_file() and file_md5(Path(out_dir) / done[b][k]) == done[b][k2]
                             for k, k2 in (("truth_file", "truth_md5"), ("input_file", "input_md5"))):
            rows.append(done[b]); continue
        rows.append(export_community(p, split, wins, names, years, sigma, out_dir, cmd5))
        out(f"   {b}: seed {rows[-1]['sim_seed']}, {rows[-1]['rounds']} rounds, {rows[-1]['stacks']} stacks")
    with open(str(man_path) + ".part", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    os.replace(str(man_path) + ".part", man_path)
    n = sum(int(r["stacks"]) for r in rows); by = sum(int(r["bytes"]) for r in rows)
    out(f"== {split}: {len(rows)} communities, {n} species-stacks, {by / 1e6:.1f} MB written to {out_dir}; habitat sigma {sigma:g}; "
        f"export_obj2.py md5 {cmd5}")
    return rows


# ---------------------------------------------------------------- habitat calibration (K2)
def k2_cases(paths, n_comm, n_rounds, draws):
    """Fixed K2 cases: per sampled community-round and target-year species with 2+ occupied and 2+ empty cells,
    `draws` record draws of 10% of occupied cells (test_13)."""
    idx = np.unique(np.round(np.linspace(0, len(paths) - 1, min(n_comm, len(paths)))).astype(int))
    cases = []
    for i in idx:
        c = read_community(paths[i])
        rr = c["rounds"]
        pick = np.unique(np.round(np.linspace(0, len(rr) - 1, min(n_rounds, len(rr)))).astype(int)) if rr else []
        for k in pick:
            r, sids, maps = rr[k]
            H = c["H"][r]
            eps = rng(c["seed"], r, 4).standard_normal((G, G))
            perm = rng(c["seed"], r, 5).permutation(NC)
            for j, s in enumerate(sids):
                P = maps[j, -1]
                occ = P.ravel().nonzero()[0]
                if len(occ) < 2 or (~P).sum() < 2:
                    continue
                kk = max(1, int(np.ceil(0.10 * len(occ))))
                g = rng(c["seed"], r, s, 6)
                for _ in range(draws):
                    rec = np.zeros(NC, bool); rec[g.choice(occ, kk, replace=False)] = True; rec = rec.reshape(G, G)
                    far = far_mask(rec)
                    if P[far].all() or not P[far].any():
                        continue
                    cases.append((H, eps, perm, rec, far, P))
    return cases


def k2_median(cases, sigma, shuffle=False):
    vals = []
    for H, eps, perm, rec, far, P in cases:
        sd = float(H.std()) or 1.0
        Hn = H if BROKEN == "no_noise" else H + sigma * sd * eps
        if shuffle:
            Hn = Hn.ravel()[perm].reshape(G, G)
        sc = -np.abs(Hn - Hn[rec].mean())
        vals.append(auc(sc[far], P[far]))
    v = np.array(vals)
    return float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else np.nan


def calibrate(train, val, n_comm=20, n_rounds=10, draws=5, out=print):
    ct, cv = k2_cases(train, n_comm, n_rounds, draws), k2_cases(val, n_comm, n_rounds, draws)
    if not ct or not cv:
        sys.exit("STOP: no K2 case (species with 2+ occupied and 2+ empty cells) in the sampled rounds")
    exact, shuf = k2_median(ct, 0.0), k2_median(ct, 0.0, shuffle=True)
    out(f"K2 on real_train sample ({len(ct)} cases): exact habitat median AUC {exact:.3f}; shuffled habitat {shuf:.3f}")
    if exact <= K2_TARGET:
        sigma = 0.0
        out(f"   exact habitat is already at or below {K2_TARGET}: no noise added")
    else:
        lo, hi = 0.0, 1.0
        while k2_median(ct, hi) > K2_TARGET and hi < 256:
            hi *= 2
        for _ in range(30):
            mid = (lo + hi) / 2
            if k2_median(ct, mid) > K2_TARGET:
                lo = mid
            else:
                hi = mid
        sigma = round((lo + hi) / 2, 4)
    tr, va, vex, vsh = k2_median(ct, sigma), k2_median(cv, sigma), k2_median(cv, 0.0), k2_median(cv, 0.0, shuffle=True)
    ok = K2_LO <= va <= K2_HI and 0.45 <= vsh <= 0.55
    out(f"habitat sigma {sigma:g} (noise sd = sigma x sd of each map): real_train median AUC {tr:.3f}")
    out(f"real_val check ({len(cv)} cases): noisy {va:.3f} (pass {K2_LO}-{K2_HI}), exact {vex:.3f}, shuffled {vsh:.3f} (pass 0.45-0.55): "
        f"{'PASS' if ok else 'FAIL'}")
    return dict(sigma=sigma, train=tr, val=va, exact=exact, shuffled=shuf, val_exact=vex, val_shuffled=vsh, ok=ok)


# ---------------------------------------------------------------- report (X8, X9) and split check
def report(dirs, out=print):
    res, seen = {}, {}
    for d in dirs:
        man = Path(d) / "manifest.csv"
        if not man.is_file():
            sys.exit(f"STOP: {man} not found")
        rows = list(csv.DictReader(open(man)))
        for row in rows:
            for key in (("seed", row["sim_seed"]), ("md5", row["source_md5"])):
                if key in seen and seen[key] != d:
                    out(f"FAIL: community {row['source']} is in {seen[key]} and {d}")
                    res["overlap"] = True
                seen[key] = d
        n_st, rec6, rec_t, cls_t, cls_true, by = 0, [], [], {}, {}, 0
        for row in rows:
            ti = np.load(Path(d) / row["truth_file"]); ii = np.load(Path(d) / row["input_file"])
            if str(ti["pair_md5"]) != str(ii["pair_md5"]):
                sys.exit(f"STOP: {row['truth_file']} and {row['input_file']} do not pair")
            rec = unpack(ii["visit1"]) | (unpack(ii["visit2"]) if BROKEN != "report_v1" else False)
            nt = rec[:, -1].reshape(len(rec), -1).sum(1)
            rec6.append(rec.any(1).reshape(len(rec), -1).sum(1)); rec_t.append(nt)
            for x in nt[nt > 0]:
                cls_t[cls(int(x))] = cls_t.get(cls(int(x)), 0) + 1
            for x in ti["true_cells"]:
                cls_true[cls(int(x))] = cls_true.get(cls(int(x)), 0) + 1
            n_st += len(rec)
            by += os.path.getsize(Path(d) / row["truth_file"]) + os.path.getsize(Path(d) / row["input_file"])
        rec6 = np.concatenate(rec6) if rec6 else np.zeros(0, int); rec_t = np.concatenate(rec_t) if rec_t else np.zeros(0, int)
        tot = sum(cls_t.values()) or 1
        out(f"== {d}: {len(rows)} communities, {n_st} species-stacks; recorded cells over 6 years per stack: median "
            f"{int(np.median(rec6)) if n_st else 0}, none {np.mean(rec6 == 0):.1%}; at the target year: median "
            f"{int(np.median(rec_t)) if n_st else 0}, none {np.mean(rec_t == 0):.1%}")
        out("   recorded class at the target year (stacks with 1+ record) against the BNM fit half:")
        for lab in BNM_SHARES:
            out(f"   {lab:>6}  {cls_t.get(lab, 0):>9}  {100 * cls_t.get(lab, 0) / tot:5.1f}%   BNM {BNM_SHARES[lab]:4.1f}%   "
                f"(true class at the target year: {cls_true.get(lab, 0)})")
        out(f"   files {by / 1e6:.1f} MB, {by / max(n_st, 1):.0f} bytes per stack")
        res[d] = dict(stacks=n_st, rec_t=rec_t, rec6=rec6, cls_t=cls_t, cls_true=cls_true, bytes=by)
    if not res.get("overlap"):
        out(f"split check: no community in two of {len(dirs)} folders: PASS")
    return res


# ---------------------------------------------------------------- self-test
def plant_community(path, seed, n_species=14, rounds=30, M=25):
    """Species occupy cells near their habitat optimum (small and large ranges), with year-to-year flicker."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:G, 0:G]
    opt = g.uniform(-1.2, 1.2, n_species)
    width = np.where(np.arange(n_species) % 3 == 0, 0.06, g.uniform(0.25, 0.7, n_species))
    Fr, Fp, Fo, Fi, Fb, Hs = [], [], [0], [], [], []
    for r in range(rounds):
        H = np.cos(2 * np.pi * xx / G + 0.1 * r) + 0.7 * np.sin(4 * np.pi * yy / G + 0.3) + 0.15 * g.standard_normal((G, G))
        Hs.append(H)
        for j in range(M):
            ids, B = [], []
            for s in range(n_species):
                m = (np.abs(H - opt[s]) < width[s]) & (g.random((G, G)) < 0.85)
                if m.any():
                    ids.append(s); B.append(m.ravel())
            Fr.append(r); Fp.append(j); Fi += ids; Fo.append(Fo[-1] + len(ids))
            Fb.append(np.packbits(np.array(B), axis=1) if B else np.zeros((0, 50), np.uint8))
    np.savez(path, ASM_F_round=np.array(Fr), ASM_F_piece=np.array(Fp), ASM_F_offsets=np.array(Fo), ASM_F_ids=np.array(Fi),
             ASM_F_bits=np.concatenate(Fb), ENV_habitat_t=np.array(Hs, np.float32), HABITAT_substeps=np.int64(M),
             SIM_SEED=np.int64(seed), P_last_final=np.zeros((n_species, G, G), bool), NICHE_opt=opt)


def plant_grids(tmp, S=30, n_win=12, years=range(2014, 2020), seed=0, bad=None):
    """Per-year grids as step 10 writes them; fit half = eastings 0, 40, 80 (6 windows), test half 120-200."""
    g = np.random.default_rng(seed)
    years = list(years)
    py, ref = Path(tmp) / "peryear", Path(tmp) / "ref"
    (py / "minlist8").mkdir(parents=True); ref.mkdir()
    for y in years[:-1]:
        (py / f"minlist8_y{y}").mkdir()
    spc = np.array([f"sp{i:02d}" for i in range(S)])
    for w in range(n_win):
        e, n = 40 * (w % 6), 40 * (w // 6)
        name = f"bnm_e{e:04d}_n{n:04d}.npz"
        base = g.random(NC) < 0.8
        for y in years:
            Ef = (base[None] & (g.random((S, NC)) < 0.9) & (g.random(NC) < 0.85)[None]).reshape(S, G, G).astype(np.uint8)
            O = (Ef.astype(bool) & (g.random((S, G, G)) < 0.1)).astype(np.uint8)
            sp = spc
            if bad == "rows" and w == 0 and y == years[0]:
                Ef = Ef[:-1]; O = O[:-1]; sp = spc[:-1]
            if bad == "names" and w == 0 and y == years[0]:
                sp = spc[::-1]
            if y == years[-1]:
                np.savez(py / "minlist8" / name, obs_mask=O, known_target=Ef, target_year=np.int64(y), species=sp,
                         window_cells=np.int64(G), cell_m=np.float64(2000))
                np.savez(ref / name, obs_mask=O, known_target=Ef, window_cells=np.int64(G), cell_m=np.float64(2000))
            else:
                np.savez(py / f"minlist8_y{y}" / name, present_year=O, known_holdout=Ef, species=sp)
    return str(py), str(ref / "bnm_*.npz")


def self_test(tools):
    global OPENED
    res = []
    quiet = lambda *a, **k: None

    def t(name, ok, what):
        res.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    def stop(fn):
        try:
            fn(); return False
        except SystemExit:
            return True

    def loadall(d):
        rows = list(csv.DictReader(open(Path(d) / "manifest.csv")))
        return [(np.load(Path(d) / r["truth_file"]), np.load(Path(d) / r["input_file"])) for r in rows], rows

    def X1(d):
        ok = True
        for ti, ii in loadall(d)[0]:
            V = unpack(ii["visit1"]) | unpack(ii["visit2"])
            ok &= not (V & ~unpack(ti["true_maps"])).any() and not (V & ~unpack(ii["effort"])).any()
        return ok

    def X2(d):
        num = {D_BIG: np.zeros(3), D_SMALL: np.zeros(3)}; den = {D_BIG: 0, D_SMALL: 0}
        for ti, ii in loadall(d)[0]:
            PE = unpack(ti["true_maps"]) & unpack(ii["effort"]); v1, v2 = unpack(ii["visit1"]), unpack(ii["visit2"])
            for dv in (D_BIG, D_SMALL):
                k = np.isclose(ti["stack_d"], dv)
                pe = PE[k]
                den[dv] += pe.sum()
                num[dv] += [(v1[k] & pe).sum(), (v1[k] & v2[k] & pe).sum(), ((v1[k] | v2[k]) & pe).sum()]
        out, ok = [], True
        for dv in (D_BIG, D_SMALL):
            sh = num[dv] / max(den[dv], 1); d1 = 1 - np.sqrt(1 - dv)
            ok &= den[dv] > 2000 and abs(sh[0] - d1) <= 0.02 and abs(sh[1] - d1 ** 2) <= 0.02 and abs(sh[2] - dv) <= 0.02
            out.append(f"d {dv}: {den[dv]} occupied surveyed cell-years, one visit {sh[0]:.3f} (d1 {d1:.3f}), both {sh[1]:.3f}, "
                       f"at least once {sh[2]:.3f}")
        return ok, "; ".join(out)

    with tempfile.TemporaryDirectory() as tmp:
        py, refpat = plant_grids(tmp)
        sim = Path(tmp) / "sim" / "real_train"; sim.mkdir(parents=True)
        simv = Path(tmp) / "sim" / "real_val"; simv.mkdir(parents=True)
        for s in (7001, 7002):
            plant_community(sim / f"c{s}_training.npz", s)
        for s in (7101, 7102):
            plant_community(simv / f"c{s}_training.npz", s)
        paths = sorted(str(p) for p in sim.glob("*_training.npz"))
        vpaths = sorted(str(p) for p in simv.glob("*_training.npz"))
        data = Path(tmp) / "data"
        run = lambda out_dir, ps=paths, sigma=1.0, split="real_train", **kw: export(ps, split, str(out_dir), sigma, py, refpat, tools, out=quiet, **kw)

        # X1, X2
        run(data / "real_train")
        t("X1", X1(data / "real_train"), "every record lies inside the true map and the surveyed cells of its year")
        ok2, txt2 = X2(data / "real_train")
        t("X2", ok2, f"recorded share per occupied surveyed cell-year: {txt2}")

        # X3: allow-list, and truth changed outside the surveyed cells leaves the input bytes the same
        (ti, ii), = loadall(data / "real_train")[0][:1]
        keys_ok = set(ii.files) <= set(ALLOW) and not (set(ii.files) & (set(TRUTH_KEYS) - set(ALLOW)))
        src = np.load(paths[0], allow_pickle=True); arr = {k: src[k] for k in src.files}
        E = unpack(ii["effort"]); st_r, st_s = ti["stack_round"], ti["stack_species"]
        k = int(np.argmax(ti["true_cells"] > 5)); r, s = int(st_r[k]), int(st_s[k])
        outside = ~E[k].any(0)
        rnd, pc, off, ids = arr["ASM_F_round"], arr["ASM_F_piece"], arr["ASM_F_offsets"], arr["ASM_F_ids"]
        bits = arr["ASM_F_bits"].copy(); changed = 0
        for row in np.nonzero(rnd == r)[0]:
            if pc[row] < 25 - WINDOW:
                continue
            for q in range(off[row], off[row + 1]):
                if ids[q] == s:
                    m = unpack(bits[q]); add = outside & ~m
                    changed += int(add.sum()); bits[q] = pack(m | add)
        arr["ASM_F_bits"] = bits
        alt = Path(tmp) / "alt" / "real_train"; alt.mkdir(parents=True)
        np.savez(alt / os.path.basename(paths[0]), **arr)
        export([str(alt / os.path.basename(paths[0]))], "real_train", str(Path(tmp) / "altdata" / "x"), 1.0, py, refpat, tools, out=quiet)
        a_in = file_md5(Path(tmp) / "altdata" / "x" / os.path.basename(loadall(data / "real_train")[1][0]["input_file"]))
        a_tr = file_md5(Path(tmp) / "altdata" / "x" / loadall(data / "real_train")[1][0]["truth_file"])
        row0 = loadall(data / "real_train")[1][0]
        same_in = a_in == row0["input_md5"] and a_tr != row0["truth_md5"] and changed > 0
        t("X3", keys_ok and same_in, f"input keys only from the allow-list {sorted(ii.files)}; {changed} true cells added outside the "
          f"surveyed cells of one stack: input file md5 unchanged, truth file md5 changed")

        # X4: fit half only; stops on bad grids
        win_ok = all(set(ti["window_names"]) <= set(bdc_keep(py, refpat, tools)) for ti, _ in loadall(data / "real_train")[0])
        OPENED = []
        load_bnm(py, refpat, tools, out=quiet)
        east_opened = [f for f in OPENED if int(Path(f).stem.split("_e")[1].split("_")[0]) >= 120]
        with tempfile.TemporaryDirectory() as t2:
            pyb, refb = plant_grids(t2, bad="rows")
            stop_rows = stop(lambda: load_bnm(pyb, refb, tools, out=quiet))
        with tempfile.TemporaryDirectory() as t2:
            pyb, refb = plant_grids(t2, bad="names")
            stop_names = stop(lambda: load_bnm(pyb, refb, tools, out=quiet))
        with tempfile.TemporaryDirectory() as t2:
            pyb, refb = plant_grids(t2)
            os.rename(Path(pyb) / "minlist8_y2016", Path(t2) / "gone")
            stop_years = stop(lambda: load_bnm(pyb, refb, tools, out=quiet))
        t("X4", win_ok and not east_opened and stop_rows and stop_names and stop_years,
          f"drawn windows all fit half; test-half files opened {len(east_opened)}; a year with other species rows stops: "
          f"{stop_rows}; other species names stop: {stop_names}; a missing year (not consecutive) stops: {stop_years}")

        # X5: rerun bit-identical; a changed community changes no other file
        run(Path(tmp) / "rerun" / "real_train")
        m1 = {r["truth_file"]: (r["truth_md5"], r["input_md5"]) for r in loadall(data / "real_train")[1]}
        m2 = {r["truth_file"]: (r["truth_md5"], r["input_md5"]) for r in loadall(Path(tmp) / "rerun" / "real_train")[1]}
        mod = Path(tmp) / "mod" / "real_train"; mod.mkdir(parents=True)
        plant_community(mod / os.path.basename(paths[0]), 7001, n_species=11)
        os.symlink(paths[1], mod / os.path.basename(paths[1]))
        run(Path(tmp) / "modrun" / "real_train", ps=sorted(str(p) for p in mod.glob("*_training.npz")))
        m3 = {r["truth_file"]: (r["truth_md5"], r["input_md5"]) for r in loadall(Path(tmp) / "modrun" / "real_train")[1]}
        f0, f1 = sorted(m1)
        t("X5", m1 == m2 and m3[f1] == m1[f1] and m3[f0] != m1[f0],
          "a rerun gives the same md5 for every file; changing the first community changes only its own files")

        # X6: real_eval refused; the same community in two splits refused
        evald = Path(tmp) / "sim" / "real_eval"; evald.mkdir()
        plant_community(evald / "c7201_training.npz", 7201)
        env = dict(os.environ); env["EXPORT_OBJ2_BROKEN"] = BROKEN
        p = subprocess.run([sys.executable, __file__, "--export", "--split", "real_eval", "--habitat-sigma", "1",
                            "--sim", str(evald / "*_training.npz"), "--out", str(data / "real_eval"), "--peryear", py,
                            "--split-ref", refpat, "--obj1-tools", tools], capture_output=True, text=True, env=env)
        two = stop(lambda: run(data / "real_val", ps=paths[:1], split="real_val"))
        t("X6", p.returncode != 0 and "real_eval" in p.stderr and two,
          "real_eval refused without --final-eval; a community already in real_train refused for real_val")

        # X7: habitat noise calibrated by K2
        cal = calibrate(paths, vpaths, n_comm=2, n_rounds=8, draws=5, out=quiet)
        t("X7", cal["ok"] and cal["val_exact"] >= 0.90 and K2_LO <= cal["val"] <= K2_HI,
          f"sigma {cal['sigma']:g}: real_val K2 noisy {cal['val']:.3f} (0.78-0.82), exact {cal['val_exact']:.3f} (0.90 or more), "
          f"shuffled {cal['val_shuffled']:.3f} (0.45-0.55)")

        # X8: report on a hand-made export with known records
        hand = Path(tmp) / "hand"; hand.mkdir()
        V1 = np.zeros((3, WINDOW, G, G), bool); V2 = np.zeros_like(V1)
        V1[0, -1, 0, 0] = True; V1[1, -1, 1, :2] = True; V2[1, -1, 1, 1:3] = True; V2[1, 0, 5, 5] = True
        save_npz(hand / "s1_truth.npz", dict(pair_md5=np.array("p"), true_cells=np.array([1, 3, 60], np.int16)))
        save_npz(hand / "s1_input.npz", dict(pair_md5=np.array("p"), visit1=pack(V1), visit2=pack(V2)))
        with open(hand / "manifest.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["sim_seed", "source", "source_md5", "truth_file", "input_file"])
            wr.writerow(["1", "s1", "m", "s1_truth.npz", "s1_input.npz"])
        rep = report([str(hand)], out=quiet)[str(hand)]
        t("X8", rep["cls_t"] == {"1": 1, "2-5": 1} and list(rep["rec_t"]) == [1, 3, 0] and list(rep["rec6"]) == [1, 4, 0]
          and rep["cls_true"] == {"1": 1, "2-5": 1, "51+": 1},
          "report: records per stack (target year 1, 3, 0; 6 years 1, 4, 0; one cell in both visits counted once) and recorded classes (1 and 2-5) from a hand-made export")

        # X9: sizes
        rep2 = report([str(data / "real_train")], out=quiet)[str(data / "real_train")]
        disk = sum(os.path.getsize(p) for p in (data / "real_train").glob("s*_*.npz"))
        t("X9", rep2["bytes"] == disk and rep2["stacks"] > 0,
          f"file sizes reported as on disk ({disk} bytes, {disk / rep2['stacks']:.0f} bytes per stack)")
    n_ok = sum(res)
    print(f"{n_ok} PASS, {len(res) - n_ok} FAIL")
    return all(res)


def bdc_keep(py, refpat, tools):
    bdc, _ = import_bdc(tools)
    names = sorted(f.name for f in (Path(py) / "minlist8").glob("bnm_*.npz"))
    return bdc.fit_half(names, bdc.split_ref(refpat))[0]


def broken_runs(tools):
    """Each planted bug must make its own test fail (run in a fresh process per bug)."""
    want = {"outside": "X1", "detect": "X2", "leak_d": "X3", "effort_from_truth": "X3", "testhalf": "X4", "eval": "X6",
            "twosplits": "X6", "sequential": "X5", "no_noise": "X7", "report_v1": "X8"}
    ok = True
    for bug, test in want.items():
        env = dict(os.environ); env["EXPORT_OBJ2_BROKEN"] = bug
        p = subprocess.run([sys.executable, __file__, "--self-test-core", "--obj1-tools", tools], capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith(test + " ")), "")
        good = line.endswith("FAIL")
        ok &= good
        print(f"broken version '{bug}': {test} {'FAILS as it must' if good else 'does not fail'}: {'PASS' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true"); ap.add_argument("--self-test-core", action="store_true")
    ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--export", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--train"); ap.add_argument("--val"); ap.add_argument("--sim"); ap.add_argument("--split")
    ap.add_argument("--out"); ap.add_argument("--data", nargs="+")
    ap.add_argument("--habitat-sigma", type=float)
    ap.add_argument("--sigma-file", default="../data/habitat_sigma.json")
    ap.add_argument("--n-comm", type=int, default=20); ap.add_argument("--n-rounds", type=int, default=10)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--peryear", default="../../AI_simulation/Empirical_data/bnm_grids_peryear")
    ap.add_argument("--split-ref", default="../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz")
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    ap.add_argument("--minlist", type=int, default=8)
    ap.add_argument("--final-eval", action="store_true"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.self_test_core:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    if a.self_test:
        ok = self_test(a.obj1_tools)
        okb = broken_runs(a.obj1_tools)
        print(f"self-test {'and broken versions ' if okb else ''}{'all PASS' if ok and okb else 'FAIL'}")
        sys.exit(0 if ok and okb else 1)
    if a.calibrate:
        if not (a.train and a.val):
            sys.exit("STOP: give --train and --val")
        tr, va = sorted(glob.glob(a.train)), sorted(glob.glob(a.val))
        if any("real_eval" in p for p in tr + va):
            sys.exit("STOP: real_eval is read once, after training")
        if not tr or not va:
            sys.exit("STOP: --train or --val matches no file")
        cal = calibrate(tr, va, a.n_comm, a.n_rounds, a.draws)
        os.makedirs(os.path.dirname(a.sigma_file) or ".", exist_ok=True)
        json.dump(dict(cal, code_md5=code_md5(), train_glob=a.train, val_glob=a.val, n_train=len(tr), n_val=len(va)),
                  open(a.sigma_file, "w"), indent=1)
        print(f"written {a.sigma_file}" + ("" if cal["ok"] else "; the real_val check failed, so --export will refuse it"))
    elif a.export:
        if not (a.sim and a.split and a.out):
            sys.exit("STOP: give --sim, --split and --out")
        if a.split not in ("real_train", "real_val", "real_eval"):
            sys.exit("STOP: --split is real_train, real_val or real_eval")
        ps = sorted(glob.glob(a.sim))
        if not ps:
            sys.exit(f"STOP: --sim matches no file: {a.sim}")
        if a.split not in a.out or any(a.split not in p for p in ps):
            sys.exit(f"STOP: --split {a.split} must appear in --out and in every --sim path")
        sigma = a.habitat_sigma
        if sigma is None:
            if not os.path.isfile(a.sigma_file):
                sys.exit(f"STOP: no --habitat-sigma and no {a.sigma_file}; run --calibrate first")
            js = json.load(open(a.sigma_file))
            if not js.get("ok"):
                sys.exit(f"STOP: {a.sigma_file} records a failed real_val check (K2 {js.get('val')}); the habitat noise is not set")
            sigma = float(js["sigma"])
            print(f"habitat sigma {sigma:g} from {a.sigma_file} (real_val K2 {js['val']:.3f})")
        export(ps, a.split, a.out, sigma, a.peryear, a.split_ref, a.obj1_tools, a.final_eval, a.resume)
    elif a.report:
        if not a.data:
            sys.exit("STOP: give --data folders")
        report(a.data)
    else:
        ap.print_help()