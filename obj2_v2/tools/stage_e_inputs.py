#!/usr/bin/env python3
"""
stage_e_inputs.py  (obj2_v2, stage E, 8 Nov 2026; plan signed off by Sara on 8 Nov)

One input builder for training, validation, inference and, later, BNM (root cause R8). It reads input files only; the
targets come from a separate loader that reads the truth files. Each stack's tensor is built from that stack's own
arrays and its draw (canvas offset, habitat blank) only, so row order, file order and species ids cannot enter.

Tensor per stack, 22 channels on a 24 x 24 canvas (claude/obj2_v2_plan.md, "Stage E plan"); the 20 x 20 window sits at
offset (oy, ox), each 0-4, and every cell outside it is 0 in every channel (edge rule C1 option 2, no wrap):
     0-5   records of years 1-6 (visit 1 or visit 2)
     6-11  halo of years 1-6: sum over that year's records of exp(-d^2 / 2), d in cells, inside the window, no wrap
    12-17  effort of years 1-6 (the species' surveyed cells)
    18     noisy habitat of the target year, z-scored with that map's own mean and sd; 0 where blanked
    19     blank flag: 1 inside the window where the habitat is blanked
    20     habitat distance |z - mean z at the stack's recorded cells (6 years)|; 0 where blanked
    21     window mask: 1 inside the window
Two numbers per stack: log(1 + recorded cells over 6 years), log(1 + surveyed cells over 6 years).
Target (separate loader): the 6 true maps of the stack, placed on the canvas at the same offset.
Draws: training draws stacks with probability proportional to the stage D weight, offset and blank from a stream seeded
by the epoch (so a resumed run repeats them); the frozen set uses its stored shift modulo 5 and its stored blank.

Gates (signed off 8 Nov; rules written before the run)
    E1 the training path (packed arrays held in memory) and the inference path (input file read afresh, visits
       unpacked apart) give bit-identical tensors
    E2 no truth file is opened while inputs are built (guard); self-test: tensors equal with every truth file removed
    E3 each stack's tensor is the same whatever the order of draws and of files; no index or id is a feature
    E4 changing true cells in the truth files leaves every input tensor bit-equal (self-test)
    E6 records equal an independent placement at the offset; every channel is 0 outside the window; every record lies
       inside the target at the same canvas position (same offset for inputs and target); halo has no wrap
    E7 habitat blank share 0.50 in every recorded and true class over one epoch of draws (within 0.01, or 4 binomial
       sd for a class with few draws); exactly half of each class in the frozen set
    E5 (holes with decoys) is not needed on simulated validation (agreed 8 Nov) and moves to stage H.
Reported: channel means and sd inside the window, the two numbers, time per batch of 256, and the md5 of the tensors
of the first 64 frozen stacks (stage F checks that its loader gives the same md5).

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_e_inputs.py --self-test | tail -1
    python3 ../tools/stage_e_inputs.py --check --data ../data | tee ../logs/stage_e_inputs_8nov.txt
Needs numpy, export_obj2.py and stage_d_sampler.py in the same folder, and the two stage D files in --data.
"""
import argparse, hashlib, importlib.util, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_e_inputs.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


D = _load("stage_d_sampler")
X = D.X
G, W = X.G, X.WINDOW
CANVAS, PAD, HALO_SIG = 24, 4, 1.0
CH = ([f"rec_y{t + 1}" for t in range(W)] + [f"halo_y{t + 1}" for t in range(W)] + [f"eff_y{t + 1}" for t in range(W)]
      + ["habitat", "blank", "hab_dist", "mask"])
NCH = len(CH)
NUMBERS = ["log1p_recorded_cells", "log1p_surveyed_cells"]
N_DRAW, N_E1, BATCH, CHUNK = 200000, 2000, 256, 2048
BROKEN = os.environ.get("STAGE_E_BROKEN", "")         # self-test only: plants one bug so its gate must fail


# ---------------------------------------------------------------- the builder (inputs only)
def kernel(sig, wrap=False):
    i = np.arange(G)
    d = np.abs(i[:, None] - i[None])
    if wrap:
        d = np.minimum(d, G - d)
    return np.exp(-(d ** 2) / (2.0 * sig * sig))


def halo(V, sig=HALO_SIG):
    """V (n, 6, G, G) -> per-year sum of Gaussians over records, inside the window, no wrap (separable)."""
    K = kernel(sig, wrap=(BROKEN == "wrap"))
    return K @ V.astype(np.float64) @ K.T              # K V K^T per year (matmul: about 60 times faster than einsum here)


def place(A, offset):
    """A (n, k, G, G) -> (n, k, CANVAS, CANVAS), window at offset (oy, ox) per stack, zero elsewhere."""
    n, k = A.shape[:2]
    out = np.zeros((n, k, CANVAS, CANVAS), np.float32)
    for oy in range(PAD + 1):
        for ox in range(PAD + 1):
            m = (offset[:, 0] == oy) & (offset[:, 1] == ox)
            if m.any():
                out[m, :, oy:oy + G, ox:ox + G] = A[m]
    return out


def build(V, E, H, offset, blank, sig=HALO_SIG):
    """Records V and effort E (n, 6, G, G) bool, noisy habitat H (n, G, G), offset (n, 2) in 0-4, blank (n,) bool
    -> tensor (n, 22, 24, 24) float32 and numbers (n, 2) float32."""
    n = len(V)
    V, E, blank = V.astype(bool), E.astype(bool), np.asarray(blank, bool)
    R6 = V.any(1)
    nrec = R6.reshape(n, -1).sum(1)
    if (nrec == 0).any():
        raise ValueError("a stack without a record reached the builder (gate D2)")
    Hf = H.astype(np.float64)
    mu = Hf.reshape(n, -1).mean(1); sd = Hf.reshape(n, -1).std(1); sd[sd == 0] = 1.0
    Z = (Hf - mu[:, None, None]) / sd[:, None, None]
    zm = (Z * R6).reshape(n, -1).sum(1) / nrec
    keep = (~blank)[:, None, None]
    win = np.zeros((n, NCH, G, G), np.float32)
    win[:, 0:W] = V
    win[:, W:2 * W] = halo(V, sig)
    win[:, 2 * W:3 * W] = E
    win[:, 18] = Z * keep
    win[:, 19] = np.broadcast_to(blank[:, None, None], (n, G, G))
    win[:, 20] = np.abs(Z - zm[:, None, None]) * keep
    win[:, 21] = 1.0
    out = place(win, offset)
    if BROKEN == "leak_outside":
        out[:, 18] = np.where(out[:, 21] > 0, out[:, 18], 0.01)
    nums = np.stack([np.log1p(nrec), np.log1p(E.any(1).reshape(n, -1).sum(1))], 1).astype(np.float32)
    return out, nums


# ---------------------------------------------------------------- training path: packed inputs held in memory
class Inputs:
    """Input files of one split, packed, in the order given (stage D 'files'); records = visit1 | visit2 on the
    packed bytes. Each file's md5 is checked against the manifest."""
    def __init__(self, d, files):
        man = {r["input_file"]: r for r in D.rows_of(d)}
        self.d, self.files = Path(d), [str(f) for f in files]
        self.rec, self.eff, self.hab, self.sh, self.seed = [], [], [], [], []
        for f in self.files:
            row = man[f]
            p = self.d / f
            if X.file_md5(p) != row["input_md5"]:
                sys.exit(f"STOP: {p} md5 differs from the manifest")
            with np.load(p, allow_pickle=False) as z:
                self.rec.append(z["visit1"] | z["visit2"]); self.eff.append(z["effort"])
                self.hab.append(z["habitat_noisy"]); self.sh.append(z["stack_hab"])
            self.seed.append(int(row["sim_seed"]))

    def arrays(self, fi, rw):
        n = len(fi)
        V = np.zeros((n, W, G, G), bool); E = np.zeros_like(V); H = np.zeros((n, G, G), np.float32)
        for i in np.unique(fi):
            k = np.nonzero(fi == i)[0]; r = rw[k]
            V[k] = X.unpack(self.rec[i][r]); E[k] = X.unpack(self.eff[i][r])
            H[k] = self.hab[i][self.sh[i][r]]
        return V, E, H

    def get(self, fi, rw, offset, blank):
        V, E, H = self.arrays(fi, rw)
        if BROKEN == "truth_peek":                      # a builder that reads the targets must be refused (E2)
            targets(self.d, self.files, fi, rw, offset)
        out, nums = build(V, E, H, offset, blank)
        if BROKEN == "row_feature":
            out[:, 21] *= (1 + rw[:, None, None] % 7).astype(np.float32)
        return out, nums


def shuffled_copy(inp, which, seed=5):
    """The same stacks with the file list reversed and the rows of the files in `which` permuted (E3; only those are
    copied, to keep memory low); returns the copy and a function mapping (file index, row) of the original to the copy."""
    c = Inputs.__new__(Inputs)
    c.d, c.files, c.seed = inp.d, inp.files[::-1], inp.seed[::-1]
    g = np.random.default_rng(seed)
    perms = [g.permutation(len(r)) if i in which else np.arange(len(r)) for i, r in enumerate(inp.rec)]
    inv = [np.argsort(p) for p in perms]
    c.rec = [r[p] if i in which else r for i, (r, p) in enumerate(zip(inp.rec, perms))][::-1]
    c.eff = [e[p] if i in which else e for i, (e, p) in enumerate(zip(inp.eff, perms))][::-1]
    c.sh = [h[p] if i in which else h for i, (h, p) in enumerate(zip(inp.sh, perms))][::-1]
    c.hab = inp.hab[::-1]
    nf = len(inp.files)
    return c, lambda fi, rw: (nf - 1 - fi, np.array([inv[f][r] for f, r in zip(fi, rw)], dtype=np.int64))


def true_class(d, files, fi, rw):
    man = {r["input_file"]: r for r in D.rows_of(d)}
    tc = np.zeros(len(fi), int)
    for i in np.unique(fi):
        k = np.nonzero(fi == i)[0]
        with np.load(Path(d) / man[str(files[i])]["truth_file"], allow_pickle=False) as zt:
            tc[k] = D.tidx(zt["true_cells"][rw[k]])
    return tc


# ---------------------------------------------------------------- inference path: one input file read afresh
def build_from_file(path, rows, offset, blank):
    with np.load(path, allow_pickle=False) as z:
        V = X.unpack(z["visit1"][rows]) | X.unpack(z["visit2"][rows])
        E = X.unpack(z["effort"][rows]); H = z["habitat_noisy"][z["stack_hab"][rows]]
    return build(V, E, H, offset, blank, sig=(1.5 if BROKEN == "two_paths" else HALO_SIG))


# ---------------------------------------------------------------- targets (truth files; separate loader)
def targets(d, files, fi, rw, offset):
    man = {r["input_file"]: r for r in D.rows_of(d)}
    n = len(fi)
    T = np.zeros((n, W, G, G), bool); tc = np.zeros(n, int)
    for i in np.unique(fi):
        k = np.nonzero(fi == i)[0]
        with np.load(Path(d) / man[str(files[i])]["truth_file"], allow_pickle=False) as zt:
            T[k] = X.unpack(zt["true_maps"][rw[k]]); tc[k] = D.tidx(zt["true_cells"][rw[k]])
    off = np.zeros_like(offset) if BROKEN == "target_no_shift" else offset
    return place(T, off), tc


# ---------------------------------------------------------------- draws
def train_draws(sampler, epoch, n):
    """Stacks with probability proportional to the stage D weight (with replacement), offset 0-4 per axis, blank
    with probability 0.5, all from one stream seeded by the epoch."""
    g = X.rng(77, epoch)
    w = sampler["weight"].astype(np.float64)
    idx = g.choice(len(w), n, replace=True, p=w / w.sum())
    off = g.integers(0, PAD + 1, size=(n, 2))
    blank = g.random(n) < 0.5
    if BROKEN == "blank_by_class":
        blank = sampler["cls"][idx] <= 1
    return idx, off, blank


def frozen_draws(frozen):
    return frozen["shift"].astype(np.int64) % (PAD + 1), frozen["blank"].astype(bool)


def ref_place(A, offset):
    """Independent placement for E6: np.pad per stack."""
    return np.stack([np.pad(a, ((0, 0), (o[0], PAD - o[0]), (o[1], PAD - o[1]))) for a, o in zip(A, offset)]).astype(np.float32)


# ---------------------------------------------------------------- check on the exported data
def check(data, train_dir, val_dir, n_draw=N_DRAW, n_e1=N_E1, out=print):
    for p in (data, train_dir, val_dir):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, after training")
    sampler = dict(np.load(Path(data) / "stage_d_train_sampler.npz"))
    frozen = dict(np.load(Path(data) / "stage_d_frozen_val.npz"))
    res = {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    cmd5 = X.file_md5(__file__)
    out(f"stage_e_inputs.py md5 {cmd5}; stage_d_sampler.py md5 {X.file_md5(HERE / 'stage_d_sampler.py')}; "
        f"export_obj2.py md5 {X.file_md5(HERE / 'export_obj2.py')}")
    out(f"   channels {NCH} on {CANVAS} x {CANVAS}: " + ", ".join(CH) + "; numbers: " + ", ".join(NUMBERS))
    idx, off, blank = train_draws(sampler, 0, n_draw)
    fi_t, rw_t = sampler["file_index"][idx], sampler["row"][idx]
    voff, vblank = frozen_draws(frozen)
    fi_v, rw_v = frozen["file_index"], frozen["row"]
    with D.TruthGuard() as guard:
        try:
            t0 = time.time()
            TR = Inputs(train_dir, sampler["files"]); VA = Inputs(val_dir, frozen["files"])
            t_load = time.time() - t0
            # E1: training path against inference path
            s = np.arange(min(n_e1, n_draw))
            a1, n1 = TR.get(fi_t[s], rw_t[s], off[s], blank[s])
            e1 = True
            for i in np.unique(fi_t[s]):
                k = s[fi_t[s] == i]
                b, nb = build_from_file(Path(train_dir) / str(sampler["files"][i]), rw_t[k], off[k], blank[k])
                e1 &= np.array_equal(a1[np.searchsorted(s, k)], b) and np.array_equal(n1[np.searchsorted(s, k)], nb)
            kv = np.arange(min(n_e1, len(rw_v)))
            av, nv = VA.get(fi_v[kv], rw_v[kv], voff[kv], vblank[kv])
            for i in np.unique(fi_v[kv]):
                k = kv[fi_v[kv] == i]
                b, nb = build_from_file(Path(val_dir) / str(frozen["files"][i]), rw_v[k], voff[k], vblank[k])
                e1 &= np.array_equal(av[k], b) and np.array_equal(nv[k], nb)
            # E3: reversed draw order; files in reversed order with the rows of every file permuted
            r = s[::-1]
            a3, _ = TR.get(fi_t[r], rw_t[r], off[r], blank[r])
            e3 = np.array_equal(a3[::-1], a1)
            which = set(np.unique(fi_t[s])[:10].tolist())
            sw = s[np.isin(fi_t[s], list(which))]
            TRp, remap = shuffled_copy(TR, which)
            fp_, rp_ = remap(fi_t[sw], rw_t[sw])
            a3b, _ = TRp.get(fp_, rp_, off[sw], blank[sw])
            e3 &= np.array_equal(a3b, a1[sw])
            n3 = len(sw)
            # E6 on the inputs: placement and zeros outside the window, frozen set in chunks; timing; fingerprint
            V64, E64, H64 = VA.arrays(fi_v[:64], rw_v[:64])
            fp_in, _ = VA.get(fi_v[:64], rw_v[:64], voff[:64], vblank[:64])
            fprint = hashlib.md5(fp_in.tobytes()).hexdigest()
            outside_bad, place_bad, stats, nums_all = 0, 0, np.zeros((NCH, 3)), []
            for a in range(0, len(rw_v), CHUNK):
                k = np.arange(a, min(a + CHUNK, len(rw_v)))
                x, nm = VA.get(fi_v[k], rw_v[k], voff[k], vblank[k])
                inside = x[:, 21:22] > 0
                outside_bad += int(((x != 0) & ~inside).sum())
                Vk, Ek, _ = VA.arrays(fi_v[k], rw_v[k])
                place_bad += int((x[:, 0:W] != ref_place(Vk, voff[k])).sum() + (x[:, 2 * W:3 * W] != ref_place(Ek, voff[k])).sum())
                m = np.broadcast_to(inside, x.shape)
                stats[:, 0] += m.sum((0, 2, 3)); stats[:, 1] += (x * m).sum((0, 2, 3)); stats[:, 2] += (x * x * m).sum((0, 2, 3))
                nums_all.append(nm)
            t0 = time.time()
            for _ in range(5):
                TR.get(fi_t[:BATCH], rw_t[:BATCH], off[:BATCH], blank[:BATCH])
            t_batch = (time.time() - t0) / 5
            crashed = None
        except RuntimeError as e:
            crashed = str(e)
    if crashed:
        gate("E2", False, crashed)
        return res, None
    gate("E1", e1, f"training path and inference path bit-identical on {len(s)} training draws and {len(kv)} frozen stacks")
    gate("E2", not guard.opened, f"no truth file opened while the inputs were built (opened {len(guard.opened)}); "
         f"inputs loaded in {t_load:.0f} s")
    gate("E3", e3, f"tensors equal with the draw order reversed ({len(s)} draws), and with the files in reversed order and the rows "
         f"of 10 files permuted ({n3} draws)")
    # truth after the build: records inside the target at the same canvas position
    rec_out = 0
    for a in range(0, len(rw_v), CHUNK):
        k = np.arange(a, min(a + CHUNK, len(rw_v)))
        Tk, _ = targets(val_dir, frozen["files"], fi_v[k], rw_v[k], voff[k])
        x, _ = VA.get(fi_v[k], rw_v[k], voff[k], vblank[k])
        rec_out += int(((x[:, 0:W] > 0) & (Tk == 0)).sum())
    corner = np.zeros((1, W, G, G), bool); corner[0, :, 0, 0] = True
    hc = halo(corner)[0, 0]
    nowrap = abs(hc[G - 1, G - 1] - np.exp(-((G - 1) ** 2) / HALO_SIG ** 2)) < 1e-12 and hc[0, G - 1] < 1e-30
    gate("E6", outside_bad == 0 and place_bad == 0 and rec_out == 0 and nowrap,
         f"frozen set: non-zero values outside the window {outside_bad}, records or effort differing from an independent "
         f"placement {place_bad}, records outside the target at the same position {rec_out}; halo without wrap {nowrap}")
    # E7: one epoch of draws, blank share per recorded class and per true class (truth read after the draws)
    tc_t = true_class(train_dir, sampler["files"], fi_t, rw_t)
    worst, lines = 0.0, []
    for name, lab, labels in (("recorded", sampler["cls"][idx], D.TCLS), ("true", tc_t, D.TRUE)):
        parts = []
        for c, l in enumerate(labels):
            m = lab == c
            if m.sum() == 0:
                continue
            share = blank[m].mean(); tol = max(0.01, 4 * np.sqrt(0.25 / m.sum()))
            worst = max(worst, abs(share - 0.5) / tol)
            parts.append(f"{l} {share:.3f} ({int(m.sum())})")
        lines.append(f"{name}: " + ", ".join(parts))
    half = all(int(vblank[frozen["cls"] == c].sum()) == int((frozen["cls"] == c).sum()) // 2 for c in range(len(D.TCLS)))
    gate("E7", worst <= 1 and half, f"blank share over {n_draw} training draws within 0.01 (or 4 binomial sd) in every class; "
         f"frozen set exactly half per class {half}")
    for l in lines:
        out("   " + l)
    n_, s1, s2 = stats[:, 0], stats[:, 1], stats[:, 2]
    mean = s1 / np.maximum(n_, 1); sdv = np.sqrt(np.maximum(s2 / np.maximum(n_, 1) - mean ** 2, 0))
    out("   channel mean / sd inside the window (frozen set): " + ", ".join(f"{c} {m:.3f}/{v:.3f}" for c, m, v in zip(CH, mean, sdv)))
    na = np.concatenate(nums_all)
    out("   numbers (frozen set) median, min, max: " + ", ".join(f"{c} {np.median(na[:, j]):.2f}, {na[:, j].min():.2f}, {na[:, j].max():.2f}"
                                                              for j, c in enumerate(NUMBERS)))
    out(f"   time per batch of {BATCH} (training path): {1000 * t_batch:.0f} ms; md5 of the tensors of the first 64 frozen stacks {fprint}")
    n_ok = sum(res.values())
    out(f"stage E: {n_ok} PASS, {len(res) - n_ok} FAIL")
    return res, dict(fprint=fprint, mean=mean, sd=sdv)


# ---------------------------------------------------------------- self-test
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    # unit: placement, halo, habitat channels, numbers
    g = np.random.default_rng(3)
    V = np.zeros((3, W, G, G), bool); V[:, 0, 5, 7] = True; V[1, 3, 0, 0] = True; V[2, 5, 19, 19] = True
    E = g.random((3, W, G, G)) < 0.4
    H = g.standard_normal((3, G, G)).astype(np.float32)
    off = np.array([[0, 0], [4, 1], [2, 4]]); bl = np.array([False, True, False])
    x, nm = build(V, E, H, off, bl)
    ok_pos = all(x[i, 0, 5 + off[i, 0], 7 + off[i, 1]] == 1 for i in range(3)) and x[1, 3, 4, 1] == 1 and x[2, 5, 21, 23] == 1
    ok_out = all(np.all(x[i][:, x[i, 21] == 0] == 0) for i in range(3))
    t("U1", ok_pos and ok_out and x.shape == (3, NCH, CANVAS, CANVAS), "records land at window cell + offset; every channel 0 outside the window")
    hz = x[0, 18, 0:G, 0:G]
    t("U2", abs(hz.mean()) < 1e-5 and abs(hz.std() - 1) < 1e-4 and np.all(x[1, 18] == 0) and np.all(x[1, 20] == 0)
      and np.all(x[1, 19][x[1, 21] > 0] == 1) and np.all(x[0, 19] == 0),
      "habitat z-scored inside the window; blanked stack: habitat and distance 0, flag 1")
    zrec = hz[5, 7]
    t("U3", abs(x[0, 20, 5, 7]) < 1e-6 and abs(x[0, 20, 0, 0] - abs(hz[0, 0] - zrec)) < 1e-5,
      "habitat distance is 0 at the only recorded cell and |z - z(record)| elsewhere")
    t("U4", abs(nm[0, 0] - np.log1p(1)) < 1e-6 and abs(nm[2, 0] - np.log1p(2)) < 1e-6
      and abs(nm[0, 1] - np.log1p(E[0].any(0).sum())) < 1e-5, "numbers: log(1 + recorded cells), log(1 + surveyed cells)")
    corner = np.zeros((1, W, G, G), bool); corner[0, :, 0, 0] = True
    hc = halo(corner)[0, 0]
    t("U5", abs(hc[0, 0] - 1) < 1e-12 and abs(hc[0, 1] - np.exp(-0.5)) < 1e-12 and hc[0, G - 1] < 1e-30,
      "halo: 1 at the record, exp(-1/2) next to it, nothing across the edge (no wrap)")
    try:
        build(np.zeros((1, W, G, G), bool), E[:1], H[:1], off[:1], bl[:1]); raised = False
    except ValueError:
        raised = True
    t("U6", raised, "a stack without a record is refused")
    with tempfile.TemporaryDirectory() as tmp:
        py, refpat = X.plant_grids(Path(tmp) / "g")
        st = Path(tmp) / "sim" / "real_train"; st.mkdir(parents=True); sv = Path(tmp) / "sim" / "real_val"; sv.mkdir(parents=True)
        for s_ in (7001, 7002, 7003):
            X.plant_community(st / f"c{s_}_training.npz", s_)
        for s_ in (7101, 7102):
            X.plant_community(sv / f"c{s_}_training.npz", s_)
        base = Path(tmp) / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.6, py, refpat, tools, out=quiet)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.6, py, refpat, tools, out=quiet)
        D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
        res, info = check(str(base), str(base / "real_train"), str(base / "real_val"), n_draw=20000, n_e1=500, out=quiet)
        t("S1", all(res.values()) and len(res) == 5, f"planted export: E1-E3, E6, E7 PASS {res}")
        # E2 and E4: truth files removed, then truth files changed: same tensors
        sampler = dict(np.load(base / "stage_d_train_sampler.npz")); frozen = dict(np.load(base / "stage_d_frozen_val.npz"))
        voff, vbl = frozen_draws(frozen)
        ref, _ = Inputs(base / "real_val", frozen["files"]).get(frozen["file_index"], frozen["row"], voff, vbl)
        nt = Path(tmp) / "notruth"; shutil.copytree(base / "real_val", nt / "real_val")
        for p in nt.glob("*/*_truth.npz"):
            p.unlink()
        with D.TruthGuard():
            a, _ = Inputs(nt / "real_val", frozen["files"]).get(frozen["file_index"], frozen["row"], voff, vbl)
        t("S2", np.array_equal(a, ref), "E2: with every truth file removed the tensors are the same")
        for p in (base / "real_val").glob("*_truth.npz"):
            z = dict(np.load(p))
            tm = X.unpack(z["true_maps"]); tm |= np.random.default_rng(1).random(tm.shape) < 0.2
            z["true_maps"] = X.pack(tm); X.save_npz(p, z)
        b, _ = Inputs(base / "real_val", frozen["files"]).get(frozen["file_index"], frozen["row"], voff, vbl)
        t("S3", np.array_equal(b, ref), "E4: true cells added in the truth files leave every input tensor bit-equal")
        # reruns give the same fingerprint
        res2, info2 = check(str(base), str(base / "real_train"), str(base / "real_val"), n_draw=20000, n_e1=500, out=quiet)
        t("S4", info2 is not None and info2["fprint"] == info["fprint"], "a rerun gives the same tensor md5 (fingerprint)")
        i0 = train_draws(sampler, 3, 1000); i1 = train_draws(sampler, 3, 1000); i2 = train_draws(sampler, 4, 1000)
        t("S5", all(np.array_equal(p, q) for p, q in zip(i0, i1)) and not np.array_equal(i0[0], i2[0]),
          "training draws repeat for the same epoch (resume) and differ between epochs")
        # restore the truth files' effect: E6 records-inside-target needs the true truth, so re-export real_val for the bugs
        shutil.rmtree(base / "real_val")
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.6, py, refpat, tools, out=quiet)
        want = {"two_paths": "E1", "truth_peek": "E2", "row_feature": "E3", "target_no_shift": "E6", "leak_outside": "E6",
                "wrap": "E6", "blank_by_class": "E7"}
        for bug, gate in want.items():
            env = dict(os.environ); env["STAGE_E_BROKEN"] = bug
            code = ("import sys; sys.argv=['x']; import importlib.util as u; s=u.spec_from_file_location('e', %r); "
                    "m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r,_=m.check(%r, %r, %r, n_draw=20000, n_e1=500, out=lambda *a, **k: None); print('RES', r)"
                    % (str(Path(__file__).resolve()), str(base), str(base / "real_train"), str(base / "real_val")))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        stop = False
        try:
            check(str(base), str(base / "real_train"), str(Path(tmp) / "real_eval"), n_draw=1000, n_e1=10, out=quiet)
        except SystemExit:
            stop = True
        t("S6", stop, "a real_eval folder is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--train", default=None); ap.add_argument("--val", default=None)
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    if a.check:
        tr, va = a.train or str(Path(a.data) / "real_train"), a.val or str(Path(a.data) / "real_val")
        res, _ = check(a.data, tr, va)
        sys.exit(0 if res and all(res.values()) else 1)
    ap.print_help()