############################################
# assembly_stepwise_ibm.py  (obj1_v2; v2 27 Oct 2026: niche mode, life table, window-end presence;
#   v2.1 28 Oct 2026: optional niche trade-off strength; v2.2 29 Oct 2026: optional dispersal rate per species)
############################################
"""
Infinite-pool community assembly for the individual-based model (IBM).

Each round adds m = 1 + floor(frac_multi * gamma) new species (gamma = current
richness; fewer if m would pass max_attempts). Each newcomer gets random
competition links (assembly_utils.draw_interactions), one individual in each of
seed_size random patches, and its own growth-rate field. The IBM then runs for
window_steps. A newcomer is kept if it reaches the detection threshold in at least
one patch; a resident is removed for good when it no longer does.

Changes from the original file (obj1_v2, 24 Oct 2026), each found by a test:
1. Field per species. Each species draws its growth-rate field once, on arrival,
   from (seed_field, species id), keeps it in every window, and the field is
   returned so the final run uses it too. The original drew all fields again by
   list position in every window, so a species' field changed whenever an earlier
   species was removed (test_02 E4, test_03 C1), and species lost in the final run
   sat on cells that were poor under their new field (test_04 E4).
2. Seeds. All random numbers come from SeedSequence(seed): one stream for
   interactions and founding cells, one for the demographic seed of every window
   and of the final run. The original gave every window numpy seed 0 (test_03 A5).
3. Last round kept. The round that reaches max_attempts is capped to the attempts
   left and recorded before stopping. The original simulated it and dropped it
   (test_03 A3; a median of 166 invaders per evaluation community, test_04 H2).
4. Assembly windows do not store trajectories (record_mode "none"); nothing used them.

v2 (27 Oct 2026, stage 2d-a part B; tests in tests/test_12_niche_assembly.py):
5. Niche mode (habitat given). Species share one habitat layer H; species sid has a
   niche (optimum, width, peak) drawn from (seed_field, sid) by environment.niche_traits,
   and its growth rate is environment.niche_r(H, ...) - cost. Field mode (habitat None)
   runs exactly as before: no random number is drawn differently (test_12 R1).
6. Life table (Sara's question, 27 Oct): for every species that ever arrived, the round
   it arrived, whether it passed its first window, the round it was removed (-1 if still
   present) and its largest range (cells at or above the threshold at a window end).
   Species ids count invasion attempts, so the id is also the attempt of arrival.
   Round of arrival -1 means present at the start of this run (the founder, or a resident
   of a resumed run).
7. Window-end presence: the map of every species present at the end of every round,
   packed 1 bit per cell, with its total count. Turnover, who took over the cells of a
   removed species, and history frames during invasions are read from this record later,
   so the simulator fixes no definition of them.

v2.1 (28 Oct 2026): the niche settings may also give "tradeoff" (strength k of the
specialist-generalist trade-off, environment.niche_traits; default 1 = equal area, as in
v2). Without it, or with 1, every result is the same as v2 (test_12 C5).

v2.2 (29 Oct 2026, stage 2d-b): disp=dict(d_min=..., d_max=...) gives every species its own
dispersal rate, log-uniform in [d_min, d_max], drawn from (seed_field, species id) by
dispersal.species_dispersal_rate on a stream of its own (niches unchanged). Each window passes the
rates of its species list to dispersal.set_species_rates and switches them off afterwards, also
on an error. The rates of the final community (disp_rate) and of every species that arrived
(LT_d) are returned. Without disp nothing is drawn or set differently, so every result equals
v2.1 (test_15 A1). Needs dispersal.py with patch_dispersal_per_species.py, and propagule dispersal.

Resuming from a checkpoint keeps species ids, so fields are rebuilt exactly; the
random streams restart from the seed (as in the original), so a resumed world is not
the world an uninterrupted run gives. Its history arrays start at the resume round
(ASM_resumed_from_round).
"""
from __future__ import annotations
import logging
import numpy as np
from models_ibm import IBMModel
from config import NUM_PATCHES_X, NUM_PATCHES_Y, THRESHOLD
from assembly_utils import draw_interactions, expand_RC, prune_extinct
from environment import generate_spatial_r
import dispersal as _disp                               # v2.2: per-species dispersal rates

try:
    import cupy as cp
    def _to_host(a):
        return cp.asnumpy(a) if isinstance(a, cp.ndarray) else a
except Exception:
    def _to_host(a):
        return a

log = logging.getLogger(__name__)
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X
NICHE_KEYS = ("opt_sd", "w_min", "w_max", "peak_max", "cost")
NICHE_OPTIONAL = ("tradeoff",)


def species_field(sid, *, base_r=1.0, length_scale=None, var_r=None, seed_field=None):
    """Growth-rate field (Ny, Nx) of species `sid`. It depends only on
    (seed_field, sid, length_scale, var_r), never on the species' list position."""
    if length_scale is None or var_r is None:
        return np.full((Ny, Nx), float(base_r))
    if seed_field is None:
        raise ValueError("seed_field is needed when length_scale and var_r are set")
    ss = np.random.SeedSequence([int(seed_field), int(sid)])
    return generate_spatial_r(1, Ny, Nx, length_scale, np.array([base_r], float), var_r, seed=ss)[0]


def niche_setup(habitat, niche, seed_field, length_scale, var_r):
    """Checks the niche-mode inputs; returns (H, traits(sid), field(sid))."""
    from environment import niche_traits, niche_r        # needs patch_environment_niche.py
    if length_scale is not None or var_r is not None:
        raise ValueError("niche mode: length_scale and var_r belong to field mode; leave them None")
    if seed_field is None:
        raise ValueError("niche mode: seed_field (seed of the species niches) is needed")
    if niche is None or not (set(NICHE_KEYS) <= set(niche) <= set(NICHE_KEYS) | set(NICHE_OPTIONAL)):
        raise ValueError(f"niche mode: niche must give {NICHE_KEYS} (optionally {NICHE_OPTIONAL}), got {None if niche is None else sorted(niche)}")
    H = np.asarray(habitat, dtype=float)
    if H.shape != (Ny, Nx) or not np.isfinite(H).all():
        raise ValueError(f"niche mode: habitat must be a finite ({Ny}, {Nx}) array, got shape {H.shape}")
    kw = {k: niche[k] for k in niche if k != "cost"}
    traits = lambda sid: niche_traits(sid, seed=seed_field, **kw)
    field = lambda sid: niche_r(H, *traits(sid), niche["cost"])
    return H, traits, field


def stepwise_assembly_ibm(
        *,
        base_r: float = 1.0,
        seed_size: int = 50,
        F_sat: float | None = 1e9,
        max_attempts: int | None = None,
        richness_cap: int | None = None,
        frac_multi: float = 0.05,
        window_steps: int = 5_000,
        record_step: int = 50,
        max_rounds: int | None,
        seed: int = 0,
        detection_threshold: float | None = None,
        checkpoint_fn=None,
        init_r=None, init_C=None, init_N=None, init_ids=None,
        init_attempts: int = 0, init_round: int = -1,
        length_scale=None, var_r=None, seed_field=None,
        habitat=None, niche=None,
        disp=None,
        **model_kw):

    ss_asm, ss_demo = np.random.SeedSequence(int(seed)).spawn(2)
    rng = np.random.default_rng(ss_asm)                 # interactions and founding cells
    demo = np.random.default_rng(ss_demo)               # demographic seed of each IBM run
    new_seed = lambda: int(demo.integers(0, 2**32 - 1))
    thr = THRESHOLD if detection_threshold is None else detection_threshold
    if habitat is None:
        if niche is not None:
            raise ValueError("niche settings given without a habitat layer")
        traits = None
        field = lambda sid: species_field(sid, base_r=base_r, length_scale=length_scale,
                                          var_r=var_r, seed_field=seed_field)
    else:
        H, traits, field = niche_setup(habitat, niche, seed_field, length_scale, var_r)
    model_kw.setdefault("record_mode", "none")
    d_of = None                                         # v2.2: per-species dispersal rate, or None
    if disp is not None:
        if set(disp) != {"d_min", "d_max"}:
            raise ValueError(f"disp must give d_min and d_max, got {sorted(disp)}")
        if not 0 < float(disp["d_min"]) <= float(disp["d_max"]):
            raise ValueError(f"disp needs 0 < d_min <= d_max, got {disp}")
        if seed_field is None:
            raise ValueError("disp: seed_field (seed of the species traits) is needed")
        if model_kw.get("dispersal_type", "propagule") != "propagule":
            raise ValueError("disp: per-species rates are for propagule dispersal only")
        if not hasattr(_disp, "set_species_rates"):
            raise ValueError("disp: dispersal.py lacks set_species_rates; apply patch_dispersal_per_species.py")
        _d_cache = {}
        def d_of(sid):
            sid = int(sid)
            if sid not in _d_cache:
                _d_cache[sid] = _disp.species_dispersal_rate(sid, seed=seed_field, d_min=float(disp["d_min"]),
                                                             d_max=float(disp["d_max"]))
            return _d_cache[sid]

    _patches_hist = [(0, 0), (0, min(Nx - 1, 1)), (min(Ny - 1, 1), 0)][:min(3, Ny * Nx)]
    hist = {k: [] for k in ("round", "attempts", "gamma", "alpha_bar", "alpha_pats", "est_cum", "pruned")}
    est_cum = 0

    resumed = (init_r is not None) and (init_C is not None) and (init_N is not None)
    if resumed:
        r = np.array(init_r, dtype=float, copy=True)
        C = np.array(init_C, dtype=float, copy=True)
        N = np.array(init_N, dtype=int, copy=True)
        ids = np.arange(len(r)) if init_ids is None else np.array(init_ids, dtype=np.int64)
        attempts, start_round = int(init_attempts), int(init_round) + 1
        log.info(f"[IBM] resume: gamma={len(r)} from round {init_round}")
    else:
        r = np.array([base_r], float)
        C = np.array([[1.0]], float)
        N = np.zeros((1, Ny, Nx), int)
        N[0, rng.integers(Ny), rng.integers(Nx)] = 1
        ids = np.array([0], dtype=np.int64)
        attempts, start_round = 0, 0
    next_id = int(ids.max()) + 1 if len(ids) else 1
    F = np.stack([field(i) for i in ids])

    # life table: one row per species id; residents at the start get round -1
    lt = {int(i): [-1, 1, -1, int((N[k] >= thr).sum())] for k, i in enumerate(ids)}   # round_in, established, round_out, peak cells
    pres_round, pres_ids, pres_bits, pres_count, pres_offsets = [], [], [], [], [0]

    rnd = start_round - 1
    rounds_iter = range(start_round, start_round + (max_rounds if max_rounds is not None else 10**12))
    for rnd in rounds_iter:
        n_cand = 1 + int(frac_multi * len(r))
        if max_attempts is not None:
            n_cand = min(n_cand, max_attempts - attempts)
            if n_cand <= 0:
                break

        r_big, C_big = r.copy(), C.copy()
        N_big = np.pad(N, [(0, n_cand), (0, 0), (0, 0)], constant_values=0)
        new_ids = np.arange(next_id, next_id + n_cand, dtype=np.int64); next_id += n_cand
        for add in range(n_cand):
            row, col = draw_interactions(len(r_big), rng=rng)
            r_big, C_big = expand_RC(r_big, C_big, base_r, row, col)
            patches = rng.choice(Ny * Nx, min(seed_size, Ny * Nx), replace=False)
            N_big[len(r) + add].flat[patches] = 1
        ids_big = np.r_[ids, new_ids]
        F_big = np.concatenate([F, np.stack([field(i) for i in new_ids])])

        model = IBMModel(r_big, C_big, r_field=F_big, nsteps=window_steps, record_step=record_step,
                         initial_N=N_big, seed=new_seed(), **model_kw)
        if d_of is None:
            model.run()
        else:                                           # v2.2: rates in the order of ids_big
            _disp.set_species_rates(np.array([d_of(i) for i in ids_big], dtype=float))
            try:
                model.run()
            finally:
                _disp.set_species_rates(None)
        N_fin = _to_host(model.N)
        attempts += n_cand

        estab_mask = (N_fin[-n_cand:] >= thr).any(axis=(1, 2))
        for i, ok in zip(new_ids.tolist(), estab_mask.tolist()):
            lt[i] = [int(rnd), int(ok), -1 if ok else int(rnd), 0]
        keep_all = np.r_[np.arange(len(r)), np.flatnonzero(estab_mask) + len(r)]
        r, C = r_big[keep_all], C_big[np.ix_(keep_all, keep_all)]
        N, F, ids = N_fin[keep_all], F_big[keep_all], ids_big[keep_all]
        established = int(estab_mask.sum())

        alive = (N >= thr).any(axis=(1, 2))
        pruned = int((~alive).sum())
        for i in ids[~alive].tolist():
            lt[i][2] = int(rnd)
        if pruned:
            r, C, N, F, ids, _ = prune_extinct(alive, r, C, N, F, ids)
        log.info(f"[IBM] round {rnd}: est={established}/{n_cand}, pruned={pruned}, gamma={len(r)} (thr={thr})")

        P = (N >= thr)
        cells = P.sum(axis=(1, 2))
        for i, c in zip(ids.tolist(), cells.tolist()):
            lt[i][3] = max(lt[i][3], int(c))
        pres_round.append(int(rnd)); pres_ids.append(ids.copy())
        pres_bits.append(np.packbits(P.reshape(len(ids), Ny * Nx), axis=1))
        pres_count.append(N.sum(axis=(1, 2)).astype(np.int64)); pres_offsets.append(pres_offsets[-1] + len(ids))

        rich_map = P.sum(axis=0)
        est_cum += established
        hist["round"].append(int(rnd)); hist["attempts"].append(int(attempts)); hist["gamma"].append(int(len(r)))
        hist["alpha_bar"].append(float(rich_map.mean()))
        hist["alpha_pats"].append([int(rich_map[y, x]) for (y, x) in _patches_hist])
        hist["est_cum"].append(int(est_cum)); hist["pruned"].append(pruned)

        if checkpoint_fn is not None:
            checkpoint_fn(dict(r=r, C=C, N=N, ids=ids, occ=(N >= thr).sum(axis=(1, 2)), attempts=attempts,
                               gamma=len(r), round=rnd, thr=thr))

        if (max_attempts is not None) and (attempts >= max_attempts):
            log.info(f"[IBM] stop: attempts = {attempts} >= max_attempts"); break
        if (richness_cap is not None) and (len(r) >= richness_cap):
            log.info(f"[IBM] stop: gamma = {len(r)} >= cap"); break
        if (F_sat not in (None, np.inf)) and attempts >= F_sat * len(r):
            log.info(f"[IBM] stop: attempts={attempts} >= {F_sat} x gamma"); break

    lt_id = np.array(sorted(lt), dtype=np.int64)
    lt_rows = np.array([lt[i] for i in lt_id.tolist()], dtype=np.int64).reshape(-1, 4)
    nb = (Ny * Nx + 7) // 8
    extra = dict(
        occ_counts=(N >= thr).sum(axis=(1, 2)), detection_threshold=thr, attempts_total=attempts,
        round_last=rnd, frac_multi=frac_multi, window_steps=window_steps, seed_size=seed_size,
        species_id=ids.astype(np.int64), r_field=F, assembly_seed=int(seed), final_run_seed=new_seed(),
        ASM_round=np.asarray(hist["round"], dtype=np.int64),
        ASM_attempts=np.asarray(hist["attempts"], dtype=np.int64),
        ASM_established=np.asarray(hist["est_cum"], dtype=np.int64),
        ASM_gamma=np.asarray(hist["gamma"], dtype=np.int64),
        ASM_pruned=np.asarray(hist["pruned"], dtype=np.int64),
        ASM_alpha_bar=np.asarray(hist["alpha_bar"], dtype=np.float32),
        ASM_alpha_patches=np.asarray(hist["alpha_pats"], dtype=np.int16),
        env_mode="field" if habitat is None else "niche",
        ASM_resumed_from_round=np.int64(start_round if resumed else -1),
        LT_id=lt_id, LT_round_in=lt_rows[:, 0], LT_established=lt_rows[:, 1].astype(np.uint8),
        LT_round_out=lt_rows[:, 2], LT_peak_cells=lt_rows[:, 3].astype(np.int32),
        ASM_P_round=np.asarray(pres_round, dtype=np.int64),
        ASM_P_offsets=np.asarray(pres_offsets, dtype=np.int64),
        ASM_P_ids=np.concatenate(pres_ids).astype(np.int64) if pres_ids else np.zeros(0, np.int64),
        ASM_P_bits=np.concatenate(pres_bits) if pres_bits else np.zeros((0, nb), np.uint8),
        ASM_P_count=np.concatenate(pres_count) if pres_count else np.zeros(0, np.int64),
    )
    if d_of is not None:                                # v2.2
        extra.update(disp_rate=np.array([d_of(i) for i in ids.tolist()], dtype=float),
                     LT_d=np.array([d_of(i) for i in lt_id.tolist()], dtype=float),
                     disp_range=np.array([float(disp["d_min"]), float(disp["d_max"])]))
    if traits is not None:
        T_now = np.array([traits(i) for i in ids.tolist()], dtype=float).reshape(-1, 3)
        T_all = np.array([traits(i) for i in lt_id.tolist()], dtype=float).reshape(-1, 3)
        extra.update(habitat=H, niche_opt=T_now[:, 0], niche_w=T_now[:, 1], niche_peak=T_now[:, 2],
                     LT_opt=T_all[:, 0], LT_w=T_all[:, 1], LT_peak=T_all[:, 2])
    return r, C, N, extra


def unpack_presence(extra_or_npz, round_index):
    """(ids, P) of the community at the end of the round_index-th recorded round,
    P of shape (len(ids), Ny, Nx), bool. Reads the ASM_P_* arrays of a run or a saved file."""
    z = extra_or_npz
    o = np.asarray(z["ASM_P_offsets"])
    a, b = int(o[round_index]), int(o[round_index + 1])
    bits = np.asarray(z["ASM_P_bits"])[a:b]
    P = np.unpackbits(bits, axis=1, count=Ny * Nx).astype(bool).reshape(b - a, Ny, Nx)
    return np.asarray(z["ASM_P_ids"])[a:b], P