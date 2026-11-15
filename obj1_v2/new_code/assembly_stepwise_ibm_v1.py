############################################
# assembly_stepwise_ibm.py  (obj1_v2)
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

Resuming from a checkpoint keeps species ids, so fields are rebuilt exactly; the
random streams restart from the seed (as in the original).
"""
from __future__ import annotations
import logging
import numpy as np
from models_ibm import IBMModel
from config import NUM_PATCHES_X, NUM_PATCHES_Y, THRESHOLD
from assembly_utils import draw_interactions, expand_RC, prune_extinct
from environment import generate_spatial_r

try:
    import cupy as cp
    def _to_host(a):
        return cp.asnumpy(a) if isinstance(a, cp.ndarray) else a
except Exception:
    def _to_host(a):
        return a

log = logging.getLogger(__name__)
Ny, Nx = NUM_PATCHES_Y, NUM_PATCHES_X


def species_field(sid, *, base_r=1.0, length_scale=None, var_r=None, seed_field=None):
    """Growth-rate field (Ny, Nx) of species `sid`. It depends only on
    (seed_field, sid, length_scale, var_r), never on the species' list position."""
    if length_scale is None or var_r is None:
        return np.full((Ny, Nx), float(base_r))
    if seed_field is None:
        raise ValueError("seed_field is needed when length_scale and var_r are set")
    ss = np.random.SeedSequence([int(seed_field), int(sid)])
    return generate_spatial_r(1, Ny, Nx, length_scale, np.array([base_r], float), var_r, seed=ss)[0]


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
        **model_kw):

    ss_asm, ss_demo = np.random.SeedSequence(int(seed)).spawn(2)
    rng = np.random.default_rng(ss_asm)                 # interactions and founding cells
    demo = np.random.default_rng(ss_demo)               # demographic seed of each IBM run
    new_seed = lambda: int(demo.integers(0, 2**32 - 1))
    thr = THRESHOLD if detection_threshold is None else detection_threshold
    field = lambda sid: species_field(sid, base_r=base_r, length_scale=length_scale,
                                      var_r=var_r, seed_field=seed_field)
    model_kw.setdefault("record_mode", "none")

    _patches_hist = [(0, 0), (0, min(Nx - 1, 1)), (min(Ny - 1, 1), 0)][:min(3, Ny * Nx)]
    hist = {k: [] for k in ("round", "attempts", "gamma", "alpha_bar", "alpha_pats", "est_cum", "pruned")}
    est_cum = 0

    if (init_r is not None) and (init_C is not None) and (init_N is not None):
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
    next_id = int(ids.max()) + 1
    F = np.stack([field(i) for i in ids])

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
        model.run()
        N_fin = _to_host(model.N)
        attempts += n_cand

        estab_mask = (N_fin[-n_cand:] >= thr).any(axis=(1, 2))
        keep_all = np.r_[np.arange(len(r)), np.flatnonzero(estab_mask) + len(r)]
        r, C = r_big[keep_all], C_big[np.ix_(keep_all, keep_all)]
        N, F, ids = N_fin[keep_all], F_big[keep_all], ids_big[keep_all]
        established = int(estab_mask.sum())

        alive = (N >= thr).any(axis=(1, 2))
        pruned = int((~alive).sum())
        if pruned:
            r, C, N, F, ids, _ = prune_extinct(alive, r, C, N, F, ids)
        log.info(f"[IBM] round {rnd}: est={established}/{n_cand}, pruned={pruned}, gamma={len(r)} (thr={thr})")

        rich_map = (N >= thr).sum(axis=0)
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
    )
    return r, C, N, extra
