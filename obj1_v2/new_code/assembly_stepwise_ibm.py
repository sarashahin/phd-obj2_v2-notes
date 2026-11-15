############################################
# assembly_stepwise_ibm.py  (obj1_v2; v2 27 Oct 2026: niche mode, life table, window-end presence;
#   v2.1 28 Oct 2026: optional niche trade-off strength; v2.2 29 Oct 2026: optional dispersal rate per species;
#   v2.3 29 Oct 2026: optional LVMCM founding, founding="best"; v2.4 31 Oct 2026: optional habitat changing
#   slowly through time, habitat_change=dict(...); v2.5 31 Oct 2026: optional regional shift of the whole habitat
#   layer, habitat_change=dict(..., shift_sd=..., shift_tau=...))
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
   Species ids count invasion attempts, so the id is also the attempt of arrival (random founding; see v2.3).
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

v2.3 (29 Oct 2026, LVMCM founding; tests in tests/test_16_founding_best.py): founding="best" founds
newcomers as LVMCM does (O'Sullivan, LVMCM_src commit ba165f8: Metacommunity.cpp invaderSample lines
311-383, main.cpp line 647).
- Each candidate is tested before it enters: g(x) = r(x) - sum_j C_new,j B_j(x), with B = N x BODY_MASS of the
  residents at the start of the round (LVMCM tests invaders on resident biomass only; invaders hold none yet).
- It enters only if max g >= 0 (LVMCM min_b = 0), as one individual in the cell of highest g (the first such
  cell in row-major order; LVMCM also takes the first).
- A candidate that fails never enters the IBM. It keeps its id and a life-table row: LT_entered 0, round in =
  round out = the round of the test, LT_g_max its best g.
- Only entered newcomers count as attempts (LVMCM counts invasions this way), so max_attempts is the number
  of entered newcomers. Species ids count tested candidates; LT_attempt gives the attempt of each entered one
  (-1 for the others).
- The world starts empty, as in LVMCM: round 0 tests candidates against no residents (g = r), so the first
  species enters in its best cell. There is no founder of round -1.
- LVMCM keeps drawing until enough candidates pass. Here a round tries at most founding_cap candidates per
  newcomer needed and enters those that passed. The assembly stops when a round finds none (found_saturated 1,
  found_stop_round, logged); that round runs no window.
- Needs seed_size 1. The test draws no extra random numbers apart from the competition links of the candidates
  that fail.
founding="random" (default) is v2.2 unchanged: every result is the same (test_16 R1-R3), and no new key is returned.

v2.4 (31 Oct 2026, root cause 23; tests in tests/test_18_habitat_change.py): habitat_change=dict(tau=..., seed=...,
length_scale=..., contrast=...) lets the habitat change slowly through time (niche mode only). Why: in every niche
world about three newcomers were kept for each resident lost, against a balance in the field-mode twins (chain of
twins, richness_balance_check.py, 30 Oct), so richness never levelled off. Each species had fixed home cells and the
habitat never changed, so residents were almost never lost.
- At the start of every round after the first, the standardised layer U (H = contrast x U) moves one step:
  U <- rho U + sqrt(1 - rho^2) Z, with Z = environment.habitat_layer(length_scale, seed=[seed, 104729, round]) and
  rho = exp(-window_steps / tau), then U is standardised again (mean 0, sd 1), so the habitat keeps its contrast and
  spatial scale and only where each value sits moves. tau is the time scale of the change, in IBM steps.
- The growth fields of all residents are recomputed from the new H before anything else in the round; newcomers
  get fields on the current H, and best founding tests candidates on it. Fields stay fixed within a window.
- Z comes from its own random stream, so the candidates, competition links, founding cells and demographic seeds are
  drawn exactly as without the option; round 0 is the same as the static world.
- Returned: habitat = the map of the final fields (the one the final run uses), habitat_start, habitat_t (the map of
  every completed round, float32), habitat_rho, habitat_tau. Refused in field mode and with a resumed run.
habitat_change=None (default) is v2.3 unchanged: no random number is drawn and no key is added (test_18 R1-R3).

v2.5 (31 Oct 2026, root cause 23, fix B part 2; tests in tests/test_19_habitat_shift.py): habitat_change may also give
shift_sd and shift_tau (both or neither). Why: resident_loss_check (31 Oct) showed losses fall almost only on small
ranges and that under v2.4 no species ever had less than half its own habitat amount (the moving map is standardised
every round), so a large-range resident is never pushed down; position-only change brought best founding to 0.89 of
the loss rate a level needs under the newcomer rule, random founding to 0.40-0.42.
- The whole layer moves up or down together: H_t = contrast x (U_t + m_t), where U_t is v2.4's moving map (same
  random numbers) and m_t is one number per round, m_t = rho_m m_t-1 + sqrt(1 - rho_m^2) shift_sd eps_t, m_0 = 0,
  rho_m = exp(-window_steps / shift_tau), eps_t from its own stream [seed, 104743, round]. shift_sd is in units of
  U's sd (1). One end of the habitat range becomes more common for a while and the other rarer, so each species'
  habitat amount and its growth relative to other species change everywhere at once.
- Round 0 is the twin's round 0 (m_0 = 0); every U_t is the twin's; only H_t and the fields differ.
- Returned as well: habitat_shift_sd, habitat_shift_tau, habitat_shift_rho, habitat_shift_t (m of every completed round,
  the m of the map in habitat_t). Without shift_sd and shift_tau every result is v2.4's (test_19 S0).

v2.6 (1 Nov 2026, root cause 24, gate 7; tests in tests/test_20_habitat_substeps.py): habitat_change may also give
substeps m (a whole number >= 1 that divides window_steps). Why: frame_change_check (1 Nov) found the candidate setting
(habitat tau 62,500, shift sd 1.0) changes about 6.6 times faster per round than BNM per year, while the static twin
changes at BNM's rate; slowing the habitat per round breaks the balance that fixed gate 1 (n57, n58, n59, n65), so the
time steps are made finer instead.
- Each window runs as m pieces of window_steps / m steps on the same IBM model and random stream. The habitat moves
  before every piece except the very first of the run, so it moves once every window_steps / m steps; the step is
  v2.4/v2.5's with rho_s = exp(-(window_steps / m) / tau) (and the same for the shift), so the change per round is the
  same as before (rho_s^m = rho). Sub-step j of round t draws Z from [seed, 104729, t, j] and eps from
  [seed, 104743, t, j]. Between pieces the growth fields of every species in the window are recomputed from the new
  map and written into the model in place; nothing else is drawn or changed.
- A map is saved after every piece: ASM_F_round, ASM_F_piece, ASM_F_offsets, ASM_F_ids, ASM_F_bits (species with at
  least one cell at the detection threshold, packed as ASM_P_*), and ASM_F_shift (m of each piece, with a shift).
  The last piece of a round holds the same species and cells as ASM_P of that round (test_20 U4).
- substeps 1: every dynamic result is v2.5's (same random numbers, same keys) and ASM_F_* are added (one map per round).
  Without substeps every result and key is v2.5's (test_20 U1).
- habitat_t and habitat_shift_t keep one entry per round: the map of the round's last piece (the one its fields end on).
- The maps of the pieces are not saved; they follow from the seeds above.

Resuming from a checkpoint keeps species ids, so fields are rebuilt exactly; the
random streams restart from the seed (as in the original), so a resumed world is not
the world an uninterrupted run gives. Its history arrays start at the resume round
(ASM_resumed_from_round).
"""
from __future__ import annotations
import logging
import numpy as np
from models_ibm import IBMModel
from config import NUM_PATCHES_X, NUM_PATCHES_Y, THRESHOLD, BODY_MASS
from assembly_utils import draw_interactions, expand_RC, prune_extinct
from environment import generate_spatial_r
import dispersal as _disp                               # v2.2: per-species dispersal rates

try:
    import cupy as cp
    def _to_host(a):
        return cp.asnumpy(a) if isinstance(a, cp.ndarray) else a
    def _like(a, ref):                                  # v2.6: a on the device of ref
        return cp.asarray(a, dtype=ref.dtype) if isinstance(ref, cp.ndarray) else np.asarray(a, dtype=ref.dtype)
except Exception:
    def _to_host(a):
        return a
    def _like(a, ref):
        return np.asarray(a, dtype=ref.dtype)

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


def invasion_growth(field_new, row_res, N_res):
    """v2.3, LVMCM invader test (Metacommunity.cpp invaderSample): growth rate per cell of a newcomer when rare,
    g = r_new - sum_j C_new,j B_j, with B = N x BODY_MASS of the residents. row_res[j] is the effect of resident j
    on the newcomer, in the order of N_res. Returns (Ny, Nx)."""
    row_res = np.asarray(row_res, dtype=float)
    B = np.asarray(N_res, dtype=float).reshape(len(row_res), Ny * Nx) * BODY_MASS
    return np.asarray(field_new, dtype=float) - (row_res @ B).reshape(Ny, Nx)


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
        founding: str = "random", founding_cap: int = 100,
        habitat_change=None,
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
    hc = None                                           # v2.4: habitat changing through time, or None
    if habitat_change is not None:
        if habitat is None:
            raise ValueError("habitat_change needs niche mode (a habitat layer)")
        _hc_keys = set(habitat_change) if isinstance(habitat_change, dict) else set()
        if not isinstance(habitat_change, dict) or _hc_keys - {"shift_sd", "shift_tau", "substeps"} != {"tau", "seed", "length_scale", "contrast"} \
                or len(_hc_keys & {"shift_sd", "shift_tau"}) == 1:   # v2.5: shift_sd and shift_tau optional, both or neither
            raise ValueError(f"habitat_change must give tau, seed, length_scale and contrast (and optionally both shift_sd "
                             f"and shift_tau), got {habitat_change!r}")
        if "shift_sd" in _hc_keys and not (float(habitat_change["shift_sd"]) > 0 and float(habitat_change["shift_tau"]) > 0):
            raise ValueError(f"habitat_change: shift_sd and shift_tau must be > 0, got {habitat_change!r}")
        if not float(habitat_change["tau"]) > 0 or not float(habitat_change["length_scale"]) > 0 \
                or not float(habitat_change["contrast"]) > 0:
            raise ValueError(f"habitat_change: tau, length_scale and contrast must be > 0, got {habitat_change!r}")
        if init_r is not None:
            raise ValueError("habitat_change: a resumed run is not supported; rerun the world from the start")
        from environment import habitat_layer, niche_r
        hc = dict(tau=float(habitat_change["tau"]), seed=int(habitat_change["seed"]),
                  ls=float(habitat_change["length_scale"]), contrast=float(habitat_change["contrast"]))
        hc["rho"] = float(np.exp(-float(window_steps) / hc["tau"]))
        hc["m"] = None                                  # v2.6: habitat sub-steps per window, or None
        if "substeps" in _hc_keys:
            _m = habitat_change["substeps"]
            if isinstance(_m, bool) or int(_m) != _m or int(_m) < 1 or int(window_steps) % int(_m):
                raise ValueError(f"habitat_change: substeps must be a whole number >= 1 that divides window_steps "
                                 f"({window_steps}), got {_m!r}")
            hc["m"] = int(_m)
            hc["rho_s"] = float(np.exp(-float(window_steps) / hc["m"] / hc["tau"]))
        hc["U"] = np.asarray(H, dtype=float) / hc["contrast"]
        sh = None                                       # v2.5: regional shift of the whole layer, or None
        if "shift_sd" in _hc_keys:
            sh = dict(sd=float(habitat_change["shift_sd"]), tau=float(habitat_change["shift_tau"]), m=0.0)
            sh["rho"] = float(np.exp(-float(window_steps) / sh["tau"]))
            if hc["m"] is not None:                     # v2.6
                sh["rho_s"] = float(np.exp(-float(window_steps) / hc["m"] / sh["tau"]))
        H_cur = [np.asarray(H, dtype=float)]
        _cost = float(niche["cost"])
        field = lambda sid: niche_r(H_cur[0], *traits(sid), _cost)

        def habitat_step(rnd_, j=None):
            """One move of the layer: v2.4/v2.5's step (j None), or v2.6 sub-step j of round rnd_."""
            if j is None:
                Z = habitat_layer(Ny, Nx, hc["ls"], seed=[hc["seed"], 104729, int(rnd_)])
                rho = hc["rho"]
            else:
                Z = habitat_layer(Ny, Nx, hc["ls"], seed=[hc["seed"], 104729, int(rnd_), int(j)])
                rho = hc["rho_s"]
            U = rho * hc["U"] + np.sqrt(1.0 - rho ** 2) * Z
            hc["U"] = (U - U.mean()) / U.std()
            H_cur[0] = hc["contrast"] * hc["U"]
            if sh is not None:                          # v2.5: the whole layer moves up or down together
                if j is None:
                    eps = float(np.random.default_rng([hc["seed"], 104743, int(rnd_)]).standard_normal())
                    rho_m = sh["rho"]
                else:
                    eps = float(np.random.default_rng([hc["seed"], 104743, int(rnd_), int(j)]).standard_normal())
                    rho_m = sh["rho_s"]
                sh["m"] = rho_m * sh["m"] + np.sqrt(1.0 - rho_m ** 2) * sh["sd"] * eps
                H_cur[0] = hc["contrast"] * (hc["U"] + sh["m"])
        hab_hist, shift_hist = [], []
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
    best = founding == "best"                           # v2.3: LVMCM founding
    if founding not in ("random", "best"):
        raise ValueError(f"founding must be 'random' or 'best', got {founding!r}")
    if best:
        if int(seed_size) != 1:
            raise ValueError(f"founding 'best' places one individual in one cell (LVMCM); seed_size must be 1, got {seed_size}")
        try:
            cap_ok = int(founding_cap) == founding_cap and founding_cap >= 1
        except (TypeError, ValueError):
            cap_ok = False
        if not cap_ok:
            raise ValueError(f"founding_cap must be a whole number >= 1, got {founding_cap!r}")
        founding_cap = int(founding_cap)

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
    elif best:                                          # v2.3: LVMCM starts empty; round 0 tests against no residents
        r = np.zeros(0, float)
        C = np.zeros((0, 0), float)
        N = np.zeros((0, Ny, Nx), int)
        ids = np.zeros(0, dtype=np.int64)
        attempts, start_round = 0, 0
    else:
        r = np.array([base_r], float)
        C = np.array([[1.0]], float)
        N = np.zeros((1, Ny, Nx), int)
        N[0, rng.integers(Ny), rng.integers(Nx)] = 1
        ids = np.array([0], dtype=np.int64)
        attempts, start_round = 0, 0
    next_id = int(ids.max()) + 1 if len(ids) else (0 if (best and not resumed) else 1)
    F = np.stack([field(i) for i in ids]) if len(ids) else np.zeros((0, Ny, Nx))

    # life table: one row per species id; residents at the start get round -1
    lt = {int(i): [-1, 1, -1, int((N[k] >= thr).sum())] for k, i in enumerate(ids)}   # round_in, established, round_out, peak cells
    pres_round, pres_ids, pres_bits, pres_count, pres_offsets = [], [], [], [], [0]
    frm = dict(round=[], piece=[], ids=[], bits=[], off=[0], shift=[])   # v2.6: map after every piece
    lt_g, lt_att, lt_fail = {}, {}, set()               # v2.3 best: best g of each tested candidate, attempt of each entered one
    tested_total, saturated, stop_round, tested_hist = 0, 0, -1, []

    rnd = start_round - 1
    rounds_iter = range(start_round, start_round + (max_rounds if max_rounds is not None else 10**12))
    for rnd in rounds_iter:
        if hc is not None and rnd > start_round:        # v2.4: the habitat moves one step; residents' fields follow
            habitat_step(rnd, None if hc["m"] in (None, 1) else 0)
            F = np.stack([field(i) for i in ids]) if len(ids) else np.zeros((0, Ny, Nx))
        n_cand = 1 + int(frac_multi * len(r))
        if max_attempts is not None:
            n_cand = min(n_cand, max_attempts - attempts)
            if n_cand <= 0:
                break

        if best:                                        # v2.3: LVMCM test before entry; a failed candidate never enters
            r_big, C_big = r.copy(), C.copy()
            new_list, found_cells, tried = [], [], 0
            while len(new_list) < n_cand and tried < founding_cap * n_cand:
                sid = next_id; next_id += 1; tried += 1
                row, col = draw_interactions(len(r_big), rng=rng)
                g = invasion_growth(field(sid), row[:len(r)], N)       # against the residents only
                lt_g[sid] = float(g.max())
                if lt_g[sid] >= 0:
                    r_big, C_big = expand_RC(r_big, C_big, base_r, row, col)
                    new_list.append(sid); found_cells.append(int(np.argmax(g)))
                    lt_att[sid] = attempts + len(new_list)
                else:
                    lt[sid] = [int(rnd), 0, int(rnd), 0]; lt_fail.add(sid)
            tested_total += tried
            if not new_list:
                saturated, stop_round = 1, int(rnd)
                log.info(f"[IBM] stop: none of {tried} candidates can grow anywhere (round {rnd}); saturated")
                break
            n_cand = len(new_list)
            new_ids = np.array(new_list, dtype=np.int64)
            N_big = np.pad(N, [(0, n_cand), (0, 0), (0, 0)], constant_values=0)
            for add, c in enumerate(found_cells):
                N_big[len(r) + add].flat[c] = 1
        else:
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

        n_piece = 1 if (hc is None or hc["m"] is None) else hc["m"]     # v2.6: the window in n_piece pieces
        model = IBMModel(r_big, C_big, r_field=F_big, nsteps=window_steps // n_piece, record_step=record_step,
                         initial_N=N_big, seed=new_seed(), **model_kw)
        def run_window():
            nonlocal F_big
            for j in range(n_piece):
                if j > 0:                               # v2.6: the habitat moves; fields written into the model
                    habitat_step(rnd, j)
                    F_big = np.stack([field(i) for i in ids_big])
                    model.r_field[...] = _like(F_big, model.r_field)
                    model.r_flat = model.r_field.reshape(len(ids_big), -1)
                model.run()
                if hc is not None and hc["m"] is not None:
                    Nh = _to_host(model.N); Pj = (Nh >= thr).reshape(len(ids_big), -1); on = Pj.any(axis=1)
                    frm["round"].append(int(rnd)); frm["piece"].append(j); frm["ids"].append(ids_big[on].copy())
                    frm["bits"].append(np.packbits(Pj[on], axis=1)); frm["off"].append(frm["off"][-1] + int(on.sum()))
                    if sh is not None:
                        frm["shift"].append(sh["m"])
        if d_of is None:
            run_window()
        else:                                           # v2.2: rates in the order of ids_big
            _disp.set_species_rates(np.array([d_of(i) for i in ids_big], dtype=float))
            try:
                run_window()
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
        log.info(f"[IBM] round {rnd}: est={established}/{n_cand}, pruned={pruned}, gamma={len(r)} (thr={thr})"
                 + (f", tested {tried}" if best else ""))

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
        tested_hist.append(int(tested_total))           # v2.3 (returned with founding "best" only)
        if hc is not None:                              # v2.4: the map this round's window used
            hab_hist.append(H_cur[0].astype(np.float32))
            if sh is not None:                          # v2.5: the shift of that map
                shift_hist.append(sh["m"])

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
    if best:                                            # v2.3: LVMCM founding record
        extra.update(founding="best", founding_cap=founding_cap, tested_total=int(tested_total),
                     found_saturated=int(saturated), found_stop_round=int(stop_round),
                     LT_entered=np.array([0 if i in lt_fail else 1 for i in lt_id.tolist()], dtype=np.uint8),
                     LT_attempt=np.array([lt_att.get(i, -1) for i in lt_id.tolist()], dtype=np.int64),
                     LT_g_max=np.array([lt_g.get(i, np.nan) for i in lt_id.tolist()], dtype=float),
                     ASM_tested=np.asarray(tested_hist, dtype=np.int64))
    if hc is not None:                                  # v2.4
        extra.update(habitat_start=np.asarray(H, dtype=float), habitat_rho=hc["rho"], habitat_tau=hc["tau"],
                     habitat_t=np.stack(hab_hist) if hab_hist else np.zeros((0, Ny, Nx), np.float32))
        if sh is not None:                              # v2.5
            extra.update(habitat_shift_sd=sh["sd"], habitat_shift_tau=sh["tau"], habitat_shift_rho=sh["rho"],
                         habitat_shift_t=np.asarray(shift_hist, dtype=float))
        if hc["m"] is not None:                         # v2.6
            extra.update(habitat_substeps=hc["m"], ASM_F_round=np.asarray(frm["round"], dtype=np.int64),
                         ASM_F_piece=np.asarray(frm["piece"], dtype=np.int64),
                         ASM_F_offsets=np.asarray(frm["off"], dtype=np.int64),
                         ASM_F_ids=np.concatenate(frm["ids"]).astype(np.int64) if frm["ids"] else np.zeros(0, np.int64),
                         ASM_F_bits=np.concatenate(frm["bits"]) if frm["bits"] else np.zeros((0, nb), np.uint8))
            if sh is not None:
                extra.update(ASM_F_shift=np.asarray(frm["shift"], dtype=float))
        H = H_cur[0]                                    # the map of the final fields, for the final run
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