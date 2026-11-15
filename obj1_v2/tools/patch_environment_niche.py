"""
patch_environment_niche.py  (obj1_v2, stage 2d-a first draft, 27 Oct 2026)

Adds three functions to environment.py for the niche model agreed on 27 Oct (2d-a). Nothing
that exists changes, so every current run behaves as before until the assembly is wired to
them (2d-a draft part B).

  habitat_layer(ny, nx, length_scale, seed=..., path=None)
      one shared habitat layer H, mean 0 and sd 1: a periodic Gaussian random field with the
      spectrum of generate_spatial_r, or a real map read from a .npy file (other datasets).
  niche_traits(sid, seed=..., opt_sd, w_min, w_max, peak_max)
      optimum, width and peak growth of species sid, drawn from (seed, sid) only, as the
      fields are now (Step 1). Width is log-uniform in [w_min, w_max]; peak x width is the
      same for every species (equal area, agreed 27 Oct), the narrowest niche peaking at
      peak_max: specialists grow fast where the habitat fits, generalists slowly almost
      everywhere.
  niche_r(H, opt, w, peak, cost)
      growth rate peak exp(-(H - opt)^2 / (2 w^2)) - cost; negative where the habitat does
      not fit, so the species dies there unless immigrants keep arriving.

peak_max must stay below 2. models_ibm.py updates counts so that the expected count next step
is N exp(g), g = r - C B (survivors times births); for one species alone this is
B' = B exp(r - B), whose fixed point B* = r is stable only while |1 - r| < 1. test_11 N4
checks this on the real IBM.

Run from obj1_v2/work:
    python3 ../tools/patch_environment_niche.py            # dry run, prints the diff
    python3 ../tools/patch_environment_niche.py --apply
    USE_GPU=0 python3 ../tests/test_11_niche.py            # about 2 minutes on CPU
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("environment.py"); MARK = "# obj1_v2 niche"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: environment.py already has the niche functions")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

NEW = f'''

{MARK} (27 Oct 2026): shared habitat layer and niche growth rates (stage 2d-a)
NICHE_PEAK_MAX = 1.5      # highest growth rate per step; must stay below 2 (see niche_traits)


def habitat_layer(ny, nx, length_scale, *, seed=None, path=None):
    """One shared habitat layer H (ny, nx) with mean 0 and sd 1.
    Simulated: periodic Gaussian random field with the spectrum used by generate_spatial_r
    (length_scale in cells; above about 5 it changes little on a 20 x 20 grid).
    path: a real map in a .npy file of shape (ny, nx), standardised the same way."""
    if path is not None:
        H = np.asarray(np.load(path), dtype=float)
        if H.shape != (ny, nx):
            raise ValueError(f"habitat map {{path}} has shape {{H.shape}}, expected {{(ny, nx)}}")
        if not np.isfinite(H).all():
            raise ValueError(f"habitat map {{path}} has missing or infinite values")
    else:
        if length_scale is None or length_scale <= 0:
            raise ValueError(f"habitat length_scale must be > 0, got {{length_scale}}")
        rng = np.random.default_rng(seed)
        spectrum = _sqrt_exp_spectrum(ny, nx, length_scale, 1.0)
        Z = rng.normal(size=(ny, nx // 2 + 1)) + 1j * rng.normal(size=(ny, nx // 2 + 1))
        H = irfft2(spectrum * Z, s=(ny, nx))
    sd = H.std()
    if not sd > 0:
        raise ValueError("habitat map is flat (sd 0)")
    return (H - H.mean()) / sd


def niche_traits(sid, *, seed, opt_sd=1.5, w_min=0.2, w_max=1.0, peak_max=NICHE_PEAK_MAX):
    """Niche of species sid: (optimum, width, peak). Drawn from (seed, sid) only, so a
    species keeps its niche whatever its place in the species list.
    optimum ~ normal(0, opt_sd) on the habitat scale (sd 1), so some species specialise on
    rare habitat; width log-uniform in [w_min, w_max]; peak = peak_max * w_min / width, so
    peak x width is the same for all (equal area) and no peak exceeds peak_max.
    peak_max < 2: the IBM's expected update for one species alone is B' = B exp(r - B),
    stable at B* = r only while |1 - r| < 1."""
    if not (0 < w_min <= w_max):
        raise ValueError(f"need 0 < w_min <= w_max, got {{w_min}}, {{w_max}}")
    if not (0 < peak_max < 2):
        raise ValueError(f"peak_max must be in (0, 2) for a stable update, got {{peak_max}}")
    if opt_sd < 0:
        raise ValueError(f"opt_sd must be >= 0, got {{opt_sd}}")
    rng = np.random.default_rng([int(seed), 7919, int(sid)])
    opt = float(rng.normal(0.0, opt_sd))
    w = float(np.exp(rng.uniform(np.log(w_min), np.log(w_max)))) if w_max > w_min else float(w_min)
    return opt, w, float(peak_max * w_min / w)


def niche_r(H, opt, w, peak, cost):
    """Growth rate on habitat H: peak * exp(-(H - opt)^2 / (2 w^2)) - cost."""
    if w <= 0:
        raise ValueError(f"niche width must be > 0, got {{w}}")
    if cost < 0:
        raise ValueError(f"cost must be >= 0, got {{cost}}")
    return peak * np.exp(-(np.asarray(H, dtype=float) - opt) ** 2 / (2.0 * w ** 2)) - cost'''

i = find(lambda s: s.strip() == "return out.astype(np.float64)", "end of generate_spatial_r")
new = L[: i + 1] + NEW.split("\n") + L[i + 1:]
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "environment.py (now)", "environment.py (patched)", lineterm="", n=1)))
compile(out, "environment.py", "exec"); print("\n1 insertion; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: environment.py")
else:
    print("dry run: nothing written (add --apply to write)")
