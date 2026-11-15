#!/usr/bin/env python3
"""
footprint_export.py  (obj1_v2, gate 5 cause check, first fix round for species per cell, turnover, 51+ share and
range-size TV; 1 Nov 2026, Sara's yes)

Why: in the gate 5 worlds (n70, n73-n75) species per window matches BNM (x0.93-1.09) but each species holds too small
and too similar a share of the window: species per cell x0.59-0.78, turnover x1.29-1.57 (1 / turnover 0.27-0.33 against
BNM 0.43), 51+ share 19.7-25.1% (BNM 30.7%), range sizes bunched at 6-50 cells (60-63% against 46%). On the 80
window-end maps the same sides hold (per cell 23 below, 0 above; turnover 38 above; 51+ 39 below, 1 above; TV 65
above). A species' range = the cells where it can grow (its footprint: saved growth rate above 0, set by niche width,
optimum and habitat) x the share of those it holds (set by competition and dispersal). This tool asks which part is short.

What it does: for each world file, writes a copy of the training file in which P_last_final is the footprint of every
final species (ENV_r_field > 0: the growth rate on the final habitat map, the map of the final run; habitat_track_check C0
checks the saved fields against the niche formula on that map), every other key unchanged. world_vs_bnm (unchanged,
--tau 1, presence from P_last_final only; frame_bands_export relies on the same) scores it through the same survey and
detection. A control copy (own end map) must reprint the end-map numbers. Species with an empty footprint are dropped
from the copy (counted and printed; expected 0, since no species lives on arrivals alone, sink_growth 30 Oct).

Reading rule (written 1 Nov, before the build; claude/n33_result_29oct.md):
  1. control copies reprint the end-map numbers (logs/world_vs_bnm_g5_1nov.txt), or nothing is read.
  2. niche side: footprint turnover above 3.06 or footprint 51+ share below 25.9% in 3 or more of 4 worlds -> the cells a
     species can grow in are themselves too few; the fix goes to niche width.
  3. filling side: footprint turnover 3.06 or less and 51+ share 25.9% or more in 3 or more of 4 worlds -> species could
     cover enough of the window but hold only part of it: competition sorting, the cause of the decay limit (row 25).
     Strength 0.2 and connectance 0.4 were tried (niche_p: turnover 3.25 and 3.02, window 60 and 57), so these bands go to
     Sara with that evidence, or one more fix round.
  4. neither in 3 of 4: discuss. Footprint TV, 1 cell share and species per cell reported.
  Expected: filling side (n66-n69 species fill 40-51% of their suitable cells).

Run from obj1_v2/work:
    python3 ../tools/footprint_export.py --self-test | tail -1
    python3 ../tools/footprint_export.py --out results/footprint_g5 "results/stage2a/niche_*[p5]/*_cn1p0_*_fb_ht62500_hs1p0t62500_hss25_*_training.npz"
Writes only under --out; reads the training files.
"""
import argparse, glob, sys, tempfile
from pathlib import Path
import numpy as np

NEED = ["P_last_final", "ENV_r_field"]


def footprint(z):
    """(footprint maps uint8 (S', Ny, Nx), kept row index, per-species numbers) or raises ValueError."""
    P = np.asarray(z["P_last_final"]).astype(bool)
    R = np.asarray(z["ENV_r_field"], float)
    if R.shape != P.shape:
        raise ValueError(f"fields {R.shape} and end map {P.shape} differ")
    FP = R > 0
    S = len(P)
    occ = P.reshape(S, -1).sum(1); fpn = FP.reshape(S, -1).sum(1); both = (P & FP).reshape(S, -1).sum(1)
    keep = np.flatnonzero(fpn > 0)
    info = dict(S=S, empty=int(S - len(keep)), occ=occ, fp=fpn,
                fill=both[keep] / fpn[keep], in_fp=both.sum() / max(occ.sum(), 1))
    return FP[keep].astype(np.uint8), keep, info


def export(f, out, control):
    z = np.load(f, allow_pickle=True)
    miss = [k for k in NEED if k not in z.files]
    if miss:
        return None, f"skipped {Path(f).name}: missing {miss}"
    try:
        FP, keep, info = footprint(z)
    except ValueError as e:
        return None, f"skipped {Path(f).name}: {e}"
    data = {k: z[k] for k in z.files}
    stem = Path(f).name[: -len("_training.npz")] if Path(f).name.endswith("_training.npz") else Path(f).stem
    data["P_last_final"] = FP
    np.savez_compressed(Path(out) / f"{stem}_fp_training.npz", **data)
    if control:
        data["P_last_final"] = np.asarray(z["P_last_final"])
        p = Path(out) / "control" / f"{stem}_fpend_training.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **data)
    info["name"] = Path(f).name
    return info, None


# ------------------------------------------------ self-test
def planted(path, rng, S=8, G=20, zero_cells=False, empty=False):
    """Species s grows (r > 0) in a disc of radius 2 + s/2 and holds a known part of it (every other cell) plus one sink."""
    yy, xx = np.mgrid[0:G, 0:G]
    R = np.full((S, G, G), -0.05); P = np.zeros((S, G, G), bool)
    for s in range(S):
        cy, cx = rng.integers(0, G, 2)
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= (2 + s / 2) ** 2
        R[s][disc] = 0.1
        P[s] = disc & ((yy + xx) % 2 == 0)
        out_y, out_x = np.argwhere(~disc)[0]
        P[s, out_y, out_x] = True                      # one sink cell outside the footprint
    if zero_cells:
        R[0, 0, 0] = 0.0                               # exactly 0: not a footprint cell (growth must be above 0)
        R[0, 0, 1] = 1e-9
    if empty:
        R[S - 1] = -0.05
    np.savez(path, P_last_final=P.astype(np.uint8), ENV_r_field=R.astype(np.float32), B_last=(P * 1e-4).astype(np.float16),
             SIM_SEED=np.int64(2303), HABITAT_shift_sd=np.float64(1.0), NICHE_w=np.linspace(0.2, 0.9, S))
    return P, R


def self_test():
    rng = np.random.default_rng(3); res = []

    def check(name, good, text):
        res.append(bool(good)); print(f"{name} {text}: {'PASS' if good else 'FAIL'}")

    with tempfile.TemporaryDirectory() as tmp:
        P, R = planted(f"{tmp}/w_training.npz", rng, zero_cells=True)
        out = Path(tmp) / "out"; out.mkdir()
        info, why = export(f"{tmp}/w_training.npz", out, control=True)
        z0 = np.load(f"{tmp}/w_training.npz"); c = np.load(out / "w_fp_training.npz")
        check("T1", why is None and np.array_equal(c["P_last_final"].astype(bool), R > 0),
              "copy holds every cell with growth rate above 0, species in the same order")
        check("T2", c["P_last_final"][0, 0, 0] == 0 and c["P_last_final"][0, 0, 1] == 1,
              "a cell with growth exactly 0 is left out, 1e-9 is kept (footprint = r > 0, as spatial_cause_check)")
        check("T3", all(np.array_equal(c[k], z0[k]) for k in z0.files if k != "P_last_final") and set(c.files) == set(z0.files),
              "every other key copied unchanged, none added or lost")
        e = np.load(out / "control" / "w_fpend_training.npz")
        check("T4", all(np.array_equal(e[k], z0[k]) for k in z0.files) and set(e.files) == set(z0.files),
              "control copy equals the training file key for key")
        S = len(P); fp = (R > 0).reshape(S, -1); occ = P.reshape(S, -1)
        fill = [(occ[s] & fp[s]).sum() / fp[s].sum() for s in range(S)]
        in_fp = (occ & fp).sum() / occ.sum()
        check("T5", np.allclose(info["fill"], fill) and abs(info["in_fp"] - in_fp) < 1e-12 and 0.4 < np.mean(fill) < 0.6,
              f"fill per species and share of occupied cells inside the footprint equal a direct loop "
              f"(mean fill {np.mean(fill):.2f}, in footprint {in_fp:.2f})")
        P2, R2 = planted(f"{tmp}/e_training.npz", rng, empty=True)
        info2, _ = export(f"{tmp}/e_training.npz", out, control=False)
        c2 = np.load(out / "e_fp_training.npz")
        check("T6", info2["empty"] == 1 and len(c2["P_last_final"]) == len(P2) - 1
              and np.array_equal(c2["P_last_final"].astype(bool), (R2 > 0)[:-1]),
              "a species with no cell above 0 is dropped and counted; the rest stay in order")
        np.savez(f"{tmp}/m_training.npz", P_last_final=np.zeros((2, 20, 20), np.uint8))
        np.savez(f"{tmp}/s_training.npz", P_last_final=np.zeros((2, 20, 20), np.uint8), ENV_r_field=np.zeros((3, 20, 20)))
        r = [export(f"{tmp}/{n}_training.npz", out, False)[1] or "" for n in ("m", "s")]
        check("T7", "missing" in r[0] and "differ" in r[1], "refused: no fields; fields and end map of different shapes")
        # broken versions: footprint from the end map (no change), r >= 0, species order reversed
        bad = [np.array_equal(P.astype(bool), R > 0), np.array_equal(R >= 0, R > 0), np.array_equal((R > 0)[::-1], R > 0)]
        check("T8", not any(bad), "broken versions differ from T1's expected copy: end map kept, r >= 0, species reversed")
    print(f"{sum(res)} of {len(res)} PASS" if all(res) else f"FAIL: {len(res) - sum(res)} of {len(res)}")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.out or not a.files:
        sys.exit("STOP: give --out DIR and one or more training-file patterns")
    files = sorted({f for p in a.files for f in glob.glob(p)})
    if not files:
        sys.exit("STOP: no file matched")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if any(Path(a.out).glob("*_training.npz")):
        sys.exit(f"STOP: {a.out} already holds copies; use an empty folder")
    for f in files:
        info, why = export(f, a.out, not a.no_control)
        if why:
            print(why); continue
        print(f"{info['name'][-60:]}: {info['S']} species ({info['empty']} with an empty footprint, dropped); "
              f"cells held median {int(np.median(info['occ']))}, footprint median {int(np.median(info['fp']))} of 400; "
              f"species hold {np.mean(info['fill']):.0%} of their footprint on average; "
              f"{info['in_fp']:.0%} of occupied cells inside the footprint")
    print(f"\nwritten to {a.out} (control copies in {a.out}/control)")