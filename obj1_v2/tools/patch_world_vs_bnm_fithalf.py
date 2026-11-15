"""
patch_world_vs_bnm_fithalf.py  (obj1_v2, 25 Oct 2026)

Makes world_vs_bnm.py read only the geographic fit half of the BNM windows, for both
--bnm and --survey. Sara's rule (25 Oct): one evaluation technique, no leak. Objectives
3 and 4 score on the test half of the geographic split, so Stage 2 (choosing d, kernel,
founding, environment) must never open a test-half window.

The split is the one steps 13, 14 and 19 use: fit = window easting below the median of
the distinct eastings of the reference set (default: bnm_grids/minlist8, the scoring
set). Any BNM folder (minlist 4 or 12 too) is cut with that reference: a window east
of the cut is left unopened, and so is a western window that shares cells with a
test window. File names are step 10's bnm_e<km>_n<km>; the window side is
window_cells x cell_m from a fit-half file.

The alternate split (every other window) shares windows with the fit half, so after
Stage 2 it is no longer a clean test; only the geographic test half stays untouched.

Run from obj1_v2/work:
    python3 ../tools/patch_world_vs_bnm_fithalf.py            # dry run, prints the diff
    python3 ../tools/patch_world_vs_bnm_fithalf.py --apply
    python3 ../tools/world_vs_bnm.py --self-test              # 13 PASS
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("../tools/world_vs_bnm.py"); MARK = "# obj1_v2 fit-half"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: world_vs_bnm.py already reads the fit half only")
L = src.split("\n")

def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]

SPLIT = f'''

{MARK} (25 Oct 2026): Stage 2 reads the geographic fit half of the BNM windows only.
REF_DEFAULT = "../../AI_simulation/Empirical_data/bnm_grids/minlist8/bnm_*.npz"
REF = None                                   # set from --split-ref before any BNM file is read


def window_en(f):
    """Window origin (easting, northing) in km from step 10's file name bnm_e<km>_n<km>."""
    s = Path(f).stem
    return int(s.split("_e")[1].split("_")[0]), int(s.split("_n")[1].split("_")[0])


def split_ref(pattern):
    """The geographic split of steps 13, 14 and 19, set on the reference windows (the
    Objective 3/4 scoring set): fit = easting below the median of the distinct eastings.
    Only one fit-half file is opened, to read the window side."""
    files = sorted(glob.glob(pattern))
    if not files: sys.exit(f"STOP: no reference BNM files matched {{pattern}}; the fit half cannot be set")
    eas = sorted({{window_en(f)[0] for f in files}}); cut = eas[len(eas) // 2]
    fit = [f for f in files if window_en(f)[0] < cut]
    if not fit: sys.exit("STOP: the reference set has no fit-half window")
    z = np.load(fit[0], allow_pickle=True)
    side = (int(z["window_cells"]) * float(z["cell_m"]) / 1000) if {{"window_cells", "cell_m"}} <= set(z.files) \\
        else z["obs_mask"].shape[-1] * 2.0
    return dict(cut=cut, side=side, n=len(files), test=[window_en(f) for f in files if window_en(f)[0] >= cut])


def fit_half(files, ref):
    """Windows Stage 2 may read: west of the cut and sharing no cell with a test window.
    Decided from file names alone; nothing is opened."""
    keep, east, overlap = [], 0, 0
    for f in sorted(files):
        e, n = window_en(f)
        if e >= ref["cut"]: east += 1
        elif any(abs(e - te) < ref["side"] and abs(n - tn) < ref["side"] for te, tn in ref["test"]): overlap += 1
        else: keep.append(f)
    return keep, east, overlap


def bnm_glob(pattern):
    """BNM files for Stage 2: the geographic fit half only. Test-half files are never opened."""
    files = sorted(glob.glob(pattern))
    keep, east, overlap = fit_half(files, REF)
    print(f"BNM files {{pattern}}: {{len(files)}} matched; {{len(keep)}} fit-half windows read; {{east}} test-half windows and "
          f"{{overlap}} sharing cells with a test window left unopened (cut at easting {{REF['cut']}} km, set on "
          f"{{REF['n']}} reference windows)")
    return keep'''

TEST = '''    import tempfile                                   # obj1_v2 fit-half: the split and that test windows stay unopened
    global REF
    with tempfile.TemporaryDirectory() as tmp:
        win = [(400, 200), (440, 200), (440, 240), (480, 240), (520, 240)]
        for e, n in win:
            o = np.zeros((3, G, G), bool)
            if e >= 480: o[:] = True                  # test half: every cell occupied (would show if read)
            else: o[:, 0, 0] = True                   # fit half: one cell per species
            np.savez(f"{tmp}/bnm_e{e:04d}_n{n:04d}.npz", obs_mask=o, known_target=np.ones_like(o), window_cells=G, cell_m=2000)
        REF = split_ref(f"{tmp}/bnm_*.npz")
        eas = sorted({e for e, _ in win}); cut19 = eas[len(eas) // 2]       # step 19's split_fit rule, written out
        want = sorted(f"bnm_e{e:04d}_n{n:04d}.npz" for e, n in win if e < cut19)
        keep, east, ov = fit_half(glob.glob(f"{tmp}/bnm_*.npz"), REF)
        good = sorted(Path(f).name for f in keep) == want and east == 2 and ov == 0 and REF["side"] == 40.0; ok &= good
        print(f"self-test, fit half = step 19's rule (easting below the median), test half left out: {'PASS' if good else 'FAIL'}")
        other = [f"x/bnm_e{e:04d}_n{n:04d}.npz" for e, n in ((470, 240), (300, 100), (500, 300))]
        keep, east, ov = fit_half(other, REF)
        good = [Path(f).name for f in keep] == ["bnm_e0300_n0100.npz"] and east == 1 and ov == 1; ok &= good
        print(f"self-test, a window from another folder that shares cells with a test window is refused: {'PASS' if good else 'FAIL'}")
        files, cases = load_bnm(f"{tmp}/bnm_*.npz")
        s = summarise(cases, wrap=False)
        good = len(files) == 3 and s["R_med"] == 1 and len(survey_masks(f"{tmp}/bnm_*.npz")) == 3; ok &= good
        print(f"self-test, --bnm and --survey read fit-half windows only (median range {s['R_med']:.0f}; 400 if a "
              f"test window were read): {'PASS' if good else 'FAIL'}")
    REF = None'''

edits = []
i = find(lambda s: s.strip() == "import argparse, glob, hashlib, sys", "imports")
edits.append((i + 1, i + 1, [f"from pathlib import Path   {MARK}"]))
i = find(lambda s: s.strip() == "STRUCT = ndimage.generate_binary_structure(2, 1)", "STRUCT")
edits.append((i + 1, i + 1, SPLIT.split("\n")))
i = find(lambda s: s.strip() == "files = sorted(glob.glob(pattern)); cases = []", "load_bnm glob")
edits.append((i, i + 1, [L[i].replace("sorted(glob.glob(pattern))", "bnm_glob(pattern)") + f"   {MARK}"]))
i = find(lambda s: s.strip() == "files = sorted(glob.glob(pattern))", "survey_masks glob")
edits.append((i, i + 1, [L[i].replace("sorted(glob.glob(pattern))", "bnm_glob(pattern)") + f"   {MARK}"]))
i = find(lambda s: s.strip() == "return ok", "end of self_test")
edits.append((i, i, TEST.split("\n")))
i = find(lambda s: s.strip() == 'ap.add_argument("--each", action="store_true")', "--each option")
edits.append((i + 1, i + 1, [f'ap.add_argument("--split-ref", default=REF_DEFAULT, help="BNM windows that set the geographic '
                             f'split (the Objective 3/4 scoring set)")   {MARK}']))
i = find(lambda s: s.strip() == "sys.exit(0 if self_test() else 1)", "self-test exit")
edits.append((i + 1, i + 1, [f"REF = split_ref(a.split_ref) if (a.bnm or a.survey) else None   {MARK}"]))
i = find(lambda s: s.strip() == 'report(f"BNM ({len(files)} files)", target)', "BNM report label")
edits.append((i, i + 1, [L[i].replace('f"BNM ({len(files)} files)"', 'f"BNM, geographic fit half ({len(files)} windows)"')]))
i = find(lambda s: s.strip() == "Run from obj1_v2/work (both arguments are optional; quote the patterns):", "docstring run line")
edits.append((i, i, ["Version 3 (25 Oct): BNM files (--bnm and --survey) are read from the geographic fit half only,",
                     "  the split of steps 13, 14 and 19 set on --split-ref (default minlist8); test-half windows and",
                     "  windows sharing cells with them are never opened. Pass bands are to be set again from the fit half."]))

new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "world_vs_bnm.py (now)", "world_vs_bnm.py (patched)", lineterm="", n=1)))
compile(out, "world_vs_bnm.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: ../tools/world_vs_bnm.py")
else:
    print("dry run: nothing written (add --apply to write)")
