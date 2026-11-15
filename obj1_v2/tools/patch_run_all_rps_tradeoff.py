"""
patch_run_all_rps_tradeoff.py  (obj1_v2, stage 2d-a, 28 Oct 2026)

Adds --niche-tradeoff (strength k of the specialist-generalist trade-off, default 1 = the
equal-area trade-off of worlds n1-n4) to work/run_all_rps.py. Needs environment.py with
patch_environment_tradeoff.py and assembly_stepwise_ibm.py v2.1.
 1. --niche-tradeoff k (k >= 0; niche mode only).
 2. k goes into the niche settings given to the assembly.
 3. World tag: "k<k>" after the niche part only when k is not 1, so n1-n4 tags do not change.
 4. Saved: NICHE_tradeoff.

Run from obj1_v2/work:
    python3 ../tools/patch_run_all_rps_tradeoff.py            # dry run, prints the diff
    python3 ../tools/patch_run_all_rps_tradeoff.py --apply
Every anchor must be found exactly once, or nothing is written.
"""
import difflib, sys
from pathlib import Path

PATH = Path("run_all_rps.py"); MARK = "# obj1_v2 tradeoff"
src = PATH.read_text()
if MARK in src:
    sys.exit("STOP: run_all_rps.py already has --niche-tradeoff")
if "# obj1_v2 niche" not in src:
    sys.exit("STOP: apply patch_run_all_rps_niche.py first")
if "tradeoff" not in Path("environment.py").read_text():
    sys.exit("STOP: environment.py has no trade-off strength; apply patch_environment_tradeoff.py first")
if "NICHE_OPTIONAL" not in Path("assembly_stepwise_ibm.py").read_text():
    sys.exit("STOP: assembly_stepwise_ibm.py is not v2.1; copy it from new_code first")
L = src.split("\n")


def find(pred, what):
    hits = [i for i, s in enumerate(L) if pred(s)]
    if len(hits) != 1:
        sys.exit(f"STOP: anchor '{what}' found {len(hits)} times (needs exactly 1); nothing written")
    return hits[0]


ind = lambda s: s[: len(s) - len(s.lstrip())]
edits = []
i = find(lambda s: s.strip().startswith("p.add_argument('--niche-peak-max'"), "--niche-peak-max option")
edits.append((i + 1, i + 1, [
    f"    p.add_argument('--niche-tradeoff', type=float, default=1.0,   {MARK}",
    "                   help='niche mode: trade-off strength k, peak = peak_max (w_min / width)^k; 1 = equal area, 0.5 softer')"]))
i = find(lambda s: "raise SystemExit('--habitat-seed and --habitat-file need --env-mode niche')" in s, "habitat-without-niche guard")
edits.append((i + 1, i + 1, [
    f"    if args.niche_tradeoff < 0:   {MARK}",
    "        raise SystemExit('--niche-tradeoff must be >= 0')",
    "    if args.env_mode != 'niche' and args.niche_tradeoff != 1.0:",
    "        raise SystemExit('--niche-tradeoff needs --env-mode niche')"]))
i = find(lambda s: s.strip() == "peak_max=NICHE_PEAK_MAX if args.niche_peak_max is None else args.niche_peak_max)", "niche_kw peak_max line")
edits.append((i, i + 1, [f"{ind(L[i])}peak_max=NICHE_PEAK_MAX if args.niche_peak_max is None else args.niche_peak_max,",
                         f"{ind(L[i])}tradeoff=args.niche_tradeoff)   {MARK}"]))
i = find(lambda s: s.strip() == "f\"p{_sanitize_slug(niche_kw['peak_max'])}\")", "tag peak part")
edits.append((i, i + 1, [f"{ind(L[i])}f\"p{{_sanitize_slug(niche_kw['peak_max'])}}\"",
                         f"{ind(L[i])}+ (f\"k{{_sanitize_slug(args.niche_tradeoff)}}\" if args.niche_tradeoff != 1.0 else \"\"))   {MARK}"]))
i = find(lambda s: s.strip() == "\"NICHE_peak_max\": _np.float32(niche_kw[\"peak_max\"]),", "saved NICHE_peak_max")
edits.append((i + 1, i + 1, [f"{ind(L[i])}\"NICHE_tradeoff\": _np.float32(args.niche_tradeoff),   {MARK}"]))

new = list(L)
for a, b, lines in sorted(edits, key=lambda e: e[0], reverse=True):
    new[a:b] = lines
out = "\n".join(new)
print("\n".join(difflib.unified_diff(L, new, "run_all_rps.py (now)", "run_all_rps.py (patched)", lineterm="", n=1)))
compile(out, "run_all_rps.py", "exec"); print(f"\n{len(edits)} edits; the edited file compiles")
if "--apply" in sys.argv:
    PATH.write_text(out); print("written: run_all_rps.py")
else:
    print("dry run: nothing written (add --apply to write)")