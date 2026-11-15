#!/usr/bin/env python3
"""
stage_a_fixab_history.py  (obj2_v2, stage A step 3, read-only, 7 Nov 2026)

What it does
    Reads the training histories (training_history.json, written by training.py every validation epoch) and the
    training log (train.log) of the old FIXAB runs, and prints, per file:
      H1 epochs covered and the curriculum phase of every epoch (training.py: phase 1 below phase1_epochs, 2 below
         phase2_epochs, 3 below phase3_epochs, else 4; boundaries are cumulative)
      H2 whether the Fix A extrapolation loss ever acted: fix_a_b_extrapolation.py adds losses['extrapolation'] only
         when phase >= 4, so the history holds train_extrapolation only for epochs that ran it
      H3 which loss terms were on (train_total, train_diffusion, train_prevalence, train_cooccurrence, train_spatial)
      H4 validation rare AUC per validation epoch and the best one (the metric that chose best_model.pt)
      L1 from train.log: phase boundaries, phase changes, Fix A / Fix B install lines, history K, batch size,
         early-stopping lines, resume lines
    Writes nothing except what it prints. Needs only the Python standard library.

Why
    Audit finding F11: if epochs 104 and 149 were phase 3, the Fix A loss never acted on the deployed model. Audit
    finding F13: the validation AUC is computed on pred_x0 from one denoising step of the noised target (training.py
    validate calls model(target, condition), which noises the target itself), so it measures denoising of the answer,
    not reconstruction from records; H4 shows the values that chose the checkpoint.

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_a_fixab_history.py --self-test | tail -1
    python3 ../tools/stage_a_fixab_history.py ../../stage2_outputs_SPATIAL_FIXAB_partial ../../stage2_outputs_SPATIAL_FIXAB_149 \
        ../../stage2_outputs_SPATIAL_FIXAB_final | tee ../logs/stage_a_fixab_history_7nov.txt
    (each argument is a run folder; the tool reads every training_history.json, FIXAB_history_*.json and train.log in it)
"""
import argparse, glob, json, math, os, re, sys, tempfile
from collections import Counter

LOG_PATTERNS = [
    ("phase boundaries", r"phase boundaries"), ("phase change", r"Entering Phase|=== Phase"),
    ("Fix A", r"Fix A installed|Fix A:"), ("Fix B", r"Fix B installed"), ("history K", r"hist sparsify K"),
    ("batch size", r"batch size:"), ("total epochs", r"total epochs:"), ("early stopping", r"Early stopping"),
    ("start or resume", r"Starting training from epoch|resumed from"), ("new best", r"new best"),
]


def ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def ranges(epochs):
    """[(first, last)] runs of consecutive epochs."""
    out = []
    for e in sorted(epochs):
        if out and e == out[-1][1] + 1:
            out[-1][1] = e
        else:
            out.append([e, e])
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in out)


def read_history(path):
    h = json.load(open(path))
    ep = [int(e) for e in h.get("epoch", [])]
    ph = [int(p) for p in h.get("phase", [])]
    print(f"\n== history {path}")
    if not ep or len(ep) != len(ph):
        print(f"H1 epoch list {len(ep)}, phase list {len(ph)}: cannot align (keys: {', '.join(sorted(h))})")
        return
    by = {}
    for e, p in zip(ep, ph):
        by.setdefault(p, []).append(e)
    print(f"H1 epochs {ep[0]}-{ep[-1]} ({len(ep)} epochs); phase of each epoch: "
          + "; ".join(f"phase {p}: epochs {ranges(v)}" for p, v in sorted(by.items())))
    ex = [v for v in h.get("train_extrapolation", []) if ok(v)]
    print(f"H2 Fix A extrapolation loss values in this history: {len(ex)}"
          + (" -> the Fix A loss acted" if ex else " -> the Fix A loss never acted in these epochs"))
    terms = [k for k in ("train_total", "train_diffusion", "train_prevalence", "train_cooccurrence", "train_spatial",
                         "train_extrapolation") if k in h]
    print("H3 loss terms on: " + ", ".join(f"{k[6:]} ({sum(ok(v) for v in h[k])} epochs)" for k in terms))
    auc = h.get("auc_rare", [])
    if auc:
        val_ep = [e for e in ep if (e + 1) % 5 == 0]
        i = max(range(len(auc)), key=lambda k: auc[k])
        where = f"epoch {val_ep[i]}" if len(val_ep) == len(auc) else f"validation number {i + 1} (epochs not aligned)"
        print(f"H4 validation rare AUC: {len(auc)} values, first {auc[0]:.4f}, last {auc[-1]:.4f}, best {auc[i]:.4f} at {where}")
    else:
        print("H4 no auc_rare in this history")


def read_log(path):
    print(f"\n== log {path}")
    lines = open(path, errors="replace").read().splitlines()
    for name, pat in LOG_PATTERNS:
        hits = [l.strip() for l in lines if re.search(pat, l)]
        shown = hits[:3] + (["..."] + hits[-2:] if len(hits) > 5 else hits[3:5])
        print(f"L1 {name}: {len(hits)} lines" + "".join(f"\n     {s[:160]}" for s in shown))


def scan(folder):
    files = sorted(set(glob.glob(f"{folder}/**/training_history.json", recursive=True))
                   | set(glob.glob(f"{folder}/**/FIXAB_history_*.json", recursive=True)))
    logs = sorted(glob.glob(f"{folder}/**/train.log", recursive=True))
    print(f"\n######## {folder}: {len(files)} histories, {len(logs)} logs")
    for f in files:
        read_history(f)
    for f in logs:
        read_log(f)


# ---------------------------------------------------------------- self-test
def planted(folder, last_epoch=149, start=0, b=(50, 100, 200), fix_a_from=None):
    os.makedirs(f"{folder}/checkpoints", exist_ok=True); os.makedirs(f"{folder}/logs", exist_ok=True)
    h = {"epoch": [], "phase": [], "train_total": [], "train_diffusion": [], "auc_rare": []}
    for e in range(start, last_epoch + 1):
        p = 1 if e < b[0] else 2 if e < b[1] else 3 if e < b[2] else 4
        h["epoch"].append(float(e)); h["phase"].append(float(p)); h["train_total"].append(0.1); h["train_diffusion"].append(0.05)
        if fix_a_from is not None and e >= fix_a_from:
            h.setdefault("train_extrapolation", []).append(0.3)
        if (e + 1) % 5 == 0:
            h["auc_rare"].append(0.99 if e == 104 else 0.94)
    json.dump(h, open(f"{folder}/checkpoints/training_history.json", "w"))
    open(f"{folder}/logs/train.log", "w").write(
        "  total epochs:     220  (phase boundaries 50/100/200/500)\n  batch size:       1\n"
        "   ✓ Fix A installed: ExtrapolationLoss (extrap_weight=0.5, presence_weight=20.0, phase>=4)\n"
        "=== Phase 1 ===\n=== Entering Phase 2 — early-stopping counter reset (was 3) ===\n")


def self_test():
    import contextlib, io
    res = []

    def t(name, cond, what):
        res.append(bool(cond)); print(f"{name} {what}: {'PASS' if cond else 'FAIL'}")
    with tempfile.TemporaryDirectory() as tmp:
        def run(folder):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                scan(folder)
            return buf.getvalue()
        planted(f"{tmp}/a")
        out = run(f"{tmp}/a")
        t("T1", "phase 3: epochs 100-149" in out and "phase 1: epochs 0-49" in out, "phases read from the history: epoch 149 in phase 3")
        t("T2", "Fix A loss never acted" in out, "no extrapolation values -> Fix A never acted")
        t("T3", "best 0.9900 at epoch 104" in out, "best validation rare AUC found at epoch 104")
        t("T4", "phase boundaries: 1 lines" in out and "Fix A: 1 lines" in out and "phase change: 2 lines" in out, "log lines found")
        planted(f"{tmp}/b", last_epoch=219, fix_a_from=200)
        out = run(f"{tmp}/b")
        t("T5", "phase 4: epochs 200-219" in out and "values in this history: 20 -> the Fix A loss acted" in out,
          "a run that reached phase 4 shows the Fix A loss acting in 20 epochs")
        planted(f"{tmp}/c", last_epoch=149, start=110)
        out = run(f"{tmp}/c")
        t("T6", "epochs 110-149 (40 epochs)" in out and "phase 3: epochs 110-149" in out, "a resumed history (110-149) is read as it is")
    print(f"{sum(res)} PASS, {len(res) - sum(res)} FAIL")
    return all(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not a.folders:
        sys.exit("STOP: give one or more run folders")
    for f in a.folders:
        if not os.path.isdir(f):
            sys.exit(f"STOP: {f} is not a folder")
        scan(f)