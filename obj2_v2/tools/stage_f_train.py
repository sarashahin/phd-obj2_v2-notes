#!/usr/bin/env python3
"""
stage_f_train.py  (obj2_v2, stage F steps 1-2, 8 Nov 2026; v4)

Training loop for obj2_model.py, with the early stopping Sara adopted on 8 Nov and the stage F gates, plus a smoke run
that measures the cost on her GPU before any real run is fixed (claude/obj2_v2_plan.md, "Stage F plan").

    data        training draws from tools/stage_e_inputs.py (stage D weights; offset and blank from a stream seeded by
                the epoch); targets = the 6 true maps of the drawn stack, from the real_train truth files, as +1 / -1
    loss        noise-prediction error over window cells (obj2_model.Diffusion.loss); class weights act through the
                draws only, so they are not applied twice
    weights     AdamW, learning rate 2e-4; an exponential moving average of the weights (0.999) is the model that is
                validated and kept
    validation  every val_every epochs: the frozen real_val set of stage D (all 16,347 stacks, or the fast subset),
                built by stage E from inputs only; K samples per stack by DDIM from a fixed starting noise; score =
                per-stack AUC of the sample frequency at the target year over window cells with no record at the
                target year, mean per true class (1, 2-5, 6-10, 11-20, 21-50, 51+), then the mean of the classes;
                AOO ratio (expected sampled cells / true cells at the target year) per true class is logged beside it
    stopping    stop when the score has not risen by more than min_delta for `patience` validations in a row; no
                resets; best.pt (by the score) and last.pt (model, moving average, optimizer, stopping state, epoch);
                the random stream of epoch e is seeded by (seed, e), so a run resumed from last.pt repeats an
                unbroken run; one CSV row per epoch; the stop reason and chosen epoch at the end of the log

Gates (rules written before the run)
    F1  file-open log: training opens real_train files only, validation real_val files only, real_eval never
    F2  predictions are built with no truth file open (guard) and their md5 written before the truth is read;
        a model that ignores its inputs (the untrained model, whose output layer starts at zero) scores 0.50 within
        0.02 (smoke run and self-test); no cell recorded at the target year is scored
    ES1 a planted score sequence stops exactly `patience` validations after its best and keeps the best
    ES2 a run stopped halfway and resumed from last.pt ends with the same weights, the same best epoch and the same
        log rows as an unbroken run (self-test, on the CPU, where the arithmetic is deterministic)
    ES4 the validation inputs are bit-identical at every validation (md5 per validation in the CSV), and the first
        64 frozen stacks give the stage E tensor md5 (0001385421f0602bd3c66584f5d1b022 on Sara's data)
    F3-F6 (null-input control, memorisation, model against the smoother, seeds and learning curve) run after training.

v3 (8 Nov 2026, Sara: save checkpoints, best, logs and what later stages need). Saving and logging only: the training
arithmetic, CFG and last.pt are unchanged, so a run started with v2 (26642116) resumes with v3.
    out/run_info.json      configuration; for every start or resume: time, device, GPU, torch and CUDA versions, md5 of
                           the five tools and of the two stage D files, the epoch it started from; at the end the stop
                           reason, best epoch and best score. A resume stops if obj2_model.py, stage_e_inputs.py or a
                           stage D file has another md5 than at the first start (training would change under the run).
    out/log.csv            rewritten after every epoch (v2 wrote it at the end only)
    out/best_val_pred.npz  at every new best: sampled counts (0..K) per canvas cell for every frozen stack, at the target
                           year, with the epoch and the prediction md5, so stage F step 3 and the trace pages can read the
                           best predictions without sampling again
    printed lines are flushed at once, so a log written through nohup shows each epoch as it ends

v4 (8 Nov 2026, Sara: no duplication when a run is restarted; seed 1 with v2 died during its first validation, after
pred_md5.txt got its epoch-9 line and before last.pt of epoch 9). Training arithmetic and CFG unchanged.
    a fresh start (no --resume) stops if the folder already holds files of a run (last.pt, best.pt, log.csv, ...), so
        two runs never mix in one folder and a later --resume can never pick up an old last.pt
    a resume first drops what a run that died left after its last saved epoch: pred_md5.txt lines of epochs not in
        last.pt, and best candidates of an epoch not in last.pt
    a new best is written as best.cand.pt and best_val_pred.cand.npz and becomes best.pt and best_val_pred.npz only
        after last.pt of the same epoch is saved, so best.pt always matches the stopping state in last.pt
    validation: only an error raised by the truth guard counts as an F2 failure; any other error (CUDA, memory) is
        raised as it is, with its own message (v2-v3 caught every RuntimeError of the seed-7 sampling as F2)

Run (from PSD_Dispersal_pool/obj2_v2/work)
    python3 ../tools/stage_f_train.py --self-test | tail -1
    python3 ../tools/stage_f_train.py --smoke --data ../data --out ../runs/smoke | tee ../logs/stage_f_smoke_8nov.txt
    python3 ../tools/stage_f_train.py --train --seed 1 --data ../data --out ../runs/f_seed1     (step 2; --resume continues)
Needs torch and numpy, and obj2_model.py, stage_e_inputs.py, stage_d_sampler.py and export_obj2.py in the same folder.
"""
import argparse, copy, csv, hashlib, importlib.util, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name):
    p = HERE / f"{name}.py"
    if not p.is_file():
        sys.exit(f"STOP: {p} not found; stage_f_train.py needs {name}.py in the same folder")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


E = _load("stage_e_inputs")
M = _load("obj2_model")
D, X = E.D, E.X
G, W, PAD = X.G, X.WINDOW, E.PAD
SCORE = [1, 2, 3, 4, 5, 6]
EXPECT_FPRINT = "0001385421f0602bd3c66584f5d1b022"
CFG = dict(base=40, emb=160, T=1000, batch=256, lr=2e-4, ema=0.999, draws=200000, val_every=10, patience=5, cap=200,
           min_delta_floor=0.002, K=8, ddim=50, seed=1, val_set="full", val_chunk=128)
# step 2 settings (Sara's smoke run, 8 Nov: 1.1 min per epoch, 9.8 min per full validation): the full frozen set every
# 10 epochs; patience 5 validations = 50 epochs = a quarter of the cap of 200; min_delta = the larger of 0.002 and twice
# the score difference between sampler seeds 7 and 8 on the first validation, then fixed for the run
BROKEN = os.environ.get("STAGE_F_BROKEN", "")           # self-test only: plants one bug so its gate must fail


# ---------------------------------------------------------------- data
class Targets:
    """Packed true maps of one split (training targets), in the stage D file order."""
    def __init__(self, d, files):
        man = {r["input_file"]: r for r in D.rows_of(d)}
        self.maps = []
        for f in files:
            with np.load(Path(d) / man[str(f)]["truth_file"], allow_pickle=False) as z:
                self.maps.append(z["true_maps"])

    def get(self, fi, rw, offset):
        n = len(fi)
        T = np.zeros((n, W, G, G), bool)
        for i in np.unique(fi):
            k = np.nonzero(fi == i)[0]
            T[k] = X.unpack(self.maps[i][rw[k]])
        return E.place(T, offset)


def draws_key(seed, epoch):
    return seed * 100000 + epoch


def epoch_batches(TR, TG, sampler, cfg, epoch):
    idx, off, blank = E.train_draws(sampler, draws_key(cfg["seed"], epoch), cfg["draws"])
    fi, rw = sampler["file_index"][idx], sampler["row"][idx]
    for a in range(0, cfg["draws"], cfg["batch"]):
        s = slice(a, min(a + cfg["batch"], cfg["draws"]))
        x, nums = TR.get(fi[s], rw[s], off[s], blank[s])
        y = TG.get(fi[s], rw[s], off[s])
        yield x, nums, 2.0 * y - 1.0


def val_select(frozen, which):
    return np.nonzero(frozen["fast"].astype(bool))[0] if which == "fast" else np.arange(len(frozen["row"]))


# ---------------------------------------------------------------- validation
@torch.no_grad()
def predict(model, diff, VA, frozen, sel, cfg, dev, seed):
    """Sample frequency at the target year (n, 24, 24) from inputs only; md5 of the inputs and of the frequencies."""
    voff, vblank = E.frozen_draws(frozen)
    fi, rw = frozen["file_index"], frozen["row"]
    K, ch = cfg["K"], cfg["val_chunk"]
    freq = np.zeros((len(sel), E.CANVAS, E.CANVAS), np.float32)
    hin = hashlib.md5()
    model.eval()
    for j, a in enumerate(range(0, len(sel), ch)):
        k = sel[a:a + ch]
        x, nums = VA.get(fi[k], rw[k], voff[k], vblank[k])
        if BROKEN == "truth_in_predict":
            E.targets(VA.d, VA.files, fi[k], rw[k], voff[k])
        if BROKEN == "cuda_error":
            raise RuntimeError("planted CUDA error (self-test)")
        hin.update(x.tobytes()); hin.update(nums.tobytes())
        cond = torch.from_numpy(x).to(dev).repeat_interleave(K, 0)
        nm = torch.from_numpy(nums).to(dev).repeat_interleave(K, 0)
        g = torch.Generator(device=dev).manual_seed(seed * 1000003 + j)
        noise = torch.randn((len(cond), W, E.CANVAS, E.CANVAS), generator=g, device=dev)
        x0 = diff.ddim(model, cond, nm, noise, steps=cfg["ddim"])
        occ = (x0[:, W - 1] > 0).float().view(len(k), K, E.CANVAS, E.CANVAS).mean(1)
        freq[a:a + len(k)] = occ.cpu().numpy()
    return freq, hin.hexdigest(), hashlib.md5(freq.tobytes()).hexdigest()


def stack_auc(score, lab):
    pos = score[lab]; neg = np.sort(score[~lab])
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    lt = np.searchsorted(neg, pos, "left"); le = np.searchsorted(neg, pos, "right")
    return (lt.sum() + 0.5 * (le - lt).sum()) / (len(pos) * len(neg))


def score(freq, VA, frozen, sel, val_dir):
    """Truth read here, after the predictions: class-balanced AUC over unrecorded window cells, per-class AUC and
    AOO ratio, and the number of recorded cells among the scored cells (must be 0)."""
    voff, _ = E.frozen_draws(frozen)
    fi, rw = frozen["file_index"][sel], frozen["row"][sel]
    T, tc = E.targets(val_dir, frozen["files"], fi, rw, voff[sel])
    V, _, _ = VA.arrays(fi, rw)
    rec_t = E.place(V[:, W - 1:W], voff[sel])[:, 0] > 0
    win = E.place(np.ones((len(sel), 1, G, G), bool), voff[sel])[:, 0] > 0
    scored = win & ~rec_t if BROKEN != "score_recorded" else win
    truth = T[:, W - 1] > 0
    a = np.array([stack_auc(freq[i][scored[i]], truth[i][scored[i]]) for i in range(len(sel))])
    per, aoo, var = {}, {}, []
    for c in SCORE:
        m = (tc == c) & ~np.isnan(a)
        per[D.TRUE[c]] = float(a[m].mean()) if m.any() else float("nan")
        if m.any():
            var.append(a[m].var() / m.sum())
        mt = tc == c
        aoo[D.TRUE[c]] = float((freq[mt] * win[mt]).sum((1, 2)).mean() / max(truth[mt].sum((1, 2)).mean(), 1e-9)) if mt.any() else float("nan")
    vals = [v for v in per.values() if v == v]
    return dict(score=float(np.mean(vals)) if vals else float("nan"), per=per, aoo=aoo,
                se=float(np.sqrt(np.sum(var)) / len(var)) if var else float("nan"),
                rec_scored=int((scored & rec_t).sum()), left=int(np.isnan(a).sum()))


# ---------------------------------------------------------------- early stopping
class EarlyStop:
    def __init__(self, patience, min_delta=None):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.bad, self.prev = -np.inf, None, 0, -np.inf

    def update(self, s, epoch):
        if s > self.best + self.min_delta:
            self.best, self.best_epoch, self.bad = s, epoch, 0
            out = "best"
        else:
            self.bad += 1
            if BROKEN == "patience_reset" and s > self.prev:
                self.bad = 0
            out = "stop" if self.bad >= self.patience else "wait"
        self.prev = s
        return out

    def state(self):
        return dict(best=self.best, best_epoch=self.best_epoch, bad=self.bad, prev=self.prev, min_delta=self.min_delta)

    def load(self, st):
        self.best, self.best_epoch, self.bad, self.prev = st["best"], st["best_epoch"], st["bad"], st["prev"]
        self.min_delta = st["min_delta"]


def es_planted_check():
    """ES1: a planted sequence must stop exactly `patience` validations after its best and keep the best."""
    seq = [0.50, 0.60, 0.62, 0.61, 0.623, 0.60, 0.59, 0.70]
    es = EarlyStop(patience=3, min_delta=0.005)
    stop_at = None
    for i, s in enumerate(seq):
        if es.update(s, i) == "stop":
            stop_at = i
            break
    return stop_at == 5 and es.best_epoch == 2 and abs(es.best - 0.62) < 1e-12


# ---------------------------------------------------------------- training
def save_atomic(obj, path):
    tmp = str(path) + ".part"
    torch.save(obj, tmp); os.replace(tmp, path)


def save_npz_atomic(path, **arrays):
    tmp = str(path)[:-4] + ".part.npz"
    np.savez(tmp, **arrays); os.replace(tmp, path)


def write_csv(rows, path):
    keys = sorted({k for r in rows for k in r})
    tmp = str(path) + ".part"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def guarded_predict(model, diff, VA, frozen, sel, cfg, dev, seed):
    """predict() under the truth guard: ((freq, md5 in, md5 out), None), or (None, message) if a truth file was opened.
    Any other error is raised unchanged."""
    with D.TruthGuard() as guard:
        try:
            return predict(model, diff, VA, frozen, sel, cfg, dev, seed), None
        except RuntimeError as e:
            if not guard.opened:
                raise
            return None, str(e)


RUN_FILES = ("last.pt", "best.pt", "best_val_pred.npz", "best.cand.pt", "best_val_pred.cand.npz", "pred_md5.txt",
             "log.csv", "run_info.json")
CANDS = (("best.cand.pt", "best.pt"), ("best_val_pred.cand.npz", "best_val_pred.npz"))


def clean_for_resume(out, start, best_epoch):
    """Drop what a run that died left after its last saved epoch (start = that epoch + 1, 0 if no last.pt):
    pred_md5.txt lines of epochs >= start; best candidates are promoted if last.pt was saved with them as its best,
    else deleted. Returns what it did, for the log."""
    did = []
    pm = out / "pred_md5.txt"
    if pm.is_file():
        lines = [l for l in pm.read_text().splitlines() if l.strip()]
        keep = [l for l in lines if int(l.split()[0]) < start]
        if len(keep) < len(lines):
            pm.write_text("".join(k + "\n" for k in keep))
            did.append(f"pred_md5.txt: {len(lines) - len(keep)} line(s) of epochs not in last.pt dropped")
    c = out / "best.cand.pt"
    if c.is_file():
        e = torch.load(c, map_location="cpu", weights_only=False)["epoch"]
        if start > 0 and e == best_epoch and e < start:
            for a, b in CANDS:
                if (out / a).is_file():
                    os.replace(out / a, out / b)
            did.append(f"best candidate of epoch {e} promoted (last.pt holds it as best)")
        else:
            for a, _ in CANDS:
                (out / a).unlink(missing_ok=True)
            did.append(f"best candidate of epoch {e} dropped (not in last.pt)")
    elif (out / "best_val_pred.cand.npz").is_file():
        (out / "best_val_pred.cand.npz").unlink()
        did.append("stray best_val_pred.cand.npz dropped")
    return did


TOOLS = ("stage_f_train", "obj2_model", "stage_e_inputs", "stage_d_sampler", "export_obj2")
DATA_FILES = ("stage_d_train_sampler.npz", "stage_d_frozen_val.npz")
FIXED = ("obj2_model", "stage_e_inputs")           # must not change between the starts of one run


def start_record(cfg, data, dev, resume, start):
    return dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), resume=bool(resume), from_epoch=int(start), device=str(dev),
                gpu=torch.cuda.get_device_name(0) if str(dev).startswith("cuda") else "cpu", torch=torch.__version__,
                cuda=torch.version.cuda, md5={n: X.file_md5(HERE / f"{n}.py") for n in TOOLS},
                data_md5={n: X.file_md5(Path(data) / n) for n in DATA_FILES})


def train(cfg, data, out_dir, dev, log=print, resume=False, stop_after=None, train_dir=None, val_dir=None):
    train_dir = train_dir or str(Path(data) / "real_train")
    val_dir = val_dir or str(Path(data) / "real_val")
    for p in (data, train_dir, val_dir, out_dir):
        if "real_eval" in str(p):
            sys.exit("STOP: real_eval is read once, after training")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    used = [f for f in RUN_FILES if (out / f).exists()]
    if used and not resume:
        sys.exit(f"STOP: {out} already holds {', '.join(used)} from another run; move that folder aside for a fresh "
                 "start, or add --resume to continue it")
    sampler = dict(np.load(Path(data) / "stage_d_train_sampler.npz"))
    frozen = dict(np.load(Path(data) / "stage_d_frozen_val.npz"))
    epoch = -1
    with _OpenLog() as flog:
        TR = E.Inputs(train_dir, sampler["files"]); TG = Targets(train_dir, sampler["files"])
        if BROKEN == "train_reads_val":                 # a training phase that opens a validation file must fail F1
            np.load(next(Path(val_dir).glob("*_truth.npz"))).close()
        n_train_open = len(flog.paths)
        VA = E.Inputs(val_dir, frozen["files"])
        torch.manual_seed(cfg["seed"])
        model = M.UNet(cfg["base"], cfg["emb"]).to(dev)
        ema = copy.deepcopy(model).eval()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0)
        diff = M.Diffusion(cfg["T"]).to(dev)
        es = EarlyStop(cfg["patience"])
        start, rows, stop_reason = 0, [], None
        epoch = start - 1
        if resume and (out / "last.pt").is_file():
            ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
            if ck["cfg"] != cfg:
                sys.exit("STOP: last.pt was written with another configuration")
            model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
            es.load(ck["es"]); start, rows = ck["epoch"] + 1, ck["rows"]
            epoch = start - 1
        if resume:
            for line in clean_for_resume(out, start, es.best_epoch):
                log(f"   resume: {line}")
        info_p = out / "run_info.json"
        rec = start_record(cfg, data, dev, resume, start)
        info = json.loads(info_p.read_text()) if (start > 0 and info_p.is_file()) else dict(cfg=cfg, starts=[])
        if start > 0 and not info["starts"]:
            info["note"] = "run started by a version without run_info.json; md5 from this resume on"
        if info["starts"]:
            first = info["starts"][0]
            changed = [n for n in FIXED if first["md5"][n] != rec["md5"][n]] + \
                      [n for n in DATA_FILES if first["data_md5"][n] != rec["data_md5"][n]]
            if changed:
                sys.exit(f"STOP: {', '.join(changed)} changed since the first start of this run; "
                         "start a new run in a new folder")
        info["starts"].append(rec)
        info_p.write_text(json.dumps(info, indent=1))
        sel = val_select(frozen, cfg["val_set"])
        if str(dev).startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        p_model, p_ema = list(model.parameters()), list(ema.parameters())
        log(f"   steps per epoch {-(-cfg['draws'] // cfg['batch'])} (batch {cfg['batch']}, no accumulation; R20); "
            f"parameters {M.n_params(model):,}; device {dev}")
        gen = torch.Generator(device=dev)
        if BROKEN == "rng_global":
            gen.manual_seed(cfg["seed"])
        for epoch in range(start, cfg["cap"]):
            if BROKEN != "rng_global":
                gen.manual_seed(cfg["seed"] * 1000003 + epoch)
            model.train(); t0 = time.time(); tot, nb = 0.0, 0
            for x, nums, y in epoch_batches(TR, TG, sampler, cfg, epoch):
                xc, nm, y0 = (torch.from_numpy(v).to(dev) for v in (x, nums, y))
                loss = diff.loss(model, y0, xc, nm, gen)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                with torch.no_grad():
                    torch._foreach_mul_(p_ema, cfg["ema"]); torch._foreach_add_(p_ema, p_model, alpha=1 - cfg["ema"])
                tot += float(loss.detach()); nb += 1
            row = dict(epoch=epoch, train_loss=tot / max(nb, 1), seconds=round(time.time() - t0, 1))
            if (epoch + 1) % cfg["val_every"] == 0:
                got, err = guarded_predict(ema, diff, VA, frozen, sel, cfg, dev, seed=7)
                if err:
                    log(f"F2 truth file opened while the predictions were built: FAIL ({err})")
                    return dict(F2=False, rows=rows)
                freq, h_in, h_pred = got
                with (out / "pred_md5.txt").open("a") as fp:
                    fp.write(f"{epoch} {h_pred}\n")
                sc = score(freq, VA, frozen, sel, val_dir)
                if es.min_delta is None:                # first validation: score noise from a second sampler seed
                    got8, err = guarded_predict(ema, diff, VA, frozen, sel, cfg, dev, seed=8)
                    if err:
                        log(f"F2 truth file opened while the predictions were built: FAIL ({err})")
                        return dict(F2=False, rows=rows)
                    s8 = score(got8[0], VA, frozen, sel, val_dir)["score"]
                    es.min_delta = max(cfg["min_delta_floor"], 2 * abs(sc["score"] - s8))
                    row.update(score_seed8=round(s8, 5), min_delta=round(es.min_delta, 5))
                    log(f"   min_delta set at the first validation: seeds 7 and 8 give {sc['score']:.4f} and {s8:.4f}; "
                        f"min_delta {es.min_delta:.4f} (twice the difference, at least {cfg['min_delta_floor']}), fixed for the run")
                state = es.update(sc["score"], epoch)
                row.update(val_score=round(sc["score"], 5), val_in_md5=h_in, pred_md5=h_pred, rec_scored=sc["rec_scored"],
                           **{f"auc_{k}": round(v, 4) for k, v in sc["per"].items()},
                           **{f"aoo_{k}": round(v, 3) for k, v in sc["aoo"].items()}, es=state)
                if state == "best":                     # candidates; promoted once last.pt of this epoch is saved
                    save_atomic(dict(ema=ema.state_dict(), cfg=cfg, epoch=epoch, score=sc["score"]), out / "best.cand.pt")
                    save_npz_atomic(out / "best_val_pred.cand.npz", counts=np.rint(freq * cfg["K"]).astype(np.uint8),
                                    sel=sel, epoch=epoch, K=cfg["K"], score=sc["score"], pred_md5=h_pred, in_md5=h_in)
                if state == "stop":
                    stop_reason = f"no rise above {es.min_delta:.4f} for {cfg['patience']} validations"
            rows.append(row)
            write_csv(rows, out / "log.csv")
            log("   " + json.dumps(row))
            if BROKEN.startswith("die_e") and epoch == int(BROKEN[5:]):
                sys.exit("planted death before last.pt (self-test)")
            save_atomic(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), es=es.state(),
                             epoch=epoch, rows=rows, cfg=cfg), out / "last.pt")
            for a, b in CANDS:                          # last.pt now holds this epoch's best: promote
                if (out / a).is_file():
                    os.replace(out / a, out / b)
            if stop_reason or (stop_after is not None and epoch + 1 >= stop_after):
                break
        if not stop_reason and not (stop_after is not None and epoch + 1 >= stop_after):
            stop_reason = f"epoch cap {cfg['cap']}"
    if rows:
        write_csv(rows, out / "log.csv")
    info["end"] = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), last_epoch=int(epoch), stop_reason=stop_reason,
                       best_epoch=es.best_epoch, best=None if not np.isfinite(es.best) else float(es.best),
                       min_delta=es.min_delta)
    info_p.write_text(json.dumps(info, indent=1))
    tr_paths = flog.paths[:n_train_open]
    later = flog.paths[n_train_open:]
    inside = lambda p, d: str(p).startswith(str(Path(d).resolve()) + os.sep)
    f1 = all(inside(p, train_dir) for p in tr_paths) and all(inside(p, val_dir) for p in later) \
        and not any("real_eval" in p for p in flog.paths) and Path(train_dir).resolve() != Path(val_dir).resolve()
    hs = {r["val_in_md5"] for r in rows if "val_in_md5" in r}
    res = dict(F1=f1, F2=all(r.get("rec_scored", 0) == 0 for r in rows), ES4=len(hs) <= 1, rows=rows,
               stop_reason=stop_reason, best_epoch=es.best_epoch, best=es.best,
               state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    log(f"F1 files opened: {len(tr_paths)} for training (all in real_train {all(inside(p, train_dir) for p in tr_paths)}), "
        f"{len(later)} for validation (all in real_val {all(inside(p, val_dir) for p in later)}), real_eval 0: {'PASS' if f1 else 'FAIL'}")
    log(f"stop: {stop_reason}; best epoch {es.best_epoch}, best score {es.best:.4f}")
    return res


class _OpenLog:
    """Logs every np.load path while active (gate F1)."""
    def __enter__(self):
        self.orig, self.paths = np.load, []
        np.load = self._load
        return self

    def _load(self, f, *a, **k):
        self.paths.append(str(Path(f).resolve()))
        return self.orig(f, *a, **k)

    def __exit__(self, *exc):
        np.load = self.orig
        return False


# ---------------------------------------------------------------- smoke run on the GPU
def smoke(data, out_dir, steps=300, expect=EXPECT_FPRINT, cpu=False, out=print):
    res = {}

    def gate(name, ok, text):
        res[name] = bool(ok); out(f"{name} {text}: {'PASS' if ok else 'FAIL'}")

    out(f"stage_f_train.py md5 {X.file_md5(__file__)}; obj2_model.py md5 {X.file_md5(HERE / 'obj2_model.py')}; "
        f"stage_e_inputs.py md5 {X.file_md5(HERE / 'stage_e_inputs.py')}")
    out(f"   torch {torch.__version__}, CUDA build {torch.version.cuda}, CUDA available {torch.cuda.is_available()}")
    dev = "cpu"
    if torch.cuda.is_available() and not cpu:
        name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
        try:
            y = torch.nn.functional.conv2d(torch.randn(2, 4, 24, 24, device="cuda"), torch.randn(8, 4, 3, 3, device="cuda"))
            ok = bool(torch.isfinite(y).all()); dev = "cuda"
        except Exception as e:                        # a torch build without kernels for this GPU fails here
            ok = False; out(f"   GPU kernel test error: {str(e)[:300]}")
        gate("GPU", ok, f"{name}, compute capability {cap[0]}.{cap[1]}, a convolution runs on it")
        if not ok:
            return res
    cfg = dict(CFG)
    sampler = dict(np.load(Path(data) / "stage_d_train_sampler.npz"))
    frozen = dict(np.load(Path(data) / "stage_d_frozen_val.npz"))
    voff, vbl = E.frozen_draws(frozen)
    with D.TruthGuard():
        VA = E.Inputs(Path(data) / "real_val", frozen["files"])
        x64, _ = VA.get(frozen["file_index"][:64], frozen["row"][:64], voff[:64], vbl[:64])
    fp = hashlib.md5(x64.tobytes()).hexdigest()
    gate("ES4", fp == expect, f"the first 64 frozen stacks give the stage E tensor md5 {fp} (expected {expect})")
    torch.manual_seed(cfg["seed"])
    model = M.UNet(cfg["base"], cfg["emb"]).to(dev)
    diff = M.Diffusion(cfg["T"]).to(dev)
    out(f"   model parameters {M.n_params(model):,} (base {cfg['base']}, embedding {cfg['emb']})")
    sel = val_select(frozen, "fast")
    t0 = time.time()
    with D.TruthGuard():
        freq, _, _ = predict(model, diff, VA, frozen, sel, cfg, dev, seed=7)
    t_fast = time.time() - t0
    sc0 = score(freq, VA, frozen, sel, Path(data) / "real_val")
    tol = max(0.02, 4 * sc0["se"])
    gate("F2", abs(sc0["score"] - 0.5) <= tol and sc0["rec_scored"] == 0,
         f"untrained model (output layer at zero, so it ignores its inputs) scores {sc0['score']:.3f} on the fast subset "
         f"(0.50 within {tol:.3f}, the larger of 0.02 and 4 standard errors); recorded cells scored {sc0['rec_scored']}")
    TR = E.Inputs(Path(data) / "real_train", sampler["files"]); TG = Targets(Path(data) / "real_train", sampler["files"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    gen = torch.Generator(device=dev).manual_seed(1)
    c2 = dict(cfg, draws=steps * cfg["batch"])
    losses, t_data, t_step = [], 0.0, 0.0
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    it = epoch_batches(TR, TG, sampler, c2, 0)
    model.train()
    for i in range(steps):
        a = time.time(); x, nums, y = next(it); t_data += time.time() - a
        a = time.time()
        xc, nm, y0 = (torch.from_numpy(v).to(dev) for v in (x, nums, y))
        loss = diff.loss(model, y0, xc, nm, gen)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
        t_step += time.time() - a
        losses.append(float(loss.detach()))
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
    out(f"   {steps} training steps of {cfg['batch']}: loss first 20 {np.mean(losses[:20]):.4f}, last 20 {np.mean(losses[-20:]):.4f}; "
        f"per step {1000 * t_data / steps:.1f} ms building inputs + {1000 * t_step / steps:.1f} ms on the {dev}; peak GPU memory {mem:.2f} GB")
    s = []
    t0 = time.time()
    for seed in (7, 8):
        with D.TruthGuard():
            f_, _, _ = predict(model, diff, VA, frozen, sel, cfg, dev, seed=seed)
        s.append(score(f_, VA, frozen, sel, Path(data) / "real_val")["score"])
    t_val = (time.time() - t0) / 2
    n_full = len(frozen["row"])
    per_epoch = cfg["draws"] / cfg["batch"] * (t_data + t_step) / steps
    out(f"   after {steps} steps, fast subset ({len(sel)} stacks, K {cfg['K']}, DDIM {cfg['ddim']}): score {s[0]:.4f} with sampler seed 7, "
        f"{s[1]:.4f} with seed 8 (difference {abs(s[0] - s[1]):.4f}); {t_val:.0f} s per validation (untrained: {t_fast:.0f} s)")
    out(f"   estimates: one epoch of {cfg['draws']:,} draws {per_epoch / 60:.1f} min; validation on the full frozen set "
        f"({n_full} stacks) {t_val * n_full / len(sel) / 60:.1f} min, on the fast subset {t_val / 60:.1f} min")
    meas = dict(device=dev, params=M.n_params(model), ms_data=1000 * t_data / steps, ms_step=1000 * t_step / steps,
                peak_gb=mem, fast_val_s=t_val, full_val_s=t_val * n_full / len(sel), epoch_s=per_epoch,
                seed_scores=s, seed_diff=abs(s[0] - s[1]), untrained_score=sc0["score"], fprint=fp)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "smoke.json").write_text(json.dumps(meas, indent=1))
    out(f"   written {Path(out_dir) / 'smoke.json'}")
    n_ok = sum(res.values())
    out(f"stage F smoke: {n_ok} PASS, {len(res) - n_ok} FAIL")
    return res


# ---------------------------------------------------------------- self-test (CPU, tiny model, planted data)
def self_test(tools):
    quiet = lambda *a, **k: None
    results = []

    def t(name, ok, what):
        results.append(bool(ok)); print(f"{name} {what}: {'PASS' if ok else 'FAIL'}")

    torch.manual_seed(0)
    big = M.UNet(CFG["base"], CFG["emb"])
    small = M.UNet(8, 32)
    xt = torch.randn(3, W, 24, 24); cond = torch.randn(3, M.N_COND, 24, 24); nums = torch.randn(3, 2)
    o = small(xt, cond, torch.tensor([0, 10, 999]), nums)
    t("U1", o.shape == (3, W, 24, 24) and float(o.detach().abs().max()) == 0.0 and 2e6 <= M.n_params(big) <= 4e6,
      f"output 6 x 24 x 24, zero before training; default model {M.n_params(big):,} parameters (2-4 M agreed)")
    diff = M.Diffusion(1000)
    x0 = torch.where(torch.rand(4, W, 24, 24) < 0.3, 1.0, -1.0)
    cond = torch.zeros(4, M.N_COND, 24, 24); cond[:, M.MASK_CH, 2:22, 1:21] = 1
    m = cond[:, M.MASK_CH:M.MASK_CH + 1]
    x0 = x0 * m
    noise = torch.randn_like(x0)

    class Oracle(torch.nn.Module):                     # knows x0: returns the noise that explains x_t
        def forward(self, x, c, tt, n):
            a = diff.ab[tt].view(-1, 1, 1, 1)
            return (x - a.sqrt() * x0) / (1 - a).sqrt()
    rec = diff.ddim(Oracle(), cond, torch.zeros(4, 2), noise, steps=20)
    t("U2", torch.allclose(rec, x0, atol=1e-4), "DDIM with a model that knows the answer returns it exactly (sampler arithmetic)")
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    y2 = x0.clone(); y2[:, :, 0, 0] = 1.0; y2[:, :, 23, 23] = 1.0
    l1 = diff.loss(small, x0, cond, torch.zeros(4, 2), g1); l2 = diff.loss(small, y2, cond, torch.zeros(4, 2), g2)
    t("U3", float(abs(l1 - l2).detach()) < 1e-7, "the loss ignores target cells outside the window")
    t("ES1", es_planted_check(), "a planted score sequence stops 3 validations after its best (patience 3) and keeps the best")
    a1, a2 = [torch.randn(5, 3) for _ in range(4)], None
    a2 = [x.clone() for x in a1]; mp = [torch.randn(5, 3) for _ in range(4)]
    for pe, pm in zip(a1, mp):
        pe.mul_(0.999).add_(pm, alpha=0.001)
    torch._foreach_mul_(a2, 0.999); torch._foreach_add_(a2, mp, alpha=0.001)
    t("U4", all(torch.equal(x, y) for x, y in zip(a1, a2)), "the fused moving-average update equals the plain loop")
    with tempfile.TemporaryDirectory() as tmp:
        py, refpat = X.plant_grids(Path(tmp) / "g")
        st = Path(tmp) / "sim" / "real_train"; st.mkdir(parents=True); sv = Path(tmp) / "sim" / "real_val"; sv.mkdir(parents=True)
        for s_ in (7001, 7002):
            X.plant_community(st / f"c{s_}_training.npz", s_)
        X.plant_community(sv / "c7101_training.npz", 7101)
        base = Path(tmp) / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.6, py, refpat, tools, out=quiet)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.6, py, refpat, tools, out=quiet)
        D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, tools, out=quiet)
        cfg = dict(CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=4, K=2, ddim=4, val_set="fast",
                   val_chunk=64, lr=1e-3)
        ra = train(cfg, str(base), str(Path(tmp) / "runA"), "cpu", log=quiet)
        t("S1", ra.get("F1") and ra.get("F2") and ra.get("ES4") and (Path(tmp) / "runA" / "best.pt").is_file()
          and len(ra["rows"]) == 4, f"planted run of 4 epochs: F1 {ra.get('F1')}, F2 {ra.get('F2')}, ES4 {ra.get('ES4')}, best.pt written")
        losses = [r["train_loss"] for r in ra["rows"]]
        t("S2", losses[-1] < losses[0], f"training loss falls ({losses[0]:.3f} -> {losses[-1]:.3f})")
        rB = Path(tmp) / "runB"
        train(cfg, str(base), str(rB), "cpu", log=quiet, stop_after=2)
        n_csv_mid = len(list(csv.DictReader(open(rB / "log.csv"))))
        rb = train(cfg, str(base), str(rB), "cpu", log=quiet, resume=True)
        same = all(torch.equal(ra["state"][k], rb["state"][k]) for k in ra["state"]) and ra["best_epoch"] == rb["best_epoch"] \
            and [r.get("val_score") for r in ra["rows"]] == [r.get("val_score") for r in rb["rows"]]
        t("ES2", same, "stopped after 2 epochs and resumed: same weights, best epoch and validation scores as the unbroken run")
        info = json.loads((rB / "run_info.json").read_text())
        n_csv = len(list(csv.DictReader(open(rB / "log.csv"))))
        ok = n_csv_mid == 2 and n_csv == 4 and [s_["from_epoch"] for s_ in info["starts"]] == [0, 2] \
            and info["starts"][1]["data_md5"]["stage_d_frozen_val.npz"] == X.file_md5(base / "stage_d_frozen_val.npz") \
            and info["end"]["stop_reason"] == "epoch cap 4" and info["end"]["best_epoch"] == rb["best_epoch"]
        t("S4", ok, f"log.csv after every epoch ({n_csv_mid} rows when stopped at 2, {n_csv} at the end); run_info.json: starts "
          f"from epochs {[s_['from_epoch'] for s_ in info['starts']]}, data md5, stop reason and best epoch at the end")
        bp = np.load(rB / "best_val_pred.npz")
        bt = torch.load(rB / "best.pt", map_location="cpu", weights_only=False)
        md5_saved = hashlib.md5((bp["counts"].astype(np.float32) / cfg["K"]).tobytes()).hexdigest()
        best_model = M.UNet(cfg["base"], cfg["emb"]); best_model.load_state_dict(bt["ema"])
        fr_ = dict(np.load(base / "stage_d_frozen_val.npz"))
        VA_ = E.Inputs(base / "real_val", fr_["files"])
        with D.TruthGuard():
            _, _, md5_again = predict(best_model, M.Diffusion(cfg["T"]), VA_, fr_, bp["sel"], cfg, "cpu", seed=7)
        logged = dict(l.split() for l in open(rB / "pred_md5.txt"))[str(int(bp["epoch"]))]
        t("S5", int(bp["epoch"]) == bt["epoch"] == rb["best_epoch"] and md5_saved == str(bp["pred_md5"]) == logged == md5_again,
          f"best_val_pred.npz holds the predictions of best.pt (epoch {int(bp['epoch'])}): counts / K give the logged md5, "
          "and sampling best.pt again gives the same md5")
        rC = Path(tmp) / "runC"
        train(cfg, str(base), str(rC), "cpu", log=quiet, stop_after=2)
        (rC / "run_info.json").unlink()                 # a folder written by v2 has no run_info.json
        rc = train(cfg, str(base), str(rC), "cpu", log=quiet, resume=True)
        ic = json.loads((rC / "run_info.json").read_text())
        t("S6", all(torch.equal(ra["state"][k], rc["state"][k]) for k in ra["state"]) and "note" in ic
          and [s_["from_epoch"] for s_ in ic["starts"]] == [2],
          "a run folder without run_info.json (as written by v2) resumes to the same weights; the gap is noted")
        rD = Path(tmp) / "runD"
        train(cfg, str(base), str(rD), "cpu", log=quiet, stop_after=2)
        idd = json.loads((rD / "run_info.json").read_text())
        idd["starts"][0]["data_md5"]["stage_d_frozen_val.npz"] = "0" * 32
        (rD / "run_info.json").write_text(json.dumps(idd))
        refused = False
        try:
            train(cfg, str(base), str(rD), "cpu", log=quiet, resume=True)
        except SystemExit:
            refused = True
        t("S7", refused, "a resume is refused when a stage D file differs from the first start (planted md5 change)")

        def no_dup(d, r):
            ep = [int(l.split()[0]) for l in open(d / "pred_md5.txt") if l.strip()]
            bt_ = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
            return (all(torch.equal(ra["state"][k], r["state"][k]) for k in ra["state"]) and ep == [0, 1, 2, 3]
                    and bt_["epoch"] == r["best_epoch"] == int(np.load(d / "best_val_pred.npz")["epoch"])
                    and not any((d / a).exists() for a, _ in CANDS)), ep
        for name, die in (("S8", 2), ("S9", 0)):
            dd = Path(tmp) / f"die{die}"
            globals()["BROKEN"] = f"die_e{die}"
            try:
                train(cfg, str(base), str(dd), "cpu", log=quiet)
            except SystemExit:
                pass
            finally:
                globals()["BROKEN"] = ""
            left = sorted(p.name for p in dd.iterdir())
            rr = train(cfg, str(base), str(dd), "cpu", log=quiet, resume=True)
            ok, ep = no_dup(dd, rr)
            t(name, ok, f"died after the epoch-{die} validation, before last.pt (left: {', '.join(left)}); resumed: same "
              f"weights as the unbroken run, pred_md5 epochs {ep} (no duplicate), best.pt = stopping state, no candidate left")
        before = X.file_md5(rB / "last.pt")
        refused = False
        try:
            train(cfg, str(base), str(rB), "cpu", log=quiet)
        except SystemExit:
            refused = True
        t("S10", refused and X.file_md5(rB / "last.pt") == before,
          "a fresh start into a folder that holds a run is refused and leaves that run untouched")
        globals()["BROKEN"] = "cuda_error"
        raised = False
        try:
            train(cfg, str(base), str(Path(tmp) / "cuda"), "cpu", log=quiet)
        except RuntimeError as e:
            raised = "planted CUDA error" in str(e)
        finally:
            globals()["BROKEN"] = ""
        t("S11", raised, "an error during validation sampling that is not the truth guard is raised as it is (not called F2)")
        frozen = dict(np.load(base / "stage_d_frozen_val.npz"))
        VA = E.Inputs(base / "real_val", frozen["files"])
        sel = np.arange(len(frozen["row"]))
        with D.TruthGuard():
            f0, _, _ = predict(M.UNet(8, 32), M.Diffusion(1000), VA, frozen, sel, dict(cfg, K=4), "cpu", seed=7)
        s0 = score(f0, VA, frozen, sel, base / "real_val")
        tol = max(0.02, 4 * s0["se"])
        t("F2", abs(s0["score"] - 0.5) <= tol and s0["rec_scored"] == 0,
          f"a model that ignores its inputs scores {s0['score']:.3f} (0.50 within {tol:.3f} on {len(sel)} planted stacks); recorded cells scored 0")
        want = {"truth_in_predict": "F2", "train_reads_val": "F1", "score_recorded": "F2"}
        for bug, gate in want.items():
            env = dict(os.environ); env["STAGE_F_BROKEN"] = bug
            code = ("import sys, json; sys.argv=['x']; import importlib.util as u; s=u.spec_from_file_location('f', %r); "
                    "m=u.module_from_spec(s); s.loader.exec_module(m); "
                    "r=m.train(json.loads(%r), %r, %r, 'cpu', log=lambda *a, **k: None); print('RES', {k: r.get(k) for k in ('F1','F2','ES4')})"
                    % (str(Path(__file__).resolve()), json.dumps(dict(cfg, cap=1)), str(base), str(Path(tmp) / f"b_{bug}")))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RES ")), "")
            t(f"bug {bug}", f"'{gate}': False" in line, f"{gate} fails as it must ({line[4:] or p.stderr.strip()[-200:]})")
        env = dict(os.environ); env["STAGE_F_BROKEN"] = "patience_reset"
        code = ("import importlib.util as u; s=u.spec_from_file_location('f', %r); m=u.module_from_spec(s); "
                "s.loader.exec_module(m); print('RES', m.es_planted_check())" % str(Path(__file__).resolve()))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        t("bug patience_reset", "RES False" in p.stdout, "a counter reset on any rise fails ES1")
        env = dict(os.environ); env["STAGE_F_BROKEN"] = "rng_global"
        code = ("import sys, json, torch; import importlib.util as u; s=u.spec_from_file_location('f', %r); m=u.module_from_spec(s); "
                "s.loader.exec_module(m); c=json.loads(%r); q=lambda *a, **k: None\n"
                "a=m.train(c, %r, %r, 'cpu', log=q)\n"
                "m.train(c, %r, %r, 'cpu', log=q, stop_after=2); b=m.train(c, %r, %r, 'cpu', log=q, resume=True)\n"
                "print('RES', all(torch.equal(a['state'][k], b['state'][k]) for k in a['state']))"
                % (str(Path(__file__).resolve()), json.dumps(cfg), str(base), str(Path(tmp) / "gA"), str(base), str(Path(tmp) / "gB"),
                   str(base), str(Path(tmp) / "gB")))
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        t("bug rng_global", "RES False" in p.stdout, "one random stream for the whole run (not reseeded per epoch) fails ES2")
        stop = False
        try:
            train(cfg, str(base), str(Path(tmp) / "e"), "cpu", log=quiet, val_dir=str(Path(tmp) / "real_eval"))
        except SystemExit:
            stop = True
        t("S3", stop, "a real_eval folder is refused")
    n_ok = sum(results)
    print(f"{n_ok} PASS, {len(results) - n_ok} FAIL")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="../data"); ap.add_argument("--out", default="../runs/smoke")
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--expect", default=EXPECT_FPRINT)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--train", action="store_true"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    if a.train:
        dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
        cfg = dict(CFG, seed=a.seed)
        print(f"stage_f_train.py md5 {X.file_md5(__file__)}; obj2_model.py md5 {X.file_md5(HERE / 'obj2_model.py')}; "
              f"stage_e_inputs.py md5 {X.file_md5(HERE / 'stage_e_inputs.py')}; torch {torch.__version__}; config {json.dumps(cfg)}",
              flush=True)
        r = train(cfg, a.data, a.out, dev, resume=a.resume, log=lambda s: print(s, flush=True))
        ok = bool(r.get("F1")) and bool(r.get("F2")) and bool(r.get("ES4"))
        print(f"stage F training seed {a.seed}: F1 {r.get('F1')}, F2 {r.get('F2')}, ES4 {r.get('ES4')}: {'PASS' if ok else 'FAIL'}",
              flush=True)
        sys.exit(0 if ok else 1)
    if a.self_test:
        torch.use_deterministic_algorithms(True)
        sys.exit(0 if self_test(a.obj1_tools) else 1)
    if a.smoke:
        r = smoke(a.data, a.out, steps=a.steps, expect=a.expect, cpu=a.cpu)
        sys.exit(0 if r and all(r.values()) else 1)
    ap.print_help()