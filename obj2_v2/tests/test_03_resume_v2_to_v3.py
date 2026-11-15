#!/usr/bin/env python3
"""
test_03_resume_v2_to_v3.py  (obj2_v2, stage F step 2, 8 Nov 2026)

Question: Sara started the real training with stage_f_train.py v2 (md5 26642116). v3 changes saving and logging only.
Can her run be stopped and resumed with v3, ending with the same weights, best epoch and validation scores as a run
done with v3 alone?

How: on planted communities (as the stage_f_train self-test), CPU, deterministic arithmetic:
    A  v3 alone, 4 epochs
    B  v2 for 2 epochs (writes last.pt, no run_info.json), then v3 --resume to 4 epochs
Pass: B's weights equal A's bit for bit, same best epoch, same validation scores; B's run_info.json notes the gap and
records the resume from epoch 2; B's log.csv has 4 rows.

Run from obj2_v2/work, with the v2 file kept beside v3 in tools as stage_f_train_v2.py:
    python3 ../tests/test_03_resume_v2_to_v3.py --v2 ../tools/stage_f_train_v2.py --v3 ../tools/stage_f_train.py
Sandbox 8 Nov: PASS.
"""
import argparse, csv, importlib.util, json, sys, tempfile
from pathlib import Path
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True); ap.add_argument("--v3", required=True)
    ap.add_argument("--obj1-tools", default="../../obj1_v2/tools")
    a = ap.parse_args()
    torch.use_deterministic_algorithms(True)
    v2, v3 = load(a.v2, "f_v2"), load(a.v3, "f_v3")
    X, D = v3.X, v3.D
    print(f"v2 md5 {X.file_md5(a.v2)}; v3 md5 {X.file_md5(a.v3)}")
    q = lambda *x, **k: None
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        py, refpat = X.plant_grids(tmp / "g")
        st = tmp / "sim" / "real_train"; st.mkdir(parents=True); sv = tmp / "sim" / "real_val"; sv.mkdir(parents=True)
        for s in (7001, 7002):
            X.plant_community(st / f"c{s}_training.npz", s)
        X.plant_community(sv / "c7101_training.npz", 7101)
        base = tmp / "data"
        X.export(sorted(map(str, st.glob("*.npz"))), "real_train", str(base / "real_train"), 0.6, py, refpat, a.obj1_tools, out=q)
        X.export(sorted(map(str, sv.glob("*.npz"))), "real_val", str(base / "real_val"), 0.6, py, refpat, a.obj1_tools, out=q)
        D.build(str(base / "real_train"), str(base / "real_val"), str(base), py, refpat, a.obj1_tools, out=q)
        cfg = dict(v3.CFG, base=8, emb=32, batch=32, draws=256, val_every=1, patience=50, cap=4, K=2, ddim=4,
                   val_set="fast", val_chunk=64, lr=1e-3)
        if dict(v2.CFG) != dict(v3.CFG):
            print("FAIL: CFG differs between v2 and v3, so last.pt from v2 would be refused"); sys.exit(1)
        ra = v3.train(cfg, str(base), str(tmp / "A"), "cpu", log=q)
        v2.train(cfg, str(base), str(tmp / "B"), "cpu", log=q, stop_after=2)
        had_info = (tmp / "B" / "run_info.json").is_file()
        rb = v3.train(cfg, str(base), str(tmp / "B"), "cpu", log=q, resume=True)
        same_w = all(torch.equal(ra["state"][k], rb["state"][k]) for k in ra["state"])
        same_s = [r.get("val_score") for r in ra["rows"]] == [r.get("val_score") for r in rb["rows"]]
        info = json.loads((tmp / "B" / "run_info.json").read_text())
        n_csv = len(list(csv.DictReader(open(tmp / "B" / "log.csv"))))
        print(f"   v2 wrote run_info.json: {had_info}; same weights {same_w}; same best epoch "
              f"{ra['best_epoch'] == rb['best_epoch']} ({rb['best_epoch']}); same validation scores {same_s}")
        print(f"   run_info starts from epochs {[s['from_epoch'] for s in info['starts']]}; note: {info.get('note')}; "
              f"log.csv rows {n_csv}; F1 {rb['F1']}, F2 {rb['F2']}, ES4 {rb['ES4']}")
        ok = (not had_info) and same_w and same_s and ra["best_epoch"] == rb["best_epoch"] and "note" in info \
            and [s["from_epoch"] for s in info["starts"]] == [2] and n_csv == 4 and rb["F1"] and rb["F2"] and rb["ES4"]
    print(f"test_03 resume v2 -> v3: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()