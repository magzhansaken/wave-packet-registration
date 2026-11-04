#!/usr/bin/env python3
"""Run 2 of the reproduction: validation checkpoint, several seeds, step-matched budget.
Changes against run 1 (checkpoint by test loss, as in the published code): 4 of the 20 training designs (every 5th, fixed for all seeds) form a validation
set; the checkpoint is chosen by validation loss (never by the test set); EPOCHS defaults to the number of epochs that
matches the authors' optimizer-step budget (~11,800 updates) with batch 16 and no accumulation; MC = 100 at test.

Reduced-budget reproduction of MF-B-NO (You et al., npj Comput. Mater. 12, 247 (2026), doi 10.1038/s41524-026-02112-y)
on its own published data (repo MF-B-NO_forSpinodal, data.7z), CPU only.

Uses the authors' code (src/train_hifi.py: classes, LF prediction, data loading) unchanged; only the training loop is
re-implemented here to (i) log every epoch, (ii) keep BOTH checkpoints: best-by-test-loss (as in the published code,
where val_loader = test set) and last epoch (no test information used), (iii) train the HF-only baseline needed for the
paper's headline number (84.1 % lower MSE than the high-fidelity baseline).

Budget: EPOCHS (env, default 30) instead of 500, seed 1 instead of 5 seeds; MC samples at test = 100 instead of 1000.
Outputs: out/repro_<variant>_history.csv, out/repro_summary.json
"""
import os, sys, json, time, glob, random
import numpy as np, torch, torch.nn as nn, joblib
from torch.utils.data import DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
import train_hifi as th
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

EPOCHS = int(os.environ.get("EPOCHS", 0)); SEED = int(os.environ.get("SEED", 1)); MC = int(os.environ.get("MC", 100))
NOACC = os.environ.get("NOACC", "1") == "1"   # 1 = optimizer step every mini-batch of 16 (step-matched budget), 0 = accumulate to 256 as published
OUT = os.path.join(HERE, os.environ.get("OUTDIR", "out")); os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("THREADS", 2)))
dev = torch.device("cpu")
PRIOR = {"prior_mu": 0.0, "prior_sigma": 1.0, "posterior_mu_init": 0.0, "posterior_rho_init": -5.0, "type": "Flipout", "moped_enable": False}
G = 41; BATCH = 16; KL_W = 1e-5; LR = 1e-3

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "repro.log"), "a") as fh: fh.write(msg + "\n")

scaler = joblib.load(os.path.join(HERE, "models/strain_scaler.gz"))
seed_all(SEED)
t0 = time.time()
lofi = th.LoFiDeepONet(branch_input_dim=3, trunk_input_dim=5, latent_dim=128); dnn_to_bnn(lofi, PRIOR)
lofi.load_state_dict(torch.load(os.path.join(HERE, "models/lofi_model.pth"), map_location=dev, weights_only=True)); lofi.eval()
work = os.path.join(OUT, f"seed_{SEED}"); os.makedirs(work, exist_ok=True)
tr_files = sorted(glob.glob(os.path.join(HERE, "data/hifi/curves/train/*.csv"))); te_files = sorted(glob.glob(os.path.join(HERE, "data/hifi/curves/test/*.csv")))
p_tr = th.predict_lofi_on_hifi(lofi, tr_files, work, scaler, G, os.path.join(HERE, "data/hifi/sdf/train"), dev, "lofi_on_train")
p_te = th.predict_lofi_on_hifi(lofi, te_files, work, scaler, G, os.path.join(HERE, "data/hifi/sdf/test"), dev, "lofi_on_test")
log(f"LF predictions: train files {len(p_tr)}/{len(tr_files)}, test files {len(p_te)}/{len(te_files)}  ({time.time()-t0:.0f}s)")
itr, ktr, ltr, ytr, sdf_tr, _ = th.load_residual_data_optimized(p_tr, scaler, G, os.path.join(HERE, "data/hifi/sdf/train"))
VAL_DESIGNS = np.arange(len(sdf_tr))[4::5]                          # 4 of 20 designs, fixed for all seeds
vmask = np.isin(np.asarray(itr).ravel(), VAL_DESIGNS)
iva, kva, lva, yva = itr[vmask], ktr[vmask], ltr[vmask], ytr[vmask]
itr, ktr, ltr, ytr = itr[~vmask], ktr[~vmask], ltr[~vmask], ytr[~vmask]
if EPOCHS == 0:
    EPOCHS = int(np.ceil(11800 / np.ceil(len(ytr) / BATCH)))
ite, kte, lte, yte, sdf_te, _ = th.load_residual_data_optimized(p_te, scaler, G, os.path.join(HERE, "data/hifi/sdf/test"))
SDF_TR = torch.from_numpy(sdf_tr).float(); SDF_TE = torch.from_numpy(sdf_te).float()
log(f"points: train {len(ytr)}, validation {len(yva)} (designs {list(VAL_DESIGNS)}), test {len(yte)} (designs {len(sdf_te)}); epochs {EPOCHS}")
mse_lf = float(np.mean((lte - yte) ** 2)); var_te = float(np.var(yte))
log(f"LF-only test MSE = {mse_lf:.6g}  (test target variance {var_te:.6g})")

def train(variant):
    """variant 'mf': residual net, trunk = [strain, phase, dir(3), LF]; target = y - LF; prediction = LF + residual
       variant 'hf': HF-only net, trunk = [strain, phase, dir(3)];      target = y;      prediction = net"""
    seed_all(SEED)
    tdim = 6 if variant == "mf" else 5
    net = th.SdfDeepONet(trunk_input_dim=tdim, latent_dim=128, sdf_grid_size=G); dnn_to_bnn(net, PRIOR)
    ds_tr = th.SDFLookupDataset(itr, ktr[:, :tdim].copy(), ltr, ytr); ds_te = th.SDFLookupDataset(ite, kte[:, :tdim].copy(), lte, yte)
    ds_va = th.SDFLookupDataset(iva, kva[:, :tdim].copy(), lva, yva)
    L_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0); L_te = DataLoader(ds_te, batch_size=64, shuffle=False)
    L_va = DataLoader(ds_va, batch_size=64, shuffle=False)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.3, patience=50)
    acc = 1 if NOACC else max(1, 256 // BATCH); crit = nn.MSELoss(); best = (float("inf"), -1); hist = []
    state_p = os.path.join(work, f"{variant}_state.pt"); ep0 = 0
    if os.path.exists(state_p):                     # resume after a stop: full state saved at the end of every epoch
        st = torch.load(state_p, weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        best, hist, ep0 = tuple(st["best"]), st["hist"], st["ep"]
        torch.set_rng_state(st["rng_torch"]); np.random.set_state(st["rng_np"]); random.setstate(st["rng_py"])
        log(f"[{variant}] resumed from epoch {ep0}/{EPOCHS}")
    for ep in range(ep0, EPOCHS):
        te0 = time.time(); net.train(); opt.zero_grad(); run = 0.0; nn_ = 0
        for i, (si, trk, lp, y) in enumerate(L_tr):
            pred = net(SDF_TR[si], trk); tgt = (y - lp) if variant == "mf" else y
            mse = crit(pred, tgt); loss = mse + KL_W * get_kl_loss(net)
            (loss / acc).backward()
            if (i + 1) % acc == 0: opt.step(); opt.zero_grad()
            run += mse.item() * len(y); nn_ += len(y)
        net.eval(); kl = get_kl_loss(net).item(); se = 0.0
        with torch.no_grad():
            for si, trk, lp, y in L_va:
                pred = net(SDF_TR[si], trk); fin = (lp + pred) if variant == "mf" else pred
                se += ((fin - y) ** 2).sum().item()
        vm = se / len(yva); vl = vm + KL_W * kl; sch.step(vl)
        hist.append({"epoch": ep + 1, "train_mse": run / nn_, "val_mse_eval_mode": vm, "sec": round(time.time() - te0, 1)})
        if vl < best[0]:
            best = (vl, ep + 1); torch.save(net.state_dict(), os.path.join(work, f"{variant}_best_by_val.pth"))
        log(f"[{variant}] epoch {ep+1}/{EPOCHS} train_mse={run/nn_:.5g} val_mse(eval-mode)={vm:.5g} ({time.time()-te0:.0f}s)")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "best": list(best), "hist": hist, "ep": ep + 1,
                    "rng_torch": torch.get_rng_state(), "rng_np": np.random.get_state(), "rng_py": random.getstate()}, state_p + ".tmp")
        os.replace(state_p + ".tmp", state_p)
    torch.save(net.state_dict(), os.path.join(work, f"{variant}_last.pth"))
    import csv
    with open(os.path.join(OUT, f"repro_{variant}_history.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)
    res = {"best_epoch_by_val": best[1]}
    for ck in ("best_by_val", "last"):
        net.load_state_dict(torch.load(os.path.join(work, f"{variant}_{ck}.pth"), weights_only=True)); net.train()  # MC sampling as in the paper
        preds = []
        with torch.no_grad():
            for _ in range(MC):
                outp = []
                for si, trk, lp, y in L_te:
                    p = net(SDF_TE[si], trk); outp.append((lp + p) if variant == "mf" else p)
                preds.append(torch.cat(outp).numpy().ravel())
        P = np.stack(preds); mu = P.mean(0); sd = P.std(0); y = yte.ravel()
        res[ck] = {"test_mse_mc_mean": float(np.mean((mu - y) ** 2)), "coverage95": float(np.mean(np.abs(y - mu) <= 1.96 * sd)),
                   "mean_std": float(sd.mean())}
    return res

summary = {"budget": {"epochs": EPOCHS, "seed": SEED, "mc_samples": MC, "optimizer_steps_per_epoch": int(np.ceil(len(ytr) / BATCH)) // (1 if NOACC else 16),
                      "optimizer_steps_total": int(EPOCHS * (np.ceil(len(ytr) / BATCH) // (1 if NOACC else 16))), "validation_designs": [int(v) for v in VAL_DESIGNS],
                      "accumulation_to_256": not NOACC, "paper": "500 epochs x ~23.7 steps = ~11,800 optimizer steps (batch 16 accumulated to 256), seeds 1-5, 1000 MC samples"},
           "points_train": int(len(ytr)), "points_test": int(len(yte)), "test_mse_LF_only": mse_lf}
_prev = os.path.join(OUT, f"repro_summary_seed{SEED}.json")
if os.environ.get("RESUME") == "1" and os.path.exists(_prev):     # after a container restart: keep finished variants
    for v, val in json.load(open(_prev)).items():
        if v in ("mf", "hf"):
            summary[v] = val; log(f"[{v}] taken from {_prev} (finished before the restart)")
for v in ("mf", "hf"):
    if v in summary:
        continue
    summary[v] = train(v); json.dump(summary, open(os.path.join(OUT, f"repro_summary_seed{SEED}.json"), "w"), indent=1)
for ck in ("best_by_val", "last"):
    m, h = summary["mf"][ck]["test_mse_mc_mean"], summary["hf"][ck]["test_mse_mc_mean"]
    summary[f"reduction_vs_HFonly_{ck}"] = 1 - m / h; summary[f"reduction_vs_LFonly_{ck}"] = 1 - m / mse_lf
summary["paper_claim"] = "84.1 % MSE reduction vs high-fidelity baseline (README; ~22 LF samples, 20 HF samples)"
summary["seconds_total"] = round(time.time() - t0)
json.dump(summary, open(os.path.join(OUT, f"repro_summary_seed{SEED}.json"), "w"), indent=1)
log(json.dumps(summary, indent=1))
