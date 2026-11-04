"""Neural baselines on one unit (path x frequency x split), CPU.

MF-B-NO (You et al., npj Comput. Mater. 2026) re-implemented for signals: Bayesian DeepONet (bayesian-torch Flipout, prior N(0,1),
posterior rho init -5, KL weight 1e-5, Adam), branch = normalised condition, trunk = (normalised time, LF value),
additive residual y = LF + r; predictive mean and std by Monte-Carlo sampling. LF variants: 'nom' - the edge reference
signal without a thermal law (what a nominal-property simulation gives), 'bss' - the edge reference stretched by the
fitted law. Checkpoint: number of steps chosen on the validation block (inner -> validation), then refit on the whole
training block for that number of steps.

f-VAE (Miele & Lomazzi, github.com/lorenzomie/VAE-generative-temperature-signal-for-CFRP-plate): dense VAE
128-64-16-2 (SiLU, sigmoid output on min-max normalised signals), KL weight 0.05, Adam 5e-3, 100 epochs, batch 20 -
the authors' hyperparameters; generation at a new temperature from the line fitted through (z1, z2, T) of the training
signals (their 'simple_PCA'), then the decoder.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

PRIOR = {"prior_mu": 0.0, "prior_sigma": 1.0, "posterior_mu_init": 0.0, "posterior_rho_init": -5.0, "type": "Flipout",
         "moped_enable": False}


def dec_factor(fk):
    """Decimation factor of the neural baselines for a frequency in kHz."""
    return 4 if fk <= 100 else 2          # 2.5 MHz -> 0.625 MHz (<= 100 kHz) or 1.25 MHz


def down(X, q):
    """Polyphase decimation of the rows of X by q."""
    return resample_poly(np.atleast_2d(X), 1, q, axis=1)


def up(X, q, m):
    """Polyphase interpolation of the rows of X by q, cut to m samples."""
    return resample_poly(np.atleast_2d(X), q, 1, axis=1)[:, :m]


class DON(nn.Module):
    """DeepONet with a branch net for the condition and a trunk net for (time, low-fidelity value)."""
    def __init__(s, h=64):
        super().__init__()
        s.b = nn.Sequential(nn.Linear(1, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        s.k = nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        s.bias = nn.Parameter(torch.zeros(1))

    def forward(s, xb, xt):
        return (s.b(xb) * s.k(xt)).sum(1, keepdim=True) + s.bias


def _tensors(T, F, R, tt, Tc, Ts, fsc, rsc):
    """Branch and trunk inputs (and the scaled residual targets, if given) as float32 tensors."""
    N, m = F.shape
    xb = np.repeat(((T - Tc) / Ts)[:, None], m, 1).reshape(-1, 1)
    xt = np.stack([np.tile(tt, N), (F / fsc).reshape(-1)], 1)
    out = [torch.tensor(xb, dtype=torch.float32), torch.tensor(xt, dtype=torch.float32)]
    if R is not None:
        out.append(torch.tensor((R / rsc).reshape(-1, 1), dtype=torch.float32))
    return out


def train_bdon(T, F, R, steps, lr=1e-3, width=64, seed=0, batch=1024, eval_every=0, val=None):
    """Train the Bayesian DeepONet residual. val = (Tv, Fv, Rv): returns validation curve every eval_every steps."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = F.shape[1]; tt = np.linspace(-1, 1, m)
    Tc, Ts = float(np.mean(T)), float(np.std(T) + 1e-6)
    fsc = float(np.abs(F).max() + 1e-12); rsc = float(R.std() + 1e-12)
    XB, XT, Y = _tensors(T, F, R, tt, Tc, Ts, fsc, rsc)
    net = DON(width); dnn_to_bnn(net, PRIOR); opt = torch.optim.Adam(net.parameters(), lr)
    n = len(Y); curve = []; best_sd, best_v = None, np.inf
    if val is not None:
        VB, VT, VY = _tensors(val[0], val[1], val[2], tt, Tc, Ts, fsc, rsc)
    for st in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,))
        loss = nn.functional.mse_loss(net(XB[idx], XT[idx]), Y[idx]) + 1e-5 * get_kl_loss(net) / n
        opt.zero_grad(); loss.backward(); opt.step()
        if eval_every and st % eval_every == 0 and val is not None:
            with torch.no_grad():
                net.eval(); pv = net(VB, VT); net.train()
            v = float(nn.functional.mse_loss(pv, VY)); curve.append((st, v))
            if v < best_v:
                best_v = v; best_sd = {k: x.clone() for k, x in net.state_dict().items()}
    if best_sd is not None:
        net.load_state_dict(best_sd)               # checkpoint chosen on the validation block
    state = dict(net=net, tt=tt, Tc=Tc, Ts=Ts, fsc=fsc, rsc=rsc)
    return state, curve


def predict_bdon(state, Tq, Fq, mc=20):
    """Predictive mean and standard deviation of the residual by Monte-Carlo sampling of the weights."""
    net = state["net"]
    XB, XT = _tensors(Tq, Fq, None, state["tt"], state["Tc"], state["Ts"], state["fsc"], state["rsc"])
    net.train()                                # Monte-Carlo over the Flipout weights, as in the authors' code
    with torch.no_grad():
        P = np.stack([net(XB, XT).numpy().reshape(Fq.shape) for _ in range(mc)]) * state["rsc"]
    return P.mean(0), P.std(0)


def mfbno_unit(t, Str, Ttr, Tte, inner, val, LFtr, LFte, fk, seed=0, lr=1e-3, width=64, steps=600, eval_every=60, LFin=None):
    """Two-stage protocol. inner, val: boolean masks over the training block. LFin = (LF at the inner temperatures,
    LF at the validation temperatures), both from the law fitted on the inner part only (correction of 03.11: the
    validation stage must not see the validation signals through the LF). Returns test mean, std (full rate)."""
    q = dec_factor(fk); m = Str.shape[1]
    Y, F, Fq = down(Str, q), down(LFtr, q), down(LFte, q)
    Fi, Fv = (down(LFin[0], q), down(LFin[1], q)) if LFin is not None else (F[inner], F[val])
    st1, curve = train_bdon(Ttr[inner], Fi, Y[inner] - Fi, steps, lr, width, seed, eval_every=eval_every,
                            val=(Ttr[val], Fv, Y[val] - Fv))
    best_steps = min(curve, key=lambda c: c[1])[0] if curve else steps
    mv, _ = predict_bdon(st1, Ttr[val], Fv)                 # validation predictions of the inner model at its checkpoint
    st, _ = train_bdon(Ttr, F, Y - F, best_steps, lr, width, seed)
    mu, sd = predict_bdon(st, Tte, Fq)
    return up(Fq + mu, q, m), up(sd, q, m), best_steps, curve, up(Fv + mv, q, m)


# --------------------------------------------------------------------------------------------- f-VAE
class VAE(nn.Module):
    """Fully connected variational autoencoder of the f-VAE baseline (sigmoid output on [0, 1])."""
    def __init__(s, n, latent=2):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(n, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 16), nn.SiLU())
        s.mu = nn.Linear(16, latent); s.lv = nn.Linear(16, latent)
        s.dec = nn.Sequential(nn.Linear(latent, 16), nn.SiLU(), nn.Linear(16, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU(),
                              nn.Linear(128, n), nn.Sigmoid())

    def forward(s, x):
        h = s.enc(x); mu, lv = s.mu(h), s.lv(h)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return s.dec(z), mu, lv


def fvae_unit(Str, Ttr, Tte, seed=0, epochs=100, batch=20, kl=0.05, lr=5e-3):
    """Train the f-VAE on the training signals of one unit and decode at the test conditions."""
    torch.manual_seed(seed); np.random.seed(seed)
    lo, hi = float(Str.min()), float(Str.max()); X = torch.tensor((Str - lo) / (hi - lo), dtype=torch.float32)
    net = VAE(X.shape[1]); opt = torch.optim.Adam(net.parameters(), lr)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for b in range(0, len(X), batch):
            xb = X[perm[b:b + batch]]
            rec, mu, lv = net(xb)
            rl = ((rec - xb) ** 2).mean(1).mean()                       # keras mse: mean over the signal
            kll = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum(1).mean() * kl
            opt.zero_grad(); (rl + kll).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Z = net.mu(net.enc(X)).numpy()
    # line through (z1, z2, T) by SVD (the authors' simple_PCA), then the point of the line at the query T
    P = np.c_[Z, Ttr]; c = P.mean(0); _, _, Vt = np.linalg.svd(P - c, full_matrices=False); d = Vt[0]
    if abs(d[2]) < 1e-9:
        zq = np.repeat(c[None, :2], len(Tte), 0)
    else:
        s_ = (Tte - c[2]) / d[2]; zq = c[None, :2] + s_[:, None] * d[None, :2]
    with torch.no_grad():
        Xq = net.dec(torch.tensor(zq, dtype=torch.float32)).numpy()
    return Xq * (hi - lo) + lo
