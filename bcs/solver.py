"""Bicharacteristic-splitting (BCS) solver for guided waves in a symmetric laminated plate.

Model: membrane (S0, SH0) and first-order shear (Mindlin) bending (A0) subsystems; for a symmetric laminate they decouple
and are integrated independently (`Subsystem(kind="membrane"|"bending")`).

One-dimensional sweep X(tau): the subsystem q_t = A_x q_x is written in Riemann invariants w_k = L_k q, which are constant
along the bicharacteristics dx/dt = -lambda_k; the new value at node i is the degree-p Lagrange interpolant of w_k at the
foot x_i + lambda_k tau. At a traction-free edge the invariants arriving from the interior are interpolated and the
incoming ones are fixed by the edge conditions (characteristic boundary condition). S(tau): exact propagator of the
lower-order rotation-shear coupling of the bending subsystem, plus the piezo source.
Time step: "strang"  X(dt/2) Y(dt/2) S(dt) Y(dt/2) X(dt/2)   - second order;
           "yoshida" composition of three Strang steps c1, c2, c1 (Yoshida 1990) - fourth order in time.
The Courant number nu = max|lambda| tau_max / h refers to the largest sweep step tau_max of the scheme; the stencil
contains the foot for nu <= 1.
"""
import numpy as np
from numba import njit, prange
from . import system as sy

KINDS = {"membrane": sy.MEMBRANE, "bending": sy.BENDING}


def lagrange_stencil(sigma, p, i, n, periodic):
    """Offsets start..start+deg and weights of the Lagrange interpolant of degree <= p at x_i + sigma h."""
    deg = p
    while True:
        if deg % 2 == 1:
            start = -(deg - 1) // 2 if sigma >= 0 else -(deg + 1) // 2
        else:
            start = -deg // 2
        if periodic or (i + start >= 0 and i + start + deg <= n - 1):
            break
        deg -= 1
        if deg == 0:
            start = 0
            break
    s = np.arange(start, start + deg + 1, dtype=float)
    w = np.ones(deg + 1)
    for j in range(deg + 1):
        for m in range(deg + 1):
            if m != j:
                w[j] *= (sigma - s[m]) / (s[j] - s[m])
    return start, deg, w


def block_characteristic(lam, axis, idx):
    """Characteristic speeds and eigenvectors restricted to the fields idx of one subsystem."""
    ev, L, R = sy.characteristic(lam, axis)
    rest = [i for i in range(sy.NF) if i not in idx]
    keep = np.array([np.abs(R[rest, k]).max() < 1e-9 * np.abs(R[:, k]).max() for k in range(len(ev))])
    return ev[keep], np.ascontiguousarray(L[keep][:, idx]), np.ascontiguousarray(R[np.ix_(idx, np.where(keep)[0])])


def edge_data(lam, axis, idx):
    """Free edge normal to `axis` by reflection: the 1-D subsystem q_t = A q_x anticommutes with J (J A J = -A), where
    J = +1 on velocities and rotation rates, -1 on all stress resultants (tractions of this axis and passive ones). Mirroring a line about
    its edge node with J therefore yields the whole-line problem whose solution keeps the tractions zero at the edge.
    Returns M = L J R (ghost invariants from interior invariants) and the energy-orthogonal projection Pp onto
    {tractions = 0} applied to edge nodes before a sweep (it removes what the tangential sweep put into the tractions)."""
    ev, L, R = block_characteristic(lam, axis, idx)
    names = [sy.F[i] for i in idx]
    tr = ("Nxx", "Nxy", "Mxx", "Mxy", "Qx") if axis == 0 else ("Nyy", "Nxy", "Myy", "Mxy", "Qy")
    vel = ("vx", "vy", "vz", "phx", "phy")
    J = np.diag([1.0 if n in vel else -1.0 for n in names])
    Mass, Bx, By, _ = sy.matrices(lam)
    A = ((Bx if axis == 0 else By) / Mass[:, None])[np.ix_(idx, idx)]
    assert np.allclose(J @ A @ J, -A, atol=1e-9 * np.abs(A).max())
    M = L @ J @ R
    rows = [names.index(n) for n in tr if n in names]
    C = np.zeros((len(rows), len(idx))); C[np.arange(len(rows)), rows] = 1
    Ei = np.linalg.inv(sy.energy_matrix(lam)[np.ix_(idx, idx)])
    Pp = Ei @ C.T @ np.linalg.solve(C @ Ei @ C.T, C)
    return np.ascontiguousarray(M), np.ascontiguousarray(Pp)


class Sweep:
    """Characteristic data of one direction for a fixed sweep step tau (may be negative)."""

    def __init__(self, lam, idx, axis, h, tau, n, p, periodic):
        self.ev, self.L, self.R = block_characteristic(lam, axis, idx)
        self.sigma = self.ev * tau / h
        self.nu = float(np.abs(self.sigma).max())
        nk = len(self.ev); self.g = (p + 1) // 2 + 1
        self.st = np.zeros(nk, np.int64); self.dg = np.zeros(nk, np.int64); self.w = np.zeros((nk, p + 1))
        for k in range(nk):
            st, dg, w = lagrange_stencil(self.sigma[k], p, n // 2, n, True)
            self.st[k], self.dg[k], self.w[k, :dg + 1] = st, dg, w
        self.periodic = periodic
        if periodic:
            self.M = np.zeros((nk, nk)); self.Pp = np.zeros((len(idx), len(idx)))
        else:
            self.M, self.Pp = edge_data(lam, axis, idx)

    def apply(self, q3):
        """q3: (nf, m, n) contiguous, lines along the last axis."""
        _sweep_lines(q3, self.L, self.R, self.st, self.dg, self.w, self.g, self.periodic, self.M, self.Pp)


@njit(parallel=True, fastmath=True, cache=True)
def _sweep_lines(q3, Lm, Rm, st_i, dg_i, w_i, g, periodic, M, Pp):
    """One characteristic sweep along every grid line (numba, parallel over lines), with edge projection and ghost invariants."""
    nf, m, n = q3.shape
    nk = Lm.shape[0]
    for line in prange(m):
        if not periodic:
            # energy-orthogonal projection of both edge nodes onto traction-free states
            for e in range(2):
                i = 0 if e == 0 else n - 1
                tmp = np.empty(nf)
                for f in range(nf):
                    acc = 0.0
                    for f2 in range(nf):
                        acc += Pp[f, f2] * q3[f2, line, i]
                    tmp[f] = acc
                for f in range(nf):
                    q3[f, line, i] -= tmp[f]
        W = np.zeros((nk, n + 2 * g))
        dW = np.zeros((nk, n))
        for k in range(nk):
            for f in range(nf):
                c = Lm[k, f]
                if c != 0.0:
                    for i in range(n):
                        W[k, g + i] += c * q3[f, line, i]
        if periodic:
            for k in range(nk):
                for j in range(g):
                    W[k, j] = W[k, n + j]
                    W[k, g + n + j] = W[k, g + j]
        else:
            for j in range(1, g + 1):
                for k in range(nk):
                    a = 0.0
                    b = 0.0
                    for k2 in range(nk):
                        a += M[k, k2] * W[k2, g + j]
                        b += M[k, k2] * W[k2, g + n - 1 - j]
                    W[k, g - j] = a
                    W[k, g + n - 1 + j] = b
        for k in range(nk):
            st = st_i[k]; dg = dg_i[k]
            for i in range(n):
                acc = 0.0
                for s in range(dg + 1):
                    acc += w_i[k, s] * W[k, g + i + st + s]
                dW[k, i] = acc - W[k, g + i]
        for f in range(nf):
            for k in range(nk):
                c = Rm[f, k]
                if c != 0.0:
                    for i in range(n):
                        q3[f, line, i] += c * dW[k, i]


def hann_burst(t, fc, ncyc=5):
    """Hann-windowed tone burst of ncyc cycles at fc, starting at t = 0."""
    T = ncyc / fc
    t = np.asarray(t, dtype=float)
    return np.where((t >= 0) & (t <= T), np.sin(2 * np.pi * fc * t) * 0.5 * (1 - np.cos(2 * np.pi * fc * t / ncyc)), 0.0)


def disk_mask(X, Y, xc, yc, r, h):
    """Smoothed indicator of a disk (cosine taper of width 2h across the rim)."""
    d = np.hypot(X - xc, Y - yc); w = 2 * h
    inner = d <= r - w / 2
    m = np.where(inner, 1.0, 0.5 * (1 + np.cos(np.pi * np.clip(d - (r - w / 2), 0, w) / w)))
    return m * (d < r + w / 2)


YOSHIDA_C1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_C2 = 1.0 - 2.0 * YOSHIDA_C1


class Subsystem:
    """Membrane or bending subsystem on [0, Lx] x [0, Ly], traction-free edges (or periodic for verification)."""

    def __init__(self, lam, kind, Lx, Ly, h, cfl=0.9, p=5, scheme="yoshida", periodic=False, dt=None):
        self.lam, self.kind, self.h, self.p, self.scheme, self.periodic = lam, kind, h, p, scheme, periodic
        self.idx = KINDS[kind]; self.names = [sy.F[i] for i in self.idx]; nf = len(self.idx)
        if periodic:
            self.nx, self.ny = int(round(Lx / h)), int(round(Ly / h))
        else:
            self.nx, self.ny = int(round(Lx / h)) + 1, int(round(Ly / h)) + 1
        self.x = np.arange(self.nx) * h; self.y = np.arange(self.ny) * h
        self.cmax = max(np.abs(block_characteristic(lam, 0, self.idx)[0]).max(), np.abs(block_characteristic(lam, 1, self.idx)[0]).max())
        c1, c2 = YOSHIDA_C1, YOSHIDA_C2
        if scheme == "yoshida":
            seq = [(c1 / 2, "X"), (c1 / 2, "Y"), (c1, "S"), (c1 / 2, "Y"), ((c1 + c2) / 2, "X"), (c2 / 2, "Y"), (c2, "S"),
                   (c2 / 2, "Y"), ((c2 + c1) / 2, "X"), (c1 / 2, "Y"), (c1, "S"), (c1 / 2, "Y"), (c1 / 2, "X")]
        elif scheme == "strang":
            seq = [(0.5, "X"), (0.5, "Y"), (1.0, "S"), (0.5, "Y"), (0.5, "X")]
        else:
            raise ValueError(scheme)
        taumax = max(abs(f) for f, a in seq if a != "S")
        self.dt = cfl * h / self.cmax / taumax if dt is None else dt
        self.cfl = self.cmax * taumax * self.dt / h
        cache = {}; self.ops = []
        self.src_names = ("phx", "phy", "Qx", "Qy")
        for f, a in seq:
            key = (a, round(f, 12))
            if key not in cache:
                if a == "S":
                    if kind == "bending":
                        P = sy.source_propagator(lam, f * self.dt)
                        si = [sy.IX[s] for s in self.src_names]
                        cache[key] = np.ascontiguousarray(P[np.ix_(si, si)])
                    else:
                        cache[key] = None
                else:
                    ax = 0 if a == "X" else 1
                    cache[key] = Sweep(lam, self.idx, ax, h, f * self.dt, self.nx if ax == 0 else self.ny, p, periodic)
            self.ops.append((a, f, cache[key]))
        self.src_loc = [self.names.index(s) for s in self.src_names] if kind == "bending" else []
        self.q = np.zeros((nf, self.nx, self.ny))
        self.t = 0.0; self.nstep = 0
        self.actuators = []; self.sensors = []

    def field(self, name):
        """View of one field of the state by its name."""
        return self.q[self.names.index(name)]

    # sources and receivers --------------------------------------------------------------------
    def add_actuator(self, xc, yc, r, signal, n0=1.0):
        """Surface-bonded piezo disk on the top face (z = +h/2): N += n0 s(t) chi (membrane), M += (h/2) n0 s(t) chi (bending)."""
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        chi = disk_mask(X, Y, xc, yc, r, self.h)
        ii = np.where(chi.any(1))[0]; jj = np.where(chi.any(0))[0]
        sl = (slice(ii[0], ii[-1] + 1), slice(jj[0], jj[-1] + 1))
        if self.kind == "membrane":
            comps = [(self.names.index("Nxx"), 1.0), (self.names.index("Nyy"), 1.0)]
        else:
            comps = [(self.names.index("Mxx"), self.lam["h"] / 2), (self.names.index("Myy"), self.lam["h"] / 2)]
        self.actuators.append((sl, chi[sl], signal, n0, comps))

    def add_sensor(self, xc, yc, r):
        """Add a receiving disk at (xc, yc) of radius r; its signal is the disk-averaged strain rate."""
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        chi = disk_mask(X, Y, xc, yc, r, self.h)
        ii = np.where(chi.any(1))[0]; jj = np.where(chi.any(0))[0]
        i0, i1 = max(ii[0], 1), min(ii[-1] + 1, self.nx - 1); j0, j1 = max(jj[0], 1), min(jj[-1] + 1, self.ny - 1)
        c = chi[i0:i1, j0:j1]
        self.sensors.append(((i0, i1, j0, j1), c / c.sum()))

    def sensor_rates(self):
        """Contribution of this subsystem to the top-surface in-plane strain rate averaged over each sensor disk:
        membrane: div v;  bending: (h/2) div phi."""
        h = self.h
        if self.kind == "membrane":
            a, b, fac = self.names.index("vx"), self.names.index("vy"), 1.0
        else:
            a, b, fac = self.names.index("phx"), self.names.index("phy"), self.lam["h"] / 2
        out = np.empty(len(self.sensors))
        for s, ((i0, i1, j0, j1), chi) in enumerate(self.sensors):
            dx = (self.q[a, i0 + 1:i1 + 1, j0:j1] - self.q[a, i0 - 1:i1 - 1, j0:j1]) / (2 * h)
            dy = (self.q[b, i0:i1, j0 + 1:j1 + 1] - self.q[b, i0:i1, j0 - 1:j1 - 1]) / (2 * h)
            out[s] = fac * float(((dx + dy) * chi).sum())
        return out

    # time stepping -------------------------------------------------------------------------------
    def _S(self, P, t0, t1):
        """Lower-order step over [t0, t1]: exact coupling operator P and the source increments of the actuators."""
        if P is not None:
            sub = self.q[self.src_loc]
            self.q[self.src_loc] = np.tensordot(P, sub, axes=(1, 0))
        for sl, chi, sig, n0, comps in self.actuators:
            ds = (float(sig(t1)) - float(sig(t0))) * n0
            if ds != 0.0:
                for ci, fac in comps:
                    self.q[(ci,) + sl] += fac * ds * chi

    def step(self):
        """One time step of the chosen composition (Strang or Yoshida) of sweeps and lower-order steps."""
        tc = self.t
        for a, f, op in self.ops:
            if a == "S":
                self._S(op, tc, tc + f * self.dt); tc += f * self.dt
            elif a == "Y":
                op.apply(self.q)
            else:
                qt = np.ascontiguousarray(self.q.transpose(0, 2, 1))
                op.apply(qt)
                self.q = np.ascontiguousarray(qt.transpose(0, 2, 1))
        self.t += self.dt; self.nstep += 1

    def run(self, T, record=True):
        """Advance to time T; returns the sample times and the integrated sensor signals if record is set."""
        n = int(np.ceil((T - self.t) / self.dt - 1e-9))
        ts = [self.t]; rs = [self.sensor_rates()] if (record and self.sensors) else []
        for _ in range(n):
            self.step()
            if record and self.sensors:
                ts.append(self.t); rs.append(self.sensor_rates())
        if not (record and self.sensors):
            return None, None
        ts, rs = np.array(ts), np.array(rs)
        strain = np.concatenate([np.zeros((1, rs.shape[1])), np.cumsum(0.5 * (rs[1:] + rs[:-1]) * np.diff(ts)[:, None], 0)])
        return ts, strain

    def energy(self):
        """Discrete energy of the subsystem (trapezoidal weights at the edges)."""
        E = sy.energy_matrix(self.lam)[np.ix_(self.idx, self.idx)]
        q = self.q.reshape(len(self.idx), -1)
        if self.periodic:
            wgt = np.ones(q.shape[1])
        else:
            wx = np.ones(self.nx); wx[[0, -1]] = 0.5; wy = np.ones(self.ny); wy[[0, -1]] = 0.5
            wgt = np.outer(wx, wy).ravel()
        return 0.5 * float(np.einsum("fn,fg,gn,n->", q, E, q, wgt)) * self.h ** 2


def simulate_pitch_catch(lam, XY, act, fc, T, h_mem, h_bend, r=5e-3, Lx=0.5, Ly=0.5, cfl=0.9, p=5, scheme="yoshida",
                         fs_out=2.5e6, ncyc=5, kinds=("membrane", "bending")):
    """Pitch-catch with actuator `act` (index into XY) and all transducers as receivers.
    Returns t (n,), signals (n_transducers, n) - top-surface strain averaged over each receiver disk; and per-kind parts."""
    t_out = np.arange(int(np.floor(T * fs_out)) + 1) / fs_out
    parts = {}
    sig = lambda t: hann_burst(t, fc, ncyc)
    for kind in kinds:
        hh = h_mem if kind == "membrane" else h_bend
        S = Subsystem(lam, kind, Lx, Ly, hh, cfl=cfl, p=p, scheme=scheme)
        S.add_actuator(XY[act][0], XY[act][1], r, sig)
        for x, y in XY:
            S.add_sensor(x, y, r)
        ts, strain = S.run(T)
        parts[kind] = np.array([np.interp(t_out, ts, strain[:, j]) for j in range(len(XY))])
    total = sum(parts.values())
    return t_out, total, parts
