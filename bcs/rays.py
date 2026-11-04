"""Packet arrival times along bicharacteristic rays of the plate model.

For a homogeneous anisotropic plate the energy of a mode travels along straight rays with the group velocity c_g(psi);
reflections from the traction-free edges are rays from image sources. Arrival time of packet k on a path:
    t_k = t_c + L_k / |c_g(psi_k)|,   t_c - centre of the excitation burst.
Group velocities come from the dispersion relation of the same hyperbolic system that the BCS solver integrates.
"""
import functools
import numpy as np
from . import system as sy
from .laminate import laminate, M21_T700, LAYUP_OGW1


@functools.lru_cache(maxsize=512)
def group_curve(f, branch, s=1.0, alpha=0.0, T=20.0, T0=20.0, n_theta=72, g12=None, alpha_f=0.0):
    """|c_g| as a function of the ray angle psi on [0, pi) (centrosymmetric), sampled through wave-normal angles."""
    m = dict(M21_T700)
    if g12 is not None:
        m["G12"] = g12
    for k in ("C11", "C12", "C13", "C22", "C23", "C33", "G23", "G13", "G12"):
        m[k] = m[k] * s
    lam = laminate(m, LAYUP_OGW1, T=T, T0=T0, alpha=alpha, alpha_f=alpha_f)
    th = np.linspace(0, np.pi, n_theta, endpoint=False)
    cg = np.array([sy.phase_group(lam, f, t, branch)[1] for t in th])
    psi = np.mod(np.arctan2(cg[:, 1], cg[:, 0]), np.pi)
    mag = np.linalg.norm(cg, axis=1)
    o = np.argsort(psi)
    psi, mag = psi[o], mag[o]
    psi = np.concatenate([psi - np.pi, psi, psi + np.pi]); mag = np.concatenate([mag, mag, mag])
    return psi, mag


def cg_along(f, branch, psi, **kw):
    """Group (energy) speed of a branch along the ray angle psi, interpolated on the group curve."""
    P, M = group_curve(f, branch, **{k: v for k, v in kw.items()})
    return np.interp(np.mod(psi, np.pi), P, M)


def images(src, rcv, plate=(0.5, 0.5), order=1):
    """Image receivers for direct (order 0) and single edge reflections (order 1); order 2 adds corner/double images."""
    Lx, Ly = plate
    x, y = rcv
    out = [("direct", np.array([x, y]))]
    if order >= 1:
        out += [("x0", np.array([-x, y])), ("xL", np.array([2 * Lx - x, y])),
                ("y0", np.array([x, -y])), ("yL", np.array([x, 2 * Ly - y]))]
    if order >= 2:
        for nx, ix in (("x0", -x), ("xL", 2 * Lx - x)):
            for ny, iy in (("y0", -y), ("yL", 2 * Ly - y)):
                out.append((nx + ny, np.array([ix, iy])))
        out += [("x0xL", np.array([x - 2 * Lx, y])), ("xLx0", np.array([x + 2 * Lx, y])),
                ("y0yL", np.array([x, y - 2 * Ly])), ("yLy0", np.array([x, y + 2 * Ly]))]
    res = []
    for name, im in out:
        d = im - np.asarray(src)
        res.append((name, float(np.linalg.norm(d)), float(np.arctan2(d[1], d[0]))))
    return res


def packet_table(src, rcv, fk, modes=("S0", "A0"), order=1, tmax=1.0e-3, ncyc=5, **kw):
    """List of (mode, image, length, psi, arrival time) sorted by time, arrivals before tmax."""
    f = fk * 1e3; tc = ncyc / 2 / f
    rows = []
    for mode in modes:
        for name, L, psi in images(src, rcv, order=order):
            c = float(cg_along(f, mode, psi, **kw))
            t = tc + L / c
            if t <= tmax:
                rows.append((mode, name, L, psi, t))
    return sorted(rows, key=lambda r: r[-1])
