"""First-order hyperbolic form of the plate model and its characteristic (bicharacteristic) structure.

State q (13): membrane [vx, vy, Nxx, Nyy, Nxy] and bending (Mindlin) [vz, phx, phy, Mxx, Myy, Mxy, Qx, Qy],
v - mid-plane velocities, ph - rotation rates, N, M, Q - stress resultants.
    Mass q_t = Bx q_x + By q_y + Sm q
For a symmetric laminate (B = 0) membrane and bending decouple. The energy matrix E makes E Mass^{-1} Bx symmetric, so
each one-dimensional subsystem q_t = Ax q_x has real eigenvalues and a complete set of eigenvectors: the Riemann
invariants w = L q are transported along the bicharacteristics dx/dt = -lambda of that direction.
"""
import numpy as np
from scipy.linalg import expm

F = ["vx", "vy", "Nxx", "Nyy", "Nxy", "vz", "phx", "phy", "Mxx", "Myy", "Mxy", "Qx", "Qy"]
IX = {n: i for i, n in enumerate(F)}
NF = len(F)


def matrices(lam):
    """Mass vector and the matrices Bx, By, Sm of the first-order plate system (symmetric laminates only)."""
    A, D, As, rh, I = lam["A"], lam["D"], lam["As"], lam["rho_h"], lam["I"]
    if np.abs(lam.get("B", np.zeros((3, 3)))).max() > 1e-9 * np.abs(A).max() * lam["h"]:
        raise ValueError("B != 0: membrane-bending coupling not implemented")
    Mass = np.ones(NF)
    for n in ("vx", "vy", "vz"):
        Mass[IX[n]] = rh
    for n in ("phx", "phy"):
        Mass[IX[n]] = I
    Bx = np.zeros((NF, NF)); By = np.zeros((NF, NF)); Sm = np.zeros((NF, NF))
    vx, vy, Nxx, Nyy, Nxy = (IX[n] for n in ("vx", "vy", "Nxx", "Nyy", "Nxy"))
    vz, px, py, Mxx, Myy, Mxy, Qx, Qy = (IX[n] for n in ("vz", "phx", "phy", "Mxx", "Myy", "Mxy", "Qx", "Qy"))
    # membrane momentum
    Bx[vx, Nxx] = 1; By[vx, Nxy] = 1; Bx[vy, Nxy] = 1; By[vy, Nyy] = 1
    # membrane constitutive (rate form)
    for r, row in enumerate((Nxx, Nyy, Nxy)):
        Bx[row, vx] += A[r, 0]; Bx[row, vy] += A[r, 2]
        By[row, vy] += A[r, 1]; By[row, vx] += A[r, 2]
    # bending momentum
    Bx[vz, Qx] = 1; By[vz, Qy] = 1
    Bx[px, Mxx] = 1; By[px, Mxy] = 1; Sm[px, Qx] = -1
    Bx[py, Mxy] = 1; By[py, Myy] = 1; Sm[py, Qy] = -1
    for r, row in enumerate((Mxx, Myy, Mxy)):
        Bx[row, px] += D[r, 0]; Bx[row, py] += D[r, 2]
        By[row, py] += D[r, 1]; By[row, px] += D[r, 2]
    # transverse shear
    Bx[Qx, vz] = As[0, 0]; Bx[Qy, vz] = As[1, 0]
    By[Qx, vz] = As[0, 1]; By[Qy, vz] = As[1, 1]
    Sm[Qx, px] = As[0, 0]; Sm[Qx, py] = As[0, 1]; Sm[Qy, px] = As[1, 0]; Sm[Qy, py] = As[1, 1]
    return Mass, Bx, By, Sm


def energy_matrix(lam):
    """Quadratic form of the energy density: e = q^T E q / 2."""
    E = np.zeros((NF, NF))
    rh, I = lam["rho_h"], lam["I"]
    for n in ("vx", "vy", "vz"):
        E[IX[n], IX[n]] = rh
    for n in ("phx", "phy"):
        E[IX[n], IX[n]] = I
    iN = [IX[n] for n in ("Nxx", "Nyy", "Nxy")]; iM = [IX[n] for n in ("Mxx", "Myy", "Mxy")]; iQ = [IX["Qx"], IX["Qy"]]
    E[np.ix_(iN, iN)] = np.linalg.inv(lam["A"]); E[np.ix_(iM, iM)] = np.linalg.inv(lam["D"]); E[np.ix_(iQ, iQ)] = np.linalg.inv(lam["As"])
    return E


def characteristic(lam, axis, tol=1e-9):
    """Eigen-decomposition of A_axis = Mass^{-1} B_axis through the symmetrised form.
    Returns lam_nz (k,), Linv (k, NF) = left eigenvectors (Riemann invariants w = Linv q), R (NF, k) right eigenvectors,
    all for the non-zero eigenvalues only (zero-speed invariants are not transported)."""
    Mass, Bx, By, _ = matrices(lam)
    Ax = (Bx if axis == 0 else By) / Mass[:, None]
    E = energy_matrix(lam)
    w, V = np.linalg.eigh(E)
    P = V @ np.diag(np.sqrt(w)) @ V.T; Pi = V @ np.diag(1 / np.sqrt(w)) @ V.T
    Sx = P @ Ax @ Pi
    asym = np.abs(Sx - Sx.T).max() / max(np.abs(Sx).max(), 1e-30)
    if asym > 1e-8:
        raise RuntimeError(f"symmetriser failed: {asym:.2e}")
    ev, U = np.linalg.eigh((Sx + Sx.T) / 2)
    scale = np.abs(ev).max()
    nz = np.abs(ev) > tol * scale
    R = Pi @ U[:, nz]; L = U[:, nz].T @ P
    return ev[nz], L, R


def boundary_matrices(lam, axis, side):
    """Characteristic boundary condition for a traction-free edge normal to `axis`.
    side = -1: edge at the start of the axis (outward normal -e), +1: at the end.
    Invariant k is transported along dx/dt = -lambda_k: at the start edge it comes from the interior when lambda_k > 0,
    at the end edge when lambda_k < 0. The incoming ones are fixed by the edge conditions C q = 0.
    Returns (lam_nz, L, R, inc mask, K) with dw_in = K @ (q_old + R_out dw_out)."""
    ev, L, R = characteristic(lam, axis)
    rows = [IX[n] for n in (("Nxx", "Nxy", "Mxx", "Mxy", "Qx") if axis == 0 else ("Nyy", "Nxy", "Myy", "Mxy", "Qy"))]
    C = np.zeros((len(rows), NF)); C[np.arange(len(rows)), rows] = 1
    inc = (ev < 0) if side < 0 else (ev > 0)
    if inc.sum() != len(rows):
        raise RuntimeError(f"incoming characteristics {inc.sum()} != boundary conditions {len(rows)}")
    K = -np.linalg.solve(C @ R[:, inc], C)
    return ev, L, R, inc, K


def source_propagator(lam, dt):
    """Exact solution operator of the lower-order coupling q_t = Mass^{-1} Sm q over dt (rotation <-> shear exchange)."""
    Mass, _, _, Sm = matrices(lam)
    return expm(Sm / Mass[:, None] * dt)


def symbol(lam, kx, ky):
    """Symbol of the system for the wave vector (kx, ky): q_t = G q for a plane wave."""
    Mass, Bx, By, Sm = matrices(lam)
    return (1j * kx * Bx + 1j * ky * By + Sm) / Mass[:, None]


def dispersion(lam, k, theta):
    """Angular frequencies (sorted, >= 0) and eigenvectors of the plane wave exp(i(k.x - w t)) for |k| = k at angle theta."""
    G = symbol(lam, k * np.cos(theta), k * np.sin(theta))
    ev, V = np.linalg.eig(G)                     # q_t = G q -> exp(ev t), ev = -i w
    w = (1j * ev).real
    o = np.argsort(w)
    return w[o], V[:, o]


MEMBRANE = [0, 1, 2, 3, 4]; BENDING = [5, 6, 7, 8, 9, 10, 11, 12]


def branch_omega(lam, k, theta, branch):
    """branch: 'A0' (lowest bending), 'S0' (fastest membrane quasi-longitudinal), 'SH0' (membrane quasi-shear)."""
    w, V = dispersion(lam, k, theta)
    pos = w > 1e-6 * max(w.max(), 1)
    wb = []
    for i in np.where(pos)[0]:
        v = V[:, i]; eb = np.linalg.norm(v[BENDING]); em = np.linalg.norm(v[MEMBRANE])
        wb.append((w[i], "b" if eb > em else "m"))
    bend = sorted(x for x, t in wb if t == "b"); memb = sorted(x for x, t in wb if t == "m")
    if branch == "A0":
        return bend[0]
    if branch == "S0":
        return memb[-1]
    if branch == "SH0":
        return memb[0]
    raise ValueError(branch)


def wavenumber_at(lam, f, theta, branch, kmax=None):
    """|k| with omega_branch(k) = 2 pi f (bisection; omega is increasing in k for these branches)."""
    w0 = 2 * np.pi * f
    lo, hi = 1e-3, (kmax or 2e4)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if branch_omega(lam, mid, theta, branch) < w0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def phase_group(lam, f, theta, branch, dk_rel=1e-4):
    """Phase speed and group velocity vector (2,) at frequency f, wave-normal angle theta."""
    k = wavenumber_at(lam, f, theta, branch)
    cp = 2 * np.pi * f / k
    dk = dk_rel * k
    def om(kx, ky):
        """Angular frequency of the branch at the wave vector (kx, ky)."""
        return branch_omega(lam, np.hypot(kx, ky), np.arctan2(ky, kx), branch)
    kx, ky = k * np.cos(theta), k * np.sin(theta)
    cg = np.array([(om(kx + dk, ky) - om(kx - dk, ky)) / (2 * dk), (om(kx, ky + dk) - om(kx, ky - dk)) / (2 * dk)])
    return cp, cg


def group_speed_along(lam, f, psi, branch, n_theta=181):
    """Energy (ray, bicharacteristic) speed along the ray direction psi: find the wave normal whose group velocity
    points along psi. Returns |cg| and the wave-normal angle."""
    th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    cgs = np.array([phase_group(lam, f, t, branch)[1] for t in th])
    ang = np.arctan2(cgs[:, 1], cgs[:, 0])
    d = np.angle(np.exp(1j * (ang - psi)))
    i = int(np.argmin(np.abs(d)))
    # refine by secant on the angle mismatch
    a, b = th[i] - 2 * np.pi / n_theta, th[i] + 2 * np.pi / n_theta
    def mis(t):
        """Angle between the group velocity at wave-normal angle t and the ray direction psi."""
        g = phase_group(lam, f, t, branch)[1]; return np.angle(np.exp(1j * (np.arctan2(g[1], g[0]) - psi)))
    fa, fb = mis(a), mis(b)
    for _ in range(30):
        if fa * fb > 0:
            break
        m = 0.5 * (a + b); fm = mis(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    t = 0.5 * (a + b)
    g = phase_group(lam, f, t, branch)[1]
    return float(np.linalg.norm(g)), float(t)
