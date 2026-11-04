"""pytest checks of the bicharacteristic-splitting solver (small grids, < 1 min in total)."""
import numpy as np
import pytest
from bcs.laminate import laminate, isotropic_plate, reduced_Q, rotate_Q, ply_C6, M21_T700
from bcs import system as sy
from bcs.solver import Subsystem, lagrange_stencil, hann_burst

LAM = laminate()


def plane_wave(S, lam, k, branch):
    """Exact plane wave of one branch on the grid of S, its eigenvalue and angular frequency."""
    G = sy.symbol(lam, *k); ev, V = np.linalg.eig(G); w = (1j * ev).real
    o = np.argsort(w); ev, V, w = ev[o], V[:, o], w[o]
    pos = [i for i in np.where(w > 1)[0] if np.linalg.norm(V[S.idx, i]) > 0.5 * np.linalg.norm(V[:, i])]
    i = pos[branch]
    X, Y = np.meshgrid(S.x, S.y, indexing="ij")
    ph = np.exp(1j * (k[0] * X + k[1] * Y))
    v = V[S.idx, i] / np.abs(V[S.idx, i]).max()
    return np.moveaxis(ph[..., None] * v, -1, 0), ev[i], w[i]


def run_plane(kind, N, scheme, cfl=0.8, p=5, periods=1.0, Lx=0.06, branch=0, lam=LAM):
    """Relative energy-norm error of the solver on an exact plane wave after the given periods."""
    k = 2 * np.pi * np.array([1, 2]) / Lx; h = Lx / N
    S = Subsystem(lam, kind, Lx, Lx, h, cfl=cfl, p=p, periodic=True, scheme=scheme)
    B, ev, w = plane_wave(S, lam, k, branch)
    T = periods * 2 * np.pi / w; ns = int(np.ceil(T / S.dt))
    S = Subsystem(lam, kind, Lx, Lx, h, dt=T / ns, p=p, periodic=True, scheme=scheme)
    S.q[:] = B.real
    for _ in range(ns):
        S.step()
    qe = (B * np.exp(ev * T)).real
    E = sy.energy_matrix(lam)[np.ix_(S.idx, S.idx)]; sc = np.sqrt(np.diag(E))[:, None, None]
    return np.sqrt((((S.q - qe) * sc) ** 2).sum() / ((qe * sc) ** 2).sum())


def test_qbar_matches_textbook():
    """Rotated reduced stiffness agrees with the textbook formula for Qbar_11."""
    Q, H = reduced_Q(ply_C6(M21_T700)); Qb, _ = rotate_Q(Q, H, 30.0)
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    q11 = Q[0, 0] * c ** 4 + 2 * (Q[0, 1] + 2 * Q[2, 2]) * s * s * c * c + Q[1, 1] * s ** 4
    assert abs(Qb[0, 0] - q11) < 1e-9 * q11


def test_symmetric_layup_has_no_coupling_and_balanced_A():
    """The OGW layup has B = 0 and no extension-shear coupling."""
    assert np.abs(LAM["B"]).max() < 1e-6
    assert abs(LAM["A"][0, 2]) < 1e-9 * LAM["A"][0, 0] and abs(LAM["A"][1, 2]) < 1e-9 * LAM["A"][0, 0]


@pytest.mark.parametrize("axis", [0, 1])
def test_characteristic_decomposition(axis):
    """Left and right eigenvectors diagonalise the one-dimensional system."""
    ev, L, R = sy.characteristic(LAM, axis)
    Mass, Bx, By, _ = sy.matrices(LAM)
    A = (Bx if axis == 0 else By) / Mass[:, None]
    assert np.allclose(L @ R, np.eye(len(ev)), atol=1e-10)
    assert np.allclose(A @ R, R * ev, rtol=1e-8, atol=1e-6 * np.abs(A).max())
    assert len(ev) == 10                                    # 5 wave pairs: 2 membrane + 3 bending


def test_isotropic_speeds():
    """Characteristic speeds of an isotropic plate are the plate, shear and corrected shear speeds."""
    lam = isotropic_plate(E=70e9, nu=0.33, rho=2700, h=2e-3)
    ev, _, _ = sy.characteristic(lam, 0)
    cL = np.sqrt(70e9 / (2700 * (1 - 0.33 ** 2))); cT = np.sqrt(70e9 / (2 * 1.33) / 2700)
    sp = np.abs(ev)
    assert np.any(np.isclose(sp, cL, rtol=1e-8)) and np.any(np.isclose(sp, cT, rtol=1e-8))
    assert np.any(np.isclose(sp, np.sqrt(5 / 6) * cT, rtol=1e-8))   # shear-corrected transverse speed


def test_lagrange_reproduces_polynomials():
    """Lagrange weights reproduce monomials up to the degree."""
    for p in range(1, 6):
        for sigma in (-0.7, -0.2, 0.0, 0.35, 0.9):
            st, dg, w = lagrange_stencil(sigma, p, 10, 40, False)
            xs = np.arange(st, st + dg + 1)
            for m in range(dg + 1):
                assert abs(np.dot(w, xs ** m) - sigma ** m) < 1e-10


def test_boundary_stencil_stays_inside():  # used for one-sided stencils of the interpolation helper
    """One-sided stencils near the edge stay inside the grid."""
    for i in range(4):
        st, dg, w = lagrange_stencil(0.4, 5, i, 50, False)
        assert i + st >= 0


def test_order_strang_membrane():
    """Strang composition is second order on a membrane plane wave."""
    e = [run_plane("membrane", N, "strang", periods=0.5) for N in (16, 32)]
    assert np.log2(e[0] / e[1]) > 1.7


def test_order_yoshida_bending_higher_than_strang():
    """Yoshida composition is more accurate than Strang on bending."""
    es = [run_plane("bending", N, "strang", cfl=0.5, periods=0.5) for N in (24, 48)]
    ey = [run_plane("bending", N, "yoshida", cfl=0.5, periods=0.5) for N in (24, 48)]
    assert ey[1] < es[1]
    assert np.log2(es[0] / es[1]) > 1.7


def test_free_plate_energy_does_not_grow_and_edge_loss_converges():
    """Energy never grows with free edges; the edge loss shrinks with the grid."""
    for kind in ("membrane", "bending"):
        loss = []
        for h in (1e-3, 0.5e-3):
            S = Subsystem(LAM, kind, 0.04, 0.03, h, cfl=0.9, p=5, scheme="strang")
            X, Y = np.meshgrid(S.x, S.y, indexing="ij")
            g = np.exp(-((X - 0.015) ** 2 + (Y - 0.012) ** 2) / (2 * 0.005 ** 2))
            S.q[S.names.index("vx" if kind == "membrane" else "vz")] = g
            E0 = S.energy(); Es = []
            while S.t < 40e-6:
                S.step(); Es.append(S.energy())
            assert max(Es) <= E0 * (1 + 1e-9)               # energy-stable with traction-free edges
            loss.append(1 - Es[-1] / E0)
        assert loss[1] < loss[0] / 2.5                      # edge dissipation vanishes with the grid


def test_cfl_limit():
    """Energy does not grow at Courant number 0.9."""
    def growth(cfl):
        """Energy ratio after 200 steps of white noise at the given Courant number."""
        S = Subsystem(LAM, "membrane", 0.03, 0.03, 1e-3, cfl=cfl, p=3, scheme="strang", periodic=True)
        rng = np.random.default_rng(0)
        S.q[:] = rng.standard_normal(S.q.shape) * np.array([1, 1, 1e4, 1e4, 1e4])[:, None, None]
        E0 = S.energy()
        for _ in range(200):
            S.step()
        return S.energy() / E0
    assert growth(0.9) <= 1.0
    assert growth(1.0) <= 1.0
    assert growth(1.3) > 1e3


def test_rotation_symmetry_and_reciprocity():
    """180-degree rotation maps the problem onto itself (centrosymmetric elasticity); reciprocity of pitch-catch."""
    from bcs.solver import simulate_pitch_catch
    XY = np.array([[0.025, 0.030], [0.015, 0.010]])
    fc = 150e3
    t, s01, _ = simulate_pitch_catch(LAM, XY, 0, fc, 30e-6, 1e-3, 1e-3, r=2e-3, Lx=0.04, Ly=0.04, kinds=("membrane",))
    t, s10, _ = simulate_pitch_catch(LAM, XY, 1, fc, 30e-6, 1e-3, 1e-3, r=2e-3, Lx=0.04, Ly=0.04, kinds=("membrane",))
    a, b = s01[1], s10[0]
    assert np.linalg.norm(a - b) < 0.05 * np.linalg.norm(a)
    XYr = 0.04 - XY
    t, r01, _ = simulate_pitch_catch(LAM, XYr, 0, fc, 30e-6, 1e-3, 1e-3, r=2e-3, Lx=0.04, Ly=0.04, kinds=("membrane",))
    assert np.linalg.norm(r01[1] - a) < 0.02 * np.linalg.norm(a)


def test_hann_burst():
    """The burst starts at zero, ends after five cycles and peaks near one."""
    t = np.linspace(0, 200e-6, 2001)
    s = hann_burst(t, 40e3)
    assert s[0] == 0 and abs(s[t > 125e-6]).max() == 0 and abs(s).max() > 0.9
