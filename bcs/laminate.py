"""Laminate stiffness for the plate model (first-order shear deformation + membrane).

Ply material: Hexply M21/34%/UD134/T700/300, the prepreg of the OGW plates. Elastic constants as published for this
material by the OGW group (Kudela et al., Data in Brief 42 (2022) 108078, Table 2): C1111 = 130, C1122 = C1133 = 6.1,
C2222 = C3333 = 11.2, C2233 = 5.2 GPa, rho = 1571 kg/m^3, ply 0.125 mm.  The table lists C1212 = 3 and C1313 = C2323 = 4.2 GPa;
for a transversely isotropic ply (fibre axis 1) G12 = G13 and G23 = (C2222 - C2233)/2 = 3.0 GPa, so we read the table as
G12 = G13 = 4.2 GPa, G23 = 3.0 GPa (assumption of the plate model in docs/PROTOCOL.md; the alternative reading is a sensitivity case).
Layup of OGW #1 (Moll et al., Sci. Data 6, 191, 2019): [45/0/-45/90/-45/0/45/90]_S, 2 mm.

Temperature: matrix-dominated constants scale as (1 + alpha (T - T0)); the fibre-dominated C1111 as (1 + alpha_f (T - T0)).
"""
import numpy as np

M21_T700 = dict(C11=130e9, C12=6.1e9, C13=6.1e9, C22=11.2e9, C23=5.2e9, C33=11.2e9,
                G23=3.0e9, G13=4.2e9, G12=4.2e9, rho=1571.0, t_ply=0.125e-3)
LAYUP_OGW1 = [45, 0, -45, 90, -45, 0, 45, 90, 90, 45, 0, -45, 90, -45, 0, 45]
KAPPA = 5.0 / 6.0


def ply_C6(m, T=None, T0=20.0, alpha=0.0, alpha_f=0.0):
    """6x6 Voigt stiffness of the ply in material axes (1 fibre, 2 transverse in-plane, 3 thickness)."""
    s = 1.0 if T is None else 1.0 + alpha * (T - T0)
    sf = 1.0 if T is None else 1.0 + alpha_f * (T - T0)
    C = np.zeros((6, 6))
    C[0, 0] = m["C11"] * sf
    C[0, 1] = C[1, 0] = m["C12"] * s
    C[0, 2] = C[2, 0] = m["C13"] * s
    C[1, 1] = m["C22"] * s
    C[1, 2] = C[2, 1] = m["C23"] * s
    C[2, 2] = m["C33"] * s
    C[3, 3] = m["G23"] * s
    C[4, 4] = m["G13"] * s
    C[5, 5] = m["G12"] * s
    return C


def reduced_Q(C6):
    """Plane-stress reduced stiffness (1, 2, 12) and transverse shear (23, 13) of a ply."""
    S = np.linalg.inv(C6)
    Q = np.linalg.inv(S[np.ix_([0, 1, 5], [0, 1, 5])])
    H = np.diag([C6[3, 3], C6[4, 4]])          # [Q44, Q55]: (yz ~ 23, xz ~ 13)
    return Q, H


def rotate_Q(Q, H, theta_deg):
    """Rotate ply stiffness by theta (fibre angle from the plate x axis)."""
    t = np.deg2rad(theta_deg); c, s = np.cos(t), np.sin(t)
    # strain transformation for engineering shear: eps_xy = T_e eps_12
    Te = np.array([[c * c, s * s, -c * s], [s * s, c * c, c * s], [2 * c * s, -2 * c * s, c * c - s * s]])
    # Qbar = Te^{-T} Q Te^{-1} with Te mapping material -> plate engineering strains
    Ti = np.linalg.inv(Te)
    Qb = Ti.T @ Q @ Ti
    # transverse shear: gamma_(yz, xz) plate = R gamma_(23, 13) material
    R = np.array([[c, s], [-s, c]])            # rows: yz, xz ; cols: 23, 13
    Ri = np.linalg.inv(R)
    Hb = Ri.T @ H @ Ri
    return Qb, Hb


def laminate(m=M21_T700, layup=LAYUP_OGW1, T=None, T0=20.0, alpha=0.0, alpha_f=0.0, kappa=KAPPA):
    """ABD of a laminate. Returns dict with A, B, D (3x3, order xx, yy, xy), As (2x2, order xz, yz incl. kappa), rho_h, I, h."""
    tp = m["t_ply"]; n = len(layup); h = n * tp
    z = np.linspace(-h / 2, h / 2, n + 1)
    Q, H = reduced_Q(ply_C6(m, T, T0, alpha, alpha_f))
    A = np.zeros((3, 3)); B = np.zeros((3, 3)); D = np.zeros((3, 3)); Hs = np.zeros((2, 2))
    for k, th in enumerate(layup):
        Qb, Hb = rotate_Q(Q, H, th)
        A += Qb * (z[k + 1] - z[k]); B += Qb * (z[k + 1] ** 2 - z[k] ** 2) / 2; D += Qb * (z[k + 1] ** 3 - z[k] ** 3) / 3
        Hs += Hb * (z[k + 1] - z[k])
    # reorder transverse shear to (xz, yz): Hs is in (yz, xz)
    As = kappa * np.array([[Hs[1, 1], Hs[1, 0]], [Hs[0, 1], Hs[0, 0]]])
    rho = m["rho"]
    return dict(A=A, B=B, D=D, As=As, rho_h=rho * h, I=rho * h ** 3 / 12.0, h=h)


def isotropic_plate(E=70e9, nu=0.33, rho=2700.0, h=2e-3, kappa=KAPPA):
    """Isotropic plate (for analytic checks)."""
    Qs = E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    G = E / (2 * (1 + nu))
    return dict(A=Qs * h, B=np.zeros((3, 3)), D=Qs * h ** 3 / 12, As=kappa * G * h * np.eye(2), rho_h=rho * h, I=rho * h ** 3 / 12, h=h)
