"""Phase 5 gate: bulk/shear moduli via Schur complement (plan 3.7, 8)."""
import numpy as np
import pytest
from scipy.optimize import minimize

import jamrl._core as core


def jam(seed, N=16, P=1e-3):
    s = core.make_system(N, seed, 0.80, P)
    core.relax(s, 0.0, 0.0, 40000)
    return s


def Hmin_fixing(sys, e, val, x0):
    """min over all dofs except e (held at val) of the unbiased enthalpy H."""
    n = len(x0)
    free = np.array([k for k in range(n) if k != e])

    def fun(xf):
        x = x0.copy()
        x[free] = xf
        x[e] = val
        sys.x = x
        ev = core.evaluate(sys)
        return ev["H"], np.asarray(ev["grad"])[free]

    res = minimize(lambda xf: fun(xf)[0], x0[free].copy(),
                   jac=lambda xf: fun(xf)[1], method="L-BFGS-B",
                   options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-13})
    return res.fun


@pytest.mark.parametrize("seed", [1, 2, 101])
def test_B_positive_G_shear_stable(seed):
    sys = jam(seed)
    B = core.bulk_modulus(sys)
    G = core.shear_modulus(sys)
    assert B > 0.0, f"B={B}"
    assert G >= -1e-8, f"G={G} (should be shear-stabilized >= 0)"


def test_moduli_match_affine_finite_difference():
    """B,G from Schur match the affine-deformation FD of the relaxed enthalpy."""
    sys = jam(7, N=16)
    x0 = np.asarray(sys.x).copy()
    A0 = sys.area
    iL, ig = 2 * sys.N, 2 * sys.N + 1

    B_schur = core.bulk_modulus(sys)
    G_schur = core.shear_modulus(sys)

    # bulk: curvature of H_min wrt ℓ, eliminate (s, γ)
    dl = 1e-4
    l0 = x0[iL]
    Hp = Hmin_fixing(sys, iL, l0 + dl, x0)
    H0 = Hmin_fixing(sys, iL, l0, x0)
    Hm = Hmin_fixing(sys, iL, l0 - dl, x0)
    sys.x = x0
    B_fd = (Hp - 2 * H0 + Hm) / (dl * dl) / (4 * A0)

    # shear: curvature of H_min wrt γ, eliminate (s, ℓ)
    dg = 1e-4
    g0 = x0[ig]
    Hp = Hmin_fixing(sys, ig, g0 + dg, x0)
    H0 = Hmin_fixing(sys, ig, g0, x0)
    Hm = Hmin_fixing(sys, ig, g0 - dg, x0)
    sys.x = x0
    G_fd = (Hp - 2 * H0 + Hm) / (dg * dg) / A0

    assert B_fd == pytest.approx(B_schur, rel=2e-2), f"B schur={B_schur:.5e} fd={B_fd:.5e}"
    assert G_fd == pytest.approx(G_schur, rel=2e-2), f"G schur={G_schur:.5e} fd={G_fd:.5e}"
