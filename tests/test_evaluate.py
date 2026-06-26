"""Phase 2 gate: evaluate() energy / gradient / biased enthalpy + gamma-wrap.

These checks are the correctness contract of plan section 3.2-3.3 / 8.
All comparisons go through the compiled C++ core (jamrl._core).
"""
import numpy as np
import pytest

import jamrl._core as core


# --------------------------------------------------------------------------- #
# Independent numpy reference for the sheared harmonic-disk energy + gradient.
# --------------------------------------------------------------------------- #
def ref_energy_grad(s, R, L, gamma):
    N = len(R)
    E = 0.0
    g = np.zeros(2 * N + 2)
    sumVr = 0.0
    Eg = 0.0
    A = L * L
    for i in range(N):
        for j in range(i + 1, N):
            dsy = s[i, 1] - s[j, 1]
            dsy -= np.round(dsy)
            u = (s[i, 0] - s[j, 0]) + gamma * dsy
            u -= np.round(u)
            rx, ry = L * u, L * dsy
            r = np.hypot(rx, ry)
            sig = R[i] + R[j]
            if r >= sig:
                continue
            delta = 1.0 - r / sig
            Vp = -delta / sig
            nx, ny = rx / r, ry / r
            E += 0.5 * delta * delta
            sumVr += Vp * r
            Eg += Vp * nx * ry
            gx = Vp * nx * L
            gy = Vp * (gamma * nx + ny) * L
            g[2 * i] += gx
            g[2 * i + 1] += gy
            g[2 * j] -= gx
            g[2 * j + 1] -= gy
    return E, g, sumVr, Eg, A


def set_x(sys, x):
    sys.x = np.asarray(x, dtype=float)


def H_unbiased(sys):
    return core.evaluate(sys, 0.0, 0.0)["H"]


def H_biased(sys, dP, sigF):
    return core.evaluate(sys, dP, sigF)["H_b"]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_gamma0_reduces_to_isotropic():
    """At gamma=0 the sheared engine matches an independent numpy reference."""
    sys = core.make_system(32, 7, 0.80, 1e-3)
    s = sys.s
    R = sys.radii
    L = sys.L
    E, g, sumVr, Eg, A = ref_energy_grad(s, R, L, 0.0)

    ev = core.evaluate(sys, 0.0, 0.0)
    assert ev["E"] == pytest.approx(E, rel=0, abs=1e-13 * max(1, abs(E)))
    # unbiased enthalpy H = E + P*A
    assert ev["H"] == pytest.approx(E + sys.P * A, rel=0, abs=1e-13 * max(1, abs(E)))
    # (s, l) gradient matches reference; box-l grad = sumVr + 2 P A
    grad = np.asarray(ev["grad"])
    assert np.allclose(grad[: 2 * 32], g[: 2 * 32], rtol=0, atol=1e-12)
    assert grad[2 * 32] == pytest.approx(sumVr + 2 * sys.P * A, abs=1e-12 * max(1, abs(sumVr)))
    # dH/dgamma equals the E_gamma formula
    assert grad[2 * 32 + 1] == pytest.approx(Eg, abs=1e-12 * max(1, abs(Eg)))
    assert ev["Egamma"] == pytest.approx(Eg, abs=1e-12 * max(1, abs(Eg)))


@pytest.mark.parametrize("biased", [False, True])
def test_fd_gradient_all_dofs(biased):
    """Central FD of H (or H_b) over all 2N+2 dofs at gamma=0.2."""
    N = 32
    sys = core.make_system(N, 3, 0.80, 1e-3)
    x = np.asarray(sys.x)
    x[2 * N + 1] = 0.2  # gamma = 0.2
    set_x(sys, x)

    dP, sigF = (3e-4, 2e-2) if biased else (0.0, 0.0)
    ev = core.evaluate(sys, dP, sigF)
    ana = np.asarray(ev["grad"])

    func = (lambda: H_biased(sys, dP, sigF)) if biased else (lambda: H_unbiased(sys))

    x0 = np.asarray(sys.x).copy()
    fd = np.zeros_like(ana)
    eps = 1e-6
    for k in range(len(x0)):
        xp = x0.copy(); xp[k] += eps; set_x(sys, xp); fp = func()
        xm = x0.copy(); xm[k] -= eps; set_x(sys, xm); fm = func()
        fd[k] = (fp - fm) / (2 * eps)
    set_x(sys, x0)

    # relative 1e-5 contract, with an absolute floor for genuinely-zero forces.
    H_scale = max(1.0, abs(ev["H_b"] if biased else ev["H"]))
    rel = np.abs(fd - ana) / np.maximum(np.abs(ana), 1e-3 * H_scale)
    assert np.max(rel) < 1e-5, f"max rel err {np.max(rel):.2e} (biased={biased})"


def test_gamma_wrap_energy_invariant():
    """gamma-wrap relabeling preserves energy to machine zero; |gamma|<=0.5."""
    N = 32
    sys = core.make_system(N, 11, 0.80, 1e-3)
    x = np.asarray(sys.x)
    x[2 * N + 1] = 1.7  # large shear
    set_x(sys, x)
    E_before = core.evaluate(sys, 0.0, 0.0)["E"]

    core.gamma_wrap(sys)
    E_after = core.evaluate(sys, 0.0, 0.0)["E"]

    assert abs(E_after - E_before) <= 1e-10 * max(1.0, abs(E_before))
    assert -0.5 - 1e-12 <= sys.gamma <= 0.5 + 1e-12
    # reduced coords wrapped into [0,1)
    s = sys.s
    assert s.min() >= 0.0 and s.max() < 1.0


def test_gamma_wrap_multiple_periods():
    """Wrap is exact across several periods of gamma."""
    N = 32
    for g0 in [0.4, 0.6, 2.3, -1.8, 5.5]:
        sys = core.make_system(N, 5, 0.80, 1e-3)
        x = np.asarray(sys.x); x[2 * N + 1] = g0; set_x(sys, x)
        E0 = core.evaluate(sys, 0.0, 0.0)["E"]
        core.gamma_wrap(sys)
        E1 = core.evaluate(sys, 0.0, 0.0)["E"]
        assert abs(E1 - E0) <= 1e-9 * max(1.0, abs(E0)), f"gamma0={g0}"
        assert -0.5 - 1e-12 <= sys.gamma <= 0.5 + 1e-12
