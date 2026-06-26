"""Phase 5 gate: sparse enthalpy Hessian + backbone DOS Hessian (plan 3.6, 8)."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import jamrl._core as core


def H_full(sys):
    data, ind, indptr, shape = core.hessian_sparse(sys)
    return csr_matrix((data, ind, indptr), shape=shape)


def grad_at(sys, x):
    sys.x = x
    return np.asarray(core.evaluate(sys)["grad"])


def _hv_vs_fd_relerr(sys, n_dirs=4, eps=1e-6, seed=0):
    H = H_full(sys)
    x0 = np.asarray(sys.x).copy()
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_dirs):
        v = rng.standard_normal(len(x0))
        v /= np.linalg.norm(v)
        gp = grad_at(sys, x0 + eps * v)
        gm = grad_at(sys, x0 - eps * v)
        sys.x = x0
        fd = (gp - gm) / (2 * eps)
        Hv = H @ v
        worst = max(worst, np.max(np.abs(Hv - fd)) / max(1e-30, np.max(np.abs(fd))))
    return worst, H


def test_hessian_matvec_vs_fd_random():
    sys = core.make_system(40, 3, 0.80, 1e-3)
    x = np.asarray(sys.x); x[2 * 40 + 1] = 0.2; sys.x = x  # gamma != 0
    relerr, H = _hv_vs_fd_relerr(sys)
    assert relerr < 1e-9, f"H·v vs FD relerr={relerr:.2e}"
    assert abs(H - H.T).max() < 1e-12


def test_hessian_matvec_vs_fd_jammed():
    sys = core.make_system(40, 2, 0.80, 1e-3)
    core.relax(sys, 0.0, 0.0, 20000)
    relerr, H = _hv_vs_fd_relerr(sys, seed=1)
    assert relerr < 1e-9, f"H·v vs FD relerr={relerr:.2e}"
    assert abs(H - H.T).max() < 1e-12


@pytest.mark.parametrize("seed", [1, 2, 101])
def test_backbone_dos_psd_two_zero_modes(seed):
    sys = core.make_system(64, seed, 0.80, 1e-3)
    core.relax(sys, 0.0, 0.0, 20000)
    ev = np.asarray(core.eigvals_dos(sys))  # ascending
    assert len(ev) >= 4
    # PSD (allow tiny negative numerical dust)
    assert ev[0] > -1e-8, f"min eig {ev[0]:.2e} (not PSD)"
    # exactly two trivial (translational) zero modes, then a gap
    assert abs(ev[0]) < 1e-8 and abs(ev[1]) < 1e-8
    assert ev[2] > 1e-6, f"3rd eig {ev[2]:.2e} not clearly positive"
    # omega_max finite
    assert np.isfinite(ev[-1]) and ev[-1] > 0


def test_hessian_cells_match_brute():
    """Cell-list Hessian == brute-force Hessian (plan test_cells extension)."""
    for N, gamma in [(64, 0.0), (256, 0.2), (256, -0.35)]:
        sys = core.make_system(N, 5, 0.80, 1e-3)
        x = np.asarray(sys.x); x[2 * N + 1] = gamma; sys.x = x
        db = core.hessian_sparse_brute(sys)
        dc = core.hessian_sparse_cells(sys)
        Hb = csr_matrix((db[0], db[1], db[2]), shape=db[3])
        Hc = csr_matrix((dc[0], dc[1], dc[2]), shape=dc[3])
        assert abs(Hb - Hc).max() < 1e-12, f"N={N} gamma={gamma}"
