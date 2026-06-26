"""Phase 3 gate: L-BFGS relaxer (plan section 3.4, 4.1, 8)."""
import numpy as np
import pytest

import jamrl._core as core


def test_random_init_converges():
    """Random init at small N converges (unbiased) for several seeds."""
    for seed in (1, 2, 101, 17):
        sys = core.make_system(32, seed, 0.80, 1e-3)
        r = core.relax(sys, 0.0, 0.0, 20000)
        feff = core.ftol_eff(sys)
        assert r["converged"], f"seed {seed} did not converge"
        assert not r["stalled"]
        assert r["fInf_unbiased"] < feff
        assert abs(r["P_int"] - 1e-3) / 1e-3 < 1e-4
        # physically sensible 2D bidisperse jammed density
        assert 0.80 < sys.phi < 0.90


def test_armijo_slack_no_subprecision_spin():
    """Fix 4.1: convergence costs O(10^3) evaluations, not O(10^6)."""
    sys = core.make_system(32, 3, 0.80, 1e-3)
    r = core.relax(sys, 0.0, 0.0, 20000)
    assert r["converged"]
    # Without the Armijo slack the prototype spun ~5e6 evals; we cap well below.
    assert r["n_eval"] < 20000, f"n_eval={r['n_eval']} (sub-precision spin?)"


def test_relax_determinism_bitwise():
    """Identical inputs -> identical relaxed state, bit for bit."""
    s1 = core.make_system(48, 5, 0.80, 1e-3)
    s2 = core.make_system(48, 5, 0.80, 1e-3)
    core.relax(s1, 0.0, 0.0, 5000)
    core.relax(s2, 0.0, 0.0, 5000)
    assert np.array_equal(np.asarray(s1.x), np.asarray(s2.x))


def test_relax_under_load_not_unbiased_converged():
    """While a load is held, P_int -> P+dP so the unbiased test stays unmet."""
    sys = core.make_system(32, 1, 0.80, 1e-3)
    core.relax(sys, 0.0, 0.0, 20000)  # jam first
    P = sys.P
    dP = 0.5 * P
    r = core.relax(sys, dP, 0.0, 2000)
    # biased state sits near P+dP, not P
    assert not r["converged"]
    assert r["P_int"] > 1.2 * P


def test_stall_flag_reports():
    """A fully converged state relaxed again should not report a stall failure."""
    sys = core.make_system(32, 2, 0.80, 1e-3)
    core.relax(sys, 0.0, 0.0, 20000)
    r = core.relax(sys, 0.0, 0.0, 50)  # already converged
    assert not r["stalled"]
    assert r["converged"]
