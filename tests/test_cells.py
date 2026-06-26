"""Phase 4 gate: Lees-Edwards cell list == brute force; sub-quadratic scaling.

(plan section 8 / test_cells). The Hessian cell-vs-brute parity lives in
test_hessian.py once the Hessian exists.
"""
import time

import numpy as np
import pytest

import jamrl._core as core


def set_gamma(sys, g):
    x = np.asarray(sys.x)
    x[2 * sys.N + 1] = g
    sys.x = x


@pytest.mark.parametrize("N", [64, 256, 1024])
@pytest.mark.parametrize("gamma", [0.0, 0.2, -0.35, 0.49])
def test_cells_match_brute_random(N, gamma):
    sys = core.make_system(N, 4, 0.80, 1e-3)
    set_gamma(sys, gamma)
    b = core.evaluate_brute(sys)
    l = core.evaluate_cells(sys)
    # identical contact set
    assert b["n_contacts"] == l["n_contacts"]
    assert l["E"] == pytest.approx(b["E"], rel=1e-12, abs=1e-12)
    assert l["P_int"] == pytest.approx(b["P_int"], rel=1e-12, abs=1e-14)
    assert l["Egamma"] == pytest.approx(b["Egamma"], rel=1e-12, abs=1e-14)
    assert l["maxOv"] == pytest.approx(b["maxOv"], rel=1e-12, abs=1e-14)
    gb = np.asarray(b["grad"]); gl = np.asarray(l["grad"])
    assert np.max(np.abs(gb - gl)) < 1e-12


@pytest.mark.parametrize("gamma", [0.0, 0.3, -0.45])
def test_cells_match_brute_relaxed(gamma):
    """Match on a relaxed (jammed) config, not just the random start."""
    sys = core.make_system(256, 8, 0.80, 1e-3)
    core.relax(sys, 0.0, 0.0, 20000)
    set_gamma(sys, gamma)
    b = core.evaluate_brute(sys)
    l = core.evaluate_cells(sys)
    assert b["n_contacts"] == l["n_contacts"]
    assert l["E"] == pytest.approx(b["E"], rel=1e-12, abs=1e-14)
    assert np.max(np.abs(np.asarray(b["grad"]) - np.asarray(l["grad"]))) < 1e-12


def _time_eval(fn, sys, reps=20):
    fn(sys)  # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(sys)
    return (time.perf_counter() - t0) / reps


def test_cells_subquadratic_scaling():
    """Cell-list evaluate grows sub-quadratically and beats brute at large N."""
    times = {}
    for N in (256, 1024, 4096):
        sys = core.make_system(N, 4, 0.80, 1e-3)
        times[N] = _time_eval(core.evaluate_cells, sys)
    # 4x particles -> well under 16x (quadratic) time; expect ~4-6x.
    ratio = times[4096] / times[1024]
    assert ratio < 10.0, f"cells scaling ratio {ratio:.1f} not sub-quadratic"

    # cells must beat brute at N=1024
    sys = core.make_system(1024, 4, 0.80, 1e-3)
    t_cells = _time_eval(core.evaluate_cells, sys)
    t_brute = _time_eval(core.evaluate_brute, sys, reps=5)
    assert t_cells < t_brute, f"cells {t_cells:.2e}s not faster than brute {t_brute:.2e}s"
