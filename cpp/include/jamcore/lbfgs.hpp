// jamcore/lbfgs.hpp — L-BFGS relaxer under the biased enthalpy (plan 3.4, 4.1).
#pragma once
#include "jamcore/system.hpp"
#include "jamcore/evaluate.hpp"

namespace jamcore {

struct RelaxResult {
  int iters = 0;       // accepted L-BFGS iterations
  int n_eval = 0;      // total evaluate() calls (line searches included)
  bool stalled = false;
  bool converged = false;  // unbiased convergence criterion met at exit
  EvalResult ev;           // final evaluation at (dP, sigF)
};

// Effective force tolerance ftol_eff = max(ftol_abs, ftol_rel_P * P * sigma_bar).
double ftol_eff(const System& sys, const Tols& tol);

// Unbiased convergence: |grad H|_inf < ftol_eff AND |P_int - P|/P < ptol.
bool is_converged_unbiased(const EvalResult& ev, const System& sys, const Tols& tol);

// Relax `sys` in place by minimizing the biased enthalpy H_b(dP, sigF) for up
// to n_steps L-BFGS iterations. Fresh history each call. Carries the section-4.1
// fixes: Armijo numeric slack, history+step reset on line-search failure, stall
// counter (stop after stall_max consecutive failures). Stops early on biased
// gradient convergence (|grad H_b|_inf < ftol_eff).
RelaxResult lbfgs_relax(System& sys, double dP, double sigF, int n_steps,
                        const LBFGSParams& p, const Tols& tol);

}  // namespace jamcore
