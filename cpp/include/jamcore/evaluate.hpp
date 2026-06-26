// jamcore/evaluate.hpp — energy / gradient / biased enthalpy for the packing.
#pragma once
#include "jamcore/system.hpp"

namespace jamcore {

// Result of a full evaluate() pass (plan section 3.2). `grad` is the gradient
// of the *biased* enthalpy H_b (what L-BFGS minimizes); convergence uses
// fInf_unbiased. With dP = sigF = 0, biased == unbiased.
struct EvalResult {
  double E = 0.0;       // pair energy
  double H = 0.0;       // unbiased enthalpy  E + P*A
  double H_b = 0.0;     // biased enthalpy    E + (P+dP)*A - sigF*gamma
  VectorXd grad;        // gradient of H_b, size 2N+2
  double fInf_unbiased = 0.0;  // |grad H|_inf  (convergence criterion)
  double fInf_biased = 0.0;    // |grad H_b|_inf
  double P_int = 0.0;   // internal (virial) pressure
  double maxOv = 0.0;   // max overlap delta (failure detection)
  double Egamma = 0.0;  // shear-stress residual  sum V' nx ry
  double sumVr = 0.0;   // sum V' r
  int n_contacts = 0;   // number of contacting pairs
};

// Shared per-pair kernel (plan section 3.2). Used by BOTH the O(N^2) path and
// the cell-list path (Phase 4) so per-pair arithmetic is bit-identical; only
// the summation order differs between paths.
inline void eval_accumulate_pair(int i, int j, const double* X, const double* R,
                                 double L, double gamma, double& E, double& sumVr,
                                 double& Eg, double& maxOv, int& ncon, double* g) {
  double dsy = X[2 * i + 1] - X[2 * j + 1];
  dsy -= std::round(dsy);
  double u = (X[2 * i] - X[2 * j]) + gamma * dsy;
  u -= std::round(u);
  const double rx = L * u, ry = L * dsy;
  const double r2 = rx * rx + ry * ry;
  const double sig = R[i] + R[j];
  if (r2 >= sig * sig) return;
  const double r = std::sqrt(r2);
  const double delta = 1.0 - r / sig;
  const double Vp = -delta / sig;  // dV/dr (< 0 in contact)
  const double nx = rx / r, ny = ry / r;
  E += 0.5 * delta * delta;
  sumVr += Vp * r;
  Eg += Vp * nx * ry;
  if (delta > maxOv) maxOv = delta;
  ++ncon;
  const double gx = Vp * nx * L;                  // dE/ds_ix
  const double gy = Vp * (gamma * nx + ny) * L;   // dE/ds_iy
  g[2 * i] += gx;
  g[2 * i + 1] += gy;
  g[2 * j] -= gx;
  g[2 * j + 1] -= gy;
}

// Finalize box-gradient components + scalars after the pair loop has filled the
// position gradient and the accumulators. Shared by both evaluate paths.
EvalResult eval_finalize(const System& sys, double dP, double sigF, double E,
                         double sumVr, double Eg, double maxOv, int ncon,
                         VectorXd&& grad);

// O(N^2) brute-force evaluate (reference). dP, sigF are applied loads (fixed).
EvalResult evaluate_brute(const System& sys, double dP = 0.0, double sigF = 0.0);

// Lees-Edwards cell-list evaluate (O(N)); matches evaluate_brute to ~1e-12.
EvalResult evaluate_cells(const System& sys, double dP = 0.0, double sigF = 0.0);

// Auto-dispatch: cell list for large N, brute force otherwise. Used everywhere
// (relaxer, env) so production paths get O(N) scaling transparently.
EvalResult evaluate(const System& sys, double dP = 0.0, double sigF = 0.0);

// Below this particle count the brute path is used (small-N reference fidelity).
constexpr int EVAL_CELLS_MIN_N = 128;

// Exact energy-invariant gamma relabeling, then wrap reduced coords into [0,1).
void gamma_wrap(System& sys);

}  // namespace jamcore
