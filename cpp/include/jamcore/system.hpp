// jamcore/system.hpp — the packing state: reduced coords + box dofs + radii.
#pragma once
#include "jamcore/types.hpp"

namespace jamcore {

// State vector layout (dimension 2N+2), plan section 3.1:
//   x = [ s_0x, s_0y, ..., s_(N-1)x, s_(N-1)y,  ln L,  gamma ]
//   s_i in [0,1)^2 are reduced (fractional) coordinates.
//   deformation matrix h = L * [[1, gamma], [0, 1]];  real pos = h * s_i.
struct System {
  int N = 0;
  VectorXd x;          // 2N+2
  VectorXd R;          // N frozen radii
  double P = 1e-3;     // target pressure
  double z_iso = 4.0;  // isostatic coordination (2d in 2D, frictionless)

  System() = default;
  System(int n) : N(n), x(VectorXd::Zero(2 * n + 2)), R(VectorXd::Constant(n, 0.5)) {}

  int dof() const { return 2 * N + 2; }
  int ix_lnL() const { return 2 * N; }
  int ix_gamma() const { return 2 * N + 1; }

  double lnL() const { return x[2 * N]; }
  double L() const { return std::exp(x[2 * N]); }
  double gamma() const { return x[2 * N + 1]; }
  double area() const { const double l = L(); return l * l; }

  double sx(int i) const { return x[2 * i]; }
  double sy(int i) const { return x[2 * i + 1]; }

  // Total disk area (frozen) and packing fraction phi = sum(pi R^2) / A.
  double disk_area() const {
    double a = 0.0;
    for (int i = 0; i < N; ++i) a += JAMCORE_PI * R[i] * R[i];
    return a;
  }
  double phi() const { return disk_area() / area(); }

  // Mean diameter sigma_bar (= 1.2 for the 50:50 0.5/0.7 mixture).
  double mean_diam() const {
    if (N == 0) return 1.2;
    double m = 0.0;
    for (int i = 0; i < N; ++i) m += R[i];
    return 2.0 * m / N;
  }
};

// Build a fresh random packing: 50:50 bidisperse (R = 0.5 / 0.7), reduced
// coordinates uniform in [0,1)^2, box sized so phi == phi0, gamma = 0.
System make_system(int N, uint64_t seed, double phi0, double P);

}  // namespace jamcore
