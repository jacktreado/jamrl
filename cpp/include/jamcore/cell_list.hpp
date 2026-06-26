// jamcore/cell_list.hpp — Lees-Edwards linked-cell neighbour enumeration.
//
// Provides a templated pair iterator shared by evaluate() and the Hessian
// assembly. The sheared minimum image (including the boundary x-shift) is
// handled entirely inside the per-pair kernels via round(); the cell stencil
// only has to be wide enough to never MISS a true contact, so over-search is
// harmless (non-contacts are filtered by the kernels).
#pragma once
#include <vector>
#include <cmath>
#include "jamcore/system.hpp"

namespace jamcore {

// Largest interaction diameter (1.4 for the 50:50 0.5/0.7 mixture).
inline double sigma_max(const System& sys) {
  double mx = 0.0;
  for (int i = 0; i < sys.N; ++i) mx = std::max(mx, sys.R[i]);
  return 2.0 * mx;
}

// Call f(i, j) once for every candidate pair i<j within the contact cutoff.
// Falls back to O(N^2) when the box is too small to host a >=(2k+1) stencil.
template <class F>
inline void for_each_pair_cells(const System& sys, F&& f) {
  const int N = sys.N;
  const double L = sys.L();
  const double gamma = sys.gamma();
  const double smax = sigma_max(sys);
  const double* X = sys.x.data();

  const int m = (smax > 0.0) ? static_cast<int>(std::floor(L / smax)) : 0;
  const double frac = (m > 0) ? (m * smax / L) : 1.0;  // <= 1
  const int ny = std::max(1, static_cast<int>(std::ceil(frac + 1e-9)));
  const int nx = std::max(1, static_cast<int>(std::ceil((1.0 + std::abs(gamma)) * frac + 1e-9)));

  // Need a non-overlapping stencil (else cells get visited twice -> double count).
  if (m < 2 * nx + 1 || m < 2 * ny + 1) {
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j) f(i, j);
    return;
  }

  std::vector<int> head(m * m, -1), nxt(N, -1), cx(N), cy(N);
  for (int i = 0; i < N; ++i) {
    double sx = X[2 * i] - std::floor(X[2 * i]);
    double sy = X[2 * i + 1] - std::floor(X[2 * i + 1]);
    int ix = static_cast<int>(sx * m);
    int iy = static_cast<int>(sy * m);
    if (ix >= m) ix = m - 1; if (ix < 0) ix = 0;
    if (iy >= m) iy = m - 1; if (iy < 0) iy = 0;
    cx[i] = ix; cy[i] = iy;
    const int c = iy * m + ix;
    nxt[i] = head[c];
    head[c] = i;
  }

  for (int i = 0; i < N; ++i) {
    const int ix = cx[i], iy = cy[i];
    for (int dy = -ny; dy <= ny; ++dy) {
      const int jy = ((iy + dy) % m + m) % m;
      for (int dx = -nx; dx <= nx; ++dx) {
        const int jx = ((ix + dx) % m + m) % m;
        const int c = jy * m + jx;
        for (int j = head[c]; j != -1; j = nxt[j]) {
          if (j > i) f(i, j);  // unordered cells -> keep i<j once
        }
      }
    }
  }
}

}  // namespace jamcore
