// jamcore: sparse enthalpy Hessian, backbone DOS Hessian, eigenvalues.
#include "jamcore/hessian.hpp"
#include "jamcore/cell_list.hpp"
#include "jamcore/evaluate.hpp"  // EVAL_CELLS_MIN_N

#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#ifdef JAMRL_USE_SPECTRA
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#endif

namespace py = pybind11;
using Eigen::Triplet;
using SpMat = Eigen::SparseMatrix<double>;

namespace jamcore {

// Iterate contacting pairs, invoking f(i, j, rx, ry, r, sig).
template <class F>
static void for_each_contact(const System& sys, bool use_cells, F&& f) {
  const double L = sys.L(), g = sys.gamma();
  const double* X = sys.x.data();
  const double* R = sys.R.data();
  auto test = [&](int i, int j) {
    double dsy = X[2 * i + 1] - X[2 * j + 1];
    dsy -= std::round(dsy);
    double u = (X[2 * i] - X[2 * j]) + g * dsy;
    u -= std::round(u);
    const double rx = L * u, ry = L * dsy;
    const double r2 = rx * rx + ry * ry;
    const double sig = R[i] + R[j];
    if (r2 < sig * sig) {
      const double r = std::sqrt(r2);
      f(i, j, rx, ry, r, sig);
    }
  };
  if (use_cells)
    for_each_pair_cells(sys, [&](int i, int j) { test(i, j); });
  else
    for (int i = 0; i < sys.N; ++i)
      for (int j = i + 1; j < sys.N; ++j) test(i, j);
}

SpMat assemble_hessian_full(const System& sys, bool use_cells) {
  const int N = sys.N;
  const int n = 2 * N + 2;
  const double L = sys.L(), g = sys.gamma();
  std::vector<Triplet<double>> trips;
  trips.reserve(static_cast<size_t>(36) * std::max(1, N));

  for_each_contact(sys, use_cells, [&](int i, int j, double rx, double ry, double r, double sig) {
    const double nx = rx / r, ny = ry / r;
    const double delta = 1.0 - r / sig;
    const double Vp = -delta / sig;
    const double Vpp = 1.0 / (sig * sig);

    // Real-space displacement derivatives D_q for the 6 local dofs:
    // {s_ix, s_iy, s_jx, s_jy, ℓ, γ}
    const double D[6][2] = {{L, 0.0}, {L * g, L}, {-L, 0.0}, {-L * g, -L}, {rx, ry}, {ry, 0.0}};
    double dr[6];
    for (int a = 0; a < 6; ++a) dr[a] = nx * D[a][0] + ny * D[a][1];

    // n · (∂²R/∂q_a∂q_b) — nonzero only for box-coupled second derivatives.
    double nHR[6][6] = {{0}};
    nHR[4][4] = r;
    nHR[4][0] = nHR[0][4] = nx * L;
    nHR[4][1] = nHR[1][4] = nx * L * g + ny * L;
    nHR[4][2] = nHR[2][4] = -nx * L;
    nHR[4][3] = nHR[3][4] = -nx * L * g - ny * L;
    nHR[4][5] = nHR[5][4] = nx * ry;
    nHR[5][1] = nHR[1][5] = nx * L;
    nHR[5][3] = nHR[3][5] = -nx * L;

    const int gid[6] = {2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * N, 2 * N + 1};
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {
        const double DdotD = D[a][0] * D[b][0] + D[a][1] * D[b][1];
        const double second = (DdotD - dr[a] * dr[b]) / r + nHR[a][b];
        const double h = Vpp * dr[a] * dr[b] + Vp * second;
        trips.emplace_back(gid[a], gid[b], h);
      }
    }
  });

  // Box term: ∂²(P·A)/∂ℓ² = 4·P·A.
  trips.emplace_back(2 * N, 2 * N, 4.0 * sys.P * sys.area());

  SpMat H(n, n);
  H.setFromTriplets(trips.begin(), trips.end());
  H.makeCompressed();
  return H;
}

Backbone backbone_set(const System& sys) {
  const int N = sys.N;
  std::vector<std::vector<int>> nbr(N);
  for_each_contact(sys, true, [&](int i, int j, double, double, double, double) {
    nbr[i].push_back(j);
    nbr[j].push_back(i);
  });

  std::vector<char> kept(N, 1);
  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = 0; i < N; ++i) {
      if (!kept[i]) continue;
      int cnt = 0;
      for (int j : nbr[i])
        if (kept[j]) ++cnt;
      if (cnt < 3) {  // < d+1 contacts -> rattler
        kept[i] = 0;
        changed = true;
      }
    }
  }

  Backbone bb;
  bb.remap.assign(N, -1);
  for (int i = 0; i < N; ++i) {
    if (kept[i]) {
      bb.remap[i] = bb.n_keep++;
      bb.keep.push_back(i);
    }
  }
  return bb;
}

SpMat assemble_hessian_dos(const System& sys, const Backbone& bb) {
  const int n = 2 * bb.n_keep;
  std::vector<Triplet<double>> trips;
  trips.reserve(static_cast<size_t>(16) * std::max(1, bb.n_keep));

  for_each_contact(sys, true, [&](int i, int j, double rx, double ry, double r, double sig) {
    const int bi = bb.remap[i], bj = bb.remap[j];
    if (bi < 0 || bj < 0) return;  // contact touches a rattler
    const double nx = rx / r, ny = ry / r;
    const double delta = 1.0 - r / sig;
    const double Vp = -delta / sig;
    const double Vpp = 1.0 / (sig * sig);
    // Real-space contact stiffness K = V''·nnᵀ + (V'/r)(I − nnᵀ).
    const double pr = Vp / r;
    const double K[2][2] = {{Vpp * nx * nx + pr * (1 - nx * nx), Vpp * nx * ny - pr * nx * ny},
                            {Vpp * nx * ny - pr * nx * ny, Vpp * ny * ny + pr * (1 - ny * ny)}};
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) {
        trips.emplace_back(2 * bi + a, 2 * bi + b, K[a][b]);
        trips.emplace_back(2 * bj + a, 2 * bj + b, K[a][b]);
        trips.emplace_back(2 * bi + a, 2 * bj + b, -K[a][b]);
        trips.emplace_back(2 * bj + a, 2 * bi + b, -K[a][b]);
      }
  });

  SpMat H(n, n);
  H.setFromTriplets(trips.begin(), trips.end());
  H.makeCompressed();
  return H;
}

static Eigen::VectorXd dense_eigvals(const SpMat& H) {
  Eigen::MatrixXd D = Eigen::MatrixXd(H);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(D, Eigen::EigenvaluesOnly);
  return es.eigenvalues();  // ascending
}

// Eigenvalues (ascending) of a symmetric sparse H. k<=0 (or k>=n-1) -> dense all;
// k>0 -> Spectra lowest-k via shift-invert near 0, with a dense fallback.
static Eigen::VectorXd eigvals_of(const SpMat& H, int k) {
  const int n = static_cast<int>(H.rows());
  if (n == 0) return Eigen::VectorXd();
  if (k <= 0 || k >= n - 1) return dense_eigvals(H);

#ifdef JAMRL_USE_SPECTRA
  try {
    const int ncv = std::min(n, std::max(2 * k + 1, 20));
    Spectra::SparseSymShiftSolve<double> op(H);
    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(op, k, ncv, -1e-6);
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn);
    if (eigs.info() == Spectra::CompInfo::Successful) {
      Eigen::VectorXd ev = eigs.eigenvalues();
      std::sort(ev.data(), ev.data() + ev.size());
      return ev;
    }
  } catch (...) {
  }
#endif
  // fallback: dense, smallest k
  Eigen::VectorXd all = dense_eigvals(H);
  return all.head(std::min(k, static_cast<int>(all.size())));
}

// Eigenpairs of a symmetric sparse H, eigenvalues ascending, eigenvectors as
// columns. k<=0 (or k>=n-1) -> dense all; k>0 -> Spectra lowest-k near 0.
static std::pair<Eigen::VectorXd, Eigen::MatrixXd> eigvecs_of(const SpMat& H, int k) {
  const int n = static_cast<int>(H.rows());
  if (n == 0) return {Eigen::VectorXd(), Eigen::MatrixXd()};

  auto dense = [&]() {
    Eigen::MatrixXd D = Eigen::MatrixXd(H);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(D);  // values + vectors, ascending
    Eigen::VectorXd vals = es.eigenvalues();
    Eigen::MatrixXd vecs = es.eigenvectors();
    if (k > 0 && k < n) return std::make_pair(Eigen::VectorXd(vals.head(k)),
                                              Eigen::MatrixXd(vecs.leftCols(k)));
    return std::make_pair(vals, vecs);
  };
  if (k <= 0 || k >= n - 1) return dense();

#ifdef JAMRL_USE_SPECTRA
  try {
    const int ncv = std::min(n, std::max(2 * k + 1, 20));
    Spectra::SparseSymShiftSolve<double> op(H);
    Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(op, k, ncv, -1e-6);
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestMagn);
    if (eigs.info() == Spectra::CompInfo::Successful) {
      Eigen::VectorXd ev = eigs.eigenvalues();
      Eigen::MatrixXd V = eigs.eigenvectors();
      // sort ascending, carrying eigenvectors along
      std::vector<int> idx(ev.size());
      for (int i = 0; i < ev.size(); ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&](int a, int b) { return ev[a] < ev[b]; });
      Eigen::VectorXd evs(ev.size());
      Eigen::MatrixXd Vs(V.rows(), V.cols());
      for (int j = 0; j < static_cast<int>(idx.size()); ++j) {
        evs[j] = ev[idx[j]];
        Vs.col(j) = V.col(idx[j]);
      }
      return {evs, Vs};
    }
  } catch (...) {
  }
#endif
  return dense();
}

Eigen::VectorXd eigvals_dos(const System& sys, int k) {
  Backbone bb = backbone_set(sys);
  SpMat H = assemble_hessian_dos(sys, bb);
  return eigvals_of(H, k);
}

Eigen::VectorXd eigvals_full_spectrum(const System& sys, int k) {
  SpMat H = assemble_hessian_full(sys, sys.N >= EVAL_CELLS_MIN_N);
  return eigvals_of(H, k);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> eigvecs_full(const System& sys, int k) {
  SpMat H = assemble_hessian_full(sys, sys.N >= EVAL_CELLS_MIN_N);
  return eigvecs_of(H, k);
}

// ---- pybind glue ----
static py::tuple csr_tuple(const SpMat& Hc) {
  Eigen::SparseMatrix<double, Eigen::RowMajor> H = Hc;
  H.makeCompressed();
  const int rows = static_cast<int>(H.rows());
  const int cols = static_cast<int>(H.cols());
  const int nnz = static_cast<int>(H.nonZeros());
  py::array_t<double> data(nnz);
  py::array_t<int> indices(nnz);
  py::array_t<int> indptr(rows + 1);
  std::memcpy(data.mutable_data(), H.valuePtr(), sizeof(double) * nnz);
  std::memcpy(indices.mutable_data(), H.innerIndexPtr(), sizeof(int) * nnz);
  std::memcpy(indptr.mutable_data(), H.outerIndexPtr(), sizeof(int) * (rows + 1));
  return py::make_tuple(data, indices, indptr, py::make_tuple(rows, cols));
}

void register_hessian_impl(py::module_& m) {
  m.def("hessian_sparse",
        [](const System& sys) { return csr_tuple(assemble_hessian_full(sys, sys.N >= EVAL_CELLS_MIN_N)); },
        py::arg("sys"), "Full enthalpy Hessian (2N+2) as CSR (data, indices, indptr, shape).");
  m.def("hessian_sparse_brute",
        [](const System& sys) { return csr_tuple(assemble_hessian_full(sys, false)); },
        py::arg("sys"));
  m.def("hessian_sparse_cells",
        [](const System& sys) { return csr_tuple(assemble_hessian_full(sys, true)); },
        py::arg("sys"));

  m.def("hessian_dos",
        [](const System& sys) {
          Backbone bb = backbone_set(sys);
          SpMat H = assemble_hessian_dos(sys, bb);
          py::dict d;
          py::tuple csr = csr_tuple(H);
          d["data"] = csr[0];
          d["indices"] = csr[1];
          d["indptr"] = csr[2];
          d["shape"] = csr[3];
          d["n_keep"] = bb.n_keep;
          d["keep"] = bb.keep;
          return d;
        },
        py::arg("sys"), "Backbone (rattler-removed) real-space position Hessian.");

  m.def("eigvals_dos",
        [](const System& sys, py::object k) {
          int kk = k.is_none() ? -1 : k.cast<int>();
          return Eigen::VectorXd(eigvals_dos(sys, kk));
        },
        py::arg("sys"), py::arg("k") = py::none(),
        "Backbone DOS eigenvalues (ascending); k=None -> dense all, k>0 -> lowest-k.");

  m.def("eigvals_full",
        [](const System& sys, py::object k) {
          int kk = k.is_none() ? -1 : k.cast<int>();
          return Eigen::VectorXd(eigvals_full_spectrum(sys, kk));
        },
        py::arg("sys"), py::arg("k") = py::none(),
        "Full enthalpy Hessian (2N+2, box DOF included) eigenvalues (ascending); "
        "k=None -> dense all, k>0 -> lowest-k. The box-inclusive relaxation spectrum.");

  m.def("eigvecs_full",
        [](const System& sys, py::object k) {
          int kk = k.is_none() ? -1 : k.cast<int>();
          auto vw = eigvecs_full(sys, kk);
          return py::make_tuple(Eigen::VectorXd(vw.first), Eigen::MatrixXd(vw.second));
        },
        py::arg("sys"), py::arg("k") = py::none(),
        "Full enthalpy Hessian eigenpairs (relaxation modes): returns "
        "(eigenvalues[m], eigenvectors[2N+2, m]); k=None -> all 2N+2, k>0 -> lowest-k.");

  m.def("n_rattlers",
        [](const System& sys) { return sys.N - backbone_set(sys).n_keep; }, py::arg("sys"));
}

}  // namespace jamcore

void register_hessian(py::module_& m) { jamcore::register_hessian_impl(m); }
