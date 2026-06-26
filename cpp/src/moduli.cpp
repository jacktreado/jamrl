// jamcore: bulk/shear moduli via Schur complement of the enthalpy Hessian.
#include "jamcore/moduli.hpp"
#include "jamcore/hessian.hpp"
#include "jamcore/evaluate.hpp"  // EVAL_CELLS_MIN_N

#include <vector>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using SpMat = Eigen::SparseMatrix<double>;
using Eigen::Triplet;
using Eigen::VectorXd;

namespace jamcore {

// Schur complement on the single kept dof `e`:
//   H(e,e) - H(e,q) · H_qq⁺ · H(q,e),  q = all dofs except e.
// Equivalently the curvature of H partially-minimized over q. Tikhonov ε lifts
// the two trivial translational zero modes so SimplicialLDLT stays PD.
static double schur_keep_one(const SpMat& Hin, int e, double tik) {
  SpMat H = Hin;  // column-major
  const int n = static_cast<int>(H.rows());
  const int m = n - 1;
  std::vector<Triplet<double>> tq;
  tq.reserve(H.nonZeros());
  VectorXd hqe = VectorXd::Zero(m);
  double Hee = 0.0;
  auto remap = [&](int idx) { return idx < e ? idx : idx - 1; };

  for (int col = 0; col < n; ++col) {
    for (SpMat::InnerIterator it(H, col); it; ++it) {
      const int row = static_cast<int>(it.row());
      const double v = it.value();
      if (row == e && col == e) {
        Hee = v;
      } else if (col == e) {  // column e, row != e -> coupling vector
        hqe[remap(row)] += v;
      } else if (row == e) {  // symmetric counterpart, skip
        continue;
      } else {
        tq.emplace_back(remap(row), remap(col), v);
      }
    }
  }
  for (int i = 0; i < m; ++i) tq.emplace_back(i, i, tik);

  SpMat Hqq(m, m);
  Hqq.setFromTriplets(tq.begin(), tq.end());
  Hqq.makeCompressed();

  VectorXd y;
  Eigen::SimplicialLDLT<SpMat> ldlt;
  ldlt.compute(Hqq);
  if (ldlt.info() == Eigen::Success) {
    y = ldlt.solve(hqe);
  } else {
    Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> cg;
    cg.setTolerance(1e-12);
    cg.compute(Hqq);
    y = cg.solve(hqe);
  }
  return Hee - hqe.dot(y);
}

double bulk_modulus(const System& sys) {
  const SpMat H = assemble_hessian_full(sys, sys.N >= EVAL_CELLS_MIN_N);
  const int e = 2 * sys.N;  // ℓ
  return schur_keep_one(H, e, 1e-9) / (4.0 * sys.area());
}

double shear_modulus(const System& sys) {
  const SpMat H = assemble_hessian_full(sys, sys.N >= EVAL_CELLS_MIN_N);
  const int e = 2 * sys.N + 1;  // γ
  return schur_keep_one(H, e, 1e-9) / sys.area();
}

void register_moduli_impl(py::module_& m) {
  m.def("bulk_modulus", &bulk_modulus, py::arg("sys"),
        "Bulk modulus B via Schur complement (eliminate s, γ).");
  m.def("shear_modulus", &shear_modulus, py::arg("sys"),
        "Shear modulus G via Schur complement (eliminate s, ℓ); >= 0 by construction.");
}

}  // namespace jamcore

void register_moduli(py::module_& m) { jamcore::register_moduli_impl(m); }
