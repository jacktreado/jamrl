// jamcore: evaluate (brute + cell list) + gamma-wrap + pybind registration.
#include "jamcore/evaluate.hpp"
#include "jamcore/cell_list.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
namespace jamcore {

EvalResult eval_finalize(const System& sys, double dP, double sigF, double E,
                         double sumVr, double Eg, double maxOv, int ncon,
                         VectorXd&& grad) {
  const int N = sys.N;
  const double A = sys.area();
  const double P = sys.P;
  const double gamma = sys.gamma();

  EvalResult res;
  res.grad = std::move(grad);

  // Box-gradient components (biased), plan section 3.2.
  const double dHdl_b = sumVr + 2.0 * (P + dP) * A;  // no shear coupling (s4.3)
  const double dHdg_b = Eg - sigF;
  res.grad[2 * N] = dHdl_b;
  res.grad[2 * N + 1] = dHdg_b;

  res.E = E;
  res.H = E + P * A;
  res.H_b = E + (P + dP) * A - sigF * gamma;
  res.P_int = -sumVr / (2.0 * A);
  res.maxOv = maxOv;
  res.Egamma = Eg;
  res.sumVr = sumVr;
  res.n_contacts = ncon;

  // Infinity norms: position part is common to biased/unbiased.
  double gs_inf = 0.0;
  for (int k = 0; k < 2 * N; ++k) gs_inf = std::max(gs_inf, std::abs(res.grad[k]));
  const double dHdl_u = sumVr + 2.0 * P * A;
  const double dHdg_u = Eg;
  res.fInf_unbiased = std::max(gs_inf, std::max(std::abs(dHdl_u), std::abs(dHdg_u)));
  res.fInf_biased = std::max(gs_inf, std::max(std::abs(dHdl_b), std::abs(dHdg_b)));
  return res;
}

EvalResult evaluate_brute(const System& sys, double dP, double sigF) {
  const int N = sys.N;
  const double L = sys.L();
  const double gamma = sys.gamma();
  const double* X = sys.x.data();
  const double* R = sys.R.data();

  VectorXd grad = VectorXd::Zero(2 * N + 2);
  double E = 0.0, sumVr = 0.0, Eg = 0.0, maxOv = 0.0;
  int ncon = 0;
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      eval_accumulate_pair(i, j, X, R, L, gamma, E, sumVr, Eg, maxOv, ncon, grad.data());

  return eval_finalize(sys, dP, sigF, E, sumVr, Eg, maxOv, ncon, std::move(grad));
}

EvalResult evaluate_cells(const System& sys, double dP, double sigF) {
  const int N = sys.N;
  const double L = sys.L();
  const double gamma = sys.gamma();
  const double* X = sys.x.data();
  const double* R = sys.R.data();

  VectorXd grad = VectorXd::Zero(2 * N + 2);
  double E = 0.0, sumVr = 0.0, Eg = 0.0, maxOv = 0.0;
  int ncon = 0;
  for_each_pair_cells(sys, [&](int i, int j) {
    eval_accumulate_pair(i, j, X, R, L, gamma, E, sumVr, Eg, maxOv, ncon, grad.data());
  });

  return eval_finalize(sys, dP, sigF, E, sumVr, Eg, maxOv, ncon, std::move(grad));
}

EvalResult evaluate(const System& sys, double dP, double sigF) {
  return (sys.N >= EVAL_CELLS_MIN_N) ? evaluate_cells(sys, dP, sigF)
                                     : evaluate_brute(sys, dP, sigF);
}

void gamma_wrap(System& sys) {
  const int N = sys.N;
  double g = sys.gamma();
  const double k = std::round(g);
  if (k != 0.0) {
    for (int i = 0; i < N; ++i) sys.x[2 * i] += k * sys.x[2 * i + 1];
    g -= k;
    sys.x[2 * N + 1] = g;
  }
  for (int i = 0; i < N; ++i) {
    sys.x[2 * i] -= std::floor(sys.x[2 * i]);
    sys.x[2 * i + 1] -= std::floor(sys.x[2 * i + 1]);
  }
}

static py::dict eval_to_dict(const EvalResult& r) {
  py::dict d;
  d["E"] = r.E;
  d["H"] = r.H;
  d["H_b"] = r.H_b;
  d["grad"] = VectorXd(r.grad);
  d["fInf_unbiased"] = r.fInf_unbiased;
  d["fInf_biased"] = r.fInf_biased;
  d["P_int"] = r.P_int;
  d["maxOv"] = r.maxOv;
  d["Egamma"] = r.Egamma;
  d["sumVr"] = r.sumVr;
  d["n_contacts"] = r.n_contacts;
  return d;
}

void register_evaluate_impl(py::module_& m) {
  m.def("evaluate",
        [](const System& sys, double dP, double sigF) {
          return eval_to_dict(evaluate(sys, dP, sigF));
        },
        py::arg("sys"), py::arg("dP") = 0.0, py::arg("sigF") = 0.0,
        "Evaluate energy/grad/biased enthalpy (auto brute/cells). Returns a dict.");
  m.def("evaluate_brute",
        [](const System& sys, double dP, double sigF) {
          return eval_to_dict(evaluate_brute(sys, dP, sigF));
        },
        py::arg("sys"), py::arg("dP") = 0.0, py::arg("sigF") = 0.0,
        "O(N^2) reference evaluate.");
  m.def("evaluate_cells",
        [](const System& sys, double dP, double sigF) {
          return eval_to_dict(evaluate_cells(sys, dP, sigF));
        },
        py::arg("sys"), py::arg("dP") = 0.0, py::arg("sigF") = 0.0,
        "Lees-Edwards cell-list evaluate (O(N)).");

  m.def("gamma_wrap", &gamma_wrap, py::arg("sys"),
        "Energy-invariant gamma relabeling + wrap of reduced coords (in place).");
}

}  // namespace jamcore

void register_evaluate(py::module_& m) { jamcore::register_evaluate_impl(m); }
