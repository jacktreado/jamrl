// jamcore: L-BFGS relaxer with the section-4.1 hazard fixes + pybind binding.
#include "jamcore/lbfgs.hpp"

#include <vector>
#include <deque>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
namespace jamcore {

double ftol_eff(const System& sys, const Tols& tol) {
  return std::max(tol.ftol_abs, tol.ftol_rel_P * sys.P * sys.mean_diam());
}

bool is_converged_unbiased(const EvalResult& ev, const System& sys, const Tols& tol) {
  const double feff = ftol_eff(sys, tol);
  if (ev.fInf_unbiased >= feff) return false;
  if (sys.P <= 0.0) return true;
  return std::abs(ev.P_int - sys.P) / sys.P < tol.ptol;
}

RelaxResult lbfgs_relax(System& sys, double dP, double sigF, int n_steps,
                        const LBFGSParams& p, const Tols& tol) {
  const double feff = ftol_eff(sys, tol);
  RelaxResult R;

  // History (most recent at back).
  std::deque<VectorXd> S, Y;
  std::deque<double> rho;

  EvalResult ev = evaluate(sys, dP, sigF);
  R.n_eval++;
  VectorXd g = ev.grad;
  double Hb = ev.H_b;
  int stall = 0;

  for (int it = 0; it < n_steps; ++it) {
    if (ev.fInf_biased < feff) break;  // biased gradient converged

    // ---- two-loop recursion: d = -H_approx * g ----
    VectorXd d = -g;
    const int m = static_cast<int>(S.size());
    if (m > 0) {
      VectorXd q = g;
      std::vector<double> alpha(m);
      for (int k = m - 1; k >= 0; --k) {
        alpha[k] = rho[k] * S[k].dot(q);
        q.noalias() -= alpha[k] * Y[k];
      }
      const double sy = S[m - 1].dot(Y[m - 1]);
      const double yy = Y[m - 1].dot(Y[m - 1]);
      const double scale = (yy > 0.0) ? sy / yy : 1.0;
      VectorXd r = scale * q;
      for (int k = 0; k < m; ++k) {
        const double beta = rho[k] * Y[k].dot(r);
        r.noalias() += S[k] * (alpha[k] - beta);
      }
      d = -r;
    }

    double gd = g.dot(d);
    if (gd >= 0.0) {  // not a descent direction -> reset to steepest descent
      S.clear(); Y.clear(); rho.clear();
      d = -g;
      gd = g.dot(d);
    }

    // Initial step length: bound the first (steepest-descent) move; otherwise 1.
    double a;
    if (S.empty()) {
      const double ginf = g.cwiseAbs().maxCoeff();
      a = std::min(1.0, 0.1 / std::max(1e-30, ginf));
    } else {
      a = 1.0;
    }

    // ---- backtracking Armijo with numeric slack (fix 4.1) ----
    const VectorXd x0 = sys.x;
    const double slack = 1e-14 * std::max(1e-30, std::abs(Hb));
    bool ok = false;
    EvalResult ev_new;
    double Hb_new = Hb;
    for (int ls = 0; ls < p.max_ls; ++ls) {
      sys.x = x0 + a * d;
      ev_new = evaluate(sys, dP, sigF);
      R.n_eval++;
      Hb_new = ev_new.H_b;
      if (Hb_new <= Hb + p.c1 * a * gd + slack) {
        ok = true;
        break;
      }
      a *= 0.5;
    }

    if (!ok) {
      // Line-search failure: revert, reset history + step scale, count a stall.
      sys.x = x0;
      S.clear(); Y.clear(); rho.clear();
      ++stall;
      if (stall >= p.stall_max) {
        R.stalled = true;
        R.iters = it + 1;
        break;
      }
      continue;
    }
    stall = 0;

    // Accept step; update curvature history if positive.
    const VectorXd s_vec = sys.x - x0;
    const VectorXd y_vec = ev_new.grad - g;
    const double ys = y_vec.dot(s_vec);
    if (ys > 1e-12 * std::max(1e-30, s_vec.norm() * y_vec.norm())) {
      S.push_back(s_vec);
      Y.push_back(y_vec);
      rho.push_back(1.0 / ys);
      if (static_cast<int>(S.size()) > p.memory) {
        S.pop_front(); Y.pop_front(); rho.pop_front();
      }
    }
    g = ev_new.grad;
    Hb = Hb_new;
    ev = ev_new;
    R.iters = it + 1;
  }

  R.ev = ev;
  R.converged = is_converged_unbiased(ev, sys, tol);
  return R;
}

static py::dict relax_to_dict(const RelaxResult& r, const System& sys) {
  py::dict d;
  d["iters"] = r.iters;
  d["n_eval"] = r.n_eval;
  d["stalled"] = r.stalled;
  d["converged"] = r.converged;
  d["E"] = r.ev.E;
  d["H"] = r.ev.H;
  d["H_b"] = r.ev.H_b;
  d["fInf_unbiased"] = r.ev.fInf_unbiased;
  d["fInf_biased"] = r.ev.fInf_biased;
  d["P_int"] = r.ev.P_int;
  d["maxOv"] = r.ev.maxOv;
  d["phi"] = sys.phi();
  d["n_contacts"] = r.ev.n_contacts;
  return d;
}

void register_lbfgs_impl(py::module_& m) {
  py::class_<LBFGSParams>(m, "LBFGSParams")
      .def(py::init<>())
      .def_readwrite("memory", &LBFGSParams::memory)
      .def_readwrite("c1", &LBFGSParams::c1)
      .def_readwrite("max_ls", &LBFGSParams::max_ls)
      .def_readwrite("stall_max", &LBFGSParams::stall_max);

  py::class_<Tols>(m, "Tols")
      .def(py::init<>())
      .def_readwrite("ftol_abs", &Tols::ftol_abs)
      .def_readwrite("ftol_rel_P", &Tols::ftol_rel_P)
      .def_readwrite("ptol", &Tols::ptol);

  m.def("ftol_eff",
        [](const System& s, const Tols& t) { return ftol_eff(s, t); },
        py::arg("sys"), py::arg("tol") = Tols{});

  m.def("relax",
        [](System& sys, double dP, double sigF, int n_steps,
           const LBFGSParams& p, const Tols& t) {
          auto r = lbfgs_relax(sys, dP, sigF, n_steps, p, t);
          return relax_to_dict(r, sys);
        },
        py::arg("sys"), py::arg("dP") = 0.0, py::arg("sigF") = 0.0,
        py::arg("n_steps") = 1000, py::arg("params") = LBFGSParams{},
        py::arg("tol") = Tols{},
        "Relax in place under the biased enthalpy; returns a result dict.");
}

}  // namespace jamcore

void register_lbfgs(py::module_& m) { jamcore::register_lbfgs_impl(m); }
