// jamrl._core — pybind11 surface.
//
// This file is assembled incrementally across the build phases. Each phase
// registers its objects/functions through a `register_*` entry point declared
// in the jamcore headers; the module body below wires them together.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#ifdef JAMRL_HAVE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Phase registration hooks (defined in src/bindings_*.cpp as the core grows).
void register_system(py::module_& m);
void register_evaluate(py::module_& m);
void register_lbfgs(py::module_& m);
void register_env(py::module_& m);
void register_hessian(py::module_& m);
void register_moduli(py::module_& m);
void register_rollout(py::module_& m);

static int core_max_threads() {
#ifdef JAMRL_HAVE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

static void core_set_num_threads(int n) {
#ifdef JAMRL_HAVE_OPENMP
  if (n > 0) omp_set_num_threads(n);
#else
  (void)n;
#endif
}

static bool core_has_openmp() {
#ifdef JAMRL_HAVE_OPENMP
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "jamrl C++ core: jamming MDP, L-BFGS relaxer, Hessian/moduli, batch rollouts";
  m.attr("__version__") = "0.1.0";

  m.def("has_openmp", &core_has_openmp, "True if compiled with OpenMP.");
  m.def("set_num_threads", &core_set_num_threads, py::arg("n"),
        "Set the OpenMP thread count for the batch runner.");
  m.def("max_threads", &core_max_threads, "Current OpenMP max thread count.");

  // Eigen's own threading (used in parallel_mode='intra'); BLAS threads are set
  // from Python via environment variables.
  m.def("eigen_set_threads", [](int n) { Eigen::setNbThreads(n); }, py::arg("n"),
        "Set the number of threads Eigen uses internally.");
  m.def("eigen_num_threads", []() { return Eigen::nbThreads(); });

  // Phase-0 smoke test: a trivial parallel reduction proving OpenMP links.
  m.def("ping", [](int n) {
    double acc = 0.0;
#ifdef JAMRL_HAVE_OPENMP
#pragma omp parallel for reduction(+ : acc)
#endif
    for (int i = 0; i < n; ++i) acc += static_cast<double>(i);
    return acc;
  }, py::arg("n") = 1, "Smoke-test reduction; returns sum(0..n-1).");

  // Wire up the physics core as it comes online phase by phase.
  register_system(m);
  register_evaluate(m);
  register_lbfgs(m);
  register_env(m);
  register_hessian(m);
  register_moduli(m);
  register_rollout(m);
}
