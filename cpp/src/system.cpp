// jamcore: System construction + pybind registration.
#include "jamcore/system.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
namespace jamcore {

System make_system(int N, uint64_t seed, double phi0, double P) {
  System sys(N);
  sys.P = P;
  sys.z_iso = 4.0;  // 2d for frictionless disks in 2D

  // Radii: 50:50 bidisperse, R = 0.5 (small) and 0.7 (large).
  const int n_small = N / 2;
  for (int i = 0; i < N; ++i) sys.R[i] = (i < n_small) ? 0.5 : 0.7;

  // Reduced coordinates uniform in [0,1)^2 from the seeded stream.
  Rng rng(seed);
  for (int i = 0; i < N; ++i) {
    sys.x[2 * i] = rng.uniform();
    sys.x[2 * i + 1] = rng.uniform();
  }

  // Box sized so that phi == phi0 (gamma starts at 0).
  const double L = std::sqrt(sys.disk_area() / phi0);
  sys.x[2 * N] = std::log(L);
  sys.x[2 * N + 1] = 0.0;
  return sys;
}

void register_system_impl(py::module_& m) {
  py::class_<System>(m, "System", "Bidisperse 2D packing: reduced coords + box dofs + radii.")
      .def_readonly("N", &System::N)
      .def_readwrite("P", &System::P)
      .def_readwrite("z_iso", &System::z_iso)
      .def_property_readonly("L", &System::L)
      .def_property_readonly("lnL", &System::lnL)
      .def_property_readonly("gamma", &System::gamma)
      .def_property_readonly("area", &System::area)
      .def_property_readonly("phi", &System::phi)
      .def_property_readonly("disk_area", &System::disk_area)
      .def_property_readonly("mean_diam", &System::mean_diam)
      // Full state vector x (2N+2): [s..., lnL, gamma].
      .def_property(
          "x",
          [](const System& s) { return VectorXd(s.x); },
          [](System& s, const Eigen::Ref<const VectorXd>& v) {
            if (v.size() != s.dof())
              throw std::invalid_argument("x must have length 2N+2");
            s.x = v;
          })
      // Reduced coordinates as an [N,2] array.
      .def_property_readonly("s",
          [](const System& s) {
            py::array_t<double> a({s.N, 2});
            auto r = a.mutable_unchecked<2>();
            for (int i = 0; i < s.N; ++i) {
              r(i, 0) = s.x[2 * i];
              r(i, 1) = s.x[2 * i + 1];
            }
            return a;
          })
      .def_property_readonly("radii", [](const System& s) { return VectorXd(s.R); })
      .def("set_gamma", [](System& s, double g) { s.x[2 * s.N + 1] = g; }, py::arg("gamma"))
      .def("set_lnL", [](System& s, double v) { s.x[2 * s.N] = v; }, py::arg("lnL"))
      .def("clone", [](const System& s) { return System(s); })
      .def("__repr__", [](const System& s) {
        return "<jamrl.System N=" + std::to_string(s.N) +
               " phi=" + std::to_string(s.phi()) +
               " gamma=" + std::to_string(s.gamma()) +
               " P=" + std::to_string(s.P) + ">";
      });

  m.def("make_system", &make_system, py::arg("N"), py::arg("seed"),
        py::arg("phi0") = 0.80, py::arg("P") = 1e-3,
        "Build a fresh random 50:50 bidisperse packing at packing fraction phi0.");
}

}  // namespace jamcore

void register_system(py::module_& m) { jamcore::register_system_impl(m); }
