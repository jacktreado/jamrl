// jamcore: the jamming MDP environment + pybind registration.
#include "jamcore/env.hpp"
#include "jamcore/moduli.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace jamcore {

static inline double clampd(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Terminal objective reward for a jammed state, selected by reward_mode.
// DENSITY: w_phi*(phi - phi_null);  SHEAR: w_G*(G - G_null) at fixed pressure.
static double objective_reward(const EnvConfig& cfg, const System& sys, double phi_now,
                               double phi_null, double G_null) {
  if (cfg.reward_mode == REWARD_SHEAR)
    return cfg.w_G * (shear_modulus(sys) - G_null);
  return cfg.w_phi * (phi_now - phi_null);  // REWARD_DENSITY (default)
}

int finish_budget(double P, int finish_cap) {
  const int scaled = static_cast<int>(std::lround(3e2 / std::sqrt(P)));
  return std::min(60000, std::max(finish_cap, scaled));
}

void Env::reset(const System& proto, double phi_null_value, double G_null_value) {
  sys = proto;
  phi_null = phi_null_value;
  G_null = G_null_value;
  t = 0;
  prev_aP = 0.0;
  prev_aS = 0.0;
  quiet = 0;
  done = false;
  outcome = ONGOING;
  total_reward = 0.0;
  last_ev = evaluate(sys, 0.0, 0.0);
}

VectorXd Env::observe() const {
  const EvalResult& ev = last_ev;
  const double P = sys.P;
  const double A = sys.area();
  const double gamma = sys.gamma();
  const double phi = sys.phi();
  const double z = 2.0 * static_cast<double>(ev.n_contacts) / std::max(1, sys.N);

  VectorXd o(OBS_DIM);
  o[0] = clampd((std::log10(ev.fInf_unbiased + 1e-16) + 14.0) / 15.0, 0.0, 1.0);
  o[1] = std::tanh(((ev.P_int - P) / P) / 5.0);
  o[2] = std::tanh((ev.Egamma / (P * A)) / 5.0);
  o[3] = std::tanh(2.0 * gamma);
  o[4] = (phi - 0.8) * 10.0;
  o[5] = (z > 0.0) ? std::tanh(5.0 * (z / sys.z_iso - 1.0)) : -1.0;
  o[6] = std::tanh(20.0 * ev.maxOv);
  o[7] = static_cast<double>(t) / std::max(1, cfg.T_cap);
  o[8] = prev_aP;
  o[9] = prev_aS;
  return o;
}

Transition Env::step(double aP_raw, double aS_raw) {
  const double aP = clampd(aP_raw, -1.0, 1.0);
  const double aS = clampd(aS_raw, -1.0, 1.0);
  const double P = sys.P;

  // --- loads held during this macro-step (plan 3.5 + fixes 4.2/4.3) ---
  const double Peff = std::max(0.1 * P, P * (1.0 + cfg.kappa_P * aP));  // pressure floor
  const double dP = Peff - P;
  const double A0 = sys.area();                                        // frozen area
  const double sigF = cfg.kappa_sigma * P * A0 * aS;                   // frozen shear force

  lbfgs_relax(sys, dP, sigF, cfg.n_relax, cfg.lbfgs, cfg.tol);
  gamma_wrap(sys);
  EvalResult ev = evaluate(sys, 0.0, 0.0);  // unbiased view
  last_ev = ev;
  t += 1;

  const double amax = std::max(std::abs(aP), std::abs(aS));
  quiet = (amax < cfg.quiesce_tol) ? quiet + 1 : 0;

  double phi_now = sys.phi();
  double reward = 0.0;
  bool dn = false;
  int oc = ONGOING;

  const bool bad = !std::isfinite(ev.H) || ev.maxOv > 0.9 || phi_now < 0.3;
  if (bad) {
    reward -= cfg.fail_pen;
    dn = true;
    if (!std::isfinite(ev.H)) oc = BLOWUP;
    else if (ev.maxOv > 0.9) oc = OVERLAP;
    else oc = MELT;
  } else if (is_converged_unbiased(ev, sys, cfg.tol)) {
    reward += objective_reward(cfg, sys, phi_now, phi_null, G_null);
    dn = true;
    oc = CONVERGED;
  } else if (t >= cfg.T_cap || quiet >= cfg.quiesce_n) {
    // finish-and-measure: release loads, relax up to the pressure-scaled budget
    const int cap = finish_budget(P, cfg.finish_cap);
    lbfgs_relax(sys, 0.0, 0.0, cap, cfg.lbfgs, cfg.tol);
    gamma_wrap(sys);
    ev = evaluate(sys, 0.0, 0.0);
    last_ev = ev;
    phi_now = sys.phi();
    if (is_converged_unbiased(ev, sys, cfg.tol)) {
      reward += objective_reward(cfg, sys, phi_now, phi_null, G_null) - cfg.trunc_pen;
      oc = (quiet >= cfg.quiesce_n) ? QUIESCED : CAPPED;
    } else {
      reward -= cfg.fail_pen;
      oc = UNFINISHED;
    }
    dn = true;
  }
  reward -= cfg.c_step;  // per macro-step cost, every step

  prev_aP = aP;
  prev_aS = aS;
  done = dn;
  outcome = oc;
  total_reward += reward;

  Transition tr;
  tr.obs = observe();
  tr.reward = reward;
  tr.done = dn;
  tr.outcome = oc;
  tr.phi = phi_now;
  tr.P_int = ev.P_int;
  tr.fInf = ev.fInf_unbiased;
  tr.gamma = sys.gamma();
  tr.maxOv = ev.maxOv;
  tr.Egamma = ev.Egamma;
  tr.n_contacts = ev.n_contacts;
  tr.t = t;
  return tr;
}

double compute_null_phi(System proto, const EnvConfig& cfg) {
  // phi_null must equal the density an actual zero-action EPISODE reaches
  // (plan 5.5): jamming minimization is path-dependent, so a one-shot relax
  // would land in a different basin than the macro-stepped null rollout.
  Env env;
  env.cfg = cfg;
  env.reset(proto, 0.0);  // baseline irrelevant to dynamics
  int guard = 0;
  while (!env.done && guard++ < cfg.T_cap + 4) {
    env.step(0.0, 0.0);
  }
  return env.sys.phi();
}

std::pair<double, double> compute_null_phi_G(System proto, EnvConfig cfg) {
  // One zero-action episode -> both phi_null and the shear modulus of the
  // null-protocol jammed state (the SHEAR-mode reward baseline). Force DENSITY
  // mode on the local copy so the per-step reward path never pays for moduli;
  // we measure G once, at the end.
  cfg.reward_mode = REWARD_DENSITY;
  Env env;
  env.cfg = cfg;
  env.reset(proto, 0.0);
  int guard = 0;
  while (!env.done && guard++ < cfg.T_cap + 4) {
    env.step(0.0, 0.0);
  }
  return {env.sys.phi(), shear_modulus(env.sys)};
}

static py::dict transition_to_dict(const Transition& tr) {
  py::dict d;
  d["obs"] = VectorXd(tr.obs);
  d["reward"] = tr.reward;
  d["done"] = tr.done;
  d["outcome"] = tr.outcome;
  d["phi"] = tr.phi;
  d["P_int"] = tr.P_int;
  d["fInf"] = tr.fInf;
  d["gamma"] = tr.gamma;
  d["maxOv"] = tr.maxOv;
  d["Egamma"] = tr.Egamma;
  d["n_contacts"] = tr.n_contacts;
  d["t"] = tr.t;
  return d;
}

void register_env_impl(py::module_& m) {
  py::class_<EnvConfig>(m, "EnvConfig")
      .def(py::init<>())
      .def_readwrite("phi0", &EnvConfig::phi0)
      .def_readwrite("kappa_P", &EnvConfig::kappa_P)
      .def_readwrite("kappa_sigma", &EnvConfig::kappa_sigma)
      .def_readwrite("n_relax", &EnvConfig::n_relax)
      .def_readwrite("T_cap", &EnvConfig::T_cap)
      .def_readwrite("reward_mode", &EnvConfig::reward_mode)
      .def_readwrite("w_phi", &EnvConfig::w_phi)
      .def_readwrite("w_G", &EnvConfig::w_G)
      .def_readwrite("c_step", &EnvConfig::c_step)
      .def_readwrite("fail_pen", &EnvConfig::fail_pen)
      .def_readwrite("trunc_pen", &EnvConfig::trunc_pen)
      .def_readwrite("quiesce_tol", &EnvConfig::quiesce_tol)
      .def_readwrite("quiesce_n", &EnvConfig::quiesce_n)
      .def_readwrite("finish_cap", &EnvConfig::finish_cap)
      .def_readwrite("tol", &EnvConfig::tol)
      .def_readwrite("lbfgs", &EnvConfig::lbfgs);

  py::class_<Env>(m, "Env", "Stateful box-control jamming MDP environment.")
      .def(py::init<>())
      .def_readwrite("cfg", &Env::cfg)
      .def_readonly("t", &Env::t)
      .def_readonly("done", &Env::done)
      .def_readonly("outcome", &Env::outcome)
      .def_readonly("total_reward", &Env::total_reward)
      .def_readonly("phi_null", &Env::phi_null)
      .def_readonly("G_null", &Env::G_null)
      .def_property_readonly("sys", [](Env& e) { return &e.sys; },
                             py::return_value_policy::reference_internal)
      .def_property_readonly("phi", [](const Env& e) { return e.sys.phi(); })
      .def("reset",
           [](Env& e, const System& proto, double phi_null, double G_null) {
             e.reset(proto, phi_null, G_null);
             return VectorXd(e.observe());
           },
           py::arg("proto"), py::arg("phi_null") = 0.0, py::arg("G_null") = 0.0,
           "Reset to `proto`; returns the initial observation.")
      .def("step",
           [](Env& e, double aP, double aS) { return transition_to_dict(e.step(aP, aS)); },
           py::arg("a_P"), py::arg("a_sigma"),
           "Apply one macro-step with raw actions; returns a transition dict.")
      .def("observe", [](const Env& e) { return VectorXd(e.observe()); });

  m.def("compute_null_phi", &compute_null_phi, py::arg("proto"), py::arg("cfg") = EnvConfig{},
        "Density reached by a zero-action episode on `proto` (reward baseline).");
  m.def("compute_null_phi_G", &compute_null_phi_G, py::arg("proto"), py::arg("cfg") = EnvConfig{},
        "Density and shear modulus of a zero-action episode on `proto` "
        "(returns (phi_null, G_null); SHEAR-mode reward baseline).");

  // Outcome-code constants for Python-side decoding.
  py::dict oc;
  oc["ongoing"] = (int)ONGOING;
  oc["converged"] = (int)CONVERGED;
  oc["capped"] = (int)CAPPED;
  oc["quiesced"] = (int)QUIESCED;
  oc["overlap"] = (int)OVERLAP;
  oc["melt"] = (int)MELT;
  oc["blowup"] = (int)BLOWUP;
  oc["unfinished"] = (int)UNFINISHED;
  m.attr("OUTCOMES") = oc;
}

}  // namespace jamcore

void register_env(py::module_& m) { jamcore::register_env_impl(m); }
