// jamcore: the jamming MDP environment + pybind registration.
#include "jamcore/env.hpp"
#include "jamcore/moduli.hpp"
#include "jamcore/hessian.hpp"  // eigvals_full_spectrum (low-frequency VDOS obs)

#include <algorithm>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace jamcore {

static inline double clampd(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Observation-normalization constants for the G + VDOS extras (o[10..15]).
// Calibrate from a sample rollout (see plan verification): pick so typical
// values span ~[-1, 1]. G: tanh(a*(G/G_null - 1)); eigenfreqs: clamp((log10 w + C)/S).
static constexpr double OBS_G_SCALE = 1.0;     // a
static constexpr double OBS_VDOS_SHIFT = 3.0;  // C  (maps log10(w) in [-6,0] -> [-1,1])
static constexpr double OBS_VDOS_SCALE = 3.0;  // S

// Terminal objective reward for a jammed state, selected by reward_mode.
//   DENSITY: w_phi*(phi - phi_null)
//   SHEAR:   w_G*(G/G_null - 1) at fixed pressure (normalized stiffening vs the
//            fixed campaign null baseline; absolute fallback if G_null <= 0).
//            G_now is precomputed by the caller (Env::G_obs) to avoid a second
//            Hessian assembly.
//   SPEED:   w_speed*(cost_null - total_eval)/cost_null when the packing meets the
//            null density floor (phi >= phi_null); otherwise the DENSITY penalty
//            term, which is <0 below the floor and pushes the policy back over it.
static double objective_reward(const EnvConfig& cfg, double phi_now, double phi_null,
                               double G_null, double cost_null, long total_eval,
                               double G_now) {
  if (cfg.reward_mode == REWARD_SPEED) {
    if (phi_now >= phi_null && cost_null > 0.0)
      return cfg.w_speed * (cost_null - static_cast<double>(total_eval)) / cost_null;
    return cfg.w_phi * (phi_now - phi_null);  // below density floor
  }
  if (cfg.reward_mode == REWARD_SHEAR)
    return (G_null > 0.0) ? cfg.w_G * (G_now / G_null - 1.0)
                          : cfg.w_G * (G_now - G_null);
  return cfg.w_phi * (phi_now - phi_null);  // REWARD_DENSITY (default)
}

// Net displacement of the full-enthalpy coordinate vector x = [s..., ln L, gamma]
// (dim 2N+2) between two configs, with minimum-image on the reduced particle
// coordinates and the shear strain (raw on ln L). Used for projecting the
// terminal relaxation motion onto the Hessian relaxation modes.
static VectorXd coord_displacement(const VectorXd& x_post, const VectorXd& x_pre, int N) {
  VectorXd d = x_post - x_pre;
  for (int i = 0; i < 2 * N; ++i) d[i] -= std::round(d[i]);  // reduced particle coords
  d[2 * N + 1] -= std::round(d[2 * N + 1]);                  // shear strain gamma
  return d;
}

int finish_budget(double P, int finish_cap, int finish_cap_max) {
  const int scaled = static_cast<int>(std::lround(3e2 / std::sqrt(P)));
  return std::min(finish_cap_max, std::max(finish_cap, scaled));
}

void Env::reset(const System& proto, double phi_null_value, double G_null_value,
                double cost_null_value) {
  sys = proto;
  phi_null = phi_null_value;
  G_null = G_null_value;
  cost_null = cost_null_value;
  t = 0;
  prev_aP = 0.0;
  prev_aS = 0.0;
  quiet = 0;
  done = false;
  outcome = ONGOING;
  total_reward = 0.0;
  total_eval = 0;
  disp = VectorXd();
  last_ev = evaluate(sys, 0.0, 0.0);
  measure_obs_extras();  // G_obs + vdos_feat for the initial observation
}

// Measure the current shear modulus and a low-frequency VDOS summary of the
// full-enthalpy spectrum into G_obs / vdos_feat. vdos_feat = [w1..w4, omega*]
// (raw eigenfrequencies); observe() applies the log/tanh normalization. Skips
// the eigensolve when cfg.vdos_obs is off, and degrades gracefully (features
// stay 0) on loose/singular configs with no nonzero modes.
void Env::measure_obs_extras() {
  vdos_feat = VectorXd::Zero(N_VDOS_FEAT);
  if (cfg.k_vdos_moves > 0) vdos_vecs.resize(0, 0);  // reset retained move modes
  if (!cfg.obs_extras) {  // null runs never read the obs -> skip all extra work
    G_obs = 0.0;
    return;
  }
  G_obs = shear_modulus(sys);
  if (!cfg.vdos_obs) return;

  const int ndof = 2 * sys.N + 2;
  // The full-enthalpy spectrum's low end is a cluster of ~0 modes (2 global
  // translations + ~2 per rattler), then the real backbone band. We report the
  // lowest N_VDOS_FEAT *real* eigenfrequencies (the soft-mode edge ~ boson-peak
  // region, which tracks shear stability). Solve enough modes to clear the
  // cluster (~2 + 2*rattler_frac*N) and expose N_VDOS_FEAT real modes; honor an
  // explicit cfg.k_vdos if given.
  int k = cfg.k_vdos > 0 ? cfg.k_vdos
                         : std::max(16, static_cast<int>(std::lround(0.07 * ndof)) + 8);
  k = std::min(k, ndof);

  // With VDOS-directed moves enabled we also need the eigenVECTORS of the lowest
  // real modes (to displace along them next step). Compute pairs then; else the
  // cheaper eigenvalues-only path (byte-identical to before when k_vdos_moves==0).
  if (cfg.k_vdos_moves > 0) {
    const std::pair<VectorXd, MatrixXd> ep = eigvecs_full(sys, k);  // asc eigvals + vecs
    const VectorXd& eig = ep.first;
    const MatrixXd& V = ep.second;
    std::vector<int> real_idx;  // spectrum indices of real (nonzero) modes, ascending
    std::vector<double> w;
    for (int i = 0; i < eig.size(); ++i) {
      const double wi = std::sqrt(std::max(eig[i], 0.0));
      if (wi >= 1e-4) { w.push_back(wi); real_idx.push_back(i); }
    }
    if (w.empty()) return;  // unjammed/singular: leave features + vecs empty
    for (int j = 0; j < N_VDOS_FEAT; ++j)
      vdos_feat[j] = (j < static_cast<int>(w.size())) ? w[j] : w.back();
    const int keep = std::min<int>(cfg.k_vdos_moves, static_cast<int>(real_idx.size()));
    vdos_vecs.resize(ndof, keep);  // eigenvectors of the lowest `keep` real modes
    for (int j = 0; j < keep; ++j) vdos_vecs.col(j) = V.col(real_idx[j]);
    return;
  }

  const VectorXd eig = eigvals_full_spectrum(sys, k);  // ascending, lowest k
  std::vector<double> w;  // real (nonzero) eigenfrequencies, ascending
  w.reserve(static_cast<size_t>(eig.size()));
  for (int i = 0; i < eig.size(); ++i) {
    const double wi = std::sqrt(std::max(eig[i], 0.0));
    if (wi >= 1e-4) w.push_back(wi);  // drop the ~0 translation/rattler cluster
  }
  if (w.empty()) return;  // unjammed/singular: leave features at 0
  for (int j = 0; j < N_VDOS_FEAT; ++j)  // lowest N real modes (saturate if fewer)
    vdos_feat[j] = (j < static_cast<int>(w.size())) ? w[j] : w.back();
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
  // current shear modulus as normalized stiffening vs the null baseline
  const double Gref = (G_null > 0.0) ? G_null : 1.0;
  o[10] = std::tanh(OBS_G_SCALE * (G_obs / Gref - 1.0));
  // low-frequency VDOS summary: log-scaled eigenfrequencies (w1..w4, omega*)
  for (int j = 0; j < N_VDOS_FEAT; ++j) {
    const double wj = (j < vdos_feat.size()) ? vdos_feat[j] : 0.0;
    o[11 + j] = clampd((std::log10(wj + 1e-12) + OBS_VDOS_SHIFT) / OBS_VDOS_SCALE, -1.0, 1.0);
  }
  return o;
}

Transition Env::step(const VectorXd& a) {
  const double aP = clampd(a.size() > 0 ? a[0] : 0.0, -1.0, 1.0);
  const double aS = clampd(a.size() > 1 ? a[1] : 0.0, -1.0, 1.0);
  const double P = sys.P;

  // --- loads held during this macro-step (plan 3.5 + fixes 4.2/4.3) ---
  const double Peff = std::max(0.1 * P, P * (1.0 + cfg.kappa_P * aP));  // pressure floor
  const double dP = Peff - P;
  const double A0 = sys.area();                                        // frozen area (box-only)
  const double sigF = cfg.kappa_sigma * P * A0 * aS;                   // frozen shear force

  // --- VDOS-directed move: nudge particles along the retained lowest soft modes
  // before the held-load relaxation settles them (eigenvector-following; plan (c)).
  // Coeffs are the action components past [aP, aS]; each mode's particle part is
  // unit-normalized so vdos_move_amp is a fixed reduced-coord kick length. ---
  if (cfg.k_vdos_moves > 0 && a.size() > ACT_DIM && vdos_vecs.cols() > 0) {
    const int n2 = 2 * sys.N;  // particle DOF (exclude box lnL, gamma)
    const int kk = std::min<int>({cfg.k_vdos_moves, static_cast<int>(a.size()) - ACT_DIM,
                                  static_cast<int>(vdos_vecs.cols())});
    for (int j = 0; j < kk; ++j) {
      const double cj = clampd(a[ACT_DIM + j], -1.0, 1.0);
      if (cj == 0.0) continue;
      VectorXd v = vdos_vecs.col(j).head(n2);  // particle-DOF part of soft mode j
      const double nrm = v.norm();
      if (nrm > 1e-12) sys.x.head(n2) += (cfg.vdos_move_amp * cj / nrm) * v;
    }
  }

  VectorXd x_pre = sys.x;  // post-kick, pre-relaxation config (terminal-displacement baseline)
  RelaxResult R = lbfgs_relax(sys, dP, sigF, cfg.n_relax, cfg.lbfgs, cfg.tol);
  total_eval += R.n_eval;
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
    G_obs = 0.0;  // skip the moduli/eigensolve on a blown-up state
    vdos_feat = VectorXd::Zero(N_VDOS_FEAT);
  } else {
    measure_obs_extras();  // current G + VDOS (per-step obs and CONVERGED reward)
    if (is_converged_unbiased(ev, sys, cfg.tol)) {
      reward += objective_reward(cfg, phi_now, phi_null, G_null, cost_null, total_eval, G_obs);
      disp = coord_displacement(sys.x, x_pre, sys.N);  // motion into the jammed state
      dn = true;
      oc = CONVERGED;
    } else if (t >= cfg.T_cap || quiet >= cfg.quiesce_n) {
      // finish-and-measure: release loads, relax up to the pressure-scaled budget
      const int cap = finish_budget(P, cfg.finish_cap, cfg.finish_cap_max);
      x_pre = sys.x;  // re-baseline: the release relaxation is the terminal motion
      R = lbfgs_relax(sys, 0.0, 0.0, cap, cfg.lbfgs, cfg.tol);
      total_eval += R.n_eval;
      gamma_wrap(sys);
      ev = evaluate(sys, 0.0, 0.0);
      last_ev = ev;
      phi_now = sys.phi();
      if (is_converged_unbiased(ev, sys, cfg.tol)) {
        measure_obs_extras();  // refresh on the measured (released) jammed state
        reward += objective_reward(cfg, phi_now, phi_null, G_null, cost_null, total_eval, G_obs) -
                  cfg.trunc_pen;
        disp = coord_displacement(sys.x, x_pre, sys.N);  // motion into the jammed state
        oc = (quiet >= cfg.quiesce_n) ? QUIESCED : CAPPED;
      } else {
        reward -= cfg.fail_pen;
        oc = UNFINISHED;
      }
      dn = true;
    }
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

// Box-only convenience overload: no VDOS move (used by null baselines + tests).
Transition Env::step(double aP_raw, double aS_raw) {
  VectorXd a(2);
  a << aP_raw, aS_raw;
  return step(a);
}

NullBaselines compute_null_baselines(System proto, EnvConfig cfg) {
  // One zero-action episode -> all reward baselines: phi_null (density reached by
  // the null protocol; jamming minimization is path-dependent so a one-shot relax
  // would land elsewhere -- plan 5.5), G_null (shear modulus of that jammed state),
  // and cost_null (its total lbfgs evaluate() count, the SPEED-mode baseline).
  // Force DENSITY mode on the local copy so the per-step reward path never pays
  // for moduli and never recurses into the speed baseline; we measure G once.
  // Disable per-step obs extras too -- the null run's observations are unused.
  cfg.reward_mode = REWARD_DENSITY;
  cfg.obs_extras = false;
  Env env;
  env.cfg = cfg;
  env.reset(proto, 0.0);
  int guard = 0;
  while (!env.done && guard++ < cfg.T_cap + 4) {
    env.step(0.0, 0.0);
  }
  NullBaselines nb;
  nb.phi = env.sys.phi();
  nb.G = shear_modulus(env.sys);
  nb.cost = static_cast<double>(env.total_eval);
  return nb;
}

double compute_null_phi(System proto, const EnvConfig& cfg) {
  return compute_null_baselines(proto, cfg).phi;
}

std::pair<double, double> compute_null_phi_G(System proto, EnvConfig cfg) {
  NullBaselines nb = compute_null_baselines(proto, cfg);
  return {nb.phi, nb.G};
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
      .def_readwrite("w_speed", &EnvConfig::w_speed)
      .def_readwrite("c_step", &EnvConfig::c_step)
      .def_readwrite("fail_pen", &EnvConfig::fail_pen)
      .def_readwrite("trunc_pen", &EnvConfig::trunc_pen)
      .def_readwrite("quiesce_tol", &EnvConfig::quiesce_tol)
      .def_readwrite("quiesce_n", &EnvConfig::quiesce_n)
      .def_readwrite("finish_cap", &EnvConfig::finish_cap)
      .def_readwrite("finish_cap_max", &EnvConfig::finish_cap_max)
      .def_readwrite("vdos_obs", &EnvConfig::vdos_obs)
      .def_readwrite("k_vdos", &EnvConfig::k_vdos)
      .def_readwrite("k_vdos_moves", &EnvConfig::k_vdos_moves)
      .def_readwrite("vdos_move_amp", &EnvConfig::vdos_move_amp)
      .def_readwrite("tol", &EnvConfig::tol)
      .def_readwrite("lbfgs", &EnvConfig::lbfgs);

  py::class_<Env>(m, "Env", "Stateful box-control jamming MDP environment.")
      .def(py::init<>())
      .def_readwrite("cfg", &Env::cfg)
      .def_readonly("t", &Env::t)
      .def_readonly("done", &Env::done)
      .def_readonly("outcome", &Env::outcome)
      .def_readonly("total_reward", &Env::total_reward)
      .def_readonly("total_eval", &Env::total_eval)
      .def_readonly("phi_null", &Env::phi_null)
      .def_readonly("G_null", &Env::G_null)
      .def_readonly("cost_null", &Env::cost_null)
      .def_readonly("G_obs", &Env::G_obs)
      .def_property_readonly("vdos_feat", [](const Env& e) { return VectorXd(e.vdos_feat); })
      .def_property_readonly("disp", [](const Env& e) { return VectorXd(e.disp); })
      .def_property_readonly("sys", [](Env& e) { return &e.sys; },
                             py::return_value_policy::reference_internal)
      .def_property_readonly("phi", [](const Env& e) { return e.sys.phi(); })
      .def("reset",
           [](Env& e, const System& proto, double phi_null, double G_null, double cost_null) {
             e.reset(proto, phi_null, G_null, cost_null);
             return VectorXd(e.observe());
           },
           py::arg("proto"), py::arg("phi_null") = 0.0, py::arg("G_null") = 0.0,
           py::arg("cost_null") = 0.0, "Reset to `proto`; returns the initial observation.")
      .def("step",
           [](Env& e, double aP, double aS) { return transition_to_dict(e.step(aP, aS)); },
           py::arg("a_P"), py::arg("a_sigma"),
           "Apply one macro-step with raw box actions; returns a transition dict.")
      .def("step",
           [](Env& e, const VectorXd& a) { return transition_to_dict(e.step(a)); },
           py::arg("a"),
           "Apply one macro-step with the full action vector [aP, aS, VDOS coeffs...].")
      .def("observe", [](const Env& e) { return VectorXd(e.observe()); });

  m.def("compute_null_phi", &compute_null_phi, py::arg("proto"), py::arg("cfg") = EnvConfig{},
        "Density reached by a zero-action episode on `proto` (reward baseline).");
  m.def("compute_null_phi_G", &compute_null_phi_G, py::arg("proto"), py::arg("cfg") = EnvConfig{},
        "Density and shear modulus of a zero-action episode on `proto` "
        "(returns (phi_null, G_null); SHEAR-mode reward baseline).");
  m.def("compute_null_baselines",
        [](System proto, EnvConfig cfg) {
          NullBaselines nb = compute_null_baselines(proto, cfg);
          py::dict d;
          d["phi"] = nb.phi;
          d["G"] = nb.G;
          d["cost"] = nb.cost;
          return d;
        },
        py::arg("proto"), py::arg("cfg") = EnvConfig{},
        "All null-protocol reward baselines from one zero-action episode: "
        "{phi, G, cost} (cost = total lbfgs evaluate() count; SPEED-mode baseline).");

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
