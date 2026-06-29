// jamcore/env.hpp — the box-control jamming MDP (plan section 3.5).
#pragma once
#include <utility>

#include "jamcore/system.hpp"
#include "jamcore/evaluate.hpp"
#include "jamcore/lbfgs.hpp"

namespace jamcore {

// Episode outcome taxonomy (plan section 3.5).
enum Outcome {
  ONGOING = -1,
  CONVERGED = 0,   // agent released and it jammed within a macro-step
  CAPPED = 1,      // T_cap hit -> finish-and-measure jammed
  QUIESCED = 2,    // quiet streak -> finish-and-measure jammed
  OVERLAP = 3,     // maxOv > 0.9
  MELT = 4,        // phi < 0.3
  BLOWUP = 5,      // non-finite enthalpy
  UNFINISHED = 6   // finish-and-measure failed to jam
};

constexpr int OBS_DIM = 10;
constexpr int ACT_DIM = 2;

// Reward objective selector (plan: multiple reward modes). DENSITY is the
// original denser-than-null objective; SHEAR rewards stiffer-than-null packings
// at the same fixed target pressure; SPEED rewards reaching a jammed state with
// fewer force/energy evaluations than the null protocol (gated on density floor).
enum RewardMode { REWARD_DENSITY = 0, REWARD_SHEAR = 1, REWARD_SPEED = 2 };

struct EnvConfig {
  // system (for per-episode config generation in the batch runner)
  double phi0 = 0.80;
  // loading
  double kappa_P = 1.0;
  double kappa_sigma = 0.5;
  int n_relax = 20;
  int T_cap = 60;
  // reward
  int reward_mode = REWARD_DENSITY;  // density | shear_modulus | speed
  double w_phi = 400.0;
  double w_G = 200.0;                 // shear-modulus reward weight (SHEAR mode)
  double w_speed = 200.0;             // speed reward weight (SPEED mode)
  double c_step = 0.01;
  double fail_pen = 2.0;
  double trunc_pen = 0.5;
  double quiesce_tol = 0.05;
  int quiesce_n = 3;
  int finish_cap = 12000;       // lower bound on finish-and-measure iterations
  int finish_cap_max = 60000;   // hard ceiling (raise to reach jamming at very low P)
  // tolerances + minimizer
  Tols tol;
  LBFGSParams lbfgs;
};

struct Transition {
  VectorXd obs;       // OBS_DIM
  double reward = 0.0;
  bool done = false;
  int outcome = ONGOING;
  // info (unbiased view after the macro-step)
  double phi = 0.0, P_int = 0.0, fInf = 0.0, gamma = 0.0, maxOv = 0.0, Egamma = 0.0;
  int n_contacts = 0, t = 0;
};

// Stateful environment: reset() then step() with raw (pre-clip) actions.
struct Env {
  System sys;
  EnvConfig cfg;
  double phi_null = 0.0;
  double G_null = 0.0;     // null-protocol shear modulus baseline (SHEAR mode)
  double cost_null = 0.0;  // null-protocol force-eval count baseline (SPEED mode)
  // episode state
  int t = 0;
  double prev_aP = 0.0, prev_aS = 0.0;
  int quiet = 0;
  bool done = false;
  int outcome = ONGOING;
  double total_reward = 0.0;
  long total_eval = 0;  // accumulated lbfgs evaluate() calls this episode (SPEED cost)
  VectorXd disp;        // terminal relaxation displacement (2N+2): set when jammed
  EvalResult last_ev;   // last unbiased evaluation

  void reset(const System& proto, double phi_null_value, double G_null_value = 0.0,
             double cost_null_value = 0.0);
  VectorXd observe() const;
  Transition step(double aP_raw, double aS_raw);
};

// Pressure-scaled finish-and-measure iteration budget (fix 4.4), capped at
// finish_cap_max (raise to chase jamming at very low target pressure).
int finish_budget(double P, int finish_cap, int finish_cap_max);

// Null (zero-action) protocol baselines from a single null episode: jammed
// density, shear modulus, and force-eval cost. cost is the total lbfgs
// evaluate() count of the null run (SPEED-mode reward baseline).
struct NullBaselines {
  double phi = 0.0;
  double G = 0.0;
  double cost = 0.0;
};
NullBaselines compute_null_baselines(System proto, EnvConfig cfg);

// Null (zero-action) jammed density for the reward baseline (plan 5.5).
double compute_null_phi(System proto, const EnvConfig& cfg);

// Null (zero-action) jammed density AND shear modulus, from one null run
// (SHEAR-mode reward baseline). Returns {phi_null, G_null}.
std::pair<double, double> compute_null_phi_G(System proto, EnvConfig cfg);

}  // namespace jamcore
