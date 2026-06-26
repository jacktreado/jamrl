// jamcore/mlp.hpp — embedded tanh-MLP policy forward pass (plan 3.8).
#pragma once
#include "jamcore/types.hpp"

namespace jamcore {

// Actor-facing policy: two tanh hidden layers -> linear action mean, plus a
// per-action log_std. Weights/normalizer must match the PyTorch learner exactly.
struct Policy {
  VectorXd obs_mean, obs_std;  // [obs_dim]
  MatrixXd W0; VectorXd b0;    // [h0, obs_dim], [h0]
  MatrixXd W1; VectorXd b1;    // [h1, h0],      [h1]
  MatrixXd Wmu; VectorXd bmu;  // [act_dim, h1], [act_dim]
  VectorXd log_std;            // [act_dim]

  // Action mean given a raw observation (normalization applied internally).
  VectorXd forward(const VectorXd& obs) const {
    VectorXd on = (obs - obs_mean).cwiseQuotient(obs_std);
    VectorXd h0 = (W0 * on + b0).array().tanh();
    VectorXd h1 = (W1 * h0 + b1).array().tanh();
    return Wmu * h1 + bmu;
  }
  int act_dim() const { return static_cast<int>(log_std.size()); }

  // Sample a raw (pre-clip) action: a = mu + exp(log_std) ⊙ xi, xi ~ N(0,1).
  // Used by both the batch runner and the Python parity loop -> bitwise match.
  VectorXd sample(const VectorXd& obs, Rng& rng) const {
    VectorXd mu = forward(obs);
    VectorXd a(act_dim());
    for (int d = 0; d < act_dim(); ++d) a[d] = mu[d] + std::exp(log_std[d]) * rng.normal();
    return a;
  }
};

}  // namespace jamcore
