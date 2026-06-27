// jamcore/rollout.hpp — batch episode runner (plan 3.8).
#pragma once
#include <vector>
#include <cstdint>
#include "jamcore/system.hpp"
#include "jamcore/env.hpp"
#include "jamcore/mlp.hpp"

namespace jamcore {

enum SaveHessian { SAVE_NONE = 0, SAVE_SPECTRUM = 1, SAVE_SPARSE = 2, SAVE_DENSE = 3 };

struct SaveFlags {
  int save_hessian = SAVE_NONE;
  int hessian_stride = 1;
  bool save_moduli = false;
  bool save_contacts = true;
};

// Pure-C++ episode result (no Python objects -> safe to fill under OpenMP).
struct EpisodeOut {
  // trajectory
  MatrixXd obs;   // [T, OBS_DIM]
  MatrixXd act;   // [T, ACT_DIM] (raw, pre-clip)
  VectorXd rew;   // [T]
  int T = 0;
  uint64_t seed = 0;
  int outcome = ONGOING;
  double phi = 0.0, phi_null = 0.0;
  int steps = 0;

  // final jammed state (only if outcome in {converged, capped, quiesced})
  bool jammed = false;
  VectorXd x_final;  // 2N+2
  double L = 0.0, gamma = 0.0, P_int = 0.0;
  int n_contacts = 0, n_rattlers = 0, n_keep = 0;
  double z = 0.0, z_iso = 4.0, dz = 0.0;
  std::vector<int> contact_i, contact_j;

  bool has_moduli = false;
  double B = 0.0, G = 0.0;

  bool has_hessian = false;  // CSR of the full enthalpy Hessian
  VectorXd H_data;
  std::vector<int> H_indices, H_indptr;
  int H_rows = 0, H_cols = 0;

  bool has_spectrum = false;  // backbone DOS eigenvalues
  VectorXd eig;
};

// Per-episode action-noise sub-seed (separate stream from config generation).
uint64_t action_subseed(uint64_t seed);

// Run E episodes (one per seed). proto supplies N and P; cfg.phi0 the density.
// phi_null aligned with seeds (empty -> computed per seed internally).
// g_null aligned with seeds for SHEAR-mode baseline (empty -> computed per seed).
// parallel_mode: 0 = OpenMP over episodes (each single-threaded, deterministic),
//                1 = serial episodes with intra-episode Eigen threading.
std::vector<EpisodeOut>
run_episodes_batch(const System& proto, const Policy& pol,
                   const std::vector<uint64_t>& seeds, const EnvConfig& cfg,
                   const SaveFlags& save, int parallel_mode,
                   const std::vector<double>& phi_null,
                   const std::vector<double>& g_null = {});

}  // namespace jamcore
