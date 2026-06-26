// jamcore/hessian.hpp — sparse enthalpy Hessian + backbone DOS Hessian.
#pragma once
#include <vector>
#include <Eigen/SparseCore>
#include "jamcore/system.hpp"

namespace jamcore {

// Full unbiased enthalpy Hessian over (s, ℓ, γ), dimension 2N+2 (plan 3.6).
// Symmetric; includes the +4·P·A box term on the ℓℓ entry.
Eigen::SparseMatrix<double> assemble_hessian_full(const System& sys, bool use_cells = true);

// Result of the rattler-removed backbone construction.
struct Backbone {
  std::vector<int> keep;          // kept particle indices (>=3 contacts, iterated)
  std::vector<int> remap;         // global index -> backbone index, or -1
  int n_keep = 0;
};

Backbone backbone_set(const System& sys);

// Backbone (rattler-removed) real-space position Hessian, dimension 2·n_keep
// (plan 3.6): K = V''·nnᵀ + (V'/r)(I − nnᵀ), fixed box, no box dofs.
Eigen::SparseMatrix<double> assemble_hessian_dos(const System& sys, const Backbone& bb);

// DOS eigenvalues (ascending). k<=0 -> dense (all 2·n_keep); k>0 -> Spectra
// smallest-magnitude k (shift-invert near 0), falling back to dense if needed.
Eigen::VectorXd eigvals_dos(const System& sys, int k);

}  // namespace jamcore
