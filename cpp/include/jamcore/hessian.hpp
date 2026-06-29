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

// Full enthalpy Hessian (2N+2, box DOF included) eigenvalues (ascending). The
// box-inclusive relaxation spectrum: these are the curvatures of the enthalpy the
// dynamics actually minimize. k<=0 -> dense (all 2N+2); k>0 -> Spectra lowest-k.
Eigen::VectorXd eigvals_full_spectrum(const System& sys, int k);

// Full enthalpy Hessian eigenpairs: lowest-k (eigenvalues ascending) with their
// eigenvectors as columns. k<=0 -> all 2N+2 modes (dense). The eigenvectors are
// the relaxation modes used to project terminal relaxation motion. Returns
// {eigenvalues (m), eigenvectors (2N+2 x m)}.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> eigvecs_full(const System& sys, int k);

}  // namespace jamcore
