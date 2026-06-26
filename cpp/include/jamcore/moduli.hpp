// jamcore/moduli.hpp — bulk/shear moduli via Schur complement (plan 3.7).
#pragma once
#include "jamcore/system.hpp"

namespace jamcore {

// B = (1/4A)[H_ℓℓ - H_ℓq H_qq⁺ H_qℓ],  q = (s, γ)
double bulk_modulus(const System& sys);

// G = (1/A)[H_γγ - H_γq H_qq⁺ H_qγ],  q = (s, ℓ)
// Shear-stabilized by construction (γ is a minimization dof) -> G >= 0.
double shear_modulus(const System& sys);

}  // namespace jamcore
