# Physics core — formulas and numerical contract

Reference for the C++ engine (`cpp/`). The analytic test gate (`tests/`) is the
source of truth; this document records the formulas it enforces. System: 2D
bidisperse harmonic disks, 50:50 radii `R = 0.5 / 0.7`, units `ε = m = 1`.

## State and box (§3.1)

State vector `x` of dimension `2N+2`:

```
x = [ s_0x, s_0y, ..., s_(N-1)x, s_(N-1)y,  ℓ = ln L,  γ ]
```

`s_i ∈ [0,1)²` reduced coords; `L` box side, area `A = L²`; `γ` Lees-Edwards
strain. Deformation `h = L·[[1,γ],[0,1]]`; real position `h·s_i`, real
separation `h·Δs`. Packing fraction `φ = (Σ π R_i²)/A`.

## Sheared minimum image + pair kernel (§3.2)

```
dsy = s_iy - s_jy ;  dsy -= round(dsy)
u   = (s_ix - s_jx) + γ·dsy ;  u -= round(u)
rx  = L·u ;  ry = L·dsy ;  r = sqrt(rx² + ry²) ;  σij = R_i + R_j
```

Contact iff `r < σij`; overlap `δ = 1 - r/σij`:

```
E_pair = ½ δ²        V' = -δ/σij       V'' = 1/σij²       n = (rx,ry)/r
gx_i += V'·n_x·L                  (− for j)
gy_i += V'·(γ·n_x + n_y)·L        (− for j)
```

Accumulators `sumVr = Σ V'·r`, `Eγ = Σ V'·n_x·r_y`, `E = Σ E_pair`:

```
P_int            = -sumVr/(2A)
∂H/∂ℓ  (unbiased) = sumVr + 2·P·A
∂H/∂ℓ  (biased)   = sumVr + 2·(P+ΔP)·A          # no shear coupling (§4.3)
∂H/∂γ  (unbiased) = Eγ
∂H/∂γ  (biased)   = Eγ - σF
H   = E + P·A
H_b = E + (P+ΔP)·A - σF·γ
```

Convergence (unbiased): `|∇H|_∞ < ftol_eff` and `|P_int−P|/P < ptol`, with
`ftol_eff = max(ftol_abs, ftol_rel_P · P · σ̄)`, `σ̄ = mean diameter = 1.2`.

## γ-wrap (§3.3)

```
k = round(γ); if k≠0: s_ix += k·s_iy for all i; γ -= k
s_i -= floor(s_i)
```

Energy-invariant relabeling (verified to machine zero); `γ ∈ [-½, ½]`.

## Numerical hazard fixes (§4)

1. **Armijo sub-precision spin** — accept when
   `H_b(x_new) ≤ H_b(x) + c1·a·gd + slack`, `slack = 1e-14·max(1e-30,|H_b|)`;
   reset history + step scale and count a stall on line-search failure; stop
   after `stall_max = 6` consecutive failures.
2. **Box runaway** — pressure floor `Peff = max(0.1·P, P·(1+κ_P·a_P))`; melt
   detection `φ < 0.3`.
3. **Joint γ/ℓ runaway** — shear is a generalized force frozen per macro-step,
   `σF = κ_σ·P·A₀·a_σ`; biased enthalpy `H_b = E + (P+ΔP)L² − σF·γ`.
4. **Low-P critical slowing** — finish budget `cap = min(60000, max(finish_cap,
   round(3e2/√P)))`.

## Cell list (§ Phase 4)

Lees-Edwards linked cell on reduced coordinates; the sheared minimum image is
handled inside the pair kernel, so the cell stencil only widens in `x` by
`⌈(1+|γ|)·frac⌉` to avoid missing contacts. Falls back to O(N²) when the box is
too small for a non-overlapping stencil. Matches brute force to machine
precision (identical contact set; `|ΔE| ≈ 0`, `|Δgrad| < 1e-13`).

### Scaling (single thread, Intel i5-1038NG7)

Per-`evaluate` wall time (random φ=0.80 config):

| N | brute (µs) | cells (µs) | speedup |
|---|---|---|---|
| 64 | 16.3 | 13.3 | 1.2× |
| 256 | 188.7 | 65.5 | 2.9× |
| 1024 | 2802.6 | 286.9 | 9.8× |
| 4096 | 35375.2 | 1225.4 | 28.9× |

Brute ≈ quadratic (≈13× per 4× N); cells ≈ linear (≈4.3× per 4× N).

## Hessian / moduli / DOS (§3.6–3.7)

Assembled in `hessian.cpp` / `moduli.cpp`; see `tests/test_hessian.py` and
`tests/test_moduli.py` for the enforced contract (`H·v` vs central FD to 1e-9,
symmetry to 1e-12, `B>0`, `G ≥ -1e-8` shear-stabilized, Schur-complement moduli
cross-checked by affine finite difference).
