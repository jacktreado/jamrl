// jamcore/types.hpp — shared aliases, deterministic RNG, parameter structs.
#pragma once
#include <cstdint>
#include <cmath>
#include <Eigen/Dense>

namespace jamcore {

using Eigen::VectorXd;
using Eigen::MatrixXd;

#ifndef JAMCORE_PI
#define JAMCORE_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Deterministic RNG: splitmix64 for seed mixing, xoshiro256** for the stream.
//
// Bitwise reproducibility per episode (plan section 4.6) relies on a fixed,
// platform-independent generator rather than <random> distributions.
// ---------------------------------------------------------------------------
struct SplitMix64 {
  uint64_t state;
  explicit SplitMix64(uint64_t seed) : state(seed) {}
  inline uint64_t next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
};

// Mix four 64-bit values into one (used for per-episode seed derivation).
inline uint64_t mix_seed(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  SplitMix64 sm(a);
  uint64_t h = sm.next();
  h ^= 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2) + b;
  SplitMix64 s2(h);
  h = s2.next();
  h ^= 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2) + c;
  SplitMix64 s3(h);
  h = s3.next();
  h ^= 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2) + d;
  SplitMix64 s4(h);
  return s4.next();
}

struct Rng {
  uint64_t s[4];
  bool has_cached = false;
  double cached = 0.0;

  explicit Rng(uint64_t seed) { reseed(seed); }
  void reseed(uint64_t seed) {
    SplitMix64 sm(seed);
    for (int i = 0; i < 4; ++i) s[i] = sm.next();
    has_cached = false;
    cached = 0.0;
  }
  static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  inline uint64_t next_u64() {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }
  // Uniform in [0, 1) with 53 bits of mantissa.
  inline double uniform() {
    return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
  }
  // Standard normal via Box-Muller (cache the paired sample).
  inline double normal() {
    if (has_cached) {
      has_cached = false;
      return cached;
    }
    double u1 = uniform();
    double u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;  // guard log(0)
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 2.0 * JAMCORE_PI * u2;
    cached = r * std::sin(theta);
    has_cached = true;
    return r * std::cos(theta);
  }
};

// ---------------------------------------------------------------------------
// Parameter structs (mirrored by the Python Config dataclass, plan section 5.1)
// ---------------------------------------------------------------------------
struct LBFGSParams {
  int memory = 8;
  double c1 = 1e-4;     // Armijo sufficient-decrease constant
  int max_ls = 30;      // max backtracking steps per line search
  int stall_max = 6;    // consecutive line-search failures before "stalled"
};

// Convergence tolerances (unbiased criterion, plan section 3.4).
struct Tols {
  double ftol_abs = 1e-10;
  double ftol_rel_P = 1e-5;
  double ptol = 1e-4;
};

}  // namespace jamcore
