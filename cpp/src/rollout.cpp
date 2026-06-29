// jamcore: batch episode runner (policy MLP + OpenMP over episodes).
#include "jamcore/rollout.hpp"
#include "jamcore/evaluate.hpp"
#include "jamcore/lbfgs.hpp"
#include "jamcore/cell_list.hpp"
#include "jamcore/hessian.hpp"
#include "jamcore/moduli.hpp"

#include <cmath>
#include <vector>
#include <Eigen/SparseCore>

#ifdef JAMRL_HAVE_OPENMP
#include <omp.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
namespace jamcore {

uint64_t action_subseed(uint64_t seed) { return mix_seed(seed, 0x5151ULL, 0, 0); }

static void gather_contacts(const System& s, std::vector<int>& ci, std::vector<int>& cj) {
  const double L = s.L(), g = s.gamma();
  const double* X = s.x.data();
  const double* R = s.R.data();
  for_each_pair_cells(s, [&](int i, int j) {
    double dsy = X[2 * i + 1] - X[2 * j + 1];
    dsy -= std::round(dsy);
    double u = (X[2 * i] - X[2 * j]) + g * dsy;
    u -= std::round(u);
    const double rx = L * u, ry = L * dsy;
    const double sig = R[i] + R[j];
    if (rx * rx + ry * ry < sig * sig) {
      ci.push_back(i);
      cj.push_back(j);
    }
  });
}

static void fill_final_state(System& s, const SaveFlags& save, bool want_hessian,
                             bool want_spectrum, bool want_moduli, EpisodeOut& out) {
  EvalResult ev = evaluate(s, 0.0, 0.0);
  out.x_final = s.x;
  out.L = s.L();
  out.gamma = s.gamma();
  out.P_int = ev.P_int;
  out.n_contacts = ev.n_contacts;
  out.z_iso = s.z_iso;

  if (save.save_contacts) gather_contacts(s, out.contact_i, out.contact_j);

  Backbone bb = backbone_set(s);
  out.n_keep = bb.n_keep;
  out.n_rattlers = s.N - bb.n_keep;
  int ncb = 0;
  {
    std::vector<int> ci, cj;
    gather_contacts(s, ci, cj);
    for (size_t k = 0; k < ci.size(); ++k)
      if (bb.remap[ci[k]] >= 0 && bb.remap[cj[k]] >= 0) ++ncb;
  }
  out.z = (bb.n_keep > 0) ? 2.0 * ncb / bb.n_keep : 0.0;
  out.dz = out.z - out.z_iso;

  if (want_moduli) {
    out.B = bulk_modulus(s);
    out.G = shear_modulus(s);
    out.has_moduli = true;
  }
  if (want_hessian) {
    Eigen::SparseMatrix<double, Eigen::RowMajor> H = assemble_hessian_full(s, s.N >= EVAL_CELLS_MIN_N);
    H.makeCompressed();
    const int nnz = static_cast<int>(H.nonZeros());
    out.H_rows = static_cast<int>(H.rows());
    out.H_cols = static_cast<int>(H.cols());
    out.H_data = Eigen::Map<const VectorXd>(H.valuePtr(), nnz);
    out.H_indices.assign(H.innerIndexPtr(), H.innerIndexPtr() + nnz);
    out.H_indptr.assign(H.outerIndexPtr(), H.outerIndexPtr() + out.H_rows + 1);
    out.has_hessian = true;
  }
  if (want_spectrum) {
    out.eig = eigvals_dos(s, -1);  // dense backbone spectrum
    out.has_spectrum = true;
  }
}

static EpisodeOut run_one_episode(const System& proto, const Policy& pol, uint64_t seed,
                                  const EnvConfig& cfg, const SaveFlags& save,
                                  double phi_null_in, double g_null_in, double cost_null_in,
                                  bool want_hessian, bool want_spectrum, bool want_moduli) {
  EpisodeOut out;
  out.seed = seed;
  out.z_iso = proto.z_iso;

  System sys = make_system(proto.N, seed, cfg.phi0, proto.P);
  double phi_null = phi_null_in;
  double g_null = g_null_in;
  double cost_null = cost_null_in;
  const bool need_phi = std::isnan(phi_null) || phi_null <= 0.0;
  const bool need_G = cfg.reward_mode == REWARD_SHEAR && (std::isnan(g_null) || g_null == 0.0);
  const bool need_cost = cfg.reward_mode == REWARD_SPEED && (std::isnan(cost_null) || cost_null <= 0.0);
  if (need_phi || need_G || need_cost) {
    NullBaselines nb = compute_null_baselines(sys, cfg);  // one null run for all baselines
    if (need_phi) phi_null = nb.phi;
    if (need_G) g_null = nb.G;
    if (need_cost) cost_null = nb.cost;
  }
  out.phi_null = phi_null;
  out.cost_null = cost_null;

  Env env;
  env.cfg = cfg;
  env.reset(sys, phi_null, g_null, cost_null);
  VectorXd obs = env.observe();

  Rng arng(action_subseed(seed));
  std::vector<VectorXd> obsL, actL;
  std::vector<double> rewL;
  int guard = 0;
  while (!env.done && guard++ < cfg.T_cap + 4) {
    VectorXd a = pol.sample(obs, arng);
    Transition tr = env.step(a[0], a[1]);
    obsL.push_back(obs);
    actL.push_back(a);
    rewL.push_back(tr.reward);
    obs = tr.obs;
  }

  const int T = static_cast<int>(obsL.size());
  out.T = T;
  out.steps = T;
  out.obs.resize(T, OBS_DIM);
  out.act.resize(T, ACT_DIM);
  out.rew.resize(T);
  for (int t = 0; t < T; ++t) {
    out.obs.row(t) = obsL[t].transpose();
    out.act.row(t) = actL[t].transpose();
    out.rew[t] = rewL[t];
  }
  out.outcome = env.outcome;
  out.phi = env.sys.phi();
  out.cost_eval = static_cast<double>(env.total_eval);

  out.jammed = (env.outcome == CONVERGED || env.outcome == CAPPED || env.outcome == QUIESCED);
  if (out.jammed) {
    out.disp = env.disp;  // terminal relaxation displacement (2N+2)
    fill_final_state(env.sys, save, want_hessian, want_spectrum, want_moduli, out);
  }
  return out;
}

std::vector<EpisodeOut>
run_episodes_batch(const System& proto, const Policy& pol,
                   const std::vector<uint64_t>& seeds, const EnvConfig& cfg,
                   const SaveFlags& save, int parallel_mode,
                   const std::vector<double>& phi_null,
                   const std::vector<double>& g_null,
                   const std::vector<double>& cost_null) {
  const int E = static_cast<int>(seeds.size());
  std::vector<EpisodeOut> out(E);

  auto want = [&](int i) {
    bool wh = (save.save_hessian == SAVE_SPARSE || save.save_hessian == SAVE_DENSE) &&
              (i % std::max(1, save.hessian_stride) == 0);
    bool ws = (save.save_hessian == SAVE_SPECTRUM);
    return std::make_pair(wh, ws);
  };
  auto pn_of = [&](int i) { return (i < static_cast<int>(phi_null.size())) ? phi_null[i] : NAN; };
  auto gn_of = [&](int i) { return (i < static_cast<int>(g_null.size())) ? g_null[i] : NAN; };
  auto cn_of = [&](int i) { return (i < static_cast<int>(cost_null.size())) ? cost_null[i] : NAN; };

  const int prev_eigen = Eigen::nbThreads();
  if (parallel_mode == 0) {
    Eigen::setNbThreads(1);
#ifdef JAMRL_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < E; ++i) {
      auto w = want(i);
      out[i] = run_one_episode(proto, pol, seeds[i], cfg, save, pn_of(i), gn_of(i), cn_of(i),
                               w.first, w.second, save.save_moduli);
    }
  } else {
    for (int i = 0; i < E; ++i) {
      auto w = want(i);
      out[i] = run_one_episode(proto, pol, seeds[i], cfg, save, pn_of(i), gn_of(i), cn_of(i),
                               w.first, w.second, save.save_moduli);
    }
  }
  Eigen::setNbThreads(prev_eigen);
  return out;
}

// ---- pybind glue ----
static py::dict episode_to_dict(const EpisodeOut& e) {
  py::dict d;
  d["obs"] = MatrixXd(e.obs);
  d["act"] = MatrixXd(e.act);
  d["rew"] = VectorXd(e.rew);
  d["T"] = e.T;
  d["seed"] = e.seed;
  d["outcome"] = e.outcome;
  d["phi"] = e.phi;
  d["phi_null"] = e.phi_null;
  d["steps"] = e.steps;
  d["cost_eval"] = e.cost_eval;
  d["cost_null"] = e.cost_null;
  d["jammed"] = e.jammed;
  if (e.jammed) {
    d["x_final"] = VectorXd(e.x_final);
    d["disp"] = VectorXd(e.disp);
    d["L"] = e.L;
    d["gamma"] = e.gamma;
    d["P_int"] = e.P_int;
    d["n_contacts"] = e.n_contacts;
    d["n_rattlers"] = e.n_rattlers;
    d["n_keep"] = e.n_keep;
    d["z"] = e.z;
    d["z_iso"] = e.z_iso;
    d["dz"] = e.dz;
    const int nc = static_cast<int>(e.contact_i.size());
    py::array_t<int> contacts({nc, 2});
    auto r = contacts.mutable_unchecked<2>();
    for (int k = 0; k < nc; ++k) {
      r(k, 0) = e.contact_i[k];
      r(k, 1) = e.contact_j[k];
    }
    d["contacts"] = contacts;
    if (e.has_moduli) {
      d["B"] = e.B;
      d["G"] = e.G;
    }
    if (e.has_hessian) {
      d["H_data"] = VectorXd(e.H_data);
      d["H_indices"] = e.H_indices;
      d["H_indptr"] = e.H_indptr;
      d["H_shape"] = py::make_tuple(e.H_rows, e.H_cols);
    }
    if (e.has_spectrum) d["eig"] = VectorXd(e.eig);
  }
  return d;
}

void register_rollout_impl(py::module_& m) {
  py::class_<Policy>(m, "Policy", "Embedded tanh-MLP actor (mean + log_std).")
      .def(py::init([](VectorXd obs_mean, VectorXd obs_std, MatrixXd W0, VectorXd b0, MatrixXd W1,
                       VectorXd b1, MatrixXd Wmu, VectorXd bmu, VectorXd log_std) {
             Policy p;
             p.obs_mean = obs_mean;
             p.obs_std = obs_std;
             p.W0 = W0; p.b0 = b0; p.W1 = W1; p.b1 = b1; p.Wmu = Wmu; p.bmu = bmu;
             p.log_std = log_std;
             return p;
           }),
           py::arg("obs_mean"), py::arg("obs_std"), py::arg("W0"), py::arg("b0"), py::arg("W1"),
           py::arg("b1"), py::arg("Wmu"), py::arg("bmu"), py::arg("log_std"))
      .def("forward", [](const Policy& p, const VectorXd& o) { return VectorXd(p.forward(o)); })
      .def("sample", [](const Policy& p, const VectorXd& o, Rng& r) { return VectorXd(p.sample(o, r)); },
           py::arg("obs"), py::arg("rng"))
      .def_readonly("log_std", &Policy::log_std);

  py::class_<Rng>(m, "Rng", "Deterministic xoshiro256** stream with Box-Muller normals.")
      .def(py::init<uint64_t>(), py::arg("seed"))
      .def("reseed", &Rng::reseed, py::arg("seed"))
      .def("uniform", &Rng::uniform)
      .def("normal", &Rng::normal)
      .def("next_u64", &Rng::next_u64);

  py::class_<SaveFlags>(m, "SaveFlags")
      .def(py::init<>())
      .def_readwrite("save_hessian", &SaveFlags::save_hessian)
      .def_readwrite("hessian_stride", &SaveFlags::hessian_stride)
      .def_readwrite("save_moduli", &SaveFlags::save_moduli)
      .def_readwrite("save_contacts", &SaveFlags::save_contacts);

  m.def("action_subseed", &action_subseed, py::arg("seed"));
  m.def("mix_seed",
        [](uint64_t a, uint64_t b, uint64_t c, uint64_t d) { return mix_seed(a, b, c, d); },
        py::arg("a"), py::arg("b") = 0, py::arg("c") = 0, py::arg("d") = 0);

  m.def(
      "run_episodes_batch",
      [](const System& proto, const Policy& pol, const std::vector<uint64_t>& seeds,
         const EnvConfig& cfg, const SaveFlags& save, int parallel_mode,
         const std::vector<double>& phi_null, const std::vector<double>& g_null,
         const std::vector<double>& cost_null) {
        std::vector<EpisodeOut> res;
        {
          py::gil_scoped_release rel;
          res = run_episodes_batch(proto, pol, seeds, cfg, save, parallel_mode, phi_null, g_null,
                                   cost_null);
        }
        py::list out;
        for (const auto& e : res) out.append(episode_to_dict(e));
        return out;
      },
      py::arg("proto"), py::arg("policy"), py::arg("seeds"), py::arg("cfg") = EnvConfig{},
      py::arg("save") = SaveFlags{}, py::arg("parallel_mode") = 0,
      py::arg("phi_null") = std::vector<double>{}, py::arg("g_null") = std::vector<double>{},
      py::arg("cost_null") = std::vector<double>{},
      "Run E episodes under the policy; returns a list of per-episode result dicts.");
}

}  // namespace jamcore

void register_rollout(py::module_& m) { jamcore::register_rollout_impl(m); }
