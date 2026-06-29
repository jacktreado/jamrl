"""Configuration: dataclass defaults + YAML + argparse merge (plan section 5.1).

Resolution order (lowest to highest precedence):
  1. dataclass defaults (here),
  2. an optional YAML file (--config run.yaml),
  3. explicit CLI flags (argparse; flags win).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
from dataclasses import dataclass, fields
from typing import Any

import yaml

# Tuple-valued fields are stored/serialized as plain lists and parsed from
# comma-separated strings on the CLI.
_TUPLE_FIELDS = {"hidden", "eval_seeds"}


@dataclass(frozen=True)
class Config:
    # system
    N: int = 1024
    P: float = 1e-3
    phi0: float = 0.80
    # agent loading
    kappa_P: float = 1.0
    kappa_sigma: float = 0.5
    n_relax: int = 20
    T_cap: int = 60
    # reward
    reward_mode: str = "density"  # density | shear_modulus | speed
    w_phi: float = 400.0          # density-mode weight: w_phi*(phi - phi_null)
    w_G: float = 200.0            # shear-mode weight: w_G*(G - G_null); tune so reward ~ O(1)
    w_speed: float = 200.0        # speed-mode weight: w_speed*(cost_null - cost)/cost_null
    c_step: float = 0.01
    fail_pen: float = 2.0
    trunc_pen: float = 0.5
    quiesce_tol: float = 0.05
    quiesce_n: int = 3
    finish_cap: int = 12000        # lower bound on finish-and-measure iterations
    finish_cap_max: int = 60000    # hard ceiling; raise to reach jamming at very low P
    # tolerances
    ftol_abs: float = 1e-10
    ftol_rel_P: float = 1e-5
    ptol: float = 1e-4
    # learner
    algo: str = "ppo"  # ppo | cem
    backend: str = "auto"  # auto | torch | numpy  (PPO learner backend)
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    ppo_epochs: int = 6
    minibatch: int = 1024
    ent_coef: float = 3e-3
    vf_coef: float = 0.5
    logstd_init: float = -0.5
    hidden: tuple = (64, 64)
    cem_pop: int = 64
    cem_elite_frac: float = 0.25
    cem_sigma0: float = 0.3
    cem_eps_per_cand: int = 4
    # campaign / parallelism
    rounds: int = 1000
    workers: int = 64
    episodes_per_worker: int = 8
    eval_seeds: tuple = tuple(range(101, 107))
    parallel_mode: str = "episode"  # episode | intra
    threads_per_task: int = 16
    # data
    save_hessian: str = "sparse"  # none | spectrum | sparse | dense
    hessian_stride: int = 1
    compression: str = "gzip"
    # box-inclusive VDOS + relaxation-mode projection (postprocess; plan parts B/C)
    dos_full: bool = False   # also diagonalize the full enthalpy Hessian (box DOF incl.)
    proj_k: int = 60         # # of lowest relaxation modes for box-VDOS + projection
    # node-local scratch staging (HPC): write heavy outputs here, copy to the
    # persistent campaign at task end. Supports $VARS (e.g. "$TMPDIR"); empty=off.
    node_scratch: str = ""
    # slurm
    partition: str = ""
    account: str = ""
    time_rollout: str = "02:00:00"
    time_learn: str = "00:20:00"
    time_post: str = "01:00:00"
    mem_rollout: str = "8G"
    mem_learn: str = "8G"
    mem_post: str = "16G"
    dependency_mode: str = "afterany"  # afterany | afterok
    min_worker_frac: float = 0.6
    # learner device
    device: str = "cpu"  # cpu | cuda
    # misc
    seed: int = 12345
    campaign_root: str = "./campaigns"
    name: str = "run"

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k in _TUPLE_FIELDS:
            d[k] = list(d[k])
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)

    def save_yaml(self, path) -> None:
        with open(path, "w") as f:
            f.write(self.to_yaml())

    def config_hash(self) -> str:
        """sha1 of the canonical YAML (for provenance)."""
        return hashlib.sha1(self.to_yaml().encode()).hexdigest()

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        clean = {}
        for k, v in d.items():
            if k not in known:
                continue
            if k in _TUPLE_FIELDS and v is not None:
                v = tuple(v)
            clean[k] = v
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})


def _parse_tuple(text: str) -> tuple:
    text = str(text).strip()
    if not text:
        return tuple()
    parts = [p for p in text.replace(" ", "").split(",") if p != ""]
    out = []
    for p in parts:
        out.append(int(p) if p.lstrip("-").isdigit() else float(p))
    return tuple(out)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Auto-generate one CLI flag per Config field (unset -> SUPPRESS)."""
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.name in _TUPLE_FIELDS:
            parser.add_argument(flag, dest=f.name, type=str, default=argparse.SUPPRESS,
                                help=f"{f.name} (comma-separated)")
        elif f.type == "int":
            parser.add_argument(flag, dest=f.name, type=int, default=argparse.SUPPRESS)
        elif f.type == "float":
            parser.add_argument(flag, dest=f.name, type=float, default=argparse.SUPPRESS)
        elif f.type == "bool":
            parser.add_argument(flag, dest=f.name, type=str, default=argparse.SUPPRESS,
                                help=f"{f.name} (true/false)")
        else:  # str fields
            parser.add_argument(flag, dest=f.name, type=str, default=argparse.SUPPRESS)


def from_args(args: argparse.Namespace) -> Config:
    """Build a Config honoring defaults < YAML < explicit CLI flags."""
    base = Config()
    cfg_path = getattr(args, "config", None)
    if cfg_path:
        base = Config.from_yaml(cfg_path)

    overrides: dict[str, Any] = {}
    field_types = {f.name: f.type for f in fields(Config)}
    for name, ftype in field_types.items():
        if hasattr(args, name):
            v = getattr(args, name)
            if name in _TUPLE_FIELDS:
                v = _parse_tuple(v)
            elif ftype == "int":
                v = int(v)
            elif ftype == "float":
                v = float(v)
            elif ftype == "bool" and isinstance(v, str):
                v = v.strip().lower() in ("1", "true", "yes", "on")
            overrides[name] = v
    return base.replace(**overrides)


# ----------------------------------------------------------------------- #
# Bridges into the C++ core's EnvConfig / SaveFlags.
# ----------------------------------------------------------------------- #
_SAVE_HESSIAN_CODE = {"none": 0, "spectrum": 1, "sparse": 2, "dense": 3}
_REWARD_MODE_CODE = {"density": 0, "shear_modulus": 1, "speed": 2}


def env_config(cfg: Config):
    """Build a _core.EnvConfig from a Config."""
    from jamrl import _core

    if cfg.reward_mode not in _REWARD_MODE_CODE:
        raise ValueError(
            f"unknown reward_mode {cfg.reward_mode!r}; choose from {sorted(_REWARD_MODE_CODE)}"
        )

    ec = _core.EnvConfig()
    ec.phi0 = cfg.phi0
    ec.kappa_P = cfg.kappa_P
    ec.kappa_sigma = cfg.kappa_sigma
    ec.n_relax = cfg.n_relax
    ec.T_cap = cfg.T_cap
    ec.reward_mode = _REWARD_MODE_CODE[cfg.reward_mode]
    ec.w_phi = cfg.w_phi
    ec.w_G = cfg.w_G
    ec.w_speed = cfg.w_speed
    ec.c_step = cfg.c_step
    ec.fail_pen = cfg.fail_pen
    ec.trunc_pen = cfg.trunc_pen
    ec.quiesce_tol = cfg.quiesce_tol
    ec.quiesce_n = cfg.quiesce_n
    ec.finish_cap = cfg.finish_cap
    ec.finish_cap_max = cfg.finish_cap_max
    ec.tol.ftol_abs = cfg.ftol_abs
    ec.tol.ftol_rel_P = cfg.ftol_rel_P
    ec.tol.ptol = cfg.ptol
    return ec


def save_flags(cfg: Config, save_moduli: bool = True):
    """Build a _core.SaveFlags from a Config."""
    from jamrl import _core

    sf = _core.SaveFlags()
    sf.save_hessian = _SAVE_HESSIAN_CODE.get(cfg.save_hessian, 2)
    sf.hessian_stride = cfg.hessian_stride
    sf.save_moduli = save_moduli
    sf.save_contacts = True
    return sf


def parallel_mode_code(cfg: Config) -> int:
    return 0 if cfg.parallel_mode == "episode" else 1
