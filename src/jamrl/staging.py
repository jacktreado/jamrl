"""Node-local scratch staging for HPC I/O — write-to-scratch, copy-out-at-end.

On many clusters the shared/parallel filesystem that holds the persistent
campaign (e.g. ``/home/data/<user>/campaigns``) is slow for the many small
writes a rollout array produces, and admins ask jobs to write to the compute
node's *local* scratch first and copy results back when the job finishes.

A jamrl campaign is a pipeline of separate SLURM jobs that may run on different
nodes (rollout array -> learn -> postprocess). Node-local scratch is per-node
and ephemeral, so one node cannot read another's scratch. The correct pattern is
therefore **per-task stage-out of heavy writes**: a task writes its big outputs
to node scratch and copies each finished file to its persistent location when
the write completes. Reads still come straight from the persistent campaign
(few, and reads do not hammer the metadata server the way concurrent writes do),
and genuinely shared state — the per-key ``null_cache/`` shards (each written
atomically) and the summary parquets — stays on the persistent filesystem.

Enable by setting ``node_scratch`` (config field / ``--node-scratch``) or the
``JAMRL_NODE_SCRATCH`` environment variable to a path, typically ``$TMPDIR``
(SLURM auto-creates and auto-cleans it per job). Empty / unset / unusable ->
writes happen in place, so local (non-SLURM) runs are unaffected.
"""
from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import sys

_ENV_VAR = "JAMRL_NODE_SCRATCH"
_taskdir: str | None = None  # memoized per process


def _warn(msg: str) -> None:
    print(f"[staging] {msg}", file=sys.stderr)


def resolve_base(cfg=None) -> str | None:
    """Return a usable node-scratch base directory, or None (staging disabled).

    Precedence: ``JAMRL_NODE_SCRATCH`` env var, then ``cfg.node_scratch``.
    ``$VARS`` and ``~`` are expanded; an unusable path falls back to None.
    """
    raw = os.environ.get(_ENV_VAR) or (getattr(cfg, "node_scratch", "") if cfg else "")
    if not raw:
        return None
    base = os.path.expandvars(os.path.expanduser(raw))
    if "$" in base:  # an unset variable survived expansion
        _warn(f"node_scratch {raw!r} expanded to {base!r} (unset var?); writing in place")
        return None
    try:
        os.makedirs(base, exist_ok=True)
        if not os.access(base, os.W_OK):
            raise OSError("not writable")
    except OSError as e:
        _warn(f"node_scratch {base!r} unusable ({e}); writing in place")
        return None
    return base


def enabled(cfg=None) -> bool:
    return resolve_base(cfg) is not None


def _task_dir(base: str) -> str:
    """A unique per-task working directory under `base` (cleaned at exit)."""
    global _taskdir
    if _taskdir and os.path.isdir(_taskdir):
        return _taskdir
    parts = ["jamrl", os.environ.get("SLURM_JOB_ID", str(os.getpid()))]
    at = os.environ.get("SLURM_ARRAY_TASK_ID")
    if at is not None:
        parts.append(f"a{at}")
    _taskdir = os.path.join(base, "-".join(parts))
    os.makedirs(_taskdir, exist_ok=True)
    atexit.register(lambda d=_taskdir: shutil.rmtree(d, ignore_errors=True))
    return _taskdir


def _atomic_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".stage.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)  # tmp and dst share the persistent FS -> atomic rename


@contextlib.contextmanager
def output(persistent_path: str, cfg=None):
    """Yield a destination path for an atomic writer.

    Staging on  -> yields a node-scratch path; on success the finished file is
                   copied to ``persistent_path`` and the scratch copy removed.
    Staging off -> yields ``persistent_path`` (writer writes in place).

    The writer must write atomically to the yielded path (e.g. via
    ``storage.atomic_path`` / ``np.savez`` over a temp file). On scratch the
    writer's own atomicity is local; the cross-filesystem copy-out is atomic on
    the persistent side.
    """
    base = resolve_base(cfg)
    if base is None:
        yield persistent_path
        return
    local = os.path.join(_task_dir(base), os.path.basename(persistent_path))
    ok = False
    try:
        yield local
        ok = True
    finally:
        if ok and os.path.exists(local):
            _atomic_copy(local, persistent_path)
        with contextlib.suppress(OSError):
            if os.path.exists(local):
                os.remove(local)
