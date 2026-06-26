#!/usr/bin/env bash
# Fast local build of jamrl._core for development / test iteration.
#
# Builds the extension directly with CMake+Ninja and drops the resulting
# _core*.so next to the package (src/jamrl/) so that
#     PYTHONPATH=src pytest
# picks it up without a full `pip install`.
#
# IMPORTANT: run this with the *target* environment active (so the default
# `python3` is the one you want to build for) — CMake's FindPython does not
# reliably honor a Python_EXECUTABLE hint that points outside the active env.
# For a robust, env-agnostic build use the canonical path instead:
#     pip install -e . --no-build-isolation
#
# Override the compiler with CC/CXX, e.g. to use GNU OpenMP:
#     CXX=g++-14 CC=gcc-14 scripts/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build/local"

PY="${PYTHON:-$(command -v python3)}"
PYBIND_DIR="$("${PY}" -m pybind11 --cmakedir)"
# Share the Python environment's own libomp (single OpenMP runtime in-process).
OMP_ROOT="$("${PY}" -c 'import sys; print(sys.prefix)')"

cmake -S "${ROOT}" -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="${PYBIND_DIR}" \
  -DPython_EXECUTABLE="${PY}" \
  -DPython_ROOT_DIR="${OMP_ROOT}" \
  -DPython_FIND_STRATEGY=LOCATION \
  -DJAMRL_OMP_ROOT="${OMP_ROOT}" \
  "$@"

cmake --build "${BUILD}" -j

# Place the freshly built module inside the package tree.
shopt -s nullglob
sofiles=("${BUILD}"/cpp/_core*.so)
if [[ ${#sofiles[@]} -eq 0 ]]; then
  echo "ERROR: no _core*.so produced under ${BUILD}/cpp" >&2
  exit 1
fi
cp "${sofiles[@]}" "${ROOT}/src/jamrl/"
echo "[build] installed: ${sofiles[*]##*/} -> src/jamrl/"
