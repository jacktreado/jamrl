cmake_minimum_required(VERSION 3.21)
project(jamrl LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standard / build type
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# -march=native is gated because cluster login nodes may differ from compute
# nodes (mismatched ISA -> SIGILL). See plan section 6.7.
option(JAMRL_NATIVE       "Compile with -march=native"                ON)
option(JAMRL_USE_SPECTRA  "Build with the Spectra sparse eigensolver" ON)
option(JAMRL_FETCH        "FetchContent deps not found on the system" ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ---------------------------------------------------------------------------
# Eigen 3.4+  (prefer system / brew install, fall back to FetchContent)
# ---------------------------------------------------------------------------
find_package(Eigen3 3.4 QUIET CONFIG)
if(NOT Eigen3_FOUND)
  if(JAMRL_FETCH)
    message(STATUS "[jamrl] Eigen not found; fetching 3.4.0")
    FetchContent_Declare(eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG 3.4.0
      GIT_SHALLOW TRUE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
  else()
    message(FATAL_ERROR "[jamrl] Eigen3 not found and JAMRL_FETCH=OFF")
  endif()
else()
  message(STATUS "[jamrl] Using system Eigen ${Eigen3_VERSION}")
endif()

# ---------------------------------------------------------------------------
# pybind11  (prefer pip-installed config, fall back to FetchContent)
# ---------------------------------------------------------------------------
find_package(pybind11 2.11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  if(JAMRL_FETCH)
    message(STATUS "[jamrl] pybind11 not found; fetching v2.13.6")
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)
  else()
    message(FATAL_ERROR "[jamrl] pybind11 not found and JAMRL_FETCH=OFF")
  endif()
else()
  message(STATUS "[jamrl] Using pybind11 ${pybind11_VERSION}")
endif()

# ---------------------------------------------------------------------------
# Spectra 1.0+  (header-only, optional; for low-frequency DOS at large N)
# ---------------------------------------------------------------------------
set(JAMRL_SPECTRA_INCLUDE "")
if(JAMRL_USE_SPECTRA)
  find_package(Spectra QUIET CONFIG)
  if(Spectra_FOUND)
    message(STATUS "[jamrl] Using system Spectra")
  elseif(JAMRL_FETCH)
    message(STATUS "[jamrl] Spectra not found; fetching v1.0.1")
    FetchContent_Declare(spectra
      GIT_REPOSITORY https://github.com/yixuan/spectra.git
      GIT_TAG v1.0.1
      GIT_SHALLOW TRUE)
    FetchContent_GetProperties(spectra)
    if(NOT spectra_POPULATED)
      FetchContent_Populate(spectra)
    endif()
    set(JAMRL_SPECTRA_INCLUDE "${spectra_SOURCE_DIR}/include")
  else()
    message(WARNING "[jamrl] Spectra requested but unavailable; disabling")
    set(JAMRL_USE_SPECTRA OFF)
  endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP  (compiler-provided; AppleClang needs Homebrew libomp hints)
# ---------------------------------------------------------------------------
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND APPLE)
  # IMPORTANT: prefer the Python environment's own libomp (JAMRL_OMP_ROOT,
  # typically the conda/venv prefix). Linking a *second* OpenMP runtime
  # (e.g. Homebrew's) into a process that already loaded one via numpy/numba
  # triggers "OMP Error #15: multiple copies of the OpenMP runtime" and may
  # crash. Sharing the env's libomp keeps a single runtime in-process.
  set(_omp_candidates "")
  if(JAMRL_OMP_ROOT)
    list(APPEND _omp_candidates "${JAMRL_OMP_ROOT}")
  endif()
  # The Python environment's own prefix (so `pip install` shares its libomp).
  if(DEFINED Python_EXECUTABLE)
    get_filename_component(_pybin "${Python_EXECUTABLE}" DIRECTORY)
    get_filename_component(_pyprefix "${_pybin}" DIRECTORY)
    list(APPEND _omp_candidates "${_pyprefix}")
  endif()
  if(DEFINED Python_PREFIX)
    list(APPEND _omp_candidates "${Python_PREFIX}")
  endif()
  list(APPEND _omp_candidates /usr/local/opt/libomp /opt/homebrew/opt/libomp)
  foreach(_omp ${_omp_candidates})
    if(EXISTS "${_omp}/include/omp.h" AND EXISTS "${_omp}/lib/libomp.dylib")
      message(STATUS "[jamrl] Trying libomp at ${_omp}")
      set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_omp}/include")
      set(OpenMP_CXX_LIB_NAMES "omp")
      set(OpenMP_omp_LIBRARY "${_omp}/lib/libomp.dylib")
      find_package(OpenMP QUIET)
      if(OpenMP_CXX_FOUND)
        set(JAMRL_OMP_RPATH "${_omp}/lib" CACHE INTERNAL "rpath for the chosen libomp")
        break()
      endif()
    endif()
  endforeach()
endif()
if(OpenMP_CXX_FOUND)
  message(STATUS "[jamrl] OpenMP enabled")
else()
  message(STATUS "[jamrl] OpenMP NOT found; #pragma omp will be ignored (serial build)")
endif()

add_subdirectory(cpp)
