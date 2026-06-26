# jamrl._core extension module
file(GLOB JAMCORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

pybind11_add_module(_core
  bindings/pybind_module.cpp
  ${JAMCORE_SOURCES}
)

target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(_core PRIVATE Eigen3::Eigen)

target_compile_options(_core PRIVATE
  -O3 -funroll-loops
  $<$<BOOL:${JAMRL_NATIVE}>:-march=native>
)

if(JAMRL_USE_SPECTRA)
  target_compile_definitions(_core PRIVATE JAMRL_USE_SPECTRA=1)
  if(TARGET Spectra::Spectra)
    target_link_libraries(_core PRIVATE Spectra::Spectra)
  elseif(JAMRL_SPECTRA_INCLUDE)
    target_include_directories(_core PRIVATE "${JAMRL_SPECTRA_INCLUDE}")
  endif()
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_core PRIVATE JAMRL_HAVE_OPENMP=1)
  # Resolve @rpath/libomp.dylib to the same physical library the Python env
  # already loaded, so dyld shares one OpenMP runtime (avoids OMP Error #15).
  if(JAMRL_OMP_RPATH)
    target_link_options(_core PRIVATE "-Wl,-rpath,${JAMRL_OMP_RPATH}")
  endif()
endif()

# Install the extension into the package directory (used by scikit-build-core).
install(TARGETS _core LIBRARY DESTINATION jamrl)
