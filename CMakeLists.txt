cmake_minimum_required(VERSION 3.20)
project(brainsort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BRAINSORT_BUILD_TESTS     "Build the library test suite"                              ON)
option(BRAINSORT_BUILD_BENCHMARK "Build the sortbench benchmark and its test suite"          ON)
option(BRAINSORT_BUILD_FUZZERS   "Build the libFuzzer targets (Clang only)"                  OFF)
option(SORTBENCH_NATIVE          "Compile with -march=native"                                OFF)
set(BRAINSORT_SANITIZE "" CACHE STRING "Sanitizers for every target, e.g. address,undefined or thread (GCC/Clang)")

# ---- the library: header-only ------------------------------------------------------
add_library(brainsort INTERFACE)
add_library(brainsort::brainsort ALIAS brainsort)
target_include_directories(brainsort INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(brainsort INTERFACE cxx_std_20)

include(GNUInstallDirs)
install(DIRECTORY include/brainsort DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES single_include/brainsort.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/brainsort/single_include)
install(TARGETS brainsort EXPORT brainsortTargets)
install(EXPORT brainsortTargets NAMESPACE brainsort:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/brainsort)

# ---- common compile options --------------------------------------------------------
# Identical optimisation flags for every algorithm of the benchmark: -O2, no
# -march=native unless asked (so numbers are comparable across machines).
# brainsort dispatches AVX2/BMI2 at run time either way.
# No fused multiply-add contraction: the benchmark's double keys come out of
# uniform_real_distribution, and a contracted a*b+c rounds differently, so the
# deterministic numbers would depend on the CPU (GCC contracts by default on
# ARM64; MSVC does not contract under its default /fp:precise).
add_library(brainsort_flags INTERFACE)
target_compile_options(brainsort_flags INTERFACE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive- /Zc:__cplusplus /EHsc /bigobj /utf-8>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -O2 -ffp-contract=off>)
if(SORTBENCH_NATIVE AND NOT MSVC)
  target_compile_options(brainsort_flags INTERFACE -march=native)
endif()
if(BRAINSORT_SANITIZE)
  if(MSVC)
    target_compile_options(brainsort_flags INTERFACE /fsanitize=${BRAINSORT_SANITIZE})
  else()
    target_compile_options(brainsort_flags INTERFACE -fsanitize=${BRAINSORT_SANITIZE} -fno-omit-frame-pointer -g)
    target_link_options(brainsort_flags INTERFACE -fsanitize=${BRAINSORT_SANITIZE})
  endif()
endif()
# Tests keep their assertions: undefine NDEBUG that the Release configuration adds.
set(BRAINSORT_ASSERTS_ON $<IF:$<CXX_COMPILER_ID:MSVC>,/UNDEBUG,-UNDEBUG>)

enable_testing()

# ---- library tests ---------------------------------------------------------------------
if(BRAINSORT_BUILD_TESTS)
  add_executable(brainsort_tests tests/api_tests.cpp)
  target_link_libraries(brainsort_tests PRIVATE brainsort brainsort_flags)
  target_compile_options(brainsort_tests PRIVATE ${BRAINSORT_ASSERTS_ON})
  add_test(NAME brainsort_tests COMMAND brainsort_tests)

  # The same suite on the portable scalar code, which is what every non-x86
  # target runs.
  add_executable(brainsort_tests_scalar tests/api_tests.cpp)
  target_link_libraries(brainsort_tests_scalar PRIVATE brainsort brainsort_flags)
  target_compile_options(brainsort_tests_scalar PRIVATE ${BRAINSORT_ASSERTS_ON})
  target_compile_definitions(brainsort_tests_scalar PRIVATE BRAINSORT_NO_SIMD=1)
  add_test(NAME brainsort_tests_scalar COMMAND brainsort_tests_scalar --quick)

  # The single-header distribution must be current and must compile on its own.
  add_executable(brainsort_single_header tests/single_header.cpp)
  target_include_directories(brainsort_single_header PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/single_include)
  target_link_libraries(brainsort_single_header PRIVATE brainsort_flags)
  target_compile_options(brainsort_single_header PRIVATE ${BRAINSORT_ASSERTS_ON})
  add_test(NAME brainsort_single_header COMMAND brainsort_single_header)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME brainsort_single_header_current
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/amalgamate.py --check)
  endif()

  # The fuzz target as a plain test: a few thousand random inputs.
  add_executable(brainsort_fuzz_smoke tests/fuzz_sort.cpp)
  target_link_libraries(brainsort_fuzz_smoke PRIVATE brainsort brainsort_flags)
  target_compile_options(brainsort_fuzz_smoke PRIVATE ${BRAINSORT_ASSERTS_ON})
  add_test(NAME brainsort_fuzz_smoke COMMAND brainsort_fuzz_smoke 3000)

  # The public API against std::sort, std::stable_sort and pdqsort.
  add_executable(brainsort_api_bench bench/api_bench.cpp)
  target_include_directories(brainsort_api_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  target_link_libraries(brainsort_api_bench PRIVATE brainsort brainsort_flags)
  # The run stamp fingerprints the measured sources; the binary finds them from here.
  target_compile_definitions(brainsort_api_bench PRIVATE NDEBUG SB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  find_package(Threads)
  if(Threads_FOUND)
    target_link_libraries(brainsort_tests PRIVATE Threads::Threads)
    target_link_libraries(brainsort_tests_scalar PRIVATE Threads::Threads)
  endif()
endif()

if(BRAINSORT_BUILD_FUZZERS)
  add_executable(brainsort_fuzz tests/fuzz_sort.cpp)
  target_link_libraries(brainsort_fuzz PRIVATE brainsort brainsort_flags)
  target_compile_definitions(brainsort_fuzz PRIVATE BRAINSORT_LIBFUZZER=1)
  target_compile_options(brainsort_fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer -g ${BRAINSORT_ASSERTS_ON})
  target_link_options(brainsort_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

# ---- the benchmark -----------------------------------------------------------------------
if(BRAINSORT_BUILD_BENCHMARK)
  add_library(sortbench_flags INTERFACE)
  target_include_directories(sortbench_flags INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  target_link_libraries(sortbench_flags INTERFACE brainsort brainsort_flags)

  add_executable(sortbench src/bench.cpp)
  target_link_libraries(sortbench PRIVATE sortbench_flags)
  target_compile_definitions(sortbench PRIVATE NDEBUG SB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  if(WIN32)
    target_link_libraries(sortbench PRIVATE psapi)
  endif()

  add_executable(sortbench_tests src/test.cpp)
  target_link_libraries(sortbench_tests PRIVATE sortbench_flags)
  # No NDEBUG: keep the bounds-check assertions in the Array view active.
  target_compile_options(sortbench_tests PRIVATE ${BRAINSORT_ASSERTS_ON})
  # The golden file lives next to the sources; the tests find it from here.
  target_compile_definitions(sortbench_tests PRIVATE SB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME sortbench_tests COMMAND sortbench_tests)
endif()
