# Host (desktop) unit tests for the ORT-free C++ headers.
#
# The shipping library is built by ../CMakeLists.txt via the Android NDK and links ONNX Runtime, so it
# cannot run on a host. But three headers carry real, fail-closed logic with no ORT dependency, and
# until now the entire C++ tree was compile/link-verified only — no C++ test target existed anywhere,
# which is why #8's save->load smoke and the C++ check_compat mirror had nowhere to live:
#
#   * handoff_io.h              — weight_handoff_map.json reader + the check_compat semver mirror
#   * constants/merger_variant.h — the typed MergerVariant that replaced string dispatch (#6)
#   * mem_probe.h               — parse_vmrss_kb, explicitly labelled "unit-testable without /proc"
#
# Build + run (host toolchain, NOT the NDK):
#   cmake -S src/main/cpp/tests -B build/cpp-tests && cmake --build build/cpp-tests && ctest --test-dir build/cpp-tests --output-on-failure
#
# `logging.h` includes <android/log.h>; shims/ provides a host stand-in that prints to stderr.

cmake_minimum_required(VERSION 3.22.1)
project(mobiletransformers_host_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

enable_testing()

add_executable(host_tests
        test_handoff_io.cpp
        test_merger_variant.cpp
        test_mem_probe.cpp)

target_include_directories(host_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/shims   # <android/log.h> stand-in
        ${CMAKE_CURRENT_SOURCE_DIR}/..)     # the headers under test

target_link_libraries(host_tests PRIVATE GTest::gtest_main nlohmann_json::nlohmann_json)

# The shared cross-language fixture (tests/fixtures/check_compat_cases.json) is read at runtime.
target_compile_definitions(host_tests PRIVATE
        MTF_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../..")

include(GoogleTest)
gtest_discover_tests(host_tests)
