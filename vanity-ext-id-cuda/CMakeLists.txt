cmake_minimum_required(VERSION 3.18)
project(vanity-ext-id-cuda VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# CUDA Configuration for A100 GPUs
# =============================================================================

# Detect CUDA
find_package(CUDAToolkit REQUIRED)

# A100 = sm_80, also support sm_86 (RTX 30xx) and sm_89 (RTX 40xx) for testing
# GTX 1070 = sm_61
# RTX 3080 = sm_86
set(CMAKE_CUDA_ARCHITECTURES "86")

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# =============================================================================
# Find Dependencies
# =============================================================================

find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# =============================================================================
# CUDA Vanity Search Executable
# =============================================================================

set(CUDA_SOURCES
    src/main.cpp
    src/generator.cpp
    src/cuda/cuda_runner.cu
    src/cuda/kernels.cu
    src/cuda/prime_validator.cpp
)

add_executable(${PROJECT_NAME} ${CUDA_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    src
    src/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
    CUDA::cudart
)

# Enable separable compilation for CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# =============================================================================
# Build Type Specific Optimizations
# =============================================================================

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -DNDEBUG>
    )
    target_link_options(${PROJECT_NAME} PRIVATE -O3)
else()
    # Debug build - enable CUDA error checking
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_DEBUG)
endif()

# Warnings
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

# =============================================================================
# Prime Pool Generator Tool (CPU only, no CUDA needed)
# =============================================================================

add_executable(generate-prime-pool tools/generate_prime_pool.cpp)

target_link_libraries(generate-prime-pool PRIVATE
    OpenSSL::Crypto
    Threads::Threads
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(generate-prime-pool PRIVATE -O3 -DNDEBUG -march=native)
endif()

target_compile_options(generate-prime-pool PRIVATE -Wall -Wextra)

# =============================================================================
# GPU Prime Pool Generator (CUDA-accelerated)
# =============================================================================

add_executable(generate-prime-pool-gpu tools/cuda_prime_gen.cu)

target_link_libraries(generate-prime-pool-gpu PRIVATE
    CUDA::cudart
    CUDA::curand
)

set_target_properties(generate-prime-pool-gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(generate-prime-pool-gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -DNDEBUG>
    )
endif()

# =============================================================================
# CGBN GPU Prime Pool Generator (High Performance)
# =============================================================================

add_executable(generate-prime-pool-cgbn tools/cgbn_prime_gen.cu)

target_include_directories(generate-prime-pool-cgbn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/CGBN/include
)

target_link_libraries(generate-prime-pool-cgbn PRIVATE
    CUDA::cudart
    CUDA::curand
    gmp
)

set_target_properties(generate-prime-pool-cgbn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(generate-prime-pool-cgbn PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -DNDEBUG>
    )
endif()

# =============================================================================
# CGBN Tunable Prime Generator (For Grid Search)
# =============================================================================

add_executable(generate-prime-pool-cgbn-tune tools/cgbn_prime_gen_tunable.cu)

target_include_directories(generate-prime-pool-cgbn-tune PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/CGBN/include
)

target_link_libraries(generate-prime-pool-cgbn-tune PRIVATE
    CUDA::cudart
    CUDA::curand
    gmp
)

set_target_properties(generate-prime-pool-cgbn-tune PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(generate-prime-pool-cgbn-tune PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -DNDEBUG>
    )
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(TARGETS generate-prime-pool DESTINATION bin)
install(TARGETS generate-prime-pool-gpu DESTINATION bin)
install(TARGETS generate-prime-pool-cgbn DESTINATION bin)

# =============================================================================
# Print Configuration Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Vanity Extension ID CUDA Build Configuration ===")
message(STATUS "  CUDA Version:     ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Archs:       ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenSSL Version:  ${OPENSSL_VERSION}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "")
