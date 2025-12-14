cmake_minimum_required(VERSION 3.14)
project(vanity-ext-id-cpu VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Find Threads
find_package(Threads REQUIRED)

# =============================================================================
# Main executable (CPU-based)
# =============================================================================

set(SOURCES
    src/main.cpp
    src/generator.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE src)

target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
)

# =============================================================================
# Prime Pool Generator Tool
# =============================================================================

add_executable(generate-prime-pool tools/generate_prime_pool.cpp)

target_link_libraries(generate-prime-pool PRIVATE
    OpenSSL::Crypto
    Threads::Threads
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(generate-prime-pool PRIVATE -O3 -DNDEBUG)
endif()

target_compile_options(generate-prime-pool PRIVATE -Wall -Wextra)

# =============================================================================
# CPU optimizations
# =============================================================================

if(APPLE)
    find_library(SECURITY_FRAMEWORK Security)
    if(SECURITY_FRAMEWORK)
        target_link_libraries(${PROJECT_NAME} PRIVATE ${SECURITY_FRAMEWORK})
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "Optimizing for Apple Silicon")
        target_compile_options(${PROJECT_NAME} PRIVATE
            -O3
            -mcpu=apple-m1
            -mtune=native
            -flto=auto
            -ffast-math
            -funroll-loops
            -fomit-frame-pointer
            -fno-exceptions
            -DNDEBUG
            -DUSE_APPLE_ACCELERATE
        )
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE
            -O3
            -march=native
            -flto
            -ffast-math
            -funroll-loops
        )
    endif()
    target_link_options(${PROJECT_NAME} PRIVATE -flto=auto)
endif()

# Profile-Guided Optimization support
option(ENABLE_PGO_GENERATE "Enable PGO profile generation" OFF)
option(ENABLE_PGO_USE "Enable PGO profile usage" OFF)

if(ENABLE_PGO_GENERATE)
    message(STATUS "PGO: Profile generation enabled")
    target_compile_options(${PROJECT_NAME} PRIVATE -fprofile-generate)
    target_link_options(${PROJECT_NAME} PRIVATE -fprofile-generate)
elseif(ENABLE_PGO_USE)
    message(STATUS "PGO: Using profile data")
    target_compile_options(${PROJECT_NAME} PRIVATE -fprofile-use -fprofile-correction)
    target_link_options(${PROJECT_NAME} PRIVATE -fprofile-use)
endif()

# Warnings for main executable
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# =============================================================================
# Installation
# =============================================================================

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(TARGETS generate-prime-pool DESTINATION bin)