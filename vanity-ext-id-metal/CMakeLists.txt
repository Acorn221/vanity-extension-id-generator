cmake_minimum_required(VERSION 3.14)
project(vanity-ext-id-metal VERSION 1.0.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Find Threads
find_package(Threads REQUIRED)

# =============================================================================
# Metal GPU executable
# =============================================================================

if(APPLE)
    message(STATUS "Apple platform detected - building Metal GPU version")

    # Find Metal and Foundation frameworks
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(SECURITY_FRAMEWORK Security)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        message(STATUS "Metal framework found")

        set(METAL_SOURCES
            src/main.cpp
            src/generator.cpp
            src/metal/metal_runner.mm
        )

        add_executable(${PROJECT_NAME} ${METAL_SOURCES})

        target_include_directories(${PROJECT_NAME} PRIVATE src src/metal)

        target_link_libraries(${PROJECT_NAME} PRIVATE
            OpenSSL::SSL
            OpenSSL::Crypto
            Threads::Threads
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
        )

        if(SECURITY_FRAMEWORK)
            target_link_libraries(${PROJECT_NAME} PRIVATE ${SECURITY_FRAMEWORK})
        endif()

        # Objective-C++ settings
        set_source_files_properties(src/metal/metal_runner.mm PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
        )

        # Copy Metal shaders to build directory
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/src/metal
                ${CMAKE_BINARY_DIR}/metal
            COMMENT "Copying Metal shaders to build directory"
        )

        # Release optimizations for Metal version
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
                target_compile_options(${PROJECT_NAME} PRIVATE
                    -O3
                    -mcpu=apple-m1
                    -mtune=native
                    -flto=auto
                    -ffast-math
                    -funroll-loops
                    -DNDEBUG
                    -DUSE_METAL
                )
            endif()
            target_link_options(${PROJECT_NAME} PRIVATE -flto=auto)
        endif()

        target_compile_options(${PROJECT_NAME} PRIVATE
            -Wall -Wextra
        )

    else()
        message(FATAL_ERROR "Metal framework not found - this is the Metal-only version")
    endif()
else()
    message(FATAL_ERROR "This is the Metal GPU version and requires macOS")
endif()

# =============================================================================
# Prime Pool Generator Tool (with Metal GPU acceleration)
# =============================================================================

if(APPLE AND METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
    # Metal-accelerated prime generator
    set(PRIME_POOL_SOURCES
        tools/generate_prime_pool.cpp
        src/metal/prime_metal_runner.mm
    )
    
    add_executable(generate-prime-pool ${PRIME_POOL_SOURCES})
    
    target_include_directories(generate-prime-pool PRIVATE src src/metal)
    
    target_link_libraries(generate-prime-pool PRIVATE
        OpenSSL::Crypto
        Threads::Threads
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    
    # Objective-C++ settings for Metal runner
    set_source_files_properties(src/metal/prime_metal_runner.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
    
    # Copy Metal shaders for prime generator
    add_custom_command(TARGET generate-prime-pool POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/src/metal
            ${CMAKE_BINARY_DIR}/metal
        COMMENT "Copying Metal shaders for prime generator"
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            target_compile_options(generate-prime-pool PRIVATE
                -O3
                -mcpu=apple-m1
                -mtune=native
                -flto=auto
                -ffast-math
                -funroll-loops
                -DNDEBUG
            )
        else()
            target_compile_options(generate-prime-pool PRIVATE -O3 -DNDEBUG)
        endif()
        target_link_options(generate-prime-pool PRIVATE -flto=auto)
    endif()
    
    target_compile_options(generate-prime-pool PRIVATE -Wall -Wextra)
    
    message(STATUS "Prime Pool Generator: Metal GPU acceleration enabled")
else()
    # CPU-only prime generator (non-Apple platforms)
    add_executable(generate-prime-pool tools/generate_prime_pool.cpp)
    
    target_link_libraries(generate-prime-pool PRIVATE
        OpenSSL::Crypto
        Threads::Threads
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(generate-prime-pool PRIVATE -O3 -DNDEBUG)
    endif()
    
    target_compile_options(generate-prime-pool PRIVATE -Wall -Wextra)
    
    message(STATUS "Prime Pool Generator: CPU-only (no Metal)")
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(TARGETS generate-prime-pool DESTINATION bin)

# =============================================================================
# Ultra-Fast Prime Generator (CPU optimized, minimal M-R rounds)
# =============================================================================

add_executable(generate-prime-pool-fast tools/generate_prime_pool_fast.cpp)

target_link_libraries(generate-prime-pool-fast PRIVATE
    OpenSSL::Crypto
    Threads::Threads
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        target_compile_options(generate-prime-pool-fast PRIVATE
            -O3 -mcpu=native -mtune=native -flto -ffast-math -funroll-loops -DNDEBUG
        )
    else()
        target_compile_options(generate-prime-pool-fast PRIVATE
            -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -DNDEBUG
        )
    endif()
    target_link_options(generate-prime-pool-fast PRIVATE -flto)
endif()

target_compile_options(generate-prime-pool-fast PRIVATE -Wall -Wextra)
install(TARGETS generate-prime-pool-fast DESTINATION bin)

# =============================================================================
# TURBO Prime Generator (trial division only - maximum speed)
# =============================================================================

add_executable(generate-prime-pool-turbo tools/generate_prime_pool_turbo.cpp)

target_link_libraries(generate-prime-pool-turbo PRIVATE Threads::Threads)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        target_compile_options(generate-prime-pool-turbo PRIVATE
            -O3 -mcpu=native -mtune=native -flto -ffast-math -funroll-loops -DNDEBUG
        )
    else()
        target_compile_options(generate-prime-pool-turbo PRIVATE
            -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -DNDEBUG
        )
    endif()
    target_link_options(generate-prime-pool-turbo PRIVATE -flto)
endif()

target_compile_options(generate-prime-pool-turbo PRIVATE -Wall -Wextra)
install(TARGETS generate-prime-pool-turbo DESTINATION bin)