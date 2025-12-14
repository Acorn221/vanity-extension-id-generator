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
# Installation
# =============================================================================

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(TARGETS generate-prime-pool DESTINATION bin)