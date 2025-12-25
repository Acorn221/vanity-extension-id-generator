/**
 * Metal GPU Prime Generator Runner
 * 
 * Manages Metal compute pipeline for generating 1024-bit primes using GPU.
 */

#pragma once

#include <string>
#include <functional>
#include <cstdint>

namespace vanity {
namespace metal {

// Progress callback: (primes_found, primes_target, primes_per_sec)
using PrimeProgressCallback = std::function<void(size_t, size_t, double)>;

/**
 * Metal-accelerated prime generator
 */
class PrimeMetalRunner {
public:
    PrimeMetalRunner();
    ~PrimeMetalRunner();
    
    /**
     * Check if Metal GPU is available for prime generation
     */
    bool isAvailable() const;
    
    /**
     * Get the name of the Metal GPU device
     */
    std::string getDeviceName() const;
    
    /**
     * Generate primes and write to file
     * 
     * @param target_count Number of primes to generate
     * @param output_file Path to output file
     * @param callback Progress callback (optional)
     * @return Number of primes actually generated
     */
    size_t generatePrimes(
        size_t target_count,
        const std::string& output_file,
        PrimeProgressCallback callback = nullptr
    );
    
    /**
     * Generate primes and return them in a buffer
     * 
     * @param target_count Number of primes to generate
     * @param output_buffer Pre-allocated buffer (128 bytes per prime)
     * @param callback Progress callback (optional)
     * @return Number of primes actually generated
     */
    size_t generatePrimesToBuffer(
        size_t target_count,
        uint8_t* output_buffer,
        PrimeProgressCallback callback = nullptr
    );
    
    /**
     * Stop ongoing prime generation
     */
    void stop();
    
    /**
     * Check if generation is currently running
     */
    bool isRunning() const;
    
    /**
     * Get estimated primes per second based on device
     */
    double getEstimatedRate() const;

private:
    class Impl;
    Impl* impl_;
};

} // namespace metal
} // namespace vanity

