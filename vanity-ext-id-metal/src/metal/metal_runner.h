/**
 * Metal GPU Runner for Vanity Extension ID Search
 * 
 * C++ interface to Metal compute shaders for searching prime combinations.
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>

namespace vanity {
namespace metal {

// Match result from GPU search
struct GPUMatch {
    uint32_t prime_idx_p;
    uint32_t prime_idx_q;
    std::string extension_id;
    uint32_t match_type;  // 1=prefix, 2=suffix, 3=both
};

// Search configuration
struct SearchConfig {
    std::string target_prefix;
    std::string target_suffix;
    uint32_t batch_size = 1024;  // Grid size per dispatch
};

// Progress callback type
using ProgressCallback = std::function<void(uint64_t pairs_checked, uint64_t total_pairs, uint64_t matches_found)>;

/**
 * Metal-based GPU search engine
 */
class MetalRunner {
public:
    MetalRunner();
    ~MetalRunner();
    
    // Disable copy
    MetalRunner(const MetalRunner&) = delete;
    MetalRunner& operator=(const MetalRunner&) = delete;
    
    /**
     * Check if Metal is available on this system
     */
    bool isAvailable() const;
    
    /**
     * Get GPU device name
     */
    std::string getDeviceName() const;
    
    /**
     * Load prime pool from file
     * @param filepath Path to prime_pool.bin
     * @return Number of primes loaded, or 0 on failure
     */
    size_t loadPrimePool(const std::string& filepath);
    
    /**
     * Get number of loaded primes
     */
    size_t getPrimeCount() const;
    
    /**
     * Get total number of unique prime pairs
     */
    uint64_t getTotalPairs() const;
    
    /**
     * Run search for specific prefix/suffix
     * @param config Search configuration
     * @param callback Progress callback (called periodically)
     * @return Vector of matches found
     */
    std::vector<GPUMatch> search(const SearchConfig& config, ProgressCallback callback = nullptr);
    
    /**
     * Run search with dictionary
     * @param words Vector of words to search for
     * @param min_len Minimum word length
     * @param callback Progress callback
     * @return Vector of matches found
     */
    std::vector<GPUMatch> searchDictionary(
        const std::vector<std::string>& words,
        size_t min_len,
        ProgressCallback callback = nullptr
    );
    
    /**
     * Stop any running search
     */
    void stop();
    
    /**
     * Check if search is running
     */
    bool isRunning() const;

private:
    class Impl;
    Impl* impl_;
};

/**
 * Reconstruct RSA key from prime indices
 * @param prime_pool_path Path to prime pool file
 * @param idx_p Index of first prime
 * @param idx_q Index of second prime
 * @return PEM-encoded private key, or empty string on failure
 */
std::string reconstructKey(const std::string& prime_pool_path, uint32_t idx_p, uint32_t idx_q);

} // namespace metal
} // namespace vanity
