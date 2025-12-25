/**
 * CUDA GPU Runner for Vanity Extension ID Search
 * 
 * Multi-GPU interface for searching prime combinations on 8x A100 GPUs.
 * Handles memory management, work distribution, and result collection.
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <atomic>

namespace vanity {
namespace cuda {

// Match result from GPU search
struct GPUMatch {
    uint32_t prime_idx_p;
    uint32_t prime_idx_q;
    std::string extension_id;
    uint32_t match_type;  // 1=prefix, 2=suffix, 3=both, 4=pattern
};

// Search configuration
struct SearchConfig {
    std::string target_prefix;
    std::string target_suffix;
    uint32_t batch_size = 2048;  // Grid size per dispatch (tuned for A100)
};

// Progress callback type
using ProgressCallback = std::function<void(uint64_t pairs_checked, uint64_t total_pairs, uint64_t matches_found)>;

// GPU device info
struct GPUInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int sm_count;
    int compute_capability_major;
    int compute_capability_minor;
};

/**
 * Multi-GPU CUDA search engine
 */
class CudaRunner {
public:
    CudaRunner();
    ~CudaRunner();
    
    // Disable copy
    CudaRunner(const CudaRunner&) = delete;
    CudaRunner& operator=(const CudaRunner&) = delete;
    
    /**
     * Check if CUDA is available on this system
     */
    bool isAvailable() const;
    
    /**
     * Get number of available GPUs
     */
    int getDeviceCount() const;
    
    /**
     * Get info for all available GPUs
     */
    std::vector<GPUInfo> getDeviceInfo() const;
    
    /**
     * Get combined GPU name string (e.g., "8x NVIDIA A100-SXM4-80GB")
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
     * Get raw pointer to prime pool (for validation)
     */
    const uint8_t* getPrimePool() const;
    
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
     * Fast "AI" search - finds extension IDs with N+ occurrences of "ai"
     * This is GPU-only with no CPU post-processing needed.
     * 
     * @param min_ai_count Minimum number of "ai" occurrences to save (e.g., 7)
     * @param callback Progress callback
     * @return Vector of matches found (match_type contains the ai count)
     */
    std::vector<GPUMatch> searchAI(
        uint32_t min_ai_count,
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
 * Validate that a prime from the pool is actually prime
 * Uses Miller-Rabin primality test with OpenSSL
 * @param prime_bytes Pointer to 128 bytes (1024-bit prime in big-endian)
 * @return true if prime passes primality test
 */
bool validatePrime(const uint8_t* prime_bytes);

/**
 * Validate both primes for a match
 * @param prime_pool Pointer to start of prime pool data
 * @param idx_p Index of first prime
 * @param idx_q Index of second prime
 * @return true if both primes pass primality test
 */
bool validatePrimePair(const uint8_t* prime_pool, uint32_t idx_p, uint32_t idx_q);

/**
 * Batch validate multiple matches
 * @param prime_pool Pointer to start of prime pool data
 * @param matches Vector of matches to validate
 * @return Vector of valid matches
 */
std::vector<GPUMatch> validateMatches(const uint8_t* prime_pool, const std::vector<GPUMatch>& matches);

/**
 * Get validation statistics
 * @param validations Total number of validations performed
 * @param cache_hits Number of cache hits
 * @param invalid_found Number of invalid primes found
 */
void getValidationStats(uint64_t& validations, uint64_t& cache_hits, uint64_t& invalid_found);

/**
 * Clear validation cache (call when loading a new prime pool)
 */
void clearValidationCache();

/**
 * Pre-validate a range of primes (useful for warming the cache)
 * @param prime_pool Pointer to start of prime pool data
 * @param start_idx Starting index
 * @param count Number of primes to validate
 * @param progress_callback Optional callback for progress updates
 * @return Number of valid primes found
 */
size_t prevalidatePrimes(
    const uint8_t* prime_pool,
    uint32_t start_idx,
    uint32_t count,
    std::function<void(uint32_t checked, uint32_t total, uint32_t invalid)> progress_callback = nullptr
);

/**
 * Reconstruct RSA key from prime indices
 * @param prime_pool_path Path to prime pool file
 * @param idx_p Index of first prime
 * @param idx_q Index of second prime
 * @return PEM-encoded private key, or empty string on failure
 */
std::string reconstructKey(const std::string& prime_pool_path, uint32_t idx_p, uint32_t idx_q);

} // namespace cuda
} // namespace vanity

