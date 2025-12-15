/**
 * Prime Validator for Vanity Extension ID Search
 * 
 * Uses OpenSSL's Miller-Rabin primality test to validate that primes
 * from the pool are actually prime. This is needed because ~8% of the
 * primes in the 100M pool may be invalid (false positives from generation).
 * 
 * Validation is performed on CPU after GPU finds matches, since:
 * 1. Only a small fraction of pairs are matches (need validation)
 * 2. Miller-Rabin is complex to implement efficiently on GPU
 * 3. OpenSSL's implementation is highly optimized
 */

#include "cuda_runner.h"

#include <openssl/bn.h>
#include <openssl/err.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace vanity {
namespace cuda {

// =============================================================================
// Thread-safe validation cache
// =============================================================================

// Cache validation results to avoid re-validating the same prime
// Key: prime index, Value: is_prime result
static std::unordered_map<uint32_t, bool> g_validation_cache;
static std::mutex g_cache_mutex;

// Statistics
static std::atomic<uint64_t> g_validations_performed{0};
static std::atomic<uint64_t> g_cache_hits{0};
static std::atomic<uint64_t> g_invalid_primes_found{0};

/**
 * Validate a single prime using Miller-Rabin
 * 
 * @param prime_bytes Pointer to 128 bytes (1024-bit prime in big-endian)
 * @return true if the value passes the primality test
 */
bool validatePrime(const uint8_t* prime_bytes) {
    if (!prime_bytes) return false;
    
    // Convert bytes to BIGNUM
    BIGNUM* n = BN_bin2bn(prime_bytes, 128, nullptr);
    if (!n) return false;
    
    // Create context for primality test
    BN_CTX* ctx = BN_CTX_new();
    if (!ctx) {
        BN_free(n);
        return false;
    }
    
    // Perform Miller-Rabin primality test
    // 64 rounds provides extremely high confidence (2^-128 false positive rate)
    // For 1024-bit primes, this is more than sufficient
    int is_prime = BN_check_prime(n, ctx, nullptr);
    
    BN_CTX_free(ctx);
    BN_free(n);
    
    if (is_prime != 1) {
        g_invalid_primes_found.fetch_add(1, std::memory_order_relaxed);
    }
    
    g_validations_performed.fetch_add(1, std::memory_order_relaxed);
    
    return is_prime == 1;
}

/**
 * Validate a prime with caching
 * 
 * @param prime_pool Pointer to start of prime pool data
 * @param idx Index of the prime in the pool
 * @return true if the prime passes the primality test
 */
bool validatePrimeCached(const uint8_t* prime_pool, uint32_t idx) {
    // Check cache first
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_validation_cache.find(idx);
        if (it != g_validation_cache.end()) {
            g_cache_hits.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
    }
    
    // Not in cache, perform validation
    const uint8_t* prime_bytes = prime_pool + idx * 128;
    bool is_valid = validatePrime(prime_bytes);
    
    // Store in cache
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        g_validation_cache[idx] = is_valid;
    }
    
    return is_valid;
}

/**
 * Validate both primes for a match
 * 
 * @param prime_pool Pointer to start of prime pool data
 * @param idx_p Index of first prime
 * @param idx_q Index of second prime
 * @return true if both primes pass primality test
 */
bool validatePrimePair(const uint8_t* prime_pool, uint32_t idx_p, uint32_t idx_q) {
    // Validate first prime
    if (!validatePrimeCached(prime_pool, idx_p)) {
        return false;
    }
    
    // Validate second prime
    if (!validatePrimeCached(prime_pool, idx_q)) {
        return false;
    }
    
    return true;
}

/**
 * Batch validate multiple matches
 * Returns vector of indices into matches that are valid
 * 
 * @param prime_pool Pointer to start of prime pool data
 * @param matches Vector of matches to validate
 * @return Vector of valid matches
 */
std::vector<GPUMatch> validateMatches(const uint8_t* prime_pool, const std::vector<GPUMatch>& matches) {
    std::vector<GPUMatch> valid_matches;
    valid_matches.reserve(matches.size());
    
    for (const auto& match : matches) {
        if (validatePrimePair(prime_pool, match.prime_idx_p, match.prime_idx_q)) {
            valid_matches.push_back(match);
        }
    }
    
    return valid_matches;
}

/**
 * Get validation statistics
 */
void getValidationStats(uint64_t& validations, uint64_t& cache_hits, uint64_t& invalid_found) {
    validations = g_validations_performed.load(std::memory_order_relaxed);
    cache_hits = g_cache_hits.load(std::memory_order_relaxed);
    invalid_found = g_invalid_primes_found.load(std::memory_order_relaxed);
}

/**
 * Clear validation cache (e.g., when loading a new prime pool)
 */
void clearValidationCache() {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    g_validation_cache.clear();
    g_validations_performed = 0;
    g_cache_hits = 0;
    g_invalid_primes_found = 0;
}

/**
 * Pre-validate a range of primes (useful for warming the cache)
 * 
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
    std::function<void(uint32_t checked, uint32_t total, uint32_t invalid)> progress_callback
) {
    size_t valid_count = 0;
    uint32_t invalid_count = 0;
    
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = start_idx + i;
        
        if (validatePrimeCached(prime_pool, idx)) {
            valid_count++;
        } else {
            invalid_count++;
        }
        
        // Progress callback every 1000 primes
        if (progress_callback && (i + 1) % 1000 == 0) {
            progress_callback(i + 1, count, invalid_count);
        }
    }
    
    if (progress_callback) {
        progress_callback(count, count, invalid_count);
    }
    
    return valid_count;
}

} // namespace cuda
} // namespace vanity

