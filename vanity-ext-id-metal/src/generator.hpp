#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <vector>
#include <array>
#include <unordered_set>
#include <fstream>
#include <openssl/evp.h>
#include <openssl/bn.h>

namespace vanity {

// ============================================================================
// OPTIMIZATION: Small prime sieve for fast candidate filtering
// ============================================================================

// First 256 primes for sieving (filters out ~77% of odd candidates)
extern const uint16_t SMALL_PRIMES[256];
extern const size_t NUM_SMALL_PRIMES;

/**
 * Pre-computed sieve data for fast prime candidate filtering
 * Checking divisibility by small primes is much faster than Miller-Rabin
 */
struct PrimeSieve {
    // Remainders for incremental sieving
    std::vector<uint32_t> remainders;
    
    // Initialize sieve for a starting number
    void init(const BIGNUM* start, BN_CTX* ctx);
    
    // Check if current candidate passes small prime test
    // Returns false if definitely composite, true if might be prime
    bool check_candidate() const;
    
    // Advance to next odd candidate (increment by 2)
    void advance();
};

/**
 * Thread-local context for optimized key generation
 * Reuses expensive resources across iterations
 */
struct ThreadContext {
    EVP_PKEY_CTX* keygen_ctx = nullptr;
    BN_CTX* bn_ctx = nullptr;
    PrimeSieve sieve_p;
    PrimeSieve sieve_q;
    
    // Scratch BIGNUMs for prime generation
    BIGNUM* candidate = nullptr;
    BIGNUM* tmp = nullptr;
    
    ThreadContext();
    ~ThreadContext();
    
    // Non-copyable
    ThreadContext(const ThreadContext&) = delete;
    ThreadContext& operator=(const ThreadContext&) = delete;
};

/**
 * Generate RSA key with optimized prime generation
 * Uses small prime sieving to filter candidates before Miller-Rabin
 */
EVP_PKEY* generate_rsa_key_fast(ThreadContext& ctx);

/**
 * Compute extension ID using Apple Accelerate SHA-256 (if available)
 * Falls back to OpenSSL on non-Apple platforms
 */
bool compute_extension_id_fast(EVP_PKEY* pkey, char* ext_id);

// Match position for target string
enum class MatchPosition {
    Start,
    End,
    Anywhere,
    StartAndEnd  // Match both start and end with different strings
};

// Search configuration
struct SearchConfig {
    std::string target;           // Primary target (or start target for StartAndEnd)
    std::string end_target;       // End target (only used for StartAndEnd mode)
    MatchPosition position = MatchPosition::Anywhere;
};

// Dictionary for multi-word matching
struct Dictionary {
    std::unordered_set<std::string> words;              // All valid words
    std::vector<std::string> prefixes;                   // Words to check at start (sorted by length desc)
    std::vector<std::string> suffixes;                   // Words to check at end (sorted by length desc)
    size_t min_length = 3;                               // Minimum word length to match
    size_t max_length = 32;                              // Maximum word length
    bool check_anywhere = true;                          // Check for words anywhere in ID
    bool check_start = true;                             // Check for words at start
    bool check_end = true;                               // Check for words at end
    
    // Per-length limits (0 = unlimited). Index = word length
    // Default: 100 for 3-char, 500 for 4-char, unlimited for 5+
    std::array<uint64_t, 33> length_limits{};
    
    bool empty() const { return words.empty(); }
    size_t size() const { return words.size(); }
};

// Result structure to hold found key data
struct Result {
    std::string extension_id;
    std::string private_key_pem;
    std::string public_key_base64;
    std::string matched_word;
    std::string match_position;  // "start", "end", "anywhere", or "start+end"
    bool found = false;
};

// Compact key storage - just the primes (256 bytes vs 1700 bytes)
struct CompactKey {
    std::string p_hex;  // Prime p as hex string (~128 bytes)
    std::string q_hex;  // Prime q as hex string (~128 bytes)
};

// Match result for dictionary mode (can have multiple matches)
struct DictMatch {
    std::string extension_id;
    CompactKey compact_key;                                    // Just p and q
    std::string public_key_base64;                             // Still useful for manifest.json
    std::vector<std::pair<std::string, std::string>> matches;  // word -> position
};

// Shared state between main thread and workers
struct SharedState {
    std::atomic<bool> found{false};
    std::atomic<bool> stop{false};                       // For dictionary mode (don't stop on first match)
    std::atomic<uint64_t> attempts{0};
    std::atomic<uint64_t> matches_found{0};              // Count of matches in dictionary mode
    Result result;
    std::mutex result_mutex;
    
    // Dictionary mode results
    std::vector<DictMatch> dict_matches;
    std::mutex dict_mutex;
    
    // Per-length match counters (index = word length, e.g., [3] = count of 3-letter matches)
    std::array<std::atomic<uint64_t>, 33> length_counts{};
};

/**
 * Validate that target string only contains valid chars (a-p)
 */
bool validate_target(const std::string& target);

/**
 * Parse match position from string
 */
MatchPosition parse_position(const std::string& pos_str);

/**
 * Convert match position to string for display
 */
const char* position_to_string(MatchPosition pos);

/**
 * Worker thread function - generates keys and checks for matches
 * 
 * @param config Search configuration with target(s) and position
 * @param state Shared state for coordination
 */
void worker_loop(const SearchConfig& config, SharedState& state);

/**
 * OPTIMIZED worker loop using fast prime generation and Apple Accelerate
 */
void worker_loop_fast(const SearchConfig& config, SharedState& state);

/**
 * Compute extension ID from an EVP_PKEY
 * 
 * @param pkey The RSA key pair
 * @param ext_id Output buffer (must be at least 33 bytes)
 * @return true on success
 */
bool compute_extension_id(EVP_PKEY* pkey, char* ext_id);

/**
 * Check if extension ID matches the search configuration
 */
bool check_match(const char* ext_id, const SearchConfig& config);

/**
 * Export private key to PEM format
 */
std::string export_private_key_pem(EVP_PKEY* pkey);

/**
 * Export public key to base64 (for manifest.json)
 */
std::string export_public_key_base64(EVP_PKEY* pkey);

/**
 * Calculate expected attempts for given search configuration
 */
uint64_t estimate_attempts(const SearchConfig& config);

/**
 * Load dictionary from file, filtering to only valid a-p words
 */
Dictionary load_dictionary(const std::string& filepath, size_t min_len = 3, size_t max_len = 10);

/**
 * Check extension ID against dictionary, return all matches
 */
std::vector<std::pair<std::string, std::string>> check_dictionary(
    const char* ext_id, const Dictionary& dict);

/**
 * Worker loop for dictionary mode - finds ALL matches, doesn't stop
 */
void worker_loop_dict(const Dictionary& dict, SharedState& state, 
                      const std::string& output_file, std::mutex& file_mutex);

/**
 * OPTIMIZED dictionary worker loop using fast prime generation
 */
void worker_loop_dict_fast(const Dictionary& dict, SharedState& state,
                           const std::string& output_file, std::mutex& file_mutex);

/**
 * Extract just the primes p and q from an RSA key (compact storage)
 */
CompactKey extract_compact_key(EVP_PKEY* pkey);

/**
 * Reconstruct full RSA key from compact representation (p and q)
 */
EVP_PKEY* reconstruct_key(const CompactKey& compact);

/**
 * Reconstruct and export private key PEM from compact key
 */
std::string compact_to_pem(const CompactKey& compact);

/**
 * State file for resuming dictionary mode
 */
struct StateFile {
    uint64_t total_attempts = 0;
    uint64_t total_matches = 0;
    std::array<uint64_t, 33> length_counts{};
    
    bool load(const std::string& filepath);
    bool save(const std::string& filepath) const;
};

/**
 * Check for cool patterns in extension ID
 * Returns pattern description if found, empty string if not cool enough
 * 
 * Detects:
 * - Repeated chars (4+): "aaaa" → "repeat:a*4"
 * - Sequences (4+): "abcd" → "seq:abcd"
 * - Palindromes (6+): "abccba" → "palindrome:6"
 * - Repeated patterns: "abcabc" → "pattern:abc*2"
 */
std::string detect_cool_pattern(const char* ext_id);

} // namespace vanity

