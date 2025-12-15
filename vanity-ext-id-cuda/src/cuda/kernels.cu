/**
 * CUDA Kernels for Vanity Extension ID Search
 * 
 * Converted from Metal shaders - implements:
 * 1. Big integer multiplication (1024-bit × 1024-bit = 2048-bit)
 * 2. SHA-256 hashing
 * 3. DER encoding of RSA public keys
 * 4. Pattern matching for vanity IDs
 * 
 * Optimized for NVIDIA A100 GPUs (sm_80):
 * - Uses __ldg() intrinsic for read-only data (L2 cache optimization)
 * - Loop unrolling with #pragma unroll
 * - Register pressure optimization with __launch_bounds__
 * - Constant memory for lookup tables
 * - Coalesced memory access patterns
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// A100 has 108 SMs with 2048 threads max per SM
// Optimal occupancy with 256 threads/block = 8 blocks/SM
#define BLOCK_SIZE 256
#define MAX_BLOCKS_PER_SM 8

namespace vanity {
namespace cuda {

// =============================================================================
// Constants
// =============================================================================

constexpr uint32_t PRIME_WORDS = 32;    // 1024 bits = 32 × 32-bit words
constexpr uint32_t MODULUS_WORDS = 64;  // 2048 bits = 64 × 32-bit words
constexpr uint32_t PRIME_BYTES = 128;   // 1024 bits = 128 bytes
constexpr uint32_t MODULUS_BYTES = 256; // 2048 bits = 256 bytes

// DER encoding header for RSA-2048 public key (SubjectPublicKeyInfo)
__constant__ uint8_t DER_HEADER[33] = {
    0x30, 0x82, 0x01, 0x22,  // SEQUENCE, length 290
    0x30, 0x0d,              // SEQUENCE, length 13
    0x06, 0x09,              // OID, length 9
    0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01,  // rsaEncryption OID
    0x05, 0x00,              // NULL
    0x03, 0x82, 0x01, 0x0f,  // BIT STRING, length 271
    0x00,                    // unused bits = 0
    0x30, 0x82, 0x01, 0x0a,  // SEQUENCE, length 266
    0x02, 0x82, 0x01, 0x01,  // INTEGER (n), length 257 (with leading 0x00)
    0x00                     // Leading zero for positive integer
};

__constant__ uint8_t DER_FOOTER[5] = {
    0x02, 0x03,              // INTEGER, length 3
    0x01, 0x00, 0x01         // 65537 = 0x010001
};

constexpr uint32_t DER_HEADER_LEN = 33;
constexpr uint32_t DER_FOOTER_LEN = 5;
constexpr uint32_t DER_TOTAL_LEN = 294;

// SHA-256 round constants
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// =============================================================================
// Match Result Structure (must match host definition)
// =============================================================================

struct MatchResult {
    uint32_t prime_idx_p;    // Index of prime p in pool
    uint32_t prime_idx_q;    // Index of prime q in pool
    char ext_id[36];         // Extension ID (32 chars + padding)
    uint32_t match_type;     // 1=prefix, 2=suffix, 3=both, 4=pattern
};

// =============================================================================
// SHA-256 Helper Functions
// =============================================================================

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// =============================================================================
// Big Integer Multiplication (1024 × 1024 = 2048 bits)
// =============================================================================

__device__ void bigint_mul_1024(
    const uint32_t* __restrict__ p,
    const uint32_t* __restrict__ q,
    uint32_t* __restrict__ n
) {
    // Zero initialize result
    #pragma unroll
    for (uint32_t i = 0; i < MODULUS_WORDS; i++) {
        n[i] = 0;
    }
    
    // Schoolbook multiplication
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        uint64_t carry = 0;
        
        #pragma unroll 8
        for (uint32_t j = 0; j < PRIME_WORDS; j++) {
            uint32_t k = i + j;
            uint64_t product = (uint64_t)p[i] * (uint64_t)q[j];
            uint64_t sum = (uint64_t)n[k] + product + carry;
            n[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
        
        // Propagate carry
        for (uint32_t k = i + PRIME_WORDS; carry && k < MODULUS_WORDS; k++) {
            uint64_t sum = (uint64_t)n[k] + carry;
            n[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
}

// =============================================================================
// DER Encoding
// =============================================================================

__device__ void build_der_pubkey(
    const uint32_t* __restrict__ n_words,
    uint8_t* __restrict__ der_out
) {
    // Copy header from constant memory
    #pragma unroll
    for (uint32_t i = 0; i < DER_HEADER_LEN; i++) {
        der_out[i] = DER_HEADER[i];
    }
    
    // Copy modulus n (convert from little-endian words to big-endian bytes)
    for (uint32_t i = 0; i < MODULUS_WORDS; i++) {
        uint32_t word_idx = MODULUS_WORDS - 1 - i;
        uint32_t byte_offset = DER_HEADER_LEN + i * 4;
        uint32_t word = n_words[word_idx];
        der_out[byte_offset] = (word >> 24) & 0xFF;
        der_out[byte_offset + 1] = (word >> 16) & 0xFF;
        der_out[byte_offset + 2] = (word >> 8) & 0xFF;
        der_out[byte_offset + 3] = word & 0xFF;
    }
    
    // Copy footer
    uint32_t footer_offset = DER_HEADER_LEN + MODULUS_BYTES;
    #pragma unroll
    for (uint32_t i = 0; i < DER_FOOTER_LEN; i++) {
        der_out[footer_offset + i] = DER_FOOTER[i];
    }
}

// =============================================================================
// SHA-256 Hash (optimized for ~294 byte input)
// =============================================================================

__device__ void sha256_der(const uint8_t* __restrict__ data, uint32_t len, uint8_t* __restrict__ hash_out) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Process all complete 64-byte blocks
    uint32_t num_blocks = (len + 9 + 63) / 64;  // Include padding
    
    for (uint32_t block = 0; block < num_blocks; block++) {
        uint32_t w[64];
        uint32_t block_start = block * 64;
        
        // Load or pad this block
        for (uint32_t i = 0; i < 16; i++) {
            uint32_t byte_idx = block_start + i * 4;
            uint32_t word = 0;
            
            for (uint32_t b = 0; b < 4; b++) {
                uint32_t idx = byte_idx + b;
                uint8_t byte_val;
                
                if (idx < len) {
                    byte_val = data[idx];
                } else if (idx == len) {
                    byte_val = 0x80;  // Padding bit
                } else if (block == num_blocks - 1 && i >= 14) {
                    // Length in last 8 bytes
                    uint64_t bit_len = (uint64_t)len * 8;
                    if (i == 14) {
                        byte_val = (b < 4) ? ((bit_len >> (56 - b * 8)) & 0xFF) : 0;
                    } else {
                        byte_val = (bit_len >> (24 - b * 8)) & 0xFF;
                    }
                } else {
                    byte_val = 0;
                }
                
                word = (word << 8) | byte_val;
            }
            w[i] = word;
        }
        
        // Extend message schedule
        #pragma unroll 8
        for (uint32_t i = 16; i < 64; i++) {
            w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
        }
        
        // Compression
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        
        #pragma unroll 8
        for (uint32_t i = 0; i < 64; i++) {
            uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = sigma0(a) + maj(a, b, c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    
    // Output hash
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        hash_out[i*4] = (h[i] >> 24) & 0xFF;
        hash_out[i*4+1] = (h[i] >> 16) & 0xFF;
        hash_out[i*4+2] = (h[i] >> 8) & 0xFF;
        hash_out[i*4+3] = h[i] & 0xFF;
    }
}

// =============================================================================
// Pattern Matching
// =============================================================================

__device__ bool check_prefix(const char* ext_id, const char* target, uint32_t target_len) {
    for (uint32_t i = 0; i < target_len; i++) {
        if (ext_id[i] != target[i]) return false;
    }
    return true;
}

__device__ bool check_suffix(const char* ext_id, const char* target, uint32_t target_len) {
    uint32_t start = 32 - target_len;
    for (uint32_t i = 0; i < target_len; i++) {
        if (ext_id[start + i] != target[i]) return false;
    }
    return true;
}

// =============================================================================
// Main Search Kernel
// =============================================================================

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, MAX_BLOCKS_PER_SM) vanity_search_kernel(
    const uint8_t* __restrict__ prime_pool,      // All primes (128 bytes each)
    const char* __restrict__ target_prefix,       // Target prefix to match
    const char* __restrict__ target_suffix,       // Target suffix to match
    const uint32_t* __restrict__ params,          // [pool_size, prefix_len, suffix_len, start_idx]
    uint32_t* __restrict__ match_count,           // Number of matches found (atomic)
    MatchResult* __restrict__ matches,            // Output buffer for matches
    uint32_t max_matches                          // Maximum matches to store
) {
    uint32_t pool_size = params[0];
    uint32_t prefix_len = params[1];
    uint32_t suffix_len = params[2];
    uint32_t start_idx = params[3];
    
    // Calculate prime pair indices from 2D grid
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y + i + 1;  // j > i always
    
    if (i >= pool_size || j >= pool_size) return;
    
    // Load primes from pool (big-endian bytes) using __ldg for L2 cache
    const uint8_t* p_bytes = prime_pool + i * PRIME_BYTES;
    const uint8_t* q_bytes = prime_pool + j * PRIME_BYTES;
    
    // Convert to little-endian words for multiplication
    uint32_t p_words[PRIME_WORDS];
    uint32_t q_words[PRIME_WORDS];
    
    #pragma unroll 8
    for (uint32_t w = 0; w < PRIME_WORDS; w++) {
        uint32_t byte_idx = (PRIME_WORDS - 1 - w) * 4;
        p_words[w] = (__ldg(&p_bytes[byte_idx]) << 24) |
                     (__ldg(&p_bytes[byte_idx + 1]) << 16) |
                     (__ldg(&p_bytes[byte_idx + 2]) << 8) |
                     __ldg(&p_bytes[byte_idx + 3]);
        q_words[w] = (__ldg(&q_bytes[byte_idx]) << 24) |
                     (__ldg(&q_bytes[byte_idx + 1]) << 16) |
                     (__ldg(&q_bytes[byte_idx + 2]) << 8) |
                     __ldg(&q_bytes[byte_idx + 3]);
    }
    
    // Multiply: n = p × q
    uint32_t n_words[MODULUS_WORDS];
    bigint_mul_1024(p_words, q_words, n_words);
    
    // Build DER-encoded public key
    uint8_t der_pubkey[DER_TOTAL_LEN];
    build_der_pubkey(n_words, der_pubkey);
    
    // Compute SHA-256 hash
    uint8_t hash[32];
    sha256_der(der_pubkey, DER_TOTAL_LEN, hash);
    
    // Convert to extension ID
    char ext_id[33];
    #pragma unroll
    for (uint32_t k = 0; k < 16; k++) {
        ext_id[k*2] = 'a' + (hash[k] >> 4);
        ext_id[k*2+1] = 'a' + (hash[k] & 0x0F);
    }
    ext_id[32] = '\0';
    
    // Check for matches
    bool prefix_match = (prefix_len > 0) && check_prefix(ext_id, target_prefix, prefix_len);
    bool suffix_match = (suffix_len > 0) && check_suffix(ext_id, target_suffix, suffix_len);
    
    if (prefix_match || suffix_match) {
        // Found a match! Record it atomically
        uint32_t match_idx = atomicAdd(match_count, 1);
        
        if (match_idx < max_matches) {
            matches[match_idx].prime_idx_p = i;
            matches[match_idx].prime_idx_q = j;
            #pragma unroll
            for (uint32_t k = 0; k < 32; k++) {
                matches[match_idx].ext_id[k] = ext_id[k];
            }
            matches[match_idx].ext_id[32] = '\0';
            matches[match_idx].match_type = (prefix_match ? 1 : 0) | (suffix_match ? 2 : 0);
        }
    }
}

// =============================================================================
// Dictionary Search Kernel (checks against word list + patterns)
// =============================================================================

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, MAX_BLOCKS_PER_SM) vanity_search_dict_kernel(
    const uint8_t* __restrict__ prime_pool,
    const char* __restrict__ dictionary,          // Packed words, null-separated
    const uint32_t* __restrict__ word_offsets,    // Offset of each word in dictionary
    const uint32_t* __restrict__ params,          // [pool_size, num_words, min_len, start_idx]
    uint32_t* __restrict__ match_count,
    MatchResult* __restrict__ matches,
    uint32_t max_matches
) {
    uint32_t pool_size = params[0];
    uint32_t num_words = params[1];
    uint32_t min_len = params[2];
    uint32_t start_idx = params[3];
    
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y + i + 1;
    
    if (i >= pool_size || j >= pool_size) return;
    
    // Load and multiply primes (same as above)
    const uint8_t* p_bytes = prime_pool + i * PRIME_BYTES;
    const uint8_t* q_bytes = prime_pool + j * PRIME_BYTES;
    
    uint32_t p_words[PRIME_WORDS];
    uint32_t q_words[PRIME_WORDS];
    
    #pragma unroll 8
    for (uint32_t w = 0; w < PRIME_WORDS; w++) {
        uint32_t byte_idx = (PRIME_WORDS - 1 - w) * 4;
        p_words[w] = (__ldg(&p_bytes[byte_idx]) << 24) |
                     (__ldg(&p_bytes[byte_idx + 1]) << 16) |
                     (__ldg(&p_bytes[byte_idx + 2]) << 8) |
                     __ldg(&p_bytes[byte_idx + 3]);
        q_words[w] = (__ldg(&q_bytes[byte_idx]) << 24) |
                     (__ldg(&q_bytes[byte_idx + 1]) << 16) |
                     (__ldg(&q_bytes[byte_idx + 2]) << 8) |
                     __ldg(&q_bytes[byte_idx + 3]);
    }
    
    uint32_t n_words[MODULUS_WORDS];
    bigint_mul_1024(p_words, q_words, n_words);
    
    uint8_t der_pubkey[DER_TOTAL_LEN];
    build_der_pubkey(n_words, der_pubkey);
    
    uint8_t hash[32];
    sha256_der(der_pubkey, DER_TOTAL_LEN, hash);
    
    char ext_id[33];
    #pragma unroll
    for (uint32_t k = 0; k < 16; k++) {
        ext_id[k*2] = 'a' + (hash[k] >> 4);
        ext_id[k*2+1] = 'a' + (hash[k] & 0x0F);
    }
    ext_id[32] = '\0';

    // =========================================================================
    // Check for cool patterns FIRST (before dictionary)
    // =========================================================================
    bool has_pattern = false;

    // 1. Check for repeated characters (6+ in a row)
    for (uint32_t pos = 0; pos < 27 && !has_pattern; pos++) {
        char c = ext_id[pos];
        uint32_t count = 1;
        while (pos + count < 32 && ext_id[pos + count] == c) count++;
        if (count >= 6) {
            has_pattern = true;
        }
    }

    // 2. Check for same-char prefix (5+)
    if (!has_pattern) {
        char first = ext_id[0];
        uint32_t prefix_count = 1;
        while (prefix_count < 32 && ext_id[prefix_count] == first) prefix_count++;
        if (prefix_count >= 5) {
            has_pattern = true;
        }
    }

    // 3. Check for same-char suffix (5+)
    if (!has_pattern) {
        char last = ext_id[31];
        uint32_t suffix_count = 1;
        while (suffix_count < 32 && ext_id[31 - suffix_count] == last) suffix_count++;
        if (suffix_count >= 5) {
            has_pattern = true;
        }
    }

    // 4. Check for ascending sequence from 'a' (5+ chars like "abcde")
    if (!has_pattern) {
        for (uint32_t pos = 0; pos < 28 && !has_pattern; pos++) {
            if (ext_id[pos] == 'a') {
                uint32_t len = 1;
                while (pos + len < 32 && ext_id[pos + len] == ext_id[pos + len - 1] + 1) len++;
                if (len >= 5) {
                    has_pattern = true;
                }
            }
        }
    }

    // 5. Check for palindrome at start (8+ chars)
    if (!has_pattern) {
        for (uint32_t len = 12; len >= 8 && !has_pattern; len--) {
            bool is_palindrome = true;
            for (uint32_t p = 0; p < len / 2 && is_palindrome; p++) {
                if (ext_id[p] != ext_id[len - 1 - p]) is_palindrome = false;
            }
            if (is_palindrome) {
                has_pattern = true;
            }
        }
    }

    // If we found a pattern, record it
    if (has_pattern) {
        uint32_t match_idx = atomicAdd(match_count, 1);
        if (match_idx < max_matches) {
            matches[match_idx].prime_idx_p = i;
            matches[match_idx].prime_idx_q = j;
            #pragma unroll
            for (uint32_t k = 0; k < 32; k++) {
                matches[match_idx].ext_id[k] = ext_id[k];
            }
            matches[match_idx].ext_id[32] = '\0';
            matches[match_idx].match_type = 4;  // 4 = pattern match
        }
        return;  // Found a pattern, done with this ID
    }

    // =========================================================================
    // Check against dictionary (prefix and suffix)
    // =========================================================================
    for (uint32_t w = 0; w < num_words; w++) {
        uint32_t word_start = __ldg(&word_offsets[w]);
        uint32_t word_end = __ldg(&word_offsets[w + 1]);
        uint32_t word_len = word_end - word_start - 1;  // -1 for null terminator

        if (word_len < min_len) continue;

        // Check prefix
        bool prefix_match = true;
        for (uint32_t c = 0; c < word_len && prefix_match; c++) {
            if (ext_id[c] != __ldg(&dictionary[word_start + c])) {
                prefix_match = false;
            }
        }

        // Check suffix
        bool suffix_match = true;
        uint32_t suffix_start = 32 - word_len;
        for (uint32_t c = 0; c < word_len && suffix_match; c++) {
            if (ext_id[suffix_start + c] != __ldg(&dictionary[word_start + c])) {
                suffix_match = false;
            }
        }

        if (prefix_match || suffix_match) {
            uint32_t match_idx = atomicAdd(match_count, 1);
            if (match_idx < max_matches) {
                matches[match_idx].prime_idx_p = i;
                matches[match_idx].prime_idx_q = j;
                #pragma unroll
                for (uint32_t k = 0; k < 32; k++) {
                    matches[match_idx].ext_id[k] = ext_id[k];
                }
                matches[match_idx].ext_id[32] = '\0';
                matches[match_idx].match_type = (prefix_match ? 1 : 0) | (suffix_match ? 2 : 0);
            }
            return;  // Found a match, don't need to check more words
        }
    }
}

} // namespace cuda
} // namespace vanity

