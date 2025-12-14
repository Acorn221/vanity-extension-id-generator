/**
 * Vanity Extension ID Search Kernel for Metal
 * 
 * Main compute shader that:
 * 1. Takes two prime indices from the pool
 * 2. Computes n = p × q
 * 3. Builds DER-encoded public key
 * 4. Computes SHA-256 hash
 * 5. Checks extension ID against patterns
 * 6. Reports matches
 */

#include <metal_stdlib>
using namespace metal;

// Include our utility headers (these would be in separate files in practice)
// For now, we inline the necessary types

// =============================================================================
// Constants
// =============================================================================

constant uint PRIME_WORDS = 32;    // 1024 bits = 32 × 32-bit words
constant uint MODULUS_WORDS = 64;  // 2048 bits = 64 × 32-bit words
constant uint PRIME_BYTES = 128;   // 1024 bits = 128 bytes
constant uint MODULUS_BYTES = 256; // 2048 bits = 256 bytes

// RSA public exponent e = 65537
constant uint32_t PUBLIC_EXPONENT = 65537;

// DER encoding template for RSA-2048 public key (SubjectPublicKeyInfo)
// Total length: 294 bytes (varies slightly based on leading zeros in n)
// Structure:
//   SEQUENCE {
//     SEQUENCE { OID rsaEncryption, NULL }
//     BIT STRING { SEQUENCE { INTEGER n, INTEGER e } }
//   }

// Fixed header before the modulus n (varies by 1-2 bytes based on n length)
constant uint8_t DER_HEADER[] = {
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
constant uint DER_HEADER_LEN = 33;

// Fixed footer after modulus n (the public exponent e = 65537)
constant uint8_t DER_FOOTER[] = {
    0x02, 0x03,              // INTEGER, length 3
    0x01, 0x00, 0x01         // 65537 = 0x010001
};
constant uint DER_FOOTER_LEN = 5;

// Total DER length: header(33) + modulus(256) + footer(5) = 294 bytes
constant uint DER_TOTAL_LEN = 294;

// =============================================================================
// SHA-256 Implementation (inline for kernel)
// =============================================================================

constant uint32_t K[64] = {
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

inline uint32_t rotr(uint32_t x, uint n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

inline uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

inline uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// =============================================================================
// Big Integer Multiplication (1024 × 1024 = 2048 bits)
// =============================================================================

// Multiply two 1024-bit primes to get 2048-bit modulus
// Input: p, q as arrays of 32 uint32_t words (little-endian)
// Output: n as array of 64 uint32_t words (little-endian)
inline void bigint_mul_1024(
    thread const uint32_t* p,
    thread const uint32_t* q,
    thread uint32_t* n
) {
    // Zero initialize result
    for (uint i = 0; i < MODULUS_WORDS; i++) {
        n[i] = 0;
    }
    
    // Schoolbook multiplication
    for (uint i = 0; i < PRIME_WORDS; i++) {
        uint64_t carry = 0;
        
        for (uint j = 0; j < PRIME_WORDS; j++) {
            uint k = i + j;
            uint64_t product = uint64_t(p[i]) * uint64_t(q[j]);
            uint64_t sum = uint64_t(n[k]) + product + carry;
            n[k] = uint32_t(sum);
            carry = sum >> 32;
        }
        
        // Propagate carry
        for (uint k = i + PRIME_WORDS; carry && k < MODULUS_WORDS; k++) {
            uint64_t sum = uint64_t(n[k]) + carry;
            n[k] = uint32_t(sum);
            carry = sum >> 32;
        }
    }
}

// =============================================================================
// DER Encoding
// =============================================================================

// Build DER-encoded public key from modulus n
// n is in little-endian word order, output is big-endian bytes
inline void build_der_pubkey(
    thread const uint32_t* n_words,
    thread uint8_t* der_out
) {
    // Copy header
    for (uint i = 0; i < DER_HEADER_LEN; i++) {
        der_out[i] = DER_HEADER[i];
    }
    
    // Copy modulus n (convert from little-endian words to big-endian bytes)
    for (uint i = 0; i < MODULUS_WORDS; i++) {
        uint word_idx = MODULUS_WORDS - 1 - i;
        uint byte_offset = DER_HEADER_LEN + i * 4;
        uint32_t word = n_words[word_idx];
        der_out[byte_offset] = (word >> 24) & 0xFF;
        der_out[byte_offset + 1] = (word >> 16) & 0xFF;
        der_out[byte_offset + 2] = (word >> 8) & 0xFF;
        der_out[byte_offset + 3] = word & 0xFF;
    }
    
    // Copy footer
    uint footer_offset = DER_HEADER_LEN + MODULUS_BYTES;
    for (uint i = 0; i < DER_FOOTER_LEN; i++) {
        der_out[footer_offset + i] = DER_FOOTER[i];
    }
}

// =============================================================================
// SHA-256 Hash (optimized for ~294 byte input)
// =============================================================================

inline void sha256_der(thread const uint8_t* data, uint len, thread uint8_t* hash_out) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Process all complete 64-byte blocks
    uint num_blocks = (len + 9 + 63) / 64;  // Include padding
    
    for (uint block = 0; block < num_blocks; block++) {
        uint32_t w[64];
        uint block_start = block * 64;
        
        // Load or pad this block
        for (uint i = 0; i < 16; i++) {
            uint byte_idx = block_start + i * 4;
            uint32_t word = 0;
            
            for (uint b = 0; b < 4; b++) {
                uint idx = byte_idx + b;
                uint8_t byte_val;
                
                if (idx < len) {
                    byte_val = data[idx];
                } else if (idx == len) {
                    byte_val = 0x80;  // Padding bit
                } else if (block == num_blocks - 1 && i >= 14) {
                    // Length in last 8 bytes
                    uint64_t bit_len = uint64_t(len) * 8;
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
        for (uint i = 16; i < 64; i++) {
            w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
        }
        
        // Compression
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        
        for (uint i = 0; i < 64; i++) {
            uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = sigma0(a) + maj(a, b, c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    
    // Output hash
    for (uint i = 0; i < 8; i++) {
        hash_out[i*4] = (h[i] >> 24) & 0xFF;
        hash_out[i*4+1] = (h[i] >> 16) & 0xFF;
        hash_out[i*4+2] = (h[i] >> 8) & 0xFF;
        hash_out[i*4+3] = h[i] & 0xFF;
    }
}

// =============================================================================
// Pattern Matching
// =============================================================================

// Check if extension ID matches a target pattern at start
inline bool check_prefix(thread const char* ext_id, device const char* target, uint target_len) {
    for (uint i = 0; i < target_len; i++) {
        if (ext_id[i] != target[i]) return false;
    }
    return true;
}

// Check if extension ID matches a target pattern at end
inline bool check_suffix(thread const char* ext_id, device const char* target, uint target_len) {
    uint start = 32 - target_len;
    for (uint i = 0; i < target_len; i++) {
        if (ext_id[start + i] != target[i]) return false;
    }
    return true;
}

// =============================================================================
// Match Result Structure
// =============================================================================

struct MatchResult {
    uint32_t prime_idx_p;    // Index of prime p in pool
    uint32_t prime_idx_q;    // Index of prime q in pool
    char ext_id[36];         // Extension ID (32 chars + padding)
    uint32_t match_type;     // 1=prefix, 2=suffix, 3=both
};

// =============================================================================
// Main Search Kernel
// =============================================================================

kernel void vanity_search(
    device const uint8_t* prime_pool [[buffer(0)]],      // All primes (128 bytes each)
    device const char* target_prefix [[buffer(1)]],       // Target prefix to match
    device const char* target_suffix [[buffer(2)]],       // Target suffix to match
    device const uint32_t* params [[buffer(3)]],          // [pool_size, prefix_len, suffix_len, start_idx]
    device atomic_uint* match_count [[buffer(4)]],        // Number of matches found
    device MatchResult* matches [[buffer(5)]],            // Output buffer for matches
    uint2 gid [[thread_position_in_grid]]                 // 2D grid position
) {
    uint pool_size = params[0];
    uint prefix_len = params[1];
    uint suffix_len = params[2];
    uint start_idx = params[3];
    
    // Calculate prime pair indices from 2D grid
    // We're searching the upper triangle of the matrix (i < j)
    uint i = gid.x + start_idx;
    uint j = gid.y + i + 1;  // j > i always
    
    if (i >= pool_size || j >= pool_size) return;
    
    // Load primes from pool (big-endian bytes)
    device const uint8_t* p_bytes = prime_pool + i * PRIME_BYTES;
    device const uint8_t* q_bytes = prime_pool + j * PRIME_BYTES;
    
    // Convert to little-endian words for multiplication
    uint32_t p_words[PRIME_WORDS];
    uint32_t q_words[PRIME_WORDS];
    
    for (uint w = 0; w < PRIME_WORDS; w++) {
        uint byte_idx = (PRIME_WORDS - 1 - w) * 4;
        p_words[w] = (uint32_t(p_bytes[byte_idx]) << 24) |
                     (uint32_t(p_bytes[byte_idx + 1]) << 16) |
                     (uint32_t(p_bytes[byte_idx + 2]) << 8) |
                     uint32_t(p_bytes[byte_idx + 3]);
        q_words[w] = (uint32_t(q_bytes[byte_idx]) << 24) |
                     (uint32_t(q_bytes[byte_idx + 1]) << 16) |
                     (uint32_t(q_bytes[byte_idx + 2]) << 8) |
                     uint32_t(q_bytes[byte_idx + 3]);
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
    for (uint k = 0; k < 16; k++) {
        ext_id[k*2] = 'a' + (hash[k] >> 4);
        ext_id[k*2+1] = 'a' + (hash[k] & 0x0F);
    }
    ext_id[32] = '\0';
    
    // Check for matches
    bool prefix_match = (prefix_len > 0) && check_prefix(ext_id, target_prefix, prefix_len);
    bool suffix_match = (suffix_len > 0) && check_suffix(ext_id, target_suffix, suffix_len);
    
    if (prefix_match || suffix_match) {
        // Found a match! Record it
        uint match_idx = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
        
        if (match_idx < 10000) {  // Limit matches to prevent overflow
            matches[match_idx].prime_idx_p = i;
            matches[match_idx].prime_idx_q = j;
            for (uint k = 0; k < 32; k++) {
                matches[match_idx].ext_id[k] = ext_id[k];
            }
            matches[match_idx].ext_id[32] = '\0';
            matches[match_idx].match_type = (prefix_match ? 1 : 0) | (suffix_match ? 2 : 0);
        }
    }
}

// =============================================================================
// Dictionary Search Kernel (checks against word list)
// =============================================================================

kernel void vanity_search_dict(
    device const uint8_t* prime_pool [[buffer(0)]],
    device const char* dictionary [[buffer(1)]],          // Packed words, null-separated
    device const uint32_t* word_offsets [[buffer(2)]],    // Offset of each word in dictionary
    device const uint32_t* params [[buffer(3)]],          // [pool_size, num_words, min_len, start_idx]
    device atomic_uint* match_count [[buffer(4)]],
    device MatchResult* matches [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pool_size = params[0];
    uint num_words = params[1];
    uint min_len = params[2];
    uint start_idx = params[3];
    
    uint i = gid.x + start_idx;
    uint j = gid.y + i + 1;
    
    if (i >= pool_size || j >= pool_size) return;
    
    // Load and multiply primes (same as above)
    device const uint8_t* p_bytes = prime_pool + i * PRIME_BYTES;
    device const uint8_t* q_bytes = prime_pool + j * PRIME_BYTES;
    
    uint32_t p_words[PRIME_WORDS];
    uint32_t q_words[PRIME_WORDS];
    
    for (uint w = 0; w < PRIME_WORDS; w++) {
        uint byte_idx = (PRIME_WORDS - 1 - w) * 4;
        p_words[w] = (uint32_t(p_bytes[byte_idx]) << 24) |
                     (uint32_t(p_bytes[byte_idx + 1]) << 16) |
                     (uint32_t(p_bytes[byte_idx + 2]) << 8) |
                     uint32_t(p_bytes[byte_idx + 3]);
        q_words[w] = (uint32_t(q_bytes[byte_idx]) << 24) |
                     (uint32_t(q_bytes[byte_idx + 1]) << 16) |
                     (uint32_t(q_bytes[byte_idx + 2]) << 8) |
                     uint32_t(q_bytes[byte_idx + 3]);
    }
    
    uint32_t n_words[MODULUS_WORDS];
    bigint_mul_1024(p_words, q_words, n_words);
    
    uint8_t der_pubkey[DER_TOTAL_LEN];
    build_der_pubkey(n_words, der_pubkey);
    
    uint8_t hash[32];
    sha256_der(der_pubkey, DER_TOTAL_LEN, hash);
    
    char ext_id[33];
    for (uint k = 0; k < 16; k++) {
        ext_id[k*2] = 'a' + (hash[k] >> 4);
        ext_id[k*2+1] = 'a' + (hash[k] & 0x0F);
    }
    ext_id[32] = '\0';
    
    // Check against dictionary (prefix and suffix)
    for (uint w = 0; w < num_words; w++) {
        uint word_start = word_offsets[w];
        uint word_end = word_offsets[w + 1];
        uint word_len = word_end - word_start - 1;  // -1 for null terminator
        
        if (word_len < min_len) continue;
        
        // Check prefix
        bool prefix_match = true;
        for (uint c = 0; c < word_len && prefix_match; c++) {
            if (ext_id[c] != dictionary[word_start + c]) {
                prefix_match = false;
            }
        }
        
        // Check suffix
        bool suffix_match = true;
        uint suffix_start = 32 - word_len;
        for (uint c = 0; c < word_len && suffix_match; c++) {
            if (ext_id[suffix_start + c] != dictionary[word_start + c]) {
                suffix_match = false;
            }
        }
        
        if (prefix_match || suffix_match) {
            uint match_idx = atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
            if (match_idx < 10000) {
                matches[match_idx].prime_idx_p = i;
                matches[match_idx].prime_idx_q = j;
                for (uint k = 0; k < 32; k++) {
                    matches[match_idx].ext_id[k] = ext_id[k];
                }
                matches[match_idx].ext_id[32] = '\0';
                matches[match_idx].match_type = (prefix_match ? 1 : 0) | (suffix_match ? 2 : 0);
            }
            return;  // Found a match, don't need to check more words
        }
    }
}
