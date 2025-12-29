/**
 * Cryptographic operations: DER encoding, SHA-256, and pattern matching
 */

#pragma once

#include "common.cuh"

namespace vanity {
namespace cuda {

// =============================================================================
// SHA-256 Helper Functions
// =============================================================================

static __device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static __device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static __device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static __device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static __device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static __device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static __device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// =============================================================================
// DER Encoding
// =============================================================================

static __device__ void build_der_pubkey(
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

static __device__ void sha256_der(const uint8_t* __restrict__ data, uint32_t len, uint8_t* __restrict__ hash_out) {
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

static __device__ bool check_prefix(const char* ext_id, const char* target, uint32_t target_len) {
    for (uint32_t i = 0; i < target_len; i++) {
        if (ext_id[i] != target[i]) return false;
    }
    return true;
}

static __device__ bool check_suffix(const char* ext_id, const char* target, uint32_t target_len) {
    uint32_t start = 32 - target_len;
    for (uint32_t i = 0; i < target_len; i++) {
        if (ext_id[start + i] != target[i]) return false;
    }
    return true;
}

// Count occurrences of "ai" in extension ID
static __device__ uint32_t count_ai_occurrences(const char* ext_id) {
    uint32_t count = 0;
    #pragma unroll 31
    for (uint32_t i = 0; i < 31; i++) {
        if (ext_id[i] == 'a' && ext_id[i + 1] == 'i') {
            count++;
        }
    }
    return count;
}

// Find longest run of duplicate characters
static __device__ uint32_t find_max_char_run(const char* ext_id) {
    uint32_t max_run = 1;
    uint32_t current_run = 1;

    #pragma unroll 31
    for (uint32_t i = 1; i < 32; i++) {
        if (ext_id[i] == ext_id[i - 1]) {
            current_run++;
            if (current_run > max_run) {
                max_run = current_run;
            }
        } else {
            current_run = 1;
        }
    }
    return max_run;
}

} // namespace cuda
} // namespace vanity
