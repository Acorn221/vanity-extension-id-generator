/**
 * Big Integer Operations for Metal
 * 
 * Implements 1024-bit and 2048-bit integer arithmetic using 32-bit words.
 * 1024-bit = 32 x uint32_t words (little-endian internally)
 * 2048-bit = 64 x uint32_t words (for multiplication result)
 */

#include <metal_stdlib>
using namespace metal;

// Number of 32-bit words for different sizes
constant uint WORDS_1024 = 32;  // 1024 bits
constant uint WORDS_2048 = 64;  // 2048 bits

// =============================================================================
// 1024-bit Integer Type
// =============================================================================

struct BigInt1024 {
    uint32_t words[WORDS_1024];  // Little-endian: words[0] is LSB
};

struct BigInt2048 {
    uint32_t words[WORDS_2048];  // Little-endian: words[0] is LSB
};

// =============================================================================
// Load/Store Operations
// =============================================================================

// Load 1024-bit integer from big-endian byte array
inline BigInt1024 bigint_load_be(device const uint8_t* data) {
    BigInt1024 result;
    
    // Convert big-endian bytes to little-endian words
    for (uint i = 0; i < WORDS_1024; i++) {
        uint byte_offset = (WORDS_1024 - 1 - i) * 4;
        result.words[i] = (uint32_t(data[byte_offset]) << 24) |
                          (uint32_t(data[byte_offset + 1]) << 16) |
                          (uint32_t(data[byte_offset + 2]) << 8) |
                          uint32_t(data[byte_offset + 3]);
    }
    
    return result;
}

// Store 2048-bit integer to big-endian byte array
inline void bigint_store_be(BigInt2048 val, device uint8_t* out) {
    for (uint i = 0; i < WORDS_2048; i++) {
        uint byte_offset = (WORDS_2048 - 1 - i) * 4;
        uint32_t word = val.words[i];
        out[byte_offset] = (word >> 24) & 0xFF;
        out[byte_offset + 1] = (word >> 16) & 0xFF;
        out[byte_offset + 2] = (word >> 8) & 0xFF;
        out[byte_offset + 3] = word & 0xFF;
    }
}

// =============================================================================
// Multiplication: 1024-bit × 1024-bit = 2048-bit
// =============================================================================

// Schoolbook multiplication
// This is O(n²) but simple and correct
inline BigInt2048 bigint_mul(BigInt1024 a, BigInt1024 b) {
    BigInt2048 result;
    
    // Zero initialize
    for (uint i = 0; i < WORDS_2048; i++) {
        result.words[i] = 0;
    }
    
    // Schoolbook multiplication
    for (uint i = 0; i < WORDS_1024; i++) {
        uint64_t carry = 0;
        
        for (uint j = 0; j < WORDS_1024; j++) {
            uint k = i + j;
            
            // Multiply and add
            uint64_t product = uint64_t(a.words[i]) * uint64_t(b.words[j]);
            uint64_t sum = uint64_t(result.words[k]) + product + carry;
            
            result.words[k] = uint32_t(sum & 0xFFFFFFFF);
            carry = sum >> 32;
        }
        
        // Propagate remaining carry
        uint k = i + WORDS_1024;
        while (carry && k < WORDS_2048) {
            uint64_t sum = uint64_t(result.words[k]) + carry;
            result.words[k] = uint32_t(sum & 0xFFFFFFFF);
            carry = sum >> 32;
            k++;
        }
    }
    
    return result;
}

// =============================================================================
// Comparison Operations
// =============================================================================

// Compare two 1024-bit integers
// Returns: -1 if a < b, 0 if a == b, 1 if a > b
inline int bigint_cmp(BigInt1024 a, BigInt1024 b) {
    for (int i = WORDS_1024 - 1; i >= 0; i--) {
        if (a.words[i] < b.words[i]) return -1;
        if (a.words[i] > b.words[i]) return 1;
    }
    return 0;
}

// =============================================================================
// Utility Functions
// =============================================================================

// Check if 1024-bit integer is zero
inline bool bigint_is_zero(BigInt1024 a) {
    for (uint i = 0; i < WORDS_1024; i++) {
        if (a.words[i] != 0) return false;
    }
    return true;
}

// Get bit length of 2048-bit integer (position of highest set bit + 1)
inline uint bigint_bitlen_2048(BigInt2048 a) {
    for (int i = WORDS_2048 - 1; i >= 0; i--) {
        if (a.words[i] != 0) {
            // Find highest bit in this word
            uint32_t word = a.words[i];
            uint bits = 0;
            while (word) {
                bits++;
                word >>= 1;
            }
            return i * 32 + bits;
        }
    }
    return 0;
}
