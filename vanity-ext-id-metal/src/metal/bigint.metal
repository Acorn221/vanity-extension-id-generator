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

// Get bit length of 1024-bit integer
inline uint bigint_bitlen_1024(BigInt1024 a) {
    for (int i = WORDS_1024 - 1; i >= 0; i--) {
        if (a.words[i] != 0) {
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

// Check if bit at position is set (0-indexed from LSB)
inline bool bigint_get_bit(BigInt1024 a, uint bit_pos) {
    uint word_idx = bit_pos / 32;
    uint bit_idx = bit_pos % 32;
    return (a.words[word_idx] >> bit_idx) & 1;
}

// Set 1024-bit integer from a single word
inline BigInt1024 bigint_from_word(uint32_t val) {
    BigInt1024 result;
    result.words[0] = val;
    for (uint i = 1; i < WORDS_1024; i++) {
        result.words[i] = 0;
    }
    return result;
}

// Check if 1024-bit integer equals 1
inline bool bigint_is_one(BigInt1024 a) {
    if (a.words[0] != 1) return false;
    for (uint i = 1; i < WORDS_1024; i++) {
        if (a.words[i] != 0) return false;
    }
    return true;
}

// =============================================================================
// Addition and Subtraction
// =============================================================================

// Add two 1024-bit integers, return result and carry
inline BigInt1024 bigint_add(BigInt1024 a, BigInt1024 b, thread bool* overflow) {
    BigInt1024 result;
    uint64_t carry = 0;
    
    for (uint i = 0; i < WORDS_1024; i++) {
        uint64_t sum = uint64_t(a.words[i]) + uint64_t(b.words[i]) + carry;
        result.words[i] = uint32_t(sum);
        carry = sum >> 32;
    }
    
    *overflow = (carry != 0);
    return result;
}

// Subtract b from a (assumes a >= b)
inline BigInt1024 bigint_sub(BigInt1024 a, BigInt1024 b) {
    BigInt1024 result;
    int64_t borrow = 0;
    
    for (uint i = 0; i < WORDS_1024; i++) {
        int64_t diff = int64_t(a.words[i]) - int64_t(b.words[i]) - borrow;
        if (diff < 0) {
            diff += 0x100000000LL;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result.words[i] = uint32_t(diff);
    }
    
    return result;
}

// =============================================================================
// Modular Arithmetic for Miller-Rabin
// =============================================================================

// Compute a mod n using repeated subtraction/shifting (for reduction)
// This is a simple but correct implementation
inline BigInt1024 bigint_mod(BigInt2048 a, BigInt1024 n) {
    // Convert n to 2048-bit for comparison
    BigInt2048 n_ext;
    for (uint i = 0; i < WORDS_1024; i++) {
        n_ext.words[i] = n.words[i];
    }
    for (uint i = WORDS_1024; i < WORDS_2048; i++) {
        n_ext.words[i] = 0;
    }
    
    // Get bit lengths
    uint a_bits = bigint_bitlen_2048(a);
    uint n_bits = bigint_bitlen_2048(n_ext);
    
    if (a_bits == 0) {
        BigInt1024 zero;
        for (uint i = 0; i < WORDS_1024; i++) zero.words[i] = 0;
        return zero;
    }
    
    if (n_bits == 0 || a_bits < n_bits) {
        BigInt1024 result;
        for (uint i = 0; i < WORDS_1024; i++) {
            result.words[i] = a.words[i];
        }
        return result;
    }
    
    // Shift n left to align with a's MSB
    int shift = int(a_bits) - int(n_bits);
    
    // Shifted n
    BigInt2048 n_shifted = n_ext;
    
    // Left shift n_shifted by 'shift' bits
    if (shift > 0) {
        int word_shift = shift / 32;
        int bit_shift = shift % 32;
        
        // Shift by words first
        if (word_shift > 0) {
            for (int i = WORDS_2048 - 1; i >= word_shift; i--) {
                n_shifted.words[i] = n_shifted.words[i - word_shift];
            }
            for (int i = 0; i < word_shift; i++) {
                n_shifted.words[i] = 0;
            }
        }
        
        // Shift by remaining bits
        if (bit_shift > 0) {
            uint32_t carry = 0;
            for (uint i = 0; i < WORDS_2048; i++) {
                uint32_t new_carry = n_shifted.words[i] >> (32 - bit_shift);
                n_shifted.words[i] = (n_shifted.words[i] << bit_shift) | carry;
                carry = new_carry;
            }
        }
    }
    
    // Subtract shifted n from a repeatedly
    for (int s = shift; s >= 0; s--) {
        // Compare a with n_shifted
        bool a_geq_n = true;
        for (int i = WORDS_2048 - 1; i >= 0; i--) {
            if (a.words[i] < n_shifted.words[i]) {
                a_geq_n = false;
                break;
            }
            if (a.words[i] > n_shifted.words[i]) {
                break;
            }
        }
        
        // If a >= n_shifted, subtract
        if (a_geq_n) {
            int64_t borrow = 0;
            for (uint i = 0; i < WORDS_2048; i++) {
                int64_t diff = int64_t(a.words[i]) - int64_t(n_shifted.words[i]) - borrow;
                if (diff < 0) {
                    diff += 0x100000000LL;
                    borrow = 1;
                } else {
                    borrow = 0;
                }
                a.words[i] = uint32_t(diff);
            }
        }
        
        // Right shift n_shifted by 1
        if (s > 0) {
            uint32_t carry = 0;
            for (int i = WORDS_2048 - 1; i >= 0; i--) {
                uint32_t new_carry = n_shifted.words[i] & 1;
                n_shifted.words[i] = (n_shifted.words[i] >> 1) | (carry << 31);
                carry = new_carry;
            }
        }
    }
    
    // Result is in lower 1024 bits of a
    BigInt1024 result;
    for (uint i = 0; i < WORDS_1024; i++) {
        result.words[i] = a.words[i];
    }
    return result;
}

// Modular addition: (a + b) mod n
inline BigInt1024 bigint_add_mod(BigInt1024 a, BigInt1024 b, BigInt1024 n) {
    bool overflow;
    BigInt1024 sum = bigint_add(a, b, &overflow);
    
    // If overflow or sum >= n, subtract n
    if (overflow || bigint_cmp(sum, n) >= 0) {
        sum = bigint_sub(sum, n);
    }
    
    return sum;
}

// Modular subtraction: (a - b) mod n
inline BigInt1024 bigint_sub_mod(BigInt1024 a, BigInt1024 b, BigInt1024 n) {
    if (bigint_cmp(a, b) >= 0) {
        return bigint_sub(a, b);
    } else {
        // a < b, so result = n - (b - a)
        BigInt1024 diff = bigint_sub(b, a);
        return bigint_sub(n, diff);
    }
}

// Modular multiplication: (a * b) mod n
inline BigInt1024 bigint_mul_mod(BigInt1024 a, BigInt1024 b, BigInt1024 n) {
    BigInt2048 product = bigint_mul(a, b);
    return bigint_mod(product, n);
}

// Modular exponentiation: base^exp mod n (square-and-multiply)
inline BigInt1024 bigint_mod_exp(BigInt1024 base, BigInt1024 exp, BigInt1024 n) {
    // Handle special cases
    if (bigint_is_zero(exp)) {
        return bigint_from_word(1);
    }
    
    BigInt1024 result = bigint_from_word(1);
    BigInt1024 b = base;
    
    // Reduce base mod n first
    if (bigint_cmp(b, n) >= 0) {
        BigInt2048 b_ext;
        for (uint i = 0; i < WORDS_1024; i++) b_ext.words[i] = b.words[i];
        for (uint i = WORDS_1024; i < WORDS_2048; i++) b_ext.words[i] = 0;
        b = bigint_mod(b_ext, n);
    }
    
    // Get bit length of exponent
    uint exp_bits = bigint_bitlen_1024(exp);
    
    // Square-and-multiply from LSB to MSB
    for (uint i = 0; i < exp_bits; i++) {
        if (bigint_get_bit(exp, i)) {
            result = bigint_mul_mod(result, b, n);
        }
        b = bigint_mul_mod(b, b, n);
    }
    
    return result;
}

// Subtract 1 from a 1024-bit integer
inline BigInt1024 bigint_sub_one(BigInt1024 a) {
    BigInt1024 one = bigint_from_word(1);
    return bigint_sub(a, one);
}

// Right shift by 1 bit
inline BigInt1024 bigint_shr_one(BigInt1024 a) {
    BigInt1024 result;
    uint32_t carry = 0;
    for (int i = WORDS_1024 - 1; i >= 0; i--) {
        uint32_t new_carry = a.words[i] & 1;
        result.words[i] = (a.words[i] >> 1) | (carry << 31);
        carry = new_carry;
    }
    return result;
}

// Check if even (LSB = 0)
inline bool bigint_is_even(BigInt1024 a) {
    return (a.words[0] & 1) == 0;
}
