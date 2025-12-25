/**
 * Prime Generator Kernel for Metal
 * 
 * Generates 1024-bit primes using GPU parallelism:
 * 1. PCG random number generator for candidate generation
 * 2. Small prime sieve to filter obvious composites
 * 3. Miller-Rabin primality test for verification
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

constant uint WORDS_1024 = 32;   // 1024 bits = 32 × 32-bit words
constant uint WORDS_2048 = 64;   // 2048 bits
constant uint PRIME_BYTES = 128; // 1024 bits = 128 bytes

// Number of Miller-Rabin rounds (64 rounds = 2^-128 error probability)
constant uint MILLER_RABIN_ROUNDS = 64;

// First 256 small primes for trial division
constant uint16_t SMALL_PRIMES[256] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
    313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613,
    617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719,
    727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827,
    829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
    947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049,
    1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283,
    1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423,
    1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619
};
constant uint NUM_SMALL_PRIMES = 256;

// =============================================================================
// BigInt Types (duplicated for self-contained shader)
// =============================================================================

struct BigInt1024 {
    uint32_t words[WORDS_1024];  // Little-endian: words[0] is LSB
};

struct BigInt2048 {
    uint32_t words[WORDS_2048];
};

// =============================================================================
// PCG Random Number Generator
// =============================================================================

// PCG-XSH-RR state (64-bit state, 32-bit output)
struct PCGState {
    uint64_t state;
    uint64_t inc;
};

// Initialize PCG from seed
inline PCGState pcg_init(uint64_t seed, uint64_t seq) {
    PCGState rng;
    rng.state = 0;
    rng.inc = (seq << 1) | 1;  // Must be odd
    
    // Warm up
    rng.state = rng.state * 6364136223846793005ULL + rng.inc;
    rng.state += seed;
    rng.state = rng.state * 6364136223846793005ULL + rng.inc;
    
    return rng;
}

// Generate next 32-bit random number
inline uint32_t pcg_next(thread PCGState* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    
    uint32_t xorshifted = uint32_t(((oldstate >> 18) ^ oldstate) >> 27);
    uint32_t rot = uint32_t(oldstate >> 59);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Generate random 1024-bit integer
inline BigInt1024 pcg_random_1024(thread PCGState* rng) {
    BigInt1024 result;
    for (uint i = 0; i < WORDS_1024; i++) {
        result.words[i] = pcg_next(rng);
    }
    return result;
}

// =============================================================================
// BigInt Operations (self-contained for this shader)
// =============================================================================

inline bool bigint_is_zero(BigInt1024 a) {
    for (uint i = 0; i < WORDS_1024; i++) {
        if (a.words[i] != 0) return false;
    }
    return true;
}

inline bool bigint_is_one(BigInt1024 a) {
    if (a.words[0] != 1) return false;
    for (uint i = 1; i < WORDS_1024; i++) {
        if (a.words[i] != 0) return false;
    }
    return true;
}

inline BigInt1024 bigint_from_word(uint32_t val) {
    BigInt1024 result;
    result.words[0] = val;
    for (uint i = 1; i < WORDS_1024; i++) {
        result.words[i] = 0;
    }
    return result;
}

inline int bigint_cmp(BigInt1024 a, BigInt1024 b) {
    for (int i = WORDS_1024 - 1; i >= 0; i--) {
        if (a.words[i] < b.words[i]) return -1;
        if (a.words[i] > b.words[i]) return 1;
    }
    return 0;
}

inline uint bigint_bitlen(BigInt1024 a) {
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

inline bool bigint_get_bit(BigInt1024 a, uint bit_pos) {
    uint word_idx = bit_pos / 32;
    uint bit_idx = bit_pos % 32;
    return (a.words[word_idx] >> bit_idx) & 1;
}

inline bool bigint_is_even(BigInt1024 a) {
    return (a.words[0] & 1) == 0;
}

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

inline BigInt2048 bigint_mul(BigInt1024 a, BigInt1024 b) {
    BigInt2048 result;
    
    for (uint i = 0; i < WORDS_2048; i++) {
        result.words[i] = 0;
    }
    
    for (uint i = 0; i < WORDS_1024; i++) {
        uint64_t carry = 0;
        
        for (uint j = 0; j < WORDS_1024; j++) {
            uint k = i + j;
            uint64_t product = uint64_t(a.words[i]) * uint64_t(b.words[j]);
            uint64_t sum = uint64_t(result.words[k]) + product + carry;
            result.words[k] = uint32_t(sum);
            carry = sum >> 32;
        }
        
        for (uint k = i + WORDS_1024; carry && k < WORDS_2048; k++) {
            uint64_t sum = uint64_t(result.words[k]) + carry;
            result.words[k] = uint32_t(sum);
            carry = sum >> 32;
        }
    }
    
    return result;
}

// Modular reduction: a mod n
inline BigInt1024 bigint_mod(BigInt2048 a, BigInt1024 n) {
    BigInt2048 n_ext;
    for (uint i = 0; i < WORDS_1024; i++) {
        n_ext.words[i] = n.words[i];
    }
    for (uint i = WORDS_1024; i < WORDS_2048; i++) {
        n_ext.words[i] = 0;
    }
    
    // Get bit lengths
    uint a_bits = 0;
    for (int i = WORDS_2048 - 1; i >= 0; i--) {
        if (a.words[i] != 0) {
            uint32_t word = a.words[i];
            uint bits = 0;
            while (word) { bits++; word >>= 1; }
            a_bits = i * 32 + bits;
            break;
        }
    }
    
    uint n_bits = 0;
    for (int i = WORDS_2048 - 1; i >= 0; i--) {
        if (n_ext.words[i] != 0) {
            uint32_t word = n_ext.words[i];
            uint bits = 0;
            while (word) { bits++; word >>= 1; }
            n_bits = i * 32 + bits;
            break;
        }
    }
    
    if (a_bits == 0 || n_bits == 0) {
        BigInt1024 zero;
        for (uint i = 0; i < WORDS_1024; i++) zero.words[i] = 0;
        return zero;
    }
    
    if (a_bits < n_bits) {
        BigInt1024 result;
        for (uint i = 0; i < WORDS_1024; i++) {
            result.words[i] = a.words[i];
        }
        return result;
    }
    
    int shift = int(a_bits) - int(n_bits);
    BigInt2048 n_shifted = n_ext;
    
    // Left shift n_shifted
    if (shift > 0) {
        int word_shift = shift / 32;
        int bit_shift = shift % 32;
        
        if (word_shift > 0) {
            for (int i = WORDS_2048 - 1; i >= word_shift; i--) {
                n_shifted.words[i] = n_shifted.words[i - word_shift];
            }
            for (int i = 0; i < word_shift; i++) {
                n_shifted.words[i] = 0;
            }
        }
        
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
        
        if (s > 0) {
            uint32_t carry = 0;
            for (int i = WORDS_2048 - 1; i >= 0; i--) {
                uint32_t new_carry = n_shifted.words[i] & 1;
                n_shifted.words[i] = (n_shifted.words[i] >> 1) | (carry << 31);
                carry = new_carry;
            }
        }
    }
    
    BigInt1024 result;
    for (uint i = 0; i < WORDS_1024; i++) {
        result.words[i] = a.words[i];
    }
    return result;
}

inline BigInt1024 bigint_mul_mod(BigInt1024 a, BigInt1024 b, BigInt1024 n) {
    BigInt2048 product = bigint_mul(a, b);
    return bigint_mod(product, n);
}

inline BigInt1024 bigint_add_mod(BigInt1024 a, BigInt1024 b, BigInt1024 n) {
    bool overflow;
    BigInt1024 sum = bigint_add(a, b, &overflow);
    
    if (overflow || bigint_cmp(sum, n) >= 0) {
        sum = bigint_sub(sum, n);
    }
    
    return sum;
}

// Modular exponentiation: base^exp mod n
inline BigInt1024 bigint_mod_exp(BigInt1024 base, BigInt1024 exp, BigInt1024 n) {
    if (bigint_is_zero(exp)) {
        return bigint_from_word(1);
    }
    
    BigInt1024 result = bigint_from_word(1);
    BigInt1024 b = base;
    
    if (bigint_cmp(b, n) >= 0) {
        BigInt2048 b_ext;
        for (uint i = 0; i < WORDS_1024; i++) b_ext.words[i] = b.words[i];
        for (uint i = WORDS_1024; i < WORDS_2048; i++) b_ext.words[i] = 0;
        b = bigint_mod(b_ext, n);
    }
    
    uint exp_bits = bigint_bitlen(exp);
    
    for (uint i = 0; i < exp_bits; i++) {
        if (bigint_get_bit(exp, i)) {
            result = bigint_mul_mod(result, b, n);
        }
        b = bigint_mul_mod(b, b, n);
    }
    
    return result;
}

// =============================================================================
// Small Prime Sieve
// =============================================================================

// Check if n is divisible by any small prime
// Returns true if n passes sieve (might be prime)
// Returns false if n is definitely composite
inline bool passes_small_prime_sieve(BigInt1024 n) {
    // Skip 2 since we'll only test odd numbers
    for (uint i = 1; i < NUM_SMALL_PRIMES; i++) {
        uint32_t p = SMALL_PRIMES[i];
        
        // Compute n mod p using Horner's method
        uint64_t rem = 0;
        for (int j = WORDS_1024 - 1; j >= 0; j--) {
            rem = (rem << 32) | n.words[j];
            rem = rem % p;
        }
        
        if (rem == 0) {
            // n is divisible by p
            // Only composite if n > p
            if (bigint_bitlen(n) > 16 || n.words[0] > p) {
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// Miller-Rabin Primality Test
// =============================================================================

// Single Miller-Rabin witness test
// Returns true if n is probably prime with respect to witness a
// Returns false if n is definitely composite
inline bool miller_rabin_witness(BigInt1024 n, BigInt1024 n_minus_1, BigInt1024 d, uint s, BigInt1024 a) {
    // Compute x = a^d mod n
    BigInt1024 x = bigint_mod_exp(a, d, n);
    
    // If x == 1 or x == n-1, n passes this round
    if (bigint_is_one(x) || bigint_cmp(x, n_minus_1) == 0) {
        return true;
    }
    
    // Square x repeatedly
    for (uint r = 1; r < s; r++) {
        x = bigint_mul_mod(x, x, n);
        
        if (bigint_cmp(x, n_minus_1) == 0) {
            return true;
        }
        if (bigint_is_one(x)) {
            return false;
        }
    }
    
    return false;
}

// Full Miller-Rabin test with multiple rounds
// Returns true if n is probably prime
inline bool miller_rabin_test(BigInt1024 n, thread PCGState* rng) {
    // Handle small cases
    if (bigint_is_zero(n)) return false;
    if (bigint_is_one(n)) return false;
    
    // Check if even
    if (bigint_is_even(n)) {
        // Only 2 is prime among evens
        BigInt1024 two = bigint_from_word(2);
        return bigint_cmp(n, two) == 0;
    }
    
    // Write n-1 as 2^s * d where d is odd
    BigInt1024 n_minus_1 = bigint_sub(n, bigint_from_word(1));
    BigInt1024 d = n_minus_1;
    uint s = 0;
    
    while (bigint_is_even(d)) {
        d = bigint_shr_one(d);
        s++;
    }
    
    // Perform Miller-Rabin rounds
    for (uint round = 0; round < MILLER_RABIN_ROUNDS; round++) {
        // Generate random witness a in [2, n-2]
        BigInt1024 a = pcg_random_1024(rng);
        
        // Reduce to range [2, n-2]
        BigInt2048 a_ext;
        for (uint i = 0; i < WORDS_1024; i++) a_ext.words[i] = a.words[i];
        for (uint i = WORDS_1024; i < WORDS_2048; i++) a_ext.words[i] = 0;
        
        BigInt1024 n_minus_3 = bigint_sub(n, bigint_from_word(3));
        a = bigint_mod(a_ext, n_minus_3);
        
        bool overflow;
        a = bigint_add(a, bigint_from_word(2), &overflow);
        
        // Test witness
        if (!miller_rabin_witness(n, n_minus_1, d, s, a)) {
            return false;  // Definitely composite
        }
    }
    
    return true;  // Probably prime
}

// =============================================================================
// Prime Candidate Generation
// =============================================================================

// Generate a valid 1024-bit prime candidate:
// - MSB set (ensures 1024 bits)
// - LSB set (ensures odd)
inline BigInt1024 generate_candidate(thread PCGState* rng) {
    BigInt1024 candidate = pcg_random_1024(rng);
    
    // Set MSB to ensure exactly 1024 bits
    candidate.words[WORDS_1024 - 1] |= 0x80000000;
    
    // Set LSB to ensure odd
    candidate.words[0] |= 1;
    
    return candidate;
}

// =============================================================================
// Main Prime Generation Kernel
// =============================================================================

kernel void generate_primes(
    device uint8_t* output_primes [[buffer(0)]],        // Output buffer for primes
    device atomic_uint* prime_count [[buffer(1)]],       // Number of primes found
    device const uint64_t* seeds [[buffer(2)]],          // Per-thread seeds
    device const uint32_t* params [[buffer(3)]],         // [target_count, max_attempts_per_thread]
    uint tid [[thread_position_in_grid]]
) {
    uint target_count = params[0];
    uint max_attempts = params[1];
    
    // Initialize RNG with thread-specific seed
    PCGState rng = pcg_init(seeds[tid], tid);
    
    // Try to find primes
    for (uint attempt = 0; attempt < max_attempts; attempt++) {
        // Check if we've found enough primes
        uint current_count = atomic_load_explicit(prime_count, memory_order_relaxed);
        if (current_count >= target_count) {
            return;
        }
        
        // Generate candidate
        BigInt1024 candidate = generate_candidate(&rng);
        
        // Quick sieve test
        if (!passes_small_prime_sieve(candidate)) {
            continue;
        }
        
        // Full Miller-Rabin test
        if (!miller_rabin_test(candidate, &rng)) {
            continue;
        }
        
        // Found a prime! Store it
        uint idx = atomic_fetch_add_explicit(prime_count, 1, memory_order_relaxed);
        if (idx < target_count) {
            // Store prime in big-endian format
            device uint8_t* out = output_primes + idx * PRIME_BYTES;
            
            for (uint w = 0; w < WORDS_1024; w++) {
                uint word_idx = WORDS_1024 - 1 - w;
                uint byte_offset = w * 4;
                uint32_t word = candidate.words[word_idx];
                out[byte_offset] = (word >> 24) & 0xFF;
                out[byte_offset + 1] = (word >> 16) & 0xFF;
                out[byte_offset + 2] = (word >> 8) & 0xFF;
                out[byte_offset + 3] = word & 0xFF;
            }
        }
    }
}

// =============================================================================
// Batch Prime Generation Kernel (processes multiple candidates per thread)
// =============================================================================

kernel void generate_primes_batch(
    device uint8_t* output_primes [[buffer(0)]],
    device atomic_uint* prime_count [[buffer(1)]],
    device const uint64_t* seeds [[buffer(2)]],
    device const uint32_t* params [[buffer(3)]],         // [target_count, batch_size, global_seed_offset]
    uint tid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    uint target_count = params[0];
    uint batch_size = params[1];
    uint64_t seed_offset = params[2];
    
    // Initialize RNG with unique seed per thread
    PCGState rng = pcg_init(seeds[tid % 1024] + seed_offset, tid + seed_offset);
    
    // Each thread processes batch_size candidates
    for (uint b = 0; b < batch_size; b++) {
        // Check if done
        uint current_count = atomic_load_explicit(prime_count, memory_order_relaxed);
        if (current_count >= target_count) {
            return;
        }
        
        // Generate and test candidate
        BigInt1024 candidate = generate_candidate(&rng);
        
        if (!passes_small_prime_sieve(candidate)) {
            continue;
        }
        
        if (!miller_rabin_test(candidate, &rng)) {
            continue;
        }
        
        // Store prime
        uint idx = atomic_fetch_add_explicit(prime_count, 1, memory_order_relaxed);
        if (idx < target_count) {
            device uint8_t* out = output_primes + idx * PRIME_BYTES;
            
            for (uint w = 0; w < WORDS_1024; w++) {
                uint word_idx = WORDS_1024 - 1 - w;
                uint byte_offset = w * 4;
                uint32_t word = candidate.words[word_idx];
                out[byte_offset] = (word >> 24) & 0xFF;
                out[byte_offset + 1] = (word >> 16) & 0xFF;
                out[byte_offset + 2] = (word >> 8) & 0xFF;
                out[byte_offset + 3] = word & 0xFF;
            }
        }
    }
}

