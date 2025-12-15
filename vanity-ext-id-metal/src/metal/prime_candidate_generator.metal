/**
 * Fast Prime Candidate Generator for Metal
 * 
 * Generates random 1024-bit odd numbers and filters them with trial division.
 * The candidates that pass are sent to CPU for final Miller-Rabin verification.
 * 
 * This is MUCH faster than doing Miller-Rabin on GPU because:
 * - Trial division is simple arithmetic (fast on GPU)
 * - Filters ~85% of candidates cheaply
 * - CPU's OpenSSL Miller-Rabin is highly optimized
 */

#include <metal_stdlib>
using namespace metal;

constant uint WORDS_1024 = 32;   // 1024 bits = 32 × 32-bit words
constant uint PRIME_BYTES = 128; // 1024 bits = 128 bytes

// First 256 small primes for trial division
// Checking these filters ~85% of random odd numbers
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

// =============================================================================
// PCG Random Number Generator (fast, good quality)
// =============================================================================

struct PCGState {
    uint64_t state;
    uint64_t inc;
};

inline PCGState pcg_init(uint64_t seed, uint64_t seq) {
    PCGState rng;
    rng.state = 0;
    rng.inc = (seq << 1) | 1;
    rng.state = rng.state * 6364136223846793005ULL + rng.inc;
    rng.state += seed;
    rng.state = rng.state * 6364136223846793005ULL + rng.inc;
    return rng;
}

inline uint32_t pcg_next(thread PCGState* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = uint32_t(((oldstate >> 18) ^ oldstate) >> 27);
    uint32_t rot = uint32_t(oldstate >> 59);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// =============================================================================
// Trial Division (check divisibility by small primes)
// =============================================================================

// Check if 1024-bit number (as 32 words, little-endian) is divisible by small prime p
// Uses Horner's method: n mod p = (...((w[31] mod p) * 2^32 + w[30]) mod p ...) mod p
inline bool is_divisible_by(thread const uint32_t* words, uint32_t p) {
    uint64_t rem = 0;
    uint64_t multiplier = (uint64_t(1) << 32) % p;  // 2^32 mod p
    
    for (int i = WORDS_1024 - 1; i >= 0; i--) {
        rem = (rem * multiplier + (words[i] % p)) % p;
    }
    
    return rem == 0;
}

// Check candidate against all small primes (skip 2 since we only generate odd numbers)
inline bool passes_trial_division(thread const uint32_t* words) {
    // Skip index 0 (prime 2) since we only generate odd numbers
    for (uint i = 1; i < 256; i++) {
        if (is_divisible_by(words, SMALL_PRIMES[i])) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Main Candidate Generation Kernel
// =============================================================================

kernel void generate_candidates(
    device uint8_t* output_candidates [[buffer(0)]],    // Output buffer for candidates
    device atomic_uint* candidate_count [[buffer(1)]],   // Number of candidates found
    device const uint64_t* seeds [[buffer(2)]],          // Per-thread seeds
    device const uint32_t* params [[buffer(3)]],         // [max_candidates, candidates_per_thread]
    uint tid [[thread_position_in_grid]]
) {
    uint max_candidates = params[0];
    uint attempts_per_thread = params[1];
    uint64_t seed_base = seeds[tid % 1024];
    
    // Initialize RNG with unique seed
    PCGState rng = pcg_init(seed_base + tid * 12345ULL, tid);
    
    // Generate candidate words
    uint32_t words[WORDS_1024];
    
    for (uint attempt = 0; attempt < attempts_per_thread; attempt++) {
        // Check if we have enough candidates
        uint current = atomic_load_explicit(candidate_count, memory_order_relaxed);
        if (current >= max_candidates) {
            return;
        }
        
        // Generate random 1024-bit number
        for (uint i = 0; i < WORDS_1024; i++) {
            words[i] = pcg_next(&rng);
        }
        
        // Set MSB to ensure exactly 1024 bits
        words[WORDS_1024 - 1] |= 0x80000000;
        
        // Set LSB to ensure odd
        words[0] |= 1;
        
        // Trial division filter
        if (!passes_trial_division(words)) {
            continue;
        }
        
        // Passed! Store candidate
        uint idx = atomic_fetch_add_explicit(candidate_count, 1, memory_order_relaxed);
        if (idx < max_candidates) {
            device uint8_t* out = output_candidates + idx * PRIME_BYTES;
            
            // Store as big-endian bytes
            for (uint w = 0; w < WORDS_1024; w++) {
                uint word_idx = WORDS_1024 - 1 - w;
                uint byte_offset = w * 4;
                uint32_t word = words[word_idx];
                out[byte_offset] = (word >> 24) & 0xFF;
                out[byte_offset + 1] = (word >> 16) & 0xFF;
                out[byte_offset + 2] = (word >> 8) & 0xFF;
                out[byte_offset + 3] = word & 0xFF;
            }
        }
    }
}

