/**
 * Prime Generator Compute Shader for WebGPU
 * 
 * Generates 1024-bit primes using GPU parallelism:
 * 1. PCG random number generator for candidate generation
 * 2. Small prime sieve to filter obvious composites
 * 3. Miller-Rabin primality test for verification
 * 
 * Ported from Metal shader (prime_generator.metal)
 * All 64-bit arithmetic emulated with 32-bit operations (WGSL has no native i64/u64)
 */

// =============================================================================
// Constants
// =============================================================================

const WORDS_1024: u32 = 32u;   // 1024 bits = 32 × 32-bit words
const WORDS_2048: u32 = 64u;   // 2048 bits
const PRIME_BYTES: u32 = 128u; // 1024 bits = 128 bytes

// Number of Miller-Rabin rounds (64 rounds = 2^-128 error probability)
const MILLER_RABIN_ROUNDS: u32 = 64u;

// Number of small primes for trial division
const NUM_SMALL_PRIMES: u32 = 256u;

// =============================================================================
// Types
// =============================================================================

struct BigInt1024 {
    words: array<u32, 32>,  // Little-endian: words[0] is LSB
}

struct BigInt2048 {
    words: array<u32, 64>,
}

// PCG state using two u32s to emulate u64
struct PCGState {
    state_lo: u32,
    state_hi: u32,
    inc_lo: u32,
    inc_hi: u32,
}

// =============================================================================
// Buffer Bindings
// =============================================================================

@group(0) @binding(0) var<storage, read_write> output_primes: array<u32>;
@group(0) @binding(1) var<storage, read_write> prime_count: atomic<u32>;
@group(0) @binding(2) var<storage, read> seeds: array<u32>;  // 2 u32s per thread for seed
@group(0) @binding(3) var<uniform> params: vec4<u32>;  // [target_count, batch_size, seed_offset, 0]
@group(0) @binding(4) var<storage, read> small_primes: array<u32>;

// =============================================================================
// 64-bit Arithmetic Helpers (emulated with 2x u32)
// =============================================================================

// Add two 64-bit numbers represented as (lo, hi) pairs
fn u64_add(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    let sum_lo = a_lo + b_lo;
    let carry = select(0u, 1u, sum_lo < a_lo);
    let sum_hi = a_hi + b_hi + carry;
    return vec2<u32>(sum_lo, sum_hi);
}

// Multiply two 32-bit numbers to get 64-bit result
fn u32_mul_u64(a: u32, b: u32) -> vec2<u32> {
    // Split into 16-bit halves
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    
    // Multiply parts
    let lo_lo = a_lo * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_lo = a_hi * b_lo;
    let hi_hi = a_hi * b_hi;
    
    // Combine with carries
    let mid = lo_hi + hi_lo;
    let mid_carry = select(0u, 0x10000u, mid < lo_hi);  // Carry to high word
    
    let result_lo = lo_lo + (mid << 16u);
    let lo_carry = select(0u, 1u, result_lo < lo_lo);
    
    let result_hi = hi_hi + (mid >> 16u) + mid_carry + lo_carry;
    
    return vec2<u32>(result_lo, result_hi);
}

// Multiply two 64-bit numbers, keep lower 64 bits
fn u64_mul(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    // (a_hi * 2^32 + a_lo) * (b_hi * 2^32 + b_lo)
    // = a_hi * b_hi * 2^64 + (a_hi * b_lo + a_lo * b_hi) * 2^32 + a_lo * b_lo
    // We only need the lower 64 bits
    
    let lo_lo = u32_mul_u64(a_lo, b_lo);
    let cross1 = a_hi * b_lo;
    let cross2 = a_lo * b_hi;
    
    let result_hi = lo_lo.y + cross1 + cross2;
    
    return vec2<u32>(lo_lo.x, result_hi);
}

fn u64_xor(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> vec2<u32> {
    return vec2<u32>(a_lo ^ b_lo, a_hi ^ b_hi);
}

fn u64_shl(a_lo: u32, a_hi: u32, shift: u32) -> vec2<u32> {
    if (shift == 0u) {
        return vec2<u32>(a_lo, a_hi);
    }
    if (shift >= 64u) {
        return vec2<u32>(0u, 0u);
    }
    if (shift >= 32u) {
        return vec2<u32>(0u, a_lo << (shift - 32u));
    }
    let new_hi = (a_hi << shift) | (a_lo >> (32u - shift));
    let new_lo = a_lo << shift;
    return vec2<u32>(new_lo, new_hi);
}

fn u64_shr(a_lo: u32, a_hi: u32, shift: u32) -> vec2<u32> {
    if (shift == 0u) {
        return vec2<u32>(a_lo, a_hi);
    }
    if (shift >= 64u) {
        return vec2<u32>(0u, 0u);
    }
    if (shift >= 32u) {
        return vec2<u32>(a_hi >> (shift - 32u), 0u);
    }
    let new_lo = (a_lo >> shift) | (a_hi << (32u - shift));
    let new_hi = a_hi >> shift;
    return vec2<u32>(new_lo, new_hi);
}

// =============================================================================
// PCG Random Number Generator
// =============================================================================

fn pcg_init(seed_lo: u32, seed_hi: u32, seq: u32) -> PCGState {
    var rng: PCGState;
    rng.state_lo = 0u;
    rng.state_hi = 0u;
    // inc must be odd: (seq << 1) | 1
    rng.inc_lo = (seq << 1u) | 1u;
    rng.inc_hi = seq >> 31u;
    
    // Warm up: state = state * 6364136223846793005 + inc
    let mult_lo = 0x5851F42Du;
    let mult_hi = 0x14057B7Eu;
    
    var state = u64_mul(rng.state_lo, rng.state_hi, mult_lo, mult_hi);
    state = u64_add(state.x, state.y, rng.inc_lo, rng.inc_hi);
    rng.state_lo = state.x;
    rng.state_hi = state.y;
    
    // Add seed
    state = u64_add(rng.state_lo, rng.state_hi, seed_lo, seed_hi);
    rng.state_lo = state.x;
    rng.state_hi = state.y;
    
    // Another multiply
    state = u64_mul(rng.state_lo, rng.state_hi, mult_lo, mult_hi);
    state = u64_add(state.x, state.y, rng.inc_lo, rng.inc_hi);
    rng.state_lo = state.x;
    rng.state_hi = state.y;
    
    return rng;
}

fn pcg_next(rng: ptr<function, PCGState>) -> u32 {
    let oldstate_lo = (*rng).state_lo;
    let oldstate_hi = (*rng).state_hi;
    
    // state = state * 6364136223846793005 + inc
    let mult_lo = 0x5851F42Du;
    let mult_hi = 0x14057B7Eu;
    
    var state = u64_mul(oldstate_lo, oldstate_hi, mult_lo, mult_hi);
    state = u64_add(state.x, state.y, (*rng).inc_lo, (*rng).inc_hi);
    (*rng).state_lo = state.x;
    (*rng).state_hi = state.y;
    
    // PCG-XSH-RR output function
    // xorshifted = ((oldstate >> 18) ^ oldstate) >> 27
    let shifted18 = u64_shr(oldstate_lo, oldstate_hi, 18u);
    let xored = u64_xor(shifted18.x, shifted18.y, oldstate_lo, oldstate_hi);
    let shifted27 = u64_shr(xored.x, xored.y, 27u);
    let xorshifted = shifted27.x;
    
    // rot = oldstate >> 59
    let rot = oldstate_hi >> 27u;
    
    // Rotate right
    let neg_rot = (32u - rot) & 31u;
    return (xorshifted >> rot) | (xorshifted << neg_rot);
}

fn pcg_random_1024(rng: ptr<function, PCGState>) -> BigInt1024 {
    var result: BigInt1024;
    for (var i = 0u; i < WORDS_1024; i++) {
        result.words[i] = pcg_next(rng);
    }
    return result;
}

// =============================================================================
// BigInt Operations
// =============================================================================

fn bigint_is_zero(a: BigInt1024) -> bool {
    for (var i = 0u; i < WORDS_1024; i++) {
        if (a.words[i] != 0u) {
            return false;
        }
    }
    return true;
}

fn bigint_is_one(a: BigInt1024) -> bool {
    if (a.words[0] != 1u) {
        return false;
    }
    for (var i = 1u; i < WORDS_1024; i++) {
        if (a.words[i] != 0u) {
            return false;
        }
    }
    return true;
}

fn bigint_from_word(val: u32) -> BigInt1024 {
    var result: BigInt1024;
    result.words[0] = val;
    for (var i = 1u; i < WORDS_1024; i++) {
        result.words[i] = 0u;
    }
    return result;
}

fn bigint_cmp(a: BigInt1024, b: BigInt1024) -> i32 {
    for (var i = i32(WORDS_1024) - 1; i >= 0; i--) {
        let idx = u32(i);
        if (a.words[idx] < b.words[idx]) {
            return -1;
        }
        if (a.words[idx] > b.words[idx]) {
            return 1;
        }
    }
    return 0;
}

fn bigint_bitlen(a: BigInt1024) -> u32 {
    for (var i = i32(WORDS_1024) - 1; i >= 0; i--) {
        let idx = u32(i);
        if (a.words[idx] != 0u) {
            var word = a.words[idx];
            var bits = 0u;
            while (word != 0u) {
                bits++;
                word >>= 1u;
            }
            return idx * 32u + bits;
        }
    }
    return 0u;
}

fn bigint_get_bit(a: BigInt1024, bit_pos: u32) -> bool {
    let word_idx = bit_pos / 32u;
    let bit_idx = bit_pos % 32u;
    return ((a.words[word_idx] >> bit_idx) & 1u) == 1u;
}

fn bigint_is_even(a: BigInt1024) -> bool {
    return (a.words[0] & 1u) == 0u;
}

// Add two BigInt1024 values using 32-bit arithmetic with carry propagation
fn bigint_add(a: BigInt1024, b: BigInt1024) -> BigInt1024 {
    var result: BigInt1024;
    var carry = 0u;
    
    for (var i = 0u; i < WORDS_1024; i++) {
        let sum = u32_mul_u64(1u, a.words[i]);  // Just to get the word
        let a_word = a.words[i];
        let b_word = b.words[i];
        
        // First add a + b
        let sum1 = a_word + b_word;
        let carry1 = select(0u, 1u, sum1 < a_word);
        
        // Then add carry
        let sum2 = sum1 + carry;
        let carry2 = select(0u, 1u, sum2 < sum1);
        
        result.words[i] = sum2;
        carry = carry1 + carry2;
    }
    
    return result;
}

// Subtract b from a using 32-bit arithmetic with borrow propagation
fn bigint_sub(a: BigInt1024, b: BigInt1024) -> BigInt1024 {
    var result: BigInt1024;
    var borrow = 0u;
    
    for (var i = 0u; i < WORDS_1024; i++) {
        let a_word = a.words[i];
        let b_word = b.words[i];
        
        // Subtract with borrow
        let diff1 = a_word - b_word;
        let borrow1 = select(0u, 1u, a_word < b_word);
        
        let diff2 = diff1 - borrow;
        let borrow2 = select(0u, 1u, diff1 < borrow);
        
        result.words[i] = diff2;
        borrow = borrow1 + borrow2;
    }
    
    return result;
}

fn bigint_shr_one(a: BigInt1024) -> BigInt1024 {
    var result: BigInt1024;
    var carry = 0u;
    for (var i = i32(WORDS_1024) - 1; i >= 0; i--) {
        let idx = u32(i);
        let new_carry = a.words[idx] & 1u;
        result.words[idx] = (a.words[idx] >> 1u) | (carry << 31u);
        carry = new_carry;
    }
    return result;
}

// Multiply two BigInt1024 values to get BigInt2048
fn bigint_mul(a: BigInt1024, b: BigInt1024) -> BigInt2048 {
    var result: BigInt2048;
    
    for (var i = 0u; i < WORDS_2048; i++) {
        result.words[i] = 0u;
    }
    
    for (var i = 0u; i < WORDS_1024; i++) {
        var carry = 0u;
        
        for (var j = 0u; j < WORDS_1024; j++) {
            let k = i + j;
            
            // Multiply a.words[i] * b.words[j] to get 64-bit product
            let prod = u32_mul_u64(a.words[i], b.words[j]);
            
            // Add to result[k] with existing value and carry
            let sum1 = result.words[k] + prod.x;
            let carry1 = select(0u, 1u, sum1 < result.words[k]);
            
            let sum2 = sum1 + carry;
            let carry2 = select(0u, 1u, sum2 < sum1);
            
            result.words[k] = sum2;
            carry = prod.y + carry1 + carry2;
        }
        
        // Propagate final carry
        var k = i + WORDS_1024;
        while (carry != 0u && k < WORDS_2048) {
            let sum = result.words[k] + carry;
            let new_carry = select(0u, 1u, sum < result.words[k]);
            result.words[k] = sum;
            carry = new_carry;
            k++;
        }
    }
    
    return result;
}

fn bigint2048_bitlen(a: BigInt2048) -> u32 {
    for (var i = i32(WORDS_2048) - 1; i >= 0; i--) {
        let idx = u32(i);
        if (a.words[idx] != 0u) {
            var word = a.words[idx];
            var bits = 0u;
            while (word != 0u) {
                bits++;
                word >>= 1u;
            }
            return idx * 32u + bits;
        }
    }
    return 0u;
}

// Modular reduction: a mod n
fn bigint_mod(a: BigInt2048, n: BigInt1024) -> BigInt1024 {
    var a_copy = a;
    var n_ext: BigInt2048;
    for (var i = 0u; i < WORDS_1024; i++) {
        n_ext.words[i] = n.words[i];
    }
    for (var i = WORDS_1024; i < WORDS_2048; i++) {
        n_ext.words[i] = 0u;
    }
    
    let a_bits = bigint2048_bitlen(a_copy);
    let n_bits = bigint2048_bitlen(n_ext);
    
    if (a_bits == 0u || n_bits == 0u) {
        var zero: BigInt1024;
        for (var i = 0u; i < WORDS_1024; i++) {
            zero.words[i] = 0u;
        }
        return zero;
    }
    
    if (a_bits < n_bits) {
        var result: BigInt1024;
        for (var i = 0u; i < WORDS_1024; i++) {
            result.words[i] = a_copy.words[i];
        }
        return result;
    }
    
    let shift = i32(a_bits) - i32(n_bits);
    var n_shifted = n_ext;
    
    // Left shift n_shifted
    if (shift > 0) {
        let word_shift = u32(shift) / 32u;
        let bit_shift = u32(shift) % 32u;
        
        if (word_shift > 0u) {
            for (var i = i32(WORDS_2048) - 1; i >= i32(word_shift); i--) {
                n_shifted.words[u32(i)] = n_shifted.words[u32(i) - word_shift];
            }
            for (var i = 0u; i < word_shift; i++) {
                n_shifted.words[i] = 0u;
            }
        }
        
        if (bit_shift > 0u) {
            var carry = 0u;
            for (var i = 0u; i < WORDS_2048; i++) {
                let new_carry = n_shifted.words[i] >> (32u - bit_shift);
                n_shifted.words[i] = (n_shifted.words[i] << bit_shift) | carry;
                carry = new_carry;
            }
        }
    }
    
    // Subtract shifted n from a repeatedly
    for (var s = shift; s >= 0; s--) {
        // Compare a >= n_shifted
        var a_geq_n = true;
        for (var i = i32(WORDS_2048) - 1; i >= 0; i--) {
            let idx = u32(i);
            if (a_copy.words[idx] < n_shifted.words[idx]) {
                a_geq_n = false;
                break;
            }
            if (a_copy.words[idx] > n_shifted.words[idx]) {
                break;
            }
        }
        
        if (a_geq_n) {
            // Subtract n_shifted from a_copy
            var borrow = 0u;
            for (var i = 0u; i < WORDS_2048; i++) {
                let a_word = a_copy.words[i];
                let n_word = n_shifted.words[i];
                
                let diff1 = a_word - n_word;
                let borrow1 = select(0u, 1u, a_word < n_word);
                
                let diff2 = diff1 - borrow;
                let borrow2 = select(0u, 1u, diff1 < borrow);
                
                a_copy.words[i] = diff2;
                borrow = borrow1 + borrow2;
            }
        }
        
        if (s > 0) {
            // Right shift n_shifted by 1
            var carry = 0u;
            for (var i = i32(WORDS_2048) - 1; i >= 0; i--) {
                let idx = u32(i);
                let new_carry = n_shifted.words[idx] & 1u;
                n_shifted.words[idx] = (n_shifted.words[idx] >> 1u) | (carry << 31u);
                carry = new_carry;
            }
        }
    }
    
    var result: BigInt1024;
    for (var i = 0u; i < WORDS_1024; i++) {
        result.words[i] = a_copy.words[i];
    }
    return result;
}

fn bigint_mul_mod(a: BigInt1024, b: BigInt1024, n: BigInt1024) -> BigInt1024 {
    let product = bigint_mul(a, b);
    return bigint_mod(product, n);
}

fn bigint_mod_exp(base: BigInt1024, exp: BigInt1024, n: BigInt1024) -> BigInt1024 {
    if (bigint_is_zero(exp)) {
        return bigint_from_word(1u);
    }
    
    var result = bigint_from_word(1u);
    var b = base;
    
    // Reduce base if needed
    if (bigint_cmp(b, n) >= 0) {
        var b_ext: BigInt2048;
        for (var i = 0u; i < WORDS_1024; i++) {
            b_ext.words[i] = b.words[i];
        }
        for (var i = WORDS_1024; i < WORDS_2048; i++) {
            b_ext.words[i] = 0u;
        }
        b = bigint_mod(b_ext, n);
    }
    
    let exp_bits = bigint_bitlen(exp);
    
    for (var i = 0u; i < exp_bits; i++) {
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

// Compute n mod p for small prime p using Horner's method with 64-bit emulation
fn bigint_mod_small(n: BigInt1024, p: u32) -> u32 {
    var rem_lo = 0u;
    var rem_hi = 0u;
    
    for (var j = i32(WORDS_1024) - 1; j >= 0; j--) {
        let idx = u32(j);
        // rem = (rem << 32) | n.words[idx]
        // Then rem = rem % p
        
        // Shift rem left by 32 and add word
        rem_hi = rem_lo;
        rem_lo = n.words[idx];
        
        // Now compute (rem_hi * 2^32 + rem_lo) mod p
        // = (rem_hi mod p) * (2^32 mod p) + rem_lo mod p
        // But 2^32 mod p = (2^32 - p * floor(2^32/p))
        
        // For small p, we can do this iteratively
        // rem = rem_hi * 2^32 + rem_lo
        // We reduce in steps
        
        // First reduce rem_hi
        let rem_hi_mod = rem_hi % p;
        
        // Compute rem_hi_mod * 2^32 mod p in parts
        // 2^32 = 2^16 * 2^16
        let pow16_mod_p = (65536u % p);
        let pow32_mod_p = (pow16_mod_p * pow16_mod_p) % p;
        
        let hi_contrib = (rem_hi_mod * pow32_mod_p) % p;
        let lo_contrib = rem_lo % p;
        
        rem_lo = (hi_contrib + lo_contrib) % p;
        rem_hi = 0u;
    }
    
    return rem_lo;
}

fn passes_small_prime_sieve(n: BigInt1024) -> bool {
    // Skip index 0 which is 2 since we only test odd numbers
    for (var i = 1u; i < NUM_SMALL_PRIMES; i++) {
        let p = small_primes[i];
        
        let rem = bigint_mod_small(n, p);
        
        if (rem == 0u) {
            // n is divisible by p - composite if n > p
            if (bigint_bitlen(n) > 16u || n.words[0] > p) {
                return false;
            }
        }
    }
    return true;
}

// =============================================================================
// Miller-Rabin Primality Test
// =============================================================================

fn miller_rabin_witness(n: BigInt1024, n_minus_1: BigInt1024, d: BigInt1024, s: u32, a: BigInt1024) -> bool {
    // Compute x = a^d mod n
    var x = bigint_mod_exp(a, d, n);
    
    // If x == 1 or x == n-1, n passes this round
    if (bigint_is_one(x) || bigint_cmp(x, n_minus_1) == 0) {
        return true;
    }
    
    // Square x repeatedly
    for (var r = 1u; r < s; r++) {
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

fn miller_rabin_test(n: BigInt1024, rng: ptr<function, PCGState>) -> bool {
    // Handle small cases
    if (bigint_is_zero(n)) {
        return false;
    }
    if (bigint_is_one(n)) {
        return false;
    }
    
    // Check if even
    if (bigint_is_even(n)) {
        let two = bigint_from_word(2u);
        return bigint_cmp(n, two) == 0;
    }
    
    // Write n-1 as 2^s * d where d is odd
    let n_minus_1 = bigint_sub(n, bigint_from_word(1u));
    var d = n_minus_1;
    var s = 0u;
    
    while (bigint_is_even(d)) {
        d = bigint_shr_one(d);
        s++;
    }
    
    // Perform Miller-Rabin rounds
    for (var round = 0u; round < MILLER_RABIN_ROUNDS; round++) {
        // Generate random witness a in [2, n-2]
        var a = pcg_random_1024(rng);
        
        // Reduce to range [2, n-2]
        var a_ext: BigInt2048;
        for (var i = 0u; i < WORDS_1024; i++) {
            a_ext.words[i] = a.words[i];
        }
        for (var i = WORDS_1024; i < WORDS_2048; i++) {
            a_ext.words[i] = 0u;
        }
        
        let n_minus_3 = bigint_sub(n, bigint_from_word(3u));
        a = bigint_mod(a_ext, n_minus_3);
        a = bigint_add(a, bigint_from_word(2u));
        
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

fn generate_candidate(rng: ptr<function, PCGState>) -> BigInt1024 {
    var candidate = pcg_random_1024(rng);
    
    // Set MSB to ensure exactly 1024 bits
    candidate.words[WORDS_1024 - 1u] |= 0x80000000u;
    
    // Set LSB to ensure odd
    candidate.words[0] |= 1u;
    
    return candidate;
}

// =============================================================================
// Main Compute Kernel
// =============================================================================

@compute @workgroup_size(64)
fn generate_primes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let target_count = params.x;
    let batch_size = params.y;
    let seed_offset = params.z;
    
    // Initialize RNG with thread-specific seed
    let seed_idx = (tid % 1024u) * 2u;
    let seed_lo = seeds[seed_idx] + seed_offset;
    let seed_hi = seeds[seed_idx + 1u];
    var rng = pcg_init(seed_lo, seed_hi, tid + seed_offset);
    
    // Process batch_size candidates
    for (var b = 0u; b < batch_size; b++) {
        // Check if we've found enough primes
        let current_count = atomicLoad(&prime_count);
        if (current_count >= target_count) {
            return;
        }
        
        // Generate candidate
        let candidate = generate_candidate(&rng);
        
        // Quick sieve test
        if (!passes_small_prime_sieve(candidate)) {
            continue;
        }
        
        // Full Miller-Rabin test
        if (!miller_rabin_test(candidate, &rng)) {
            continue;
        }
        
        // Found a prime! Store it
        let idx = atomicAdd(&prime_count, 1u);
        if (idx < target_count) {
            // Store prime in big-endian format (as u32 words)
            let out_offset = idx * 32u;  // 32 words per prime
            
            for (var w = 0u; w < WORDS_1024; w++) {
                let word_idx = WORDS_1024 - 1u - w;
                let word = candidate.words[word_idx];
                // Store as big-endian bytes packed into u32
                // Each output u32 = one input u32 byte-swapped
                let byte0 = (word >> 24u) & 0xFFu;
                let byte1 = (word >> 16u) & 0xFFu;
                let byte2 = (word >> 8u) & 0xFFu;
                let byte3 = word & 0xFFu;
                output_primes[out_offset + w] = (byte0 << 24u) | (byte1 << 16u) | (byte2 << 8u) | byte3;
            }
        }
    }
}
