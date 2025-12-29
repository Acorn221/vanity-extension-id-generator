/**
 * Unified CUDA kernels for vanity extension ID search
 * 
 * All kernels consolidated into single file to avoid CGBN multiple definition issues.
 * CGBN host functions (error reporting) are defined once here.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <gmp.h>
#include <cgbn/cgbn.h>

namespace vanity {
namespace cuda {

// =============================================================================
// Constants
// =============================================================================

constexpr uint32_t PRIME_WORDS = 32;    // 1024 bits = 32 × 32-bit words
constexpr uint32_t MODULUS_WORDS = 64;  // 2048 bits = 64 × 32-bit words
constexpr uint32_t PRIME_BYTES = 128;   // 1024 bits = 128 bytes
constexpr uint32_t MODULUS_BYTES = 256; // 2048 bits = 256 bytes

constexpr uint32_t BLOCK_SIZE = 256;
constexpr uint32_t MAX_BLOCKS_PER_SM = 2;  // Conservative value for CGBN kernels

// =============================================================================
// CGBN Configuration (TPI=8 optimized for 1024-bit multiplication)
// Based on grid search: TPI=8, WINDOW=4 gives best throughput
// =============================================================================

template<uint32_t tpi, uint32_t bits>
class cgbn_params_t {
public:
    static const uint32_t TPB = 0;              // Use blockDim.x
    static const uint32_t MAX_ROTATION = 4;
    static const uint32_t SHM_LIMIT = 0;
    static const bool CONSTANT_TIME = false;
    static const uint32_t TPI = tpi;
    static const uint32_t BITS = bits;
};

typedef cgbn_params_t<8, 1024> mul_params_t;  // TPI=8 is optimal for multiplication
typedef cgbn_context_t<mul_params_t::TPI, mul_params_t> mul_context_t;
typedef cgbn_env_t<mul_context_t, 1024> mul_env_t;

constexpr uint32_t TPI = 8;   // 8 threads cooperate on one multiplication (optimized)
constexpr uint32_t INSTANCES_PER_BLOCK = BLOCK_SIZE / TPI;  // 256/8 = 32 instances per block

// =============================================================================
// DER Encoding Constants
// =============================================================================

static __constant__ uint8_t DER_HEADER[33] = {
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

static __constant__ uint8_t DER_FOOTER[5] = {
    0x02, 0x03,              // INTEGER, length 3
    0x01, 0x00, 0x01         // 65537 = 0x010001
};

constexpr uint32_t DER_HEADER_LEN = 33;
constexpr uint32_t DER_FOOTER_LEN = 5;
constexpr uint32_t DER_TOTAL_LEN = 294;

// =============================================================================
// SHA-256 Constants
// =============================================================================

static __constant__ uint32_t K[64] = {
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
// Match Result Structure
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
// DER Encoding (Parallelized across TPI threads)
// =============================================================================

static __device__ void build_der_pubkey_parallel(
    const uint32_t* __restrict__ n_words,
    uint8_t* __restrict__ der_out,
    uint32_t lane  // Thread lane within TPI group (0-7)
) {
    // Parallel header copy: 33 bytes / 8 threads ≈ 4-5 bytes each
    #pragma unroll
    for (uint32_t i = lane; i < DER_HEADER_LEN; i += TPI) {
        der_out[i] = DER_HEADER[i];
    }

    // Parallel modulus copy: 64 words / 8 threads = 8 words each
    #pragma unroll
    for (uint32_t i = lane; i < MODULUS_WORDS; i += TPI) {
        uint32_t word_idx = MODULUS_WORDS - 1 - i;
        uint32_t byte_offset = DER_HEADER_LEN + i * 4;
        uint32_t word = n_words[word_idx];
        der_out[byte_offset] = (word >> 24) & 0xFF;
        der_out[byte_offset + 1] = (word >> 16) & 0xFF;
        der_out[byte_offset + 2] = (word >> 8) & 0xFF;
        der_out[byte_offset + 3] = word & 0xFF;
    }

    // Parallel footer copy
    uint32_t footer_offset = DER_HEADER_LEN + MODULUS_BYTES;
    if (lane < DER_FOOTER_LEN) {
        der_out[footer_offset + lane] = DER_FOOTER[lane];
    }
}

// =============================================================================
// SHA-256 Hash (Parallelized across TPI threads)
// =============================================================================

// Shared memory structure for parallel SHA-256 and extension ID
struct SHA256SharedState {
    uint32_t w[64];       // Message schedule
    uint32_t h[8];        // Hash state
    uint8_t hash_out[32]; // Output hash (shared among TPI threads)
    char ext_id[36];      // Extension ID output (shared, with padding)
    uint32_t n_words[64]; // Modulus words (shared for DER encoding)
};

static __device__ void sha256_der_parallel(
    const uint8_t* __restrict__ data, 
    uint32_t len, 
    uint8_t* __restrict__ hash_out,
    uint32_t lane,  // Thread lane (0 to TPI-1)
    SHA256SharedState* shared  // Shared memory for this instance
) {
    // Initialize hash state (all threads do this for consistency)
    if (lane == 0) {
        shared->h[0] = 0x6a09e667; shared->h[1] = 0xbb67ae85;
        shared->h[2] = 0x3c6ef372; shared->h[3] = 0xa54ff53a;
        shared->h[4] = 0x510e527f; shared->h[5] = 0x9b05688c;
        shared->h[6] = 0x1f83d9ab; shared->h[7] = 0x5be0cd19;
    }
    __syncwarp();

    uint32_t num_blocks = (len + 9 + 63) / 64;

    for (uint32_t block = 0; block < num_blocks; block++) {
        uint32_t block_start = block * 64;

        // PARALLEL: Load first 16 words - each of 8 threads loads 2 words
        for (uint32_t i = lane; i < 16; i += TPI) {
            uint32_t byte_idx = block_start + i * 4;
            uint32_t word = 0;

            for (uint32_t b = 0; b < 4; b++) {
                uint32_t idx = byte_idx + b;
                uint8_t byte_val;

                if (idx < len) {
                    byte_val = data[idx];
                } else if (idx == len) {
                    byte_val = 0x80;
                } else if (block == num_blocks - 1 && i >= 14) {
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
            shared->w[i] = word;
        }
        __syncwarp();

        // Message schedule expansion - sequential due to dependencies
        // w[i] depends on w[i-2] which may be computed in same "wave"
        if (lane == 0) {
            #pragma unroll 8
            for (uint32_t i = 16; i < 64; i++) {
                shared->w[i] = gamma1(shared->w[i-2]) + shared->w[i-7] + 
                               gamma0(shared->w[i-15]) + shared->w[i-16];
            }
        }
        __syncwarp();

        // SEQUENTIAL: Compression - only thread 0 does this
        if (lane == 0) {
            uint32_t a = shared->h[0], b = shared->h[1], c = shared->h[2], d = shared->h[3];
            uint32_t e = shared->h[4], f = shared->h[5], g = shared->h[6], hh = shared->h[7];

            #pragma unroll 8
            for (uint32_t i = 0; i < 64; i++) {
                uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + K[i] + shared->w[i];
                uint32_t t2 = sigma0(a) + maj(a, b, c);
                hh = g; g = f; f = e; e = d + t1;
                d = c; c = b; b = a; a = t1 + t2;
            }

            shared->h[0] += a; shared->h[1] += b; shared->h[2] += c; shared->h[3] += d;
            shared->h[4] += e; shared->h[5] += f; shared->h[6] += g; shared->h[7] += hh;
        }
        __syncwarp();
    }

    // PARALLEL: Output hash to shared memory - each thread writes 1 word (we have 8 threads, 8 words)
    if (lane < 8) {
        uint32_t word = shared->h[lane];
        shared->hash_out[lane*4] = (word >> 24) & 0xFF;
        shared->hash_out[lane*4+1] = (word >> 16) & 0xFF;
        shared->hash_out[lane*4+2] = (word >> 8) & 0xFF;
        shared->hash_out[lane*4+3] = word & 0xFF;
    }
    __syncwarp();
    
    // Copy from shared to output (all threads read from shared)
    #pragma unroll
    for (uint32_t i = lane; i < 32; i += TPI) {
        hash_out[i] = shared->hash_out[i];
    }
    __syncwarp();
}


// =============================================================================
// Pattern Matching Helpers
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

// =============================================================================
// CGBN Multiplication (TPI=8 optimized)
// =============================================================================

static __device__ void bigint_mul_1024_cgbn(
    const uint32_t* __restrict__ p,
    const uint32_t* __restrict__ q,
    uint32_t* __restrict__ n,
    cgbn_mem_t<1024>* p_mem,
    cgbn_mem_t<1024>* q_mem,
    cgbn_mem_t<1024>* result_low_mem,
    cgbn_mem_t<1024>* result_high_mem,
    uint32_t instance_idx
) {
    uint32_t local_idx = threadIdx.x / TPI;
    uint32_t lane = threadIdx.x % TPI;

    cgbn_error_report_t *report = nullptr;
    mul_context_t ctx(cgbn_no_checks, report, instance_idx);
    mul_env_t env(ctx);

    typename mul_env_t::cgbn_t p_bn, q_bn;
    typename mul_env_t::cgbn_wide_t result_wide;

    // Each of the 8 threads loads 4 words (32 words total)
    #pragma unroll
    for (uint32_t i = lane; i < PRIME_WORDS; i += TPI) {
        p_mem[local_idx]._limbs[i] = p[i];
        q_mem[local_idx]._limbs[i] = q[i];
    }

    cgbn_load(env, p_bn, &p_mem[local_idx]);
    cgbn_load(env, q_bn, &q_mem[local_idx]);

    cgbn_mul_wide(env, result_wide, p_bn, q_bn);

    cgbn_store(env, &result_low_mem[local_idx], result_wide._low);
    cgbn_store(env, &result_high_mem[local_idx], result_wide._high);

    // Only thread 0 of each group copies back to output
    if (lane == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < PRIME_WORDS; i++) {
            n[i] = result_low_mem[local_idx]._limbs[i];
            n[PRIME_WORDS + i] = result_high_mem[local_idx]._limbs[i];
        }
    }
}

// =============================================================================
// Helper: Convert pair index to (i, j) coordinates
// For pair k in upper triangle: i < j, pairs ordered as (0,1), (0,2), ..., (0,N-1), (1,2), ...
// =============================================================================

static __device__ __forceinline__ void pair_idx_to_ij(uint64_t pair_idx, uint32_t pool_size, uint32_t& i, uint32_t& j) {
    // Use quadratic formula with double precision for accuracy
    double n = (double)pool_size;
    double k = (double)pair_idx;
    double discriminant = (n - 0.5) * (n - 0.5) - 2.0 * k;
    double i_float = n - 0.5 - sqrt(fmax(discriminant, 0.0));
    i = (uint32_t)i_float;
    
    // Clamp and compute j
    if (i >= pool_size) i = pool_size - 1;
    uint64_t pairs_before_i = ((uint64_t)i * (2ULL * pool_size - i - 1)) >> 1;
    
    // Verify we have the right row (handle precision issues)
    while (pairs_before_i + (pool_size - i - 1) <= pair_idx && i < pool_size - 1) {
        pairs_before_i += (pool_size - i - 1);
        i++;
    }
    
    j = (uint32_t)(pair_idx - pairs_before_i) + i + 1;
}

// =============================================================================
// Helper: Load primes and compute extension ID (PARALLELIZED)
// =============================================================================

static __device__ void compute_extension_id_parallel(
    const uint8_t* __restrict__ prime_pool,
    uint32_t i, uint32_t j,
    cgbn_mem_t<1024>* p_mem,
    cgbn_mem_t<1024>* q_mem,
    cgbn_mem_t<1024>* result_low_mem,
    cgbn_mem_t<1024>* result_high_mem,
    uint8_t* der_shared,        // Shared memory for DER (294 bytes per instance)
    SHA256SharedState* sha_shared,  // Shared memory for SHA + ext_id
    char* ext_id_out,           // Output: pointer to copy ext_id to (can be local)
    uint32_t* n_words_out       // Output: modulus words
) {
    const uint8_t* p_bytes = prime_pool + i * PRIME_BYTES;
    const uint8_t* q_bytes = prime_pool + j * PRIME_BYTES;
    uint32_t lane = threadIdx.x % TPI;
    uint32_t instance_idx = threadIdx.x / TPI;
    
    // Pointer to shared ext_id for this instance
    char* ext_id = sha_shared[instance_idx].ext_id;

    uint32_t p_words[PRIME_WORDS];
    uint32_t q_words[PRIME_WORDS];

    // Each thread loads all prime words (needed for CGBN)
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

    // PARALLEL: CGBN multiplication (all 8 threads cooperate)
    // Result goes to n_words_out which is per-thread, but only lane 0 fills it
    bigint_mul_1024_cgbn(p_words, q_words, n_words_out,
                         p_mem, q_mem, result_low_mem, result_high_mem, instance_idx);
    __syncwarp();
    
    // Copy n_words to shared memory so all threads can access it
    uint32_t* shared_n = sha_shared[instance_idx].n_words;
    if (lane == 0) {
        #pragma unroll
        for (uint32_t k = 0; k < MODULUS_WORDS; k++) {
            shared_n[k] = n_words_out[k];
        }
    }
    __syncwarp();

    // PARALLEL: DER encoding (all 8 threads help) - use shared n_words
    uint8_t* my_der = der_shared + instance_idx * DER_TOTAL_LEN;
    build_der_pubkey_parallel(shared_n, my_der, lane);
    __syncwarp();

    // PARALLEL: SHA-256 (all 8 threads help with loading/expansion)
    uint8_t hash[32];
    sha256_der_parallel(my_der, DER_TOTAL_LEN, hash, lane, &sha_shared[instance_idx]);
    // After this, all threads have the hash in their local array

    // PARALLEL: Convert to extension ID in SHARED memory
    // Each thread handles 2 hash bytes (16 bytes / 8 threads = 2 each)
    #pragma unroll
    for (uint32_t k = lane; k < 16; k += TPI) {
        ext_id[k*2] = 'a' + (hash[k] >> 4);
        ext_id[k*2+1] = 'a' + (hash[k] & 0x0F);
    }
    if (lane == 0) {
        ext_id[32] = '\0';
    }
    __syncwarp();
    
    // Lane 0 copies the full shared ext_id to output for checking
    // All other lanes can ignore ext_id_out
    if (lane == 0) {
        #pragma unroll
        for (uint32_t k = 0; k < 33; k++) {
            ext_id_out[k] = ext_id[k];
        }
    }
}


// =============================================================================
// KERNEL: Vanity Search (Prefix/Suffix)
// =============================================================================

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, MAX_BLOCKS_PER_SM) vanity_search_kernel(
    const uint8_t* __restrict__ prime_pool,
    const char* __restrict__ target_prefix,
    const char* __restrict__ target_suffix,
    const uint32_t* __restrict__ params,
    uint32_t* __restrict__ match_count,
    MatchResult* __restrict__ matches,
    uint32_t max_matches,
    uint64_t start_pair_idx,
    uint64_t pairs_this_batch
) {
    // Shared memory for CGBN
    __shared__ cgbn_mem_t<1024> p_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> q_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> result_low_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> result_high_mem[INSTANCES_PER_BLOCK];
    
    // Shared memory for parallel DER and SHA
    __shared__ uint8_t der_shared[INSTANCES_PER_BLOCK * DER_TOTAL_LEN];
    __shared__ SHA256SharedState sha_shared[INSTANCES_PER_BLOCK];

    uint32_t pool_size = params[0];
    uint32_t prefix_len = params[1];
    uint32_t suffix_len = params[2];
    uint32_t lane = threadIdx.x % TPI;

    // Calculate which pair this thread group handles
    uint64_t local_pair_idx = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (local_pair_idx >= pairs_this_batch) return;
    
    uint64_t pair_idx = start_pair_idx + local_pair_idx;

    uint32_t i, j;
    pair_idx_to_ij(pair_idx, pool_size, i, j);

    if (i >= pool_size || j >= pool_size) return;

    char ext_id[33];
    uint32_t n_words[MODULUS_WORDS];
    compute_extension_id_parallel(prime_pool, i, j, p_mem, q_mem, result_low_mem, result_high_mem,
                                  der_shared, sha_shared, ext_id, n_words);

    // All threads now have ext_id - only lane 0 does matching and output
    if (lane == 0) {
        bool prefix_match = (prefix_len > 0) && check_prefix(ext_id, target_prefix, prefix_len);
        bool suffix_match = (suffix_len > 0) && check_suffix(ext_id, target_suffix, suffix_len);

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
        }
    }
}

// =============================================================================
// KERNEL: Dictionary Search
// =============================================================================

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, MAX_BLOCKS_PER_SM) vanity_search_dict_kernel(
    const uint8_t* __restrict__ prime_pool,
    const char* __restrict__ dictionary,
    const uint32_t* __restrict__ word_offsets,
    const uint32_t* __restrict__ params,
    uint32_t* __restrict__ match_count,
    MatchResult* __restrict__ matches,
    uint32_t max_matches,
    uint64_t start_pair_idx,
    uint64_t pairs_this_batch
) {
    __shared__ cgbn_mem_t<1024> p_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> q_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> result_low_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> result_high_mem[INSTANCES_PER_BLOCK];
    __shared__ uint8_t der_shared[INSTANCES_PER_BLOCK * DER_TOTAL_LEN];
    __shared__ SHA256SharedState sha_shared[INSTANCES_PER_BLOCK];

    uint32_t pool_size = params[0];
    uint32_t num_words = params[1];
    uint32_t min_len = params[2];
    uint32_t lane = threadIdx.x % TPI;

    uint64_t local_pair_idx = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (local_pair_idx >= pairs_this_batch) return;
    
    uint64_t pair_idx = start_pair_idx + local_pair_idx;

    uint32_t i, j;
    pair_idx_to_ij(pair_idx, pool_size, i, j);

    if (i >= pool_size || j >= pool_size) return;

    char ext_id[33];
    uint32_t n_words[MODULUS_WORDS];
    compute_extension_id_parallel(prime_pool, i, j, p_mem, q_mem, result_low_mem, result_high_mem,
                                  der_shared, sha_shared, ext_id, n_words);

    if (lane == 0) {
        // Check for cool patterns FIRST
        bool has_pattern = false;

        // 1. Repeated characters (6+ in a row)
        for (uint32_t pos = 0; pos < 27 && !has_pattern; pos++) {
            char c = ext_id[pos];
            uint32_t count = 1;
            while (pos + count < 32 && ext_id[pos + count] == c) count++;
            if (count >= 6) has_pattern = true;
        }

        // 2. Same-char prefix (5+)
        if (!has_pattern) {
            char first = ext_id[0];
            uint32_t prefix_count = 1;
            while (prefix_count < 32 && ext_id[prefix_count] == first) prefix_count++;
            if (prefix_count >= 5) has_pattern = true;
        }

        // 3. Same-char suffix (5+)
        if (!has_pattern) {
            char last = ext_id[31];
            uint32_t suffix_count = 1;
            while (suffix_count < 32 && ext_id[31 - suffix_count] == last) suffix_count++;
            if (suffix_count >= 5) has_pattern = true;
        }

        // 4. Ascending sequence from 'a' (5+ chars)
        if (!has_pattern) {
            for (uint32_t pos = 0; pos < 28 && !has_pattern; pos++) {
                if (ext_id[pos] == 'a') {
                    uint32_t len = 1;
                    while (pos + len < 32 && ext_id[pos + len] == ext_id[pos + len - 1] + 1) len++;
                    if (len >= 5) has_pattern = true;
                }
            }
        }

        // 5. Palindrome at start (8+ chars)
        if (!has_pattern) {
            for (uint32_t len = 12; len >= 8 && !has_pattern; len--) {
                bool is_palindrome = true;
                for (uint32_t p = 0; p < len / 2 && is_palindrome; p++) {
                    if (ext_id[p] != ext_id[len - 1 - p]) is_palindrome = false;
                }
                if (is_palindrome) has_pattern = true;
            }
        }

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
                matches[match_idx].match_type = 4;
            }
            return;
        }

        // Check dictionary
        for (uint32_t w = 0; w < num_words; w++) {
            uint32_t word_start = __ldg(&word_offsets[w]);
            uint32_t word_end = __ldg(&word_offsets[w + 1]);
            uint32_t word_len = word_end - word_start - 1;

            if (word_len < min_len) continue;

            bool prefix_match = true;
            for (uint32_t c = 0; c < word_len && prefix_match; c++) {
                if (ext_id[c] != __ldg(&dictionary[word_start + c])) {
                    prefix_match = false;
                }
            }

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
                return;
            }
        }
    }
}

// =============================================================================
// KERNEL: AI Search (fast "ai" occurrence counting)
// =============================================================================

extern "C" __global__ void __launch_bounds__(BLOCK_SIZE, MAX_BLOCKS_PER_SM) vanity_search_ai_kernel(
    const uint8_t* __restrict__ prime_pool,
    const uint32_t* __restrict__ params,
    uint32_t* __restrict__ match_count,
    MatchResult* __restrict__ matches,
    uint32_t max_matches,
    uint64_t start_pair_idx,
    uint64_t pairs_this_batch
) {
    __shared__ cgbn_mem_t<1024> p_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> q_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> result_low_mem[INSTANCES_PER_BLOCK];
    __shared__ cgbn_mem_t<1024> result_high_mem[INSTANCES_PER_BLOCK];
    __shared__ uint8_t der_shared[INSTANCES_PER_BLOCK * DER_TOTAL_LEN];
    __shared__ SHA256SharedState sha_shared[INSTANCES_PER_BLOCK];

    uint32_t pool_size = params[0];
    uint32_t min_ai_count = params[1];
    uint32_t lane = threadIdx.x % TPI;

    uint64_t local_pair_idx = (uint64_t)(blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (local_pair_idx >= pairs_this_batch) return;
    
    uint64_t pair_idx = start_pair_idx + local_pair_idx;

    uint32_t i, j;
    pair_idx_to_ij(pair_idx, pool_size, i, j);

    if (i >= pool_size || j >= pool_size) return;

    char ext_id[33];
    uint32_t n_words[MODULUS_WORDS];
    compute_extension_id_parallel(prime_pool, i, j, p_mem, q_mem, result_low_mem, result_high_mem,
                                  der_shared, sha_shared, ext_id, n_words);

    if (lane == 0) {
        uint32_t ai_count = count_ai_occurrences(ext_id);
        uint32_t max_run = find_max_char_run(ext_id);

        if (ai_count >= min_ai_count || max_run >= 10) {
            uint32_t match_idx = atomicAdd(match_count, 1);

            if (match_idx < max_matches) {
                matches[match_idx].prime_idx_p = i;
                matches[match_idx].prime_idx_q = j;
                #pragma unroll
                for (uint32_t k = 0; k < 32; k++) {
                    matches[match_idx].ext_id[k] = ext_id[k];
                }
                matches[match_idx].ext_id[32] = '\0';
                matches[match_idx].match_type = ai_count | (max_run << 8);
            }
        }
    }
}

} // namespace cuda
} // namespace vanity

