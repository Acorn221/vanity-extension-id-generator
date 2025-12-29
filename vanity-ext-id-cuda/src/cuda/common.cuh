/**
 * Common CUDA definitions for vanity extension ID search
 */

#pragma once

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
constexpr uint32_t MAX_BLOCKS_PER_SM = 8;

// =============================================================================
// CGBN Configuration (TPI=32 for 1024-bit multiplication)
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

typedef cgbn_params_t<32, 1024> mul_params_t;
typedef cgbn_context_t<mul_params_t::TPI, mul_params_t> mul_context_t;
typedef cgbn_env_t<mul_context_t, 1024> mul_env_t;

constexpr uint32_t TPI = 32;  // 32 threads cooperate on one multiplication
constexpr uint32_t INSTANCES_PER_BLOCK = BLOCK_SIZE / TPI;  // 256/32 = 8 instances per block

// =============================================================================
// DER Encoding Constants
// =============================================================================

// DER encoding header for RSA-2048 public key (SubjectPublicKeyInfo)
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

// SHA-256 round constants
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

} // namespace cuda
} // namespace vanity
