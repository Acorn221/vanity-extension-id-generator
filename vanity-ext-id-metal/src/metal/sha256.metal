/**
 * SHA-256 Implementation for Metal
 * 
 * Computes SHA-256 hash of input data.
 * Optimized for hashing RSA public key DER encodings (~270 bytes).
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// SHA-256 Constants
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

constant uint32_t H_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// =============================================================================
// SHA-256 Helper Functions
// =============================================================================

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
// SHA-256 State
// =============================================================================

struct SHA256State {
    uint32_t h[8];
    uint64_t total_len;
};

inline SHA256State sha256_init() {
    SHA256State state;
    for (int i = 0; i < 8; i++) {
        state.h[i] = H_INIT[i];
    }
    state.total_len = 0;
    return state;
}

// Process a single 64-byte block
inline void sha256_process_block(thread SHA256State& state, thread const uint8_t* block) {
    uint32_t w[64];
    
    // Prepare message schedule (first 16 words from block)
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t(block[i*4]) << 24) |
               (uint32_t(block[i*4+1]) << 16) |
               (uint32_t(block[i*4+2]) << 8) |
               uint32_t(block[i*4+3]);
    }
    
    // Extend message schedule
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }
    
    // Initialize working variables
    uint32_t a = state.h[0];
    uint32_t b = state.h[1];
    uint32_t c = state.h[2];
    uint32_t d = state.h[3];
    uint32_t e = state.h[4];
    uint32_t f = state.h[5];
    uint32_t g = state.h[6];
    uint32_t h = state.h[7];
    
    // Main loop
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Add to hash
    state.h[0] += a;
    state.h[1] += b;
    state.h[2] += c;
    state.h[3] += d;
    state.h[4] += e;
    state.h[5] += f;
    state.h[6] += g;
    state.h[7] += h;
}

// =============================================================================
// SHA-256 Complete Hash Function
// =============================================================================

// Hash a message of known length (up to 512 bytes for our use case)
// Output: 32 bytes (256 bits)
inline void sha256_hash(thread const uint8_t* data, uint len, thread uint8_t* out) {
    SHA256State state = sha256_init();
    
    // Process complete blocks
    uint blocks = len / 64;
    for (uint i = 0; i < blocks; i++) {
        sha256_process_block(state, data + i * 64);
    }
    
    // Handle final block with padding
    uint8_t final_block[128];  // May need 2 blocks for padding
    uint remaining = len % 64;
    
    // Copy remaining bytes
    for (uint i = 0; i < remaining; i++) {
        final_block[i] = data[blocks * 64 + i];
    }
    
    // Add padding bit
    final_block[remaining] = 0x80;
    
    // Zero rest of block
    for (uint i = remaining + 1; i < 64; i++) {
        final_block[i] = 0;
    }
    
    // Check if we need two blocks
    if (remaining >= 56) {
        // Process first padding block
        sha256_process_block(state, final_block);
        
        // Zero second block
        for (uint i = 0; i < 56; i++) {
            final_block[i] = 0;
        }
    }
    
    // Add length (in bits) as big-endian 64-bit integer
    uint64_t bit_len = uint64_t(len) * 8;
    final_block[56] = (bit_len >> 56) & 0xFF;
    final_block[57] = (bit_len >> 48) & 0xFF;
    final_block[58] = (bit_len >> 40) & 0xFF;
    final_block[59] = (bit_len >> 32) & 0xFF;
    final_block[60] = (bit_len >> 24) & 0xFF;
    final_block[61] = (bit_len >> 16) & 0xFF;
    final_block[62] = (bit_len >> 8) & 0xFF;
    final_block[63] = bit_len & 0xFF;
    
    // Process final block
    sha256_process_block(state, final_block);
    
    // Output hash (big-endian)
    for (int i = 0; i < 8; i++) {
        out[i*4] = (state.h[i] >> 24) & 0xFF;
        out[i*4+1] = (state.h[i] >> 16) & 0xFF;
        out[i*4+2] = (state.h[i] >> 8) & 0xFF;
        out[i*4+3] = state.h[i] & 0xFF;
    }
}

// =============================================================================
// Extension ID Generation
// =============================================================================

// Convert SHA-256 hash to Chrome extension ID (first 16 bytes -> 32 chars a-p)
inline void hash_to_extension_id(thread const uint8_t* hash, thread char* ext_id) {
    for (int i = 0; i < 16; i++) {
        ext_id[i*2] = 'a' + (hash[i] >> 4);
        ext_id[i*2+1] = 'a' + (hash[i] & 0x0F);
    }
    ext_id[32] = '\0';
}
