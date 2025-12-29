/**
 * Minimal CGBN Multiplication Test
 * Tests if CGBN can multiply two 1024-bit primes correctly
 */

#include <cuda_runtime.h>
#include <gmp.h>
#include <cgbn/cgbn.h>
#include <cstdio>
#include <cstdint>
#include <fstream>

// CGBN params - same as vanity search
template<uint32_t tpi, uint32_t bits>
class cgbn_params_t {
public:
    static const uint32_t TPB = 0;
    static const uint32_t MAX_ROTATION = 4;
    static const uint32_t SHM_LIMIT = 0;
    static const bool CONSTANT_TIME = false;
    static const uint32_t TPI = tpi;
    static const uint32_t BITS = bits;
};

typedef cgbn_params_t<32, 1024> mul_params_t;  // TPI=32 for 1024-bit (CGBN requires 4/8/16/32, not 1!)
typedef cgbn_context_t<mul_params_t::TPI, mul_params_t> mul_context_t;
typedef cgbn_env_t<mul_context_t, 1024> mul_env_t;

__global__ void test_cgbn_multiply(uint32_t *p, uint32_t *q, uint32_t *result, int *status) {
    cgbn_error_report_t *report = nullptr;
    mul_context_t ctx(cgbn_no_checks, report, 0);
    mul_env_t env(ctx);

    typename mul_env_t::cgbn_t p_bn, q_bn;
    typename mul_env_t::cgbn_wide_t result_wide;

    // Use __shared__ memory so all 32 threads can see the result
    __shared__ cgbn_mem_t<1024> p_mem, q_mem, result_low_mem, result_high_mem;

    // All threads load data
    for (uint32_t i = 0; i < 32; i++) {
        p_mem._limbs[i] = p[i];
        q_mem._limbs[i] = q[i];
    }

    // Load into CGBN
    cgbn_load(env, p_bn, &p_mem);
    cgbn_load(env, q_bn, &q_mem);

    // Multiply
    cgbn_mul_wide(env, result_wide, p_bn, q_bn);

    // Store result
    cgbn_store(env, &result_low_mem, result_wide._low);
    cgbn_store(env, &result_high_mem, result_wide._high);

    // Only thread 0 of the TPI=32 group should copy back and print
    if (threadIdx.x == 0) {
        printf("STORE DONE\n");
        printf("result_low_mem._limbs[31]=%08x\n", result_low_mem._limbs[31]);
        printf("result_high_mem._limbs[0]=%08x\n", result_high_mem._limbs[0]);

        // Copy back
        for (uint32_t i = 0; i < 32; i++) {
            result[i] = result_low_mem._limbs[i];
            result[32 + i] = result_high_mem._limbs[i];
        }

        printf("KERNEL END: result[0]=%08x result[63]=%08x\n", result[0], result[63]);
        *status = 1;  // Success
    }
}

// Schoolbook multiplication for comparison
void schoolbook_mul(const uint32_t *p, const uint32_t *q, uint32_t *n) {
    // Zero init
    for (int i = 0; i < 64; i++) n[i] = 0;

    // Multiply
    for (int i = 0; i < 32; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 32; j++) {
            int k = i + j;
            uint64_t product = (uint64_t)p[i] * (uint64_t)q[j];
            uint64_t sum = (uint64_t)n[k] + product + carry;
            n[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
        for (int k = i + 32; carry && k < 64; k++) {
            uint64_t sum = (uint64_t)n[k] + carry;
            n[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
}

int main() {
    printf("=== CGBN Multiplication Test ===\n\n");

    // Load first two primes from pool
    std::ifstream pool("test_pool_10k.bin", std::ios::binary);
    if (!pool) {
        printf("ERROR: Can't open test_pool_10k.bin\n");
        return 1;
    }

    // Skip header
    pool.seekg(16);

    // Read first two primes (128 bytes each, big-endian)
    uint8_t p_bytes[128], q_bytes[128];
    pool.read((char*)p_bytes, 128);
    pool.read((char*)q_bytes, 128);
    pool.close();

    // Convert to little-endian words (same as kernel does)
    uint32_t h_p[32], h_q[32];
    for (int w = 0; w < 32; w++) {
        int byte_idx = (31 - w) * 4;
        h_p[w] = (p_bytes[byte_idx] << 24) |
                 (p_bytes[byte_idx + 1] << 16) |
                 (p_bytes[byte_idx + 2] << 8) |
                 p_bytes[byte_idx + 3];
        h_q[w] = (q_bytes[byte_idx] << 24) |
                 (q_bytes[byte_idx + 1] << 16) |
                 (q_bytes[byte_idx + 2] << 8) |
                 q_bytes[byte_idx + 3];
    }

    printf("Prime p[0]=%08x p[1]=%08x ... p[31]=%08x\n", h_p[0], h_p[1], h_p[31]);
    printf("Prime q[0]=%08x q[1]=%08x ... q[31]=%08x\n\n", h_q[0], h_q[1], h_q[31]);

    // Compute GMP result (ground truth)
    mpz_t gmp_p, gmp_q, gmp_result;
    mpz_init(gmp_p);
    mpz_init(gmp_q);
    mpz_init(gmp_result);

    mpz_import(gmp_p, 32, -1, sizeof(uint32_t), 0, 0, h_p);  // little-endian
    mpz_import(gmp_q, 32, -1, sizeof(uint32_t), 0, 0, h_q);
    mpz_mul(gmp_result, gmp_p, gmp_q);

    uint32_t gmp_result_words[64] = {0};
    size_t count;
    mpz_export(gmp_result_words, &count, -1, sizeof(uint32_t), 0, 0, gmp_result);

    printf("GMP result[0]=%08x result[32]=%08x result[63]=%08x\n",
           gmp_result_words[0], gmp_result_words[32], gmp_result_words[63]);

    // Compute schoolbook result for comparison
    uint32_t schoolbook_result[64];
    schoolbook_mul(h_p, h_q, schoolbook_result);
    printf("Schoolbook result[0]=%08x result[32]=%08x result[63]=%08x\n\n",
           schoolbook_result[0], schoolbook_result[32], schoolbook_result[63]);

    // Allocate GPU memory
    uint32_t *d_p, *d_q, *d_result;
    int *d_status;
    cudaMalloc(&d_p, 32 * sizeof(uint32_t));
    cudaMalloc(&d_q, 32 * sizeof(uint32_t));
    cudaMalloc(&d_result, 64 * sizeof(uint32_t));
    cudaMalloc(&d_status, sizeof(int));

    // Copy to GPU
    cudaMemcpy(d_p, h_p, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, h_q, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    int status = 0;
    cudaMemcpy(d_status, &status, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 32 threads (TPI=32 means 32 threads cooperate on one instance)
    printf("Launching CGBN kernel...\n");
    test_cgbn_multiply<<<1, 32>>>(d_p, d_q, d_result, d_status);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KERNEL LAUNCH ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("KERNEL EXECUTION ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\n");

    // Get results
    uint32_t cgbn_result[64];
    cudaMemcpy(cgbn_result, d_result, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost);

    printf("CGBN result[0]=%08x result[63]=%08x\n", cgbn_result[0], cgbn_result[63]);
    printf("Status: %d\n\n", status);

    // Print boundary words
    printf("\nBoundary between _low and _high (around word 32):\n");
    for (int i = 28; i < 36 && i < 64; i++) {
        printf("  word[%2d]: schoolbook=%08x cgbn=%08x %s\n",
               i, schoolbook_result[i], cgbn_result[i],
               (schoolbook_result[i] != cgbn_result[i]) ? "← DIFF" : "");
    }
    printf("\n");

    // Compare all three
    printf("Comparing CGBN vs GMP (ground truth):\n");
    bool cgbn_vs_gmp_match = true;
    for (int i = 0; i < 64; i++) {
        if (gmp_result_words[i] != cgbn_result[i]) {
            printf("  word[%2d]: gmp=%08x cgbn=%08x ← DIFF\n",
                   i, gmp_result_words[i], cgbn_result[i]);
            cgbn_vs_gmp_match = false;
        }
    }

    printf("\nComparing Schoolbook vs GMP (ground truth):\n");
    bool schoolbook_vs_gmp_match = true;
    for (int i = 0; i < 64; i++) {
        if (gmp_result_words[i] != schoolbook_result[i]) {
            printf("  word[%2d]: gmp=%08x schoolbook=%08x ← DIFF\n",
                   i, gmp_result_words[i], schoolbook_result[i]);
            schoolbook_vs_gmp_match = false;
        }
    }

    printf("\n");
    if (cgbn_vs_gmp_match) {
        printf("✅ CGBN CORRECT (matches GMP)\n");
    } else {
        printf("❌ CGBN WRONG (differs from GMP)\n");
    }

    if (schoolbook_vs_gmp_match) {
        printf("✅ Schoolbook CORRECT (matches GMP)\n");
    } else {
        printf("❌ Schoolbook WRONG (differs from GMP)\n");
    }

    mpz_clear(gmp_p);
    mpz_clear(gmp_q);
    mpz_clear(gmp_result);

    bool match = cgbn_vs_gmp_match;

    // Cleanup
    cudaFree(d_p);
    cudaFree(d_q);
    cudaFree(d_result);
    cudaFree(d_status);

    return match ? 0 : 1;
}
