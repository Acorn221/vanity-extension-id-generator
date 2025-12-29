/**
 * Benchmark: CGBN vs Schoolbook for Vanity Search
 *
 * Tests actual throughput of both methods in realistic conditions:
 * - Load two primes from pool
 * - Multiply them
 * - Measure pairs/sec
 */

#include <cuda_runtime.h>
#include <gmp.h>
#include <cgbn/cgbn.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <chrono>

// =============================================================================
// Schoolbook Kernel (current approach - 1 thread per pair)
// =============================================================================

constexpr uint32_t PRIME_WORDS = 32;
constexpr uint32_t MODULUS_WORDS = 64;

__global__ void schoolbook_multiply_kernel(
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ results,
    uint32_t num_pairs
) {
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) return;

    // Load primes (simulate loading from pool)
    const uint32_t* p = primes;
    const uint32_t* q = primes + PRIME_WORDS;
    uint32_t* n = results + pair_idx * MODULUS_WORDS;

    // Zero init
    for (uint32_t i = 0; i < MODULUS_WORDS; i++) {
        n[i] = 0;
    }

    // Schoolbook multiplication
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        uint64_t carry = 0;
        for (uint32_t j = 0; j < PRIME_WORDS; j++) {
            uint32_t k = i + j;
            uint64_t product = (uint64_t)p[i] * (uint64_t)q[j];
            uint64_t sum = (uint64_t)n[k] + product + carry;
            n[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
        // Propagate carry
        for (uint32_t k = i + PRIME_WORDS; carry && k < MODULUS_WORDS; k++) {
            uint64_t sum = (uint64_t)n[k] + carry;
            n[k] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
}

// =============================================================================
// CGBN Kernel (TPI=32, 32 threads per pair)
// =============================================================================

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

typedef cgbn_params_t<32, 1024> mul_params_t;
typedef cgbn_context_t<mul_params_t::TPI, mul_params_t> mul_context_t;
typedef cgbn_env_t<mul_context_t, 1024> mul_env_t;

__global__ void cgbn_multiply_kernel(
    const uint32_t* __restrict__ primes,
    uint32_t* __restrict__ results,
    uint32_t num_pairs
) {
    // Each group of 32 threads handles one pair
    uint32_t pair_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (pair_idx >= num_pairs) return;

    cgbn_error_report_t *report = nullptr;
    mul_context_t ctx(cgbn_no_checks, report, pair_idx);
    mul_env_t env(ctx);

    typename mul_env_t::cgbn_t p_bn, q_bn;
    typename mul_env_t::cgbn_wide_t result_wide;

    // Shared memory for cooperative access
    __shared__ cgbn_mem_t<1024> p_mem[8];  // 8 instances per block (256 threads / 32 TPI)
    __shared__ cgbn_mem_t<1024> q_mem[8];
    __shared__ cgbn_mem_t<1024> result_low_mem[8];
    __shared__ cgbn_mem_t<1024> result_high_mem[8];

    uint32_t local_idx = (threadIdx.x / 32);  // Which instance within the block

    // Load primes (all threads participate)
    const uint32_t* p = primes;
    const uint32_t* q = primes + PRIME_WORDS;
    for (uint32_t i = 0; i < 32; i++) {
        p_mem[local_idx]._limbs[i] = p[i];
        q_mem[local_idx]._limbs[i] = q[i];
    }

    // Load into CGBN
    cgbn_load(env, p_bn, &p_mem[local_idx]);
    cgbn_load(env, q_bn, &q_mem[local_idx]);

    // Multiply
    cgbn_mul_wide(env, result_wide, p_bn, q_bn);

    // Store result
    cgbn_store(env, &result_low_mem[local_idx], result_wide._low);
    cgbn_store(env, &result_high_mem[local_idx], result_wide._high);

    // Copy back (only thread 0 of each group)
    if (threadIdx.x % 32 == 0) {
        uint32_t* n = results + pair_idx * MODULUS_WORDS;
        for (uint32_t i = 0; i < 32; i++) {
            n[i] = result_low_mem[local_idx]._limbs[i];
            n[32 + i] = result_high_mem[local_idx]._limbs[i];
        }
    }
}

// =============================================================================
// Benchmark Runner
// =============================================================================

void benchmark_schoolbook(uint32_t* d_primes, uint32_t* d_results, uint32_t num_pairs) {
    const int THREADS = 256;
    const int BLOCKS = (num_pairs + THREADS - 1) / THREADS;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    schoolbook_multiply_kernel<<<BLOCKS, THREADS>>>(d_primes, d_results, num_pairs);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        schoolbook_multiply_kernel<<<BLOCKS, THREADS>>>(d_primes, d_results, num_pairs);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double total_pairs = num_pairs * 100.0;
    double seconds = milliseconds / 1000.0;
    double pairs_per_sec = total_pairs / seconds;

    printf("Schoolbook (1 thread/pair):\n");
    printf("  Threads: %d\n", num_pairs);
    printf("  Time: %.2f ms (100 iterations)\n", milliseconds);
    printf("  Throughput: %.2f M pairs/sec\n\n", pairs_per_sec / 1e6);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmark_cgbn(uint32_t* d_primes, uint32_t* d_results, uint32_t num_pairs) {
    const int THREADS = 256;  // Must be multiple of 32
    const int INSTANCES_PER_BLOCK = THREADS / 32;  // 8 instances per block
    const int BLOCKS = (num_pairs + INSTANCES_PER_BLOCK - 1) / INSTANCES_PER_BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cgbn_multiply_kernel<<<BLOCKS, THREADS>>>(d_primes, d_results, num_pairs);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        cgbn_multiply_kernel<<<BLOCKS, THREADS>>>(d_primes, d_results, num_pairs);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double total_pairs = num_pairs * 100.0;
    double seconds = milliseconds / 1000.0;
    double pairs_per_sec = total_pairs / seconds;

    printf("CGBN (32 threads/pair):\n");
    printf("  Threads: %d (for %d pairs)\n", BLOCKS * THREADS, num_pairs);
    printf("  Time: %.2f ms (100 iterations)\n", milliseconds);
    printf("  Throughput: %.2f M pairs/sec\n\n", pairs_per_sec / 1e6);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== CGBN vs Schoolbook Benchmark ===\n\n");

    // Load test primes from pool
    std::ifstream pool("test_pool_10k.bin", std::ios::binary);
    if (!pool) {
        printf("ERROR: Can't open test_pool_10k.bin\n");
        return 1;
    }

    pool.seekg(16);  // Skip header

    uint8_t p_bytes[128], q_bytes[128];
    pool.read((char*)p_bytes, 128);
    pool.read((char*)q_bytes, 128);
    pool.close();

    // Convert to little-endian words
    uint32_t h_primes[64];  // p and q
    for (int w = 0; w < 32; w++) {
        int byte_idx = (31 - w) * 4;
        h_primes[w] = (p_bytes[byte_idx] << 24) |
                      (p_bytes[byte_idx + 1] << 16) |
                      (p_bytes[byte_idx + 2] << 8) |
                      p_bytes[byte_idx + 3];
        h_primes[32 + w] = (q_bytes[byte_idx] << 24) |
                           (q_bytes[byte_idx + 1] << 16) |
                           (q_bytes[byte_idx + 2] << 8) |
                           q_bytes[byte_idx + 3];
    }

    // Test different scales
    uint32_t test_sizes[] = {1024, 4096, 16384, 65536, 262144};  // Up to 256K pairs

    for (uint32_t num_pairs : test_sizes) {
        printf("========================================\n");
        printf("Testing with %u pairs:\n", num_pairs);
        printf("========================================\n\n");

        // Allocate GPU memory
        uint32_t *d_primes, *d_results;
        cudaMalloc(&d_primes, 64 * sizeof(uint32_t));
        cudaMalloc(&d_results, num_pairs * MODULUS_WORDS * sizeof(uint32_t));

        cudaMemcpy(d_primes, h_primes, 64 * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Benchmark both
        benchmark_schoolbook(d_primes, d_results, num_pairs);
        benchmark_cgbn(d_primes, d_results, num_pairs);

        // Verify results match (sample check)
        uint32_t schoolbook_result[64], cgbn_result[64];

        // Run once more to get results
        schoolbook_multiply_kernel<<<1, 1>>>(d_primes, d_results, 1);
        cudaMemcpy(schoolbook_result, d_results, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        cgbn_multiply_kernel<<<1, 32>>>(d_primes, d_results, 1);
        cudaMemcpy(cgbn_result, d_results, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        bool match = true;
        for (int i = 0; i < 64; i++) {
            if (schoolbook_result[i] != cgbn_result[i]) {
                match = false;
                break;
            }
        }

        printf("Results match: %s\n\n", match ? "✅ YES" : "❌ NO");

        cudaFree(d_primes);
        cudaFree(d_results);
    }

    return 0;
}
