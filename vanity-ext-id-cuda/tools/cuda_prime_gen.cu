/**
 * GPU-Accelerated Prime Generator using CUDA
 *
 * Generates 1024-bit primes on GPU using:
 * - Parallel random number generation
 * - Small prime sieving
 * - Miller-Rabin primality testing on GPU
 *
 * This is MUCH faster than CPU-only generation for large batches.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <unistd.h>

// Prime generation parameters
constexpr uint32_t PRIME_WORDS = 32;      // 1024 bits = 32 × 32-bit words
constexpr uint32_t PRIME_BYTES = 128;     // 1024 bits = 128 bytes
constexpr uint32_t MILLER_RABIN_ROUNDS = 20;  // Security parameter

// Small primes for quick filtering (first 2048 primes)
__constant__ uint16_t SMALL_PRIMES[256] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
    313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509,
    521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617,
    619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727,
    733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829,
    839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
    953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051,
    1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171,
    1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289,
    1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427,
    1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523,
    1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621
};

constexpr uint32_t NUM_SMALL_PRIMES = 256;

struct PrimePoolHeader {
    uint32_t magic;         // 'PRMP' = 0x504D5250
    uint32_t version;       // 1
    uint32_t count;         // Number of primes
    uint32_t prime_bytes;   // 128
};

// =============================================================================
// Big Integer Arithmetic (1024-bit)
// =============================================================================

// Compare: returns 1 if a > b, -1 if a < b, 0 if equal
__device__ int bigint_cmp(const uint32_t* a, const uint32_t* b) {
    for (int i = PRIME_WORDS - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

// Add: c = a + b, returns carry
__device__ uint32_t bigint_add(const uint32_t* a, const uint32_t* b, uint32_t* c) {
    uint64_t carry = 0;
    #pragma unroll 8
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        c[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    return (uint32_t)carry;
}

// Subtract: c = a - b, returns borrow
__device__ uint32_t bigint_sub(const uint32_t* a, const uint32_t* b, uint32_t* c) {
    uint64_t borrow = 0;
    #pragma unroll 8
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        uint64_t diff = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
        c[i] = (uint32_t)diff;
        borrow = (diff >> 32) & 1;
    }
    return (uint32_t)borrow;
}

// Modular multiplication: result = (a * b) % m
__device__ void bigint_mul_mod(const uint32_t* a, const uint32_t* b, const uint32_t* m, uint32_t* result) {
    // For simplicity, using repeated addition (Montgomery multiplication would be faster)
    // This is a simplified version - a full implementation would use Barrett or Montgomery reduction

    // Zero initialize
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        result[i] = 0;
    }

    // Multiply using schoolbook algorithm with modular reduction
    // Note: This is not the most efficient, but works for demonstration
    // A production version would use Montgomery multiplication

    uint32_t temp_a[PRIME_WORDS];
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        temp_a[i] = a[i];
    }

    for (uint32_t bit = 0; bit < PRIME_WORDS * 32; bit++) {
        // If bit i of b is set, add temp_a to result (mod m)
        uint32_t word_idx = bit / 32;
        uint32_t bit_idx = bit % 32;

        if (b[word_idx] & (1u << bit_idx)) {
            uint32_t temp_result[PRIME_WORDS];
            bigint_add(result, temp_a, temp_result);

            // Reduce mod m
            while (bigint_cmp(temp_result, m) >= 0) {
                bigint_sub(temp_result, m, temp_result);
            }

            for (uint32_t j = 0; j < PRIME_WORDS; j++) {
                result[j] = temp_result[j];
            }
        }

        // Double temp_a (mod m)
        uint32_t doubled[PRIME_WORDS];
        bigint_add(temp_a, temp_a, doubled);
        while (bigint_cmp(doubled, m) >= 0) {
            bigint_sub(doubled, m, doubled);
        }
        for (uint32_t j = 0; j < PRIME_WORDS; j++) {
            temp_a[j] = doubled[j];
        }
    }
}

// Modular exponentiation: result = base^exp % mod
__device__ void bigint_pow_mod(const uint32_t* base, const uint32_t* exp, const uint32_t* mod, uint32_t* result) {
    // Initialize result to 1
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        result[i] = 0;
    }
    result[0] = 1;

    uint32_t base_copy[PRIME_WORDS];
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        base_copy[i] = base[i];
    }

    // Square-and-multiply algorithm
    for (uint32_t bit = 0; bit < PRIME_WORDS * 32; bit++) {
        uint32_t word_idx = bit / 32;
        uint32_t bit_idx = bit % 32;

        if (exp[word_idx] & (1u << bit_idx)) {
            uint32_t temp[PRIME_WORDS];
            bigint_mul_mod(result, base_copy, mod, temp);
            for (uint32_t j = 0; j < PRIME_WORDS; j++) {
                result[j] = temp[j];
            }
        }

        // Square base_copy
        uint32_t squared[PRIME_WORDS];
        bigint_mul_mod(base_copy, base_copy, mod, squared);
        for (uint32_t j = 0; j < PRIME_WORDS; j++) {
            base_copy[j] = squared[j];
        }
    }
}

// Check if number is divisible by small prime
__device__ bool is_divisible_by_small_primes(const uint32_t* n) {
    // Simple trial division by small primes
    // Note: This is simplified - a full implementation would use proper modular arithmetic
    for (uint32_t i = 0; i < NUM_SMALL_PRIMES; i++) {
        uint16_t p = SMALL_PRIMES[i];

        // Compute n % p
        uint32_t remainder = 0;
        for (int j = PRIME_WORDS - 1; j >= 0; j--) {
            uint64_t temp = ((uint64_t)remainder << 32) | n[j];
            remainder = temp % p;
        }

        if (remainder == 0) return true;
    }
    return false;
}

// =============================================================================
// Miller-Rabin Primality Test
// =============================================================================

__device__ bool miller_rabin_test(const uint32_t* n, curandState* state, uint32_t rounds) {
    // Check if n is even
    if ((n[0] & 1) == 0) return false;

    // Check against small primes
    if (is_divisible_by_small_primes(n)) return false;

    // Write n-1 as d * 2^r
    uint32_t n_minus_1[PRIME_WORDS];
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        n_minus_1[i] = n[i];
    }
    n_minus_1[0]--;  // n - 1

    // Find r (number of trailing zeros in binary)
    uint32_t r = 0;
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        if (n_minus_1[i] == 0) {
            r += 32;
        } else {
            r += __ffs(n_minus_1[i]) - 1;  // Find first set bit
            break;
        }
    }

    // Compute d = (n-1) / 2^r
    uint32_t d[PRIME_WORDS];
    for (uint32_t i = 0; i < PRIME_WORDS; i++) {
        d[i] = n_minus_1[i];
    }

    // Right shift by r bits
    uint32_t shift_words = r / 32;
    uint32_t shift_bits = r % 32;

    for (uint32_t i = 0; i < PRIME_WORDS - shift_words; i++) {
        d[i] = d[i + shift_words];
    }
    for (uint32_t i = PRIME_WORDS - shift_words; i < PRIME_WORDS; i++) {
        d[i] = 0;
    }

    if (shift_bits > 0) {
        for (uint32_t i = 0; i < PRIME_WORDS - 1; i++) {
            d[i] = (d[i] >> shift_bits) | (d[i + 1] << (32 - shift_bits));
        }
        d[PRIME_WORDS - 1] >>= shift_bits;
    }

    // Perform rounds of testing
    for (uint32_t round = 0; round < rounds; round++) {
        // Pick random a in [2, n-2]
        uint32_t a[PRIME_WORDS];
        for (uint32_t i = 0; i < PRIME_WORDS; i++) {
            a[i] = curand(state);
        }

        // Make sure a < n and a >= 2
        while (bigint_cmp(a, n) >= 0) {
            for (uint32_t i = 0; i < PRIME_WORDS; i++) {
                a[i] >>= 1;
            }
        }
        if (a[0] < 2) a[0] = 2;

        // Compute x = a^d mod n
        uint32_t x[PRIME_WORDS];
        bigint_pow_mod(a, d, n, x);

        // Check if x == 1 or x == n-1
        uint32_t one[PRIME_WORDS] = {1};
        if (bigint_cmp(x, one) == 0 || bigint_cmp(x, n_minus_1) == 0) {
            continue;
        }

        // Repeat r-1 times
        bool composite = true;
        for (uint32_t i = 0; i < r - 1; i++) {
            // x = x^2 mod n
            bigint_mul_mod(x, x, n, x);

            if (bigint_cmp(x, n_minus_1) == 0) {
                composite = false;
                break;
            }
        }

        if (composite) return false;
    }

    return true;  // Probably prime
}

// =============================================================================
// Prime Generation Kernel
// =============================================================================

__global__ void generate_primes_kernel(
    uint8_t* prime_pool,           // Output: array of primes
    uint32_t* prime_count,         // Output: number of primes found
    uint32_t max_primes,           // Maximum number of primes to find
    uint64_t seed                  // Random seed
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize random state
    curandState state;
    curand_init(seed, tid, 0, &state);

    uint32_t candidate[PRIME_WORDS];

    while (atomicAdd(prime_count, 0) < max_primes) {
        // Generate random 1024-bit number
        for (uint32_t i = 0; i < PRIME_WORDS; i++) {
            candidate[i] = curand(&state);
        }

        // Set top bit to ensure 1024 bits
        candidate[PRIME_WORDS - 1] |= 0x80000000;

        // Set bottom bit to ensure odd
        candidate[0] |= 1;

        // Test for primality
        if (miller_rabin_test(candidate, &state, MILLER_RABIN_ROUNDS)) {
            // Found a prime! Add it to the pool
            uint32_t idx = atomicAdd(prime_count, 1);

            if (idx < max_primes) {
                // Convert to big-endian bytes
                uint8_t* output = prime_pool + idx * PRIME_BYTES;

                for (uint32_t i = 0; i < PRIME_WORDS; i++) {
                    uint32_t word_idx = PRIME_WORDS - 1 - i;
                    uint32_t byte_offset = i * 4;
                    uint32_t word = candidate[word_idx];

                    output[byte_offset] = (word >> 24) & 0xFF;
                    output[byte_offset + 1] = (word >> 16) & 0xFF;
                    output[byte_offset + 2] = (word >> 8) & 0xFF;
                    output[byte_offset + 3] = word & 0xFF;
                }
            }
        }
    }
}

// =============================================================================
// Host Code
// =============================================================================

int main(int argc, char* argv[]) {
    size_t target_count = 100000;
    std::string output_file = "prime_pool_gpu.bin";

    if (argc > 1) {
        target_count = std::stoull(argv[1]);
    }
    if (argc > 2) {
        output_file = argv[2];
    }

    std::cout << "GPU Prime Pool Generator (CUDA)\n";
    std::cout << "================================\n";
    std::cout << "Target count: " << target_count << " primes\n";
    std::cout << "Prime size:   1024 bits (" << PRIME_BYTES << " bytes)\n";
    std::cout << "Output file:  " << output_file << "\n";
    std::cout << "Output size:  ~" << (target_count * PRIME_BYTES / 1024 / 1024) << " MB\n";
    std::cout << "\n";

    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n\n";

    // Allocate device memory
    uint8_t* d_prime_pool;
    uint32_t* d_prime_count;

    size_t pool_size = target_count * PRIME_BYTES;
    cudaMalloc(&d_prime_pool, pool_size);
    cudaMalloc(&d_prime_count, sizeof(uint32_t));
    cudaMemset(d_prime_count, 0, sizeof(uint32_t));

    // Launch kernel with many threads
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = prop.multiProcessorCount * 8;  // 8 blocks per SM

    std::cout << "Launching kernel with " << num_blocks << " blocks × "
              << threads_per_block << " threads = "
              << (num_blocks * threads_per_block) << " total threads\n\n";

    auto start_time = std::chrono::steady_clock::now();

    // Generate random seed
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();

    generate_primes_kernel<<<num_blocks, threads_per_block>>>(
        d_prime_pool,
        d_prime_count,
        target_count,
        seed
    );

    // Monitor progress
    uint32_t h_count = 0;
    while (h_count < target_count) {
        cudaMemcpy(&h_count, d_prime_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double rate = (elapsed > 0) ? h_count / elapsed : 0;
        double progress = 100.0 * h_count / target_count;
        double eta = (rate > 0) ? (target_count - h_count) / rate : 0;

        std::cout << "\rProgress: " << std::setw(6) << h_count << " / " << target_count
                  << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                  << " | Rate: " << std::setw(4) << static_cast<int>(rate) << "/s"
                  << " | ETA: " << std::setw(4) << static_cast<int>(eta) << "s"
                  << "     " << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    cudaDeviceSynchronize();

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "\n\nGeneration complete!\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_time << "s\n";
    std::cout << "Average rate: " << static_cast<int>(target_count / total_time) << " primes/s\n\n";

    // Copy results to host
    std::vector<uint8_t> h_prime_pool(pool_size);
    cudaMemcpy(h_prime_pool.data(), d_prime_pool, pool_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_prime_pool);
    cudaFree(d_prime_count);

    // Write to file
    std::cout << "Writing to " << output_file << "...\n";

    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open output file\n";
        return 1;
    }

    // Write header
    PrimePoolHeader header;
    header.magic = 0x504D5250;  // 'PRMP'
    header.version = 1;
    header.count = static_cast<uint32_t>(target_count);
    header.prime_bytes = PRIME_BYTES;

    out.write(reinterpret_cast<char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(h_prime_pool.data()), pool_size);
    out.close();

    std::cout << "Done! Wrote " << target_count << " primes to " << output_file << "\n";
    std::cout << "File size: " << (sizeof(header) + pool_size) << " bytes\n";

    return 0;
}
