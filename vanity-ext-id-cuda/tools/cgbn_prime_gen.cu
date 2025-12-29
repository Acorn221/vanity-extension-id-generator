/**
 * GPU Prime Pool Generator using NVIDIA CGBN
 *
 * Based on CGBN sample_04_miller_rabin
 * Generates 1024-bit primes using GPU-accelerated Miller-Rabin testing
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <gmp.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <atomic>
#include "../CGBN/include/cgbn/cgbn.h"

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#define CGBN_CHECK(report) \
  do { \
    if(cgbn_error_report_check(report)) { \
      fprintf(stderr, "CGBN error at %s:%d\n", __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// Prime pool header
struct PrimePoolHeader {
    uint32_t magic;         // 'PRMP' = 0x504D5250
    uint32_t version;       // 1
    uint32_t count;         // Number of primes
    uint32_t prime_bytes;   // 128
};

constexpr uint32_t PRIME_BYTES = 128;

// Miller-Rabin parameters
template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class mr_params_t {
  public:
  static const uint32_t TPB=0;
  static const uint32_t MAX_ROTATION=4;
  static const uint32_t SHM_LIMIT=0;
  static const bool CONSTANT_TIME=false;
  static const uint32_t TPI=tpi;
  static const uint32_t BITS=bits;
  static const uint32_t WINDOW_BITS=window_bits;
};

template<class params>
class miller_rabin_t {
  public:
  static const uint32_t window_bits=params::WINDOW_BITS;

  typedef struct {
    cgbn_mem_t<params::BITS> candidate;
    uint32_t passed;
    uint32_t is_prime;  // 1 if prime, 0 if composite
  } instance_t;

  typedef cgbn_context_t<params::TPI, params> context_t;
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;
  typedef typename env_t::cgbn_local_t bn_local_t;
  typedef typename env_t::cgbn_wide_t bn_wide_t;

  context_t _context;
  env_t _env;
  int32_t _instance;

  __device__ __forceinline__ miller_rabin_t(cgbn_monitor_t monitor,
    cgbn_error_report_t *report, int32_t instance)
    : _context(monitor, report, (uint32_t)instance), _env(_context),
      _instance(instance) { }

  __device__ __forceinline__ void powm(bn_t &x, const bn_t &power,
    const bn_t &modulus) {
    bn_t t;
    bn_local_t window[1<<window_bits];
    int32_t index, position, offset;
    uint32_t np0;

    cgbn_negate(_env, t, modulus);
    cgbn_store(_env, window+0, t);

    np0=cgbn_bn2mont(_env, x, x, modulus);
    cgbn_store(_env, window+1, x);
    cgbn_set(_env, t, x);

    #pragma nounroll
    for(index=2;index<(1<<window_bits);index++) {
      cgbn_mont_mul(_env, x, x, t, modulus, np0);
      cgbn_store(_env, window+index, x);
    }

    position=params::BITS - cgbn_clz(_env, power);

    offset=position % window_bits;
    if(offset==0)
      position=position-window_bits;
    else
      position=position-offset;
    index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
    cgbn_load(_env, x, window+index);

    while(position>0) {
      #pragma nounroll
      for(int sqr_count=0;sqr_count<window_bits;sqr_count++)
        cgbn_mont_sqr(_env, x, x, modulus, np0);

      position=position-window_bits;
      index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
      cgbn_load(_env, t, window+index);
      cgbn_mont_mul(_env, x, x, t, modulus, np0);
    }

    cgbn_mont2bn(_env, x, x, modulus, np0);
  }

  __device__ __forceinline__ uint32_t miller_rabin(const bn_t &candidate,
    uint32_t *primes, uint32_t prime_count) {
    int k, trailing, count;
    bn_t x, power, minus_one;
    bn_wide_t w;

    cgbn_sub_ui32(_env, power, candidate, 1);
    trailing=cgbn_ctz(_env, power);
    cgbn_rotate_right(_env, power, power, trailing);

    for(k=0;k<prime_count;k++) {
      cgbn_set_ui32(_env, x, primes[k]);
      powm(x, power, candidate);
      cgbn_sub_ui32(_env, minus_one, candidate, 1);
      if(!cgbn_equals_ui32(_env, x, 1) && !cgbn_equals(_env, x, minus_one)) {
        if(trailing==1)
          return k;

        count=trailing;
        while(true) {
          cgbn_sqr_wide(_env, w, x);
          cgbn_rem_wide(_env, x, w, candidate);
          if(cgbn_equals(_env, x, minus_one))
            break;
          if(--count==0 || cgbn_equals_ui32(_env, x, 1))
            return k;
        }
      }
    }
    return prime_count;
  }
};

// Kernel to generate random candidates
__global__ void generate_candidates_kernel(
    typename miller_rabin_t<mr_params_t<8, 1024, 4>>::instance_t *instances,
    uint32_t instance_count,
    uint64_t seed,
    uint32_t offset) {

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= instance_count) return;

  // Initialize curand
  curandState state;
  curand_init(seed, idx + offset, 0, &state);

  // Generate random 1024-bit number
  for(int i = 0; i < 32; i++) {
    instances[idx].candidate._limbs[i] = curand(&state);
  }

  // Ensure it's odd and 1024 bits
  instances[idx].candidate._limbs[0] |= 1;           // Make odd
  instances[idx].candidate._limbs[31] |= 0x80000000; // Set top bit
  instances[idx].passed = 0;
  instances[idx].is_prime = 0;
}

// Miller-Rabin testing kernel
template<class params>
__global__ void kernel_miller_rabin(cgbn_error_report_t *report,
  typename miller_rabin_t<params>::instance_t *instances,
  uint32_t instance_count, uint32_t *primes, uint32_t prime_count) {

  int32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance>=instance_count)
    return;

  typedef miller_rabin_t<params> local_mr_t;

  local_mr_t mr(cgbn_report_monitor, report, instance);
  typename local_mr_t::bn_t candidate;
  uint32_t passed;

  cgbn_load(mr._env, candidate, &(instances[instance].candidate));

  passed=mr.miller_rabin(candidate, primes, prime_count);

  instances[instance].passed=passed;
  instances[instance].is_prime = (passed == prime_count) ? 1 : 0;
}

// Generate small primes for testing
uint32_t *generate_small_primes(uint32_t count) {
  uint32_t *list=(uint32_t *)malloc(sizeof(uint32_t)*count);
  int test, current, index;

  list[0]=2;
  current=3;
  index=1;
  while(index<count) {
    for(test=1;test<index;test++)
      if(current%list[test]==0)
        break;
    if(test==index)
      list[index++]=current;
    current=current+2;
  }
  return list;
}

// Convert CGBN number to bytes
void cgbn_to_bytes(uint32_t *limbs, uint8_t *bytes) {
  for(int i = 0; i < 32; i++) {
    uint32_t word_idx = 31 - i;
    uint32_t byte_offset = i * 4;
    uint32_t word = limbs[word_idx];

    bytes[byte_offset] = (word >> 24) & 0xFF;
    bytes[byte_offset + 1] = (word >> 16) & 0xFF;
    bytes[byte_offset + 2] = (word >> 8) & 0xFF;
    bytes[byte_offset + 3] = word & 0xFF;
  }
}

int main(int argc, char* argv[]) {
  size_t target_count = 100000;
  std::string output_file = "prime_pool_cgbn.bin";

  if (argc > 1) {
    target_count = std::stoull(argv[1]);
  }
  if (argc > 2) {
    output_file = argv[2];
  }

  std::cout << "CGBN GPU Prime Pool Generator\n";
  std::cout << "==============================\n";
  std::cout << "Target count: " << target_count << " primes\n";
  std::cout << "Prime size:   1024 bits (" << PRIME_BYTES << " bytes)\n";
  std::cout << "Output file:  " << output_file << "\n";
  std::cout << "Output size:  ~" << (target_count * PRIME_BYTES / 1024 / 1024) << " MB\n";
  std::cout << "\n";

  // Check GPU
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No CUDA devices found!\n";
    return 1;
  }

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Using GPU: " << prop.name << "\n";
  std::cout << "Compute: " << prop.major << "." << prop.minor << "\n";
  std::cout << "SMs: " << prop.multiProcessorCount << "\n\n";

  // Generate small primes for Miller-Rabin
  const uint32_t PRIME_COUNT = 20;
  uint32_t *h_primes = generate_small_primes(PRIME_COUNT);

  uint32_t *d_primes;
  CUDA_CHECK(cudaMalloc(&d_primes, sizeof(uint32_t) * PRIME_COUNT));
  CUDA_CHECK(cudaMemcpy(d_primes, h_primes, sizeof(uint32_t) * PRIME_COUNT,
    cudaMemcpyHostToDevice));

  // Allocate CGBN error reporting
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  typedef mr_params_t<8, 1024, 4> params;  // OPTIMIZED: TPI=8, WINDOW=4 (1089 primes/s)
  typedef miller_rabin_t<params>::instance_t instance_t;

  const int TPI = 8;   // Threads per instance (OPTIMIZED from grid search)
  const int TPB = 128; // Threads per block
  const int IPB = TPB / TPI;  // Instances per block (= 16)
  const uint32_t BATCH_SIZE = 50000;  // Candidates per batch (OPTIMIZED)

  std::vector<uint8_t> prime_pool;
  prime_pool.reserve(target_count * PRIME_BYTES);

  uint64_t candidates_tested = 0;
  uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();

  auto start_time = std::chrono::steady_clock::now();

  std::cout << "Generating primes (batch size: " << BATCH_SIZE << ")...\n\n";

  while(prime_pool.size() < target_count * PRIME_BYTES) {
    // Allocate batch
    instance_t *h_instances = (instance_t *)malloc(sizeof(instance_t) * BATCH_SIZE);
    instance_t *d_instances;
    CUDA_CHECK(cudaMalloc(&d_instances, sizeof(instance_t) * BATCH_SIZE));

    // Generate random candidates
    int gen_blocks = (BATCH_SIZE + 255) / 256;
    generate_candidates_kernel<<<gen_blocks, 256>>>(d_instances, BATCH_SIZE,
      seed, candidates_tested);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Test with Miller-Rabin
    kernel_miller_rabin<params><<<(BATCH_SIZE+IPB-1)/IPB, TPB>>>(
      report, d_instances, BATCH_SIZE, d_primes, PRIME_COUNT);
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_instances, d_instances, sizeof(instance_t) * BATCH_SIZE,
      cudaMemcpyDeviceToHost));

    // Extract primes
    uint32_t primes_found = 0;
    for(uint32_t i = 0; i < BATCH_SIZE && prime_pool.size() < target_count * PRIME_BYTES; i++) {
      if(h_instances[i].is_prime) {
        uint8_t bytes[PRIME_BYTES];
        cgbn_to_bytes(h_instances[i].candidate._limbs, bytes);
        prime_pool.insert(prime_pool.end(), bytes, bytes + PRIME_BYTES);
        primes_found++;
      }
    }

    candidates_tested += BATCH_SIZE;

    // Progress update
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    uint32_t primes_so_far = prime_pool.size() / PRIME_BYTES;
    double rate = (elapsed > 0) ? primes_so_far / elapsed : 0;
    double progress = 100.0 * primes_so_far / target_count;
    double eta = (rate > 0) ? (target_count - primes_so_far) / rate : 0;

    printf("\rProgress: %6u / %zu (%5.1f%%) | Rate: %4.0f/s | "
           "Tested: %llu | Found: %u | ETA: %4.0fs     ",
           primes_so_far, target_count, progress, rate,
           (unsigned long long)candidates_tested, primes_found, eta);
    fflush(stdout);

    free(h_instances);
    CUDA_CHECK(cudaFree(d_instances));
  }

  auto end_time = std::chrono::steady_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();

  printf("\n\nGeneration complete!\n");
  printf("Total time: %.1fs\n", total_time);
  printf("Average rate: %.0f primes/s\n", target_count / total_time);
  printf("Candidates tested: %llu\n", (unsigned long long)candidates_tested);
  printf("Success rate: %.2f%%\n", 100.0 * target_count / candidates_tested);

  // Write to file
  std::cout << "\nWriting to " << output_file << "...\n";

  std::ofstream out(output_file, std::ios::binary);
  if (!out) {
    std::cerr << "Error: Could not open output file\n";
    return 1;
  }

  PrimePoolHeader header;
  header.magic = 0x504D5250;
  header.version = 1;
  header.count = target_count;
  header.prime_bytes = PRIME_BYTES;

  out.write(reinterpret_cast<char*>(&header), sizeof(header));
  out.write(reinterpret_cast<const char*>(prime_pool.data()),
    target_count * PRIME_BYTES);
  out.close();

  std::cout << "Done! Wrote " << target_count << " primes\n";
  std::cout << "File size: " << (sizeof(header) + target_count * PRIME_BYTES) << " bytes\n";

  // Cleanup
  free(h_primes);
  CUDA_CHECK(cudaFree(d_primes));
  CUDA_CHECK(cgbn_error_report_free(report));

  return 0;
}
