/**
 * Tunable CGBN GPU Prime Pool Generator
 * For grid-search optimization of parameters
 *
 * Usage: ./generate-prime-pool-cgbn-tune COUNT OUTPUT [BATCH_SIZE] [TPB] [TPI] [WINDOW_BITS] [MR_ROUNDS]
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
#include <signal.h>
#include <csignal>
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

struct PrimePoolHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t prime_bytes;
};

constexpr uint32_t PRIME_BYTES = 128;

// Global state for signal handling
static volatile sig_atomic_t g_interrupted = 0;
static void signal_handler(int signum) {
  g_interrupted = 1;
  fprintf(stderr, "\n\nReceived signal %d, cleaning up...\n", signum);
}

// Checkpoint helper function
bool save_checkpoint(const std::string &filename, const std::vector<uint8_t> &prime_pool,
                     size_t target_count, uint64_t candidates_tested) {
  std::string checkpoint_file = filename + ".checkpoint";
  std::ofstream out(checkpoint_file, std::ios::binary);
  if (!out) {
    fprintf(stderr, "Warning: Failed to write checkpoint\n");
    return false;
  }

  uint32_t current_count = prime_pool.size() / PRIME_BYTES;
  PrimePoolHeader header = {0x504D5250, 1, current_count, PRIME_BYTES};

  out.write(reinterpret_cast<char*>(&header), sizeof(header));
  out.write(reinterpret_cast<const char*>(prime_pool.data()), prime_pool.size());
  out.write(reinterpret_cast<const char*>(&candidates_tested), sizeof(candidates_tested));
  out.close();

  fprintf(stderr, "\nCheckpoint saved: %u primes, %llu candidates tested\n",
          current_count, (unsigned long long)candidates_tested);
  return true;
}

// Load checkpoint if exists
bool load_checkpoint(const std::string &filename, std::vector<uint8_t> &prime_pool,
                     uint64_t &candidates_tested) {
  std::string checkpoint_file = filename + ".checkpoint";
  std::ifstream in(checkpoint_file, std::ios::binary);
  if (!in) return false;

  PrimePoolHeader header;
  in.read(reinterpret_cast<char*>(&header), sizeof(header));

  if (header.magic != 0x504D5250 || header.version != 1 || header.prime_bytes != PRIME_BYTES) {
    fprintf(stderr, "Warning: Invalid checkpoint file, starting fresh\n");
    return false;
  }

  prime_pool.resize(header.count * PRIME_BYTES);
  in.read(reinterpret_cast<char*>(prime_pool.data()), prime_pool.size());
  in.read(reinterpret_cast<char*>(&candidates_tested), sizeof(candidates_tested));
  in.close();

  fprintf(stderr, "Loaded checkpoint: %u primes, %llu candidates tested\n",
          header.count, (unsigned long long)candidates_tested);
  return true;
}

// Template parameters configurable at compile-time
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
    uint32_t is_prime;
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
template<typename instance_t>
__global__ void generate_candidates_kernel_tunable(instance_t *instances, uint32_t instance_count,
                                                     uint64_t seed, uint32_t offset) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= instance_count) return;
  curandState state;
  curand_init(seed, idx + offset, 0, &state);
  for(int i = 0; i < 32; i++) {
    instances[idx].candidate._limbs[i] = curand(&state);
  }
  instances[idx].candidate._limbs[0] |= 1;
  instances[idx].candidate._limbs[31] |= 0x80000000;
  instances[idx].passed = 0;
  instances[idx].is_prime = 0;
}

// Miller-Rabin kernel
template<class params>
__global__ void kernel_miller_rabin_tunable(cgbn_error_report_t *report,
                                             typename miller_rabin_t<params>::instance_t *instances,
                                             uint32_t instance_count, uint32_t *primes,
                                             uint32_t prime_count) {
  int32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=instance_count) return;

  typedef miller_rabin_t<params> local_mr_t;
  local_mr_t mr(cgbn_report_monitor, report, instance);
  typename local_mr_t::bn_t candidate;

  cgbn_load(mr._env, candidate, &(instances[instance].candidate));
  uint32_t passed = mr.miller_rabin(candidate, primes, prime_count);
  instances[instance].passed = passed;
  instances[instance].is_prime = (passed == prime_count) ? 1 : 0;
}

// Runtime-configured kernel launcher helper
template<uint32_t TPI, uint32_t WINDOW_BITS>
void run_with_params(
    size_t target_count,
    std::string output_file,
    uint32_t batch_size,
    uint32_t tpb,
    uint32_t prime_count) {

  typedef mr_params_t<TPI, 1024, WINDOW_BITS> params;
  typedef typename miller_rabin_t<params>::instance_t instance_t;

  // Generate small primes for Miller-Rabin
  std::vector<uint32_t> h_primes;
  h_primes.push_back(2);
  uint32_t current = 3;
  while(h_primes.size() < prime_count) {
    bool is_prime = true;
    for(uint32_t p : h_primes) {
      if(current % p == 0) { is_prime = false; break; }
    }
    if(is_prime) h_primes.push_back(current);
    current += 2;
  }

  uint32_t *d_primes;
  CUDA_CHECK(cudaMalloc(&d_primes, sizeof(uint32_t) * prime_count));
  CUDA_CHECK(cudaMemcpy(d_primes, h_primes.data(), sizeof(uint32_t) * prime_count,
    cudaMemcpyHostToDevice));

  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  const int IPB = tpb / TPI;

  // Setup signal handlers
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  std::vector<uint8_t> prime_pool;
  prime_pool.reserve(target_count * PRIME_BYTES);

  uint64_t candidates_tested = 0;
  uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();

  // Try to load checkpoint
  if (load_checkpoint(output_file, prime_pool, candidates_tested)) {
    printf("Resuming from checkpoint with %zu primes\n",
           prime_pool.size() / PRIME_BYTES);
  }

  auto start_time = std::chrono::steady_clock::now();
  uint32_t batches_since_checkpoint = 0;
  const uint32_t CHECKPOINT_INTERVAL = 10; // Save every 10 batches

  while(prime_pool.size() < target_count * PRIME_BYTES && !g_interrupted) {
    instance_t *h_instances = (instance_t *)malloc(sizeof(instance_t) * batch_size);
    instance_t *d_instances;
    CUDA_CHECK(cudaMalloc(&d_instances, sizeof(instance_t) * batch_size));

    // Generate candidates with timeout check
    auto kernel_start = std::chrono::steady_clock::now();
    int gen_blocks = (batch_size + 255) / 256;
    generate_candidates_kernel_tunable<instance_t><<<gen_blocks, 256>>>(
        d_instances, batch_size, seed, candidates_tested);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    auto kernel_end = std::chrono::steady_clock::now();
    double gen_time = std::chrono::duration<double>(kernel_end - kernel_start).count();
    if (gen_time > 30.0) {
      fprintf(stderr, "\nWarning: Candidate generation took %.1fs (may indicate GPU issue)\n", gen_time);
    }

    // Miller-Rabin test with timeout check
    kernel_start = std::chrono::steady_clock::now();
    kernel_miller_rabin_tunable<params><<<(batch_size+IPB-1)/IPB, tpb>>>(
        report, d_instances, batch_size, d_primes, prime_count);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_end = std::chrono::steady_clock::now();
    double test_time = std::chrono::duration<double>(kernel_end - kernel_start).count();
    if (test_time > 60.0) {
      fprintf(stderr, "\nWarning: Miller-Rabin test took %.1fs (may indicate GPU issue)\n", test_time);
    }
    CGBN_CHECK(report);

    CUDA_CHECK(cudaMemcpy(h_instances, d_instances, sizeof(instance_t) * batch_size,
      cudaMemcpyDeviceToHost));

    uint32_t primes_found = 0;
    for(uint32_t i = 0; i < batch_size && prime_pool.size() < target_count * PRIME_BYTES; i++) {
      if(h_instances[i].is_prime) {
        uint8_t bytes[PRIME_BYTES];
        for(int j = 0; j < 32; j++) {
          uint32_t word_idx = 31 - j;
          uint32_t byte_offset = j * 4;
          uint32_t word = h_instances[i].candidate._limbs[word_idx];
          bytes[byte_offset] = (word >> 24) & 0xFF;
          bytes[byte_offset + 1] = (word >> 16) & 0xFF;
          bytes[byte_offset + 2] = (word >> 8) & 0xFF;
          bytes[byte_offset + 3] = word & 0xFF;
        }
        prime_pool.insert(prime_pool.end(), bytes, bytes + PRIME_BYTES);
        primes_found++;
      }
    }

    candidates_tested += batch_size;

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    uint32_t primes_so_far = prime_pool.size() / PRIME_BYTES;
    double rate = (elapsed > 0) ? primes_so_far / elapsed : 0;

    printf("\rProgress: %6u / %zu | Rate: %4.0f/s | Tested: %llu     ",
           primes_so_far, target_count, rate, (unsigned long long)candidates_tested);
    fflush(stdout);

    free(h_instances);
    CUDA_CHECK(cudaFree(d_instances));

    // Save checkpoint periodically
    batches_since_checkpoint++;
    if (batches_since_checkpoint >= CHECKPOINT_INTERVAL) {
      save_checkpoint(output_file, prime_pool, target_count, candidates_tested);
      batches_since_checkpoint = 0;
    }

    // Check for interruption
    if (g_interrupted) {
      fprintf(stderr, "\n\nInterrupted! Saving checkpoint...\n");
      save_checkpoint(output_file, prime_pool, target_count, candidates_tested);
      break;
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  double total_time = std::chrono::duration<double>(end_time - start_time).count();

  uint32_t final_count = prime_pool.size() / PRIME_BYTES;

  if (g_interrupted) {
    printf("\n\nINTERRUPTED: %.1fs | %u/%zu primes | Tested: %llu\n",
           total_time, final_count, target_count,
           (unsigned long long)candidates_tested);
    printf("Resume by running the same command - checkpoint will be loaded automatically.\n");
  } else {
    printf("\n\nDONE: %.1fs | %.0f primes/s | Tested: %llu | Success: %.3f%%\n",
           total_time, (double)final_count / total_time,
           (unsigned long long)candidates_tested,
           100.0 * final_count / candidates_tested);

    // Only write final file if we completed
    std::ofstream out(output_file, std::ios::binary);
    PrimePoolHeader header = {0x504D5250, 1, final_count, PRIME_BYTES};
    out.write(reinterpret_cast<char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(prime_pool.data()), final_count * PRIME_BYTES);
    out.close();

    // Remove checkpoint file on successful completion
    std::string checkpoint_file = output_file + ".checkpoint";
    remove(checkpoint_file.c_str());
  }

  CUDA_CHECK(cudaFree(d_primes));
  CUDA_CHECK(cgbn_error_report_free(report));
}

// Dispatcher for different TPI/WINDOW combinations
void dispatch_params(uint32_t tpi, uint32_t window_bits,
                     size_t target_count, std::string output_file,
                     uint32_t batch_size, uint32_t tpb, uint32_t prime_count) {

  #define DISPATCH(T, W) \
    if(tpi == T && window_bits == W) { \
      run_with_params<T, W>(target_count, output_file, batch_size, tpb, prime_count); \
      return; \
    }

  // TPI=4 not supported for 1024-bit numbers in CGBN
  DISPATCH(8, 3)   DISPATCH(8, 4)   DISPATCH(8, 5)   DISPATCH(8, 6)
  DISPATCH(16, 3)  DISPATCH(16, 4)  DISPATCH(16, 5)  DISPATCH(16, 6)
  DISPATCH(32, 3)  DISPATCH(32, 4)  DISPATCH(32, 5)  DISPATCH(32, 6)  DISPATCH(32, 7)

  #undef DISPATCH

  fprintf(stderr, "Unsupported TPI=%u WINDOW=%u combination\n", tpi, window_bits);
  exit(1);
}

void print_usage(const char* prog) {
  printf("Usage: %s COUNT OUTPUT [BATCH_SIZE] [TPB] [TPI] [WINDOW_BITS] [MR_ROUNDS]\n\n", prog);
  printf("Arguments:\n");
  printf("  COUNT        Number of primes to generate\n");
  printf("  OUTPUT       Output filename\n");
  printf("  BATCH_SIZE   Candidates per batch (default: 100000)\n");
  printf("  TPB          Threads per block (default: 128, must be multiple of TPI)\n");
  printf("  TPI          Threads per instance (default: 32, options: 8/16/32)\n");
  printf("  WINDOW_BITS  Modexp window size (default: 5, range: 3-7)\n");
  printf("  MR_ROUNDS    Miller-Rabin rounds (default: 20)\n\n");
  printf("Examples:\n");
  printf("  %s 1000 test.bin\n", prog);
  printf("  %s 1000 test.bin 200000 256 16 5 20\n", prog);
  printf("  %s 100000 pool.bin 500000 256 32 6 15\n\n", prog);
  printf("Grid Search Example:\n");
  printf("  for tpi in 8 16 32; do\n");
  printf("    for wb in 4 5 6; do\n");
  printf("      %s 10000 /dev/null 100000 128 $tpi $wb 20\n", prog);
  printf("    done\n");
  printf("  done\n");
  printf("\nCheckpoint Support:\n");
  printf("  - Checkpoints are automatically saved every 10 batches\n");
  printf("  - Resume interrupted runs by running the same command\n");
  printf("  - Checkpoint file: OUTPUT.checkpoint\n");
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  size_t target_count = std::stoull(argv[1]);
  std::string output_file = argv[2];
  uint32_t batch_size = (argc > 3) ? std::stoul(argv[3]) : 100000;
  uint32_t tpb = (argc > 4) ? std::stoul(argv[4]) : 128;
  uint32_t tpi = (argc > 5) ? std::stoul(argv[5]) : 32;
  uint32_t window_bits = (argc > 6) ? std::stoul(argv[6]) : 5;
  uint32_t prime_count = (argc > 7) ? std::stoul(argv[7]) : 20;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("CGBN Tunable Prime Generator\n");
  printf("============================\n");
  printf("GPU:         %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  printf("Target:      %zu primes\n", target_count);
  printf("Output:      %s\n", output_file.c_str());
  printf("Batch Size:  %u\n", batch_size);
  printf("TPB:         %u\n", tpb);
  printf("TPI:         %u\n", tpi);
  printf("Window Bits: %u\n", window_bits);
  printf("MR Rounds:   %u\n", prime_count);
  printf("IPB:         %u\n", tpb / tpi);
  printf("============================\n\n");

  dispatch_params(tpi, window_bits, target_count, output_file,
                  batch_size, tpb, prime_count);

  return 0;
}
