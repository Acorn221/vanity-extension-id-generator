/**
 * Prime Pool Generator for Vanity Extension ID GPU Search
 * 
 * Generates a pool of certified 1024-bit primes and stores them in a binary file.
 * These primes are used by the Metal GPU search to try all combinations.
 * 
 * Supports two backends:
 *   - Metal GPU (default on macOS): Much faster, uses GPU parallelism
 *   - CPU (fallback): Uses OpenSSL with multi-threading
 * 
 * Output format:
 *   - Header: uint32_t magic, version, count, prime_bytes (128)
 *   - Primes: count × 128 bytes (big-endian, zero-padded)
 * 
 * Usage: ./generate_prime_pool [count] [output_file] [--cpu]
 *        Default: 100000 primes to prime_pool.bin
 *        --cpu: Force CPU backend even if Metal is available
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>

#include <openssl/bn.h>
#include <openssl/rand.h>

// Metal support (macOS only)
#ifdef __APPLE__
#define HAS_METAL 1
#include "metal/prime_metal_runner.h"
#else
#define HAS_METAL 0
#endif

constexpr size_t PRIME_BITS = 1024;
constexpr size_t PRIME_BYTES = PRIME_BITS / 8;  // 128 bytes

struct PrimePoolHeader {
    uint32_t magic;         // 'PRMP' = 0x504D5250
    uint32_t version;       // 1
    uint32_t count;         // Number of primes
    uint32_t prime_bytes;   // 128
};

// =============================================================================
// CPU Backend (OpenSSL)
// =============================================================================

// Convert BIGNUM to fixed-size big-endian byte array
bool bn_to_bytes(const BIGNUM* bn, uint8_t* out, size_t out_len) {
    int bn_bytes = BN_num_bytes(bn);
    if (bn_bytes > static_cast<int>(out_len)) {
        return false;
    }
    
    // Zero-pad the beginning
    memset(out, 0, out_len);
    
    // Write big-endian bytes at the end
    BN_bn2bin(bn, out + (out_len - bn_bytes));
    return true;
}

// Generate a single 1024-bit prime using OpenSSL
BIGNUM* generate_prime_cpu(BN_CTX* /* ctx */) {
    BIGNUM* prime = BN_new();
    if (!prime) return nullptr;
    
    // Generate prime with BN_generate_prime_ex
    // safe=0 means we don't need (p-1)/2 to also be prime
    if (!BN_generate_prime_ex(prime, PRIME_BITS, 0, nullptr, nullptr, nullptr)) {
        BN_free(prime);
        return nullptr;
    }
    
    return prime;
}

// Worker thread for parallel CPU prime generation
void cpu_worker_thread(
    std::vector<std::vector<uint8_t>>& primes,
    std::mutex& primes_mutex,
    std::atomic<size_t>& generated,
    std::atomic<bool>& done,
    size_t target_count
) {
    BN_CTX* ctx = BN_CTX_new();
    std::vector<uint8_t> prime_bytes(PRIME_BYTES);
    
    while (!done.load()) {
        BIGNUM* prime = generate_prime_cpu(ctx);
        if (!prime) continue;
        
        bn_to_bytes(prime, prime_bytes.data(), PRIME_BYTES);
        BN_free(prime);
        
        {
            std::lock_guard<std::mutex> lock(primes_mutex);
            if (primes.size() < target_count) {
                primes.push_back(prime_bytes);
                generated.fetch_add(1);
            }
            if (primes.size() >= target_count) {
                done.store(true);
            }
        }
    }
    
    BN_CTX_free(ctx);
}

// Generate primes using CPU
size_t generate_primes_cpu(size_t target_count, const std::string& output_file) {
    std::cout << "Using CPU backend (OpenSSL)\n";
    
    // Determine thread count
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::cout << "Using " << num_threads << " CPU threads\n\n";
    
    // Storage for generated primes
    std::vector<std::vector<uint8_t>> primes;
    primes.reserve(target_count);
    std::mutex primes_mutex;
    std::atomic<size_t> generated{0};
    std::atomic<bool> done{false};
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Spawn worker threads
    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < num_threads; i++) {
        workers.emplace_back(cpu_worker_thread, 
            std::ref(primes), std::ref(primes_mutex),
            std::ref(generated), std::ref(done), target_count);
    }
    
    // Progress monitoring
    size_t last_count = 0;
    auto last_time = start_time;
    
    while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        size_t current = generated.load();
        auto now = std::chrono::steady_clock::now();
        
        double elapsed = std::chrono::duration<double>(now - last_time).count();
        double total_elapsed = std::chrono::duration<double>(now - start_time).count();
        
        size_t delta = current - last_count;
        double rate = (elapsed > 0) ? delta / elapsed : 0;
        double avg_rate = (total_elapsed > 0) ? current / total_elapsed : 0;
        
        double progress = 100.0 * current / target_count;
        double eta = (avg_rate > 0) ? (target_count - current) / avg_rate : 0;
        
        std::cout << "\rProgress: " << std::setw(6) << current << " / " << target_count
                  << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                  << " | Rate: " << std::setw(4) << static_cast<int>(rate) << "/s"
                  << " | ETA: " << std::setw(4) << static_cast<int>(eta) << "s"
                  << "     " << std::flush;
        
        last_count = current;
        last_time = now;
    }
    
    // Wait for all workers
    for (auto& worker : workers) {
        worker.join();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "\n\nGeneration complete!\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_time << "s\n";
    std::cout << "Average rate: " << static_cast<int>(primes.size() / total_time) << " primes/s\n\n";
    
    // Write to file
    std::cout << "Writing to " << output_file << "...\n";
    
    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open output file\n";
        return 0;
    }
    
    // Write header
    PrimePoolHeader header;
    header.magic = 0x504D5250;  // 'PRMP'
    header.version = 1;
    header.count = static_cast<uint32_t>(primes.size());
    header.prime_bytes = PRIME_BYTES;
    
    out.write(reinterpret_cast<char*>(&header), sizeof(header));
    
    // Write primes
    for (const auto& prime : primes) {
        out.write(reinterpret_cast<const char*>(prime.data()), PRIME_BYTES);
    }
    
    out.close();
    
    std::cout << "Done! Wrote " << primes.size() << " primes to " << output_file << "\n";
    
    return primes.size();
}

// =============================================================================
// Metal GPU Backend
// =============================================================================

#if HAS_METAL

size_t generate_primes_metal(size_t target_count, const std::string& output_file) {
    std::cout << "Using Metal GPU backend\n\n";
    
    vanity::metal::PrimeMetalRunner runner;
    
    if (!runner.isAvailable()) {
        std::cerr << "Metal GPU not available, falling back to CPU\n\n";
        return generate_primes_cpu(target_count, output_file);
    }
    
    std::cout << "GPU Device: " << runner.getDeviceName() << "\n";
    std::cout << "Estimated rate: ~" << static_cast<int>(runner.getEstimatedRate()) << " primes/sec\n\n";
    
    auto start_time = std::chrono::steady_clock::now();
    size_t last_count = 0;
    auto last_time = start_time;
    
    // Progress callback
    auto progress_callback = [&](size_t current, size_t target, double rate) {
        auto now = std::chrono::steady_clock::now();
        (void)start_time;  // Used for potential future enhancements
        
        double progress = 100.0 * current / target;
        double eta = (rate > 0) ? (target - current) / rate : 0;
        
        // Format ETA nicely for long runs
        std::string eta_str;
        if (eta > 3600) {
            int hours = static_cast<int>(eta / 3600);
            int mins = static_cast<int>((eta - hours * 3600) / 60);
            eta_str = std::to_string(hours) + "h " + std::to_string(mins) + "m";
        } else if (eta > 60) {
            int mins = static_cast<int>(eta / 60);
            int secs = static_cast<int>(eta - mins * 60);
            eta_str = std::to_string(mins) + "m " + std::to_string(secs) + "s";
        } else {
            eta_str = std::to_string(static_cast<int>(eta)) + "s";
        }
        
        std::cout << "\rProgress: " << std::setw(10) << current << " / " << target
                  << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                  << " | Rate: " << std::setw(6) << static_cast<int>(rate) << "/s"
                  << " | ETA: " << std::setw(10) << eta_str
                  << "     " << std::flush;
        
        last_count = current;
        last_time = now;
    };
    
    size_t generated = runner.generatePrimes(target_count, output_file, progress_callback);
    
    std::cout << "\n";
    
    return generated;
}

#endif

// =============================================================================
// Main
// =============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [count] [output_file] [options]\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  count        Number of primes to generate (default: 100000)\n";
    std::cout << "  output_file  Output file path (default: prime_pool.bin)\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --cpu        Force CPU backend (OpenSSL)\n";
    std::cout << "  --metal      Force Metal GPU backend (macOS only)\n";
    std::cout << "  --help       Show this help message\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog << " 50000000 prime_pool_50m.bin\n";
    std::cout << "  " << prog << " 1000000 --cpu\n";
}

int main(int argc, char* argv[]) {
    size_t target_count = 100000;
    std::string output_file = "prime_pool.bin";
    bool force_cpu = false;
    bool force_metal = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--cpu") {
            force_cpu = true;
        } else if (arg == "--metal") {
            force_metal = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            // Positional argument
            if (i == 1 || (i == 2 && !force_cpu && !force_metal)) {
                // Try to parse as count first
                try {
                    size_t val = std::stoull(arg);
                    if (target_count == 100000) {
                        target_count = val;
                    } else {
                        output_file = arg;
                    }
                } catch (...) {
                    output_file = arg;
                }
            } else {
                output_file = arg;
            }
        }
    }
    
    std::cout << "Prime Pool Generator\n";
    std::cout << "====================\n";
    std::cout << "Target count: " << target_count << " primes\n";
    std::cout << "Prime size:   " << PRIME_BITS << " bits (" << PRIME_BYTES << " bytes)\n";
    std::cout << "Output file:  " << output_file << "\n";
    
    // Calculate expected file size
    size_t expected_size = sizeof(PrimePoolHeader) + target_count * PRIME_BYTES;
    if (expected_size > 1024 * 1024 * 1024) {
        std::cout << "Output size:  ~" << (expected_size / 1024 / 1024 / 1024) << " GB\n";
    } else {
        std::cout << "Output size:  ~" << (expected_size / 1024 / 1024) << " MB\n";
    }
    std::cout << "\n";
    
    size_t generated = 0;
    
#if HAS_METAL
    if (!force_cpu) {
        generated = generate_primes_metal(target_count, output_file);
    } else {
        generated = generate_primes_cpu(target_count, output_file);
    }
#else
    if (force_metal) {
        std::cerr << "Error: Metal is only available on macOS\n";
        return 1;
    }
    generated = generate_primes_cpu(target_count, output_file);
#endif
    
    if (generated == 0) {
        std::cerr << "Error: Failed to generate primes\n";
        return 1;
    }
    
    // Print final stats
    std::cout << "\nFile size: " << (sizeof(PrimePoolHeader) + generated * PRIME_BYTES) << " bytes\n";
    std::cout << "\nPool statistics:\n";
    std::cout << "  Total primes: " << generated << "\n";
    std::cout << "  Total combinations: " << (generated * (generated - 1) / 2) << " unique keys\n";
    
    // Estimate search time
    double pairs = static_cast<double>(generated) * (generated - 1) / 2;
    std::cout << "\nEstimated vanity search times (at 100M pairs/sec):\n";
    std::cout << "  Full search: " << std::fixed << std::setprecision(1) 
              << (pairs / 100e6 / 60) << " minutes\n";
    
    return 0;
}
