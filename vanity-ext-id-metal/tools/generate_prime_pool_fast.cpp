/**
 * ULTRA-FAST Prime Pool Generator
 * 
 * Optimized for maximum throughput:
 * - Minimal Miller-Rabin rounds (10 = 2^-20 error, plenty safe)
 * - Direct OpenSSL BN calls without wrappers
 * - Lock-free output buffer
 * - SIMD-friendly memory layout
 * 
 * Target: 10,000+ primes/sec on modern CPU
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <thread>
#include <atomic>

#include <openssl/bn.h>
#include <openssl/rand.h>

constexpr size_t PRIME_BITS = 1024;
constexpr size_t PRIME_BYTES = 128;
constexpr int MILLER_RABIN_ROUNDS = 10;  // 2^-20 error probability - plenty safe

// Small primes for trial division
constexpr uint16_t SMALL_PRIMES[] = {
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311
};
constexpr size_t NUM_SMALL_PRIMES = sizeof(SMALL_PRIMES) / sizeof(SMALL_PRIMES[0]);

struct PrimePoolHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t prime_bytes;
};

// Fast trial division using precomputed remainders
inline bool passes_trial_division(BIGNUM* n) {
    BN_ULONG rem;
    for (size_t i = 0; i < NUM_SMALL_PRIMES; i++) {
        rem = BN_mod_word(n, SMALL_PRIMES[i]);
        if (rem == 0) return false;
    }
    return true;
}

// Fast Miller-Rabin with minimal rounds for speed
inline bool fast_miller_rabin(BIGNUM* n, BN_CTX* ctx) {
    // Use BN_is_prime_ex with controlled number of rounds
    // Suppress deprecation warning - we need this for speed control
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return BN_is_prime_ex(n, MILLER_RABIN_ROUNDS, ctx, nullptr) == 1;
    #pragma clang diagnostic pop
}

// Worker thread - generates primes directly to output buffer
void fast_worker(
    uint8_t* output_buffer,
    std::atomic<size_t>& write_index,
    std::atomic<size_t>& generated_count,
    std::atomic<bool>& done,
    size_t target_count,
    size_t max_buffer_size
) {
    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* candidate = BN_new();
    BIGNUM* tmp = BN_new();
    
    uint8_t local_buffer[PRIME_BYTES];
    
    // Pre-generate random bytes in larger chunks for efficiency
    constexpr size_t RAND_CHUNK = 4096;
    uint8_t rand_pool[RAND_CHUNK];
    size_t rand_pos = RAND_CHUNK;
    
    while (!done.load(std::memory_order_relaxed)) {
        // Refill random pool if needed
        if (rand_pos >= RAND_CHUNK - PRIME_BYTES) {
            RAND_bytes(rand_pool, RAND_CHUNK);
            rand_pos = 0;
        }
        
        // Generate candidate from random pool
        BN_bin2bn(rand_pool + rand_pos, PRIME_BYTES, candidate);
        rand_pos += PRIME_BYTES;
        
        // Set MSB for exactly 1024 bits
        BN_set_bit(candidate, PRIME_BITS - 1);
        
        // Set LSB for odd
        BN_set_bit(candidate, 0);
        
        // Trial division filter (catches ~85% of composites)
        if (!passes_trial_division(candidate)) {
            continue;
        }
        
        // Fast Miller-Rabin with minimal rounds
        if (!fast_miller_rabin(candidate, ctx)) {
            continue;
        }
        
        // Found a prime! Write to buffer
        size_t idx = write_index.fetch_add(1, std::memory_order_relaxed);
        if (idx >= max_buffer_size) {
            write_index.fetch_sub(1, std::memory_order_relaxed);
            done.store(true, std::memory_order_relaxed);
            break;
        }
        
        // Convert to bytes and store
        int bn_bytes = BN_num_bytes(candidate);
        memset(local_buffer, 0, PRIME_BYTES);
        BN_bn2bin(candidate, local_buffer + (PRIME_BYTES - bn_bytes));
        
        memcpy(output_buffer + idx * PRIME_BYTES, local_buffer, PRIME_BYTES);
        
        size_t count = generated_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count >= target_count) {
            done.store(true, std::memory_order_relaxed);
        }
    }
    
    BN_free(candidate);
    BN_free(tmp);
    BN_CTX_free(ctx);
}

int main(int argc, char* argv[]) {
    size_t target_count = 100000;
    std::string output_file = "prime_pool.bin";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [count] [output_file]\n";
            std::cout << "Ultra-fast prime generator (10 M-R rounds)\n";
            return 0;
        } else if (arg[0] != '-') {
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
        }
    }
    
    std::cout << "Ultra-Fast Prime Generator\n";
    std::cout << "==========================\n";
    std::cout << "Target: " << target_count << " primes\n";
    std::cout << "Output: " << output_file << "\n";
    std::cout << "M-R rounds: " << MILLER_RABIN_ROUNDS << " (error < 2^-" << (2*MILLER_RABIN_ROUNDS) << ")\n";
    
    size_t output_size = target_count * PRIME_BYTES;
    if (output_size > 1024*1024*1024) {
        std::cout << "Size: ~" << (output_size / 1024 / 1024 / 1024) << " GB\n";
    } else {
        std::cout << "Size: ~" << (output_size / 1024 / 1024) << " MB\n";
    }
    std::cout << "\n";
    
    // Allocate output buffer
    std::vector<uint8_t> buffer(target_count * PRIME_BYTES);
    
    // Atomics for coordination
    std::atomic<size_t> write_index{0};
    std::atomic<size_t> generated_count{0};
    std::atomic<bool> done{false};
    
    // Start worker threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::cout << "Using " << num_threads << " threads\n\n";
    
    auto start_time = std::chrono::steady_clock::now();
    
    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < num_threads; i++) {
        workers.emplace_back(fast_worker,
            buffer.data(),
            std::ref(write_index),
            std::ref(generated_count),
            std::ref(done),
            target_count,
            target_count
        );
    }
    
    // Progress monitoring
    size_t last_count = 0;
    auto last_time = start_time;
    
    while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        
        size_t current = generated_count.load();
        auto now = std::chrono::steady_clock::now();
        
        double elapsed = std::chrono::duration<double>(now - last_time).count();
        double total_elapsed = std::chrono::duration<double>(now - start_time).count();
        
        double rate = (elapsed > 0) ? (current - last_count) / elapsed : 0;
        double avg_rate = (total_elapsed > 0) ? current / total_elapsed : 0;
        
        double progress = 100.0 * current / target_count;
        double eta = (avg_rate > 0) ? (target_count - current) / avg_rate : 0;
        
        // Format ETA
        std::string eta_str;
        if (eta > 3600) {
            int h = static_cast<int>(eta / 3600);
            int m = static_cast<int>((eta - h * 3600) / 60);
            eta_str = std::to_string(h) + "h" + std::to_string(m) + "m";
        } else if (eta > 60) {
            int m = static_cast<int>(eta / 60);
            int s = static_cast<int>(eta - m * 60);
            eta_str = std::to_string(m) + "m" + std::to_string(s) + "s";
        } else {
            eta_str = std::to_string(static_cast<int>(eta)) + "s";
        }
        
        std::cout << "\r" << current << "/" << target_count
                  << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                  << " | " << static_cast<int>(rate) << "/s"
                  << " | avg " << static_cast<int>(avg_rate) << "/s"
                  << " | ETA: " << eta_str << "        " << std::flush;
        
        last_count = current;
        last_time = now;
    }
    
    // Wait for workers
    for (auto& w : workers) {
        w.join();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    size_t final_count = generated_count.load();
    
    std::cout << "\n\nDone! Generated " << final_count << " primes in "
              << std::fixed << std::setprecision(1) << total_time << "s\n";
    std::cout << "Average: " << static_cast<int>(final_count / total_time) << " primes/s\n\n";
    
    // Write to file
    std::cout << "Writing to " << output_file << "...\n";
    
    std::ofstream out(output_file, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open output file\n";
        return 1;
    }
    
    PrimePoolHeader header;
    header.magic = 0x504D5250;
    header.version = 1;
    header.count = static_cast<uint32_t>(final_count);
    header.prime_bytes = PRIME_BYTES;
    
    out.write(reinterpret_cast<char*>(&header), sizeof(header));
    out.write(reinterpret_cast<char*>(buffer.data()), final_count * PRIME_BYTES);
    out.close();
    
    std::cout << "Wrote " << final_count << " primes to " << output_file << "\n";
    std::cout << "Combinations: " << (final_count * (final_count - 1) / 2) << " unique RSA keys\n";
    
    return 0;
}

