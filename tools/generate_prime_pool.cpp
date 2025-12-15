/**
 * Prime Pool Generator for Vanity Extension ID GPU Search
 * 
 * Generates a pool of certified 1024-bit primes and stores them in a binary file.
 * These primes are used by the Metal GPU search to try all combinations.
 * 
 * Output format:
 *   - Header: uint32_t count, uint32_t prime_bytes (128)
 *   - Primes: count × 128 bytes (big-endian, zero-padded)
 * 
 * Usage: ./generate_prime_pool [count] [output_file]
 *        Default: 100000 primes to prime_pool.bin
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

constexpr size_t PRIME_BITS = 1024;
constexpr size_t PRIME_BYTES = PRIME_BITS / 8;  // 128 bytes

struct PrimePoolHeader {
    uint32_t magic;         // 'PRMP' = 0x504D5250
    uint32_t version;       // 1
    uint32_t count;         // Number of primes
    uint32_t prime_bytes;   // 128
};

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

// Generate a single 1024-bit prime
BIGNUM* generate_prime(BN_CTX* ctx) {
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

// Worker thread for parallel prime generation
void worker_thread(
    std::vector<std::vector<uint8_t>>& primes,
    std::mutex& primes_mutex,
    std::atomic<size_t>& generated,
    std::atomic<bool>& done,
    size_t target_count
) {
    BN_CTX* ctx = BN_CTX_new();
    std::vector<uint8_t> prime_bytes(PRIME_BYTES);
    
    while (!done.load()) {
        BIGNUM* prime = generate_prime(ctx);
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

int main(int argc, char* argv[]) {
    size_t target_count = 100000;
    std::string output_file = "prime_pool.bin";
    
    if (argc > 1) {
        target_count = std::stoull(argv[1]);
    }
    if (argc > 2) {
        output_file = argv[2];
    }
    
    std::cout << "Prime Pool Generator\n";
    std::cout << "====================\n";
    std::cout << "Target count: " << target_count << " primes\n";
    std::cout << "Prime size:   " << PRIME_BITS << " bits (" << PRIME_BYTES << " bytes)\n";
    std::cout << "Output file:  " << output_file << "\n";
    std::cout << "Output size:  ~" << (target_count * PRIME_BYTES / 1024 / 1024) << " MB\n";
    std::cout << "\n";
    
    // Determine thread count
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::cout << "Using " << num_threads << " threads\n\n";
    
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
        workers.emplace_back(worker_thread, 
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
        return 1;
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
    std::cout << "File size: " << (sizeof(header) + primes.size() * PRIME_BYTES) << " bytes\n";
    
    // Print stats
    std::cout << "\nPool statistics:\n";
    std::cout << "  Total combinations: " << (primes.size() * (primes.size() - 1) / 2) << " unique keys\n";
    
    return 0;
}
