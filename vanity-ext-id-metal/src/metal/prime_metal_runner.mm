/**
 * Fast Hybrid Prime Generator - Metal GPU + CPU
 * 
 * Strategy:
 * 1. GPU generates random 1024-bit candidates with trial division (FAST)
 * 2. CPU verifies with OpenSSL Miller-Rabin (optimized)
 * 
 * This is 10-50x faster than either pure GPU or pure CPU alone.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "prime_metal_runner.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <atomic>
#include <chrono>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

// OpenSSL for fast Miller-Rabin
#include <openssl/bn.h>

namespace vanity {
namespace metal {

struct PrimePoolHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t prime_bytes;
};

constexpr size_t PRIME_BYTES = 128;

// Thread-safe queue for candidates
template<typename T>
class ThreadSafeQueue {
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }
    
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    void wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
        item = std::move(queue_.front());
        queue_.pop();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cv_;
};

class PrimeMetalRunner::Impl {
public:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> commandQueue_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState> candidatePipeline_ = nil;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};
    
    std::string deviceName_;
    double estimatedRate_ = 5000.0;
    
    Impl() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            std::cerr << "Metal: No GPU device found\n";
            return;
        }
        
        deviceName_ = [device_.name UTF8String];
        
        // Estimate based on device
        if (deviceName_.find("M3") != std::string::npos) {
            estimatedRate_ = 25000.0;
        } else if (deviceName_.find("M2") != std::string::npos) {
            estimatedRate_ = 15000.0;
        } else if (deviceName_.find("M1") != std::string::npos) {
            estimatedRate_ = 8000.0;
        }
        
        commandQueue_ = [device_ newCommandQueue];
        if (!commandQueue_) {
            std::cerr << "Metal: Failed to create command queue\n";
            return;
        }
        
        // Load candidate generator shader
        NSError* error = nil;
        NSArray* shaderPaths = @[
            @"metal/prime_candidate_generator.metal",
            @"src/metal/prime_candidate_generator.metal",
            @"../src/metal/prime_candidate_generator.metal"
        ];
        
        NSString* shaderSource = nil;
        for (NSString* path in shaderPaths) {
            shaderSource = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
            if (shaderSource) break;
        }
        
        if (shaderSource) {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeFast;
            }
            library_ = [device_ newLibraryWithSource:shaderSource options:options error:&error];
            
            if (!library_) {
                std::cerr << "Metal: Shader compile error: " << [[error localizedDescription] UTF8String] << "\n";
            }
        }
        
        if (library_) {
            id<MTLFunction> func = [library_ newFunctionWithName:@"generate_candidates"];
            if (func) {
                candidatePipeline_ = [device_ newComputePipelineStateWithFunction:func error:&error];
            }
        }
        
        if (candidatePipeline_) {
            std::cout << "Metal Prime: GPU candidate generator ready on " << deviceName_ << "\n";
        } else {
            std::cerr << "Metal Prime: Failed to create pipeline\n";
        }
    }
    
    ~Impl() {}
    
    bool isAvailable() const {
        return device_ != nil && commandQueue_ != nil && candidatePipeline_ != nil;
    }
    
    // CPU worker thread - verifies candidates with Miller-Rabin
    static void cpu_verifier_thread(
        ThreadSafeQueue<std::vector<uint8_t>>& candidate_queue,
        std::vector<std::vector<uint8_t>>& verified_primes,
        std::mutex& primes_mutex,
        std::atomic<size_t>& verified_count,
        std::atomic<bool>& done,
        size_t target_count
    ) {
        BN_CTX* ctx = BN_CTX_new();
        
        while (!done.load()) {
            std::vector<uint8_t> candidate;
            
            // Try to get a candidate
            if (!candidate_queue.try_pop(candidate)) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            
            // Convert to BIGNUM
            BIGNUM* n = BN_bin2bn(candidate.data(), PRIME_BYTES, nullptr);
            if (!n) continue;
            
            // Miller-Rabin test (OpenSSL uses optimal number of rounds)
            // Use BN_check_prime for OpenSSL 3.0+
            int is_prime = BN_check_prime(n, ctx, nullptr);
            
            if (is_prime == 1) {
                std::lock_guard<std::mutex> lock(primes_mutex);
                if (verified_primes.size() < target_count) {
                    verified_primes.push_back(candidate);
                    verified_count.fetch_add(1);
                }
                if (verified_primes.size() >= target_count) {
                    done.store(true);
                }
            }
            
            BN_free(n);
        }
        
        BN_CTX_free(ctx);
    }
    
    size_t generatePrimes(
        size_t target_count,
        const std::string& output_file,
        PrimeProgressCallback callback
    ) {
        if (!isAvailable()) {
            std::cerr << "Metal Prime: Not available\n";
            return 0;
        }
        
        running_ = true;
        stopRequested_ = false;
        
        std::cout << "Metal Prime: Hybrid GPU+CPU mode\n";
        std::cout << "  GPU: Generates filtered candidates\n";
        std::cout << "  CPU: Verifies with Miller-Rabin\n\n";
        std::cout << std::flush;
        
        // Generate random seeds
        std::random_device rd;
        std::mt19937_64 seed_gen(rd());
        
        constexpr size_t NUM_SEEDS = 1024;
        std::vector<uint64_t> seeds(NUM_SEEDS);
        for (size_t i = 0; i < NUM_SEEDS; i++) {
            seeds[i] = seed_gen();
        }
        
        id<MTLBuffer> seedBuffer = [device_ newBufferWithBytes:seeds.data()
                                                        length:seeds.size() * sizeof(uint64_t)
                                                       options:MTLResourceStorageModeShared];
        
        // GPU generates candidates in batches
        // Tuned for balance between GPU throughput and memory
        constexpr size_t GPU_BATCH_SIZE = 20000;  // Max candidates per GPU dispatch
        constexpr size_t GPU_THREADS = 8192;      // GPU threads
        constexpr size_t ATTEMPTS_PER_THREAD = 20; // Each thread tries this many candidates
        
        size_t candidate_buffer_size = GPU_BATCH_SIZE * PRIME_BYTES;
        id<MTLBuffer> candidateBuffer = [device_ newBufferWithLength:candidate_buffer_size
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> countBuffer = [device_ newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        
        uint32_t params[2] = {
            static_cast<uint32_t>(GPU_BATCH_SIZE),
            static_cast<uint32_t>(ATTEMPTS_PER_THREAD)
        };
        id<MTLBuffer> paramsBuffer = [device_ newBufferWithBytes:params
                                                          length:sizeof(params)
                                                         options:MTLResourceStorageModeShared];
        
        // Verified primes storage
        std::vector<std::vector<uint8_t>> verified_primes;
        verified_primes.reserve(target_count);
        std::mutex primes_mutex;
        std::atomic<size_t> verified_count{0};
        std::atomic<bool> done{false};
        
        // Candidate queue (GPU -> CPU)
        ThreadSafeQueue<std::vector<uint8_t>> candidate_queue;
        
        // Start CPU verifier threads
        unsigned int num_cpu_threads = std::thread::hardware_concurrency();
        if (num_cpu_threads == 0) num_cpu_threads = 4;
        
        std::cout << "Using " << num_cpu_threads << " CPU threads for verification\n\n";
        std::cout << std::flush;
        
        std::vector<std::thread> cpu_threads;
        for (unsigned int i = 0; i < num_cpu_threads; i++) {
            cpu_threads.emplace_back(cpu_verifier_thread,
                std::ref(candidate_queue),
                std::ref(verified_primes),
                std::ref(primes_mutex),
                std::ref(verified_count),
                std::ref(done),
                target_count
            );
        }
        
        auto start_time = std::chrono::steady_clock::now();
        auto last_time = start_time;
        uint32_t batch_number = 0;
        size_t total_candidates_generated = 0;
        
        // Main loop: GPU generates, CPU verifies
        while (!done.load() && !stopRequested_) {
            @autoreleasepool {
                // Reset candidate count
                uint32_t zero = 0;
                memcpy([countBuffer contents], &zero, sizeof(uint32_t));
                
                // Update seeds periodically
                if (batch_number % 10 == 0) {
                    for (size_t i = 0; i < NUM_SEEDS; i++) {
                        seeds[i] = seed_gen();
                    }
                    memcpy([seedBuffer contents], seeds.data(), seeds.size() * sizeof(uint64_t));
                }
                
                // Launch GPU kernel
                id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
                if (!commandBuffer) {
                    std::cerr << "Failed to create command buffer\n";
                    break;
                }
                
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) {
                    std::cerr << "Failed to create encoder\n";
                    break;
                }
                
                [encoder setComputePipelineState:candidatePipeline_];
                [encoder setBuffer:candidateBuffer offset:0 atIndex:0];
                [encoder setBuffer:countBuffer offset:0 atIndex:1];
                [encoder setBuffer:seedBuffer offset:0 atIndex:2];
                [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
                
                MTLSize gridSize = MTLSizeMake(GPU_THREADS, 1, 1);
                NSUInteger tgSize = std::min(static_cast<NSUInteger>(256),
                                             candidatePipeline_.maxTotalThreadsPerThreadgroup);
                MTLSize threadGroup = MTLSizeMake(tgSize, 1, 1);
                
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroup];
                [encoder endEncoding];
                
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                
                // Check for GPU errors
                if (commandBuffer.status == MTLCommandBufferStatusError) {
                    std::cerr << "GPU Error: " << [[commandBuffer.error localizedDescription] UTF8String] << "\n";
                    break;
                }
                
                // Get generated candidates
                uint32_t num_candidates = *static_cast<uint32_t*>([countBuffer contents]);
                total_candidates_generated += num_candidates;
                
                // Queue candidates for CPU verification
                // Limit queue size to prevent memory explosion (128 bytes * 50k = 6.4MB max)
                while (candidate_queue.size() > 50000 && !done.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                
                if (done.load()) break;
                
                uint8_t* candidate_data = static_cast<uint8_t*>([candidateBuffer contents]);
                for (uint32_t i = 0; i < num_candidates && !done.load(); i++) {
                    std::vector<uint8_t> candidate(PRIME_BYTES);
                    memcpy(candidate.data(), candidate_data + i * PRIME_BYTES, PRIME_BYTES);
                    candidate_queue.push(std::move(candidate));
                }
                
                batch_number++;
                
                // Progress update (every 0.5 seconds)
                auto now = std::chrono::steady_clock::now();
                double elapsed_since_last = std::chrono::duration<double>(now - last_time).count();
                
                if (elapsed_since_last >= 0.5) {
                    size_t current_verified = verified_count.load();
                    double total_elapsed = std::chrono::duration<double>(now - start_time).count();
                    
                    double rate = (total_elapsed > 0) ? current_verified / total_elapsed : 0;
                    double progress = 100.0 * current_verified / target_count;
                    double eta = (rate > 0) ? (target_count - current_verified) / rate : 0;
                    
                    // Format ETA
                    std::string eta_str;
                    if (eta > 3600) {
                        int hours = static_cast<int>(eta / 3600);
                        int mins = static_cast<int>((eta - hours * 3600) / 60);
                        eta_str = std::to_string(hours) + "h" + std::to_string(mins) + "m";
                    } else if (eta > 60) {
                        int mins = static_cast<int>(eta / 60);
                        int secs = static_cast<int>(eta - mins * 60);
                        eta_str = std::to_string(mins) + "m" + std::to_string(secs) + "s";
                    } else {
                        eta_str = std::to_string(static_cast<int>(eta)) + "s";
                    }
                    
                    std::cout << "\rVerified: " << current_verified << "/" << target_count
                              << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                              << " | " << static_cast<int>(rate) << "/s"
                              << " | Queue: " << candidate_queue.size()
                              << " | ETA: " << eta_str << "        " << std::flush;
                    
                    if (callback) {
                        callback(current_verified, target_count, rate);
                    }
                    
                    last_time = now;
                }
            }
        }
        
        // Signal done and wait for CPU threads
        done.store(true);
        for (auto& t : cpu_threads) {
            t.join();
        }
        
        auto end_time = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "\n\nGeneration complete!\n";
        std::cout << "Verified primes: " << verified_primes.size() << "\n";
        std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_time << "s\n";
        std::cout << "Average rate: " << static_cast<int>(verified_primes.size() / total_time) << " primes/s\n";
        std::cout << "GPU candidates generated: " << total_candidates_generated << "\n";
        std::cout << "Hit rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * verified_primes.size() / total_candidates_generated) << "%\n\n";
        
        // Write to file
        std::cout << "Writing to " << output_file << "...\n";
        std::ofstream out(output_file, std::ios::binary);
        if (!out) {
            std::cerr << "Error: Could not open output file\n";
            running_ = false;
            return 0;
        }
        
        PrimePoolHeader header;
        header.magic = 0x504D5250;
        header.version = 1;
        header.count = static_cast<uint32_t>(verified_primes.size());
        header.prime_bytes = PRIME_BYTES;
        
        out.write(reinterpret_cast<char*>(&header), sizeof(header));
        
        for (const auto& prime : verified_primes) {
            out.write(reinterpret_cast<const char*>(prime.data()), PRIME_BYTES);
        }
        
        out.close();
        
        std::cout << "Done! Wrote " << verified_primes.size() << " primes\n";
        
        running_ = false;
        return verified_primes.size();
    }
    
    size_t generatePrimesToBuffer(
        size_t target_count,
        uint8_t* output_buffer,
        PrimeProgressCallback callback
    ) {
        // Similar to generatePrimes but writes to buffer
        // For now, just use file-based approach
        std::string temp_file = "/tmp/prime_pool_temp.bin";
        size_t count = generatePrimes(target_count, temp_file, callback);
        
        if (count > 0 && output_buffer) {
            std::ifstream in(temp_file, std::ios::binary);
            in.seekg(sizeof(PrimePoolHeader));  // Skip header
            in.read(reinterpret_cast<char*>(output_buffer), count * PRIME_BYTES);
        }
        
        std::remove(temp_file.c_str());
        return count;
    }
    
    void stop() { stopRequested_ = true; }
    bool isRunning() const { return running_; }
};

// Public interface
PrimeMetalRunner::PrimeMetalRunner() : impl_(new Impl()) {}
PrimeMetalRunner::~PrimeMetalRunner() { delete impl_; }
bool PrimeMetalRunner::isAvailable() const { return impl_->isAvailable(); }
std::string PrimeMetalRunner::getDeviceName() const { return impl_->deviceName_; }
size_t PrimeMetalRunner::generatePrimes(size_t target_count, const std::string& output_file, PrimeProgressCallback callback) {
    return impl_->generatePrimes(target_count, output_file, callback);
}
size_t PrimeMetalRunner::generatePrimesToBuffer(size_t target_count, uint8_t* output_buffer, PrimeProgressCallback callback) {
    return impl_->generatePrimesToBuffer(target_count, output_buffer, callback);
}
void PrimeMetalRunner::stop() { impl_->stop(); }
bool PrimeMetalRunner::isRunning() const { return impl_->isRunning(); }
double PrimeMetalRunner::getEstimatedRate() const { return impl_->estimatedRate_; }

} // namespace metal
} // namespace vanity
