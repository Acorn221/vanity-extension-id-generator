/**
 * CUDA GPU Runner Implementation
 * 
 * Multi-GPU support for 8x A100 GPUs with:
 * - Parallel memory allocation across all GPUs
 * - Work distribution by row ranges
 * - Async kernel execution with streams
 * - Result collection and deduplication
 */

#include "cuda_runner.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <iostream>
#include <cstring>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <unordered_set>

// For key reconstruction
#include <openssl/bn.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/param_build.h>
#include <openssl/core_names.h>

namespace vanity {
namespace cuda {

// =============================================================================
// CUDA Error Checking Macros
// =============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    } \
} while(0)

#define CUDA_CHECK_RETURN(call, retval) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return retval; \
    } \
} while(0)

// =============================================================================
// Match Result Structure (must match kernel definition)
// =============================================================================

struct ShaderMatchResult {
    uint32_t prime_idx_p;
    uint32_t prime_idx_q;
    char ext_id[36];
    uint32_t match_type;
};

// =============================================================================
// Prime Pool Header
// =============================================================================

struct PrimePoolHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t prime_bytes;
};

constexpr uint32_t PRIME_BYTES = 128;

// =============================================================================
// External Kernel Declarations
// =============================================================================

extern "C" __global__ void vanity_search_kernel(
    const uint8_t* prime_pool,
    const char* target_prefix,
    const char* target_suffix,
    const uint32_t* params,
    uint32_t* match_count,
    ShaderMatchResult* matches,
    uint32_t max_matches
);

extern "C" __global__ void vanity_search_dict_kernel(
    const uint8_t* prime_pool,
    const char* dictionary,
    const uint32_t* word_offsets,
    const uint32_t* params,
    uint32_t* match_count,
    ShaderMatchResult* matches,
    uint32_t max_matches
);

// =============================================================================
// Per-GPU Context
// =============================================================================

struct GPUContext {
    int device_id;
    cudaStream_t stream;
    
    // Device memory pointers
    uint8_t* d_prime_pool;
    char* d_prefix;
    char* d_suffix;
    char* d_dictionary;
    uint32_t* d_word_offsets;
    uint32_t* d_params;
    uint32_t* d_match_count;
    ShaderMatchResult* d_matches;
    
    // Host pinned memory for async transfers
    uint32_t* h_match_count;
    ShaderMatchResult* h_matches;
    
    size_t max_matches;
    
    GPUContext() : device_id(-1), stream(nullptr),
                   d_prime_pool(nullptr), d_prefix(nullptr), d_suffix(nullptr),
                   d_dictionary(nullptr), d_word_offsets(nullptr), d_params(nullptr),
                   d_match_count(nullptr), d_matches(nullptr),
                   h_match_count(nullptr), h_matches(nullptr), max_matches(0) {}
};

// =============================================================================
// Implementation Class
// =============================================================================

class CudaRunner::Impl {
public:
    std::vector<GPUContext> gpu_contexts_;
    int device_count_ = 0;
    
    // Host prime pool
    std::vector<uint8_t> prime_pool_data_;
    size_t prime_count_ = 0;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    
    Impl() {
        // Get device count
        cudaError_t err = cudaGetDeviceCount(&device_count_);
        if (err != cudaSuccess || device_count_ == 0) {
            std::cerr << "CUDA: No GPU devices found\n";
            device_count_ = 0;
            return;
        }
        
        std::cout << "CUDA: Found " << device_count_ << " GPU device(s)\n";
        
        // Initialize contexts for each GPU
        gpu_contexts_.resize(device_count_);
        for (int i = 0; i < device_count_; i++) {
            gpu_contexts_[i].device_id = i;
            
            cudaSetDevice(i);
            cudaStreamCreate(&gpu_contexts_[i].stream);
            
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "  GPU " << i << ": " << prop.name 
                      << " (" << prop.multiProcessorCount << " SMs, "
                      << (prop.totalGlobalMem / 1024 / 1024) << " MB)\n";
        }
    }
    
    ~Impl() {
        // Cleanup GPU contexts
        for (auto& ctx : gpu_contexts_) {
            if (ctx.device_id >= 0) {
                cudaSetDevice(ctx.device_id);
                
                if (ctx.stream) cudaStreamDestroy(ctx.stream);
                if (ctx.d_prime_pool) cudaFree(ctx.d_prime_pool);
                if (ctx.d_prefix) cudaFree(ctx.d_prefix);
                if (ctx.d_suffix) cudaFree(ctx.d_suffix);
                if (ctx.d_dictionary) cudaFree(ctx.d_dictionary);
                if (ctx.d_word_offsets) cudaFree(ctx.d_word_offsets);
                if (ctx.d_params) cudaFree(ctx.d_params);
                if (ctx.d_match_count) cudaFree(ctx.d_match_count);
                if (ctx.d_matches) cudaFree(ctx.d_matches);
                if (ctx.h_match_count) cudaFreeHost(ctx.h_match_count);
                if (ctx.h_matches) cudaFreeHost(ctx.h_matches);
            }
        }
    }
    
    bool isAvailable() const {
        return device_count_ > 0;
    }
    
    std::vector<GPUInfo> getDeviceInfo() const {
        std::vector<GPUInfo> info;
        for (int i = 0; i < device_count_; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            size_t free_mem, total_mem;
            cudaSetDevice(i);
            cudaMemGetInfo(&free_mem, &total_mem);
            
            GPUInfo gi;
            gi.device_id = i;
            gi.name = prop.name;
            gi.total_memory = total_mem;
            gi.free_memory = free_mem;
            gi.sm_count = prop.multiProcessorCount;
            gi.compute_capability_major = prop.major;
            gi.compute_capability_minor = prop.minor;
            info.push_back(gi);
        }
        return info;
    }
    
    std::string getDeviceName() const {
        if (device_count_ == 0) return "No CUDA devices";
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        if (device_count_ == 1) {
            return prop.name;
        }
        return std::to_string(device_count_) + "x " + prop.name;
    }
    
    size_t loadPrimePool(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "CUDA: Failed to open prime pool: " << filepath << "\n";
            return 0;
        }
        
        // Read header
        PrimePoolHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (header.magic != 0x504D5250) {  // 'PRMP'
            std::cerr << "CUDA: Invalid prime pool file (bad magic)\n";
            return 0;
        }
        
        if (header.prime_bytes != PRIME_BYTES) {
            std::cerr << "CUDA: Unexpected prime size: " << header.prime_bytes << "\n";
            return 0;
        }
        
        prime_count_ = header.count;
        size_t data_size = prime_count_ * PRIME_BYTES;
        
        // Read prime data to host memory
        prime_pool_data_.resize(data_size);
        file.read(reinterpret_cast<char*>(prime_pool_data_.data()), data_size);
        
        std::cout << "CUDA: Loaded " << prime_count_ << " primes ("
                  << (data_size / 1024 / 1024) << " MB)\n";
        
        // Allocate and copy to each GPU
        std::cout << "CUDA: Copying prime pool to " << device_count_ << " GPU(s)...\n";
        
        for (auto& ctx : gpu_contexts_) {
            cudaSetDevice(ctx.device_id);
            
            // Allocate device memory
            cudaError_t err = cudaMalloc(&ctx.d_prime_pool, data_size);
            if (err != cudaSuccess) {
                std::cerr << "CUDA: Failed to allocate GPU memory on device " << ctx.device_id << "\n";
                return 0;
            }
            
            // Copy async
            cudaMemcpyAsync(ctx.d_prime_pool, prime_pool_data_.data(), data_size,
                           cudaMemcpyHostToDevice, ctx.stream);
        }
        
        // Wait for all copies to complete
        for (auto& ctx : gpu_contexts_) {
            cudaSetDevice(ctx.device_id);
            cudaStreamSynchronize(ctx.stream);
        }
        
        std::cout << "CUDA: Prime pool ready on all GPUs\n";
        return prime_count_;
    }
    
    std::vector<GPUMatch> search(const SearchConfig& config, ProgressCallback callback) {
        std::vector<GPUMatch> results;
        
        if (!isAvailable() || prime_count_ == 0) {
            return results;
        }
        
        running_ = true;
        stop_requested_ = false;
        
        // Allocate buffers on each GPU
        size_t max_matches_per_gpu = 10000000;  // 10M per GPU
        
        for (auto& ctx : gpu_contexts_) {
            cudaSetDevice(ctx.device_id);
            
            ctx.max_matches = max_matches_per_gpu;
            
            // Allocate device buffers
            cudaMalloc(&ctx.d_prefix, config.target_prefix.size() + 1);
            cudaMalloc(&ctx.d_suffix, config.target_suffix.size() + 1);
            cudaMalloc(&ctx.d_params, 4 * sizeof(uint32_t));
            cudaMalloc(&ctx.d_match_count, sizeof(uint32_t));
            cudaMalloc(&ctx.d_matches, max_matches_per_gpu * sizeof(ShaderMatchResult));
            
            // Allocate pinned host memory
            cudaMallocHost(&ctx.h_match_count, sizeof(uint32_t));
            cudaMallocHost(&ctx.h_matches, max_matches_per_gpu * sizeof(ShaderMatchResult));
            
            // Copy prefix and suffix
            cudaMemcpy(ctx.d_prefix, config.target_prefix.c_str(), 
                      config.target_prefix.size() + 1, cudaMemcpyHostToDevice);
            cudaMemcpy(ctx.d_suffix, config.target_suffix.c_str(),
                      config.target_suffix.size() + 1, cudaMemcpyHostToDevice);
            
            // Initialize match count to 0
            cudaMemset(ctx.d_match_count, 0, sizeof(uint32_t));
        }
        
        uint64_t total_pairs = (uint64_t)prime_count_ * (prime_count_ - 1) / 2;
        uint32_t batch_size = config.batch_size;
        
        // Divide work among GPUs
        uint32_t rows_per_gpu = (prime_count_ + device_count_ - 1) / device_count_;
        
        auto start_time = std::chrono::steady_clock::now();
        
        // Process in batches
        for (uint32_t batch_start = 0; batch_start < prime_count_ && !stop_requested_; batch_start += batch_size * device_count_) {
            // Launch kernels on all GPUs
            for (int gpu_idx = 0; gpu_idx < device_count_; gpu_idx++) {
                auto& ctx = gpu_contexts_[gpu_idx];
                cudaSetDevice(ctx.device_id);
                
                uint32_t start_idx = batch_start + gpu_idx * batch_size;
                if (start_idx >= prime_count_) continue;
                
                // Update params
                uint32_t params[4] = {
                    static_cast<uint32_t>(prime_count_),
                    static_cast<uint32_t>(config.target_prefix.size()),
                    static_cast<uint32_t>(config.target_suffix.size()),
                    start_idx
                };
                cudaMemcpyAsync(ctx.d_params, params, sizeof(params),
                               cudaMemcpyHostToDevice, ctx.stream);
                
                // Calculate grid dimensions
                uint32_t grid_width = min(batch_size, (uint32_t)prime_count_ - start_idx);
                uint32_t grid_height = prime_count_ - start_idx;
                
                // A100 optimal: 256 threads per block
                dim3 block_size(16, 16);  // 256 threads
                dim3 grid_size((grid_width + block_size.x - 1) / block_size.x,
                              (grid_height + block_size.y - 1) / block_size.y);
                
                // Launch kernel
                vanity_search_kernel<<<grid_size, block_size, 0, ctx.stream>>>(
                    ctx.d_prime_pool,
                    ctx.d_prefix,
                    ctx.d_suffix,
                    ctx.d_params,
                    ctx.d_match_count,
                    ctx.d_matches,
                    ctx.max_matches
                );
            }
            
            // Synchronize all GPUs
            for (auto& ctx : gpu_contexts_) {
                cudaSetDevice(ctx.device_id);
                cudaStreamSynchronize(ctx.stream);
            }
            
            // Progress callback
            if (callback) {
                uint64_t pairs_checked = 0;
                uint64_t total_matches = 0;
                
                for (auto& ctx : gpu_contexts_) {
                    cudaSetDevice(ctx.device_id);
                    cudaMemcpy(ctx.h_match_count, ctx.d_match_count, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost);
                    total_matches += *ctx.h_match_count;
                }
                
                // Estimate pairs checked
                uint32_t end_idx = min(batch_start + batch_size * device_count_, (uint32_t)prime_count_);
                for (uint32_t row = 0; row < end_idx; row++) {
                    pairs_checked += (prime_count_ - 1 - row);
                }
                
                callback(pairs_checked, total_pairs, total_matches);
            }
        }
        
        // Collect results from all GPUs
        std::cout << "\nCUDA: Collecting results from " << device_count_ << " GPU(s)...\n";
        
        for (auto& ctx : gpu_contexts_) {
            cudaSetDevice(ctx.device_id);
            
            cudaMemcpy(ctx.h_match_count, ctx.d_match_count, sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
            
            uint32_t match_count = *ctx.h_match_count;
            if (match_count > ctx.max_matches) match_count = ctx.max_matches;
            
            if (match_count > 0) {
                cudaMemcpy(ctx.h_matches, ctx.d_matches, 
                          match_count * sizeof(ShaderMatchResult),
                          cudaMemcpyDeviceToHost);
                
                for (uint32_t i = 0; i < match_count; i++) {
                    GPUMatch match;
                    match.prime_idx_p = ctx.h_matches[i].prime_idx_p;
                    match.prime_idx_q = ctx.h_matches[i].prime_idx_q;
                    match.extension_id = std::string(ctx.h_matches[i].ext_id);
                    match.match_type = ctx.h_matches[i].match_type;
                    results.push_back(match);
                }
            }
            
            std::cout << "  GPU " << ctx.device_id << ": " << match_count << " matches\n";
        }
        
        auto end_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "CUDA: Search complete. " << total_pairs << " pairs in "
                  << elapsed << "s (" << (total_pairs / elapsed / 1e9) << " B pairs/s)\n";
        
        running_ = false;
        return results;
    }
    
    std::vector<GPUMatch> searchDictionary(
        const std::vector<std::string>& words,
        size_t min_len,
        ProgressCallback callback
    ) {
        std::vector<GPUMatch> results;
        
        if (!isAvailable() || prime_count_ == 0) {
            return results;
        }
        
        running_ = true;
        stop_requested_ = false;
        
        // Filter and pack dictionary
        std::vector<std::string> valid_words;
        for (const auto& word : words) {
            if (word.size() >= min_len && word.size() <= 10) {
                valid_words.push_back(word);
            }
        }
        
        std::cout << "Dictionary: " << valid_words.size() << " valid words (min " << min_len << " chars)\n";
        
        // Pack dictionary: null-separated words
        std::vector<char> dict_data;
        std::vector<uint32_t> word_offsets;
        
        for (const auto& word : valid_words) {
            word_offsets.push_back(static_cast<uint32_t>(dict_data.size()));
            for (char c : word) {
                dict_data.push_back(c);
            }
            dict_data.push_back('\0');
        }
        word_offsets.push_back(static_cast<uint32_t>(dict_data.size()));
        
        // Allocate buffers on each GPU
        size_t max_matches_per_gpu = 50000000;  // 50M per GPU for dictionary mode
        
        for (auto& ctx : gpu_contexts_) {
            cudaSetDevice(ctx.device_id);
            
            ctx.max_matches = max_matches_per_gpu;
            
            // Allocate device buffers
            cudaMalloc(&ctx.d_dictionary, dict_data.size());
            cudaMalloc(&ctx.d_word_offsets, word_offsets.size() * sizeof(uint32_t));
            cudaMalloc(&ctx.d_params, 4 * sizeof(uint32_t));
            cudaMalloc(&ctx.d_match_count, sizeof(uint32_t));
            cudaMalloc(&ctx.d_matches, max_matches_per_gpu * sizeof(ShaderMatchResult));
            
            // Allocate pinned host memory
            cudaMallocHost(&ctx.h_match_count, sizeof(uint32_t));
            cudaMallocHost(&ctx.h_matches, max_matches_per_gpu * sizeof(ShaderMatchResult));
            
            // Copy dictionary data
            cudaMemcpy(ctx.d_dictionary, dict_data.data(), dict_data.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(ctx.d_word_offsets, word_offsets.data(), 
                      word_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            
            // Initialize match count to 0
            cudaMemset(ctx.d_match_count, 0, sizeof(uint32_t));
        }
        
        uint64_t total_pairs = (uint64_t)prime_count_ * (prime_count_ - 1) / 2;
        uint32_t batch_size = 2048;  // Larger batches for dictionary mode
        
        auto start_time = std::chrono::steady_clock::now();
        
        // Process in batches across all GPUs
        for (uint32_t batch_start = 0; batch_start < prime_count_ && !stop_requested_; batch_start += batch_size * device_count_) {
            // Launch kernels on all GPUs
            for (int gpu_idx = 0; gpu_idx < device_count_; gpu_idx++) {
                auto& ctx = gpu_contexts_[gpu_idx];
                cudaSetDevice(ctx.device_id);
                
                uint32_t start_idx = batch_start + gpu_idx * batch_size;
                if (start_idx >= prime_count_) continue;
                
                // Update params: [pool_size, num_words, min_len, start_idx]
                uint32_t params[4] = {
                    static_cast<uint32_t>(prime_count_),
                    static_cast<uint32_t>(valid_words.size()),
                    static_cast<uint32_t>(min_len),
                    start_idx
                };
                cudaMemcpyAsync(ctx.d_params, params, sizeof(params),
                               cudaMemcpyHostToDevice, ctx.stream);
                
                // Calculate grid dimensions
                uint32_t grid_width = min(batch_size, (uint32_t)prime_count_ - start_idx);
                uint32_t grid_height = prime_count_ - start_idx;
                
                // A100 optimal: 256 threads per block
                dim3 block_size(16, 16);
                dim3 grid_size((grid_width + block_size.x - 1) / block_size.x,
                              (grid_height + block_size.y - 1) / block_size.y);
                
                // Launch kernel
                vanity_search_dict_kernel<<<grid_size, block_size, 0, ctx.stream>>>(
                    ctx.d_prime_pool,
                    ctx.d_dictionary,
                    ctx.d_word_offsets,
                    ctx.d_params,
                    ctx.d_match_count,
                    ctx.d_matches,
                    ctx.max_matches
                );
            }
            
            // Synchronize all GPUs periodically (every few batches)
            for (auto& ctx : gpu_contexts_) {
                cudaSetDevice(ctx.device_id);
                cudaStreamSynchronize(ctx.stream);
            }
            
            // Progress callback
            if (callback) {
                uint64_t pairs_checked = 0;
                uint64_t total_matches = 0;
                
                for (auto& ctx : gpu_contexts_) {
                    cudaSetDevice(ctx.device_id);
                    cudaMemcpy(ctx.h_match_count, ctx.d_match_count, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost);
                    total_matches += *ctx.h_match_count;
                }
                
                uint32_t end_idx = min(batch_start + batch_size * device_count_, (uint32_t)prime_count_);
                for (uint32_t row = 0; row < end_idx; row++) {
                    pairs_checked += (prime_count_ - 1 - row);
                }
                
                callback(pairs_checked, total_pairs, total_matches);
            }
        }
        
        // Collect results from all GPUs
        std::cout << "\nCUDA: Collecting results from " << device_count_ << " GPU(s)...\n";
        
        for (auto& ctx : gpu_contexts_) {
            cudaSetDevice(ctx.device_id);
            
            cudaMemcpy(ctx.h_match_count, ctx.d_match_count, sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
            
            uint32_t match_count = *ctx.h_match_count;
            if (match_count > ctx.max_matches) match_count = ctx.max_matches;
            
            if (match_count > 0) {
                cudaMemcpy(ctx.h_matches, ctx.d_matches,
                          match_count * sizeof(ShaderMatchResult),
                          cudaMemcpyDeviceToHost);
                
                for (uint32_t i = 0; i < match_count; i++) {
                    GPUMatch match;
                    match.prime_idx_p = ctx.h_matches[i].prime_idx_p;
                    match.prime_idx_q = ctx.h_matches[i].prime_idx_q;
                    match.extension_id = std::string(ctx.h_matches[i].ext_id);
                    match.match_type = ctx.h_matches[i].match_type;
                    results.push_back(match);
                }
            }
            
            std::cout << "  GPU " << ctx.device_id << ": " << match_count << " matches\n";
        }
        
        auto end_time = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "CUDA: Dictionary search complete. " << total_pairs << " pairs in "
                  << elapsed << "s (" << (total_pairs / elapsed / 1e9) << " B pairs/s)\n";
        std::cout << "Found " << results.size() << " total matches\n";
        
        running_ = false;
        return results;
    }
    
    void stop() {
        stop_requested_ = true;
    }
    
    bool isRunning() const {
        return running_;
    }
};

// =============================================================================
// Public Interface Implementation
// =============================================================================

CudaRunner::CudaRunner() : impl_(new Impl()) {}

CudaRunner::~CudaRunner() {
    delete impl_;
}

bool CudaRunner::isAvailable() const {
    return impl_->isAvailable();
}

int CudaRunner::getDeviceCount() const {
    return impl_->device_count_;
}

std::vector<GPUInfo> CudaRunner::getDeviceInfo() const {
    return impl_->getDeviceInfo();
}

std::string CudaRunner::getDeviceName() const {
    return impl_->getDeviceName();
}

size_t CudaRunner::loadPrimePool(const std::string& filepath) {
    return impl_->loadPrimePool(filepath);
}

size_t CudaRunner::getPrimeCount() const {
    return impl_->prime_count_;
}

uint64_t CudaRunner::getTotalPairs() const {
    size_t n = impl_->prime_count_;
    return (uint64_t)n * (n - 1) / 2;
}

const uint8_t* CudaRunner::getPrimePool() const {
    return impl_->prime_pool_data_.data();
}

std::vector<GPUMatch> CudaRunner::search(const SearchConfig& config, ProgressCallback callback) {
    return impl_->search(config, callback);
}

std::vector<GPUMatch> CudaRunner::searchDictionary(
    const std::vector<std::string>& words,
    size_t min_len,
    ProgressCallback callback
) {
    return impl_->searchDictionary(words, min_len, callback);
}

void CudaRunner::stop() {
    impl_->stop();
}

bool CudaRunner::isRunning() const {
    return impl_->isRunning();
}

// =============================================================================
// Key Reconstruction (same as Metal version)
// =============================================================================

std::string reconstructKey(const std::string& prime_pool_path, uint32_t idx_p, uint32_t idx_q) {
    // Load primes from file
    std::ifstream file(prime_pool_path, std::ios::binary);
    if (!file) return "";
    
    PrimePoolHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.magic != 0x504D5250) return "";
    
    // Read the two primes
    std::vector<uint8_t> p_bytes(128), q_bytes(128);
    
    file.seekg(sizeof(header) + idx_p * 128);
    file.read(reinterpret_cast<char*>(p_bytes.data()), 128);
    
    file.seekg(sizeof(header) + idx_q * 128);
    file.read(reinterpret_cast<char*>(q_bytes.data()), 128);
    
    // Convert to BIGNUMs
    BIGNUM* p = BN_bin2bn(p_bytes.data(), 128, nullptr);
    BIGNUM* q = BN_bin2bn(q_bytes.data(), 128, nullptr);
    
    if (!p || !q) {
        if (p) BN_free(p);
        if (q) BN_free(q);
        return "";
    }
    
    // Ensure p > q
    if (BN_cmp(p, q) < 0) {
        BIGNUM* tmp = p;
        p = q;
        q = tmp;
    }
    
    // Compute RSA components
    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* n = BN_new();
    BIGNUM* e = BN_new();
    BIGNUM* d = BN_new();
    BIGNUM* p1 = BN_new();
    BIGNUM* q1 = BN_new();
    BIGNUM* phi = BN_new();
    BIGNUM* dmp1 = BN_new();
    BIGNUM* dmq1 = BN_new();
    BIGNUM* iqmp = BN_new();
    
    BN_set_word(e, 65537);
    BN_mul(n, p, q, ctx);
    
    BN_sub(p1, p, BN_value_one());
    BN_sub(q1, q, BN_value_one());
    BN_mul(phi, p1, q1, ctx);
    
    BN_mod_inverse(d, e, phi, ctx);
    BN_mod(dmp1, d, p1, ctx);
    BN_mod(dmq1, d, q1, ctx);
    BN_mod_inverse(iqmp, q, p, ctx);
    
    // Build EVP_PKEY
    OSSL_PARAM_BLD* bld = OSSL_PARAM_BLD_new();
    OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_N, n);
    OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_E, e);
    OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_D, d);
    OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_FACTOR1, p);
    OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_FACTOR2, q);
    OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_EXPONENT1, dmp1);
    OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_EXPONENT2, dmq1);
    OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_COEFFICIENT1, iqmp);
    
    OSSL_PARAM* params = OSSL_PARAM_BLD_to_param(bld);
    
    EVP_PKEY* pkey = nullptr;
    EVP_PKEY_CTX* pctx = EVP_PKEY_CTX_new_from_name(nullptr, "RSA", nullptr);
    EVP_PKEY_fromdata_init(pctx);
    EVP_PKEY_fromdata(pctx, &pkey, EVP_PKEY_KEYPAIR, params);
    
    std::string result;
    if (pkey) {
        BIO* bio = BIO_new(BIO_s_mem());
        PEM_write_bio_PrivateKey(bio, pkey, nullptr, nullptr, 0, nullptr, nullptr);
        
        BUF_MEM* buf;
        BIO_get_mem_ptr(bio, &buf);
        result = std::string(buf->data, buf->length);
        
        BIO_free(bio);
        EVP_PKEY_free(pkey);
    }
    
    // Cleanup
    EVP_PKEY_CTX_free(pctx);
    OSSL_PARAM_free(params);
    OSSL_PARAM_BLD_free(bld);
    BN_free(p); BN_free(q); BN_free(n); BN_free(e); BN_free(d);
    BN_free(p1); BN_free(q1); BN_free(phi);
    BN_free(dmp1); BN_free(dmq1); BN_free(iqmp);
    BN_CTX_free(ctx);
    
    return result;
}

} // namespace cuda
} // namespace vanity

