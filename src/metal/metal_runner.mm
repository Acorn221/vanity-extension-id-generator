/**
 * Metal GPU Runner Implementation
 * 
 * Uses Metal compute shaders to search prime combinations in parallel.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_runner.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <atomic>
#include <chrono>
#include <set>
#include <unordered_set>

// OpenSSL for key reconstruction
#include <openssl/bn.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
#include <openssl/param_build.h>
#include <openssl/core_names.h>

namespace vanity {
namespace metal {

// Prime pool header structure
struct PrimePoolHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t count;
    uint32_t prime_bytes;
};

// Match result from shader (must match Metal struct)
struct ShaderMatchResult {
    uint32_t prime_idx_p;
    uint32_t prime_idx_q;
    char ext_id[36];
    uint32_t match_type;
};

class MetalRunner::Impl {
public:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> commandQueue_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState> searchPipeline_ = nil;
    id<MTLComputePipelineState> dictPipeline_ = nil;
    
    id<MTLBuffer> primePoolBuffer_ = nil;
    size_t primeCount_ = 0;
    
    std::atomic<bool> running_{false};
    std::atomic<bool> stopRequested_{false};
    bool quietMode_ = false;  // Suppress per-search logs in dictionary mode

    std::string deviceName_;
    
    Impl() {
        // Get default Metal device
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            std::cerr << "Metal: No GPU device found\n";
            return;
        }
        
        deviceName_ = [device_.name UTF8String];
        
        // Create command queue
        commandQueue_ = [device_ newCommandQueue];
        if (!commandQueue_) {
            std::cerr << "Metal: Failed to create command queue\n";
            return;
        }
        
        // Load shader library
        NSError* error = nil;
        
        // Try to load from default library first (compiled into app)
        library_ = [device_ newDefaultLibrary];
        
        if (!library_) {
            // Try to compile from source file
            NSString* shaderPath = @"src/metal/vanity_search.metal";
            NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                               encoding:NSUTF8StringEncoding
                                                                  error:&error];
            if (!shaderSource) {
                // Try relative to executable
                shaderPath = @"../src/metal/vanity_search.metal";
                shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            }
            
            if (shaderSource) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                library_ = [device_ newLibraryWithSource:shaderSource options:options error:&error];
            }
        }
        
        if (!library_) {
            std::cerr << "Metal: Failed to load shader library\n";
            if (error) {
                std::cerr << "  Error: " << [[error localizedDescription] UTF8String] << "\n";
            }
            return;
        }
        
        // Create compute pipelines
        id<MTLFunction> searchFunc = [library_ newFunctionWithName:@"vanity_search"];
        if (searchFunc) {
            searchPipeline_ = [device_ newComputePipelineStateWithFunction:searchFunc error:&error];
            if (!searchPipeline_ && error) {
                std::cerr << "Metal: Failed to create search pipeline: "
                          << [[error localizedDescription] UTF8String] << "\n";
            }
        }
        
        id<MTLFunction> dictFunc = [library_ newFunctionWithName:@"vanity_search_dict"];
        if (dictFunc) {
            dictPipeline_ = [device_ newComputePipelineStateWithFunction:dictFunc error:&error];
        }
        
        std::cout << "Metal: Initialized on " << deviceName_ << "\n";
    }
    
    ~Impl() {
        // ARC handles cleanup
    }
    
    bool isAvailable() const {
        return device_ != nil && commandQueue_ != nil && searchPipeline_ != nil;
    }
    
    size_t loadPrimePool(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Metal: Failed to open prime pool: " << filepath << "\n";
            return 0;
        }
        
        // Read header
        PrimePoolHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        
        if (header.magic != 0x504D5250) {  // 'PRMP'
            std::cerr << "Metal: Invalid prime pool file (bad magic)\n";
            return 0;
        }
        
        if (header.prime_bytes != 128) {
            std::cerr << "Metal: Unexpected prime size: " << header.prime_bytes << "\n";
            return 0;
        }
        
        // Allocate Metal buffer
        size_t dataSize = header.count * header.prime_bytes;
        primePoolBuffer_ = [device_ newBufferWithLength:dataSize
                                                options:MTLResourceStorageModeShared];
        
        if (!primePoolBuffer_) {
            std::cerr << "Metal: Failed to allocate GPU buffer (" << dataSize << " bytes)\n";
            return 0;
        }
        
        // Read prime data
        file.read(static_cast<char*>([primePoolBuffer_ contents]), dataSize);
        
        primeCount_ = header.count;
        
        std::cout << "Metal: Loaded " << primeCount_ << " primes ("
                  << (dataSize / 1024 / 1024) << " MB)\n";
        
        return primeCount_;
    }
    
    std::vector<GPUMatch> search(const SearchConfig& config, ProgressCallback callback) {
        std::vector<GPUMatch> results;
        
        if (!isAvailable() || primeCount_ == 0) {
            return results;
        }
        
        running_ = true;
        stopRequested_ = false;
        
        // Create buffers for targets
        id<MTLBuffer> prefixBuffer = [device_ newBufferWithLength:config.target_prefix.size() + 1
                                                          options:MTLResourceStorageModeShared];
        memcpy([prefixBuffer contents], config.target_prefix.c_str(), config.target_prefix.size() + 1);
        
        id<MTLBuffer> suffixBuffer = [device_ newBufferWithLength:config.target_suffix.size() + 1
                                                          options:MTLResourceStorageModeShared];
        memcpy([suffixBuffer contents], config.target_suffix.c_str(), config.target_suffix.size() + 1);
        
        // Parameters buffer
        uint32_t params[4] = {
            static_cast<uint32_t>(primeCount_),
            static_cast<uint32_t>(config.target_prefix.size()),
            static_cast<uint32_t>(config.target_suffix.size()),
            0  // start_idx
        };
        id<MTLBuffer> paramsBuffer = [device_ newBufferWithBytes:params
                                                          length:sizeof(params)
                                                         options:MTLResourceStorageModeShared];
        
        // Match count buffer (atomic)
        id<MTLBuffer> matchCountBuffer = [device_ newBufferWithLength:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        memset([matchCountBuffer contents], 0, sizeof(uint32_t));
        
        // Matches buffer
        size_t maxMatches = 10000;
        id<MTLBuffer> matchesBuffer = [device_ newBufferWithLength:maxMatches * sizeof(ShaderMatchResult)
                                                           options:MTLResourceStorageModeShared];
        
        // Calculate total pairs
        uint64_t totalPairs = (uint64_t)primeCount_ * (primeCount_ - 1) / 2;
        uint64_t pairsChecked = 0;
        
        auto startTime = std::chrono::steady_clock::now();
        
        // Process in batches
        uint32_t batchSize = config.batch_size;
        
        for (uint32_t startIdx = 0; startIdx < primeCount_ && !stopRequested_; startIdx += batchSize) {
            @autoreleasepool {
                // Update start index
                uint32_t* paramsPtr = static_cast<uint32_t*>([paramsBuffer contents]);
                paramsPtr[3] = startIdx;
                
                // Create command buffer
                id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                
                [encoder setComputePipelineState:searchPipeline_];
                [encoder setBuffer:primePoolBuffer_ offset:0 atIndex:0];
                [encoder setBuffer:prefixBuffer offset:0 atIndex:1];
                [encoder setBuffer:suffixBuffer offset:0 atIndex:2];
                [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
                [encoder setBuffer:matchCountBuffer offset:0 atIndex:4];
                [encoder setBuffer:matchesBuffer offset:0 atIndex:5];
                
                // Calculate grid size (2D: i × j pairs)
                uint32_t gridWidth = MIN(batchSize, primeCount_ - startIdx);
                uint32_t gridHeight = primeCount_ - startIdx;
                
                MTLSize gridSize = MTLSizeMake(gridWidth, gridHeight, 1);
                MTLSize threadGroupSize = MTLSizeMake(
                    MIN(32u, gridWidth),
                    MIN(32u, gridHeight),
                    1
                );
                
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
                [encoder endEncoding];
                
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                
                // Update progress
                pairsChecked += (uint64_t)gridWidth * gridHeight / 2;
                
                uint32_t matchCount = *static_cast<uint32_t*>([matchCountBuffer contents]);
                
                if (callback) {
                    callback(pairsChecked, totalPairs, matchCount);
                }
            }
        }
        
        // Collect results
        uint32_t matchCount = *static_cast<uint32_t*>([matchCountBuffer contents]);
        ShaderMatchResult* matchData = static_cast<ShaderMatchResult*>([matchesBuffer contents]);
        
        for (uint32_t i = 0; i < matchCount && i < maxMatches; i++) {
            GPUMatch match;
            match.prime_idx_p = matchData[i].prime_idx_p;
            match.prime_idx_q = matchData[i].prime_idx_q;
            match.extension_id = std::string(matchData[i].ext_id);
            match.match_type = matchData[i].match_type;
            results.push_back(match);
        }
        
        auto endTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(endTime - startTime).count();

        if (!quietMode_) {
            std::cout << "Metal: Search complete. " << pairsChecked << " pairs in "
                      << elapsed << "s (" << (pairsChecked / elapsed / 1e6) << "M pairs/s)\n";
        }
        
        running_ = false;
        return results;
    }
    
    std::vector<GPUMatch> searchDictionary(
        const std::vector<std::string>& words,
        size_t min_len,
        ProgressCallback callback
    ) {
        std::vector<GPUMatch> results;

        if (!isAvailable() || primeCount_ == 0 || !dictPipeline_) {
            std::cerr << "Metal: Dictionary pipeline not available\n";
            return results;
        }

        running_ = true;
        stopRequested_ = false;

        // Filter words by min length and build packed dictionary buffer
        std::vector<std::string> validWords;
        for (const auto& word : words) {
            if (word.size() >= min_len && word.size() <= 10) {
                validWords.push_back(word);
            }
        }

        std::cout << "Dictionary: " << validWords.size() << " valid words (min " << min_len << " chars)\n";

        // Pack dictionary: null-separated words
        std::vector<char> dictData;
        std::vector<uint32_t> wordOffsets;

        for (const auto& word : validWords) {
            wordOffsets.push_back(static_cast<uint32_t>(dictData.size()));
            for (char c : word) {
                dictData.push_back(c);
            }
            dictData.push_back('\0');
        }
        wordOffsets.push_back(static_cast<uint32_t>(dictData.size()));  // End marker

        std::cout << "Dictionary buffer: " << dictData.size() << " bytes, "
                  << wordOffsets.size() - 1 << " words\n";

        // Create GPU buffers for dictionary
        id<MTLBuffer> dictBuffer = [device_ newBufferWithBytes:dictData.data()
                                                        length:dictData.size()
                                                       options:MTLResourceStorageModeShared];

        id<MTLBuffer> offsetsBuffer = [device_ newBufferWithBytes:wordOffsets.data()
                                                           length:wordOffsets.size() * sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];

        // Parameters: [pool_size, num_words, min_len, start_idx]
        uint32_t params[4] = {
            static_cast<uint32_t>(primeCount_),
            static_cast<uint32_t>(validWords.size()),
            static_cast<uint32_t>(min_len),
            0  // start_idx
        };
        id<MTLBuffer> paramsBuffer = [device_ newBufferWithBytes:params
                                                          length:sizeof(params)
                                                         options:MTLResourceStorageModeShared];

        // Match count buffer (atomic)
        id<MTLBuffer> matchCountBuffer = [device_ newBufferWithLength:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        memset([matchCountBuffer contents], 0, sizeof(uint32_t));

        // Matches buffer - needs to be large enough for all potential matches
        // With 10k primes and 50M pairs, dictionary can find millions of matches
        size_t maxMatches = 10000000;  // 10M matches max
        id<MTLBuffer> matchesBuffer = [device_ newBufferWithLength:maxMatches * sizeof(ShaderMatchResult)
                                                           options:MTLResourceStorageModeShared];

        if (!matchesBuffer) {
            // If 10M is too big, try smaller
            maxMatches = 1000000;
            matchesBuffer = [device_ newBufferWithLength:maxMatches * sizeof(ShaderMatchResult)
                                                 options:MTLResourceStorageModeShared];
        }
        std::cout << "Match buffer: " << (maxMatches * sizeof(ShaderMatchResult) / 1024 / 1024) << " MB (max " << maxMatches << " matches)\n";

        uint64_t totalPairs = (uint64_t)primeCount_ * (primeCount_ - 1) / 2;

        auto startTime = std::chrono::steady_clock::now();

        // Process in batches - use larger batch for better GPU utilization
        uint32_t batchSize = 2048;

        for (uint32_t startIdx = 0; startIdx < primeCount_ && !stopRequested_; startIdx += batchSize) {
            @autoreleasepool {
                // Update start index
                uint32_t* paramsPtr = static_cast<uint32_t*>([paramsBuffer contents]);
                paramsPtr[3] = startIdx;

                // Create command buffer
                id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                [encoder setComputePipelineState:dictPipeline_];
                [encoder setBuffer:primePoolBuffer_ offset:0 atIndex:0];
                [encoder setBuffer:dictBuffer offset:0 atIndex:1];
                [encoder setBuffer:offsetsBuffer offset:0 atIndex:2];
                [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
                [encoder setBuffer:matchCountBuffer offset:0 atIndex:4];
                [encoder setBuffer:matchesBuffer offset:0 atIndex:5];

                // Calculate grid size (2D: i × j pairs)
                uint32_t gridWidth = MIN(batchSize, static_cast<uint32_t>(primeCount_) - startIdx);
                uint32_t gridHeight = static_cast<uint32_t>(primeCount_) - startIdx;

                MTLSize gridSize = MTLSizeMake(gridWidth, gridHeight, 1);
                MTLSize threadGroupSize = MTLSizeMake(
                    MIN(32u, gridWidth),
                    MIN(32u, gridHeight),
                    1
                );

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                // Calculate progress based on how many "rows" we've completed
                // Each row i processes pairs (i, i+1), (i, i+2), ..., (i, n-1)
                // So row i has (n-1-i) pairs. Rows 0..startIdx-1 are complete.
                uint32_t endIdx = MIN(startIdx + batchSize, static_cast<uint32_t>(primeCount_));
                uint64_t pairsChecked = 0;
                for (uint32_t row = 0; row < endIdx; row++) {
                    pairsChecked += (primeCount_ - 1 - row);
                }

                uint32_t matchCount = *static_cast<uint32_t*>([matchCountBuffer contents]);

                if (callback) {
                    callback(pairsChecked, totalPairs, matchCount);
                }
            }
        }

        // Collect results
        uint32_t matchCount = *static_cast<uint32_t*>([matchCountBuffer contents]);
        ShaderMatchResult* matchData = static_cast<ShaderMatchResult*>([matchesBuffer contents]);

        std::cout << "\nCollecting " << matchCount << " GPU matches";
        if (matchCount > maxMatches) {
            std::cout << " (capped at " << maxMatches << ")";
        }
        std::cout << "...\n";

        size_t collectCount = std::min(static_cast<size_t>(matchCount), maxMatches);
        for (size_t i = 0; i < collectCount; i++) {
            GPUMatch match;
            match.prime_idx_p = matchData[i].prime_idx_p;
            match.prime_idx_q = matchData[i].prime_idx_q;
            match.extension_id = std::string(matchData[i].ext_id);
            match.match_type = matchData[i].match_type;
            results.push_back(match);
        }

        auto endTime = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(endTime - startTime).count();

        std::cout << "Metal: Dictionary search complete. " << totalPairs << " pairs in "
                  << elapsed << "s (" << (totalPairs / elapsed / 1e6) << "M pairs/s)\n";
        std::cout << "Found " << results.size() << " matches\n";

        running_ = false;
        return results;
    }
    
    void stop() {
        stopRequested_ = true;
    }
    
    bool isRunning() const {
        return running_;
    }
};

// Public interface implementation

MetalRunner::MetalRunner() : impl_(new Impl()) {}

MetalRunner::~MetalRunner() {
    delete impl_;
}

bool MetalRunner::isAvailable() const {
    return impl_->isAvailable();
}

std::string MetalRunner::getDeviceName() const {
    return impl_->deviceName_;
}

size_t MetalRunner::loadPrimePool(const std::string& filepath) {
    return impl_->loadPrimePool(filepath);
}

size_t MetalRunner::getPrimeCount() const {
    return impl_->primeCount_;
}

uint64_t MetalRunner::getTotalPairs() const {
    size_t n = impl_->primeCount_;
    return (uint64_t)n * (n - 1) / 2;
}

std::vector<GPUMatch> MetalRunner::search(const SearchConfig& config, ProgressCallback callback) {
    return impl_->search(config, callback);
}

std::vector<GPUMatch> MetalRunner::searchDictionary(
    const std::vector<std::string>& words,
    size_t min_len,
    ProgressCallback callback
) {
    return impl_->searchDictionary(words, min_len, callback);
}

void MetalRunner::stop() {
    impl_->stop();
}

bool MetalRunner::isRunning() const {
    return impl_->isRunning();
}

// Key reconstruction from prime indices
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

} // namespace metal
} // namespace vanity
