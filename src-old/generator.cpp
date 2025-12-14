#include "generator.hpp"

#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/err.h>
#include <openssl/bn.h>
#include <openssl/core_names.h>
#include <openssl/param_build.h>

#include <cstring>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace vanity {

bool validate_target(const std::string& target) {
    if (target.empty()) {
        return false;
    }
    for (char c : target) {
        if (c < 'a' || c > 'p') {
            return false;
        }
    }
    return true;
}

MatchPosition parse_position(const std::string& pos_str) {
    if (pos_str == "start") return MatchPosition::Start;
    if (pos_str == "end") return MatchPosition::End;
    return MatchPosition::Anywhere;
}

const char* position_to_string(MatchPosition pos) {
    switch (pos) {
        case MatchPosition::Start: return "start";
        case MatchPosition::End: return "end";
        case MatchPosition::Anywhere: return "anywhere";
    }
    return "anywhere";
}

bool compute_extension_id(EVP_PKEY* pkey, char* ext_id) {
    // Get the public key in SPKI/DER format
    unsigned char* der_buf = nullptr;
    int der_len = i2d_PUBKEY(pkey, &der_buf);
    
    if (der_len <= 0 || der_buf == nullptr) {
        return false;
    }
    
    // SHA-256 hash of the DER-encoded public key
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(der_buf, static_cast<size_t>(der_len), hash);
    
    // Free the DER buffer (OpenSSL allocated it)
    OPENSSL_free(der_buf);
    
    // Convert first 16 bytes to extension ID (a-p encoding)
    // Each byte becomes two characters: high nibble + low nibble
    for (int i = 0; i < 16; i++) {
        ext_id[i * 2]     = 'a' + (hash[i] >> 4);
        ext_id[i * 2 + 1] = 'a' + (hash[i] & 0x0f);
    }
    ext_id[32] = '\0';
    
    return true;
}

__attribute__((hot))
bool check_match(const char* ext_id, const SearchConfig& config) {
    const size_t ext_len = 32;
    const size_t target_len = config.target.length();
    
    switch (config.position) {
        case MatchPosition::Start:
            return std::strncmp(ext_id, config.target.c_str(), target_len) == 0;
            
        case MatchPosition::End:
            if (target_len > ext_len) return false;
            return std::strncmp(ext_id + ext_len - target_len, config.target.c_str(), target_len) == 0;
            
        case MatchPosition::Anywhere:
            return std::strstr(ext_id, config.target.c_str()) != nullptr;
            
        case MatchPosition::StartAndEnd: {
            // Must match BOTH start AND end
            const size_t end_len = config.end_target.length();
            if (target_len + end_len > ext_len) return false;
            
            bool start_match = std::strncmp(ext_id, config.target.c_str(), target_len) == 0;
            bool end_match = std::strncmp(ext_id + ext_len - end_len, config.end_target.c_str(), end_len) == 0;
            return start_match && end_match;
        }
    }
    return false;
}

std::string export_private_key_pem(EVP_PKEY* pkey) {
    BIO* bio = BIO_new(BIO_s_mem());
    if (!bio) return "";
    
    if (!PEM_write_bio_PrivateKey(bio, pkey, nullptr, nullptr, 0, nullptr, nullptr)) {
        BIO_free(bio);
        return "";
    }
    
    BUF_MEM* buf = nullptr;
    BIO_get_mem_ptr(bio, &buf);
    
    std::string result(buf->data, buf->length);
    BIO_free(bio);
    
    return result;
}

std::string export_public_key_base64(EVP_PKEY* pkey) {
    // Get public key in DER format
    unsigned char* der_buf = nullptr;
    int der_len = i2d_PUBKEY(pkey, &der_buf);
    
    if (der_len <= 0 || der_buf == nullptr) {
        return "";
    }
    
    // Base64 encode
    BIO* bio = BIO_new(BIO_s_mem());
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);
    
    BIO_write(bio, der_buf, der_len);
    BIO_flush(bio);
    
    OPENSSL_free(der_buf);
    
    BUF_MEM* buf = nullptr;
    BIO_get_mem_ptr(bio, &buf);
    
    std::string result(buf->data, buf->length);
    BIO_free_all(bio);
    
    return result;
}

uint64_t estimate_attempts(const SearchConfig& config) {
    // 16 possible chars per position
    size_t target_length = config.target.length();
    
    uint64_t total_combinations = 1;
    for (size_t i = 0; i < target_length; i++) {
        total_combinations *= 16;
    }
    
    if (config.position == MatchPosition::Anywhere) {
        // More positions to match = higher chance
        size_t positions = 32 - target_length + 1;
        return total_combinations / positions;
    }
    
    if (config.position == MatchPosition::StartAndEnd) {
        // Both start AND end must match - multiply probabilities
        uint64_t end_combinations = 1;
        for (size_t i = 0; i < config.end_target.length(); i++) {
            end_combinations *= 16;
        }
        return total_combinations * end_combinations;
    }
    
    // Start or End: only one position
    return total_combinations;
}

__attribute__((hot))
void worker_loop(const SearchConfig& config, SharedState& state) {
    // Create key generation context once per thread
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    if (!ctx) {
        return;
    }
    
    if (EVP_PKEY_keygen_init(ctx) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return;
    }
    
    if (EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 2048) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return;
    }
    
    char ext_id[33];
    uint64_t local_count = 0;
    constexpr uint64_t BATCH_SIZE = 100;
    
    while (!state.found.load(std::memory_order_relaxed)) {
        EVP_PKEY* pkey = nullptr;
        
        if (EVP_PKEY_keygen(ctx, &pkey) <= 0) {
            // Key generation failed, skip
            continue;
        }
        
        if (compute_extension_id(pkey, ext_id)) {
            if (check_match(ext_id, config)) {
                // Found a match!
                bool expected = false;
                if (state.found.compare_exchange_strong(expected, true, 
                        std::memory_order_release, std::memory_order_relaxed)) {
                    // We're the first to find it
                    std::lock_guard<std::mutex> lock(state.result_mutex);
                    state.result.found = true;
                    state.result.extension_id = ext_id;
                    state.result.private_key_pem = export_private_key_pem(pkey);
                    state.result.public_key_base64 = export_public_key_base64(pkey);
                }
            }
        }
        
        EVP_PKEY_free(pkey);
        
        // Batch progress updates to reduce atomic contention
        if (++local_count % BATCH_SIZE == 0) {
            state.attempts.fetch_add(BATCH_SIZE, std::memory_order_relaxed);
        }
    }
    
    // Add remaining count
    uint64_t remaining = local_count % BATCH_SIZE;
    if (remaining > 0) {
        state.attempts.fetch_add(remaining, std::memory_order_relaxed);
    }
    
    EVP_PKEY_CTX_free(ctx);
}

Dictionary load_dictionary(const std::string& filepath, size_t min_len, size_t max_len) {
    Dictionary dict;
    dict.min_length = min_len;
    dict.max_length = max_len;
    
    // Set default per-length limits:
    // 3-char: 100, 4-char: 200, 5-char: 500, 6+: unlimited
    dict.length_limits.fill(0);  // 0 = unlimited
    dict.length_limits[3] = 100;
    dict.length_limits[4] = 200;
    dict.length_limits[5] = 500;
    // 6+ chars are rare and valuable, no limit
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open dictionary file: " << filepath << "\n";
        return dict;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        
        std::string word = line.substr(start, end - start + 1);
        
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Check length
        if (word.length() < min_len || word.length() > max_len) continue;
        
        // Validate: only a-p characters
        if (!validate_target(word)) continue;
        
        // Add to dictionary
        dict.words.insert(word);
        dict.prefixes.push_back(word);
        dict.suffixes.push_back(word);
    }
    
    // Sort prefixes and suffixes by length (longest first for greedy matching)
    auto by_length_desc = [](const std::string& a, const std::string& b) {
        return a.length() > b.length();
    };
    std::sort(dict.prefixes.begin(), dict.prefixes.end(), by_length_desc);
    std::sort(dict.suffixes.begin(), dict.suffixes.end(), by_length_desc);
    
    return dict;
}

__attribute__((hot))
std::vector<std::pair<std::string, std::string>> check_dictionary(
    const char* ext_id, const Dictionary& dict) {
    
    std::vector<std::pair<std::string, std::string>> matches;
    std::string id_str(ext_id);
    
    // Check for words at start (longest match wins)
    if (dict.check_start) {
        for (const auto& word : dict.prefixes) {
            if (id_str.compare(0, word.length(), word) == 0) {
                matches.emplace_back(word, "start");
                break;  // Only report longest prefix match
            }
        }
    }
    
    // Check for words at end (longest match wins)
    if (dict.check_end) {
        for (const auto& word : dict.suffixes) {
            size_t pos = 32 - word.length();
            if (id_str.compare(pos, word.length(), word) == 0) {
                // Don't duplicate if same word matched at start
                bool already_matched = false;
                for (const auto& m : matches) {
                    if (m.first == word) { already_matched = true; break; }
                }
                if (!already_matched) {
                    matches.emplace_back(word, "end");
                }
                break;  // Only report longest suffix match
            }
        }
    }
    
    // For long words (7+ chars), also check ANYWHERE - these are rare and valuable!
    constexpr size_t LONG_WORD_THRESHOLD = 7;
    for (const auto& word : dict.words) {
        if (word.length() < LONG_WORD_THRESHOLD) continue;
        
        // Skip if already matched at start or end
        bool already_matched = false;
        for (const auto& m : matches) {
            if (m.first == word) { already_matched = true; break; }
        }
        if (already_matched) continue;
        
        // Search for long word anywhere in ID
        size_t pos = id_str.find(word);
        if (pos != std::string::npos) {
            std::string pos_desc;
            if (pos == 0) {
                pos_desc = "start";
            } else if (pos == 32 - word.length()) {
                pos_desc = "end";
            } else {
                pos_desc = "pos:" + std::to_string(pos);
            }
            matches.emplace_back(word, pos_desc);
        }
    }
    
    return matches;
}

CompactKey extract_compact_key(EVP_PKEY* pkey) {
    CompactKey compact;
    
    BIGNUM* p = nullptr;
    BIGNUM* q = nullptr;
    
    // Extract p and q from the key
    if (!EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_FACTOR1, &p) ||
        !EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_FACTOR2, &q)) {
        if (p) BN_free(p);
        if (q) BN_free(q);
        return compact;
    }
    
    // Convert to hex strings
    char* p_hex = BN_bn2hex(p);
    char* q_hex = BN_bn2hex(q);
    
    if (p_hex) {
        compact.p_hex = p_hex;
        OPENSSL_free(p_hex);
    }
    if (q_hex) {
        compact.q_hex = q_hex;
        OPENSSL_free(q_hex);
    }
    
    BN_free(p);
    BN_free(q);
    
    return compact;
}

EVP_PKEY* reconstruct_key(const CompactKey& compact) {
    if (compact.p_hex.empty() || compact.q_hex.empty()) {
        return nullptr;
    }
    
    BIGNUM* p = nullptr;
    BIGNUM* q = nullptr;
    BIGNUM* n = nullptr;
    BIGNUM* e = nullptr;
    BIGNUM* d = nullptr;
    BIGNUM* dmp1 = nullptr;
    BIGNUM* dmq1 = nullptr;
    BIGNUM* iqmp = nullptr;
    BN_CTX* bn_ctx = nullptr;
    EVP_PKEY* pkey = nullptr;
    
    // Parse p and q from hex
    if (!BN_hex2bn(&p, compact.p_hex.c_str()) ||
        !BN_hex2bn(&q, compact.q_hex.c_str())) {
        goto cleanup;
    }
    
    // Allocate other components
    n = BN_new();
    e = BN_new();
    d = BN_new();
    dmp1 = BN_new();
    dmq1 = BN_new();
    iqmp = BN_new();
    bn_ctx = BN_CTX_new();
    
    if (!n || !e || !d || !dmp1 || !dmq1 || !iqmp || !bn_ctx) {
        goto cleanup;
    }
    
    // e = 65537 (standard public exponent)
    BN_set_word(e, 65537);
    
    // n = p * q
    BN_mul(n, p, q, bn_ctx);
    
    // Compute d = e^(-1) mod (p-1)(q-1)
    {
        BIGNUM* p1 = BN_new();
        BIGNUM* q1 = BN_new();
        BIGNUM* phi = BN_new();
        
        BN_sub(p1, p, BN_value_one());
        BN_sub(q1, q, BN_value_one());
        BN_mul(phi, p1, q1, bn_ctx);
        BN_mod_inverse(d, e, phi, bn_ctx);
        
        // dmp1 = d mod (p-1)
        BN_mod(dmp1, d, p1, bn_ctx);
        // dmq1 = d mod (q-1)
        BN_mod(dmq1, d, q1, bn_ctx);
        
        BN_free(p1);
        BN_free(q1);
        BN_free(phi);
    }
    
    // iqmp = q^(-1) mod p
    BN_mod_inverse(iqmp, q, p, bn_ctx);
    
    // Build the key using OSSL_PARAM
    {
        OSSL_PARAM_BLD* bld = OSSL_PARAM_BLD_new();
        if (!bld) goto cleanup;
        
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_N, n);
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_E, e);
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_D, d);
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_FACTOR1, p);
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_FACTOR2, q);
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_EXPONENT1, dmp1);
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_EXPONENT2, dmq1);
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_COEFFICIENT1, iqmp);
        
        OSSL_PARAM* params = OSSL_PARAM_BLD_to_param(bld);
        OSSL_PARAM_BLD_free(bld);
        
        if (!params) goto cleanup;
        
        EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_from_name(nullptr, "RSA", nullptr);
        if (ctx) {
            EVP_PKEY_fromdata_init(ctx);
            EVP_PKEY_fromdata(ctx, &pkey, EVP_PKEY_KEYPAIR, params);
            EVP_PKEY_CTX_free(ctx);
        }
        
        OSSL_PARAM_free(params);
    }
    
cleanup:
    if (p) BN_free(p);
    if (q) BN_free(q);
    if (n) BN_free(n);
    if (e) BN_free(e);
    if (d) BN_free(d);
    if (dmp1) BN_free(dmp1);
    if (dmq1) BN_free(dmq1);
    if (iqmp) BN_free(iqmp);
    if (bn_ctx) BN_CTX_free(bn_ctx);
    
    return pkey;
}

std::string compact_to_pem(const CompactKey& compact) {
    EVP_PKEY* pkey = reconstruct_key(compact);
    if (!pkey) return "";
    
    std::string pem = export_private_key_pem(pkey);
    EVP_PKEY_free(pkey);
    return pem;
}

bool StateFile::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        
        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        
        if (key == "total_attempts") {
            total_attempts = std::stoull(value);
        } else if (key == "total_matches") {
            total_matches = std::stoull(value);
        } else if (key.rfind("len_", 0) == 0) {
            // len_3=100, len_4=200, etc.
            size_t len = std::stoul(key.substr(4));
            if (len < length_counts.size()) {
                length_counts[len] = std::stoull(value);
            }
        }
    }
    
    return true;
}

bool StateFile::save(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# Vanity Extension ID Generator - State File\n";
    file << "# Do not edit manually\n";
    file << "total_attempts=" << total_attempts << "\n";
    file << "total_matches=" << total_matches << "\n";
    
    for (size_t i = 3; i < length_counts.size(); i++) {
        if (length_counts[i] > 0) {
            file << "len_" << i << "=" << length_counts[i] << "\n";
        }
    }
    
    file.flush();
    return true;
}

std::string detect_cool_pattern(const char* ext_id) {
    std::string id(ext_id);
    std::vector<std::string> patterns;
    
    // 1. Check for repeated characters (6+ in a row) - rare!
    for (size_t i = 0; i < 32; ) {
        char c = id[i];
        size_t count = 1;
        while (i + count < 32 && id[i + count] == c) count++;
        
        if (count >= 6) {
            patterns.push_back(std::string(1, c) + "*" + std::to_string(count));
        }
        i += count;
    }
    
    // 2. Check for ascending sequences starting from 'a' only (5+ chars like abcde)
    for (size_t i = 0; i < 28; i++) {
        if (id[i] != 'a') continue;  // Only care about sequences starting with 'a'
        
        size_t len = 1;
        while (i + len < 32 && id[i + len] == id[i + len - 1] + 1) len++;
        
        if (len >= 5) {
            patterns.push_back("seq:" + id.substr(i, len));
        }
    }
    
    // 3. Check for descending sequences ending at 'a' (5+ chars like edcba)
    for (size_t i = 0; i < 28; i++) {
        size_t len = 1;
        while (i + len < 32 && id[i + len] == id[i + len - 1] - 1) len++;
        
        // Only keep if it ends at 'a' (like "edcba", "dcba")
        if (len >= 5 && id[i + len - 1] == 'a') {
            patterns.push_back("seq:" + id.substr(i, len));
        }
    }
    
    // 4. Check for palindrome at start (8+ chars)
    for (size_t len = 12; len >= 8; len--) {
        bool is_palindrome = true;
        for (size_t j = 0; j < len / 2; j++) {
            if (id[j] != id[len - 1 - j]) {
                is_palindrome = false;
                break;
            }
        }
        if (is_palindrome) {
            patterns.push_back("palindrome:" + id.substr(0, len));
            break;
        }
    }
    
    // 5. Check for palindrome at end (8+ chars)
    for (size_t len = 12; len >= 8; len--) {
        size_t start = 32 - len;
        bool is_palindrome = true;
        for (size_t j = 0; j < len / 2; j++) {
            if (id[start + j] != id[start + len - 1 - j]) {
                is_palindrome = false;
                break;
            }
        }
        if (is_palindrome) {
            patterns.push_back("palindrome:" + id.substr(start, len));
            break;
        }
    }
    
    // 6. Check for repeated 2-3 char patterns (4+ reps like abababab, abcabcabcabc)
    for (size_t plen = 2; plen <= 3; plen++) {
        std::string pat = id.substr(0, plen);
        size_t reps = 1;
        for (size_t i = plen; i + plen <= 32 && id.substr(i, plen) == pat; i += plen) {
            reps++;
        }
        if (reps >= 4) {
            patterns.push_back(pat + "*" + std::to_string(reps));
        }
    }
    
    // 7. Check for same char prefix/suffix (5+)
    size_t prefix_count = 1;
    while (prefix_count < 32 && id[prefix_count] == id[0]) prefix_count++;
    if (prefix_count >= 5) {
        patterns.push_back("prefix:" + std::string(prefix_count, id[0]));
    }
    
    size_t suffix_count = 1;
    while (suffix_count < 32 && id[31 - suffix_count] == id[31]) suffix_count++;
    if (suffix_count >= 5) {
        patterns.push_back("suffix:" + std::string(suffix_count, id[31]));
    }
    
    // Build result string
    if (patterns.empty()) return "";
    
    std::string result;
    for (size_t i = 0; i < patterns.size(); i++) {
        if (i > 0) result += ";";
        result += patterns[i];
    }
    return result;
}

void worker_loop_dict(const Dictionary& dict, SharedState& state,
                      const std::string& output_file, std::mutex& file_mutex) {
    // Create key generation context once per thread
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    if (!ctx) {
        return;
    }
    
    if (EVP_PKEY_keygen_init(ctx) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return;
    }
    
    if (EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 2048) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return;
    }
    
    char ext_id[33];
    uint64_t local_count = 0;
    constexpr uint64_t BATCH_SIZE = 100;
    
    while (!state.stop.load(std::memory_order_relaxed)) {
        EVP_PKEY* pkey = nullptr;
        
        if (EVP_PKEY_keygen(ctx, &pkey) <= 0) {
            continue;
        }
        
        if (compute_extension_id(pkey, ext_id)) {
            auto matches = check_dictionary(ext_id, dict);
            
            // Also check for cool patterns
            std::string cool_pattern = detect_cool_pattern(ext_id);
            if (!cool_pattern.empty()) {
                matches.emplace_back("PATTERN", cool_pattern);
            }
            
            if (!matches.empty()) {
                // Find the longest match to determine if we should save
                size_t longest_match_len = 0;
                for (const auto& m : matches) {
                    if (m.first.length() > longest_match_len) {
                        longest_match_len = m.first.length();
                    }
                }
                
                // Check if we've hit the limit for this word length
                uint64_t limit = dict.length_limits[longest_match_len];
                if (limit > 0) {
                    uint64_t current = state.length_counts[longest_match_len].load(std::memory_order_relaxed);
                    if (current >= limit) {
                        // Skip this match - we have enough of this length
                        EVP_PKEY_free(pkey);
                        if (++local_count % BATCH_SIZE == 0) {
                            state.attempts.fetch_add(BATCH_SIZE, std::memory_order_relaxed);
                        }
                        continue;
                    }
                    // Increment counter for this length
                    state.length_counts[longest_match_len].fetch_add(1, std::memory_order_relaxed);
                }
                
                // Extract compact key (just p and q) - ~256 bytes vs ~1700 bytes
                CompactKey compact = extract_compact_key(pkey);
                std::string pub_b64 = export_public_key_base64(pkey);
                
                // Build match description
                std::ostringstream match_desc;
                for (size_t i = 0; i < matches.size(); i++) {
                    if (i > 0) match_desc << ",";
                    match_desc << matches[i].first << "@" << matches[i].second;
                }
                
                // Write compact format to CSV file
                {
                    std::lock_guard<std::mutex> lock(file_mutex);
                    std::ofstream out(output_file, std::ios::app);
                    if (out.is_open()) {
                        // CSV format: ext_id,matches,p_hex,q_hex,pub_b64
                        out << ext_id << ","
                            << match_desc.str() << ","
                            << compact.p_hex << ","
                            << compact.q_hex << ","
                            << pub_b64 << "\n";
                        out.flush();
                    }
                }
                
                // Update match counter
                state.matches_found.fetch_add(1, std::memory_order_relaxed);
                
                // Store in memory for final report (limited to avoid memory issues)
                if (state.dict_matches.size() < 10000) {
                    std::lock_guard<std::mutex> lock(state.dict_mutex);
                    DictMatch dm;
                    dm.extension_id = ext_id;
                    dm.compact_key = compact;
                    dm.public_key_base64 = pub_b64;
                    dm.matches = matches;
                    state.dict_matches.push_back(std::move(dm));
                }
            }
        }
        
        EVP_PKEY_free(pkey);
        
        if (++local_count % BATCH_SIZE == 0) {
            state.attempts.fetch_add(BATCH_SIZE, std::memory_order_relaxed);
        }
    }
    
    uint64_t remaining = local_count % BATCH_SIZE;
    if (remaining > 0) {
        state.attempts.fetch_add(remaining, std::memory_order_relaxed);
    }
    
    EVP_PKEY_CTX_free(ctx);
}

} // namespace vanity

