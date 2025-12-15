/**
 * Vanity Extension ID Generator - CUDA Multi-GPU Version
 * 
 * Uses pre-computed prime pool + CUDA GPU acceleration to search
 * for vanity Chrome extension IDs on 8x A100 GPUs.
 * 
 * Features:
 *   - Multi-GPU support (automatically uses all available GPUs)
 *   - Dictionary mode with wordlist
 *   - Pattern detection (cool patterns)
 *   - Prime validation (filters out invalid primes from pool)
 *   - Per-length limits on matches
 *   - CSV output with compact key format
 *   - State file for resuming
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <csignal>
#include <cstring>
#include <unordered_set>
#include <algorithm>
#include <array>
#include <thread>
#include <mutex>
#include <atomic>

#include "cuda/cuda_runner.h"
#include "generator.hpp"

using namespace vanity;
using namespace vanity::cuda;

// Global runner for signal handling
static CudaRunner* g_runner = nullptr;
static std::atomic<bool> g_interrupted{false};

void signal_handler(int sig) {
    (void)sig;
    std::cout << "\n\nReceived interrupt signal. Stopping gracefully...\n";
    g_interrupted = true;
    if (g_runner) {
        g_runner->stop();
    }
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " -p POOL_FILE [OPTIONS]\n\n"
              << "Search for vanity Chrome extension IDs using CUDA GPU acceleration.\n"
              << "Supports multiple GPUs for maximum throughput.\n\n"
              << "Required:\n"
              << "  -p, --pool FILE     Prime pool file (from generate-prime-pool)\n\n"
              << "Search Options:\n"
              << "  -d, --dict FILE     Search for words from dictionary file\n"
              << "  -s, --start STR     Find IDs starting with STR\n"
              << "  -e, --end STR       Find IDs ending with STR\n\n"
              << "Output Options:\n"
              << "  -o, --output FILE   Output CSV file (default: cuda_results.csv)\n"
              << "  --min-len N         Minimum word length for dictionary (default: 3)\n"
              << "  --max-len N         Maximum word length for dictionary (default: 10)\n\n"
              << "Validation Options:\n"
              << "  --validate          Validate primes before saving (recommended for 100M pool)\n"
              << "  --no-validate       Skip prime validation (faster but may include invalid keys)\n\n"
              << "Other Options:\n"
              << "  -h, --help          Show this help message\n\n"
              << "Examples:\n"
              << "  " << prog_name << " -p pool.bin -d wordlist.txt -o results.csv --validate\n"
              << "  " << prog_name << " -p pool.bin -s cafe\n"
              << "  " << prog_name << " -p pool.bin -s cia -e fbi\n\n"
              << "Output CSV format: ext_id,matches,p_idx,q_idx,validated\n";
}

std::string format_number(uint64_t n) {
    std::string s = std::to_string(n);
    int insert_pos = static_cast<int>(s.length()) - 3;
    while (insert_pos > 0) {
        s.insert(insert_pos, ",");
        insert_pos -= 3;
    }
    return s;
}

std::string format_time(double seconds) {
    if (seconds < 60.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << seconds << "s";
        return oss.str();
    } else if (seconds < 3600.0) {
        int mins = static_cast<int>(seconds / 60);
        int secs = static_cast<int>(seconds) % 60;
        return std::to_string(mins) + "m " + std::to_string(secs) + "s";
    } else if (seconds < 86400.0) {
        int hours = static_cast<int>(seconds / 3600);
        int mins = (static_cast<int>(seconds) % 3600) / 60;
        return std::to_string(hours) + "h " + std::to_string(mins) + "m";
    } else {
        int days = static_cast<int>(seconds / 86400);
        int hours = (static_cast<int>(seconds) % 86400) / 3600;
        return std::to_string(days) + "d " + std::to_string(hours) + "h";
    }
}

// =============================================================================
// Dictionary and Pattern Logic
// =============================================================================

struct GPUDictionary {
    std::unordered_set<std::string> words;
    std::vector<std::string> prefixes;  // Sorted by length desc
    std::vector<std::string> suffixes;  // Sorted by length desc
    size_t min_length = 3;
    size_t max_length = 10;
    
    // Per-length limits
    std::array<uint64_t, 33> length_limits{};
    std::array<uint64_t, 33> length_counts{};
    
    bool empty() const { return words.empty(); }
    size_t size() const { return words.size(); }
};

GPUDictionary load_dictionary_gpu(const std::string& filepath, size_t min_len, size_t max_len) {
    GPUDictionary dict;
    dict.min_length = min_len;
    dict.max_length = max_len;
    
    // Default per-length limits
    dict.length_limits.fill(0);  // 0 = unlimited
    dict.length_limits[3] = 100;
    dict.length_limits[4] = 200;
    dict.length_limits[5] = 500;
    // 6+ chars are rare and valuable, no limit
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dictionary file: " << filepath << "\n";
        return dict;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        
        std::string word = line.substr(start, end - start + 1);
        
        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        // Check length and valid characters
        if (word.length() < min_len || word.length() > max_len) continue;
        if (!vanity::validate_target(word)) continue;
        
        dict.words.insert(word);
        dict.prefixes.push_back(word);
        dict.suffixes.push_back(word);
    }
    
    // Sort by length descending
    auto by_length_desc = [](const std::string& a, const std::string& b) {
        return a.length() > b.length();
    };
    std::sort(dict.prefixes.begin(), dict.prefixes.end(), by_length_desc);
    std::sort(dict.suffixes.begin(), dict.suffixes.end(), by_length_desc);
    
    return dict;
}

// Pattern detection
std::string detect_cool_pattern_cpu(const std::string& id) {
    std::vector<std::string> patterns;

    if (id.length() != 32) {
        return "";
    }

    // 1. Repeated characters (6+ in a row)
    for (size_t i = 0; i < 32; ) {
        char c = id[i];
        size_t count = 1;
        while (i + count < 32 && id[i + count] == c) count++;
        if (count >= 6) {
            patterns.push_back(std::string(1, c) + "*" + std::to_string(count));
        }
        i += count;
    }
    
    // 2. Ascending sequences from 'a' (5+)
    for (size_t i = 0; i < 28; i++) {
        if (id[i] != 'a') continue;
        size_t len = 1;
        while (i + len < 32 && id[i + len] == id[i + len - 1] + 1) len++;
        if (len >= 5) {
            patterns.push_back("seq:" + id.substr(i, len));
        }
    }
    
    // 3. Descending sequences to 'a' (5+)
    for (size_t i = 0; i < 28; i++) {
        size_t len = 1;
        while (i + len < 32 && id[i + len] == id[i + len - 1] - 1) len++;
        if (len >= 5 && id[i + len - 1] == 'a') {
            patterns.push_back("seq:" + id.substr(i, len));
        }
    }
    
    // 4. Palindrome at start (8+)
    for (size_t len = 12; len >= 8; len--) {
        bool is_palindrome = true;
        for (size_t j = 0; j < len / 2; j++) {
            if (id[j] != id[len - 1 - j]) { is_palindrome = false; break; }
        }
        if (is_palindrome) {
            patterns.push_back("palindrome:" + id.substr(0, len));
            break;
        }
    }
    
    // 5. Same char prefix/suffix (5+)
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
    
    if (patterns.empty()) return "";
    
    std::string result;
    for (size_t i = 0; i < patterns.size(); i++) {
        if (i > 0) result += ";";
        result += patterns[i];
    }
    return result;
}

// =============================================================================
// State File
// =============================================================================

struct GPUStateFile {
    uint64_t total_pairs_checked = 0;
    uint64_t total_matches = 0;
    uint64_t validated_matches = 0;
    uint64_t invalid_matches = 0;
    std::array<uint64_t, 33> length_counts{};
    
    bool load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            
            std::string key = line.substr(0, eq);
            std::string value = line.substr(eq + 1);
            
            if (key == "total_pairs") total_pairs_checked = std::stoull(value);
            else if (key == "total_matches") total_matches = std::stoull(value);
            else if (key == "validated_matches") validated_matches = std::stoull(value);
            else if (key == "invalid_matches") invalid_matches = std::stoull(value);
            else if (key.rfind("len_", 0) == 0) {
                size_t len = std::stoul(key.substr(4));
                if (len < length_counts.size()) length_counts[len] = std::stoull(value);
            }
        }
        return true;
    }
    
    bool save(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) return false;
        
        file << "# Vanity Extension ID Generator (CUDA) - State File\n";
        file << "total_pairs=" << total_pairs_checked << "\n";
        file << "total_matches=" << total_matches << "\n";
        file << "validated_matches=" << validated_matches << "\n";
        file << "invalid_matches=" << invalid_matches << "\n";
        for (size_t i = 3; i < length_counts.size(); i++) {
            if (length_counts[i] > 0) {
                file << "len_" << i << "=" << length_counts[i] << "\n";
            }
        }
        file.flush();
        return true;
    }
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    std::string pool_file;
    std::string target_prefix;
    std::string target_suffix;
    std::string dict_file;
    std::string output_file = "cuda_results.csv";
    size_t min_word_len = 3;
    size_t max_word_len = 10;
    bool validate_primes = true;  // Default to validation for safety
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-p" || arg == "--pool") {
            if (i + 1 >= argc) { std::cerr << "Error: --pool requires a file\n"; return 1; }
            pool_file = argv[++i];
        } else if (arg == "-s" || arg == "--start") {
            if (i + 1 >= argc) { std::cerr << "Error: --start requires a string\n"; return 1; }
            target_prefix = argv[++i];
        } else if (arg == "-e" || arg == "--end") {
            if (i + 1 >= argc) { std::cerr << "Error: --end requires a string\n"; return 1; }
            target_suffix = argv[++i];
        } else if (arg == "-d" || arg == "--dict") {
            if (i + 1 >= argc) { std::cerr << "Error: --dict requires a file\n"; return 1; }
            dict_file = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) { std::cerr << "Error: --output requires a file\n"; return 1; }
            output_file = argv[++i];
        } else if (arg == "--min-len") {
            if (i + 1 >= argc) { std::cerr << "Error: --min-len requires a number\n"; return 1; }
            min_word_len = std::stoull(argv[++i]);
        } else if (arg == "--max-len") {
            if (i + 1 >= argc) { std::cerr << "Error: --max-len requires a number\n"; return 1; }
            max_word_len = std::stoull(argv[++i]);
        } else if (arg == "--validate") {
            validate_primes = true;
        } else if (arg == "--no-validate") {
            validate_primes = false;
        } else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            return 1;
        }
    }
    
    // Validate arguments
    if (pool_file.empty()) {
        std::cerr << "Error: Prime pool file required (-p)\n";
        print_usage(argv[0]);
        return 1;
    }
    
    if (target_prefix.empty() && target_suffix.empty() && dict_file.empty()) {
        std::cerr << "Error: Need at least one of -s, -e, or -d\n";
        return 1;
    }
    
    if (!target_prefix.empty() && !vanity::validate_target(target_prefix)) {
        std::cerr << "Error: Prefix must only contain letters a-p\n";
        return 1;
    }
    if (!target_suffix.empty() && !vanity::validate_target(target_suffix)) {
        std::cerr << "Error: Suffix must only contain letters a-p\n";
        return 1;
    }
    
    // Initialize CUDA runner
    CudaRunner runner;
    g_runner = &runner;
    
    if (!runner.isAvailable()) {
        std::cerr << "Error: CUDA GPUs not available\n";
        return 1;
    }
    
    // Load dictionary if specified
    GPUDictionary dict;
    bool use_dictionary = !dict_file.empty();
    if (use_dictionary) {
        dict = load_dictionary_gpu(dict_file, min_word_len, max_word_len);
        if (dict.empty()) {
            std::cerr << "Error: No valid words in dictionary\n";
            return 1;
        }
    }
    
    // State file
    std::string state_path = output_file + ".state";
    GPUStateFile state;
    bool resumed = state.load(state_path);
    
    // Copy state to dict limits
    if (resumed) {
        for (size_t i = 0; i < state.length_counts.size(); i++) {
            dict.length_counts[i] = state.length_counts[i];
        }
    }
    
    // Print banner
    std::cout << "=======================================================\n";
    std::cout << "Vanity Extension ID Generator - CUDA Multi-GPU Mode\n";
    std::cout << "=======================================================\n";
    std::cout << "GPU Devices:    " << runner.getDeviceName() << "\n";
    std::cout << "GPU Count:      " << runner.getDeviceCount() << "\n";
    
    // Print GPU details
    auto gpu_info = runner.getDeviceInfo();
    for (const auto& info : gpu_info) {
        std::cout << "  GPU " << info.device_id << ": " << info.name 
                  << " (SM " << info.compute_capability_major << "." << info.compute_capability_minor
                  << ", " << info.sm_count << " SMs, "
                  << (info.total_memory / 1024 / 1024 / 1024) << " GB)\n";
    }
    
    // Load prime pool
    size_t primeCount = runner.loadPrimePool(pool_file);
    if (primeCount == 0) {
        std::cerr << "Error: Failed to load prime pool\n";
        return 1;
    }
    
    uint64_t totalPairs = runner.getTotalPairs();
    std::cout << "Prime Pool:     " << format_number(primeCount) << " primes\n";
    std::cout << "Total Pairs:    " << format_number(totalPairs) << " unique keys\n";
    
    if (use_dictionary) {
        std::cout << "Dictionary:     " << dict_file << " (" << dict.size() << " words)\n";
        std::cout << "Word Length:    " << min_word_len << "-" << max_word_len << " chars\n";
    }
    if (!target_prefix.empty()) std::cout << "Target Prefix:  \"" << target_prefix << "\"\n";
    if (!target_suffix.empty()) std::cout << "Target Suffix:  \"" << target_suffix << "\"\n";
    std::cout << "Output File:    " << output_file << "\n";
    std::cout << "Validation:     " << (validate_primes ? "ENABLED (Miller-Rabin)" : "DISABLED") << "\n";
    if (resumed) {
        std::cout << "RESUMED:        " << format_number(state.total_pairs_checked) 
                  << " pairs, " << state.total_matches << " matches\n";
    }
    std::cout << "=======================================================\n\n";
    std::cout << "Running. Press Ctrl+C to stop.\n\n";
    
    // Set up signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Build target list for GPU
    std::vector<std::string> targets;
    if (use_dictionary) {
        for (const auto& word : dict.words) {
            targets.push_back(word);
        }
    } else {
        if (!target_prefix.empty()) targets.push_back(target_prefix);
        if (!target_suffix.empty()) targets.push_back(target_suffix);
    }
    
    // Search configuration
    cuda::SearchConfig search_config;
    search_config.target_prefix = target_prefix;
    search_config.target_suffix = target_suffix;
    
    uint64_t matches_found = 0;
    uint64_t validated_count = 0;
    uint64_t invalid_count = 0;
    
    // Open output file for appending
    std::ofstream csv_out(output_file, std::ios::app);
    std::mutex csv_mutex;
    
    // Progress callback
    auto progress_callback = [&](uint64_t checked, uint64_t total, uint64_t raw_matches) {
        (void)raw_matches;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double rate = (elapsed > 0) ? checked / elapsed : 0;
        double progress = 100.0 * checked / total;
        double eta = (rate > 0) ? (total - checked) / rate : 0;
        
        std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << progress << "%"
                  << " | Pairs: " << format_number(checked)
                  << " | Rate: " << format_number(static_cast<uint64_t>(rate)) << "/s"
                  << " | GPU Matches: " << raw_matches
                  << " | ETA: " << format_time(eta)
                  << "     " << std::flush;
    };
    
    // Run GPU search
    std::vector<GPUMatch> gpu_results;
    if (use_dictionary) {
        gpu_results = runner.searchDictionary(targets, min_word_len, progress_callback);
    } else {
        gpu_results = runner.search(search_config, progress_callback);
    }
    
    if (g_interrupted) {
        std::cout << "\nSearch interrupted by user.\n";
    }
    
    // Process GPU results on CPU
    std::cout << "\n\nProcessing " << gpu_results.size() << " GPU matches...\n";
    if (validate_primes) {
        std::cout << "Validating primes (this may take a while for large result sets)...\n";
    }

    auto process_start = std::chrono::steady_clock::now();
    size_t processed = 0;
    size_t total_to_process = gpu_results.size();
    
    // Get pointer to prime pool for validation
    const uint8_t* prime_pool_ptr = runner.getPrimePool();

    for (const auto& gpu_match : gpu_results) {
        processed++;

        // Progress update every 1000 items
        if (processed % 1000 == 0 || processed == total_to_process) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - process_start).count();
            double rate = (elapsed > 0) ? processed / elapsed : 0;
            double eta = (rate > 0) ? (total_to_process - processed) / rate : 0;
            double pct = 100.0 * processed / total_to_process;

            std::cout << "\rProcessing: " << std::fixed << std::setprecision(1) << pct << "%"
                      << " | " << format_number(processed) << "/" << format_number(total_to_process)
                      << " | Valid: " << validated_count << " | Invalid: " << invalid_count
                      << " | ETA: " << format_time(eta)
                      << "     " << std::flush;
        }

        // Validate primes if enabled
        if (validate_primes) {
            if (!validatePrimePair(prime_pool_ptr, gpu_match.prime_idx_p, gpu_match.prime_idx_q)) {
                invalid_count++;
                continue;  // Skip invalid prime pairs
            }
            validated_count++;
        }

        std::string ext_id = gpu_match.extension_id;
        uint32_t match_type = gpu_match.match_type;

        std::vector<std::pair<std::string, std::string>> word_matches;

        if (match_type == 4) {
            // Pattern match
            std::string cool_pattern = detect_cool_pattern_cpu(ext_id);
            if (!cool_pattern.empty()) {
                word_matches.emplace_back("PATTERN", cool_pattern);
            }
        } else if (use_dictionary) {
            // Dictionary match
            if (match_type & 1) {
                for (const auto& word : dict.prefixes) {
                    if (ext_id.compare(0, word.length(), word) == 0) {
                        word_matches.emplace_back(word, "start");
                        break;
                    }
                }
            }
            if (match_type & 2) {
                for (const auto& word : dict.suffixes) {
                    size_t pos = 32 - word.length();
                    if (ext_id.compare(pos, word.length(), word) == 0) {
                        bool already = false;
                        for (const auto& m : word_matches) {
                            if (m.first == word) { already = true; break; }
                        }
                        if (!already) {
                            word_matches.emplace_back(word, "end");
                        }
                        break;
                    }
                }
            }
        } else {
            // Single target mode
            if (!target_prefix.empty() && ext_id.compare(0, target_prefix.length(), target_prefix) == 0) {
                word_matches.emplace_back(target_prefix, "start");
            }
            if (!target_suffix.empty()) {
                size_t pos = 32 - target_suffix.length();
                if (ext_id.compare(pos, target_suffix.length(), target_suffix) == 0) {
                    word_matches.emplace_back(target_suffix, "end");
                }
            }
        }

        if (word_matches.empty()) continue;
        
        // Find longest match for limit checking
        size_t longest = 0;
        for (const auto& m : word_matches) {
            if (m.first.length() > longest && m.first != "PATTERN") {
                longest = m.first.length();
            }
        }
        
        // Check per-length limits
        if (use_dictionary && longest < dict.length_limits.size()) {
            uint64_t limit = dict.length_limits[longest];
            if (limit > 0 && dict.length_counts[longest] >= limit) {
                continue;
            }
            dict.length_counts[longest]++;
        }
        
        // Build match description
        std::ostringstream match_desc;
        for (size_t i = 0; i < word_matches.size(); i++) {
            if (i > 0) match_desc << ",";
            match_desc << word_matches[i].first << "@" << word_matches[i].second;
        }
        
        // Write to CSV: ext_id,matches,p_idx,q_idx,validated
        {
            std::lock_guard<std::mutex> lock(csv_mutex);
            csv_out << ext_id << ","
                    << match_desc.str() << ","
                    << gpu_match.prime_idx_p << ","
                    << gpu_match.prime_idx_q << ","
                    << (validate_primes ? "true" : "false") << "\n";
        }
        
        matches_found++;
    }
    
    csv_out.flush();
    csv_out.close();
    
    // Update and save state
    state.total_pairs_checked = runner.getTotalPairs();
    state.total_matches = matches_found;
    state.validated_matches = validated_count;
    state.invalid_matches = invalid_count;
    for (size_t i = 0; i < dict.length_counts.size(); i++) {
        state.length_counts[i] = dict.length_counts[i];
    }
    state.save(state_path);
    
    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "\n\n";
    std::cout << "=======================================================\n";
    std::cout << "SEARCH COMPLETE\n";
    std::cout << "=======================================================\n";
    std::cout << "Total Pairs:       " << format_number(runner.getTotalPairs()) << "\n";
    std::cout << "Total Time:        " << format_time(total_time) << "\n";
    std::cout << "Avg Rate:          " << format_number(static_cast<uint64_t>(runner.getTotalPairs() / total_time)) << " pairs/s\n";
    std::cout << "GPU Matches:       " << gpu_results.size() << "\n";
    if (validate_primes) {
        std::cout << "Validated:         " << validated_count << " (" 
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * validated_count / (validated_count + invalid_count)) << "%)\n";
        std::cout << "Invalid (skipped): " << invalid_count << "\n";
    }
    std::cout << "Final Matches:     " << matches_found << "\n";
    std::cout << "Results Saved:     " << output_file << "\n";
    std::cout << "=======================================================\n";
    
    // Print match summary by length
    if (use_dictionary) {
        std::cout << "\nMatches by word length:\n";
        for (size_t i = 3; i <= 10; i++) {
            if (dict.length_counts[i] > 0) {
                std::cout << "  " << i << "-char: " << dict.length_counts[i];
                if (dict.length_limits[i] > 0) {
                    std::cout << " / " << dict.length_limits[i] << " (limit)";
                }
                std::cout << "\n";
            }
        }
    }
    
    g_runner = nullptr;
    return g_interrupted ? 130 : 0;  // 130 = 128 + SIGINT
}
