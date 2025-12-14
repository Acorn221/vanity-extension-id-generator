#include "generator.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <csignal>

using namespace vanity;

// Global state for signal handling
static SharedState* g_state = nullptr;

static bool g_dict_mode = false;

void signal_handler(int) {
    if (g_state) {
        if (g_dict_mode) {
            g_state->stop.store(true, std::memory_order_release);
        } else {
            g_state->found.store(true, std::memory_order_release);
        }
    }
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS] [TARGET]\n\n"
              << "Generate RSA-2048 key pairs until finding one whose Chrome extension ID\n"
              << "contains the target substring.\n\n"
              << "Arguments:\n"
              << "  TARGET              String to find (only letters a-p allowed)\n\n"
              << "Options:\n"
              << "  -p, --position POS  Match position: start|end|anywhere (default: anywhere)\n"
              << "  -s, --start STR     Match this string at the START of the ID\n"
              << "  -e, --end STR       Match this string at the END of the ID\n"
              << "  -w, --workers N     Number of worker threads (default: CPU cores)\n"
              << "  -o, --output PATH   Output PEM file path (default: ./key.pem)\n"
              << "  -h, --help          Show this help message\n\n"
              << "Dictionary Mode (run continuously, find ALL matches):\n"
              << "  -d, --dict FILE     Load wordlist and find ALL matching words\n"
              << "  --min-len N         Minimum word length (default: 3)\n"
              << "  --max-len N         Maximum word length (default: 8)\n\n"
              << "Reconstruct Mode (convert compact CSV to full PEM):\n"
              << "  -r, --reconstruct   Reconstruct full key from compact format\n"
              << "                      Reads: ext_id,matches,p_hex,q_hex,pub_b64\n"
              << "                      Use with: echo 'line' | " << prog_name << " -r\n"
              << "                      Or: " << prog_name << " -r < results.csv\n\n"
              << "Examples:\n"
              << "  " << prog_name << " cafe                    # Find 'cafe' anywhere\n"
              << "  " << prog_name << " -p start dead           # Find 'dead' at start\n"
              << "  " << prog_name << " -s cia -e fbi           # Start with 'cia' AND end with 'fbi'\n"
              << "  " << prog_name << " -d wordlist.txt         # Dictionary mode: find all words\n"
              << "  " << prog_name << " -d words.txt -o results.csv --min-len 4\n"
              << "  " << prog_name << " -r < results.csv        # Reconstruct PEM keys\n\n"
              << "Valid target characters: a b c d e f g h i j k l m n o p\n";
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
    if (seconds < 1.0) {
        return std::to_string(static_cast<int>(seconds * 1000)) + "ms";
    } else if (seconds < 60.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << seconds << "s";
        return oss.str();
    } else if (seconds < 3600.0) {
        int mins = static_cast<int>(seconds / 60);
        int secs = static_cast<int>(seconds) % 60;
        return std::to_string(mins) + "m " + std::to_string(secs) + "s";
    } else {
        int hours = static_cast<int>(seconds / 3600);
        int mins = (static_cast<int>(seconds) % 3600) / 60;
        return std::to_string(hours) + "h " + std::to_string(mins) + "m";
    }
}

int main(int argc, char* argv[]) {
    // Default values
    std::string target;
    std::string start_target;
    std::string end_target;
    std::string position_str = "anywhere";
    std::string dict_file;
    bool reconstruct_mode = false;
    unsigned int num_workers = std::thread::hardware_concurrency();
    std::string output_path = "./key.pem";
    size_t min_word_len = 3;
    size_t max_word_len = 8;
    
    if (num_workers == 0) num_workers = 4;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-p" || arg == "--position") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --position requires an argument\n";
                return 1;
            }
            position_str = argv[++i];
        } else if (arg == "-s" || arg == "--start") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --start requires an argument\n";
                return 1;
            }
            start_target = argv[++i];
        } else if (arg == "-e" || arg == "--end") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --end requires an argument\n";
                return 1;
            }
            end_target = argv[++i];
        } else if (arg == "-d" || arg == "--dict") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --dict requires a file path\n";
                return 1;
            }
            dict_file = argv[++i];
        } else if (arg == "-r" || arg == "--reconstruct") {
            reconstruct_mode = true;
        } else if (arg == "--min-len") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --min-len requires a number\n";
                return 1;
            }
            min_word_len = static_cast<size_t>(std::stoi(argv[++i]));
        } else if (arg == "--max-len") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --max-len requires a number\n";
                return 1;
            }
            max_word_len = static_cast<size_t>(std::stoi(argv[++i]));
        } else if (arg == "-w" || arg == "--workers") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --workers requires an argument\n";
                return 1;
            }
            num_workers = static_cast<unsigned int>(std::stoi(argv[++i]));
            if (num_workers == 0) {
                std::cerr << "Error: workers must be > 0\n";
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --output requires an argument\n";
                return 1;
            }
            output_path = argv[++i];
        } else if (arg[0] == '-') {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        } else {
            target = arg;
        }
    }
    
    // =========================================================================
    // RECONSTRUCT MODE - Convert compact CSV to full PEM keys
    // =========================================================================
    if (reconstruct_mode) {
        std::cout << "Reconstruct Mode - Reading compact keys from stdin...\n";
        std::cout << "Format: ext_id,matches,p_hex,q_hex,pub_b64\n\n";
        
        std::string line;
        int count = 0;
        
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            
            // Parse CSV: ext_id,matches,p_hex,q_hex,pub_b64
            std::vector<std::string> parts;
            std::stringstream ss(line);
            std::string part;
            while (std::getline(ss, part, ',')) {
                parts.push_back(part);
            }
            
            if (parts.size() < 4) {
                std::cerr << "Skipping malformed line: " << line.substr(0, 50) << "...\n";
                continue;
            }
            
            std::string ext_id = parts[0];
            std::string matches = parts[1];
            CompactKey compact;
            compact.p_hex = parts[2];
            compact.q_hex = parts[3];
            
            // Reconstruct and export
            std::string pem = compact_to_pem(compact);
            if (pem.empty()) {
                std::cerr << "Failed to reconstruct key for: " << ext_id << "\n";
                continue;
            }
            
            std::cout << "================================================================================\n";
            std::cout << "Extension ID: " << ext_id << "\n";
            std::cout << "Matches: " << matches << "\n";
            if (parts.size() >= 5) {
                std::cout << "Public Key: " << parts[4] << "\n";
            }
            std::cout << "Private Key:\n" << pem << "\n";
            
            count++;
        }
        
        std::cout << "================================================================================\n";
        std::cout << "Reconstructed " << count << " keys.\n";
        return 0;
    }
    
    // =========================================================================
    // DICTIONARY MODE
    // =========================================================================
    if (!dict_file.empty()) {
        g_dict_mode = true;
        
        // Load dictionary
        Dictionary dict = load_dictionary(dict_file, min_word_len, max_word_len);
        if (dict.empty()) {
            std::cerr << "Error: No valid words loaded from " << dict_file << "\n";
            std::cerr << "Words must be " << min_word_len << "-" << max_word_len 
                      << " chars using only letters a-p\n";
            return 1;
        }
        
        // State file path (same as output but .state extension)
        std::string state_path = output_path + ".state";
        
        // Set up shared state
        SharedState state;
        g_state = &state;
        std::mutex file_mutex;
        
        // Try to load previous state
        StateFile saved_state;
        bool resumed = saved_state.load(state_path);
        if (resumed) {
            state.attempts.store(saved_state.total_attempts, std::memory_order_relaxed);
            state.matches_found.store(saved_state.total_matches, std::memory_order_relaxed);
            for (size_t i = 0; i < saved_state.length_counts.size(); i++) {
                state.length_counts[i].store(saved_state.length_counts[i], std::memory_order_relaxed);
            }
        }
        
        std::cout << "Chrome Extension Vanity ID Generator - DICTIONARY MODE\n";
        std::cout << "======================================================\n";
        std::cout << "Dictionary:       " << dict_file << "\n";
        std::cout << "Valid words:      " << dict.size() << "\n";
        std::cout << "Word length:      " << min_word_len << "-" << max_word_len << " chars\n";
        std::cout << "Workers:          " << num_workers << "\n";
        std::cout << "Output file:      " << output_path << "\n";
        std::cout << "State file:       " << state_path << "\n";
        if (resumed) {
            std::cout << "RESUMED:          " << format_number(saved_state.total_attempts) 
                      << " attempts, " << saved_state.total_matches << " matches\n";
        }
        std::cout << "======================================================\n\n";
        std::cout << "Running continuously. Matches saved to: " << output_path << "\n";
        std::cout << "State saved to: " << state_path << " (auto-saves every 30s)\n";
        std::cout << "Press Ctrl+C to stop.\n\n";
        
        // Set up signal handler
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
        
        // Record start time
        auto start_time = std::chrono::steady_clock::now();
        
        // Spawn worker threads
        std::vector<std::thread> workers;
        workers.reserve(num_workers);
        
        for (unsigned int i = 0; i < num_workers; i++) {
            // Use optimized worker loop with Apple Accelerate and larger batches
            workers.emplace_back(worker_loop_dict_fast, std::cref(dict), std::ref(state),
                                std::cref(output_path), std::ref(file_mutex));
        }
        
        // Progress monitoring loop
        uint64_t last_attempts = 0;
        auto last_time = start_time;
        auto last_save_time = start_time;
        constexpr int SAVE_INTERVAL_SECS = 30;  // Save state every 30 seconds
        
        while (!state.stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            auto now = std::chrono::steady_clock::now();
            uint64_t current_attempts = state.attempts.load(std::memory_order_relaxed);
            uint64_t matches = state.matches_found.load(std::memory_order_relaxed);
            
            double elapsed_interval = std::chrono::duration<double>(now - last_time).count();
            double total_elapsed = std::chrono::duration<double>(now - start_time).count();
            
            // Periodic state save
            double since_last_save = std::chrono::duration<double>(now - last_save_time).count();
            if (since_last_save >= SAVE_INTERVAL_SECS) {
                StateFile sf;
                sf.total_attempts = current_attempts;
                sf.total_matches = matches;
                for (size_t i = 0; i < state.length_counts.size(); i++) {
                    sf.length_counts[i] = state.length_counts[i].load(std::memory_order_relaxed);
                }
                sf.save(state_path);
                last_save_time = now;
            }
            
            uint64_t delta = current_attempts - last_attempts;
            double rate = (elapsed_interval > 0) ? delta / elapsed_interval : 0;
            
            std::cout << "\r" << std::flush;
            std::cout << "Keys: " << std::setw(12) << format_number(current_attempts)
                      << " | Rate: " << std::setw(8) << format_number(static_cast<uint64_t>(rate)) << "/s"
                      << " | Matches: " << std::setw(6) << matches
                      << " | Time: " << std::setw(10) << format_time(total_elapsed)
                      << "   " << std::flush;
            
            last_attempts = current_attempts;
            last_time = now;
        }
        
        // Wait for all workers to finish
        for (auto& worker : workers) {
            worker.join();
        }
        
        auto end_time = std::chrono::steady_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        // Final state save
        {
            StateFile sf;
            sf.total_attempts = state.attempts.load();
            sf.total_matches = state.matches_found.load();
            for (size_t i = 0; i < state.length_counts.size(); i++) {
                sf.length_counts[i] = state.length_counts[i].load(std::memory_order_relaxed);
            }
            sf.save(state_path);
        }
        
        std::cout << "\n\n";
        std::cout << "======================================================\n";
        std::cout << "DICTIONARY MODE COMPLETE\n";
        std::cout << "======================================================\n";
        std::cout << "Total keys generated: " << format_number(state.attempts.load()) << "\n";
        std::cout << "Total matches found:  " << state.matches_found.load() << "\n";
        std::cout << "Total time:           " << format_time(total_time) << "\n";
        std::cout << "Average rate:         " << format_number(static_cast<uint64_t>(state.attempts.load() / total_time)) << "/s\n";
        std::cout << "Results saved to:     " << output_path << "\n";
        std::cout << "State saved to:       " << state_path << "\n\n";
        
        // Print summary of matches
        if (!state.dict_matches.empty()) {
            std::cout << "Match Summary:\n";
            std::cout << "--------------\n";
            size_t show_count = std::min(state.dict_matches.size(), size_t(20));
            for (size_t i = 0; i < show_count; i++) {
                const auto& m = state.dict_matches[i];
                std::cout << "  " << m.extension_id << " -> ";
                for (size_t j = 0; j < m.matches.size(); j++) {
                    if (j > 0) std::cout << ", ";
                    std::cout << m.matches[j].first << "@" << m.matches[j].second;
                }
                std::cout << "\n";
            }
            if (state.dict_matches.size() > 20) {
                std::cout << "  ... and " << (state.dict_matches.size() - 20) << " more\n";
            }
        }
        
        return 0;
    }
    
    // Build search configuration
    SearchConfig config;
    
    // Check if using --start/--end mode vs positional TARGET
    if (!start_target.empty() || !end_target.empty()) {
        // Using explicit --start and/or --end
        if (!target.empty()) {
            std::cerr << "Error: Cannot use positional TARGET with --start/--end options\n";
            return 1;
        }
        
        if (!start_target.empty() && !validate_target(start_target)) {
            std::cerr << "Error: --start value must only contain letters a-p\n";
            std::cerr << "Invalid: \"" << start_target << "\"\n";
            return 1;
        }
        
        if (!end_target.empty() && !validate_target(end_target)) {
            std::cerr << "Error: --end value must only contain letters a-p\n";
            std::cerr << "Invalid: \"" << end_target << "\"\n";
            return 1;
        }
        
        if (!start_target.empty() && !end_target.empty()) {
            // Both start AND end specified
            if (start_target.length() + end_target.length() > 32) {
                std::cerr << "Error: Combined --start and --end length cannot exceed 32\n";
                return 1;
            }
            config.target = start_target;
            config.end_target = end_target;
            config.position = MatchPosition::StartAndEnd;
        } else if (!start_target.empty()) {
            // Only start
            config.target = start_target;
            config.position = MatchPosition::Start;
        } else {
            // Only end
            config.target = end_target;
            config.position = MatchPosition::End;
        }
    } else {
        // Using positional TARGET with optional --position
        if (target.empty()) {
            std::cerr << "Error: TARGET is required (or use --start/--end)\n\n";
            print_usage(argv[0]);
            return 1;
        }
        
        if (!validate_target(target)) {
            std::cerr << "Error: TARGET must only contain letters a-p\n";
            std::cerr << "Invalid target: \"" << target << "\"\n";
            return 1;
        }
        
        if (target.length() > 32) {
            std::cerr << "Error: TARGET cannot be longer than 32 characters\n";
            return 1;
        }
        
        config.target = target;
        config.position = parse_position(position_str);
        if (position_str != "start" && position_str != "end" && position_str != "anywhere") {
            std::cerr << "Warning: Unknown position '" << position_str << "', using 'anywhere'\n";
        }
    }
    
    // Calculate expected attempts
    uint64_t expected = estimate_attempts(config);
    
    std::cout << "Chrome Extension Vanity ID Generator\n";
    std::cout << "=====================================\n";
    if (config.position == MatchPosition::StartAndEnd) {
        std::cout << "Start with:       \"" << config.target << "\"\n";
        std::cout << "End with:         \"" << config.end_target << "\"\n";
    } else {
        std::cout << "Target:           \"" << config.target << "\"\n";
        std::cout << "Position:         " << position_to_string(config.position) << "\n";
    }
    std::cout << "Workers:          " << num_workers << "\n";
    std::cout << "Output:           " << output_path << "\n";
    std::cout << "Expected tries:   ~" << format_number(expected) << "\n";
    std::cout << "=====================================\n\n";
    std::cout << "Searching... (Ctrl+C to cancel)\n\n";
    
    // Set up shared state
    SharedState state;
    g_state = &state;
    
    // Set up signal handler for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Record start time
    auto start_time = std::chrono::steady_clock::now();
    
    // Spawn worker threads
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    
    for (unsigned int i = 0; i < num_workers; i++) {
        // Use optimized worker loop with Apple Accelerate and larger batches
        workers.emplace_back(worker_loop_fast, std::cref(config), std::ref(state));
    }
    
    // Progress monitoring loop
    uint64_t last_attempts = 0;
    auto last_time = start_time;
    
    while (!state.found.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        auto now = std::chrono::steady_clock::now();
        uint64_t current_attempts = state.attempts.load(std::memory_order_relaxed);
        
        // Calculate rate
        double elapsed_interval = std::chrono::duration<double>(now - last_time).count();
        double total_elapsed = std::chrono::duration<double>(now - start_time).count();
        
        uint64_t delta = current_attempts - last_attempts;
        double rate = (elapsed_interval > 0) ? delta / elapsed_interval : 0;
        
        // Calculate ETA
        double progress = static_cast<double>(current_attempts) / expected;
        std::string eta_str = "calculating...";
        if (rate > 0 && current_attempts > 0) {
            double remaining_attempts = expected - current_attempts;
            if (remaining_attempts > 0) {
                double eta = remaining_attempts / rate;
                eta_str = format_time(eta);
            } else {
                eta_str = "any moment...";
            }
        }
        
        // Print progress
        std::cout << "\r" << std::flush;
        std::cout << "Attempts: " << std::setw(12) << format_number(current_attempts)
                  << " | Rate: " << std::setw(8) << format_number(static_cast<uint64_t>(rate)) << "/s"
                  << " | Elapsed: " << std::setw(8) << format_time(total_elapsed)
                  << " | ETA: " << std::setw(12) << eta_str
                  << "   " << std::flush;
        
        last_attempts = current_attempts;
        last_time = now;
    }
    
    // Wait for all workers to finish
    for (auto& worker : workers) {
        worker.join();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "\n\n";
    
    // Check if we found a match or were interrupted
    if (!state.result.found) {
        std::cout << "Search cancelled after " << format_number(state.attempts.load()) 
                  << " attempts (" << format_time(total_time) << ")\n";
        return 1;
    }
    
    // Success! Print results
    std::cout << "=====================================\n";
    std::cout << "MATCH FOUND!\n";
    std::cout << "=====================================\n\n";
    
    std::cout << "Attempts:     " << format_number(state.attempts.load()) << "\n";
    std::cout << "Time:         " << format_time(total_time) << "\n";
    std::cout << "Rate:         " << format_number(static_cast<uint64_t>(state.attempts.load() / total_time)) << "/s\n\n";
    
    // Display extension ID with match highlighted
    std::cout << "Extension ID: " << state.result.extension_id << "\n";
    
    // Find and highlight the match(es)
    std::cout << "              ";
    if (config.position == MatchPosition::StartAndEnd) {
        // Highlight both start and end
        for (size_t i = 0; i < config.target.length(); i++) std::cout << "^";
        for (size_t i = config.target.length(); i < 32 - config.end_target.length(); i++) std::cout << " ";
        for (size_t i = 0; i < config.end_target.length(); i++) std::cout << "^";
    } else {
        size_t match_pos = 0;
        switch (config.position) {
            case MatchPosition::Start:
                match_pos = 0;
                break;
            case MatchPosition::End:
                match_pos = 32 - config.target.length();
                break;
            case MatchPosition::Anywhere:
                match_pos = state.result.extension_id.find(config.target);
                break;
            case MatchPosition::StartAndEnd:
                break; // Already handled above
        }
        for (size_t i = 0; i < match_pos; i++) std::cout << " ";
        for (size_t i = 0; i < config.target.length(); i++) std::cout << "^";
    }
    std::cout << "\n\n";
    
    // Save private key
    std::ofstream key_file(output_path);
    if (!key_file) {
        std::cerr << "Error: Could not write to " << output_path << "\n";
        return 1;
    }
    key_file << state.result.private_key_pem;
    key_file.close();
    
    std::cout << "Private key saved to: " << output_path << "\n\n";
    
    std::cout << "Public key (for manifest.json \"key\" field):\n";
    std::cout << state.result.public_key_base64 << "\n\n";
    
    std::cout << "To verify, add to manifest.json:\n";
    std::cout << "  \"key\": \"" << state.result.public_key_base64 << "\"\n";
    
    return 0;
}

