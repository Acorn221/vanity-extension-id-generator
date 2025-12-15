# Chrome Extension Vanity ID Generator

This repository has been split into two separate implementations:

## 🖥️ CPU Version
**Location:** [`vanity-ext-id-cpu/`](vanity-ext-id-cpu/)

A high-performance CPU-based C++ tool for brute-forcing RSA-2048 key pairs to find Chrome extension IDs containing desired substrings. Works on any platform with OpenSSL.

- **Best for:** Cross-platform usage, servers, non-Apple hardware
- **Performance:** Multi-threaded CPU processing
- **Dependencies:** OpenSSL, CMake, C++17 compiler

[→ CPU Version README](vanity-ext-id-cpu/README.md)

## 🚀 Metal GPU Version
**Location:** [`vanity-ext-id-metal/`](vanity-ext-id-metal/)

Apple Silicon GPU-accelerated version using Metal framework for massive performance gains (~100,000x faster than CPU).

- **Best for:** Apple Silicon Macs (M1/M2/M3/M4)
- **Performance:** ~50M key pairs/second on modern Apple Silicon
- **Dependencies:** macOS, Metal framework, OpenSSL, CMake

[→ Metal GPU Version README](vanity-ext-id-metal/README.md)

## Quick Comparison

| Feature | CPU Version | Metal GPU Version |
|---------|-------------|-------------------|
| **Platform** | Linux, macOS, Windows | macOS (Apple Silicon) |
| **Performance** | ~100-1000 keys/sec | ~50M keys/sec |
| **Memory Usage** | Low | High (prime pools) |
| **Setup Complexity** | Simple | Moderate |
| **Dependencies** | OpenSSL + CMake | Metal + OpenSSL + CMake |

## Migration

If you were using the old combined repository:

1. **For CPU usage:** Switch to `vanity-ext-id-cpu/`
2. **For Metal GPU:** Switch to `vanity-ext-id-metal/`
3. **Both versions** include the prime pool generator, reconstruction scripts, and share the `wordlist.txt` file

The split improves maintainability and allows each version to be optimized for its specific use case.