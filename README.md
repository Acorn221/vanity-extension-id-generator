# Chrome Extension Vanity ID Generator

A high-performance C++ tool for brute-forcing RSA-2048 key pairs to find one whose Chrome extension ID contains a desired substring.

## Quick Start

```bash
# 1. Install dependencies (macOS)
brew install cmake openssl

# 2. Build
cd vanity-ext-id
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=$(brew --prefix openssl) ..
make -j

# 3. Run dictionary mode (finds all words from wordlist)
./vanity-ext-id -d ../wordlist.txt -o results.csv

# 4. Let it run... Ctrl+C to stop (state auto-saves every 30s)

# 5. Extract a key you like
cd ..
pip install cryptography
python reconstruct.py
# Paste a line from build/results.csv
```

## How It Works

Chrome extension IDs are derived from the RSA public key:
1. Take the RSA public key (SPKI/DER format)
2. SHA-256 hash it
3. Take first 16 bytes (128 bits)
4. Convert to a-p encoding: `0→a, 1→b, 2→c, ... f→p`

Result: 32-character extension ID using only letters a-p (e.g., `cafedeadbeefabcd...`)

## Building

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+)
- CMake 3.14+
- OpenSSL 1.1+ development libraries

**Ubuntu/Debian:**
```bash
sudo apt install build-essential cmake libssl-dev
```

**macOS (Homebrew):**
```bash
brew install cmake openssl
```

**Fedora/RHEL:**
```bash
sudo dnf install gcc-c++ cmake openssl-devel
```

### Build Instructions

```bash
cd vanity-ext-id
mkdir build && cd build

# Linux
cmake -DCMAKE_BUILD_TYPE=Release ..

# macOS (need to specify OpenSSL path)
cmake -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=$(brew --prefix openssl) ..

# Build
make -j$(nproc)
```

## Usage Modes

### Mode 1: Single Target (stops when found)

```bash
# Find "cafe" anywhere
./vanity-ext-id cafe

# Find "dead" at start only
./vanity-ext-id -p start dead

# Find key that starts with "cia" AND ends with "fbi"
./vanity-ext-id -s cia -e fbi

# Custom output path
./vanity-ext-id -o mykey.pem cafe
```

Output: `key.pem` (full PKCS#8 private key)

### Mode 2: Dictionary Mode (runs continuously)

```bash
# Run against the included wordlist
./vanity-ext-id -d ../wordlist.txt -o results.csv

# With word length filters
./vanity-ext-id -d ../wordlist.txt -o results.csv --min-len 4 --max-len 8
```

**Features:**
- Runs continuously until Ctrl+C
- Saves ALL matches to CSV (compact format)
- Auto-saves state every 30s → resume on restart
- Per-length limits: 100×3-char, 200×4-char, 500×5-char, unlimited 6+
- 7+ letter words saved even if found in middle of ID
- Detects cool patterns (sequences, palindromes, repeated chars)

**Output format (CSV):**
```
ext_id,matches,p_hex,q_hex,pub_b64
cafeabcdefgh...,cafe@start,A1B2C3...,D4E5F6...,MIIBIj...
```

## Extracting Keys from CSV

The CSV stores keys in compact format (just the two primes p,q). Use the Python script to reconstruct full PEM keys:

```bash
# Install dependency
pip install cryptography

# Run the script
python reconstruct.py
```

Then paste a line from your CSV:
```
> cafeabcdefghijklmnopqrstuvwxyzab,cafe@start,A1B2C3...,D4E5F6...,MIIBIj...

Extension ID: cafeabcdefghijklmnopqrstuvwxyzab
Matches: cafe@start
✓ Verified: Extension ID matches!

============================================================
PRIVATE KEY (save as key.pem):
============================================================
-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASC...
-----END PRIVATE KEY-----

============================================================
PUBLIC KEY for manifest.json:
============================================================
"key": "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A..."
============================================================
```

## Using the Key in Your Extension

Add the public key to your `manifest.json`:

```json
{
  "manifest_version": 3,
  "name": "My Extension",
  "version": "1.0",
  "key": "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A..."
}
```

Keep the private key safe or get rid of it, the chrome web store will store it for updates but if this is an internal extension then store it and be careful with it.

## Valid Characters

Extension IDs only use letters **a through p** (hex digits 0-f).

✅ **Valid:** cafe, face, fade, beef, dead, deed, feed, babe, badge, linkedin  
❌ **Invalid:** cool (has 'o'), test (has 's','t'), zero (has 'r','z')

## Difficulty Estimates

| Target | Position | Expected Attempts | @ 10k/s |
|--------|----------|-------------------|---------|
| 4 chars | start | ~65K | ~7 sec |
| 5 chars | start | ~1M | ~2 min |
| 6 chars | start | ~17M | ~28 min |
| 7 chars | start | ~268M | ~7.5 hrs |
| 8 chars | start | ~4.3B | ~5 days |

## Performance

| Hardware | Keys/sec |
|----------|----------|
| 4-core laptop | ~8,000 |
| 8-core desktop | ~16,000 |
| 16-core workstation | ~30,000+ |

RSA-2048 key generation is CPU-bound (~50-100μs per key).

## Files

```
vanity-ext-id/
├── CMakeLists.txt      # Build config
├── README.md           # This file
├── wordlist.txt        # ~3600 words (filtered to ~1000 valid a-p)
├── reconstruct.py      # Python script to extract keys from CSV
├── src/
│   ├── main.cpp        # CLI and threading
│   ├── generator.hpp   # Headers
│   └── generator.cpp   # RSA generation, hashing, matching
└── build/              # Build output (created by cmake)
    ├── vanity-ext-id   # The executable
    ├── results.csv     # Your matches
    └── results.csv.state  # Resume state
```

## License

MIT License
