# Chrome Extension Vanity ID Generator

A high-performance C++ tool for brute-forcing RSA-2048 key pairs to find one whose Chrome extension ID contains a desired substring.

## How It Works

Chrome extension IDs are derived from the RSA public key:
1. Take the RSA public key (SPKI/DER format)
2. SHA-256 hash it
3. Take first 16 bytes (128 bits)
4. Convert to a-p encoding: `0→a, 1→b, 2→c, ... f→p`

Result: 32-character extension ID using only letters a-p (e.g., `aapbdbdomjkkjkaonfhkkikfgjllcleb`)

## Building

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+)
- CMake 3.14+
- OpenSSL 1.1+ development libraries

#### Installing dependencies

**Ubuntu/Debian:**
```bash
sudo apt install build-essential cmake libssl-dev
```

**macOS (with Homebrew):**
```bash
brew install cmake openssl
```

**Fedora/RHEL:**
```bash
sudo dnf install gcc-c++ cmake openssl-devel
```

### Build Instructions

```bash
# Clone or navigate to the project directory
cd vanity-ext-id

# Create build directory
mkdir build && cd build

# Configure (Release mode for best performance)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j$(nproc)

# (Optional) Install system-wide
sudo make install
```

On macOS with Homebrew OpenSSL, you may need to specify the OpenSSL path:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=$(brew --prefix openssl) ..
```

## Usage

```bash
vanity-ext-id [OPTIONS] <TARGET>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `TARGET` | String to find (only letters a-p allowed) |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --position` | Match position: `start`, `end`, or `anywhere` | `anywhere` |
| `-w, --workers` | Number of worker threads | CPU cores |
| `-o, --output` | Output PEM file path | `./key.pem` |
| `-h, --help` | Show help message | - |

### Examples

```bash
# Find "cafe" anywhere in the extension ID
./vanity-ext-id cafe

# Find "dead" at the start of the extension ID
./vanity-ext-id -p start dead

# Use 4 workers and custom output path
./vanity-ext-id -w 4 -o mykey.pem beef

# Find "face" at the end
./vanity-ext-id --position end --output face-key.pem face

# Start AND end match
./vanity-ext-id -s cia -e fbi
```

## Dictionary Mode

Run continuously and find ALL matches from a wordlist:

```bash
# From the build directory
./vanity-ext-id -d ../wordlist.txt -o results.csv

# With custom word length filters
./vanity-ext-id -d ../wordlist.txt -o results.csv --min-len 4 --max-len 8
```

The included `wordlist.txt` at the project root contains ~1000 valid a-p words.

### Dictionary Mode Features

- **Runs continuously** until you hit Ctrl+C
- **Saves all matches** to CSV with compact key storage
- **Auto-saves state** every 30s to `results.csv.state`
- **Resumes automatically** if you restart
- **Per-length limits**: 100 for 3-char, 200 for 4-char, 500 for 5-char, unlimited for 6+
- **Long word anywhere**: 7+ letter words saved even if found in middle of ID
- **Pattern detection**: Finds cool patterns like repeated chars, sequences, palindromes

### Output Format (CSV)

```
ext_id,matches,p_hex,q_hex,pub_b64
cafeabcd...,cafe@start,A1B2C3...,D4E5F6...,MIIBIj...
```

### Reconstructing Keys

Use the included Python script to convert CSV lines back to full PEM keys:

```bash
pip install cryptography
python reconstruct.py
# Then paste a line from your CSV
```

## Valid Target Characters

Extension IDs only use letters **a through p** (representing hex digits 0-f).

**Valid targets:** cafe, face, fade, beef, dead, deed, feed, bead, babe, cage, edge, badge

**Invalid targets:** cool (has 'o'), zero (has 'r', 'z'), test (has 's', 't')

## Difficulty Estimates

| Target Length | Position | Expected Attempts | @ 10,000/s |
|---------------|----------|-------------------|------------|
| 3 chars | anywhere | ~137 | <1 sec |
| 4 chars | anywhere | ~2,185 | <1 sec |
| 4 chars | start | ~65,536 | ~7 sec |
| 5 chars | anywhere | ~37,449 | ~4 sec |
| 5 chars | start | ~1,048,576 | ~2 min |
| 6 chars | anywhere | ~639,456 | ~1 min |
| 6 chars | start | ~16,777,216 | ~28 min |

## Output

When a match is found, the tool outputs:

1. **key.pem** - PKCS#8 private key file
2. **Console output** with:
   - Extension ID (with match highlighted)
   - Base64 public key for manifest.json
   - Statistics (attempts, time, rate)

### Using the Generated Key

Add the public key to your extension's `manifest.json`:

```json
{
  "manifest_version": 3,
  "name": "My Extension",
  "version": "1.0",
  "key": "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A..."
}
```

Keep the private key (`key.pem`) safe - you'll need it to sign updates to your extension.

## Performance

Expected throughput on modern hardware:

| Hardware | Keys/sec |
|----------|----------|
| 4-core laptop | ~8,000 |
| 8-core desktop | ~16,000 |
| 16-core workstation | ~30,000+ |

RSA-2048 key generation is the bottleneck (~50-100μs per key on modern CPUs).

## Verifying the Result

You can verify the generated key produces the expected extension ID:

```javascript
const crypto = require('crypto');
const fs = require('fs');

const pem = fs.readFileSync('key.pem');
const key = crypto.createPrivateKey(pem);
const publicKey = crypto.createPublicKey(key);

const spki = publicKey.export({ type: 'spki', format: 'der' });
const hash = crypto.createHash('sha256').update(spki).digest();

let extId = '';
for (let i = 0; i < 16; i++) {
    extId += String.fromCharCode(97 + (hash[i] >> 4));
    extId += String.fromCharCode(97 + (hash[i] & 0x0f));
}
console.log('Extension ID:', extId);
```

## License

MIT License

