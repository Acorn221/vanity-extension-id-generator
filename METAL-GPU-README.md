# Vanity Extension ID Generator - Metal GPU Version

Generate vanity Chrome extension IDs using Apple Silicon GPU acceleration.

## Overview

This tool searches for Chrome extension IDs that contain dictionary words or cool patterns. It uses a pre-computed pool of RSA primes and tests all combinations on the GPU, achieving ~50M key pairs/second on Apple Silicon.

## Quick Start

```bash
# 1. Build
cd build
cmake .. && make -j8

# 2. Generate a prime pool (do this once)
./generate-prime-pool 10000 prime_pool_10k.bin

# 3. Search for vanity IDs
./vanity-ext-id-metal -p prime_pool_10k.bin -d ../wordlist.txt -o results.csv

# 4. Reconstruct a key from results
cd ..
python reconstruct_gpu.py -p build/prime_pool_10k.bin -i
# Then paste a line from results.csv
```

## Step-by-Step Guide

### 1. Generate Prime Pool

The prime pool is a collection of pre-computed 1024-bit RSA primes. The GPU tests all unique pairs (p, q) from this pool.

```bash
./generate-prime-pool <count> <output_file>
```

**Pool size vs search space:**

| Primes | Unique Pairs | Search Time* |
|--------|--------------|--------------|
| 1,500  | ~1.1M        | ~10 seconds  |
| 10,000 | ~50M         | ~8 minutes   |
| 50,000 | ~1.25B       | ~3.5 hours   |
| 100,000| ~5B          | ~14 hours    |

*Estimated for dictionary mode on M1/M2/M3 Mac

**Examples:**
```bash
# Quick test pool
./generate-prime-pool 1500 prime_pool_test.bin

# Standard pool (good balance of speed vs coverage)
./generate-prime-pool 10000 prime_pool_10k.bin

# Large overnight run
./generate-prime-pool 100000 prime_pool_100k.bin
```

### 2. Prepare a Wordlist

Create a text file with words you want to find (one per line). Words must:
- Be 3-10 characters (configurable)
- Only contain letters a-p (Chrome extension ID alphabet)
- Examples: `cafe`, `code`, `hack`, `ninja`, `epic`

Invalid words (containing q-z) are automatically filtered out.

**Sample wordlist.txt:**
```
cafe
code
hack
cool
epic
ninja
apple
chrome
```

### 3. Run the Search

**Dictionary mode (recommended):**
```bash
./vanity-ext-id-metal -p prime_pool_10k.bin -d wordlist.txt -o results.csv
```

**Search for specific prefix:**
```bash
./vanity-ext-id-metal -p prime_pool_10k.bin -s cafe -o results.csv
```

**Search for specific suffix:**
```bash
./vanity-ext-id-metal -p prime_pool_10k.bin -e code -o results.csv
```

**Combined prefix + suffix:**
```bash
./vanity-ext-id-metal -p prime_pool_10k.bin -s abc -e def -o results.csv
```

**Options:**
```
-p, --pool FILE     Prime pool file (required)
-d, --dict FILE     Dictionary file for word search
-s, --start STR     Find IDs starting with STR
-e, --end STR       Find IDs ending with STR
-o, --output FILE   Output CSV file (default: gpu_results.csv)
--min-len N         Minimum word length (default: 3)
--max-len N         Maximum word length (default: 10)
```

### 4. Understanding the Output

The CSV output format is:
```
ext_id,matches,p_idx,q_idx
```

- `ext_id`: The 32-character Chrome extension ID
- `matches`: What was found (e.g., `cafe@start`, `code@end`, `PATTERN@i*7`)
- `p_idx`: Index of first prime in the pool
- `q_idx`: Index of second prime in the pool

**Example output:**
```csv
cafeabcdefghijklmnopabcdefghijkl,cafe@start,123,456
abcdefghijklmnopabcdefghijklcode,code@end,789,1011
aaaaaaabcdefghijklmnopqrstuvwxyz,PATTERN@a*7,100,200
```

**Pattern types detected:**
- `word@start` - Word at beginning of ID
- `word@end` - Word at end of ID
- `word@pos:N` - Long word (7+ chars) found at position N
- `PATTERN@X*N` - Character X repeated N times (6+)
- `PATTERN@seq:abc` - Sequential characters (5+)
- `PATTERN@palindrome:X` - Palindrome at start/end (8+)
- `PATTERN@prefix:X` - Same character prefix (5+)
- `PATTERN@suffix:X` - Same character suffix (5+)

### 5. Reconstruct Keys

Once you find an ID you like, reconstruct the full RSA private key:

**Interactive mode:**
```bash
python reconstruct_gpu.py -p build/prime_pool_10k.bin -i
```
Then paste a line from your results CSV.

**Batch mode (export all to PEM files):**
```bash
python reconstruct_gpu.py -p build/prime_pool_10k.bin -c results.csv -o keys/
```

**Single key with limit:**
```bash
python reconstruct_gpu.py -p build/prime_pool_10k.bin -c results.csv -n 1
```

**Output includes:**
- Verification that the key produces the expected extension ID
- PEM-formatted private key (save as `key.pem`)
- Base64 public key for `manifest.json`

### 6. Use in Your Extension

1. Save the private key as `key.pem`

2. Add the public key to your `manifest.json`:
```json
{
  "manifest_version": 3,
  "name": "My Extension",
  "version": "1.0",
  "key": "MIIBIjANBgkqh..."
}
```

3. Pack the extension:
```bash
# Using Chrome
chrome --pack-extension=/path/to/extension --pack-extension-key=key.pem

# Or load unpacked in developer mode - the ID will match!
```

## Important Notes

### Prime Pool + Results Are Linked

The CSV results contain prime **indices**, not the actual primes. You must use the same prime pool file that was used during the search to reconstruct keys.

```bash
# WRONG - pool mismatch!
./vanity-ext-id-metal -p pool_A.bin -o results.csv
python reconstruct_gpu.py -p pool_B.bin -c results.csv  # Will fail!

# CORRECT
./vanity-ext-id-metal -p pool_A.bin -o results.csv
python reconstruct_gpu.py -p pool_A.bin -c results.csv  # Works!
```

### Output File Appends

Results are appended to the output CSV, not overwritten. Delete or use a new filename between runs with different pools:

```bash
rm results.csv  # Clear old results
./vanity-ext-id-metal -p new_pool.bin -d wordlist.txt -o results.csv
```

### Per-Length Limits

To avoid millions of short matches, dictionary mode limits:
- 3-char words: 100 matches max
- 4-char words: 200 matches max
- 5-char words: 500 matches max
- 6+ char words: unlimited (they're rare!)

### Ctrl+C Safe

Press Ctrl+C to stop gracefully. A state file (`results.csv.state`) tracks progress for potential resume support.

## Troubleshooting

**"Metal GPU not available"**
- Requires macOS with Apple Silicon (M1/M2/M3) or AMD GPU
- Make sure you built with Metal support: `cmake .. && make`

**"Prime index out of range"**
- You're using a different prime pool than was used for the search
- Check which pool file the search used and use the same one for reconstruction

**"Invalid prime pool file"**
- File is corrupted or not a valid prime pool
- Regenerate with `./generate-prime-pool`

**Search seems stuck**
- In dictionary mode, it searches each 3-char prefix/suffix pattern sequentially
- Watch the `[N/M] Searching prefix: xxx` progress indicator
- Large dictionaries = many patterns = longer runtime

## Performance Tips

1. **Smaller dictionary = faster search** - Only include words you actually want
2. **Longer minimum word length** - Use `--min-len 4` to skip 3-char words
3. **Larger prime pool = more results** - But takes longer to search
4. **Run overnight** - Use a 50k-100k pool for best coverage
