# Vanity Chrome Extension ID Generator (CUDA)

Generate custom Chrome extension IDs like `cafebeef00000000000000000000aaaa` using GPU acceleration!

## WTF is This?

Chrome extension IDs are **32-character strings** made from letters `a-p` (base-16). They're derived from the SHA-256 hash of your extension's RSA public key.

**Example IDs:**
- `nkbihfbeogaeaoehlefnkodbefgpgknn` (MetaMask)
- `aeblfdkhhhdcdjpifhhbdiojplfjncoa` (1Password)
- `cafe000000000000000000000000aaaa` (Vanity ID - starts with "cafe")

This tool lets you **generate custom IDs** by brute-forcing RSA key pairs until you find one that produces your desired ID pattern.

---

## How Chrome Extension IDs Work

### The Process:

```
1. Generate RSA-2048 key pair (p, q are 1024-bit primes)
   └─> Public key: n = p × q (2048 bits)

2. Encode public key as DER format (294 bytes)
   └─> Standard X.509 SubjectPublicKeyInfo structure

3. Hash with SHA-256
   └─> 256-bit hash (32 bytes)

4. Convert to base-16 using letters a-p
   └─> 32-character extension ID!
```

### Example:

```
Prime p: 0xd5f2a1... (1024 bits)
Prime q: 0x8c3b7e... (1024 bits)
    ↓
Modulus n = p × q (2048 bits)
    ↓
DER encoding: 30 82 01 22 30 0d 06 09...
    ↓
SHA-256: a3 f5 2c 89 b1 4e...
    ↓
Extension ID: cafe12ab34cd56ef... (32 chars, a-p only)
```

**Why it's hard:**
- SHA-256 is a **one-way function** - you can't reverse it
- Each prime pair produces a pseudo-random ID
- To get a specific pattern, you need to test **millions of combinations**

---

## The Two-Phase Approach

### Phase 1: Prime Pool Generation (Slow, Do Once)

Generate a large pool of 1024-bit primes and save them to a file.

**Why pre-generate?**
- Prime generation is **slow** (Miller-Rabin testing)
- But you only need to do it **once**
- Reuse the same pool for all searches

**Tools:**
- `generate-prime-pool` - CPU version (multithreaded, ~55 primes/s)
- **`generate-prime-pool-cgbn` - GPU version (1150 primes/s!) ← USE THIS**

**Output:** Binary file with N primes (128 bytes each)

```
File: prime_pool_100k.bin (13 MB)
├─ Header (16 bytes): magic, version, count, prime_bytes
└─ Primes (100,000 × 128 bytes): raw big-endian 1024-bit numbers
```

### Phase 2: Vanity Search (Fast, GPU-Accelerated)

Try all combinations of primes from the pool until you find matching IDs.

**Math:**
- 100K primes = **~5 billion unique combinations** (100k × 99.9k / 2)
- RTX 3080 can test **~300 million/second**
- Find patterns in seconds to minutes!

**Tool:**
- `vanity-ext-id-cuda` - Multi-GPU CUDA search engine

---

## Quick Start

### 1. Build Everything

```bash
cd vanity-ext-id-cuda
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

### 2. Generate Prime Pool (90 seconds for 100K primes)

```bash
./generate-prime-pool-cgbn 100000 pool_100k.bin
```

### 3. Search for Vanity IDs

```bash
# Find IDs starting with "cafe"
./vanity-ext-id-cuda -p pool_100k.bin -s cafe -o results.csv

# Find IDs ending with "beef"
./vanity-ext-id-cuda -p pool_100k.bin -e beef -o results.csv

# Both prefix and suffix
./vanity-ext-id-cuda -p pool_100k.bin -s dead -e beef -o results.csv

# Dictionary mode (search for words)
echo -e "crypto\nbitcoin\nwallet" > words.txt
./vanity-ext-id-cuda -p pool_100k.bin -d words.txt -o dict_results.csv
```

### 4. Extract Private Key

```bash
# Get prime indices from CSV (e.g., p_idx=1234, q_idx=5678)
python3 ../tools/reconstruct_key.py pool_100k.bin 1234 5678 > key.pem

# Get public key for manifest.json
openssl rsa -in key.pem -pubout -outform DER | base64 -w 0
```

---

## How the GPU Search Works

### CPU Approach (Slow):
```cpp
for (int i = 0; i < primes.size(); i++) {
  for (int j = i+1; j < primes.size(); j++) {
    n = primes[i] * primes[j];           // Multiply
    der = encode_der(n);                 // Encode
    hash = sha256(der);                  // Hash
    id = to_extension_id(hash);          // Convert
    if (matches_pattern(id)) save(i,j);  // Check
  }
}
```

**Problem:** Testing 5 billion pairs takes **hours** on CPU!

### GPU Approach (Fast):

**Key Insight:** All operations (multiply, SHA-256, pattern matching) can run **in parallel** on thousands of GPU threads!

```
GPU has 10,752 CUDA cores (RTX 3080)
    ↓
Launch 10,752 threads in parallel
    ↓
Each thread tests one (p,q) pair
    ↓
~300 million pairs/second!
```

### CUDA Kernel Pipeline:

```cuda
__global__ void vanity_search_kernel(...) {
  // Each thread processes one (i, j) pair
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y + i + 1;

  // 1. Load primes from global memory
  prime_p = load_prime(pool, i);
  prime_q = load_prime(pool, j);

  // 2. Multiply 1024×1024 → 2048 bits
  n = bigint_mul_1024(prime_p, prime_q);

  // 3. Build DER encoding (294 bytes)
  der = build_der_pubkey(n);

  // 4. SHA-256 hash
  hash = sha256(der);

  // 5. Convert to extension ID
  ext_id = hash_to_extension_id(hash);

  // 6. Check pattern match
  if (matches(ext_id, pattern)) {
    save_match(ext_id, i, j);
  }
}
```

**Optimizations:**
- **Coalesced memory access** - threads read adjacent primes
- **Shared memory** - cache SHA-256 constants
- **Register optimization** - minimize memory traffic
- **2D grid** - efficient (i,j) pair distribution

---

## Prime Generation: CPU vs GPU

> 📊 **Want the full technical breakdown?** See [PERFORMANCE.md](PERFORMANCE.md) for a deep dive into how CGBN achieves 20x speedup!

### CPU Method (OpenSSL):
```cpp
BIGNUM* prime = BN_new();
BN_generate_prime_ex(prime, 1024, ...);  // Miller-Rabin built-in
```

**Speed:** ~50-100 primes/second (multithreaded on 16 cores)

### GPU Method (CGBN):

**CGBN** = CUDA GPU BigNum library by NVIDIA
- Optimized for **arbitrary-precision arithmetic** on GPU
- Uses **Montgomery multiplication** for fast modular math
- Parallel Miller-Rabin testing across thousands of threads

```cuda
// 1. Generate random 1024-bit candidates on GPU
for (int i = 0; i < 100000; i++) {
  candidate[i] = curand_generate_1024bit();
}

// 2. Test all candidates in parallel (CGBN)
__global__ void miller_rabin_kernel(...) {
  // Montgomery form for fast modular exponentiation
  cgbn_powm(result, base, power, candidate);
  if (is_probably_prime(result)) {
    save_prime(candidate);
  }
}
```

**Speed:** ~1150 primes/second (RTX 3080)
**Speedup:** **20x faster than CPU!**

### Miller-Rabin Primality Test

**How it works:**
```
1. Write n-1 = 2^r × d (factor out powers of 2)
2. Pick random base a
3. Compute x = a^d mod n (using Montgomery multiplication)
4. Check if x = 1 or x = n-1
5. If not, square x repeatedly (r-1 times)
6. If we ever get n-1, probably prime
7. Otherwise, definitely composite

Repeat 20 times with different bases
→ Error probability < 2^-80 (cryptographically negligible)
```

**GPU Advantage:** Test 10,000 candidates **simultaneously**!

---

## Performance

### Prime Generation (100K primes):

| Tool                      | Time      | Rate          | Hardware      |
|---------------------------|-----------|---------------|---------------|
| `generate-prime-pool`     | ~30 min   | ~55 primes/s  | 16-core CPU   |
| `generate-prime-pool-cgbn`| **90 sec**| **~1150 primes/s** | **RTX 3080** |

**Speedup: 20x**

### Vanity Search (100K pool = 5B pairs):

| Pattern           | Matches Found | Time    | Rate           |
|-------------------|---------------|---------|----------------|
| 3-char prefix     | ~40,000       | 20 sec  | 250M pairs/s   |
| 4-char prefix     | ~2,500        | 20 sec  | 250M pairs/s   |
| 5-char prefix     | ~150          | 20 sec  | 250M pairs/s   |
| Dictionary (10K words) | ~500,000 | 25 sec | 200M pairs/s |

**Note:** Search time is constant (tests all pairs), matches depend on pattern rarity.

### GPU Utilization:

```
RTX 3080 during search:
├─ GPU: 100%
├─ Memory: 2.5 GB / 10 GB
├─ Power: 300W / 320W
└─ Compute: 10,752 cores @ 100%
```

---

## Usage Guide

### Prime Pool Generation

**Small pool (10K primes, ~1 MB, 10 seconds):**
```bash
./generate-prime-pool-cgbn 10000 pool_10k.bin
```

**Medium pool (100K primes, ~13 MB, 90 seconds):**
```bash
./generate-prime-pool-cgbn 100000 pool_100k.bin
```

**Large pool (1M primes, ~128 MB, 15 minutes):**
```bash
./generate-prime-pool-cgbn 1000000 pool_1m.bin
```

### Vanity Search Modes

#### 1. Prefix/Suffix Search
```bash
./vanity-ext-id-cuda -p pool.bin -s cafe        # Starts with "cafe"
./vanity-ext-id-cuda -p pool.bin -e dead        # Ends with "dead"
./vanity-ext-id-cuda -p pool.bin -s ca -e fe    # Both!
```

#### 2. Dictionary Search
```bash
# Create wordlist
echo -e "crypto\nbitcoin\nwallet\nmetamask" > words.txt

./vanity-ext-id-cuda -p pool.bin -d words.txt -o results.csv --min-len 5
```

Searches for **any word** from the dictionary at **start or end** of ID.

**Auto pattern detection:**
- Repeated chars (6+): `aaaaaa`, `pppppppp`
- Sequences: `abcdefg`, `ponmlkji`
- Palindromes (8+): `abcdcba...`

#### 3. AI Mode (Fast!)
```bash
./vanity-ext-id-cuda -p pool.bin --ai 7 -o ai_results.csv
```

Finds IDs with **7+ occurrences of "ai"** OR **10+ consecutive duplicates**.
- Pure GPU mode (no CPU post-processing)
- Fastest search mode!

#### 4. Validation (Recommended)
```bash
./vanity-ext-id-cuda -p pool.bin -d words.txt --validate -o results.csv
```

Validates primes with Miller-Rabin before saving (filters out false positives).

---

## File Formats

### Prime Pool (`.bin`):

```c
struct PrimePoolHeader {
    uint32_t magic;         // 0x504D5250 ("PRMP")
    uint32_t version;       // 1
    uint32_t count;         // Number of primes
    uint32_t prime_bytes;   // 128
};

// Followed by N × 128-byte primes (big-endian)
```

### Results CSV:

```csv
ext_id,matches,p_idx,q_idx,validated
cafebeef00000000000000000000aaaa,cafe@start;aaaa@end,1234,5678,true
deadface11111111111111111111bbbb,dead@start,2345,6789,true
```

**Columns:**
- `ext_id` - 32-char extension ID
- `matches` - What matched (word@position or pattern:type)
- `p_idx` - Index of prime p in pool
- `q_idx` - Index of prime q in pool
- `validated` - Miller-Rabin validation passed

---

## Optimization

### Pool Size vs Search Time

```
Pool Size | Unique Pairs    | Search Time (RTX 3080)
----------|-----------------|------------------------
1K        | 500K            | <1 sec
10K       | 50M             | ~0.2 sec
100K      | 5B              | ~20 sec
1M        | 500B            | ~30 min
```

**Recommendation:**
- **100K pool** for most use cases (good balance)
- **1M pool** for rare patterns (5+ char prefixes)

### Match Probability

Probability of finding N-character prefix in M pairs:

```
P(match) ≈ M / 16^N

Examples (100K pool = 5B pairs):
- 3 chars: 5B / 16^3 ≈ 1,220,000 matches (guaranteed)
- 4 chars: 5B / 16^4 ≈ 76,000 matches (guaranteed)
- 5 chars: 5B / 16^5 ≈ 4,700 matches (likely)
- 6 chars: 5B / 16^6 ≈ 300 matches (possible)
- 7 chars: 5B / 16^7 ≈ 18 matches (rare)
- 8 chars: 5B / 16^8 ≈ 1 match (very rare)
```

**For guaranteed 6+ char matches:** Use 1M prime pool!

### CGBN Parameters (Auto-Optimized)

The CGBN prime generator has been **tuned via grid search**:

```cpp
// Optimized settings (1150 primes/s on RTX 3080):
TPI = 8           // Threads per instance
WINDOW_BITS = 4   // Modular exponentiation window
TPB = 128         // Threads per block
BATCH_SIZE = 50000 // Candidates per batch
```

Found by testing 36+ configurations. **Already optimal for RTX 3080!**

---

## Requirements

- **GPU:** NVIDIA with Compute Capability 6.1+ (GTX 1070+, RTX 20xx/30xx/40xx)
- **CUDA:** 12.0+ (tested with 12.6)
- **CMake:** 3.18+
- **OpenSSL:** 3.0+
- **GMP:** libgmp-dev (for CGBN)

### Installation (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libssl-dev libgmp-dev
```

---

## Troubleshooting

### "CUDA Error: out of memory"

**Cause:** Prime pool too large for GPU RAM.

**Solution:** Reduce batch size in `src/cuda/cuda_runner.cu`.

### "No matches found"

**Causes:**
1. Pattern too specific (6+ chars with small pool)
2. Invalid characters (use only a-p)

**Solutions:**
- Use larger prime pool (1M instead of 100K)
- Try shorter pattern

### Slow prime generation on older GPUs

**Issue:** CGBN is optimized for Volta+ (sm_70+).

**Solution:** Use CPU version for older GPUs:
```bash
./generate-prime-pool 100000 pool.bin
```

---

## Technical Details

### Why RSA-2048?

Chrome uses **RSA-2048** for extension signatures:
- **Security:** 2048-bit RSA ≈ 112-bit symmetric security
- **Speed:** Fast enough for signature verification
- **Standard:** Widely supported

### Why Pre-compute Primes?

**On-the-fly generation:**
```
Generate p (20ms) + Generate q (20ms) + Check ID (0.01ms) = 40ms
→ 25 attempts/second
→ 1M attempts = 11 hours
```

**Pre-computed pool:**
```
Generate 100K primes once (90s) + Test 5B pairs (20s) = 110s total
→ Amortized: infinitely faster!
```

### SHA-256 to Extension ID

```python
hash = sha256(der_pubkey)  # 32 bytes

ext_id = ""
for i in range(16):
    byte = hash[i]
    ext_id += chr(ord('a') + (byte >> 4))      # High nibble
    ext_id += chr(ord('a') + (byte & 0x0F))    # Low nibble

# Result: 32 characters, a-p only
```

**Why a-p?**
- 16 letters = 4 bits per character
- 32 chars × 4 bits = 128 bits (first half of SHA-256)

---

## Credits

**CGBN Library:**
- https://github.com/NVlabs/CGBN
- NVIDIA's official big-number library for CUDA

**Chrome Extension Format:**
- https://developer.chrome.com/docs/extensions/mv3/manifest/key/

Built with CUDA 12.6, CGBN, OpenSSL 3.0, and pure GPU power! 🚀

---

## License

MIT License - do whatever you want!

**Warning:** Generating vanity IDs for malicious extensions is unethical and probably illegal. Use responsibly!

---

## FAQ

**Q: Will Chrome ban my extension?**
A: No! The ID is cryptographically valid.

**Q: Can I use this offline?**
A: Yes! All tools work completely offline.

**Q: Can someone else generate the same ID?**
A: Probability: 1 in 2^128 ≈ 10^38 (essentially impossible)

**Q: Why not just use UUIDs?**
A: Chrome specifically uses RSA-based IDs for security/signing.

---

**Now go make some sick extension IDs!** 🎯
