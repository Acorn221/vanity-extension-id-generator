# Vanity Extension ID Generator - CUDA Multi-GPU Version

High-performance vanity Chrome extension ID generator using CUDA for 8x A100 GPUs.

## Features

- **Multi-GPU Support**: Automatically uses all available CUDA GPUs
- **8x A100 Optimized**: Tuned for NVIDIA A100 (sm_80) with support for sm_86/sm_89
- **Prime Validation**: Miller-Rabin primality test filters invalid primes
- **Dictionary Mode**: Search for words from a wordlist
- **Pattern Detection**: Find cool patterns (palindromes, sequences, repeats)
- **Resume Support**: State file allows resuming interrupted searches

## Requirements

- NVIDIA GPU with CUDA Compute Capability 8.0+ (A100, RTX 30xx, RTX 40xx)
- CUDA Toolkit 11.0+
- OpenSSL 3.0+
- CMake 3.18+

## Building

```bash
cd vanity-ext-id-cuda

# Create build directory
mkdir build && cd build

# Configure with Release mode
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
make -j$(nproc)
```

## Usage

### Basic Search

```bash
# Search for IDs starting with "cafe"
./vanity-ext-id-cuda -p prime_pool_100m.bin -s cafe -o results.csv

# Search for IDs ending with "dead"
./vanity-ext-id-cuda -p prime_pool_100m.bin -e dead -o results.csv

# Search for both prefix and suffix
./vanity-ext-id-cuda -p prime_pool_100m.bin -s cia -e fbi -o results.csv
```

### Dictionary Search

```bash
# Search with dictionary
./vanity-ext-id-cuda -p prime_pool_100m.bin -d wordlist.txt -o results.csv --validate
```

### Options

```
Required:
  -p, --pool FILE     Prime pool file (from generate-prime-pool)

Search Options:
  -d, --dict FILE     Search for words from dictionary file
  -s, --start STR     Find IDs starting with STR
  -e, --end STR       Find IDs ending with STR

Output Options:
  -o, --output FILE   Output CSV file (default: cuda_results.csv)
  --min-len N         Minimum word length for dictionary (default: 3)
  --max-len N         Maximum word length for dictionary (default: 10)

Validation Options:
  --validate          Validate primes before saving (recommended)
  --no-validate       Skip prime validation (faster but may include invalid keys)

Other Options:
  -h, --help          Show help message
```

## Prime Pool

Generate a prime pool using the included tool:

```bash
# Generate 100 million primes (takes ~1-2 hours on CPU)
./generate-prime-pool 100000000 prime_pool_100m.bin
```

**Note**: About 8% of primes in a large pool may be invalid (false positives from generation). Use `--validate` to filter these out when searching.

## Output Format

Results are saved as CSV:

```
ext_id,matches,p_idx,q_idx,validated
cafebabecafebabecafebabecafebabe,cafe@start,12345,67890,true
```

Use the included `reconstruct.py` script to reconstruct full RSA keys:

```bash
python3 reconstruct.py results.csv prime_pool_100m.bin
```

## Performance

Expected performance on 8x A100-80GB:

| Pool Size | Pairs | Estimated Time |
|-----------|-------|----------------|
| 50K | 1.25B | ~1 second |
| 1M | 500B | ~5 minutes |
| 10M | 50T | ~8 hours |
| 100M | 5P | ~3-7 days |

Actual performance depends on:
- GPU interconnect (NVLink vs PCIe)
- Memory bandwidth
- Dictionary size
- Match rate (more matches = more validation overhead)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Host CPU                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │  Prime   │  │  Work    │  │  Prime   │  │   CSV   ││
│  │  Pool    │  │ Scheduler│  │ Validator│  │  Writer ││
│  │ (12.8GB) │  │          │  │(OpenSSL) │  │         ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘│
└───────┼─────────────┼─────────────┼─────────────┼──────┘
        │             │             │             │
   ┌────▼────┐   ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
   │  GPU 0  │   │  GPU 1  │  │  GPU 2  │  │  GPU 7  │
   │ Rows    │   │ Rows    │  │ Rows    │  │ Rows    │
   │ 0-12.5M │   │12.5-25M │  │  ...    │  │87.5-100M│
   │         │   │         │  │         │  │         │
   │ ┌─────┐ │   │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │
   │ │Kern │ │   │ │Kern │ │  │ │Kern │ │  │ │Kern │ │
   │ └─────┘ │   │ └─────┘ │  │ └─────┘ │  │ └─────┘ │
   └─────────┘   └─────────┘  └─────────┘  └─────────┘
```

Each kernel:
1. Loads two primes from pool
2. Computes n = p × q (1024×1024 → 2048 bit)
3. Builds DER-encoded public key
4. Computes SHA-256 hash
5. Converts to extension ID
6. Checks patterns/dictionary
7. Reports matches atomically

## Troubleshooting

### Out of Memory
- Reduce batch size in search config
- Use a smaller prime pool
- Check available GPU memory with `nvidia-smi`

### CUDA Not Found
```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Wrong GPU Architecture
The default build targets sm_80 (A100). For other GPUs, modify `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt:
- RTX 3090: sm_86
- RTX 4090: sm_89
- V100: sm_70
- T4: sm_75

## License

MIT License
