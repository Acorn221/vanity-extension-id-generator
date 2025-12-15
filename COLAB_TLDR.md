# Chrome Extension Vanity ID Generator - TLDR

## Problem
Chrome extension IDs are derived from RSA public keys. We want to brute-force find keys whose extension ID contains specific words.

## How Extension IDs Work
1. Generate RSA-2048 key pair
2. Export public key as SPKI/DER format
3. SHA-256 hash it
4. Take first 16 bytes → convert each hex digit to letter a-p (0=a, 1=b, ... f=p)
5. Result: 32-char ID like `cafedeadbeefabcd...`

## Valid Characters
Only letters **a through p** (representing hex 0-f). Words like "cafe", "dead", "beef", "linkedin" work. Words with q-z don't.

## Algorithm
```
loop:
    key = generate_rsa_2048()
    der = export_public_key_spki_der(key)
    hash = sha256(der)
    ext_id = hex_to_ap(hash[0:16])  # first 16 bytes → 32 chars
    
    if ext_id matches target word(s):
        save(key)
```

## Performance
- RSA-2048 generation is the bottleneck (~50-100μs per key)
- Target: 10,000-20,000 keys/sec with parallelism
- Use all CPU cores

## Storage Optimization
Instead of saving full PEM (~1700 bytes), save just primes p and q (~256 bytes). Full key can be reconstructed from p,q since n=p*q and d=e^(-1) mod φ(n).

