#!/usr/bin/env python3
"""
Reconstruct RSA keypair from GPU search results.

Usage:
  python reconstruct_gpu.py -p prime_pool.bin -c results.csv
  python reconstruct_gpu.py -p prime_pool.bin --interactive

CSV format from GPU: ext_id,matches,p_idx,q_idx
"""

import argparse
import struct
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateNumbers, RSAPublicNumbers,
    rsa_crt_iqmp, rsa_crt_dmp1, rsa_crt_dmq1
)
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import hashlib
import base64


def load_prime_pool(filepath: str) -> list[bytes]:
    """Load primes from binary pool file."""
    with open(filepath, 'rb') as f:
        # Read header: magic, version, count, prime_bytes
        header = f.read(16)
        magic, version, count, prime_bytes = struct.unpack('<IIII', header)

        if magic != 0x504D5250:  # 'PRMP'
            raise ValueError(f"Invalid prime pool file (bad magic: {hex(magic)})")

        if prime_bytes != 128:
            raise ValueError(f"Unexpected prime size: {prime_bytes}")

        primes = []
        for _ in range(count):
            prime_data = f.read(prime_bytes)
            if len(prime_data) != prime_bytes:
                break
            primes.append(prime_data)

        print(f"Loaded {len(primes)} primes from {filepath}")
        return primes


def reconstruct_key(p_bytes: bytes, q_bytes: bytes):
    """Reconstruct RSA private key from prime bytes."""
    p = int.from_bytes(p_bytes, 'big')
    q = int.from_bytes(q_bytes, 'big')
    e = 65537

    # Ensure p > q for CRT
    if p < q:
        p, q = q, p

    n = p * q
    phi = (p - 1) * (q - 1)
    d = pow(e, -1, phi)

    # CRT components
    dmp1 = rsa_crt_dmp1(d, p)
    dmq1 = rsa_crt_dmq1(d, q)
    iqmp = rsa_crt_iqmp(p, q)

    public_numbers = RSAPublicNumbers(e, n)
    private_numbers = RSAPrivateNumbers(p, q, d, dmp1, dmq1, iqmp, public_numbers)

    return private_numbers.private_key(default_backend())


def compute_extension_id(private_key) -> str:
    """Compute Chrome extension ID from private key."""
    public_key = private_key.public_key()
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    hash_bytes = hashlib.sha256(der).digest()[:16]
    ext_id = ''.join(chr(ord('a') + (b >> 4)) + chr(ord('a') + (b & 0x0f)) for b in hash_bytes)

    return ext_id


def export_key(private_key, ext_id: str, output_dir: Path = None):
    """Export key in PEM format and show manifest.json key."""
    # Verify extension ID
    computed_id = compute_extension_id(private_key)
    if computed_id != ext_id:
        print(f"⚠ Warning: Extension ID mismatch!")
        print(f"  Expected: {ext_id}")
        print(f"  Computed: {computed_id}")
    else:
        print(f"✓ Verified: Extension ID matches!")

    # Export PEM
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode()

    # Export public key base64 for manifest.json
    pub_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    pub_b64 = base64.b64encode(pub_der).decode()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        key_file = output_dir / f"{ext_id}.pem"
        key_file.write_text(pem)
        print(f"\nSaved: {key_file}")
    else:
        print("\n" + "=" * 60)
        print("PRIVATE KEY (save as key.pem):")
        print("=" * 60)
        print(pem)

    print("=" * 60)
    print("PUBLIC KEY for manifest.json:")
    print("=" * 60)
    print(f'"key": "{pub_b64}"')
    print("=" * 60)

    return pem, pub_b64


def process_csv_line(line: str, primes: list[bytes], output_dir: Path = None):
    """Process a single CSV line and reconstruct the key."""
    parts = line.strip().split(',')
    if len(parts) < 4:
        print(f"Error: Need at least 4 comma-separated values, got {len(parts)}")
        return None

    ext_id = parts[0]
    matches = parts[1]
    p_idx = int(parts[2])
    q_idx = int(parts[3])

    print(f"\nExtension ID: {ext_id}")
    print(f"Matches: {matches}")
    print(f"Prime indices: p={p_idx}, q={q_idx}")

    if p_idx >= len(primes) or q_idx >= len(primes):
        print(f"Error: Prime index out of range (pool has {len(primes)} primes)")
        return None

    p_bytes = primes[p_idx]
    q_bytes = primes[q_idx]

    print("Reconstructing key...")
    key = reconstruct_key(p_bytes, q_bytes)

    return export_key(key, ext_id, output_dir)


def interactive_mode(primes: list[bytes]):
    """Interactive mode - paste CSV lines one at a time."""
    print("\nPaste a line from GPU results CSV (or 'q' to quit):")
    print("Format: ext_id,matches,p_idx,q_idx")
    print("-" * 60)

    while True:
        try:
            line = input("\n> ").strip()
            if line.lower() == 'q':
                break
            if not line:
                continue

            process_csv_line(line, primes)

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(primes: list[bytes], csv_file: str, output_dir: Path = None, limit: int = None):
    """Process entire CSV file."""
    with open(csv_file, 'r') as f:
        lines = f.readlines()

    # Skip header if present
    if lines and lines[0].startswith('ext_id'):
        lines = lines[1:]

    if limit:
        lines = lines[:limit]

    print(f"\nProcessing {len(lines)} entries from {csv_file}...")

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        print(f"\n[{i}/{len(lines)}] ", end="")
        try:
            process_csv_line(line, primes, output_dir)
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Reconstruct RSA keys from GPU search results')
    parser.add_argument('-p', '--pool', required=True, help='Prime pool binary file')
    parser.add_argument('-c', '--csv', help='CSV file with results (batch mode)')
    parser.add_argument('-o', '--output', help='Output directory for PEM files')
    parser.add_argument('-n', '--limit', type=int, help='Limit number of keys to process')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    # Load prime pool
    try:
        primes = load_prime_pool(args.pool)
    except Exception as e:
        print(f"Error loading prime pool: {e}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else None

    if args.csv:
        batch_mode(primes, args.csv, output_dir, args.limit)
    elif args.interactive:
        interactive_mode(primes)
    else:
        # Default to interactive if no CSV provided
        interactive_mode(primes)


if __name__ == "__main__":
    main()
