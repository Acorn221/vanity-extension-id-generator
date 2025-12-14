#!/usr/bin/env python3
"""
Reconstruct RSA keypair from compact CSV format.
Usage: python reconstruct.py
       Then paste a line from overnight.csv
"""

from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateNumbers, RSAPublicNumbers, rsa_crt_iqmp, rsa_crt_dmp1, rsa_crt_dmq1
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import hashlib

def reconstruct_key(p_hex: str, q_hex: str):
    """Reconstruct RSA private key from primes p and q"""
    p = int(p_hex, 16)
    q = int(q_hex, 16)
    e = 65537
    
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

def verify_extension_id(private_key, expected_id: str) -> bool:
    """Verify the key produces the expected extension ID"""
    public_key = private_key.public_key()
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    hash_bytes = hashlib.sha256(der).digest()[:16]
    ext_id = ''.join(chr(ord('a') + (b >> 4)) + chr(ord('a') + (b & 0x0f)) for b in hash_bytes)
    
    return ext_id == expected_id

def main():
    print("Paste a line from overnight.csv (or type 'q' to quit):")
    print("Format: ext_id,matches,p_hex,q_hex,pub_b64")
    print("-" * 60)
    
    while True:
        try:
            line = input("\n> ").strip()
            if line.lower() == 'q':
                break
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 4:
                print("Error: Need at least 4 comma-separated values")
                continue
            
            ext_id = parts[0]
            matches = parts[1]
            p_hex = parts[2]
            q_hex = parts[3]
            
            print(f"\nExtension ID: {ext_id}")
            print(f"Matches: {matches}")
            print("\nReconstructing key...")
            
            key = reconstruct_key(p_hex, q_hex)
            
            # Verify
            if verify_extension_id(key, ext_id):
                print("✓ Verified: Extension ID matches!")
            else:
                print("⚠ Warning: Extension ID mismatch!")
            
            # Export PEM
            pem = key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode()
            
            # Export public key base64 for manifest.json
            pub_der = key.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            import base64
            pub_b64 = base64.b64encode(pub_der).decode()
            
            print("\n" + "=" * 60)
            print("PRIVATE KEY (save as key.pem):")
            print("=" * 60)
            print(pem)
            
            print("=" * 60)
            print("PUBLIC KEY for manifest.json:")
            print("=" * 60)
            print(f'"key": "{pub_b64}"')
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

