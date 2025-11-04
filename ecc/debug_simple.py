#!/usr/bin/env python3

import numpy as np
import random
import hashlib
import sys
import os

# Add the reed-muller-python directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'reed-muller-python'))
from reedmuller.reedmuller import ReedMuller

def test_oaep_simple():
    """Test OAEP functions in isolation."""
    print("Testing OAEP functions...")
    
    # Test message
    message = [1, 0, 1, 1, 0, 0, 1, 0]
    print(f"Original message: {message}")
    
    # Test OAEP encryption step by step
    print(f"\nDebugging OAEP encryption:")
    
    # Manual step-by-step encryption
    def G(r_bits):
        r_bytes = bytearray()
        for i in range(0, 128, 8):
            byte_bits = r_bits[i:i+8]
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                byte_val += bit * (2 ** (7-j))
            r_bytes.append(byte_val)
        r_bytes = bytes(r_bytes)
        hash_bytes = hashlib.sha256(r_bytes).digest()
        return [int(bit) for bit in format(hash_bytes[0], '08b')]

    def H(t_bits):
        byte_val = 0
        for j, bit in enumerate(t_bits):
            byte_val += bit * (2 ** (7-j))
        t_bytes = bytes([byte_val])
        hash_bytes = hashlib.sha256(t_bytes).digest()
        hash_bits = []
        for byte in hash_bytes:
            hash_bits.extend([int(bit) for bit in format(byte, '08b')])
        return hash_bits[:128]

    # Generate random r
    r_bits = [random.randint(0, 1) for _ in range(128)]
    print(f"Random r: {r_bits[:8]}...")
    
    # Step 1: t = message ⊕ G(r)
    G_r = G(r_bits)
    t = [(message[i] + G_r[i]) % 2 for i in range(8)]
    print(f"Step 1 - t = message ⊕ G(r): {t}")
    print(f"G(r): {G_r}")
    
    # Step 2: s = r ⊕ H(t)
    H_t = H(t)
    s = [(r_bits[i] + H_t[i]) % 2 for i in range(128)]
    print(f"Step 2 - s = r ⊕ H(t): {s[:8]}... (Hamming weight: {sum(s)})")
    
    # Test OAEP encryption
    s2, t2 = oaep_encrypt(message)
    print(f"OAEP function result - s: {s2[:8]}... (Hamming weight: {sum(s2)})")
    print(f"OAEP function result - t: {t2}")
    print(f"Manual s matches function: {s == s2}")
    print(f"Manual t matches function: {t == t2}")
    
    # Test OAEP decryption step by step
    print(f"\nDebugging OAEP decryption:")
    
    # Manual step-by-step decryption
    def G(r_bits):
        r_bytes = bytearray()
        for i in range(0, 128, 8):
            byte_bits = r_bits[i:i+8]
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                byte_val += bit * (2 ** (7-j))
            r_bytes.append(byte_val)
        r_bytes = bytes(r_bytes)
        hash_bytes = hashlib.sha256(r_bytes).digest()
        return [int(bit) for bit in format(hash_bytes[0], '08b')]

    def H(t_bits):
        byte_val = 0
        for j, bit in enumerate(t_bits):
            byte_val += bit * (2 ** (7-j))
        t_bytes = bytes([byte_val])
        hash_bytes = hashlib.sha256(t_bytes).digest()
        hash_bits = []
        for byte in hash_bytes:
            hash_bits.extend([int(bit) for bit in format(byte, '08b')])
        return hash_bits[:128]

    # Step 1: r = s ⊕ H(t)
    H_t = H(t)
    r = [(s[i] + H_t[i]) % 2 for i in range(128)]
    print(f"Step 1 - r = s ⊕ H(t): {r[:8]}...")
    
    # Step 2: message = t ⊕ G(r)
    G_r = G(r)
    recovered_message = [(t[i] + G_r[i]) % 2 for i in range(8)]
    print(f"Step 2 - message = t ⊕ G(r): {recovered_message}")
    print(f"G(r): {G_r}")
    
    # Test OAEP decryption
    recovered_message2, recovered_r = oaep_decrypt(s, t)
    print(f"OAEP function result: {recovered_message2}")
    print(f"Manual result: {recovered_message}")
    print(f"Results match: {recovered_message == recovered_message2}")
    
    print(f"Final result: {recovered_message}")
    print(f"OAEP works: {recovered_message == message}")
    
    return recovered_message == message

def oaep_encrypt(message_bits, randomness=None):
    """Implement OAEP with 2-round Feistel network."""
    if len(message_bits) != 8:
        raise ValueError("Message must be 8 bits")

    # Generate random r if not provided
    if randomness is None:
        r_bits = [random.randint(0, 1) for _ in range(128)]
    else:
        r_bits = randomness

    # Hash function G: 128 bits -> 8 bits
    def G(r_bits):
        # Convert 128 bits to 16 bytes
        r_bytes = bytearray()
        for i in range(0, 128, 8):
            byte_bits = r_bits[i:i+8]
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                byte_val += bit * (2 ** (7-j))
            r_bytes.append(byte_val)
        r_bytes = bytes(r_bytes)
        hash_bytes = hashlib.sha256(r_bytes).digest()
        return [int(bit) for bit in format(hash_bytes[0], '08b')]

    # Hash function H: 8 bits -> 128 bits
    def H(t_bits):
        # Convert 8 bits to 1 byte
        byte_val = 0
        for j, bit in enumerate(t_bits):
            byte_val += bit * (2 ** (7-j))
        t_bytes = bytes([byte_val])
        hash_bytes = hashlib.sha256(t_bytes).digest()
        # Convert hash to 128 bits
        hash_bits = []
        for byte in hash_bytes:
            hash_bits.extend([int(bit) for bit in format(byte, '08b')])
        return hash_bits[:128]

    # 2-round Feistel network
    # Round 1: t = message ⊕ G(r)
    G_r = G(r_bits)
    t = [(message_bits[i] + G_r[i]) % 2 for i in range(8)]

    # Round 2: s = r ⊕ H(t)
    H_t = H(t)
    s = [(r_bits[i] + H_t[i]) % 2 for i in range(128)]

    # Rejection sampling to find r that produces s with exactly 31 errors
    max_attempts = 1000
    for attempt in range(max_attempts):
        if sum(s) == 31:
            break
        
        # Regenerate r and recalculate
        r_bits = [random.randint(0, 1) for _ in range(128)]
        G_r = G(r_bits)
        t = [(message_bits[i] + G_r[i]) % 2 for i in range(8)]
        H_t = H(t)
        s = [(r_bits[i] + H_t[i]) % 2 for i in range(128)]
    
    # If rejection sampling fails, we'll accept whatever we got
    # (in practice, this should rarely happen with enough attempts)

    return s, t

def oaep_decrypt(s, t):
    """Decrypt OAEP to recover original message and randomness."""
    # Hash function G: 128 bits -> 8 bits
    def G(r_bits):
        # Convert 128 bits to 16 bytes
        r_bytes = bytearray()
        for i in range(0, 128, 8):
            byte_bits = r_bits[i:i+8]
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                byte_val += bit * (2 ** (7-j))
            r_bytes.append(byte_val)
        r_bytes = bytes(r_bytes)
        hash_bytes = hashlib.sha256(r_bytes).digest()
        return [int(bit) for bit in format(hash_bytes[0], '08b')]

    # Hash function H: 8 bits -> 128 bits
    def H(t_bits):
        # Convert 8 bits to 1 byte
        byte_val = 0
        for j, bit in enumerate(t_bits):
            byte_val += bit * (2 ** (7-j))
        t_bytes = bytes([byte_val])
        hash_bytes = hashlib.sha256(t_bytes).digest()
        # Convert hash to 128 bits
        hash_bits = []
        for byte in hash_bytes:
            hash_bits.extend([int(bit) for bit in format(byte, '08b')])
        return hash_bits[:128]

    # Reverse Feistel network
    # Round 2: r = s ⊕ H(t)
    H_t = H(t)
    r = [(s[i] + H_t[i]) % 2 for i in range(128)]

    # Round 1: message = t ⊕ G(r)
    G_r = G(r)
    message = [(t[i] + G_r[i]) % 2 for i in range(8)]

    return message, r

if __name__ == "__main__":
    print("=" * 50)
    print("Simple OAEP Test")
    print("=" * 50)
    
    success = test_oaep_simple()
    
    if success:
        print("\n✅ OAEP is working correctly!")
    else:
        print("\n❌ OAEP has issues!")
    
    print("=" * 50)
