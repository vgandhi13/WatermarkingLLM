#!/usr/bin/env python3
"""
Test script for McEliece with Reed-Muller codes implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mceliece_reed_muller import McElieceReedMuller, ReedMullerCode, print_binary
import numpy as np

def test_reed_muller_code():
    """Test the Reed-Muller code wrapper."""
    print("Testing Reed-Muller Code Wrapper...")
    
    # Test RM(1,6) - small code for testing
    rm = ReedMullerCode(1, 6)
    print(f"RM(1,6) parameters: n={rm.n}, k={rm.k}, t={rm.t}")
    
    # Test encoding/decoding - use a message that fits in 7 bits
    test_message = b"H"  # 1 byte = 8 bits, but we'll truncate to 7 bits
    print(f"\nTesting message: {test_message}")
    
    try:
        # Encode
        encoded = rm.encode(test_message)
        print_binary(encoded, "Encoded Message")
        
        # Decode
        decoded = rm.decode(encoded)
        print_binary(decoded, "Decoded Message")
        
        success = test_message == decoded
        print(f"Encoding/Decoding Success: {success}")
        
        # Test error correction
        print("\nTesting error correction...")
        corrupted = rm.introduce_errors(encoded, 5)  # Introduce 5 errors
        print_binary(corrupted, "Corrupted Message (5 errors)")
        
        corrected = rm.decode(corrupted)
        print_binary(corrected, "Corrected Message")
        
        correction_success = test_message == corrected
        print(f"Error Correction Success: {correction_success}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def test_mceliece_encryption():
    """Test the McEliece encryption/decryption."""
    print("\n" + "="*60)
    print("Testing McEliece Encryption/Decryption")
    print("="*60)
    
    # Test with RM(1,6) - small enough for testing
    try:
        mc = McElieceReedMuller(1, 6)
        print(f"McEliece parameters: n={mc.n}, k={mc.k}, t={mc.t}")
        
        # Test messages - use small messages that fit in the code
        test_messages = [
            b"H",  # 1 byte = 8 bits
            b"Hi",  # 2 bytes = 16 bits
        ]
        
        for msg in test_messages:
            if len(msg) * 8 > mc.k:
                print(f"Skipping '{msg}' - too long ({len(msg)*8} bits > {mc.k} bits)")
                continue
                
            print(f"\nTesting message: {msg}")
            print_binary(msg, "Original Message")
            
            try:
                # Encrypt
                cipher, num_errors = mc.encrypt(msg)
                print_binary(cipher, f"Encrypted Message ({num_errors} errors)")
                
                # Decrypt
                decrypted = mc.decrypt(cipher)
                print_binary(decrypted, "Decrypted Message")
                
                success = msg == decrypted
                print(f"Encryption/Decryption Success: {success}")
                
                if not success:
                    print(f"Expected: {msg}")
                    print(f"Got: {decrypted}")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                
    except Exception as e:
        print(f"Failed to initialize McEliece: {str(e)}")

def test_different_parameters():
    """Test different Reed-Muller parameters."""
    print("\n" + "="*60)
    print("Testing Different Reed-Muller Parameters")
    print("="*60)
    
    # Test different parameter combinations
    test_params = [
        (1, 5),  # RM(1,5): n=32, k=6, t=7
        (1, 6),  # RM(1,6): n=64, k=7, t=15
        (2, 5),  # RM(2,5): n=32, k=16, t=3
    ]
    
    for r, m in test_params:
        print(f"\nTesting RM({r},{m})...")
        try:
            mc = McElieceReedMuller(r, m)
            print(f"Parameters: n={mc.n}, k={mc.k}, t={mc.t}")
            
            # Test with a simple message
            test_msg = b"H"  # 1 byte
            if len(test_msg) * 8 <= mc.k:
                cipher, num_errors = mc.encrypt(test_msg)
                decrypted = mc.decrypt(cipher)
                success = test_msg == decrypted
                print(f"Test result: {success}")
            else:
                print("Message too long for this code")
                
        except Exception as e:
            print(f"Failed: {str(e)}")

def test_security_analysis():
    """Basic security analysis - check key sizes and error correction."""
    print("\n" + "="*60)
    print("Security Analysis")
    print("="*60)
    
    # Test different parameter sets and analyze security
    security_params = [
        (1, 8),   # RM(1,8): n=256, k=9, t=63
        (2, 8),   # RM(2,8): n=256, k=37, t=31
        (3, 8),   # RM(3,8): n=256, k=93, t=15
        (1, 10),  # RM(1,10): n=1024, k=11, t=255
    ]
    
    for r, m in security_params:
        try:
            mc = McElieceReedMuller(r, m)
            
            # Calculate key sizes
            public_key_size = mc.n * mc.k  # Size of G' matrix
            private_key_size = mc.k * mc.k + mc.n * mc.n + mc.n * mc.k  # S + P + G matrices
            
            print(f"\nRM({r},{m}):")
            print(f"  Code parameters: n={mc.n}, k={mc.k}, t={mc.t}")
            print(f"  Public key size: {public_key_size} bits ({public_key_size//8} bytes)")
            print(f"  Private key size: {private_key_size} bits ({private_key_size//8} bytes)")
            print(f"  Error correction: {mc.t} errors")
            print(f"  Security level: ~{mc.t} bits (based on error correction)")
            
        except Exception as e:
            print(f"RM({r},{m}) failed: {str(e)}")

def main():
    """Main test function."""
    print("McEliece with Reed-Muller Codes - Test Suite")
    print("="*60)
    
    # Run all tests
    test_reed_muller_code()
    test_mceliece_encryption()
    test_different_parameters()
    test_security_analysis()
    
    print("\n" + "="*60)
    print("Test Suite Complete")
    print("="*60)

if __name__ == "__main__":
    main()
