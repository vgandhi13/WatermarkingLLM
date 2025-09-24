#!/usr/bin/env python3
"""
Simple test script for McEliece with Reed-Muller codes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mceliece_reed_muller import McElieceReedMuller, ReedMullerCode, print_binary

def test_simple_encryption():
    """Test simple encryption/decryption with appropriate message sizes."""
    print("Simple McEliece with Reed-Muller Codes Test")
    print("="*50)
    
    # Test with RM(1,7) which has k=8 bits (can fit 1 byte)
    try:
        mc = McElieceReedMuller(r=1, m=7)
        print(f"Code parameters: n={mc.n}, k={mc.k}, t={mc.t}")
        
        # Create a message that fits in k bits (8 bits = 1 byte)
        test_message = b"H"  # 1 byte = 8 bits
        
        print(f"\nTesting message: {test_message}")
        print_binary(test_message, "Original Message")
        
        # Encrypt
        ciphertext, num_errors = mc.encrypt(test_message)
        print_binary(ciphertext, f"Encrypted Message ({num_errors} errors)")
        
        # Decrypt
        decrypted = mc.decrypt(ciphertext)
        print_binary(decrypted, "Decrypted Message")
        
        success = test_message == decrypted
        print(f"Encryption/Decryption Success: {success}")
        
        if success:
            print("✅ McEliece with Reed-Muller codes is working correctly!")
        else:
            print("❌ Decryption failed")
            print(f"Expected: {test_message}")
            print(f"Got: {decrypted}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def test_reed_muller_only():
    """Test just the Reed-Muller code functionality."""
    print("\nReed-Muller Code Test")
    print("="*30)
    
    # Test RM(1,7) - k=8 bits
    rm = ReedMullerCode(1, 7)
    print(f"RM(1,7) parameters: n={rm.n}, k={rm.k}, t={rm.t}")
    
    # Test with a 1-byte message
    test_message = b"H"
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
        corrupted = rm.introduce_errors(encoded, 5)  # Introduce 5 errors (within error correction capability)
        print_binary(corrupted, "Corrupted Message (5 errors)")
        
        corrected = rm.decode(corrupted)
        print_binary(corrected, "Corrected Message")
        
        correction_success = test_message == corrected
        print(f"Error Correction Success: {correction_success}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def test_parameter_comparison():
    """Compare different Reed-Muller parameters."""
    print("\nParameter Comparison")
    print("="*30)
    
    parameters = [
        (1, 6),  # RM(1,6): n=64, k=7, t=15
        (2, 6),  # RM(2,6): n=64, k=22, t=7
        (1, 7),  # RM(1,7): n=128, k=8, t=31
        (2, 7),  # RM(2,7): n=128, k=29, t=15
    ]
    
    for r, m in parameters:
        try:
            rm = ReedMullerCode(r, m)
            print(f"RM({r},{m}): n={rm.n}, k={rm.k}, t={rm.t}")
            print(f"  Message capacity: {rm.k} bits ({rm.k//8} bytes)")
            print(f"  Error correction: {rm.t} errors")
            print(f"  Code rate: {rm.k/rm.n:.3f}")
        except Exception as e:
            print(f"RM({r},{m}): Error - {str(e)}")

if __name__ == "__main__":
    test_reed_muller_only()
    test_parameter_comparison()
    test_simple_encryption()
