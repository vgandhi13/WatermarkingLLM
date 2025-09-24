#!/usr/bin/env python3
"""
Detailed analysis of Reed-Muller codes vs McEliece ciphertexts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mceliece_reed_muller import McElieceReedMuller, ReedMullerCode, print_binary

def analyze_code_vs_ciphertext():
    """Analyze the relationship between Reed-Muller codes and McEliece ciphertexts."""
    print("Reed-Muller Code vs McEliece Ciphertext Analysis")
    print("="*60)
    
    # Use RM(1,7) for analysis - this gives 128-bit codewords
    r, m = 1, 7
    rm = ReedMullerCode(r, m)
    mc = McElieceReedMuller(r, m)
    
    print(f"Reed-Muller Code RM({r},{m}):")
    print(f"  Code length (n): {rm.n} bits")
    print(f"  Message length (k): {rm.k} bits")
    print(f"  Error correction (t): {rm.t} errors")
    print(f"  Code rate: {rm.k/rm.n:.3f}")
    print()
    
    # Test with a simple message
    message = b"H"  # 1 byte = 8 bits (fits in RM(1,7) which has k=8)
    print(f"Original message: {message}")
    print(f"Message length: {len(message)} bytes = {len(message)*8} bits")
    print()
    
    # Step 1: Reed-Muller encoding only
    print("Step 1: Reed-Muller Encoding Only")
    print("-" * 40)
    rm_encoded = rm.encode(message)
    print_binary(rm_encoded, "Reed-Muller Encoded")
    print(f"Reed-Muller output: {rm.n} bits ({rm.n//8} bytes)")
    print()
    
    # Step 2: McEliece encryption
    print("Step 2: McEliece Encryption")
    print("-" * 40)
    mc_ciphertext, num_errors = mc.encrypt(message)
    print_binary(mc_ciphertext, "McEliece Ciphertext")
    print(f"McEliece output: {len(mc_ciphertext)*8} bits ({len(mc_ciphertext)} bytes)")
    print(f"Errors added: {num_errors}")
    print()
    
    # Step 3: Compare sizes
    print("Step 3: Size Comparison")
    print("-" * 40)
    print(f"Original message: {len(message)*8} bits")
    print(f"Reed-Muller encoded: {rm.n} bits")
    print(f"McEliece ciphertext: {len(mc_ciphertext)*8} bits")
    print()
    print(f"Expansion factor (Reed-Muller): {rm.n/(len(message)*8):.2f}x")
    print(f"Expansion factor (McEliece): {len(mc_ciphertext)*8/(len(message)*8):.2f}x")
    print()
    
    # Step 4: Decryption
    print("Step 4: McEliece Decryption")
    print("-" * 40)
    decrypted = mc.decrypt(mc_ciphertext)
    print_binary(decrypted, "Decrypted Message")
    success = message == decrypted
    print(f"Decryption success: {success}")
    print()

def analyze_different_parameters():
    """Analyze different Reed-Muller parameters."""
    print("Parameter Analysis")
    print("="*60)
    
    parameters = [
        (1, 6),  # RM(1,6): n=64, k=7, t=15
        (2, 6),  # RM(2,6): n=64, k=22, t=7
        (1, 7),  # RM(1,7): n=128, k=8, t=31
        (2, 7),  # RM(2,7): n=128, k=29, t=15
        (1, 8),  # RM(1,8): n=256, k=9, t=63
    ]
    
    print(f"{'Code':<10} {'n':<6} {'k':<6} {'t':<6} {'Rate':<8} {'1-byte':<10} {'2-byte':<10}")
    print("-" * 60)
    
    for r, m in parameters:
        try:
            rm = ReedMullerCode(r, m)
            rate = rm.k / rm.n
            one_byte_expansion = rm.n / 8
            two_byte_expansion = rm.n / 16
            
            print(f"RM({r},{m}): {rm.n:<6} {rm.k:<6} {rm.t:<6} {rate:<8.3f} {one_byte_expansion:<10.1f}x {two_byte_expansion:<10.1f}x")
        except Exception as e:
            print(f"RM({r},{m}): Error - {str(e)}")

def demonstrate_security():
    """Demonstrate the security aspect of McEliece."""
    print("\nSecurity Demonstration")
    print("="*60)
    
    mc = McElieceReedMuller(1, 7)
    message = b"H"
    
    print(f"Original message: {message}")
    print()
    
    # Encrypt the same message multiple times
    print("Multiple encryptions of the same message:")
    for i in range(3):
        ciphertext, num_errors = mc.encrypt(message)
        print(f"Encryption {i+1}: {ciphertext.hex()} (errors: {num_errors})")
    
    print()
    print("Note: Each encryption produces a different ciphertext due to random errors!")
    print("This is the security mechanism of McEliece.")

def compare_with_other_codes():
    """Compare Reed-Muller with other code parameters."""
    print("\nComparison with Other Code Parameters")
    print("="*60)
    
    # Theoretical comparison
    print("Code Type Comparison:")
    print(f"{'Code Type':<15} {'n':<6} {'k':<6} {'t':<6} {'Rate':<8} {'Security':<10}")
    print("-" * 60)
    
    # Reed-Muller examples
    print(f"{'RM(1,6)':<15} {64:<6} {7:<6} {15:<6} {7/64:<8.3f} {'~15 bits':<10}")
    print(f"{'RM(2,6)':<15} {64:<6} {22:<6} {7:<6} {22/64:<8.3f} {'~7 bits':<10}")
    print(f"{'RM(1,8)':<15} {256:<6} {9:<6} {63:<6} {9/256:<8.3f} {'~63 bits':<10}")
    
    print()
    print("Note: Reed-Muller codes have lower code rates but simpler structure.")
    print("For higher security, larger parameters are needed.")

if __name__ == "__main__":
    analyze_code_vs_ciphertext()
    analyze_different_parameters()
    demonstrate_security()
    compare_with_other_codes()
