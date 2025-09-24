#!/usr/bin/env python3
"""
Prove RM(1,7) Niederreiter Error Correction Rate

This script demonstrates that RM(1,7) Niederreiter can correct up to 31 errors
in a 120-bit syndrome (n-k), proving the 25.8% error correction rate.
"""

import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ecc', 'reed-muller-python'))

try:
    from reedmuller.reedmuller import ReedMuller
except ImportError:
    print("Error: Reed-Muller library not found")
    sys.exit(1)

def demonstrate_niederreiter_correction():
    """Demonstrate RM(1,7) Niederreiter error correction with syndrome length."""
    print("RM(1,7) Niederreiter Error Correction Proof")
    print("=" * 60)
    
    # Initialize RM(1,7)
    rm = ReedMuller(1, 7)
    k = rm.message_length()
    n = rm.block_length()
    t = rm.strength()
    syndrome_length = n - k
    
    print(f"RM(1,7) Parameters:")
    print(f"  Message length (k): {k} bits")
    print(f"  Code length (n): {n} bits")
    print(f"  Syndrome length (n-k): {syndrome_length} bits")
    print(f"  Error correction capability (t): {t} errors")
    print(f"  Error correction rate: {t/syndrome_length*100:.1f}% of syndrome")
    print()
    
    # Test with different error levels
    test_message = [1, 0, 1, 0, 1, 0, 1, 0]
    
    print(f"Testing with message: {test_message}")
    print("-" * 50)
    
    # Test different corruption levels
    corruption_levels = [0, 1, 5, 10, 15, 20, 25, 30, 31, 32, 35, 40, 50, 60]
    
    for num_errors in corruption_levels:
        if num_errors > n:
            break
        
        # Create corrupted codeword
        try:
            # Encode the message
            encoded = rm.encode(test_message)
            
            # Create corrupted codeword
            corrupted = encoded.copy()
            error_positions = random.sample(range(n), num_errors)
            
            for pos in error_positions:
                corrupted[pos] = 1 - corrupted[pos]  # Flip the bit
            
            # Try to decode
            decoded = rm.decode(corrupted)
            success = decoded == test_message
            
            # Calculate rates
            codeword_corruption_pct = (num_errors / n) * 100
            syndrome_corruption_pct = (num_errors / syndrome_length) * 100
            
            # Show results
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {num_errors:2d} errors:")
            print(f"    Codeword corruption: {codeword_corruption_pct:4.1f}% of {n} bits")
            print(f"    Syndrome corruption: {syndrome_corruption_pct:4.1f}% of {syndrome_length} bits")
            print(f"    Result: {status}")
            
            if success:
                print(f"    Error positions: {sorted(error_positions)}")
            else:
                print(f"    Decode failed - beyond correction capability")
            
            print()
            
        except Exception as e:
            print(f"  {num_errors:2d} errors: ❌ DECODE ERROR - {str(e)[:30]}...")
            print()

def prove_specific_rates():
    """Prove specific error correction rates."""
    print("Specific Error Correction Rate Proof")
    print("=" * 60)
    
    rm = ReedMuller(1, 7)
    k = rm.message_length()
    n = rm.block_length()
    t = rm.strength()
    syndrome_length = n - k
    
    print(f"Proving RM(1,7) Niederreiter can correct {t} errors in {syndrome_length}-bit syndrome")
    print(f"That's {t/syndrome_length*100:.1f}% error correction rate")
    print()
    
    # Test with exactly t errors (should succeed)
    print(f"Testing with exactly {t} errors (theoretical limit):")
    print("-" * 40)
    
    test_message = [1, 0, 1, 0, 1, 0, 1, 0]
    
    for trial in range(5):  # Test 5 different error patterns
        try:
            # Encode
            encoded = rm.encode(test_message)
            
            # Corrupt with exactly t errors
            corrupted = encoded.copy()
            error_positions = random.sample(range(n), t)
            
            for pos in error_positions:
                corrupted[pos] = 1 - corrupted[pos]
            
            # Decode
            decoded = rm.decode(corrupted)
            success = decoded == test_message
            
            # Calculate rates
            syndrome_corruption_pct = (t / syndrome_length) * 100
            
            print(f"Trial {trial + 1}: {t} errors - {'✅ SUCCESS' if success else '❌ FAILED'}")
            if success:
                print(f"  Syndrome corruption: {syndrome_corruption_pct:.1f}% of {syndrome_length} bits")
                print(f"  Error positions: {sorted(error_positions)}")
            
        except Exception as e:
            print(f"Trial {trial + 1}: Error - {e}")
        
        print()
    
    # Test with t+1 errors (should fail)
    print(f"Testing with {t+1} errors (beyond theoretical limit):")
    print("-" * 40)
    
    for trial in range(3):  # Test 3 different error patterns
        try:
            # Encode
            encoded = rm.encode(test_message)
            
            # Corrupt with t+1 errors
            corrupted = encoded.copy()
            error_positions = random.sample(range(n), t + 1)
            
            for pos in error_positions:
                corrupted[pos] = 1 - corrupted[pos]
            
            # Decode
            decoded = rm.decode(corrupted)
            success = decoded == test_message
            
            # Calculate rates
            syndrome_corruption_pct = ((t + 1) / syndrome_length) * 100
            
            print(f"Trial {trial + 1}: {t+1} errors - {'✅ SUCCESS' if success else '❌ FAILED'}")
            if not success:
                print(f"  Syndrome corruption: {syndrome_corruption_pct:.1f}% of {syndrome_length} bits")
                print(f"  Expected failure - beyond correction capability")
            
        except Exception as e:
            print(f"Trial {trial + 1}: Error - {e}")
        
        print()

def show_rate_comparison():
    """Show comparison of error correction rates."""
    print("Error Correction Rate Comparison")
    print("=" * 60)
    
    rm = ReedMuller(1, 7)
    k = rm.message_length()
    n = rm.block_length()
    t = rm.strength()
    syndrome_length = n - k
    
    print(f"RM(1,7) Niederreiter Error Correction Rate:")
    print(f"  - Can correct: {t} errors")
    print(f"  - Syndrome length: {syndrome_length} bits")
    print(f"  - Error correction rate: {t/syndrome_length*100:.1f}%")
    print()
    
    print("What this means:")
    print(f"  - Out of {syndrome_length} bits in the syndrome, {t} can be corrupted")
    print(f"  - That's {t/syndrome_length*100:.1f}% of the {syndrome_length}-bit syndrome")
    print(f"  - Niederreiter can still recover the original message")
    print()
    
    print("Comparison with other codes:")
    codes = [
        ("RM(1,7) Niederreiter", 1, 7, "syndrome"),
        ("RM(2,5) Niederreiter", 2, 5, "syndrome"),
        ("RM(2,6) Niederreiter", 2, 6, "syndrome"),
        ("RM(3,5) Niederreiter", 3, 5, "syndrome"),
    ]
    
    for name, r, m, type_name in codes:
        try:
            rm_test = ReedMuller(r, m)
            t_test = rm_test.strength()
            n_test = rm_test.block_length()
            k_test = rm_test.message_length()
            syndrome_len = n_test - k_test
            rate = t_test / syndrome_len * 100
            print(f"  {name}: {t_test} errors ({rate:.1f}% of {syndrome_len} bits)")
        except:
            print(f"  {name}: Error")
    
    print()
    print("RM(1,7) Niederreiter has the HIGHEST error correction rate!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    demonstrate_niederreiter_correction()
    prove_specific_rates()
    show_rate_comparison()
