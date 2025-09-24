#!/usr/bin/env python3
"""
Prove RM(1,7) Error Correction Capability

This script demonstrates that RM(1,7) Niederreiter can correct up to 31 errors
by showing actual codewords with varying levels of corruption and successful recovery.
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

def demonstrate_rm17_correction():
    """Demonstrate RM(1,7) error correction with various corruption levels."""
    print("RM(1,7) Error Correction Demonstration")
    print("=" * 60)
    
    # Initialize RM(1,7)
    rm = ReedMuller(1, 7)
    k = rm.message_length()
    n = rm.block_length()
    t = rm.strength()
    
    print(f"RM(1,7) Parameters:")
    print(f"  Message length (k): {k} bits")
    print(f"  Code length (n): {n} bits")
    print(f"  Error correction capability (t): {t} errors")
    print(f"  Error correction rate: {t/n*100:.1f}%")
    print()
    
    # Create test messages
    test_messages = [
        [1, 0, 1, 0, 1, 0, 1, 0],  # 8-bit test message
        [0, 1, 1, 0, 0, 1, 1, 0],  # Another 8-bit test message
        [1, 1, 1, 1, 0, 0, 0, 0],  # Another 8-bit test message
    ]
    
    # Test different corruption levels
    corruption_levels = [0, 1, 5, 10, 15, 20, 25, 30, 31, 32, 35, 40, 50]
    
    for msg_idx, original_message in enumerate(test_messages, 1):
        print(f"Test Message {msg_idx}: {original_message}")
        print("-" * 50)
        
        try:
            # Encode the message
            encoded = rm.encode(original_message)
            print(f"Encoded codeword ({len(encoded)} bits): {encoded}")
            print()
            
            for num_errors in corruption_levels:
                if num_errors > n:
                    break
                
                # Create corrupted codeword
                corrupted = encoded.copy()
                error_positions = random.sample(range(n), num_errors)
                
                for pos in error_positions:
                    corrupted[pos] = 1 - corrupted[pos]  # Flip the bit
                
                # Try to decode
                try:
                    decoded = rm.decode(corrupted)
                    
                    # Check if decoding was successful
                    success = decoded == original_message
                    
                    # Calculate corruption percentage
                    corruption_pct = (num_errors / n) * 100
                    
                    # Show results
                    status = "✅ SUCCESS" if success else "❌ FAILED"
                    print(f"  {num_errors:2d} errors ({corruption_pct:4.1f}%): {status}")
                    
                    if success:
                        print(f"    Original:  {original_message}")
                        print(f"    Decoded:   {decoded}")
                        print(f"    Errors at: {sorted(error_positions)}")
                    else:
                        print(f"    Original:  {original_message}")
                        print(f"    Decoded:   {decoded}")
                        print(f"    Mismatch at: {[i for i, (o, d) in enumerate(zip(original_message, decoded)) if o != d]}")
                    
                    print()
                    
                except Exception as e:
                    print(f"  {num_errors:2d} errors: ❌ DECODE ERROR - {str(e)[:30]}...")
                    print()
        
        except Exception as e:
            print(f"  Encoding error: {e}")
            print()
        
        print("=" * 60)
        print()

def demonstrate_specific_correction():
    """Demonstrate specific error correction scenarios."""
    print("Specific Error Correction Scenarios")
    print("=" * 60)
    
    rm = ReedMuller(1, 7)
    k = rm.message_length()
    n = rm.block_length()
    t = rm.strength()
    
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
            
            print(f"Trial {trial + 1}: {t} errors - {'✅ SUCCESS' if success else '❌ FAILED'}")
            if success:
                print(f"  Error positions: {sorted(error_positions)}")
                print(f"  Corruption rate: {t/n*100:.1f}%")
            
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
            
            print(f"Trial {trial + 1}: {t+1} errors - {'✅ SUCCESS' if success else '❌ FAILED'}")
            if not success:
                print(f"  Error positions: {sorted(error_positions)}")
                print(f"  Corruption rate: {(t+1)/n*100:.1f}%")
                print(f"  Expected failure - beyond correction capability")
            
        except Exception as e:
            print(f"Trial {trial + 1}: Error - {e}")
        
        print()

def show_error_correction_summary():
    """Show a summary of error correction capabilities."""
    print("Error Correction Summary")
    print("=" * 60)
    
    rm = ReedMuller(1, 7)
    k = rm.message_length()
    n = rm.block_length()
    t = rm.strength()
    
    print(f"RM(1,7) can correct up to {t} errors in a {n}-bit codeword.")
    print(f"This means {t/n*100:.1f}% of the codeword can be corrupted.")
    print()
    
    print("Practical implications:")
    print(f"  - Out of {n} bits, {t} can be wrong")
    print(f"  - That's {t} flipped bits out of {n} total bits")
    print(f"  - The decoder can still recover the original message")
    print()
    
    print("Comparison with other codes:")
    codes = [
        ("RM(1,7)", 1, 7),
        ("RM(2,5)", 2, 5),
        ("RM(2,6)", 2, 6),
        ("RM(3,5)", 3, 5),
    ]
    
    for name, r, m in codes:
        try:
            rm_test = ReedMuller(r, m)
            t_test = rm_test.strength()
            n_test = rm_test.block_length()
            print(f"  {name}: {t_test} errors ({t_test/n_test*100:.1f}% of {n_test} bits)")
        except:
            print(f"  {name}: Error")
    
    print()
    print("RM(1,7) has the HIGHEST error correction capability!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    demonstrate_rm17_correction()
    demonstrate_specific_correction()
    show_error_correction_summary()
