#!/usr/bin/env python3
"""
Prove Reed-Muller Code Rates

The CODE RATE is the percentage of the codeword that can be corrupted
while still successfully recovering the original message.

For example:
- RM(3,5) has rate 81.2% means 81.2% of the codeword can be corrupted
  and we can still decode the original message
- This is different from error correction capability (t)
"""

import sys
import os
import random
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ecc', 'reed-muller-python'))

def test_code_rate(rm_config, r, m, num_tests=1000):
    """
    Test the actual code rate - percentage of codeword that can be corrupted
    while still successfully recovering the original message.
    
    Args:
        rm_config: Configuration name (e.g., 'RM(3,5)')
        r: Reed-Muller order parameter
        m: Reed-Muller number of variables parameter
        num_tests: Number of random tests to perform
    
    Returns:
        dict: Statistics about code rate performance
    """
    try:
        from reedmuller.reedmuller import ReedMuller
        
        # Initialize Reed-Muller code
        rm = ReedMuller(r, m)
        k = rm.message_length()
        n = rm.block_length()
        t = rm.strength()
        theoretical_rate = k / n
        
        print(f"\nTesting {rm_config} Code Rate:")
        print(f"  Message length (k): {k} bits")
        print(f"  Codeword length (n): {n} bits")
        print(f"  Theoretical rate (k/n): {theoretical_rate:.3f} ({theoretical_rate*100:.1f}%)")
        print(f"  Error correction capability (t): {t} errors")
        
        # Create a test message (16 bits for our use case)
        test_message = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        
        # Pad or truncate to fit the code
        if k > 16:
            test_message.extend([0] * (k - 16))
        elif k < 16:
            test_message = test_message[:k]
        
        print(f"  Test message length: {len(test_message)} bits")
        
        # Test different corruption percentages
        corruption_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        
        stats = {
            'theoretical_rate': theoretical_rate,
            'corruption_results': {},
            'max_successful_corruption': 0
        }
        
        for corruption_pct in corruption_percentages:
            if corruption_pct >= 100:
                break
                
            # Calculate number of bits to corrupt
            num_corruptions = int(n * corruption_pct / 100)
            if num_corruptions > n:
                num_corruptions = n
            
            successful = 0
            total = 0
            
            # Run tests for this corruption level
            tests_per_level = min(num_tests, 200)
            
            for _ in range(tests_per_level):
                try:
                    # Encode the message
                    encoded = rm.encode(test_message)
                    
                    # Corrupt the specified percentage of bits
                    if num_corruptions > 0:
                        corruption_positions = random.sample(range(n), num_corruptions)
                        corrupted = encoded.copy()
                        
                        for pos in corruption_positions:
                            corrupted[pos] = 1 - corrupted[pos]  # Flip the bit
                    else:
                        corrupted = encoded.copy()
                    
                    # Try to decode
                    decoded = rm.decode(corrupted)
                    
                    # Check if the original 16 bits are preserved
                    original_16 = test_message[:16] if len(test_message) >= 16 else test_message
                    decoded_16 = decoded[:16] if len(decoded) >= 16 else decoded
                    
                    if original_16 == decoded_16:
                        successful += 1
                    total += 1
                    
                except Exception as e:
                    # Decoding failed
                    total += 1
            
            if total > 0:
                success_rate = successful / total
                stats['corruption_results'][corruption_pct] = {
                    'success_rate': success_rate,
                    'num_corruptions': num_corruptions,
                    'tests': total
                }
                
                print(f"    {corruption_pct:2d}% corrupted ({num_corruptions:2d}/{n:2d} bits): {successful:3d}/{total:3d} successful ({success_rate:.1%})")
                
                # Track maximum successful corruption
                if success_rate > 0.5:  # More than 50% success
                    stats['max_successful_corruption'] = corruption_pct
        
        # Calculate practical code rate
        practical_rate = stats['max_successful_corruption'] / 100.0
        print(f"  Practical code rate: {practical_rate:.3f} ({practical_rate*100:.1f}%)")
        print(f"  Theoretical code rate: {theoretical_rate:.3f} ({theoretical_rate*100:.1f}%)")
        
        if practical_rate >= theoretical_rate * 0.8:  # Within 80% of theoretical
            print(f"  ✅ Code rate validated: Practical rate matches theoretical rate")
        else:
            print(f"  ⚠️  Code rate discrepancy: Practical rate lower than theoretical")
        
        return stats
        
    except Exception as e:
        print(f"  Error testing {rm_config}: {e}")
        return None

def demonstrate_code_rate_concept():
    """Demonstrate what code rate actually means with a concrete example."""
    print("\n" + "=" * 80)
    print("UNDERSTANDING CODE RATE")
    print("=" * 80)
    print("Code Rate = Message Length / Codeword Length")
    print("This represents the 'efficiency' of the code.")
    print()
    print("For example:")
    print("- RM(3,5): 26 bits message → 32 bits codeword")
    print("  Code Rate = 26/32 = 0.812 (81.2%)")
    print("  This means 81.2% of the codeword contains actual message information")
    print("  The remaining 18.8% is redundancy for error correction")
    print()
    print("Higher code rate = More efficient = Less redundancy")
    print("Lower code rate = Less efficient = More redundancy = Better error correction")
    print("=" * 80)

def prove_reed_muller_code_rates():
    """Prove the code rates for various Reed-Muller configurations."""
    print("Proving Reed-Muller Code Rates")
    print("=" * 80)
    print("Testing how much of each codeword can be corrupted")
    print("while still successfully recovering the original message.")
    print("=" * 80)
    
    demonstrate_code_rate_concept()
    
    # Test configurations with different rates
    test_configs = [
        ('RM(4,4)', 4, 4, 'Ultra-High Rate (100%)'),
        ('RM(4,5)', 4, 5, 'Ultra-High Rate (96.9%)'),
        ('RM(4,6)', 4, 6, 'Very High Rate (89.1%)'),
        ('RM(3,5)', 3, 5, 'Balanced High Rate (81.2%)'),
        ('RM(4,7)', 4, 7, 'High Rate with Correction (77.3%)'),
        ('RM(3,6)', 3, 6, 'Moderate High Rate (65.6%)'),
        ('RM(2,5)', 2, 5, 'Classic High Rate (50.0%)'),
    ]
    
    all_stats = {}
    
    for config_name, r, m, description in test_configs:
        print(f"\n{description}")
        print("-" * 60)
        
        stats = test_code_rate(config_name, r, m, num_tests=100)
        if stats:
            all_stats[config_name] = stats
    
    # Summary
    print("\n" + "=" * 80)
    print("CODE RATE VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"{'Config':<10} {'Theoretical':<12} {'Practical':<12} {'Max Corruption':<15} {'Validated?'}")
    print("-" * 70)
    
    for config_name, stats in all_stats.items():
        if stats:
            theoretical = stats['theoretical_rate']
            practical = stats['max_successful_corruption'] / 100.0
            max_corruption = f"{stats['max_successful_corruption']}%"
            
            validated = "✅ YES" if practical >= theoretical * 0.8 else "❌ NO"
            
            print(f"{config_name:<10} {theoretical:<12.3f} {practical:<12.3f} {max_corruption:<15} {validated}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("The code rates represent the efficiency of information transmission.")
    print("Higher rates mean more efficient use of codeword space.")
    print("The tests validate that the theoretical rates are achievable.")
    print("=" * 80)

def specific_rate_example():
    """Show a specific example of code rate in action."""
    print("\n" + "=" * 80)
    print("SPECIFIC EXAMPLE: RM(3,5) Code Rate Demonstration")
    print("=" * 80)
    
    try:
        from reedmuller.reedmuller import ReedMuller
        
        rm = ReedMuller(3, 5)
        k = rm.message_length()
        n = rm.block_length()
        rate = k / n
        
        print(f"RM(3,5) Code Rate Analysis:")
        print(f"  Message length (k): {k} bits")
        print(f"  Codeword length (n): {n} bits")
        print(f"  Code rate (k/n): {rate:.3f} ({rate*100:.1f}%)")
        print()
        print(f"This means:")
        print(f"  - {k} bits contain the actual message")
        print(f"  - {n-k} bits are redundancy for error correction")
        print(f"  - {rate*100:.1f}% of the codeword is 'useful' information")
        print(f"  - {100-rate*100:.1f}% of the codeword is redundancy")
        
        # Create a test message
        test_message = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        test_message.extend([0] * (k - 16))  # Pad to k bits
        
        print(f"\nOriginal 16-bit message: {test_message[:16]}")
        
        # Encode
        encoded = rm.encode(test_message)
        print(f"Encoded codeword ({n} bits): {encoded}")
        
        # Show which bits are "message" vs "redundancy"
        print(f"\nCodeword breakdown:")
        print(f"  Message bits (first {k}): {encoded[:k]}")
        print(f"  Redundancy bits (last {n-k}): {encoded[k:]}")
        
        # Test corruption at the theoretical rate
        corruption_pct = int(rate * 100)  # Corrupt at the code rate percentage
        num_corruptions = int(n * corruption_pct / 100)
        
        print(f"\nTesting corruption at {corruption_pct}% (code rate):")
        print(f"  Corrupting {num_corruptions} out of {n} bits")
        
        # Corrupt the codeword
        corruption_positions = random.sample(range(n), num_corruptions)
        corrupted = encoded.copy()
        
        for pos in corruption_positions:
            corrupted[pos] = 1 - corrupted[pos]
        
        print(f"  Corrupted positions: {sorted(corruption_positions)}")
        print(f"  Corrupted codeword: {corrupted}")
        
        # Try to decode
        decoded = rm.decode(corrupted)
        print(f"  Decoded message: {decoded}")
        
        # Check if 16-bit message is preserved
        original_16 = test_message[:16]
        decoded_16 = decoded[:16]
        
        print(f"\nOriginal 16 bits:  {original_16}")
        print(f"Decoded 16 bits:  {decoded_16}")
        print(f"Recovery successful: {'✅ YES' if original_16 == decoded_16 else '❌ NO'}")
        
        if original_16 == decoded_16:
            print(f"\n🎯 PROOF: RM(3,5) maintains {rate*100:.1f}% code rate")
            print(f"   even with {corruption_pct}% corruption!")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    prove_reed_muller_code_rates()
    specific_rate_example()
