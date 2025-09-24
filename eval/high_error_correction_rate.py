#!/usr/bin/env python3
"""
Find Reed-Muller codes with high error correction rates.

Error correction rate = (t / n) * 100%
Where t = number of errors that can be corrected
      n = codeword length

This shows what percentage of the codeword can be corrupted
while still successfully recovering the original message.
"""

import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ecc', 'reed-muller-python'))

def find_high_error_correction_rate_codes():
    """Find Reed-Muller codes with high error correction rates."""
    print("Reed-Muller Codes with High Error Correction Rates")
    print("=" * 80)
    print("Error Correction Rate = (t / n) * 100%")
    print("Where t = errors that can be corrected, n = codeword length")
    print("=" * 80)
    
    try:
        from reedmuller.reedmuller import ReedMuller
        
        # Test a wide range of Reed-Muller parameters
        test_configs = []
        
        # Test different r and m combinations
        for r in range(1, 6):  # Order from 1 to 5
            for m in range(4, 12):  # Variables from 4 to 11
                try:
                    rm = ReedMuller(r, m)
                    k = rm.message_length()
                    n = rm.block_length()
                    t = rm.strength()
                    
                    # Only consider codes that support 16-bit messages
                    if k >= 16:
                        error_correction_rate = (t / n) * 100
                        code_rate = k / n
                        
                        test_configs.append({
                            'config': f'RM({r},{m})',
                            'r': r,
                            'm': m,
                            'k': k,
                            'n': n,
                            't': t,
                            'error_correction_rate': error_correction_rate,
                            'code_rate': code_rate
                        })
                        
                except Exception:
                    # Skip invalid parameter combinations
                    continue
        
        # Sort by error correction rate (higher is better)
        test_configs.sort(key=lambda x: x['error_correction_rate'], reverse=True)
        
        print(f"{'Config':<10} {'k':<4} {'n':<6} {'t':<4} {'Error Corr Rate':<15} {'Code Rate':<10} {'Category'}")
        print("-" * 80)
        
        for config in test_configs:
            # Categorize by error correction rate
            if config['error_correction_rate'] >= 20:
                category = "🚀 Ultra-High"
            elif config['error_correction_rate'] >= 15:
                category = "⚡ Very High"
            elif config['error_correction_rate'] >= 10:
                category = "📈 High"
            elif config['error_correction_rate'] >= 5:
                category = "📊 Moderate"
            else:
                category = "📉 Low"
            
            print(f"{config['config']:<10} {config['k']:<4} {config['n']:<6} {config['t']:<4} "
                  f"{config['error_correction_rate']:<15.1f}% {config['code_rate']:<10.3f} {category}")
        
        # Show top recommendations
        print("\n" + "=" * 80)
        print("TOP RECOMMENDATIONS FOR HIGH ERROR CORRECTION RATE")
        print("=" * 80)
        
        top_configs = test_configs[:10]  # Top 10
        
        for i, config in enumerate(top_configs, 1):
            print(f"{i:2d}. {config['config']}: {config['error_correction_rate']:.1f}% error correction rate")
            print(f"    - Can correct {config['t']} errors in {config['n']}-bit codeword")
            print(f"    - Supports {config['k']}-bit messages")
            print(f"    - Code rate: {config['code_rate']:.3f} ({config['code_rate']*100:.1f}%)")
            print()
        
        return test_configs
        
    except ImportError as e:
        print(f"Error importing Reed-Muller library: {e}")
        return []

def test_error_correction_rate(rm_config, r, m, num_tests=500):
    """
    Test the actual error correction rate by corrupting different percentages
    of the codeword and seeing how many can be successfully corrected.
    """
    try:
        from reedmuller.reedmuller import ReedMuller
        
        rm = ReedMuller(r, m)
        k = rm.message_length()
        n = rm.block_length()
        t = rm.strength()
        theoretical_rate = (t / n) * 100
        
        print(f"\nTesting {rm_config} Error Correction Rate:")
        print(f"  Message length (k): {k} bits")
        print(f"  Codeword length (n): {n} bits")
        print(f"  Error correction capability (t): {t} errors")
        print(f"  Theoretical error correction rate: {theoretical_rate:.1f}%")
        
        # Create a test message (16 bits)
        test_message = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        if k > 16:
            test_message.extend([0] * (k - 16))
        elif k < 16:
            test_message = test_message[:k]
        
        # Test different corruption percentages
        corruption_percentages = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        results = {}
        
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
            for _ in range(num_tests):
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
                    
                except Exception:
                    total += 1
            
            if total > 0:
                success_rate = successful / total
                results[corruption_pct] = success_rate
                
                print(f"    {corruption_pct:2d}% corrupted ({num_corruptions:2d}/{n:2d} bits): {successful:3d}/{total:3d} successful ({success_rate:.1%})")
        
        # Find the maximum corruption percentage with >50% success
        max_successful_corruption = 0
        for corruption_pct, success_rate in results.items():
            if success_rate > 0.5:
                max_successful_corruption = corruption_pct
        
        practical_rate = max_successful_corruption
        print(f"  Practical error correction rate: {practical_rate:.1f}%")
        print(f"  Theoretical error correction rate: {theoretical_rate:.1f}%")
        
        if practical_rate >= theoretical_rate * 0.8:
            print(f"  ✅ Error correction rate validated")
        else:
            print(f"  ⚠️  Error correction rate lower than theoretical")
        
        return results
        
    except Exception as e:
        print(f"  Error testing {rm_config}: {e}")
        return {}

def demonstrate_high_error_correction():
    """Demonstrate high error correction rate codes."""
    print("\n" + "=" * 80)
    print("DEMONSTRATING HIGH ERROR CORRECTION RATE CODES")
    print("=" * 80)
    
    # Find the best codes
    all_configs = find_high_error_correction_rate_codes()
    
    if not all_configs:
        print("No suitable configurations found.")
        return
    
    # Test the top 3 configurations
    top_configs = all_configs[:3]
    
    print(f"\nTesting Top {len(top_configs)} High Error Correction Rate Codes:")
    print("-" * 60)
    
    for config in top_configs:
        test_error_correction_rate(
            config['config'], 
            config['r'], 
            config['m'], 
            num_tests=200
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: High Error Correction Rate Codes")
    print("=" * 80)
    
    print(f"{'Config':<10} {'Error Corr Rate':<15} {'Code Rate':<10} {'t':<4} {'n':<6} {'Description'}")
    print("-" * 80)
    
    for config in top_configs:
        description = f"Can correct {config['t']} errors in {config['n']}-bit codeword"
        print(f"{config['config']:<10} {config['error_correction_rate']:<15.1f}% "
              f"{config['code_rate']:<10.3f} {config['t']:<4} {config['n']:<6} {description}")

def specific_high_correction_example():
    """Show a specific example of high error correction rate."""
    print("\n" + "=" * 80)
    print("SPECIFIC EXAMPLE: High Error Correction Rate")
    print("=" * 80)
    
    try:
        from reedmuller.reedmuller import ReedMuller
        
        # Use RM(1,8) which should have high error correction rate
        rm = ReedMuller(1, 8)
        k = rm.message_length()
        n = rm.block_length()
        t = rm.strength()
        error_correction_rate = (t / n) * 100
        code_rate = k / n
        
        print(f"RM(1,8) High Error Correction Rate Example:")
        print(f"  Message length (k): {k} bits")
        print(f"  Codeword length (n): {n} bits")
        print(f"  Error correction capability (t): {t} errors")
        print(f"  Error correction rate: {error_correction_rate:.1f}%")
        print(f"  Code rate: {code_rate:.3f} ({code_rate*100:.1f}%)")
        print()
        print(f"This means:")
        print(f"  - Can correct {t} errors out of {n} bits")
        print(f"  - {error_correction_rate:.1f}% of the codeword can be corrupted")
        print(f"  - Still successfully recover the original message")
        
        # Create a test message
        test_message = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        if k > 16:
            test_message.extend([0] * (k - 16))
        elif k < 16:
            test_message = test_message[:k]
        
        print(f"\nOriginal 16-bit message: {test_message[:16]}")
        
        # Encode
        encoded = rm.encode(test_message)
        print(f"Encoded codeword ({n} bits): {encoded}")
        
        # Corrupt at the error correction rate
        corruption_pct = int(error_correction_rate)
        num_corruptions = int(n * corruption_pct / 100)
        
        print(f"\nCorrupting {corruption_pct}% of codeword ({num_corruptions} out of {n} bits):")
        
        # Corrupt the codeword
        corruption_positions = random.sample(range(n), num_corruptions)
        corrupted = encoded.copy()
        
        for pos in corruption_positions:
            corrupted[pos] = 1 - corrupted[pos]
        
        print(f"Corrupted positions: {sorted(corruption_positions)}")
        print(f"Corrupted codeword: {corrupted}")
        
        # Try to decode
        decoded = rm.decode(corrupted)
        print(f"Decoded message: {decoded}")
        
        # Check if 16-bit message is preserved
        original_16 = test_message[:16]
        decoded_16 = decoded[:16]
        
        print(f"\nOriginal 16 bits:  {original_16}")
        print(f"Decoded 16 bits:  {decoded_16}")
        print(f"Recovery successful: {'✅ YES' if original_16 == decoded_16 else '❌ NO'}")
        
        if original_16 == decoded_16:
            print(f"\n🎯 PROOF: RM(1,8) can correct {error_correction_rate:.1f}% corruption!")
            print(f"   This is a HIGH error correction rate!")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    demonstrate_high_error_correction()
    specific_high_correction_example()
