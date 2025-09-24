#!/usr/bin/env python3
"""
Find Reed-Muller parameters that support specific message length requirements.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'reed-muller-python'))

def find_rm_params_for_message_length(target_k):
    """
    Find Reed-Muller parameters (r,m) that support a specific message length k.
    
    Args:
        target_k: Target message length in bits
        
    Returns:
        List of tuples (r, m, k, n, t) where k >= target_k
    """
    try:
        from reedmuller.reedmuller import ReedMuller
        
        suitable_params = []
        
        # Search through reasonable ranges of r and m
        # r: order (degree of polynomials), typically 1-4
        # m: number of variables, typically 4-10
        for r in range(1, 5):
            for m in range(4, 11):
                try:
                    rm = ReedMuller(r, m)
                    k = rm.message_length()
                    n = rm.block_length()
                    t = rm.strength()
                    
                    # Check if this code supports our target message length
                    if k >= target_k:
                        suitable_params.append((r, m, k, n, t))
                        
                except Exception as e:
                    # Skip invalid parameter combinations
                    continue
        
        # Sort by efficiency (k/n ratio) and then by code length
        suitable_params.sort(key=lambda x: (x[2]/x[3], x[3]))
        
        return suitable_params
        
    except ImportError:
        print("Error: Reed-Muller library not found")
        return []

def analyze_rm_params_for_16bit():
    """
    Specifically analyze Reed-Muller parameters for 16-bit messages.
    """
    print("Reed-Muller Parameters for 16-bit Messages")
    print("=" * 60)
    
    # Find parameters that support 16-bit messages
    suitable_params = find_rm_params_for_message_length(16)
    
    if not suitable_params:
        print("No suitable Reed-Muller parameters found for 16-bit messages.")
        return
    
    print(f"Found {len(suitable_params)} suitable parameter combinations:")
    print()
    print(f"{'r':<3} | {'m':<3} | {'k':<4} | {'n':<6} | {'t':<4} | {'k/n':<6} | {'Efficiency':<12}")
    print("-" * 60)
    
    for r, m, k, n, t in suitable_params:
        efficiency = k / n
        efficiency_pct = efficiency * 100
        
        # Categorize efficiency
        if efficiency_pct >= 25:
            eff_label = "High"
        elif efficiency_pct >= 15:
            eff_label = "Medium"
        else:
            eff_label = "Low"
        
        print(f"{r:<3} | {m:<3} | {k:<4} | {n:<6} | {t:<4} | {efficiency:.3f} | {eff_label:<12}")
    
    print()
    print("Parameter Analysis:")
    print("-" * 30)
    
    # Find the most efficient option
    most_efficient = min(suitable_params, key=lambda x: x[2]/x[3])
    r, m, k, n, t = most_efficient
    print(f"Most efficient: RM({r},{m}) with k/n = {k/n:.3f}")
    print(f"  - Message length: {k} bits")
    print(f"  - Code length: {n} bits")
    print(f"  - Error correction: {t} bits")
    
    # Find the shortest code length
    shortest = min(suitable_params, key=lambda x: x[3])
    r, m, k, n, t = shortest
    print(f"Shortest code: RM({r},{m}) with n = {n} bits")
    print(f"  - Message length: {k} bits")
    print(f"  - Code length: {n} bits")
    print(f"  - Error correction: {t} bits")
    
    # Find the highest error correction capability
    highest_t = max(suitable_params, key=lambda x: x[4])
    r, m, k, n, t = highest_t
    print(f"Highest error correction: RM({r},{m}) with t = {t} bits")
    print(f"  - Message length: {k} bits")
    print(f"  - Code length: {n} bits")
    print(f"  - Error correction: {t} bits")

def test_specific_rm_params():
    """
    Test specific Reed-Muller parameters to verify they work.
    """
    print("\n" + "=" * 60)
    print("Testing Specific Reed-Muller Parameters")
    print("=" * 60)
    
    try:
        from reedmuller.reedmuller import ReedMuller
        
        # Test some promising parameters
        test_params = [
            (1, 6),  # RM(1,6) - might be close to 16 bits
            (2, 5),  # RM(2,5) - might be close to 16 bits
            (1, 7),  # RM(1,7) - current implementation
            (2, 6),  # RM(2,6) - might be suitable
        ]
        
        for r, m in test_params:
            try:
                rm = ReedMuller(r, m)
                k = rm.message_length()
                n = rm.block_length()
                t = rm.strength()
                
                print(f"RM({r},{m}):")
                print(f"  - Message length (k): {k} bits")
                print(f"  - Code length (n): {n} bits")
                print(f"  - Error correction (t): {t} bits")
                print(f"  - Efficiency (k/n): {k/n:.3f}")
                
                # Test encoding/decoding if k >= 16
                if k >= 16:
                    print(f"  - ✅ Supports 16-bit messages")
                    
                    # Test with a 16-bit message
                    test_message = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                    if k > 16:
                        # Pad with zeros if needed
                        test_message.extend([0] * (k - 16))
                    
                    try:
                        encoded = rm.encode(test_message)
                        decoded = rm.decode(encoded)
                        
                        if decoded == test_message:
                            print(f"  - ✅ Encoding/decoding test passed")
                        else:
                            print(f"  - ❌ Encoding/decoding test failed")
                    except Exception as e:
                        print(f"  - ❌ Encoding/decoding error: {e}")
                else:
                    print(f"  - ❌ Too small for 16-bit messages")
                
                print()
                
            except Exception as e:
                print(f"RM({r},{m}): Error - {e}")
                print()
                
    except ImportError:
        print("Error: Reed-Muller library not found")

if __name__ == "__main__":
    analyze_rm_params_for_16bit()
    test_specific_rm_params()


