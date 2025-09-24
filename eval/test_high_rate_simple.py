#!/usr/bin/env python3
"""
Simple test for high-rate Reed-Muller configurations without external dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ecc', 'reed-muller-python'))

def test_high_rate_reed_muller():
    """Test high-rate Reed-Muller configurations."""
    print("High-Rate Reed-Muller Configurations for 16-bit Messages")
    print("=" * 80)
    
    try:
        from reedmuller.reedmuller import ReedMuller
        
        # High-rate configurations sorted by efficiency
        high_rate_configs = [
            ('RM(4,4)', 4, 4, 'Ultra-High Rate'),
            ('RM(4,5)', 4, 5, 'Ultra-High Rate'),
            ('RM(4,6)', 4, 6, 'Very High Rate'),
            ('RM(3,5)', 3, 5, 'Balanced High Rate'),
            ('RM(4,7)', 4, 7, 'High Rate with Correction'),
            ('RM(3,6)', 3, 6, 'Moderate High Rate'),
            ('RM(2,5)', 2, 5, 'Classic High Rate'),
            ('RM(3,7)', 3, 7, 'High Rate'),
            ('RM(2,6)', 2, 6, 'Standard High Rate'),
        ]
        
        print(f"{'Config':<10} {'k':<4} {'n':<6} {'t':<4} {'Rate':<8} {'Category':<20} {'16-bit Support'}")
        print("-" * 80)
        
        working_configs = []
        
        for config_name, r, m, category in high_rate_configs:
            try:
                rm = ReedMuller(r, m)
                k = rm.message_length()
                n = rm.block_length()
                t = rm.strength()
                rate = k / n
                
                # Check 16-bit support
                if k >= 16:
                    support = "✅ Perfect" if k == 16 else f"✅ Up to {k} bits"
                    working_configs.append((config_name, r, m, k, n, t, rate, category))
                else:
                    support = "❌ Too small"
                
                print(f"{config_name:<10} {k:<4} {n:<6} {t:<4} {rate:<8.3f} {category:<20} {support}")
                
            except Exception as e:
                print(f"{config_name:<10} Error: {str(e)[:30]}...")
        
        print("\n" + "=" * 80)
        print("RECOMMENDED HIGH-RATE OPTIONS FOR 16-BIT MESSAGES")
        print("=" * 80)
        
        if working_configs:
            # Sort by rate (efficiency) - higher is better
            working_configs.sort(key=lambda x: x[6], reverse=True)
            
            print(f"{'Rank':<4} {'Config':<10} {'Rate':<8} {'n':<6} {'t':<4} {'Description'}")
            print("-" * 70)
            
            for i, (config_name, r, m, k, n, t, rate, category) in enumerate(working_configs[:8], 1):
                if rate >= 0.8:
                    description = "🚀 Ultra-high efficiency"
                elif rate >= 0.6:
                    description = "⚡ Very high efficiency"
                elif rate >= 0.3:
                    description = "📈 High efficiency"
                else:
                    description = "📊 Good efficiency"
                
                print(f"{i:<4} {config_name:<10} {rate:<8.3f} {n:<6} {t:<4} {description}")
            
            print(f"\n🎯 TOP RECOMMENDATION: {working_configs[0][0]}")
            print(f"   Rate: {working_configs[0][6]:.3f} ({working_configs[0][6]*100:.1f}%)")
            print(f"   Code length: {working_configs[0][4]} bits")
            print(f"   Error correction: {working_configs[0][5]} bits")
            print(f"   Message capacity: {working_configs[0][3]} bits")
        
        print("\n" + "=" * 80)
        print("TESTING ENCODING/DECODING")
        print("=" * 80)
        
        # Test the top 3 configurations
        for i, (config_name, r, m, k, n, t, rate, category) in enumerate(working_configs[:3], 1):
            print(f"\n{i}. Testing {config_name} (Rate: {rate:.3f}):")
            try:
                rm = ReedMuller(r, m)
                
                # Create a 16-bit test message
                test_message = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
                
                # Pad if needed
                if k > 16:
                    test_message.extend([0] * (k - 16))
                    print(f"   Padded message to {k} bits for encoding")
                elif k < 16:
                    test_message = test_message[:k]
                    print(f"   Truncated message to {k} bits for encoding")
                
                # Test encoding/decoding
                encoded = rm.encode(test_message)
                decoded = rm.decode(encoded)
                
                # Check if original 16 bits are preserved
                original_16 = test_message[:16] if len(test_message) >= 16 else test_message
                decoded_16 = decoded[:16] if len(decoded) >= 16 else decoded
                
                success = original_16 == decoded_16
                print(f"   Encoding/Decoding: {'✅ Success' if success else '❌ Failed'}")
                print(f"   Original 16 bits: {original_16}")
                print(f"   Decoded 16 bits:  {decoded_16}")
                print(f"   Code length: {len(encoded)} bits")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
    except ImportError as e:
        print(f"Error importing Reed-Muller library: {e}")
        print("Make sure the reed-muller-python library is available")

if __name__ == "__main__":
    test_high_rate_reed_muller()
