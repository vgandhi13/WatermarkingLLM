#!/usr/bin/env python3
"""
Debug script to check Reed-Muller encoding and conversion.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reed-muller-python'))

from reedmuller.reedmuller import ReedMuller

def debug_reed_muller_encoding():
    """Debug the Reed-Muller encoding process."""
    print("Debugging Reed-Muller RM(1,7) encoding...")
    
    # Create RM(1,7) code
    rm = ReedMuller(1, 7)
    print(f"RM(1,7) parameters: n={rm.block_length()}, k={rm.message_length()}, t={rm.strength()}")
    
    # Test with the actual message "H" (01001000 in binary)
    message = [0, 1, 0, 0, 1, 0, 0, 0]  # "H" in binary
    print(f"Input message: {message} (letter 'H')")
    print(f"Input length: {len(message)} bits")
    
    # Encode
    encoded = rm.encode(message)
    print(f"Encoded length: {len(encoded)} bits")
    print(f"First 32 bits: {encoded[:32]}")
    print(f"Last 32 bits: {encoded[-32:]}")
    
    # Convert to bytes
    encoded_bytes = bytearray()
    for i in range(0, len(encoded), 8):
        byte_bits = encoded[i:i+8]
        if len(byte_bits) < 8:
            byte_bits.extend([0] * (8 - len(byte_bits)))
        byte_val = int(''.join(str(bit) for bit in byte_bits), 2)
        encoded_bytes.append(byte_val)
    
    print(f"Converted to {len(encoded_bytes)} bytes")
    print(f"Bytes: {encoded_bytes}")
    print(f"Hex: {encoded_bytes.hex()}")
    
    # Convert back to bitstring for display
    bitstring = ''.join([format(b, '08b') for b in encoded_bytes])
    print(f"Bitstring: {bitstring}")
    print(f"Bitstring length: {len(bitstring)} bits")
    
    # Show the pattern in chunks of 8 bits for readability
    print(f"\nBitstring in 8-bit chunks:")
    for i in range(0, len(bitstring), 8):
        chunk = bitstring[i:i+8]
        print(f"  {i//8:2d}: {chunk}")

if __name__ == "__main__":
    debug_reed_muller_encoding()
