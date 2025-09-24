#!/usr/bin/env python3
"""
Function to calculate Hamming distance between two messages.
Hamming distance is the number of positions at which corresponding symbols differ.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reed-muller-python'))

def hamming_distance(msg1, msg2):
    """
    Calculate the Hamming distance between two messages.
    For strings, converts to bitstrings (8 bits per character) before comparison.
    
    Args:
        msg1: First message (string, bytes, or list of integers)
        msg2: Second message (string, bytes, or list of integers)
    
    Returns:
        int: Hamming distance between the messages
    
    Example:
        >>> hamming_distance("hello", "world")
        4
        >>> hamming_distance(b"hello", b"world")
        4
        >>> hamming_distance([1,0,1,0], [1,1,0,0])
        2
    """
    # Convert messages to bitstrings if they're strings or bytes
    if isinstance(msg1, str):
        msg1 = ''.join([format(ord(c), '08b') for c in msg1])
    elif isinstance(msg1, bytes):
        msg1 = ''.join([format(b, '08b') for b in msg1])
    
    if isinstance(msg2, str):
        msg2 = ''.join([format(ord(c), '08b') for c in msg2])
    elif isinstance(msg2, bytes):
        msg2 = ''.join([format(b, '08b') for b in msg2])
    
    # For lists, convert to bitstrings
    if isinstance(msg1, list):
        msg1 = ''.join([format(int(b), '08b') for b in msg1])
    if isinstance(msg2, list):
        msg2 = ''.join([format(int(b), '08b') for b in msg2])
    
    # Ensure both messages have the same length
    if len(msg1) != len(msg2):
        raise ValueError("Messages must have the same length for Hamming distance")
    
    # Count positions where they differ
    distance = sum(1 for a, b in zip(msg1, msg2) if a != b)
    return distance

def analyze_reed_muller_decoding(expected_message, codeword, r=1, m=7):
    """
    Decode a Reed-Muller codeword and analyze the distance from expected message.
    
    Args:
        expected_message: The original/expected message (string, bytes, or list)
        codeword: The Reed-Muller codeword to decode (list of bits)
        r: Reed-Muller order parameter (default: 1)
        m: Reed-Muller number of variables parameter (default: 7)
    
    Returns:
        dict: Dictionary containing analysis results
    """
    try:
        # Import Reed-Muller library
        from reedmuller.reedmuller import ReedMuller
        
        # Initialize Reed-Muller code
        rm = ReedMuller(r, m)
        
        print(f"Reed-Muller Code Parameters:")
        print(f"  Order (r): {r}")
        print(f"  Variables (m): {m}")
        print(f"  Code length (n): {rm.block_length()}")
        print(f"  Message length (k): {rm.message_length()}")
        print(f"  Error correction capability (t): {rm.strength()}")
        print()
        
        # Decode the codeword
        print(f"Input codeword: {codeword}")
        print(f"Codeword length: {len(codeword)} bits")
        
        decoded_message = rm.decode(codeword)
        print(f"Decoded message: {decoded_message}")
        
        # Convert decoded message to appropriate format for comparison
        if isinstance(expected_message, str):
            # Convert decoded bits to string
            decoded_str = ''.join([str(bit) for bit in decoded_message])
            # Convert binary string to actual string (assuming ASCII)
            try:
                # Pad to 8-bit boundaries if needed
                if len(decoded_str) % 8 != 0:
                    decoded_str = decoded_str.zfill(((len(decoded_str) + 7) // 8) * 8)
                
                # Convert 8-bit chunks to characters
                decoded_chars = []
                for i in range(0, len(decoded_str), 8):
                    byte_str = decoded_str[i:i+8]
                    if len(byte_str) == 8:
                        char_code = int(byte_str, 2)
                        decoded_chars.append(chr(char_code))
                
                decoded_final = ''.join(decoded_chars)
                print(f"Decoded as string: {decoded_final}")
                
                # Calculate distance
                distance = hamming_distance(expected_message, decoded_final)
                print(f"Hamming distance between expected and decoded: {distance}")
                
            except Exception as e:
                print(f"Could not convert decoded bits to string: {e}")
                # Fall back to bit-level comparison
                distance = hamming_distance(expected_message, decoded_message)
                print(f"Hamming distance (bit-level): {distance}")
        
        elif isinstance(expected_message, bytes):
            # Convert decoded bits to bytes
            decoded_str = ''.join([str(bit) for bit in decoded_message])
            # Pad to 8-bit boundaries if needed
            if len(decoded_str) % 8 != 0:
                decoded_str = decoded_str.zfill(((len(decoded_str) + 7) // 8) * 8)
            
            # Convert 8-bit chunks to bytes
            decoded_bytes = []
            for i in range(0, len(decoded_str), 8):
                byte_str = decoded_str[i:i+8]
                if len(byte_str) == 8:
                    byte_val = int(byte_str, 2)
                    decoded_bytes.append(byte_val)
            
            decoded_final = bytes(decoded_bytes)
            print(f"Decoded as bytes: {decoded_final}")
            
            # Calculate distance
            distance = hamming_distance(expected_message, decoded_final)
            print(f"Hamming distance between expected and decoded: {distance}")
        
        else:
            # For lists, compare directly
            distance = hamming_distance(expected_message, decoded_message)
            print(f"Hamming distance: {distance}")
        
        return {
            'expected': expected_message,
            'codeword': codeword,
            'decoded': decoded_message,
            'distance': distance,
            'rm_params': {'r': r, 'm': m, 'n': rm.block_length(), 'k': rm.message_length(), 't': rm.strength()}
        }
        
    except ImportError:
        print("Error: Reed-Muller library not found. Please install it first.")
        return None
    except Exception as e:
        print(f"Error during Reed-Muller decoding: {e}")
        return None

def decode_and_compare(expected_message, codeword, r=1, m=7, verbose=True):
    """
    Decode a Reed-Muller codeword and return the distance from expected message.
    This is a cleaner version that returns just the essential information.
    
    Args:
        expected_message: The original/expected message (string, bytes, or list)
        codeword: The Reed-Muller codeword to decode (list of bits)
        r: Reed-Muller order parameter (default: 1)
        m: Reed-Muller number of variables parameter (default: 7)
        verbose: Whether to print detailed output (default: True)
    
    Returns:
        dict: Dictionary containing {'distance': int, 'decoded': list, 'success': bool}
    """
    try:
        # Import Reed-Muller library
        from reedmuller.reedmuller import ReedMuller
        
        # Initialize Reed-Muller code
        rm = ReedMuller(r, m)
        
        if verbose:
            print(f"Reed-Muller RM({r},{m}): n={rm.block_length()}, k={rm.message_length()}, t={rm.strength()}")
        
        # Decode the codeword
        decoded_message = rm.decode(codeword)
        
        if decoded_message is None:
            if verbose:
                print("Decoding failed - too many errors")
            return {'distance': None, 'decoded': None, 'success': False}
        
        if verbose:
            print(f"Decoded: {decoded_message}")
        
        # Calculate distance based on message type
        if isinstance(expected_message, str):
            # Convert decoded bits to string
            decoded_str = ''.join([str(bit) for bit in decoded_message])
            # Pad to 8-bit boundaries if needed
            if len(decoded_str) % 8 != 0:
                decoded_str = decoded_str.zfill(((len(decoded_str) + 7) // 8) * 8)
            
            # Convert 8-bit chunks to characters
            decoded_chars = []
            for i in range(0, len(decoded_str), 8):
                byte_str = decoded_str[i:i+8]
                if len(byte_str) == 8:
                    char_code = int(byte_str, 2)
                    decoded_chars.append(chr(char_code))
            
            decoded_final = ''.join(decoded_chars)
            distance = hamming_distance(expected_message, decoded_final)
            
            if verbose:
                print(f"Decoded as string: '{decoded_final}'")
                print(f"Hamming distance: {distance}")
        
        elif isinstance(expected_message, bytes):
            # Convert decoded bits to bytes
            decoded_str = ''.join([str(bit) for bit in decoded_message])
            # Pad to 8-bit boundaries if needed
            if len(decoded_str) % 8 != 0:
                decoded_str = decoded_str.zfill(((len(decoded_str) + 7) // 8) * 8)
            
            # Convert 8-bit chunks to bytes
            decoded_bytes = []
            for i in range(0, len(decoded_str), 8):
                byte_str = decoded_str[i:i+8]
                if len(byte_str) == 8:
                    byte_val = int(byte_str, 2)
                    decoded_bytes.append(byte_val)
            
            decoded_final = bytes(decoded_bytes)
            distance = hamming_distance(expected_message, decoded_final)
            
            if verbose:
                print(f"Decoded as bytes: {decoded_final}")
                print(f"Hamming distance: {distance}")
        
        else:
            # For lists, compare directly
            distance = hamming_distance(expected_message, decoded_message)
            decoded_final = decoded_message
            
            if verbose:
                print(f"Hamming distance: {distance}")
        
        return {
            'distance': distance,
            'decoded': decoded_final,
            'success': True
        }
        
    except ImportError:
        if verbose:
            print("Error: Reed-Muller library not found")
        return {'distance': None, 'decoded': None, 'success': False}
    except Exception as e:
        if verbose:
            print(f"Error during decoding: {e}")
        return {'distance': None, 'decoded': None, 'success': False}

def test_hamming_distance():
    """Test the Hamming distance function."""
    print("Testing Hamming Distance Function")
    print("="*40)
    
    # Test cases
    test_cases = [
        ("hello", "world"),
        (b"hello", b"world"),
        ([1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 2, 3], [4, 5, 6])
    ]
    
    for i, (msg1, msg2) in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"  Message 1: {msg1}")
        print(f"  Message 2: {msg2}")
        
        # Show bitstring representation for strings and bytes
        if isinstance(msg1, str):
            bits1 = ''.join([format(ord(c), '08b') for c in msg1])
            bits2 = ''.join([format(ord(c), '08b') for c in msg2])
            print(f"  Bitstring 1: {bits1}")
            print(f"  Bitstring 2: {bits2}")
        elif isinstance(msg1, bytes):
            bits1 = ''.join([format(b, '08b') for b in msg1])
            bits2 = ''.join([format(b, '08b') for b in msg2])
            print(f"  Bitstring 2: {bits2}")
        
        print(f"  Hamming distance: {hamming_distance(msg1, msg2)}")

def test_reed_muller_analysis():
    """Test the Reed-Muller analysis function."""
    print("\n" + "="*60)
    print("Testing Reed-Muller Analysis Function")
    print("="*60)
    
    try:
        # Import Reed-Muller library
        from reedmuller.reedmuller import ReedMuller
        
        # Test with a simple case
        expected_msg = "H"
        
        # Create a Reed-Muller code and encode the message first
        rm = ReedMuller(1, 7)
        print(f"Reed-Muller Code RM(1,7):")
        print(f"  Code length (n): {rm.block_length()}")
        print(f"  Message length (k): {rm.message_length()}")
        print(f"  Error correction capability (t): {rm.strength()}")
        print()
        
        # Convert message to binary
        message_bits = []
        for c in expected_msg:
            message_bits.extend([int(b) for b in format(ord(c), '08b')])
        
        # Pad or truncate to exact length k
        if len(message_bits) > rm.message_length():
            message_bits = message_bits[:rm.message_length()]
        elif len(message_bits) < rm.message_length():
            message_bits.extend([0] * (rm.message_length() - len(message_bits)))
        
        print(f"Message '{expected_msg}' as bits: {message_bits}")
        print(f"Message bit length: {len(message_bits)}")
        
        # Encode the message
        encoded_codeword = rm.encode(message_bits)
        print(f"Encoded codeword: {encoded_codeword}")
        print(f"Codeword length: {len(encoded_codeword)} bits")
        
        # Now test the analysis function
        print(f"\n" + "="*40)
        print("Testing Analysis Function")
        print("="*40)
        
        result = analyze_reed_muller_decoding(expected_msg, encoded_codeword)
        
        if result:
            print(f"\nAnalysis complete!")
            print(f"Result summary: {result}")
        
        # Test the new cleaner function
        print(f"\n" + "="*40)
        print("Testing Clean Decode and Compare Function")
        print("="*40)
        
        clean_result = decode_and_compare(expected_msg, encoded_codeword)
        print(f"Clean result: {clean_result}")
    
    except ImportError:
        print("Error: Reed-Muller library not found. Please install it first.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hamming_distance()
    test_reed_muller_analysis()
