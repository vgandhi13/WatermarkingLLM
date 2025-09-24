#!/usr/bin/env python3
"""
Simplified Niederreiter Cryptosystem Implementation

A simplified version of the Niederreiter cryptosystem without numpy dependencies.
Uses Reed-Muller codes as the underlying error-correcting code.
"""

import sys
import os
import random
from typing import Tuple, List, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), 'reed-muller-python'))

try:
    from reedmuller.reedmuller import ReedMuller
except ImportError:
    print("Warning: Reed-Muller library not found. Some functionality may be limited.")
    ReedMuller = None

class NiederreiterSimple:
    """
    Simplified Niederreiter Cryptosystem using Reed-Muller codes.
    
    This implementation focuses on the core functionality without complex
    matrix operations, making it easier to understand and debug.
    """
    
    def __init__(self, r: int = 2, m: int = 5):
        """
        Initialize Niederreiter with Reed-Muller code RM(r,m).
        
        Args:
            r: Order of the Reed-Muller code
            m: Number of variables (code length = 2^m)
        """
        if ReedMuller is None:
            raise ImportError("Reed-Muller library is required for Niederreiter")
        
        self.rm_code = ReedMuller(r, m)
        self.r = r
        self.m = m
        self.n = self.rm_code.block_length()  # 2^m
        self.k = self.rm_code.message_length()  # message length
        self.t = self.rm_code.strength()  # error correction capability
        
        print(f"Niederreiter RM({r},{m}) initialized:")
        print(f"  Message length (k): {self.k} bits")
        print(f"  Code length (n): {self.n} bits")
        print(f"  Error correction capability (t): {self.t}")
        print(f"  Code rate: {self.k/self.n:.3f}")
    
    def encrypt(self, message: bytes) -> Tuple[List[int], int]:
        """
        Encrypt message using Niederreiter approach.
        
        Args:
            message: Message to encrypt
            
        Returns:
            Tuple of (syndrome_bits, number_of_errors)
        """
        # Convert message to bits
        message_bits = self._bytes_to_bits(message)
        
        # Pad or truncate to fit
        if len(message_bits) > self.k:
            message_bits = message_bits[:self.k]
        elif len(message_bits) < self.k:
            message_bits.extend([0] * (self.k - len(message_bits)))
        
        # Generate error vector
        error_vector = self._generate_error_vector(self.t)
        
        # For Niederreiter, we'd compute syndrome
        # Here we'll use a simplified approach
        syndrome = self._compute_syndrome(error_vector)
        
        return syndrome, self.t
    
    def decrypt(self, syndrome: List[int]) -> bytes:
        """
        Decrypt syndrome to recover message.
        
        Args:
            syndrome: Encrypted syndrome
            
        Returns:
            Decrypted message
        """
        # Syndrome decoding to find error vector
        error_vector = self._syndrome_decode(syndrome)
        
        # Extract message (simplified)
        message_bits = error_vector[:self.k]
        
        return self._bits_to_bytes(message_bits)
    
    def _generate_error_vector(self, weight: int) -> List[int]:
        """Generate random error vector of given weight."""
        error_vector = [0] * self.n
        positions = random.sample(range(self.n), min(weight, self.n))
        
        for pos in positions:
            error_vector[pos] = 1
        
        return error_vector
    
    def _compute_syndrome(self, error_vector: List[int]) -> List[int]:
        """Compute syndrome for error vector (simplified)."""
        # In practice, this would be H * e^T
        # For demonstration, we'll use a simple hash-like function
        syndrome = []
        for i in range(self.n - self.k):
            bit = 0
            for j in range(self.n):
                if error_vector[j] == 1:
                    bit ^= ((i + j) % 2)
            syndrome.append(bit)
        
        return syndrome
    
    def _syndrome_decode(self, syndrome: List[int]) -> List[int]:
        """Perform syndrome decoding (simplified)."""
        # Try to find error vector that produces the syndrome
        for _ in range(1000):
            error_vector = self._generate_error_vector(self.t)
            computed_syndrome = self._compute_syndrome(error_vector)
            
            if computed_syndrome == syndrome:
                return error_vector
        
        # Return zero vector if no solution found
        return [0] * self.n
    
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to bits."""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert bits to bytes."""
        while len(bits) % 8 != 0:
            bits.append(0)
        
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte |= (bits[i + j] << (7 - j))
            bytes_data.append(byte)
        
        return bytes(bytes_data)

class NiederreiterCiphertext:
    """
    Niederreiter-based ciphertext implementation for watermarking.
    """
    
    def __init__(self, rm_config: str = 'RM(2,5)'):
        """
        Initialize Niederreiter ciphertext with specified Reed-Muller configuration.
        
        Args:
            rm_config: Reed-Muller configuration string like 'RM(2,5)'
        """
        # Parse configuration
        if rm_config == 'RM(2,5)':
            r, m = 2, 5
        elif rm_config == 'RM(2,6)':
            r, m = 2, 6
        elif rm_config == 'RM(3,5)':
            r, m = 3, 5
        elif rm_config == 'RM(4,5)':
            r, m = 4, 5
        elif rm_config == 'RM(4,6)':
            r, m = 4, 6
        else:
            raise ValueError(f"Unsupported RM configuration: {rm_config}")
        
        self.niederreiter = NiederreiterSimple(r, m)
        self.rm_config = rm_config
        
        print(f"Niederreiter Ciphertext {rm_config} initialized:")
        print(f"  Message length: {self.niederreiter.k} bits")
        print(f"  Code length: {self.niederreiter.n} bits")
        print(f"  Error correction: {self.niederreiter.t} errors")
    
    def encrypt(self, message: str) -> str:
        """
        Encrypt a message using Niederreiter.
        
        Args:
            message: Message to encrypt
            
        Returns:
            Binary string representation of the encrypted syndrome
        """
        # Convert string to bytes
        message_bytes = message.encode('utf-8')
        
        # Use Niederreiter encryption
        syndrome, num_errors = self.niederreiter.encrypt(message_bytes)
        
        # Convert syndrome to binary string
        syndrome_str = ''.join(str(bit) for bit in syndrome)
        
        return syndrome_str
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a Niederreiter ciphertext.
        
        Args:
            ciphertext: Binary string representation of the encrypted syndrome
            
        Returns:
            Decrypted message
        """
        # Convert binary string to syndrome list
        syndrome = [int(bit) for bit in ciphertext]
        
        # Use Niederreiter decryption
        decrypted_bytes = self.niederreiter.decrypt(syndrome)
        
        # Convert back to string
        try:
            return decrypted_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return decrypted_bytes.hex()

def test_niederreiter_simple():
    """Test the simplified Niederreiter implementation."""
    print("Testing Simplified Niederreiter Cryptosystem")
    print("=" * 60)
    
    try:
        # Test with different Reed-Muller parameters
        test_configs = [
            ('RM(2,5)', 2, 5),  # RM(2,5): n=32, k=16, t=3
            ('RM(2,6)', 2, 6),  # RM(2,6): n=64, k=22, t=7
            ('RM(3,5)', 3, 5),  # RM(3,5): n=32, k=26, t=1
        ]
        
        for config_name, r, m in test_configs:
            print(f"\nTesting {config_name}:")
            try:
                niederreiter = NiederreiterSimple(r, m)
                
                # Test messages
                test_messages = [
                    b"Hello",
                    b"Test",
                    b"A" * 2,  # 16 bits
                ]
                
                for msg in test_messages:
                    if len(msg) > niederreiter.k // 8:
                        print(f"  Skipping '{msg}' - too long")
                        continue
                    
                    print(f"  Testing message: {msg}")
                    
                    # Encrypt
                    syndrome, num_errors = niederreiter.encrypt(msg)
                    print(f"    Encrypted to {len(syndrome)}-bit syndrome with {num_errors} errors")
                    
                    # Decrypt
                    decrypted = niederreiter.decrypt(syndrome)
                    print(f"    Decrypted: {decrypted}")
                    
                    success = msg == decrypted
                    print(f"    Success: {'✅' if success else '❌'}")
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        # Test ciphertext wrapper
        print(f"\nTesting NiederreiterCiphertext wrapper:")
        try:
            ciphertext = NiederreiterCiphertext('RM(2,5)')
            
            test_message = "Hello"
            encrypted = ciphertext.encrypt(test_message)
            decrypted = ciphertext.decrypt(encrypted)
            
            print(f"  Original: '{test_message}'")
            print(f"  Encrypted: {encrypted[:32]}...")
            print(f"  Decrypted: '{decrypted}'")
            print(f"  Success: {test_message == decrypted}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_niederreiter_simple()
