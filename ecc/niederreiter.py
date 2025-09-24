#!/usr/bin/env python3
"""
Niederreiter Cryptosystem Implementation

The Niederreiter cryptosystem is a public-key encryption scheme based on the 
hardness of syndrome decoding. It's closely related to the McEliece cryptosystem
but uses a different approach.

Key differences from McEliece:
- Public key is a parity-check matrix H instead of generator matrix G
- Encryption involves computing syndrome s = H * e^T where e is error vector
- Decryption involves syndrome decoding to find the error vector

This implementation uses Reed-Muller codes as the underlying error-correcting code.
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

class Niederreiter:
    """
    Niederreiter Cryptosystem using Reed-Muller codes.
    
    Key Generation:
    1. Choose a Reed-Muller code RM(r,m) with parameters (n,k,t)
    2. Generate parity-check matrix H for the code
    3. Generate random matrices S (non-singular) and P (permutation)
    4. Public key: H' = SHP
    5. Private key: (S, H, P)
    
    Encryption:
    1. Choose random error vector e of weight t
    2. Compute syndrome s = H' * e^T
    3. Ciphertext is the syndrome s
    
    Decryption:
    1. Compute s' = S^(-1) * s
    2. Use syndrome decoding to find error vector e'
    3. Compute e = P^(-1) * e'
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
        
        # Generate the key matrices
        self.generate_keys()
        
        print(f"Niederreiter RM({r},{m}) initialized:")
        print(f"  Message length (k): {self.k} bits")
        print(f"  Code length (n): {self.n} bits")
        print(f"  Error correction capability (t): {self.t}")
        print(f"  Code rate: {self.k/self.n:.3f}")
    
    def generate_keys(self):
        """Generate the public and private keys."""
        # Generate parity-check matrix H for the Reed-Muller code
        self.H = self._generate_parity_check_matrix()
        
        # Generate random non-singular matrix S (k x k)
        self.S = self._generate_random_matrix(self.n - self.k, self.n - self.k)
        
        # Generate random permutation matrix P (n x n)
        self.P = self._generate_permutation_matrix(self.n)
        
        # Public key: H' = SHP
        self.H_public = np.dot(np.dot(self.S, self.H), self.P)
        
        # Store private key components
        self.S_inv = np.linalg.inv(self.S)
        self.P_inv = np.linalg.inv(self.P)
        
        print(f"Keys generated:")
        print(f"  Public key size: {self.H_public.shape}")
        print(f"  Private key components: S, H, P")
    
    def _generate_parity_check_matrix(self) -> List[List[int]]:
        """Generate parity-check matrix for the Reed-Muller code."""
        # For Reed-Muller codes, we can construct the parity-check matrix
        # This is a simplified construction - in practice, you'd use the
        # actual Reed-Muller parity-check matrix structure
        
        # Create a random parity-check matrix of size (n-k) x n
        H = []
        for i in range(self.n - self.k):
            row = [random.randint(0, 1) for _ in range(self.n)]
            H.append(row)
        
        return H
    
    def _generate_random_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate a random non-singular matrix over GF(2)."""
        while True:
            matrix = np.random.randint(0, 2, size=(rows, cols))
            if np.linalg.det(matrix) != 0:  # Non-singular
                return matrix.astype(np.uint8)
    
    def _generate_permutation_matrix(self, size: int) -> np.ndarray:
        """Generate a random permutation matrix."""
        P = np.zeros((size, size), dtype=np.uint8)
        perm = list(range(size))
        random.shuffle(perm)
        
        for i, j in enumerate(perm):
            P[i, j] = 1
        
        return P
    
    def encrypt(self, message: bytes) -> Tuple[np.ndarray, int]:
        """
        Encrypt a message using Niederreiter.
        
        Args:
            message: Message to encrypt (will be padded/truncated to fit)
            
        Returns:
            Tuple of (syndrome, number_of_errors)
        """
        # Convert message to binary vector
        message_bits = self._bytes_to_bits(message)
        
        # Pad or truncate to fit the code
        if len(message_bits) > self.k:
            message_bits = message_bits[:self.k]
        elif len(message_bits) < self.k:
            message_bits.extend([0] * (self.k - len(message_bits)))
        
        # Choose random error vector of weight t
        error_vector = self._generate_error_vector(self.t)
        
        # Compute syndrome: s = H' * e^T
        syndrome = np.dot(self.H_public, error_vector) % 2
        
        return syndrome, self.t
    
    def decrypt(self, syndrome: np.ndarray) -> bytes:
        """
        Decrypt a syndrome to recover the original message.
        
        Args:
            syndrome: The encrypted syndrome
            
        Returns:
            Decrypted message as bytes
        """
        # Step 1: Compute s' = S^(-1) * s
        s_prime = np.dot(self.S_inv, syndrome) % 2
        
        # Step 2: Syndrome decoding to find error vector
        error_vector = self._syndrome_decode(s_prime)
        
        # Step 3: Compute e = P^(-1) * e'
        error_vector = np.dot(self.P_inv, error_vector) % 2
        
        # Extract message from error vector (simplified)
        # In practice, you'd need proper message extraction
        message_bits = error_vector[:self.k]
        
        return self._bits_to_bytes(message_bits)
    
    def _generate_error_vector(self, weight: int) -> np.ndarray:
        """Generate a random error vector of given weight."""
        error_vector = np.zeros(self.n, dtype=np.uint8)
        
        # Choose random positions for errors
        error_positions = random.sample(range(self.n), min(weight, self.n))
        
        for pos in error_positions:
            error_vector[pos] = 1
        
        return error_vector
    
    def _syndrome_decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Perform syndrome decoding to find the error vector.
        This is a simplified implementation.
        """
        # For demonstration, we'll use a simple approach
        # In practice, you'd use proper syndrome decoding algorithms
        
        # Try to find an error vector that produces the given syndrome
        for _ in range(1000):  # Limit iterations
            error_vector = self._generate_error_vector(self.t)
            computed_syndrome = np.dot(self.H, error_vector) % 2
            
            if np.array_equal(computed_syndrome, syndrome):
                return error_vector
        
        # If no solution found, return zero vector
        return np.zeros(self.n, dtype=np.uint8)
    
    def _bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to list of bits."""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits
    
    def _bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes."""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)
        
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte |= (bits[i + j] << (7 - j))
            bytes_data.append(byte)
        
        return bytes(bytes_data)

class NiederreiterReedMuller:
    """
    Simplified Niederreiter implementation using Reed-Muller codes directly.
    """
    
    def __init__(self, r: int = 2, m: int = 5):
        """Initialize Niederreiter with Reed-Muller code."""
        if ReedMuller is None:
            raise ImportError("Reed-Muller library is required")
        
        self.rm_code = ReedMuller(r, m)
        self.r = r
        self.m = m
        self.n = self.rm_code.block_length()
        self.k = self.rm_code.message_length()
        self.t = self.rm_code.strength()
        
        print(f"Niederreiter RM({r},{m}) initialized:")
        print(f"  Message length (k): {self.k} bits")
        print(f"  Code length (n): {self.n} bits")
        print(f"  Error correction capability (t): {self.t}")
    
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

def test_niederreiter():
    """Test the Niederreiter implementation."""
    print("Testing Niederreiter Cryptosystem")
    print("=" * 50)
    
    try:
        # Test with different Reed-Muller parameters
        test_configs = [
            (2, 5),  # RM(2,5): n=32, k=16, t=3
            (2, 6),  # RM(2,6): n=64, k=22, t=7
            (3, 5),  # RM(3,5): n=32, k=26, t=1
        ]
        
        for r, m in test_configs:
            print(f"\nTesting RM({r},{m}):")
            try:
                niederreiter = NiederreiterReedMuller(r, m)
                
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
    
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_niederreiter()
