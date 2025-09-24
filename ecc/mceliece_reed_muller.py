# McEliece Cryptosystem using Reed-Muller Codes
# Based on the original McEliece implementation but adapted for Reed-Muller codes
# Reference: https://github.com/jkrauze/mceliece/blob/master/mceliece/mceliececipher.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reed-muller-python'))

import numpy as np
from reedmuller.reedmuller import ReedMuller

class ReedMullerCode:
    """Wrapper class for Reed-Muller codes to provide a consistent interface."""
    
    def __init__(self, r=1, m=7):
        """
        Initialize Reed-Muller code RM(r,m)
        r: order of the code (degree of polynomials)
        m: number of variables (code length will be 2^m)
        """
        self.rm = ReedMuller(r, m)
        self.r = r
        self.m = m
        self.n = self.rm.block_length()  # 2^m
        self.k = self.rm.message_length()  # sum of binomial coefficients
        self.t = self.rm.strength()  # error correction capability
    
    def encode(self, message: bytes) -> bytes:
        """Encode a message using Reed-Muller code."""
        # Convert bytes to binary list
        binary_list = []
        for byte in message:
            binary_list.extend([int(b) for b in format(byte, '08b')])
        
        # Truncate or pad to exact length k
        if len(binary_list) > self.k:
            binary_list = binary_list[:self.k]
        elif len(binary_list) < self.k:
            binary_list.extend([0] * (self.k - len(binary_list)))
        
        # Encode using Reed-Muller
        encoded_list = self.rm.encode(binary_list)
        
        # Convert back to bytes
        encoded_bytes = bytearray()
        for i in range(0, len(encoded_list), 8):
            byte_bits = encoded_list[i:i+8]
            if len(byte_bits) < 8:
                byte_bits.extend([0] * (8 - len(byte_bits)))
            byte_val = int(''.join(str(bit) for bit in byte_bits), 2)
            encoded_bytes.append(byte_val)
        
        return bytes(encoded_bytes)
    
    def decode(self, encoded_message: bytes) -> bytes:
        """Decode a message using Reed-Muller code."""
        try:
            # Convert bytes to binary list
            binary_list = []
            for byte in encoded_message:
                binary_list.extend([int(b) for b in format(byte, '08b')])
            
            # Truncate or pad to exact length n
            if len(binary_list) > self.n:
                binary_list = binary_list[:self.n]
            elif len(binary_list) < self.n:
                binary_list.extend([0] * (self.n - len(binary_list)))
            
            # Decode using Reed-Muller
            decoded_list = self.rm.decode(binary_list)
            
            if decoded_list is None:
                raise ValueError("Unable to decode message - too many errors")
            
            # Convert back to bytes
            decoded_bytes = bytearray()
            for i in range(0, len(decoded_list), 8):
                byte_bits = decoded_list[i:i+8]
                if len(byte_bits) < 8:
                    byte_bits.extend([0] * (8 - len(byte_bits)))
                byte_val = int(''.join(str(bit) for bit in byte_bits), 2)
                decoded_bytes.append(byte_val)
            
            return bytes(decoded_bytes).rstrip(b'\0')
            
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def introduce_errors(self, encoded_message: bytes, num_errors: int) -> bytes:
        """Introduce random bit errors in the encoded message."""
        # Convert to binary list
        binary_list = []
        for byte in encoded_message:
            binary_list.extend([int(b) for b in format(byte, '08b')])
        
        # Introduce bit errors
        positions = np.random.choice(len(binary_list), min(num_errors, len(binary_list)), replace=False)
        for pos in positions:
            binary_list[pos] = 1 - binary_list[pos]  # Flip the bit
        
        # Convert back to bytes
        encoded_bytes = bytearray()
        for i in range(0, len(binary_list), 8):
            byte_bits = binary_list[i:i+8]
            if len(byte_bits) < 8:
                byte_bits.extend([0] * (8 - len(byte_bits)))
            byte_val = int(''.join(str(bit) for bit in byte_bits), 2)
            encoded_bytes.append(byte_val)
        
        return bytes(encoded_bytes)

class McElieceReedMuller:
    """
    McEliece Cryptosystem using Reed-Muller codes as the underlying error-correcting code.
    
    Key Generation:
    1. Choose a Reed-Muller code RM(r,m) that can correct t errors
    2. Generate three matrices:
       - G (the generator matrix of the code)
       - S (a random non-singular matrix)
       - P (a random permutation matrix)
    3. Public key: G' = SGP
    4. Private key: (S, G, P)
    
    Encryption:
    1. Convert message m to binary
    2. Compute c = mG' + e
       where e is a random error vector of weight t
    
    Decryption:
    1. Compute cP^(-1)
    2. Use the Reed-Muller decoding algorithm to remove errors
    3. Multiply by S^(-1) to get m
    """
    
    def __init__(self, r=1, m=7):
        """
        Initialize McEliece with Reed-Muller code RM(r,m)
        r: order of the Reed-Muller code
        m: number of variables (code length = 2^m)
        """
        self.rm_code = ReedMullerCode(r, m)
        self.r = r
        self.m = m
        self.n = self.rm_code.n  # 2^m
        self.k = self.rm_code.k  # message length
        self.t = self.rm_code.t  # error correction capability
        
        # Generate the key matrices
        self.generate_keys()
    
    def generate_keys(self):
        """Generate the public and private keys."""
        # Generate random binary matrices
        self.S = np.random.randint(0, 2, (self.k, self.k))
        self.P = np.random.permutation(np.eye(self.n))
        
        # Make sure S is invertible (over GF(2)) - use identity matrix as base
        self.S = np.eye(self.k, dtype=int)
        # Add some random rows to make it non-trivial but still invertible
        for i in range(self.k):
            for j in range(i+1, self.k):
                if np.random.random() > 0.5:
                    self.S[i] = (self.S[i] + self.S[j]) % 2
        
        # Calculate inverses (over GF(2))
        self.S_inv = self._matrix_inverse_gf2(self.S)
        self.P_inv = np.linalg.inv(self.P)
        
        # Generate the generator matrix G for the Reed-Muller code
        self.G = self._generate_reed_muller_generator_matrix()
        
        # Public key: G' = SGP
        self.G_prime = (self.S @ self.G @ self.P) % 2  # Ensure binary
    
    def _matrix_inverse_gf2(self, matrix):
        """Calculate matrix inverse over GF(2)."""
        n = matrix.shape[0]
        # Create augmented matrix [A|I]
        augmented = np.hstack([matrix, np.eye(n)])
        
        # Gaussian elimination over GF(2)
        for i in range(n):
            # Find pivot
            pivot_row = i
            for j in range(i + 1, n):
                if augmented[j, i] == 1:
                    pivot_row = j
                    break
            
            if augmented[pivot_row, i] == 0:
                raise ValueError("Matrix is not invertible over GF(2)")
            
            # Swap rows if necessary
            if pivot_row != i:
                augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
            
            # Eliminate column i
            for j in range(n):
                if j != i and augmented[j, i] == 1:
                    augmented[j] = (augmented[j] + augmented[i]) % 2
        
        return augmented[:, n:] % 2
    
    def _generate_reed_muller_generator_matrix(self):
        """Generate the generator matrix for the Reed-Muller code."""
        # Use the encoding function to build the matrix
        matrix = []
        for i in range(self.k):
            # Create unit vector with 1 at position i
            unit_vector = [0] * self.k
            unit_vector[i] = 1
            # Encode this unit vector
            encoded = self.rm_code.rm.encode(unit_vector)
            matrix.append(encoded)
        
        # Return as k x n matrix (transpose of what we built)
        return np.array(matrix)
    
    def encrypt(self, message: bytes, t=None) -> tuple:
        """
        Encrypt a message using the public key.
        
        Args:
            message: Message to encrypt (bytes)
            t: Number of errors to introduce (default: use code's error correction capability)
        
        Returns:
            tuple: (ciphertext, number_of_errors)
        """
        if t is None:
            t = self.t
        
        # Convert to binary vector
        binary_vector = []
        for byte in message:
            binary_vector.extend([int(b) for b in format(byte, '08b')])
        
        # Truncate or pad to exact length k
        if len(binary_vector) > self.k:
            binary_vector = binary_vector[:self.k]
        elif len(binary_vector) < self.k:
            binary_vector.extend([0] * (self.k - len(binary_vector)))
        
        # Convert to numpy array
        m = np.array(binary_vector)
        
        # Compute c = mG' + e
        c = (m @ self.G_prime) % 2
        
        # Generate random error vector e of weight t
        e = np.zeros(self.n, dtype=int)
        error_positions = np.random.choice(self.n, t, replace=False)
        e[error_positions] = 1
        
        # Add error vector
        ciphertext = (c + e) % 2
        
        # Convert back to bytes
        ciphertext_bytes = bytearray()
        for i in range(0, len(ciphertext), 8):
            byte_bits = ciphertext[i:i+8]
            if len(byte_bits) < 8:
                byte_bits = np.append(byte_bits, [0] * (8 - len(byte_bits)))
            byte_val = int(''.join(str(int(bit)) for bit in byte_bits), 2)
            ciphertext_bytes.append(byte_val)
        
        return bytes(ciphertext_bytes), t
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt a message using the private key.
        
        Args:
            ciphertext: Encrypted message (bytes)
        
        Returns:
            bytes: Decrypted message
        """
        try:
            # Convert ciphertext to binary vector
            binary_vector = []
            for byte in ciphertext:
                binary_vector.extend([int(b) for b in format(byte, '08b')])
            
            # Truncate or pad to exact length n
            if len(binary_vector) > self.n:
                binary_vector = binary_vector[:self.n]
            elif len(binary_vector) < self.n:
                binary_vector.extend([0] * (self.n - len(binary_vector)))
            
            # Convert to numpy array
            c = np.array(binary_vector)
            
            # Step 1: Compute cP^(-1)
            c_prime = (c @ self.P_inv) % 2
            
            # Step 2: Use Reed-Muller decoding to remove errors
            # Convert to list for the RM decoder
            c_prime_list = c_prime.tolist()
            decoded_list = self.rm_code.rm.decode(c_prime_list)
            
            if decoded_list is None:
                raise ValueError("Unable to decode message - too many errors")
            
            # Step 3: Multiply by S^(-1) to get the original message
            decoded_vector = np.array(decoded_list)
            m = (decoded_vector @ self.S_inv) % 2
            
            # Convert back to bytes
            decoded_bytes = bytearray()
            for i in range(0, len(m), 8):
                byte_bits = m[i:i+8]
                if len(byte_bits) < 8:
                    byte_bits = np.append(byte_bits, [0] * (8 - len(byte_bits)))
                byte_val = int(''.join(str(int(bit)) for bit in byte_bits), 2)
                decoded_bytes.append(byte_val)
            
            return bytes(decoded_bytes).rstrip(b'\0')
            
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")

def print_binary(data: bytes, label: str):
    """Utility function to print binary representation of data."""
    print("\n" + label + ":")
    print("Bytes: " + str(data))
    print("Binary: " + " ".join([format(b, '08b') for b in data[:8]]))  # First 8 bytes in binary
    print("Length: " + str(len(data)) + " bytes")
    
    # Count bits
    total_0, total_1 = 0, 0
    for byte in data:
        binary = format(byte, '08b')
        total_0 += binary.count('0')
        total_1 += binary.count('1')
    total_bits = total_0 + total_1
    if total_bits > 0:
        print(f"Average 0s: {total_0/total_bits:.3f}, Average 1s: {total_1/total_bits:.3f}")

def test_mceliece_reed_muller():
    """Test function for the Reed-Muller based McEliece implementation."""
    print("Testing McEliece with Reed-Muller codes...")
    
    # Test different Reed-Muller parameters
    test_cases = [
        (1, 6),  # RM(1,6): n=64, k=7, t=15
        (2, 6),  # RM(2,6): n=64, k=22, t=7
        (1, 7),  # RM(1,7): n=128, k=8, t=31
    ]
    
    for r, m in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing RM({r},{m}) - n=2^{m}={2**m}, k={sum([__import__('math').comb(m, i) for i in range(r+1)])}")
        print(f"{'='*60}")
        
        try:
            mc = McElieceReedMuller(r, m)
            print(f"Code parameters: n={mc.n}, k={mc.k}, t={mc.t}")
            
            # Test messages
            messages = [
                b"Hello",
                b"McEliece",
                b"Reed-Muller",
                b"Test message for RM codes"
            ]
            
            for msg in messages:
                if len(msg) > mc.k // 8:
                    print(f"Skipping '{msg}' - too long ({len(msg)} > {mc.k // 8})")
                    continue
                
                print(f"\nTesting message: {msg}")
                print_binary(msg, "Original Message")
                
                try:
                    # Encrypt
                    cipher, num_errors = mc.encrypt(msg)
                    print_binary(cipher, f"Encrypted Message (with {num_errors} errors)")
                    
                    # Decrypt
                    decrypted = mc.decrypt(cipher)
                    print_binary(decrypted, "Decrypted Message")
                    
                    success = msg == decrypted
                    print(f"Decryption Success: {success}")
                    
                    if not success:
                        print(f"Expected: {msg}")
                        print(f"Got: {decrypted}")
                        
                except Exception as e:
                    print(f"Error: {str(e)}")
                    
        except Exception as e:
            print(f"Failed to initialize RM({r},{m}): {str(e)}")

if __name__ == "__main__":
    test_mceliece_reed_muller()
