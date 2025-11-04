#!/usr/bin/env python3

import numpy as np
import random
import hashlib
import sys
import os
import math
# Add the reed-muller-python directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'reed-muller-python'))
from reedmuller.reedmuller import ReedMuller


# McEliece parameters for RM(1,7)
MCELIECE_R = 1
MCELIECE_M = 7
MCELIECE_N = 2**MCELIECE_M  # 128
MCELIECE_K = MCELIECE_M + 1  # 8 (for RM(1,7))
MCELIECE_T = 2**(MCELIECE_M - MCELIECE_R - 1) - 1  # 31


class McEliece:
    """
    McEliece cryptosystem implementation using Reed-Muller codes.
    """
    
    def __init__(self):
        """
        Key generation for McEliece cryptosystem.
        Implements the algorithm:
        1. S ←$ GL(k, F_q)  (random invertible k×k matrix)
        2. P ←$ S_n         (random permutation matrix)
        3. Ĝ ← SGP          (public key generator matrix)
        4. tk_M ← (S, P)    (private key)
        5. Return (Ĝ, tk_M)
        """
        # Initialize Reed-Muller code RM(1,7)
        self.rm_code = ReedMuller(MCELIECE_R, MCELIECE_M)
        
        # Get the generator matrix G from Reed-Muller code
        # Convert from list of tuples to numpy array for easier manipulation
        # The Reed-Muller M is (n, k), but we need (k, n) for McEliece
        self.G = np.array(self.rm_code.M, dtype=int).T
        
        # Parameters
        self.k = MCELIECE_K  # 8
        self.n = MCELIECE_N  # 128
        self.t = MCELIECE_T  # 31
        
        # Step 1: Generate random invertible k×k matrix S
        self.S = self._generate_random_invertible_matrix(self.k)
        
        # Step 2: Generate random permutation matrix P
        self.P = self._generate_random_permutation_matrix(self.n)
        
        # Step 3: Compute public key Ĝ = SGP
        self.G_hat = self._matrix_multiply_mod2(
            self._matrix_multiply_mod2(self.S, self.G), 
            self.P
        )
        
        # Step 4: Private key is (S, P)
        self.private_key = (self.S, self.P)
        
        # Public key is Ĝ
        self.public_key = self.G_hat

        self.lazy_perm = format(random.getrandbits(8), f'08b')

    
    def _generate_random_invertible_matrix(self, size):
        """
        Generate a random invertible k×k matrix over F_2.
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            # Generate random binary matrix
            matrix = np.random.randint(0, 2, (size, size), dtype=int)
            
            # Check if invertible (determinant is 1 mod 2)
            if self._is_invertible_mod2(matrix):
                return matrix
        
        raise RuntimeError("Failed to generate invertible matrix after maximum attempts")
    
    def _is_invertible_mod2(self, matrix):
        """
        Check if matrix is invertible over F_2 using Gaussian elimination.
        """
        matrix = matrix.copy()
        n = matrix.shape[0]
        
        # Gaussian elimination mod 2
        for i in range(n):
            # Find pivot
            if matrix[i, i] == 0:
                # Look for a row to swap
                for j in range(i + 1, n):
                    if matrix[j, i] == 1:
                        matrix[[i, j]] = matrix[[j, i]]
                        break
                else:
                    return False  # No pivot found, matrix is singular
            
            # Eliminate column
            for j in range(i + 1, n):
                if matrix[j, i] == 1:
                    matrix[j] = (matrix[j] + matrix[i]) % 2
        
        return True
    
    def _generate_random_permutation_matrix(self, size):
        """
        Generate a random n×n permutation matrix.
        """
        # Generate random permutation
        permutation = np.random.permutation(size)
        
        # Create permutation matrix
        P = np.zeros((size, size), dtype=int)
        for i, j in enumerate(permutation):
            P[i, j] = 1
        
        return P
    def lazy_prf(self, message): 
        #random string of length m
        m = len(message)
        return format(int(message, 2) ^ int(self.lazy_perm, 2), f'0{m}b')
    def bijection_function(self, integer, n, k):
        # convert an integer to a binary string of length n with hamming weight k 
        # This implements the unranking algorithm for combinations
        s = ['0'] * n
        rank = integer
        ones_remaining = k
        
        for i in range(n-1, -1, -1):
            if ones_remaining == 0:
                break
            if i < ones_remaining - 1:
                break
                
            # Calculate C(i, ones_remaining)
            combination_count = math.comb(i, ones_remaining)
            
            if rank < combination_count:
                # Place a 0 at position i
                s[i] = '0'
            else:
                # Place a 1 at position i
                s[i] = '1'
                rank = rank - combination_count
                ones_remaining = ones_remaining - 1
                
        return ''.join(s)   
    def _matrix_multiply_mod2(self, A, B):
        """
        Matrix multiplication mod 2.
        """
        result = np.dot(A, B) % 2
        return result.astype(int)
    
    def get_public_key(self):
        """Return the public key (Ĝ)."""
        return self.G_hat
    
    def get_private_key(self):
        """Return the private key (S, P)."""
        return self.private_key
    
    def get_parameters(self):
        """Return the McEliece parameters."""
        return {
            'k': self.k,
            'n': self.n, 
            't': self.t,
            'r': MCELIECE_R,
            'm': MCELIECE_M
        }
    
    def encrypt(self, message, error_vector=None):
        """
        McEliece encryption algorithm M[C].Ev.
        
        Args:
            message: A string of k bits (0s and 1s) representing the message
            
        Returns:
            ciphertext: A string of n bits representing the encrypted message
        """
        # Convert string to list of integers if needed
        if isinstance(message, str):
            message_list = [int(c) for c in message]
        else:
            message_list = message
            
        # Validate inputs
        if len(message_list) != self.k:
            message_list = message_list[:self.k]
        
        # Generate random error vector with weight t
        if not error_vector:
            integer = int(self.lazy_prf(message) , 2)
            error_vector = self.bijection_function(integer, 128, self.t//2)
        
        # Convert error_vector to list of integers if it's a string
        if isinstance(error_vector, str):
            error_vector_list = [int(c) for c in error_vector]
        else:
            error_vector_list = error_vector
        
        # McEliece encryption
        message = np.array(message_list, dtype=int)
        error_vector = np.array(error_vector_list, dtype=int)
        
        # Compute m^T * Ĝ (message transpose times public key matrix)
        message_encoded = self._matrix_multiply_mod2(message.reshape(1, -1), self.G_hat)
        message_encoded = message_encoded.flatten()
        
        # Add error vector: y = m^T * Ĝ + e
        ciphertext = (message_encoded + error_vector) % 2
        return ''.join(map(str, ciphertext.tolist()))

    
    def generate_random_error_vector(self, weight=None):
        """
        Generate a random error vector of exactly t weight.
        
        Args:
            weight: Number of 1s in the error vector. If None, uses t.
            
        Returns:
            error_vector: A list of n bits with exactly the specified weight
        """
        if weight is None:
            weight = self.t
        elif weight > self.t:
            raise ValueError(f"Error weight must be ≤ {self.t}, got {weight}")
        
        # Create error vector with exactly the specified weight
        error_vector = [0] * self.n
        
        # Generate t different numbers in the range [0, n-1]
        positions = random.sample(range(self.n), weight)
        
        # Set those bits to 1
        for pos in positions:
            error_vector[pos] = 1
        
        return error_vector
    
    def decrypt(self, ciphertext):
        """
        McEliece decryption algorithm M[C].Inv.
        
        Args:
            ciphertext: A string of n bits representing the encrypted message
            
        Returns:
            tuple: (decoded_message, recovered_error) as bitstrings
        """
        # Convert string to list of integers if needed
        if isinstance(ciphertext, str):
            ciphertext_list = [int(c) for c in ciphertext]
        else:
            ciphertext_list = ciphertext
            
        # Validate input
        if len(ciphertext_list) != self.n:
            raise ValueError(f"Ciphertext length must be {self.n}, got {len(ciphertext_list)}")
        
        # Step 1: Unpack private key (S, P) ← tk_M
        S, P = self.private_key
        
        # Step 2: c_hat ← c * P^-1 (un-permute the ciphertext)
        # For permutation matrix, P^-1 = P^T
        P_inv = P.T
        c_hat = self._matrix_multiply_mod2(np.array(ciphertext_list).reshape(1, -1), P_inv).flatten()
        
        # Step 3: m_hat ← A(c_hat) (decode using Reed-Muller decoder)
        c_hat_list = c_hat.tolist()
        m_hat = self.rm_code.decode(c_hat_list)
        
        if m_hat is None:
            return None  # Decoding failed
        
        # Step 4: m ← m_hat * S^-1 (recover original message)
        S_inv = self._matrix_inverse_mod2(S)
        decoded_message = self._matrix_multiply_mod2(
            np.array(m_hat).reshape(1, -1), S_inv
        ).flatten().tolist()
        
        # Calculate the error vector for verification
        integer = int(self.lazy_prf(''.join(map(str, decoded_message))) , 2)
        recovered_error_integer = self.bijection_function(integer, 128, self.t//2)
        
        # Return the decoded message and error vector
        decoded_message_str = ''.join(map(str, decoded_message))
        recovered_error_str = ''.join(map(str, recovered_error_integer))
        return (decoded_message_str, recovered_error_str)


    
    
    
    def _is_valid_codeword(self, codeword):
        """
        Check if a codeword is valid by trying to decode it.
        """
        # Try to decode the codeword
        decoded = self.rm_code.decode(codeword)
        if decoded is None:
            return False
        
        # Re-encode and check if we get the same codeword
        re_encoded = self.rm_code.encode(decoded)
        return codeword == re_encoded
    
    def _matrix_inverse_mod2(self, matrix):
        """
        Calculate the inverse of a matrix modulo 2 using Gaussian elimination.
        
        Args:
            matrix: Square matrix to invert
            
        Returns:
            inverse: Inverse matrix modulo 2
        """
        n = matrix.shape[0]
        # Create augmented matrix [A | I]
        augmented = np.hstack([matrix, np.eye(n, dtype=int)])
        
        # Gaussian elimination mod 2
        for i in range(n):
            # Find pivot
            if augmented[i, i] == 0:
                # Look for a row to swap
                for j in range(i + 1, n):
                    if augmented[j, i] == 1:
                        augmented[[i, j]] = augmented[[j, i]]
                        break
                else:
                    raise ValueError("Matrix is not invertible")
            
            # Eliminate column
            for j in range(n):
                if j != i and augmented[j, i] == 1:
                    augmented[j] = (augmented[j] + augmented[i]) % 2
        
        # Extract inverse matrix
        inverse = augmented[:, n:]
        return inverse.astype(int)








def test_mceliece():
    """
    Test of the McEliece cryptosystem.
    """
    # Initialize McEliece
    mceliece = McEliece()
    # print(mceliece.bijection_function(44120, 10, 5))
    # Test single message
    char = "H"
    binary = "01001000"
    
    print(f"1. Message '{char}' in bytes (characters)")
    print(f"2. Message '{char}' in binary: {binary}")
    
    # Encrypt
    ciphertext = mceliece.encrypt(binary)
    print(f"3. Encoded message ciphertext as binary string: {ciphertext}")
    
    # Decrypt
    result = mceliece.decrypt(ciphertext)
    
    if result is not None:
        decoded_message, recovered_error = result
        print(f"4. Decoded message in binary: {decoded_message}")
        print(f"   Recovered error bitstring: {recovered_error}")
        
        # Convert back to character
        try:
            decoded_char = chr(int(decoded_message, 2))
            print(f"5. Decoded message in bytes (characters): '{decoded_char}'")
        except:
            print(f"5. Decoded message in bytes (characters): Invalid UTF-8")
    else:
        print("4. Decryption failed")
        print("5. No decoded message")
    print("Reencoded message: ", mceliece.encrypt(decoded_message, recovered_error))
    ciphertext_modified = list(ciphertext)
    for i in range(10):
        # flip a random bit 
        pos = random.randint(0, len(ciphertext_modified) - 1)
        ciphertext_modified[pos] = '1' if ciphertext_modified[pos] == '0' else '0'
    ciphertext_modified = ''.join(ciphertext_modified)
    message, error = mceliece.decrypt(ciphertext_modified)
    print("Decoded message: ", chr(int(message, 2)))
    print("Error: ", error)




def main():
    """
    Run the McEliece test suite.
    """
    test_mceliece()




if __name__ == "__main__":
    main()
