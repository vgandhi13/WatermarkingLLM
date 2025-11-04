import numpy as np
import random

class RandomLinearCode:
    def __init__(self, n, k, seed=None):
        self.n = n  # Code length
        self.k = k  # Code dimension
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Generate random generator matrix G of size k x n
        self.G = self._generate_random_generator_matrix()
        
        # Store all possible messages and their codewords
        self.message_codeword_dict = self._generate_all_codewords()
        
    def _generate_random_generator_matrix(self):
        # Binary field: generate random 0s and 1s
        G = np.random.randint(0, 2, size=(self.k, self.n), dtype=np.uint8)
        
        # Ensure the matrix has full rank (linearly independent rows)
        # If not, regenerate until we get a full-rank matrix
        while np.linalg.matrix_rank(G) < self.k:
            G = np.random.randint(0, 2, size=(self.k, self.n), dtype=np.uint8)
        
        return G
    
    def _generate_all_codewords(self):
        message_codeword_dict = {}
        
        # Generate all possible k-bit messages (2^k combinations)
        for i in range(2**self.k):
            # Convert integer to binary representation
            message = [(i >> j) & 1 for j in range(self.k-1, -1, -1)]
            # Encode the message
            codeword = self.encode(message)
            # Store as tuple for hashability
            message_codeword_dict[tuple(codeword)] = message
        
        return message_codeword_dict
    
    def encode(self, message):
        """
        Universal encode function that handles both string and binary array inputs.
        
        Args:
            message: Either a string or a list/array of binary values (0s and 1s)
            
        Returns:
            np.ndarray: The encoded codeword of length n
        """
        # Check input type
        if isinstance(message, str):
            # String input: convert to binary with padding/truncation
            message_bytes = message.encode('utf-8')
            binary_string = ''.join([f'{byte:08b}' for byte in message_bytes])
            
            # Handle length adjustment to exactly k bits
            if len(binary_string) > self.k:
                # Truncate to k bits (take first k bits)
                binary_string = binary_string[:self.k]
            elif len(binary_string) < self.k:
                # Pad with zeros to k bits (pad on the right)
                binary_string = binary_string.ljust(self.k, '0')
            
            # Convert binary string to list of integers
            message = [int(bit) for bit in binary_string]
        elif isinstance(message, list) and len(message) > 0 and isinstance(message[0], str):
            # List of strings (e.g., ['1', '0', '1']): convert to list of integers
            message = [int(bit) for bit in message]
        
        # Convert to numpy array and validate length
        message = np.array(message, dtype=np.uint8)
        if len(message) != self.k:
            raise ValueError(f"Message length must be {self.k}, got {len(message)}")
        
        # Encode: c = m * G (mod 2)
        codeword = np.dot(message, self.G) % 2
        
        # Convert to binary string
        binary_string = ''.join([str(bit) for bit in codeword])
        return binary_string
    
    def decode(self, candidate_codeword, decoded_index_bit_map=None):
        # Normalize candidate_codeword to np.uint8 array of bits
        if isinstance(candidate_codeword, str):
            candidate_codeword = np.array([int(bit) for bit in candidate_codeword], dtype=np.uint8)
        elif isinstance(candidate_codeword, list) and len(candidate_codeword) > 0 and isinstance(candidate_codeword[0], str):
            candidate_codeword = np.array([int(bit) for bit in candidate_codeword], dtype=np.uint8)
        else:
            candidate_codeword = np.array(candidate_codeword, dtype=np.uint8)

        # Find the codeword with minimum Hamming distance
        min_distance = float('inf')
        best_message = None
        best_codeword = None

        # Use standard Hamming distance calculation
        for stored_codeword, message in self.message_codeword_dict.items():
            # Calculate Hamming distance
            distance = np.sum(candidate_codeword != np.array(stored_codeword, dtype=np.uint8))
            
            if distance < min_distance:
                min_distance = distance
                best_message = message
                best_codeword = list(stored_codeword)
        min_distance = float('inf')
        if decoded_index_bit_map is not None:
            # Use decoded_index_bit_map for distance calculation
            for stored_codeword, message in self.message_codeword_dict.items():
                total_errors = 0
                total_bits = 0
                for index, bits in decoded_index_bit_map.items():
                    for bit in bits:
                        if bit != stored_codeword[index]:
                            total_errors += 1
                        total_bits += 1
                average = total_errors / total_bits if total_bits > 0 else 0
                
                min_distance = min(min_distance, average)
                

        
        
        # Convert recovered message from list of integers to binary string
        if best_message is not None:
            recovered_message_str = ''.join([str(bit) for bit in best_message])
        else:
            recovered_message_str = None
        
        return recovered_message_str, best_codeword, min_distance
    
    def get_generator_matrix(self):
        return self.G.copy()
    
    def get_code_parameters(self):
        return {
            'n': self.n,
            'k': self.k
        }
    
    def __str__(self):
        return f"RandomLinearCode(n={self.n}, k={self.k})"
    
    def __repr__(self):
        return self.__str__()


# Example usage
if __name__ == "__main__":
    # Create a random (128,20) binary linear code
    code = RandomLinearCode(n=128, k=20, seed=42)
    
    print(f"Code: {code}")
    print(f"Generator matrix G shape: {code.G.shape}")
    
    # Test string encoding with different lengths
    test_strings = [
        "Hi",           # Short string (needs padding)
        "Hello World",  # Medium string (needs truncation)
        "A",            # Very short string (needs padding)
        "This is a very long string that will definitely need truncation to fit in 20 bits",  # Long string (needs truncation)
        "",             # Empty string (needs padding)
    ]
    
    print("\n=== Testing String Encoding ===")
    for test_str in test_strings:
        print(f"\nOriginal string: '{test_str}'")
        
        # Convert to binary to show the process
        message_bytes = test_str.encode('utf-8')
        binary_string = ''.join([f'{byte:08b}' for byte in message_bytes])
        print(f"Binary representation: {binary_string} (length: {len(binary_string)})")
        
        # Encode using universal method
        codeword = code.encode(test_str)
        print(f"Codeword length: {len(codeword)}")
        print(f"Codeword: {codeword}")
        print(f"Codeword type: {type(codeword)}")
    
    # Test with original binary message method
    print("\n=== Testing Binary Message Encoding ===")
    message = [1]*20  # 20-bit message
    codeword = code.encode(message)
    print(f"Message length: {len(message)}")
    print(f"Codeword length: {len(codeword)}")
    print(f"Message: {message}")
    print(f"Codeword: {codeword}")
    print(f"Codeword type: {type(codeword)}")
    
    # Test decoding
    decoded_message, expected_codeword, distance = code.decode(codeword)
    print(f"Decoded message: {decoded_message}")
    print(f"Expected codeword: {expected_codeword}")
    print(f"Hamming distance: {distance}")
    original_message_str = ''.join(str(b) for b in message)
    print(f"Decoding successful: {original_message_str == decoded_message}")

    avg_distance = 0
    keys_list = list(code.message_codeword_dict.keys())
    for i in range(len(keys_list) - 1):
        for j in range(i+1, len(keys_list)):
            
            stored_codeword1 = list(keys_list[i])
            stored_codeword2 = list(keys_list[j])
            distance = np.sum(stored_codeword1 != stored_codeword2)
            avg_distance += distance
    avg_distance /= len(code.message_codeword_dict) * (len(code.message_codeword_dict) - 1) / 2
    print(f"Average Hamming distance between codewords: {avg_distance}")

