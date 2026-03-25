import numpy as np
import random
from collections import defaultdict

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
    def decode_codeword_to_message(self, candidate_codeword):
        # this function decodes the candidate codewrod to the message using the dictionary of all the codewords and messages
        if isinstance(candidate_codeword, str):
            candidate_codeword = np.array([int(bit) for bit in candidate_codeword], dtype=np.uint8)
        elif isinstance(candidate_codeword, list) and len(candidate_codeword) > 0 and isinstance(candidate_codeword[0], str):
            candidate_codeword = np.array([int(bit) for bit in candidate_codeword], dtype=np.uint8)
        else:
            candidate_codeword = np.array(candidate_codeword, dtype=np.uint8)
        # return self.decode(candidate_codeword)
        # Find the codeword with minimum Hamming distance
        min_distance = float('inf')
        best_message = None
        best_codeword = None
        # print("Candidate codeword: ", candidate_codeword)
        # Use standard Hamming distance calculation
        for stored_codeword, message in self.message_codeword_dict.items():
            # Calculate Hamming distance
            stored_codeword_array = np.array([int(bit) for bit in stored_codeword], dtype=np.uint8)
            distance = np.sum(candidate_codeword != stored_codeword_array)
            # if stored_codeword == np.array([int(bit) for bit in '11001101001101110110111110010011001000010011001111111010011001110101101000010100010001100101010001101001010011101000001000011110'], dtype=np.uint8):
            #     print(distance)
            if distance < min_distance:
                min_distance = distance
                best_message = message
                best_codeword = list(stored_codeword)
        # print("Final min distance: ", min_distance)
        return best_message, best_codeword, min_distance
    
    def decode_bit_map_to_message(self, decoded_index_bit_map):
        # this function decodes the decoded index bit map to some message and outputs the minimum distance as well, using the stored dictionary 
        for stored_codeword, message in self.message_codeword_dict.items():
                total_errors = 0
                total_bits = 0
                # print(f"Decoded index and bit map: {decoded_index_bit_map}")
                for index, bits in decoded_index_bit_map.items():
                    # print(f"Index: {index} and bits: {bits}")
                    # print(stored_codeword[index])
                    for bit in bits:
                        # print(bit)
                        if bit != stored_codeword[index]:
                            total_errors += 1
                            # print(f"Error at index {index} and bit {bit} and stored codeword bit {stored_codeword[index]}")
                        total_bits += 1
                average = total_errors / total_bits if total_bits > 0 else 0
                # print(f"Average error rate: {average}")
                min_distance = min(min_distance, average)
                # print(f"Min distance: {min_distance}")
                # print("min codeword: ", stored_codeword)
        return best_message, best_codeword, min_distance

    def decode_bit_map_to_fixed_codeword(self, decoded_index_bit_map, fixed_codeword):
        # this function decodes the decoded index bit map to the fixed codeword 
        min_distance = float('inf')
        best_message = None
        best_codeword = None
        if fixed_codeword is not None:
                    total_errors = 0
                    total_bits = 0
                    
                    for index, bits in decoded_index_bit_map.items():
                        for bit in bits:
                            # print(bit)
                            if bit != fixed_codeword[index]:
                                total_errors += 1
                                # print(f"Error at index {index} and bit {bit} and stored codeword bit {stored_codeword[index]}")
                            total_bits += 1
                    average = total_errors / total_bits if total_bits > 0 else 0
                    # print(f"Average error rate: {average}")
                    if average == 0 and min_distance == float('inf'):
                        min_distance = 0 
                    if average != 0:
                        min_distance = min(min_distance, average)
                    

        return best_message, best_codeword, min_distance

    
    
    def calculate_prediction_score(self, decoded_index_bit_map=None, fixed_codeword=None):
        # print("Decoding...")
        # print(f"Decoded index and bit map: {decoded_index_bit_map}")
        # Normalize candidate_codeword to np.uint8 array of bits
        # if isinstance(candidate_codeword, str):
        #     candidate_codeword = np.array([int(bit) for bit in candidate_codeword], dtype=np.uint8)
        # elif isinstance(candidate_codeword, list) and len(candidate_codeword) > 0 and isinstance(candidate_codeword[0], str):
        #     candidate_codeword = np.array([int(bit) for bit in candidate_codeword], dtype=np.uint8)
        # else:
        #     candidate_codeword = np.array(candidate_codeword, dtype=np.uint8)

        prediction_score1 = None 
        prediction_score2 = None
        # Find the codeword with minimum Hamming distance
        min_distance = float('inf')
        best_message = None
        best_codeword = None
        # print("Candidate codeword: ", candidate_codeword)
        # Use standard Hamming distance calculation
        # for stored_codeword, message in self.message_codeword_dict.items():
        #     # Calculate Hamming distance
        #     stored_codeword_array = np.array([int(bit) for bit in stored_codeword], dtype=np.uint8)
        #     distance = np.sum(candidate_codeword != stored_codeword_array)
        #     # if stored_codeword == np.array([int(bit) for bit in '11001101001101110110111110010011001000010011001111111010011001110101101000010100010001100101010001101001010011101000001000011110'], dtype=np.uint8):
        #     #     print(distance)
        #     if distance < min_distance:
        #         min_distance = distance
        #         best_message = message
        #         best_codeword = list(stored_codeword)
        # print("Final min distance: ", min_distance)
        min_distance = float('inf')
        if decoded_index_bit_map is not None:
            # Use decoded_index_bit_map for distance calculation
            for stored_codeword, message in self.message_codeword_dict.items():
                total_errors = 0
                total_bits = 0
                # print(f"Decoded index and bit map: {decoded_index_bit_map}")
                for index, bits in decoded_index_bit_map.items():
                    # print(f"Index: {index} and bits: {bits}")
                    # print(stored_codeword[index])
                    for bit in bits:
                        # print(bit)
                        if bit != stored_codeword[index]:
                            total_errors += 1
                            # print(f"Error at index {index} and bit {bit} and stored codeword bit {stored_codeword[index]}")
                        total_bits += 1
                average = total_errors / total_bits if total_bits > 0 else 0
                # print(f"Average error rate: {average}")
                min_distance = min(min_distance, average)
                # print("Total errors1: ", total_errors)
                # print("Total bits1: ", total_bits)
                observed_bits1 = total_bits - total_errors
                # print("Observed bits1: ", observed_bits1)
                prediction_score1 = observed_bits1 / total_bits if total_bits > 0 else 0
                # print("Prediction score1: ", prediction_score1)
            min_distance = float('inf')
            if fixed_codeword is not None:
                    total_errors = 0
                    total_bits = 0
                    
                    for index, bits in decoded_index_bit_map.items():
                        for bit in bits:
                            # print(bit)
                            if bit != fixed_codeword[index]:
                                total_errors += 1
                                # print(f"Error at index {index} and bit {bit} and stored codeword bit {stored_codeword[index]}")
                            total_bits += 1
                    average = total_errors / total_bits if total_bits > 0 else 0
                    # print(f"Average error rate: {average}")
                    if average == 0 and min_distance == float('inf'):
                        min_distance = 0 
                    if average != 0:
                        min_distance = min(min_distance, average)
                    # print("Total errors2: ", total_errors)
                    # print("Total bits2: ", total_bits)
                    observed_bits2 = total_bits - total_errors
                    # print("Observed bits2: ", observed_bits2)
                    prediction_score2 = observed_bits2 / total_bits if total_bits > 0 else 0
                    # print("Prediction score2: ", prediction_score2)
                    

        return prediction_score1, prediction_score2

    
    
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
    
    def print_pairwise_hamming_distances(self):
        """
        Calculates the minimum Hamming distance between codewords and counts
        how many codeword pairs are at this minimum distance.
        """
        # Get all codewords as a list
        codewords = list(self.message_codeword_dict.keys())
        num_codewords = len(codewords)
        
        if num_codewords < 2:
            print("Need at least 2 codewords to compute pairwise distances.")
            return
        
        # Convert codewords (tuples of characters) to numpy arrays for comparison
        codeword_arrays = []
        for codeword_tuple in codewords:
            # Convert tuple of characters to numpy array of integers
            codeword_array = np.array([int(bit) for bit in codeword_tuple], dtype=np.uint8)
            codeword_arrays.append(codeword_array)
        
        min_distance = float('inf')
        min_distance_count = 0
        
        print("Calculating minimum distance...")
        print("-" * 60)

        # Calculate distances and track minimum
        for i in range(num_codewords):
            for j in range(i + 1, num_codewords):
                codeword1 = codeword_arrays[i]
                codeword2 = codeword_arrays[j]
                
                # Calculate Hamming distance
                distance = np.sum(codeword1 != codeword2)
                
                if distance < min_distance:
                    min_distance = distance
                    min_distance_count = 1
                    print(f"Current minimum distance: {min_distance} (pairs at min: {min_distance_count})", flush=True)
                elif distance == min_distance:
                    min_distance_count += 1
                    print(f"Current minimum distance: {min_distance} (pairs at min: {min_distance_count})", flush=True)
        
        # Print results
        if min_distance != float('inf'):
            print(f"Minimum distance (d_min): {min_distance}")
            print(f"Number of codeword pairs at minimum distance: {min_distance_count}")
        else:
            print("No pairs found.")


# Example usage
if __name__ == "__main__":
    # Create a random (128,20) binary linear code
    code = RandomLinearCode(n=128, k=20, seed=42)
    
    # print(f"Code: {code}")
    # print(f"Generator matrix G shape: {code.G.shape}")
    
    # # Test string encoding with different lengths
    # test_strings = [
    #     "Hi",           # Short string (needs padding)
    #     "Hello World",  # Medium string (needs truncation)
    #     "A",            # Very short string (needs padding)
    #     "This is a very long string that will definitely need truncation to fit in 20 bits",  # Long string (needs truncation)
    #     "",             # Empty string (needs padding)
    # ]
    
    # print("\n=== Testing String Encoding ===")
    # for test_str in test_strings:
    #     print(f"\nOriginal string: '{test_str}'")
        
    #     # Convert to binary to show the process
    #     message_bytes = test_str.encode('utf-8')
    #     binary_string = ''.join([f'{byte:08b}' for byte in message_bytes])
    #     print(f"Binary representation: {binary_string} (length: {len(binary_string)})")
        
    #     # Encode using universal method
    #     codeword = code.encode(test_str)
    #     print(f"Codeword length: {len(codeword)}")
    #     print(f"Codeword: {codeword}")
    #     print(f"Codeword type: {type(codeword)}")
    
    # # Test with original binary message method
    # print("\n=== Testing Binary Message Encoding ===")
    message = [1]*20  # 20-bit message
    codeword = code.encode(message)
    # print(f"Message length: {len(message)}")
    # print(f"Codeword length: {len(codeword)}")
    # print(f"Message: {message}")
    # print(f"Codeword: {codeword}")
    # print(f"Codeword type: {type(codeword)}")
    
    corrupted_codeword = codeword[:2] + '1' + codeword[3:]
    print(f"Corrupted codeword: {corrupted_codeword}")
    # Test decoding
    # decoded_message, expected_codeword, distance = code.decode(corrupted_codeword)
    # print(f"Decoded message: {decoded_message}")
    # print(f"Expected codeword: {expected_codeword}")
    # print(f"Hamming distance: {distance}")
    # original_message_str = ''.join(str(b) for b in message)
    # print(f"Decoding successful: {original_message_str == decoded_message}")
    expected_codeword_str = '11001101001101110110111110010011001000010011001111111010011001110101101000010100010001100101010001101001010011101000001000011110'
    # # convert corrupted_codeword to a bit map
    # corrupted_bit_map = defaultdict(list)
    # for i in range(len(corrupted_codeword)):
    #     corrupted_bit_map[i].append(corrupted_codeword[i])
    # decoded_message, expected_codeword, distance = code.decode(corrupted_codeword, corrupted_bit_map)
    # print('DISTANCE: ', distance)
    # corrupted_array = np.array([int(bit) for bit in corrupted_codeword], dtype=np.uint8)
    # expected_array = np.array([int(bit) for bit in expected_codeword_str], dtype=np.uint8)
    # print(np.sum(corrupted_array != expected_array))


# need to debug the decode function second score 

    corrupted_bit_map = {111: ['1', '1', '1'], 118: ['1', '0'], 106: ['1', '1', '1', '1', '1', '1'], 64: ['0', '1', '1', '1', '1'], 114: ['1', '0', '0', '1', '0', '1', '0', '0', '1', '0', '0', '0', '0'], 38: ['1', '0'], 75: ['1', '1', '1'], 33: ['0'], 91: ['1', '0', '1', '1', '1'], 125: ['1', '1', '0', '1', '0', '1', '1', '1', '1', '0', '0'], 29: ['0', '1', '1', '1'], 107: ['0', '1', '0', '1', '1', '1', '1', '0', '1'], 86: ['0'], 48: ['0', '1', '1', '1'], 20: ['1', '0', '1', '1', '1', '1', '1', '0', '0', '0', '0'], 34: ['1', '0', '1', '0', '1'], 59: ['1', '1'], 8: ['0', '0', '1', '1'], 70: ['1'], 103: ['0', '0', '1'], 24: ['1', '1', '0'], 31: ['0', '1', '1', '0', '1', '1', '1'], 87: ['0', '0', '1', '0', '0', '1', '0'], 49: ['1', '0', '0', '1', '0', '0', '1', '0', '1', '0', '0', '1', '1', '1'], 74: ['1', '1', '1'], 54: ['0', '1', '0', '1', '0', '0', '1', '1', '1'], 68: ['1', '1', '0', '1', '0', '0', '0', '0'], 51: ['1', '0'], 14: ['0', '0', '1', '0'], 124: ['0', '1', '1', '1', '1'], 122: ['0', '1', '1', '1'], 65: ['1', '0', '1'], 25: ['0', '1', '1'], 22: ['0', '1', '0'], 28: ['0', '1', '1'], 3: ['1', '1', '1', '0', '1', '0', '1', '1', '0'], 121: ['0'], 97: ['1', '1', '0', '0'], 66: ['1', '1', '0', '1', '1', '1', '1', '0'], 9: ['0', '0', '1', '0', '0', '1'], 32: ['1'], 21: ['0'], 113: ['1', '0', '1', '0', '1', '0', '1', '1', '0', '1'], 95: ['1', '1', '0', '1', '1', '1', '1', '1'], 123: ['0', '0', '0', '1'], 76: ['1', '1', '0', '0', '1', '1', '0', '0', '0', '1', '0'], 120: ['1', '0', '0', '1', '1', '1'], 5: ['0', '1', '0', '1'], 101: ['0', '1', '0', '1', '0'], 10: ['0', '1', '1'], 119: ['1', '1', '0', '1', '1'], 27: ['0', '1', '1', '1', '1', '0', '1', '0'], 15: ['0'], 100: ['0', '1', '0', '1', '0', '1', '1', '0'], 88: ['1', '1'], 115: ['1', '1'], 62: ['1', '1', '0', '1', '0'], 94: ['1', '1', '0', '1', '0'], 61: ['1', '1', '0', '1', '0'], 127: ['1', '1'], 80: ['1', '1', '1', '0'], 43: ['1'], 35: ['0'], 18: ['1', '1', '0'], 78: ['0'], 19: ['1', '1', '1', '0', '1'], 98: ['1', '0', '1'], 104: ['0', '1'], 6: ['1', '1', '1', '1', '1', '1'], 117: ['1', '1'], 105: ['1', '1', '1', '1'], 11: ['1'], 42: ['0', '1', '0'], 40: ['1'], 110: ['0', '1', '0', '1'], 52: ['1', '1', '0', '0', '1', '1', '0'], 89: ['1', '0'], 126: ['0', '1'], 39: ['0', '1'], 90: ['0', '0', '1'], 81: ['0'], 2: ['0'], 0: ['0', '0'], 26: ['0'], 73: ['1'], 102: ['0', '1'], 36: ['1', '1', '1'], 53: ['1', '1', '1'], 79: ['1', '0', '1', '0'], 56: ['1'], 116: ['1', '1', '0'], 30: ['0'], 109: ['1', '1', '0'], 47: ['0'], 63: ['1', '1', '1'], 67: ['1', '1', '1', '0'], 57: ['1']}

    decoded_message, expected_codeword, distance = code.decode(corrupted_codeword, decoded_index_bit_map=corrupted_bit_map)
    print('DISTANCE: ', distance)





    