# Key Generation:
# 1. Choose a binary Goppa code that can correct t errors
# 2. Generate three matrices:
#    - G (the generator matrix of the code)
#    - S (a random non-singular matrix)
#    - P (a random permutation matrix)
# 3. Public key: G' = SGP
# 4. Private key: (S, G, P)

# Encryption:
# 1. Convert message m to binary
# 2. Compute c = mG' + e
#    where e is a random error vector of weight t

# Decryption:
# 1. Compute cP^(-1)
# 2. Use the decoding algorithm for G to remove errors
# 3. Multiply by S^(-1) to get m

# Source: https://github.com/joseph-garcia/mceliece-python
# Source: https://stackoverflow.com/questions/42274010/mceliece-encryption-decryption-algorithm




import numpy as np
from ecc.reed_solomon import ReedSolomonCode

class McEliece:
    def __init__(self, n=255, k=223, type="ReedSolomon"):
        if type == "ReedSolomon":
            self.rs = ReedSolomonCode(n, k)
        else:
            raise ValueError("Invalid type. Choose from one of the available types.")
        self.n = n
        self.k = k
        self.generate_keys()
    
    def generate_keys(self):
        # Generate random binary matrices
        self.S = np.random.randint(0, 2, (self.k, self.k))
        self.P = np.random.permutation(np.eye(self.n))
        
        # Make sure S is invertible
        while np.linalg.matrix_rank(self.S) != self.k:
            self.S = np.random.randint(0, 2, (self.k, self.k))
        
        self.S_inv = np.linalg.inv(self.S)
        self.P_inv = np.linalg.inv(self.P)
    
    def encrypt(self, message: bytes, t=10):
        if len(message) > self.k:
            raise ValueError(f"Message too long. Max length is {self.k} bytes")
        
        # Pad message if needed
        if len(message) < self.k:
            message = message + b'\0' * (self.k - len(message))
        
        # Convert message to bit array and encode
        m = np.frombuffer(message, dtype=np.uint8)
        encoded = self.rs.encode(message)
        
        # Add t random errors using RS error introduction
        corrupted = self.rs.introduce_errors(encoded, t)
        
        return corrupted, t

    def decrypt(self, ciphertext: bytes):
        try:
            # Use RS decoder to correct errors and get original message
            decoded = self.rs.decode(ciphertext)
            # Convert back to bytes and strip padding
            if isinstance(decoded, str):
                decoded = decoded.encode('utf-8')
            return decoded.rstrip(b'\0')
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
def print_binary(data: bytes, label: str):
    print("\n" + label + ":")
    print("Bytes: " + str(data))
    print("Binary: " + " ".join([format(b, '08b') for b in data[:8]]) + "...")  # First 8 bytes in binary
    print("Length: " + str(len(data)) + " bytes")

def test_mceliece():
    mc = McEliece()
    
    messages = [
        b"Hello world, PLS WORK!",
        b"this is a test message pls work pls",
        b"short msg",
        b" @##4135135 plswork pretty pls long message" * 5
    ]
    
    error_counts = [5, 10, 15]
    
    for msg in messages:
        print("\n" + "="*50)
        print_binary(msg, "Original Message")
        
        if len(msg) > mc.k:
            print("Skipping - message too long (" + str(len(msg)) + " > " + str(mc.k) + ")")
            continue
            
        for t in error_counts:
            print("\nTesting with " + str(t) + " errors:")
            try:
                # Encrypt
                cipher, num_errors = mc.encrypt(msg, t)
                print_binary(cipher, "Encrypted Message")
                
                corrupted = mc.rs.introduce_errors(cipher, t)
                print_binary(corrupted, "Corrupted Message (with " + str(t) + " errors)")
                

                decrypted = mc.decrypt(corrupted)
                print_binary(decrypted, "Decrypted Message")
                print("\nDecryption Success: " + str(msg == decrypted))
                
            except Exception as e:
                print("Error: " + str(e))

if __name__ == "__main__":
    test_mceliece()