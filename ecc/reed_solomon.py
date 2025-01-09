from unireedsolomon import rs
import numpy as np

class ReedSolomonCode:
    def __init__(self, n=255, k=223):
        self.n = n  # Total codeword length
        self.k = k  # Message length
        self.coder = rs.RSCoder(n, k)
    
    def encode(self, message: bytes) -> bytes:
        encoded = self.coder.encode(message)
        if isinstance(encoded, str):
            return encoded.encode('latin1')
        return encoded
    
    def decode(self, encoded_message: bytes) -> str:
        try:
            if isinstance(encoded_message, bytes):
                encoded_message = encoded_message.decode('latin1')
            
            decoded_bytes = self.coder.decode(encoded_message)[0]
            
            if isinstance(decoded_bytes, bytes):
                return decoded_bytes.decode('utf-8')
            return decoded_bytes
        except Exception as e:
            print(f"Debug - Decode error: {str(e)}")
            raise ValueError("Unable to decode message - too many errors")
    
    def introduce_errors(self, encoded_message: bytes, num_errors: int) -> bytes:
        # randomly introduces errors
        message_array = bytearray(encoded_message)
        positions = np.random.choice(len(message_array), num_errors, replace=False)
        
        for pos in positions:
            # Modify the byte at the chosen position
            message_array[pos] = (message_array[pos] + np.random.randint(1, 255)) % 256
            
        return bytes(message_array) 