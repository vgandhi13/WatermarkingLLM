# seed = aes(key=key, ctr=ctr)
# ciphertext = prg(seed, length = length)
import os 
import sys
import random
# Add the reed-muller-python directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'reed-muller-python'))

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.algorithms import ChaCha20
# Import the McEliece with Reed-Muller implementation
sys.path.append(os.path.dirname(__file__))
from mceliece_reed_muller import McElieceReedMuller

def bytes_to_bits(byte_data):
    """Convert bytes to a string of bits."""
    if isinstance(byte_data, bytes):
        return ''.join(format(byte, '08b') for byte in byte_data)
    else:
        # Handle other types by converting to bytes first
        try:
            if isinstance(byte_data, str):
                byte_data = byte_data.encode('utf-8')
            elif isinstance(byte_data, list):
                # Convert list of bits to bytes
                bitstring = ''.join(str(bit) for bit in byte_data)
                byte_data = bytearray()
                for i in range(0, len(bitstring), 8):
                    byte_str = bitstring[i:i+8]
                    if len(byte_str) == 8:
                        byte_val = int(byte_str, 2)
                        byte_data.append(byte_val)
                    else:
                        # Pad with zeros if incomplete byte
                        byte_str = byte_str.ljust(8, '0')
                        byte_val = int(byte_str, 2)
                        byte_data.append(byte_val)
                byte_data = bytes(byte_data)
            
            return ''.join(format(byte, '08b') for byte in byte_data)
        except Exception as e:
            print(f"Error converting to bits: {e}")
            return None

def prg(seed, length):
    """
    Implement PRG using ChaCha20.
    seed: bytes object used as key (16 bytes)
    length: desired output length in bits
    """
    # ChaCha20 needs a 32-byte key, so we'll double our 16-byte seed
    chacha_key = seed + seed
    nonce = b'\x00' * 16
    
    # Create ChaCha20 cipher
    algorithm = ChaCha20(chacha_key, nonce)
    cipher = Cipher(algorithm, mode=None)
    encryptor = cipher.encryptor()
    
    # Generate required number of bytes (rounding up)
    num_bytes = (length + 7) // 8  # Convert bits to bytes, rounding up
    # Generate random bytes by encrypting a zero message
    random_bytes = encryptor.update(b'\x00' * num_bytes)
    
    # Convert to bits and truncate to exact length
    bits = bytes_to_bits(random_bytes)
    return bits[:length]

embedding_modes = ["normal", "encrypt", "prompt_correlated"]
# BEGIN OPTIONS TO TEST
mode = embedding_modes[0]
KEY = b'=\x0fs\xf1q\xccQ\x9fhi\xa7\x89\x8f\xc5#\xbf'

class Ciphertext:
    def __init__(self):
        # Initialize McEliece with Reed-Muller (RM(1,7))
        self.mceliece = McElieceReedMuller(r=1, m=7)
        print(f"McEliece RM(1,7) initialized:")
        print(f"  Message length (k): {self.mceliece.k} bits")
        print(f"  Code length (n): {self.mceliece.n} bits")
        print(f"  Error correction capability (t): {self.mceliece.t}")

    def encrypt(self, message):
        """
        Encrypt a message using McEliece with Reed-Muller codes.
        
        Args:
            message: String message to encrypt
            
        Returns:
            str: Binary string representation of the encrypted ciphertext
        """
        try:
            # Convert string message to bytes
            if isinstance(message, str):
                message_bytes = message.encode('utf-8')
            else:
                message_bytes = message
            
            # Check if message is too long
            if len(message_bytes) * 8 > self.mceliece.k:
                print(f"Warning: Message too long ({len(message_bytes)} bytes, {len(message_bytes) * 8} bits). Max is {self.mceliece.k} bits.")
                # Truncate message if too long
                max_bytes = self.mceliece.k // 8
                message_bytes = message_bytes[:max_bytes]
            
            # Encrypt using McEliece (returns tuple: (ciphertext, num_errors))
            encrypted_result = self.mceliece.encrypt(message_bytes)
            
            # Extract just the ciphertext from the tuple
            if isinstance(encrypted_result, tuple):
                encrypted_bytes = encrypted_result[0]
            else:
                encrypted_bytes = encrypted_result
            
            # Convert encrypted bytes to binary string
            binary_string = bytes_to_bits(encrypted_bytes)
            
            return binary_string
            
        except Exception as e:
            print(f"Encryption error: {e}")
            return None
        
    def decrypt(self, ciphertext):
        """
        Decrypt a ciphertext using McEliece with Reed-Muller codes.
        
        Args:
            ciphertext: Binary string or list of bits representing the ciphertext
            
        Returns:
            str: Decrypted message
        """
        try:
            # Convert string ciphertext to bytes if needed
            if isinstance(ciphertext, str):
                # Convert binary string to bytes
                ciphertext_bytes = bytearray()
                for i in range(0, len(ciphertext), 8):
                    byte_str = ciphertext[i:i+8]
                    if len(byte_str) == 8:
                        byte_val = int(byte_str, 2)
                        ciphertext_bytes.append(byte_val)
                    else:
                        # Pad with zeros if incomplete byte
                        byte_str = byte_str.ljust(8, '0')
                        byte_val = int(byte_str, 2)
                        ciphertext_bytes.append(byte_val)
                ciphertext_bytes = bytes(ciphertext_bytes)
            else:
                # Assume it's already bytes
                ciphertext_bytes = ciphertext
            
            # Ensure the ciphertext length matches the expected length
            expected_bytes = self.mceliece.n // 8
            if len(ciphertext_bytes) != expected_bytes:
                print(f"Warning: ciphertext length {len(ciphertext_bytes)} bytes doesn't match expected length {expected_bytes} bytes")
                # Pad or truncate to match expected length
                if len(ciphertext_bytes) < expected_bytes:
                    ciphertext_bytes = ciphertext_bytes + b'\x00' * (expected_bytes - len(ciphertext_bytes))
                else:
                    ciphertext_bytes = ciphertext_bytes[:expected_bytes]
            
            # Decrypt using McEliece
            decrypted_bytes = self.mceliece.decrypt(ciphertext_bytes)
            
            if decrypted_bytes is None:
                print("Decryption failed - unable to decode")
                return None
            
            # Convert decrypted bytes to string
            try:
                decrypted_text = decrypted_bytes.decode('utf-8')
                # Remove null characters and other non-printable characters
                decrypted_text = ''.join(char for char in decrypted_text if char.isprintable())
                return decrypted_text
            except UnicodeDecodeError:
                print("Warning: Could not decode as UTF-8, returning raw bytes")
                return str(decrypted_bytes)
                
        except Exception as e:
            print(f"Decryption error: {e}")
            return None

class Mceliece:
    def __init__(self):
        # Initialize McEliece with Reed-Muller (RM(1,7))
        self.mceliece = McElieceReedMuller(r=1, m=7)
        print(f"McEliece RM(1,7) initialized:")
        print(f"  Message length (k): {self.mceliece.k} bits")
        print(f"  Code length (n): {self.mceliece.n} bits")
        print(f"  Error correction capability (t): {self.mceliece.t}")
    
    def encrypt(self, message):
        """
        Encrypt a message using McEliece with Reed-Muller codes.
        
        Args:
            message: Message to encrypt (string, bytes, or list of bits)
            
        Returns:
            bytes: Encrypted ciphertext
        """
        try:
            # Convert message to bytes if it's a string
            if isinstance(message, str):
                message_bytes = message.encode('utf-8')
            elif isinstance(message, list):
                # Convert list of bits to bytes
                bitstring = ''.join(str(bit) for bit in message)
                message_bytes = bytearray()
                for i in range(0, len(bitstring), 8):
                    byte_str = bitstring[i:i+8]
                    if len(byte_str) == 8:
                        byte_val = int(byte_str, 2)
                        message_bytes.append(byte_val)
                    else:
                        # Pad with zeros if incomplete byte
                        byte_str = byte_str.ljust(8, '0')
                        byte_val = int(byte_str, 2)
                        message_bytes.append(byte_val)
                message_bytes = bytes(message_bytes)
            else:
                message_bytes = message
            
            # Check if message is too long
            if len(message_bytes) * 8 > self.mceliece.k:
                print(f"Warning: Message too long ({len(message_bytes)} bytes, {len(message_bytes) * 8} bits). Max is {self.mceliece.k} bits.")
                # Truncate message if too long
                max_bytes = self.mceliece.k // 8
                message_bytes = message_bytes[:max_bytes]
            
            # Encrypt using McEliece (returns tuple: (ciphertext, num_errors))
            encrypted_result = self.mceliece.encrypt(message_bytes)
            
            # Extract just the ciphertext from the tuple
            if isinstance(encrypted_result, tuple):
                return encrypted_result[0]
            else:
                return encrypted_result
            
        except Exception as e:
            print(f"Encryption error: {e}")
            return None
    
    def decrypt(self, ciphertext):
        """
        Decrypt a ciphertext using McEliece with Reed-Muller codes.
        
        Args:
            ciphertext: Ciphertext to decrypt (bytes or list of bits)
            
        Returns:
            bytes: Decrypted message
        """
        try:
            # Convert list of bits to bytes if needed
            if isinstance(ciphertext, list):
                bitstring = ''.join(str(bit) for bit in ciphertext)
                ciphertext_bytes = bytearray()
                for i in range(0, len(bitstring), 8):
                    byte_str = bitstring[i:i+8]
                    if len(byte_str) == 8:
                        byte_val = int(byte_str, 2)
                        ciphertext_bytes.append(byte_val)
                    else:
                        # Pad with zeros if incomplete byte
                        byte_str = byte_str.ljust(8, '0')
                        byte_val = int(byte_str, 2)
                        ciphertext_bytes.append(byte_val)
                ciphertext_bytes = bytes(ciphertext_bytes)
            else:
                ciphertext_bytes = ciphertext
            
            # Ensure the ciphertext length matches the expected length
            expected_bytes = self.mceliece.n // 8
            if len(ciphertext_bytes) != expected_bytes:
                print(f"Warning: ciphertext length {len(ciphertext_bytes)} bytes doesn't match expected length {expected_bytes} bytes")
                # Pad or truncate to match expected length
                if len(ciphertext_bytes) < expected_bytes:
                    ciphertext_bytes = ciphertext_bytes + b'\x00' * (expected_bytes - len(ciphertext_bytes))
                else:
                    ciphertext_bytes = ciphertext_bytes[:expected_bytes]
            
            # Decrypt using McEliece
            return self.mceliece.decrypt(ciphertext_bytes)
            
        except Exception as e:
            print(f"Decryption error: {e}")
            return None

if __name__ == "__main__":
    print("Testing McEliece with Reed-Muller implementation")
    print("="*60)
    
    # Test the Ciphertext class
    print("\n1. Testing Ciphertext class:")
    ciphertext = Ciphertext()
    
    test_message = "H"
    print(f"\nEncrypting: '{test_message}'")
    encrypted = ciphertext.encrypt(test_message)
    print(f"Encrypted (binary): {encrypted}")
    print(f"Encrypted length: {len(encrypted)} bits")
    
    print(f"\nDecrypting...")
    decrypted = ciphertext.decrypt(encrypted)
    print(f"Decrypted: '{decrypted}'")
    
    # Test the Mceliece class
    print("\n2. Testing Mceliece class:")
    mceliece = Mceliece()
    
    test_message = "H"
    print(f"\nEncrypting: '{test_message}'")
    encrypted_bytes = mceliece.encrypt(test_message)
    print(f"Encrypted (bytes): {encrypted_bytes}")
    print(f"Encrypted length: {len(encrypted_bytes)} bytes")
    
    print(f"\nDecrypting...")
    decrypted_bytes = mceliece.decrypt(encrypted_bytes)
    print(f"Decrypted (bytes): {decrypted_bytes}")
    if decrypted_bytes:
        decrypted_text = decrypted_bytes.decode('utf-8')
        print(f"Decrypted (text): '{decrypted_text}'")

