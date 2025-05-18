# seed = aes(key=key, ctr=ctr)
# ciphertext = prg(seed, length = length)
import os 
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.algorithms import ChaCha20

def bytes_to_bits(byte_data):
    """Convert bytes to a string of bits."""
    return ''.join(format(byte, '08b') for byte in byte_data)

def prg(seed, length):
    """
    Implement PRG using ChaCha20.
    seed: bytes object used as key (16 bytes)
    length: desired output length in bits
    """
    # ChaCha20 needs a 32-byte key, so we'll double our 16-byte seed
    chacha_key = seed + seed  # Simple way to expand from 16 to 32 bytes
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
        if mode=="normal":
            self.ctr = 0
            self.key = KEY # AES-128 key
        else:
            raise ValueError("Not implemented yet lol")

    def get_key(self):
        """Return the 128-bit AES key as a bytes object."""
        return self.key

    def aes_block(self, counter):
        """Generate one AES block from counter."""
        cipher = Cipher(algorithms.AES(self.key), modes.ECB())
        encryptor = cipher.encryptor()
        # Convert counter to 16 bytes
        counter_bytes = counter.to_bytes(16, byteorder='big')
        block_bytes = encryptor.update(counter_bytes) + encryptor.finalize()
        # Convert to bits - AES block is always 128 bits
        return bytes_to_bits(block_bytes)

    def encrypt(self, length, message=None):
        self.ctr += 1
        if mode=="normal":
            # Generate seed using AES in counter mode - returns 128 bits
            seed_bits = self.aes_block(self.ctr)
            # Convert bits back to bytes for PRG
            seed_bytes = int(seed_bits, 2).to_bytes(16, byteorder='big')
            # Use PRG to expand the seed to desired length
            return prg(seed_bytes, length)
        
    def decrypt(self, ciphertext):
        pass
        #only needed when mode == "encrypt")

if __name__ == "__main__":
    ciphertext = Ciphertext()
    print(ciphertext.encrypt(100))
    print(ciphertext.encrypt(100))
    print("avg 0s and 1s")
    
    for i in range(1, 200):
        total_0, total_1 = 0, 0
        print(ciphertext.encrypt(i))
        for j in range(len(ciphertext.encrypt(i))):
            if ciphertext.encrypt(i)[j] == '0':
                total_0 += 1
            else:
                total_1 += 1
        if total_0/len(ciphertext.encrypt(i)) < 0.51 and total_0/len(ciphertext.encrypt(i)) > 0.49:
            print(i)
            print(f"Average 0s: {total_0/len(ciphertext.encrypt(i))}, Average 1s: {total_1/len(ciphertext.encrypt(i))}")

