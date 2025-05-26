from ecc.reed_solomon import ReedSolomonCode
import numpy as np

class PermutedReedSolomon:
    def __init__(self, n=255, k=223):
        self.rs = ReedSolomonCode(n, k)
        self.n = n
        self.k = k
        self.perm = None
        self.inv_perm = None
        
    def generate_key(self):
        self.perm = np.random.permutation(self.n)
        self.inv_perm = np.argsort(self.perm)
        return {'perm': self.perm, 'inv_perm': self.inv_perm}
    
    def set_key(self, key):
        self.perm = key['perm']
        self.inv_perm = key['inv_perm']
        
    def apply_permutation(self, data, perm):
        data_array = bytearray(data)
        permuted = bytearray(len(data_array))
        for i, p in enumerate(perm):
            if p < len(data_array):
                permuted[i] = data_array[p]
        return bytes(permuted)
        
    def encode(self, message):
        if self.perm is None:
            self.generate_key()
        encoded = self.rs.encode(message)
        return self.apply_permutation(encoded, self.perm)
        
    def decode(self, encoded_message):
        if self.inv_perm is None:
            raise ValueError("can't decode without key")
        depermuted = self.apply_permutation(encoded_message, self.inv_perm)
        return self.rs.decode(depermuted)
        
    def introduce_errors(self, encoded_message, num_errors):
        if self.perm is None:
            raise ValueError("can't introduce errors without key being set first")
        return self.rs.introduce_errors(encoded_message, num_errors) 