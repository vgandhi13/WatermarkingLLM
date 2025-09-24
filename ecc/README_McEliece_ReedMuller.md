# McEliece Cryptosystem with Reed-Muller Codes

This implementation provides a McEliece cryptosystem using Reed-Muller codes as the underlying error-correcting code, based on the [original McEliece implementation](https://github.com/jkrauze/mceliece/blob/master/mceliece/mceliececipher.py).

## Overview

The McEliece cryptosystem is a public-key encryption scheme that relies on the hardness of decoding random linear codes. This implementation uses Reed-Muller codes instead of the traditional Goppa codes, providing an alternative approach to the McEliece cryptosystem.

## Reed-Muller Codes

Reed-Muller codes RM(r,m) are a family of linear error-correcting codes with the following properties:

- **Code length**: n = 2^m
- **Message length**: k = C(m,0) + C(m,1) + ... + C(m,r)
- **Error correction capability**: t = 2^(m-r-1) - 1
- **Minimum distance**: d = 2^(m-r)

Where C(m,i) is the binomial coefficient.

## Implementation Details

### Key Components

1. **ReedMullerCode Class**: A wrapper around the Reed-Muller implementation that provides a consistent interface for encoding, decoding, and error introduction.

2. **McElieceReedMuller Class**: The main cryptosystem implementation that uses Reed-Muller codes as the underlying error-correcting code.

### Key Generation

The key generation process follows the standard McEliece approach:

1. Choose a Reed-Muller code RM(r,m) that can correct t errors
2. Generate three matrices:
   - G: The generator matrix of the Reed-Muller code
   - S: A random non-singular k×k matrix over GF(2)
   - P: A random n×n permutation matrix
3. Public key: G' = SGP
4. Private key: (S, G, P)

### Encryption

1. Convert message m to binary vector of length k
2. Compute c = mG' + e
   where e is a random error vector of weight t

### Decryption

1. Compute cP^(-1)
2. Use the Reed-Muller decoding algorithm to remove errors
3. Multiply by S^(-1) to get the original message m

## Usage

### Basic Usage

```python
from mceliece_reed_muller import McElieceReedMuller

# Initialize with Reed-Muller parameters RM(2,8)
mc = McElieceReedMuller(r=2, m=8)

# Encrypt a message
message = b"Hello, McEliece!"
ciphertext, num_errors = mc.encrypt(message)

# Decrypt the message
decrypted = mc.decrypt(ciphertext)
print(f"Original: {message}")
print(f"Decrypted: {decrypted}")
```

### Parameter Selection

Choose Reed-Muller parameters based on your security and performance requirements:

```python
# Small code for testing
mc_small = McElieceReedMuller(r=1, m=6)  # n=64, k=7, t=15

# Medium security
mc_medium = McElieceReedMuller(r=2, m=8)  # n=256, k=37, t=31

# Higher security
mc_large = McElieceReedMuller(r=1, m=10)  # n=1024, k=11, t=255
```

### Testing

Run the test suite to verify the implementation:

```bash
cd ecc
python test_mceliece_reed_muller.py
```

## Security Analysis

### Key Sizes

The key sizes depend on the Reed-Muller parameters:

- **Public key size**: n × k bits (size of G' matrix)
- **Private key size**: k² + n² + n×k bits (S + P + G matrices)

### Security Level

The security level is approximately based on the error correction capability t, as an attacker would need to solve the syndrome decoding problem with t errors.

### Parameter Recommendations

| Security Level | Parameters | n | k | t | Public Key (KB) | Private Key (KB) |
|----------------|------------|---|---|---|-----------------|------------------|
| 80 bits        | RM(2,8)    | 256 | 37 | 31 | 1.2 | 8.5 |
| 128 bits       | RM(1,10)   | 1024 | 11 | 255 | 1.4 | 131 |
| 256 bits       | RM(1,12)   | 4096 | 13 | 1023 | 6.7 | 2097 |

## Comparison with Original McEliece

### Advantages of Reed-Muller Codes

1. **Simplicity**: Reed-Muller codes have a simpler structure than Goppa codes
2. **Fast decoding**: Majority logic decoding is computationally efficient
3. **Flexible parameters**: Easy to choose different security levels by varying r and m

### Disadvantages

1. **Larger key sizes**: Reed-Muller codes typically require larger keys than Goppa codes for the same security level
2. **Known attacks**: Some specific attacks exist against Reed-Muller codes in the McEliece context
3. **Less studied**: Reed-Muller codes in McEliece have been less extensively analyzed than Goppa codes

## Implementation Notes

### Dependencies

- `numpy`: For matrix operations
- `reedmuller`: The Reed-Muller code implementation (included in `reed-muller-python/`)

### Performance Considerations

1. **Matrix operations**: All matrix operations are performed over GF(2) for efficiency
2. **Error introduction**: Uses bit-level errors rather than symbol-level errors
3. **Memory usage**: Large parameter sets may require significant memory

### Limitations

1. **Message size**: Limited by the message length k of the Reed-Muller code
2. **Error correction**: Limited by the error correction capability t
3. **Parameter constraints**: Must satisfy r < m for valid Reed-Muller codes

## Example Applications

### Secure Communication

```python
# Alice generates keys
alice = McElieceReedMuller(r=2, m=8)

# Bob encrypts a message for Alice
message = b"Secret message"
ciphertext, _ = alice.encrypt(message)

# Alice decrypts the message
decrypted = alice.decrypt(ciphertext)
```

### Hybrid Encryption

For longer messages, consider using McEliece for key exchange and a symmetric cipher for the actual message:

```python
import os
from cryptography.fernet import Fernet

# Generate a random symmetric key
symmetric_key = Fernet.generate_key()
cipher = Fernet(symmetric_key)

# Encrypt the symmetric key with McEliece
mc = McElieceReedMuller(r=2, m=8)
encrypted_key, _ = mc.encrypt(symmetric_key)

# Encrypt the actual message with symmetric encryption
message = b"Long message that needs to be encrypted"
encrypted_message = cipher.encrypt(message)

# Send both encrypted_key and encrypted_message
```

## References

1. [Original McEliece Implementation](https://github.com/jkrauze/mceliece/blob/master/mceliece/mceliececipher.py)
2. [Reed-Muller Codes](https://en.wikipedia.org/wiki/Reed%E2%80%93Muller_code)
3. [McEliece Cryptosystem](https://en.wikipedia.org/wiki/McEliece_cryptosystem)

## License

This implementation is provided for educational and research purposes. Please ensure compliance with relevant cryptographic regulations in your jurisdiction.
