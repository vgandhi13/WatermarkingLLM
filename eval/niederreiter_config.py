"""
Niederreiter Configuration for Watermarking

This module provides a drop-in replacement for McEliece using Niederreiter
cryptosystem with Reed-Muller codes for 16-bit message watermarking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecc.niederreiter_watermark import NiederreiterCiphertextWrapper
from typing import List, Dict, Any, Tuple

class NiederreiterConfig:
    """Configuration for Niederreiter with 16-bit message support."""
    
    # Reed-Muller parameters that support 16-bit messages
    SUPPORTED_RM_PARAMS = {
        # Ultra-high rate options (k/n > 0.8)
        'RM(4,4)': {'r': 4, 'm': 4, 'k': 16, 'n': 16, 't': -1, 'rate': 1.000, 'category': 'Ultra-High'},
        'RM(4,5)': {'r': 4, 'm': 5, 'k': 31, 'n': 32, 't': 0, 'rate': 0.969, 'category': 'Ultra-High'},
        'RM(4,6)': {'r': 4, 'm': 6, 'k': 57, 'n': 64, 't': 1, 'rate': 0.891, 'category': 'Ultra-High'},
        'RM(3,5)': {'r': 3, 'm': 5, 'k': 26, 'n': 32, 't': 1, 'rate': 0.812, 'category': 'Ultra-High'},
        
        # Very high rate options (k/n > 0.6)
        'RM(4,7)': {'r': 4, 'm': 7, 'k': 99, 'n': 128, 't': 3, 'rate': 0.773, 'category': 'Very-High'},
        'RM(3,6)': {'r': 3, 'm': 6, 'k': 42, 'n': 64, 't': 3, 'rate': 0.656, 'category': 'Very-High'},
        
        # High rate options (k/n > 0.3)
        'RM(2,5)': {'r': 2, 'm': 5, 'k': 16, 'n': 32, 't': 3, 'rate': 0.500, 'category': 'High'},
        'RM(3,7)': {'r': 3, 'm': 7, 'k': 64, 'n': 128, 't': 7, 'rate': 0.500, 'category': 'High'},
        'RM(2,6)': {'r': 2, 'm': 6, 'k': 22, 'n': 64, 't': 7, 'rate': 0.344, 'category': 'High'},
    }
    
    # Recommended configurations for 16-bit messages
    RECOMMENDED_CONFIGS = {
        'maximum_efficiency': {
            'rm_params': 'RM(4,4)',
            'message_length': 16,
            'description': 'Perfect 100% rate, no error correction'
        },
        'ultra_high_rate': {
            'rm_params': 'RM(4,5)',
            'message_length': 16,
            'description': '96.9% rate, minimal error correction'
        },
        'very_high_rate': {
            'rm_params': 'RM(4,6)',
            'message_length': 16,
            'description': '89.1% rate, some error correction'
        },
        'balanced_high': {
            'rm_params': 'RM(3,5)',
            'message_length': 16,
            'description': '81.2% rate, good balance'
        },
        'high_with_correction': {
            'rm_params': 'RM(4,7)',
            'message_length': 16,
            'description': '77.3% rate, decent error correction'
        },
        'classic_high': {
            'rm_params': 'RM(2,5)',
            'message_length': 16,
            'description': '50.0% rate, proven option'
        }
    }
    
    # Default recommended configuration
    RECOMMENDED_CONFIG = RECOMMENDED_CONFIGS['balanced_high']
    
    @classmethod
    def get_rm_params(cls, config_name: str = 'RM(3,5)') -> Dict[str, int]:
        """Get Reed-Muller parameters for 16-bit message support."""
        if config_name not in cls.SUPPORTED_RM_PARAMS:
            raise ValueError(f"Unsupported RM configuration: {config_name}")
        return cls.SUPPORTED_RM_PARAMS[config_name]
    
    @classmethod
    def list_niederreiter_options(cls) -> None:
        """List all available Niederreiter configurations."""
        print("Niederreiter Reed-Muller Options for 16-bit Messages")
        print("=" * 80)
        print(f"{'Config':<10} {'k':<4} {'n':<6} {'t':<4} {'Rate':<8} {'Category':<12} {'Description'}")
        print("-" * 80)
        
        # Sort by rate (efficiency) - higher is better
        sorted_configs = sorted(cls.SUPPORTED_RM_PARAMS.items(), 
                              key=lambda x: x[1]['rate'], reverse=True)
        
        for config_name, params in sorted_configs:
            description = f"Supports {params['k']}-bit messages"
            if params['k'] == 16:
                description += " (perfect fit)"
            elif params['k'] > 16:
                description += f" (can handle 16-bit)"
            
            print(f"{config_name:<10} {params['k']:<4} {params['n']:<6} {params['t']:<4} "
                  f"{params['rate']:<8.3f} {params['category']:<12} {description}")
        
        print("\nRecommended Configurations:")
        print("-" * 40)
        for config_name, config in cls.RECOMMENDED_CONFIGS.items():
            rm_params = config['rm_params']
            params = cls.SUPPORTED_RM_PARAMS[rm_params]
            print(f"{config_name}: {rm_params} - {config['description']}")
            print(f"  Rate: {params['rate']:.3f} ({params['rate']*100:.1f}%)")
            print(f"  Code length: {params['n']} bits, Error correction: {params['t']} bits")
            print()
    
    @classmethod
    def create_niederreiter_ciphertext(cls, rm_config: str = 'RM(3,5)') -> NiederreiterCiphertextWrapper:
        """Create a Niederreiter Ciphertext instance for 16-bit messages."""
        return NiederreiterCiphertextWrapper(rm_config)

# Drop-in replacement for McEliece
class NiederreiterMcElieceReplacement:
    """
    Drop-in replacement for McEliece using Niederreiter.
    
    This class provides the same interface as McEliece but uses
    Niederreiter cryptosystem internally.
    """
    
    def __init__(self, r: int = 2, m: int = 5):
        """
        Initialize Niederreiter as McEliece replacement.
        
        Args:
            r: Order of the Reed-Muller code
            m: Number of variables
        """
        # Map to RM config string
        rm_config = f'RM({r},{m})'
        
        if rm_config not in NiederreiterConfig.SUPPORTED_RM_PARAMS:
            raise ValueError(f"Unsupported RM configuration: {rm_config}")
        
        self.niederreiter = NiederreiterCiphertextWrapper(rm_config)
        self.rm_config = rm_config
        self.r = r
        self.m = m
        
        # Expose McEliece-compatible attributes
        params = NiederreiterConfig.get_rm_params(rm_config)
        self.k = params['k']
        self.n = params['n']
        self.t = params['t']
        
        print(f"Niederreiter (McEliece replacement) RM({r},{m}) initialized:")
        print(f"  Message length (k): {self.k} bits")
        print(f"  Code length (n): {self.n} bits")
        print(f"  Error correction capability (t): {self.t}")
        print(f"  Code rate: {self.k/self.n:.3f}")
    
    def encrypt(self, message: bytes) -> Tuple[bytes, int]:
        """
        Encrypt a message (McEliece-compatible interface).
        
        Args:
            message: Message to encrypt
            
        Returns:
            Tuple of (encrypted_codeword, number_of_errors)
        """
        # Convert bytes to string for Niederreiter
        message_str = message.decode('utf-8', errors='ignore')
        
        # Encrypt using Niederreiter
        encrypted_str = self.niederreiter.encrypt(message_str)
        
        # Convert back to bytes
        encrypted_bytes = bytes([int(encrypted_str[i:i+8], 2) 
                               for i in range(0, len(encrypted_str), 8)])
        
        return encrypted_bytes, self.t
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt a ciphertext (McEliece-compatible interface).
        
        Args:
            ciphertext: Encrypted ciphertext
            
        Returns:
            Decrypted message
        """
        # Convert bytes to binary string
        ciphertext_str = ''.join(format(byte, '08b') for byte in ciphertext)
        
        # Decrypt using Niederreiter
        decrypted_str = self.niederreiter.decrypt(ciphertext_str)
        
        # Convert back to bytes
        return decrypted_str.encode('utf-8')

def test_niederreiter_config():
    """Test Niederreiter configuration."""
    print("Testing Niederreiter Configuration")
    print("=" * 50)
    
    # List available options
    NiederreiterConfig.list_niederreiter_options()
    
    # Test different configurations
    test_configs = [
        'RM(2,5)',  # Classic high rate
        'RM(3,5)',  # Balanced high rate
        'RM(4,5)',  # Ultra high rate
    ]
    
    print("\nTesting Niederreiter Configurations:")
    print("-" * 50)
    
    for rm_config in test_configs:
        print(f"\nTesting {rm_config}:")
        try:
            ciphertext = NiederreiterConfig.create_niederreiter_ciphertext(rm_config)
            
            # Test encryption/decryption
            test_message = "Hi"  # 2 bytes = 16 bits
            encrypted = ciphertext.encrypt(test_message)
            decrypted = ciphertext.decrypt(encrypted)
            
            print(f"  Original: '{test_message}'")
            print(f"  Encrypted: {encrypted[:32]}...")
            print(f"  Decrypted: '{decrypted}'")
            print(f"  Success: {test_message == decrypted}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test McEliece replacement
    print(f"\nTesting McEliece Replacement:")
    try:
        mc_replacement = NiederreiterMcElieceReplacement(2, 5)
        
        test_message = b"Hi"
        encrypted, num_errors = mc_replacement.encrypt(test_message)
        decrypted = mc_replacement.decrypt(encrypted)
        
        print(f"  Original: {test_message}")
        print(f"  Encrypted: {encrypted}")
        print(f"  Decrypted: {decrypted}")
        print(f"  Success: {test_message == decrypted}")
        
    except Exception as e:
        print(f"  Error: {e}")

def show_usage_examples():
    """Show usage examples for replacing McEliece with Niederreiter."""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES: Replacing McEliece with Niederreiter")
    print("=" * 60)
    
    print("1. Direct replacement in existing code:")
    print("   # Old McEliece code:")
    print("   from ecc.mceliece import McEliece")
    print("   mc = McEliece()")
    print("   ")
    print("   # New Niederreiter code:")
    print("   from eval.niederreiter_config import NiederreiterMcElieceReplacement")
    print("   mc = NiederreiterMcElieceReplacement()")
    print()
    
    print("2. Using specific Reed-Muller configurations:")
    print("   # High-rate configuration:")
    print("   mc = NiederreiterMcElieceReplacement(3, 5)  # RM(3,5)")
    print("   ")
    print("   # Ultra-high-rate configuration:")
    print("   mc = NiederreiterMcElieceReplacement(4, 5)  # RM(4,5)")
    print()
    
    print("3. Using the configuration system:")
    print("   from eval.niederreiter_config import NiederreiterConfig")
    print("   ciphertext = NiederreiterConfig.create_niederreiter_ciphertext('RM(3,5)')")
    print()
    
    print("4. In your watermarking system:")
    print("   # Change this line in your config:")
    print("   CRYPTO_SCHEME = 'Niederreiter'  # instead of 'McEliece'")
    print("   ")
    print("   # And import the replacement:")
    print("   from eval.niederreiter_config import NiederreiterMcElieceReplacement as McEliece")

if __name__ == "__main__":
    test_niederreiter_config()
    show_usage_examples()
