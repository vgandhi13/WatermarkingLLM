"""
Niederreiter 16-bit Message Configuration

This module provides configuration and utilities for using Niederreiter cryptosystem
with 16-bit message length support, as an alternative to McEliece.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecc.niederreiter_simple import NiederreiterSimple, NiederreiterCiphertext
from typing import List, Dict, Any

class Niederreiter16BitConfig:
    """Configuration for Niederreiter with 16-bit message support."""
    
    # Reed-Muller parameters that support 16-bit messages
    # Sorted by efficiency (k/n ratio) - higher is better
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
    
    # Recommended configurations for 16-bit messages by use case
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
    
    # Default recommended configuration (balanced approach)
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
    def create_niederreiter_ciphertext(cls, rm_config: str = 'RM(3,5)') -> 'NiederreiterCiphertext16Bit':
        """Create a Niederreiter Ciphertext instance for 16-bit messages."""
        return NiederreiterCiphertext16Bit(rm_config)

class NiederreiterCiphertext16Bit:
    """Niederreiter Ciphertext implementation for 16-bit messages."""
    
    def __init__(self, rm_config: str = 'RM(3,5)'):
        """
        Initialize Niederreiter Ciphertext with 16-bit message support.
        
        Args:
            rm_config: Reed-Muller configuration
        """
        self.rm_config = rm_config
        rm_params = Niederreiter16BitConfig.get_rm_params(rm_config)
        
        # Initialize Niederreiter with the specified RM parameters
        self.niederreiter = NiederreiterSimple(r=rm_params['r'], m=rm_params['m'])
        
        print(f"Niederreiter {rm_config} initialized for 16-bit messages:")
        print(f"  Message length (k): {self.niederreiter.k} bits")
        print(f"  Code length (n): {self.niederreiter.n} bits")
        print(f"  Error correction capability (t): {self.niederreiter.t}")
        print(f"  Code rate (k/n): {rm_params['rate']:.3f} ({rm_params['rate']*100:.1f}%)")
        print(f"  Category: {rm_params['category']}")
        
        if self.niederreiter.k < 16:
            print(f"  ⚠️  Warning: Message length {self.niederreiter.k} < 16 bits")
        elif self.niederreiter.k == 16:
            print(f"  ✅ Perfect fit for 16-bit messages")
        else:
            print(f"  ℹ️  Supports 16-bit messages (can handle up to {self.niederreiter.k} bits)")
        
        # Rate analysis
        if rm_params['rate'] >= 0.8:
            print(f"  🚀 Ultra-high rate code - excellent efficiency!")
        elif rm_params['rate'] >= 0.6:
            print(f"  ⚡ Very high rate code - great efficiency!")
        elif rm_params['rate'] >= 0.3:
            print(f"  📈 High rate code - good efficiency!")
        else:
            print(f"  📊 Moderate rate code")
    
    def encrypt(self, message: bytes) -> str:
        """
        Encrypt a message using Niederreiter.
        
        Args:
            message: Message to encrypt
            
        Returns:
            str: Binary string representation of the encrypted syndrome
        """
        # Use Niederreiter encryption
        syndrome, num_errors = self.niederreiter.encrypt(message)
        
        # Convert syndrome to binary string
        syndrome_str = ''.join(str(bit) for bit in syndrome)
        
        return syndrome_str
    
    def decrypt(self, ciphertext: str) -> bytes:
        """
        Decrypt a Niederreiter ciphertext.
        
        Args:
            ciphertext: Binary string representation of the encrypted syndrome
            
        Returns:
            bytes: Decrypted message
        """
        # Convert binary string to syndrome list
        syndrome = [int(bit) for bit in ciphertext]
        
        # Use Niederreiter decryption
        decrypted_bytes = self.niederreiter.decrypt(syndrome)
        
        return decrypted_bytes
    
    def encrypt_16bit_message(self, message: str) -> str:
        """
        Encrypt a message ensuring it fits within 16-bit constraints.
        
        Args:
            message: String message to encrypt
            
        Returns:
            str: Binary string representation of the encrypted syndrome
        """
        # Convert string to bytes and then to 16-bit representation
        if isinstance(message, str):
            message_bytes = message.encode('utf-8')
        else:
            message_bytes = message
        
        # Ensure we don't exceed 16 bits (2 bytes)
        if len(message_bytes) > 2:
            message_bytes = message_bytes[:2]
            print(f"Warning: Message truncated to 2 bytes for 16-bit compatibility")
        
        # Pad to exactly 2 bytes if needed
        if len(message_bytes) < 2:
            message_bytes = message_bytes + b'\x00' * (2 - len(message_bytes))
        
        # Use Niederreiter encryption
        return self.encrypt(message_bytes)
    
    def decrypt_16bit_message(self, ciphertext: str) -> str:
        """
        Decrypt a 16-bit message.
        
        Args:
            ciphertext: Binary string representation of the encrypted syndrome
            
        Returns:
            str: Decrypted message
        """
        # Use Niederreiter decryption
        decrypted_bytes = self.decrypt(ciphertext)
        
        # Remove null padding
        decrypted_bytes = decrypted_bytes.rstrip(b'\x00')
        
        # Convert back to string
        try:
            return decrypted_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return decrypted_bytes.hex()

def test_niederreiter_16bit():
    """Test Niederreiter with 16-bit message support."""
    print("Testing Niederreiter with 16-bit Message Support")
    print("=" * 60)
    
    # List available options
    Niederreiter16BitConfig.list_niederreiter_options()
    
    # Test different RM configurations
    test_configs = [
        'RM(4,4)',  # Maximum efficiency
        'RM(4,5)',  # Ultra-high rate
        'RM(4,6)',  # Very high rate
        'RM(3,5)',  # Balanced high
        'RM(2,5)',  # Classic high
    ]
    
    print("\nTesting Niederreiter Configurations:")
    print("-" * 50)
    
    for rm_config in test_configs:
        print(f"\nTesting {rm_config}:")
        try:
            ciphertext = Niederreiter16BitConfig.create_niederreiter_ciphertext(rm_config)
            
            # Test encryption/decryption
            test_message = "Hi"  # 2 bytes = 16 bits
            encrypted = ciphertext.encrypt_16bit_message(test_message)
            decrypted = ciphertext.decrypt_16bit_message(encrypted)
            
            print(f"  Original: '{test_message}'")
            print(f"  Encrypted: {encrypted[:32]}...")
            print(f"  Decrypted: '{decrypted}'")
            print(f"  Success: {test_message == decrypted}")
            
        except Exception as e:
            print(f"  Error: {e}")

def compare_niederreiter_vs_mceliece():
    """Compare Niederreiter vs McEliece for 16-bit messages."""
    print("\n" + "=" * 60)
    print("Niederreiter vs McEliece Comparison")
    print("=" * 60)
    
    print("Niederreiter Advantages:")
    print("  - Based on syndrome decoding problem")
    print("  - Smaller public key size")
    print("  - Different security assumptions")
    print("  - Can be more efficient for certain applications")
    print()
    
    print("McEliece Advantages:")
    print("  - More mature and widely studied")
    print("  - Based on general decoding problem")
    print("  - Larger body of research")
    print("  - More implementations available")
    print()
    
    print("For 16-bit messages:")
    print("  - Both can use the same Reed-Muller codes")
    print("  - Same error correction capabilities")
    print("  - Same code rates and efficiency")
    print("  - Choice depends on specific security requirements")

if __name__ == "__main__":
    test_niederreiter_16bit()
    compare_niederreiter_vs_mceliece()
