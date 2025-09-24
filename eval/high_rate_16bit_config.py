"""
High-Rate Reed-Muller Configuration for 16-bit Messages

This module provides high-rate Reed-Muller configurations for 16-bit messages
with longer codewords for better efficiency.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oeap_protection import OEAPProtection
from oeap_integration import OEAPIntegratedWatermarker
from ecc.mceliece_reed_muller import McElieceReedMuller
from ecc.ciphertext import Ciphertext
import numpy as np
from typing import List, Dict, Any

class HighRate16BitConfig:
    """Configuration for high-rate Reed-Muller codes with 16-bit message support."""
    
    # High-rate Reed-Muller parameters that support 16-bit messages
    # Sorted by efficiency (k/n ratio) - higher is better
    HIGH_RATE_RM_PARAMS = {
        # Ultra-high rate options (k/n > 0.8)
        'RM(4,4)': {'r': 4, 'm': 4, 'k': 16, 'n': 16, 't': -1, 'rate': 1.000, 'category': 'Ultra-High'},
        'RM(4,5)': {'r': 4, 'm': 5, 'k': 31, 'n': 32, 't': 0, 'rate': 0.969, 'category': 'Ultra-High'},
        'RM(4,6)': {'r': 4, 'm': 6, 'k': 57, 'n': 64, 't': 1, 'rate': 0.891, 'category': 'Ultra-High'},
        'RM(3,5)': {'r': 3, 'm': 5, 'k': 26, 'n': 32, 't': 1, 'rate': 0.812, 'category': 'Ultra-High'},
        
        # Very high rate options (k/n > 0.6)
        'RM(4,7)': {'r': 4, 'm': 7, 'k': 99, 'n': 128, 't': 3, 'rate': 0.773, 'category': 'Very-High'},
        'RM(3,6)': {'r': 3, 'm': 6, 'k': 42, 'n': 64, 't': 3, 'rate': 0.656, 'category': 'Very-High'},
        'RM(4,8)': {'r': 4, 'm': 8, 'k': 163, 'n': 256, 't': 7, 'rate': 0.637, 'category': 'Very-High'},
        
        # High rate options (k/n > 0.3)
        'RM(2,5)': {'r': 2, 'm': 5, 'k': 16, 'n': 32, 't': 3, 'rate': 0.500, 'category': 'High'},
        'RM(3,7)': {'r': 3, 'm': 7, 'k': 64, 'n': 128, 't': 7, 'rate': 0.500, 'category': 'High'},
        'RM(4,9)': {'r': 4, 'm': 9, 'k': 256, 'n': 512, 't': 15, 'rate': 0.500, 'category': 'High'},
        'RM(2,6)': {'r': 2, 'm': 6, 'k': 22, 'n': 64, 't': 7, 'rate': 0.344, 'category': 'High'},
        'RM(3,8)': {'r': 3, 'm': 8, 'k': 93, 'n': 256, 't': 15, 'rate': 0.363, 'category': 'High'},
        'RM(4,10)': {'r': 4, 'm': 10, 'k': 386, 'n': 1024, 't': 31, 'rate': 0.377, 'category': 'High'},
    }
    
    # Recommended configurations by use case
    RECOMMENDED_CONFIGS = {
        'maximum_efficiency': 'RM(4,4)',      # Perfect 1.0 rate, but no error correction
        'ultra_high_rate': 'RM(4,5)',         # 96.9% rate, minimal error correction
        'very_high_rate': 'RM(4,6)',          # 89.1% rate, some error correction
        'balanced_high': 'RM(3,5)',           # 81.2% rate, good balance
        'high_with_correction': 'RM(4,7)',    # 77.3% rate, decent error correction
        'moderate_high': 'RM(3,6)',           # 65.6% rate, good error correction
        'classic_high': 'RM(2,5)',            # 50.0% rate, proven option
    }
    
    @classmethod
    def get_rm_params(cls, config_name: str = 'RM(4,5)') -> Dict[str, Any]:
        """Get Reed-Muller parameters for high-rate 16-bit message support."""
        if config_name not in cls.HIGH_RATE_RM_PARAMS:
            raise ValueError(f"Unsupported high-rate RM configuration: {config_name}")
        return cls.HIGH_RATE_RM_PARAMS[config_name]
    
    @classmethod
    def list_available_configs(cls) -> None:
        """List all available high-rate configurations."""
        print("High-Rate Reed-Muller Configurations for 16-bit Messages")
        print("=" * 80)
        print(f"{'Config':<10} {'k':<4} {'n':<6} {'t':<4} {'Rate':<8} {'Category':<12} {'Description'}")
        print("-" * 80)
        
        for config_name, params in cls.HIGH_RATE_RM_PARAMS.items():
            description = f"Supports {params['k']}-bit messages"
            if params['k'] == 16:
                description += " (perfect fit)"
            elif params['k'] > 16:
                description += f" (can handle 16-bit)"
            
            print(f"{config_name:<10} {params['k']:<4} {params['n']:<6} {params['t']:<4} "
                  f"{params['rate']:<8.3f} {params['category']:<12} {description}")
    
    @classmethod
    def create_high_rate_ciphertext(cls, rm_config: str = 'RM(4,5)') -> 'HighRateCiphertext16Bit':
        """Create a high-rate Ciphertext instance for 16-bit messages."""
        return HighRateCiphertext16Bit(rm_config)
    
    @classmethod
    def create_high_rate_watermarker(cls, model_name: str = "gpt2-medium", 
                                   rm_config: str = 'RM(4,5)',
                                   protection_level: str = 'medium') -> OEAPIntegratedWatermarker:
        """Create an OEAP watermarker with high-rate Reed-Muller configuration."""
        return OEAPIntegratedWatermarker(
            model_name=model_name,
            crypto_scheme="Ciphertext",
            enc_dec_method="Next",
            hash_scheme="kmeans",
            protection_level=protection_level,
            message_length=16
        )

class HighRateCiphertext16Bit(Ciphertext):
    """High-rate Ciphertext implementation for 16-bit messages."""
    
    def __init__(self, rm_config: str = 'RM(4,5)'):
        """
        Initialize high-rate Ciphertext with 16-bit message support.
        
        Args:
            rm_config: High-rate Reed-Muller configuration
        """
        self.rm_config = rm_config
        rm_params = HighRate16BitConfig.get_rm_params(rm_config)
        
        # Initialize McEliece with the specified RM parameters
        self.mceliece = McElieceReedMuller(r=rm_params['r'], m=rm_params['m'])
        
        print(f"High-Rate McEliece {rm_config} initialized:")
        print(f"  Message length (k): {self.mceliece.k} bits")
        print(f"  Code length (n): {self.mceliece.n} bits")
        print(f"  Error correction capability (t): {self.mceliece.t}")
        print(f"  Code rate (k/n): {rm_params['rate']:.3f} ({rm_params['rate']*100:.1f}%)")
        print(f"  Category: {rm_params['category']}")
        
        if self.mceliece.k < 16:
            print(f"  ⚠️  Warning: Message length {self.mceliece.k} < 16 bits")
        elif self.mceliece.k == 16:
            print(f"  ✅ Perfect fit for 16-bit messages")
        else:
            print(f"  ℹ️  Supports 16-bit messages (can handle up to {self.mceliece.k} bits)")
        
        # Rate analysis
        if rm_params['rate'] >= 0.8:
            print(f"  🚀 Ultra-high rate code - excellent efficiency!")
        elif rm_params['rate'] >= 0.6:
            print(f"  ⚡ Very high rate code - great efficiency!")
        elif rm_params['rate'] >= 0.3:
            print(f"  📈 High rate code - good efficiency!")
        else:
            print(f"  📊 Moderate rate code")
    
    def encrypt_16bit_message(self, message: str) -> str:
        """
        Encrypt a message ensuring it fits within 16-bit constraints.
        
        Args:
            message: String message to encrypt
            
        Returns:
            str: Binary string representation of the encrypted ciphertext
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
        
        # Use the parent class encryption method
        return self.encrypt(message_bytes)
    
    def decrypt_16bit_message(self, ciphertext: str) -> str:
        """
        Decrypt a 16-bit message.
        
        Args:
            ciphertext: Binary string representation of the encrypted ciphertext
            
        Returns:
            str: Decrypted message
        """
        # Use the parent class decryption method
        decrypted_bytes = self.decrypt(ciphertext)
        
        # Remove null padding
        decrypted_bytes = decrypted_bytes.rstrip(b'\x00')
        
        # Convert back to string
        try:
            return decrypted_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return decrypted_bytes.hex()

def test_high_rate_configurations():
    """Test high-rate Reed-Muller configurations."""
    print("Testing High-Rate Reed-Muller Configurations")
    print("=" * 60)
    
    # List available configurations
    HighRate16BitConfig.list_available_configs()
    
    print("\n" + "=" * 60)
    print("Testing Specific High-Rate Configurations")
    print("=" * 60)
    
    # Test different high-rate configurations
    test_configs = [
        'RM(4,5)',  # Ultra-high rate
        'RM(4,6)',  # Very high rate
        'RM(3,5)',  # Balanced high
        'RM(4,7)',  # High with correction
        'RM(3,6)',  # Moderate high
    ]
    
    for rm_config in test_configs:
        print(f"\nTesting {rm_config}:")
        try:
            ciphertext = HighRate16BitConfig.create_high_rate_ciphertext(rm_config)
            
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

def compare_rate_options():
    """Compare different rate options for 16-bit messages."""
    print("\n" + "=" * 60)
    print("Rate Comparison for 16-bit Messages")
    print("=" * 60)
    
    comparison_configs = [
        ('RM(2,5)', 'Classic High Rate'),
        ('RM(3,5)', 'Balanced High Rate'),
        ('RM(4,5)', 'Ultra High Rate'),
        ('RM(4,6)', 'Very High Rate'),
        ('RM(4,7)', 'High Rate with Correction'),
    ]
    
    print(f"{'Config':<10} {'Rate':<8} {'n':<6} {'t':<4} {'Description'}")
    print("-" * 60)
    
    for config, description in comparison_configs:
        try:
            params = HighRate16BitConfig.get_rm_params(config)
            print(f"{config:<10} {params['rate']:<8.3f} {params['n']:<6} {params['t']:<4} {description}")
        except Exception as e:
            print(f"{config:<10} Error: {e}")

if __name__ == "__main__":
    test_high_rate_configurations()
    compare_rate_options()
