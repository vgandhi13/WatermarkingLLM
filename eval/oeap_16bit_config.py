"""
OEAP 16-bit Message Configuration

This module provides configuration and utilities for using OEAP protection
with 16-bit message length support, compatible with RM(2,5) Reed-Muller codes.
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

class OEAP16BitConfig:
    """Configuration for OEAP with 16-bit message support."""
    
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
            'protection_level': 'medium',
            'crypto_scheme': 'Ciphertext',
            'description': 'Perfect 100% rate, no error correction'
        },
        'ultra_high_rate': {
            'rm_params': 'RM(4,5)',
            'message_length': 16,
            'protection_level': 'medium',
            'crypto_scheme': 'Ciphertext',
            'description': '96.9% rate, minimal error correction'
        },
        'very_high_rate': {
            'rm_params': 'RM(4,6)',
            'message_length': 16,
            'protection_level': 'medium',
            'crypto_scheme': 'Ciphertext',
            'description': '89.1% rate, some error correction'
        },
        'balanced_high': {
            'rm_params': 'RM(3,5)',
            'message_length': 16,
            'protection_level': 'medium',
            'crypto_scheme': 'Ciphertext',
            'description': '81.2% rate, good balance'
        },
        'high_with_correction': {
            'rm_params': 'RM(4,7)',
            'message_length': 16,
            'protection_level': 'medium',
            'crypto_scheme': 'Ciphertext',
            'description': '77.3% rate, decent error correction'
        },
        'classic_high': {
            'rm_params': 'RM(2,5)',
            'message_length': 16,
            'protection_level': 'medium',
            'crypto_scheme': 'Ciphertext',
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
    def list_high_rate_options(cls) -> None:
        """List all available high-rate Reed-Muller configurations."""
        print("High-Rate Reed-Muller Options for 16-bit Messages")
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
    def create_16bit_ciphertext(cls, rm_config: str = 'RM(3,5)') -> 'Ciphertext16Bit':
        """Create a Ciphertext instance configured for 16-bit messages."""
        return Ciphertext16Bit(rm_config)
    
    @classmethod
    def create_oeap_watermarker(cls, model_name: str = "gpt2-medium", 
                               rm_config: str = 'RM(3,5)',
                               protection_level: str = 'medium') -> OEAPIntegratedWatermarker:
        """Create an OEAP watermarker configured for 16-bit messages."""
        return OEAPIntegratedWatermarker(
            model_name=model_name,
            crypto_scheme="Ciphertext",
            enc_dec_method="Next",
            hash_scheme="kmeans",
            protection_level=protection_level,
            message_length=16
        )

class Ciphertext16Bit(Ciphertext):
    """Ciphertext implementation optimized for 16-bit messages using RM(2,5)."""
    
    def __init__(self, rm_config: str = 'RM(2,5)'):
        """
        Initialize Ciphertext with 16-bit message support.
        
        Args:
            rm_config: Reed-Muller configuration ('RM(2,5)' or 'RM(2,6)')
        """
        self.rm_config = rm_config
        rm_params = OEAP16BitConfig.get_rm_params(rm_config)
        
        # Initialize McEliece with the specified RM parameters
        self.mceliece = McElieceReedMuller(r=rm_params['r'], m=rm_params['m'])
        
        print(f"McEliece {rm_config} initialized for 16-bit messages:")
        print(f"  Message length (k): {self.mceliece.k} bits")
        print(f"  Code length (n): {self.mceliece.n} bits")
        print(f"  Error correction capability (t): {self.mceliece.t}")
        
        if self.mceliece.k < 16:
            print(f"  ⚠️  Warning: Message length {self.mceliece.k} < 16 bits")
        elif self.mceliece.k == 16:
            print(f"  ✅ Perfect fit for 16-bit messages")
        else:
            print(f"  ℹ️  Supports 16-bit messages (can handle up to {self.mceliece.k} bits)")
    
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

def test_16bit_oeap():
    """Test OEAP with 16-bit message support."""
    print("Testing OEAP with 16-bit Message Support")
    print("=" * 50)
    
    # Test different RM configurations
    for rm_config in ['RM(2,5)', 'RM(2,6)']:
        print(f"\nTesting {rm_config}:")
        try:
            ciphertext = OEAP16BitConfig.create_16bit_ciphertext(rm_config)
            
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
    
    # Test OEAP protection with 16-bit messages
    print(f"\nTesting OEAP Protection with 16-bit Messages:")
    try:
        oeap = OEAPProtection(message_length=16)
        
        # Test watermark bits (16 bits)
        watermark_bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        print(f"  Original 16-bit watermark: {watermark_bits}")
        
        # Test protection levels
        for level in ['low', 'medium', 'high']:
            protected_bits = oeap.protect_watermark(watermark_bits, level)
            redundancy_ratio = len(protected_bits) / len(watermark_bits)
            print(f"  {level.capitalize()} protection: {len(watermark_bits)} -> {len(protected_bits)} bits "
                  f"(redundancy: {redundancy_ratio:.2f}x)")
        
    except Exception as e:
        print(f"  OEAP Error: {e}")
    
    # Test integrated watermarker
    print(f"\nTesting Integrated OEAP Watermarker:")
    try:
        watermarker = OEAP16BitConfig.create_oeap_watermarker(
            model_name="gpt2-medium",
            rm_config='RM(2,5)',
            protection_level='medium'
        )
        
        print(f"  Model: {watermarker.model_name}")
        print(f"  Message length: {watermarker.message_length} bits")
        print(f"  Protection level: {watermarker.protection_level}")
        print(f"  Crypto scheme: {watermarker.crypto_scheme}")
        print(f"  ✅ Integrated watermarker ready for 16-bit messages")
        
    except Exception as e:
        print(f"  Integration Error: {e}")

def compare_8bit_vs_16bit():
    """Compare 8-bit vs 16-bit message support."""
    print("\n" + "=" * 50)
    print("Comparing 8-bit vs 16-bit Message Support")
    print("=" * 50)
    
    # 8-bit configuration (current default)
    print("8-bit Configuration (RM(1,7)):")
    try:
        ciphertext_8bit = Ciphertext()  # Uses RM(1,7)
        print(f"  Message length: {ciphertext_8bit.mceliece.k} bits")
        print(f"  Code length: {ciphertext_8bit.mceliece.n} bits")
        print(f"  Error correction: {ciphertext_8bit.mceliece.t}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # 16-bit configuration
    print("\n16-bit Configuration (RM(2,5)):")
    try:
        ciphertext_16bit = OEAP16BitConfig.create_16bit_ciphertext('RM(2,5)')
        print(f"  Message length: {ciphertext_16bit.mceliece.k} bits")
        print(f"  Code length: {ciphertext_16bit.mceliece.n} bits")
        print(f"  Error correction: {ciphertext_16bit.mceliece.t}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"  8-bit:  Higher error correction (t=31) but limited message size")
    print(f"  16-bit: Lower error correction (t=3) but supports larger messages")
    print(f"  Recommendation: Use RM(2,5) for 16-bit messages when needed")

def test_high_rate_options():
    """Test high-rate Reed-Muller options."""
    print("Testing High-Rate Reed-Muller Options")
    print("=" * 50)
    
    # List all available options
    OEAP16BitConfig.list_high_rate_options()
    
    # Test specific high-rate configurations
    high_rate_configs = [
        'RM(4,4)',  # Maximum efficiency
        'RM(4,5)',  # Ultra-high rate
        'RM(4,6)',  # Very high rate
        'RM(3,5)',  # Balanced high
    ]
    
    print("\nTesting High-Rate Configurations:")
    print("-" * 40)
    
    for rm_config in high_rate_configs:
        print(f"\nTesting {rm_config}:")
        try:
            ciphertext = OEAP16BitConfig.create_16bit_ciphertext(rm_config)
            
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

if __name__ == "__main__":
    test_high_rate_options()
    print("\n" + "="*50)
    test_16bit_oeap()
    compare_8bit_vs_16bit()
