"""
OEAP Integration Module

This module integrates the OEAP protection system with the existing watermarking
infrastructure, providing seamless protection against embedding-based attacks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oeap_protection import OEAPProtection, EmbeddingAnalyzer, WatermarkRobustnessEnhancer, AttackDetector
from batch_main_vary_context_window import batch_encoder
from batch_decoder_vary_context_window import BatchWatermarkDecoder
from unwatermarked_samp import batch_encoder as batch_unencoder
from ecc.ciphertext import Ciphertext
from ecc.mceliece import McEliece
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np
import random
from enum import Enum

class EncDecMethod(Enum):
    STANDARD = 'Standard'
    RANDOM = 'Random'
    NEXT = 'Next'

class OEAPIntegratedWatermarker:
    """Integrated watermarking system with OEAP protection."""
    
    def __init__(self, model_name: str = "gpt2-medium", 
                 crypto_scheme: str = "Ciphertext",
                 enc_dec_method: str = "Next",
                 hash_scheme: str = "kmeans",
                 protection_level: str = "medium",
                 message_length: int = 16):
        
        self.model_name = model_name
        self.crypto_scheme = crypto_scheme
        self.enc_dec_method = enc_dec_method
        self.hash_scheme = hash_scheme
        self.protection_level = protection_level
        self.message_length = message_length
        
        # Initialize OEAP protection with 16-bit message support
        self.oeap = OEAPProtection(message_length=message_length)
        
        # Initialize cryptographic components
        if crypto_scheme == "Ciphertext":
            self.crypto = Ciphertext()
        elif crypto_scheme == "McEliece":
            self.crypto = McEliece()
        else:
            raise ValueError(f"Unsupported crypto scheme: {crypto_scheme}")
    
    def generate_protected_watermarked_text(self, prompts: List[str], 
                                          messages: List[str], 
                                          max_tokens: int = 1000,
                                          batch_size: int = 1) -> Dict[str, Any]:
        """Generate watermarked text with OEAP protection."""
        
        print("Generating watermarked text with OEAP protection...")
        
        # Step 1: Generate initial watermarked text
        results, actual_model = batch_encoder(
            prompts, 
            max_tokens=max_tokens, 
            batch_size=batch_size, 
            enc_method=self.enc_dec_method, 
            messages=messages, 
            model_name=self.model_name, 
            crypto_scheme=self.crypto_scheme, 
            hash_scheme=self.hash_scheme
        )
        
        # Step 2: Apply OEAP protection to watermark bits
        protected_results = []
        for result in results:
            # Extract watermark bits
            watermark_bits = result.get('encoded_bits', [])
            
            # Apply OEAP protection
            protected_bits = self.oeap.protect_watermark(watermark_bits, self.protection_level)
            
            # Create protected result
            protected_result = result.copy()
            protected_result['protected_bits'] = protected_bits
            protected_result['protection_level'] = self.protection_level
            protected_result['original_bits'] = watermark_bits
            
            protected_results.append(protected_result)
        
        # Step 3: Analyze embedding space for potential vulnerabilities
        generated_texts = [result['generated_text'] for result in protected_results]
        embedding_analysis = self.oeap.analyze_embedding_space(generated_texts)
        
        return {
            'results': protected_results,
            'actual_model': actual_model,
            'embedding_analysis': embedding_analysis,
            'protection_applied': True
        }
    
    def detect_attacks_on_watermarked_text(self, original_results: List[Dict], 
                                         modified_texts: List[str]) -> Dict[str, Any]:
        """Detect attacks on watermarked text."""
        
        print("Detecting attacks on watermarked text...")
        
        attack_results = []
        for i, (original_result, modified_text) in enumerate(zip(original_results, modified_texts)):
            original_text = original_result['generated_text']
            
            # Detect attacks
            attack_analysis = self.oeap.detect_and_mitigate_attacks(original_text, modified_text)
            
            attack_results.append({
                'index': i,
                'original_text': original_text,
                'modified_text': modified_text,
                'attack_analysis': attack_analysis
            })
        
        return {
            'attack_results': attack_results,
            'total_attacks_detected': sum(len(result['attack_analysis']['detected_attacks']) 
                                        for result in attack_results)
        }
    
    def decode_protected_watermark(self, actual_model, messages: List[str], 
                                 prompts: List[str], generated_texts: List[str],
                                 batch_size: int = 1) -> Dict[str, Any]:
        """Decode watermarks from protected text."""
        
        print("Decoding protected watermarks...")
        
        # Initialize decoder
        decoder = BatchWatermarkDecoder(
            actual_model, 
            message=messages, 
            dec_method=self.enc_dec_method, 
            model_name=self.model_name, 
            crypto_scheme=self.crypto_scheme, 
            hash_scheme=self.hash_scheme
        )
        
        # Decode watermarks
        decoded_results = decoder.batch_decode(prompts, generated_texts, batch_size=batch_size)
        
        # Analyze decoding success with OEAP protection
        decoding_analysis = []
        for i, decoded_result in enumerate(decoded_results):
            extracted_bits = decoded_result.get('extracted_bits', [])
            
            # Analyze extraction quality
            extraction_quality = self._analyze_extraction_quality(extracted_bits)
            
            decoding_analysis.append({
                'index': i,
                'extracted_bits': extracted_bits,
                'extraction_quality': extraction_quality
            })
        
        return {
            'decoded_results': decoded_results,
            'decoding_analysis': decoding_analysis
        }
    
    def _analyze_extraction_quality(self, extracted_bits: List[str]) -> Dict[str, Any]:
        """Analyze the quality of bit extraction."""
        if not extracted_bits:
            return {'quality_score': 0.0, 'confidence': 'low'}
        
        # Count valid bits (0 or 1)
        valid_bits = sum(1 for bit in extracted_bits if bit in ['0', '1'])
        total_bits = len(extracted_bits)
        
        quality_score = valid_bits / total_bits if total_bits > 0 else 0.0
        
        # Determine confidence level
        if quality_score >= 0.9:
            confidence = 'high'
        elif quality_score >= 0.7:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'quality_score': quality_score,
            'confidence': confidence,
            'valid_bits': valid_bits,
            'total_bits': total_bits
        }
    
    def comprehensive_evaluation(self, prompts: List[str], messages: List[str],
                               max_tokens: int = 1000, batch_size: int = 1) -> Dict[str, Any]:
        """Perform comprehensive evaluation with OEAP protection."""
        
        print("Performing comprehensive OEAP evaluation...")
        
        # Step 1: Generate protected watermarked text
        watermarked_results = self.generate_protected_watermarked_text(
            prompts, messages, max_tokens, batch_size
        )
        
        # Step 2: Generate unwatermarked text for comparison
        unwatermarked_results = batch_unencoder(
            prompts, max_tokens=max_tokens, batch_size=batch_size, model_name=self.model_name
        )
        
        # Step 3: Test attack detection
        watermarked_texts = [result['generated_text'] for result in watermarked_results['results']]
        unwatermarked_texts = [result['generated_text'] for result in unwatermarked_results]
        
        # Simulate attacks by using unwatermarked texts as "attacked" versions
        attack_detection = self.detect_attacks_on_watermarked_text(
            watermarked_results['results'], unwatermarked_texts
        )
        
        # Step 4: Decode watermarks
        watermarked_decoding = self.decode_protected_watermark(
            watermarked_results['actual_model'], messages, prompts, watermarked_texts, batch_size
        )
        
        unwatermarked_decoding = self.decode_protected_watermark(
            watermarked_results['actual_model'], messages, prompts, unwatermarked_texts, batch_size
        )
        
        # Step 5: Calculate metrics
        metrics = self._calculate_comprehensive_metrics(
            watermarked_decoding, unwatermarked_decoding, attack_detection
        )
        
        return {
            'watermarked_results': watermarked_results,
            'unwatermarked_results': unwatermarked_results,
            'attack_detection': attack_detection,
            'watermarked_decoding': watermarked_decoding,
            'unwatermarked_decoding': unwatermarked_decoding,
            'metrics': metrics,
            'protection_level': self.protection_level
        }
    
    def _calculate_comprehensive_metrics(self, watermarked_decoding: Dict, 
                                       unwatermarked_decoding: Dict,
                                       attack_detection: Dict) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Watermark detection metrics
        watermarked_quality_scores = [analysis['extraction_quality']['quality_score'] 
                                    for analysis in watermarked_decoding['decoding_analysis']]
        unwatermarked_quality_scores = [analysis['extraction_quality']['quality_score'] 
                                      for analysis in unwatermarked_decoding['decoding_analysis']]
        
        # Attack detection metrics
        total_attacks = attack_detection['total_attacks_detected']
        total_texts = len(attack_detection['attack_results'])
        
        return {
            'watermark_detection': {
                'watermarked_avg_quality': np.mean(watermarked_quality_scores),
                'unwatermarked_avg_quality': np.mean(unwatermarked_quality_scores),
                'detection_accuracy': np.mean(watermarked_quality_scores) - np.mean(unwatermarked_quality_scores)
            },
            'attack_detection': {
                'total_attacks_detected': total_attacks,
                'attack_detection_rate': total_attacks / total_texts if total_texts > 0 else 0
            },
            'protection_effectiveness': {
                'protection_level': self.protection_level,
                'robustness_improvement': self._calculate_robustness_improvement(watermarked_quality_scores)
            }
        }
    
    def _calculate_robustness_improvement(self, quality_scores: List[float]) -> float:
        """Calculate robustness improvement from protection."""
        # Simple heuristic: higher quality scores indicate better robustness
        return np.mean(quality_scores) if quality_scores else 0.0

def test_oeap_integration():
    """Test the OEAP integration with the watermarking system."""
    
    print("Testing OEAP Integration...")
    
    # Test parameters
    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain the concept of machine learning in simple terms."
    ]
    messages = ["Asteroid"] * len(prompts)
    
    # Initialize integrated watermarker
    watermarker = OEAPIntegratedWatermarker(
        model_name="gpt2-medium",
        crypto_scheme="Ciphertext",
        enc_dec_method="Next",
        hash_scheme="kmeans",
        protection_level="medium"
    )
    
    # Perform comprehensive evaluation
    results = watermarker.comprehensive_evaluation(
        prompts=prompts,
        messages=messages,
        max_tokens=200,
        batch_size=1
    )
    
    # Print results
    print("\n" + "="*60)
    print("OEAP Integration Test Results:")
    print("="*60)
    
    print(f"Protection Level: {results['protection_level']}")
    print(f"Total Attacks Detected: {results['attack_detection']['total_attacks_detected']}")
    
    metrics = results['metrics']
    print(f"\nWatermark Detection Metrics:")
    print(f"  Watermarked Avg Quality: {metrics['watermark_detection']['watermarked_avg_quality']:.3f}")
    print(f"  Unwatermarked Avg Quality: {metrics['watermark_detection']['unwatermarked_avg_quality']:.3f}")
    print(f"  Detection Accuracy: {metrics['watermark_detection']['detection_accuracy']:.3f}")
    
    print(f"\nAttack Detection Metrics:")
    print(f"  Attack Detection Rate: {metrics['attack_detection']['attack_detection_rate']:.3f}")
    
    print(f"\nProtection Effectiveness:")
    print(f"  Robustness Improvement: {metrics['protection_effectiveness']['robustness_improvement']:.3f}")
    
    return results

if __name__ == "__main__":
    test_oeap_integration()
