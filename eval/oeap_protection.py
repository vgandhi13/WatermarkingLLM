"""
Optimal Embedding Attack Protection (OEAP) for Watermarking LLM

This module implements protection mechanisms against embedding-based attacks
on watermarked text, including detection, mitigation, and robustness enhancement.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

class EmbeddingAnalyzer:
    """Analyzes embedding spaces to detect potential attacks."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        return np.array(embeddings)
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between embeddings."""
        return cosine_similarity(embeddings)
    
    def detect_anomalies(self, embeddings: np.ndarray, threshold: float = 0.1) -> List[int]:
        """Detect anomalous embeddings that might indicate attacks."""
        # Use PCA to reduce dimensionality for anomaly detection
        pca = PCA(n_components=min(10, embeddings.shape[1]))
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Compute distances from centroid
        centroid = np.mean(reduced_embeddings, axis=0)
        distances = np.linalg.norm(reduced_embeddings - centroid, axis=1)
        
        # Identify outliers
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        anomaly_threshold = mean_dist + threshold * std_dist
        
        anomalies = [i for i, dist in enumerate(distances) if dist > anomaly_threshold]
        return anomalies

class WatermarkRobustnessEnhancer:
    """Enhances watermark robustness against embedding attacks."""
    
    def __init__(self, embedding_dim: int = 384, message_length: int = 16):
        self.embedding_dim = embedding_dim
        self.robustness_threshold = 0.8
        self.message_length = message_length  # Support for 16-bit messages
        
    def add_noise_resistance(self, watermark_bits: List[int], noise_level: float = 0.1) -> List[int]:
        """Add noise resistance to watermark bits."""
        # Ensure we have exactly the required message length
        if len(watermark_bits) < self.message_length:
            # Pad with zeros if too short
            watermark_bits = watermark_bits + [0] * (self.message_length - len(watermark_bits))
        elif len(watermark_bits) > self.message_length:
            # Truncate if too long
            watermark_bits = watermark_bits[:self.message_length]
        
        # Add redundancy to make watermark more robust
        enhanced_bits = []
        for bit in watermark_bits:
            # Create multiple copies with slight variations
            for _ in range(3):  # Triple redundancy
                enhanced_bits.append(bit)
        return enhanced_bits
    
    def apply_error_correction(self, watermark_bits: List[int]) -> List[int]:
        """Apply simple error correction coding to watermark."""
        # Ensure we have exactly the required message length
        if len(watermark_bits) < self.message_length:
            watermark_bits = watermark_bits + [0] * (self.message_length - len(watermark_bits))
        elif len(watermark_bits) > self.message_length:
            watermark_bits = watermark_bits[:self.message_length]
        
        # Simple parity check implementation
        corrected_bits = watermark_bits.copy()
        
        # Add parity bits every 8 bits (suitable for 16-bit messages)
        for i in range(0, len(watermark_bits), 8):
            chunk = watermark_bits[i:i+8]
            if len(chunk) == 8:
                parity = sum(chunk) % 2
                corrected_bits.append(parity)
        
        return corrected_bits
    
    def enhance_robustness(self, watermark_bits: List[int]) -> List[int]:
        """Apply multiple robustness enhancement techniques."""
        # Step 1: Add noise resistance
        robust_bits = self.add_noise_resistance(watermark_bits)
        
        # Step 2: Apply error correction
        robust_bits = self.apply_error_correction(robust_bits)
        
        return robust_bits

class AttackDetector:
    """Detects various types of embedding-based attacks."""
    
    def __init__(self):
        self.attack_patterns = {
            'paraphrase': self._detect_paraphrase_attack,
            'substitution': self._detect_substitution_attack,
            'insertion': self._detect_insertion_attack,
            'deletion': self._detect_deletion_attack
        }
    
    def _detect_paraphrase_attack(self, original_text: str, modified_text: str) -> Dict[str, Any]:
        """Detect paraphrase-based attacks."""
        # Simple heuristic: check for high semantic similarity but different surface forms
        words_original = set(original_text.lower().split())
        words_modified = set(modified_text.lower().split())
        
        jaccard_similarity = len(words_original.intersection(words_modified)) / len(words_original.union(words_modified))
        
        return {
            'attack_type': 'paraphrase',
            'confidence': jaccard_similarity,
            'is_attack': jaccard_similarity > 0.7 and original_text != modified_text
        }
    
    def _detect_substitution_attack(self, original_text: str, modified_text: str) -> Dict[str, Any]:
        """Detect word substitution attacks."""
        words_original = original_text.split()
        words_modified = modified_text.split()
        
        if len(words_original) != len(words_modified):
            return {'attack_type': 'substitution', 'confidence': 0.0, 'is_attack': False}
        
        substitutions = sum(1 for orig, mod in zip(words_original, words_modified) if orig != mod)
        substitution_rate = substitutions / len(words_original)
        
        return {
            'attack_type': 'substitution',
            'confidence': substitution_rate,
            'is_attack': substitution_rate > 0.1 and substitution_rate < 0.5
        }
    
    def _detect_insertion_attack(self, original_text: str, modified_text: str) -> Dict[str, Any]:
        """Detect insertion attacks."""
        words_original = original_text.split()
        words_modified = modified_text.split()
        
        if len(words_modified) <= len(words_original):
            return {'attack_type': 'insertion', 'confidence': 0.0, 'is_attack': False}
        
        insertion_rate = (len(words_modified) - len(words_original)) / len(words_original)
        
        return {
            'attack_type': 'insertion',
            'confidence': insertion_rate,
            'is_attack': insertion_rate > 0.1
        }
    
    def _detect_deletion_attack(self, original_text: str, modified_text: str) -> Dict[str, Any]:
        """Detect deletion attacks."""
        words_original = original_text.split()
        words_modified = modified_text.split()
        
        if len(words_modified) >= len(words_original):
            return {'attack_type': 'deletion', 'confidence': 0.0, 'is_attack': False}
        
        deletion_rate = (len(words_original) - len(words_modified)) / len(words_original)
        
        return {
            'attack_type': 'deletion',
            'confidence': deletion_rate,
            'is_attack': deletion_rate > 0.1
        }
    
    def detect_attacks(self, original_text: str, modified_text: str) -> List[Dict[str, Any]]:
        """Detect all types of attacks on the text."""
        detected_attacks = []
        
        for attack_type, detector_func in self.attack_patterns.items():
            result = detector_func(original_text, modified_text)
            if result['is_attack']:
                detected_attacks.append(result)
        
        return detected_attacks

class OEAPProtection:
    """Main OEAP protection system."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 message_length: int = 16):
        self.embedding_analyzer = EmbeddingAnalyzer(model_name)
        self.robustness_enhancer = WatermarkRobustnessEnhancer(message_length=message_length)
        self.attack_detector = AttackDetector()
        self.protection_history = []
        self.message_length = message_length
        
    def analyze_embedding_space(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze the embedding space for potential attacks."""
        embeddings = self.embedding_analyzer.get_embeddings(texts)
        similarity_matrix = self.embedding_analyzer.compute_similarity_matrix(embeddings)
        anomalies = self.embedding_analyzer.detect_anomalies(embeddings)
        
        return {
            'embeddings': embeddings,
            'similarity_matrix': similarity_matrix,
            'anomalies': anomalies,
            'mean_similarity': np.mean(similarity_matrix),
            'std_similarity': np.std(similarity_matrix)
        }
    
    def protect_watermark(self, watermark_bits: List[int], protection_level: str = 'medium') -> List[int]:
        """Apply protection mechanisms to watermark bits."""
        if protection_level == 'low':
            return watermark_bits
        elif protection_level == 'medium':
            return self.robustness_enhancer.enhance_robustness(watermark_bits)
        elif protection_level == 'high':
            # Apply maximum protection
            protected_bits = self.robustness_enhancer.enhance_robustness(watermark_bits)
            # Add additional redundancy
            return protected_bits + protected_bits[:len(protected_bits)//2]
        else:
            raise ValueError("Protection level must be 'low', 'medium', or 'high'")
    
    def detect_and_mitigate_attacks(self, original_text: str, modified_text: str) -> Dict[str, Any]:
        """Detect attacks and suggest mitigation strategies."""
        # Detect attacks
        detected_attacks = self.attack_detector.detect_attacks(original_text, modified_text)
        
        # Analyze embedding space
        embedding_analysis = self.analyze_embedding_space([original_text, modified_text])
        
        # Generate mitigation strategies
        mitigation_strategies = []
        for attack in detected_attacks:
            if attack['attack_type'] == 'paraphrase':
                mitigation_strategies.append("Increase watermark redundancy and use semantic-aware embedding")
            elif attack['attack_type'] == 'substitution':
                mitigation_strategies.append("Use character-level watermarking and increase bit density")
            elif attack['attack_type'] == 'insertion':
                mitigation_strategies.append("Use position-independent watermarking")
            elif attack['attack_type'] == 'deletion':
                mitigation_strategies.append("Use distributed watermarking across multiple positions")
        
        return {
            'detected_attacks': detected_attacks,
            'embedding_analysis': embedding_analysis,
            'mitigation_strategies': mitigation_strategies,
            'risk_level': self._assess_risk_level(detected_attacks, embedding_analysis)
        }
    
    def _assess_risk_level(self, attacks: List[Dict], embedding_analysis: Dict) -> str:
        """Assess the overall risk level based on detected attacks and embedding analysis."""
        if not attacks:
            return 'low'
        
        # Count high-confidence attacks
        high_confidence_attacks = sum(1 for attack in attacks if attack['confidence'] > 0.7)
        
        # Check for embedding anomalies
        has_anomalies = len(embedding_analysis['anomalies']) > 0
        
        if high_confidence_attacks >= 2 or has_anomalies:
            return 'high'
        elif high_confidence_attacks >= 1:
            return 'medium'
        else:
            return 'low'
    
    def generate_protection_report(self, texts: List[str], watermark_bits: List[int]) -> Dict[str, Any]:
        """Generate a comprehensive protection report."""
        # Analyze embedding space
        embedding_analysis = self.analyze_embedding_space(texts)
        
        # Test protection levels
        protection_results = {}
        for level in ['low', 'medium', 'high']:
            protected_bits = self.protect_watermark(watermark_bits, level)
            protection_results[level] = {
                'original_length': len(watermark_bits),
                'protected_length': len(protected_bits),
                'redundancy_ratio': len(protected_bits) / len(watermark_bits)
            }
        
        return {
            'embedding_analysis': embedding_analysis,
            'protection_results': protection_results,
            'recommendations': self._generate_recommendations(embedding_analysis, protection_results)
        }
    
    def _generate_recommendations(self, embedding_analysis: Dict, protection_results: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Check embedding space characteristics
        if embedding_analysis['std_similarity'] > 0.3:
            recommendations.append("High embedding variance detected - consider using more robust watermarking")
        
        if len(embedding_analysis['anomalies']) > 0:
            recommendations.append("Anomalous embeddings detected - investigate potential attacks")
        
        # Check protection effectiveness
        high_protection_ratio = protection_results['high']['redundancy_ratio']
        if high_protection_ratio > 3.0:
            recommendations.append("High protection level may impact performance - consider medium level")
        
        return recommendations

def test_oeap_protection():
    """Test the OEAP protection system."""
    print("Testing OEAP Protection System...")
    
    # Initialize OEAP protection
    oeap = OEAPProtection()
    
    # Test texts
    original_text = "The quick brown fox jumps over the lazy dog."
    modified_texts = [
        "A fast brown fox leaps over a sleepy dog.",  # Paraphrase
        "The quick brown fox jumps over the lazy cat.",  # Substitution
        "The quick brown fox jumps over the very lazy dog.",  # Insertion
        "The quick brown fox jumps over the dog.",  # Deletion
        "The quick brown fox jumps over the lazy dog."  # No change
    ]
    
    # Test watermark bits
    watermark_bits = [1, 0, 1, 1, 0, 0, 1, 0]
    
    print(f"Original text: {original_text}")
    print(f"Watermark bits: {watermark_bits}")
    print("\n" + "="*60)
    
    # Test each modified text
    for i, modified_text in enumerate(modified_texts):
        print(f"\nTest {i+1}: {modified_text}")
        
        # Detect and mitigate attacks
        result = oeap.detect_and_mitigate_attacks(original_text, modified_text)
        
        print(f"Detected attacks: {[attack['attack_type'] for attack in result['detected_attacks']]}")
        print(f"Risk level: {result['risk_level']}")
        print(f"Mitigation strategies: {result['mitigation_strategies']}")
    
    # Test protection levels
    print("\n" + "="*60)
    print("Testing Protection Levels:")
    
    for level in ['low', 'medium', 'high']:
        protected_bits = oeap.protect_watermark(watermark_bits, level)
        print(f"{level.capitalize()} protection: {len(watermark_bits)} -> {len(protected_bits)} bits")
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("Generating Protection Report...")
    
    all_texts = [original_text] + modified_texts
    report = oeap.generate_protection_report(all_texts, watermark_bits)
    
    print(f"Embedding analysis - Mean similarity: {report['embedding_analysis']['mean_similarity']:.3f}")
    print(f"Embedding analysis - Anomalies detected: {len(report['embedding_analysis']['anomalies'])}")
    print(f"Recommendations: {report['recommendations']}")

if __name__ == "__main__":
    test_oeap_protection()
