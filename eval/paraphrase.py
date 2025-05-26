import openai
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from typing import Optional, List, Dict
from dotenv import load_dotenv
import asyncio
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')  # Download the punkt tokenizer data
from datasets import load_dataset
from batch_main import batch_encoder
from batch_decoder import BatchWatermarkDecoder
from collections import defaultdict
from datetime import datetime
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

#----------------USER INPUT VARIABLES BEGIN------------------
BATCH_SIZE = 5
CRYPTO_SCHEME = 'Ciphertext'  # ['McEliece', 'Ciphertext']
MAX_TOKENS = 300
HASH_SCHEME = 'kmeans'  # ['hashlib', 'kmeans']
ENC_DEC_METHOD = 'Standard'  # ['Standard', 'Random', 'Next']
MODEL_NAME = "gpt2"

# Load dataset
def load_alpaca_dataset():
    """Load Alpaca evaluation set with outputs"""
    alpaca_dataset = load_dataset("tatsu-lab/alpaca_eval", trust_remote_code=True)
    # Get both instructions and outputs
    prompts_and_outputs = [
        {
            'prompt': instruction,
            'expected_output': output
        }
        for instruction, output in zip(
            alpaca_dataset['eval']['instruction'],
            alpaca_dataset['eval']['output']
        )
    ]
    return prompts_and_outputs

alpaca_prompts = load_alpaca_dataset()
PROMPTS = [x['prompt'] for x in alpaca_prompts]
PROMPTS = PROMPTS[:10]  # Take first 10 samples

MESSAGES = ['asteroid'] * len(PROMPTS)

class Paraphraser:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Paraphraser with an OpenAI API key."""
        load_dotenv()
        openai.api_key_path = "openai_key.txt"

        # Add translation pipelines
        self.en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-fr")
        self.fr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-fr-en")
        self.en_ru = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
        self.ru_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

    async def paraphrase(self, text: str) -> str:
        """Paraphrase the given text using GPT-3.5-turbo."""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at paraphrasing text while maintaining its original meaning."},
                    {"role": "user", "content": f"Paraphrase this text while maintaining its meaning:\n\n{text}"}
                ],
                temperature=1,
                max_tokens=1000
            )
            time.sleep(1)  # Rate limit
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during paraphrasing: {str(e)}")
            return None

    async def paraphrase_sentence(self, text: str) -> str:
        """Split text into sentences, paraphrase each one with context, and rejoin."""
        sentences = sent_tokenize(text)
        paraphrased_sentences = []

        # index into the sentences and ask for an example of what you expect, parse the output back
        
        for i, sentence in enumerate(sentences):
            context = " ".join(sentences[:i]) if i > 0 else ""
            prompt = (
                "Given some previous context and a sentence "
                "following that context, paraphrase the "
                "current sentence. Only return the "
                "paraphrased sentence in your response.\n"
                f"Previous context: {context}\n"
                f"Current sentence to paraphrase: {sentence}\n"
                "Your paraphrase of the current sentence:"
            )
            
            try:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert at paraphrasing sentences while maintaining their meaning and contextual relevance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                time.sleep(1)  # Rate limit
                paraphrased = response.choices[0].message.content.strip()
                paraphrased_sentences.append(paraphrased)
            except Exception as e:
                print(f"Error during paraphrasing sentence {i+1}: {str(e)}")
                paraphrased_sentences.append(sentence)
        
        return " ".join(paraphrased_sentences)

    async def translation_attack(self, text: str, language: str = "french") -> str:
        """Perform a translation attack by translating text to target language and back."""
        try:
            if language.lower() == "french":
                french = self.en_fr(text)[0]['translation_text']
                result = self.fr_en(french)[0]['translation_text']
            elif language.lower() == "russian":
                russian = self.en_ru(text)[0]['translation_text']
                result = self.ru_en(russian)[0]['translation_text']
            else:
                raise ValueError("Unsupported language. Use 'french' or 'russian'.")
            return result
        except Exception as e:
            print(f"Error during translation attack: {str(e)}")
            return None

    async def test_on_datasets(self) -> Dict:
        """Test paraphrasing attacks on OpenGen dataset."""
        print("\n" + "="*50)
        print("Testing OpenGen Dataset")
        print("="*50)
        opengen_results = []
        
        try:
            # Generate watermarked text using batch_encoder
            watermarked_results, actual_model = batch_encoder(
                PROMPTS,
                max_tokens=MAX_TOKENS,
                batch_size=BATCH_SIZE,
                messages=MESSAGES,
                enc_method=ENC_DEC_METHOD,
                model_name=MODEL_NAME,
                crypto_scheme=CRYPTO_SCHEME,
                hash_scheme=HASH_SCHEME
            )
            
            if watermarked_results:
                for i, result in enumerate(watermarked_results):
                    watermarked_text = result["generated_text"]
                    prompt = PROMPTS[i]
                    # Run attacks on the watermarked text
                    attack_results = await self.run_attacks(prompt, watermarked_text, decoder)
                    opengen_results.extend(attack_results)
            
            return {"opengen": opengen_results}
            
        except Exception as e:
            print(f"Error during dataset testing: {str(e)}")
            return {"opengen": []}

    async def run_attacks(self, prompt: str, watermarked_text: str, decoder: BatchWatermarkDecoder) -> List[tuple]:
        """Run different paraphrasing attacks on watermarked text and check detection."""
        results = []
        
        # Run all attacks
        attacks = [
            ("French Translation", lambda: self.translation_attack(watermarked_text, "french")),
            ("Russian Translation", lambda: self.translation_attack(watermarked_text, "russian")),
            ("GPT Paraphrase", lambda: self.paraphrase(watermarked_text)),
            ("Sentence Paraphrase", lambda: self.paraphrase_sentence(watermarked_text))
        ]
        
        for attack_name, attack_func in attacks:
            transformed_text = await attack_func()
            if transformed_text:
                decoded_results = decoder.batch_decode(
                    [prompt],
                    [transformed_text],
                    batch_size=1
                )
                
                # Calculate precision/recall
                extracted_bits = decoded_results[0]['extracted_bits']
                extracted_indices = decoded_results[0]['extracted_indices']
                
                # Get expected bits based on crypto scheme
                if CRYPTO_SCHEME == 'Ciphertext':
                    from ecc.ciphertext import Ciphertext
                    ciphertext = Ciphertext()
                    codeword = ciphertext.encrypt(100)
                    encoded_bits = [c for c in codeword]
                    encoded_bit_indices = list(range(len(encoded_bits)))
                else:  # McEliece
                    from ecc.mceliece import McEliece
                    codeword = McEliece().encrypt("asteroid".encode('utf-8'))[0]
                    codeword = ''.join(format(byte, '08b') for byte in codeword)
                    encoded_bits = [c for c in codeword]
                    encoded_bit_indices = list(range(len(encoded_bits)))
                
                # Create bit maps
                enc_idx_bit_map = defaultdict(list)
                ext_idx_bit_map = defaultdict(list)
                
                for i, bit in enumerate(encoded_bits):
                    if bit != '?':
                        enc_idx_bit_map[encoded_bit_indices[i]].append(bit)
                        
                for i, bit in enumerate(extracted_bits):
                    if bit != '?':
                        ext_idx_bit_map[extracted_indices[i]].append(bit)
                
                # Calculate matches
                matches = 0
                num_enc_bits = 0
                num_dec_bits = 0
                
                for idx, enc_arr in enc_idx_bit_map.items():
                    if idx not in ext_idx_bit_map:
                        continue
                    dec_arr = ext_idx_bit_map[idx]
                    for j in range(len(enc_arr)):
                        if j >= len(dec_arr):
                            break
                        if enc_arr[j] == dec_arr[j]:
                            matches += 1
                    num_enc_bits += len(enc_arr)
                    num_dec_bits += len(dec_arr)
                
                precision = matches/num_dec_bits if num_dec_bits > 0 else 0
                recall = matches/num_enc_bits if num_enc_bits > 0 else 0
                
                results.append((attack_name, watermarked_text, transformed_text, precision > 0.5))
                print_detection_result(watermarked_text, transformed_text, precision > 0.5)
        
        return results

def print_detection_result(original_text: str, transformed_text: str, detected: bool):
    """Print the detection results for a single text transformation."""
    print("\nDetection Results:")
    print("-" * 50)
    print(f"Original text: {original_text[:100]}...")
    print(f"Transformed text: {transformed_text[:100]}...")
    print(f"Detected: {'Yes' if detected else 'No'}")
    print("-" * 50)

def calculate_detection_rate(results: list) -> dict:
    """Calculate detection rates for each attack type."""
    attack_stats = {}
    
    for attack_type, _, _, detected in results:
        if attack_type not in attack_stats:
            attack_stats[attack_type] = {"total": 0, "detected": 0}
        
        attack_stats[attack_type]["total"] += 1
        if detected:
            attack_stats[attack_type]["detected"] += 1
    
    print("\nDetection Rates by Attack Type:")
    print("-" * 50)
    for attack_type, stats in attack_stats.items():
        rate = (stats["detected"] / stats["total"]) * 100
        print(f"{attack_type}:")
        print(f"Total tests: {stats['total']}")
        print(f"Detected: {stats['detected']}")
        print(f"Detection rate: {rate:.2f}%")
        print("-" * 25)
    
    return attack_stats

async def main():
    """Main function to run the paraphrase testing."""
    try:
        paraphraser = Paraphraser()
        
        print("Testing paraphrasing attacks on datasets...")
        results = await paraphraser.test_on_datasets()
        
        # Print overall summary
        print("\n" + "="*50)
        print("OVERALL SUMMARY")
        print("="*50)
        print("\nOpenGen Dataset:")
        calculate_detection_rate(results["opengen"])
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 