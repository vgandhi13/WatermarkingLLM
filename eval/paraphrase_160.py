import openai
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from typing import Optional, List, Dict
from dotenv import load_dotenv
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')  # Download the punkt tokenizer data
from datasets import load_dataset
from batch_main import batch_encoder
from batch_decoder import BatchWatermarkDecoder
from collections import defaultdict
from datetime import datetime
from ecc.mceliece import McEliece
from ecc.ciphertext import Ciphertext
import csv
from huggingface_hub import login
import json
import numpy as np
import matplotlib.pyplot as plt

#----------------USER INPUT VARIABLES BEGIN------------------
BATCH_SIZE = 5
CRYPTO_SCHEME = 'Ciphertext'  # ['McEliece', 'Ciphertext']
MAX_TOKENS = 300
HASH_SCHEME = 'kmeans'  # ['hashlib', 'kmeans']
ENC_DEC_METHOD = 'Standard'  # ['Standard', 'Random', 'Next']
token = os.getenv("HF_TOKEN")  # Replace with your actual Hugging Face token
login(token = token)
MODEL_NAME = "meta-llama/Llama-3.1-8B"

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
PROMPTS = PROMPTS[:5]  # Take first 1 samples

MESSAGES = ['asteroid'] * len(PROMPTS)

load_dotenv()
openai.api_key_path = "openai_key.txt"

def watermarked_detected(watermarked_results, decoded_results, i):
        encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en, t_enc, probs_start_enc = watermarked_results[i]['encoded_bits'], watermarked_results[i]['encoded_indices'], watermarked_results[i]['generated_text'], watermarked_results[i]['sampled_tokens'], watermarked_results[i]['token_probs'], watermarked_results[i]['t_values'], watermarked_results[i]['prob_starts']
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoded_results[i]['extracted_bits'], decoded_results[i]['extracted_indices'], decoded_results[i]['sampled_tokens'], decoded_results[i]['token_probs'], decoded_results[i]['threshold_values'], decoded_results[i]['prob_starts']
        
        

        bits_match = encoded_bits == extracted_bits
        indices_match = encoded_bit_indices == extracted_indices
        # tokens_match = sampled_tokens_en == sampled_tokens_dec
        # probs_match = token_sampled_probs_en == token_sampled_probs_dec

        # print(f"Encoded bits match extracted bits: {GREEN if bits_match else RED}{bits_match}{RESET}")
        # print(f"Encoded bit indices match extracted indices: {GREEN if indices_match else RED}{indices_match}{RESET}")

        # print('Num of encoded bits', len(encoded_bits))
        # print('Num of decoded bits', len(extracted_bits))
        # Count matching bits and indices
        matching_bits = sum(1 for i in range(min(len(extracted_bits), len(encoded_bits))) if extracted_bits[i] == encoded_bits[i])
        matching_indices = sum(1 for i in range(min(len(extracted_indices), len(encoded_bit_indices))) if extracted_indices[i] == encoded_bit_indices[i])

        # print(f"Columnwise Matching bits: {matching_bits} / {len(extracted_bits)}")
        # print(f"Columnwise Matching indices: {matching_indices} / {len(extracted_indices)}")

       

        enc_idx_bit_map = defaultdict(list)
        ext_idx_bit_map = defaultdict(list)


        for i in range(len(encoded_bits)):
            if encoded_bits[i]  != '?':
                enc_idx_bit_map[encoded_bit_indices[i]].append(encoded_bits[i])

        for i in range(len(extracted_bits)):
            if extracted_bits[i] != '?':
                ext_idx_bit_map[extracted_indices[i]].append(extracted_bits[i])
        
        # print("Encoded index and their bits", enc_idx_bit_map)
        # print("Decoded index and their bits", ext_idx_bit_map)

        
        # print('Indices which were encoded but not decoded ',not_decoded)
        # print('Indices were bits decoded dont match bits encoded', bit_not_same)

        matches = 0
        num_enc_bits = 0
        num_dec_bits = 0


        for i, enc_arr in enc_idx_bit_map.items(): #change the decoding
            if i not in ext_idx_bit_map:
                continue
            
            dec_arr = ext_idx_bit_map[i]
            for j in range(len(enc_arr)):
                if j>= len(dec_arr):
                    break
                if enc_arr[j] == dec_arr[j]:
                    matches += 1
            num_enc_bits += len(enc_arr)
            num_dec_bits += len(dec_arr)
        
        # print(matches, num_enc_bits, num_dec_bits)
        print("Precision_send is ", matches/num_dec_bits)
        # print(enc_idx_bit_map)
        # print(ext_idx_bit_map)
        #need to calcualte percision based on the decoded bits and the ground truth.
        ground_truth_bit_map = defaultdict(list)
        if CRYPTO_SCHEME == 'Ciphertext':
            ciphertext = Ciphertext()
            ground_truth_ciphertext = ciphertext.encrypt(160)
            for i in range(len(ground_truth_ciphertext)):
                ground_truth_bit_map[i].append(ground_truth_ciphertext[i])
            # print("Ground truth bit map", ground_truth_bit_map)
            matches = 0
            num_enc_bits = 0
            num_dec_bits = 0
            # print(ext_idx_bit_map)
            for i, dec_arr in ext_idx_bit_map.items(): #change the decoding
                if i not in ground_truth_bit_map:
                    continue
                bit = ground_truth_bit_map[i]
                for j in range(len(dec_arr)):
                    if bit[0] == dec_arr[j]:
                        matches += 1
                
                num_dec_bits += len(dec_arr)
            print("Precision_watermarked is ", matches/num_dec_bits)
            matches = 0
            num_enc_bits = 0
            num_dec_bits = 0
            # print(ext_idx_bit_map)
            for i, dec_arr in ext_idx_bit_map.items(): #change the decoding
                if i not in enc_idx_bit_map:
                    continue
                bit = ground_truth_bit_map[i]
                for j in range(len(dec_arr)):
                    if bit[0] == dec_arr[j]:
                        matches += 1
                
                num_dec_bits += len(dec_arr)
            print("Precision_watermarked_correct_hashing is ", matches/num_dec_bits)
        else:
            # need to just check if the codeword decodes correctly.
            pass
            



def paraphrase_sentence(text: str) -> str:
    """Split text into sentences, paraphrase each one with context, and rejoin."""
    sentences = sent_tokenize(text)
    paraphrased_sentences = []
    
    for i, sentence in enumerate(sentences):
        context = " ".join(sentences[:i]) if i > 0 else ""
        prompt = (
            "Given some previous context and a sentence "
            "following that context, paraphrase the "
            "current sentence. Only return the "
            "paraphrased sentence in your response. Do not add any other text to your response.\n"
            f"Previous context: {context}\n"
            f"Current sentence to paraphrase: {sentence}\n"
            "Your paraphrase of the current sentence:"
        )
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at paraphrasing sentences while maintaining their meaning and contextual relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=MAX_TOKENS
            )
            # time.sleep(1)  # Rate limit
            paraphrased = response.choices[0].message.content
            paraphrased_sentences.append(paraphrased)
        except Exception as e:
            print(f"Error during paraphrasing sentence {i+1}: {str(e)}")
            paraphrased_sentences.append(sentence)
    
    return " ".join(paraphrased_sentences)

def main():
    """Main function to run the paraphrase testing."""
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
    decoder = BatchWatermarkDecoder(actual_model, message=MESSAGES, dec_method=ENC_DEC_METHOD, model_name = MODEL_NAME, crypto_scheme=CRYPTO_SCHEME, hash_scheme=HASH_SCHEME)
    
    paraphrased_results = []
    watermarked_results_before = []
    
    for i in range(len(watermarked_results)):
        #print("Original Text")
        #print(watermarked_results[i]['generated_text'])
        #print("Paraphrased Text")
        paraphrased_results.append(paraphrase_sentence(watermarked_results[i]['generated_text']).strip())
        watermarked_results_before.append(watermarked_results[i]['generated_text'].strip())
        #print(paraphrased_results[-1])
    decoded_results_before = decoder.batch_decode(
        [r["prompt"] for r in watermarked_results],
        watermarked_results_before,
        batch_size=BATCH_SIZE)  
    decoded_results_after = decoder.batch_decode(
        [r["prompt"] for r in watermarked_results],
        paraphrased_results,
        batch_size=BATCH_SIZE)
    for i in range(len(decoded_results_after)):
        print("Original Text: ")
        print(" ")
        print(watermarked_results[i]['generated_text'])
        print("Decoding Results: ")
        watermarked_detected(watermarked_results, decoded_results_before, i)
        watermarked_results[i]['generated_text'] = paraphrased_results[i]
        print("Paraphrased Text: ")
        print(" ")
        print(paraphrased_results[i])
        print("Decoding Results")
        watermarked_detected(watermarked_results, decoded_results_after, i)
if __name__ == "__main__":
    main()