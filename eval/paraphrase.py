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
from transformers import AutoTokenizer
from datasets import load_dataset
from batch_main_vary_context_window import batch_encoder
from batch_decoder_vary_context_window import BatchWatermarkDecoder
from collections import defaultdict
from datetime import datetime
from ecc.mceliece import McEliece
from ecc.ciphertext import Ciphertext
import csv
from huggingface_hub import login
import json
import numpy as np
from enum import Enum
import datasets
import matplotlib.pyplot as plt
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
#----------------USER INPUT VARIABLES BEGIN------------------

login(token = os.getenv("HF_TOKEN"))
class EncDecMethod(Enum):
    STANDARD = 'Standard'
    RANDOM = 'Random'
    NEXT = 'Next'

MODEL_NAMES = ['gpt2', 'gpt2-medium',   "meta-llama/Llama-3.2-1B",'ministral/Ministral-3b-instruct', "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]

#----------------USER INPUT VARIABLES BEGIN------------------
print("Testing with kmeans clustering variation")

BATCH_SIZE = 5
CRYPTO_SCHEME = 'Ciphertext' # ['McEliece', 'Ciphertext']
MAX_TOKENS = 500
ENC_DEC_METHOD = EncDecMethod.STANDARD.value
HASH_SCHEME = 'kmeans' # ['hashlib', 'kmeans']
MODEL = MODEL_NAMES[-1]
print('Model used was ', MODEL)

KMEANS_MODEL = "kmeans_model3.pkl"  # Path to the KMeans model file
print('KMeans model used was: ', MODEL)
window_size = 3
print("Window size used was: ", window_size)

def load_dataset():
        """Load Alpaca evaluation set with outputs"""
        dataset = datasets.load_dataset("llm-aes/writing-prompts", trust_remote_code=True)
        # Get both instructions and outputs
        # titles_and_prompts = [
        #     {
        #         'instruction': instruction,
        #         'output': output
        #     }
        #     for instruction, output in zip(
        #         dataset['train']['instruction'],
        #         dataset['train']['output']
        #     )
        # ]
        # return titles_and_prompts
        prompts = dataset['train']['prompt']
        return prompts

prompts = load_dataset()
PROMPTS = [p.strip().replace('\n', '').strip('"').strip("'") for p in prompts]

PROMPTS = PROMPTS[:10]
print(PROMPTS)
print("Number of prompts: ", len(PROMPTS))

def encode_prompt(prompt):
    return '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant for story generation. You are given a story and you need to continue it. <|eot_id|><|start_header_id|>user<|end_header_id|>''' + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


PROMPTS = [encode_prompt(p) for p in PROMPTS]


MESSAGES = [
    'Asteroid',
] * len(PROMPTS)
    

load_dotenv()
openai.api_key_path = "openai_key.txt"


def watermarked_detected(watermarked_results, decoded_results, i, when, avg_before, avg_after):
        encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en, t_enc, probs_start_enc = watermarked_results[i]['encoded_bits'], watermarked_results[i]['encoded_indices'], watermarked_results[i]['generated_text'], watermarked_results[i]['sampled_tokens'], watermarked_results[i]['token_probs'], watermarked_results[i]['t_values'], watermarked_results[i]['prob_starts']
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoded_results[i]['extracted_bits'], decoded_results[i]['extracted_indices'], decoded_results[i]['sampled_tokens'], decoded_results[i]['token_probs'], decoded_results[i]['threshold_values'], decoded_results[i]['prob_starts']
        
        

        bits_match = encoded_bits == extracted_bits
        indices_match = encoded_bit_indices == extracted_indices
        tokens_match = sampled_tokens_en == sampled_tokens_dec
        probs_match = token_sampled_probs_en == token_sampled_probs_dec

        # print(f"Encoded bits match extracted bits: {GREEN if bits_match else RED}{bits_match}{RESET}")
        # print(f"Encoded bit indices match extracted indices: {GREEN if indices_match else RED}{indices_match}{RESET}")

        # print('Num of encoded bits', len(encoded_bits))
        # print('Num of decoded bits', len(extracted_bits))
        # # Count matching bits and indices
        # matching_bits = sum(1 for i in range(min(len(extracted_bits), len(encoded_bits))) if extracted_bits[i] == encoded_bits[i])
        # matching_indices = sum(1 for i in range(min(len(extracted_indices), len(encoded_bit_indices))) if extracted_indices[i] == encoded_bit_indices[i])

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

        not_encoded = 0
        bit_not_same = 0
        for k, v in ext_idx_bit_map.items():
            if k not in enc_idx_bit_map:
                not_encoded += 1
            else:
                if v != enc_idx_bit_map[k]:
                    bit_not_same+=1
        # print('Indices which were encoded but not decoded ',not_encoded)
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
        print("Precision_send is ", matches/num_dec_bits if num_dec_bits != 0 else 0)
        if when == 'before':
            avg_before += matches/num_dec_bits if num_dec_bits != 0 else 0
        else:
            avg_after += matches/num_dec_bits if num_dec_bits != 0 else 0
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
            print("Precision_watermarked is ", matches/num_dec_bits if num_dec_bits != 0 else 0)
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
            print("Precision_watermarked_correct_hashing is ", matches/num_dec_bits if num_dec_bits != 0 else 0)
        else:
            # need to just check if the codeword decodes correctly.
            pass
        # if not (bits_match and indices_match and tokens_match and probs_match):
        #     print("\n⚠️ Mismatch Detected! Side-by-Side Comparison Table:\n")
        #     print(f"{'Idx':<5} | {'EBit':<3} | {'ExBit':<3} | {'EIdx':<3} | {'ExIdx':<3} | "
        #         f"{'Encoded Token':<20} | {'Extracted Token':<20} | "
        #         f"{'Encoded Prob':<12} | {'Extracted Prob':<12} | "
        #         f"{'Enc Start Prob':<14} | {'Ext Start Prob':<14} | "
        #         f"{'Enc T':<8} | {'Ext T':<8}")
        #     print("-" * 150)

        #     min_len = min(len(encoded_bits), len(extracted_bits), len(encoded_bit_indices), len(extracted_indices), 
        #                 len(sampled_tokens_en), len(sampled_tokens_dec), len(token_sampled_probs_en), len(token_sampled_probs_dec))

        #     for i in range(min_len):
        #         e_bit = encoded_bits[i] if i < len(encoded_bits) else "N/A"
        #         x_bit = extracted_bits[i] if i < len(extracted_bits) else "N/A"
        #         e_idx = encoded_bit_indices[i] if i < len(encoded_bit_indices) else "N/A"
        #         x_idx = extracted_indices[i] if i < len(extracted_indices) else "N/A"
        #         e_tok = sampled_tokens_en[i] if i < len(sampled_tokens_en) else "N/A"
        #         x_tok = sampled_tokens_dec[i] if i < len(sampled_tokens_dec) else "N/A"
        #         e_prob = token_sampled_probs_en[i] if i < len(token_sampled_probs_en) else "N/A"
        #         x_prob = token_sampled_probs_dec[i] if i < len(token_sampled_probs_dec) else "N/A"
        #         e_start_prob = probs_start_enc[i] if i < len(probs_start_enc) else "N/A"
        #         x_start_prob = probs_start_ext[i] if i < len(probs_start_ext) else "N/A"
        #         e_t = t_enc[i] if i < len(t_enc) else "N/A"
        #         x_t = t_ext[i] if i < len(t_ext) else "N/A"

        #         # Highlight mismatches
        #         e_bit = f"{RED}{e_bit}{RESET}" if e_bit != x_bit else e_bit
        #         e_idx = f"{RED}{e_idx}{RESET}" if e_idx != x_idx else e_idx
        #         e_tok = f"{RED}{e_tok}{RESET}" if e_tok != x_tok else e_tok
        #         e_prob = f"{RED}{e_prob:.6f}{RESET}" if e_prob != x_prob else f"{e_prob:.6f}"

        #         print(f"{i:<5} | {e_bit:<3} | {x_bit:<3} | {e_idx:<3} | {x_idx:<3} | "
        #             f"{e_tok:<20} | {x_tok:<20} | {e_prob:<12} | {x_prob:<12.6f} | "
        #             f"{e_start_prob:<14} | {x_start_prob:<14.6f} | {e_t:<8} | {x_t:<8.6f}")
        return avg_before, avg_after

    
def paraphrase_overall(text: str) -> str:
    
    prompt = (
            "Given the following text, paraphrase it while retaining its entire original meaning and contextual relevance. Pretend you are a high school student who wants to make their text not look AI generated, but remember that you only have the capabalities of a high school student. Only return the paraphrased text in your response.\n"
        )
        
    try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at paraphrasing texts while maintaining their meaning and contextual relevance. "},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                max_tokens=MAX_TOKENS
            )
            # time.sleep(1)  # Rate limit
            paraphrased_text = response.choices[0].message.content
    except Exception as e:
            print(f"Error during paraphrasing text: {str(e)}")
            
    
    return paraphrased_text


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
    avg_before = 0 
    avg_after = 0
    watermarked_results, actual_model = batch_encoder(
            PROMPTS,
            max_tokens=MAX_TOKENS,
            batch_size=BATCH_SIZE,
            messages=MESSAGES,
            enc_method=ENC_DEC_METHOD,
            model_name=MODEL,
            crypto_scheme=CRYPTO_SCHEME,
            hash_scheme=HASH_SCHEME,kmeans_model_path=KMEANS_MODEL,window_size=window_size
        )
    decoder = BatchWatermarkDecoder(actual_model, message=MESSAGES, dec_method=ENC_DEC_METHOD, model_name = MODEL, crypto_scheme=CRYPTO_SCHEME, hash_scheme=HASH_SCHEME, kmeans_model_path=KMEANS_MODEL,
        window_size=window_size)
    
    paraphrased_results = []
    watermarked_results_before = []
    
    skip_indices = []

    for i in range(len(watermarked_results)):
        #print("Original Text")
        #print(watermarked_results[i]['generated_text'])
        #print("Paraphrased Text")
        paraphrased_results.append(paraphrase_overall(watermarked_results[i]['generated_text']))
        watermarked_results_before.append(watermarked_results[i]['generated_text'])
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
        if i not in skip_indices:
            print("Original Text: ")
            print(" ")
            print(watermarked_results[i]['generated_text'])
            print("Length of original text: ", len(watermarked_results[i]['generated_text'].split(" ")))
            print("Decoding Results: ")
            avg_before, avg_after = watermarked_detected(watermarked_results, decoded_results_before, i, 'before', avg_before, avg_after)
            watermarked_results[i]['generated_text'] = paraphrased_results[i]
            print("Paraphrased Text: ")
            print(" ")
            print(paraphrased_results[i])
            print("Decoding Results")
            avg_before, avg_after = watermarked_detected(watermarked_results, decoded_results_after, i, 'after', avg_before, avg_after)
        # print(avg_before, avg_after)
    print("Average precision before watermarking: ", avg_before/len(PROMPTS))
    print("Average precision after watermarking: ", avg_after/len(PROMPTS))
if __name__ == "__main__":

    main()