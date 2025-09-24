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
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from batch_main_vary_context_window import batch_encoder
from batch_decoder_vary_context_window import BatchWatermarkDecoder
from unwatermarked_samp import batch_encoder as batch_unencoder
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
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from collections import Counter
import random
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
#----------------USER INPUT VARIABLES BEGIN------------------


load_dotenv()
login(token = os.getenv('HF_TOKEN'))
class EncDecMethod(Enum):
    STANDARD = 'Standard'
    RANDOM = 'Random'
    NEXT = 'Next'

MODEL_NAMES = ['gpt2', 'gpt2-medium',   "meta-llama/Llama-3.2-1B",'mistralai/Mistral-7B-v0.1', "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]

#----------------USER INPUT VARIABLES BEGIN------------------
print("Testing with kmeans clustering variation")

BATCH_SIZE = 1
CRYPTO_SCHEME = 'Ciphertext' # ['McEliece', 'Ciphertext']
MAX_TOKENS = 300
ENC_DEC_METHOD = EncDecMethod.STANDARD.value
HASH_SCHEME = 'kmeans' # ['hashlib', 'kmeans']
MODEL = MODEL_NAMES[3]
print('Model used was ', MODEL)
paraphrase_prompt = "Given the following text, please paraphrase it by changing up to 40 percent of the words, making sure to keep the same meaning, style, tone, and context. Return the paraphrased text only. \n"
print('Paraphrase prompt used was: ', paraphrase_prompt)
KMEANS_MODEL = "kmeans_model3.pkl"  # Path to the KMeans model file
print('KMeans model used was: ', MODEL)
window_size = 3
print("Window size used was: ", window_size)

def load_dataset():
    """Load Alpaca evaluation set with outputs"""
    dataset = datasets.load_dataset("tatsu-lab/alpaca")
    
    # Get both instructions and outputs
    titles_and_prompts = [
        {
            'instruction': instruction,
            'input': input,
            'output': output
        }
        for instruction, input, output in zip(
            dataset['train']['instruction'],
            dataset['train']['input'],
            dataset['train']['output']
        )
    ]
    return titles_and_prompts


prompts = load_dataset()
# Extract instructions from the Alpaca dataset, excluding those with input content
PROMPTS = [p['instruction'] for p in prompts if( not p.get('input') or p['input'].strip() == '')]

PROMPTS = PROMPTS[:3]


def encode_prompt(prompt):
    return '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant who answers questions. <|eot_id|><|start_header_id|>user<|end_header_id|>''' + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

model = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def encode_prompt_mistral(prompt):
    device = "cuda" # the device to load the model onto

    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)


# PROMPTS = [encode_prompt_mistral(p) for p in PROMPTS]
print(PROMPTS)
print("Number of prompts: ", len(PROMPTS))

MESSAGES = [
    'Asteroid',
] * len(PROMPTS)
    

load_dotenv()
openai.api_key_path = "openai_key.txt"


def watermarked_detected(watermarked_results, decoded_results, i, when, avg_before, avg_after, total_guesses_before, total_guesses_after):
        # print(f"DEBUG: Entering watermarked_detected for sample {i}, when={when}")
        # print("GENERATED TEXT: ", watermarked_results[i]['generated_text'])
        encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en, t_enc, probs_start_enc = watermarked_results[i]['encoded_bits'], watermarked_results[i]['encoded_indices'], watermarked_results[i]['generated_text'], watermarked_results[i]['sampled_tokens'], watermarked_results[i]['token_probs'], watermarked_results[i]['t_values'], watermarked_results[i]['prob_starts']
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoded_results[i]['extracted_bits'], decoded_results[i]['extracted_indices'], decoded_results[i]['sampled_tokens'], decoded_results[i]['token_probs'], decoded_results[i]['threshold_values'], decoded_results[i]['prob_starts']
        
        # print(f"DEBUG: Extracted data for sample {i}")

        bits_match = encoded_bits == extracted_bits
        indices_match = encoded_bit_indices == extracted_indices
        tokens_match = sampled_tokens_en == sampled_tokens_dec
        probs_match = token_sampled_probs_en == token_sampled_probs_dec

        # print(f"DEBUG: Basic comparisons done for sample {i}")

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

        # print(f"DEBUG: Creating bit maps for sample {i}")

        for i in range(len(encoded_bits)):
            if encoded_bits[i]  != '?':
                enc_idx_bit_map[encoded_bit_indices[i]].append(encoded_bits[i])

        for i in range(len(extracted_bits)):
            if extracted_bits[i] != '?':
                ext_idx_bit_map[extracted_indices[i]].append(extracted_bits[i])
        
        # print(f"DEBUG: Bit maps created for sample {i}")
        
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

        # print(f"DEBUG: Starting precision calculations for sample {i}")

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
        
        # print(f"DEBUG: Basic precision calculation done for sample {i}")
        
        # print(matches, num_enc_bits, num_dec_bits)
        # print("Precision_send is ", matches/num_dec_bits if num_dec_bits != 0 else 0)
        # print(enc_idx_bit_map)
        # print(ext_idx_bit_map)
        #need to calcualte percision based on the decoded bits and the ground truth.
        ground_truth_bit_map = defaultdict(list)
        precision_watermarked = 0
        if CRYPTO_SCHEME == 'Ciphertext':
            # print(f"DEBUG: Starting Ciphertext calculations for sample {i}")
            ciphertext = Ciphertext()
            ground_truth_ciphertext = ciphertext.encrypt('Asteroid')
            for i in range(len(ground_truth_ciphertext)):
                ground_truth_bit_map[i].append(ground_truth_ciphertext[i])
            # print("Ground truth bit map", ground_truth_bit_map)
            codeword = [0]*128
            for i in range(len(codeword)):
                codeword[i] = random.randint(0, 1)
            for i, ext_arr in ext_idx_bit_map.items():
                counts = Counter(ext_arr)
                most_common_element = counts.most_common(1)[0][0]
                codeword[i] = most_common_element
                
            print(codeword)
            print("Decoded codeword: ", ciphertext.decrypt("".join(str(bit) for bit in codeword)))
            if ciphertext.decrypt("".join(str(bit) for bit in codeword)) == 'A':
                print("Codeword decoded correctly")
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
            precision_watermarked = matches/num_dec_bits if num_dec_bits != 0 else 0
            print("Precision_watermarked is ", precision_watermarked)
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
            # print("Precision_watermarked_correct_hashing is ", matches/num_dec_bits if num_dec_bits != 0 else 0)
            if when == 'before':
                avg_before += precision_watermarked
                total_guesses_before.append(precision_watermarked)
            else:
                avg_after += precision_watermarked
                total_guesses_after.append(precision_watermarked)
        else:
            # need to just check if the codeword decodes correctly.
            pass
        # print(f"DEBUG: About to return from watermarked_detected for sample {i}")
        return avg_before, avg_after, precision_watermarked, total_guesses_before, total_guesses_after

    
def paraphrase_overall(text: str) -> str:
    
    prompt = (
            paraphrase_prompt
        )
        
    response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant who is an expert at following instructions."},
                    {"role": "user", "content": prompt + "'"+text+"'"}
                ],
                temperature=1,
                max_tokens=MAX_TOKENS
            )
            # time.sleep(1)  # Rate limit
    paraphrased_text = response.choices[0].message.content
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
        
        response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at paraphrasing sentences while maintaining their meaning and contextual relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=MAX_TOKENS
            )
        paraphrased = response.choices[0].message.content
        paraphrased_sentences.append(paraphrased)
    
    return " ".join(paraphrased_sentences)

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics using scikit-learn functions."""
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix with explicit labels to ensure correct shape
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Extract values from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (same as recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    return {
        'precision': precision,
        'recall': recall,
        'tpr': tpr,
        'tnr': tnr,
        'fpr': fpr,
        'fnr': fnr,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def calculate_non_watermarked_precision(decoded_results, i):
    """Calculate precision for non-watermarked texts without requiring encoded_bits."""
    try:
        extracted_bits = decoded_results[i]['extracted_bits']
        extracted_indices = decoded_results[i]['extracted_indices']
        
        # For non-watermarked texts, we expect very few or no extracted bits
        # The precision should be very low since there shouldn't be any watermark
        if len(extracted_bits) == 0:
            return 0.0
        
        # Count how many bits were "detected" (should be very few for non-watermarked)
        detected_bits = sum(1 for bit in extracted_bits if bit != '?')
        total_possible_bits = len(extracted_bits)
        
        # Precision is the ratio of detected bits to total possible bits
        # For non-watermarked texts, this should be very low
        precision = detected_bits / total_possible_bits if total_possible_bits > 0 else 0.0
        
        return precision
    except Exception as e:
        print(f"Error calculating non-watermarked precision for sample {i}: {e}")
        return 0.0

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
    original_watermarked_texts = []  # Same watermarked text, not paraphrased
    
    skip_indices = []

    for i in range(len(watermarked_results)):
        #print("Original Text")
        #print(watermarked_results[i]['generated_text'])
        #print("Paraphrased Text")
        text_length = len(watermarked_results[i]['generated_text'])
        if text_length < MAX_TOKENS*0.5:
            skip_indices.append(i)
        paraphrased_results.append(paraphrase_overall(watermarked_results[i]['generated_text']))
        original_watermarked_texts.append(watermarked_results[i]['generated_text'])  # Keep original text as-is
        #print(paraphrased_results[-1])
    
    # Decode the original watermarked text (not paraphrased)
    decoded_results_original = decoder.batch_decode(
        [r["prompt"] for r in watermarked_results],
        original_watermarked_texts,
        batch_size=BATCH_SIZE)  
    # Decode the paraphrased watermarked text
    decoded_results_paraphrased = decoder.batch_decode(
        [r["prompt"] for r in watermarked_results],
        paraphrased_results,
        batch_size=BATCH_SIZE)
    
    
    total_guesses_before = []
    total_guesses_after = []
    # Reset for detailed output
    for i in range(len(decoded_results_paraphrased)):
        avg_difference = 0
        if i not in skip_indices:
            print("Original Text: ")
            print(" ")
            print(watermarked_results[i]['generated_text'])
            print("Length of original text: ", len(watermarked_results[i]['generated_text'].split(" ")))
            print("Decoding Results: ")
            avg_before, avg_after, precision_watermarked, total_guesses_before, total_guesses_after = watermarked_detected(watermarked_results, decoded_results_original, i, 'before', avg_before, avg_after, total_guesses_before, total_guesses_after)
            watermarked_results[i]['generated_text'] = paraphrased_results[i]
            print("Paraphrased Text: ")
            print(" ")
            print(paraphrased_results[i])
            print("Decoding Results")
            avg_before, avg_after, precision_watermarked, total_guesses_before, total_guesses_after = watermarked_detected(watermarked_results, decoded_results_paraphrased, i, 'after', avg_before, avg_after, total_guesses_before, total_guesses_after)
            avg_difference = avg_difference + (avg_before - avg_after)
        # print(avg_before, avg_after)
    print("Average precision before watermarking: ", avg_before/(len(PROMPTS)-len(skip_indices)))
    print("Average precision after watermarking: ", avg_after/(len(PROMPTS)-len(skip_indices)))
    print("Percentage decrease in precision_watermarked: ", (avg_difference)/(len(PROMPTS)-len(skip_indices)))
    print("Total guesses before: ", total_guesses_before)
    print("Total guesses after: ", total_guesses_after)
def graph_precision_send(avg_befores, avg_afters):
    pass
if __name__ == "__main__":

    main()