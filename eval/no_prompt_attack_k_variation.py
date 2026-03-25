import csv
import datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
import torch
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from datetime import datetime
from enum import Enum
from huggingface_hub import login
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, PrecisionRecallDisplay, average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from dotenv import load_dotenv
from datasets import load_dataset

from batch_main_vary_context_window import batch_encoder
from batch_decoder_vary_context_window import BatchWatermarkDecoder
from ecc.ciphertext import McEliece

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, auc, precision_recall_curve, PrecisionRecallDisplay
# GREEN = "\033[92m"
# RED = "\033[91m"
# RESET = "\033[0m"
GREEN = ""
RED = ""
RESET = ""
load_dotenv()
login(token = os.getenv('HF_TOKEN'))

class EncDecMethod(Enum):
    STANDARD = 'Standard'
    RANDOM = 'Random'
    NEXT = 'Next'

MODEL_NAMES = ['gpt2', 'gpt2-medium', "meta-llama/Llama-3.2-1B", 'mistralai/Mistral-7B-v0.1', "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k", "meta-llama/Meta-Llama-3-8B"]

#----------------USER INPUT VARIABLES BEGIN------------------
BATCH_SIZE = 1
CRYPTO_SCHEME = 'RANDOM' # ['McEliece', 'Ciphertext']
MAX_TOKENS = 1000
ENC_DEC_METHOD = EncDecMethod.STANDARD.value
MODEL = MODEL_NAMES[1]
KMEANS_MODEL = "kmeans_model3.pkl"  # Path to the KMeans model file
print('Model used was ', MODEL)
window_size = 3

def load_alpaca_dataset():
    """Load Alpaca evaluation set with outputs"""

    alpaca_dataset = load_dataset("reciprocate/alpaca-eval", trust_remote_code=True, split='train')
    # Get both instructions and outputs
    prompts = [x['prompt'] for x in alpaca_dataset]
    return prompts

alpaca_prompts = load_alpaca_dataset()
PROMPTS = alpaca_prompts

PROMPTS = PROMPTS[:10]
# print(len(PROMPTS))

MESSAGES = [
['1']*20,
] * len(PROMPTS)

HASH_SCHEME = 'kmeans' # ['hashlib', 'kmeans']    

#First entry of each batch table will be printed
#----------------USER INPUT VARIABLES END------------------

def sanitize_filename(name):
    """Sanitize a string to be safe for use in filenames"""
    # Replace all problematic characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove any consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized
actual_model_fr = None


def decode_watermarked_text(decoder, watermarked_predictions, expected_codeword, instance):
    decoded_results = decoder.batch_decode( [r["prompt"] for r in results], [r["generated_text"] for r in results], batch_size=BATCH_SIZE)
    for i in range(0, len(results), BATCH_SIZE):
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoded_results[i]['extracted_bits'], decoded_results[i]['extracted_indices'], decoded_results[i]['sampled_tokens'], decoded_results[i]['token_probs'], decoded_results[i]['threshold_values'], decoded_results[i]['prob_starts']
        codeword = [0]*128
        ext_idx_bit_map = defaultdict(list)
        for i in range(len(extracted_bits)):
            if extracted_bits[i] != '?':
                ext_idx_bit_map[extracted_indices[i]].append(extracted_bits[i])
        print("Decoded index and their bits", ext_idx_bit_map)
        recovered_message, recovered_codeword, distance = instance.decode(codeword, ext_idx_bit_map, fixed_codeword=expected_codeword)
        print("Distance: ", distance)
        watermarked_predictions.append(1 - distance)
        print("Watermarked predictions", watermarked_predictions)

    return watermarked_predictions




def decode_no_prompt_text(decoder, no_prompt_predictions, codeword, instance):
    decoded_results = decoder.batch_decode( ["" for r in results], [r["generated_text"] for r in results], batch_size=BATCH_SIZE)
    for i in range(0, len(results), BATCH_SIZE):
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoded_results[i]['extracted_bits'], decoded_results[i]['extracted_indices'], decoded_results[i]['sampled_tokens'], decoded_results[i]['token_probs'], decoded_results[i]['threshold_values'], decoded_results[i]['prob_starts']
        codeword = [0]*128
        ext_idx_bit_map = defaultdict(list)
        for i in range(len(extracted_bits)):
            if extracted_bits[i] != '?':
                ext_idx_bit_map[extracted_indices[i]].append(extracted_bits[i])
        print("Decoded index and their bits", ext_idx_bit_map)
        recovered_message, recovered_codeword, distance = instance.decode(codeword, ext_idx_bit_map, fixed_codeword=expected_codeword)
        print("Distance: ", distance)
        no_prompt_predictions.append(1 - distance)
        print("No Prompt predictions", no_prompt_predictions)

    return no_prompt_predictions
    
 

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # generate watermarked text
    results, actual_model = batch_encoder(PROMPTS, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE, enc_method = ENC_DEC_METHOD, messages=MESSAGES, model_name = MODEL, crypto_scheme = CRYPTO_SCHEME, hash_scheme=HASH_SCHEME)
    encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en, t_enc, probs_start_enc = results[0]['encoded_bits'], results[0]['encoded_indices'], results[0]['generated_text'], results[0]['sampled_tokens'], results[0]['token_probs'], results[0]['t_values'], results[0]['prob_starts']
    enc_idx_bit_map = defaultdict(list)
    for i in range(len(encoded_bits)):
        if encoded_bits[i]  != '?':
            enc_idx_bit_map[encoded_bit_indices[i]].append(encoded_bits[i])
    print("Encoded index and their bits", enc_idx_bit_map)

    decoder = BatchWatermarkDecoder(actual_model, message=MESSAGES, dec_method=ENC_DEC_METHOD, model_name = MODEL, crypto_scheme=CRYPTO_SCHEME, hash_scheme=HASH_SCHEME)
    print("Expected codeword: ", results[0]['codeword'])
    expected_codeword, instance = results[0]['codeword']
    watermarked_predictions = []
    no_prompt_predictions = [] 
    watermarked_predictions = decode_watermarked_text(decoder,watermarked_predictions, expected_codeword, instance)
    no_prompt_predictions = decode_no_prompt_text(decoder, no_prompt_predictions, expected_codeword, instance)
    
    
    print("Watermarked predictions: ", watermarked_predictions)
    print("no prompt predictions: ", no_prompt_predictions)

    all_predictions = watermarked_predictions + no_prompt_predictions
    all_ground_truth = [1] * len(watermarked_predictions) + [0] * len(no_prompt_predictions)
    print("All predictions: ", all_predictions)
    print("All ground truth: ", all_ground_truth)
    
    # Get precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(all_ground_truth, all_predictions, pos_label=1)
    auc_score = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve for No Prompt Attack (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Model: ' + MODEL + ' - Crypto Scheme: ' + CRYPTO_SCHEME + ' - Enc Method: ' + ENC_DEC_METHOD)
    plt.legend()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Sanitize all components for filename safety
    safe_model_name = sanitize_filename(MODEL)
    safe_crypto_scheme = sanitize_filename(CRYPTO_SCHEME)
    safe_enc_method = sanitize_filename(ENC_DEC_METHOD)
    filename = f'precision_recall_curve_no_prompt_attack_model_mistral_crypto_scheme_{safe_crypto_scheme}_enc_method_{safe_enc_method}_{timestamp}.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved plot to: {filename}')
    plt.close()




    