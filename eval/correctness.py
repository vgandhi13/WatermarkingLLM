import openai
import os
import sys
import random
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
import numpy as np
from collections import Counter
from enum import Enum
import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, auc, precision_recall_curve, PrecisionRecallDisplay

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
MAX_TOKENS = 1000
ENC_DEC_METHOD = EncDecMethod.STANDARD.value
HASH_SCHEME = 'kmeans' # ['hashlib', 'kmeans']
MODEL = MODEL_NAMES[1]
print('Model used was ', MODEL)
KMEANS_MODEL = "kmeans_model3.pkl"  # Path to the KMeans model file
print('KMeans model used was: ', KMEANS_MODEL)
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

PROMPTS = PROMPTS[:2]

# Add the missing MESSAGES variable
MESSAGES = [
    'Asteroid',
] * len(PROMPTS)

#First entry of each batch table will be printed
#----------------USER INPUT VARIABLES END------------------
watermarked_predictions =  []
actual_model_fr = None
def generate_watermarked_text():
    results, actual_model = batch_encoder(PROMPTS, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE, enc_method = ENC_DEC_METHOD, messages=MESSAGES, model_name = MODEL, crypto_scheme = CRYPTO_SCHEME, hash_scheme=HASH_SCHEME)

    actual_model_fr = actual_model
    decoder = BatchWatermarkDecoder(actual_model, message=MESSAGES, dec_method=ENC_DEC_METHOD, model_name = MODEL, crypto_scheme=CRYPTO_SCHEME, hash_scheme=HASH_SCHEME)
    decoded_results = decoder.batch_decode(
    [r["prompt"] for r in results],
    [r["generated_text"] for r in results],
    batch_size=BATCH_SIZE
    )

    for i in range(0, len(results), BATCH_SIZE):

        encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en, t_enc, probs_start_enc = results[i]['encoded_bits'], results[i]['encoded_indices'], results[i]['generated_text'], results[i]['sampled_tokens'], results[i]['token_probs'], results[i]['t_values'], results[i]['prob_starts']
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoded_results[i]['extracted_bits'], decoded_results[i]['extracted_indices'], decoded_results[i]['sampled_tokens'], decoded_results[i]['token_probs'], decoded_results[i]['threshold_values'], decoded_results[i]['prob_starts']
        
        bits_match = encoded_bits == extracted_bits
        indices_match = encoded_bit_indices == extracted_indices
        tokens_match = sampled_tokens_en == sampled_tokens_dec
        probs_match = token_sampled_probs_en == token_sampled_probs_dec

        print(f"Encoded bits match extracted bits: {GREEN if bits_match else RED}{bits_match}{RESET}")
        print(f"Encoded bit indices match extracted indices: {GREEN if indices_match else RED}{indices_match}{RESET}")

        print('Num of encoded bits', len(encoded_bits))
        print('Num of decoded bits', len(extracted_bits))
        # Count matching bits and indices
        matching_bits = sum(1 for i in range(min(len(extracted_bits), len(encoded_bits))) if extracted_bits[i] == encoded_bits[i])
        matching_indices = sum(1 for i in range(min(len(extracted_indices), len(encoded_bit_indices))) if extracted_indices[i] == encoded_bit_indices[i])

        print(f"Columnwise Matching bits: {matching_bits} / {len(extracted_bits)}")
        print(f"Columnwise Matching indices: {matching_indices} / {len(extracted_indices)}")

        '''
        create hashmap with index bit mapping for encoded, then for extracted check if those indices are present if they are check if bits match
        '''

        enc_idx_bit_map = defaultdict(list)
        ext_idx_bit_map = defaultdict(list)


        for i in range(len(encoded_bits)):
            if encoded_bits[i]  != '?':
                enc_idx_bit_map[encoded_bit_indices[i]].append(encoded_bits[i])

        for i in range(len(extracted_bits)):
            if extracted_bits[i] != '?':
                ext_idx_bit_map[extracted_indices[i]].append(extracted_bits[i])
        
        print("Encoded index and their bits", enc_idx_bit_map)
        print("Decoded index and their bits", ext_idx_bit_map)

        not_decoded = 0
        bit_not_same = 0
        for k, v in ext_idx_bit_map.items():
            if k not in enc_idx_bit_map:
                not_decoded += 1
            else:
                if v != enc_idx_bit_map[k]:
                    bit_not_same+=1
        
        print('Indices which were encoded but not decoded ',not_decoded)
        print('Indices were bits decoded dont match bits encoded', bit_not_same)

        matches = 0
        num_enc_bits = 0
        num_dec_bits = 0
        # calculate how many indices are failign to be sent 
        set_of_indices = set(range(128))
        set_of_indices = set_of_indices - set(enc_idx_bit_map.keys())
        print('Indices which were not even encoded', set_of_indices)
        print('Indices which were not decoded or encoded', set_of_indices-set(ext_idx_bit_map.keys()))
        for i, enc_arr in enc_idx_bit_map.items(): #change the decoding
            if i not in ext_idx_bit_map:
                break
            dec_arr = ext_idx_bit_map[i]
            for j in range(len(enc_arr)):
                if j>= len(dec_arr):
                    break
                if enc_arr[j] == dec_arr[j]:
                    matches += 1
            num_enc_bits += len(enc_arr)
            num_dec_bits += len(dec_arr)
        # print(matches, num_enc_bits, num_dec_bits)
        watermarked_predictions.append(matches/num_dec_bits if num_dec_bits>0 else 0)
        if CRYPTO_SCHEME == 'Ciphertext':
            # Create ciphertext instance for decoding
            ciphertext = Ciphertext()
            codeword = [0]*128
            for i in range(len(codeword)):
                codeword[i] = random.randint(0, 1)
            for i, ext_arr in ext_idx_bit_map.items():
                counts = Counter(ext_arr)
                most_common_element = counts.most_common(1)[0][0]
                codeword[i] = most_common_element
                
            print("Codeword: ", "".join(str(bit) for bit in codeword))
            print("Decoded codeword: ", ciphertext.decrypt("".join(str(bit) for bit in codeword)))
            if ciphertext.decrypt("".join(str(bit) for bit in codeword)) == 'A':
                print("Codeword decoded correctly")
            print("Precision is ", matches/num_dec_bits)
            print("Recall is ", matches/num_enc_bits)
        else:
            # need to just check if the codeword decodes correctly.
            pass
            



        # Print unified table if there's a mismatch
        if not (bits_match and indices_match and tokens_match and probs_match):
            print("\n⚠️ Mismatch Detected! Side-by-Side Comparison Table:\n")
            print(f"{'Idx':<5} | {'EBit':<3} | {'ExBit':<3} | {'EIdx':<3} | {'ExIdx':<3} | "
                f"{'Encoded Token':<20} | {'Extracted Token':<20} | "
                f"{'Encoded Prob':<12} | {'Extracted Prob':<12} | "
                f"{'Enc Start Prob':<14} | {'Ext Start Prob':<14} | "
                f"{'Enc T':<8} | {'Ext T':<8}")
            print("-" * 150)

            min_len = min(len(encoded_bits), len(extracted_bits), len(encoded_bit_indices), len(extracted_indices), 
                        len(sampled_tokens_en), len(sampled_tokens_dec), len(token_sampled_probs_en), len(token_sampled_probs_dec))

            for i in range(min_len):
                e_bit = encoded_bits[i] if i < len(encoded_bits) else "N/A"
                x_bit = extracted_bits[i] if i < len(extracted_bits) else "N/A"
                e_idx = encoded_bit_indices[i] if i < len(encoded_bit_indices) else "N/A"
                x_idx = extracted_indices[i] if i < len(extracted_indices) else "N/A"
                e_tok = sampled_tokens_en[i] if i < len(sampled_tokens_en) else "N/A"
                x_tok = sampled_tokens_dec[i] if i < len(sampled_tokens_dec) else "N/A"
                e_prob = token_sampled_probs_en[i] if i < len(token_sampled_probs_en) else "N/A"
                x_prob = token_sampled_probs_dec[i] if i < len(token_sampled_probs_dec) else "N/A"
                e_start_prob = probs_start_enc[i] if i < len(probs_start_enc) else "N/A"
                x_start_prob = probs_start_ext[i] if i < len(probs_start_ext) else "N/A"
                e_t = t_enc[i] if i < len(t_enc) else "N/A"
                x_t = t_ext[i] if i < len(t_ext) else "N/A"

                # Highlight mismatches
                e_bit = f"{RED}{e_bit}{RESET}" if e_bit != x_bit else e_bit
                e_idx = f"{RED}{e_idx}{RESET}" if e_idx != x_idx else e_idx
                e_tok = f"{RED}{e_tok}{RESET}" if e_tok != x_tok else e_tok
                e_prob = f"{RED}{e_prob:.6f}{RESET}" if e_prob != x_prob else f"{e_prob:.6f}"

                print(f"{i:<5} | {e_bit:<3} | {x_bit:<3} | {e_idx:<3} | {x_idx:<3} | "
                    f"{e_tok:<20} | {x_tok:<20} | {e_prob:<12} | {x_prob:<12.6f} | "
                    f"{e_start_prob:<14} | {x_start_prob:<14.6f} | {e_t:<8} | {x_t:<8.6f}")

            print("\n" + "-" * 150 + "\n")


        
    return actual_model
    
    
    


def calculate_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # same as tpr
    
    return {
        'precision': precision,
        'recall': recall,
        'tpr': tpr,
        'tnr': tnr,
        'fpr': fpr,
        'fnr': fnr
    }

unwatermarked_predictions =  []
def generate_unwatermarked_text(actual_model_fr):
    results = batch_unencoder(PROMPTS, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE, model_name = MODEL)
    print("UNWATERMARKED TEXT GENERATED")

    decoder = BatchWatermarkDecoder(actual_model_fr, message=MESSAGES, dec_method=ENC_DEC_METHOD, model_name = MODEL, crypto_scheme=CRYPTO_SCHEME, hash_scheme=HASH_SCHEME)
    print("UNWATERMARKED TEXT DETECTED")
    decoded_results = decoder.batch_decode(
    [r["prompt"] for r in results],
    [r["generated_text"] for r in results],
    batch_size=BATCH_SIZE
    )

    for i in range(0, len(results), BATCH_SIZE):

        encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en, t_enc, probs_start_enc = [], [], results[i]['generated_text'], [], [], [], []
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoded_results[i]['extracted_bits'], decoded_results[i]['extracted_indices'], decoded_results[i]['sampled_tokens'], decoded_results[i]['token_probs'], decoded_results[i]['threshold_values'], decoded_results[i]['prob_starts']
        
        bits_match = encoded_bits == extracted_bits
        indices_match = encoded_bit_indices == extracted_indices
        tokens_match = sampled_tokens_en == sampled_tokens_dec
        probs_match = token_sampled_probs_en == token_sampled_probs_dec

        print(f"Encoded bits match extracted bits: {GREEN if bits_match else RED}{bits_match}{RESET}")
        print(f"Encoded bit indices match extracted indices: {GREEN if indices_match else RED}{indices_match}{RESET}")

        print('Num of encoded bits', len(encoded_bits))
        print('Num of decoded bits', len(extracted_bits))
        # Count matching bits and indices
        matching_bits = sum(1 for i in range(min(len(extracted_bits), len(encoded_bits))) if extracted_bits[i] == encoded_bits[i])
        matching_indices = sum(1 for i in range(min(len(extracted_indices), len(encoded_bit_indices))) if extracted_indices[i] == encoded_bit_indices[i])

        print(f"Columnwise Matching bits: {matching_bits} / {len(extracted_bits)}")
        print(f"Columnwise Matching indices: {matching_indices} / {len(extracted_indices)}")

        '''
        create hashmap with index bit mapping for encoded, then for extracted check if those indices are present if they are check if bits match
        '''

        enc_idx_bit_map = defaultdict(list)
        ext_idx_bit_map = defaultdict(list)
        if CRYPTO_SCHEME == 'McEliece':
            codeword = McEliece().encrypt(MESSAGES[0].encode('utf-8'))[0]
            codeword = ''.join(format(byte, '08b') for byte in codeword)
            #codeword = '100110'
        elif CRYPTO_SCHEME == 'Ciphertext':
            ciphertext = Ciphertext()
            codeword = ciphertext.encrypt('Asteroid')
        encoded_bits = [c for c in codeword]
        encoded_bit_indices = [i for i in range(len(encoded_bits))]

        for i in range(len(encoded_bits)):
            if encoded_bits[i]  != '?':
                enc_idx_bit_map[encoded_bit_indices[i]].append(encoded_bits[i])

        for i in range(len(extracted_bits)):
            if extracted_bits[i] != '?':
                ext_idx_bit_map[extracted_indices[i]].append(extracted_bits[i])
        
        

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
        
        print(matches, num_enc_bits, num_dec_bits)
        unwatermarked_predictions.append((matches/num_dec_bits if num_dec_bits>0 else 0))
        print("Precision is ", (matches/num_dec_bits if num_dec_bits>0 else 0))
        print("Recall is ", (matches/num_enc_bits if num_enc_bits>0 else 0))

        # Print unified table if there's a mismatch
        
            
actual_model_fr = generate_watermarked_text()
generate_unwatermarked_text(actual_model_fr)
all_predictions = watermarked_predictions + unwatermarked_predictions
print(all_predictions)
all_ground_truth = [1]*len(watermarked_predictions) + [0]*len(unwatermarked_predictions)
metrics = calculate_metrics(all_ground_truth, all_predictions)


# get ground truth and predictiosn from paraphrase

all_predictions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9288702928870293, 1.0, 1.0, 0.9551020408163265, 1.0, 0.9612068965517241, 0.9748743718592965, 1.0, 1.0, 
0.9789029535864979, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9441624365482234, 0.9879032258064516, 1.0, 1.0, 1.0, 0.995475113122172, 1.0, 1.0, 1.0, 0.6696035242290749, 
0.7244444444444444, 0.7342995169082126, 0.592274678111588, 0.6172248803827751, 0.5971563981042654, 0.7112068965517241, 0.6529411764705882, 0.7105263157894737, 
0.714859437751004, 0.6417112299465241, 0.676595744680851, 0.648936170212766, 0.6180257510729614, 0.6373626373626373, 0.6936936936936937, 0.7096774193548387, 0.7, 
0.7123287671232876, 0.6824644549763034, 0.7009345794392523, 0.7616580310880829, 0.6262626262626263, 0.7711864406779662, 0.72, 0.6695652173913044, 0.776255707762557, 
0.6274509803921569, 0.6181818181818182, 0.7587939698492462] 
all_ground_truth = [1]*len(all_predictions) + [0]*len(unwatermarked_predictions)
all_predictions = all_predictions + unwatermarked_predictions
precision, recall, _ = precision_recall_curve(all_ground_truth, all_predictions)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
disp.plot().figure_.savefig('precision_recall_curve_ciphertext_next.png')

    
print("\nWatermarking Detection Metrics:")
print("-" * 50)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"True Positive Rate (recall) (TPR): {metrics['tpr']:.4f}")
print(f"True Negative Rate (TNR): {metrics['tnr']:.4f}")
print(f"False Positive Rate (FPR): {metrics['fpr']:.4f}")
print(f"False Negative Rate (FNR): {metrics['fnr']:.4f}")