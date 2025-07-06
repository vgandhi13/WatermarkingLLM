import csv
import datasets
import matplotlib.pyplot as plt
import numpy as np
import os
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

from batch_main_vary_context_window import batch_encoder
from batch_decoder_vary_context_window import BatchWatermarkDecoder
from ecc.ciphertext import Ciphertext
from ecc.mceliece import McEliece
from unwatermarked_samp import batch_encoder as batch_unencoder

# GREEN = "\033[92m"
# RED = "\033[91m"
# RESET = "\033[0m"
GREEN = ""
RED = ""
RESET = ""
login(token = os.getenv("HF_TOKEN"))

class EncDecMethod(Enum):
    STANDARD = 'Standard'
    RANDOM = 'Random'
    NEXT = 'Next'

MODEL_NAMES = ['gpt2', 'gpt2-medium', "meta-llama/Llama-3.2-1B", 'ministral/Ministral-3b-instruct', "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k", "meta-llama/Meta-Llama-3-8B"]

#----------------USER INPUT VARIABLES BEGIN------------------
BATCH_SIZE = 2
CRYPTO_SCHEME = 'Ciphertext' # ['McEliece', 'Ciphertext']
MAX_TOKENS = 200
ENC_DEC_METHOD = EncDecMethod.STANDARD.value
MODEL = MODEL_NAMES[0]
KMEANS_MODEL = "kmeans_model_160.pkl"  # Path to the KMeans model file
print('Model used was ', MODEL)
window_size = 5

def load_alpaca_dataset():
    """Load Alpaca evaluation set with outputs"""
    alpaca_dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", trust_remote_code=True)
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

# PROMPTS = PROMPTS[:100]
# print(len(PROMPTS))

MESSAGES = [
    'Asteroid',
] * len(PROMPTS)

HASH_SCHEME = 'kmeans' # ['hashlib', 'kmeans']    

#First entry of each batch table will be printed
#----------------USER INPUT VARIABLES END------------------
watermarked_predictions =  []
actual_model_fr = None

def generate_watermarked_text():
    global watermarked_texts, watermarked_recalls
    watermarked_texts = []
    watermarked_recalls = []
    
    print("Starting watermarking process...")
    results, actual_model = batch_encoder(
        PROMPTS,
        max_tokens=MAX_TOKENS,
        batch_size=BATCH_SIZE,
        enc_method = ENC_DEC_METHOD,
        messages=MESSAGES,
        model_name = MODEL,
        crypto_scheme = CRYPTO_SCHEME,
        hash_scheme=HASH_SCHEME,
        kmeans_model_path=KMEANS_MODEL,
        window_size=window_size)
    
    print(f"Generated {len(results)} watermarked texts")
    
    decoder = BatchWatermarkDecoder(
        actual_model,
        message=MESSAGES,
        dec_method=ENC_DEC_METHOD,
        model_name = MODEL,
        crypto_scheme=CRYPTO_SCHEME,
        hash_scheme=HASH_SCHEME,
        kmeans_model_path=KMEANS_MODEL,
        window_size=window_size)
    
    print("Starting decoding process...")
    decoded_results = decoder.batch_decode(
            ["" for r in results],
            [r["generated_text"] for r in results],
            batch_size=BATCH_SIZE)
    
    print(f"Decoded {len(decoded_results)} texts")

    for i in range(0, len(results), BATCH_SIZE):
        watermarked_texts.append(results[i]['generated_text'])

        encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en, t_enc, probs_start_enc = results[i]['encoded_bits'], results[i]['encoded_indices'], results[i]['generated_text'], results[i]['sampled_tokens'], results[i]['token_probs'], results[i]['t_values'], results[i]['prob_starts']
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoded_results[i]['extracted_bits'], decoded_results[i]['extracted_indices'], decoded_results[i]['sampled_tokens'], decoded_results[i]['token_probs'], decoded_results[i]['threshold_values'], decoded_results[i]['prob_starts']
        
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}:")
        print(f"Number of encoded bits: {len(encoded_bits)}")
        print(f"Number of extracted bits: {len(extracted_bits)}")
        
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

        enc_idx_bit_map = defaultdict(list)
        ext_idx_bit_map = defaultdict(list)

        for j in range(len(encoded_bits)):
            if encoded_bits[j] != '?':
                enc_idx_bit_map[encoded_bit_indices[j]].append(encoded_bits[j])

        for j in range(len(extracted_bits)):
            if extracted_bits[j] != '?':
                ext_idx_bit_map[extracted_indices[j]].append(extracted_bits[j])
        
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

        for idx, enc_arr in enc_idx_bit_map.items():
            if idx not in ext_idx_bit_map:
                # ques - why do we not add len(enc_arr)?
                continue
            dec_arr = ext_idx_bit_map[idx]
            for j in range(len(enc_arr)):
                if j >= len(dec_arr):
                    break
                if enc_arr[j] == dec_arr[j]:
                    matches += 1
            num_enc_bits += len(enc_arr)
            num_dec_bits += len(dec_arr)
        
        print(f"Matches: {matches}, Encoded bits: {num_enc_bits}, Decoded bits: {num_dec_bits}")
        precision = matches / num_dec_bits if num_dec_bits > 0 else 0
        recall = matches / num_enc_bits if num_enc_bits > 0 else 0
        watermarked_predictions.append(precision)
        print("Watermarked predictions", watermarked_predictions)
        watermarked_recalls.append(recall)
        print("Precision is ", precision)
        print("Recall is ", recall)

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

    all_predictions = watermarked_predictions + unwatermarked_predictions
    all_ground_truth = [1] * len(watermarked_predictions) + [0] * len(unwatermarked_predictions)
    ap = average_precision_score(all_ground_truth, all_predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'tpr': tpr,
        'tnr': tnr,
        'fpr': fpr,
        'fnr': fnr,
        'ap': ap
    }

unwatermarked_predictions =  []
def generate_unwatermarked_text():
    global unwatermarked_texts, unwatermarked_recalls
    unwatermarked_texts = []
    unwatermarked_recalls = []
    
    results = batch_unencoder(
        PROMPTS,
        max_tokens=MAX_TOKENS,
        batch_size=BATCH_SIZE,
        model_name = MODEL)
    print("UNWATERMARKED TEXT GENERATED")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_model_fr = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto").to(device)
    actual_model_fr.eval()
    decoder = BatchWatermarkDecoder(
        actual_model_fr,
        message=MESSAGES,
        dec_method=ENC_DEC_METHOD,
        model_name = MODEL,
        crypto_scheme=CRYPTO_SCHEME,
        hash_scheme=HASH_SCHEME,
        kmeans_model_path=KMEANS_MODEL,
        window_size=window_size)
            
    print("UNWATERMARKED TEXT DETECTED")
    decoded_results = decoder.batch_decode(
            ["" for r in results],
            [r["generated_text"] for r in results],
            batch_size=BATCH_SIZE
        )

    for i in range(0, len(results), BATCH_SIZE):
        unwatermarked_texts.append(results[i]['generated_text'])

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

        enc_idx_bit_map = defaultdict(list)
        ext_idx_bit_map = defaultdict(list)

        if CRYPTO_SCHEME == 'McEliece':
            codeword = McEliece().encrypt(MESSAGES[0].encode('utf-8'))[0]
            codeword = ''.join(format(byte, '08b') for byte in codeword)
            #codeword = '100110'
        elif CRYPTO_SCHEME == 'Ciphertext':
            ciphertext = Ciphertext()
            codeword = ciphertext.encrypt(100)

        encoded_bits = [c for c in codeword]
        encoded_bit_indices = [i for i in range(len(encoded_bits))]

        for i in range(len(encoded_bits)):
            if encoded_bits[i] != '?':
                enc_idx_bit_map[encoded_bit_indices[i]].append(encoded_bits[i])

        for i in range(len(extracted_bits)):
            if extracted_bits[i] != '?':
                ext_idx_bit_map[extracted_indices[i]].append(extracted_bits[i])

        matches = 0
        num_enc_bits = 0
        num_dec_bits = 0

        for i, enc_arr in enc_idx_bit_map.items():
            if i not in ext_idx_bit_map:
                continue
            dec_arr = ext_idx_bit_map[i]
            for j in range(len(enc_arr)):
                if j >= len(dec_arr):
                    break
                if enc_arr[j] == dec_arr[j]:
                    matches += 1
            num_enc_bits += len(enc_arr)
            num_dec_bits += len(dec_arr)
        
        print(matches, num_enc_bits, num_dec_bits)
        unwatermarked_predictions.append((matches/num_dec_bits if num_dec_bits>0 else 0))
        print("Unwatermarked predictions", unwatermarked_predictions)
        unwatermarked_recalls.append((matches/num_enc_bits if num_enc_bits>0 else 0))
        print("Precision is ", (matches/num_dec_bits if num_dec_bits>0 else 0))
        print("Recall is ", (matches/num_enc_bits if num_enc_bits>0 else 0))

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


def run_experiment(crypto_scheme, enc_method, kmeans_model_path):
    global CRYPTO_SCHEME, ENC_DEC_METHOD, KMEANS_MODEL
    CRYPTO_SCHEME = crypto_scheme
    ENC_DEC_METHOD = enc_method
    KMEANS_MODEL = kmeans_model_path
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create CSV file for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'no_prompt_attack_{kmeans_model_path}_{crypto_scheme}_{enc_method}_{timestamp}.csv'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Type', 'Text', 'Precision', 'Recall'])
            
    actual_model_fr = generate_watermarked_text()
    generate_unwatermarked_text()
    
    # Save watermarked texts
    with open(csv_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i, (text, precision) in enumerate(zip(watermarked_texts, watermarked_predictions)):
            csvwriter.writerow(['Watermarked', text, precision, watermarked_recalls[i] if i < len(watermarked_recalls) else 'N/A'])
    
    # Save unwatermarked texts
    with open(csv_filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i, (text, precision) in enumerate(zip(unwatermarked_texts, unwatermarked_predictions)):
            csvwriter.writerow(['Unwatermarked', text, precision, unwatermarked_recalls[i] if i < len(unwatermarked_recalls) else 'N/A'])
    
    print("Watermarked predictions: ", watermarked_predictions)
    print("Unwatermarked predictions: ", unwatermarked_predictions)

    all_predictions = watermarked_predictions + unwatermarked_predictions
    all_ground_truth = [1] * len(watermarked_predictions) + [0] * len(unwatermarked_predictions)
    
    # Check if we have any predictions
    if not all_predictions or not all_ground_truth:
        print(f"No predictions generated for {crypto_scheme} - {enc_method}")
        return [0], [0], {'precision': 0, 'recall': 0, 'tpr': 0, 'tnr': 0, 'fpr': 0, 'fnr': 0}
    
    # Get precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(all_ground_truth, all_predictions, pos_label=1)
    ap = average_precision_score(all_ground_truth, all_predictions)

    # Compute F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # add small value to avoid division by zero

    # Find the best threshold
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5  # thresholds is len-1 of precision/recall
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best F1: {best_f1:.3f}")
    print(f"Precision at best F1: {best_precision:.3f}")
    print(f"Recall at best F1: {best_recall:.3f}")

    # For reporting metrics at the best threshold:
    binary_predictions = [1 if p >= best_threshold else 0 for p in all_predictions]
    print("Binary predictions: ", binary_predictions)
    print("Ground truth: ", all_ground_truth)
    metrics = calculate_metrics(all_ground_truth, binary_predictions)

    print(f"\nResults have been saved to {csv_filename}")
    return precision, recall, metrics

def main():
    # Define configurations
    configs = [
        ('McEliece', EncDecMethod.STANDARD.value),          
        ('McEliece', EncDecMethod.RANDOM.value),
        ('McEliece', EncDecMethod.NEXT.value),
        ('Ciphertext', EncDecMethod.STANDARD.value),
        ('Ciphertext', EncDecMethod.RANDOM.value)
    ]
    
    # Colors for different configurations
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    kmeans_model_paths = ['kmeans_model_160_n5.pkl', 'kmeans_model_250_n5.pkl', 'kmeans_model_540_n5.pkl', 'kmeans_model_1020_n5.pkl', 'kmeans_model_2040_n5.pkl']
    
    # Plot setup
    plt.figure(figsize=(10, 8))
    
    # Run experiments and plot
    for kmeans_model_path in kmeans_model_paths:
        for (crypto_scheme, enc_method), color in zip(configs, colors):
            print(f"\nRunning experiment with {crypto_scheme} and {enc_method} encoding and kmeans model {kmeans_model_path}...")
            precision, recall, metrics = run_experiment(crypto_scheme, enc_method, kmeans_model_path)
                
            # Only plot if we have real data (not just [0])
            if len(precision) > 1 and len(recall) > 1:
                plt.plot(recall, precision, color=color, linewidth=2,
                    label=f'{crypto_scheme} - {enc_method}\n'
                    f'Precision: {metrics["precision"]:.4f}\n'
                    f'Recall: {metrics["recall"]:.4f}\n'
                    f'TPR: {metrics["tpr"]:.4f}\n'
                    f'FPR: {metrics["fpr"]:.4f}\n'
                    f'AP: {metrics["ap"]:.2f})')
            else:
                print(f"Skipping plot for {crypto_scheme} - {enc_method} - {kmeans_model_path} due to lack of data.")
        
        # Customize plot
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves for Different Configurations\nNo Prompt Attack', fontsize=14)
        
        # Enhanced legend
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
            fontsize=10, frameon=True, framealpha=0.9,
            edgecolor='black', fancybox=True)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save plot with extra space for legend
        plt.savefig(f'PR_no_prompt_{kmeans_model_path}.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    main()