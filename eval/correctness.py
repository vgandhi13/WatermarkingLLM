import numpy as np
from sklearn.metrics import precision_score, recall_score
from typing import List, Dict
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from enum import Enum
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_main import batch_encoder
from batch_decoder import BatchWatermarkDecoder
from unwatermarked_samp import batch_encoder as batch_unencoder
import datasets

from huggingface_hub import login
token = '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'
login(token = token)

class EncDecMethod(Enum):
    STANDARD = 'Standard'
    RANDOM = 'Random'
    NEXT = 'Next'

MODEL_NAMES = ['gpt2', 'gpt2-medium', "meta-llama/Llama-3.1-1B",'ministral/Ministral-3b-instruct', "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k", "meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.2-3B-Instruct"]

BATCH_SIZE = 3
MAX_TOKENS = 100
CRYPTO_SCHEME = 'Ciphertext' # ['McEliece', 'Ciphertext']
ENC_DEC_METHOD = EncDecMethod.STANDARD.value
MODEL = MODEL_NAMES[-1]
print('Model used was ', MODEL)

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

PROMPTS = [
        "Tell me about artificial intelligence"]

MESSAGES = [
    'asteroid',
] * len(PROMPTS)



        
        
def detect_watermark(results, actual_model) -> List[int]:
    try:
        decoder = BatchWatermarkDecoder(actual_model, message=MESSAGES, dec_method=ENC_DEC_METHOD, model_name=MODEL, crypto_scheme=CRYPTO_SCHEME)
        decoded_results = decoder.batch_decode(
            [r["prompt"] for r in results],
            [r["generated_text"] for r in results],
            batch_size=BATCH_SIZE
        )
        
        print("decoded results")
        print([x['watermark_detected'] for x in decoded_results])
        for x, y in zip(results, decoded_results):
            print("Prompt: ", x["prompt"])
            print("Generated Text: ", x["generated_text"])
            print("Detection Score: ", y["watermark_detected"])
        return [1 if x['watermark_detected'] >= 0.5 else 0 for x in decoded_results]
    except Exception as e:
        print(f"Detection error: {e}")
        return [0]*len(results)

def generate_watermarked_text() -> str:
    results, actual_model = batch_encoder(PROMPTS, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE, enc_method = ENC_DEC_METHOD, messages=MESSAGES, model_name = MODEL, crypto_scheme = CRYPTO_SCHEME) 
    print("watermarked")
    #for i in range(len(results)):
        #results[i]['prompt'] = results[i]['prompt'][57:-55]
        #results[i]['generated_text'] = results[i]['generated_text'][4+len(results[i]['prompt'])+9:]

    return results, actual_model

def generate_unwatermarked_text():
    outputs = batch_unencoder(PROMPTS, model_name=MODEL, max_tokens=100, batch_size=4)
    #for i in range(len(outputs)):
        #outputs[i]['prompt'] = outputs[i]['prompt'][57:-55]
        #outputs[i]['genereated_text'] = outputs[i]['generated_text'][4+len(outputs[i]['prompt'])+9:]
    return outputs

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

def main():
    
    print("Generating watermarked texts...")
    watermarked_results, actual_model = generate_watermarked_text()
    watermarked_predictions = detect_watermark(watermarked_results, actual_model)
    
    print("Generating unwatermarked texts...")
    unwatermarked_results = generate_unwatermarked_text()
    unwatermarked_predictions = detect_watermark(unwatermarked_results, actual_model)
    
    
    
    all_predictions = watermarked_predictions + unwatermarked_predictions
    all_ground_truth = [1]*len(watermarked_predictions) + [0]*len(unwatermarked_predictions)
    
    metrics = calculate_metrics(all_ground_truth, all_predictions)
    
    print("\nWatermarking Detection Metrics:")
    print("-" * 50)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"True Positive Rate (recall) (TPR): {metrics['tpr']:.4f}")
    print(f"True Negative Rate (TNR): {metrics['tnr']:.4f}")
    print(f"False Positive Rate (FPR): {metrics['fpr']:.4f}")
    print(f"False Negative Rate (FNR): {metrics['fnr']:.4f}")

if __name__ == "__main__":
    main()