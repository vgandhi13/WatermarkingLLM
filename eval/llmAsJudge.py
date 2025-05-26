import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from ecc.reed_solomon import ReedSolomonCode
import openai
import time
import json
from typing import List, Dict
import pandas as pd
from enum import Enum
from ecc.reed_solomon import ReedSolomonCode
from ecc.permuted_reed_solomon import PermutedReedSolomon
from ecc.mceliece import McEliece
import datasets
from random import sample
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import asyncio
from dotenv import load_dotenv
from batch_main import batch_encoder
from unwatermarked_samp import batch_encoder as batch_unencoder
import csv
from datetime import datetime


class EncDecMethod(Enum):
    STANDARD = 'Standard'
    RANDOM = 'Random'
    NEXT = 'Next'

MODEL_NAMES = ['gpt2', 'gpt2-medium',   "meta-llama/Llama-3.2-1B",'ministral/Ministral-3b-instruct', "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k", "meta-llama/Meta-Llama-3-8B"]

#----------------USER INPUT VARIABLES BEGIN------------------
BATCH_SIZE = 4
CRYPTO_SCHEME = 'McEliece' # ['McEliece', 'Ciphertext']
MAX_TOKENS = 100
HASH_SCHEME = 'kmeans' # ['hashlib', 'kmeans']
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

PROMPTS = PROMPTS[:100]

MESSAGES = [
    "Asteroid"*27 + "aaa" 
] * len(PROMPTS)

class LLMJudge:
    def __init__(self, api_key: str, model_type: str = "both"):
        self.model_type = model_type
        openai.api_key_path = "openai_key.txt"
        # Don't load datasets in init, load them when needed
        
    
    def load_c4_dataset(self, limit=1000):
        """Load C4 dataset with target outputs"""
        c4_dataset = datasets.load_dataset("allenai/c4", "en", streaming=True, trust_remote_code=True)
        prompts_and_outputs = []
        count = 0
        for text in c4_dataset['train']:
            if count >= limit:
                break
            if text.get('url', '').endswith(('.com', '.org', '.net')):
                # For C4, we'll use the full text as expected output
                # and create a summarization prompt
                prompts_and_outputs.append({
                    'prompt': f"Summarize the following text: {text['text']}",
                    'expected_output': text['text']
                })
                count += 1
        return prompts_and_outputs
    

    async def evaluate_text(self, generated_text: str, expected_output: str) -> Dict:
        """
        Evaluate text by comparing to expected output
        """
        prompt = f"""
        Please evaluate the following generated text by comparing it to the expected output.
        Consider these aspects:
        1. Relevancy (1-10): How well does it match the expected content
        2. Coherence (1-10): Logical flow, clarity, and organization
        3. Informativeness (1-10): Coverage of key points from expected output
        4. Factuality (1-10): Accuracy compared to expected output
        5. Interestingness (1-10): Engagement and creativity while maintaining accuracy
        6. Overall Quality (1-10): Holistic assessment of the match with expected output

        This is an example of a high quality response:
        AI works by using algorithms and data to mimic human intelligence. At its core, it involves training models on large datasets so they can identify patterns and make predictions. Techniques like deep learning use neural networks with many layers to process information similarly to the human brain. These models are used in tasks like language understanding, image recognition, and decision-making. As more data is processed, the AI improves through a process called learning. AI also includes fields like natural language processing, computer vision, and robotics.

        Relevancy: 10/10

        Coherence: 10/10

        Informativeness: 10/10

        Factuality: 10/10

        Interestingness: 9/10

        Overall Quality: 10/10
        This is an example of a medium quality response:
        AI is when computers are made to act smart. It uses stuff like data and algorithms to solve problems. Sometimes it learns things on its own, like figuring out what a cat looks like from pictures. AI is used in phones, websites, and robots. Some AI is really advanced and can write or talk like a person. Scientists keep making it better by feeding it more data and checking how it does. It's kind of like teaching a machine to think, but it's not really thinking like humans do.

        Relevancy: 6/10

        Coherence: 5/10 

        Informativeness: 5/10

        Factuality: 6/10

        Interestingness: 4/10

        Overall Quality: 5/10



        This is an example of a low quality response:
        AI is cool and it works with computers. You can make it do things and it's smart. It does jobs and helps. Like when you talk to your phone and it talks back, that's AI. Or when a game plays itself. It just uses codes and learns maybe. AI is like future stuff. It's everywhere. Robots have AI too. AI knows things from data. Sometimes it guesses and is right. It's not magic but close.

        Relevancy: 3/10 

        Coherence: 2/10

        Informativeness: 2/10

        Factuality: 2/10

        Interestingness: 3/10

        Overall Quality: 2/10


        Provide evaluation in this exact JSON format:
        {{
            "relevancy": score,
            "coherence": score,
            "informativeness": score,
            "factuality": score,
            "interestingness": score,
            "overall_quality": score
        }}
        
        Expected output:
        {expected_output}
        
        Generated text to evaluate:
        {generated_text}
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Compare the generated text to the expected output and respond only with the JSON format specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1
            )
            time.sleep(10)
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

async def compare_watermarked_pairs(prompts_and_outputs: List[Dict], watermarked_texts: List[str], unwatermarked_texts: List[str]) -> Dict:
    """Compare quality metrics between original, watermarked, and unwatermarked text pairs"""
    results = {
        "watermarked": {
            "relevancy": [],
            "coherence": [],
            "informativeness": [],
            "factuality": [],
            "interestingness": [],
            "overall_quality": []
        },
        "unwatermarked": {
            "relevancy": [],
            "coherence": [],
            "informativeness": [],
            "factuality": [],
            "interestingness": [],
            "overall_quality": []
        }
    }
    
    # Create CSV file for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'llm_judge_outputs_{timestamp}.csv'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        csvwriter.writerow(['Type', 'Text', 'Relevancy', 'Coherence', 'Informativeness', 
                          'Factuality', 'Interestingness', 'Overall Quality'])
    
    judge = LLMJudge(os.getenv("OPENAI_API_KEY"))
    
    for prompt_data, water, unwater in zip(prompts_and_outputs, watermarked_texts, unwatermarked_texts):
        print("\nEvaluating new text pair...")
        
        water_eval = await judge.evaluate_text(water, prompt_data['expected_output'])
        if not water_eval:
            print("Failed to evaluate watermarked text, skipping pair")
            continue
            
        unwater_eval = await judge.evaluate_text(unwater, prompt_data['expected_output'])
        if not unwater_eval:
            print("Failed to evaluate unwatermarked text, skipping pair")
            continue
        
        print("Successfully evaluated text pair")
        
        # Store raw scores
        metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness", "overall_quality"]
        for metric in metrics:
            if metric in water_eval and metric in unwater_eval:
                results["watermarked"][metric].append(water_eval[metric])
                results["unwatermarked"][metric].append(unwater_eval[metric])
        
        # Write to CSV
        with open(csv_filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write watermarked text and scores
            csvwriter.writerow(['Watermarked', water] + [water_eval.get(m, 0) for m in metrics])
            # Write unwatermarked text and scores
            csvwriter.writerow(['Unwatermarked', unwater] + [unwater_eval.get(m, 0) for m in metrics])
    
    print(f"\nResults have been saved to {csv_filename}")
    return results

def plot_quality_metrics(results: Dict, scheme_name: str):
    """Plot quality metrics for watermarked and unwatermarked texts"""
    metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness"]
    
    # Set up the plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot watermarked metrics
    water_values = [float(np.mean(results["watermarked"][m])) for m in metrics]
    x = np.arange(len(metrics))
    ax1.bar(x, water_values)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.set_title(f'Quality Scores for Watermarked Text\n{scheme_name}')
    ax1.set_ylabel('Quality Score')
    ax1.set_ylim(0, 5)  # Set y-axis limit to 5
    
    # Plot unwatermarked metrics
    unwater_values = [float(np.mean(results["unwatermarked"][m])) for m in metrics]
    ax2.bar(x, unwater_values)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45)
    ax2.set_title(f'Quality Scores for Unwatermarked Text\n{scheme_name}')
    ax2.set_ylabel('Quality Score')
    ax2.set_ylim(0, 5)  # Set y-axis limit to 5
    
    # Plot overall quality comparison
    water_overall = float(np.mean(results["watermarked"]["overall_quality"]))
    unwater_overall = float(np.mean(results["unwatermarked"]["overall_quality"]))
    
    ax3.bar(['Watermarked', 'Unwatermarked'], [water_overall, unwater_overall], 
            color=['blue', 'orange'])
    ax3.set_title(f'Overall Quality Comparison\n{scheme_name}')
    ax3.set_ylabel('Quality Score')
    ax3.set_ylim(0, 5)  # Set y-axis limit to 5
    
    plt.tight_layout()
    plt.savefig(f'quality_metrics_{scheme_name.lower().replace("-", "_")}.png')
    plt.close()

def plot_quality_comparison(results: Dict, scheme_name: str):
    """Create a side-by-side comparison of watermarked vs unwatermarked metrics"""
    metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness", "overall_quality"]
    
    # Calculate average values for each metric using raw scores
    water_values = [float(np.mean(results["watermarked"][m])) for m in metrics]
    unwater_values = [float(np.mean(results["unwatermarked"][m])) for m in metrics]
    
    # Set up the plot
    plt.figure(figsize=(15, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, water_values, width, label='Watermarked', color='blue', alpha=0.7)
    plt.bar(x + width/2, unwater_values, width, label='Unwatermarked', color='orange', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Quality Metrics')
    plt.ylabel('Quality Score')
    plt.title(f'Quality Comparison: Watermarked vs Unwatermarked Text\n{scheme_name}')
    plt.xticks(x, [m.capitalize() for m in metrics], rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits to 5
    plt.ylim(0, 5)
    
    # Add value labels on top of bars
    for i, v in enumerate(water_values):
        plt.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(unwater_values):
        plt.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'quality_comparison_{scheme_name.lower().replace("-", "_")}.png')
    plt.close()

def analyze_quality_scores(baseline_eval, water_eval, unwater_eval, scheme_name, dataset_name):
    """Analyze and plot quality scores for a single combination."""
    metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness", "overall_quality"]
    
    # Create bar plot
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, [float(np.mean(water_eval[m])) for m in metrics], width, label='Watermarked')
    rects2 = ax.bar(x + width/2, [float(np.mean(unwater_eval[m])) for m in metrics], width, label='Unwatermarked')
    
    ax.set_ylabel('Score')
    ax.set_title(f'Quality Scores - {scheme_name} on {dataset_name}')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'quality_scores_{scheme_name}_{dataset_name}.png')
    plt.close()

def plot_combined_overall_scores(all_scores):
    """Create combined plot of overall quality scores across all combinations."""
    combinations = list(all_scores.keys())
    water_values = [float(np.mean(scores['watermarked']['overall_quality'])) for scores in all_scores.values()]
    unwater_values = [float(np.mean(scores['unwatermarked']['overall_quality'])) for scores in all_scores.values()]
    
    x = np.arange(len(combinations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, water_values, width, label='Watermarked')
    rects2 = ax.bar(x + width/2, unwater_values, width, label='Unwatermarked')
    
    ax.set_ylabel('Overall Quality Score')
    ax.set_title('Overall Quality Scores Across All Combinations')
    ax.set_xticks(x)
    ax.set_xticklabels(combinations, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('combined_overall_quality.png')
    plt.close()

def print_results(baseline_eval: Dict, water_eval: Dict, unwater_eval: Dict, scheme_name: str, dataset_name: str):
    """Helper function to print raw quality scores and create plots"""
    metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness", "overall_quality"]
    
    print(f"\nQuality Scores for {scheme_name} on {dataset_name}:")
    
    print("\nWatermarked Text Scores:")
    for metric in metrics:
        avg_score = float(np.mean(water_eval[metric])) if water_eval[metric] else 0
        print(f"{metric.capitalize()}: {avg_score:.2f}")
        
    print("\nUnwatermarked Text Scores:")
    for metric in metrics:
        avg_score = float(np.mean(unwater_eval[metric])) if unwater_eval[metric] else 0
        print(f"{metric.capitalize()}: {avg_score:.2f}")
    
    # Create individual plot for this combination
    analyze_quality_scores(baseline_eval, water_eval, unwater_eval, scheme_name, dataset_name)

async def main():
    try:
        # Initialize storage for all scores
        all_scores = {}
        
        # Create judge instance
        judge = LLMJudge(os.getenv("OPENAI_API_KEY"))
        
        # Define configurations
        configs = [
            ('McEliece', EncDecMethod.STANDARD.value),
            ('McEliece', EncDecMethod.RANDOM.value),
            ('McEliece', EncDecMethod.NEXT.value)
            # ('Ciphertext', EncDecMethod.STANDARD.value),
            # ('Ciphertext', EncDecMethod.RANDOM.value)
        ]
        
        # Initialize counters for total 0s and 1s
        
        
        # Run tests on Alpaca dataset for each configuration
        for crypto_scheme, enc_method in configs:
            print(f"\nTesting with {crypto_scheme} and {enc_method} encoding:")
            total_0s = 0
            total_1s = 0
            total_bits = 0
            global CRYPTO_SCHEME, ENC_DEC_METHOD
            CRYPTO_SCHEME = crypto_scheme
            ENC_DEC_METHOD = enc_method
            
            alpaca_prompts = load_alpaca_dataset()
            alpaca_promptsfr = [x['prompt'] for x in alpaca_prompts]
            
            watermarked_texts = []
            unwatermarked_texts = []
            results, actual_model = batch_encoder(PROMPTS, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE, 
                                                enc_method=ENC_DEC_METHOD, messages=MESSAGES, 
                                                model_name=MODEL, crypto_scheme=CRYPTO_SCHEME, hash_scheme=HASH_SCHEME)
            for i in range(len(results)):
                encoded_bits, encoded_bit_indices = results[i]['encoded_bits'], results[i]['encoded_indices']
                enc_idx_bit_map = defaultdict(list)
                for j in range(len(encoded_bits)):
                    if encoded_bits[j]  != '?':
                        enc_idx_bit_map[encoded_bit_indices[j]].append(encoded_bits[j])
                batch_0s, batch_1s = 0, 0
                for j in (encoded_bit_indices):
                    if j in enc_idx_bit_map and enc_idx_bit_map[j]:  # Check if index exists and list is not empty
                        last_bit = enc_idx_bit_map[j][-1]
                        if last_bit == '0':
                            batch_0s += 1
                        if last_bit == '1':
                            batch_1s += 1
                total_0s += batch_0s
                total_1s += batch_1s
                total_bits += (batch_0s + batch_1s)
            print(enc_idx_bit_map)

            watermarked_texts.extend([result['generated_text'] for result in results])
            outputs = batch_unencoder(PROMPTS, model_name=MODEL , 
                                    max_tokens=MAX_TOKENS, batch_size=4)
            unwatermarked_texts.extend([output['generated_text'] for output in outputs])
            
            # Compare quality
            results = await compare_watermarked_pairs(alpaca_prompts, 
                                                    watermarked_texts, 
                                                    unwatermarked_texts)
            
            if not results["watermarked"]["relevancy"]:
                print(f"No successful evaluations for {crypto_scheme} - {enc_method}")
                continue
                
            # Generate plots
            scheme_name = f"{crypto_scheme}_{enc_method}"
            plot_quality_metrics(results, scheme_name)
            plot_quality_comparison(results, scheme_name)
            
            # Store scores for this combination
            combination_name = f"{crypto_scheme}_{enc_method}"
            all_scores[combination_name] = {
                'baseline': None,
                'watermarked': results["watermarked"],
                'unwatermarked': results["unwatermarked"]
            }
            
            # Print results and create individual plot
            print_results(None, results["watermarked"], results["unwatermarked"], scheme_name, "alpaca")
            print(f"\nBit Distribution for {scheme_name} and {enc_method}:")
            print(f"Average 0's: {total_0s/total_bits:.4f}")
            print(f"Average 1's: {total_1s/total_bits:.4f}")
            
        
        # Create combined plot of overall scores
        plot_combined_overall_scores(all_scores)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    print("pls work")
    asyncio.run(main())