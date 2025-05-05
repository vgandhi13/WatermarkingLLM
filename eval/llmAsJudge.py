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
from ecc.reed_solomon import ReedSolomonCode
from ecc.permuted_reed_solomon import PermutedReedSolomon
from ecc.mceliece import McEliece
import datasets
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import asyncio
from dotenv import load_dotenv
from batch_main import batch_encoder
from unwatermarked_samp import batch_encoder as batch_unencoder


class LLMJudge:
    def __init__(self, api_key: str, model_type: str = "both"):
        self.model_type = model_type
        openai.api_key_path = "openai_key.txt"
        # Don't load datasets in init, load them when needed
        
    def load_alpaca_dataset(self):
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
            time.sleep(20)
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

async def compare_watermarked_pairs(prompts_and_outputs: List[Dict], watermarked_texts: List[str], unwatermarked_texts: List[str]) -> Dict:
    """Compare quality metrics between original, watermarked, and unwatermarked text pairs"""
    results = {
        "watermarked": {
            "relevancy_diff": [],
            "coherence_diff": [],
            "informativeness_diff": [],
            "factuality_diff": [],
            "interestingness_diff": [],
            "overall_quality_diff": [],
            "overall_quality_impact": []
        },
        "unwatermarked": {
            "relevancy_diff": [],
            "coherence_diff": [],
            "informativeness_diff": [],
            "factuality_diff": [],
            "interestingness_diff": [],
            "overall_quality_diff": [],
            "overall_quality_impact": []
        }
    }
    
    judge = LLMJudge(os.getenv("OPENAI_API_KEY"))
    
    for prompt_data, water, unwater in zip(prompts_and_outputs, watermarked_texts, unwatermarked_texts):
        print("\nEvaluating new text pair...")
        
        baseline_eval = await judge.evaluate_text(prompt_data['expected_output'], prompt_data['expected_output'])
        if not baseline_eval:
            print("Failed to evaluate baseline text, skipping pair")
            continue
        
        water_eval = await judge.evaluate_text(water, prompt_data['expected_output'])
        if not water_eval:
            print("Failed to evaluate watermarked text, skipping pair")
            continue
            
        unwater_eval = await judge.evaluate_text(unwater, prompt_data['expected_output'])
        if not unwater_eval:
            print("Failed to evaluate unwatermarked text, skipping pair")
            continue
        
        print("Successfully evaluated text pair")
        
        if baseline_eval and water_eval and unwater_eval:
            print_results(baseline_eval, water_eval, unwater_eval, "Watermarking", "Current Dataset")
            
            # Still collect the differences for plotting
            metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness", "overall_quality"]
            for metric in metrics:
                results["watermarked"][f"{metric}_diff"].append(baseline_eval[metric] - water_eval[metric])
                results["unwatermarked"][f"{metric}_diff"].append(baseline_eval[metric] - unwater_eval[metric])
    
    return results

def plot_quality_metrics(results: Dict, scheme_name: str):
    """Plot quality metrics for watermarked and unwatermarked texts"""
    metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness"]
    
    # Set up the plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot watermarked metrics
    water_values = [-sum(results["watermarked"][f"{m}_diff"])/len(results["watermarked"][f"{m}_diff"]) for m in metrics]
    x = np.arange(len(metrics))
    ax1.bar(x, water_values)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.set_title(f'Quality Impact on Watermarked Text\n{scheme_name}')
    ax1.set_ylabel('Impact (negative means quality decrease)')
    
    # Plot unwatermarked metrics
    unwater_values = [-sum(results["unwatermarked"][f"{m}_diff"])/len(results["unwatermarked"][f"{m}_diff"]) for m in metrics]
    ax2.bar(x, unwater_values)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45)
    ax2.set_title(f'Quality Impact on Unwatermarked Text\n{scheme_name}')
    ax2.set_ylabel('Impact (negative means quality decrease)')
    
    # Plot overall quality comparison
    water_overall = -sum(results["watermarked"]["overall_quality_impact"])/len(results["watermarked"]["overall_quality_impact"])
    unwater_overall = -sum(results["unwatermarked"]["overall_quality_impact"])/len(results["unwatermarked"]["overall_quality_impact"])
    
    ax3.bar(['Watermarked', 'Unwatermarked'], [water_overall, unwater_overall], 
            color=['blue', 'orange'])
    ax3.set_title(f'Overall Quality Impact Comparison\n{scheme_name}')
    ax3.set_ylabel('Impact (negative means quality decrease)')
    
    plt.tight_layout()
    plt.savefig(f'quality_metrics_{scheme_name.lower().replace("-", "_")}.png')
    plt.close()

async def main():
    try:
        
        # Create judge instance
        judge = LLMJudge(os.getenv("OPENAI_API_KEY"))
        
        # Run tests on Alpaca dataset first
        print("\nTesting with Alpaca dataset:")
        alpaca_prompts = judge.load_alpaca_dataset()
        alpaca_promptsfr = [x['prompt'] for x in alpaca_prompts]
        print(type(alpaca_prompts))
        print(alpaca_prompts[0])
            
        watermarked_texts = []
        unwatermarked_texts = []
        results, actual_model = batch_encoder(alpaca_promptsfr, max_tokens=300, batch_size=5, enc_method = 'Standard', messages=["asteroid"]*len(alpaca_promptsfr), model_name = "gpt2-medium") 
        print(results[0])
        watermarked_texts.extend([result['generated_text'] for result in results])
        outputs = batch_unencoder(alpaca_promptsfr, model_name="meta-llama/Llama-3.2-3B", max_tokens=100, batch_size=4)
        unwatermarked_texts.extend([output['generated_text'] for output in outputs])
        
        
        
        # Compare quality
        results = await compare_watermarked_pairs(alpaca_prompts, 
                                                watermarked_texts, 
                                                unwatermarked_texts)
        
        if not results["watermarked"]["relevancy_diff"]:
            print(f"No successful evaluations for Alpaca")

        
        #  C4 dataset
        print("\nTesting with C4 dataset:")
        c4_prompts = judge.load_c4_dataset(limit=1000)
        c4_promptsfr = [x['prompt'] for x in c4_prompts]
        print(f"\nTesting watermarking on C4:")
            
        watermarked_texts = []
        unwatermarked_texts = []
        results, actual_model = batch_encoder(c4_promptsfr[:50], max_tokens=300, batch_size=5, enc_method = 'Standard', messages=["asteroid"]*50, model_name = "gpt2-medium") 
        watermarked_texts.append([result['generated_text'] for result in results])
        for p in c4_promptsfr:
            unwatermarked_texts.append(encoder(prompt=p, model_name="meta-llama/Llama-3.2-3B"))
        
        # Compare quality
        results = await compare_watermarked_pairs(c4_prompts[:50], 
                                                watermarked_texts, 
                                                unwatermarked_texts)
        
        if not results["watermarked"]["relevancy_diff"]:
            print(f"No successful evaluations for c4")
            
            
        # Generate plots
        plot_quality_metrics(results, "alpaca")
        plot_quality_metrics(results, "c4")
            

    except Exception as e:
        print(f"Error in main: {str(e)}")

def print_results(baseline_eval: Dict, water_eval: Dict, unwater_eval: Dict, scheme_name: str, dataset_name: str):
    """Helper function to print raw quality scores"""
    metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness", "overall_quality"]
    
    print(f"\nQuality Scores for {scheme_name} on {dataset_name}:")
    print("\nExpected Output Scores:")
    for metric in metrics:
        print(f"{metric.capitalize()}: {baseline_eval[metric]:.2f}")
    
    print("\nWatermarked Text Scores:")
    for metric in metrics:
        print(f"{metric.capitalize()}: {water_eval[metric]:.2f}")
        
    print("\nUnwatermarked Text Scores:")
    for metric in metrics:
        print(f"{metric.capitalize()}: {unwater_eval[metric]:.2f}")

if __name__ == "__main__":
    print("pls work")
    asyncio.run(main())