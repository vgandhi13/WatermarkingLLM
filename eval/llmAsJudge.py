import openai
import json
from typing import List, Dict
import os
import pandas as pd
from ecc.reed_solomon import ReedSolomonCode
from ecc.permuted_reed_solomon import PermutedReedSolomon
from ecc.mceliece import McEliece
import datasets
from random import sample
import matplotlib.pyplot as plt
import numpy as np

class LLMJudge:
    def __init__(self, api_key: str, model_type: str = "alpaca"):
        openai.api_key = api_key
        self.model_type = model_type
        self.evaluation_data = self.load_evaluation_dataset()
        
    def load_evaluation_dataset(self):
        """Load and sample 200 prompts based on model type"""
        if self.model_type.lower() == "alpaca":
            # Load Alpaca evaluation set
            dataset = datasets.load_dataset("tatsu-lab/alpaca_eval")
            eval_prompts = dataset['eval']['instruction']
            # Add word limit to each prompt
            eval_prompts = [f"{prompt} Please limit your response to 300 words." for prompt in eval_prompts]
        else:
            # Load C4 dataset news-like subset
            dataset = datasets.load_dataset("c4", "en", streaming=True)
            # Filter for news-like content and add word limit
            news_texts = [text for text in dataset['train'] 
                         if text.get('url', '').endswith(('.com', '.org', '.net'))]
            eval_prompts = [f"{text['text']} Please limit your response to 300 words." for text in news_texts]
        
        # Sample 200 prompts randomly
        sampled_prompts = sample(eval_prompts, 200)
        return sampled_prompts
        
    async def evaluate_text(self, text: str) -> Dict:
        """
        Evaluate text using GPT-4 based on sampled reference texts
        """
        reference_texts = sample(self.evaluation_data, 3)
        
        prompt = f"""
        Please evaluate the following text by comparing it to high-quality reference texts.
        Consider these aspects:
        1. Relevancy (1-10): How well does it follow instructions and stay on topic
        2. Coherence (1-10): Logical flow, clarity, and organization
        3. Informativeness (1-10): Depth and usefulness of information provided
        4. Factuality (1-10): Accuracy of facts and claims
        5. Interestingness (1-10): Engagement and creativity
        6. Overall Quality (1-10): Holistic assessment of the text's quality

        Provide a concise evaluation in JSON format with scores and brief explanations.
        
        Reference texts for comparison:
        {' '.join(reference_texts)}
        
        Text to evaluate:
        {text}
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

def compare_watermarked_pairs(original_texts: List[str], watermarked_texts: List[str], unwatermarked_texts: List[str]) -> Dict:
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
    
    for orig, water, unwater in zip(original_texts, watermarked_texts, unwatermarked_texts):
        orig_eval = judge.evaluate_text(orig)
        water_eval = judge.evaluate_text(water)
        unwater_eval = judge.evaluate_text(unwater)
        
        if orig_eval and water_eval and unwater_eval:
            metrics = ["relevancy", "coherence", "informativeness", "factuality", "interestingness", "overall_quality"]
            
            # Compare watermarked
            for metric in metrics:
                water_diff = orig_eval[metric] - water_eval[metric]
                results["watermarked"][f"{metric}_diff"].append(water_diff)
            
            water_overall = sum(results["watermarked"][k][-1] for k in [f"{m}_diff" for m in metrics]) / len(metrics)
            results["watermarked"]["overall_quality_impact"].append(water_overall)
            
            # Compare unwatermarked
            for metric in metrics:
                unwater_diff = orig_eval[metric] - unwater_eval[metric]
                results["unwatermarked"][f"{metric}_diff"].append(unwater_diff)
            
            unwater_overall = sum(results["unwatermarked"][k][-1] for k in [f"{m}_diff" for m in metrics]) / len(metrics)
            results["unwatermarked"]["overall_quality_impact"].append(unwater_overall)
    
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

def main():
    # Get prompts from correctness.py
    from eval.correctness import prompts
    
    # Initialize different watermarking schemes
    rs_code = ReedSolomonCode(255, 223)
    prs_code = PermutedReedSolomon(255, 223)
    mce_code = McEliece(255, 223)
    
    watermarking_schemes = {
        "Reed-Solomon": rs_code,
        "Permuted Reed-Solomon": prs_code,
        "McEliece": mce_code
    }
    
    # Test each watermarking scheme
    for scheme_name, scheme in watermarking_schemes.items():
        print(f"\nTesting {scheme_name} watermarking:")
        
        # Generate watermarked and unwatermarked versions
        watermarked_texts = []
        unwatermarked_texts = []
        for prompt in prompts[:3]:  # Test first 3 prompts to save API calls
            codeword = scheme.encode("Watermark")
            watermarked = watermarker(prompt, codeword)
            watermarked_texts.append(watermarked)
            unwatermarked_texts.append(unwatermarker(watermarked, codeword))
        
        # Compare quality
        results = compare_watermarked_pairs(prompts[:3], watermarked_texts, unwatermarked_texts)
        
        # Print results
        for version in ["watermarked", "unwatermarked"]:
            print(f"\nResults for {scheme_name} ({version}):")
            print(f"Average Relevancy Impact: {sum(results[version]['relevancy_diff'])/len(results[version]['relevancy_diff']):.2f}")
            print(f"Average Coherence Impact: {sum(results[version]['coherence_diff'])/len(results[version]['coherence_diff']):.2f}")
            print(f"Average Informativeness Impact: {sum(results[version]['informativeness_diff'])/len(results[version]['informativeness_diff']):.2f}")
            print(f"Average Factuality Impact: {sum(results[version]['factuality_diff'])/len(results[version]['factuality_diff']):.2f}")
            print(f"Average Interestingness Impact: {sum(results[version]['interestingness_diff'])/len(results[version]['interestingness_diff']):.2f}")
            print(f"Average Overall Quality Impact: {sum(results[version]['overall_quality_diff'])/len(results[version]['overall_quality_diff']):.2f}")
            print(f"Overall Impact Across All Metrics: {sum(results[version]['overall_quality_impact'])/len(results[version]['overall_quality_impact']):.2f}")
        
        # Generate plots
        plot_quality_metrics(results, scheme_name)

if __name__ == "__main__":
    main()
