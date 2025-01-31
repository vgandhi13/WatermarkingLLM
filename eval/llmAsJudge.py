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
        # Sample 3 reference texts for comparison
        reference_texts = sample(self.evaluation_data, 3)
        
        prompt = f"""
        Please evaluate the following text by comparing it to high-quality reference texts.
        Consider these aspects:
        1. Relevancy (1-10): How well does it follow instructions and stay on topic
        2. Coherence (1-10): Logical flow, clarity, and organization
        3. Informativeness (1-10): Depth and usefulness of information provided
        4. {self.model_type.lower() == "alpaca" and "Factuality" or "Interestingness"} (1-10): {self.model_type.lower() == "alpaca" and "Accuracy of facts and claims" or "Engagement and creativity"}

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
                temperature=0.3
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

def compare_watermarked_pairs(original_texts: List[str], watermarked_texts: List[str]) -> Dict:
    """Compare quality metrics between original and watermarked text pairs"""
    results = {
        "relevancy_diff": [],
        "coherence_diff": [],
        "informativeness_diff": [],
        "factuality_interestingness_diff": [],
        "overall_quality_impact": []
    }
    
    judge = LLMJudge(os.getenv("OPENAI_API_KEY"))
    
    for orig, water in zip(original_texts, watermarked_texts):
        orig_eval = judge.evaluate_text(orig)
        water_eval = judge.evaluate_text(water)
        
        if orig_eval and water_eval:
            for metric in ["relevancy", "coherence", "informativeness", "factuality_interestingness"]:
                diff = orig_eval[metric] - water_eval[metric]
                results[f"{metric}_diff"].append(diff)
            
            overall_diff = sum(results[k][-1] for k in results.keys() if k != "overall_quality_impact") / 4
            results["overall_quality_impact"].append(overall_diff)
    
    return results

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
        
        # Generate watermarked versions
        watermarked_texts = []
        for prompt in prompts[:3]:  # Test first 3 prompts to save API calls
            codeword = scheme.encode("Watermark")
            watermarked_texts.append(watermarker(prompt, codeword))
        
        # Compare quality
        results = compare_watermarked_pairs(prompts[:3], watermarked_texts)
        
        # Print results
        print(f"\nResults for {scheme_name}:")
        print(f"Average Relevancy Impact: {sum(results['relevancy_diff'])/len(results['relevancy_diff']):.2f}")
        print(f"Average Coherence Impact: {sum(results['coherence_diff'])/len(results['coherence_diff']):.2f}")
        print(f"Average Informativeness Impact: {sum(results['informativeness_diff'])/len(results['informativeness_diff']):.2f}")
        print(f"Average {'Factuality' if model_type.lower() == 'alpaca' else 'Interestingness'} Impact: {sum(results['factuality_interestingness_diff'])/len(results['factuality_interestingness_diff']):.2f}")
        print(f"Overall Quality Impact: {sum(results['overall_quality_impact'])/len(results['overall_quality_impact']):.2f}")

if __name__ == "__main__":
    main()
