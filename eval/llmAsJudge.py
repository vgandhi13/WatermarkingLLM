import openai
import json
from typing import List, Dict
import os
from ecc.reed_solomon import ReedSolomonCode
from ecc.permuted_reed_solomon import PermutedReedSolomon
from ecc.mceliece import McEliece

class LLMJudge:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        
    async def evaluate_text(self, text: str) -> Dict:
        """
        Evaluate text using GPT-4 based on multiple criteria
        """
        prompt = f"""
        Please evaluate the following text based on these criteria:
        1. Coherence (1-10): How well the ideas flow and connect
        2. Grammar (1-10): Correctness of grammar and syntax
        3. Natural Flow (1-10): How natural and human-like the text sounds
        4. Content Quality (1-10): Depth and value of the content
        
        Provide your evaluation in JSON format with scores and brief explanations.
        
        Text to evaluate:
        {text}
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert text evaluator. Provide honest, objective assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3 #idk what this should be set to as a default
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return None

def compare_watermarked_pairs(original_texts: List[str], watermarked_texts: List[str]) -> Dict:
    """
    Compare quality metrics between original and watermarked text pairs
    """
    results = {
        "coherence_diff": [],
        "grammar_diff": [],
        "natural_flow_diff": [],
        "content_quality_diff": [],
        "overall_quality_impact": []
    }
    
    judge = LLMJudge(os.getenv("OPENAI_API_KEY"))
    
    for orig, water in zip(original_texts, watermarked_texts):
        orig_eval = judge.evaluate_text(orig)
        water_eval = judge.evaluate_text(water)
        
        if orig_eval and water_eval:
            for metric in ["coherence", "grammar", "natural_flow", "content_quality"]:
                diff = orig_eval[metric] - water_eval[metric]
                results[f"{metric}_diff"].append(diff)
            
            # Calculate overall impact
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
        print(f"Average Coherence Impact: {sum(results['coherence_diff'])/len(results['coherence_diff']):.2f}")
        print(f"Average Grammar Impact: {sum(results['grammar_diff'])/len(results['grammar_diff']):.2f}")
        print(f"Average Natural Flow Impact: {sum(results['natural_flow_diff'])/len(results['natural_flow_diff']):.2f}")
        print(f"Average Content Quality Impact: {sum(results['content_quality_diff'])/len(results['content_quality_diff']):.2f}")
        print(f"Overall Quality Impact: {sum(results['overall_quality_impact'])/len(results['overall_quality_impact']):.2f}")

if __name__ == "__main__":
    main()
