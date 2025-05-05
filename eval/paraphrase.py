import openai
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from typing import Optional
from dotenv import load_dotenv
import asyncio
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')  # Download the punkt tokenizer data
from datasets import load_dataset
from batch_main import batch_encoder
from batch_decoder import BatchWatermarkDecoder

class Paraphraser:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Paraphraser with an OpenAI API key.
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
        """
        load_dotenv()
        openai.api_key_path = "openai_key.txt"

        # Add translation pipelines
        self.en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-fr")
        self.fr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-fr-en")
        self.en_ru = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
        self.ru_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

    async def paraphrase(self, text: str) -> str:
        """
        Paraphrase the given text using GPT-3.5-turbo.
        
        Args:
            text: The text to paraphrase
        
        Returns:
            str: The paraphrased text
        """
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at paraphrasing text while maintaining its original meaning."},
                    {"role": "user", "content": f"Paraphrase this text while maintaining its meaning:\n\n{text}"}
                ],
                temperature=1,
                max_tokens=1000
            )
            
            # Add delay to respect rate limits
            time.sleep(1)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error during paraphrasing: {str(e)}")
            return None

    async def paraphrase_sentence(self, text: str) -> str:
        """
        Split text into sentences, paraphrase each one with context, and rejoin.
        
        Args:
            text: The full text to paraphrase sentence by sentence
        
        Returns:
            str: The full text with each sentence paraphrased
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        paraphrased_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Get context (previous sentences)
            context = " ".join(sentences[:i]) if i > 0 else ""
            
            prompt = (
                "Given some previous context and a sentence "
                "following that context, paraphrase the "
                "current sentence. Only return the "
                "paraphrased sentence in your response.\n"
                f"Previous context: {context}\n"
                f"Current sentence to paraphrase: {sentence}\n"
                "Your paraphrase of the current sentence:"
            )
            
            try:
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert at paraphrasing sentences while maintaining their meaning and contextual relevance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                
                # Add delay to respect rate limits
                time.sleep(1)
                
                paraphrased = response.choices[0].message.content.strip()
                paraphrased_sentences.append(paraphrased)
                
            except Exception as e:
                print(f"Error during paraphrasing sentence {i+1}: {str(e)}")
                # If error occurs, keep original sentence
                paraphrased_sentences.append(sentence)
        
        # Rejoin sentences
        return " ".join(paraphrased_sentences)

    async def translation_attack(self, text: str, language: str = "french") -> str:
        """
        Perform a translation attack by translating text to target language and back.
        
        Args:
            text: The text to transform
            language: Target language for translation ("french" or "russian")
        
        Returns:
            str: The text after being translated to target language and back to English
        """
        try:
            if language.lower() == "french":
                # English -> French
                french = self.en_fr(text)[0]['translation_text']
                # French -> English
                result = self.fr_en(french)[0]['translation_text']
            
            elif language.lower() == "russian":
                # English -> Russian
                russian = self.en_ru(text)[0]['translation_text']
                # Russian -> English
                result = self.ru_en(russian)[0]['translation_text']
            
            else:
                raise ValueError("Unsupported language. Use 'french' or 'russian'.")
            
            return result
            
        except Exception as e:
            print(f"Error during translation attack: {str(e)}")
            return None

    async def test_on_datasets(self):
        """Test paraphrasing attacks on OpenGen and LFQA datasets separately"""
        # Test OpenGen dataset
        print("\n" + "="*50)
        print("Testing OpenGen Dataset")
        print("="*50)
        opengen = load_dataset("OpenGen")['train']
        opengen_samples = opengen.select(range(50))  # Take first 50 samples
        opengen_results = []
        
        for sample in opengen_samples:
            prompt = sample['question'] if 'question' in sample else sample['input']
            
            # Generate watermarked text using batch_encoder directly
            watermarked_results, actual_model = batch_encoder(
                [prompt]*5,     # Match batch size of 5
                max_tokens=300, 
                batch_size=5,
                messages=["asteroid"]*5,
                enc_method='Standard',  
                model_name="gpt2-medium"
            )
            
            if watermarked_results:
                watermarked_text = watermarked_results[0]["generated_text"]
                # Run attacks on the watermarked text
                attack_results = await self.run_attacks(prompt, watermarked_text, actual_model)
                opengen_results.extend(attack_results)

    async def run_attacks(self, prompt: str, watermarked_text: str, actual_model):
        """Run different paraphrasing attacks on watermarked text and check detection"""
        results = []
        
        # Create decoder for this specific test
        decoder = BatchWatermarkDecoder(
            actual_model,
            message=["asteroid"]*5, 
            dec_method='Standard', 
            model_name="gpt2-medium"
        )
        
        # 1. French translation attack
        french_result = await self.translation_attack(watermarked_text, "french")
        if french_result:
            decoded_results = decoder.batch_decode(
                [prompt],
                [french_result],
                batch_size=1
            )
            results.append(("French Translation", watermarked_text, french_result, decoded_results[0]['detected']))
            print_detection_result(watermarked_text, french_result, decoded_results[0]['detected'])
        
        # 2. Russian translation attack
        russian_result = await self.translation_attack(watermarked_text, "russian")
        if russian_result:
            decoded_results = decoder.batch_decode(
                [prompt],
                [russian_result],
                batch_size=1
            )
            results.append(("Russian Translation", watermarked_text, russian_result, decoded_results[0]['detected']))
            print_detection_result(watermarked_text, russian_result, decoded_results[0]['detected'])
        
        # 3. GPT paraphrase
        paraphrase_result = await self.paraphrase(watermarked_text)
        if paraphrase_result:
            decoded_results = decoder.batch_decode(
                [prompt],
                [paraphrase_result],
                batch_size=1
            )
            results.append(("GPT Paraphrase", watermarked_text, paraphrase_result, decoded_results[0]['detected']))
            print_detection_result(watermarked_text, paraphrase_result, decoded_results[0]['detected'])
        
        # 4. Sentence-by-sentence paraphrase
        sentence_result = await self.paraphrase_sentence(watermarked_text)
        if sentence_result:
            decoded_results = decoder.batch_decode(
                [prompt],
                [sentence_result],
                batch_size=1
            )
            results.append(("Sentence Paraphrase", watermarked_text, sentence_result, decoded_results[0]['detected']))
            print_detection_result(watermarked_text, sentence_result, decoded_results[0]['detected'])
        
        return results

def print_detection_result(original_text: str, transformed_text: str, detected: bool):
    """
    Print the detection results for a single text transformation.
    
    Args:
        original_text: The original input text
        transformed_text: The text after transformation (paraphrase/translation)
        detected: Whether the transformation was detected
    """
    print("\nDetection Results:")
    print("-" * 50)
    print(f"Original text: {original_text[:100]}...")  # Show first 100 chars
    print(f"Transformed text: {transformed_text[:100]}...")
    print(f"Detected: {'Yes' if detected else 'No'}")
    print("-" * 50)

def calculate_detection_rate(results: list) -> dict:
    """
    Calculate detection rates for each attack type
    """
    attack_stats = {}
    
    for attack_type, _, _, detected in results:
        if attack_type not in attack_stats:
            attack_stats[attack_type] = {"total": 0, "detected": 0}
        
        attack_stats[attack_type]["total"] += 1
        if detected:
            attack_stats[attack_type]["detected"] += 1
    
    # Calculate rates
    print("\nDetection Rates by Attack Type:")
    print("-" * 50)
    for attack_type, stats in attack_stats.items():
        rate = (stats["detected"] / stats["total"]) * 100
        print(f"{attack_type}:")
        print(f"Total tests: {stats['total']}")
        print(f"Detected: {stats['detected']}")
        print(f"Detection rate: {rate:.2f}%")
        print("-" * 25)
    
    return attack_stats

async def main():
    """Example usage with separate dataset testing"""
    try:
        paraphraser = Paraphraser()
        
        print("Testing paraphrasing attacks on datasets...")
        results = await paraphraser.test_on_datasets()
        
        # Print overall summary
        print("\n" + "="*50)
        print("OVERALL SUMMARY")
        print("="*50)
        print("\nOpenGen Dataset:")
        calculate_detection_rate(results["opengen"])
        print("\nLFQA Dataset:")
        calculate_detection_rate(results["lfqa"])
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 