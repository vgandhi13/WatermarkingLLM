import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib
# from ecc.mceliece import McEliece

class BatchWatermarkDecoder:
    def __init__(self, actual_model, message, dec_method, model_name):
        self.dec_method = dec_method
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer1 = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer2 = AutoTokenizer.from_pretrained(model_name, padding_side='right')
        self.tokenizer1.pad_token = self.tokenizer1.eos_token  # Use the EOS token as the padding token
        self.tokenizer2.pad_token = self.tokenizer2.eos_token
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to(self.device)
        # self.model.eval()
        self.model = actual_model
        self.model.eval()

        # For demonstration purposes:
        # In production implementation, use:
        # codeword = McEliece().encrypt(message.encode('utf-8'))[0]
        # self.expected_codeword = ''.join(format(byte, '08b') for byte in codeword)
        self.expected_codeword = ''.join(['1']*2040)
        
    
    def batch_decode(self, prompts, generated_texts, batch_size=1):
        """
        Decode watermarks from multiple texts simultaneously.
        
        Args:
            prompts: List of original prompts used for generation
            generated_texts: List of texts to analyze for watermarks
            batch_size: Number of texts to process simultaneously
        
        Returns:
            List of dictionaries containing decoded watermark information
        """
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_texts = generated_texts[i:i+batch_size]
            actual_batch_size = len(batch_prompts)
            
            # Tokenize prompts and generated texts together with padding
            batch_inputs = self.tokenizer1(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            # batch_full_inputs = self.tokenizer([p + g for p, g in zip(batch_prompts, batch_texts)], return_tensors="pt", padding=True).to(self.device)
            batch_full_inputs = self.tokenizer2(batch_texts, return_tensors="pt", padding=True).to(self.device)
            batch_full_inputs = torch.cat((batch_inputs.input_ids, batch_full_inputs.input_ids), dim = 1)
            # print('Decpder IDs ',batch_inputs.input_ids)

            with torch.no_grad():
                outputs = self.model(batch_full_inputs)
                logits = outputs.logits[:, :-1, :]  # Ignore last token

            batch_results = []
            for b in range(actual_batch_size):
                result = self.decode(batch_prompts[b], batch_texts[b], batch_inputs.input_ids[b], batch_full_inputs[b], logits[b])
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def decode(self, prompt, generated_text, prompt_input_ids, input_ids, logits):
        """
        Decode watermark from a single text.
        
        Args:
            prompt: Original prompt used for generation
            generated_text: Text to analyze for watermark
        
        Returns:
            Tuple containing extracted bits, indices, sampled tokens, and probabilities
        """
        
        # Initialize context window (previous 3 tokens)
        prev_tokens = ["<empty>", "<empty>", "<empty>"]
        prev_decoded_indices = set()
        for i in range(max(0, len(prompt_input_ids) - 3), len(prompt_input_ids)):
            token = prompt_input_ids[i]
            prev_tokens.append(self.tokenizer1.decode(token.item()))
        prev_tokens = prev_tokens[-3:]
        
        # Initialize result trackers
        extracted_bits = []
        extracted_indices = []
        sampled_tokens = []
        token_sampled_probs = []
        t_ext = []
        probs_start = []
        
        # Threshold for bit determination
        t = 0.5
        bit = 'unassigned'
        question_mark = ''
        
        # Start decoding after the prompt tokens
        prompt_length = len(prompt_input_ids)
        for i in range(prompt_length - 1, len(logits)):

            if question_mark != '?':
                # Determine the bit index based on previous tokens
                idx = new_index = int(hashlib.md5("".join(prev_tokens).encode()).hexdigest(), 16) % len(self.expected_codeword)
                #Next Bit Approach
                if self.dec_method == 'Next':
                    while new_index in prev_decoded_indices:
                        new_index += 1
                        new_index = new_index % len(self.expected_codeword)
                        if new_index == idx:
                            print("Error: All bits were Decoded")
                            exit()
                    prev_decoded_indices.add(new_index)
                index = new_index
            # Update context window
            prev_tokens = prev_tokens[1:] + [self.tokenizer1.decode(input_ids[i + 1].item())]
            
            # Get probability distribution
            sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            
            # Find position of the actual next token
            next_token_id = input_ids[i + 1].item()
            next_token_position = (sorted_indices == next_token_id).nonzero(as_tuple=True)[0].item()
            
            # Get probability information
            prob_of_next_token = cumulative_probs[next_token_position].item()
            prob_of_start_of_token = 0 if prob_of_next_token == cumulative_probs[0].item() else cumulative_probs[next_token_position - 1].item()
            
            # Track threshold
            if i != (len(logits) - 1):
                t_ext.append(t)
            
            if prob_of_next_token < t:
                bit = '1'
                question_mark = ''
                t = 0.5
            elif prob_of_start_of_token > t:
                bit = '0'
                question_mark = ''
                t = 0.5
            else:
                question_mark = '?'
                bit = '?'
                t = (t - prob_of_start_of_token) / (prob_of_next_token - prob_of_start_of_token)
            
            # Record information
            if i != (len(logits) - 1):
                extracted_bits.append(bit)
                extracted_indices.append(index)
                sampled_tokens.append(question_mark + self.tokenizer1.decode(next_token_id))
                token_sampled_probs.append(prob_of_next_token)
                probs_start.append(prob_of_start_of_token)
        
        return {
            "extracted_bits": extracted_bits,
            "extracted_indices": extracted_indices,
            "sampled_tokens": sampled_tokens,
            "token_probs": token_sampled_probs,
            "threshold_values": t_ext,
            "prob_starts": probs_start,
            "expected_codeword": self.expected_codeword,
            "watermark_detected": self.evaluate_watermark(extracted_bits, extracted_indices)
        }
    
    def evaluate_watermark(self, extracted_bits, extracted_indices):
        """
        Evaluate if the extracted bits match the expected watermark.
        
        Returns a confidence score between 0.0 and 1.0
        """
        if not extracted_bits:
            return 0.0
        
        # Count matches
        matches = 0
        total_bits = 0
        
        for bit, index in zip(extracted_bits, extracted_indices):
            if bit != '?':  # Skip uncertain bits
                expected_bit = self.expected_codeword[index]
                if bit == expected_bit:
                    matches += 1
                total_bits += 1
        
        # Calculate confidence
        return matches / total_bits if total_bits > 0 else 0.0

# Example usage
if __name__ == '__main__':
    decoder = BatchWatermarkDecoder()
    
    prompts = ["Once upon a time", "The quick brown fox"]
    generated_texts = [", there was a magical kingdom.", " jumped over the lazy dog."]
    
    results = decoder.batch_decode(prompts, generated_texts)
    
    for i, result in enumerate(results):
        print(f"\nText {i+1}: {prompts[i]}{generated_texts[i]}")
        print(f"Watermark detected: {result['watermark_detected']:.2f} confidence")
        print(f"Extracted bits: {result['extracted_bits'][:10]}...")
