import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib
from ecc.mceliece import McEliece
from ecc.ciphertext import Ciphertext
from collections import defaultdict

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import joblib
from gensim.models import FastText
import time

#from ecc.ciphertext import Ciphertext
# KEY = b'=\x0fs\xf1q\xccQ\x9fhi\xa7\x89\x8f\xc5#\xbf'

class BatchWatermarkDecoder:
    def __init__(self, actual_model, message, dec_method, model_name, crypto_scheme):
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
        start = time.time()
        self.fasttext_model = FastText.load_fasttext_format("cc.en.300.bin")
        end = time.time()
        elapsed = end - start
        print("Time taken to load fastext model in decoder: ", elapsed, "seconds")
        start = time.time()
        self.kmeans_model = joblib.load("kmeans_model3.pkl")
        end = time.time()
        elapsed = end - start
        print("Time taken to load kmeans_model in decoder: ", elapsed, "seconds")

        # For demonstration purposes:
        # In production implementation, use:
        # codeword = McEliece().encrypt(message[0].encode('utf-8'))[0]
        # self.expected_codeword = ''.join(format(byte, '08b') for byte in codeword)

        self.messages = message
        self.crypto_scheme = crypto_scheme
        
    
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
            batch_messages = self.messages[i:i+batch_size]
            actual_batch_size = len(batch_prompts)
            
            # Tokenize prompts and generated texts together with padding
            batch_inputs = self.tokenizer1(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            # batch_full_inputs = self.tokenizer([p + g for p, g in zip(batch_prompts, batch_texts)], return_tensors="pt", padding=True).to(self.device)
            batch_full_inputs = self.tokenizer2(batch_texts, return_tensors="pt", padding=True, add_special_tokens = False).to(self.device)
            batch_full_inputs = torch.cat((batch_inputs.input_ids, batch_full_inputs.input_ids), dim = 1)
            # print('Decpder IDs ',batch_inputs.input_ids)

            with torch.no_grad():
                outputs = self.model(batch_full_inputs)
                logits = outputs.logits[:, :-1, :]  # Ignore last token

            batch_results = []
            for b in range(actual_batch_size):
                if self.crypto_scheme == 'McEliece':
                    codeword = McEliece().encrypt(batch_messages[b].encode('utf-8'))[0]
                    codeword = ''.join(format(byte, '08b') for byte in codeword)
                    #codeword = '100110'
                elif self.crypto_scheme == 'Ciphertext':
                    ciphertext = Ciphertext()
                    codeword = ciphertext.encrypt(100)
                    self.encoded_bits = [c for c in codeword]
                    self.encoded_bit_indices = [i for i in range(len(self.encoded_bits))]
                result = self.decode(batch_prompts[b], batch_texts[b], batch_inputs.input_ids[b], batch_full_inputs[b], logits[b], codeword)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def decode(self, prompt, generated_text, prompt_input_ids, input_ids, logits, codeword):
        """
        Decode watermark from a single text.
        
        Args:
            prompt: Original prompt used for generation
            generated_text: Text to analyze for watermark
        
        Returns:
            Tuple containing extracted bits, indices, sampled tokens, and probabilities
        """
        

        '''  
        OLD APPROACH
        # Initialize context window (previous 3 tokens)
        prev_tokens = ["<empty>", "<empty>", "<empty>"]
        prev_decoded_indices = set()
        for i in range(max(0, len(prompt_input_ids) - 3), len(prompt_input_ids)):
            token = prompt_input_ids[i]
            prev_tokens.append(self.tokenizer1.decode(token.item()))
        prev_tokens = prev_tokens[-3:]
        '''
        prev_decoded_indices = set()
        # Initialize last_twenty_tokens and prev3_generated_tokens
        last_twenty_tokens = []

        # Fill last_twenty_tokens with decoded prompt tokens
        for token in prompt_input_ids:
            last_twenty_tokens.append(self.tokenizer1.decode(token.item()))

        # Cap at 20 tokens
        if len(last_twenty_tokens) > 20:
            last_twenty_tokens = last_twenty_tokens[-20:]

        # Merge tokens and tokenize into words
        merged_string = ''.join(last_twenty_tokens)
        # Find the last complete word boundary (last space)
        last_space_idx = merged_string.rfind(' ')

        if last_space_idx != -1:
            merged_string = merged_string[:last_space_idx]  # Cut off incomplete word
        else:
            merged_string = ''  # No spaces found, so treat as empty (no complete words)

        words = word_tokenize(merged_string)

        # Initialize prev3_generated_tokens (last 3 words, pad with '.')
        prev_tokens = ['.'] * (3 - len(words)) + words[-3:]

        print('Decoder last 3 words: ', prev_tokens)

        
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
                # idx = new_index = int(hashlib.md5("".join(prev_tokens).encode()).hexdigest(), 16) % len(self.expected_codeword)
                embeddings = [self.fasttext_model.wv[token] for token in prev_tokens]
                avg_embedding = sum(embeddings) / len(embeddings)
                            # Reshape to (1, -1) since predict expects a 2D array
                avg_embedding = avg_embedding.reshape(1, -1)
                # Hash the context to get the bit index
                # idx = int(hashlib.md5("".join(prev_tokens).encode()).hexdigest(), 16) % len(self.codewords[b])
                idx = new_index = self.kmeans_model.predict(avg_embedding)[0] % len(codeword)
                print('Index from Kmeans - decoder', idx)

                #Next Bit Approach
                if self.dec_method == 'Next':
                    while new_index in prev_decoded_indices:
                        new_index += 1
                        new_index = new_index % len(codeword)
                        if new_index == idx:
                            print("Error: All bits were Decoded")
                            exit()
                    prev_decoded_indices.add(new_index)
                index = new_index
            '''            
            OLD APPROACH
            # Update context window
            prev_tokens = prev_tokens[1:] + [self.tokenizer1.decode(input_ids[i + 1].item())]'''
            # Decode the new token
            decoded_token = self.tokenizer1.decode(input_ids[i + 1].item())

            # Update last_twenty_tokens
            last_twenty_tokens.append(decoded_token)

            # Cap last_twenty_tokens to 20 tokens
            if len(last_twenty_tokens) > 20:
                last_twenty_tokens = last_twenty_tokens[-20:]

            # Merge tokens into a string
            merged_string = ''.join(last_twenty_tokens)
                        # Find the last complete word boundary (last space)
            last_space_idx = merged_string.rfind(' ')

            if last_space_idx != -1:
                merged_string = merged_string[:last_space_idx]  # Cut off incomplete word
            else:
                merged_string = ''  # No spaces found, so treat as empty (no complete words)
            words = word_tokenize(merged_string)

            # Initialize prev3_generated_tokens (last 3 words, pad with '.')
            prev_tokens = ['.'] * (3 - len(words)) + words[-3:]

            print('Decoder last 3 words: ', prev_tokens)
            
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
        self.extracted_bits = extracted_bits
        self.extracted_indices = extracted_indices
        self.sampled_tokens = sampled_tokens
        self.token_probs = token_sampled_probs
        self.threshold_values = t_ext
        self.prob_starts = probs_start
        
        return {
            "extracted_bits": extracted_bits,
            "extracted_indices": extracted_indices,
            "sampled_tokens": sampled_tokens,
            "token_probs": token_sampled_probs,
            "threshold_values": t_ext,
            "prob_starts": probs_start,
            "expected_codeword": codeword,
            "watermark_detected": self.evaluate_watermark(extracted_bits, extracted_indices, codeword) if self.crypto_scheme ==  'Ciphertext' else None
        }
    
    def evaluate_watermark(self, extracted_bits, extracted_indices, codeword):
        # should return precsion 

        encoded_bits, encoded_bit_indices = self.encoded_bits, self.encoded_bit_indices
        extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = self.extracted_bits, self.extracted_indices, self.sampled_tokens, self.token_probs, self.threshold_values, self.prob_starts
    
        bits_match = encoded_bits == extracted_bits
        indices_match = encoded_bit_indices == extracted_indices


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
        print("Precision is ", matches/num_dec_bits)
        print("Recall is ", matches/num_enc_bits)
        return matches/num_dec_bits
   




# # Example usage
# if __name__ == '__main__':
#     decoder = BatchWatermarkDecoder()
    
#     prompts = ["Once upon a time", "The quick brown fox"]
#     generated_texts = [", there was a magical kingdom.", " jumped over the lazy dog."]
    
#     results = decoder.batch_decode(prompts, generated_texts)
    
#     for i, result in enumerate(results):
#         print(f"\nText {i+1}: {prompts[i]}{generated_texts[i]}")
#         print(f"Watermark detected: {result['watermark_detected']:.2f} confidence")
#         print(f"Extracted bits: {result['extracted_bits'][:10]}...")
