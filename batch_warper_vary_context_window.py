import inspect
import math
from transformers import AutoTokenizer
from typing import Callable, Iterable, List, Optional, Tuple, Union
import hashlib
import pickle
from ecc.ciphertext import McEliece
import numpy as np
import torch
from ecc.random_linear_code import RandomLinearCode
import random
# Using Ciphertext class which implements OAEP-McEliece
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from custom_kmeans_wrapper import SimpleKMeansWrapper


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

class BatchTopPLogitsWarper(LogitsProcessor):
    """
    Batch-aware version of TopPLogitsWarper that can process multiple sequences simultaneously.
    """

    def _decode_tokens_with_spacing(self, token_ids):
        """
        Decode individual tokens while preserving spacing information.
        This is important for SentencePiece tokenizers like Mistral that don't
        include spaces when decoding individual tokens.
        
        Strategy: Decode consecutive pairs of tokens and extract the difference
        to infer what each token contributes, including spaces.
        
        Args:
            token_ids: List or tensor of token IDs
            
        Returns:
            List of decoded token strings with proper spacing inferred
        """
        if len(token_ids) == 0:
            return []
        
        # Convert to list if tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if len(token_ids) == 1:
            return [self.tokenizer.decode([token_ids[0]], skip_special_tokens=False)]
        
        decoded_tokens = []
        
        # Decode first token alone
        prev_decoded = self.tokenizer.decode([token_ids[0]], skip_special_tokens=False)
        decoded_tokens.append(prev_decoded)
        
        # For each subsequent token, decode it with the previous one and extract the difference
        for i in range(1, len(token_ids)):
            # Decode pair [prev, current]
            pair_decoded = self.tokenizer.decode(token_ids[i-1:i+1], skip_special_tokens=False)
            
            # Extract what the new token added (this includes any spaces)
            if pair_decoded.startswith(prev_decoded):
                new_token_text = pair_decoded[len(prev_decoded):]
            else:
                # Fallback: if prefix doesn't match, decode individually
                new_token_text = self.tokenizer.decode([token_ids[i]], skip_special_tokens=False)
            
            decoded_tokens.append(new_token_text)
            prev_decoded = pair_decoded  # Update for next iteration
        
        return decoded_tokens

    def flush_positions(self):
        """
        Returns a list of position-strings (one for each batch element),
        exactly like the old flush_position() in the original watermark class.
        """
        results = []
        for b in range(self.batch_size):
            pos_str = "".join(str(p) for p in self.sampled_token_positions[b])
            results.append(pos_str)

            # Reset after flushing (same behavior as old code)
            self.sampled_token_positions[b] = []

        return results

    def __init__(self, batch_encoded_bits,
            batch_encoded_indices, 
            batch_sampled_tokens, 
            batch_token_probs,
            batch_t_enc, 
            batch_prob_start,
            batch_messages,
            enc_method,
            input_ids,
            batch_size,
            crypto_scheme,
            fasttext_model,
            kmeans_model,
            hash_scheme = "kmeans", model_name = "gpt2", t: float = 0.5, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, window_size: int = 3):
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.batch_size = batch_size
        self.encoded_bits = batch_encoded_bits
        self.encoded_bit_indices = batch_encoded_indices
        self.sampled_tokens = batch_sampled_tokens
        self.token_sampled_probs = batch_token_probs
        self.t_enc = batch_t_enc
        self.probs_start = batch_prob_start
        self.enc_method = enc_method
        self.fasttext_model = fasttext_model
        self.kmeans_model = kmeans_model
        self.hash_scheme = hash_scheme
        self.window_size = window_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize batch-specific state variables
        self.questionmark_token = []
        self.questionmark_token_lower_prob = []
        self.questionmark_token_higher_prob = []
        self.sampled_once = []
        self.sorted_logits = []
        self.sorted_indices = []
        self.cum_probs = []
        self.index = []
        self.bit = []
        self.t = []
        self.prev_generated_tokens = []
        self.prev_encoded_indices = []
        
        # Sachin - new
        self.sampled_token_positions = [[] for _ in range(batch_size)]
        

        # In production, you would use:
        codewords = []
        if crypto_scheme == 'RANDOM':
            for message in batch_messages:
                randomLinearCode = RandomLinearCode(n=128, k=20, seed=42)
                codeword = randomLinearCode.encode(message)
                codewords.append([codeword, randomLinearCode])
                print("USED RANDOM LINEAR CODE")
        if crypto_scheme == 'McEliece':
            for message in batch_messages:
                mceliece = McEliece()
                codeword = mceliece.encrypt(''.join([f'{byte:08b}' for byte in message.encode('utf-8')]))      
                codewords.append([codeword, mceliece])

        elif crypto_scheme == 'Ciphertext':
            for message in batch_messages:
                mceliece = McEliece()
                codeword = mceliece.encrypt(''.join([f'{byte:08b}' for byte in message.encode('utf-8')]))             
                codewords.append([codeword, mceliece])
        self.codewords = codewords
        
        with open('/work/pi_adamoneill_umass_edu/WatermarkingLLM/saved_wrapper_kmeans_2040_n3_minibatch.pkl', 'rb') as f:
            mapping = pickle.load(f)
        self.loaded_wrapper = SimpleKMeansWrapper.__new__(SimpleKMeansWrapper)
        self.loaded_wrapper.mapping = mapping
        self.loaded_wrapper.codeword_length = 128
        print('USING saved_wrapper_kmeans_2040_n3_minibatch.pkl - minibatch one')
        
        # Initialize state for each sequence in the batch
        for b in range(batch_size):
            if hash_scheme == 'hashlib':
                # Initialize context window (previous self.window_size tokens)
                prev_tokens = ["<empty>"] * self.window_size
                for i in range(max(0, len(input_ids[b]) - self.window_size), len(input_ids[b])):
                    token = input_ids[b, i]
                    prev_tokens.append(self.tokenizer.decode(token.item()))
                prev_tokens = prev_tokens[-self.window_size:]
                
                # Hash the context to get the bit index
                idx = int(hashlib.md5("".join(prev_tokens).encode()).hexdigest(), 16) % len(self.codewords[b][0])
            elif hash_scheme == 'kmeans':
                # Decode the entire sequence at once to preserve proper spacing
                # This is important for SentencePiece tokenizers like Mistral
                merged_string = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
                
                # Find the last complete word boundary (last space)
                last_space_idx = merged_string.rfind(' ')

                if last_space_idx != -1:
                    merged_string = merged_string[:last_space_idx]  # Cut off incomplete word
                else:
                    merged_string = ''  # No spaces found, so treat as empty (no complete words)
                
                # Debug: print what we're tokenizing
                # print(f'Debug merged_string: {repr(merged_string)}')
                
                # Tokenize into words
                words = word_tokenize(merged_string)
                # print(f'Debug words after tokenize: {words}')
                
                # Get last self.window_size words (pad with periods if needed)
                prev_tokens = ['.'] * (self.window_size - len(words)) + words[-self.window_size:]
                
                # print('Encoder last self.window_size words constructor: ', prev_tokens)
                # print('Prev tokens:', prev_tokens)
                embeddings = [self.fasttext_model.wv[token] for token in prev_tokens]
                avg_embedding = sum(embeddings) / len(embeddings)
                # Reshape to (1, -1) since predict expects a 2D array
                avg_embedding = np.array(avg_embedding).reshape(1, -1)
                # idx = self.kmeans_model.predict(avg_embedding)[0] % len(self.codewords[b][0])
                idx = self.loaded_wrapper.predict(avg_embedding, "/work/pi_adamoneill_umass_edu/WatermarkingLLM/kmeans_model_2040_n3_minibatch.pkl")
                # print('Avg embedding:', avg_embedding)
                # idx = self.loaded_wrapper.predict(avg_embedding)
                # print('Index:', idx)
                # print('Wanted to encode index:', idx)
            # Append the last self.window_size words for this batch index
            self.prev_generated_tokens.append(prev_tokens)
            self.prev_encoded_indices.append(set([idx]))
            self.index.append(idx)
            # print('Encoder index kmeans:', idx)
            
            # Get the bit to encode
            # print(self.codewords[b][0])
            # print(self.codewords[b][0][0])
            # print('Bit to encode:', self.codewords[b][0][idx])
            self.bit.append(self.codewords[b][0][idx])
            
            # Initialize other state variables
            self.questionmark_token.append(None)
            self.questionmark_token_lower_prob.append(-1)
            self.questionmark_token_higher_prob.append(-1)
            self.sampled_once.append(False)
            self.sorted_logits.append(None)
            self.sorted_indices.append(None)
            self.cum_probs.append(None)
            self.t.append(t)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        # Make a copy of scores that we'll modify
        processed_scores = scores.clone()
        
        # Process each sequence in the batch
        for b in range(self.batch_size):
            # Track previously sampled token if we've already started generation
            if self.sampled_once[b]:
                # Get the token that was sampled in the previous step
                sampled_token = input_ids[b, -1]

                if self.hash_scheme == 'hashlib':
                    # Update context window
                    if len(self.prev_generated_tokens[b]) == self.window_size:
                        self.prev_generated_tokens[b] = self.prev_generated_tokens[b][1:]
                    self.prev_generated_tokens[b].append(self.tokenizer.decode(sampled_token.item()))
                elif self.hash_scheme == 'kmeans':
                    # Decode the entire sequence at once to preserve proper spacing
                    # This is important for SentencePiece tokenizers like Mistral
                    merged_string = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)

                    # Find the last complete word boundary (last space)
                    last_space_idx = merged_string.rfind(' ')

                    if last_space_idx != -1:
                        merged_string = merged_string[:last_space_idx]  # Cut off incomplete word
                    else:
                        merged_string = ''  # No spaces found, so treat as empty (no complete words)

                    # Debug: print what we're tokenizing
                    # print(f'Debug merged_string in call: {repr(merged_string)}')
                    
                    # Tokenize into words
                    words = word_tokenize(merged_string)
                    # print(f'Debug words after tokenize in call: {words}')
                    # Get last self.window_size words (pad with periods if needed)
                    prev_tokens = ['.'] * (self.window_size - len(words)) + words[-self.window_size:]
                    # print('Prevprev tokens:', prev_tokens)

                    # Update prev_generated_tokens[b] with the new context
                    self.prev_generated_tokens[b] = prev_tokens
                    # print('Encoder last self.window_size words call: ', prev_tokens)
                
                
                # Initial token tracking (will be updated if it's a question mark token)
                qm = ''
                self.encoded_bits [b].append(self.bit[b])
                self.encoded_bit_indices[b].append(self.index[b])
                self.t_enc [b].append(self.t[b])
                
                # Check if this was a question mark token (partially encoded bit)
                if self.questionmark_token[b] is not None and sampled_token.item() == self.questionmark_token[b].item():
                    qm = '?'
                    self.encoded_bits [b].pop()
                    self.encoded_bits [b].append('?')
                    # Adjust threshold for next token
                    self.t[b] = (self.t[b] - self.questionmark_token_lower_prob[b]) / (
                        self.questionmark_token_higher_prob[b] - self.questionmark_token_lower_prob[b])
                else:
                    if self.hash_scheme == 'hashlib':
                        idx = new_index = int(hashlib.md5("".join(self.prev_generated_tokens[b]).encode()).hexdigest(), 16) % len(self.codewords[b][0])
                    elif self.hash_scheme == 'kmeans':
                        # print('Prev tokens in call!!!:', prev_tokens)
                        
                        embeddings = [self.fasttext_model.wv[token] for token in prev_tokens]
                        # print('Previous tokens:', prev_tokens)
                        avg_embedding = sum(embeddings) / len(embeddings)
                                    # Reshape to (1, -1) since predict expects a 2D array
                        # print('Avg embedding:', avg_embedding)
                        avg_embedding = np.array(avg_embedding).reshape(1, -1)
                        idx = new_index = self.loaded_wrapper.predict(avg_embedding, "/work/pi_adamoneill_umass_edu/WatermarkingLLM/kmeans_model_2040_n3_minibatch.pkl")
                        # idx = new_index = self.kmeans_model.predict(avg_embedding)[0] % len(self.codewords[b][0])
                        #print('Encoder index kmeans:', idx)

                    #Next Bit Approach

                    if self.enc_method == 'Next':
                        while new_index in self.prev_encoded_indices[b]:
                            #print(new_index,' already encoded for idx',idx,' in batch ',b)
                            new_index += 1
                            new_index = new_index % len(self.codewords[b])
                            if new_index == idx:
                                print("Error: All bits were Encoded")
                                exit()

                    # if self.enc_method == 'Next' or self.enc_method == 'Standard':
                    # print('New index:', new_index)
                    # print('self.bit[b]:', self.bit[b])
                    self.bit[b] = self.codewords[b][0][new_index]

                    #random bit approach
                    if self.enc_method == 'Random':
                        if new_index in self.prev_encoded_indices[b]:
                            print('There was a collision for index', new_index, 'in batch', b)
                            print('Currently encoded bits:', self.prev_encoded_indices[b])
                            self.bit[b] = str(random.randint(0,1))
                            print('Randomly assigned bit:', self.bit[b])

                    self.index[b] = new_index #Earlier: self.index[b] = int(hashlib.md5("".join(self.prev3_generated_tokens[b]).encode()).hexdigest(), 16) % len(self.E)
                    self.prev_encoded_indices[b].add(new_index)
                    self.t[b] = 0.5  # Reset threshold
                
                # Record token and probability information
                # if self.sorted_indices[b] is not None:
                token_pos = (self.sorted_indices[b] == sampled_token.item()).nonzero(as_tuple=True)[0].item()
                prob_of_token_in_cum = self.cum_probs[b][token_pos].item()
                
                # Sachin - storing sampled positions in a list:
                self.sampled_token_positions[b].append(token_pos)

                if prob_of_token_in_cum == self.cum_probs[b][0].item():
                    prob_of_start_of_token = 0
                else:
                    prob_of_start_of_token = self.cum_probs[b][token_pos - 1].item()
                    
                self.sampled_tokens [b].append(qm + self.tokenizer.decode(sampled_token.item()))
                self.token_sampled_probs [b].append(prob_of_token_in_cum)
                self.probs_start [b].append(prob_of_start_of_token)
            else:
                # First token generation
                self.sampled_once[b] = True
            
            # Sort logits and compute cumulative probabilities
            sorted_logits, sorted_indices = torch.sort(scores[b], descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            
            # Determine which tokens to keep based on the bit to encode
            question_mark_token_ind = -1
            # print('Bit to encode in call:', self.bit[b])
            # print('bit in codeword:', self.codewords[b][0][self.index[b]])
            if self.bit[b] == '0':
                # For bit 0, remove tokens with cumulative probability < threshold
                sorted_indices_to_remove = cumulative_probs < self.t[b]
                # Find the boundary token (question mark token)
                question_mark_token_ind = (sorted_indices_to_remove == False).nonzero(as_tuple=True)[0][0]
            else:  # self.bit[b] == '1'
                # For bit 1, remove tokens with cumulative probability > threshold
                sorted_indices_to_remove = cumulative_probs > self.t[b]
                # Find the boundary token and ensure we keep it
                question_mark_token_ind = (sorted_indices_to_remove == True).nonzero(as_tuple=True)[0][0]
                sorted_indices_to_remove[question_mark_token_ind] = False
            
            # Store information about the question mark token
            self.questionmark_token[b] = sorted_indices[question_mark_token_ind]
            self.questionmark_token_higher_prob[b] = cumulative_probs[question_mark_token_ind].item()
            self.questionmark_token_lower_prob[b] = cumulative_probs[question_mark_token_ind - 1].item() if question_mark_token_ind > 0 else 0
            
            # Store for later use
            self.cum_probs[b] = cumulative_probs
            self.sorted_logits[b], self.sorted_indices[b] = sorted_logits, sorted_indices
            
            # Get indices to remove for the current sequence
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            
            # Calculate adjustment factor for question mark token
            adjustment_factor = 0
            if self.bit[b] == '0':
                # For bit=0, only keep probability worth (token_cum_prob - threshold_t)
                adjustment_factor = (
                    self.questionmark_token_higher_prob[b] - self.t[b]) / (
                    self.questionmark_token_higher_prob[b] - self.questionmark_token_lower_prob[b])
            else:  # self.bit[b] == '1'
                # For bit=1, only keep probability worth (threshold_t - token_lower_prob)
                adjustment_factor = (
                    self.t[b] - self.questionmark_token_lower_prob[b]) / (
                    self.questionmark_token_higher_prob[b] - self.questionmark_token_lower_prob[b])
            
            # Adjust the question mark token's logit
            if adjustment_factor > 0:
                qm_token_ind = self.questionmark_token[b].item()
                original_logit = processed_scores[b, qm_token_ind].item()
                new_logit = torch.log(torch.tensor(adjustment_factor)) + original_logit
                processed_scores[b, qm_token_ind] = new_logit
            
            # Apply the filter to remove tokens
            processed_scores[b] = processed_scores[b].masked_fill(indices_to_remove, self.filter_value)
        
        return processed_scores
    