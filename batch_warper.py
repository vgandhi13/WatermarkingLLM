import inspect
import math
from transformers import AutoTokenizer
from typing import Callable, Iterable, List, Optional, Tuple, Union
import hashlib

import numpy as np
import torch
import random
from ecc.mceliece import McEliece

# Uncomment if you're using McEliece for your actual implementation
# from ecc.mceliece import McEliece

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

    def __init__(self, batch_encoded_bits,
            batch_encoded_indices, 
            batch_sampled_tokens, 
            batch_token_probs,
            batch_t_enc, 
            batch_prob_start,
                 batch_messages, enc_method, input_ids, batch_size, model_name = "gpt2", t: float = 0.5, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if t < 0 or t > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {t}")
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
        self.prev3_generated_tokens = []
        self.prev_encoded_indices = []
        
        # For demo purposes, using a simple codeword
        # In production, you would use:
        codewords = []
        for message in batch_messages:
            codeword = McEliece().encrypt(message.encode('utf-8'))[0]
            E = ''.join(format(byte, '08b') for byte in codeword)
            codewords.append(E)
        self.codewords = codewords
        # self.E = '100110'
        
        # Initialize state for each sequence in the batch
        for b in range(batch_size):
            # Initialize context window (previous 3 tokens)
            prev_tokens = ["<empty>", "<empty>", "<empty>"]
            for i in range(max(0, len(input_ids[b]) - 3), len(input_ids[b])):
                token = input_ids[b, i]
                prev_tokens.append(self.tokenizer.decode(token.item()))
            prev_tokens = prev_tokens[-3:]
            self.prev3_generated_tokens.append(prev_tokens)
            
            # Hash the context to get the bit index
            idx = int(hashlib.md5("".join(prev_tokens).encode()).hexdigest(), 16) % len(self.codewords[b])
            self.prev_encoded_indices.append(set([idx]))
            self.index.append(idx)
            
            # Get the bit to encode
            self.bit.append(self.codewords[b][idx])
            
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

                # Update context window
                if len(self.prev3_generated_tokens[b]) == 3:
                    self.prev3_generated_tokens[b] = self.prev3_generated_tokens[b][1:]
                self.prev3_generated_tokens[b].append(self.tokenizer.decode(sampled_token.item()))
                
                # Initial token tracking (will be updated if it's a question mark token)
                qm = ''
                self.encoded_bits [b].append(self.bit[b])
                self.encoded_bit_indices [b].append(self.index[b])
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
                    # Complete bit encoded, move to next bit
                    idx = new_index = int(hashlib.md5("".join(self.prev3_generated_tokens[b]).encode()).hexdigest(), 16) % len(self.codewords[b])
                    #Next Bit Approach
                    if self.enc_method == 'Next':
                        while new_index in self.prev_encoded_indices[b]:
                            print(new_index,' already encoded for idx',idx,' in batch ',b)
                            new_index += 1
                            new_index = new_index % len(self.codewords[b])
                            if new_index == idx:
                                print("Error: All bits were Encoded")
                                exit()
                    if self.enc_method == 'Next' or self.enc_method == 'Standard':
                        self.bit[b] = self.codewords[b][new_index]
                    #random bit approach
                    elif self.enc_method == 'Random':
                        if new_index in self.prev_encoded_indices[b]:
                            self.bit[b] = str(random.randint(0,1))
                    self.index[b] = new_index #Earlier: self.index[b] = int(hashlib.md5("".join(self.prev3_generated_tokens[b]).encode()).hexdigest(), 16) % len(self.E)
                    self.prev_encoded_indices[b].add(new_index)
                    self.t[b] = 0.5  # Reset threshold
                
                # Record token and probability information
                # if self.sorted_indices[b] is not None:
                token_pos = (self.sorted_indices[b] == sampled_token.item()).nonzero(as_tuple=True)[0].item()
                prob_of_token_in_cum = self.cum_probs[b][token_pos].item()
                
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
    
    # def debug_res(self, batch_size, input_ids, batch_prompts):
    #     # Decode generated text for each sequence
    #     for b in range(batch_size):
    #         full_text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
    #         prompt_text = batch_prompts[b]
            
    #         # Find where the generated text starts after the prompt
    #         if full_text.startswith(prompt_text):
    #             generated_text = full_text[len(prompt_text):]
    #         else:
    #             # Fallback if exact prompt isn't found at the beginning
    #             generated_text = full_text
            
    #         return {
    #             "prompt": prompt_text,
    #             "generated_text": generated_text,
    #             "full_text": full_text,
    #             "encoded_bits": self.encoded_bits[b],
    #             "encoded_indices": self.encoded_bit_indices[b],
    #             "sampled_tokens": self.sampled_tokens[b],
    #             "token_probs": self.token_sampled_probs[b],
    #             "t_values": self.t_enc[b],
    #             "prob_starts": self.probs_start[b]
    #         }