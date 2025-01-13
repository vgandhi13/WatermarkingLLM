from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# Choose 1B or 3B model
# model_name = "LLaMA-3B"  # Or "LLaMA-1B"
model_name = "gpt2"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = model.to(device)

import hashlib

def hash_function(context, n):
    """
    Hashes a 3-gram context into an index for the codeword.
    """
    context_str = ' '.join(context)
    hashed_value = int(hashlib.md5(context_str.encode()).hexdigest(), 16)
    return hashed_value % n


def update_threshold(t, lower, upper):
    """
    Updates the threshold t based on the sampled token's cumulative range.
    """
    return (t - lower) / (upper - lower)

import random

def embed_watermark(input_text, codeword, threshold=0.5):
    """
    Iteratively embeds the watermark into the text.
    """
    context = []  # Initial empty context
    prev3 = []
    next_token_id = None

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    print(input_ids)
    generated_ids = input_ids

    for i in range(5):

        outputs = model(generated_ids)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        # next_token_id = torch.argmax(next_token_logits, dim=-1)
        # Convert logits to probabilities
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        #print(next_token_probs)
        sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
        #print(sorted_probs, sorted_indices)

        cumulative_dist = torch.cumsum(sorted_probs, dim=1)
        #print(cumulative_dist)

        #WITH TOP N SAMPLING

    # If you want to sample only from the top-N tokens
        # top_n_probs = sorted_probs[:, :10]
        # top_n_indices = sorted_indices[:, :10]
        
        # # Sample from the top-N tokens using multinomial
        # next_token_id = torch.multinomial(top_n_probs, num_samples=1).squeeze(-1)
        
        # # Map the sampled index back to the actual token in the sorted list
        # next_token_id = top_n_indices.gather(1, next_token_id.unsqueeze(-1)).squeeze(-1)
        
        # generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

        # # Decode the generated text so far
        # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    #WITHOUT TOP N SAMPLING

        # next_token_id = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        # generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)

        # # Decode the generated text so far
        # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # # Print the current generated text
        # print(generated_text)

        # Assign messages
        token_messages = []
        for j in range(cumulative_dist.size(1)):
            if cumulative_dist[0, j] < threshold:
                token_messages.append(1)
            elif cumulative_dist[0, j] >= threshold and (j == 0 or cumulative_dist[0, j-1] < threshold):
                token_messages.append("?")
            else:
                token_messages.append(0)
        
        # Filter tokens based on messages
        filtered_probs = []
        filtered_indices = []
        for idx, message in enumerate(token_messages):
            if message != 0:  # Keep tokens with messages 1 or ?
                filtered_probs.append(sorted_probs[0, idx].item())
                filtered_indices.append(sorted_indices[0, idx].item())
        
        # Convert to tensors and renormalize
        filtered_probs = torch.tensor(filtered_probs, device=device)
        filtered_indices = torch.tensor(filtered_indices, device=device)
        renormalized_probs = filtered_probs / filtered_probs.sum()
        
        # Sample next token
        sampled_idx = torch.multinomial(renormalized_probs, 1).item()
        next_token_id = filtered_indices[sampled_idx]
        
        # Update generated sequence
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        
        # Decode and print the generated text so far
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(generated_text)

    #bit = hash_function(prev3)

# Input data
prompt = "The cat is"
codeword = "0110"

# Generate watermarked text
watermarked_text = embed_watermark(prompt, codeword, threshold=0.5)

