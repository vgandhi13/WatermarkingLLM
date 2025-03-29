from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from batch_warper import BatchTopPLogitsWarper

class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


def batch_encoder(prompts, enc_method, messages, model_name, max_tokens=100, batch_size=4):
    """
    Encode watermarks into text generated from multiple prompts simultaneously.
    
    Args:
        prompts: List of input prompts to generate from
        max_tokens: Maximum number of tokens to generate for each prompt
        batch_size: Number of sequences to process simultaneously
        message: Message to encode as watermark
        
    Returns:
        List of dictionaries containing generation results and watermark data
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the padding token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to(device)
    model.eval()
    
    results = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_messages = messages[i:i+batch_size]
        #print(batch_prompts)
        actual_batch_size = len(batch_prompts)
        #print(actual_batch_size)
        
        # Initialize batch trackers
        batch_encoded_bits = [[] for _ in range(actual_batch_size)]
        batch_encoded_indices = [[] for _ in range(actual_batch_size)]
        batch_sampled_tokens = [[] for _ in range(actual_batch_size)]
        batch_token_probs = [[] for _ in range(actual_batch_size)]
        batch_t_enc = [[] for _ in range(actual_batch_size)]
        batch_prob_start = [[] for _ in range(actual_batch_size)]
        
        # Tokenize all prompts in batch
        encoded_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = encoded_inputs.input_ids.to(device)
        # attention_mask = encoded_inputs.attention_mask.to(device)
        
        # Create batch-aware logits processor
        logits_processor_i = BatchTopPLogitsWarper(
            batch_encoded_bits,
            batch_encoded_indices, 
            batch_sampled_tokens, 
            batch_token_probs,
            batch_t_enc, 
            batch_prob_start,
            batch_messages=batch_messages, 
            enc_method = enc_method,
            input_ids=input_ids,
            batch_size=actual_batch_size,
            model_name=model_name
        )
        
        # # Generate tokens for the batch
        # for step in range(max_tokens): #to be changed using generation function
        #     # Get model outputs
        #     outputs = model(input_ids)
        #     # outputs = model(input_ids, attention_mask=attention_mask)
        #     logits = outputs.logits[:, -1, :]  # Last token for each sequence
        #     #print(logits)
            
        #     # Process logits for all sequences in batch
        #     scores_processed = logits_processor_i(
        #         input_ids, 
        #         logits
        #     )
        #     #print('reached here')
            
        #     # Sample next tokens for all sequences
        #     probs = F.softmax(scores_processed, dim=-1)
        #     next_tokens = torch.multinomial(probs, 1)
            
        #     # Add new tokens to input_ids
        #     input_ids = torch.cat([input_ids, next_tokens], dim=1)
        
        logits_processor = LogitsProcessorList()
        logits_processor.append(logits_processor_i)
        output_sequences = model.generate(input_ids=input_ids.to(device), pad_token_id=tokenizer.eos_token_id, logits_processor=logits_processor, max_new_tokens=max_tokens, do_sample=True )
        
        # Decode generated text for each sequence
        for b in range(actual_batch_size):
            # full_text = tokenizer.decode(input_ids[b], skip_special_tokens=True)
            output_con = output_sequences[b]  # Extract generated content
            full_text = tokenizer.decode(output_con, skip_special_tokens=True)
            
            prompt_text = batch_prompts[b]
            
            # Find where the generated text starts after the prompt
            if full_text.startswith(prompt_text):
                generated_text = full_text[len(prompt_text):]
            else:
                # Fallback if exact prompt isn't found at the beginning
                generated_text = full_text
            
            results.append({
                "prompt": prompt_text,
                "generated_text": generated_text,
                "full_text": full_text,
                "encoded_bits": batch_encoded_bits[b],
                "encoded_indices": batch_encoded_indices[b],
                "sampled_tokens": batch_sampled_tokens[b],
                "token_probs": batch_token_probs[b],
                "t_values": batch_t_enc[b],
                "prob_starts": batch_prob_start[b]
            })
    
    return results, model

# Example usage
if __name__ == '__main__':
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far away",
        "It was a dark and stormy night"
    ]
    
    results = batch_encoder(test_prompts, max_tokens=30, batch_size=2)
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"Encoded bits ({len(result['encoded_bits'])}): {result['encoded_bits'][:10]}...")