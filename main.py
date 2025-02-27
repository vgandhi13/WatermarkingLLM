from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from Warper import TopPLogitsWarper

def encoder():
    # Choose 1B or 3B model
    # model_name = "LLaMA-3B"  # Or "LLaMA-1B"
    model_name = "gpt2"
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to(device)
    model.eval()
    # Example: Initial input

    input_ids = tokenizer.encode("Once upon a time", return_tensors="pt").to(device)
    next_token = -1
    logits_processor = TopPLogitsWarper(message="Asteroid", input_ids = input_ids)
    encoded_bits = []
    encoded_bit_indices = []
    sampled_tokens, token_sampled_probs = [], []
    prob_start = []
    t_enc = []

    # Loop for generating tokens, er
    #while True:
    for step in range(100):
        # Get the logits from the model
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Only consider the last token

        # Apply top-p filtering using our LogitsProcessor
        
        scores_processed = logits_processor(input_ids, logits, encoded_bits, encoded_bit_indices, sampled_tokens, token_sampled_probs, t_enc, prob_start)

        # Apply softmax to get probabilities
        probs = F.softmax(scores_processed, dim=-1)
        # Sample the next token
        next_token = torch.multinomial(probs, 1)
        
        #break if end-of-sequence token is generated
        # if next_token == tokenizer.eos_token_id:
        #     break

        # Append the token to the input for the next step
        input_ids = torch.cat((input_ids, next_token), dim=-1)
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        #print("Text: ", generated_text)

    #print(encoded_bits, encoded_bit_indices, generated_text[16:])
    return encoded_bits, encoded_bit_indices, generated_text[16:], sampled_tokens, token_sampled_probs, t_enc, prob_start
