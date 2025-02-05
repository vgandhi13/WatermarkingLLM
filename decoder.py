import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib
from ecc.mceliece import McEliece
import torch.nn.functional as F

class WatermarkDecoder:
    def __init__(self, model_name="gpt2", message="Asteroid"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to(self.device)
        
        # Generate expected codeword
        # codeword = McEliece().encrypt(message.encode('utf-8'))[0]
        # self.expected_codeword = ''.join(format(byte, '08b') for byte in codeword)
        self.expected_codeword = '100110'
    
    def decode(self, prompt, generated_text):
        prompt_input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prev_tokens = ["<empty>", "<empty>", "<empty>"]
        for i in range(max(0, len(prompt_input_ids[0]) - 3), len(prompt_input_ids[0])):
            token = prompt_input_ids[0, i]
            #print(i, token)
            prev_tokens.append(self.tokenizer.decode(token.item()))
        prev_tokens = prev_tokens[-3:]

        print(prev_tokens)

        input_ids = self.tokenizer.encode(prev_tokens[-1] + generated_text, return_tensors="pt").to(self.device) #look into this

        print(input_ids)
        # Get logits for all tokens
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :] #since the last token has no "next token" to predict.
        print(len(logits[0]))
        extracted_bits = []
        extracted_indices = []
        t = 0.5
        bit = 'unassigned'
        index = -1

        
        for i in range(len(logits[0])):
            
            if bit != '?':
                index = int(hashlib.md5("".join(prev_tokens).encode()).hexdigest(), 16) % len(self.expected_codeword)

            prev_tokens = prev_tokens[1:] + [self.tokenizer.decode(input_ids[0, i + 1].item())]
           
            # Compute top-p probabilities
            sorted_logits, sorted_indices = torch.sort(logits[0, i], descending=True)
            cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Get the token ID of the next token in the sequence
            next_token_id = input_ids[0, i + 1].item()

            # Find the index of this token in the sorted indices
            next_token_position = (sorted_indices == next_token_id).nonzero(as_tuple=True)[0].item()

            # Retrieve its cumulative probability
            prob_of_next_token = cumulative_probs[next_token_position].item()
            if prob_of_next_token == cumulative_probs[0].item():
                prob_of_start_of_token = 0
            else:
                prob_of_start_of_token = cumulative_probs[next_token_position - 1].item()

            print(cumulative_probs)
            print(prob_of_next_token, prob_of_start_of_token)
            
            # Determine encoded bit based on threshold
            if prob_of_next_token < t:
                bit = '1'
            elif prob_of_start_of_token > t: 
                bit = '0'
            else:
                bit = '?'
                t = (t - prob_of_start_of_token)/(prob_of_next_token - prob_of_start_of_token)
            
            if bit != '?':
                extracted_bits.append(bit)
                extracted_indices.append(index)
        
        print(t)
        for x, y in zip(extracted_bits, extracted_indices):
            print('bit', x, ' in index ', y)

# Example usage
decoder = WatermarkDecoder()
result = decoder.decode("Once upon a time", ", I was fishing on the Delaware River towards Pat")
print(result)