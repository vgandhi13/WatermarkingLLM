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
        codeword = McEliece().encrypt(message.encode('utf-8'))[0]
        self.expected_codeword = ''.join(format(byte, '08b') for byte in codeword)
    
    def decode(self, prompt, generated_text):
        prompt_input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prev_tokens = ["<empty>", "<empty>", "<empty>"]
        for i in range(max(0, len(prompt_input_ids[0]) - 3), len(prompt_input_ids[0])):
            token = prompt_input_ids[0, i]
            #print(i, token)
            prev_tokens.append(self.tokenizer.decode(token.item()))
        prev_tokens = prev_tokens[-3:]

        print(prev_tokens)

        input_ids = self.tokenizer.encode(prev_tokens[-1] + generated_text, return_tensors="pt").to(self.device)

        print(input_ids)
        # Get logits for all tokens
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :] #since the last token has no "next token" to predict.
        print(len(logits[0]))
        extracted_bits = []
        extracted_indices = []
        
        
        for i in range(3, input_ids.shape[1]):
            prev_tokens = prev_tokens[1:] + [self.tokenizer.decode(input_ids[0, i - 1].item())]
            
            # Compute hash-based index
            index = int(hashlib.md5("".join(prev_tokens).encode()).hexdigest(), 16) % len(self.expected_codeword)
            
            # Compute top-p probabilities
            sorted_logits, sorted_indices = torch.sort(logits[0, i - 3], descending=True)
            cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            
            # Determine encoded bit based on threshold
            if cumulative_probs[0] < 0.5:
                bit = '0'
            else:
                bit = '1'
            
            extracted_bits.append(bit)
            extracted_indices.append(index)
        
        print(extracted_bits, extracted_indices)

# Example usage
decoder = WatermarkDecoder()
result = decoder.decode("Once upon a time", " the kingdom flourished under wise")
print(result)
