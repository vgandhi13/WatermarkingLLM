import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib
from ecc.mceliece import McEliece
import torch.nn.functional as F

class WatermarkDecoder:
    def __init__(self, model_name="gpt2", message="Asteroid"):
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to(self.device)
        self.model.eval()
        # Generate expected codeword
        # codeword = McEliece().encrypt(message.encode('utf-8'))[0]
        # self.expected_codeword = ''.join(format(byte, '08b') for byte in codeword)
        self.expected_codeword = '100110'
    
    def decode(self, prompt, generated_text):
        prompt_input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prev_tokens = ["<empty>", "<empty>", "<empty>"]
        for i in range(max(0, len(prompt_input_ids[0]) - 3), len(prompt_input_ids[0])):
            token = prompt_input_ids[0, i]
            prev_tokens.append(self.tokenizer.decode(token.item()))
        prev_tokens = prev_tokens[-3:]

        # print(prev_tokens)
        # print(self.tokenizer.encode(prompt, return_tensors="pt"))

        input_ids = self.tokenizer.encode(prompt + generated_text, return_tensors="pt").to(self.device) #look into this, to change

        # print(input_ids)
        # Get logits for all tokens
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :] #since the last token has no "next token" to predict.
        # print(len(logits[0]))
        extracted_bits = []
        extracted_indices = []
        sampled_tokens = []
        token_sampled_probs = []
        t_ext = []
        probs_start = []
        t = 0.5
        bit = 'unassigned'
        index = -1
        question_mark = ''

        #print(len(self.tokenizer.encode(prompt, return_tensors="pt")), len(logits[0]))
        for i in range(3, len(logits[0])):
            
            if question_mark != '?':
                index = int(hashlib.md5("".join(prev_tokens).encode()).hexdigest(), 16) % len(self.expected_codeword)

            prev_tokens = prev_tokens[1:] + [self.tokenizer.decode(input_ids[0, i + 1].item())] #to change
           
            # Compute top-p probabilities
            sorted_logits, sorted_indices = torch.sort(logits[0, i], descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
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

            if  i != (len(logits[0]) - 1):
                t_ext.append(t)
            #print("token is ", self.tokenizer.decode(next_token_id)," prob of token is ", prob_of_next_token)
            # Determine encoded bit based on threshold
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
                t = (t - prob_of_start_of_token)/(prob_of_next_token - prob_of_start_of_token)
            
            #print("t is ", t)
            # if bit != '?' and i != (len(logits[0]) - 1):
            if  i != (len(logits[0]) - 1):
                extracted_bits.append(bit)
                extracted_indices.append(index)
                sampled_tokens.append(question_mark + self.tokenizer.decode(next_token_id))
                token_sampled_probs.append(prob_of_next_token)
                probs_start.append(prob_of_start_of_token)
                #print('bit', extracted_bits[-1], ' in index ', extracted_indices[-1])
        
        return extracted_bits, extracted_indices, sampled_tokens, token_sampled_probs, t_ext, probs_start

# Example usage
if __name__ == '__main__':
    decoder = WatermarkDecoder()
    result = decoder.decode("Once upon a time", ", I saw a lot of karasala")
    print(result)