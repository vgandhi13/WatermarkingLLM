from main import encoder
from decoder import WatermarkDecoder

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

for i in range(10):
    prompt = "Once upon a time"
    encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en = encoder()

    decoder = WatermarkDecoder()
    extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec = decoder.decode(prompt, generated_text)

    bits_match = encoded_bits == extracted_bits
    indices_match = encoded_bit_indices == extracted_indices
    tokens_match = sampled_tokens_en == sampled_tokens_dec
    probs_match = token_sampled_probs_en == token_sampled_probs_dec

    print(f"Encoded bits match extracted bits: {GREEN if bits_match else RED}{bits_match}{RESET}")
    print(f"Encoded bit indices match extracted indices: {GREEN if indices_match else RED}{indices_match}{RESET}")

    # Print unified table if there's a mismatch
    if not (bits_match and indices_match and tokens_match and probs_match):
        print("\n⚠️ Mismatch Detected! Side-by-Side Comparison Table:\n")
        print(f"{'Index':<6} | {'Encoded Bit':<12} | {'Extracted Bit':<14} | "
              f"{'Encoded Index':<14} | {'Extracted Index':<14} | "
              f"{'Encoded Token':<20} | {'Extracted Token':<20} | "
              f"{'Encoded Prob':<12} | {'Extracted Prob':<12}")
        print("-" * 150)

        max_len = max(len(encoded_bits), len(extracted_bits), len(encoded_bit_indices), len(extracted_indices), 
                      len(sampled_tokens_en), len(sampled_tokens_dec), len(token_sampled_probs_en), len(token_sampled_probs_dec))

        for i in range(max_len):
            e_bit = encoded_bits[i] if i < len(encoded_bits) else "N/A"
            x_bit = extracted_bits[i] if i < len(extracted_bits) else "N/A"
            e_idx = encoded_bit_indices[i] if i < len(encoded_bit_indices) else "N/A"
            x_idx = extracted_indices[i] if i < len(extracted_indices) else "N/A"
            e_tok = sampled_tokens_en[i] if i < len(sampled_tokens_en) else "N/A"
            x_tok = sampled_tokens_dec[i] if i < len(sampled_tokens_dec) else "N/A"
            e_prob = token_sampled_probs_en[i] if i < len(token_sampled_probs_en) else "N/A"
            x_prob = token_sampled_probs_dec[i] if i < len(token_sampled_probs_dec) else "N/A"
   # Highlight mismatches
            e_bit = f"{RED}{e_bit}{RESET}" if e_bit != x_bit else e_bit
            e_idx = f"{RED}{e_idx}{RESET}" if e_idx != x_idx else e_idx
            e_tok = f"{RED}{e_tok}{RESET}" if e_tok != x_tok else e_tok
            e_prob = f"{RED}{e_prob:.6f}{RESET}" if e_prob != x_prob else f"{e_prob:.6f}"

            print(f"{i:<6} | {e_bit:<12} | {x_bit:<14} | {e_idx:<14} | {x_idx:<14} | {e_tok:<20} | {x_tok:<20} | {e_prob:<12} | {x_prob:<12.6f}")

    print("\n" + "-" * 150 + "\n")
