from main import encoder
from decoder import WatermarkDecoder
from collections import defaultdict

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

for i in range(1):
    prompt = "Once upon a time"
    encoded_bits, encoded_bit_indices, generated_text, sampled_tokens_en, token_sampled_probs_en, t_enc, probs_start_enc = encoder()

    decoder = WatermarkDecoder()
    extracted_bits, extracted_indices, sampled_tokens_dec, token_sampled_probs_dec, t_ext, probs_start_ext = decoder.decode(prompt, generated_text)

    bits_match = encoded_bits == extracted_bits
    indices_match = encoded_bit_indices == extracted_indices
    tokens_match = sampled_tokens_en == sampled_tokens_dec
    probs_match = token_sampled_probs_en == token_sampled_probs_dec

    print(f"Encoded bits match extracted bits: {GREEN if bits_match else RED}{bits_match}{RESET}")
    print(f"Encoded bit indices match extracted indices: {GREEN if indices_match else RED}{indices_match}{RESET}")

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

    idx_mismatch_map = dict()
    total_mismatches = 0
    denominator = 0 #likely less than total because of question marks
    for i, enc_arr in enc_idx_bit_map.items():
        dec_arr = ext_idx_bit_map[i]
        dont_match = 0
        for b1, b2 in zip(enc_arr.copy(), dec_arr.copy()):
            if b1 != b2:
                dont_match+=1
        total_mismatches += dont_match
        idx_mismatch_map[i] = (dont_match, len(enc_arr), len(dec_arr))
        denominator += max(len(enc_arr), len(dec_arr))
    print("Index wise mismatch ", idx_mismatch_map)
    print("Mismatch accuracy ", total_mismatches/denominator)
    # Print unified table if there's a mismatch
    if not (bits_match and indices_match and tokens_match and probs_match):
        print("\n⚠️ Mismatch Detected! Side-by-Side Comparison Table:\n")
        print(f"{'Idx':<5} | {'EBit':<3} | {'ExBit':<3} | {'EIdx':<3} | {'ExIdx':<3} | "
            f"{'Encoded Token':<20} | {'Extracted Token':<20} | "
            f"{'Encoded Prob':<12} | {'Extracted Prob':<12} | "
            f"{'Enc Start Prob':<14} | {'Ext Start Prob':<14} | "
            f"{'Enc T':<8} | {'Ext T':<8}")
        print("-" * 150)

        min_len = min(len(encoded_bits), len(extracted_bits), len(encoded_bit_indices), len(extracted_indices), 
                    len(sampled_tokens_en), len(sampled_tokens_dec), len(token_sampled_probs_en), len(token_sampled_probs_dec))

        for i in range(min_len):
            e_bit = encoded_bits[i] if i < len(encoded_bits) else "N/A"
            x_bit = extracted_bits[i] if i < len(extracted_bits) else "N/A"
            e_idx = encoded_bit_indices[i] if i < len(encoded_bit_indices) else "N/A"
            x_idx = extracted_indices[i] if i < len(extracted_indices) else "N/A"
            e_tok = sampled_tokens_en[i] if i < len(sampled_tokens_en) else "N/A"
            x_tok = sampled_tokens_dec[i] if i < len(sampled_tokens_dec) else "N/A"
            e_prob = token_sampled_probs_en[i] if i < len(token_sampled_probs_en) else "N/A"
            x_prob = token_sampled_probs_dec[i] if i < len(token_sampled_probs_dec) else "N/A"
            e_start_prob = probs_start_enc[i] if i < len(probs_start_enc) else "N/A"
            x_start_prob = probs_start_ext[i] if i < len(probs_start_ext) else "N/A"
            e_t = t_enc[i] if i < len(t_enc) else "N/A"
            x_t = t_ext[i] if i < len(t_ext) else "N/A"

            # Highlight mismatches
            e_bit = f"{RED}{e_bit}{RESET}" if e_bit != x_bit else e_bit
            e_idx = f"{RED}{e_idx}{RESET}" if e_idx != x_idx else e_idx
            e_tok = f"{RED}{e_tok}{RESET}" if e_tok != x_tok else e_tok
            e_prob = f"{RED}{e_prob:.6f}{RESET}" if e_prob != x_prob else f"{e_prob:.6f}"

            print(f"{i:<5} | {e_bit:<3} | {x_bit:<3} | {e_idx:<3} | {x_idx:<3} | "
                f"{e_tok:<20} | {x_tok:<20} | {e_prob:<12} | {x_prob:<12.6f} | "
                f"{e_start_prob:<14} | {x_start_prob:<14.6f} | {e_t:<8} | {x_t:<8.6f}")

        print("\n" + "-" * 150 + "\n")
