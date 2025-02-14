from main import encoder
from decoder import WatermarkDecoder


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

for i in range(10):
    prompt = "Once upon a time"
    encoded_bits, encoded_bit_indices, generated_text = encoder()

    decoder = WatermarkDecoder()
    extracted_bits, extracted_indices = decoder.decode(prompt, generated_text)

    # print(encoded_bits, encoded_bit_indices)
    # print(extracted_bits, extracted_indices)

    # Compare encoded_bits with extracted_bits
    bits_match = encoded_bits == extracted_bits
    indices_match = encoded_bit_indices == extracted_indices

    bits_match_str = f"{GREEN}True{RESET}" if bits_match else f"{RED}False{RESET}"
    indices_match_str = f"{GREEN}True{RESET}" if indices_match else f"{RED}False{RESET}"

    print(f"Encoded bits match extracted bits: {bits_match_str}")
    print(f"Encoded bit indices match extracted indices: {indices_match_str}")

    # If you want to see mismatches
    if not bits_match:
        print("Mismatched bits:")
        for i, (e, x) in enumerate(zip(encoded_bits, extracted_bits)):
            if e != x:
                print(f"Index {i}: Encoded {e} - Extracted {x}")

    if not indices_match:
        print("Mismatched indices:")
        for i, (e, x) in enumerate(zip(encoded_bit_indices, extracted_indices)):
            if e != x:
                print(f"Index {i}: Encoded {e} - Extracted {x}")
    print()


#todo, add sampled probs and t value to list to check why deviating