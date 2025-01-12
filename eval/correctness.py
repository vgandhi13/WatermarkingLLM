from ecc.reed_solomon import ReedSolomonCode
from ecc.permuted_reed_solomon import PermutedReedSolomon
from ecc.mceliece import McEliece



prompts = ["What are the pros and cons of using Generative AI to generate code? Answer this in at least 300 words.", 
           "What is a brief history of Generative AI? Answer this in at least 300 words.", 
           "How does one use Generative AI in a safe and ethical way? Answer this in at least 300 words.",
           "What are the main challenges in using Generative AI? Answer this in at least 300 words.",
           "What are the main benefits of using Generative AI? Answer this in at least 300 words.",
           "Write a short story about a person who uses Generative AI to create a new world. Answer this in at least 300 words.",
           "Write a haiku about Generative AI. Answer this in at least 300 words.",
           "How can Generative AI be used in MedTech? Answer this in at least 300 words.",
           "How can Generative AI be used in education? Answer this in at least 300 words.",
           "Breifly describe some influential people in the history of Generative AI. Answer this in at least 300 words."]
codewords = [ReedSolomonCode(255, 223).encode("Codeword"), PermutedReedSolomon(255, 223).encode("Codeword"), McEliece(255, 223).encode("Codeword")]
watermarked = [watermarker(prompt, codeword) for prompt, codeword in zip(prompts, codewords)]
#here u can do some stuff with the watermarked prompts to pretend to be an adv 
#default testing for now
detects = [detect(text, codeword) for text, codeword in zip(watermarked, codewords)]