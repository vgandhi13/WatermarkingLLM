"""
Three-way watermark evaluation: Asteroid vs Stanford vs OrZamir
Evaluates: text quality (LLM judge), watermark detectability, robustness to paraphrasing, diversity.
"""
import os
import sys
import json
import csv
import random
import time
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import datasets
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, auc
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Load modules from explicit paths to avoid name collisions ----
# Stanford's generate.py and OrZamir's generate.py share the same filename,
# so we use importlib to load them under distinct module names.
# We add their directories to sys.path first so their internal imports
# (mersenne, levenshtein, dynamic_ecc, utils, compact_text) can resolve.

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# All external source files are copied into Existing_code/ for self-containment.
# Subdirectory layout:
#   Existing_code/stanford/             ← watermark/demo/
#   Existing_code/stanford_watermarking/ ← watermark/watermarking/
#   Existing_code/orzamir/              ← OrZamir/
#   Existing_code/asteroid/             ← WatermarkingLLM/ (batch_decoder is patched copy)
#     ecc/                              ← WatermarkingLLM/ecc/ + reedmuller package

_VARUN = os.path.abspath(os.path.dirname(__file__))
_EC = os.path.join(_VARUN, 'Existing_code')
_RESULTS = os.path.join(_VARUN, 'results')
_PLOTS = os.path.join(_RESULTS, 'plots')

# Ensure output directories exist
os.makedirs(_PLOTS, exist_ok=True)
os.makedirs(os.path.join(_RESULTS, 'diversity_csvs'), exist_ok=True)

sys.path.insert(0, os.path.join(_EC, 'stanford'))
sys.path.insert(0, os.path.join(_EC, 'stanford_watermarking'))
sys.path.insert(0, os.path.join(_EC, 'orzamir'))
# Existing_code/asteroid/ contains the patched batch_decoder; added first for priority
sys.path.insert(0, os.path.join(_EC, 'asteroid'))
# reedmuller is a package inside ecc/ — its parent must be on sys.path
sys.path.insert(0, os.path.join(_EC, 'asteroid', 'ecc'))

# Stanford — use importlib for generate.py to avoid collision with OrZamir's generate.py
_stanford_gen = _load_module("stanford_generate_mod", os.path.join(_EC, "stanford/generate.py"))
_stanford_det = _load_module("stanford_detect_mod", os.path.join(_EC, "stanford/detect.py"))
_stanford_generation = _load_module("stanford_generation_mod", os.path.join(_EC, "stanford_watermarking/generation.py"))

stanford_generate = _stanford_gen.main2
stanford_detect = _stanford_det.main
stanford_generate_rnd = _stanford_generation.generate_rnd

# Asteroid (unique filenames, no collision)
from batch_main_vary_context_window import batch_encoder as asteroid_encoder
from batch_decoder_vary_context_window import BatchWatermarkDecoder
from unwatermarked_samp import batch_encoder as unwatermarked_encoder

# OrZamir — use importlib for generate.py to avoid collision with Stanford's generate.py
_orzamir_gen = _load_module("orzamir_generate_mod", os.path.join(_EC, "orzamir/generate.py"))

orzamir_watermark = _orzamir_gen.generate_watermarked_response
orzamir_unwatermarked = _orzamir_gen.generate_response_binarize
orzamir_detect_score = _orzamir_gen.compute_score
orzamir_payload = _orzamir_gen.generate_payloaded_response
# OrZamir's utils/compact_text are importable via Existing_code/orzamir/ on sys.path
from utils import start_model as orzamir_start_model
from compact_text import CompactText

# Diversity
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk import word_tokenize
from nltk.util import ngrams
import string
import re

load_dotenv()

# ---- Config from environment (set in run.sh) ----
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", 20))
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-v0.1")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 300))
WATERMARK_KEY = int(os.environ.get("WATERMARK_KEY", 42))

# Asteroid-specific config
BATCH_SIZE = 1
CRYPTO_SCHEME = 'RANDOM'
HASH_SCHEME = 'kmeans'
ENC_DEC_METHOD = 'STANDARD'
KMEANS_MODEL = "/work/pi_adamoneill_umass_edu/WatermarkingLLM/kmeans_model_2040_n3_minibatch.pkl"
WINDOW_SIZE = 3

# OpenAI for LLM judge + paraphrasing
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
OPENAI_WORKERS = 8  # parallel threads for API calls

# ======================================================================
# 1. LOAD PROMPTS
# ======================================================================
def load_prompts():
    dataset = datasets.load_dataset("tatsu-lab/alpaca")
    all_prompts = [
        p['instruction'] for p in
        [{'instruction': inst, 'input': inp}
         for inst, inp in zip(dataset['train']['instruction'], dataset['train']['input'])]
        if not p.get('input') or p['input'].strip() == ''
    ]
    return all_prompts[:NUM_PROMPTS]

# ======================================================================
# 2. LLM JUDGE
# ======================================================================
def llm_judge(text):
    llm_prompt = f"""Please evaluate the following generated text on the following metrics.
        Consider these aspects:
        1. Relevancy (1-10)
        2. Coherence (1-10)
        3. Informativeness (1-10)
        4. Factuality (1-10)
        5. Interestingness (1-10)
        6. Overall Quality (1-10)

        This is an example of a high quality response:
        AI works by using algorithms and data to mimic human intelligence. At its core, it involves training models on large datasets so they can identify patterns and make predictions. Techniques like deep learning use neural networks with many layers to process information similarly to the human brain. These models are used in tasks like language understanding, image recognition, and decision-making. As more data is processed, the AI improves through a process called learning. AI also includes fields like natural language processing, computer vision, and robotics.

        Relevancy: 10/10

        Coherence: 10/10

        Informativeness: 10/10

        Factuality: 10/10

        Interestingness: 9/10

        Overall Quality: 10/10
        This is an example of a medium quality response:
        AI is when computers are made to act smart. It uses stuff like data and algorithms to solve problems. Sometimes it learns things on its own, like figuring out what a cat looks like from pictures. AI is used in phones, websites, and robots. Some AI is really advanced and can write or talk like a person. Scientists keep making it better by feeding it more data and checking how it does. It's kind of like teaching a machine to think, but it's not really thinking like humans do.

        Relevancy: 6/10

        Coherence: 5/10

        Informativeness: 5/10

        Factuality: 6/10

        Interestingness: 4/10

        Overall Quality: 5/10



        This is an example of a low quality response:
        AI is cool and it works with computers. You can make it do things and it's smart. It does jobs and helps. Like when you talk to your phone and it talks back, that's AI. Or when a game plays itself. It just uses codes and learns maybe. AI is like future stuff. It's everywhere. Robots have AI too. AI knows things from data. Sometimes it guesses and is right. It's not magic but close.

        Relevancy: 3/10

        Coherence: 2/10

        Informativeness: 2/10

        Factuality: 2/10

        Interestingness: 3/10

        Overall Quality: 2/10


        Provide evaluation in this exact JSON format:
        {{
            "relevancy": score,
            "coherence": score,
            "informativeness": score,
            "factuality": score,
            "interestingness": score,
            "overall_quality": score
        }}


        Generated text to evaluate:
        """
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "developer", "content": "You are an expert at judging the quality of texts based on their properties. You return the scores in the format provided, and nothing else."},
                {"role": "user", "content": llm_prompt + text}
            ],
            max_completion_tokens=1024
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM judge error: {e}")
        return {"relevancy": 0, "coherence": 0, "informativeness": 0, "factuality": 0, "interestingness": 0, "overall_quality": 0}

# ======================================================================
# 3. PARAPHRASER
# ======================================================================
def paraphrase(text):
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "developer", "content": "You are an expert at paraphrasing text, making sure to keep the same meaning, style, tone, and context. Return the paraphrased text only."},
                {"role": "user", "content": "Paraphrase the following text:\n" + text}
            ],
            max_completion_tokens=1024
        )
        return str(response.choices[0].message.content)
    except Exception as e:
        print(f"Paraphrase error: {e}")
        return text

# ======================================================================
# 3b. PARALLEL WRAPPERS FOR OPENAI CALLS
# ======================================================================
def llm_judge_batch(texts):
    """Run LLM judge on all texts in parallel."""
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=OPENAI_WORKERS) as executor:
        future_to_idx = {executor.submit(llm_judge, t): i for i, t in enumerate(texts)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results


def paraphrase_batch(texts):
    """Paraphrase all texts in parallel."""
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=OPENAI_WORKERS) as executor:
        future_to_idx = {executor.submit(paraphrase, t): i for i, t in enumerate(texts)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results


# ======================================================================
# 4. DIVERSITY
# ======================================================================
def compute_diversity(texts):
    """Compute n-gram diversity across a list of texts."""
    combined = " ".join(texts).lower()
    combined = re.sub(f"[{string.punctuation}]", "", combined)
    tokens = word_tokenize(combined)
    results = {}
    for n in [2, 3, 4]:
        grams = list(ngrams(tokens, n))
        total = len(grams)
        unique = len(set(grams))
        results[f'diversity_{n}g'] = unique / total if total > 0 else 0
    results['product_diversity'] = results['diversity_2g'] * results['diversity_3g'] * results['diversity_4g']
    return results

# ======================================================================
# 5. GENERATION FUNCTIONS
# ======================================================================

_stanford_wat_model = None
_stanford_wat_tokenizer = None

def generate_stanford(prompts, model_name, max_tokens, key, seed=0):
    """Generate watermarked texts using Stanford's method. Model loaded once and cached."""
    global _stanford_wat_model, _stanford_wat_tokenizer
    if _stanford_wat_model is None:
        _stanford_wat_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _stanford_wat_model = AutoModelForCausalLM.from_pretrained(model_name).to(
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        print(f"  Stanford watermark model loaded once")
    torch.manual_seed(seed)
    texts = []
    for p in prompts:
        tokens = _stanford_wat_tokenizer.encode(p, return_tensors='pt', truncation=True, max_length=2048)
        watermarked_tokens = _stanford_gen.generate_shift(
            _stanford_wat_model, tokens, len(_stanford_wat_tokenizer), 256, max_tokens, key
        )[0]
        text = _stanford_wat_tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
        text = text[len(p):]
        texts.append(text)
        print(f"  Stanford watermarked: {text[:80]}...")
    return texts


_stanford_det_tokenizer = None

def detect_stanford(texts, model_name, key):
    """Detect watermark in texts using Stanford's method. Returns list of p-values."""
    global _stanford_det_tokenizer
    if _stanford_det_tokenizer is None:
        _stanford_det_tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  Stanford detect tokenizer loaded once")
    pvals = []
    for text in texts:
        tokens = _stanford_det_tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
        pval = _stanford_det.permutation_test(tokens, key, 256, len(tokens), len(_stanford_det_tokenizer))
        pvals.append(pval)
    return pvals


def generate_asteroid(prompts, model_name, max_tokens):
    """Generate watermarked texts using Asteroid's method.
    Each prompt gets a unique random 20-bit message so different codewords are embedded.
    """
    messages = [random.choices(['0', '1'], k=20) for _ in prompts]
    results, model = asteroid_encoder(
        prompts,
        max_tokens=max_tokens,
        batch_size=BATCH_SIZE,
        kmeans_model_path=KMEANS_MODEL,
        enc_method=ENC_DEC_METHOD,
        messages=messages,
        model_name=model_name,
        crypto_scheme=CRYPTO_SCHEME,
        hash_scheme=HASH_SCHEME,
        window_size=WINDOW_SIZE
    )
    texts = [r["generated_text"] for r in results]
    for t in texts:
        print(f"  Asteroid watermarked: {t[:80]}...")
    return texts, results, model, messages


def detect_asteroid(prompts, texts, model, model_name, messages):
    """Detect watermark using Asteroid's decoder. Returns list of precision scores."""
    decoder = BatchWatermarkDecoder(
        actual_model=model,
        message=messages,
        dec_method=ENC_DEC_METHOD,
        model_name=model_name,
        crypto_scheme=CRYPTO_SCHEME,
        hash_scheme=HASH_SCHEME,
        kmeans_model_path=KMEANS_MODEL,
        window_size=WINDOW_SIZE
    )
    decode_results = decoder.batch_decode(prompts, texts, batch_size=BATCH_SIZE)
    precisions = []
    for dr in decode_results:
        bits = dr['extracted_bits']
        codeword = dr['expected_codeword']
        indices = dr['extracted_indices']
        # Compute match rate as detection score
        matches = 0
        total = 0
        for b, idx in zip(bits, indices):
            if b != '?':
                total += 1
                if idx < len(codeword) and b == codeword[idx]:
                    matches += 1
        precision = matches / total if total > 0 else 0
        precisions.append(precision)
    return precisions


def detect_asteroid_no_prompt(texts, model, model_name, messages):
    """Detect watermark using Asteroid's decoder WITHOUT providing the prompt.
    Simulates a no-prompt attack: attacker only has the generated text, not the original prompt.
    Returns list of precision scores (same metric as detect_asteroid).
    """
    decoder = BatchWatermarkDecoder(
        actual_model=model,
        message=messages,
        dec_method=ENC_DEC_METHOD,
        model_name=model_name,
        crypto_scheme=CRYPTO_SCHEME,
        hash_scheme=HASH_SCHEME,
        kmeans_model_path=KMEANS_MODEL,
        window_size=WINDOW_SIZE
    )
    empty_prompts = [''] * len(texts)
    decode_results = decoder.batch_decode(empty_prompts, texts, batch_size=BATCH_SIZE)
    precisions = []
    for dr in decode_results:
        bits = dr['extracted_bits']
        codeword = dr['expected_codeword']
        indices = dr['extracted_indices']
        matches = 0
        total = 0
        for b, idx in zip(bits, indices):
            if b != '?':
                total += 1
                if idx < len(codeword) and b == codeword[idx]:
                    matches += 1
        precisions.append(matches / total if total > 0 else 0)
    return precisions


def generate_orzamir(prompts, model, tokenizer, key, max_tokens, mode="simple"):
    """Generate watermarked texts using OrZamir's method."""
    texts = []
    for p in prompts:
        if mode == "simple":
            text = orzamir_watermark(key, model, tokenizer, p, length=max_tokens)
            # Remove the prompt from the output
            text = text[len(p):]
        elif mode == "steganographic":
            payload = CompactText.text_to_bits("EXAMPLE PAYLOAD")
            text, ecc = orzamir_payload(key, model, tokenizer, p, payload, length=max_tokens)
            # generate_payloaded_response already strips the prompt
        texts.append(text)
        print(f"  OrZamir watermarked ({mode}): {text[:80]}...")
    return texts


def detect_orzamir(texts, tokenizer, key):
    """Detect watermark using OrZamir's scorer. Returns list of scores."""
    scores = []
    for text in texts:
        score = orzamir_detect_score(key, text, tokenizer)
        scores.append(score)
    return scores


def generate_unwatermarked_stanford(prompts, model_name, max_tokens, seed=0):
    """Generate unwatermarked text. Reuses the cached Stanford model."""
    global _stanford_wat_model, _stanford_wat_tokenizer
    if _stanford_wat_model is None:
        _stanford_wat_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _stanford_wat_model = AutoModelForCausalLM.from_pretrained(model_name).to(
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
    torch.manual_seed(seed)
    texts = []
    for p in prompts:
        tokens = _stanford_wat_tokenizer.encode(p, return_tensors='pt', truncation=True, max_length=2048 - 20)[0]
        text_tokens = stanford_generate_rnd(torch.vstack([tokens]), max_tokens, _stanford_wat_model)
        text = _stanford_wat_tokenizer.decode(text_tokens[0], skip_special_tokens=True)
        if text.startswith(p):
            text = text[len(p):]
        texts.append(text)
        print(f"  Stanford unwatermarked: {text[:80]}...")
    return texts


def generate_unwatermarked_asteroid(prompts, model_name, max_tokens):
    """Generate unwatermarked text using Asteroid's unwatermarked sampler."""
    results = unwatermarked_encoder(prompts, model_name=model_name, max_tokens=max_tokens, batch_size=BATCH_SIZE)
    texts = [r["generated_text"] for r in results]
    for t in texts:
        print(f"  Asteroid unwatermarked: {t[:80]}...")
    return texts


def generate_unwatermarked_orzamir(prompts, model, tokenizer, max_tokens):
    """Generate unwatermarked text using OrZamir's binarized generation."""
    texts = []
    for p in prompts:
        text = orzamir_unwatermarked(model, tokenizer, p, length=max_tokens)
        text = text[len(p):]
        texts.append(text)
        print(f"  OrZamir unwatermarked: {text[:80]}...")
    return texts


# ======================================================================
# 6. MAIN EVALUATION
# ======================================================================
def main():
    print("=" * 60)
    print("THREE-WAY WATERMARK EVALUATION")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompts: {NUM_PROMPTS}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"OrZamir mode: {"simple"}")
    print("=" * 60)

    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts from Alpaca dataset")

    # ------------------------------------------------------------------
    # Load OrZamir model once (Stanford & Asteroid load their own internally)
    # ------------------------------------------------------------------
    print("\n--- Loading OrZamir model ---")
    oz_model, oz_tokenizer = orzamir_start_model(MODEL_NAME)

    # ------------------------------------------------------------------
    # STEP 1 & 2: Generate watermarked + unwatermarked texts
    # Grouped by method to avoid GPU OOM (only one method's model on GPU at a time)
    # ------------------------------------------------------------------
    print("\n--- Generating texts (grouped by method to manage GPU memory) ---")

    # --- Stanford (watermarked + unwatermarked) ---
    print("\n[Stanford watermarked]")
    oz_model.cpu(); torch.cuda.empty_cache()
    stanford_wat_texts = generate_stanford(prompts, MODEL_NAME, MAX_TOKENS, WATERMARK_KEY)
    print("\n[Stanford unwatermarked]")
    stanford_unwat_texts = generate_unwatermarked_stanford(prompts, MODEL_NAME, MAX_TOKENS)
    # Free Stanford model from GPU
    global _stanford_wat_model
    _stanford_wat_model.cpu(); torch.cuda.empty_cache()

    # --- Asteroid (watermarked + unwatermarked) ---
    print("\n[Asteroid watermarked]")
    asteroid_wat_texts, asteroid_enc_results, asteroid_model, asteroid_messages = generate_asteroid(prompts, MODEL_NAME, MAX_TOKENS)

    # Compute bits sent for Asteroid (from encoder results)
    asteroid_bits_sent = []
    for r in asteroid_enc_results:
        enc_idx_bit_map = defaultdict(list)
        for bit, idx in zip(r['encoded_bits'], r['encoded_indices']):
            if bit != '?':
                enc_idx_bit_map[idx].append(bit)
        asteroid_bits_sent.append(sum(len(v) for v in enc_idx_bit_map.values()))
    print(f"  Asteroid avg bits sent: {np.mean(asteroid_bits_sent):.1f}")

    # Move asteroid model to CPU (needed later for detection), free GPU for unwatermarked
    asteroid_model.cpu(); torch.cuda.empty_cache()
    print("\n[Asteroid unwatermarked]")
    asteroid_unwat_texts = generate_unwatermarked_asteroid(prompts, MODEL_NAME, MAX_TOKENS)
    torch.cuda.empty_cache()

    # --- OrZamir (watermarked + unwatermarked) ---
    oz_model.cuda()
    print("\n[OrZamir watermarked]")
    orzamir_wat_texts = generate_orzamir(prompts, oz_model, oz_tokenizer, WATERMARK_KEY, MAX_TOKENS, mode="simple")

    # Compute bits sent for OrZamir (run payloaded generation to measure capacity)
    print("\n[OrZamir bits capacity measurement]")
    orzamir_bits_sent = []
    for p in prompts:
        payload = CompactText.text_to_bits("EXAMPLE PAYLOAD" * 5)
        _, ecc = orzamir_payload(random.random(), oz_model, oz_tokenizer, p, payload, length=MAX_TOKENS)
        bits = ecc.last_index_written + 1
        orzamir_bits_sent.append(bits)
        print(f"  OrZamir bits sent for prompt: {bits}")
    print(f"  OrZamir avg bits sent: {np.mean(orzamir_bits_sent):.1f}")

    print("\n[OrZamir unwatermarked]")
    orzamir_unwat_texts = generate_unwatermarked_orzamir(prompts, oz_model, oz_tokenizer, MAX_TOKENS)
    # Keep OrZamir on GPU for now (detection is CPU-only for OrZamir)

    # ------------------------------------------------------------------
    # STEP 3: Text quality (LLM judge) — parallel API calls
    # ------------------------------------------------------------------
    print("\n--- Evaluating text quality (LLM judge, parallel) ---")
    all_quality_texts = []
    all_quality_labels = []
    for label, texts in [
        ("stanford_wat", stanford_wat_texts),
        ("stanford_unwat", stanford_unwat_texts),
        ("asteroid_wat", asteroid_wat_texts),
        ("asteroid_unwat", asteroid_unwat_texts),
        ("orzamir_wat", orzamir_wat_texts),
        ("orzamir_unwat", orzamir_unwat_texts),
    ]:
        all_quality_texts.extend(texts)
        all_quality_labels.extend([label] * len(texts))

    print(f"  Judging {len(all_quality_texts)} texts in parallel...")
    all_scores = llm_judge_batch(all_quality_texts)

    quality_results = defaultdict(list)
    for label, score in zip(all_quality_labels, all_scores):
        quality_results[label].append(score)
    quality_results = dict(quality_results)

    # ------------------------------------------------------------------
    # STEP 4: Watermark detection (TPR on watermarked, FPR on unwatermarked)
    # ------------------------------------------------------------------
    print("\n--- Watermark detection ---")

    # Stanford detection (p-value based: lower = more likely watermarked)
    print("  Stanford detection on watermarked...")
    stanford_det_wat = detect_stanford(stanford_wat_texts, MODEL_NAME, WATERMARK_KEY)
    print("  Stanford detection on unwatermarked...")
    stanford_det_unwat = detect_stanford(stanford_unwat_texts, MODEL_NAME, WATERMARK_KEY)

    # Asteroid detection (precision based: higher = more likely watermarked)
    # Move OrZamir off GPU, bring Asteroid model back for detection
    oz_model.cpu(); torch.cuda.empty_cache()
    asteroid_model.cuda()
    print("  Asteroid detection on watermarked...")
    asteroid_det_wat = detect_asteroid(prompts, asteroid_wat_texts, asteroid_model, MODEL_NAME, asteroid_messages)
    print("  Asteroid detection on unwatermarked...")
    asteroid_det_unwat = detect_asteroid(prompts, asteroid_unwat_texts, asteroid_model, MODEL_NAME, asteroid_messages)
    asteroid_model.cpu(); torch.cuda.empty_cache()

    # OrZamir detection (score based: higher = more likely watermarked)
    print("  OrZamir detection on watermarked...")
    orzamir_det_wat = detect_orzamir(orzamir_wat_texts, oz_tokenizer, WATERMARK_KEY)
    print("  OrZamir detection on unwatermarked...")
    orzamir_det_unwat = detect_orzamir(orzamir_unwat_texts, oz_tokenizer, WATERMARK_KEY)

    # ------------------------------------------------------------------
    # STEP 5: Robustness to paraphrasing
    # ------------------------------------------------------------------
    print("\n--- Robustness to paraphrasing (parallel API calls) ---")

    # Paraphrase all 60 texts in one parallel batch
    all_para_texts = stanford_wat_texts + asteroid_wat_texts + orzamir_wat_texts
    print(f"  Paraphrasing {len(all_para_texts)} texts in parallel...")
    all_paraphrased = paraphrase_batch(all_para_texts)

    n = len(prompts)
    stanford_paraphrased = all_paraphrased[:n]
    asteroid_paraphrased = all_paraphrased[n:2*n]
    orzamir_paraphrased = all_paraphrased[2*n:]

    print("  Detecting watermark in paraphrased Stanford texts...")
    stanford_det_para = detect_stanford(stanford_paraphrased, MODEL_NAME, WATERMARK_KEY)
    print("  Detecting watermark in paraphrased Asteroid texts...")
    asteroid_model.cuda()
    asteroid_det_para = detect_asteroid(prompts, asteroid_paraphrased, asteroid_model, MODEL_NAME, asteroid_messages)
    asteroid_model.cpu(); torch.cuda.empty_cache()
    print("  Detecting watermark in paraphrased OrZamir texts...")
    orzamir_det_para = detect_orzamir(orzamir_paraphrased, oz_tokenizer, WATERMARK_KEY)

    # No-prompt attack (Asteroid only): decode without providing the prompt
    print("\n--- No-prompt attack detection (Asteroid only) ---")
    asteroid_model.cuda()
    print("  No-prompt detection on watermarked Asteroid texts...")
    asteroid_no_prompt_wat = detect_asteroid_no_prompt(asteroid_wat_texts, asteroid_model, MODEL_NAME, asteroid_messages)
    print("  No-prompt detection on unwatermarked Asteroid texts...")
    asteroid_no_prompt_unwat = detect_asteroid_no_prompt(asteroid_unwat_texts, asteroid_model, MODEL_NAME, asteroid_messages)
    asteroid_model.cpu(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # STEP 6: Diversity (Stanford + Asteroid, 3 rounds with seeds 1/2/3)
    # Following paraphrase-compare.py pattern:
    #   - Triple-column CSV: one round per column (Story1/Story2/Story3), all prompts joined
    #   - Single-column CSV: all 3 rounds joined into one string
    #   - Seed varies per round (1, 2, 3); watermark key stays constant
    #   - Batch size = 1; only Stanford and Asteroid compared
    # ------------------------------------------------------------------
    print("\n--- Diversity evaluation (3 rounds, seeds 1/2/3, Stanford + Asteroid only) ---")

    rounds_stanford_wat = []
    rounds_stanford_unwat = []
    rounds_asteroid_wat = []
    rounds_asteroid_unwat = []

    for round_seed in [1, 2, 3]:
        print(f"  Diversity round seed={round_seed}")

        # Stanford (watermarked + unwatermarked) with varying seed
        _stanford_wat_model.cuda()
        r_stanford_wat = generate_stanford(prompts, MODEL_NAME, MAX_TOKENS, WATERMARK_KEY, seed=round_seed)
        r_stanford_unwat = generate_unwatermarked_stanford(prompts, MODEL_NAME, MAX_TOKENS, seed=round_seed)
        _stanford_wat_model.cpu(); torch.cuda.empty_cache()

        # Asteroid (watermarked + unwatermarked) — seed passed via torch.manual_seed before call
        torch.manual_seed(round_seed)
        r_asteroid_wat, _, _div_model, _ = generate_asteroid(prompts, MODEL_NAME, MAX_TOKENS)
        _div_model.cpu(); del _div_model; torch.cuda.empty_cache()
        torch.manual_seed(round_seed)
        r_asteroid_unwat = generate_unwatermarked_asteroid(prompts, MODEL_NAME, MAX_TOKENS)
        torch.cuda.empty_cache()

        rounds_stanford_wat.append(r_stanford_wat)
        rounds_stanford_unwat.append(r_stanford_unwat)
        rounds_asteroid_wat.append(r_asteroid_wat)
        rounds_asteroid_unwat.append(r_asteroid_unwat)

    # Write CSV files and compute diversity (matching paraphrase-compare.py format)
    csv_dir = os.path.join(_RESULTS, "diversity_csvs")
    os.makedirs(csv_dir, exist_ok=True)

    def write_diversity_csvs_and_score(label, rounds_texts):
        """Write triple-column and single-column CSVs; return diversity scores for each."""
        col1 = ' '.join(rounds_texts[0])
        col2 = ' '.join(rounds_texts[1])
        col3 = ' '.join(rounds_texts[2])
        all_joined = ' '.join(rounds_texts[0] + rounds_texts[1] + rounds_texts[2])

        # Triple-column CSV: Story1/Story2/Story3 (one row, each column = one round joined)
        triple_path = os.path.join(csv_dir, f"{label}_triple.csv")
        with open(triple_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Story1', 'Story2', 'Story3'])
            writer.writerows(zip([col1], [col2], [col3]))
        print(f"  Saved {triple_path}")

        # Single-column CSV: all rounds + all prompts joined into one string
        single_path = os.path.join(csv_dir, f"{label}_single.csv")
        with open(single_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Story1'])
            writer.writerow([all_joined])
        print(f"  Saved {single_path}")

        # Diversity on triple-column: across the 3 story strings
        triple_score = compute_diversity([col1, col2, col3])
        # Diversity on single-column: within the one combined string
        single_score = compute_diversity([all_joined])

        return {"triple": triple_score, "single": single_score}

    diversity_scores = {}
    for label, rounds in [
        ("Stanford-watermarked", rounds_stanford_wat),
        ("Stanford-unwatermarked", rounds_stanford_unwat),
        ("Asteroid-watermarked", rounds_asteroid_wat),
        ("Asteroid-unwatermarked", rounds_asteroid_unwat),
    ]:
        diversity_scores[label] = write_diversity_csvs_and_score(label, rounds)
        print(f"  {label} triple product_diversity: {diversity_scores[label]['triple']['product_diversity']:.4f}")
        print(f"  {label} single product_diversity: {diversity_scores[label]['single']['product_diversity']:.4f}")

    # ------------------------------------------------------------------
    # STEP 7: Save all results
    # ------------------------------------------------------------------
    print("\n--- Saving results ---")

    all_results = {
        "config": {
            "model": MODEL_NAME,
            "num_prompts": NUM_PROMPTS,
            "max_tokens": MAX_TOKENS,
            "orzamir_mode": "simple",
            "watermark_key": WATERMARK_KEY,
        },
        "quality": {k: v for k, v in quality_results.items()},
        "detection": {
            "stanford": {"watermarked_pvals": stanford_det_wat, "unwatermarked_pvals": stanford_det_unwat, "paraphrased_pvals": stanford_det_para},
            "asteroid": {"watermarked_precision": asteroid_det_wat, "unwatermarked_precision": asteroid_det_unwat, "paraphrased_precision": asteroid_det_para,
                         "no_prompt_watermarked": asteroid_no_prompt_wat, "no_prompt_unwatermarked": asteroid_no_prompt_unwat},
            "orzamir": {"watermarked_scores": orzamir_det_wat, "unwatermarked_scores": orzamir_det_unwat, "paraphrased_scores": orzamir_det_para},
        },
        "diversity": diversity_scores,
        "bits_sent": {
            "asteroid": {"per_prompt": asteroid_bits_sent, "avg": float(np.mean(asteroid_bits_sent))},
            "orzamir": {"per_prompt": orzamir_bits_sent, "avg": float(np.mean(orzamir_bits_sent))},
        },
    }

    with open(os.path.join(_RESULTS, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  Saved results/results.json")

    # ------------------------------------------------------------------
    # STEP 8: Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Quality averages
    print("\n--- Text Quality (avg scores) ---")
    for label in ["stanford_wat", "stanford_unwat", "asteroid_wat", "asteroid_unwat", "orzamir_wat", "orzamir_unwat"]:
        scores = quality_results[label]
        avg = {k: sum(d[k] for d in scores) / len(scores) for k in scores[0]}
        print(f"  {label}: {avg}")

    # Detection rates
    print("\n--- Detection (watermarked | unwatermarked | paraphrased) ---")
    # Stanford: fraction with p-value < 0.05
    threshold_s = 0.05
    stanford_tpr = sum(1 for p in stanford_det_wat if p < threshold_s) / len(stanford_det_wat)
    stanford_fpr = sum(1 for p in stanford_det_unwat if p < threshold_s) / len(stanford_det_unwat)
    stanford_para_rate = sum(1 for p in stanford_det_para if p < threshold_s) / len(stanford_det_para)
    print(f"  Stanford (p<{threshold_s}):  TPR={stanford_tpr:.2f}  FPR={stanford_fpr:.2f}  Paraphrased={stanford_para_rate:.2f}")

    # Asteroid: average precision
    ast_tpr = np.mean(asteroid_det_wat)
    ast_fpr = np.mean(asteroid_det_unwat)
    ast_para = np.mean(asteroid_det_para)
    print(f"  Asteroid (avg prec):  Watermarked={ast_tpr:.2f}  Unwatermarked={ast_fpr:.2f}  Paraphrased={ast_para:.2f}")

    # OrZamir: average score
    oz_wat_avg = np.mean(orzamir_det_wat)
    oz_unwat_avg = np.mean(orzamir_det_unwat)
    oz_para_avg = np.mean(orzamir_det_para)
    print(f"  OrZamir (avg score):  Watermarked={oz_wat_avg:.2f}  Unwatermarked={oz_unwat_avg:.2f}  Paraphrased={oz_para_avg:.2f}")

    # Bits sent
    print("\n--- Bits Sent (steganographic capacity) ---")
    print(f"  Asteroid: avg={np.mean(asteroid_bits_sent):.1f} bits per {MAX_TOKENS} tokens")
    print(f"  OrZamir:  avg={np.mean(orzamir_bits_sent):.1f} bits per {MAX_TOKENS} tokens")

    # Diversity
    print("\n--- Diversity (product diversity) ---")
    for label, scores in diversity_scores.items():
        print(f"  {label}  triple={scores['triple']['product_diversity']:.4f}  single={scores['single']['product_diversity']:.4f}")

    # ------------------------------------------------------------------
    # STEP 9: Generate comparison plots
    # ------------------------------------------------------------------
    print("\n--- Generating plots ---")

    # Plot 1: Quality comparison
    labels = ["relevancy", "coherence", "informativeness", "factuality", "interestingness", "overall_quality"]
    method_labels = ["Stanford Wat", "Stanford Unwat", "Asteroid Wat", "Asteroid Unwat", "OrZamir Wat", "OrZamir Unwat"]
    method_keys = ["stanford_wat", "stanford_unwat", "asteroid_wat", "asteroid_unwat", "orzamir_wat", "orzamir_unwat"]
    colors = ['blue', 'lightblue', 'red', 'lightsalmon', 'green', 'lightgreen']

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels))
    width = 0.13
    for i, (mk, ml, c) in enumerate(zip(method_keys, method_labels, colors)):
        scores = quality_results[mk]
        avgs = [sum(d[k] for d in scores) / len(scores) for k in labels]
        ax.bar(x + (i - 2.5) * width, avgs, width, label=ml, color=c, edgecolor='black')

    ax.set_ylabel('Score')
    ax.set_title('Text Quality Comparison (LLM Judge)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(_PLOTS, 'quality_comparison.png'), dpi=300, bbox_inches='tight')
    print("  Saved results/plots/quality_comparison.png")

    # Plot 2: Detection comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Stanford p-values
    axes[0].bar(['Watermarked', 'Unwatermarked', 'Paraphrased'],
                [np.mean(stanford_det_wat), np.mean(stanford_det_unwat), np.mean(stanford_det_para)],
                color=['blue', 'lightblue', 'cornflowerblue'], edgecolor='black')
    axes[0].set_title('Stanford (avg p-value, lower=detected)')
    axes[0].set_ylabel('p-value')

    # Asteroid precision
    axes[1].bar(['Watermarked', 'Unwatermarked', 'Paraphrased'],
                [ast_tpr, ast_fpr, ast_para],
                color=['red', 'lightsalmon', 'indianred'], edgecolor='black')
    axes[1].set_title('Asteroid (avg precision)')
    axes[1].set_ylabel('Precision')

    # OrZamir score
    axes[2].bar(['Watermarked', 'Unwatermarked', 'Paraphrased'],
                [oz_wat_avg, oz_unwat_avg, oz_para_avg],
                color=['green', 'lightgreen', 'mediumseagreen'], edgecolor='black')
    axes[2].set_title('OrZamir (avg score, higher=detected)')
    axes[2].set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(os.path.join(_PLOTS, 'detection_comparison.png'), dpi=300, bbox_inches='tight')
    print("  Saved results/plots/detection_comparison.png")

    def _pr_curve_data(scores_pos, scores_neg):
        """Build labels+scores and return (precision, recall, auc) or None if trivial."""
        scores = scores_pos + scores_neg
        labels = [1] * len(scores_pos) + [0] * len(scores_neg)
        if len(set(labels)) < 2 or len(scores) < 2:
            return None
        prec, rec, _ = precision_recall_curve(labels, scores)
        return prec, rec, auc(rec, prec)

    # Plot 2b-1: Correctness PR — watermarked vs unwatermarked, all 3 methods
    fig, ax = plt.subplots(figsize=(8, 6))
    r = _pr_curve_data([-p for p in stanford_det_wat], [-p for p in stanford_det_unwat])
    if r: ax.plot(r[1], r[0], 'b-', label=f'Stanford (AUC={r[2]:.2f})')
    r = _pr_curve_data(asteroid_det_wat, asteroid_det_unwat)
    if r: ax.plot(r[1], r[0], 'r-', label=f'Asteroid (AUC={r[2]:.2f})')
    r = _pr_curve_data(orzamir_det_wat, orzamir_det_unwat)
    if r: ax.plot(r[1], r[0], 'g-', label=f'OrZamir (AUC={r[2]:.2f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Correctness PR Curve (Watermarked vs Unwatermarked)')
    ax.legend(); ax.set_xlim([0, 1.05]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(_PLOTS, 'pr_correctness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved results/plots/pr_correctness.png")

    # Plot 2b-2: Paraphrase PR — paraphrased vs unwatermarked, all 3 methods
    fig, ax = plt.subplots(figsize=(8, 6))
    r = _pr_curve_data([-p for p in stanford_det_para], [-p for p in stanford_det_unwat])
    if r: ax.plot(r[1], r[0], 'b-', label=f'Stanford (AUC={r[2]:.2f})')
    r = _pr_curve_data(asteroid_det_para, asteroid_det_unwat)
    if r: ax.plot(r[1], r[0], 'r-', label=f'Asteroid (AUC={r[2]:.2f})')
    r = _pr_curve_data(orzamir_det_para, orzamir_det_unwat)
    if r: ax.plot(r[1], r[0], 'g-', label=f'OrZamir (AUC={r[2]:.2f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Paraphrase PR Curve (Paraphrased Watermarked vs Unwatermarked)')
    ax.legend(); ax.set_xlim([0, 1.05]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(_PLOTS, 'pr_paraphrase.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved results/plots/pr_paraphrase.png")

    # Plot 2b-3: No-prompt attack PR — Asteroid only, decoded without prompt
    fig, ax = plt.subplots(figsize=(8, 6))
    r = _pr_curve_data(asteroid_no_prompt_wat, asteroid_no_prompt_unwat)
    if r: ax.plot(r[1], r[0], 'r-', label=f'Asteroid no-prompt (AUC={r[2]:.2f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('No-Prompt Attack PR Curve (Asteroid, decoded without prompt)')
    ax.legend(); ax.set_xlim([0, 1.05]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(_PLOTS, 'pr_no_prompt_attack.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved results/plots/pr_no_prompt_attack.png")

    # Plot 3: Diversity comparison (triple-column and single-column side by side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    div_labels = list(diversity_scores.keys())
    div_colors = ['blue', 'lightblue', 'red', 'lightsalmon']

    triple_values = [diversity_scores[k]['triple']['product_diversity'] for k in div_labels]
    axes[0].bar(div_labels, triple_values, color=div_colors, edgecolor='black')
    axes[0].set_ylabel('Product Diversity')
    axes[0].set_title('Diversity (Triple-Column: across 3 rounds)')
    axes[0].set_xticklabels(div_labels, rotation=15, ha='right')

    single_values = [diversity_scores[k]['single']['product_diversity'] for k in div_labels]
    axes[1].bar(div_labels, single_values, color=div_colors, edgecolor='black')
    axes[1].set_ylabel('Product Diversity')
    axes[1].set_title('Diversity (Single-Column: all rounds combined)')
    axes[1].set_xticklabels(div_labels, rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(_PLOTS, 'diversity_comparison.png'), dpi=300, bbox_inches='tight')
    print("  Saved results/plots/diversity_comparison.png")

    print("\n--- Evaluation complete ---")


if __name__ == '__main__':
    main()
