# Watermark Evaluation Framework

Three-way empirical comparison of LLM text watermarking methods:
**Asteroid** (ours), **Stanford**, and **OrZamir**.

## What it evaluates

| Metric | Asteroid | Stanford | OrZamir |
|---|---|---|---|
| Text quality (LLM judge) | ✓ | ✓ | ✓ |
| Watermark detectability (PR curve) | ✓ | ✓ | ✓ |
| Paraphrase robustness (PR curve) | ✓ | ✓ | ✓ |
| No-prompt attack (PR curve) | ✓ | — | — |
| Generation diversity | ✓ | ✓ | — |

## Structure

```
Varun/
├── evaluate.py          # Main evaluation script
├── run.sh               # SLURM job submission script
├── requirements.txt
├── .gitignore
├── Existing_code/       # Exact copies of all external source files
│   ├── stanford/        # generate.py, detect.py, mersenne.py, levenshtein.pyx
│   ├── stanford_watermarking/  # generation.py
│   ├── orzamir/         # generate.py, utils.py, dynamic_ecc.py, compact_text.py
│   └── asteroid/        # warper, decoder (patched), encoder, ecc/
└── results/             # Generated outputs (gitignored)
    ├── results.json
    ├── plots/
    └── diversity_csvs/
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in this directory:

```
OPENAI_API_KEY=sk-...
```

You also need the KMeans model for Asteroid (not included in repo):
```
/work/pi_adamoneill_umass_edu/WatermarkingLLM/kmeans_model_2040_n3_minibatch.pkl
```

## Running

### On SLURM (HPC)

Edit the parameters at the top of `run.sh`, then:

```bash
sbatch run.sh
```

Key parameters in `run.sh`:

| Variable | Default | Description |
|---|---|---|
| `NUM_PROMPTS` | `1` | Number of prompts to evaluate (use 20+ for full results) |
| `MODEL_NAME` | `mistralai/Mistral-7B-v0.1` | HuggingFace model for generation |
| `MAX_TOKENS` | `256` | Max tokens per generated response |
| `WATERMARK_KEY` | `42` | Shared key for watermarking |

Requires a GPU node with 48GB VRAM (`--constraint=vram48`). Logs go to `logs/eval-<jobid>.out`.

### Locally

```bash
export NUM_PROMPTS=1
export MODEL_NAME="mistralai/Mistral-7B-v0.1"
export MAX_TOKENS=256
export WATERMARK_KEY=42
python3 evaluate.py
```

## Outputs

All outputs are written to `results/`:

- `results/results.json` — raw scores for all methods and metrics
- `results/plots/quality_comparison.png` — LLM judge scores per method
- `results/plots/detection_comparison.png` — detection accuracy bar chart
- `results/plots/pr_correctness.png` — PR curve: watermarked vs unwatermarked
- `results/plots/pr_paraphrase.png` — PR curve: paraphrased watermarked vs unwatermarked
- `results/plots/pr_no_prompt_attack.png` — PR curve: Asteroid decoded without prompt
- `results/plots/diversity_comparison.png` — n-gram diversity comparison
- `results/diversity_csvs/` — per-method diversity CSVs (triple-column and single-column)

## Methods

### Asteroid (ours)
Context-aware watermarking using KMeans clustering and FastText embeddings to assign bit values based on semantic context. Uses error-correcting codes (random linear + Reed-Muller) for robustness.

### Stanford
Exponential minimum sampling watermark ([Kuditipudi et al., 2023](https://arxiv.org/abs/2307.15593)). Detected via a permutation test on the generated token sequence.

### OrZamir
Binary token decomposition watermark using a pseudorandom function keyed on context. Detected via a score computed over the binary representation of generated tokens.
