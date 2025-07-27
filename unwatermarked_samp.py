import torch
from transformers.utils import add_start_docstrings
from transformers.generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING

class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
    
class TopPLogitsWarper(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Often used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

def batch_encoder(prompts, model_name="gpt2", max_tokens=100, batch_size=4, top_p=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have pad token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()

    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        actual_batch_size = len(batch_prompts)

        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        print("device", device)
        input_ids = encoded["input_ids"].to(device)

        logits_processor = LogitsProcessorList()
        logits_processor.append(TopPLogitsWarper(top_p=top_p))

        output_sequences = model.generate(
            input_ids=input_ids.to(device),
            max_new_tokens=max_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor
        )

        for j in range(actual_batch_size):
            full_text = tokenizer.decode(output_sequences[j], skip_special_tokens=True)
            prompt_text = batch_prompts[j]
            generated = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text

            results.append({
                "prompt": prompt_text,
                "generated_text": generated,
                "full_text": full_text
            })

    return results


if __name__ == "__main__":
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far away",
        "It was a dark and stormy night"
    ]

    outputs = batch_encoder(test_prompts, model_name="gpt2", max_tokens=50, batch_size=2)

    for i, res in enumerate(outputs):
        print(f"\nPrompt {i+1}: {res['prompt']}")
        print(f"Generated: {res['generated_text']}")