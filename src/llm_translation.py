import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

SEED = 42
torch.manual_seed(SEED)

FEW_SHOT_PAIRS = [
    ("The book is on the table.", "Il libro è sul tavolo."),
    ("We are learning generative artificial intelligence.", "Stiamo imparando l'intelligenza artificiale generativa."),
    ("She lives in a beautiful house near the sea.", "Lei vive in una bella casa vicino al mare."),
]


def format_llama_prompt(english_text: str, few_shot_pairs=FEW_SHOT_PAIRS) -> str:
    """Constructs a clean 3-shot in-context prompt for LLaMA 3.2."""
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert English-to-Italian translator. Translate the given English sentence into fluent, natural Italian. "
        "Output ONLY the Italian translation without any explanations or additional commentary.<|eot_id|>"
    )
    for src, tgt in few_shot_pairs:
        prompt += (
            f"<|start_header_id|>user<|end_header_id|>\n\nEnglish: {src}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\nItalian: {tgt}<|eot_id|>"
        )
    prompt += (
        f"<|start_header_id|>user<|end_header_id|>\n\nEnglish: {english_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\nItalian:"
    )
    return prompt


def format_gemma_prompt(english_text: str, few_shot_pairs=FEW_SHOT_PAIRS) -> str:
    """Constructs a prompt for Gemma 2. Gemma does NOT support a system role; instructions are in the user turn."""
    user_turn = "Translate the following English sentences into fluent, natural Italian.\n\n"
    for src, tgt in few_shot_pairs:
        user_turn += f"English: {src}\nItalian: {tgt}\n\n"
    user_turn += f"English: {english_text}\nItalian:"

    prompt = f"<start_of_turn>user\n{user_turn}<end_of_turn>\n<start_of_turn>model\n"
    return prompt


def load_llama_model(model_id="meta-llama/Llama-3.2-1B-Instruct", device=None):
    """Loads LLaMA 3.2 1B in bfloat16 with eager attention."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading LLaMA model ({model_id}) on {device} (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tokenizer


def load_gemma_model(model_id="google/gemma-2-2b-it", device=None):
    """Loads Gemma 2 2B in bfloat16 with eager attention."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Gemma model ({model_id}) on {device} (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def translate_llm_single(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Translates a single prompt with deterministic or sampled decoding."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": [tokenizer.eos_token_id],
    }
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 0.05)
        gen_kwargs["top_p"] = top_p

    outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][input_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Post-clean: take the first line or strip preceding label if any
    first_line = output_text.split("\n")[0].strip()
    if first_line.lower().startswith("italian:"):
        first_line = first_line[8:].strip()
    return first_line


def batch_translate_llm(
    model,
    tokenizer,
    english_sentences,
    model_type="llama",
    device=None,
    do_sample=False,
    temperature=1.0,
    max_new_tokens=64,
):
    """Translates a list of English sentences using the specified LLM."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    format_fn = format_llama_prompt if model_type == "llama" else format_gemma_prompt
    translations = []

    t0 = time.time()
    for text in english_sentences:
        prompt = format_fn(text)
        pred = translate_llm_single(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        translations.append(pred)

    elapsed = time.time() - t0
    latency_per_sent_ms = (elapsed / max(len(english_sentences), 1)) * 1000.0
    return translations, latency_per_sent_ms

