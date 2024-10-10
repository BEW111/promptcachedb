import os, copy
from itertools import chain

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # type: ignore
from safetensors.torch import save_file, load_file

from promptcachedb.on_disk_cache import cache_prompt_and_save_to_disk, load_prompt_cache_from_disk

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

PROMPT_CACHE_PATH = "./prompt_cache"

INITIAL_PROMPT="""
Prompt caching + persistent prompt db

# Goal

- Release a library that can be used in conjunction with any HF model, that provides the following:
    - cache_activation(model, prompt)
    - run_with_activation(model, cached_prompt, prompt_suffix)
    - The cached activations should be stored in a persistent database
- I really like one of the extensions—making a publicly available prompt cache api
"""


def main() -> int:
    print("Demo running!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Saving cached prompts to disk...")
    cache_prompt_and_save_to_disk(model, tokenizer, INITIAL_PROMPT, device, PROMPT_CACHE_PATH)

    print("Loading cached prompts from disk...")
    reloaded_prompt_cache = load_prompt_cache_from_disk(INITIAL_PROMPT, device, PROMPT_CACHE_PATH)

    print("Running model with cached prompt prefix and different prompts")
    prompts = ["\n# Project Name", "\n# Next Steps", "\n# Potential issues"]
    responses = []

    for prompt in prompts:
        full_prompt = INITIAL_PROMPT + prompt
        new_inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        # We need to make a copy of the original cache for each prompt, since
        # it gets modified by each of the generation runs in `prompts`
        cache = copy.deepcopy(reloaded_prompt_cache)
        outputs = model.generate(**new_inputs, past_key_values=cache, max_new_tokens=25)
        response = tokenizer.batch_decode(outputs)[0]
        responses.append(response)

    for prompt, response in zip(prompts, responses):
        full_prompt = INITIAL_PROMPT + prompt
        print(prompt)
        print(response[len(full_prompt):])

    return 0