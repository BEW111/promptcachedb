import os
import glob

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline  # type: ignore
from promptcachedb_client.cache_pipeline import PipelineWithPromptCache
from promptcachedb_client.client import PromptCacheClient

from timeit import timeit


MODEL_NAME = "Qwen/Qwen2.5-0.5B"

INITIAL_PROMPT="""
Prompt caching + persistent prompt db

# Goal

- Release a library that can be used in conjunction with any HF model, that provides the following:
    - cache_activation(model, prompt)
    - run_with_activation(model, cached_prompt, prompt_suffix)
    - The cached activations should be stored in a persistent database
- I really like one of the extensions—making a publicly available prompt cache api
"""

PROMPT_SUFFIXES = ["\n# Project Name", "\n# Next Steps", "\n# Potential issues"]


def run_prompt_with_suffixes(use_prompt_cache: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_prompt_cache:
        # TODO: make this into a pc_pipeline wrapper where we just have to input the model name
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        pc_client = PromptCacheClient(storage_type="server", path_or_url="http://localhost:8000")
        pc_pipe = PipelineWithPromptCache(model, tokenizer, device, client=pc_client)
        pc_pipe.cache_and_upload_prompt(prompt=INITIAL_PROMPT, prompt_name="project_description")

        for prompt in PROMPT_SUFFIXES:
            response = pc_pipe.generate_with_cache(
                cached_prompt_name="project_description",
                prompt=prompt,
                max_new_tokens=25
            )
    else:
        pipe = pipeline(model=MODEL_NAME, device=device)
    
        for prompt in PROMPT_SUFFIXES:
            response = pipe(
                INITIAL_PROMPT + prompt,
                max_new_tokens=25
            )


def main() -> int:
    print("Running model without persistent prompt cache")
    without_cache_time = timeit('run_prompt_with_suffixes(use_prompt_cache=False)', globals=globals(), number=1)
    print(f"Took {without_cache_time} seconds to run")

    print("Running model with persistent prompt cache")
    with_cache_time = timeit('run_prompt_with_suffixes(use_prompt_cache=True)', globals=globals(), number=1)
    print(f"Took {with_cache_time} seconds to run")

    return 0