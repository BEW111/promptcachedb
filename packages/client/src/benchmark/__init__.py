import os
import glob
from timeit import timeit

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline  # type: ignore

from promptcachedb_client import pipeline as pc_pipeline, PromptCacheClient, PipelineWithPromptCache
from .prompts import WIKIPEDIA_PROMPT_PREFIX, WIKIPEDIA_PROMPT_SUFFIXES, SHORT_PROMPT_PREFIX, SHORT_PROMPT_SUFFIXES
from .profile_utils import cprofile_function_and_save


device = "cuda" if torch.cuda.is_available() else "cpu"


INITIAL_PROMPT = SHORT_PROMPT_PREFIX
PROMPT_SUFFIXES = SHORT_PROMPT_SUFFIXES
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def run_without_cache():
    responses = []

    pipe = pipeline(model=MODEL_NAME, device=device)
    
    for prompt in PROMPT_SUFFIXES:
        response = pipe(
            INITIAL_PROMPT + prompt,
            max_new_tokens=25
        )
        responses.append(response)

    return responses


def run_with_cache():
    responses = []
    
    pc_client = PromptCacheClient(storage_type="server", path_or_url="http://localhost:8000")
    pc_pipe = pc_pipeline(model=MODEL_NAME, device=device, client=pc_client)
    pc_pipe.cache_and_upload_prompt(prompt=INITIAL_PROMPT, prompt_name="project_description")

    for prompt in PROMPT_SUFFIXES:
        response = pc_pipe.generate_with_cache(
            cached_prompt_name="project_description",
            prompt=prompt,
            max_new_tokens=25
        )
        responses.append(response)

    return responses


def benchmark() -> int:
    print("Running model without persistent prompt cache")
    without_cache_responses = run_without_cache()

    print("Running model with persistent prompt cache")
    with_cache_responses = run_with_cache()

    assert without_cache_responses == with_cache_responses, "Responses don't match!"
    return 0


def main() -> int:
    benchmark()
    return 0