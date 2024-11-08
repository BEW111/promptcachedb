import os
import glob
import itertools
from timeit import timeit
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline  # type: ignore

from promptcachedb_client import pipeline as pc_pipeline, PromptCacheClient, PipelineWithPromptCache
from .prompts import prompts
from .benchmark_config import BenchmarkConfig
from .profile_utils import time_and_log


device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def run_with_benchmark_config(benchmark_config: BenchmarkConfig):
    initial_prompt, prompt_suffixes = prompts[benchmark_config.prompt_name]
    prompt_suffixes = prompt_suffixes[:benchmark_config.number_suffixes]

    responses: list[str] = []

    if benchmark_config.mode == "no_cache":
        responses = []

        with time_and_log(section_name="create_pipeline", benchmark_config=benchmark_config):
            pipe = pipeline(model=benchmark_config.model_name, device=device)
        
        for prompt in prompt_suffixes:
            with time_and_log(section_name="generate_response", benchmark_config=benchmark_config):
                response = pipe(
                    initial_prompt + prompt,
                    max_new_tokens=benchmark_config.max_new_tokens
                )
            responses.append(response)
    else:
        storage_type: Literal['local', 'server'] = "local"
        path_or_url: str = "./server_prompt_cache"

        if benchmark_config.mode == "server_cache":
            storage_type = "server"
            path_or_url = "http://localhost:8000"

        with time_and_log(section_name="create_pc_client", benchmark_config=benchmark_config):
            pc_client = PromptCacheClient(storage_type=storage_type, path_or_url=path_or_url)

        with time_and_log(section_name="create_pipeline", benchmark_config=benchmark_config):
            pc_pipe = pc_pipeline(model=benchmark_config.model_name, device=device, client=pc_client)

        with time_and_log(section_name="cache_and_upload_prompt", benchmark_config=benchmark_config):
            pc_pipe.cache_and_upload_prompt(prompt=initial_prompt, prompt_name=benchmark_config.prompt_name)

        for prompt in prompt_suffixes:
            with time_and_log(section_name="generate_response", benchmark_config=benchmark_config):
                response = pc_pipe.generate_with_cache(
                    cached_prompt_name=benchmark_config.prompt_name,
                    prompt=prompt,
                    max_new_tokens=benchmark_config.max_new_tokens
                )
            responses.append(response)

    return responses


def clean_previous_cache():
    # clean server if necessary
    # clean local cache if necessary
    ...


'''
- add clean previous cache
- add actually logging down the times of various events
  - i think ill just make a "start timing event" and "end timing event" function
    - then just insert that in various places
    - the main thing to be concerned about here is if logging takes extra time and it's nested
'''

def run_benchmark():
    model = MODEL_NAME
    metadata = "running locally nov 7"

    mode_options = ["no_cache", "local_cache", "server_cache"]
    prompt_name_options = ["short_markdown"]
    number_suffixes_options = [1, 2, 3]
    max_new_tokens_options = [10]

    for mode, prompt_name, number_suffixes, max_new_tokens in itertools.product(
        mode_options, 
        prompt_name_options, 
        number_suffixes_options, 
        max_new_tokens_options
    ):
        config = BenchmarkConfig(
            mode=mode,
            prompt_name=prompt_name,
            number_suffixes=number_suffixes,
            max_new_tokens=max_new_tokens,
            model_name=model,
            metadata=metadata
        )
        print("Running with config:", config)
        run_with_benchmark_config(config)


def main() -> int:
    run_benchmark()
    return 0