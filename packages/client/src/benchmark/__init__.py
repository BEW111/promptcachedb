import argparse
import os
import glob
import itertools
import requests
from timeit import timeit
from typing import Literal
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, pipeline  # type: ignore

from promptcachedb_client import pipeline as pc_pipeline, PromptCacheClient, PipelineWithPromptCache
from .prompts import prompts, PromptName
from .benchmark_config import BenchmarkConfig
from .profile_utils import time_and_log


device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_PROMPT_CACHE_PATH = "./local_prompt_cache"
DEFAULT_SERVER_URL = "http://localhost:8000"


def run_with_benchmark_config(benchmark_config: BenchmarkConfig, server_url: str):
    initial_prompt, prompt_suffixes = prompts[benchmark_config.prompt_name]
    prompt_suffixes = prompt_suffixes[:benchmark_config.number_suffixes]

    responses: list[str] = []

    if benchmark_config.client_type == "no_cache":
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
        client_type: Literal['local', 'server'] = benchmark_config.client_type

        with time_and_log(section_name="create_pc_client", benchmark_config=benchmark_config):
            pc_client = PromptCacheClient(client_type=client_type, cache_server_url=server_url, local_cache_path=LOCAL_PROMPT_CACHE_PATH)

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



def delete_safetensors_files(folder_path: str):
    folder = Path(folder_path)
    for file in folder.glob("*.safetensors"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Failed to delete {file}. Reason: {e}")


def clear_previous_cache(server_url: str):
    delete_safetensors_files(LOCAL_PROMPT_CACHE_PATH)
    
    response = requests.post(f"{server_url}/clear_cache/")
    response.raise_for_status()



def run_benchmark(run_name: str, server_url: str):
    clear_previous_cache(server_url)

    model = MODEL_NAME
    metadata = run_name

    client_type_options: list[Literal['no_cache', 'local', 'server']] = ["server"]
    prompt_name_options: list[PromptName] = ["wikipedia_llms"]
    number_suffixes_options = [1, 2, 3, 5, 7, 10, 20, 30]
    max_new_tokens_options = [10]

    for client_type, prompt_name, number_suffixes, max_new_tokens in itertools.product(
        client_type_options, 
        prompt_name_options, 
        number_suffixes_options, 
        max_new_tokens_options
    ):
        config = BenchmarkConfig(
            client_type=client_type,
            prompt_name=prompt_name,
            number_suffixes=number_suffixes,
            max_new_tokens=max_new_tokens,
            model_name=model,
            metadata=metadata
        )
        print("Running with config:", config)
        run_with_benchmark_config(config, server_url)

        if client_type == "local" or client_type == "server":
            clear_previous_cache(server_url)


def main() -> int:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--run-name")
    arg_parser.add_argument("--server-url")
    args = arg_parser.parse_args()
    server_url = args.server_url if args.server_url else DEFAULT_SERVER_URL
    run_name = args.run_name if args.run_name else 'run_name_not_provided'

    run_benchmark(run_name=args.run_name, server_url=server_url)
    return 0