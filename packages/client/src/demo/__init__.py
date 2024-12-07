import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # type: ignore
from promptcachedb_client.cache_pipeline import pipeline as pc_pipeline
from promptcachedb_client.client import PromptCacheClient


PROMPT_CACHE_PATH = "./demo_prompt_cache"
os.makedirs(PROMPT_CACHE_PATH, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_SERVER_URL = "http://localhost:8000"

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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--client-type")
    arg_parser.add_argument("--server-url")
    args = arg_parser.parse_args()
    client_type = args.client_type if args.client_type else "server"
    server_url = args.server_url if args.server_url else DEFAULT_SERVER_URL

    print("Demo running!")
    pc_client = PromptCacheClient(client_type=client_type, cache_server_url=server_url, local_cache_path=PROMPT_CACHE_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc_pipe = pc_pipeline(model=MODEL_NAME, device=device, client=pc_client)

    print("Uploading cached prompts...")
    pc_pipe.cache_and_upload_prompt(prompt=INITIAL_PROMPT, prompt_name="project_description")

    print("Running model with cached prompt prefix and different prompts")
    prompts = ["\n# Project Name", "\n# Next Steps", "\n# Potential issues"]
    
    for prompt in prompts:
        response = pc_pipe.generate_with_cache(
            cached_prompt_name="project_description",
            prompt=prompt,
            max_new_tokens=25
        )
        print(prompt)
        print(response)

    return 0