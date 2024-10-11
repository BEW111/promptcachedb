from itertools import chain
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # type: ignore
from safetensors.torch import save_file, load_file


@dataclass(frozen=True)
class PromptMetadata:
    prompt_name: str
    model_name: str

    def get_file_name(self) -> str:
        return str(hash(self))


def _save_cache_to_disk(prompt_cache: DynamicCache, path: str):
    tensors = {
        # We should investigate why the KV tensors are not contiguous
        f"{key_or_value}_{layer_index}": tensor.contiguous()
        for key_or_value, layer_index, tensor in
        chain(
            (("k", layer_idx, tensor) for layer_idx, tensor in enumerate(prompt_cache.key_cache)),
            (("v", layer_idx, tensor) for layer_idx, tensor in enumerate(prompt_cache.value_cache))
        )
    }

    save_file(tensors, path)


def _load_cache_from_disk(path: str, device: str) -> DynamicCache:
    cache = DynamicCache()
    tensors = load_file(path)
    num_layers = len(tensors) // 2

    for layer_idx in range(num_layers):
        key_states = tensors[f"k_{layer_idx}"].to(device)
        value_states = tensors[f"v_{layer_idx}"].to(device)
        cache.update(key_states, value_states, layer_idx)

    return cache


def cache_prompt_and_save_to_disk(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    prompt: str,
    prompt_name: str,
    device: str,
    path: str,
):
    original_prompt_cache = DynamicCache()
    inputs_initial_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        original_prompt_cache = model(**inputs_initial_prompt, past_key_values=original_prompt_cache).past_key_values

    prompt_metadata = PromptMetadata(prompt_name, model.config._name_or_path)

    full_path = f"{path}/{prompt_metadata.get_file_name()}.safetensors"
    _save_cache_to_disk(original_prompt_cache, full_path)


def load_prompt_cache_from_disk(
    prompt_metadata: PromptMetadata,
    device: str,
    path: str
) -> DynamicCache:
    full_path = f"{path}/{prompt_metadata.get_file_name()}.safetensors"
    return _load_cache_from_disk(full_path, device)