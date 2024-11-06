from itertools import chain

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # type: ignore

from .prompt_metadata import PromptMetadata
from .client import PromptCacheClient


class PipelineWithPromptCache:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str, client: PromptCacheClient) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.client = client

        # TODO: Temp way to store prompt name -> prompt; we should do this in the server later
        self.prompts: dict[PromptMetadata, str] = {}


    def _cache_prompt(self, prompt: str) -> DynamicCache:
        prompt_cache = DynamicCache()
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            prompt_cache = self.model(**tokenized_prompt, past_key_values=prompt_cache).past_key_values

        return prompt_cache


    def _dynamiccache_to_kv_tensors(self, prompt_cache: DynamicCache) -> dict[str, torch.Tensor]:
        tensors = {
            # We should investigate why the KV tensors are not contiguous
            # - I think this just has to do with the cache using a view of the tensor?
            # - And since saving tensors seems to be fast, I don't think we have to worry
            f"{key_or_value}_{layer_index}": tensor.contiguous()
            for key_or_value, layer_index, tensor in
            chain(
                (("k", layer_idx, tensor) for layer_idx, tensor in enumerate(prompt_cache.key_cache)),
                (("v", layer_idx, tensor) for layer_idx, tensor in enumerate(prompt_cache.value_cache))
            )
        }

        return tensors


    def _kv_tensors_to_dynamiccache(self, kv_tensors: dict[str, torch.Tensor]) -> DynamicCache:
        cache = DynamicCache()
        num_layers = len(kv_tensors) // 2

        for layer_idx in range(num_layers):
            key_states = kv_tensors[f"k_{layer_idx}"].to(self.device)
            value_states = kv_tensors[f"v_{layer_idx}"].to(self.device)
            cache.update(key_states, value_states, layer_idx)

        return cache

    
    def cache_and_upload_prompt(self, prompt: str, prompt_name: str) -> None:
        prompt_metadata = PromptMetadata(prompt_name, self.model.config._name_or_path)
        prompt_cache = self._cache_prompt(prompt)
        tensors = self._dynamiccache_to_kv_tensors(prompt_cache)

        self.prompts[prompt_metadata] = prompt
        self.client._upload_cache(tensors, prompt_metadata)        

    
    def generate_with_cache(
        self, 
        cached_prompt_name: str, 
        prompt: str, 
        max_new_tokens: int
    ) -> str:
        cached_prompt_metadata = PromptMetadata(cached_prompt_name, self.model.config._name_or_path)
        kv_tensors = self.client._load_cache(cached_prompt_metadata)
        prompt_cache = self._kv_tensors_to_dynamiccache(kv_tensors)

        prompt_with_prefix = self.prompts[cached_prompt_metadata] + prompt
        model_inputs = self.tokenizer(prompt_with_prefix, return_tensors="pt").to(self.device)

        outputs = self.model.generate(**model_inputs, past_key_values=prompt_cache, max_new_tokens=max_new_tokens)
        response = self.tokenizer.batch_decode(outputs)[0]

        return response


def pipeline(model: str, device: str, client: PromptCacheClient):
    model_instance = AutoModelForCausalLM.from_pretrained(model).to(device)
    tokenizer_instance = AutoTokenizer.from_pretrained(model)
    return PipelineWithPromptCache(model_instance, tokenizer_instance, device, client)