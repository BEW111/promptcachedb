import io
import os
import requests
from typing import Union, Literal

import torch
from safetensors.torch import save, save_file, load, load_file

from .prompt_metadata import PromptMetadata


class PromptCacheClient:
    def __init__(self, storage_type: Literal["local", "server"], path_or_url: str):
        self.storage_type = storage_type
        self.path_or_url = path_or_url


    def _upload_cache(self, tensors: dict[str, torch.Tensor], prompt_metadata: PromptMetadata) -> None:
        cache_file_name = f"{prompt_metadata.get_file_name()}.safetensors"

        if self.storage_type == "local":
            cache_file_path = os.path.join(self.path_or_url, cache_file_name)
            save_file(tensors, cache_file_path)
        else:
            byte_data = save(tensors)
            files = {"prompt_cache_file": (cache_file_name, byte_data)}
            response = requests.post(f"{self.path_or_url}/upload", files=files)
            response.raise_for_status()


    def _load_cache(self, prompt_metadata: PromptMetadata) -> dict[str, torch.Tensor]:
        cache_file_name = f"{prompt_metadata.get_file_name()}.safetensors"

        if self.storage_type == "local":
            file_path = os.path.join(self.path_or_url, cache_file_name)
            return load_file(file_path)
        else:
            response = requests.get(f"{self.path_or_url}/load/{prompt_metadata.get_file_name()}")
            response.raise_for_status()
            return load(response.content)