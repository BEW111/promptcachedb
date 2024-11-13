import os
import requests
from typing import Literal, Optional

import torch
from safetensors.torch import save, save_file, load, load_file

from .prompt_metadata import PromptMetadata


class PromptCacheClient:
    def __init__(self, client_type: Literal['local', 'server'], cache_server_url: Optional[str], local_cache_path: str):
        self.client_type = client_type
        self.cache_server_url = cache_server_url
        self.local_cache_path = local_cache_path


    # TODO: add an `additionally_save_to_local_cache=False` parameter
    def _upload_cache(self, tensors: dict[str, torch.Tensor], prompt_metadata: PromptMetadata) -> None:
        cache_file_name = f"{prompt_metadata.get_file_name()}.safetensors"

        if self.client_type == "local":
            cache_file_path = os.path.join(self.local_cache_path, cache_file_name)
            save_file(tensors, cache_file_path)
        elif self.cache_server_url != None:
            byte_data = save(tensors)
            files = {"prompt_cache_file": (cache_file_name, byte_data)}
            response = requests.post(f"{self.cache_server_url}/upload", files=files)
            response.raise_for_status()

    
    # def _upload_cache_by_layers(self, tensors: dict[str, torch.Tensor], prompt_metadata: PromptMetadata) -> None:
    #     for layer_name, kv_cache in tensors.items():
    #         cache_file_name = f"{prompt_metadata.get_file_name()}_{layer_name}.safetensors"

    #         if self.storage_type == "local":
    #             cache_file_path = os.path.join(self.path_or_url, cache_file_name)
    #             save_file({layer_name: kv_cache}, cache_file_path)
    #         else:
    #             byte_data = save(tensors)
    #             files = {"prompt_cache_file": (cache_file_name, byte_data)}
    #             response = requests.post(f"{self.path_or_url}/upload", files=files)
    #             response.raise_for_status()


    def _load_cache(self, prompt_metadata: PromptMetadata, check_and_save_to_local_cache=True) -> dict[str, torch.Tensor]:
        cache_file_name = f"{prompt_metadata.get_file_name()}.safetensors"
        cache_file_path = os.path.join(self.local_cache_path, cache_file_name)

        if self.client_type == "local":
            cache_file_path = os.path.join(self.local_cache_path, cache_file_name)
            return load_file(cache_file_path)
        else:
            if check_and_save_to_local_cache and os.path.isfile(cache_file_path):
                return load_file(cache_file_path)

            if self.cache_server_url == None:
                raise RuntimeError("Cache server url not specified")
            
            response = requests.get(f"{self.cache_server_url}/load/{prompt_metadata.get_file_name()}")
            response.raise_for_status()
            tensors = load(response.content)

            if check_and_save_to_local_cache:
                save_file(tensors, cache_file_path)

            return tensors

            