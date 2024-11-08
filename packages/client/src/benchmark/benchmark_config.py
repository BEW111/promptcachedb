from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from .prompts import prompts, PromptName


@dataclass
class BenchmarkConfig:
    mode: Literal["no_cache", "local_cache", "server_cache"]
    prompt_name: PromptName
    number_suffixes: int
    max_new_tokens: int
    model_name: str
    metadata: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())