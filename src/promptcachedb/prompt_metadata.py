from dataclasses import dataclass


@dataclass(frozen=True)
class PromptMetadata:
    prompt_name: str
    model_name: str

    def get_file_name(self) -> str:
        return str(hash(self))